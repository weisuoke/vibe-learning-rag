# InMemoryVectorStore 源码分析

**文件路径**: `sourcecode/langchain/libs/core/langchain_core/vectorstores/in_memory.py`

## 类定义

```python
class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation.

    Uses a dictionary, and computes cosine similarity for search using numpy.
    """
```

## 核心属性

```python
def __init__(self, embedding: Embeddings) -> None:
    """Initialize with the given embedding function.

    Args:
        embedding: embedding function to use.
    """
    self.store: dict[str, dict[str, Any]] = {}
    self.embedding = embedding
```

**存储结构**：
```python
self.store = {
    "doc_id_1": {
        "id": "doc_id_1",
        "vector": [0.1, 0.2, ...],  # embedding vector
        "text": "document content",
        "metadata": {"key": "value"}
    },
    "doc_id_2": {...},
    ...
}
```

## 核心方法

### 1. 文档添加

#### 同步添加

```python
@override
def add_documents(
    self,
    documents: list[Document],
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    texts = [doc.page_content for doc in documents]
    vectors = self.embedding.embed_documents(texts)

    if ids and len(ids) != len(texts):
        msg = (
            f"ids must be the same length as texts. "
            f"Got {len(ids)} ids and {len(texts)} texts."
        )
        raise ValueError(msg)

    id_iterator: Iterator[str | None] = (
        iter(ids) if ids else iter(doc.id for doc in documents)
    )

    ids_ = []

    for doc, vector in zip(documents, vectors, strict=False):
        doc_id = next(id_iterator)
        doc_id_ = doc_id or str(uuid.uuid4())
        ids_.append(doc_id_)
        self.store[doc_id_] = {
            "id": doc_id_,
            "vector": vector,
            "text": doc.page_content,
            "metadata": doc.metadata,
        }

    return ids_
```

**关键点**：
- 批量 embedding：一次性对所有文档进行 embedding
- ID 生成：如果没有提供 ID，使用 UUID 生成
- 存储格式：字典结构，key 为 doc_id，value 为文档信息

#### 异步添加

```python
@override
async def aadd_documents(
    self, documents: list[Document], ids: list[str] | None = None, **kwargs: Any
) -> list[str]:
    texts = [doc.page_content for doc in documents]
    vectors = await self.embedding.aembed_documents(texts)

    # ... 相同的逻辑
```

### 2. 文档检索

#### get_by_ids

```python
@override
def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
    """Get documents by their ids.

    Args:
        ids: The IDs of the documents to get.

    Returns:
        A list of Document objects.
    """
    documents = []

    for doc_id in ids:
        doc = self.store.get(doc_id)
        if doc:
            documents.append(
                Document(
                    id=doc["id"],
                    page_content=doc["text"],
                    metadata=doc["metadata"],
                )
            )
    return documents
```

### 3. 相似度搜索

#### 核心搜索方法

```python
def _similarity_search_with_score_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    filter: Callable[[Document], bool] | None = None,
) -> list[tuple[Document, float, list[float]]]:
    # Get all docs with fixed order in list
    docs = list(self.store.values())

    if filter is not None:
        docs = [
            doc
            for doc in docs
            if filter(
                Document(
                    id=doc["id"], page_content=doc["text"], metadata=doc["metadata"]
                )
            )
        ]

    if not docs:
        return []

    similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

    # Get the indices ordered by similarity score
    top_k_idx = similarity.argsort()[::-1][:k]

    return [
        (
            Document(
                id=doc_dict["id"],
                page_content=doc_dict["text"],
                metadata=doc_dict["metadata"],
            ),
            float(similarity[idx].item()),
            doc_dict["vector"],
        )
        for idx in top_k_idx
        if (doc_dict := docs[idx])
    ]
```

**关键点**：
- 使用 cosine_similarity 计算相似度
- 支持元数据过滤（filter 参数）
- 返回 top-k 结果
- 返回值包含：Document、相似度分数、向量

#### similarity_search_with_score

```python
@override
def similarity_search_with_score(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[tuple[Document, float]]:
    embedding = self.embedding.embed_query(query)
    return self.similarity_search_with_score_by_vector(
        embedding,
        k,
        **kwargs,
    )
```

#### similarity_search

```python
@override
def similarity_search(
    self, query: str, k: int = 4, **kwargs: Any
) -> list[Document]:
    return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]
```

### 4. MMR 搜索

```python
@override
def max_marginal_relevance_search_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    *,
    filter: Callable[[Document], bool] | None = None,
    **kwargs: Any,
) -> list[Document]:
    prefetch_hits = self._similarity_search_with_score_by_vector(
        embedding=embedding,
        k=fetch_k,
        filter=filter,
    )

    if not _HAS_NUMPY:
        msg = (
            "numpy must be installed to use max_marginal_relevance_search "
            "pip install numpy"
        )
        raise ImportError(msg)

    mmr_chosen_indices = maximal_marginal_relevance(
        np.array(embedding, dtype=np.float32),
        [vector for _, _, vector in prefetch_hits],
        k=k,
        lambda_mult=lambda_mult,
    )
    return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]
```

**关键点**：
- 先获取 fetch_k 个候选文档
- 使用 maximal_marginal_relevance 算法选择 k 个文档
- 需要 numpy 支持

```python
@override
def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]:
    embedding_vector = self.embedding.embed_query(query)
    return self.max_marginal_relevance_search_by_vector(
        embedding_vector,
        k,
        fetch_k,
        lambda_mult=lambda_mult,
        **kwargs,
    )
```

### 5. 文档删除

```python
@override
def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
    if ids:
        for id_ in ids:
            self.store.pop(id_, None)
```

### 6. 持久化

#### 保存

```python
def dump(self, path: str) -> None:
    """Dump the vector store to a file.

    Args:
        path: The path to dump the vector store to.
    """
    path_: Path = Path(path)
    path_.parent.mkdir(exist_ok=True, parents=True)
    with path_.open("w", encoding="utf-8") as f:
        json.dump(dumpd(self.store), f, indent=2)
```

#### 加载

```python
@classmethod
def load(
    cls, path: str, embedding: Embeddings, **kwargs: Any
) -> InMemoryVectorStore:
    """Load a vector store from a file.

    Args:
        path: The path to load the vector store from.
        embedding: The embedding to use.
        **kwargs: Additional arguments to pass to the constructor.

    Returns:
        A VectorStore object.
    """
    path_: Path = Path(path)
    with path_.open("r", encoding="utf-8") as f:
        store = load(json.load(f), allowed_objects=[Document])
    vectorstore = cls(embedding=embedding, **kwargs)
    vectorstore.store = store
    return vectorstore
```

### 7. 工厂方法

```python
@classmethod
@override
def from_texts(
    cls,
    texts: list[str],
    embedding: Embeddings,
    metadatas: list[dict] | None = None,
    **kwargs: Any,
) -> InMemoryVectorStore:
    store = cls(
        embedding=embedding,
    )
    store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
    return store
```

## 使用示例

### 基础用法

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 创建 VectorStore
vector_store = InMemoryVectorStore(OpenAIEmbeddings())

# 添加文档
from langchain_core.documents import Document

document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
document_3 = Document(id="3", page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
vector_store.add_documents(documents=documents)
```

### 检索

```python
# 相似度搜索
results = vector_store.similarity_search(query="thud", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# 带分数的搜索
results = vector_store.similarity_search_with_score(query="qux", k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

### 过滤

```python
def _filter_function(doc: Document) -> bool:
    return doc.metadata.get("bar") == "baz"

results = vector_store.similarity_search(
    query="thud", k=1, filter=_filter_function
)
```

### 转换为 Retriever

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)
retriever.invoke("thud")
```

### 持久化

```python
# 保存
vector_store.dump("vectorstore.json")

# 加载
loaded_store = InMemoryVectorStore.load("vectorstore.json", embedding=embeddings)
```

## 关键设计模式

### 1. 简单存储

使用 Python 字典作为存储，简单高效：
```python
self.store: dict[str, dict[str, Any]] = {}
```

### 2. 批量 Embedding

一次性对所有文档进行 embedding，提高效率：
```python
vectors = self.embedding.embed_documents(texts)
```

### 3. 余弦相似度

使用 cosine_similarity 计算相似度：
```python
similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]
```

### 4. 元数据过滤

支持自定义过滤函数：
```python
filter: Callable[[Document], bool] | None = None
```

### 5. MMR 算法

使用 maximal_marginal_relevance 实现多样性检索：
```python
mmr_chosen_indices = maximal_marginal_relevance(
    np.array(embedding, dtype=np.float32),
    [vector for _, _, vector in prefetch_hits],
    k=k,
    lambda_mult=lambda_mult,
)
```

## 性能特点

### 优点

1. **简单易用**：纯 Python 实现，无需外部依赖
2. **快速原型**：适合快速开发和测试
3. **完整功能**：支持所有 VectorStore 接口
4. **持久化**：支持保存和加载

### 缺点

1. **内存限制**：所有数据存储在内存中
2. **线性搜索**：O(n) 复杂度，不适合大规模数据
3. **无索引**：没有使用高级索引结构（如 HNSW、IVF）
4. **单机限制**：无法分布式扩展

## 适用场景

1. **开发测试**：快速原型和单元测试
2. **小规模数据**：文档数量 < 10,000
3. **学习示例**：理解 VectorStore 接口
4. **临时存储**：短期使用，不需要持久化

## 关键洞察

1. **接口实现**：完整实现了 VectorStore 接口，是学习 VectorStore 的最佳示例
2. **简单高效**：对于小规模数据，性能足够好
3. **易于扩展**：可以作为自定义 VectorStore 的模板
4. **测试友好**：非常适合单元测试和集成测试
5. **Retriever 集成**：通过 as_retriever() 无缝转换为 Retriever

## 与生产级 VectorStore 的对比

| 特性 | InMemoryVectorStore | 生产级 VectorStore (如 Pinecone, Weaviate) |
|------|---------------------|-------------------------------------------|
| 存储 | 内存 | 持久化存储 |
| 索引 | 无 | HNSW, IVF 等 |
| 搜索复杂度 | O(n) | O(log n) |
| 扩展性 | 单机 | 分布式 |
| 数据规模 | < 10K | 百万级+ |
| 适用场景 | 开发测试 | 生产环境 |
