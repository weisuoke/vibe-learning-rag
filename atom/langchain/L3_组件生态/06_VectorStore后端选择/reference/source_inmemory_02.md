---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/vectorstores/in_memory.py
analyzed_files:
  - sourcecode/langchain/libs/core/langchain_core/vectorstores/in_memory.py
analyzed_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
---

# 源码分析：InMemoryVectorStore 实现

## 分析的文件

- `sourcecode/langchain/libs/core/langchain_core/vectorstores/in_memory.py` - 内存向量存储实现

## 关键发现

### 1. InMemoryVectorStore 特点

InMemoryVectorStore 是 LangChain 提供的最简单的向量存储实现：
- **纯内存存储**：使用 Python 字典存储向量和文档
- **无外部依赖**：只依赖 numpy（可选）
- **适合场景**：测试、原型开发、小规模数据

### 2. 数据结构

```python
self.store: dict[str, dict[str, Any]] = {}
```

每个文档存储为：
```python
{
    "id": "document_id",
    "vector": [0.1, 0.2, ...],  # embedding 向量
    "text": "document content",
    "metadata": {"key": "value"}
}
```

### 3. 核心实现

#### 添加文档

```python
def add_documents(
    self,
    documents: list[Document],
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    texts = [doc.page_content for doc in documents]
    vectors = self.embedding.embed_documents(texts)  # 批量 embedding

    # 生成或使用提供的 ID
    if ids and len(ids) != len(texts):
        raise ValueError(...)

    ids = ids or [str(uuid.uuid4()) for _ in texts]

    # 存储到字典
    for doc_id, doc, vector in zip(ids, documents, vectors, strict=False):
        self.store[doc_id] = {
            "id": doc_id,
            "vector": vector,
            "text": doc.page_content,
            "metadata": doc.metadata,
        }

    return ids
```

#### 相似度检索

```python
def similarity_search_with_score_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    filter: Callable[[Document], bool] | None = None,
    **kwargs: Any,
) -> list[tuple[Document, float]]:
    # 1. 提取所有向量
    vectors = [item["vector"] for item in self.store.values()]

    # 2. 计算余弦相似度
    similarities = cosine_similarity([embedding], vectors)[0]

    # 3. 排序并返回 top-k
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # 4. 应用过滤器（如果有）
    results = []
    for idx in top_k_indices:
        doc_id = list(self.store.keys())[idx]
        item = self.store[doc_id]
        doc = Document(
            id=item["id"],
            page_content=item["text"],
            metadata=item["metadata"]
        )
        if filter is None or filter(doc):
            results.append((doc, similarities[idx]))

    return results
```

#### MMR 检索

```python
def max_marginal_relevance_search_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]:
    # 1. 先检索 fetch_k 个候选
    # 2. 使用 MMR 算法重排序
    # 3. 返回 top-k
    ...
```

### 4. 文档中的使用示例

源码中包含了完整的使用示例（在 docstring 中）：

```python
# 初始化
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore(OpenAIEmbeddings())

# 添加文档
from langchain_core.documents import Document

document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
documents = [document_1, document_2]
vector_store.add_documents(documents=documents)

# 检索
results = vector_store.similarity_search(query="thud", k=1)

# 带过滤器的检索
def _filter_function(doc: Document) -> bool:
    return doc.metadata.get("bar") == "baz"

results = vector_store.similarity_search(
    query="thud", k=1, filter=_filter_function
)

# 带分数的检索
results = vector_store.similarity_search_with_score(query="qux", k=1)

# 作为 Retriever 使用
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)
retriever.invoke("thud")
```

## 代码片段

### 初始化

```python
class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation.

    Uses a dictionary, and computes cosine similarity for search using numpy.
    """

    def __init__(self, embedding: Embeddings) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        self.store: dict[str, dict[str, Any]] = {}
        self.embedding = embedding

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding
```

### 删除文档

```python
@override
def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
    if ids:
        for id_ in ids:
            self.store.pop(id_, None)  # 安全删除，不存在也不报错

@override
async def adelete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
    self.delete(ids)  # 异步版本直接调用同步版本
```

### 检查文档

```python
# 遍历存储的文档
top_n = 10
for index, (id, doc) in enumerate(vector_store.store.items()):
    if index < top_n:
        # docs have keys 'id', 'vector', 'text', 'metadata'
        print(f"{id}: {doc['text']}")
    else:
        break
```

## 优缺点分析

### 优点
1. **零配置**：无需安装额外依赖或启动服务
2. **快速原型**：适合快速验证想法
3. **完整功能**：支持所有 VectorStore 接口
4. **代码简单**：易于理解和调试

### 缺点
1. **不持久化**：程序重启后数据丢失
2. **内存限制**：只能处理小规模数据
3. **性能较差**：大规模数据检索慢
4. **无并发优化**：不适合生产环境

## 适用场景

- ✅ 单元测试
- ✅ 原型开发
- ✅ 教学演示
- ✅ 小规模数据（< 1000 文档）
- ❌ 生产环境
- ❌ 大规模数据
- ❌ 需要持久化
- ❌ 高并发场景

## 与其他后端的对比

| 特性 | InMemory | Chroma | FAISS | Pinecone |
|------|----------|--------|-------|----------|
| 持久化 | ❌ | ✅ | ✅ | ✅ |
| 外部依赖 | 无 | chromadb | faiss | pinecone-client |
| 性能 | 低 | 中 | 高 | 高 |
| 适合规模 | < 1K | < 100K | < 1M | 无限 |
| 部署复杂度 | 极低 | 低 | 低 | 中 |
