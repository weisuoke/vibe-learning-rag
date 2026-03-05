# 核心概念 1：VectorStore 抽象接口

## 概述

VectorStore 是 LangChain 中所有向量存储后端的统一抽象接口，定义了向量存储的核心操作规范。通过这个抽象层，开发者可以轻松切换不同的向量存储后端（Chroma、Qdrant、Pinecone 等），而无需修改业务代码。

**核心价值**：
- 统一接口：所有向量存储后端实现相同的接口
- 易于切换：更换后端只需修改初始化代码
- 功能完整：支持增删改查、检索、过滤等完整功能
- 异步支持：原生支持异步操作，适合高并发场景

## 1. VectorStore 抽象基类设计

### 1.1 类定义

```python
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class VectorStore(ABC):
    """向量存储的抽象基类

    所有向量存储后端都必须继承这个类并实现核心方法。
    """

    @property
    @abstractmethod
    def embeddings(self) -> Embeddings:
        """返回 embedding 函数"""
        pass
```

**设计要点**：
- 使用 ABC（Abstract Base Class）确保子类实现必需方法
- `embeddings` 属性提供统一的 embedding 函数访问
- 所有子类必须实现抽象方法

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

### 1.2 核心设计模式

**模板方法模式**：
```python
def add_texts(self, texts: Iterable[str], **kwargs) -> list[str]:
    """添加文本（模板方法）"""
    # 默认实现：调用 add_documents()
    if type(self).add_documents != VectorStore.add_documents:
        docs = [Document(page_content=text) for text in texts]
        return self.add_documents(docs, **kwargs)
    raise NotImplementedError()

def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
    """添加文档（模板方法）"""
    # 默认实现：调用 add_texts()
    if type(self).add_texts != VectorStore.add_texts:
        texts = [doc.page_content for doc in documents]
        return self.add_texts(texts, **kwargs)
    raise NotImplementedError()
```

**关键特性**：
- `add_texts()` 和 `add_documents()` 互相调用
- 子类只需实现其中一个方法
- 框架自动适配另一个方法

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

## 2. 核心接口方法

### 2.1 数据操作方法

#### 添加数据

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """添加文本到向量存储

    Args:
        texts: 要添加的文本列表
        metadatas: 元数据列表（可选）
        ids: 文档 ID 列表（可选）
        **kwargs: 后端特定参数

    Returns:
        添加的文档 ID 列表
    """
    pass

def add_documents(
    self,
    documents: list[Document],
    **kwargs: Any,
) -> list[str]:
    """添加文档到向量存储

    Args:
        documents: Document 对象列表
        **kwargs: 后端特定参数

    Returns:
        添加的文档 ID 列表
    """
    pass
```

**使用示例**：
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 初始化向量存储
vector_store = InMemoryVectorStore(OpenAIEmbeddings())

# 方式1：添加文本
ids = vector_store.add_texts(
    texts=["文档1内容", "文档2内容"],
    metadatas=[{"source": "file1"}, {"source": "file2"}]
)

# 方式2：添加文档
docs = [
    Document(page_content="文档1内容", metadata={"source": "file1"}),
    Document(page_content="文档2内容", metadata={"source": "file2"})
]
ids = vector_store.add_documents(docs)
```

[来源: reference/source_vectorstore_base_01.md, reference/source_inmemory_02.md | LangChain 源码分析]

#### 删除数据

```python
def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
    """删除向量

    Args:
        ids: 要删除的文档 ID 列表
        **kwargs: 后端特定参数
    """
    pass
```

**使用示例**：
```python
# 删除指定 ID 的文档
vector_store.delete(ids=["doc1", "doc2"])
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

#### 查询数据

```python
def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
    """根据 ID 获取文档

    Args:
        ids: 文档 ID 列表

    Returns:
        Document 对象列表
    """
    pass
```

**使用示例**：
```python
# 根据 ID 获取文档
docs = vector_store.get_by_ids(["doc1", "doc2"])
for doc in docs:
    print(doc.page_content)
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

### 2.2 检索方法

#### 基础相似度检索

```python
@abstractmethod
def similarity_search(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any
) -> list[Document]:
    """相似度检索（抽象方法，必须实现）

    Args:
        query: 查询文本
        k: 返回结果数量
        **kwargs: 后端特定参数

    Returns:
        最相似的 Document 列表
    """
    pass
```

**使用示例**：
```python
# 基础检索
results = vector_store.similarity_search(
    query="什么是向量数据库？",
    k=5
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

#### 带分数的检索

```python
def similarity_search_with_score(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[tuple[Document, float]]:
    """带分数的相似度检索

    Args:
        query: 查询文本
        k: 返回结果数量
        **kwargs: 后端特定参数

    Returns:
        (Document, score) 元组列表
    """
    pass
```

**使用示例**：
```python
# 带分数的检索
results = vector_store.similarity_search_with_score(
    query="什么是向量数据库？",
    k=5
)

for doc, score in results:
    print(f"分数: {score:.4f}")
    print(f"内容: {doc.page_content}")
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

#### 带相关性分数的检索

```python
def similarity_search_with_relevance_scores(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[tuple[Document, float]]:
    """带相关性分数的检索

    相关性分数归一化到 [0, 1] 区间，1 表示最相关。

    Args:
        query: 查询文本
        k: 返回结果数量
        **kwargs: 后端特定参数

    Returns:
        (Document, relevance_score) 元组列表
    """
    pass
```

**使用示例**：
```python
# 带相关性分数的检索
results = vector_store.similarity_search_with_relevance_scores(
    query="什么是向量数据库？",
    k=5
)

for doc, relevance in results:
    print(f"相关性: {relevance:.4f}")
    print(f"内容: {doc.page_content}")
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

#### MMR 检索

```python
def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]:
    """最大边际相关性（MMR）检索

    MMR 平衡相关性和多样性：
    - lambda_mult = 1: 只考虑相关性
    - lambda_mult = 0: 只考虑多样性
    - lambda_mult = 0.5: 平衡相关性和多样性

    Args:
        query: 查询文本
        k: 返回结果数量
        fetch_k: 初始检索数量
        lambda_mult: 相关性权重 (0-1)
        **kwargs: 后端特定参数

    Returns:
        Document 列表
    """
    pass
```

**使用示例**：
```python
# MMR 检索（平衡相关性和多样性）
results = vector_store.max_marginal_relevance_search(
    query="什么是向量数据库？",
    k=5,
    fetch_k=20,
    lambda_mult=0.5  # 平衡相关性和多样性
)

for doc in results:
    print(doc.page_content)
```

[来源: reference/source_vectorstore_base_01.md, reference/source_chroma_03.md | LangChain 源码分析]

### 2.3 统一检索接口

```python
def search(
    self,
    query: str,
    search_type: str,
    **kwargs: Any
) -> list[Document]:
    """统一检索接口

    Args:
        query: 查询文本
        search_type: 检索类型
            - 'similarity': 基础相似度检索
            - 'similarity_score_threshold': 带分数阈值的检索
            - 'mmr': MMR 检索
        **kwargs: 检索参数

    Returns:
        Document 列表
    """
    if search_type == "similarity":
        return self.similarity_search(query, **kwargs)
    elif search_type == "similarity_score_threshold":
        docs_and_scores = self.similarity_search_with_relevance_scores(
            query, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]
    elif search_type == "mmr":
        return self.max_marginal_relevance_search(query, **kwargs)
    else:
        raise ValueError(f"不支持的检索类型: {search_type}")
```

**使用示例**：
```python
# 使用统一接口进行不同类型的检索
results = vector_store.search(
    query="什么是向量数据库？",
    search_type="mmr",
    k=5,
    lambda_mult=0.5
)
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

## 3. 异步支持

### 3.1 异步方法

所有核心方法都有对应的异步版本（以 `a` 开头）：

```python
async def aadd_texts(
    self,
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    **kwargs: Any,
) -> list[str]:
    """异步添加文本"""
    return await run_in_executor(None, self.add_texts, texts, metadatas, **kwargs)

async def asimilarity_search(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[Document]:
    """异步相似度检索"""
    return await run_in_executor(None, self.similarity_search, query, k, **kwargs)

async def adelete(
    self,
    ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    """异步删除"""
    return await run_in_executor(None, self.delete, ids, **kwargs)
```

**使用示例**：
```python
import asyncio

async def main():
    # 异步添加文档
    ids = await vector_store.aadd_texts(
        texts=["文档1", "文档2"]
    )

    # 异步检索
    results = await vector_store.asimilarity_search(
        query="查询文本",
        k=5
    )

    # 异步删除
    await vector_store.adelete(ids=["doc1"])

asyncio.run(main())
```

[来源: reference/source_vectorstore_base_01.md, reference/source_qdrant_04.md | LangChain 源码分析]

### 3.2 异步回退机制

Qdrant 实现了优雅的异步回退机制：

```python
def sync_call_fallback(method: Callable) -> Callable:
    """异步方法回退到同步方法的装饰器"""
    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            # 如果异步方法未实现，调用同步方法
            sync_method_name = method.__name__[1:]  # 移除 'a' 前缀
            return await run_in_executor(
                None, getattr(self, sync_method_name), *args, **kwargs
            )
    return wrapper
```

**优势**：
- 子类可以只实现同步方法
- 框架自动提供异步版本
- 如果子类实现了异步方法，则使用原生异步

[来源: reference/source_qdrant_04.md | LangChain 源码分析]

## 4. 相关性分数转换

### 4.1 转换函数

VectorStore 提供三种相关性分数转换函数：

```python
@staticmethod
def _euclidean_relevance_score_fn(distance: float) -> float:
    """欧几里得距离转换为相似度分数

    距离范围: [0, sqrt(2)]
    分数范围: [0, 1]
    """
    return 1.0 - distance / math.sqrt(2)

@staticmethod
def _cosine_relevance_score_fn(distance: float) -> float:
    """余弦距离转换为相似度分数

    距离范围: [0, 2]
    分数范围: [0, 1]
    """
    return 1.0 - distance

@staticmethod
def _max_inner_product_relevance_score_fn(distance: float) -> float:
    """最大内积转换为相似度分数

    距离范围: [-∞, +∞]
    分数范围: [0, 1]
    """
    if distance > 0:
        return 1.0 - distance
    return -1.0 * distance
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

### 4.2 使用场景

| 距离度量 | 转换函数 | 适用场景 |
|---------|---------|---------|
| 欧几里得距离 | `_euclidean_relevance_score_fn` | 未归一化向量 |
| 余弦距离 | `_cosine_relevance_score_fn` | 归一化向量（文本 embedding） |
| 最大内积 | `_max_inner_product_relevance_score_fn` | 归一化向量（性能最快） |

[来源: reference/source_vectorstore_base_01.md, reference/source_qdrant_04.md | LangChain 源码分析]

## 5. 检索类型支持

### 5.1 三种检索类型

```python
# 1. 基础相似度检索
results = vector_store.search(
    query="查询文本",
    search_type="similarity",
    k=5
)

# 2. 带分数阈值的检索
results = vector_store.search(
    query="查询文本",
    search_type="similarity_score_threshold",
    k=5,
    score_threshold=0.8  # 只返回分数 >= 0.8 的结果
)

# 3. MMR 检索（平衡相关性和多样性）
results = vector_store.search(
    query="查询文本",
    search_type="mmr",
    k=5,
    fetch_k=20,
    lambda_mult=0.5
)
```

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

### 5.2 作为 Retriever 使用

```python
# 将 VectorStore 转换为 Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

# 使用 Retriever 检索
results = retriever.invoke("查询文本")
```

[来源: reference/source_inmemory_02.md | LangChain 源码分析]

## 6. 实际应用：实现自定义 VectorStore

### 6.1 最小实现

```python
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import numpy as np

class CustomVectorStore(VectorStore):
    """自定义向量存储实现"""

    def __init__(self, embedding: Embeddings):
        self._embedding = embedding
        self.store: dict[str, dict] = {}

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs,
    ) -> list[str]:
        """添加文本"""
        texts_list = list(texts)
        vectors = self._embedding.embed_documents(texts_list)

        ids = [str(uuid.uuid4()) for _ in texts_list]
        metadatas = metadatas or [{} for _ in texts_list]

        for id_, text, vector, metadata in zip(ids, texts_list, vectors, metadatas):
            self.store[id_] = {
                "text": text,
                "vector": vector,
                "metadata": metadata
            }

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> list[Document]:
        """相似度检索"""
        query_vector = self._embedding.embed_query(query)

        # 计算相似度
        similarities = []
        for id_, item in self.store.items():
            sim = np.dot(query_vector, item["vector"])
            similarities.append((id_, sim))

        # 排序并返回 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        results = []
        for id_, _ in top_k:
            item = self.store[id_]
            results.append(Document(
                page_content=item["text"],
                metadata=item["metadata"]
            ))

        return results

    def delete(self, ids: Sequence[str] | None = None, **kwargs) -> None:
        """删除文档"""
        if ids:
            for id_ in ids:
                self.store.pop(id_, None)
```

**使用示例**：
```python
from langchain_openai import OpenAIEmbeddings

# 初始化自定义向量存储
vector_store = CustomVectorStore(OpenAIEmbeddings())

# 添加文档
ids = vector_store.add_texts(
    texts=["文档1", "文档2"],
    metadatas=[{"source": "file1"}, {"source": "file2"}]
)

# 检索
results = vector_store.similarity_search("查询文本", k=5)

# 删除
vector_store.delete(ids=["doc1"])
```

[来源: reference/source_vectorstore_base_01.md, reference/source_inmemory_02.md | LangChain 源码分析]

### 6.2 实现建议

**必须实现的方法**：
1. `similarity_search()` - 核心检索方法
2. `add_documents()` 或 `add_texts()` - 数据添加方法（二选一）

**可选实现的方法**：
1. `delete()` - 数据删除方法
2. `get_by_ids()` - ID 查询方法
3. `similarity_search_with_score()` - 带分数的检索
4. `max_marginal_relevance_search()` - MMR 检索

**异步方法**：
- 如果不实现异步方法，框架会自动使用 `run_in_executor()` 转换
- 如果后端支持原生异步，建议实现异步方法以获得更好的性能

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

## 7. 架构设计要点

### 7.1 设计模式

1. **抽象基类模式**：使用 ABC 确保子类实现必需方法
2. **模板方法模式**：`add_texts()` 和 `add_documents()` 互相调用
3. **策略模式**：支持多种检索策略（similarity, mmr, threshold）
4. **适配器模式**：统一不同后端的接口

[来源: reference/source_vectorstore_base_01.md | LangChain 源码分析]

### 7.2 扩展性

**后端特定参数**：
```python
# 通过 **kwargs 支持后端特定参数
results = vector_store.similarity_search(
    query="查询文本",
    k=5,
    # Qdrant 特定参数
    distance_strategy="COSINE",
    # Chroma 特定参数
    where={"category": "tech"}
)
```

**类型安全**：
```python
# 使用类型注解提供 IDE 提示
def similarity_search(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[Document]:
    pass
```

[来源: reference/source_vectorstore_base_01.md, reference/source_qdrant_04.md | LangChain 源码分析]

## 8. 总结

### 8.1 核心价值

1. **统一接口**：所有向量存储后端实现相同的接口
2. **易于切换**：更换后端只需修改初始化代码
3. **功能完整**：支持增删改查、检索、过滤等完整功能
4. **异步支持**：原生支持异步操作，适合高并发场景
5. **可扩展性**：通过 `**kwargs` 支持后端特定参数

### 8.2 最佳实践

1. **选择合适的检索类型**：
   - 基础检索：`similarity`
   - 质量控制：`similarity_score_threshold`
   - 多样性：`mmr`

2. **使用异步方法**：
   - 高并发场景使用异步方法
   - 如果后端支持原生异步，优先使用

3. **实现自定义后端**：
   - 至少实现 `similarity_search()` 和 `add_texts()`
   - 可选实现 `delete()` 和 `get_by_ids()`
   - 如果后端支持，实现异步方法

4. **利用统一接口**：
   - 使用 `search()` 方法支持多种检索类型
   - 使用 `as_retriever()` 转换为 Retriever

### 8.3 参考资料

- [LangChain VectorStore 源码](reference/source_vectorstore_base_01.md)
- [InMemoryVectorStore 实现](reference/source_inmemory_02.md)
- [Chroma 集成实现](reference/source_chroma_03.md)
- [Qdrant 集成实现](reference/source_qdrant_04.md)

---

**文档版本**: v1.0
**最后更新**: 2026-02-25
**维护者**: Claude Code
