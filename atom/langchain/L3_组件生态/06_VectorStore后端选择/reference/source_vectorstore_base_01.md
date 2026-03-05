---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/vectorstores/base.py
analyzed_files:
  - sourcecode/langchain/libs/core/langchain_core/vectorstores/base.py
analyzed_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
---

# 源码分析：VectorStore 抽象基类

## 分析的文件

- `sourcecode/langchain/libs/core/langchain_core/vectorstores/base.py` - VectorStore 抽象基类定义

## 关键发现

### 1. VectorStore 是抽象基类（ABC）

VectorStore 是 LangChain 中所有向量存储后端的统一接口，定义了核心的抽象方法和默认实现。

### 2. 核心接口方法

#### 数据操作方法
- `add_texts()` - 添加文本到向量存储
- `add_documents()` - 添加文档到向量存储
- `delete()` - 删除向量
- `get_by_ids()` - 根据 ID 获取文档

#### 检索方法
- `similarity_search()` - 相似度检索（抽象方法，必须实现）
- `similarity_search_with_score()` - 带分数的相似度检索
- `similarity_search_with_relevance_scores()` - 带相关性分数的检索
- `max_marginal_relevance_search()` - MMR 检索
- `search()` - 统一检索接口，支持多种检索类型

#### 异步方法
所有核心方法都有对应的异步版本（以 `a` 开头），如：
- `aadd_texts()`
- `asimilarity_search()`
- `adelete()`

### 3. 相关性分数函数

VectorStore 提供了三种相关性分数转换函数：
- `_euclidean_relevance_score_fn()` - 欧几里得距离转换
- `_cosine_relevance_score_fn()` - 余弦相似度转换
- `_max_inner_product_relevance_score_fn()` - 最大内积转换

### 4. 检索类型支持

通过 `search()` 方法支持三种检索类型：
- `similarity` - 基础相似度检索
- `similarity_score_threshold` - 带分数阈值的检索
- `mmr` - 最大边际相关性检索

### 5. 设计模式

- **模板方法模式**：`add_texts()` 和 `add_documents()` 互相调用，子类只需实现其中一个
- **异步支持**：默认使用 `run_in_executor()` 将同步方法转换为异步方法
- **可选实现**：大部分方法有默认实现或抛出 `NotImplementedError`，子类可选择性实现

## 代码片段

### VectorStore 类定义

```python
class VectorStore(ABC):
    """Interface for vector store."""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: `VectorStore` specific parameters.

        Returns:
            List of IDs from adding the texts into the `VectorStore`.
        """
        # 默认实现：调用 add_documents()
        if type(self).add_documents != VectorStore.add_documents:
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            # ... 转换为 Document 对象
            docs = [
                Document(id=id_, page_content=text, metadata=metadata_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_, strict=False)
            ]
            return self.add_documents(docs, **kwargs)
        msg = f"`add_texts` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
```

### 抽象检索方法

```python
@abstractmethod
def similarity_search(
    self, query: str, k: int = 4, **kwargs: Any
) -> list[Document]:
    """Return docs most similar to query.

    Args:
        query: Input text.
        k: Number of `Document` objects to return.
        **kwargs: Arguments to pass to the search method.

    Returns:
        List of `Document` objects most similar to the query.
    """
```

### 统一检索接口

```python
def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
    """Return docs most similar to query using a specified search type.

    Args:
        query: Input text.
        search_type: Type of search to perform.
            Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.
        **kwargs: Arguments to pass to the search method.

    Returns:
        List of `Document` objects most similar to the query.
    """
    if search_type == "similarity":
        return self.similarity_search(query, **kwargs)
    if search_type == "similarity_score_threshold":
        docs_and_similarities = self.similarity_search_with_relevance_scores(
            query, **kwargs
        )
        return [doc for doc, _ in docs_and_similarities]
    if search_type == "mmr":
        return self.max_marginal_relevance_search(query, **kwargs)
    msg = (
        f"search_type of {search_type} not allowed. Expected "
        "search_type to be 'similarity', 'similarity_score_threshold'"
        " or 'mmr'."
    )
    raise ValueError(msg)
```

### 相关性分数转换

```python
@staticmethod
def _euclidean_relevance_score_fn(distance: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # 欧几里得距离转换为相似度分数
    # 0 是最相似，sqrt(2) 最不相似
    return 1.0 - distance / math.sqrt(2)

@staticmethod
def _cosine_relevance_score_fn(distance: float) -> float:
    """Normalize the distance to a score on a scale [0, 1]."""
    return 1.0 - distance

@staticmethod
def _max_inner_product_relevance_score_fn(distance: float) -> float:
    """Normalize the distance to a score on a scale [0, 1]."""
    if distance > 0:
        return 1.0 - distance
    return -1.0 * distance
```

## 架构设计要点

1. **统一接口**：所有向量存储后端都实现相同的接口，便于切换
2. **灵活实现**：子类可以选择实现 `add_texts()` 或 `add_documents()`，框架会自动适配
3. **异步优先**：提供完整的异步支持，适合高并发场景
4. **可扩展性**：通过 `**kwargs` 支持后端特定的参数
5. **类型安全**：使用类型注解，便于 IDE 提示和静态检查

## 实现建议

对于自定义 VectorStore 后端，至少需要实现：
1. `similarity_search()` - 核心检索方法
2. `add_documents()` 或 `add_texts()` - 数据添加方法
3. `delete()` - 数据删除方法（可选）
4. `get_by_ids()` - ID 查询方法（可选）
