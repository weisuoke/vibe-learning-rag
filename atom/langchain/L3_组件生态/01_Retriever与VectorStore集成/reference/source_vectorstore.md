# VectorStore 与 VectorStoreRetriever 源码分析

**文件路径**: `sourcecode/langchain/libs/core/langchain_core/vectorstores/base.py`

## VectorStore 基类

### 类定义

```python
class VectorStore(ABC):
    """Interface for vector store."""
```

### 核心方法

#### 1. 文档操作

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Run more texts through the embeddings and add to the VectorStore."""
```

```python
def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
    """Add or update documents in the VectorStore."""
```

```python
def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
    """Delete by vector ID or other criteria."""
```

```python
def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
    """Get documents by their IDs."""
```

#### 2. 检索方法

##### 相似度搜索（Similarity Search）

```python
@abstractmethod
def similarity_search(
    self, query: str, k: int = 4, **kwargs: Any
) -> list[Document]:
    """Return docs most similar to query.

    Args:
        query: Input text.
        k: Number of Document objects to return.
        **kwargs: Arguments to pass to the search method.

    Returns:
        List of Document objects most similar to the query.
    """
```

```python
def similarity_search_with_score(
    self, *args: Any, **kwargs: Any
) -> list[tuple[Document, float]]:
    """Run similarity search with distance.

    Returns:
        List of tuples of (doc, similarity_score).
    """
```

##### 相似度阈值搜索（Similarity Score Threshold）

```python
def similarity_search_with_relevance_scores(
    self,
    query: str,
    k: int = 4,
    **kwargs: Any,
) -> list[tuple[Document, float]]:
    """Return docs and relevance scores in the range [0, 1].

    0 is dissimilar, 1 is most similar.

    Args:
        query: Input text.
        k: Number of Document objects to return.
        **kwargs: Kwargs to be passed to similarity search.
            Should include score_threshold, an optional floating point value
            between 0 to 1 to filter the resulting set of retrieved docs.

    Returns:
        List of tuples of (doc, similarity_score).
    """
    score_threshold = kwargs.pop("score_threshold", None)

    docs_and_similarities = self._similarity_search_with_relevance_scores(
        query, k=k, **kwargs
    )

    if score_threshold is not None:
        docs_and_similarities = [
            (doc, similarity)
            for doc, similarity in docs_and_similarities
            if similarity >= score_threshold
        ]
        if len(docs_and_similarities) == 0:
            logger.warning(
                "No relevant docs were retrieved using the "
                "relevance score threshold %s",
                score_threshold,
            )
    return docs_and_similarities
```

##### MMR 搜索（Maximal Marginal Relevance）

```python
def max_marginal_relevance_search(
    self,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]:
    """Return docs selected using the maximal marginal relevance.

    Maximal marginal relevance optimizes for similarity to query AND diversity
    among selected documents.

    Args:
        query: Text to look up documents similar to.
        k: Number of Document objects to return.
        fetch_k: Number of Document objects to fetch to pass to MMR algorithm.
        lambda_mult: Number between 0 and 1 that determines the degree of
            diversity among the results with 0 corresponding to maximum diversity
            and 1 to minimum diversity.
        **kwargs: Arguments to pass to the search method.

    Returns:
        List of Document objects selected by maximal marginal relevance.
    """
```

#### 3. 统一搜索接口

```python
def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
    """Return docs most similar to query using a specified search type.

    Args:
        query: Input text.
        search_type: Type of search to perform.
            Can be 'similarity', 'mmr', or 'similarity_score_threshold'.
        **kwargs: Arguments to pass to the search method.

    Returns:
        List of Document objects most similar to the query.

    Raises:
        ValueError: If search_type is not one of 'similarity',
            'mmr', or 'similarity_score_threshold'.
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

#### 4. as_retriever() 方法

```python
def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
    """Return VectorStoreRetriever initialized from this VectorStore.

    Args:
        **kwargs: Keyword arguments to pass to the search function.
            Can include:
            * search_type: Defines the type of search that the Retriever should
                perform. Can be 'similarity' (default), 'mmr', or
                'similarity_score_threshold'.
            * search_kwargs: Keyword arguments to pass to the search function.
                Can include things like:
                * k: Amount of documents to return (Default: 4)
                * score_threshold: Minimum relevance threshold
                    for similarity_score_threshold
                * fetch_k: Amount of documents to pass to MMR algorithm
                    (Default: 20)
                * lambda_mult: Diversity of results returned by MMR;
                    1 for minimum diversity and 0 for maximum. (Default: 0.5)
                * filter: Filter by document metadata

    Returns:
        Retriever class for VectorStore.

    Examples:
        # Retrieve more documents with higher diversity
        docsearch.as_retriever(
            search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
        )

        # Fetch more documents for the MMR algorithm to consider
        docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})

        # Only retrieve documents that have a relevance score above a threshold
        docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8},
        )

        # Only get the single most similar document
        docsearch.as_retriever(search_kwargs={"k": 1})

        # Use a filter to only retrieve documents from a specific paper
        docsearch.as_retriever(
            search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
        )
    """
    tags = kwargs.pop("tags", None) or [*self._get_retriever_tags()]
    return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)
```

#### 5. 辅助方法

```python
def _get_retriever_tags(self) -> list[str]:
    """Get tags for retriever."""
    tags = [self.__class__.__name__]
    if self.embeddings:
        tags.append(self.embeddings.__class__.__name__)
    return tags
```

```python
@property
def embeddings(self) -> Embeddings | None:
    """Access the query embedding object if available."""
    return None
```

### 相似度分数转换函数

```python
@staticmethod
def _euclidean_relevance_score_fn(distance: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
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

## VectorStoreRetriever 类

### 类定义

```python
class VectorStoreRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    vectorstore: VectorStore
    """VectorStore to use for retrieval."""

    search_type: str = "similarity"
    """Type of search to perform."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )
```

### 验证器

```python
@model_validator(mode="before")
@classmethod
def validate_search_type(cls, values: dict) -> Any:
    """Validate search type.

    Raises:
        ValueError: If search_type is not one of the allowed search types.
        ValueError: If score_threshold is not specified with a float value(0~1)
    """
    search_type = values.get("search_type", "similarity")
    if search_type not in cls.allowed_search_types:
        msg = (
            f"search_type of {search_type} not allowed. Valid values are: "
            f"{cls.allowed_search_types}"
        )
        raise ValueError(msg)
    if search_type == "similarity_score_threshold":
        score_threshold = values.get("search_kwargs", {}).get("score_threshold")
        if (score_threshold is None) or (not isinstance(score_threshold, float)):
            msg = (
                "`score_threshold` is not specified with a float value(0~1) "
                "in `search_kwargs`."
            )
            raise ValueError(msg)
    return values
```

### LangSmith 集成

```python
def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
    """Get standard params for tracing."""
    kwargs_ = self.search_kwargs | kwargs

    ls_params = super()._get_ls_params(**kwargs_)

    ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

    if self.vectorstore.embeddings:
        ls_params["ls_embedding_provider"] = (
            self.vectorstore.embeddings.__class__.__name__
        )
    elif hasattr(self.vectorstore, "embedding") and isinstance(
        self.vectorstore.embedding, Embeddings
    ):
        ls_params["ls_embedding_provider"] = (
            self.vectorstore.embedding.__class__.__name__
        )

    return ls_params
```

### 核心检索方法

#### 同步检索

```python
@override
def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
) -> list[Document]:
    kwargs_ = self.search_kwargs | kwargs
    if self.search_type == "similarity":
        docs = self.vectorstore.similarity_search(query, **kwargs_)
    elif self.search_type == "similarity_score_threshold":
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **kwargs_
            )
        )
        docs = [doc for doc, _ in docs_and_similarities]
    elif self.search_type == "mmr":
        docs = self.vectorstore.max_marginal_relevance_search(query, **kwargs_)
    else:
        msg = f"search_type of {self.search_type} not allowed."
        raise ValueError(msg)
    return docs
```

#### 异步检索

```python
@override
async def _aget_relevant_documents(
    self,
    query: str,
    *,
    run_manager: AsyncCallbackManagerForRetrieverRun,
    **kwargs: Any,
) -> list[Document]:
    kwargs_ = self.search_kwargs | kwargs
    if self.search_type == "similarity":
        docs = await self.vectorstore.asimilarity_search(query, **kwargs_)
    elif self.search_type == "similarity_score_threshold":
        docs_and_similarities = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                query, **kwargs_
            )
        )
        docs = [doc for doc, _ in docs_and_similarities]
    elif self.search_type == "mmr":
        docs = await self.vectorstore.amax_marginal_relevance_search(
            query, **kwargs_
        )
    else:
        msg = f"search_type of {self.search_type} not allowed."
        raise ValueError(msg)
    return docs
```

### 文档操作方法

```python
def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
    """Add documents to the VectorStore."""
    return self.vectorstore.add_documents(documents, **kwargs)

async def aadd_documents(
    self, documents: list[Document], **kwargs: Any
) -> list[str]:
    """Async add documents to the VectorStore."""
    return await self.vectorstore.aadd_documents(documents, **kwargs)
```

## 关键设计模式

### 1. 适配器模式

VectorStoreRetriever 是一个适配器，将 VectorStore 的检索方法适配到 BaseRetriever 接口：

```
VectorStore (多种检索方法) → VectorStoreRetriever → BaseRetriever (统一接口)
```

### 2. 策略模式

通过 `search_type` 参数选择不同的检索策略：
- `"similarity"`: 基础相似度搜索
- `"similarity_score_threshold"`: 阈值过滤搜索
- `"mmr"`: 最大边际相关性搜索

### 3. 参数传递机制

```python
# 参数合并：search_kwargs + 运行时 kwargs
kwargs_ = self.search_kwargs | kwargs

# 调用 VectorStore 方法
docs = self.vectorstore.similarity_search(query, **kwargs_)
```

### 4. 标签管理

```python
# 自动生成标签
tags = [VectorStore类名, Embeddings类名]

# 传递给 Retriever
retriever = VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)
```

## 三种检索策略详解

### 1. Similarity Search

**原理**: 基础向量相似度计算

**参数**:
- `k`: 返回文档数量（默认 4）
- `filter`: 元数据过滤条件

**实现**:
```python
docs = self.vectorstore.similarity_search(query, **kwargs_)
```

### 2. Similarity Score Threshold

**原理**: 只返回相似度分数高于阈值的文档

**参数**:
- `k`: 最大返回文档数量
- `score_threshold`: 相似度阈值（0-1 之间的浮点数）

**实现**:
```python
docs_and_similarities = (
    self.vectorstore.similarity_search_with_relevance_scores(
        query, **kwargs_
    )
)
# 过滤低于阈值的文档
docs = [doc for doc, score in docs_and_similarities if score >= score_threshold]
```

**验证**:
```python
if search_type == "similarity_score_threshold":
    score_threshold = values.get("search_kwargs", {}).get("score_threshold")
    if (score_threshold is None) or (not isinstance(score_threshold, float)):
        raise ValueError("score_threshold must be a float value(0~1)")
```

### 3. MMR (Maximal Marginal Relevance)

**原理**: 平衡相关性和多样性

**参数**:
- `k`: 最终返回文档数量（默认 4）
- `fetch_k`: 候选文档数量（默认 20）
- `lambda_mult`: 多样性参数（0-1 之间）
  - 0: 最大多样性
  - 1: 最小多样性（最大相关性）
  - 0.5: 平衡（默认）

**实现**:
```python
docs = self.vectorstore.max_marginal_relevance_search(query, **kwargs_)
```

**算法流程**:
1. 获取 `fetch_k` 个候选文档
2. 使用 MMR 算法从候选中选择 `k` 个文档
3. 确保选中的文档既相关又多样

## 关键洞察

1. **统一接口**: VectorStore 提供 `search()` 方法统一三种检索策略
2. **灵活转换**: `as_retriever()` 方法轻松将 VectorStore 转换为 Retriever
3. **参数验证**: VectorStoreRetriever 在初始化时验证参数有效性
4. **LangSmith 集成**: 自动追踪 VectorStore 和 Embeddings 提供商信息
5. **异步支持**: 所有检索方法都有对应的异步版本
6. **标签管理**: 自动生成和传递标签用于追踪和监控

## 使用示例

### 基础用法

```python
# 创建 VectorStore
vectorstore = InMemoryVectorStore(embedding=embeddings)

# 转换为 Retriever（默认 similarity 搜索）
retriever = vectorstore.as_retriever()

# 检索文档
docs = retriever.invoke("query")
```

### MMR 搜索

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.25}  # 高多样性
)
```

### 阈值过滤

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)
```

### 元数据过滤

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
)
```
