# Retriever与VectorStore集成 - 核心概念：Retriever抽象层

> **BaseRetriever 协议定义**：理解 LangChain 检索器的统一接口设计

---

## 概述

Retriever 抽象层是 LangChain 中所有检索实现的统一接口。通过 BaseRetriever 类，LangChain 定义了标准的检索协议，使得不同的检索实现（VectorStore、BM25、Hybrid、Web Search）可以无缝集成到 LCEL 管道中。

---

## BaseRetriever 类定义

### 继承关系

```python
class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    """Base Retriever class for all retriever implementations."""
```

**关键点**：
- 继承自 `RunnableSerializable`：Retriever 是标准的 Runnable
- 泛型参数：`RetrieverInput = str`, `RetrieverOutput = list[Document]`
- 抽象基类：子类必须实现 `_get_relevant_documents` 方法

**类型定义**：
```python
RetrieverInput = str
RetrieverOutput = list[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]
```

---

## 核心属性

### 1. tags - 标签列表

```python
tags: list[str] | None = None
"""Optional list of tags associated with the retriever."""
```

**用途**：
- 用于 LangSmith 追踪和分类
- 可以标记检索器的类型、版本、环境等
- 自动传递给回调管理器

**示例**：
```python
retriever = vectorstore.as_retriever(
    tags=["production", "v1.0", "tech-docs"]
)
```

---

### 2. metadata - 元数据字典

```python
metadata: dict[str, Any] | None = None
"""Optional metadata associated with the retriever."""
```

**用途**：
- 存储检索器的配置信息
- 用于追踪和监控
- 可以包含任意键值对

**示例**：
```python
retriever = vectorstore.as_retriever(
    metadata={
        "index_name": "tech-docs",
        "embedding_model": "text-embedding-3-small",
        "version": "1.0"
    }
)
```

---

## 核心方法

### 1. invoke() - 同步调用

```python
def invoke(
    self, input: str, config: RunnableConfig | None = None, **kwargs: Any
) -> list[Document]:
    """Invoke the retriever to get relevant documents.

    Main entry point for synchronous retriever invocations.

    Args:
        input: Query string.
        config: Optional configuration for the invocation.
        **kwargs: Additional keyword arguments.

    Returns:
        List of relevant documents.
    """
```

**执行流程**：

```python
def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
    # 1. 配置初始化
    config = ensure_config(config)

    # 2. 准备 LangSmith 追踪元数据
    inheritable_metadata = {
        **(config.get("metadata") or {}),
        **self._get_ls_params(**kwargs),
    }

    # 3. 创建回调管理器
    callback_manager = CallbackManager.configure(
        config.get("callbacks"),
        None,
        verbose=kwargs.get("verbose", False),
        inheritable_tags=config.get("tags"),
        local_tags=self.tags,
        inheritable_metadata=inheritable_metadata,
        local_metadata=self.metadata,
    )

    # 4. 触发检索开始事件
    run_manager = callback_manager.on_retriever_start(
        None,
        input,
        name=config.get("run_name") or self.get_name(),
        run_id=kwargs.pop("run_id", None),
    )

    try:
        # 5. 执行实际检索（调用子类实现）
        kwargs_ = kwargs if self._expects_other_args else {}
        if self._new_arg_supported:
            result = self._get_relevant_documents(
                input, run_manager=run_manager, **kwargs_
            )
        else:
            result = self._get_relevant_documents(input, **kwargs_)
    except Exception as e:
        # 6. 触发错误事件
        run_manager.on_retriever_error(e)
        raise
    else:
        # 7. 触发检索结束事件
        run_manager.on_retriever_end(result)
        return result
```

**关键点**：
- 模板方法模式：定义固定的执行流程
- 回调机制：在关键节点触发事件
- 错误处理：捕获异常并触发错误事件
- LangSmith 集成：自动追踪检索过程

---

### 2. ainvoke() - 异步调用

```python
async def ainvoke(
    self,
    input: str,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> list[Document]:
    """Asynchronously invoke the retriever to get relevant documents.

    Args:
        input: Query string.
        config: Optional configuration for the invocation.
        **kwargs: Additional keyword arguments.

    Returns:
        List of relevant documents.
    """
```

**执行流程**：
- 与 `invoke` 类似，但使用 `AsyncCallbackManager`
- 调用 `_aget_relevant_documents` 而非 `_get_relevant_documents`
- 支持异步上下文管理

**示例**：
```python
# 异步调用
docs = await retriever.ainvoke("What is LangChain?")

# 批量异步调用
tasks = [retriever.ainvoke(query) for query in queries]
results = await asyncio.gather(*tasks)
```

---

### 3. _get_relevant_documents() - 抽象方法

```python
@abstractmethod
def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
) -> list[Document]:
    """Get documents relevant to a query.

    Args:
        query: String to find relevant documents for.
        run_manager: The callback handler to use.

    Returns:
        List of relevant documents.
    """
```

**关键点**：
- 抽象方法：子类必须实现
- 接收查询字符串和 run_manager
- 返回相关文档列表
- 这是检索器的核心逻辑所在

**子类实现示例**：
```python
class VectorStoreRetriever(BaseRetriever):
    vectorstore: VectorStore
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        kwargs_ = self.search_kwargs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **kwargs_)
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **kwargs_)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
```

---

### 4. _aget_relevant_documents() - 异步抽象方法

```python
async def _aget_relevant_documents(
    self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
) -> list[Document]:
    """Asynchronously get documents relevant to a query.

    Args:
        query: String to find relevant documents for.
        run_manager: The async callback handler to use.

    Returns:
        List of relevant documents.
    """
    return await run_in_executor(
        None,
        self._get_relevant_documents,
        query,
        run_manager=run_manager.get_sync(),
    )
```

**默认实现**：
- 在 executor 中运行同步版本
- 子类可以覆盖以提供原生异步实现

**原生异步实现示例**：
```python
class VectorStoreRetriever(BaseRetriever):
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        kwargs_ = self.search_kwargs
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(query, **kwargs_)
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **kwargs_
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
```

---

### 5. batch() - 批量调用

```python
def batch(
    self,
    inputs: list[str],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    **kwargs: Any,
) -> list[list[Document]]:
    """Batch invoke the retriever.

    Args:
        inputs: List of query strings.
        config: Optional configuration(s) for the invocations.
        **kwargs: Additional keyword arguments.

    Returns:
        List of lists of relevant documents.
    """
```

**继承自 Runnable**：
- 自动支持批量处理
- 可以配置并发数
- 支持批量配置

**示例**：
```python
# 批量检索
queries = ["What is LangChain?", "How to use LCEL?", "What is RAG?"]
batch_results = retriever.batch(queries)

# 配置并发数
batch_results = retriever.batch(
    queries,
    config={"max_concurrency": 5}
)
```

---

### 6. abatch() - 异步批量调用

```python
async def abatch(
    self,
    inputs: list[str],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    **kwargs: Any,
) -> list[list[Document]]:
    """Async batch invoke the retriever."""
```

**示例**：
```python
# 异步批量检索
queries = ["What is LangChain?", "How to use LCEL?", "What is RAG?"]
batch_results = await retriever.abatch(queries)
```

---

## LangSmith 集成

### _get_ls_params() - 获取追踪参数

```python
def _get_ls_params(self, **_kwargs: Any) -> LangSmithRetrieverParams:
    """Get standard params for tracing.

    Returns:
        Dictionary with LangSmith parameters.
    """
    default_retriever_name = self.get_name()
    if default_retriever_name.startswith("Retriever"):
        default_retriever_name = default_retriever_name[9:]
    elif default_retriever_name.endswith("Retriever"):
        default_retriever_name = default_retriever_name[:-9]
    default_retriever_name = default_retriever_name.lower()

    return LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)
```

**LangSmithRetrieverParams 定义**：
```python
class LangSmithRetrieverParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""
    ls_retriever_name: str
    """Retriever name."""
    ls_vector_store_provider: str | None
    """Vector store provider."""
    ls_embedding_provider: str | None
    """Embedding provider."""
    ls_embedding_model: str | None
    """Embedding model."""
```

**VectorStoreRetriever 的扩展**：
```python
class VectorStoreRetriever(BaseRetriever):
    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""
        kwargs_ = self.search_kwargs | kwargs

        ls_params = super()._get_ls_params(**kwargs_)

        # 添加 VectorStore 提供商信息
        ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

        # 添加 Embedding 提供商信息
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

**追踪信息示例**：
```json
{
  "ls_retriever_name": "vectorstore",
  "ls_vector_store_provider": "InMemoryVectorStore",
  "ls_embedding_provider": "OpenAIEmbeddings"
}
```

---

## 回调机制

### 回调事件

BaseRetriever 在执行过程中触发以下事件：

1. **on_retriever_start**：检索开始
```python
run_manager = callback_manager.on_retriever_start(
    serialized=None,
    query=input,
    name=config.get("run_name") or self.get_name(),
    run_id=kwargs.pop("run_id", None),
)
```

2. **on_retriever_end**：检索成功结束
```python
run_manager.on_retriever_end(result)
```

3. **on_retriever_error**：检索失败
```python
run_manager.on_retriever_error(e)
```

### 自定义回调示例

```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomRetrieverCallback(BaseCallbackHandler):
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"[START] Retrieving documents for: {query}")

    def on_retriever_end(self, documents, **kwargs):
        print(f"[END] Retrieved {len(documents)} documents")

    def on_retriever_error(self, error, **kwargs):
        print(f"[ERROR] Retrieval failed: {error}")

# 使用自定义回调
retriever = vectorstore.as_retriever()
docs = retriever.invoke(
    "What is LangChain?",
    config={"callbacks": [CustomRetrieverCallback()]}
)
```

---

## Runnable 集成

### 为什么 Retriever 要实现 Runnable？

**核心原因**：LCEL 管道需要统一的组件接口

**Runnable 接口**：
```python
class Runnable[Input, Output]:
    def invoke(self, input: Input) -> Output:
        """Invoke the runnable."""
        pass

    async def ainvoke(self, input: Input) -> Output:
        """Async invoke the runnable."""
        pass

    def batch(self, inputs: list[Input]) -> list[Output]:
        """Batch invoke the runnable."""
        pass

    async def abatch(self, inputs: list[Input]) -> list[Output]:
        """Async batch invoke the runnable."""
        pass

    def __or__(self, other: Runnable) -> RunnableSequence:
        """Support | operator for chaining."""
        return RunnableSequence(self, other)
```

**BaseRetriever 作为 Runnable**：
```python
class BaseRetriever(RunnableSerializable[str, list[Document]]):
    """Retriever is Runnable[str, list[Document]]"""
    pass
```

### LCEL 管道集成

**基础用法**：
```python
# Retriever 可以直接用于 LCEL 管道
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

上下文：
{context}

问题：{question}

回答：
""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 构建 RAG 管道
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI(model="gpt-4")
)

# 调用管道
answer = rag_chain.invoke("What is LangChain?")
```

**高级用法**：
```python
# 多个 Retriever 并行
from langchain_core.runnables import RunnableParallel

retriever1 = vectorstore1.as_retriever()
retriever2 = vectorstore2.as_retriever()

parallel_retrieval = RunnableParallel(
    docs1=retriever1,
    docs2=retriever2
)

results = parallel_retrieval.invoke("query")
# results = {"docs1": [...], "docs2": [...]}
```

---

## 向后兼容机制

### 签名检查

BaseRetriever 通过 `__init_subclass__` 检查子类的 `_get_relevant_documents` 签名，确保向后兼容：

```python
_new_arg_supported: bool = False
_expects_other_args: bool = False

@classmethod
def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)

    # 检查 _get_relevant_documents 签名
    sig = inspect.signature(cls._get_relevant_documents)

    # 检查是否支持 run_manager 参数
    if "run_manager" in sig.parameters:
        cls._new_arg_supported = True

    # 检查是否接受其他参数
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ):
        cls._expects_other_args = True
```

**兼容性处理**：
```python
def invoke(self, input: str, **kwargs: Any) -> list[Document]:
    # ...
    kwargs_ = kwargs if self._expects_other_args else {}
    if self._new_arg_supported:
        result = self._get_relevant_documents(
            input, run_manager=run_manager, **kwargs_
        )
    else:
        # 旧版本 Retriever 不接受 run_manager
        result = self._get_relevant_documents(input, **kwargs_)
    # ...
```

---

## 实现自定义 Retriever

### 简单示例

```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class SimpleRetriever(BaseRetriever):
    """Simple retriever that returns the first k documents."""

    docs: list[Document]
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return the first k documents from the list."""
        return self.docs[:self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Async version."""
        return self.docs[:self.k]

# 使用
docs = [
    Document(page_content="Doc 1"),
    Document(page_content="Doc 2"),
    Document(page_content="Doc 3"),
]
retriever = SimpleRetriever(docs=docs, k=2)
results = retriever.invoke("query")
```

### TF-IDF Retriever 示例

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

class TFIDFRetriever(BaseRetriever, BaseModel):
    """TF-IDF based retriever."""

    vectorizer: Any
    docs: list[Document]
    tfidf_array: Any
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Retrieve documents using TF-IDF similarity."""
        # 向量化查询
        query_vec = self.vectorizer.transform([query])

        # 计算相似度
        results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))

        # 返回 top-k
        top_k_indices = results.argsort()[-self.k:][::-1]
        return [self.docs[i] for i in top_k_indices]

# 创建和使用
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    Document(page_content="LangChain is a framework for LLM applications"),
    Document(page_content="LCEL is LangChain Expression Language"),
    Document(page_content="RAG is Retrieval-Augmented Generation"),
]

texts = [doc.page_content for doc in docs]
vectorizer = TfidfVectorizer()
tfidf_array = vectorizer.fit_transform(texts)

retriever = TFIDFRetriever(
    vectorizer=vectorizer,
    docs=docs,
    tfidf_array=tfidf_array,
    k=2
)

results = retriever.invoke("What is LangChain?")
```

---

## 核心设计模式

### 1. 模板方法模式

BaseRetriever 的 `invoke` 方法定义了固定的执行流程：

```
1. 配置初始化
2. 准备追踪元数据
3. 创建回调管理器
4. 触发开始事件
5. 执行检索（子类实现）
6. 触发结束/错误事件
7. 返回结果
```

子类只需实现 `_get_relevant_documents` 方法。

---

### 2. 策略模式

不同的 Retriever 实现不同的检索策略：
- VectorStoreRetriever：向量检索
- BM25Retriever：词法检索
- HybridRetriever：混合检索
- WebSearchRetriever：网络搜索

所有策略都实现相同的接口：`str → list[Document]`

---

### 3. 适配器模式

VectorStoreRetriever 是适配器，将 VectorStore 适配到 BaseRetriever 接口：

```
VectorStore (多种方法) → VectorStoreRetriever (适配器) → BaseRetriever (统一接口)
```

---

## 关键洞察

1. **统一接口**：BaseRetriever 定义了 `str → list[Document]` 的标准接口
2. **Runnable 集成**：Retriever 是标准 Runnable，可在 LCEL 中使用
3. **回调机制**：通过回调实现追踪和监控
4. **LangSmith 集成**：自动追踪检索过程
5. **异步支持**：默认提供异步实现，子类可覆盖
6. **向后兼容**：通过签名检查确保旧版本 Retriever 仍可用
7. **模板方法**：定义固定流程，子类实现核心逻辑
8. **可扩展性**：易于实现自定义 Retriever

---

## 最佳实践

### 1. 实现自定义 Retriever

```python
class CustomRetriever(BaseRetriever):
    # 1. 定义属性
    config: dict

    # 2. 实现同步检索
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # 实现检索逻辑
        pass

    # 3. 实现异步检索（可选）
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # 实现异步检索逻辑
        pass
```

---

### 2. 使用标签和元数据

```python
retriever = vectorstore.as_retriever(
    tags=["production", "v1.0"],
    metadata={
        "index_name": "tech-docs",
        "embedding_model": "text-embedding-3-small"
    }
)
```

---

### 3. 配置回调

```python
from langchain_core.callbacks import StdOutCallbackHandler

retriever = vectorstore.as_retriever()
docs = retriever.invoke(
    "query",
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

---

### 4. 批量处理

```python
# 批量检索
queries = ["query1", "query2", "query3"]
batch_results = retriever.batch(queries, config={"max_concurrency": 5})
```

---

## 总结

BaseRetriever 是 LangChain 检索器的核心抽象层，它：

1. **定义统一接口**：`str → list[Document]`
2. **实现 Runnable**：可在 LCEL 管道中使用
3. **提供回调机制**：支持追踪和监控
4. **集成 LangSmith**：自动追踪检索过程
5. **支持异步**：默认异步实现，可覆盖
6. **向后兼容**：确保旧版本 Retriever 可用
7. **易于扩展**：简单实现自定义 Retriever

通过理解 BaseRetriever 的设计，我们可以：
- 正确使用 VectorStoreRetriever
- 实现自定义检索器
- 优化检索性能
- 集成到 LCEL 管道

---

**版本**：v1.0
**最后更新**：2026-02-24
**数据来源**：LangChain 源码分析 (`langchain_core/retrievers.py`)
