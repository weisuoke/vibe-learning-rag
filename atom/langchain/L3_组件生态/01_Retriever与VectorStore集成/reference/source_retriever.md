# BaseRetriever 源码分析

**文件路径**: `sourcecode/langchain/libs/core/langchain_core/retrievers.py`

## 核心定义

### 类型定义

```python
RetrieverInput = str
RetrieverOutput = list[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]
```

### LangSmith 追踪参数

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

## BaseRetriever 类

### 继承关系

```python
class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
```

- 继承自 `RunnableSerializable`，这意味着 Retriever 是一个标准的 Runnable
- 输入类型：`str` (查询字符串)
- 输出类型：`list[Document]` (文档列表)

### 核心属性

```python
tags: list[str] | None = None
"""Optional list of tags associated with the retriever."""

metadata: dict[str, Any] | None = None
"""Optional metadata associated with the retriever."""
```

### 核心方法

#### 1. invoke() - 同步调用

```python
def invoke(
    self, input: str, config: RunnableConfig | None = None, **kwargs: Any
) -> list[Document]:
    """Invoke the retriever to get relevant documents.

    Main entry point for synchronous retriever invocations.
    """
    config = ensure_config(config)
    inheritable_metadata = {
        **(config.get("metadata") or {}),
        **self._get_ls_params(**kwargs),
    }
    callback_manager = CallbackManager.configure(
        config.get("callbacks"),
        None,
        verbose=kwargs.get("verbose", False),
        inheritable_tags=config.get("tags"),
        local_tags=self.tags,
        inheritable_metadata=inheritable_metadata,
        local_metadata=self.metadata,
    )
    run_manager = callback_manager.on_retriever_start(
        None,
        input,
        name=config.get("run_name") or self.get_name(),
        run_id=kwargs.pop("run_id", None),
    )
    try:
        kwargs_ = kwargs if self._expects_other_args else {}
        if self._new_arg_supported:
            result = self._get_relevant_documents(
                input, run_manager=run_manager, **kwargs_
            )
        else:
            result = self._get_relevant_documents(input, **kwargs_)
    except Exception as e:
        run_manager.on_retriever_error(e)
        raise
    else:
        run_manager.on_retriever_end(result)
        return result
```

**关键点**：
- 配置 LangSmith 追踪元数据
- 创建 CallbackManager 管理回调
- 调用 `_get_relevant_documents` 实现实际检索
- 错误处理和追踪

#### 2. ainvoke() - 异步调用

```python
async def ainvoke(
    self,
    input: str,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> list[Document]:
    """Asynchronously invoke the retriever to get relevant documents."""
    # 类似 invoke，但使用 AsyncCallbackManager
    # 调用 _aget_relevant_documents
```

#### 3. _get_relevant_documents() - 抽象方法

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
- 这是子类必须实现的核心方法
- 接收查询字符串和 run_manager
- 返回相关文档列表

#### 4. _aget_relevant_documents() - 异步版本

```python
async def _aget_relevant_documents(
    self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
) -> list[Document]:
    """Asynchronously get documents relevant to a query."""
    return await run_in_executor(
        None,
        self._get_relevant_documents,
        query,
        run_manager=run_manager.get_sync(),
    )
```

**关键点**：
- 默认实现：在 executor 中运行同步版本
- 子类可以覆盖以提供原生异步实现

### LangSmith 集成

```python
def _get_ls_params(self, **_kwargs: Any) -> LangSmithRetrieverParams:
    """Get standard params for tracing."""
    default_retriever_name = self.get_name()
    if default_retriever_name.startswith("Retriever"):
        default_retriever_name = default_retriever_name[9:]
    elif default_retriever_name.endswith("Retriever"):
        default_retriever_name = default_retriever_name[:-9]
    default_retriever_name = default_retriever_name.lower()

    return LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)
```

## 设计模式

### 1. Runnable 集成

BaseRetriever 继承自 `RunnableSerializable`，这意味着：
- 可以使用标准的 Runnable 方法：`invoke`, `ainvoke`, `batch`, `abatch`, `stream`, `astream`
- 可以在 LCEL 管道中使用
- 支持配置和回调

### 2. 回调机制

```python
callback_manager.on_retriever_start(...)  # 开始检索
run_manager.on_retriever_end(result)      # 检索成功
run_manager.on_retriever_error(e)         # 检索失败
```

### 3. 向后兼容

```python
_new_arg_supported: bool = False
_expects_other_args: bool = False
```

通过 `__init_subclass__` 检查子类的 `_get_relevant_documents` 签名，确保向后兼容。

## 使用示例

### 简单 Retriever

```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class SimpleRetriever(BaseRetriever):
    docs: list[Document]
    k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Return the first k documents from the list of documents"""
        return self.docs[:self.k]

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """(Optional) async native implementation."""
        return self.docs[:self.k]
```

### TF-IDF Retriever

```python
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever(BaseRetriever, BaseModel):
    vectorizer: Any
    docs: list[Document]
    tfidf_array: Any
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_vec = self.vectorizer.transform([query])
        results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
        return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
```

## 关键洞察

1. **Runnable 优先**：Retriever 是标准的 Runnable，可以无缝集成到 LCEL 管道中
2. **回调驱动**：通过 CallbackManager 实现追踪和监控
3. **LangSmith 集成**：内置 LangSmith 追踪支持
4. **异步支持**：默认提供异步实现，子类可以覆盖以提供原生异步
5. **向后兼容**：通过签名检查确保旧版本 Retriever 仍然可用

## 与 VectorStore 的关系

BaseRetriever 是抽象层，VectorStoreRetriever 是具体实现：
- VectorStoreRetriever 继承 BaseRetriever
- 通过 `VectorStore.as_retriever()` 创建
- 将 VectorStore 的检索方法适配到 Retriever 接口
