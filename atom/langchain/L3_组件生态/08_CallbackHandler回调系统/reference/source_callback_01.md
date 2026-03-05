---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/callbacks/base.py
  - libs/core/langchain_core/callbacks/manager.py
  - libs/core/langchain_core/callbacks/__init__.py
  - libs/core/langchain_core/callbacks/stdout.py
  - libs/core/langchain_core/callbacks/streaming_stdout.py
analyzed_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
---

# 源码分析：CallbackHandler 核心架构

## 分析的文件

- `libs/core/langchain_core/callbacks/base.py` - 回调接口定义和 Mixin 架构
- `libs/core/langchain_core/callbacks/manager.py` - 回调管理器实现
- `libs/core/langchain_core/callbacks/__init__.py` - 公共 API 导出
- `libs/core/langchain_core/callbacks/stdout.py` - 标准输出回调实现
- `libs/core/langchain_core/callbacks/streaming_stdout.py` - 流式输出回调实现

## 关键发现

### 1. Mixin 架构设计

LangChain 的 CallbackHandler 采用了 **Mixin 模式**，将不同组件的回调方法分散到不同的 Mixin 类中：

- **RetrieverManagerMixin**: Retriever 相关回调
  - `on_retriever_error()` - Retriever 错误时触发
  - `on_retriever_end()` - Retriever 结束时触发

- **LLMManagerMixin**: LLM 相关回调
  - `on_llm_new_token()` - 新 token 生成时触发（流式）
  - `on_llm_end()` - LLM 完成时触发
  - `on_llm_error()` - LLM 错误时触发

- **ChainManagerMixin**: Chain 相关回调
  - `on_chain_end()` - Chain 结束时触发
  - `on_chain_error()` - Chain 错误时触发
  - `on_agent_action()` - Agent 执行动作时触发
  - `on_agent_finish()` - Agent 完成时触发

- **ToolManagerMixin**: Tool 相关回调
  - `on_tool_end()` - Tool 结束时触发
  - `on_tool_error()` - Tool 错误时触发

- **CallbackManagerMixin**: 启动相关回调
  - `on_llm_start()` - LLM 开始时触发
  - `on_chat_model_start()` - Chat Model 开始时触发
  - `on_retriever_start()` - Retriever 开始时触发
  - `on_chain_start()` - Chain 开始时触发
  - `on_tool_start()` - Tool 开始时触发

- **RunManagerMixin**: 运行时相关回调
  - `on_text()` - 任意文本输出时触发
  - `on_retry()` - 重试时触发
  - `on_custom_event()` - 自定义事件时触发

### 2. BaseCallbackHandler 设计

`BaseCallbackHandler` 继承所有 Mixin，提供完整的回调接口：

```python
class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler."""

    raise_error: bool = False  # 是否抛出异常
    run_inline: bool = False   # 是否内联运行

    # 忽略特定类型回调的属性
    @property
    def ignore_llm(self) -> bool: ...
    @property
    def ignore_chain(self) -> bool: ...
    @property
    def ignore_agent(self) -> bool: ...
    @property
    def ignore_retriever(self) -> bool: ...
    @property
    def ignore_chat_model(self) -> bool: ...
    @property
    def ignore_custom_event(self) -> bool: ...
```

### 3. 回调管理器架构

从 `manager.py` 中发现了回调管理器的层次结构：

- **CallbackManager**: 同步回调管理器
- **AsyncCallbackManager**: 异步回调管理器
- **RunManager**: 运行管理器（管理单次运行的回调）
  - `CallbackManagerForChainRun`
  - `CallbackManagerForLLMRun`
  - `CallbackManagerForToolRun`
  - `CallbackManagerForRetrieverRun`
- **trace_as_chain_group()**: 链组追踪上下文管理器
- **atrace_as_chain_group()**: 异步链组追踪上下文管理器

### 4. 内置回调处理器

**StdOutCallbackHandler** (标准输出):
- 打印 Chain 的开始和结束
- 打印 Agent 的动作和完成
- 打印 Tool 的输出
- 支持彩色输出

**StreamingStdOutCallbackHandler** (流式输出):
- 专门用于流式 LLM 输出
- 实现 `on_llm_new_token()` 方法
- 实时输出每个 token 到 stdout

### 5. 回调参数标准化

所有回调方法都遵循统一的参数模式：
- `run_id: UUID` - 当前运行的唯一标识
- `parent_run_id: UUID | None` - 父运行的标识（支持嵌套）
- `tags: list[str] | None` - 标签（用于分类和过滤）
- `metadata: dict[str, Any] | None` - 元数据
- `**kwargs: Any` - 额外参数

### 6. 异步支持

`AsyncCallbackHandler` 提供异步版本的所有回调方法：
- 所有方法都是 `async def`
- 支持异步 LLM 调用
- 支持异步 Chain 执行

### 7. 公共 API 导出

从 `__init__.py` 中发现导出的核心类：
- 基础类：`BaseCallbackHandler`, `AsyncCallbackHandler`
- 管理器：`CallbackManager`, `AsyncCallbackManager`
- 运行管理器：各种 `CallbackManagerFor*Run`
- 内置处理器：`StdOutCallbackHandler`, `StreamingStdOutCallbackHandler`, `FileCallbackHandler`, `UsageMetadataCallbackHandler`
- 自定义事件：`dispatch_custom_event`, `adispatch_custom_event`

## 代码片段

### Mixin 示例：LLMManagerMixin

```python
class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new output token.

        Only available when streaming is enabled.
        """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
```

### StreamingStdOutCallbackHandler 实现

```python
class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming.

    !!! warning "Only works with LLMs that support streaming."
    """

    @override
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
```

### trace_as_chain_group 上下文管理器

```python
@contextmanager
def trace_as_chain_group(
    group_name: str,
    callback_manager: CallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Get a callback manager for a chain group in a context manager.

    Useful for grouping different calls together as a single run even if they aren't
    composed in a single chain.
    """
    # 实现细节...
```

## 架构设计亮点

1. **Mixin 模式**：通过组合而非继承实现灵活的回调接口
2. **统一参数**：所有回调方法使用一致的参数模式（run_id, parent_run_id, tags, metadata）
3. **层次化管理**：通过 parent_run_id 支持嵌套回调追踪
4. **异步优先**：完整的异步支持
5. **可选忽略**：通过 `ignore_*` 属性选择性忽略特定类型的回调
6. **错误处理**：通过 `raise_error` 控制异常传播
7. **自定义事件**：支持用户定义的自定义事件

## 依赖库识别

从源码中识别出的关键依赖：
- `langsmith` - 用于追踪和可观测性（LangChainTracer）
- `tenacity` - 用于重试机制（RetryCallState）
- `uuid` - 用于生成唯一标识
- `typing_extensions` - 用于类型提示
