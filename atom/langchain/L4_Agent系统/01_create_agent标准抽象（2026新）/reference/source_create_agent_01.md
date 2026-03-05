---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain_v1/langchain/agents/__init__.py
  - libs/langchain_v1/langchain/agents/factory.py
  - libs/langchain_v1/langchain/agents/middleware/types.py
analyzed_at: 2026-02-28
knowledge_point: 01_create_agent标准抽象（2026新）
---

# 源码分析：create_agent 工厂函数与核心类型

## 分析的文件

- `libs/langchain_v1/langchain/agents/__init__.py` - 公共 API 导出（仅导出 create_agent 和 AgentState）
- `libs/langchain_v1/langchain/agents/factory.py` - create_agent() 主实现（1822行）
- `libs/langchain_v1/langchain/agents/middleware/types.py` - 核心类型定义

## 关键发现

### 1. create_agent() 函数签名

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]
```

返回值是 CompiledStateGraph，即基于 LangGraph 的编译状态图。

### 2. AgentState 定义

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

### 3. AgentMiddleware 基类

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    state_schema: type[StateT]
    tools: Sequence[BaseTool]

    # 生命周期钩子
    def before_agent(self, state, runtime) -> dict | None
    def after_agent(self, state, runtime) -> dict | None
    def before_model(self, state, runtime) -> dict | None
    def after_model(self, state, runtime) -> dict | None

    # 包装器钩子
    def wrap_model_call(self, request, handler) -> ModelResponse | AIMessage | ExtendedModelResponse
    def wrap_tool_call(self, request, execute) -> ToolMessage | Command

    # 异步版本
    async def abefore_agent/abefore_model/aafter_model/aafter_agent
    async def awrap_model_call/awrap_tool_call
```

### 4. ModelRequest 数据类

```python
@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    model: BaseChatModel
    messages: list[AnyMessage]
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any]

    def override(self, **overrides) -> ModelRequest[ContextT]:
        """不可变模式：返回新实例"""
```

### 5. ModelResponse 数据类

```python
@dataclass
class ModelResponse(Generic[ResponseT]):
    result: list[BaseMessage]
    structured_response: ResponseT | None = None
```

### 6. 内部实现要点

- create_agent 内部构建 LangGraph StateGraph
- 支持 middleware 链式组合（_chain_model_call_handlers, _chain_tool_call_wrappers）
- 结构化输出支持三种策略：ToolStrategy, ProviderStrategy, AutoStrategy
- model 参数支持字符串标识符（如 "openai:gpt-4"）通过 init_chat_model 转换
- 工具分为 built_in_tools（dict 格式）和 regular_tools（BaseTool/Callable）

### 7. 19个内置中间件

middleware/ 目录包含：
- ModelRetryMiddleware, ModelFallbackMiddleware, ModelCallLimitMiddleware
- ToolRetryMiddleware, ToolCallLimitMiddleware, LLMToolEmulator, LLMToolSelectorMiddleware
- HumanInTheLoopMiddleware, PIIMiddleware, ShellToolMiddleware
- FilesystemFileSearchMiddleware, ContextEditingMiddleware, SummarizationMiddleware
- TodoListMiddleware 等
