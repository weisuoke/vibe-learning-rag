---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/agents/agent.py
  - libs/langchain/langchain_classic/agents/initialize.py
  - libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
  - libs/core/langchain_core/tools/base.py
  - libs/core/langchain_core/callbacks/base.py
  - libs/langchain/langchain_classic/agents/utils.py
analyzed_at: 2026-03-06
knowledge_point: 10_Agent最佳实践
---

# 源码分析：Agent 设计模式与架构

## 分析的文件

| 文件 | 描述 | 代码行数 |
|------|------|----------|
| `agents/agent.py` | Agent 核心类：`BaseSingleActionAgent`、`BaseMultiActionAgent`、`AgentExecutor`、`RunnableAgent` 等 | ~1793 行 |
| `agents/initialize.py` | Agent 工厂函数 `initialize_agent`（已废弃） | ~117 行 |
| `agents/tool_calling_agent/base.py` | 现代 Tool Calling Agent 创建函数 `create_tool_calling_agent` | ~118 行 |
| `tools/base.py` | `BaseTool` 基类、`ToolException`、输入验证体系 | ~1100+ 行 |
| `callbacks/base.py` | 回调系统基类 `BaseCallbackHandler`、各种 Mixin | ~1118 行 |
| `agents/utils.py` | 工具验证实用函数 | ~20 行 |

---

## 关键发现

### 1. 分层架构（Layered Architecture）

LangChain Agent 系统采用清晰的三层架构：

```
┌─────────────────────────────────┐
│  AgentExecutor（执行层/编排层）    │  → 控制循环、错误处理、超时管理
├─────────────────────────────────┤
│  Agent（决策层）                  │  → plan() / aplan() 决定下一步
├─────────────────────────────────┤
│  Tools + Callbacks（能力层）      │  → 实际执行工具、观察回调
└─────────────────────────────────┘
```

**源码证据：** `AgentExecutor` (line 1012) 持有 `agent` 和 `tools`，自己只负责循环编排。

### 2. 策略模式（Strategy Pattern）

Agent 的核心设计是策略模式——`AgentExecutor` 不关心 Agent 如何做决策，只通过 `plan()` 接口调用。

```python
# agent.py, line 67-84 — 抽象策略接口
class BaseSingleActionAgent(BaseModel):
    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do."""
```

不同的 Agent 实现（`RunnableAgent`、`Agent`、`LLMSingleActionAgent`）各自实现 `plan()` 方法，但 `AgentExecutor` 的调用代码完全一致：

```python
# agent.py, line 1317-1321 — 统一调用
output = self._action_agent.plan(
    intermediate_steps,
    callbacks=run_manager.get_child() if run_manager else None,
    **inputs,
)
```

### 3. 模板方法模式（Template Method Pattern）

`AgentExecutor._call()` 是一个经典的模板方法，定义了 Agent 执行的骨架流程：

```python
# agent.py, line 1570-1622 — 核心执行循环
def _call(self, inputs, run_manager=None):
    # 1. 构建工具映射
    name_to_tool_map = {tool.name: tool for tool in self.tools}
    color_mapping = get_color_mapping(...)

    # 2. 初始化状态
    intermediate_steps = []
    iterations = 0
    time_elapsed = 0.0
    start_time = time.time()

    # 3. 执行循环（可覆写的子步骤）
    while self._should_continue(iterations, time_elapsed):
        next_step_output = self._take_next_step(...)  # 可覆写
        if isinstance(next_step_output, AgentFinish):
            return self._return(...)  # 可覆写

        intermediate_steps.extend(next_step_output)
        # 检查工具是否要求直接返回
        if len(next_step_output) == 1:
            tool_return = self._get_tool_return(next_step_output[0])
            if tool_return is not None:
                return self._return(tool_return, ...)

        iterations += 1
        time_elapsed = time.time() - start_time

    # 4. 超过限制时的停止策略
    output = self._action_agent.return_stopped_response(
        self.early_stopping_method, intermediate_steps, **inputs,
    )
    return self._return(output, intermediate_steps, run_manager=run_manager)
```

### 4. 适配器模式（Adapter Pattern）

`RunnableAgent` 和 `RunnableMultiActionAgent` 是适配器，将任意 `Runnable` 适配为 Agent 接口：

```python
# agent.py, line 389-494 — Runnable → Agent 适配
class RunnableAgent(BaseSingleActionAgent):
    runnable: Runnable[dict, AgentAction | AgentFinish]

    def plan(self, intermediate_steps, callbacks=None, **kwargs):
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        # 将 Runnable 的 stream/invoke 适配为 plan 的返回值
        if self.stream_runnable:
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})
        return final_output
```

自动适配发生在 `AgentExecutor` 的验证器中：

```python
# agent.py, line 1110-1144 — 自动检测并适配 Runnable
@model_validator(mode="before")
@classmethod
def validate_runnable_agent(cls, values):
    agent = values.get("agent")
    if agent and isinstance(agent, Runnable):
        try:
            output_type = agent.OutputType
        except TypeError:
            multi_action = False
        else:
            multi_action = output_type == list[AgentAction] | AgentFinish

        if multi_action:
            values["agent"] = RunnableMultiActionAgent(runnable=agent, ...)
        else:
            values["agent"] = RunnableAgent(runnable=agent, ...)
    return values
```

### 5. 管道/链式组合模式（Pipeline / LCEL Composition）

`create_tool_calling_agent` 展示了 LCEL（LangChain Expression Language）的管道式组合：

```python
# tool_calling_agent/base.py, line 110-117
def create_tool_calling_agent(llm, tools, prompt, *, message_formatter=...):
    # 前置验证
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError("This function requires a bind_tools() method...")

    llm_with_tools = llm.bind_tools(tools)

    # 管道式组合：数据准备 → 提示词 → LLM → 输出解析
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: message_formatter(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
```

**设计要点：**
- 每一步都是独立的 `Runnable`，可替换
- `|` 运算符实现了声明式的管道构建
- 前置验证确保组件兼容性

### 6. Mixin 模式（回调系统）

回调系统使用 Mixin 模式实现关注点分离：

```python
# callbacks/base.py, line 435-442 — 通过多重继承组合能力
class BaseCallbackHandler(
    LLMManagerMixin,       # on_llm_new_token, on_llm_end, on_llm_error
    ChainManagerMixin,     # on_chain_end, on_chain_error, on_agent_action, on_agent_finish
    ToolManagerMixin,      # on_tool_end, on_tool_error
    RetrieverManagerMixin, # on_retriever_error, on_retriever_end
    CallbackManagerMixin,  # on_llm_start, on_chat_model_start, on_chain_start, on_tool_start
    RunManagerMixin,       # on_text, on_retry, on_custom_event
):
    raise_error: bool = False
    run_inline: bool = False
```

每个 Mixin 负责一类组件的回调，Handler 可以通过 `ignore_*` 属性选择性忽略：

```python
# callbacks/base.py, line 451-484
@property
def ignore_llm(self) -> bool: return False
@property
def ignore_chain(self) -> bool: return False
@property
def ignore_agent(self) -> bool: return False
@property
def ignore_retriever(self) -> bool: return False
```

### 7. 工厂模式（Factory Pattern）

Agent 创建使用了多种工厂模式：

```python
# 工厂方法 1：类方法工厂（经典模式，已废弃）
# agent.py, line 884-918
@classmethod
def from_llm_and_tools(cls, llm, tools, callback_manager=None, output_parser=None, **kwargs):
    cls._validate_tools(tools)
    llm_chain = LLMChain(llm=llm, prompt=cls.create_prompt(tools), ...)
    tool_names = [tool.name for tool in tools]
    _output_parser = output_parser or cls._get_default_output_parser()
    return cls(llm_chain=llm_chain, allowed_tools=tool_names, output_parser=_output_parser, ...)

# 工厂方法 2：函数式工厂（现代推荐方式）
# tool_calling_agent/base.py, line 18-117
def create_tool_calling_agent(llm, tools, prompt, *, message_formatter=...):
    # 验证 → 组装 → 返回 Runnable
    ...

# 工厂方法 3：注册表工厂（已废弃）
# initialize.py, line 79-94
if agent not in AGENT_TO_CLASS:
    raise ValueError(f"Got unknown agent type: {agent}. Valid types are: {AGENT_TO_CLASS.keys()}.")
agent_cls = AGENT_TO_CLASS[agent]
agent_obj = agent_cls.from_llm_and_tools(llm, tools, ...)
```

### 8. 观察者模式（Observer Pattern — 回调系统）

回调系统是一个完整的观察者模式实现：

```python
# callbacks/base.py — 生命周期钩子覆盖 Agent 执行的每个阶段
class BaseCallbackHandler:
    # LLM 层
    def on_llm_start(self, serialized, prompts, *, run_id, ...): ...
    def on_llm_new_token(self, token, *, chunk, run_id, ...): ...
    def on_llm_end(self, response, *, run_id, ...): ...
    def on_llm_error(self, error, *, run_id, ...): ...

    # Chain 层
    def on_chain_start(self, serialized, inputs, *, run_id, ...): ...
    def on_chain_end(self, outputs, *, run_id, ...): ...
    def on_chain_error(self, error, *, run_id, ...): ...

    # Agent 层
    def on_agent_action(self, action, *, run_id, ...): ...
    def on_agent_finish(self, finish, *, run_id, ...): ...

    # Tool 层
    def on_tool_start(self, serialized, input_str, *, run_id, ...): ...
    def on_tool_end(self, output, *, run_id, ...): ...
    def on_tool_error(self, error, *, run_id, ...): ...

    # Retriever 层
    def on_retriever_start(self, serialized, query, *, run_id, ...): ...
    def on_retriever_end(self, documents, *, run_id, ...): ...
    def on_retriever_error(self, error, *, run_id, ...): ...

    # 通用
    def on_retry(self, retry_state, *, run_id, ...): ...
    def on_custom_event(self, name, data, *, run_id, ...): ...
```

**关键设计：每个回调方法都包含 `run_id` 和 `parent_run_id`，形成可追踪的执行树。**

### 9. 可继承的回调管理器（Inheritable Callback Manager）

```python
# callbacks/base.py, line 898-932
class BaseCallbackManager(CallbackManagerMixin):
    def __init__(self, handlers, inheritable_handlers=None, parent_run_id=None,
                 *, tags=None, inheritable_tags=None,
                 metadata=None, inheritable_metadata=None):
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers or []  # 子节点继承
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []  # 子节点继承的 tags
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}  # 子节点继承的 metadata
```

**设计意图：** `inheritable_*` 字段允许父级回调管理器的配置自动传递给子运行（如 Agent → Tool 调用），而 `handlers`/`tags`/`metadata` 只在当前级别生效。这是一种**作用域继承模式**。

---

## 配置最佳实践

### 1. AgentExecutor 关键配置项

```python
# agent.py, line 1012-1056
class AgentExecutor(Chain):
    agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable
    tools: Sequence[BaseTool]

    # 安全限制
    max_iterations: int | None = 15        # 防止无限循环
    max_execution_time: float | None = None # 时间限制

    # 错误处理策略
    handle_parsing_errors: bool | str | Callable = False  # 多态错误处理
    early_stopping_method: str = "force"    # "force" 或 "generate"

    # 调试与观测
    return_intermediate_steps: bool = False  # 返回中间步骤
    trim_intermediate_steps: int | Callable = -1  # 裁剪中间步骤（内存管理）
```

### 2. Tool 配置体系

```python
# tools/base.py, line 405-531
class BaseTool(RunnableSerializable):
    name: str                    # 唯一标识，LLM 用来选择工具
    description: str             # LLM 用来理解何时使用工具
    args_schema: ArgsSchema      # Pydantic 模型验证输入

    return_direct: bool = False  # True = 工具结果直接作为最终答案
    verbose: bool = False

    # 错误处理
    handle_tool_error: bool | str | Callable = False
    handle_validation_error: bool | str | Callable = False

    # 响应格式
    response_format: Literal["content", "content_and_artifact"] = "content"

    # 可观测性
    callbacks: Callbacks = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    # 扩展字段
    extras: dict[str, Any] | None = None  # 提供商特定配置
```

### 3. 多态配置模式

LangChain 大量使用 `bool | str | Callable` 类型的配置，提供灵活的错误处理策略：

```python
# handle_parsing_errors 的三种使用方式：
AgentExecutor(handle_parsing_errors=True)                    # 默认消息发回 LLM
AgentExecutor(handle_parsing_errors="请重新格式化你的回答")     # 自定义消息
AgentExecutor(handle_parsing_errors=lambda e: f"错误: {e}")  # 自定义处理函数
```

---

## 安全考量

### 1. 迭代限制防止无限循环

```python
# agent.py, line 1023-1038
max_iterations: int | None = 15  # 默认 15 次
max_execution_time: float | None = None
early_stopping_method: str = "force"  # 强制停止，不让 LLM 再生成

# agent.py, line 1235-1238
def _should_continue(self, iterations, time_elapsed):
    if self.max_iterations is not None and iterations >= self.max_iterations:
        return False
    return self.max_execution_time is None or time_elapsed < self.max_execution_time
```

### 2. 工具白名单验证

```python
# agent.py, line 1084-1108 — 验证 Agent 的允许工具与实际提供的工具一致
@model_validator(mode="after")
def validate_tools(self):
    allowed_tools = agent.get_allowed_tools()
    if allowed_tools is not None and set(allowed_tools) != {tool.name for tool in tools}:
        raise ValueError(
            f"Allowed tools ({allowed_tools}) different than "
            f"provided tools ({[tool.name for tool in tools]})"
        )
    return self
```

### 3. 未知工具的安全处理

```python
# agent.py, line 1390-1416 — 当 LLM 调用不存在的工具时
def _perform_agent_action(self, name_to_tool_map, ...):
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        observation = tool.run(agent_action.tool_input, ...)
    else:
        # 不崩溃，而是返回错误信息让 Agent 重试
        observation = InvalidTool().run(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            }, ...
        )
```

### 4. 输入验证与注入参数过滤

```python
# tools/base.py, line 803-837 — 过滤注入参数，防止泄露
def _filter_injected_args(self, tool_input: dict) -> dict:
    filtered_keys = set(FILTERED_ARGS)  # "run_manager", "callbacks"
    filtered_keys.update(self._injected_args_keys)
    # 从 args_schema 中识别 InjectedToolArg 类型
    if self.args_schema is not None and not isinstance(self.args_schema, dict):
        try:
            annotations = get_all_basemodel_annotations(self.args_schema)
            for field_name, field_type in annotations.items():
                if _is_injected_arg_type(field_type):
                    filtered_keys.add(field_name)
        except Exception:
            _logger.debug("Failed to get args_schema annotations for filtering.", exc_info=True)
    return {k: v for k, v in tool_input.items() if k not in filtered_keys}
```

### 5. Schema 类型注解验证（编译时安全）

```python
# tools/base.py, line 413-444 — 子类定义时验证 args_schema 注解
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    args_schema_type = cls.__annotations__.get("args_schema", None)
    if args_schema_type is not None and args_schema_type == BaseModel:
        raise SchemaAnnotationError(
            f"Tool definition for {name} must include valid type annotations"
            f" for argument 'args_schema' to behave as expected.\n"
            f"Expected annotation of 'Type[BaseModel]'"
        )
```

---

## 同步/异步对称设计

LangChain 的每个核心方法都有同步和异步版本，形成对称 API：

| 同步方法 | 异步方法 | 所在类 |
|----------|----------|--------|
| `plan()` | `aplan()` | `BaseSingleActionAgent` |
| `_call()` | `_acall()` | `AgentExecutor` |
| `_take_next_step()` | `_atake_next_step()` | `AgentExecutor` |
| `_iter_next_step()` | `_aiter_next_step()` | `AgentExecutor` |
| `_perform_agent_action()` | `_aperform_agent_action()` | `AgentExecutor` |
| `_return()` | `_areturn()` | `AgentExecutor` |
| `run()` | `arun()` | `BaseTool` |
| `_run()` | `_arun()` | `BaseTool` |
| `stream()` | `astream()` | `AgentExecutor` |

**异步版本的额外特性：并发工具执行**

```python
# agent.py, line 1510-1521 — 异步版本使用 asyncio.gather 并发执行多个工具
result = await asyncio.gather(
    *[
        self._aperform_agent_action(
            name_to_tool_map, color_mapping, agent_action, run_manager,
        )
        for agent_action in actions
    ],
)
```

**异步版本的额外特性：asyncio 超时**

```python
# agent.py, line 1643-1695 — 异步版本使用 asyncio_timeout 实现更精确的超时
async def _acall(self, inputs, run_manager=None):
    try:
        async with asyncio_timeout(self.max_execution_time):
            while self._should_continue(iterations, time_elapsed):
                ...
    except (TimeoutError, asyncio.TimeoutError):
        output = self._action_agent.return_stopped_response(...)
        return await self._areturn(output, intermediate_steps, ...)
```

---

## 中间步骤管理（Memory Management）

```python
# agent.py, line 1718-1729 — 裁剪中间步骤，控制 context window 消耗
def _prepare_intermediate_steps(self, intermediate_steps):
    if isinstance(self.trim_intermediate_steps, int) and self.trim_intermediate_steps > 0:
        return intermediate_steps[-self.trim_intermediate_steps:]  # 保留最近 N 步
    if callable(self.trim_intermediate_steps):
        return self.trim_intermediate_steps(intermediate_steps)  # 自定义裁剪
    return intermediate_steps  # 默认不裁剪
```

**设计启示：** 长时间运行的 Agent 需要管理中间步骤的增长，否则会超出 LLM 的 context window 限制。

---

## 架构演进：从 Legacy 到现代

LangChain Agent 架构经历了清晰的演进：

| 阶段 | 代表 | 状态 |
|------|------|------|
| V1: LLMChain 驱动 | `Agent`、`LLMSingleActionAgent` | `@deprecated("0.1.0")` |
| V2: 函数式工厂 | `initialize_agent` | `@deprecated("0.1.0")` |
| V3: Runnable 组合 | `create_tool_calling_agent` + `AgentExecutor` | 当前经典模式 |
| V4: LangGraph | `create_agent` (langchain v1 包) | 推荐的未来方向 |

**废弃注释揭示的迁移方向：**

```python
# initialize.py, line 19-23
@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,  # 指向 LangGraph 的迁移指南
    removal="1.0",
)
```

---

## 设计模式总结表

| 设计模式 | 应用位置 | 目的 |
|----------|----------|------|
| **策略模式** | `BaseSingleActionAgent.plan()` | 解耦决策与执行 |
| **模板方法** | `AgentExecutor._call()` | 固定执行骨架，允许覆写步骤 |
| **适配器模式** | `RunnableAgent` / `RunnableMultiActionAgent` | 将 Runnable 适配为 Agent 接口 |
| **管道模式** | `create_tool_calling_agent` (LCEL) | 声明式组合处理链 |
| **工厂模式** | `from_llm_and_tools()` / `create_tool_calling_agent()` | 封装复杂的创建逻辑 |
| **观察者模式** | `BaseCallbackHandler` + `CallbackManager` | 可插拔的生命周期监听 |
| **Mixin 模式** | `LLMManagerMixin` 等 | 按组件类型分离回调接口 |
| **注册表模式** | `AGENT_TO_CLASS` | 按类型字符串查找 Agent 类 |
| **空对象模式** | `InvalidTool` / `ExceptionTool` | 安全处理无效工具调用和异常 |
| **多态配置** | `handle_parsing_errors: bool \| str \| Callable` | 灵活的错误处理策略 |
