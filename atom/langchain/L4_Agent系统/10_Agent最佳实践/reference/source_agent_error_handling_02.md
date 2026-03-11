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

# 源码分析：Agent 错误处理、重试策略与弹性模式

## 分析的文件

| 文件 | 关注点 |
|------|--------|
| `agents/agent.py` | Agent 执行循环中的错误处理、超时管理、解析错误恢复 |
| `agents/initialize.py` | 初始化阶段的参数验证 |
| `agents/tool_calling_agent/base.py` | 前置条件验证（Fail-Fast） |
| `tools/base.py` | 工具执行中的异常分层处理、输入验证 |
| `callbacks/base.py` | 回调系统中的错误传播与可观测性 |
| `agents/utils.py` | 工具兼容性验证 |

---

## 错误处理架构总览

LangChain Agent 系统的错误处理分为 **五个层次**：

```
┌──────────────────────────────────────────────────────────┐
│ Layer 5: 全局保护 — 迭代限制 + 时间限制                      │
├──────────────────────────────────────────────────────────┤
│ Layer 4: 执行循环错误 — OutputParserException 处理          │
├──────────────────────────────────────────────────────────┤
│ Layer 3: 工具执行错误 — ToolException + ValidationError     │
├──────────────────────────────────────────────────────────┤
│ Layer 2: 输入验证 — args_schema + Pydantic 验证             │
├──────────────────────────────────────────────────────────┤
│ Layer 1: 前置条件 — 组件兼容性检查（Fail-Fast）              │
└──────────────────────────────────────────────────────────┘
```

---

## Layer 1: 前置条件验证（Fail-Fast 模式）

### 1.1 Prompt 变量验证

```python
# tool_calling_agent/base.py, line 96-101
def create_tool_calling_agent(llm, tools, prompt, ...):
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)
```

**最佳实践：** 在组装阶段就检查必需变量，而不是等到运行时才发现缺少 `agent_scratchpad`。

### 1.2 LLM 能力验证

```python
# tool_calling_agent/base.py, line 103-107
    if not hasattr(llm, "bind_tools"):
        msg = "This function requires a bind_tools() method be implemented on the LLM."
        raise ValueError(msg)
```

**最佳实践：** 使用 `hasattr` 进行鸭子类型检查，确保 LLM 支持工具调用能力。

### 1.3 工具兼容性验证

```python
# utils.py, line 6-19
def validate_tools_single_input(class_name: str, tools: Sequence[BaseTool]) -> None:
    for tool in tools:
        if not tool.is_single_input:
            msg = f"{class_name} does not support multi-input tool {tool.name}."
            raise ValueError(msg)
```

### 1.4 初始化阶段的互斥参数验证

```python
# initialize.py, line 71-109
def initialize_agent(tools, llm, agent=None, agent_path=None, ...):
    # 互斥检查
    if agent is not None and agent_path is not None:
        msg = "Both `agent` and `agent_path` are specified, but at most only one should be."
        raise ValueError(msg)

    # 类型注册表验证
    if agent not in AGENT_TO_CLASS:
        msg = f"Got unknown agent type: {agent}. Valid types are: {AGENT_TO_CLASS.keys()}."
        raise ValueError(msg)

    # 防御性编程：理论上不可达的分支
    else:
        msg = "Somehow both `agent` and `agent_path` are None, this should never happen."
        raise ValueError(msg)
```

### 1.5 Agent-Tool 一致性验证

```python
# agent.py, line 1084-1108
@model_validator(mode="after")
def validate_tools(self):
    agent = self.agent
    tools = self.tools
    allowed_tools = agent.get_allowed_tools()
    if allowed_tools is not None and set(allowed_tools) != {tool.name for tool in tools}:
        msg = (
            f"Allowed tools ({allowed_tools}) different than "
            f"provided tools ({[tool.name for tool in tools]})"
        )
        raise ValueError(msg)
    return self
```

### 1.6 Prompt 自动修复（宽容验证）

```python
# agent.py, line 819-847 — 不只是报错，而是尝试自动修复
@model_validator(mode="after")
def validate_prompt(self):
    prompt = self.llm_chain.prompt
    if "agent_scratchpad" not in prompt.input_variables:
        logger.warning(
            "`agent_scratchpad` should be a variable in prompt.input_variables."
            " Did not find it, so adding it at the end.",
        )
        prompt.input_variables.append("agent_scratchpad")
        if isinstance(prompt, PromptTemplate):
            prompt.template += "\n{agent_scratchpad}"
        elif isinstance(prompt, FewShotPromptTemplate):
            prompt.suffix += "\n{agent_scratchpad}"
        else:
            msg = f"Got unexpected prompt type {type(prompt)}"
            raise ValueError(msg)
    return self
```

**设计启示：** 对于非致命的配置问题，先尝试自动修复 + 发出警告，实在无法修复时才抛错。

---

## Layer 2: 输入验证（Pydantic 集成）

### 2.1 工具输入验证

```python
# tools/base.py, line 656-775 — 多种输入类型的验证路径
def _parse_input(self, tool_input, tool_call_id):
    input_args = self.args_schema

    # 路径 1: 字符串输入
    if isinstance(tool_input, str):
        if input_args is not None:
            if isinstance(input_args, dict):
                raise ValueError("String tool inputs not allowed with JSON schema args_schema.")
            key_ = next(iter(get_fields(input_args).keys()))
            if issubclass(input_args, BaseModel):
                input_args.model_validate({key_: tool_input})  # Pydantic v2
            elif issubclass(input_args, BaseModelV1):
                input_args.parse_obj({key_: tool_input})  # Pydantic v1
        return tool_input

    # 路径 2: 字典输入 + Pydantic 模型验证
    if input_args is not None:
        if issubclass(input_args, BaseModel):
            result = input_args.model_validate(tool_input)
            result_dict = result.model_dump()
        elif issubclass(input_args, BaseModelV1):
            result = input_args.parse_obj(tool_input)
            result_dict = result.dict()
    return tool_input
```

### 2.2 Schema 注解验证（元编程层）

```python
# tools/base.py, line 413-444 — 在类定义时就验证注解
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    args_schema_type = cls.__annotations__.get("args_schema", None)
    if args_schema_type is not None and args_schema_type == BaseModel:
        # 常见错误：使用 BaseModel 而非 Type[BaseModel]
        raise SchemaAnnotationError(
            f"Expected annotation of 'Type[BaseModel]' but got '{args_schema_type}'."
        )
```

### 2.3 构造函数类型验证

```python
# tools/base.py, line 533-551
def __init__(self, **kwargs):
    if (
        "args_schema" in kwargs
        and kwargs["args_schema"] is not None
        and not is_basemodel_subclass(kwargs["args_schema"])
        and not isinstance(kwargs["args_schema"], dict)
    ):
        msg = (
            "args_schema must be a subclass of pydantic BaseModel or "
            f"a JSON schema dict. Got: {kwargs['args_schema']}."
        )
        raise TypeError(msg)
    super().__init__(**kwargs)
```

---

## Layer 3: 工具执行错误处理

### 3.1 异常分层体系

LangChain 定义了清晰的异常层次：

```
BaseException
├── Exception
│   ├── ToolException          → 业务错误，可恢复
│   ├── ValidationError        → 输入验证错误，可恢复
│   ├── OutputParserException  → LLM 输出解析错误，可恢复
│   └── SchemaAnnotationError  → 开发时错误，不可恢复
└── KeyboardInterrupt          → 用户中断
```

### 3.2 工具执行的三层 try-except

```python
# tools/base.py, line 957-1004 — BaseTool.run() 的完整错误处理
def run(self, tool_input, ...):
    content = None
    artifact = None
    status = "success"
    error_to_raise: Exception | KeyboardInterrupt | None = None

    try:
        # 正常执行
        child_config = patch_config(config, callbacks=run_manager.get_child())
        with set_config_context(child_config) as context:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            if signature(self._run).parameters.get("run_manager"):
                tool_kwargs |= {"run_manager": run_manager}
            response = context.run(self._run, *tool_args, **tool_kwargs)

        # 响应格式验证
        if self.response_format == "content_and_artifact":
            if not isinstance(response, tuple):
                error_to_raise = ValueError(msg)
            else:
                try:
                    content, artifact = response
                except ValueError:
                    error_to_raise = ValueError(msg)
        else:
            content = response

    # 第 1 层：验证错误（输入不合法）
    except (ValidationError, ValidationErrorV1) as e:
        if not self.handle_validation_error:
            error_to_raise = e
        else:
            content = _handle_validation_error(e, flag=self.handle_validation_error)
            status = "error"

    # 第 2 层：工具业务错误（可预期的失败）
    except ToolException as e:
        if not self.handle_tool_error:
            error_to_raise = e
        else:
            content = _handle_tool_error(e, flag=self.handle_tool_error)
            status = "error"

    # 第 3 层：未预期的异常 + KeyboardInterrupt
    except (Exception, KeyboardInterrupt) as e:
        error_to_raise = e

    # 统一的错误通知与抛出
    if error_to_raise:
        run_manager.on_tool_error(error_to_raise, tool_call_id=tool_call_id)
        raise error_to_raise

    output = _format_output(content, artifact, tool_call_id, self.name, status)
    run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
    return output
```

**关键设计模式：**

1. **延迟抛出（Deferred Raise）：** 不在 except 块中直接 raise，而是先存储到 `error_to_raise`，确保 callback 通知（`on_tool_error`）总能执行。
2. **错误也是有效输出：** 当 `handle_tool_error=True` 时，错误被转化为 `content`（字符串），作为正常观察返回给 Agent。
3. **状态跟踪：** `status` 变量区分成功和错误状态。

### 3.3 多态错误处理器

```python
# 三种错误处理模式（handle_tool_error / handle_validation_error / handle_parsing_errors）

# 模式 1: bool — 使用默认错误消息
handle_tool_error = True
# → content = f"Tool execution error: {str(e)}"

# 模式 2: str — 使用固定的自定义消息
handle_tool_error = "工具调用失败，请尝试其他方法"
# → content = "工具调用失败，请尝试其他方法"

# 模式 3: Callable — 自定义错误处理函数
handle_tool_error = lambda e: f"错误详情: {e.args[0]}, 建议: ..."
# → content = handle_tool_error(e)
```

### 3.4 ToolException — 可控的错误信号

```python
# tools/base.py, line 390-397
class ToolException(Exception):
    """Exception thrown when a tool execution error occurs.

    This exception allows tools to signal errors without stopping the agent.
    The error is handled according to the tool's `handle_tool_error` setting,
    and the result is returned as an observation to the agent.
    """
```

**使用模式：** 工具开发者通过 `raise ToolException("...")` 发出可恢复的业务错误，与系统级异常（如网络超时、内存不足）区分开来。

---

## Layer 4: Agent 执行循环错误处理

### 4.1 OutputParserException — LLM 输出格式错误

这是 Agent 循环中最关键的错误处理，因为 LLM 输出格式不可控。

```python
# agent.py, line 1313-1361 — _iter_next_step 中的解析错误处理
def _iter_next_step(self, name_to_tool_map, color_mapping, inputs,
                    intermediate_steps, run_manager=None):
    try:
        intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
        output = self._action_agent.plan(
            intermediate_steps,
            callbacks=run_manager.get_child() if run_manager else None,
            **inputs,
        )
    except OutputParserException as e:
        # 决定是否要将错误传回 LLM 重试
        if isinstance(self.handle_parsing_errors, bool):
            raise_error = not self.handle_parsing_errors
        else:
            raise_error = False

        if raise_error:
            msg = (
                "An output parsing error occurred. "
                "In order to pass this error back to the agent and have it try "
                "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                f"This is the error: {e!s}"
            )
            raise ValueError(msg) from e

        # 构建观察信息
        text = str(e)
        if isinstance(self.handle_parsing_errors, bool):
            if e.send_to_llm:
                observation = str(e.observation)
                text = str(e.llm_output)
            else:
                observation = "Invalid or incomplete response"
        elif isinstance(self.handle_parsing_errors, str):
            observation = self.handle_parsing_errors
        elif callable(self.handle_parsing_errors):
            observation = self.handle_parsing_errors(e)
        else:
            raise ValueError("Got unexpected type of `handle_parsing_errors`") from e

        # 将错误作为 _Exception 工具的调用，创建错误观察步骤
        output = AgentAction("_Exception", observation, text)
        if run_manager:
            run_manager.on_agent_action(output, color="green")
        tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
        observation = ExceptionTool().run(
            output.tool_input,
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
        yield AgentStep(action=output, observation=observation)
        return  # 结束本步，让循环继续下一次迭代
```

**核心机制：** 解析错误被转化为一个虚拟的 `_Exception` 工具调用，错误信息作为 `observation` 添加到中间步骤中，让 Agent 在下一次 `plan()` 时能看到自己之前的格式错误，从而自我纠正。

### 4.2 ExceptionTool — 错误回馈机制

```python
# agent.py, line 983-1005
class ExceptionTool(BaseTool):
    """Tool that just returns the query."""
    name: str = "_Exception"
    description: str = "Exception tool"

    def _run(self, query: str, run_manager=None) -> str:
        return query  # 直接返回错误信息
```

**设计意图：** `ExceptionTool` 是一个恒等函数，它的唯一目的是让错误信息以标准的 `AgentStep(action, observation)` 格式进入中间步骤。这样 Agent 的 scratchpad 中会包含错误记录，让 LLM 能理解之前发生了什么。

### 4.3 未知工具处理

```python
# agent.py, line 1390-1416
def _perform_agent_action(self, name_to_tool_map, color_mapping, agent_action, run_manager):
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        observation = tool.run(agent_action.tool_input, ...)
    else:
        # 安全降级：使用 InvalidTool 返回可用工具列表
        observation = InvalidTool().run(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            },
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    return AgentStep(action=agent_action, observation=observation)
```

**最佳实践：** 不直接抛异常，而是通过 `InvalidTool` 返回"你请求的工具不存在，可用的工具有..."的信息，让 Agent 自我纠正。

---

## Layer 5: 全局保护机制

### 5.1 迭代次数限制

```python
# agent.py, line 1023-1027
max_iterations: int | None = 15  # 默认 15 次迭代
"""Setting to 'None' could lead to an infinite loop."""  # 文档明确警告

# agent.py, line 1235-1238
def _should_continue(self, iterations, time_elapsed):
    if self.max_iterations is not None and iterations >= self.max_iterations:
        return False
    return self.max_execution_time is None or time_elapsed < self.max_execution_time
```

### 5.2 时间限制（同步版本）

```python
# agent.py, line 1570-1622 — 同步版通过 time.time() 检查
def _call(self, inputs, run_manager=None):
    iterations = 0
    time_elapsed = 0.0
    start_time = time.time()
    while self._should_continue(iterations, time_elapsed):
        ...
        iterations += 1
        time_elapsed = time.time() - start_time  # 每次循环更新
    # 超出限制
    output = self._action_agent.return_stopped_response(
        self.early_stopping_method, intermediate_steps, **inputs,
    )
```

### 5.3 时间限制（异步版本 — asyncio 超时）

```python
# agent.py, line 1624-1695 — 异步版使用 asyncio_timeout 实现更精确的超时
async def _acall(self, inputs, run_manager=None):
    try:
        async with asyncio_timeout(self.max_execution_time):
            while self._should_continue(iterations, time_elapsed):
                ...
    except (TimeoutError, asyncio.TimeoutError):
        # 超时时优雅停止
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs,
        )
        return await self._areturn(output, intermediate_steps, run_manager=run_manager)
```

**同步 vs 异步的差异：**
- 同步版本：通过循环开始时检查时间来判断，粒度较粗（无法打断正在执行的 LLM 调用或工具调用）
- 异步版本：使用 `asyncio_timeout` 可以在任何 `await` 点中断执行

### 5.4 早停策略（Early Stopping）

```python
# agent.py, line 1032-1041
early_stopping_method: str = "force"
"""
`"force"` — 返回固定消息"Agent stopped due to iteration limit"
`"generate"` — 让 LLM 基于已有步骤做最后一次总结
"""

# agent.py, line 920-973 — Agent.return_stopped_response 的 "generate" 策略
def return_stopped_response(self, early_stopping_method, intermediate_steps, **kwargs):
    if early_stopping_method == "force":
        return AgentFinish(
            {"output": "Agent stopped due to iteration limit or time limit."}, "",
        )
    if early_stopping_method == "generate":
        # 构建已有步骤的思考链
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        thoughts += "\n\nI now need to return a final answer based on the previous steps:"

        # 让 LLM 做最后一次推理
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        full_output = self.llm_chain.predict(**full_inputs)

        # 尝试解析最终答案
        parsed_output = self.output_parser.parse(full_output)
        if isinstance(parsed_output, AgentFinish):
            return parsed_output
        # 如果仍然无法解析出 AgentFinish，直接使用原始输出
        return AgentFinish({"output": full_output}, full_output)
```

**设计取舍：**
- `"force"` 更安全但用户体验差（突然中断）
- `"generate"` 更优雅但有额外 LLM 调用成本，且不保证一定能解析出有效答案

---

## 回调系统中的错误可观测性

### 5.1 完整的生命周期错误钩子

每个组件类型都有三个阶段的回调：`on_*_start` → `on_*_end` 或 `on_*_error`

```python
# callbacks/base.py — 对称的成功/错误回调
class BaseCallbackHandler:
    # LLM
    def on_llm_start(...): ...
    def on_llm_end(...): ...
    def on_llm_error(self, error: BaseException, *, run_id, ...): ...

    # Chain
    def on_chain_start(...): ...
    def on_chain_end(...): ...
    def on_chain_error(self, error: BaseException, *, run_id, ...): ...

    # Tool
    def on_tool_start(...): ...
    def on_tool_end(...): ...
    def on_tool_error(self, error: BaseException, *, run_id, ...): ...

    # Retriever
    def on_retriever_start(...): ...
    def on_retriever_end(...): ...
    def on_retriever_error(self, error: BaseException, *, run_id, ...): ...

    # 重试
    def on_retry(self, retry_state: RetryCallState, *, run_id, ...): ...
```

### 5.2 回调错误控制

```python
# callbacks/base.py, line 445-449
class BaseCallbackHandler:
    raise_error: bool = False  # 回调自身的错误是否应该中断执行
    run_inline: bool = False   # 是否在主线程中运行（用于调试）
```

**设计考量：** 默认 `raise_error=False`，这意味着回调处理器中的错误不会影响 Agent 的正常执行。这是**防御性编程**——可观测性代码不应该干扰业务逻辑。

### 5.3 回调降级机制

```python
# callbacks/base.py, line 270-300
def on_chat_model_start(self, serialized, messages, *, run_id, ...):
    """NotImplementedError is thrown intentionally.
    Callback handler will fall back to on_llm_start if this exception is thrown."""
    msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
    raise NotImplementedError(msg)
```

**设计模式：** 使用 `NotImplementedError` 作为信号触发回退（fallback）——如果 handler 没有实现 `on_chat_model_start`，系统会自动降级到 `on_llm_start`。这是一种优雅的**渐进增强（Progressive Enhancement）**模式。

### 5.4 回调中的选择性忽略

```python
# callbacks/base.py, line 451-484
class BaseCallbackHandler:
    @property
    def ignore_llm(self) -> bool: return False
    @property
    def ignore_chain(self) -> bool: return False
    @property
    def ignore_agent(self) -> bool: return False
    @property
    def ignore_retriever(self) -> bool: return False
    @property
    def ignore_retry(self) -> bool: return False
    @property
    def ignore_custom_event(self) -> bool: return False
```

**使用场景：** 当只需要监控工具执行而不关心 LLM 调用时，设置 `ignore_llm=True` 可以减少不必要的回调开销。

---

## 弹性模式总结

### 模式 1: 错误转化为观察（Error as Observation）

```
LLM 输出解析失败
  → OutputParserException 捕获
  → 创建 _Exception 工具调用
  → 错误信息作为 observation 添加到中间步骤
  → 下次 plan() 时 LLM 看到错误并自我纠正
```

**这是 LangChain Agent 最核心的弹性机制。**

### 模式 2: 安全降级（Graceful Degradation）

```
Agent 请求不存在的工具
  → 不抛异常
  → 使用 InvalidTool 返回可用工具列表
  → Agent 在下次迭代中选择正确的工具
```

### 模式 3: 双重保险（Belt and Suspenders）

```
迭代限制 (max_iterations=15)  ← 防止 Agent 无限循环
  +
时间限制 (max_execution_time)  ← 防止单步骤耗时过长
  +
早停策略 (early_stopping_method)  ← 超限时的响应策略
```

### 模式 4: 多态错误处理（Polymorphic Error Handling）

```python
# bool → 自动处理（默认消息）
handle_parsing_errors = True

# str → 固定消息
handle_parsing_errors = "请重新格式化你的输出"

# Callable → 自定义逻辑
handle_parsing_errors = lambda e: f"解析错误: {e}, 请使用正确的格式"
```

**三种模式统一的类型签名：** `bool | str | Callable[[Exception], str]`

### 模式 5: 延迟抛出 + 确保清理（Deferred Raise with Guaranteed Cleanup）

```python
# tools/base.py — 确保错误回调总是被调用
error_to_raise = None
try:
    response = self._run(...)
except ToolException as e:
    if self.handle_tool_error:
        content = _handle_tool_error(e, ...)
    else:
        error_to_raise = e  # 不立即 raise
except (Exception, KeyboardInterrupt) as e:
    error_to_raise = e      # 不立即 raise

# 确保回调通知
if error_to_raise:
    run_manager.on_tool_error(error_to_raise)  # 总是执行
    raise error_to_raise                        # 然后再 raise
```

### 模式 6: 中间步骤裁剪（Memory Pressure Relief）

```python
# 当 Agent 运行时间过长，中间步骤会占用越来越多的 context window
trim_intermediate_steps = 5  # 只保留最近 5 步
# 或
trim_intermediate_steps = custom_trimmer  # 自定义裁剪逻辑
```

---

## 错误处理配置推荐

### 开发环境

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                    # 详细输出
    handle_parsing_errors=True,      # 自动恢复解析错误
    max_iterations=25,               # 较大的迭代限制
    return_intermediate_steps=True,  # 返回所有中间步骤用于调试
)
```

### 生产环境

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=lambda e: f"请重新格式化: {str(e)[:200]}",  # 截断错误信息
    max_iterations=10,               # 更严格的限制
    max_execution_time=30.0,         # 30 秒超时
    early_stopping_method="force",   # 超时时强制停止（不做额外 LLM 调用）
    trim_intermediate_steps=5,       # 控制内存
)

# 为每个 Tool 配置错误处理
for tool in tools:
    tool.handle_tool_error = True                  # 业务错误返回给 Agent
    tool.handle_validation_error = True            # 验证错误返回给 Agent
```

---

## 关键教训

1. **永远不要信任 LLM 的输出格式** — `handle_parsing_errors=True` 是生产环境必备
2. **迭代限制是安全底线** — `max_iterations=None` 是极其危险的配置
3. **错误信息是 Agent 学习的素材** — 将错误转化为观察，让 Agent 自我纠正
4. **回调异常不应中断业务** — `raise_error=False` 是正确的默认值
5. **异步版本更优雅** — `asyncio_timeout` 可以在 await 点中断，而同步版只能在循环开始时检查
6. **防御未知工具调用** — 使用 `InvalidTool` 返回可用工具列表，而非直接崩溃
7. **区分可恢复与不可恢复的错误** — `ToolException` vs 其他 `Exception` 的分层处理
