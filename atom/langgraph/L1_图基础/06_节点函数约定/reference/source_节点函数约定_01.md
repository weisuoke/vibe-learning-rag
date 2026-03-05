---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py
analyzed_files:
  - state.py (lines 1-1708)
analyzed_at: 2026-02-25
knowledge_point: 节点函数约定
---

# 源码分析：StateGraph 节点函数定义

## 分析的文件
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py` - StateGraph 核心实现

## 关键发现

### 1. 节点函数签名约定

从 `add_node` 方法的实现可以看到，节点函数的签名约定：

```python
# 基础签名
def node_function(state: State) -> dict:
    return {"key": "value"}

# 带 Config 参数
def node_function(state: State, config: RunnableConfig) -> dict:
    return {"key": "value"}

# 带 Runtime 参数（Context）
def node_function(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)
    return {"key": "value"}
```

**源码位置**：`state.py:289-354`

### 2. 节点函数类型推断

LangGraph 会自动推断节点函数的输入输出类型：

```python
# state.py:698-743
if (
    isfunction(action)
    or ismethod(action)
    or ismethod(getattr(action, "__call__", None))
) and (
    hints := get_type_hints(getattr(action, "__call__"))
    or get_type_hints(action)
):
    if input_schema is None:
        first_parameter_name = next(
            iter(
                inspect.signature(
                    cast(FunctionType, action)
                ).parameters.keys()
            )
        )
        if input_hint := hints.get(first_parameter_name):
            if isinstance(input_hint, type) and get_type_hints(input_hint):
                inferred_input_schema = input_hint
```

**关键点**：
- 自动推断第一个参数的类型作为 input_schema
- 支持 TypedDict 类型推断
- 支持 Command 返回值类型推断

### 3. 节点函数返回值处理

节点函数可以返回多种类型：

```python
# state.py:1206-1242
def _get_updates(
    input: None | dict | Any,
) -> Sequence[tuple[str, Any]] | None:
    if input is None:
        return None
    elif isinstance(input, dict):
        return [(k, v) for k, v in input.items() if k in output_keys]
    elif isinstance(input, Command):
        if input.graph == Command.PARENT:
            return None
        return [
            (k, v) for k, v in input._update_as_tuples() if k in output_keys
        ]
    elif (
        isinstance(input, (list, tuple))
        and input
        and any(isinstance(i, Command) for i in input)
    ):
        updates: list[tuple[str, Any]] = []
        for i in input:
            if isinstance(i, Command):
                if i.graph == Command.PARENT:
                    continue
                updates.extend(
                    (k, v) for k, v in i._update_as_tuples() if k in output_keys
                )
            else:
                updates.extend(_get_updates(i) or ())
        return updates
    elif (t := type(input)) and get_cached_annotated_keys(t):
        return get_update_as_tuples(input, output_keys)
    else:
        msg = create_error_message(
            message=f"Expected dict, got {input}",
            error_code=ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE,
        )
        raise InvalidUpdateError(msg)
```

**支持的返回值类型**：
1. `dict` - 部分状态更新
2. `Command` - 控制流程
3. `None` - 不更新状态
4. `list[Command | dict]` - 多个更新
5. 带 Annotated 的自定义类型

### 4. 节点配置选项

节点可以配置多个选项：

```python
# state.py:289-301
def add_node(
    self,
    node: str | StateNode[NodeInputT, ContextT],
    action: StateNode[NodeInputT, ContextT] | None = None,
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: type[NodeInputT] | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> Self:
```

**配置选项**：
- `defer`: 延迟执行
- `metadata`: 元数据
- `input_schema`: 输入 schema
- `retry_policy`: 重试策略
- `cache_policy`: 缓存策略
- `destinations`: 目标节点（用于 Command）

### 5. 异步节点支持

LangGraph 自动检测并处理异步节点：

```python
# _call.py:175-192
if is_async_callable(func):
    run = RunnableCallable(
        None, func, name=func.__name__, trace=False, recurse=False
    )
else:
    afunc = functools.update_wrapper(
        functools.partial(run_in_executor, None, func), func
    )
    run = RunnableCallable(
        func,
        afunc,
        name=func.__name__,
        trace=False,
        recurse=False,
    )
```

**关键点**：
- 自动检测 `async` 函数
- 同步函数会被包装成异步执行
- 使用 `run_in_executor` 在线程池中执行同步函数

## 代码片段

### 节点函数定义示例

```python
# 基础节点函数
def my_node(state: State, config: RunnableConfig) -> State:
    return {"x": state["x"] + 1}

# 异步节点函数
async def my_async_node(state: State) -> dict:
    result = await some_async_api()
    return {"result": result}

# 使用 Command 的节点函数
def my_command_node(state: State) -> Command[Literal["next", "end"]]:
    if state["x"] > 10:
        return Command(goto="end")
    return Command(goto="next", update={"x": state["x"] + 1})
```

### 节点添加示例

```python
builder = StateGraph(State)

# 基础添加
builder.add_node("my_node", my_node)

# 带配置添加
builder.add_node(
    "my_node",
    my_node,
    retry_policy=RetryPolicy(max_attempts=3),
    metadata={"description": "My node"}
)

# 带 input_schema 添加
builder.add_node(
    "my_node",
    my_node,
    input_schema=NodeInput
)
```
