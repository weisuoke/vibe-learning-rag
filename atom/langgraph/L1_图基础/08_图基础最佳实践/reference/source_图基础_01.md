---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/langgraph/graph/
analyzed_files:
  - state.py (前200行)
  - _node.py (完整)
  - _branch.py (完整)
analyzed_at: 2026-02-25
knowledge_point: 08_图基础最佳实践
---

# 源码分析：StateGraph 核心实现

## 分析的文件

- `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py` - StateGraph 核心类实现
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/_node.py` - 节点定义和规范
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/_branch.py` - 条件路由实现

## 关键发现

### 1. StateGraph 设计模式

**Builder 模式**:
```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    The signature of each node is `State -> Partial<State>`.
    """
```

**关键特性**:
- StateGraph 是 Builder 类,不能直接执行
- 必须调用 `.compile()` 创建可执行图
- 编译后支持 `invoke()`, `stream()`, `astream()`, `ainvoke()` 等方法

**最佳实践**:
- ✅ 使用 Builder 模式构建图,保持构建和执行分离
- ✅ 编译前完成所有节点和边的定义
- ✅ 编译后的图是不可变的

### 2. 状态 Schema 设计

**状态定义方式**:
```python
# 方式1: TypedDict
class State(TypedDict):
    x: Annotated[list, reducer]

# 方式2: Pydantic BaseModel
class State(BaseModel):
    x: list
```

**Reducer 函数**:
```python
def reducer(a: list, b: int | None) -> list:
    """聚合多个节点的状态更新"""
    if b is not None:
        return a + [b]
    return a
```

**状态验证**:
```python
def _warn_invalid_state_schema(schema: type[Any] | Any) -> None:
    """验证状态 schema 的有效性"""
    if isinstance(schema, type):
        return
    if typing.get_args(schema):
        return
    warnings.warn(
        f"Invalid state_schema: {schema}. Expected a type or Annotated[type, reducer]. "
        "Please provide a valid schema to ensure correct updates."
    )
```

**最佳实践**:
- ✅ 使用 TypedDict 或 Pydantic 明确定义状态类型
- ✅ 使用 Annotated[type, reducer] 定义需要聚合的字段
- ✅ Reducer 函数签名: `(Value, Value) -> Value`
- ✅ 节点可以返回部分状态更新(不需要返回完整状态)

### 3. Context Schema (运行时上下文)

**Context 设计**:
```python
class Context(TypedDict):
    r: float  # 不可变的运行时参数

graph = StateGraph(state_schema=State, context_schema=Context)

def node(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)  # 访问运行时上下文
    x = state["x"][-1]
    next_value = x * r * (1 - x)
    return {"x": next_value}
```

**最佳实践**:
- ✅ 使用 context_schema 传递不可变的运行时参数
- ✅ Context 适合存储: user_id, db_conn, 配置参数等
- ✅ Context 在整个图执行过程中保持不变
- ✅ 通过 Runtime[Context] 访问上下文

### 4. 节点函数签名设计

**多种节点签名**:
```python
# 1. 基础节点: 只接收 state
_Node[NodeInputT]
def node(state: State) -> dict: ...

# 2. 带配置: 接收 state + config
_NodeWithConfig[NodeInputT]
def node(state: State, config: RunnableConfig) -> dict: ...

# 3. 带流式输出: 接收 state + writer
_NodeWithWriter[NodeInputT]
def node(state: State, *, writer: StreamWriter) -> dict: ...

# 4. 带存储: 接收 state + store
_NodeWithStore[NodeInputT]
def node(state: State, *, store: BaseStore) -> dict: ...

# 5. 带运行时上下文: 接收 state + runtime
_NodeWithRuntime[NodeInputT, ContextT]
def node(state: State, *, runtime: Runtime[Context]) -> dict: ...

# 6. 组合形式: 可以同时接收多个参数
_NodeWithConfigWriterStore[NodeInputT]
def node(state: State, *, config: RunnableConfig, writer: StreamWriter, store: BaseStore) -> dict: ...
```

**StateNodeSpec 配置**:
```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
```

**最佳实践**:
- ✅ 根据节点需求选择合适的签名
- ✅ 简单节点只接收 state,保持简洁
- ✅ 需要配置时使用 config 参数
- ✅ 需要流式输出时使用 writer 参数
- ✅ 需要持久化存储时使用 store 参数
- ✅ 需要运行时上下文时使用 runtime 参数
- ✅ 使用 StateNodeSpec 配置重试策略、缓存策略等

### 5. 条件路由设计

**BranchSpec 实现**:
```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None
```

**路由函数类型推断**:
```python
@classmethod
def from_path(
    cls,
    path: Runnable[Any, Hashable | list[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None,
    infer_schema: bool = False,
) -> BranchSpec:
    # 从函数返回类型推断路由目标
    if rtn_type := get_type_hints(func).get("return"):
        if get_origin(rtn_type) is Literal:
            path_map_ = {name: name for name in get_args(rtn_type)}
```

**路由函数示例**:
```python
# 使用 Literal 类型提示
def route(state: State) -> Literal["node_a", "node_b"]:
    if state["x"] > 0:
        return "node_a"
    return "node_b"

# 返回多个目标
def route(state: State) -> list[str]:
    return ["node_a", "node_b"]  # 并行执行

# 使用 Send 动态路由
def route(state: State) -> Send | list[Send]:
    return Send("node_a", {"x": state["x"] + 1})
```

**路由验证**:
```python
def _finish(self, writer, input, result, config):
    # 验证路由结果
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")
    if any(p.node == END for p in destinations if isinstance(p, Send)):
        raise InvalidUpdateError("Cannot send a packet to the END node")
```

**最佳实践**:
- ✅ 使用 Literal 类型提示明确路由选项
- ✅ 路由函数可以返回单个值或列表
- ✅ 返回列表时,多个目标会并行执行
- ✅ 使用 Send 机制进行动态路由和参数传递
- ✅ 路由函数不能返回 None 或 START
- ✅ 不能向 END 节点发送 Send 消息

### 6. 错误处理机制

**错误类型**:
```python
from langgraph.errors import (
    ErrorCode,
    InvalidUpdateError,
    ParentCommand,
    create_error_message,
)
```

**验证机制**:
- 状态 schema 验证 (_warn_invalid_state_schema)
- 路由目标验证 (不能是 None 或 START)
- Send 目标验证 (不能是 END)

**最佳实践**:
- ✅ 使用明确的错误类型
- ✅ 在构建时进行验证,而不是运行时
- ✅ 提供清晰的错误消息

## 代码片段

### 完整的 StateGraph 使用示例

```python
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


def reducer(a: list, b: int | None) -> list:
    if b is not None:
        return a + [b]
    return a


class State(TypedDict):
    x: Annotated[list, reducer]


class Context(TypedDict):
    r: float


graph = StateGraph(state_schema=State, context_schema=Context)


def node(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)
    x = state["x"][-1]
    next_value = x * r * (1 - x)
    return {"x": next_value}


graph.add_node("A", node)
graph.set_entry_point("A")
graph.set_finish_point("A")
compiled = graph.compile()

step1 = compiled.invoke({"x": 0.5}, context={"r": 3.0})
# {'x': [0.5, 0.75]}
```

## 总结

从源码分析中提取的核心最佳实践:

1. **图设计**: 使用 Builder 模式,保持构建和执行分离
2. **状态设计**: 使用 TypedDict/Pydantic + Annotated[type, reducer]
3. **上下文设计**: 使用 context_schema 传递不可变参数
4. **节点设计**: 根据需求选择合适的节点签名
5. **路由设计**: 使用 Literal 类型提示 + Send 机制
6. **错误处理**: 构建时验证 + 明确的错误类型
