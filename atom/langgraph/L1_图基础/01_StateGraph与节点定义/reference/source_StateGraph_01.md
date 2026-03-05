---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py
analyzed_files:
  - sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py (lines 1-400)
  - sourcecode/langgraph/libs/langgraph/langgraph/graph/__init__.py
  - sourcecode/langgraph/libs/langgraph/langgraph/graph/_node.py
analyzed_at: 2026-02-25
knowledge_point: 01_StateGraph与节点定义
---

# 源码分析：StateGraph 核心实现

## 分析的文件

### 1. `state.py` - StateGraph 主类
- **路径**: `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py`
- **行数**: 1-400（部分）
- **作用**: StateGraph 类的核心实现

### 2. `__init__.py` - 公共 API
- **路径**: `sourcecode/langgraph/libs/langgraph/langgraph/graph/__init__.py`
- **作用**: 定义公共导出接口

### 3. `_node.py` - 节点协议定义
- **路径**: `sourcecode/langgraph/libs/langgraph/langgraph/graph/_node.py`
- **作用**: 定义节点函数的各种协议和规范

## 关键发现

### 1. StateGraph 类的设计哲学

**Builder 模式**（state.py:112-127）:
```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    !!! warning
        `StateGraph` is a builder class and cannot be used directly for execution.
        You must first call `.compile()` to create an executable graph that supports
        methods like `invoke()`, `stream()`, `astream()`, and `ainvoke()`.
    """
```

**关键点**:
- StateGraph 是 builder，不可直接执行
- 必须调用 `.compile()` 生成 `CompiledStateGraph`
- 泛型设计：`StateT`, `ContextT`, `InputT`, `OutputT`

### 2. StateGraph 的核心属性

**内部数据结构**（state.py:183-195）:
```python
edges: set[tuple[str, str]]                    # 边集合
nodes: dict[str, StateNodeSpec[Any, ContextT]] # 节点字典
branches: defaultdict[str, dict[str, BranchSpec]] # 分支字典
channels: dict[str, BaseChannel]               # 通道字典
managed: dict[str, ManagedValueSpec]           # 托管值字典
schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]] # Schema 字典
waiting_edges: set[tuple[tuple[str, ...], str]] # 等待边集合

compiled: bool                                 # 编译状态
state_schema: type[StateT]                     # 状态 Schema
context_schema: type[ContextT] | None          # 上下文 Schema
input_schema: type[InputT]                     # 输入 Schema
output_schema: type[OutputT]                   # 输出 Schema
```

### 3. StateGraph 初始化过程

**构造函数**（state.py:197-250）:
```python
def __init__(
    self,
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> None:
    # 初始化内部数据结构
    self.nodes = {}
    self.edges = set()
    self.branches = defaultdict(dict)
    self.schemas = {}
    self.channels = {}
    self.managed = {}
    self.compiled = False
    self.waiting_edges = set()

    # 设置 schema
    self.state_schema = state_schema
    self.input_schema = cast(type[InputT], input_schema or state_schema)
    self.output_schema = cast(type[OutputT], output_schema or state_schema)
    self.context_schema = context_schema

    # 添加 schema 到内部字典
    self._add_schema(self.state_schema)
    self._add_schema(self.input_schema, allow_managed=False)
    self._add_schema(self.output_schema, allow_managed=False)
```

**关键点**:
- `state_schema` 是必需参数
- `input_schema` 和 `output_schema` 默认使用 `state_schema`
- `context_schema` 是可选的，用于运行时上下文
- `_add_schema` 方法处理 schema 并提取 channels 和 managed values

### 4. add_node 方法签名

**方法重载**（state.py:289-354）:
```python
@overload
def add_node(
    self,
    node: StateNode[NodeInputT, ContextT],
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> Self:
    """Add a new node to the `StateGraph`, input schema is inferred as the state schema.

    Will take the name of the function/runnable as the node name.
    """
```

**参数说明**:
- `node`: 节点函数或 Runnable
- `defer`: 是否延迟执行
- `metadata`: 节点元数据
- `input_schema`: 节点输入 schema（默认使用图的 state_schema）
- `retry_policy`: 重试策略
- `cache_policy`: 缓存策略
- `destinations`: 目标节点（用于图渲染）

**返回值**: `Self`，支持链式调用

### 5. 节点函数协议

**9种节点签名**（_node.py:16-81）:
```python
# 1. 基础节点：只接受 state
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...

# 2. 带 config 的节点
class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...

# 3. 带 writer 的节点
class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...

# 4. 带 store 的节点
class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...

# 5. 带 writer 和 store 的节点
class _NodeWithWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, writer: StreamWriter, store: BaseStore
    ) -> Any: ...

# 6. 带 config 和 writer 的节点
class _NodeWithConfigWriter(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, writer: StreamWriter
    ) -> Any: ...

# 7. 带 config 和 store 的节点
class _NodeWithConfigStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, store: BaseStore
    ) -> Any: ...

# 8. 带 config、writer 和 store 的节点
class _NodeWithConfigWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self,
        state: NodeInputT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...

# 9. 带 runtime 的节点（新版本推荐）
class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...

# StateNode 类型别名
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | _NodeWithStore[NodeInputT]
    | _NodeWithWriterStore[NodeInputT]
    | _NodeWithConfigWriter[NodeInputT]
    | _NodeWithConfigStore[NodeInputT]
    | _NodeWithConfigWriterStore[NodeInputT]
    | _NodeWithRuntime[NodeInputT, ContextT]
    | Runnable[NodeInputT, Any]
)
```

### 6. StateNodeSpec 数据类

**节点规范**（_node.py:84-93）:
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

### 7. 公共 API 导出

**__init__.py**:
```python
from langgraph.constants import END, START
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.graph.state import StateGraph

__all__ = (
    "END",
    "START",
    "StateGraph",
    "add_messages",
    "MessagesState",
    "MessageGraph",
)
```

**关键常量**:
- `START`: 图的入口点
- `END`: 图的出口点

## 架构设计洞察

### 1. Builder 模式的应用
- StateGraph 作为 builder，负责构建图结构
- 通过 `add_node`, `add_edge` 等方法逐步构建
- 最后通过 `compile()` 生成可执行的 `CompiledStateGraph`

### 2. 泛型设计的优势
- `StateT`: 状态类型，通常是 TypedDict
- `ContextT`: 运行时上下文类型
- `InputT`: 输入类型（默认等于 StateT）
- `OutputT`: 输出类型（默认等于 StateT）
- 提供类型安全和 IDE 支持

### 3. Channel 机制
- 状态通过 Channel 进行通信
- 不同的 Channel 类型：`LastValue`, `EphemeralValue`, `BinaryOperatorAggregate` 等
- Channel 支持 reducer 函数进行状态聚合

### 4. 节点函数的灵活性
- 支持 9 种不同的节点签名
- 可以注入 `config`, `writer`, `store`, `runtime` 等依赖
- 支持 Runnable 接口，与 LangChain 无缝集成

### 5. Schema 驱动的设计
- 通过 TypedDict 定义状态结构
- 使用 Annotated 类型添加 reducer 函数
- 自动提取 channels 和 managed values

## 需要深入研究的部分

1. **compile() 方法的实现**
   - 如何将 builder 转换为可执行图
   - Pregel 执行引擎的工作原理

2. **Channel 机制的细节**
   - 不同 Channel 类型的使用场景
   - Reducer 函数的工作原理

3. **边的添加方法**
   - `add_edge` 的实现
   - `add_conditional_edges` 的实现
   - 条件路由的工作原理

4. **图的执行流程**
   - `invoke`, `stream`, `astream` 等方法
   - 异步执行的支持

5. **START 和 END 节点**
   - 特殊节点的实现
   - 入口点和出口点的设置

## 代码片段

### 示例 1：基础 StateGraph 创建（state.py:141-180）
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

### 示例 2：add_node 使用（state.py:328-349）
```python
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph


class State(TypedDict):
    x: int


def my_node(state: State, config: RunnableConfig) -> State:
    return {"x": state["x"] + 1}


builder = StateGraph(State)
builder.add_node(my_node)  # node name will be 'my_node'
builder.add_edge(START, "my_node")
graph = builder.compile()
graph.invoke({"x": 1})
# {'x': 2}
```

## 依赖关系

### 核心依赖
- `langchain_core.runnables`: Runnable 接口
- `langgraph.checkpoint.base`: Checkpoint 机制
- `langgraph.store.base`: Store 接口
- `langgraph.pregel`: Pregel 执行引擎
- `langgraph.channels`: Channel 通信机制
- `pydantic`: 类型验证
- `typing_extensions`: 类型扩展

### 内部模块
- `langgraph._internal._constants`: 内部常量
- `langgraph._internal._fields`: 字段处理
- `langgraph._internal._pydantic`: Pydantic 工具
- `langgraph._internal._runnable`: Runnable 工具
- `langgraph._internal._typing`: 类型工具
