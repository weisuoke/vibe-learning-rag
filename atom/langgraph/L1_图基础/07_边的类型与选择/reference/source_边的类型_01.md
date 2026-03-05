---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/graph/_branch.py
  - libs/langgraph/langgraph/graph/_node.py
analyzed_at: 2026-02-25
knowledge_point: 07_边的类型与选择
---

# 源码分析：LangGraph 边的类型与实现

## 分析的文件

- `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py` - StateGraph 核心类
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/_branch.py` - 条件边实现
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/_node.py` - 节点定义

## 关键发现

### 1. 普通边（Normal Edge）

**定义位置**: `state.py:785-837`

```python
def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
    """Add a directed edge from the start node (or list of start nodes) to the end node.

    When a single start node is provided, the graph will wait for that node to complete
    before executing the end node. When multiple start nodes are provided,
    the graph will wait for ALL of the start nodes to complete before executing the end node.
    """
```

**核心特征**:
- 直接连接两个节点
- 单个起始节点：`add_edge("A", "B")`
- 多个起始节点（等待边）：`add_edge(["A", "B"], "C")`
- 存储在 `self.edges` 集合中（单起始）或 `self.waiting_edges` 集合中（多起始）

**实现细节**:
```python
# 单起始节点
if isinstance(start_key, str):
    self.edges.add((start_key, end_key))
    return self

# 多起始节点（等待边）
self.waiting_edges.add((tuple(start_key), end_key))
```

### 2. 条件边（Conditional Edge）

**定义位置**: `state.py:839-887`

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    """Add a conditional edge from the starting node to any number of destination nodes.

    Args:
        source: The starting node. This conditional edge will run when exiting this node.
        path: The callable that determines the next node or nodes.
        path_map: Optional mapping of paths to node names.
    """
```

**核心特征**:
- 根据路径函数动态决定下一个节点
- 路径函数返回节点名称或 `Send` 对象
- 支持返回多个目标节点（并行执行）
- 存储在 `self.branches[source][name]` 中

**BranchSpec 实现** (`_branch.py:83-226`):
```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None

    def _route(self, input: Any, config: RunnableConfig, ...) -> Runnable:
        # 执行路径函数
        result = self.path.invoke(value, config)
        # 根据结果路由到目标节点
        if self.ends:
            destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
        else:
            destinations = result
```

**路径函数类型推断**:
- 如果路径函数有 `Literal` 返回类型注解，自动推断 `path_map`
- 如果提供 `path_map`，将路径函数返回值映射到节点名称
- 如果都没有，路径函数必须直接返回节点名称

### 3. 等待边（Waiting Edge）

**定义位置**: `state.py:826-836`

```python
# 当 start_key 是列表时
for start in start_key:
    if start == END:
        raise ValueError("END cannot be a start node")
    if start not in self.nodes:
        raise ValueError(f"Need to add_node `{start}` first")
if end_key == START:
    raise ValueError("START cannot be an end node")
if end_key != END and end_key not in self.nodes:
    raise ValueError(f"Need to add_node `{end_key}` first")

self.waiting_edges.add((tuple(start_key), end_key))
```

**核心特征**:
- 等待多个节点全部完成后才执行目标节点
- 使用 `NamedBarrierValue` 或 `NamedBarrierValueAfterFinish` 实现同步
- 编译时创建 `join:start1+start2:end` 通道

**编译时实现** (`state.py:1306-1321`):
```python
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        # 普通边：直接写入目标节点的触发通道
        if end != END:
            self.nodes[starts].writers.append(
                ChannelWrite((ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),))
            )
    elif end != END:
        # 等待边：创建 barrier 通道
        channel_name = f"join:{'+'.join(starts)}:{end}"
        if self.builder.nodes[end].defer:
            self.channels[channel_name] = NamedBarrierValueAfterFinish(str, set(starts))
        else:
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
        # 订阅通道
        self.nodes[end].triggers.append(channel_name)
        # 发布到通道
        for start in starts:
            self.nodes[start].writers.append(
                ChannelWrite((ChannelWriteEntry(channel_name, start),))
            )
```

### 4. 序列边（Sequence Edge）

**定义位置**: `state.py:889-934`

```python
def add_sequence(
    self,
    nodes: Sequence[
        StateNode[NodeInputT, ContextT]
        | tuple[str, StateNode[NodeInputT, ContextT]]
    ],
) -> Self:
    """Add a sequence of nodes that will be executed in the provided order.

    Args:
        nodes: A sequence of `StateNode` (callables that accept a `state` arg) or `(name, StateNode)` tuples.
    """
    if len(nodes) < 1:
        raise ValueError("Sequence requires at least one node.")

    previous_name: str | None = None
    for node in nodes:
        if isinstance(node, tuple) and len(node) == 2:
            name, node = node
        else:
            name = _get_node_name(node)

        if name in self.nodes:
            raise ValueError(f"Node names must be unique: node with the name '{name}' already exists.")

        self.add_node(name, node)
        if previous_name is not None:
            self.add_edge(previous_name, name)

        previous_name = name
```

**核心特征**:
- 语法糖，简化顺序节点的添加
- 自动调用 `add_node()` 和 `add_edge()`
- 支持自动推断节点名称或手动指定

### 5. 动态路由（Send）

**定义位置**: `_branch.py:202-210`

```python
if self.ends:
    destinations: Sequence[Send | str] = [
        r if isinstance(r, Send) else self.ends[r] for r in result
    ]
else:
    destinations = cast(Sequence[Send | str], result)

if any(p.node == END for p in destinations if isinstance(p, Send)):
    raise InvalidUpdateError("Cannot send a packet to the END node")
```

**核心特征**:
- 条件边可以返回 `Send` 对象
- `Send(node, state)` 允许向目标节点发送自定义状态
- 支持动态并行执行（fan-out）

## 边的内部实现机制

### 通道系统

LangGraph 使用通道（Channel）系统实现边的通信：

1. **普通边**: 写入 `branch:to:{target}` 通道
2. **条件边**: 通过 `BranchSpec.run()` 动态写入通道
3. **等待边**: 使用 `NamedBarrierValue` 通道同步多个起始节点

### 编译过程

在 `compile()` 时，边被转换为通道写入操作：

```python
# state.py:1143-1151
for start, end in self.edges:
    compiled.attach_edge(start, end)

for starts, end in self.waiting_edges:
    compiled.attach_edge(starts, end)

for start, branches in self.branches.items():
    for name, branch in branches.items():
        compiled.attach_branch(start, name, branch)
```

## 边的选择策略

### 1. 普通边 vs 条件边

- **普通边**: 固定路由，适合确定性流程
- **条件边**: 动态路由，适合需要根据状态决策的场景

### 2. 单起始 vs 多起始

- **单起始**: 节点完成后立即触发下一个节点
- **多起始**: 等待所有起始节点完成后才触发（AND 逻辑）

### 3. 序列边

- 语法糖，简化线性流程的定义
- 内部使用普通边实现

## 关键代码片段

### 条件边路由逻辑

```python
# _branch.py:146-167
def _route(
    self,
    input: Any,
    config: RunnableConfig,
    *,
    reader: Callable[[RunnableConfig], Any] | None,
    writer: _Writer,
) -> Runnable:
    if reader:
        value = reader(config)
        # passthrough additional keys from node to branch
        if isinstance(value, dict) and isinstance(input, dict) and self.input_schema is None:
            value = {**input, **value}
    else:
        value = input
    result = self.path.invoke(value, config)
    return self._finish(writer, input, result, config)
```

### 等待边同步机制

```python
# state.py:1307-1321
channel_name = f"join:{'+'.join(starts)}:{end}"
if self.builder.nodes[end].defer:
    self.channels[channel_name] = NamedBarrierValueAfterFinish(str, set(starts))
else:
    self.channels[channel_name] = NamedBarrierValue(str, set(starts))
self.nodes[end].triggers.append(channel_name)
for start in starts:
    self.nodes[start].writers.append(
        ChannelWrite((ChannelWriteEntry(channel_name, start),))
    )
```

## 总结

LangGraph 的边系统设计精巧，通过通道机制实现了：
1. **灵活的路由**: 支持固定路由和动态路由
2. **并行控制**: 支持等待多个节点完成（AND）和动态并行执行（fan-out）
3. **类型安全**: 通过类型注解自动推断路由映射
4. **可扩展性**: 通过 `Send` 对象支持自定义状态传递
