---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/types.py
  - libs/langgraph/langgraph/graph/_branch.py
  - libs/langgraph/langgraph/pregel/main.py
analyzed_at: 2026-02-27
knowledge_point: 01_并行执行与分支合并
---

# 源码分析：LangGraph 并行执行机制

## 分析的文件

- `libs/langgraph/langgraph/types.py` - Send 类定义
- `libs/langgraph/langgraph/graph/_branch.py` - 分支路由实现
- `libs/langgraph/langgraph/pregel/main.py` - Pregel 算法实现

## 关键发现

### 1. Send 类 - 并行执行的核心

**位置**: `types.py:289-362`

**定义**:
```python
class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.

    One such example is a "map-reduce" workflow where your graph invokes
    the same node multiple times in parallel with different states,
    before aggregating the results back into the main graph's state.
    """

    __slots__ = ("node", "arg")

    node: str  # 目标节点名称
    arg: Any   # 发送的状态或消息
```

**核心特性**:
1. **动态节点调用**: 在条件边中动态调用节点
2. **自定义状态**: 发送的状态可以与图的核心状态不同
3. **Map-Reduce 模式**: 支持同一节点多次并行调用，使用不同的状态

**使用示例** (来自源码文档):
```python
from typing import Annotated
from langgraph.types import Send
from langgraph.graph import END, START
from langgraph.graph import StateGraph
import operator

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

def continue_to_jokes(state: OverallState):
    # 返回多个 Send 对象实现并行执行
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder = StateGraph(OverallState)
builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)
graph = builder.compile()

# 调用时会并行生成多个笑话
graph.invoke({"subjects": ["cats", "dogs"]})
# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
```

### 2. Command 类 - 高级控制

**位置**: `types.py:368-388`

**定义**:
```python
@dataclass
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    Args:
        graph: Graph to send the command to
        update: Update to apply to the graph's state
        resume: Value to resume execution with
        goto: Can be one of the following:
            - Name of the node to navigate to next
            - Sequence of node names to navigate to next
            - Send object (to execute a node with the input provided)
            - Sequence of Send objects
    """
```

**核心特性**:
1. **支持 Send 对象**: `goto` 参数可以是 `Send` 对象或 `Send` 对象序列
2. **状态更新**: 可以同时更新状态和发送消息
3. **灵活导航**: 支持多种导航方式

### 3. BranchSpec 类 - 分支路由实现

**位置**: `_branch.py:83-226`

**核心方法**:

#### `_finish` 方法 - 处理路由结果
```python
def _finish(
    self,
    writer: _Writer,
    input: Any,
    result: Any,
    config: RunnableConfig,
) -> Runnable | Any:
    # 将结果转换为列表
    if not isinstance(result, (list, tuple)):
        result = [result]

    # 处理目标节点
    if self.ends:
        destinations: Sequence[Send | str] = [
            r if isinstance(r, Send) else self.ends[r] for r in result
        ]
    else:
        destinations = cast(Sequence[Send | str], result)

    # 验证目标节点
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")
    if any(p.node == END for p in destinations if isinstance(p, Send)):
        raise InvalidUpdateError("Cannot send a packet to the END node")

    # 写入目标节点
    entries = writer(destinations, False)
    # ...
```

**关键逻辑**:
1. **支持多个目标**: 结果可以是列表或元组
2. **混合类型**: 可以混合使用 `Send` 对象和节点名称字符串
3. **验证机制**: 检查目标节点的有效性
4. **写入操作**: 将目标节点写入执行队列

### 4. Pregel 算法 - 并行执行模型

**位置**: `pregel/main.py:336-344`

**核心概念**:
```python
"""
following the **Pregel Algorithm**/**Bulk Synchronous Parallel** model.

- **Execution**: Execute all selected **actors** in parallel,
"""
```

**Bulk Synchronous Parallel (BSP) 模型**:
1. **超步 (Superstep)**: 执行分为多个超步
2. **并行执行**: 在每个超步中，所有选中的 actors 并行执行
3. **同步点**: 超步之间有同步点，确保所有 actors 完成后再进入下一个超步

**在 LangGraph 中的体现**:
- 当条件边返回多个 `Send` 对象时，这些 `Send` 对象对应的节点会在同一个超步中并行执行
- 所有并行节点执行完毕后，结果会被合并到图的状态中
- 然后进入下一个超步

## 测试用例中的并行执行示例

### 示例 1: 简单并行执行
**位置**: `test_pregel.py:1115`
```python
return [Send("2", state), Send("2", state)]
```
- 同一个节点 "2" 被调用两次，使用相同的状态

### 示例 2: 混合并行执行
**位置**: `test_pregel.py:1145-1148`
```python
return [Send("2", 1), Send("2", 2), "3.1"]
```
- 节点 "2" 被调用两次，使用不同的状态 (1 和 2)
- 同时还会执行节点 "3.1"

### 示例 3: Map-Reduce 模式
**位置**: `test_large_cases_async.py:3466`
```python
return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```
- 对每个 subject 并行生成笑话
- 典型的 Map-Reduce 模式

### 示例 4: 工具调用并行执行
**位置**: `test_large_cases_async.py:1410`
```python
return [Send("tools", tool_call) for tool_call in tool_calls]
```
- 并行执行多个工具调用
- 常见于 Agent 系统中

## 并行执行的关键机制

### 1. 并行节点创建
- 通过条件边返回多个 `Send` 对象
- 每个 `Send` 对象指定目标节点和输入状态

### 2. 结果合并
- 使用 `Annotated` 类型和 reducer 函数（如 `operator.add`）
- 自动将并行节点的结果合并到图的状态中

### 3. 同步机制
- 基于 Bulk Synchronous Parallel 模型
- 所有并行节点在同一个超步中执行
- 超步之间有同步点，确保所有节点完成后再继续

## 实现细节

### 并行执行流程
1. **条件边评估**: 条件边函数返回多个 `Send` 对象
2. **任务创建**: 为每个 `Send` 对象创建一个执行任务
3. **并行执行**: 所有任务在同一个超步中并行执行
4. **结果收集**: 收集所有任务的结果
5. **状态合并**: 使用 reducer 函数合并结果到图的状态
6. **同步点**: 等待所有任务完成后进入下一个超步

### 状态管理
- **独立状态**: 每个 `Send` 对象可以携带独立的状态
- **状态合并**: 使用 `Annotated[type, reducer]` 定义合并策略
- **状态隔离**: 并行节点之间的状态是隔离的

## 性能考虑

### 并行度
- 并行度取决于返回的 `Send` 对象数量
- 没有硬性限制，但需要考虑系统资源

### 同步开销
- 超步之间的同步会带来一定开销
- 适合计算密集型任务，不适合 I/O 密集型任务

### 内存使用
- 每个并行任务都有独立的状态副本
- 大量并行任务可能导致内存压力

## 总结

LangGraph 的并行执行机制基于以下核心组件：

1. **Send 类**: 定义并行任务的目标节点和输入状态
2. **条件边**: 返回多个 `Send` 对象来触发并行执行
3. **Pregel 算法**: 基于 Bulk Synchronous Parallel 模型实现同步并行
4. **Reducer 函数**: 自动合并并行节点的结果

这种设计使得 LangGraph 能够优雅地处理 Map-Reduce、多工具调用、多智能体协作等复杂的并行工作流场景。
