---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-28
knowledge_point: 03_循环与迭代
context7_query: loops iteration cycles conditional edges
---

# LangGraph 循环与迭代 - Context7 文档参考

> 来源: Context7 `/websites/langchain_oss_python_langgraph` (Benchmark Score: 86.9, 900 snippets)

---

## Query 1: 循环、条件边与图中的环 (Loops, Cycles, Conditional Edges)

### 创建和控制循环

Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api

当创建带有循环的图时，需要一种方式来停止执行。最常见的方法是使用条件边（conditional edge），在满足特定终止条件时将流程导向 `END` 节点。此外，可以通过在调用或流式处理时设置递归限制（recursion limit）来控制图可以运行的最大步数。此限制通过在图超过允许的超级步数（supersteps）时引发错误来防止无限循环。

### 示例 1: 基础条件循环

定义一个带有循环的图，使用 `StateGraph`。包括添加节点、定义基于状态的终止条件边，以及编译图。`State` 使用 `TypedDict` 定义，其中 `aggregate` 列表通过 `operator.add` 实现追加。

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

### 示例 2: 条件分支 (Conditional Branching)

Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api

演示如何在 LangGraph 中设置条件分支。定义状态、更新状态的节点，并使用 `add_conditional_edges` 配合自定义函数根据状态确定下一个节点。这允许图的执行流在运行时动态变化。

```python
import operator
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"], "which": "c"}  # [!code highlight]

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)

def conditional_edge(state: State) -> Literal["b", "c"]:
    # Fill in arbitrary logic here that uses the state
    # to determine the next node
    return state["which"]

builder.add_conditional_edges("a", conditional_edge)  # [!code highlight]

graph = builder.compile()
```

### 示例 3: 带分支的循环与递归限制

Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api

一个复杂示例，展示递归限制如何在具有分支路径的循环中运作。有助于理解当单个步骤可以到达多个节点时的流控制。

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Node C sees {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Node D sees {state["aggregate"]}')
    return {"aggregate": ["D"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)

# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "c")
builder.add_edge("b", "d")
builder.add_edge(["c", "d"], "a")
graph = builder.compile()
```

---

## Query 2: 递归限制、RemainingSteps 与循环终止

### 设置递归限制

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

在某些应用中，我们可能无法保证会达到给定的终止条件。在这些情况下，可以设置图的递归限制。这将在给定数量的超级步（supersteps）后引发 `GraphRecursionError`。然后可以捕获并处理此异常。

```python
# 通过 config 字典在 invoke 或 stream 方法中设置递归限制
graph.invoke(inputs, config={"recursion_limit": 5})
```

### 示例 4: 使用 RemainingSteps 返回状态而非抛出异常

Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api

展示如何使用 `RemainingSteps` 注解来管理递归限制。这种方法不是引发错误，而是修改状态以跟踪剩余步数，允许受控终止。

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    remaining_steps: RemainingSteps

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# Define edges
def route(state: State) -> Literal["b", END]:
    if state["remaining_steps"] <= 2:
        return END
    else:
        return "b"

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()

# Test it out
result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
print(result)
```

### 主动式 vs 被动式递归限制处理

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

比较 LangGraph 中处理递归限制的主动式和被动式方法。推荐使用 `RemainingSteps` 的主动式方法，以便在图逻辑内实现优雅降级。

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps
from langgraph.errors import GraphRecursionError

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps

# Proactive Approach (recommended) - using RemainingSteps
def agent_with_monitoring(state: State) -> dict:
    """Proactively monitor and handle recursion within the graph"""
    remaining = state["remaining_steps"]

    # Early detection - route to internal handling
    if remaining <= 2:
        return {
            "messages": ["Approaching limit, returning partial result"]
        }

    # Normal processing
    return {"messages": [f"Processing... ({remaining} steps remaining)"]}

def route_decision(state: State) -> Literal["agent", END]:
    if state["remaining_steps"] <= 2:
        return END
    return "agent"

# Build graph
builder = StateGraph(State)
builder.add_node("agent", agent_with_monitoring)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_decision)
graph = builder.compile()

# Proactive: Graph completes gracefully
result = graph.invoke({"messages": []}, {"recursion_limit": 10})
```

### 示例 5: 带 Fallback 节点的主动递归处理

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

展示使用 `RemainingSteps` 管理值的主动递归处理。通过在达到递归限制之前检查剩余步数并相应路由逻辑，实现优雅降级。

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps  # Managed value - tracks steps until limit

def reasoning_node(state: State) -> dict:
    # RemainingSteps is automatically populated by LangGraph
    remaining = state["remaining_steps"]

    # Check if we're running low on steps
    if remaining <= 2:
        return {"messages": ["Approaching limit, wrapping up..."]}

    # Normal processing
    return {"messages": ["thinking..."]}

def route_decision(state: State) -> Literal["reasoning_node", "fallback_node"]:
    """Route based on remaining steps"""
    if state["remaining_steps"] <= 2:
        return "fallback_node"
    return "reasoning_node"

def fallback_node(state: State) -> dict:
    """Handle cases where recursion limit is approaching"""
    return {"messages": ["Reached complexity limit, providing best effort answer"]}

# Build graph
builder = StateGraph(State)
builder.add_node("reasoning_node", reasoning_node)
builder.add_node("fallback_node", fallback_node)
builder.add_edge(START, "reasoning_node")
builder.add_conditional_edges("reasoning_node", route_decision)
builder.add_edge("fallback_node", END)

graph = builder.compile()

# RemainingSteps works with any recursion_limit
result = graph.invoke({"messages": []}, {"recursion_limit": 10})
```

---

## Query 3: Command、goto 与动态路由模式

### Command API: 组合状态更新与路由

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

`Command` API 允许节点在同一个函数调用中同时执行状态更新和确定下一个要转换到的节点。这对于需要同时修改状态和动态控制流的场景非常有用，例如多智能体交接（multi-agent handoffs）。

**核心用法:**

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

**带条件的 Command:**

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

### Command 与静态边的交互

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

**重要:** `Command` 只添加动态边，静态边仍然会执行。换句话说，`Command` 不会覆盖静态边。

```python
from langgraph.types import Command
from typing import Literal

def node_a(state: State) -> Command[Literal["my_other_node"]]:
   if state["foo"] == "bar":
       return Command(update={"foo": "baz"}, goto="my_other_node")

# Add a static edge from "node_a" to "node_b"
graph.add_edge("node_a", "node_b")

# Command will NOT prevent "node_a" from going to "node_b"
# In the example above, "node_a" will go to both "node_b" and "my_other_node"
```

### 何时使用 Command vs 条件边?

Source: https://docs.langchain.com/oss/python/langgraph/graph-api

- **使用 `Command`**: 当你需要**同时**更新图状态**并且**路由到不同节点时。例如，在实现多智能体交接时，需要路由到不同的 agent 并传递一些信息给该 agent。
- **使用条件边**: 当你只需要在节点之间进行条件路由而**不需要**更新状态时。

### 示例 6: 端到端 Command 示例

Source: https://docs.langchain.com/oss/python/langgraph/use-graph-api

一个端到端的 LangGraph 示例，使用 `Command` 进行控制流。定义三个节点（A、B、C），其中节点 A 根据随机选择决定路由到 B 或 C，同时更新图状态。

```python
import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command

class State(TypedDict):
    foo: str

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["b", "c"])
    if value == "b":
        goto = "node_b"
    else:
        goto = "node_c"

    return Command(
        update={"foo": value},
        goto=goto,
    )

def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}

builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
```

**类型注解要求:** 当在节点函数中返回 `Command` 时，必须添加返回类型注解，列出节点可以路由到的节点名称列表，例如 `Command[Literal["my_other_node"]]`。这对于图的渲染是必要的，并告诉 LangGraph 该节点可以导航到哪些节点。

---

## 关键概念总结

### 1. 循环创建的三种方式

| 方式 | 适用场景 | 特点 |
|------|----------|------|
| `add_conditional_edges` + 路由函数 | 标准循环控制 | 最常用，清晰分离路由逻辑 |
| `Command` + `goto` | 需要同时更新状态和路由 | 适合多智能体交接 |
| 静态边 `add_edge("b", "a")` + 条件退出 | 简单循环 | 配合条件边使用 |

### 2. 循环终止的三种策略

| 策略 | 方法 | 推荐度 |
|------|------|--------|
| 条件终止 | 路由函数返回 `END` | 最常用 |
| 主动监控 (推荐) | `RemainingSteps` 管理值 | 优雅降级 |
| 被动限制 | `recursion_limit` 配置 | 兜底保护 |

### 3. 核心 API 速查

```python
# 条件边
builder.add_conditional_edges("node_a", route_function)

# 静态边
builder.add_edge("node_a", "node_b")

# 多源汇聚边
builder.add_edge(["node_c", "node_d"], "node_a")

# Command 动态路由
from langgraph.types import Command
return Command(update={"key": "value"}, goto="next_node")

# 递归限制
graph.invoke(inputs, config={"recursion_limit": 10})

# 剩余步数监控
from langgraph.managed import RemainingSteps
# 或
from langgraph.managed.is_last_step import RemainingSteps
```
