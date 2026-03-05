---
type: fetched_content
source: https://medium.com/fundamentals-of-artificial-intelligence/langgraph-cycles-and-conditional-edges-fb4c4839e0a4
title: LangGraph Cycles and Conditional Edges
fetched_at: 2026-02-28
status: fetched
knowledge_point: 03_循环与迭代
fetch_tool: grok-mcp
---

# LangGraph: Cycles and Conditional Edges

*作者：Pankaj Chandravanshi*
*刊物：Fundamentals of Artificial Intelligence*

> **导读**：本文重点讲解如何使用 **LangGraph** 实现带有**条件分支**（conditional edges）和**循环**（cycles）的复杂工作流。作者将通过一个简单但完整的示例，展示状态更新、条件判断与循环执行的结合方式。

## Graph modifies a state and decides whether to continue or finish depending on the value inside the state

在 LangGraph 中，我们可以构建一个能够**修改状态**，并根据**状态内部的值**来决定是**继续执行**还是**结束流程**的图（Graph）。

本文将聚焦于如何使用 LangGraph 创建一个同时包含**条件逻辑**（conditional logic）和**循环行为**（cyclic behavior）的工作流。

### The State and Functions

我们首先定义状态（State）。这里使用 `TypedDict` 来定义一个带有计数器（counter）的简单状态：

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    counter: Annotated[int, operator.add]
    # 你可以根据需要添加更多字段
```

> **说明**：
> - 使用 `Annotated[int, operator.add]` 表示每次更新时会对 counter 字段进行**累加**操作（这是 LangGraph 中 reducer 的常见用法）。

接下来定义几个核心节点函数：

1. **节点：增加计数器**

```python
def increment_counter(state: State) -> State:
    return {"counter": 1}   # 每次调用 +1（配合 reducer）
```

2. **条件判断函数**（用于决定路由）

```python
def should_continue(state: State) -> str:
    if state["counter"] < 5:
        return "continue"   # 继续循环
    else:
        return "finish"     # 结束
```

> 注意：LangGraph 中的条件边（conditional edge）会调用此类函数，并根据返回值决定下一步走向哪个节点。

### 构建 Graph 的完整示例代码

以下是一个典型的带有循环和条件边的 LangGraph 实现：

```python
from langgraph.graph import StateGraph, END

# 创建 StateGraph
workflow = StateGraph(State)

# 添加节点
workflow.add_node("increment", increment_counter)

# 设置入口点
workflow.set_entry_point("increment")

# 添加条件边
workflow.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",   # 形成循环
        "finish": END
    }
)

# 编译图
graph = workflow.compile()

# 执行示例
initial_state = {"counter": 0}
result = graph.invoke(initial_state)

print(result)  # 最终应输出 {'counter': 5}
```

### 代码解释

- **循环的实现**：通过条件边将节点指向**自己**（"increment" -> "increment"），实现了循环
- **退出条件**：当 `counter >= 5` 时，路由到 `END`，结束执行
- **状态累加**：得益于 `Annotated[..., operator.add]`，每次返回的 `{"counter": 1}` 都会累加到现有值上

### 实际应用场景

这种**条件 + 循环**的模式非常常见，例如：

- Agent 循环调用工具直到满足停止条件
- 多轮对话中反复精炼答案
- 文档处理中反复拆分/合并直到达到质量阈值
- 自动调试/自我修正循环

### 总结

LangGraph 通过**条件边（conditional edges）**和**指向自身的边**（形成 cycle）提供了极大的灵活性，让开发者可以轻松构建带有循环和动态路由的复杂 AI 工作流。

希望这篇简单示例能帮助你更好地理解 LangGraph 中循环与条件分支的核心实现方式。

后续可以尝试：

- 加入更多节点与分支
- 使用 `human-in-the-loop` 中断循环
- 结合工具调用（tools）实现真正的 Agent 行为
