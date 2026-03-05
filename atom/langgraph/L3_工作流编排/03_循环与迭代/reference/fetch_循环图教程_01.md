---
type: fetched_content
source: https://medium.com/towardsdev/built-with-langgraph-9-looping-graphs-b689e42677d7
title: Built with LangGraph #9 Looping Graphs
fetched_at: 2026-02-28
status: fetched
knowledge_point: 03_循环与迭代
fetch_tool: grok-mcp
---

# Built with LangGraph! #9: Looping Graphs

**Okan Yenigün**
Jul 7, 2025
3 min read

## How Looping and Conditional Logic Can Be Orchestrated?

在上一章中，我们实现了一个条件图（conditional graph）。现在，我们将利用它来创建一个**循环图（looping graph）**，其中某个节点作为循环节点，重复执行直到满足特定条件为止。

首先，我们设计一个简单的有状态图（stateful graph），它会反复生成随机数并将它们累积到一个列表中，直到满足某个条件。

```python
import random
from langgraph.graph import StateGraph, END
from typing import List, TypedDict

class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int
```

我们先定义两个核心节点：一个用于问候用户并初始化状态，另一个用于生成随机数并跟踪迭代次数。

```python
def greeting_node(state: AgentState) -> AgentState:
    """This node greets the user and initializes the state."""
    state["name"] = f"Hi, {state['name']}!"
    state["counter"] = 0
    return state

def random_node(state: AgentState) -> AgentState:
    """This node generates a random number and appends it to the list."""
    state["number"].append(random.randint(1, 10))
    state["counter"] += 1
    return state
```

为了控制图的流向并决定何时继续循环或结束，我们需要一个决策节点，它会检查当前状态并返回相应的下一步。

```python
def should_continue(state: AgentState) -> str:
    """This node checks if the counter is less than 5."""
    if state["counter"] < 5:
        print(f"Entering loop with counter: {state['counter']}")
        return "loop"
    else:
        return "exit"
```

如果迭代次数少于 5 次，它返回字符串 "loop"，指示图继续重复随机数生成步骤。

我们从创建一个新的 `StateGraph` 开始：

```python
graph = StateGraph(AgentState)

# Add nodes representing the main actions of the workflow
graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

# Connect the greeting node directly to the random node
graph.add_edge("greeting", "random")

# Add a conditional edge to enable looping or exiting based on the counter
graph.add_conditional_edges(
    "random",
    should_continue,  # Decision function
    {
        "loop": "random",  # If 'loop', repeat the random node
        "exit": END,       # If 'exit', terminate the workflow
    }
)

# Specify where the workflow should start
graph.set_entry_point("greeting")

# Compile the graph into an executable application
app = graph.compile()
```

- `greeting` 节点：用于初始化和问候用户
- `random` 节点：生成随机数并更新状态

工作流总是先从 `greeting` 节点过渡到 `random` 节点。

从 `random` 节点出发，使用 `should_continue` 函数设置条件边：

- 如果返回 `loop`，则重复执行 `random` 节点
- 如果返回 `exit`，则工作流结束

```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

```python
response = app.invoke({"name": "Okan", "number": [], "counter": -1})

"""
Entering loop with counter: 1
Entering loop with counter: 2
Entering loop with counter: 3
Entering loop with counter: 4
"""

print(response)

"""
{'name': 'Hi, Okan!', 'number': [2, 7, 7, 2, 2], 'counter': 5}
"""
```

### Sources

- [https://www.youtube.com/watch?v=jGg_1h0qzaM](https://www.youtube.com/watch?v=jGg_1h0qzaM)
