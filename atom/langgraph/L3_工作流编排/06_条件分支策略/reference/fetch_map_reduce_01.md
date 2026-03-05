---
type: fetched_content
source: https://medium.com/@astropomeai/implementing-map-reduce-with-langgraph-creating-flexible-branches-for-parallel-execution-b6dc44327c0e
title: Implementing Map-Reduce with LangGraph - Send API
fetched_at: 2026-02-28
status: success
knowledge_point: 06_条件分支策略
fetch_tool: grok-mcp
---

# Send API 实现 Map-Reduce 并行分支

## 核心问题

Map-Reduce 实现的两大挑战：
1. 子任务数量在图设计时未知
2. 每个子任务需要不同的输入状态

## Send API 解决方案

Send API 通过条件边分发不同状态到多个节点实例：
- 发送的状态可以与核心图状态不同
- 支持动态并行分支

## Tree of Thoughts 完整示例

```python
import operator
from typing import Annotated, TypedDict
from langgraph.constants import Send
from langgraph.graph import END, StateGraph, START

class OverallState(TypedDict):
    input: str
    perfect_factors: str
    solutions: Annotated[list[str], operator.add]
    reviews: Annotated[list[str], operator.add]
    deep_thoughts: Annotated[list[str], operator.add]
    ranked_solutions: str

class SolutionState(TypedDict):
    solution: str

def continue_to_evaluation(state: OverallState):
    return [Send("evaluate_solution", {"solution": s}) for s in state["solutions"]]

def continue_to_deep_thought(state: OverallState):
    return [Send("deepen_thought", {"solution": r}) for r in state["reviews"]]

# 构建图
graph = StateGraph(OverallState)
graph.add_node("generate_solutions", generate_solutions)
graph.add_node("evaluate_solution", evaluate_solution)
graph.add_node("deepen_thought", deepen_thought)
graph.add_node("rank_solutions", rank_solutions)

graph.add_edge(START, "generate_solutions")
graph.add_conditional_edges("generate_solutions", continue_to_evaluation, ["evaluate_solution"])
graph.add_conditional_edges("evaluate_solution", continue_to_deep_thought, ["deepen_thought"])
graph.add_edge("deepen_thought", "rank_solutions")
graph.add_edge("rank_solutions", END)
```

## 关键模式

- **Map**: 条件边 + Send 分发子任务
- **Reduce**: Annotated[list, operator.add] 自动聚合结果
- **状态隔离**: SolutionState 与 OverallState 分离
