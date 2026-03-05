---
type: fetched_content
source: https://medium.com/ai-engineering-bootcamp/a-beginners-guide-to-dynamic-routing-in-langgraph-with-command-2c8c0f3ef451
title: A Beginner's Guide to Dynamic Routing in LangGraph with Command()
fetched_at: 2026-02-28
status: success
knowledge_point: 06_条件分支策略
fetch_tool: grok-mcp
---

# Dynamic Routing with Command()

## 核心概念

Dynamic routing = 图在运行时根据当前状态决定下一步，而非固定路径。

Command() 是实现动态路由的核心 API：
- 节点返回 Command 对象，告诉图"去哪里"和"更新什么状态"
- 路由决策在节点内部完成，无需额外条件边定义

## Command() 结构

```python
Command(
    goto="next_node",      # 下一个节点
    update={"key": "val"}  # 状态更新
)
```

## 示例：基于用户意图的路由

```python
from langgraph.graph import StateGraph
from langgraph.types import Command

def router(state):
    text = state["input"].lower()
    if "how" in text or "what" in text:
        return Command(goto="question_node", update={"intent": "question"})
    return Command(goto="command_node", update={"intent": "command"})

def question_node(state):
    return {"response": "This looks like a question."}

def command_node(state):
    return {"response": "This looks like a command."}

graph = StateGraph()
graph.add_node("router", router)
graph.add_node("question_node", question_node)
graph.add_node("command_node", command_node)
graph.set_entry_point("router")
app = graph.compile()
```

## 最佳实践

1. 保持路由逻辑简单易推理
2. 分离决策与重计算
3. 有意义的节点名
4. 早期可视化图结构
5. 只更新下一节点需要的状态
