---
type: fetched_content
source: https://medium.com/ai-engineering-bootcamp/a-beginners-guide-to-dynamic-routing-in-langgraph-with-command-2c8c0f3ef451
title: A Beginner's Guide to Dynamic Routing in LangGraph with Command()
fetched_at: 2026-02-28
status: success
author: Damilola Oyedunmade
knowledge_point: 05_动态图构建
fetch_tool: grok-mcp
---

# A Beginner's Guide to Dynamic Routing in LangGraph with Command()

by Damilola Oyedunmade | AI Engineering BootCamp | Medium

## 核心观点

动态路由意味着图在运行时根据当前状态决定下一步做什么。Command() 是实现动态路由的核心 API。

## 什么是动态路由

- 图不必遵循固定路径
- 下一步可以在运行时根据刚发生的事情改变
- 与静态路由不同，动态路由灵活、表达力强

## Command() 如何实现动态路由

Command() 允许节点在运行时决定接下来发生什么：
- 指定下一个节点（goto）
- 应用状态更新（update）
- 路由决策逻辑在节点内部

## 简单示例：基于用户意图的路由

```python
from langgraph.graph import StateGraph
from langgraph.types import Command

def router(state):
    text = state["input"].lower()
    if "how" in text or "what" in text:
        return Command(
            goto="question_node",
            update={"intent": "question"}
        )
    return Command(
        goto="command_node",
        update={"intent": "command"}
    )

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

1. 保持路由逻辑简单易懂
2. 分离决策与计算 - 路由节点只做选择，下游节点做实际工作
3. 有意识地更新状态 - 只包含下一个节点需要的数据
4. 使用有意义的节点名称
5. 尽早可视化图结构

## 关键洞察

- Command() 让节点有了"声音"，可以说"去这里"、"现在停止"、"路由到另一条路径"
- 路由逻辑靠近推理逻辑，图更易读
- 动态流感觉自然，而非强制
