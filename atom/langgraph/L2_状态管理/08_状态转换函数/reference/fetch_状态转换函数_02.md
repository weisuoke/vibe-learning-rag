---
type: fetched_content
source: https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025
title: Mastering LangGraph State Management in 2025
fetched_at: 2026-02-27
status: success
author: SparkCo
knowledge_point: 08_状态转换函数
fetch_tool: grok-mcp
---

# Mastering LangGraph State Management in 2025

## 核心要点

### 1. 显式状态 Schema
```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[str]
    counter: Annotated[int, add]
```

### 2. Reducer 在状态转换中的作用
- Reducer 定义了新状态信息如何与现有状态合并
- 避免多 agent 并发更新时的数据丢失
- 灵感来自函数式编程范式

### 3. 状态转换最佳实践
- 使用 TypedDict 定义明确的状态结构
- 为需要累积的字段指定 reducer
- 不需要累积的字段使用默认覆盖行为
- 结合 checkpointing 实现持久化

### 4. 多 Agent 协调
- 显式 reducer 确保并发安全
- 消息历史使用 add_messages
- 计数器使用 operator.add
- 文档列表可以使用默认覆盖或 operator.add
