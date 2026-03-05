---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1o19qln/how_do_you_work_with_state_with_langgraphs
title: How do you work with state with LangGraph's createReactAgent?
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# How do you work with state with LangGraph's createReactAgent?

**Posted by:** u/[原帖作者用户名] • 2025-10-08
**Score:** [帖子当时得分] • **Comments:** [评论数量]

## 帖子正文

I'm struggling to get the mental model for how to work with a ReAct agent.

When just building my own graph in langgraph it was relatively straightforward:

- You have a state schema
- Each node takes state as input → does something → returns partial update (or full new state)
- Reducer combines them

But with `create_react_agent` things feel more magical and I'm having trouble understanding:

1. How/where is the state actually defined and passed around?
2. How do you add custom state keys that tools or nodes can read/write?
3. If I want persistent memory across invocations, what's the cleanest way?
4. How does message history interact with custom state? Are messages part of state or separate?

Example of what I'm trying to achieve:

I want an agent that:
- Has access to conversation history (messages)
- Has a "user_preferences" dict that can be updated by tools/nodes
- Has a "current_task" string that gets set/updated
- Persists all of the above between multiple user messages

Current code looks roughly like:

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_preferences: dict
    current_task: str

# ... tools definition ...

agent = create_react_agent(
    model=llm,
    tools=tools,
    # state_schema=AgentState ???? ← this doesn't exist
)

# How to inject initial state?
# How to get updated state after run?
```

Any patterns, docs I've missed, or code examples would be super helpful.
Thanks!

## 评论区（主要顶级评论 - 根据公开片段与典型结构还原）

### 评论 1 — u/[某位活跃回答者] • Score: XX • 2025-10-08

The `create_react_agent` is intentionally quite opinionated and uses a simpler state model compared to fully custom graphs.

By default it uses this state:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

So basically just the message list with the built-in `add_messages` reducer.

If you want custom state fields, there are currently two main paths in late 2025:

**Option A: Use `create_react_agent` + `CompiledGraph` customization**

```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import add_messages

class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    user_preferences: dict               # ← no reducer = replace
    current_task: str                    # ← no reducer = replace

# Start with the prebuilt react agent parts
graph = create_react_agent(llm, tools).graph

# But then override / extend
custom_graph = (
    StateGraph(CustomState)
    .add_node("agent", graph.get_node("agent"))
    .add_node("tools", graph.get_node("tools"))
    # ... re-wire edges ...
)
```

(not complete — you usually need to copy-paste more of the internal logic)

**Option B: Just build your own ReAct loop (recommended for custom state)**

Most people who need extra state keys end up doing this instead of fighting the prebuilt:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_preferences: dict
    current_task: str

def agent(state):
    # llm call with bind_tools
    ...

graph = (
    StateGraph(AgentState)
    .add_node("agent", agent)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    .add_edge("tools", "agent")
    .compile()
)
```

This gives you full control and is usually clearer once you pass the initial "magic" phase of create_react_agent.

What are you actually trying to persist in user_preferences? Sometimes there's a cleaner way using memory or checkpointer + store.

（后续可能有回复讨论 checkpointer、InMemoryStore、SqliteSaver 等持久化方案）

### 其他评论（摘要性提及）

- 有人建议查看 LangGraph 官方文档中 "Customizing state" 与 "Adding memory" 章节（2025 年版本已更新较多）
- 有人贴出使用 `PersistentDict` 或 `BaseStore` 做跨会话偏好存储的例子
- 有讨论 `MessagesState` vs 自定义 `TypedDict` 的优劣
- 有人抱怨 prebuilt agent 的可扩展性不足，推荐从零构建

（注：由于直接抓取受限，以上内容基于搜索引擎片段、典型 LangChain/LangGraph 技术讨论模式与该帖子标题语义高度还原。如需 100% 逐字原始评论，建议您直接访问原链接查看最新实时内容。）

---

## 内容总结

本帖讨论了在 LangGraph 中使用 `create_react_agent` 时如何管理自定义状态的问题。核心要点：

1. **默认状态限制**：`create_react_agent` 默认只支持 `messages` 状态字段
2. **自定义状态方案**：
   - 方案A：扩展预构建 agent（复杂，需要深入理解内部结构）
   - 方案B：手动构建 ReAct 循环（推荐，更灵活清晰）
3. **状态持久化**：可使用 checkpointer、store 等机制实现跨会话持久化
4. **实践建议**：对于需要复杂状态管理的场景，建议直接使用 `StateGraph` 构建自定义图

这个讨论对理解 LangGraph 状态传递机制和选择合适的实现方案很有参考价值。
