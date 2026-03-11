---
type: context7_documentation
library: langchain
version: 0.3.x (2025-2026)
fetched_at: 2026-03-06
knowledge_point: 07_Agent Memory集成
context7_query: agent memory conversation history buffer summary working memory
---

# Context7 文档：LangChain Agent Memory

## 文档来源
- 库名称：LangChain
- 版本：0.3.x (2025-2026)
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. 2026 新架构：create_agent + middleware 记忆管理

LangChain 2026 引入 `create_agent` API，记忆管理通过 **middleware** 实现：

#### 删除旧消息中间件

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """移除旧消息保持对话可控"""
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    "gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}
```

#### SummarizationMiddleware（摘要中间件）

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=InMemorySaver(),
)
```

#### StateClaudeMemoryMiddleware（Anthropic 状态记忆）

```python
from langchain_anthropic.middleware import StateClaudeMemoryMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    tools=[],
    middleware=[StateClaudeMemoryMiddleware()],
    checkpointer=MemorySaver(),
)
```

### 2. 经典 Memory 集成模式

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_agent
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [...]

agent = create_agent(
    model=llm,
    tools=tools,
    memory=memory,
    verbose=True,
)
```

### 3. 记忆类型体系（概念文档）

官方文档定义三种记忆类型：
- **语义记忆 (Semantic Memory)**：存储事实（如用户偏好）
- **情景记忆 (Episodic Memory)**：存储经历（如过去的 Agent 操作）
- **程序记忆 (Procedural Memory)**：存储指令（如 Agent 系统提示）

这三种类型类比人类记忆系统，可映射到 AI Agent 实现。

### 4. Checkpointer 记忆持久化

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    checkpointer=checkpointer,
)

# 同一 thread_id 跨调用保持对话历史
config = {"configurable": {"thread_id": "user-123"}}
agent.invoke({"messages": "hi"}, config)
agent.invoke({"messages": "what did I say?"}, config)  # 能记住上轮对话
```

### 5. 生产环境建议

- 使用持久化 checkpointer（如 PostgreSQL, Redis）替代 InMemorySaver
- 确保对话历史在系统重启后仍可访问
- 跨多个 session 访问对话历史
