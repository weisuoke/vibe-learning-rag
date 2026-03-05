---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-24
knowledge_point: Memory与对话历史管理
context7_query: migrating memory deprecated, chat message history storage, memory conversation history management
---

# Context7 文档：LangChain Memory 与对话历史管理

## 文档来源

- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/oss/python/langchain
- Context7 库 ID：/websites/langchain

## 关键信息提取

### 1. **Memory 迁移指南**

**重要发现**：
- LangChain 的传统 Memory 类已被弃用
- 官方推荐迁移到新的架构
- 迁移指南：https://python.langchain.com/docs/versions/migrating_memory/

**弃用示例**：
```python
# 已弃用的导入
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_ai21 import AI21LLM
```

**弃用政策**：
- 当弃用功能时，会标记为 deprecated 并提供清晰的迁移指南
- 弃用功能至少保留一个次要版本，为用户提供迁移的宽限期
- 功能仅在主要版本发布时移除
- 迁移支持包括迁移指南，并在可能的情况下提供自动化工具

### 2. **现代对话历史管理方法**

#### A. **InMemoryChatMessageHistory**

**基础用法**：
```python
from langchain_core.chat_history import InMemoryChatMessageHistory

chats_by_session_id = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history
```

**特点**：
- 基于字典的存储系统
- 按 session ID 索引
- 支持多个并发对话独立跟踪

#### B. **RunnableWithMessageHistory**

**核心模式**：
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 存储字典，映射 session ID 到对话历史
store = {}  # 内存维护在链外部

# 返回给定 session ID 的对话历史
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = ChatNVIDIA(
    model="mistralai/mixtral-8x22b-instruct-v0.1",
    temperature=0.1,
    max_tokens=100,
    top_p=1.0,
)

# 定义 RunnableConfig 对象，包含 configurable 键
config = {"configurable": {"session_id": "1"}}

conversation = RunnableWithMessageHistory(
    chat,
    get_session_history,
)

conversation.invoke(
    "Hi I'm Srijan Dubey.",
    config=config,
)
```

**特点**：
- 使用 `RunnableWithMessageHistory` 包装聊天模型
- 支持基于 session 的对话历史管理
- 内存维护在链外部
- 通过 `config` 传递 session ID

#### C. **手动管理对话状态**

**直接管理消息列表**：
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", use_responses_api=True)

first_query = "Hi, I'm Bob."
messages = [{"role": "user", "content": first_query}]

response = llm.invoke(messages)
print(response.text)

second_query = "What is my name?"

messages.extend(
    [
        response,
        {"role": "user", "content": second_query},
    ]
)
second_response = llm.invoke(messages)
print(second_response.text)
```

**特点**：
- 手动维护消息列表
- 每次调用前将响应追加到历史
- 模型能够跨交互维护上下文
- 简单直接，适合简单场景

### 3. **持久化对话历史存储**

#### A. **CockroachDB 存储**

**配置示例**：
```python
from langchain_cockroachdb import CockroachDBChatMessageHistory
import uuid

chat_history = CockroachDBChatMessageHistory(
    session_id=str(uuid.uuid4()),
    connection_string=CONNECTION_STRING,
    table_name="chat_history",
)

from langchain.messages import HumanMessage, AIMessage

# 添加单条消息
await chat_history.aadd_message(HumanMessage(content="Hello!"))
await chat_history.aadd_message(AIMessage(content="Hi there!"))

# 获取消息
messages = await chat_history.aget_messages()
```

**批量添加消息**：
```python
messages = [
    HumanMessage(content="What are vector indexes?"),
    AIMessage(content="Vector indexes enable fast similarity search..."),
    HumanMessage(content="How do they work?"),
    AIMessage(content="They use approximate nearest neighbor algorithms..."),
]

await chat_history.aadd_messages(messages)
```

**特点**：
- 支持持久化存储
- 基于 session 的组织
- 强一致性保证
- 支持异步操作
- 批量操作更高效

### 4. **LangGraph 中的对话历史管理**

#### A. **SummarizationNode**

**完整示例**：
```python
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

model = init_chat_model("claude-sonnet-4-5-20250929")
summarization_model = model.bind(max_tokens=128)

class State(MessagesState):
    context: dict[str, RunningSummary]  # 运行摘要的上下文

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

# 调用图
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
final_response = graph.invoke({"messages": "what's my name?"}, config)
```

**特点**：
- 使用 `SummarizationNode` 管理和总结对话历史
- 自定义 `State` 包含运行摘要的上下文
- 配置 Token 限制的总结过程
- 集成到 LangGraph 工作流中
- 展示如何在对话 AI 中维护长期记忆

#### B. **Checkpointer 机制**

**对话记忆维护**：
- 使用 checkpointer 存储对话状态
- 通过唯一的 thread ID 标识
- 允许 agent 引用和理解之前的交互
- 支持上下文感知的响应
- checkpointer 持久化对话历史
- 允许 agent 维护连贯的多轮对话

### 5. **核心设计模式**

#### A. **Session 管理模式**

**关键要素**：
1. **Session ID**：唯一标识每个对话
2. **存储字典**：映射 session ID 到对话历史
3. **获取函数**：根据 session ID 检索或创建历史
4. **配置对象**：传递 session ID 到 Runnable

**示例模式**：
```python
# 1. 定义存储
store = {}

# 2. 定义获取函数
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. 包装 Runnable
conversation = RunnableWithMessageHistory(
    chat_model,
    get_session_history,
)

# 4. 使用配置调用
config = {"configurable": {"session_id": "1"}}
conversation.invoke("message", config=config)
```

#### B. **消息管理模式**

**两种主要方法**：

1. **自动管理**（推荐）：
   - 使用 `RunnableWithMessageHistory`
   - 自动处理消息追加和检索
   - 支持持久化存储

2. **手动管理**：
   - 直接维护消息列表
   - 手动追加响应到历史
   - 适合简单场景

### 6. **最佳实践**

#### A. **选择存储方案**

**内存存储**：
- 适合：开发、测试、短期会话
- 优点：简单、快速
- 缺点：不持久、不可扩展

**持久化存储**：
- 适合：生产环境、长期会话
- 优点：持久、可扩展、可靠
- 缺点：需要额外配置

#### B. **Session 管理**

**关键考虑**：
1. **Session ID 生成**：使用 UUID 或用户 ID
2. **Session 清理**：定期清理过期 session
3. **并发控制**：处理多个并发对话
4. **错误处理**：处理存储失败

#### C. **消息格式**

**标准格式**：
```python
from langchain.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="用户输入"),
    AIMessage(content="AI 回复"),
]
```

**特点**：
- 使用 `HumanMessage` 和 `AIMessage`
- 支持异步操作（`aadd_message`、`aget_messages`）
- 支持批量操作（`aadd_messages`）

### 7. **迁移路径**

#### A. **从传统 Memory 迁移**

**旧方式**（已弃用）：
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
```

**新方式**（推荐）：
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chat_model,
    get_session_history,
)
```

#### B. **迁移步骤**

1. **识别当前使用的 Memory 类型**
2. **选择新的存储方案**（内存或持久化）
3. **实现 session 管理函数**
4. **使用 `RunnableWithMessageHistory` 包装模型**
5. **更新调用代码以传递 config**
6. **测试迁移后的功能**

### 8. **LangGraph 集成**

#### A. **状态管理**

**自定义状态**：
```python
class State(MessagesState):
    context: dict[str, RunningSummary]
```

**特点**：
- 继承 `MessagesState`
- 添加自定义字段（如 `context`）
- 支持复杂的状态管理

#### B. **Checkpointer**

**使用示例**：
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "message"}, config)
```

**特点**：
- 持久化对话状态
- 支持多线程对话
- 通过 thread ID 隔离

### 9. **性能优化**

#### A. **批量操作**

**推荐**：
```python
await chat_history.aadd_messages(messages)  # 批量添加
```

**不推荐**：
```python
for message in messages:
    await chat_history.aadd_message(message)  # 逐个添加
```

#### B. **异步操作**

**使用异步方法**：
- `aadd_message()` 而非 `add_message()`
- `aget_messages()` 而非 `get_messages()`
- 提高并发性能

### 10. **常见问题**

#### Q1: 为什么传统 Memory 类被弃用？

**A**:
- 不支持原生工具调用
- 架构限制
- 不适合现代 LLM 应用
- 官方推荐使用 LangGraph 和 `RunnableWithMessageHistory`

#### Q2: 如何选择存储方案？

**A**:
- **开发/测试**：`InMemoryChatMessageHistory`
- **生产环境**：CockroachDB、PostgreSQL、Redis 等持久化存储
- **简单场景**：手动管理消息列表
- **复杂场景**：LangGraph + Checkpointer

#### Q3: 如何处理长对话历史？

**A**:
- 使用 `SummarizationNode` 总结历史
- 配置 Token 限制
- 定期清理旧消息
- 使用向量存储检索相关历史

## 总结

**核心变化**：
1. **传统 Memory 类已弃用**
2. **推荐使用 `RunnableWithMessageHistory`**
3. **LangGraph 提供更强大的状态管理**
4. **支持多种持久化存储方案**
5. **Session 管理成为核心模式**

**迁移建议**：
- 尽快迁移到新架构
- 使用官方迁移指南
- 选择合适的存储方案
- 采用 Session 管理模式
- 考虑使用 LangGraph 处理复杂场景

**学习重点**：
- `InMemoryChatMessageHistory` 基础用法
- `RunnableWithMessageHistory` 集成模式
- Session 管理最佳实践
- 持久化存储配置
- LangGraph 状态管理
