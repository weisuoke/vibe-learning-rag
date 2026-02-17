# 核心概念6：LangGraph 消息列表管理

## 概念定义

**LangGraph**：LangChain 生态系统中用于构建有状态 AI Agent 的框架，通过图结构管理 Agent 的状态流转。

**MessagesState**：LangGraph 提供的内置状态类型，使用 `Annotated[list, add_messages]` 模式管理对话历史。

---

## LangGraph 状态管理的核心模式

### 1. MessagesState 基础

```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

# MessagesState 的定义（简化）
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
    # messages 是一个 List，使用 add_messages reducer
```

**关键特性：**
- `messages` 字段是 Python List（动态数组）
- `add_messages` 是一个 reducer 函数，处理消息合并逻辑
- 支持消息去重（基于 ID）

---

### 2. add_messages Reducer

```python
from langgraph.graph import add_messages

# add_messages 的简化实现
def add_messages(left: list, right: list) -> list:
    """合并两个消息列表，去重"""
    # 创建消息 ID 到消息的映射
    messages_by_id = {msg.id: msg for msg in left}

    # 添加或更新右侧消息
    for msg in right:
        messages_by_id[msg.id] = msg

    # 返回合并后的列表（保持顺序）
    return list(messages_by_id.values())
```

**工作原理：**
1. 左侧列表（现有状态）
2. 右侧列表（新消息）
3. 基于消息 ID 去重
4. 返回合并后的列表

---

## 实战示例：LangGraph Agent

### 示例1：基础对话 Agent

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 定义 Agent 状态
class AgentState(MessagesState):
    pass  # 继承 MessagesState，自动获得 messages 字段

# 创建 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 定义 Agent 节点
def agent_node(state: AgentState) -> AgentState:
    """Agent 节点：调用 LLM 生成回复"""
    messages = state["messages"]

    # 调用 LLM
    response = llm.invoke(messages)

    # 返回新消息（add_messages 会自动合并）
    return {"messages": [response]}

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# 编译
app = graph.compile()

# 使用
result = app.invoke({
    "messages": [
        SystemMessage(content="你是一个有帮助的助手"),
        HumanMessage(content="你好")
    ]
})

print(result["messages"])
# [SystemMessage(...), HumanMessage(...), AIMessage(...)]
```

---

### 示例2：带记忆的对话 Agent

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点保存器（内存）
memory = MemorySaver()

# 编译时传入检查点
app = graph.compile(checkpointer=memory)

# 第一轮对话
config = {"configurable": {"thread_id": "user_123"}}
result1 = app.invoke({
    "messages": [HumanMessage(content="我叫张三")]
}, config)

print(result1["messages"][-1].content)
# "你好，张三！很高兴认识你。"

# 第二轮对话（自动加载历史）
result2 = app.invoke({
    "messages": [HumanMessage(content="我叫什么名字？")]
}, config)

print(result2["messages"][-1].content)
# "你叫张三。"

# 查看完整对话历史
print(f"总消息数: {len(result2['messages'])}")
# 总消息数: 4（2 轮 * 2 条/轮）
```

---

### 示例3：多轮对话的状态演化

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "conversation_1"}}

# 第 1 轮
result = app.invoke({
    "messages": [HumanMessage(content="介绍一下 RAG")]
}, config)
print(f"第 1 轮后消息数: {len(result['messages'])}")  # 2

# 第 2 轮
result = app.invoke({
    "messages": [HumanMessage(content="它有什么优势？")]
}, config)
print(f"第 2 轮后消息数: {len(result['messages'])}")  # 4

# 第 3 轮
result = app.invoke({
    "messages": [HumanMessage(content="给个代码示例")]
}, config)
print(f"第 3 轮后消息数: {len(result['messages'])}")  # 6

# 访问特定轮次
turn_2_user = result["messages"][2]
turn_2_ai = result["messages"][3]
print(f"第 2 轮对话:")
print(f"  用户: {turn_2_user.content}")
print(f"  AI: {turn_2_ai.content}")
```

---

## 消息列表的性能特性

### 1. 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|------------|------|
| **添加消息** | O(1) 摊销 | List.append() |
| **访问特定轮次** | O(1) | List[index] |
| **获取最近 N 条** | O(N) | List[-N:] |
| **遍历所有消息** | O(N) | for msg in messages |
| **消息去重** | O(N) | add_messages reducer |

---

### 2. 内存占用

```python
import sys
from langchain_core.messages import HumanMessage, AIMessage

# 100 轮对话（200 条消息）
messages = []
for i in range(100):
    messages.append(HumanMessage(content=f"用户输入 {i}"))
    messages.append(AIMessage(content=f"AI 回复 {i}"))

# List 本身的内存
list_size = sys.getsizeof(messages)
print(f"List 本身: {list_size} 字节")  # ~1752 字节

# 每条消息的内存（包含内容）
msg_size = sys.getsizeof(messages[0])
print(f"单条消息: {msg_size} 字节")  # ~200-500 字节

# 总内存
total = list_size + sum(sys.getsizeof(msg) for msg in messages)
print(f"总内存: {total / 1024:.2f} KB")  # ~50-100 KB
```

---

## 高级用法

### 1. 自定义状态字段

```python
from typing import Annotated
from langgraph.graph import MessagesState, add_messages

class CustomAgentState(MessagesState):
    # 继承 messages 字段

    # 添加自定义字段
    user_id: str
    session_start: float
    action_count: int

# 使用
def agent_node(state: CustomAgentState) -> CustomAgentState:
    messages = state["messages"]
    user_id = state["user_id"]

    # ... Agent 逻辑

    return {
        "messages": [AIMessage(content="回复")],
        "action_count": state["action_count"] + 1
    }
```

---

### 2. 消息过滤和修剪

```python
def trim_messages(state: AgentState) -> AgentState:
    """保留最近 20 条消息"""
    messages = state["messages"]

    if len(messages) > 20:
        # 保留系统消息 + 最近 19 条
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_msgs = messages[-19:]
        trimmed = system_msgs + recent_msgs

        return {"messages": trimmed}

    return state

# 在图中添加修剪节点
graph.add_node("trim", trim_messages)
graph.add_edge("agent", "trim")
graph.add_edge("trim", END)
```

---

### 3. 消息转换

```python
from langchain_core.messages import BaseMessage

def transform_messages(state: AgentState) -> AgentState:
    """转换消息格式（如添加时间戳）"""
    messages = state["messages"]

    transformed = []
    for msg in messages:
        # 添加元数据
        msg.additional_kwargs["timestamp"] = time.time()
        transformed.append(msg)

    return {"messages": transformed}
```

---

## 与其他框架的对比

### LangGraph vs LangChain

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| **状态管理** | 手动管理 | 自动管理（MessagesState） |
| **对话历史** | 需要 Memory 类 | 内置 checkpointer |
| **消息去重** | 手动实现 | 自动（add_messages） |
| **持久化** | 需要外部存储 | 内置 MemorySaver |

**LangChain 示例：**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 手动管理内存
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

# 需要手动保存/加载
chain.predict(input="你好")
```

**LangGraph 示例：**
```python
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 自动管理状态
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 自动保存/加载
app.invoke({"messages": [HumanMessage(content="你好")]}, config)
```

---

### LangGraph vs OpenAI SDK

| 特性 | OpenAI SDK | LangGraph |
|------|------------|-----------|
| **消息管理** | 手动传递 List | 自动管理 |
| **持久化** | 需要自己实现 | 内置 checkpointer |
| **状态流转** | 线性调用 | 图结构 |

**OpenAI SDK 示例：**
```python
from openai import OpenAI

client = OpenAI()
messages = []  # 手动管理

# 第 1 轮
messages.append({"role": "user", "content": "你好"})
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# 第 2 轮
messages.append({"role": "user", "content": "介绍 RAG"})
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
```

**LangGraph 示例：**
```python
# 自动管理，无需手动追加
app.invoke({"messages": [HumanMessage(content="你好")]}, config)
app.invoke({"messages": [HumanMessage(content="介绍 RAG")]}, config)
```

---

## 最佳实践

### 1. 使用 thread_id 隔离会话

```python
# 不同用户使用不同 thread_id
user1_config = {"configurable": {"thread_id": "user_123"}}
user2_config = {"configurable": {"thread_id": "user_456"}}

app.invoke({"messages": [HumanMessage(content="你好")]}, user1_config)
app.invoke({"messages": [HumanMessage(content="你好")]}, user2_config)
```

---

### 2. 定期修剪消息历史

```python
def should_trim(state: AgentState) -> str:
    """判断是否需要修剪"""
    if len(state["messages"]) > 50:
        return "trim"
    return "continue"

graph.add_conditional_edges("agent", should_trim, {
    "trim": "trim_node",
    "continue": END
})
```

---

### 3. 使用系统消息设置上下文

```python
system_prompt = SystemMessage(content="""
你是一个 RAG 系统专家。
- 回答要简洁明了
- 提供代码示例
- 引用相关文档
""")

app.invoke({
    "messages": [system_prompt, HumanMessage(content="介绍 RAG")]
}, config)
```

---

## 关键要点

1. **MessagesState 核心**
   - 使用 Python List 存储消息
   - `add_messages` reducer 自动去重
   - 支持持久化（checkpointer）

2. **性能特性**
   - 添加消息：O(1) 摊销
   - 访问特定轮次：O(1)
   - 消息去重：O(N)

3. **与其他框架对比**
   - LangChain：手动管理内存
   - OpenAI SDK：手动传递消息列表
   - LangGraph：自动状态管理

4. **最佳实践**
   - 使用 thread_id 隔离会话
   - 定期修剪消息历史
   - 系统消息设置上下文

5. **内存占用**
   - List 本身：~1.7 KB（200 条消息）
   - 总内存：~50-100 KB（包含消息内容）

---

## 参考来源（2025-2026）

### LangGraph 官方文档
- **LangGraph Memory Overview** (2026)
  - URL: https://langchain-ai.github.io/langgraph/concepts/memory/
  - 描述：LangGraph 官方内存管理文档

- **LangGraph MessagesState** (2026)
  - URL: https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate
  - 描述：MessagesState 详细说明

- **LangGraph Checkpointers** (2026)
  - URL: https://langchain-ai.github.io/langgraph/concepts/persistence/
  - 描述：持久化机制文档

### LangChain 消息系统
- **LangChain Message Types** (2026)
  - URL: https://python.langchain.com/docs/concepts/messages/
  - 描述：消息类型系统文档

### 实战教程
- **Building Stateful Agents with LangGraph** (2026)
  - URL: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  - 描述：LangGraph 入门教程

- **LangGraph How-to Guides** (2026)
  - URL: https://langchain-ai.github.io/langgraph/how-tos/
  - 描述：LangGraph 实战指南
