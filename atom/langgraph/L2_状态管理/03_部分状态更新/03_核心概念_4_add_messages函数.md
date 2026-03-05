# 核心概念4：add_messages 函数

> 专门用于消息列表的 Reducer 函数，实现消息累积而非替换

---

## 概述

`add_messages` 是 LangGraph 提供的内置 Reducer 函数，专门用于处理消息列表的状态更新。它实现了消息追加而非替换的语义，是构建对话式 AI 应用的核心工具。

**[来源: reference/context7_langgraph_01.md, reference/source_部分状态更新_01.md]**

---

## 1. 核心定义

### 什么是 add_messages？

**add_messages 是一个预定义的 Reducer 函数，用于将新消息追加到现有消息列表中，而不是替换整个列表。**

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 使用 add_messages 作为 Reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**[来源: reference/context7_langgraph_01.md:49-58]**

### 为什么需要 add_messages？

在对话式应用中，我们需要维护完整的对话历史：
- 用户的每次输入
- AI 的每次回复
- 系统消息和工具调用

如果使用默认的覆盖策略，每次更新都会丢失之前的消息。`add_messages` 解决了这个问题。

---

## 2. 工作原理

### 2.1 默认行为 vs add_messages

**默认行为（覆盖）：**
```python
class State(TypedDict):
    messages: list[str]  # 没有 Reducer

# 节点返回
def node(state):
    return {"messages": ["new message"]}

# 结果：旧消息被完全替换
# state["messages"] = ["new message"]
```

**使用 add_messages（追加）：**
```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[str], add_messages]

# 节点返回
def node(state):
    return {"messages": ["new message"]}

# 结果：新消息被追加到列表末尾
# state["messages"] = [...旧消息, "new message"]
```

**[来源: reference/context7_langgraph_01.md:125-147]**

### 2.2 消息类型支持

`add_messages` 支持 LangChain 的所有消息类型：

```python
from langchain_core.messages import (
    HumanMessage,      # 用户消息
    AIMessage,         # AI 回复
    SystemMessage,     # 系统消息
    ToolMessage,       # 工具调用结果
    FunctionMessage,   # 函数调用结果
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**[来源: reference/context7_langgraph_01.md:232-237]**

---

## 3. 实战示例

### 3.1 基础对话流

```python
"""
add_messages 基础示例
演示：构建简单的对话流
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ===== 1. 定义状态 =====
class ChatState(TypedDict):
    """对话状态，使用 add_messages 维护消息历史"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 2. 定义节点 =====
def chatbot(state: ChatState) -> dict:
    """简单的聊天机器人节点"""
    # 获取最后一条用户消息
    last_message = state["messages"][-1]

    # 生成回复（这里简化处理）
    response = AIMessage(content=f"你说: {last_message.content}")

    # 返回新消息（会被追加到列表）
    return {"messages": [response]}

# ===== 3. 构建图 =====
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# ===== 4. 运行对话 =====
print("=== 对话示例 ===\n")

# 第一轮对话
result1 = graph.invoke({
    "messages": [HumanMessage(content="你好")]
})
print(f"第一轮后的消息数量: {len(result1['messages'])}")
for msg in result1["messages"]:
    print(f"  {msg.__class__.__name__}: {msg.content}")

print()

# 第二轮对话（继续之前的对话）
result2 = graph.invoke({
    "messages": result1["messages"] + [HumanMessage(content="今天天气怎么样？")]
})
print(f"第二轮后的消息数量: {len(result2['messages'])}")
for msg in result2["messages"]:
    print(f"  {msg.__class__.__name__}: {msg.content}")
```

**运行输出：**
```
=== 对话示例 ===

第一轮后的消息数量: 2
  HumanMessage: 你好
  AIMessage: 你说: 你好

第二轮后的消息数量: 4
  HumanMessage: 你好
  AIMessage: 你说: 你好
  HumanMessage: 今天天气怎么样？
  AIMessage: 你说: 今天天气怎么样？
```

**[来源: reference/context7_langgraph_01.md:149-196]**

### 3.2 与 LLM 集成

```python
"""
add_messages 与 LLM 集成
演示：构建真实的对话 AI
"""

import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    """代理状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 2. 初始化 LLM =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ===== 3. 定义节点 =====
def call_model(state: AgentState) -> dict:
    """调用 LLM 生成回复"""
    # 添加系统消息（如果是第一条消息）
    messages = state["messages"]
    if len(messages) == 1:
        messages = [
            SystemMessage(content="你是一个友好的助手。"),
            *messages
        ]

    # 调用 LLM
    response = llm.invoke(messages)

    # 返回 AI 消息（会被追加）
    return {"messages": [response]}

# ===== 4. 构建图 =====
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()

# ===== 5. 多轮对话 =====
print("=== 多轮对话示例 ===\n")

# 初始化对话
conversation_state = {"messages": []}

# 第一轮
conversation_state = graph.invoke({
    "messages": conversation_state["messages"] + [
        HumanMessage(content="我叫小明，请记住我的名字。")
    ]
})
print(f"AI: {conversation_state['messages'][-1].content}\n")

# 第二轮（测试记忆）
conversation_state = graph.invoke({
    "messages": conversation_state["messages"] + [
        HumanMessage(content="我叫什么名字？")
    ]
})
print(f"AI: {conversation_state['messages'][-1].content}\n")

print(f"总消息数: {len(conversation_state['messages'])}")
```

**[来源: reference/context7_langgraph_01.md:179-196]**

### 3.3 RAG 代理示例

```python
"""
add_messages 在 RAG 中的应用
演示：构建检索增强生成代理
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ===== 1. 定义状态 =====
class RAGState(TypedDict):
    """RAG 代理状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 2. 定义节点 =====
def retrieve_documents(state: RAGState) -> dict:
    """检索相关文档"""
    query = state["messages"][-1].content

    # 模拟检索（实际应用中调用向量数据库）
    docs = [
        "文档1: LangGraph 是一个用于构建状态化工作流的框架。",
        "文档2: add_messages 用于追加消息到列表。"
    ]

    # 将检索结果作为系统消息添加
    context_msg = SystemMessage(
        content=f"检索到的相关文档:\n" + "\n".join(docs)
    )

    return {"messages": [context_msg]}

def generate_answer(state: RAGState) -> dict:
    """基于上下文生成答案"""
    # 获取最后的用户问题和检索的文档
    messages = state["messages"]

    # 模拟 LLM 生成（实际应用中调用 LLM）
    answer = AIMessage(
        content="根据检索到的文档，LangGraph 是一个状态化工作流框架，"
                "add_messages 函数用于追加消息。"
    )

    return {"messages": [answer]}

# ===== 3. 构建图 =====
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ===== 4. 运行 RAG 查询 =====
print("=== RAG 查询示例 ===\n")

result = graph.invoke({
    "messages": [HumanMessage(content="什么是 LangGraph？")]
})

print("对话历史:")
for i, msg in enumerate(result["messages"], 1):
    print(f"{i}. {msg.__class__.__name__}: {msg.content[:50]}...")
```

**[来源: reference/context7_langgraph_01.md:179-196, reference/search_部分状态更新_01.md:98-146]**

---

## 4. 类比理解

### 前端类比

**add_messages 就像 React 的 useState 数组更新：**

```javascript
// React 中追加数组元素
const [messages, setMessages] = useState([]);

// ❌ 错误：直接修改
messages.push(newMessage);

// ✅ 正确：创建新数组
setMessages([...messages, newMessage]);
```

```python
# LangGraph 中的 add_messages 自动处理追加
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 节点只需返回新消息
def node(state):
    return {"messages": [new_message]}  # 自动追加
```

### 日常生活类比

**add_messages 就像聊天记录：**
- 每次发送消息，都会追加到聊天记录末尾
- 不会删除之前的消息
- 可以随时查看完整的对话历史

**[来源: reference/search_部分状态更新_01.md:98-126]**

---

## 5. 常见误区

### 误区1：以为 add_messages 会去重 ❌

**错误理解：**
"add_messages 会自动去除重复的消息"

**正确理解：**
`add_messages` 只是简单地追加消息，不会进行去重。如果需要去重，需要自定义 Reducer。

```python
# add_messages 不会去重
def node1(state):
    return {"messages": [HumanMessage(content="hello")]}

def node2(state):
    return {"messages": [HumanMessage(content="hello")]}  # 会重复

# 结果：["hello", "hello"]
```

**[来源: reference/search_部分状态更新_01.md:119-126]**

### 误区2：以为可以用于任意列表 ❌

**错误理解：**
"add_messages 可以用于任何列表类型"

**正确理解：**
`add_messages` 专门设计用于 LangChain 的消息类型（`BaseMessage` 及其子类）。对于普通列表，应该使用 `operator.add`。

```python
# ✅ 正确：用于消息列表
messages: Annotated[Sequence[BaseMessage], add_messages]

# ❌ 错误：用于普通列表
items: Annotated[list[str], add_messages]  # 类型不匹配

# ✅ 正确：用于普通列表
items: Annotated[list[str], operator.add]
```

**[来源: reference/context7_langgraph_01.md:232-237]**

### 误区3：以为必须返回列表 ❌

**错误理解：**
"使用 add_messages 时，节点必须返回消息列表"

**正确理解：**
节点可以返回单个消息或消息列表，`add_messages` 都能正确处理。

```python
# ✅ 返回单个消息
def node1(state):
    return {"messages": AIMessage(content="hello")}

# ✅ 返回消息列表
def node2(state):
    return {"messages": [AIMessage(content="hello")]}

# 两种方式都会正确追加
```

**[来源: reference/source_部分状态更新_01.md:27-48]**

---

## 6. 最佳实践

### 6.1 使用 Sequence 类型注解

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage

# ✅ 推荐：使用 Sequence（更通用）
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ⚠️ 可以但不推荐：使用 list
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**原因：** `Sequence` 是更通用的类型，支持列表、元组等多种序列类型。

**[来源: reference/context7_langgraph_01.md:49-58, reference/search_部分状态更新_01.md:119-136]**

### 6.2 节点应该是纯函数

```python
# ✅ 推荐：纯函数，返回新消息
def chatbot(state: State) -> dict:
    response = generate_response(state["messages"])
    return {"messages": [response]}

# ❌ 避免：直接修改状态
def chatbot(state: State) -> dict:
    state["messages"].append(response)  # 不要这样做！
    return state
```

**原因：** 纯函数更容易测试、调试和理解。LangGraph 会自动处理状态合并。

**[来源: reference/search_部分状态更新_01.md:119-126]**

### 6.3 合理使用系统消息

```python
def add_system_context(state: State) -> dict:
    """在需要时添加系统消息"""
    messages = state["messages"]

    # 只在第一次添加系统消息
    if not any(isinstance(m, SystemMessage) for m in messages):
        return {"messages": [
            SystemMessage(content="你是一个专业的助手。")
        ]}

    return {"messages": []}  # 不添加新消息
```

**[来源: reference/search_部分状态更新_01.md:98-146]**

---

## 7. 与其他 Reducer 的对比

### add_messages vs operator.add

| 特性 | add_messages | operator.add |
|------|--------------|--------------|
| 适用类型 | `BaseMessage` 及子类 | 任意支持 `+` 的类型 |
| 主要用途 | 对话历史维护 | 通用列表/数值累加 |
| 特殊处理 | 理解消息语义 | 简单的 `+` 操作 |
| 类型安全 | 强类型检查 | 依赖 Python 类型系统 |

```python
import operator

class State(TypedDict):
    # 用于消息
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 用于普通列表
    items: Annotated[list[str], operator.add]

    # 用于数值累加
    counter: Annotated[int, operator.add]
```

**[来源: reference/context7_langgraph_01.md:24-45, reference/source_部分状态更新_01.md:79-97]**

---

## 8. 源码实现原理

### 8.1 add_messages 的内部实现

虽然我们不需要深入源码，但了解基本原理有助于更好地使用：

```python
# 简化的 add_messages 实现逻辑
def add_messages(current: Sequence[BaseMessage], new: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """
    将新消息追加到当前消息列表

    Args:
        current: 当前的消息列表
        new: 要追加的新消息

    Returns:
        合并后的消息列表
    """
    # 处理单个消息的情况
    if isinstance(new, BaseMessage):
        new = [new]

    # 追加新消息
    return list(current) + list(new)
```

**关键点：**
1. 接收当前值和新值两个参数
2. 返回合并后的新列表
3. 不修改原始列表（不可变性）

**[来源: reference/source_部分状态更新_01.md:49-73]**

### 8.2 Reducer 的调用时机

```
节点执行 → 返回部分状态 → 调用 Reducer → 合并到完整状态
    ↓
{"messages": [new_msg]}
    ↓
add_messages(current_messages, [new_msg])
    ↓
[...current_messages, new_msg]
```

**[来源: reference/source_部分状态更新_01.md:183-209, reference/search_部分状态更新_01.md:137-146]**

---

## 9. 实际应用场景

### 场景1：客服机器人

```python
class CustomerServiceState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    customer_id: str
    issue_resolved: bool

# 维护完整的客服对话历史
# 用于后续分析和质量评估
```

### 场景2：多轮问答系统

```python
class QAState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

# 支持上下文相关的多轮问答
# 每次回答都基于完整的对话历史
```

### 场景3：代码助手

```python
class CodeAssistantState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    code: str
    errors: list[str]

# 维护代码修改的完整对话
# 帮助理解用户的意图和上下文
```

**[来源: reference/context7_langgraph_01.md:149-196]**

---

## 10. 总结

### 核心要点

1. **专用性**：`add_messages` 专门用于 LangChain 消息类型
2. **追加语义**：实现消息追加而非替换
3. **对话历史**：是构建对话式 AI 的核心工具
4. **类型安全**：提供强类型检查和 IDE 支持
5. **纯函数**：节点应该返回新消息，不修改状态

### 一句话总结

**`add_messages` 是 LangGraph 提供的专用 Reducer 函数，用于将新消息追加到对话历史中，是构建对话式 AI 应用的核心工具。**

---

## 参考资料

### 官方文档
- [LangGraph 官方文档 - State Management](https://github.com/langchain-ai/langgraph)
- [Context7 文档](https://context7.com/langchain-ai/langgraph/llms.txt)

### 源码分析
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/message.py` - add_messages 实现
- `sourcecode/langgraph/libs/langgraph/tests/test_pregel.py` - 测试用例

### 社区资源
- [LangGraph Best Practices](https://www.swarnendu.de/blog/langgraph-best-practices)
- [LangGraph for Beginners](https://dev.to/petrashka/langgraph-for-beginners-a-complete-guide-2310)

---

**版本：** v1.0
**创建时间：** 2026-02-26
**维护者：** Claude Code
