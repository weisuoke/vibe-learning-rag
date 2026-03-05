# 核心概念 08：add_messages 深度解析

> 本文档深入讲解 LangGraph 内置的 add_messages Reducer 函数，包括按 ID 合并、RemoveMessage 机制、REMOVE_ALL_MESSAGES 和 OpenAI 格式支持

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/message.py` (行 60-244)

**官方文档**:
- Context7 LangGraph 文档 (2026-02-17)
- https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

---

## 1. 概念定义

### 什么是 add_messages？

**add_messages 是 LangGraph 内置的专用 Reducer 函数，用于智能合并消息列表，支持按 ID 更新、删除和格式转换。**

与简单的列表拼接（`operator.add`）不同，add_messages 提供了更智能的消息管理功能，特别适合聊天机器人和对话系统。

### 核心特征

1. **按 ID 合并**: 相同 ID 的消息会被替换，而非追加
2. **自动生成 ID**: 没有 ID 的消息会自动分配 UUID
3. **删除机制**: 支持删除特定消息或清空所有消息
4. **格式转换**: 支持 OpenAI 消息格式转换
5. **Append-only**: 默认行为是追加新消息

---

## 2. 函数签名

### 2.1 标准签名

```python
from langgraph.graph.message import add_messages

def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """
    合并两个消息列表，按 ID 更新现有消息

    Args:
        left: 基础消息列表（旧值）
        right: 要合并的消息列表（新值）
        format: 可选的格式转换（"langchain-openai" 或 None）

    Returns:
        合并后的消息列表
    """
```

### 2.2 使用方式

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 使用
# messages 字段会使用 add_messages 合并
```

---

## 3. 按 ID 合并机制

### 3.1 基本原理

```python
# 初始消息
messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2")
]

# 新消息
new_messages = [
    HumanMessage(content="Hello again", id="1"),  # 相同 ID
    AIMessage(content="How are you?", id="3")     # 新 ID
]

# 合并结果
result = add_messages(messages, new_messages)
# [
#     HumanMessage(content="Hello again", id="1"),  # 被替换
#     AIMessage(content="Hi", id="2"),              # 保留
#     AIMessage(content="How are you?", id="3")     # 追加
# ]
```

### 3.2 ID 分配规则

```python
# 场景 1: 消息有 ID
msg = HumanMessage(content="Hello", id="custom-id")
# 使用提供的 ID

# 场景 2: 消息没有 ID
msg = HumanMessage(content="Hello")
# 自动生成 UUID: id="uuid-generated-id"
```

### 3.3 合并逻辑

```python
# 来源: libs/langgraph/langgraph/graph/message.py (行 187-244)

def add_messages(left: Messages, right: Messages, *, format=None) -> Messages:
    # 1. 转换为列表
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]

    # 2. 分配 ID
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for m in right:
        if m.id is None:
            m.id = str(uuid.uuid4())

    # 3. 按 ID 合并
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}

    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            # ID 已存在，替换
            merged[existing_idx] = m
        else:
            # ID 不存在，追加
            merged_by_id[m.id] = len(merged)
            merged.append(m)

    return merged
```

---

## 4. RemoveMessage 机制

### 4.1 删除特定消息

```python
from langchain_core.messages import RemoveMessage

# 创建删除标记
remove_msg = RemoveMessage(id="1")

# 初始消息
messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2")
]

# 合并（删除 ID="1" 的消息）
result = add_messages(messages, [remove_msg])
# [AIMessage(content="Hi", id="2")]
```

### 4.2 删除逻辑

```python
# 来源: libs/langgraph/langgraph/graph/message.py (行 98-110)

ids_to_remove = set()

for m in right:
    if isinstance(m, RemoveMessage):
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            ids_to_remove.add(m.id)
        else:
            raise ValueError(
                f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
            )

# 过滤掉要删除的消息
merged = [m for m in merged if m.id not in ids_to_remove]
```

### 4.3 错误处理

```python
# ❌ 错误：删除不存在的消息
remove_msg = RemoveMessage(id="non-existent-id")
result = add_messages(messages, [remove_msg])
# ValueError: Attempting to delete a message with an ID that doesn't exist ('non-existent-id')
```

---

## 5. REMOVE_ALL_MESSAGES

### 5.1 清空所有消息

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

# 创建清空标记
remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)

# 初始消息
messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2"),
    HumanMessage(content="How are you?", id="3")
]

# 清空并添加新消息
new_messages = [
    remove_all,
    HumanMessage(content="New conversation", id="4")
]

result = add_messages(messages, new_messages)
# [HumanMessage(content="New conversation", id="4")]
# 所有历史消息被清空
```

### 5.2 实现逻辑

```python
# 来源: libs/langgraph/langgraph/graph/message.py (行 85-90)

REMOVE_ALL_MESSAGES = "__remove_all__"

# 检测 REMOVE_ALL_MESSAGES
remove_all_idx = None
for idx, m in enumerate(right):
    if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
        remove_all_idx = idx

# 如果找到，返回 remove_all 之后的所有消息
if remove_all_idx is not None:
    return right[remove_all_idx + 1:]
```

### 5.3 使用场景

```python
# 场景 1: 重置对话
def reset_conversation(state: ChatState) -> dict:
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(content="对话已重置。有什么可以帮您？")
        ]
    }

# 场景 2: 清空历史开始新话题
def start_new_topic(state: ChatState) -> dict:
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content="新话题：...")
        ]
    }
```

---

## 6. OpenAI 格式支持

### 6.1 格式转换

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]

def chatbot_node(state: State) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's an image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "1234"
                        }
                    }
                ]
            }
        ]
    }

# 结果会自动转换为 LangChain 格式
# HumanMessage(
#     content=[
#         {"type": "text", "text": "Here's an image:"},
#         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,1234"}}
#     ]
# )
```

### 6.2 支持的格式

```python
# OpenAI 格式
{
    "role": "user",
    "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "source": {...}}
    ]
}

# 转换为 LangChain 格式
HumanMessage(
    content=[
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "..."}}
    ]
)
```

---

## 7. 实际应用场景

### 7.1 聊天机器人

```python
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def user_input(state: ChatState) -> dict:
    user_message = input("You: ")
    return {"messages": [("user", user_message)]}

def chatbot(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(ChatState)
builder.add_node("user", user_input)
builder.add_node("bot", chatbot)
builder.add_edge("__start__", "user")
builder.add_edge("user", "bot")
builder.add_edge("bot", "user")

graph = builder.compile()
```

### 7.2 消息编辑

```python
def edit_message(state: ChatState, message_id: str, new_content: str) -> dict:
    """编辑特定消息"""
    return {
        "messages": [
            HumanMessage(content=new_content, id=message_id)
        ]
    }

# 使用
# 原消息: HumanMessage(content="Hello", id="1")
# 编辑后: HumanMessage(content="Hi there", id="1")
```

### 7.3 消息删除

```python
def delete_message(state: ChatState, message_id: str) -> dict:
    """删除特定消息"""
    return {
        "messages": [RemoveMessage(id=message_id)]
    }

# 使用
# 删除 ID="1" 的消息
```

### 7.4 对话重置

```python
def reset_conversation(state: ChatState) -> dict:
    """重置对话"""
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(content="对话已重置。")
        ]
    }
```

### 7.5 多模态消息

```python
def send_image(state: ChatState, image_data: str) -> dict:
    """发送图片消息"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }
        ]
    }
```

---

## 8. 常见问题

### Q1: add_messages 和 operator.add 有什么区别？

**A**: add_messages 按 ID 合并，operator.add 简单拼接。

```python
# operator.add: 简单拼接
[msg1, msg2] + [msg3, msg4] = [msg1, msg2, msg3, msg4]

# add_messages: 按 ID 合并
[msg(id="1"), msg(id="2")] + [msg(id="1", new_content), msg(id="3")]
= [msg(id="1", new_content), msg(id="2"), msg(id="3")]
```

### Q2: 如何更新消息内容？

**A**: 返回相同 ID 的新消息。

```python
# 原消息
messages = [HumanMessage(content="Hello", id="1")]

# 更新消息
new_messages = [HumanMessage(content="Hi there", id="1")]

# 结果
result = add_messages(messages, new_messages)
# [HumanMessage(content="Hi there", id="1")]
```

### Q3: 如何批量删除消息？

**A**: 返回多个 RemoveMessage。

```python
def delete_multiple(state: ChatState, ids: list[str]) -> dict:
    return {
        "messages": [RemoveMessage(id=id) for id in ids]
    }
```

### Q4: OpenAI 格式转换需要什么版本？

**A**: 需要 `langchain-core>=0.3.11`。

```python
# 检查版本
import langchain_core
print(langchain_core.__version__)  # 应该 >= 0.3.11
```

---

## 9. 最佳实践

### 9.1 使用 add_messages 而非 operator.add

```python
# ✅ 推荐：使用 add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ❌ 不推荐：使用 operator.add（无法更新/删除消息）
class State(TypedDict):
    messages: Annotated[list, operator.add]
```

### 9.2 显式提供消息 ID

```python
# ✅ 好：显式提供 ID（便于后续更新/删除）
msg = HumanMessage(content="Hello", id="user-msg-1")

# ❌ 可以但不推荐：依赖自动生成的 ID
msg = HumanMessage(content="Hello")  # id 会自动生成
```

### 9.3 处理删除错误

```python
def safe_delete(state: ChatState, message_id: str) -> dict:
    """安全删除消息"""
    # 检查消息是否存在
    message_ids = {m.id for m in state["messages"]}
    if message_id not in message_ids:
        return {}  # 不删除

    return {"messages": [RemoveMessage(id=message_id)]}
```

### 9.4 使用 REMOVE_ALL_MESSAGES 重置对话

```python
# ✅ 好：使用 REMOVE_ALL_MESSAGES
def reset(state: ChatState) -> dict:
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(content="已重置")
        ]
    }

# ❌ 坏：手动删除所有消息
def reset(state: ChatState) -> dict:
    return {
        "messages": [
            RemoveMessage(id=m.id) for m in state["messages"]
        ] + [AIMessage(content="已重置")]
    }
```

---

## 10. 与前端开发的类比

### React 消息列表管理

**add_messages** 类似于 **React 的消息列表状态管理**：

```python
# LangGraph
messages: Annotated[list, add_messages]

# React (JavaScript)
const [messages, setMessages] = useState([]);

// 添加消息
setMessages(prev => [...prev, newMessage]);

// 更新消息
setMessages(prev => prev.map(m =>
    m.id === messageId ? {...m, content: newContent} : m
));

// 删除消息
setMessages(prev => prev.filter(m => m.id !== messageId));
```

---

## 11. 调试技巧

### 11.1 打印消息 ID

```python
def debug_messages(state: ChatState) -> dict:
    """打印所有消息 ID"""
    for msg in state["messages"]:
        print(f"ID: {msg.id}, Content: {msg.content}")
    return {}
```

### 11.2 追踪消息变化

```python
def track_changes(old_messages: list, new_messages: list):
    """追踪消息变化"""
    old_ids = {m.id for m in old_messages}
    new_ids = {m.id for m in new_messages}

    added = new_ids - old_ids
    removed = old_ids - new_ids
    updated = old_ids & new_ids

    print(f"Added: {added}")
    print(f"Removed: {removed}")
    print(f"Updated: {updated}")
```

### 11.3 验证消息格式

```python
def validate_messages(messages: list):
    """验证消息格式"""
    for msg in messages:
        assert hasattr(msg, "id"), "Message must have ID"
        assert hasattr(msg, "content"), "Message must have content"
        assert msg.id is not None, "Message ID cannot be None"
```

---

## 12. 总结

**add_messages 是 LangGraph 中最强大的内置 Reducer**：

1. **按 ID 合并**: 相同 ID 的消息会被替换
2. **自动生成 ID**: 没有 ID 的消息会自动分配 UUID
3. **删除机制**: 支持删除特定消息或清空所有消息
4. **格式转换**: 支持 OpenAI 消息格式转换
5. **Append-only**: 默认行为是追加新消息

**关键要点**:
- 优先使用 add_messages 而非 operator.add
- 显式提供消息 ID 便于后续管理
- 使用 RemoveMessage 删除消息
- 使用 REMOVE_ALL_MESSAGES 重置对话
- 使用 format="langchain-openai" 转换格式

**使用场景**:
- 聊天机器人（维护对话历史）
- 消息编辑（更新特定消息）
- 消息删除（删除特定消息）
- 对话重置（清空历史）
- 多模态消息（图片、文本混合）

---

## 参考资源

1. **源码**: `libs/langgraph/langgraph/graph/message.py`
2. **官方文档**: https://langchain-ai.github.io/langgraph/
3. **示例**: https://github.com/langchain-ai/langgraph/tree/main/examples

---

**版本**: v1.0
**最后更新**: 2026-02-26
**维护者**: Claude Code
