# 核心概念 04：内置 Reducer 函数

> 本文档详细讲解 LangGraph 提供的内置 Reducer 函数，包括 operator.add、operator.or_、add_messages 的使用方式和场景对比

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/message.py` (行 60-244)
- `libs/langgraph/langgraph/channels/binop.py` (行 41-135)

**官方文档**:
- Context7 LangGraph 文档 (2026-02-17)
- https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

---

## 1. 概念定义

### 什么是内置 Reducer 函数？

**内置 Reducer 函数是 LangGraph 框架预定义的、经过优化的状态合并函数，用于处理常见的状态更新场景。**

LangGraph 提供了三类主要的内置 Reducer：
1. **operator.add**: 用于列表拼接、字符串拼接、数值相加
2. **operator.or_**: 用于字典合并
3. **add_messages**: 用于消息列表的智能合并

---

## 2. operator.add

### 2.1 基本用法

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    items: Annotated[list, operator.add]
    text: Annotated[str, operator.add]
    count: Annotated[int, operator.add]
```

### 2.2 列表拼接

```python
import operator
from langgraph.graph import StateGraph

class State(TypedDict):
    items: Annotated[list, operator.add]

def node_a(state: State) -> dict:
    return {"items": [1, 2, 3]}

def node_b(state: State) -> dict:
    return {"items": [4, 5, 6]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge("__start__", "a")
builder.add_edge("a", "b")
builder.add_edge("b", "__end__")

graph = builder.compile()
result = graph.invoke({"items": []})
print(result)
# {'items': [1, 2, 3, 4, 5, 6]}
```

**合并逻辑**:
```python
[1, 2, 3] + [4, 5, 6] = [1, 2, 3, 4, 5, 6]
```

### 2.3 字符串拼接

```python
class State(TypedDict):
    text: Annotated[str, operator.add]

def node_a(state: State) -> dict:
    return {"text": "Hello "}

def node_b(state: State) -> dict:
    return {"text": "World!"}

# 结果
# {'text': 'Hello World!'}
```

**合并逻辑**:
```python
"Hello " + "World!" = "Hello World!"
```

### 2.4 数值累加

```python
class State(TypedDict):
    count: Annotated[int, operator.add]

def node_a(state: State) -> dict:
    return {"count": 10}

def node_b(state: State) -> dict:
    return {"count": 20}

# 结果
# {'count': 30}
```

**合并逻辑**:
```python
10 + 20 = 30
```

### 2.5 使用场景

| 场景 | 示例 | 说明 |
|------|------|------|
| 数据收集 | 多个节点收集数据到列表 | 适合累积型数据 |
| 日志拼接 | 多个节点生成日志文本 | 适合文本累积 |
| 计数器 | 多个节点累加计数 | 适合数值统计 |
| 结果聚合 | 多个节点返回结果列表 | 适合结果汇总 |

---

## 3. operator.or_

### 3.1 基本用法

```python
import operator

class State(TypedDict):
    config: Annotated[dict, operator.or_]
```

### 3.2 字典合并

```python
import operator
from langgraph.graph import StateGraph

class State(TypedDict):
    config: Annotated[dict, operator.or_]

def node_a(state: State) -> dict:
    return {"config": {"timeout": 30, "retries": 3}}

def node_b(state: State) -> dict:
    return {"config": {"timeout": 60, "max_workers": 5}}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge("__start__", "a")
builder.add_edge("a", "b")
builder.add_edge("b", "__end__")

graph = builder.compile()
result = graph.invoke({"config": {}})
print(result)
# {'config': {'timeout': 60, 'retries': 3, 'max_workers': 5}}
```

**合并逻辑**:
```python
{"timeout": 30, "retries": 3} | {"timeout": 60, "max_workers": 5}
= {"timeout": 60, "retries": 3, "max_workers": 5}
# 后者覆盖前者的相同键
```

### 3.3 覆盖规则

```python
# 场景 1: 无冲突
{"a": 1, "b": 2} | {"c": 3, "d": 4}
= {"a": 1, "b": 2, "c": 3, "d": 4}

# 场景 2: 有冲突（后者覆盖前者）
{"a": 1, "b": 2} | {"b": 3, "c": 4}
= {"a": 1, "b": 3, "c": 4}

# 场景 3: 嵌套字典（不递归合并）
{"a": {"x": 1}} | {"a": {"y": 2}}
= {"a": {"y": 2}}  # 整个 "a" 被替换
```

### 3.4 使用场景

| 场景 | 示例 | 说明 |
|------|------|------|
| 配置管理 | 合并默认配置和用户配置 | 适合配置覆盖 |
| 元数据聚合 | 多个节点添加元数据 | 适合元数据累积 |
| 特征合并 | 多个节点提取特征 | 适合特征字典 |
| 参数传递 | 多个节点设置参数 | 适合参数覆盖 |

---

## 4. add_messages

### 4.1 基本用法

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### 4.2 消息累积

```python
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def user_input(state: ChatState) -> dict:
    return {"messages": [("user", "Hello")]}

def chatbot(state: ChatState) -> dict:
    return {"messages": [("assistant", "Hi! How can I help?")]}

builder = StateGraph(ChatState)
builder.add_node("user", user_input)
builder.add_node("bot", chatbot)
builder.add_edge("__start__", "user")
builder.add_edge("user", "bot")
builder.add_edge("bot", "__end__")

graph = builder.compile()
result = graph.invoke({"messages": []})
print(result)
# {
#     'messages': [
#         HumanMessage(content='Hello', id='...'),
#         AIMessage(content='Hi! How can I help?', id='...')
#     ]
# }
```

### 4.3 按 ID 合并

```python
from langchain_core.messages import HumanMessage, AIMessage

# 初始消息
messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2")
]

# 更新消息（相同 ID）
new_messages = [
    HumanMessage(content="Hello again", id="1"),  # 替换 ID="1" 的消息
    AIMessage(content="How are you?", id="3")     # 追加新消息
]

# 合并结果
result = add_messages(messages, new_messages)
# [
#     HumanMessage(content="Hello again", id="1"),  # 被替换
#     AIMessage(content="Hi", id="2"),              # 保留
#     AIMessage(content="How are you?", id="3")     # 追加
# ]
```

### 4.4 RemoveMessage

```python
from langchain_core.messages import RemoveMessage

# 删除特定消息
remove_msg = RemoveMessage(id="1")

messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2")
]

result = add_messages(messages, [remove_msg])
# [AIMessage(content="Hi", id="2")]  # ID="1" 的消息被删除
```

### 4.5 REMOVE_ALL_MESSAGES

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

# 清空所有历史消息
remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)

messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi", id="2")
]

new_messages = [
    remove_all,
    HumanMessage(content="New conversation", id="3")
]

result = add_messages(messages, new_messages)
# [HumanMessage(content="New conversation", id="3")]
# 所有历史消息被清空
```

### 4.6 OpenAI 格式支持

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

### 4.7 使用场景

| 场景 | 示例 | 说明 |
|------|------|------|
| 聊天机器人 | 维护对话历史 | 适合多轮对话 |
| 消息更新 | 替换或删除特定消息 | 适合消息编辑 |
| 对话重置 | 清空历史开始新对话 | 适合会话管理 |
| 格式转换 | OpenAI 格式转换 | 适合多模型兼容 |

---

## 5. 三种 Reducer 对比

### 5.1 功能对比

| Reducer | 数据类型 | 合并策略 | 特殊功能 |
|---------|---------|---------|---------|
| `operator.add` | list, str, int | 拼接/相加 | 无 |
| `operator.or_` | dict | 字典合并（后者覆盖前者） | 无 |
| `add_messages` | list[Message] | 按 ID 合并 | RemoveMessage, REMOVE_ALL, 格式转换 |

### 5.2 使用场景对比

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 场景 1: 数据收集（使用 operator.add）
    results: Annotated[list, operator.add]

    # 场景 2: 配置管理（使用 operator.or_）
    config: Annotated[dict, operator.or_]

    # 场景 3: 对话历史（使用 add_messages）
    messages: Annotated[list, add_messages]

    # 场景 4: 简单覆盖（不使用 Reducer）
    status: str
```

### 5.3 性能对比

| Reducer | 时间复杂度 | 空间复杂度 | 说明 |
|---------|-----------|-----------|------|
| `operator.add` | O(n) | O(n) | n 是列表长度 |
| `operator.or_` | O(n) | O(n) | n 是字典键数量 |
| `add_messages` | O(n*m) | O(n) | n 是消息数量，m 是平均 ID 查找时间 |

---

## 6. 实际应用示例

### 6.1 数据收集系统

```python
import operator
from langgraph.graph import StateGraph

class DataState(TypedDict):
    results: Annotated[list, operator.add]
    metadata: Annotated[dict, operator.or_]

def fetch_source_a(state: DataState) -> dict:
    return {
        "results": [{"source": "A", "data": [1, 2, 3]}],
        "metadata": {"source_a_timestamp": "2026-02-26"}
    }

def fetch_source_b(state: DataState) -> dict:
    return {
        "results": [{"source": "B", "data": [4, 5, 6]}],
        "metadata": {"source_b_timestamp": "2026-02-26"}
    }

builder = StateGraph(DataState)
builder.add_node("a", fetch_source_a)
builder.add_node("b", fetch_source_b)
builder.add_edge("__start__", "a")
builder.add_edge("a", "b")
builder.add_edge("b", "__end__")

graph = builder.compile()
result = graph.invoke({"results": [], "metadata": {}})
print(result)
# {
#     'results': [
#         {'source': 'A', 'data': [1, 2, 3]},
#         {'source': 'B', 'data': [4, 5, 6]}
#     ],
#     'metadata': {
#         'source_a_timestamp': '2026-02-26',
#         'source_b_timestamp': '2026-02-26'
#     }
# }
```

### 6.2 配置管理系统

```python
import operator

class ConfigState(TypedDict):
    config: Annotated[dict, operator.or_]

def load_default_config(state: ConfigState) -> dict:
    return {
        "config": {
            "timeout": 30,
            "retries": 3,
            "max_workers": 10
        }
    }

def load_user_config(state: ConfigState) -> dict:
    return {
        "config": {
            "timeout": 60,  # 覆盖默认值
            "debug": True   # 添加新配置
        }
    }

# 结果
# {
#     'config': {
#         'timeout': 60,      # 被覆盖
#         'retries': 3,       # 保留
#         'max_workers': 10,  # 保留
#         'debug': True       # 新增
#     }
# }
```

### 6.3 聊天机器人系统

```python
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def user_input(state: ChatState) -> dict:
    return {"messages": [("user", "What's the weather?")]}

def chatbot(state: ChatState) -> dict:
    return {"messages": [("assistant", "It's sunny today!")]}

def user_followup(state: ChatState) -> dict:
    return {"messages": [("user", "Thanks!")]}

# 结果
# {
#     'messages': [
#         HumanMessage(content="What's the weather?"),
#         AIMessage(content="It's sunny today!"),
#         HumanMessage(content="Thanks!")
#     ]
# }
```

---

## 7. 常见问题

### Q1: 如何选择合适的 Reducer？

**A**: 根据数据类型和合并需求选择：

```python
# 列表追加 -> operator.add
items: Annotated[list, operator.add]

# 字典合并 -> operator.or_
config: Annotated[dict, operator.or_]

# 消息列表 -> add_messages
messages: Annotated[list, add_messages]

# 简单覆盖 -> 不使用 Reducer
status: str
```

### Q2: operator.or_ 会递归合并嵌套字典吗？

**A**: 不会。operator.or_ 只合并顶层键，嵌套字典会被整体替换。

```python
# 示例
{"a": {"x": 1, "y": 2}} | {"a": {"z": 3}}
= {"a": {"z": 3}}  # 整个 "a" 被替换，不是递归合并
```

如果需要递归合并，使用自定义 Reducer：

```python
def deep_merge(old: dict, new: dict) -> dict:
    """递归合并字典"""
    result = old.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

### Q3: add_messages 如何处理没有 ID 的消息？

**A**: 自动生成 UUID。

```python
# 没有 ID 的消息
msg = HumanMessage(content="Hello")

# add_messages 自动生成 ID
# msg.id = "uuid-generated-id"
```

### Q4: 如何清空对话历史？

**A**: 使用 REMOVE_ALL_MESSAGES。

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage

return {
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        HumanMessage(content="New conversation")
    ]
}
```

---

## 8. 最佳实践

### 8.1 优先使用内置 Reducer

```python
# ✅ 推荐：使用内置 Reducer
import operator
from langgraph.graph.message import add_messages

class State(TypedDict):
    items: Annotated[list, operator.add]
    config: Annotated[dict, operator.or_]
    messages: Annotated[list, add_messages]

# ❌ 不推荐：自定义简单的 Reducer
def my_list_reducer(old: list, new: list) -> list:
    return old + new  # 等同于 operator.add
```

### 8.2 理解合并语义

```python
# operator.add: 拼接
[1, 2] + [3, 4] = [1, 2, 3, 4]

# operator.or_: 后者覆盖前者
{"a": 1} | {"a": 2} = {"a": 2}

# add_messages: 按 ID 合并
[Message(id="1", content="A")] + [Message(id="1", content="B")]
= [Message(id="1", content="B")]  # 相同 ID 被替换
```

### 8.3 处理边界情况

```python
# 处理空值
class State(TypedDict):
    items: Annotated[list, operator.add]

# 初始化为空列表
result = graph.invoke({"items": []})

# 不要初始化为 None
# result = graph.invoke({"items": None})  # 可能导致错误
```

### 8.4 避免过度使用

```python
# ✅ 好：只在需要合并时使用 Reducer
class State(TypedDict):
    items: Annotated[list, operator.add]  # 需要累积
    status: str  # 简单覆盖，不需要 Reducer

# ❌ 坏：所有字段都使用 Reducer
class State(TypedDict):
    items: Annotated[list, operator.add]
    status: Annotated[str, operator.add]  # 不需要拼接字符串
```

---

## 9. 与前端开发的类比

### Array.concat() vs operator.add

```python
# LangGraph
items: Annotated[list, operator.add]
[1, 2] + [3, 4] = [1, 2, 3, 4]

# JavaScript
const items = [1, 2].concat([3, 4]);
// [1, 2, 3, 4]
```

### Object.assign() vs operator.or_

```python
# LangGraph
config: Annotated[dict, operator.or_]
{"a": 1} | {"b": 2} = {"a": 1, "b": 2}

# JavaScript
const config = Object.assign({a: 1}, {b: 2});
// {a: 1, b: 2}
```

### Chat History vs add_messages

```python
# LangGraph
messages: Annotated[list, add_messages]

# React (类比)
const [messages, setMessages] = useState([]);
setMessages(prev => [...prev, newMessage]);
```

---

## 10. 总结

**内置 Reducer 函数是 LangGraph 状态管理的核心工具**：

1. **operator.add**: 列表拼接、字符串拼接、数值相加
2. **operator.or_**: 字典合并（后者覆盖前者）
3. **add_messages**: 消息列表智能合并（按 ID）

**关键要点**:
- 优先使用内置 Reducer
- 理解每种 Reducer 的合并语义
- 根据数据类型和需求选择合适的 Reducer
- 处理边界情况（空值、None）

---

## 参考资源

1. **源码**: `libs/langgraph/langgraph/graph/message.py`
2. **官方文档**: https://langchain-ai.github.io/langgraph/
3. **示例**: https://github.com/langchain-ai/langgraph/tree/main/examples

---

**版本**: v1.0
**最后更新**: 2026-02-26
**维护者**: Claude Code
