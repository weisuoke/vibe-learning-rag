# 核心概念 3：内置 Reducer 函数

## 概述

LangGraph 提供了三个常用的内置 Reducer 函数：`operator.add`、`operator.or_` 和 `add_messages`。这些函数覆盖了大部分常见的状态合并场景，无需编写自定义逻辑。理解它们的工作原理和适用场景，是高效使用 LangGraph 的关键。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## 第一性原理

### 为什么需要内置 Reducer？

在状态管理中，我们经常遇到以下场景：
1. **列表累积**：将新项追加到列表末尾
2. **字典合并**：将新键值对合并到字典中
3. **消息管理**：按 ID 更新或追加消息

如果每次都要编写自定义 Reducer，会导致：
- 代码重复
- 容易出错
- 维护成本高

因此，LangGraph 提供了三个内置 Reducer，覆盖最常见的场景：
- `operator.add`：适用于列表、字符串等支持 `+` 运算的类型
- `operator.or_`：适用于字典合并
- `add_messages`：专门用于消息列表管理

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## operator.add - 列表/字符串拼接

### 基本用法

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    # 列表累积
    items: Annotated[list[str], operator.add]

    # 字符串拼接
    text: Annotated[str, operator.add]

    # 数字列表
    numbers: Annotated[list[int], operator.add]
```

[来源: reference/context7_langgraph_01.md]

### 工作原理

`operator.add` 使用 Python 的 `+` 运算符合并值：

```python
# 列表
old = [1, 2, 3]
new = [4, 5]
result = operator.add(old, new)  # [1, 2, 3, 4, 5]

# 字符串
old = "Hello "
new = "World"
result = operator.add(old, new)  # "Hello World"

# 元组
old = (1, 2)
new = (3, 4)
result = operator.add(old, new)  # (1, 2, 3, 4)
```

**关键特性**：
- 不修改原始值（创建新对象）
- 保持顺序（旧值在前，新值在后）
- 支持任何实现了 `__add__` 方法的类型

[来源: reference/source_annotated_02.md | 源码分析]

### 完整示例

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    counter: int

def node1(state: State) -> dict:
    """第一个节点：添加消息"""
    return {
        "messages": ["Message from node1"],
        "counter": 1
    }

def node2(state: State) -> dict:
    """第二个节点：添加更多消息"""
    return {
        "messages": ["Message from node2"],
        "counter": 2
    }

# 构建图
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()

# 执行
result = graph.invoke({"messages": [], "counter": 0})
print(result)
# {
#     "messages": ["Message from node1", "Message from node2"],
#     "counter": 2
# }
```

**执行流程**：
1. 初始状态：`{"messages": [], "counter": 0}`
2. `node1` 返回：`{"messages": ["Message from node1"], "counter": 1}`
   - `messages`：`[] + ["Message from node1"]` = `["Message from node1"]`
   - `counter`：覆盖为 `1`
3. `node2` 返回：`{"messages": ["Message from node2"], "counter": 2}`
   - `messages`：`["Message from node1"] + ["Message from node2"]`
   - `counter`：覆盖为 `2`

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 适用场景

1. **日志收集**
```python
class State(TypedDict):
    logs: Annotated[list[str], operator.add]
```

2. **步骤记录**
```python
class State(TypedDict):
    steps: Annotated[list[dict], operator.add]
```

3. **文本拼接**
```python
class State(TypedDict):
    content: Annotated[str, operator.add]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## operator.or_ - 字典合并

### 基本用法

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    # 字典合并
    metadata: Annotated[dict, operator.or_]

    # 配置合并
    config: Annotated[dict[str, any], operator.or_]
```

[来源: reference/context7_langgraph_01.md]

### 工作原理

`operator.or_` 使用 Python 的 `|` 运算符合并字典（Python 3.9+）：

```python
# 字典合并
old = {"a": 1, "b": 2}
new = {"b": 3, "c": 4}
result = operator.or_(old, new)  # {"a": 1, "b": 3, "c": 4}

# 新值覆盖旧值
old = {"key": "old_value"}
new = {"key": "new_value"}
result = operator.or_(old, new)  # {"key": "new_value"}
```

**关键特性**：
- 新值覆盖旧值（相同键）
- 保留旧值中不存在于新值的键
- 创建新字典（不修改原始字典）

[来源: reference/source_annotated_02.md | 源码分析]

### 完整示例

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator

class State(TypedDict):
    metadata: Annotated[dict, operator.or_]
    result: str

def collect_user_info(state: State) -> dict:
    """收集用户信息"""
    return {
        "metadata": {
            "user_id": "123",
            "name": "Alice"
        }
    }

def collect_session_info(state: State) -> dict:
    """收集会话信息"""
    return {
        "metadata": {
            "session_id": "abc",
            "timestamp": "2026-02-26"
        }
    }

def collect_preferences(state: State) -> dict:
    """收集偏好设置"""
    return {
        "metadata": {
            "theme": "dark",
            "language": "zh"
        }
    }

# 构建图
builder = StateGraph(State)
builder.add_node("user", collect_user_info)
builder.add_node("session", collect_session_info)
builder.add_node("prefs", collect_preferences)
builder.add_edge(START, "user")
builder.add_edge("user", "session")
builder.add_edge("session", "prefs")
builder.add_edge("prefs", END)

graph = builder.compile()

# 执行
result = graph.invoke({"metadata": {}, "result": ""})
print(result["metadata"])
# {
#     "user_id": "123",
#     "name": "Alice",
#     "session_id": "abc",
#     "timestamp": "2026-02-26",
#     "theme": "dark",
#     "language": "zh"
# }
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 适用场景

1. **元数据收集**
```python
class State(TypedDict):
    metadata: Annotated[dict, operator.or_]
```

2. **配置合并**
```python
class State(TypedDict):
    config: Annotated[dict, operator.or_]
```

3. **统计信息**
```python
class State(TypedDict):
    stats: Annotated[dict[str, int], operator.or_]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## add_messages - 消息列表管理

### 基本用法

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

[来源: reference/context7_langgraph_01.md]

### 工作原理

`add_messages` 是一个智能的消息合并函数，支持：
1. **按 ID 更新**：相同 ID 的消息会被更新而非追加
2. **自动追加**：新消息（无 ID 或 ID 不存在）会被追加
3. **删除消息**：支持 `RemoveMessage` 删除特定消息

```python
from langchain_core.messages import HumanMessage, AIMessage

# 场景 1：追加新消息
old = [HumanMessage(content="Hello", id="1")]
new = [AIMessage(content="Hi!", id="2")]
result = add_messages(old, new)
# [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi!", id="2")]

# 场景 2：更新现有消息
old = [HumanMessage(content="Hello", id="1")]
new = [HumanMessage(content="Hello again", id="1")]
result = add_messages(old, new)
# [HumanMessage(content="Hello again", id="1")]

# 场景 3：混合操作
old = [HumanMessage(content="Hello", id="1")]
new = [
    HumanMessage(content="Hello updated", id="1"),  # 更新
    AIMessage(content="Hi!", id="2")  # 追加
]
result = add_messages(old, new)
# [HumanMessage(content="Hello updated", id="1"), AIMessage(content="Hi!", id="2")]
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 完整示例

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def user_input(state: ConversationState) -> dict:
    """用户输入"""
    return {
        "messages": [HumanMessage(content="What is LangGraph?", id="msg1")]
    }

def ai_response(state: ConversationState) -> dict:
    """AI 响应"""
    return {
        "messages": [AIMessage(content="LangGraph is a framework...", id="msg2")]
    }

def user_followup(state: ConversationState) -> dict:
    """用户追问"""
    return {
        "messages": [HumanMessage(content="Can you explain more?", id="msg3")]
    }

# 构建图
builder = StateGraph(ConversationState)
builder.add_node("user1", user_input)
builder.add_node("ai1", ai_response)
builder.add_node("user2", user_followup)
builder.add_edge(START, "user1")
builder.add_edge("user1", "ai1")
builder.add_edge("ai1", "user2")
builder.add_edge("user2", END)

graph = builder.compile()

# 执行
result = graph.invoke({"messages": []})
print(f"Total messages: {len(result['messages'])}")
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")

# 输出:
# Total messages: 3
# HumanMessage: What is LangGraph?
# AIMessage: LangGraph is a framework...
# HumanMessage: Can you explain more?
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 消息更新示例

```python
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# 初始消息
messages = [
    HumanMessage(content="Hello", id="1"),
    AIMessage(content="Hi there!", id="2")
]

# 更新第一条消息
new_messages = [HumanMessage(content="Hello, updated!", id="1")]
messages = add_messages(messages, new_messages)

print(messages)
# [
#     HumanMessage(content="Hello, updated!", id="1"),
#     AIMessage(content="Hi there!", id="2")
# ]
```

[来源: reference/context7_langgraph_01.md]

### 适用场景

1. **对话系统**
```python
class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

2. **聊天机器人**
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
```

3. **多轮对话**
```python
class DialogState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: dict
```

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

## 内置 Reducer 对比

### 功能对比表

| Reducer | 适用类型 | 合并策略 | 典型场景 |
|---------|---------|---------|---------|
| `operator.add` | list, str, tuple | 拼接（旧+新） | 日志收集、步骤记录 |
| `operator.or_` | dict | 合并（新覆盖旧） | 元数据、配置 |
| `add_messages` | list[BaseMessage] | 按 ID 更新或追加 | 对话系统、聊天 |

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 性能对比

```python
import time
from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

# 测试 operator.add
def benchmark_operator_add(iterations=1000):
    class State(TypedDict):
        items: Annotated[list, operator.add]

    start = time.time()
    items = []
    for i in range(iterations):
        items = operator.add(items, [i])
    end = time.time()

    return (end - start) * 1000  # ms

# 测试 operator.or_
def benchmark_operator_or(iterations=1000):
    class State(TypedDict):
        data: Annotated[dict, operator.or_]

    start = time.time()
    data = {}
    for i in range(iterations):
        data = operator.or_(data, {f"key{i}": i})
    end = time.time()

    return (end - start) * 1000  # ms

# 测试 add_messages
def benchmark_add_messages(iterations=1000):
    start = time.time()
    messages = []
    for i in range(iterations):
        messages = add_messages(messages, [HumanMessage(content=f"msg{i}", id=str(i))])
    end = time.time()

    return (end - start) * 1000  # ms

print(f"operator.add: {benchmark_operator_add():.2f}ms")
print(f"operator.or_: {benchmark_operator_or():.2f}ms")
print(f"add_messages: {benchmark_add_messages():.2f}ms")

# 输出（示例）:
# operator.add: 45.23ms
# operator.or_: 52.18ms
# add_messages: 89.45ms
```

**结论**：
- `operator.add` 最快（简单拼接）
- `operator.or_` 稍慢（字典合并）
- `add_messages` 最慢（需要按 ID 查找和更新）

[来源: reference/search_annotated_github_01.md | 性能优化实践]

## 最佳实践

### 1. 选择合适的 Reducer

```python
# ✅ 好：根据数据类型选择
class State(TypedDict):
    logs: Annotated[list[str], operator.add]  # 列表用 add
    metadata: Annotated[dict, operator.or_]  # 字典用 or_
    messages: Annotated[list[BaseMessage], add_messages]  # 消息用 add_messages

# ❌ 不好：类型不匹配
class State(TypedDict):
    logs: Annotated[list, operator.or_]  # 错误：列表不能用 or_
    metadata: Annotated[dict, operator.add]  # 错误：字典不能用 add
```

[来源: reference/context7_langgraph_01.md]

### 2. 避免状态膨胀

```python
# ❌ 不好：所有数据都累积
class State(TypedDict):
    data: Annotated[list, operator.add]  # 无限增长

# ✅ 好：分离不同类型的数据
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    recent_logs: Annotated[list[str], operator.add]  # 定期清理
    metadata: Annotated[dict, operator.or_]  # 固定大小
```

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

### 3. 使用类型注解

```python
# ✅ 好：明确类型
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    items: Annotated[list[str], operator.add]
    config: Annotated[dict[str, any], operator.or_]

# ❌ 不好：类型不明确
class State(TypedDict):
    messages: Annotated[list, add_messages]
    items: Annotated[list, operator.add]
    config: Annotated[dict, operator.or_]
```

[来源: reference/context7_langgraph_01.md]

## 常见问题

### 问题 1：消息重复

```python
# ❌ 错误：返回完整列表
def node(state):
    return {"messages": state["messages"] + [new_message]}

# ✅ 正确：只返回新消息
def node(state):
    return {"messages": [new_message]}
```

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

### 问题 2：字典键冲突

```python
# 问题：新值会覆盖旧值
old = {"key": "old_value"}
new = {"key": "new_value"}
result = operator.or_(old, new)  # {"key": "new_value"}

# 解决方案：使用嵌套字典
class State(TypedDict):
    data: Annotated[dict[str, dict], operator.or_]

# 或使用自定义 reducer
def merge_nested(old: dict, new: dict) -> dict:
    result = old.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 问题 3：Token 限制

```python
# 问题：消息列表过长
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 解决方案 1：使用滑动窗口
def keep_recent_messages(messages: list, max_count: int = 10):
    return messages[-max_count:]

# 解决方案 2：分离工具结果
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: Annotated[list[dict], operator.add]  # 分离存储
```

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

## 总结

### 核心要点

1. **operator.add**：适用于列表、字符串等支持 `+` 运算的类型
2. **operator.or_**：适用于字典合并，新值覆盖旧值
3. **add_messages**：专门用于消息列表，支持按 ID 更新
4. **性能**：add < or_ < add_messages
5. **类型匹配**：确保 Reducer 与数据类型匹配

### 选择指南

- 需要累积列表？使用 `operator.add`
- 需要合并字典？使用 `operator.or_`
- 需要管理消息？使用 `add_messages`
- 需要复杂逻辑？编写自定义 Reducer

### 下一步

在理解了内置 Reducer 函数后，下一个核心概念将深入讲解**自定义 Reducer 函数**，包括：
- Lambda 函数作为 Reducer
- 普通函数作为 Reducer
- 带参数的 Reducer 工厂函数
- 去重合并、限制大小等实用模式

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

**参考资料**：
- [LangGraph 源码分析](reference/source_annotated_01.md)
- [Annotated 字段解析机制](reference/source_annotated_02.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [Reddit 社区讨论](reference/search_annotated_reddit_01.md)
- [技术文章与教程](reference/search_annotated_github_01.md)
