# 核心概念：Annotated 与 Reducer 绑定

> LangGraph 状态类型系统的核心机制之一

---

## 概述

在 LangGraph 中，`Annotated` 类型用于将 Reducer 函数绑定到状态字段，实现状态的累积更新而非简单覆盖。这是 LangGraph 状态管理的核心机制之一。

**核心价值**：
- 实现状态的累积更新（如消息列表、历史记录）
- 避免状态覆盖导致的数据丢失
- 提供灵活的状态合并策略

---

## 1. Annotated 类型基础

### 1.1 什么是 Annotated

`Annotated` 是 Python 3.9+ 引入的类型注解特性（PEP 593），用于在类型提示旁边附加元数据。

**基础语法**：
```python
from typing import Annotated

# 语法：Annotated[类型, 元数据1, 元数据2, ...]
x: Annotated[int, "这是一个整数", "范围: 0-100"]
```

**关键特性**：
- 第一个参数是实际类型
- 后续参数是元数据，存储在 `__metadata__` 属性中
- 不影响类型检查，仅提供额外信息

[来源: reference/context7_typing_extensions_01.md | typing_extensions 官方文档]

### 1.2 在 LangGraph 中的应用

LangGraph 使用 `Annotated` 的最后一个元数据作为 Reducer 函数：

```python
from typing import Annotated
from typing_extensions import TypedDict

def add_messages(left: list, right: list) -> list:
    """Reducer 函数：合并两个消息列表"""
    return left + right

class State(TypedDict):
    # 绑定 Reducer 函数到 messages 字段
    messages: Annotated[list, add_messages]
    user_id: str  # 普通字段，使用默认覆盖行为
```

**工作原理**：
- `messages` 字段使用 `add_messages` 函数合并新旧值
- `user_id` 字段使用默认行为（新值覆盖旧值）

---

## 2. Reducer 函数详解

### 2.1 Reducer 函数签名

Reducer 函数必须是**二元操作符**（接受两个参数）：

```python
def reducer(current_value: T, new_value: T) -> T:
    """
    Args:
        current_value: 当前状态中的值
        new_value: 节点返回的新值

    Returns:
        合并后的值
    """
    pass
```

**关键要求**：
- 必须接受恰好 2 个位置参数
- 参数类型应与字段类型一致
- 返回值类型应与字段类型一致

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]

### 2.2 源码实现机制

LangGraph 通过 `_is_field_binop` 函数检测 Reducer：

```python
from inspect import signature

def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    """检查类型是否绑定了二元操作符 Reducer"""
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            # 检查是否恰好有 2 个位置参数
            if (
                sum(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                == 2
            ):
                return BinaryOperatorAggregate(typ, meta[-1])
    return None
```

**检查逻辑**：
1. 检查类型是否有 `__metadata__` 属性
2. 获取最后一个元数据（应该是 Reducer 函数）
3. 检查是否可调用
4. 检查函数签名是否恰好有 2 个位置参数
5. 如果满足条件，包装为 `BinaryOperatorAggregate`

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]

---

## 3. 常见 Reducer 模式

### 3.1 列表累积

**场景**：消息历史、操作日志、事件流

```python
from typing import Annotated
from typing_extensions import TypedDict

def add_messages(left: list, right: list | None) -> list:
    """累积消息列表"""
    if right is None:
        return left
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**使用示例**：
```python
from langgraph.graph import StateGraph

graph = StateGraph(State)

def node1(state: State) -> dict:
    return {"messages": ["Hello"]}

def node2(state: State) -> dict:
    return {"messages": ["World"]}

graph.add_node("node1", node1)
graph.add_node("node2", node2)
graph.add_edge("node1", "node2")
graph.set_entry_point("node1")
graph.set_finish_point("node2")

# 执行后 state["messages"] = ["Hello", "World"]
```

[来源: reference/search_状态类型系统_01.md | LangGraph 社区实践]

### 3.2 字典合并

**场景**：配置累积、元数据聚合

```python
def merge_dicts(left: dict, right: dict | None) -> dict:
    """合并字典，右侧优先"""
    if right is None:
        return left
    return {**left, **right}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

**使用示例**：
```python
def node1(state: State) -> dict:
    return {"metadata": {"user": "alice", "version": 1}}

def node2(state: State) -> dict:
    return {"metadata": {"version": 2, "timestamp": "2026-02-26"}}

# 执行后 state["metadata"] = {
#     "user": "alice",
#     "version": 2,  # 被覆盖
#     "timestamp": "2026-02-26"  # 新增
# }
```

### 3.3 数值累加

**场景**：计数器、统计信息

```python
def add_numbers(left: int, right: int | None) -> int:
    """累加数值"""
    if right is None:
        return left
    return left + right

class State(TypedDict):
    total_tokens: Annotated[int, add_numbers]
```

### 3.4 集合去重

**场景**：标签收集、唯一 ID 追踪

```python
def merge_sets(left: set, right: set | None) -> set:
    """合并集合（自动去重）"""
    if right is None:
        return left
    return left | right

class State(TypedDict):
    tags: Annotated[set, merge_sets]
```

### 3.5 自定义合并逻辑

**场景**：复杂业务规则

```python
from datetime import datetime

def merge_with_timestamp(
    left: list[dict],
    right: list[dict] | None
) -> list[dict]:
    """按时间戳排序合并"""
    if right is None:
        return left

    combined = left + right
    # 按时间戳排序
    return sorted(
        combined,
        key=lambda x: x.get("timestamp", datetime.min)
    )

class State(TypedDict):
    events: Annotated[list[dict], merge_with_timestamp]
```

---

## 4. BinaryOperatorAggregate 包装机制

### 4.1 包装类的作用

LangGraph 使用 `BinaryOperatorAggregate` 包装 Reducer 函数：

```python
class BinaryOperatorAggregate:
    """包装二元操作符 Reducer"""

    def __init__(self, typ: type, func: Callable):
        self.typ = typ  # 字段类型
        self.func = func  # Reducer 函数

    def __call__(self, left, right):
        """执行 Reducer"""
        return self.func(left, right)
```

**包装的好处**：
- 统一接口：所有 Reducer 都通过相同的方式调用
- 类型信息保留：可以访问原始类型
- 元数据管理：方便框架内部处理

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]

### 4.2 Channel 创建流程

LangGraph 在创建 Channel 时处理 Reducer：

```python
def _get_channel(name: str, typ: type) -> BaseChannel:
    """根据类型创建 Channel"""

    # 检查是否绑定了 Reducer
    if binop := _is_field_binop(typ):
        # 创建带 Reducer 的 Channel
        return BinaryOperatorChannel(
            name=name,
            reducer=binop.func,
            default=get_field_default(name, typ, schema)
        )
    else:
        # 创建普通 Channel（覆盖行为）
        return LastValue(
            name=name,
            default=get_field_default(name, typ, schema)
        )
```

**关键点**：
- 有 Reducer：使用 `BinaryOperatorChannel`（累积更新）
- 无 Reducer：使用 `LastValue`（覆盖更新）

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]

---

## 5. 实战代码示例

### 5.1 基础消息累积

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

def add_messages(left: list, right: list | None) -> list:
    """累积消息"""
    if right is None:
        return left
    return left + right

class ChatState(TypedDict):
    messages: Annotated[list[str], add_messages]
    user_id: str

# 创建图
graph = StateGraph(ChatState)

def user_input(state: ChatState) -> dict:
    """用户输入节点"""
    return {"messages": ["User: Hello"]}

def bot_response(state: ChatState) -> dict:
    """机器人响应节点"""
    return {"messages": ["Bot: Hi there!"]}

graph.add_node("user", user_input)
graph.add_node("bot", bot_response)
graph.add_edge("user", "bot")
graph.add_edge("bot", END)
graph.set_entry_point("user")

# 编译并运行
app = graph.compile()
result = app.invoke({"user_id": "alice"})

print(result["messages"])
# 输出: ["User: Hello", "Bot: Hi there!"]
```

### 5.2 多类型 Reducer 组合

```python
from typing import Annotated
from typing_extensions import TypedDict

def add_list(left: list, right: list | None) -> list:
    return left + (right or [])

def merge_dict(left: dict, right: dict | None) -> dict:
    return {**left, **(right or {})}

def add_int(left: int, right: int | None) -> int:
    return left + (right or 0)

class ComplexState(TypedDict):
    messages: Annotated[list, add_list]
    metadata: Annotated[dict, merge_dict]
    token_count: Annotated[int, add_int]
    user_id: str  # 普通字段

# 使用示例
graph = StateGraph(ComplexState)

def node1(state: ComplexState) -> dict:
    return {
        "messages": ["msg1"],
        "metadata": {"source": "node1"},
        "token_count": 10
    }

def node2(state: ComplexState) -> dict:
    return {
        "messages": ["msg2"],
        "metadata": {"timestamp": "2026-02-26"},
        "token_count": 15
    }

graph.add_node("n1", node1)
graph.add_node("n2", node2)
graph.add_edge("n1", "n2")
graph.add_edge("n2", END)
graph.set_entry_point("n1")

app = graph.compile()
result = app.invoke({"user_id": "alice"})

print(result)
# {
#     "messages": ["msg1", "msg2"],
#     "metadata": {"source": "node1", "timestamp": "2026-02-26"},
#     "token_count": 25,
#     "user_id": "alice"
# }
```

### 5.3 条件 Reducer

```python
def smart_merge(left: list, right: list | None) -> list:
    """智能合并：去重并保持顺序"""
    if right is None:
        return left

    # 使用字典保持插入顺序（Python 3.7+）
    seen = {item: None for item in left}
    for item in right:
        seen[item] = None

    return list(seen.keys())

class State(TypedDict):
    unique_items: Annotated[list, smart_merge]

# 测试
graph = StateGraph(State)

def node1(state: State) -> dict:
    return {"unique_items": ["a", "b", "c"]}

def node2(state: State) -> dict:
    return {"unique_items": ["b", "c", "d"]}

graph.add_node("n1", node1)
graph.add_node("n2", node2)
graph.add_edge("n1", "n2")
graph.add_edge("n2", END)
graph.set_entry_point("n1")

app = graph.compile()
result = app.invoke({})

print(result["unique_items"])
# 输出: ["a", "b", "c", "d"]  # 去重且保持顺序
```

---

## 6. 常见错误与解决

### 6.1 错误：Reducer 参数数量不对

```python
# ❌ 错误：只有 1 个参数
def bad_reducer(value: list) -> list:
    return value + ["new"]

class State(TypedDict):
    items: Annotated[list, bad_reducer]  # 不会被识别为 Reducer
```

**解决方案**：
```python
# ✅ 正确：2 个参数
def good_reducer(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right

class State(TypedDict):
    items: Annotated[list, good_reducer]
```

### 6.2 错误：类型不匹配

```python
# ❌ 错误：Reducer 返回类型与字段类型不匹配
def bad_reducer(left: list, right: list) -> str:
    return str(left + right)

class State(TypedDict):
    items: Annotated[list, bad_reducer]  # 运行时错误
```

**解决方案**：
```python
# ✅ 正确：类型一致
def good_reducer(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right

class State(TypedDict):
    items: Annotated[list, good_reducer]
```

### 6.3 错误：忘记处理 None

```python
# ❌ 错误：未处理 None 情况
def bad_reducer(left: list, right: list) -> list:
    return left + right  # 如果 right 是 None 会报错

class State(TypedDict):
    items: Annotated[list, bad_reducer]
```

**解决方案**：
```python
# ✅ 正确：处理 None
def good_reducer(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right

class State(TypedDict):
    items: Annotated[list, good_reducer]
```

---

## 7. 最佳实践

### 7.1 Reducer 函数设计原则

1. **纯函数**：不修改输入参数，返回新值
2. **处理 None**：始终检查 `right` 是否为 None
3. **类型一致**：参数和返回值类型与字段类型一致
4. **幂等性**：多次应用相同输入应产生相同结果
5. **文档化**：清晰说明合并逻辑

### 7.2 何时使用 Reducer

**适合使用 Reducer**：
- 消息历史、聊天记录
- 操作日志、事件流
- 累积统计（计数、求和）
- 配置合并、元数据聚合
- 标签收集、ID 追踪

**不适合使用 Reducer**：
- 简单状态（用户 ID、配置项）
- 需要完全覆盖的字段
- 临时计算结果

### 7.3 性能考虑

```python
# ❌ 低效：每次都创建新列表
def inefficient_reducer(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right  # 创建新列表，O(n)

# ✅ 高效：使用 extend（如果可以修改 left）
def efficient_reducer(left: list, right: list | None) -> list:
    if right is None:
        return left
    result = left.copy()  # 浅拷贝
    result.extend(right)  # O(k)，k 是 right 的长度
    return result
```

**注意**：
- 对于大列表，考虑使用 `deque` 或其他数据结构
- 避免在 Reducer 中执行昂贵的操作
- 考虑使用生成器或惰性求值

---

## 8. 与其他特性的集成

### 8.1 与 Pydantic 集成

```python
from pydantic import BaseModel, Field
from typing import Annotated

def add_messages(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right

class State(BaseModel):
    messages: Annotated[list[str], add_messages] = Field(default_factory=list)
    user_id: str = Field(..., min_length=1)

# 使用
graph = StateGraph(State)
```

**注意事项**：
- Pydantic 会验证字段类型
- Reducer 返回值也会被验证
- 默认值通过 `Field(default_factory=...)` 设置

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 8.2 与 Required/NotRequired 集成

```python
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

def add_list(left: list, right: list | None) -> list:
    if right is None:
        return left
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_list]  # 必需字段 + Reducer
    metadata: NotRequired[Annotated[dict, merge_dict]]  # 可选字段 + Reducer
```

[来源: reference/context7_typing_extensions_01.md | typing_extensions 官方文档]

---

## 9. 调试技巧

### 9.1 打印 Reducer 调用

```python
def debug_reducer(left: list, right: list | None) -> list:
    """带调试信息的 Reducer"""
    print(f"Reducer called: left={left}, right={right}")
    if right is None:
        return left
    result = left + right
    print(f"Reducer result: {result}")
    return result

class State(TypedDict):
    items: Annotated[list, debug_reducer]
```

### 9.2 检查 Reducer 是否生效

```python
from langgraph.graph import StateGraph

graph = StateGraph(State)

# 检查 Channel 类型
print(graph.channels)
# 如果 Reducer 生效，应该看到 BinaryOperatorChannel
# 否则是 LastValue
```

---

## 10. 总结

**核心要点**：
1. `Annotated` 用于绑定 Reducer 函数到状态字段
2. Reducer 必须是二元操作符（2 个参数）
3. LangGraph 通过 `BinaryOperatorAggregate` 包装 Reducer
4. 常见模式：列表累积、字典合并、数值累加、集合去重
5. 最佳实践：纯函数、处理 None、类型一致、文档化

**关键优势**：
- 避免状态覆盖导致的数据丢失
- 灵活的状态合并策略
- 与 Python 类型系统无缝集成

**参考资源**：
- [PEP 593: Flexible function and variable annotations](https://peps.python.org/pep-0593/)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/)
- [typing_extensions 文档](https://github.com/python/typing_extensions)

---

**文档版本**：v1.0
**最后更新**：2026-02-26
**知识点层级**：L2_状态管理 > 04_状态类型系统
