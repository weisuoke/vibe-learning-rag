# 核心概念 01：Reducer 函数的定义与签名

> 本文档详细讲解 LangGraph 中 Reducer 函数的定义、签名规范、验证机制和使用方式

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/state.py` (行 112-180, 1633-1651)
- `libs/langgraph/langgraph/channels/binop.py` (行 41-135)

**官方文档**:
- Context7 LangGraph 文档 (2026-02-17)
- https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

---

## 1. 概念定义

### 什么是 Reducer 函数？

**Reducer 函数是一个接收两个值并返回合并后值的纯函数，用于定义状态字段的更新策略。**

在 LangGraph 中，当多个节点同时更新同一个状态字段时，Reducer 函数决定如何合并这些更新值。

### 核心特征

1. **函数签名固定**: `(Value, Value) -> Value`
2. **纯函数**: 不产生副作用，相同输入总是产生相同输出
3. **绑定到状态字段**: 使用 `Annotated` 类型注解绑定
4. **自动调用**: 框架在状态更新时自动调用

---

## 2. 函数签名规范

### 2.1 标准签名

```python
from typing import Callable, TypeVar

Value = TypeVar('Value')

# Reducer 函数的标准签名
ReducerFunction = Callable[[Value, Value], Value]
```

**签名要求**:
- **必须接收 2 个参数**: 第一个是旧值，第二个是新值
- **必须返回 1 个值**: 合并后的结果
- **类型一致**: 输入和输出类型必须相同

### 2.2 参数说明

```python
def my_reducer(old_value: Value, new_value: Value) -> Value:
    """
    Args:
        old_value: 当前状态中的旧值
        new_value: 节点返回的新值

    Returns:
        合并后的值
    """
    return merge(old_value, new_value)
```

**参数顺序**:
- **第一个参数**: 当前状态中的值（旧值）
- **第二个参数**: 节点返回的值（新值）

---

## 3. 使用 Annotated 绑定 Reducer

### 3.1 基本语法

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    # 绑定 Reducer 到状态字段
    messages: Annotated[list[str], operator.add]
    counter: int  # 无 Reducer，使用默认覆盖策略
```

**语法结构**:
```python
field_name: Annotated[type, reducer_function]
```

### 3.2 完整示例

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
import operator

# 定义状态 Schema
class State(TypedDict):
    # 列表追加：使用 operator.add
    messages: Annotated[list[str], operator.add]

    # 字典合并：使用 operator.or_
    config: Annotated[dict, operator.or_]

    # 数值累加：使用 operator.add
    total: Annotated[int, operator.add]

    # 无 Reducer：直接覆盖
    status: str

# 定义节点
def node_a(state: State) -> dict:
    return {
        "messages": ["A"],
        "config": {"source": "A"},
        "total": 1,
        "status": "processing"
    }

def node_b(state: State) -> dict:
    return {
        "messages": ["B"],
        "config": {"priority": "high"},
        "total": 2,
        "status": "completed"
    }

# 构建图
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge("__start__", "a")
builder.add_edge("a", "b")
builder.add_edge("b", "__end__")

graph = builder.compile()

# 执行
result = graph.invoke({"messages": [], "config": {}, "total": 0, "status": "init"})

print(result)
# {
#     "messages": ["A", "B"],  # operator.add 合并
#     "config": {"source": "A", "priority": "high"},  # operator.or_ 合并
#     "total": 3,  # operator.add 累加 (0 + 1 + 2)
#     "status": "completed"  # 直接覆盖
# }
```

---

## 4. Reducer 验证机制

### 4.1 源码实现

```python
# 来源: libs/langgraph/langgraph/graph/state.py (行 1633-1651)

from inspect import signature
from typing import Any
from langgraph.channels.binop import BinaryOperatorAggregate

def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    """检查类型是否有有效的 Reducer 函数"""
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if len(params) == 2:
                # 验证通过，创建 BinaryOperatorAggregate
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None
```

### 4.2 验证规则

**验证步骤**:
1. 检查类型是否有 `__metadata__` 属性（来自 `Annotated`）
2. 检查最后一个元数据是否可调用（是否是函数）
3. 检查函数签名是否有恰好 2 个参数
4. 如果不符合，抛出 `ValueError`

**错误示例**:

```python
# ❌ 错误：参数数量不对
def wrong_reducer(value: list) -> list:
    return value + [1]

class State(TypedDict):
    data: Annotated[list, wrong_reducer]  # ValueError: Expected (a, b) -> c

# ❌ 错误：参数数量过多
def wrong_reducer2(a: list, b: list, c: list) -> list:
    return a + b + c

class State(TypedDict):
    data: Annotated[list, wrong_reducer2]  # ValueError: Expected (a, b) -> c
```

**正确示例**:

```python
# ✅ 正确：恰好 2 个参数
def correct_reducer(old: list, new: list) -> list:
    return old + new

class State(TypedDict):
    data: Annotated[list, correct_reducer]  # 验证通过
```

---

## 5. 常见 Reducer 函数

### 5.1 operator.add

**用途**: 列表拼接、字符串拼接、数值相加

```python
import operator

# 列表拼接
class State(TypedDict):
    items: Annotated[list, operator.add]

# 使用
# [1, 2] + [3, 4] -> [1, 2, 3, 4]

# 字符串拼接
class State(TypedDict):
    text: Annotated[str, operator.add]

# 使用
# "Hello" + " World" -> "Hello World"

# 数值相加
class State(TypedDict):
    count: Annotated[int, operator.add]

# 使用
# 5 + 3 -> 8
```

### 5.2 operator.or_

**用途**: 字典合并

```python
import operator

class State(TypedDict):
    config: Annotated[dict, operator.or_]

# 使用
# {"a": 1} | {"b": 2} -> {"a": 1, "b": 2}
# {"a": 1} | {"a": 2} -> {"a": 2}  # 后者覆盖前者
```

### 5.3 add_messages

**用途**: 消息列表合并（按 ID）

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 使用
# 按消息 ID 合并，相同 ID 的消息会被替换
```

---

## 6. 自定义 Reducer 函数

### 6.1 基础自定义

```python
def custom_list_reducer(old: list, new: list) -> list:
    """自定义列表合并：去重追加"""
    result = old.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

class State(TypedDict):
    unique_items: Annotated[list, custom_list_reducer]

# 使用
# [1, 2, 3] + [2, 3, 4] -> [1, 2, 3, 4]  # 去重
```

### 6.2 处理 None 值

```python
def safe_list_reducer(old: list | None, new: list | None) -> list:
    """安全的列表合并：处理 None 值"""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new

class State(TypedDict):
    items: Annotated[list, safe_list_reducer]
```

### 6.3 条件合并

```python
def conditional_reducer(old: int, new: int) -> int:
    """条件合并：只保留较大的值"""
    return max(old, new)

class State(TypedDict):
    max_score: Annotated[int, conditional_reducer]

# 使用
# max(5, 3) -> 5
# max(5, 8) -> 8
```

---

## 7. 实际应用场景

### 7.1 聊天机器人：消息累积

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, AnyMessage

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def chatbot_node(state: ChatState) -> dict:
    # 添加 AI 回复
    return {"messages": [("assistant", "Hello! How can I help?")]}

builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")

graph = builder.compile()

# 执行
result = graph.invoke({"messages": [("user", "Hi")]})
# messages: [HumanMessage("Hi"), AIMessage("Hello! How can I help?")]
```

### 7.2 配置管理：字典合并

```python
import operator

class ConfigState(TypedDict):
    config: Annotated[dict, operator.or_]

def load_default_config(state: ConfigState) -> dict:
    return {"config": {"timeout": 30, "retries": 3}}

def load_user_config(state: ConfigState) -> dict:
    return {"config": {"timeout": 60}}  # 覆盖 timeout

# 结果: {"timeout": 60, "retries": 3}
```

### 7.3 计数器：数值累加

```python
import operator

class CounterState(TypedDict):
    total: Annotated[int, operator.add]
    processed: Annotated[int, operator.add]

def process_batch_a(state: CounterState) -> dict:
    return {"total": 10, "processed": 8}

def process_batch_b(state: CounterState) -> dict:
    return {"total": 15, "processed": 12}

# 结果: {"total": 25, "processed": 20}
```

---

## 8. 常见问题

### Q1: Reducer 什么时候被调用？

**A**: 当多个节点返回同一个状态字段的更新时，Reducer 会被调用来合并这些值。

```python
# 场景 1: 顺序执行
# node_a 返回 {"items": [1]}
# node_b 返回 {"items": [2]}
# Reducer 调用: reducer([1], [2]) -> [1, 2]

# 场景 2: 并行执行
# node_a 和 node_b 并行执行，都返回 {"items": [...]}
# Reducer 调用: reducer(initial, node_a_value)
#              reducer(result, node_b_value)
```

### Q2: 第一个值会调用 Reducer 吗？

**A**: 不会。第一个值直接赋值，不调用 Reducer。

```python
# 初始状态: {"items": []}
# node_a 返回: {"items": [1]}
# 结果: {"items": [1]}  # 直接赋值，不调用 Reducer

# node_b 返回: {"items": [2]}
# 结果: {"items": [1, 2]}  # 调用 Reducer: [1] + [2]
```

### Q3: Reducer 可以修改输入值吗？

**A**: 不推荐。Reducer 应该是纯函数，不应该修改输入值。

```python
# ❌ 错误：修改输入值
def bad_reducer(old: list, new: list) -> list:
    old.extend(new)  # 修改了 old
    return old

# ✅ 正确：创建新值
def good_reducer(old: list, new: list) -> list:
    return old + new  # 创建新列表
```

### Q4: 如何处理复杂类型的合并？

**A**: 自定义 Reducer 函数，实现复杂的合并逻辑。

```python
from typing import Annotated
from typing_extensions import TypedDict

class Document:
    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content

def merge_documents(old: list[Document], new: list[Document]) -> list[Document]:
    """按 ID 合并文档，相同 ID 的文档会被替换"""
    result = {doc.id: doc for doc in old}
    for doc in new:
        result[doc.id] = doc
    return list(result.values())

class State(TypedDict):
    documents: Annotated[list[Document], merge_documents]
```

---

## 9. 最佳实践

### 9.1 使用内置 Reducer

**优先使用内置 Reducer**，它们经过充分测试和优化。

```python
import operator
from langgraph.graph.message import add_messages

# ✅ 推荐：使用内置 Reducer
class State(TypedDict):
    messages: Annotated[list, add_messages]
    items: Annotated[list, operator.add]
    config: Annotated[dict, operator.or_]
```

### 9.2 保持 Reducer 简单

**Reducer 应该简单、快速、无副作用**。

```python
# ✅ 好：简单的合并逻辑
def simple_reducer(old: list, new: list) -> list:
    return old + new

# ❌ 坏：复杂的业务逻辑
def complex_reducer(old: list, new: list) -> list:
    # 调用外部 API
    # 修改数据库
    # 复杂的计算
    return result
```

### 9.3 处理边界情况

**考虑 None、空值等边界情况**。

```python
def safe_reducer(old: list | None, new: list | None) -> list:
    """处理 None 值"""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new
```

### 9.4 类型注解

**使用类型注解提高代码可读性**。

```python
from typing import Annotated

def typed_reducer(old: list[str], new: list[str]) -> list[str]:
    """带类型注解的 Reducer"""
    return old + new

class State(TypedDict):
    items: Annotated[list[str], typed_reducer]
```

---

## 10. 与前端开发的类比

### Redux Reducer

**LangGraph Reducer** 类似于 **Redux Reducer**：

| LangGraph | Redux |
|-----------|-------|
| `Reducer(old, new) -> merged` | `reducer(state, action) -> newState` |
| 合并多个节点的更新 | 合并多个 action 的更新 |
| 使用 `Annotated` 绑定 | 使用 `combineReducers` 组合 |
| 自动调用 | 手动 dispatch |

```python
# LangGraph
class State(TypedDict):
    items: Annotated[list, operator.add]

# Redux (JavaScript)
const reducer = (state = [], action) => {
    switch (action.type) {
        case 'ADD_ITEM':
            return [...state, action.payload];
        default:
            return state;
    }
};
```

---

## 11. 总结

**Reducer 函数是 LangGraph 状态管理的核心机制**：

1. **标准签名**: `(Value, Value) -> Value`
2. **绑定方式**: `Annotated[type, reducer]`
3. **验证严格**: 必须恰好 2 个参数
4. **自动调用**: 框架在状态更新时自动调用
5. **纯函数**: 不产生副作用

**关键要点**:
- 优先使用内置 Reducer (`operator.add`, `operator.or_`, `add_messages`)
- 保持 Reducer 简单、快速、无副作用
- 处理边界情况（None、空值）
- 使用类型注解提高可读性

---

## 参考资源

1. **源码**: `libs/langgraph/langgraph/graph/state.py`
2. **官方文档**: https://langchain-ai.github.io/langgraph/
3. **示例**: https://github.com/langchain-ai/langgraph/tree/main/examples

---

**版本**: v1.0
**最后更新**: 2026-02-26
**维护者**: Claude Code
