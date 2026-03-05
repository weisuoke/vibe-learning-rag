---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/graph/message.py
  - libs/langgraph/langgraph/_internal/_fields.py
  - libs/langgraph/langgraph/types.py
analyzed_at: 2026-02-26
knowledge_point: 05_Annotated字段
---

# 源码分析：Annotated 字段在 LangGraph 中的实现

## 分析的文件

### 1. `libs/langgraph/langgraph/graph/state.py`
**核心类**: `StateGraph`

**关键发现**:
- StateGraph 类的文档字符串（行 112-180）明确说明：
  - "Each state key can optionally be annotated with a reducer function"
  - "The signature of a reducer function is `(Value, Value) -> Value`"

**示例代码**（行 149-156）:
```python
def reducer(a: list, b: int | None) -> list:
    if b is not None:
        return a + [b]
    return a

class State(TypedDict):
    x: Annotated[list, reducer]
```

**关键方法**:
- `_add_schema()` (行 257-287): 处理状态 schema，调用 `_get_channels()` 来解析 Annotated 字段
- `_warn_invalid_state_schema()` (行 93-102): 验证 state schema 的有效性

### 2. `libs/langgraph/langgraph/graph/message.py`
**核心函数**: `add_messages`

**关键发现**:
- `add_messages` 是一个预定义的 reducer 函数（行 60-244）
- 用于合并消息列表，支持按 ID 更新现有消息
- 支持 `RemoveMessage` 和 `REMOVE_ALL_MESSAGES` 特殊操作

**使用示例**（行 116-117）:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**MessagesState**（行 307-308）:
```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### 3. `libs/langgraph/langgraph/_internal/_fields.py`
**核心函数**: 字段处理相关的内部实现

**关键发现**:
- `get_cached_annotated_keys()` (行 196-213): 获取带注解的键
  - 使用 `WeakKeyDictionary` 缓存结果
  - 遍历类的 MRO（Method Resolution Order）
  - 从 `__annotations__` 中提取键

- `_is_optional_type()` (行 15-37): 检查类型是否为 Optional
  - 处理 `Annotated` 类型的递归检查
  - 支持新的 union 语法（PEP 604）

- `_is_required_type()` (行 40-56): 检查是否标记为 Required/NotRequired
  - 处理 `Annotated` 类型的递归检查

- `_is_readonly_type()` (行 59-73): 检查是否标记为 ReadOnly
  - 处理 `Annotated` 类型的递归检查

### 4. `libs/langgraph/langgraph/types.py`
**关键导入**:
```python
from langgraph._internal._fields import get_cached_annotated_keys, get_update_as_tuples
```

## Annotated 字段的使用模式

### 模式 1: 使用内置 operator
```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    counters: Annotated[dict, operator.or_]
```

### 模式 2: 使用自定义 reducer
```python
def custom_reducer(old: list, new: list) -> list:
    return old + new

class State(TypedDict):
    items: Annotated[list, custom_reducer]
```

### 模式 3: 使用 lambda 函数
```python
class State(TypedDict):
    value: Annotated[str, lambda x, y: y or x]  # 优先使用新值
    data: Annotated[dict | None, lambda x, y: y if y is not None else x]
```

### 模式 4: 使用预定义函数（如 add_messages）
```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

## 实际使用场景（从 Grep 结果）

### 1. 消息累积
```python
messages: Annotated[list, operator.add]
```

### 2. 字典合并
```python
slack_participants: Annotated[dict, operator.or_]
```

### 3. 条件覆盖
```python
primary_issue_medium: Annotated[str, lambda x, y: y or x]
autoresponse: Annotated[dict | None, lambda _, y: y]  # 总是覆盖
```

### 4. 条件保留
```python
issue: Annotated[dict | None, lambda x, y: y if y else x]
user_info: Annotated[dict | None, lambda x, y: y if y is not None else x]
```

## 关键技术点

1. **类型提示解析**: 使用 `get_type_hints()` 和 `get_origin()` 解析 Annotated 类型
2. **Reducer 函数签名**: `(Value, Value) -> Value`
3. **缓存机制**: 使用 `WeakKeyDictionary` 缓存 annotated keys
4. **MRO 遍历**: 支持继承的状态类
5. **特殊类型处理**: Optional, Required, NotRequired, ReadOnly

## 设计模式

1. **装饰器模式**: Annotated 作为类型装饰器，附加 reducer 元数据
2. **策略模式**: 不同的 reducer 函数实现不同的合并策略
3. **缓存模式**: 使用弱引用字典缓存类型信息

## 下一步调研方向

1. `_get_channels()` 函数的实现细节
2. Annotated 字段如何转换为 Channel 对象
3. BinaryOperatorAggregate 的实现
4. 实际执行时如何调用 reducer 函数
