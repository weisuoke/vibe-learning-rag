# 核心概念 1：Annotated 类型注解语法

## 概述

`Annotated` 是 Python 3.9+ 引入的类型注解增强机制，允许在类型提示中附加元数据。在 LangGraph 中，`Annotated` 被用来为状态字段指定 reducer 函数，控制状态更新的合并策略。

[来源: reference/source_annotated_01.md | LangGraph 源码分析]

## 第一性原理

### 为什么需要 Annotated？

在 LangGraph 的状态管理中，我们需要解决一个核心问题：**如何在不改变类型定义的前提下，为字段附加额外的行为信息？**

传统的类型注解只能表达"这是什么类型"，无法表达"如何处理这个类型"。例如：

```python
class State(TypedDict):
    messages: list  # 只能表达"这是一个列表"
```

但在实际应用中，我们需要表达：
- "这是一个列表，新值应该追加到旧值后面"
- "这是一个字典，新值应该合并到旧值中"
- "这是一个消息列表，按 ID 更新或追加"

`Annotated` 提供了一种优雅的解决方案：**在类型注解中附加元数据，而不改变类型本身**。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## Annotated 语法结构

### 基本语法

```python
from typing import Annotated

# 语法格式
Annotated[type, metadata1, metadata2, ...]

# 示例
Annotated[list, operator.add]
Annotated[dict, operator.or_]
Annotated[str, "这是一个描述", "可以有多个元数据"]
```

**关键点**：
- 第一个参数是实际的类型（如 `list`, `dict`, `str`）
- 后续参数是元数据，可以是任意对象
- 元数据不影响类型检查，只是附加信息

[来源: reference/source_annotated_01.md]

### 在 LangGraph 中的使用

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    # 使用 operator.add 作为 reducer
    messages: Annotated[list[str], operator.add]

    # 使用 operator.or_ 作为 reducer
    metadata: Annotated[dict, operator.or_]

    # 不使用 Annotated，默认覆盖策略
    counter: int
```

**执行效果**：
- `messages` 字段：新值会追加到旧值后面
- `metadata` 字段：新值会合并到旧值中
- `counter` 字段：新值直接覆盖旧值

[来源: reference/context7_langgraph_01.md]

## Annotated 的内部属性

### `__metadata__` 属性

`__metadata__` 是一个元组，包含所有附加的元数据：

```python
from typing import Annotated
import operator

# 定义 Annotated 类型
typ = Annotated[list, operator.add, "额外信息"]

# 访问元数据
print(typ.__metadata__)
# 输出: (<built-in function add>, '额外信息')

# LangGraph 只使用最后一个元数据作为 reducer
reducer = typ.__metadata__[-1]
```

**关键发现**：
- `__metadata__` 是一个元组，按定义顺序存储
- LangGraph 使用 `meta[-1]` 获取最后一个元数据作为 reducer
- 如果最后一个元数据是可调用对象，则作为 reducer 函数

[来源: reference/source_annotated_02.md | 源码分析]

### `__origin__` 和 `__args__` 属性

这两个属性用于获取原始类型信息：

```python
from typing import Annotated

typ = Annotated[list[str], operator.add]

# 获取原始类型
print(typ.__origin__)  # 输出: list

# 获取类型参数
print(typ.__args__)    # 输出: (list[str],)
```

**用途**：
- `__origin__`：获取去除 `Annotated` 包装后的类型
- `__args__`：获取类型参数（包括泛型参数）

[来源: reference/source_annotated_02.md]

## Python 类型系统中的 Annotated

### typing.get_type_hints() 的关键参数

LangGraph 使用 `get_type_hints(include_extras=True)` 来解析 `Annotated` 类型：

```python
from typing import get_type_hints, Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    counter: int

# 不包含 extras（默认）
hints = get_type_hints(State)
print(hints)
# {'messages': list, 'counter': int}
# ❌ Annotated 信息丢失！

# 包含 extras
hints = get_type_hints(State, include_extras=True)
print(hints)
# {'messages': Annotated[list, operator.add], 'counter': int}
# ✅ Annotated 信息保留！
```

**关键点**：
- `include_extras=False`（默认）：剥离 `Annotated`，只保留原始类型
- `include_extras=True`：保留 `Annotated` 及其元数据
- LangGraph 必须使用 `include_extras=True` 才能获取 reducer 信息

[来源: reference/source_annotated_02.md | 源码分析]

### 类型检查器的行为

```python
from typing import Annotated

# 对于类型检查器（如 mypy），以下两者等价
x: list = [1, 2, 3]
y: Annotated[list, operator.add] = [1, 2, 3]

# 类型检查器只关心 list，忽略 operator.add
```

**设计哲学**：
- `Annotated` 的元数据对类型检查器透明
- 不影响类型兼容性和类型推断
- 只在运行时通过反射访问

[来源: reference/context7_langgraph_01.md]

## 完整示例：解析 Annotated 类型

### 手写解析器

```python
from typing import Annotated, get_type_hints, get_origin, get_args
from typing_extensions import TypedDict
import operator
import inspect

class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    metadata: Annotated[dict, operator.or_]
    counter: int

def parse_annotated_field(name: str, typ: type) -> dict:
    """解析 Annotated 字段，提取类型和 reducer 信息"""
    result = {
        "name": name,
        "is_annotated": False,
        "origin_type": typ,
        "reducer": None,
        "metadata": ()
    }

    # 检查是否是 Annotated 类型
    if hasattr(typ, "__metadata__"):
        result["is_annotated"] = True
        result["metadata"] = typ.__metadata__
        result["origin_type"] = get_origin(typ) or get_args(typ)[0]

        # 检查最后一个元数据是否是可调用对象
        if result["metadata"] and callable(result["metadata"][-1]):
            reducer = result["metadata"][-1]
            sig = inspect.signature(reducer)
            params = list(sig.parameters.values())

            # 验证 reducer 签名：必须有恰好 2 个位置参数
            positional_params = sum(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in params
            )

            if positional_params == 2:
                result["reducer"] = reducer
            else:
                raise ValueError(
                    f"Invalid reducer signature for {name}. "
                    f"Expected (a, b) -> c. Got {sig}"
                )

    return result

# 解析所有字段
hints = get_type_hints(State, include_extras=True)
for name, typ in hints.items():
    info = parse_annotated_field(name, typ)
    print(f"\n字段: {name}")
    print(f"  是否 Annotated: {info['is_annotated']}")
    print(f"  原始类型: {info['origin_type']}")
    print(f"  Reducer: {info['reducer']}")
    print(f"  元数据: {info['metadata']}")
```

**输出**：
```
字段: messages
  是否 Annotated: True
  原始类型: list
  Reducer: <built-in function add>
  元数据: (<built-in function add>,)

字段: metadata
  是否 Annotated: True
  原始类型: dict
  Reducer: <built-in function or_>
  元数据: (<built-in function or_>,)

字段: counter
  是否 Annotated: False
  原始类型: <class 'int'>
  Reducer: None
  元数据: ()
```

[来源: reference/source_annotated_02.md | 源码分析]

## LangGraph 的解析流程

### 完整流程图

```
用户定义状态
    ↓
class State(TypedDict):
    messages: Annotated[list, operator.add]
    ↓
StateGraph.__init__(State)
    ↓
_add_schema(State)
    ↓
_get_channels(State)
    ↓
get_type_hints(State, include_extras=True)
    ↓
返回: {'messages': Annotated[list, operator.add]}
    ↓
_get_channel('messages', Annotated[list, operator.add])
    ↓
_is_field_binop(Annotated[list, operator.add])
    ↓
检查 __metadata__ = (operator.add,)
    ↓
验证 operator.add 签名: (a, b) -> c
    ↓
创建 BinaryOperatorAggregate(list, operator.add)
    ↓
存储到 self.channels['messages']
```

[来源: reference/source_annotated_02.md]

### 源码实现（简化版）

```python
from typing import get_type_hints
from inspect import signature

def _get_channels(schema: type[dict]) -> dict:
    """从状态 schema 中提取所有 channels"""
    if not hasattr(schema, "__annotations__"):
        return {}

    # 关键：include_extras=True
    type_hints = get_type_hints(schema, include_extras=True)

    channels = {}
    for name, typ in type_hints.items():
        channel = _get_channel(name, typ)
        channels[name] = channel

    return channels

def _get_channel(name: str, typ: type):
    """为单个字段创建 channel"""
    # 检查是否是 binop (reducer 函数)
    if channel := _is_field_binop(typ):
        channel.key = name
        return channel

    # 默认使用 LastValue（覆盖模式）
    from langgraph.channels import LastValue
    fallback = LastValue(typ)
    fallback.key = name
    return fallback

def _is_field_binop(typ: type):
    """检查是否是 Annotated 类型且包含 reducer 函数"""
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())

            # 验证签名：必须有恰好 2 个位置参数
            positional_count = sum(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in params
            )

            if positional_count == 2:
                from langgraph.channels import BinaryOperatorAggregate
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )

    return None
```

[来源: reference/source_annotated_02.md | 源码分析]

## 实际应用场景

### 场景 1：对话系统

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    # 消息列表：按 ID 更新或追加
    messages: Annotated[list[BaseMessage], add_messages]

    # 用户信息：直接覆盖
    user_id: str
    session_id: str

    # 元数据：合并
    metadata: Annotated[dict, operator.or_]
```

**使用效果**：
- `messages`：新消息追加，相同 ID 的消息更新
- `user_id`, `session_id`：每次更新直接覆盖
- `metadata`：新元数据合并到旧元数据中

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

### 场景 2：多步推理系统

```python
class ReasoningState(TypedDict):
    # 问题：直接覆盖
    question: str

    # 推理步骤：累积
    thoughts: Annotated[list[str], operator.add]

    # 行动记录：累积
    actions: Annotated[list[dict], operator.add]

    # 观察结果：累积
    observations: Annotated[list[str], operator.add]

    # 最终答案：直接覆盖
    answer: str
```

**使用效果**：
- 每个节点可以添加新的推理步骤、行动、观察
- 所有历史记录自动累积
- 最终答案可以随时更新

[来源: reference/search_annotated_github_01.md | 技术文章]

### 场景 3：代码生成系统

```python
from langgraph.graph.message import AnyMessage, add_messages

class CodeGenState(TypedDict):
    # 错误标志：直接覆盖
    error: str

    # 消息历史：按 ID 管理
    messages: Annotated[list[AnyMessage], add_messages]

    # 生成的代码：直接覆盖
    generation: str

    # 迭代次数：直接覆盖
    iterations: int
```

**使用效果**：
- 每次生成新代码时，`generation` 被覆盖
- 消息历史自动管理，避免重复
- 错误信息和迭代次数实时更新

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

## 常见误区

### 误区 1：忘记使用 Annotated

```python
# ❌ 错误：没有使用 Annotated
class State(TypedDict):
    messages: list  # 会被覆盖，不会累积

# ✅ 正确：使用 Annotated
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 会累积
```

[来源: reference/context7_langgraph_01.md]

### 误区 2：元数据顺序错误

```python
# ❌ 错误：类型和元数据顺序颠倒
messages: Annotated[operator.add, list]  # 类型检查失败

# ✅ 正确：类型在前，元数据在后
messages: Annotated[list, operator.add]
```

### 误区 3：多个元数据时的混淆

```python
# LangGraph 只使用最后一个元数据作为 reducer
messages: Annotated[list, "描述信息", operator.add]  # ✅ 使用 operator.add
messages: Annotated[list, operator.add, "描述信息"]  # ❌ "描述信息" 不是函数
```

[来源: reference/source_annotated_02.md]

## 性能考虑

### 类型提示解析的开销

```python
import time
from typing import get_type_hints

class LargeState(TypedDict):
    field1: Annotated[list, operator.add]
    field2: Annotated[list, operator.add]
    # ... 100 个字段

# 测试解析性能
start = time.time()
for _ in range(1000):
    hints = get_type_hints(LargeState, include_extras=True)
end = time.time()

print(f"解析 1000 次耗时: {end - start:.3f}s")
# 输出: 解析 1000 次耗时: 0.050s
```

**结论**：
- `get_type_hints()` 的开销很小
- LangGraph 在初始化时解析一次，后续不再重复
- 对运行时性能影响可忽略

[来源: reference/search_annotated_github_01.md | 性能优化实践]

## 调试技巧

### 检查 Annotated 是否生效

```python
from typing import get_type_hints

class State(TypedDict):
    messages: Annotated[list, operator.add]

# 检查是否正确解析
hints = get_type_hints(State, include_extras=True)
msg_type = hints["messages"]

print(f"是否 Annotated: {hasattr(msg_type, '__metadata__')}")
print(f"元数据: {msg_type.__metadata__ if hasattr(msg_type, '__metadata__') else None}")
print(f"Reducer: {msg_type.__metadata__[-1] if hasattr(msg_type, '__metadata__') else None}")
```

**输出**：
```
是否 Annotated: True
元数据: (<built-in function add>,)
Reducer: <built-in function add>
```

[来源: reference/source_annotated_02.md]

### 验证 Reducer 签名

```python
from inspect import signature

def validate_reducer(reducer):
    """验证 reducer 函数签名"""
    sig = signature(reducer)
    params = list(sig.parameters.values())

    positional_count = sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    if positional_count != 2:
        raise ValueError(f"Reducer 必须有恰好 2 个位置参数，当前有 {positional_count} 个")

    print(f"✅ Reducer 签名有效: {sig}")

# 测试
validate_reducer(operator.add)  # ✅
validate_reducer(lambda x, y: x + y)  # ✅
validate_reducer(lambda x: x)  # ❌ ValueError
```

[来源: reference/source_annotated_02.md | 源码分析]

## 与其他类型注解的组合

### 与 Optional 组合

```python
from typing import Annotated, Optional

class State(TypedDict):
    # 可选的累积列表
    messages: Annotated[Optional[list], operator.add]

    # 或者使用新语法
    items: Annotated[list | None, operator.add]
```

**注意**：
- `Annotated` 可以包装任何类型，包括 `Optional`
- Reducer 函数需要处理 `None` 值

[来源: reference/source_annotated_01.md]

### 与泛型组合

```python
from typing import Annotated, TypeVar, Generic

T = TypeVar('T')

class GenericState(TypedDict, Generic[T]):
    items: Annotated[list[T], operator.add]

# 使用
IntState = GenericState[int]
StrState = GenericState[str]
```

[来源: reference/context7_langgraph_01.md]

## 总结

### 核心要点

1. **语法结构**：`Annotated[type, metadata1, metadata2, ...]`
2. **元数据访问**：通过 `__metadata__` 属性访问
3. **类型提示解析**：必须使用 `get_type_hints(include_extras=True)`
4. **LangGraph 约定**：最后一个元数据作为 reducer 函数
5. **签名要求**：Reducer 必须有恰好 2 个位置参数

### 最佳实践

1. **明确类型**：始终指定具体的类型参数（如 `list[str]` 而非 `list`）
2. **单一职责**：每个字段只使用一个 reducer
3. **验证签名**：确保 reducer 函数签名正确
4. **文档注释**：为自定义 reducer 添加清晰的文档

### 下一步

在理解了 `Annotated` 类型注解语法后，下一个核心概念将深入讲解 **Reducer 函数签名与验证机制**，包括：
- Reducer 函数的签名要求
- LangGraph 如何验证签名
- 常见签名错误及解决方案

[来源: reference/source_annotated_02.md | 源码分析]

---

**参考资料**：
- [LangGraph 源码分析](reference/source_annotated_01.md)
- [Annotated 字段解析机制](reference/source_annotated_02.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [Reddit 社区讨论](reference/search_annotated_reddit_01.md)
- [技术文章与教程](reference/search_annotated_github_01.md)
