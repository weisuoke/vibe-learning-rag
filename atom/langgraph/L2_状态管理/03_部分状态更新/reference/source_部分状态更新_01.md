---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/channels/binop.py
  - libs/langgraph/langgraph/_internal/_fields.py
  - libs/langgraph/langgraph/pregel/_write.py
  - libs/langgraph/tests/test_pregel.py
analyzed_at: 2026-02-26
knowledge_point: 03_部分状态更新
---

# 源码分析：LangGraph 部分状态更新机制

## 分析的文件

- `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py` - StateGraph 核心实现
- `sourcecode/langgraph/libs/langgraph/langgraph/channels/binop.py` - Reducer 函数实现
- `sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py` - 字段处理工具
- `sourcecode/langgraph/libs/langgraph/langgraph/pregel/_write.py` - 状态写入机制
- `sourcecode/langgraph/libs/langgraph/tests/test_pregel.py` - 测试用例

## 关键发现

### 1. 部分状态返回机制

**节点函数可以返回部分状态字段：**

```python
# 示例 1：返回单个字段
def node_a(state: State):
    return {"hello": "world"}

# 示例 2：返回多个字段
def node_b(state: State):
    return {
        "now": 123,
        "hello": "again",
    }

# 示例 3：使用 Command 对象更新
def node_c(state: State):
    return Command(goto="b", update={"foo": "bar"})
```

**来源：** `sourcecode/langgraph/libs/langgraph/tests/test_pregel.py:108-283`

### 2. Reducer 函数机制（BinaryOperatorAggregate）

**核心类：BinaryOperatorAggregate**

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    Example:
        import operator
        total = Channels.BinaryOperatorAggregate(int, operator.add)
    """

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        # Initialize with empty value
        try:
            self.value = typ()
        except Exception:
            self.value = MISSING
```

**来源：** `sourcecode/langgraph/libs/langgraph/langgraph/channels/binop.py:41-68`

**关键特性：**
- 使用二元操作符（如 `operator.add`）合并状态
- 支持覆盖模式（Overwrite）
- 自动初始化空值

### 3. Annotated 字段定义

**使用 Annotated 定义字段的更新策略：**

```python
from typing import Annotated
import operator

class MyState(TypedDict):
    myval: Annotated[int, operator.add]  # 使用加法合并
    otherval: bool  # 默认覆盖
```

**来源：** `sourcecode/langgraph/libs/langgraph/tests/test_pregel.py:557-558`

**支持的操作符：**
- `operator.add` - 加法（数值累加、列表拼接）
- 自定义 Reducer 函数

### 4. Pydantic 模型的部分更新

**get_update_as_tuples 函数：**

```python
def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Get Pydantic state update as a list of (key, value) tuples."""
    if isinstance(input, BaseModel):
        keep = input.model_fields_set
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}

    # Only update values that are different from defaults or in the keep set
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (
            value is not None
            or defaults.get(k, MISSING) is not None
            or (keep is not None and k in keep)
        )
    ]
```

**来源：** `sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py:166-188`

**关键特性：**
- 只更新被显式设置的字段（`model_fields_set`）
- 跳过默认值字段
- 支持 None 值的特殊处理

### 5. 状态写入机制

**ChannelWrite 类：**

```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    """Channel name to write to."""
    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the input."""
    skip_none: bool = False
    """Whether to skip writing if the value is None."""
    mapper: Callable | None = None
    """Function to transform the value before writing."""

class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]
    """Function to extract tuples from value."""
    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the input."""
    static: Sequence[tuple[str, Any, str | None]] | None = None
    """Optional, declared writes for static analysis."""
```

**来源：** `sourcecode/langgraph/libs/langgraph/langgraph/pregel/_write.py:26-43`

**关键特性：**
- 支持 PASSTHROUGH 模式（直接传递输入）
- 支持 skip_none（跳过 None 值）
- 支持 mapper 函数（转换值）

### 6. Overwrite 模式

**_get_overwrite 函数：**

```python
def _get_overwrite(value: Any) -> tuple[bool, Any]:
    """Inspects the given value and returns (is_overwrite, overwrite_value)."""
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict) and set(value.keys()) == {OVERWRITE}:
        return True, value[OVERWRITE]
    return False, None
```

**来源：** `sourcecode/langgraph/libs/langgraph/langgraph/channels/binop.py:32-38`

**关键特性：**
- 支持显式覆盖模式
- 可以使用 `Overwrite(value)` 或 `{OVERWRITE: value}`

## 核心概念总结

### 1. 返回部分字段
- 节点函数可以只返回状态的部分字段
- 未返回的字段保持原值不变
- 支持字典、Pydantic 模型、Command 对象

### 2. 状态增量更新
- 通过 Reducer 函数实现增量更新
- 使用 `BinaryOperatorAggregate` 类
- 支持 `operator.add` 等标准操作符

### 3. 更新策略
- **默认策略**：覆盖（LastValue）
- **Reducer 策略**：使用 Annotated 定义
- **Overwrite 模式**：显式覆盖 Reducer

### 4. Pydantic 集成
- 只更新被显式设置的字段
- 跳过默认值字段
- 支持 None 值的特殊处理

### 5. Command 对象
- 支持在返回值中使用 Command 对象
- 可以同时更新状态和控制流程

## 技术点识别

1. **节点返回值处理** - 如何处理节点返回的部分状态
2. **Reducer 函数机制** - 如何使用 Reducer 函数合并状态
3. **Annotated 字段** - 如何使用 Annotated 定义字段的更新策略
4. **Command 对象** - 如何使用 Command 对象更新状态
5. **BinaryOperatorAggregate** - 二元操作符聚合类
6. **get_update_as_tuples** - 获取 Pydantic 状态更新为元组列表
7. **Overwrite 模式** - 显式覆盖 Reducer

## 依赖库识别

1. **langgraph** - 核心框架
2. **typing_extensions** - Annotated 类型支持
3. **pydantic** - 状态验证（可选）
4. **operator** - 标准库，提供操作符函数
