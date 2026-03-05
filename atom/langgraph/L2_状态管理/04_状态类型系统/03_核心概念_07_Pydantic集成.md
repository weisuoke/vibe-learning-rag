# 核心概念 07：Pydantic 集成

## 概述

LangGraph 对 Pydantic 模型提供了特殊的集成支持，通过 `model_fields_set` 属性实现智能的状态更新机制。这种集成允许框架区分"显式设置的字段"和"使用默认值的字段"，从而实现更精确的状态管理。

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py]

---

## 核心机制

### 1. model_fields_set 跟踪机制

Pydantic 的 `model_fields_set` 属性记录了在模型初始化时被显式提供的字段名称集合。

**源码实现**：

```python
def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Get Pydantic state update as a list of (key, value) tuples."""
    if isinstance(input, BaseModel):
        keep = input.model_fields_set  # 跟踪显式设置的字段
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}

    # 只更新与默认值不同的字段或在 model_fields_set 中的字段
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

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py:166-189]

**关键特性**：
- `model_fields_set`：存储显式设置的字段名称
- `model_fields`：包含所有字段的元数据（类型、默认值等）
- 只有满足条件的字段才会被更新到状态中

---

### 2. 字段更新逻辑

LangGraph 使用三个条件判断字段是否应该更新：

```python
# 条件 1：值不是 None
value is not None

# 条件 2：默认值不是 None（即使当前值是 None）
or defaults.get(k, MISSING) is not None

# 条件 3：字段在 model_fields_set 中（显式设置）
or (keep is not None and k in keep)
```

**更新规则**：
1. **非 None 值**：总是更新
2. **None 值但默认值不是 None**：更新（表示用户显式设置为 None）
3. **字段在 model_fields_set 中**：更新（即使值等于默认值）

---

### 3. 特殊的 None 值处理

None 值的处理是 Pydantic 集成中最微妙的部分。

**场景分析**：

```python
from pydantic import BaseModel

class State(BaseModel):
    x: int = 10
    y: int | None = None
    z: int | None = 20

# 场景 1：显式设置为 None
state1 = State(x=5, y=None)
# model_fields_set = {'x', 'y'}
# 更新：x=5, y=None（y 在 model_fields_set 中）

# 场景 2：使用默认值 None
state2 = State(x=5)
# model_fields_set = {'x'}
# 更新：x=5（y 不在 model_fields_set 中，不更新）

# 场景 3：None 值但默认值不是 None
state3 = State(x=5, z=None)
# model_fields_set = {'x', 'z'}
# 更新：x=5, z=None（z 的默认值是 20，不是 None）
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 实战示例

### 示例 1：基础 Pydantic 状态

```python
from pydantic import BaseModel
from langgraph.graph import StateGraph

class State(BaseModel):
    count: int = 0
    message: str = "default"
    data: list[str] = []

def increment(state: State) -> State:
    # 只更新 count，message 和 data 使用默认值
    return State(count=state.count + 1)

graph = StateGraph(State)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.set_finish_point("increment")

compiled = graph.compile()

# 初始状态
initial = State(count=0, message="hello")
# model_fields_set = {'count', 'message'}

# 执行节点
result = compiled.invoke(initial)
# increment 返回 State(count=1)
# model_fields_set = {'count'}
# 只更新 count，message 保持 "hello"
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]

---

### 示例 2：None 值处理

```python
from pydantic import BaseModel
from typing import Optional

class State(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = 18

def clear_name(state: State) -> State:
    # 显式设置 name 为 None
    return State(user_id=state.user_id, name=None)

def use_default(state: State) -> State:
    # 不设置 name，使用默认值
    return State(user_id=state.user_id)

# 场景 1：显式设置为 None
state1 = State(user_id="123", name="Alice")
result1 = clear_name(state1)
# result1.model_fields_set = {'user_id', 'name'}
# 更新：user_id="123", name=None

# 场景 2：使用默认值
state2 = State(user_id="123", name="Alice")
result2 = use_default(state2)
# result2.model_fields_set = {'user_id'}
# 更新：user_id="123"（name 保持 "Alice"）
```

---

### 示例 3：部分更新模式

```python
from pydantic import BaseModel, Field

class UserState(BaseModel):
    id: str
    name: str = "Unknown"
    email: str = ""
    age: int = 0
    verified: bool = False

def update_email(state: UserState) -> UserState:
    # 只更新 email，其他字段保持不变
    return UserState(id=state.id, email="new@example.com")

def update_multiple(state: UserState) -> UserState:
    # 更新多个字段
    return UserState(
        id=state.id,
        name="Alice",
        verified=True
    )

# 初始状态
initial = UserState(id="123", name="Bob", age=25)
# model_fields_set = {'id', 'name', 'age'}

# 只更新 email
result1 = update_email(initial)
# result1.model_fields_set = {'id', 'email'}
# 最终状态：id="123", name="Bob", age=25, email="new@example.com"

# 更新多个字段
result2 = update_multiple(initial)
# result2.model_fields_set = {'id', 'name', 'verified'}
# 最终状态：id="123", name="Alice", age=25, verified=True
```

---

### 示例 4：与 TypedDict 对比

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

# TypedDict 版本
class TypedDictState(TypedDict):
    count: int
    message: str

def typed_dict_update(state: TypedDictState) -> TypedDictState:
    # 必须返回所有字段或使用字典合并
    return {"count": state["count"] + 1}
    # 问题：message 字段会丢失！

# Pydantic 版本
class PydanticState(BaseModel):
    count: int
    message: str = "default"

def pydantic_update(state: PydanticState) -> PydanticState:
    # 只返回需要更新的字段
    return PydanticState(count=state.count + 1)
    # message 字段自动保留！
```

[来源: reference/search_状态类型系统_01.md]

---

## 向后兼容性

### 兼容性考虑

源码注释中提到：

```python
# NOTE: This behavior for Pydantic is somewhat inelegant,
# but we keep around for backwards compatibility
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_fields.py:310-312]

**历史原因**：
- 早期版本可能有不同的行为
- 当前实现保证了向后兼容性
- 未来版本可能会优化这个逻辑

---

### Pydantic v1 vs v2

```python
# Pydantic v1
from pydantic import BaseModel

class StateV1(BaseModel):
    x: int

    class Config:
        # v1 配置
        validate_assignment = True

# Pydantic v2
from pydantic import BaseModel, ConfigDict

class StateV2(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    x: int
```

**LangGraph 支持**：
- 同时支持 Pydantic v1 和 v2
- `model_fields_set` 在两个版本中都可用
- 推荐使用 Pydantic v2（性能更好）

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 性能考虑

### 性能对比

| 操作 | TypedDict | Pydantic |
|------|-----------|----------|
| 创建实例 | 快 | 慢（需要验证） |
| 字段访问 | 快 | 快 |
| 类型检查 | 静态 | 运行时 |
| 内存占用 | 小 | 大 |

**性能测试**：

```python
import time
from typing_extensions import TypedDict
from pydantic import BaseModel

class TypedDictState(TypedDict):
    x: int
    y: str

class PydanticState(BaseModel):
    x: int
    y: str

# TypedDict 性能
start = time.time()
for _ in range(100000):
    state = {"x": 1, "y": "test"}
typed_dict_time = time.time() - start

# Pydantic 性能
start = time.time()
for _ in range(100000):
    state = PydanticState(x=1, y="test")
pydantic_time = time.time() - start

print(f"TypedDict: {typed_dict_time:.3f}s")
print(f"Pydantic: {pydantic_time:.3f}s")
# TypedDict: 0.015s
# Pydantic: 0.450s（约 30 倍慢）
```

[来源: reference/search_状态类型系统_01.md]

---

### 优化建议

**1. 使用 model_construct 跳过验证**：

```python
from pydantic import BaseModel

class State(BaseModel):
    x: int
    y: str

# 慢：完整验证
state1 = State(x=1, y="test")

# 快：跳过验证（信任数据源）
state2 = State.model_construct(x=1, y="test")
```

**2. 混合使用 TypedDict 和 Pydantic**：

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

# 内部状态使用 TypedDict（快）
class InternalState(TypedDict):
    messages: list[str]
    context: dict

# API 边界使用 Pydantic（安全）
class APIInput(BaseModel):
    query: str
    user_id: str

def process(api_input: APIInput) -> InternalState:
    # 验证后转换为内部状态
    return {
        "messages": [api_input.query],
        "context": {"user_id": api_input.user_id}
    }
```

[来源: reference/search_状态类型系统_01.md]

---

## 最佳实践

### 1. 何时使用 Pydantic

**推荐场景**：
- API 输入验证
- 复杂数据结构
- 需要类型转换
- 需要自定义验证器

**不推荐场景**：
- 简单内部状态
- 性能敏感的场景
- 高频率状态更新

---

### 2. 字段设计原则

```python
from pydantic import BaseModel, Field

class GoodState(BaseModel):
    # 必需字段：不提供默认值
    id: str

    # 可选字段：提供合理的默认值
    name: str = "Unknown"

    # 可空字段：使用 Optional
    email: str | None = None

    # 带验证的字段：使用 Field
    age: int = Field(default=0, ge=0, le=150)

    # 列表字段：使用 default_factory
    tags: list[str] = Field(default_factory=list)
```

---

### 3. 避免常见陷阱

**陷阱 1：可变默认值**：

```python
# 错误：共享可变对象
class BadState(BaseModel):
    items: list = []  # 所有实例共享同一个列表！

# 正确：使用 default_factory
class GoodState(BaseModel):
    items: list = Field(default_factory=list)
```

**陷阱 2：过度验证**：

```python
# 过度：每个字段都验证
class OverValidated(BaseModel):
    x: int = Field(..., ge=0, le=100)
    y: int = Field(..., ge=0, le=100)
    z: int = Field(..., ge=0, le=100)

# 适度：只验证关键字段
class Balanced(BaseModel):
    x: int
    y: int
    z: int = Field(..., ge=0, le=100)  # 只验证 z
```

---

## 调试技巧

### 1. 检查 model_fields_set

```python
from pydantic import BaseModel

class State(BaseModel):
    x: int = 0
    y: str = "default"

state = State(x=5)
print(f"Fields set: {state.model_fields_set}")
# 输出：Fields set: {'x'}

print(f"All fields: {state.model_fields.keys()}")
# 输出：All fields: dict_keys(['x', 'y'])
```

---

### 2. 追踪状态更新

```python
def debug_update(state: State) -> State:
    new_state = State(x=state.x + 1)
    print(f"Old fields_set: {state.model_fields_set}")
    print(f"New fields_set: {new_state.model_fields_set}")
    return new_state
```

---

### 3. 验证更新逻辑

```python
from langgraph._internal._fields import get_update_as_tuples

state = State(x=5, y="hello")
keys = ["x", "y", "z"]
updates = get_update_as_tuples(state, keys)
print(f"Updates: {updates}")
# 输出：Updates: [('x', 5), ('y', 'hello')]
```

---

## 总结

### 核心要点

1. **model_fields_set**：跟踪显式设置的字段
2. **智能更新**：只更新必要的字段
3. **None 值处理**：区分显式 None 和默认 None
4. **向后兼容**：保持历史行为
5. **性能权衡**：验证 vs 速度

### 选择建议

| 场景 | 推荐方案 |
|------|----------|
| 简单内部状态 | TypedDict |
| API 边界验证 | Pydantic |
| 复杂数据结构 | Pydantic |
| 高性能要求 | TypedDict |
| 需要类型转换 | Pydantic |

---

**文件长度**：约 450 行
**最后更新**：2026-02-26
