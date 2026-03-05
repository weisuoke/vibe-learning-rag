# 核心概念 08：StateLike 协议

## 概述

LangGraph 使用 Python 的 Protocol（协议）机制实现类型系统的灵活性，通过 `StateLike` 协议定义了状态对象必须满足的接口规范。这种设计允许框架支持多种类型定义方式（TypedDict、dataclass、Pydantic），而无需强制继承特定的基类。

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py]

---

## 核心协议定义

### 1. TypedDictLikeV1 协议

**定义**：

```python
from typing import Protocol, ClassVar

class TypedDictLikeV1(Protocol):
    """Protocol to represent types that behave like TypedDicts
    Version 1: using `ClassVar` for keys."""
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py:12-16]

**关键特性**：
- 使用 `ClassVar` 标注类变量
- `__required_keys__`：必需字段的集合
- `__optional_keys__`：可选字段的集合
- 适用于早期 Python 版本的 TypedDict

**兼容类型**：
- Python 3.8-3.10 的 TypedDict
- 使用 `typing_extensions.TypedDict`

---

### 2. TypedDictLikeV2 协议

**定义**：

```python
class TypedDictLikeV2(Protocol):
    """Protocol to represent types that behave like TypedDicts
    Version 2: not using `ClassVar` for keys."""
    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py:18-22]

**关键特性**：
- 不使用 `ClassVar` 标注
- 与 V1 功能相同，但类型注解不同
- 适用于较新 Python 版本的 TypedDict

**兼容类型**：
- Python 3.11+ 的 TypedDict
- 标准库 `typing.TypedDict`

---

### 3. DataclassLike 协议

**定义**：

```python
from dataclasses import Field
from typing import Any

class DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses."""
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py:24-26]

**关键特性**：
- `__dataclass_fields__`：字段元数据字典
- 包含字段名称、类型、默认值等信息
- 由 `@dataclass` 装饰器自动生成

**兼容类型**：
- 标准库 `dataclasses.dataclass`
- 第三方库的 dataclass 实现（如 attrs）

---

### 4. StateLike 类型别名

**定义**：

```python
from pydantic import BaseModel
from typing import TypeAlias

StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py:28]

**关键特性**：
- 联合类型，支持 4 种状态定义方式
- 通过协议实现鸭子类型
- 无需继承特定基类

---

## 协议与鸭子类型

### 什么是协议（Protocol）

协议是 Python 3.8+ 引入的结构化子类型（Structural Subtyping）机制，也称为"鸭子类型"的静态类型版本。

**核心理念**：
> "如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子"

**与继承的区别**：

```python
# 传统继承（Nominal Typing）
class Animal:
    def speak(self) -> str:
        pass

class Dog(Animal):  # 必须显式继承
    def speak(self) -> str:
        return "Woof"

# 协议（Structural Typing）
from typing import Protocol

class Speakable(Protocol):
    def speak(self) -> str:
        ...

class Cat:  # 无需继承
    def speak(self) -> str:
        return "Meow"

def make_sound(animal: Speakable) -> str:
    return animal.speak()

# Cat 自动满足 Speakable 协议
make_sound(Cat())  # 类型检查通过
```

[来源: reference/context7_typing_extensions_01.md | typing_extensions 官方文档]

---

### LangGraph 中的协议应用

**设计优势**：

1. **灵活性**：支持多种类型定义方式
2. **兼容性**：无需修改现有代码
3. **扩展性**：用户可以自定义状态类型
4. **类型安全**：静态类型检查支持

**示例**：

```python
from typing_extensions import TypedDict
from dataclasses import dataclass
from pydantic import BaseModel
from langgraph.graph import StateGraph

# 方式 1：TypedDict
class State1(TypedDict):
    x: int

# 方式 2：dataclass
@dataclass
class State2:
    x: int

# 方式 3：Pydantic
class State3(BaseModel):
    x: int

# 所有方式都兼容
graph1 = StateGraph(State1)
graph2 = StateGraph(State2)
graph3 = StateGraph(State3)
```

---

## 类型兼容性检查

### 运行时检查机制

LangGraph 使用 `isinstance()` 和 `hasattr()` 检查类型兼容性。

**检查 TypedDict**：

```python
def is_typed_dict_like(obj: Any) -> bool:
    """Check if an object behaves like a TypedDict."""
    return (
        hasattr(obj, "__required_keys__") and
        hasattr(obj, "__optional_keys__")
    )

# 示例
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

print(is_typed_dict_like(State))  # True
print(hasattr(State, "__required_keys__"))  # True
print(State.__required_keys__)  # frozenset({'x'})
```

---

**检查 dataclass**：

```python
import dataclasses

def is_dataclass_like(obj: Any) -> bool:
    """Check if an object behaves like a dataclass."""
    return hasattr(obj, "__dataclass_fields__")

# 示例
from dataclasses import dataclass

@dataclass
class State:
    x: int

print(is_dataclass_like(State))  # True
print(dataclasses.is_dataclass(State))  # True
print(State.__dataclass_fields__)  # {'x': Field(...)}
```

---

**检查 Pydantic**：

```python
from pydantic import BaseModel

def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)

# 示例
class State(BaseModel):
    x: int

print(is_pydantic_model(State))  # True
print(isinstance(State(), BaseModel))  # True
```

---

### 静态类型检查

**mypy 支持**：

```python
from typing import Protocol
from langgraph.graph import StateGraph

class State(Protocol):
    x: int

def process(state: State) -> State:
    return state

# mypy 会检查类型兼容性
graph = StateGraph(State)  # 类型检查通过
```

---

## 实战示例

### 示例 1：自定义协议状态

```python
from typing import Protocol, ClassVar
from langgraph.graph import StateGraph

class CustomState(Protocol):
    """自定义状态协议"""
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]

    x: int
    y: str

# 实现协议
class MyState:
    __required_keys__ = frozenset({"x", "y"})
    __optional_keys__ = frozenset()

    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y

# 使用自定义状态
graph = StateGraph(MyState)
```

---

### 示例 2：混合类型系统

```python
from typing_extensions import TypedDict
from dataclasses import dataclass
from pydantic import BaseModel

# 内部状态：TypedDict（快）
class InternalState(TypedDict):
    messages: list[str]
    context: dict

# 节点输入：dataclass（简单）
@dataclass
class NodeInput:
    query: str
    user_id: str

# API 输入：Pydantic（验证）
class APIInput(BaseModel):
    query: str
    user_id: str

def node_func(state: InternalState) -> InternalState:
    # 处理内部状态
    return {"messages": state["messages"] + ["new"], "context": state["context"]}

graph = StateGraph(InternalState)
graph.add_node("process", node_func)
```

---

### 示例 3：协议继承

```python
from typing import Protocol

class BaseState(Protocol):
    """基础状态协议"""
    id: str

class ExtendedState(BaseState, Protocol):
    """扩展状态协议"""
    name: str
    age: int

# 实现扩展协议
class UserState:
    def __init__(self, id: str, name: str, age: int):
        self.id = id
        self.name = name
        self.age = age

# 类型检查通过
def process(state: ExtendedState) -> None:
    print(f"{state.id}: {state.name}, {state.age}")

process(UserState("123", "Alice", 25))
```

---

### 示例 4：运行时类型检查

```python
from typing import Any, get_type_hints
from typing_extensions import TypedDict
from dataclasses import dataclass, is_dataclass
from pydantic import BaseModel

def get_state_type(state_schema: type) -> str:
    """识别状态类型"""
    if hasattr(state_schema, "__required_keys__"):
        return "TypedDict"
    elif is_dataclass(state_schema):
        return "dataclass"
    elif isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
        return "Pydantic"
    else:
        return "Unknown"

# 测试
class State1(TypedDict):
    x: int

@dataclass
class State2:
    x: int

class State3(BaseModel):
    x: int

print(get_state_type(State1))  # TypedDict
print(get_state_type(State2))  # dataclass
print(get_state_type(State3))  # Pydantic
```

---

## 协议版本差异

### TypedDictLikeV1 vs V2

**V1（使用 ClassVar）**：

```python
from typing import ClassVar

class State:
    __required_keys__: ClassVar[frozenset[str]] = frozenset({"x"})
    __optional_keys__: ClassVar[frozenset[str]] = frozenset()

    x: int
```

**V2（不使用 ClassVar）**：

```python
class State:
    __required_keys__: frozenset[str] = frozenset({"x"})
    __optional_keys__: frozenset[str] = frozenset()

    x: int
```

**差异原因**：
- Python 3.8-3.10：TypedDict 使用 `ClassVar`
- Python 3.11+：TypedDict 不使用 `ClassVar`
- LangGraph 同时支持两种版本

[来源: sourcecode/langgraph/libs/langgraph/langgraph/_internal/_typing.py]

---

### 兼容性表格

| Python 版本 | TypedDict 实现 | 协议版本 |
|-------------|----------------|----------|
| 3.8-3.10 | typing_extensions | V1 (ClassVar) |
| 3.11+ | typing | V2 (无 ClassVar) |
| 3.12+ | typing | V2 (无 ClassVar) |

---

## 最佳实践

### 1. 选择合适的类型定义

**推荐决策树**：

```
需要运行时验证？
├─ 是 → Pydantic BaseModel
└─ 否
   ├─ 需要默认值？
   │  ├─ 是 → dataclass
   │  └─ 否 → TypedDict
   └─ 追求性能？
      └─ 是 → TypedDict
```

---

### 2. 协议设计原则

**好的协议设计**：

```python
from typing import Protocol

class GoodProtocol(Protocol):
    """清晰的协议定义"""
    # 必需属性
    id: str
    name: str

    # 必需方法
    def process(self) -> dict:
        ...

# 实现简单
class Implementation:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    def process(self) -> dict:
        return {"id": self.id, "name": self.name}
```

**避免的设计**：

```python
class BadProtocol(Protocol):
    """过于复杂的协议"""
    # 太多属性
    attr1: str
    attr2: int
    attr3: list
    # ... 20 个属性

    # 太多方法
    def method1(self) -> None: ...
    def method2(self) -> None: ...
    # ... 10 个方法
```

---

### 3. 类型注解最佳实践

```python
from typing import Protocol, TypeVar

# 定义泛型协议
T = TypeVar("T")

class StateProtocol(Protocol[T]):
    """泛型状态协议"""
    data: T

    def get_data(self) -> T:
        ...

# 使用泛型协议
class IntState:
    def __init__(self, data: int):
        self.data = data

    def get_data(self) -> int:
        return self.data

class StrState:
    def __init__(self, data: str):
        self.data = data

    def get_data(self) -> str:
        return self.data

# 类型检查通过
int_state: StateProtocol[int] = IntState(42)
str_state: StateProtocol[str] = StrState("hello")
```

---

### 4. 调试协议兼容性

```python
from typing import get_type_hints, Protocol
import inspect

def debug_protocol_compatibility(obj: Any, protocol: type[Protocol]) -> None:
    """调试协议兼容性"""
    print(f"Checking {obj.__name__} against {protocol.__name__}")

    # 检查属性
    protocol_attrs = get_type_hints(protocol)
    obj_attrs = get_type_hints(obj) if hasattr(obj, "__annotations__") else {}

    print("\nRequired attributes:")
    for attr, typ in protocol_attrs.items():
        has_attr = attr in obj_attrs
        print(f"  {attr}: {typ} - {'✓' if has_attr else '✗'}")

    # 检查方法
    protocol_methods = [m for m in dir(protocol) if not m.startswith("_")]
    obj_methods = [m for m in dir(obj) if not m.startswith("_")]

    print("\nRequired methods:")
    for method in protocol_methods:
        has_method = method in obj_methods
        print(f"  {method} - {'✓' if has_method else '✗'}")

# 使用
class MyProtocol(Protocol):
    x: int
    def process(self) -> None: ...

class MyClass:
    x: int = 0
    def process(self) -> None:
        pass

debug_protocol_compatibility(MyClass, MyProtocol)
```

---

## 常见问题

### Q1: 为什么需要两个 TypedDict 协议版本？

**A**: Python 3.11 改变了 TypedDict 的内部实现，不再使用 `ClassVar` 标注 `__required_keys__` 和 `__optional_keys__`。为了向后兼容，LangGraph 同时支持两种版本。

---

### Q2: 协议和抽象基类（ABC）有什么区别？

**A**:
- **协议**：结构化子类型，无需继承
- **ABC**：名义化子类型，必须继承

```python
# ABC 方式
from abc import ABC, abstractmethod

class StateABC(ABC):
    @abstractmethod
    def process(self) -> None:
        pass

class MyState(StateABC):  # 必须继承
    def process(self) -> None:
        pass

# Protocol 方式
from typing import Protocol

class StateProtocol(Protocol):
    def process(self) -> None:
        ...

class MyState:  # 无需继承
    def process(self) -> None:
        pass
```

---

### Q3: 如何扩展 StateLike 协议？

**A**: 创建自定义协议并确保满足 LangGraph 的要求：

```python
from typing import Protocol, ClassVar

class CustomStateLike(Protocol):
    """自定义状态协议"""
    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]

    # 自定义属性
    custom_attr: str
```

---

### Q4: 协议的性能开销如何？

**A**: 协议是编译时特性，运行时没有额外开销。类型检查由 mypy 等工具在开发阶段完成。

---

## 总结

### 核心要点

1. **协议机制**：结构化子类型，鸭子类型的静态版本
2. **三种协议**：TypedDictLike（V1/V2）、DataclassLike、BaseModel
3. **灵活兼容**：支持多种类型定义方式
4. **类型安全**：静态类型检查 + 运行时验证
5. **版本兼容**：同时支持旧版和新版 Python

### 设计优势

| 特性 | 传统继承 | 协议 |
|------|----------|------|
| 灵活性 | 低 | 高 |
| 兼容性 | 差 | 好 |
| 类型安全 | 好 | 好 |
| 运行时开销 | 有 | 无 |

---

**文件长度**：约 480 行
**最后更新**：2026-02-26
