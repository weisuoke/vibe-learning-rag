# 核心概念 3：Protocol 协议

## 一句话定义

**Protocol 是 Python 中定义结构类型（structural typing）的机制，通过"鸭子类型"让类型检查器验证对象是否实现了特定方法，而不需要显式继承，是 LangChain Runnable 接口设计的核心基础。**

---

## 为什么需要 Protocol？

### 问题：名义类型 vs 结构类型

Python 传统的类型系统是**名义类型**（nominal typing）：必须显式继承才能满足类型要求。

```python
from abc import ABC, abstractmethod

# 名义类型：必须显式继承
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):  # 必须继承 Animal
    def speak(self) -> str:
        return "Woof!"

class Cat:  # 没有继承 Animal
    def speak(self) -> str:
        return "Meow!"

def make_sound(animal: Animal) -> str:
    return animal.speak()

# ✅ Dog 继承了 Animal
make_sound(Dog())

# ❌ Cat 没有继承 Animal，即使有 speak 方法
make_sound(Cat())  # 类型错误：Cat is not a subclass of Animal
```

**问题**：
- Cat 有 `speak` 方法，但因为没有继承 Animal 而被拒绝
- 无法对第三方库的类进行类型检查（无法修改它们的继承关系）
- 违反了"鸭子类型"哲学："如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子"

**解决方案：Protocol（结构类型）**

```python
from typing import Protocol

# 结构类型：只要有 speak 方法就行
class Speakable(Protocol):
    def speak(self) -> str:
        ...

class Dog:  # 不需要继承
    def speak(self) -> str:
        return "Woof!"

class Cat:  # 不需要继承
    def speak(self) -> str:
        return "Meow!"

def make_sound(animal: Speakable) -> str:
    return animal.speak()

# ✅ Dog 有 speak 方法
make_sound(Dog())

# ✅ Cat 也有 speak 方法
make_sound(Cat())
```

---

## Protocol 基础

### 1. 定义 Protocol

```python
from typing import Protocol

# 基础 Protocol
class Drawable(Protocol):
    def draw(self) -> None:
        ...

# 带属性的 Protocol
class Named(Protocol):
    name: str

# 带多个方法的 Protocol
class Comparable(Protocol):
    def __lt__(self, other: 'Comparable') -> bool:
        ...

    def __eq__(self, other: object) -> bool:
        ...
```

**规则**：
- Protocol 方法体通常是 `...`（Ellipsis）
- 不需要实现方法
- 只定义接口契约

### 2. 使用 Protocol

```python
from typing import Protocol

class Runnable(Protocol):
    def run(self) -> None:
        ...

# 实现类不需要继承 Protocol
class Task:
    def run(self) -> None:
        print("Running task")

class Job:
    def run(self) -> None:
        print("Running job")

# 类型检查器自动验证
def execute(runnable: Runnable) -> None:
    runnable.run()

execute(Task())  # ✅
execute(Job())   # ✅
```

### 3. Protocol 与 ABC 的对比

```python
from abc import ABC, abstractmethod
from typing import Protocol

# ABC：名义类型（必须继承）
class AnimalABC(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class DogABC(AnimalABC):  # 必须继承
    def speak(self) -> str:
        return "Woof!"

# Protocol：结构类型（不需要继承）
class AnimalProtocol(Protocol):
    def speak(self) -> str:
        ...

class DogProtocol:  # 不需要继承
    def speak(self) -> str:
        return "Woof!"
```

**何时使用 ABC vs Protocol**：

| 场景 | 推荐 | 原因 |
|------|------|------|
| 定义库的公共接口 | Protocol | 不强制用户继承 |
| 需要共享实现代码 | ABC | 可以提供默认实现 |
| 第三方类型检查 | Protocol | 无法修改第三方代码 |
| 严格的继承层次 | ABC | 明确的继承关系 |
| 大型代码库 | Protocol | 更灵活，易于重构 |

**2025 最佳实践**：
- ✅ 优先使用 Protocol（更灵活）
- ✅ 只在需要共享实现时使用 ABC
- ✅ 大型代码库推荐 Protocol

---

## 泛型 Protocol

### 1. 单类型参数

```python
from typing import Protocol, TypeVar

T = TypeVar('T')

class Container(Protocol[T]):
    def get(self) -> T:
        ...

    def set(self, value: T) -> None:
        ...

# 实现
class Box:
    def __init__(self, value: int):
        self._value = value

    def get(self) -> int:
        return self._value

    def set(self, value: int) -> None:
        self._value = value

# 类型检查
box: Container[int] = Box(42)  # ✅
value: int = box.get()
```

### 2. 多类型参数

```python
from typing import Protocol, TypeVar

K = TypeVar('K')
V = TypeVar('V')

class Mapping(Protocol[K, V]):
    def get(self, key: K) -> V | None:
        ...

    def set(self, key: K, value: V) -> None:
        ...

# 实现
class SimpleDict:
    def __init__(self):
        self._data: dict[str, int] = {}

    def get(self, key: str) -> int | None:
        return self._data.get(key)

    def set(self, key: str, value: int) -> None:
        self._data[key] = value

# 类型检查
mapping: Mapping[str, int] = SimpleDict()  # ✅
```

### 3. 协变和逆变 Protocol

```python
from typing import Protocol, TypeVar

# 协变：只读
T_co = TypeVar('T_co', covariant=True)

class Producer(Protocol[T_co]):
    def produce(self) -> T_co:
        ...

# 逆变：只写
T_contra = TypeVar('T_contra', contravariant=True)

class Consumer(Protocol[T_contra]):
    def consume(self, item: T_contra) -> None:
        ...

# 使用
class Animal: pass
class Dog(Animal): pass

class DogProducer:
    def produce(self) -> Dog:
        return Dog()

# 协变：Producer[Dog] 可以赋值给 Producer[Animal]
producer: Producer[Animal] = DogProducer()  # ✅
```

---

## Protocol 的高级特性

### 1. 运行时检查（runtime_checkable）

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def paint(self) -> None:  # 方法名不同
        print("Painting square")

# 运行时检查
print(isinstance(Circle(), Drawable))  # True
print(isinstance(Square(), Drawable))  # False
```

**注意**：
- `@runtime_checkable` 只检查方法名，不检查签名
- 主要用于调试和运行时验证
- 类型检查器仍然会检查完整签名

### 2. Protocol 继承

```python
from typing import Protocol

class Readable(Protocol):
    def read(self) -> str:
        ...

class Writable(Protocol):
    def write(self, data: str) -> None:
        ...

# 组合多个 Protocol
class ReadWritable(Readable, Writable, Protocol):
    pass

# 实现
class File:
    def read(self) -> str:
        return "data"

    def write(self, data: str) -> None:
        print(f"Writing: {data}")

# 类型检查
file: ReadWritable = File()  # ✅
```

### 3. Protocol 与泛型方法

```python
from typing import Protocol, TypeVar

T = TypeVar('T')

class Transformer(Protocol):
    def transform(self, value: T) -> T:
        ...

# 实现
class Doubler:
    def transform(self, value: int) -> int:
        return value * 2

class Uppercaser:
    def transform(self, value: str) -> str:
        return value.upper()

# 使用
def apply_transform(transformer: Transformer, value: T) -> T:
    return transformer.transform(value)

# 类型推断
result: int = apply_transform(Doubler(), 21)  # 42
result: str = apply_transform(Uppercaser(), "hello")  # "HELLO"
```

---

## 在 LangChain 中的应用

### 1. Runnable Protocol

LangChain 的 Runnable 实际上是一个 Protocol（简化版）：

```python
from typing import Protocol, TypeVar, Any

Input = TypeVar('Input', contravariant=True)
Output = TypeVar('Output', covariant=True)

class Runnable(Protocol[Input, Output]):
    """LangChain Runnable 协议"""

    def invoke(self, input: Input, config: Any = None) -> Output:
        """同步调用"""
        ...

    async def ainvoke(self, input: Input, config: Any = None) -> Output:
        """异步调用"""
        ...

    def batch(self, inputs: list[Input], config: Any = None) -> list[Output]:
        """批量调用"""
        ...

    def stream(self, input: Input, config: Any = None):
        """流式调用"""
        ...
```

**为什么使用 Protocol**：
- ✅ 不强制用户继承特定基类
- ✅ 可以将任何实现了这些方法的对象当作 Runnable
- ✅ 第三方库可以无缝集成
- ✅ 更灵活的组合和扩展

### 2. 自定义 Runnable（不继承基类）

```python
from typing import Any

# 不继承任何基类，只实现 Protocol 要求的方法
class MyTransformer:
    def invoke(self, input: str, config: Any = None) -> str:
        return input.upper()

    async def ainvoke(self, input: str, config: Any = None) -> str:
        return input.upper()

    def batch(self, inputs: list[str], config: Any = None) -> list[str]:
        return [x.upper() for x in inputs]

    def stream(self, input: str, config: Any = None):
        for char in input.upper():
            yield char

# 类型检查器认为这是 Runnable[str, str]
from langchain_core.runnables import Runnable

transformer: Runnable[str, str] = MyTransformer()  # ✅
result = transformer.invoke("hello")  # "HELLO"
```

### 3. 组合不同来源的 Runnable

```python
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

# 1. LangChain 内置 Runnable
model: Runnable = ChatOpenAI(model="gpt-4o-mini")

# 2. 自定义 Runnable
class CustomProcessor:
    def invoke(self, input: str, config=None) -> str:
        return input.strip().lower()

processor: Runnable[str, str] = CustomProcessor()

# 3. Lambda Runnable
formatter: Runnable[str, str] = RunnableLambda(lambda x: f"Query: {x}")

# 4. 组合（都满足 Runnable Protocol）
chain = formatter | processor | model
```

### 4. Protocol 让 LCEL 更灵活

```python
from typing import Protocol, Any

# 定义自己的 Protocol
class Processor(Protocol):
    def process(self, data: str) -> str:
        ...

# 适配器：将 Processor 转换为 Runnable
class ProcessorAdapter:
    def __init__(self, processor: Processor):
        self.processor = processor

    def invoke(self, input: str, config: Any = None) -> str:
        return self.processor.process(input)

# 使用
class MyProcessor:
    def process(self, data: str) -> str:
        return data.upper()

# 适配到 Runnable
from langchain_core.runnables import Runnable
adapter: Runnable[str, str] = ProcessorAdapter(MyProcessor())
```

---

## 2025-2026 最佳实践

### 1. 优先使用 Protocol

```python
# ❌ 不推荐：强制继承
from abc import ABC, abstractmethod

class Handler(ABC):
    @abstractmethod
    def handle(self, data: str) -> str:
        pass

# ✅ 推荐：使用 Protocol
from typing import Protocol

class Handler(Protocol):
    def handle(self, data: str) -> str:
        ...
```

### 2. Protocol 命名约定

```python
# 2025-2026 命名规范

# ✅ 形容词形式（推荐）
class Readable(Protocol): ...
class Writable(Protocol): ...
class Comparable(Protocol): ...

# ✅ 名词形式（也可以）
class Reader(Protocol): ...
class Writer(Protocol): ...
class Comparator(Protocol): ...

# ❌ 避免 I 前缀（C# 风格）
class IReadable(Protocol): ...  # 不推荐
```

### 3. 最小化 Protocol

```python
# ❌ Protocol 太大
class DataProcessor(Protocol):
    def load(self) -> None: ...
    def validate(self) -> bool: ...
    def transform(self) -> None: ...
    def save(self) -> None: ...
    def cleanup(self) -> None: ...

# ✅ 拆分成小 Protocol
class Loadable(Protocol):
    def load(self) -> None: ...

class Validatable(Protocol):
    def validate(self) -> bool: ...

class Transformable(Protocol):
    def transform(self) -> None: ...

# 组合使用
class DataProcessor(Loadable, Validatable, Transformable, Protocol):
    pass
```

### 4. 使用 Python 3.12+ 新语法

```python
# Python 3.12+ 泛型 Protocol 新语法
from typing import Protocol

# ❌ 旧语法
T = TypeVar('T')
class Container(Protocol[T]):
    def get(self) -> T: ...

# ✅ 新语法（Python 3.12+）
class Container[T](Protocol):
    def get(self) -> T: ...
```

---

## Protocol vs ABC vs TypedDict

### 对比表

| 特性 | Protocol | ABC | TypedDict |
|------|----------|-----|-----------|
| 类型系统 | 结构类型 | 名义类型 | 结构类型 |
| 需要继承 | ❌ | ✅ | ❌ |
| 运行时检查 | 可选 | ✅ | ❌ |
| 共享实现 | ❌ | ✅ | ❌ |
| 适用场景 | 接口定义 | 抽象基类 | 字典结构 |
| 灵活性 | 高 | 中 | 高 |
| 2025 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 使用场景

```python
from typing import Protocol, TypedDict
from abc import ABC, abstractmethod

# 1. Protocol：定义接口（推荐）
class Runnable(Protocol):
    def run(self) -> None: ...

# 2. ABC：需要共享实现
class BaseHandler(ABC):
    def handle(self, data: str) -> str:
        # 共享的预处理逻辑
        data = self._preprocess(data)
        return self._process(data)

    def _preprocess(self, data: str) -> str:
        return data.strip()

    @abstractmethod
    def _process(self, data: str) -> str:
        pass

# 3. TypedDict：定义字典结构
class UserData(TypedDict):
    name: str
    age: int
    email: str
```

---

## 实战示例：类型安全的插件系统

```python
"""
类型安全的插件系统
演示：Protocol 在实际项目中的应用
"""

from typing import Protocol, TypeVar, Any
from langchain_core.runnables import Runnable, RunnableLambda

# ===== 1. 定义插件 Protocol =====
Input = TypeVar('Input', contravariant=True)
Output = TypeVar('Output', covariant=True)

class Plugin(Protocol[Input, Output]):
    """插件协议"""

    @property
    def name(self) -> str:
        """插件名称"""
        ...

    def process(self, input: Input) -> Output:
        """处理数据"""
        ...

# ===== 2. 实现具体插件（不继承任何基类）=====

class UppercasePlugin:
    """大写转换插件"""

    @property
    def name(self) -> str:
        return "uppercase"

    def process(self, input: str) -> str:
        return input.upper()

class LengthPlugin:
    """长度计算插件"""

    @property
    def name(self) -> str:
        return "length"

    def process(self, input: str) -> int:
        return len(input)

class ReversePlugin:
    """反转插件"""

    @property
    def name(self) -> str:
        return "reverse"

    def process(self, input: str) -> str:
        return input[::-1]

# ===== 3. 插件管理器 =====

class PluginManager:
    """插件管理器"""

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        """注册插件（接受任何满足 Protocol 的对象）"""
        self._plugins[plugin.name] = plugin
        print(f"✅ 注册插件: {plugin.name}")

    def get(self, name: str) -> Plugin | None:
        """获取插件"""
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """列出所有插件"""
        return list(self._plugins.keys())

# ===== 4. 适配器：Plugin -> Runnable =====

class PluginAdapter(Runnable[Input, Output]):
    """将 Plugin 适配为 Runnable"""

    def __init__(self, plugin: Plugin[Input, Output]):
        self.plugin = plugin

    def invoke(self, input: Input, config: Any = None) -> Output:
        return self.plugin.process(input)

# ===== 5. 使用示例 =====

print("=== 类型安全的插件系统 ===\n")

# 创建插件管理器
manager = PluginManager()

# 注册插件（不需要继承任何基类）
manager.register(UppercasePlugin())
manager.register(LengthPlugin())
manager.register(ReversePlugin())

print(f"\n已注册插件: {manager.list_plugins()}\n")

# 使用插件
uppercase_plugin = manager.get("uppercase")
if uppercase_plugin:
    result: str = uppercase_plugin.process("hello")
    print(f"Uppercase: {result}")

length_plugin = manager.get("length")
if length_plugin:
    result: int = length_plugin.process("hello")
    print(f"Length: {result}")

# 适配为 Runnable
print("\n=== 适配为 Runnable ===\n")
reverse_plugin = manager.get("reverse")
if reverse_plugin:
    runnable: Runnable[str, str] = PluginAdapter(reverse_plugin)
    result = runnable.invoke("hello")
    print(f"Reverse (via Runnable): {result}")

# 组合到 LCEL 链
print("\n=== 组合到 LCEL 链 ===\n")
uppercase_runnable = PluginAdapter(UppercasePlugin())
reverse_runnable = PluginAdapter(ReversePlugin())

chain = uppercase_runnable | reverse_runnable
result = chain.invoke("hello")
print(f"Chain result: {result}")  # "OLLEH"

# ===== 6. 第三方插件（无需修改代码）=====

class ThirdPartyProcessor:
    """第三方库的处理器（我们无法修改它的代码）"""

    @property
    def name(self) -> str:
        return "third_party"

    def process(self, input: str) -> str:
        return f"[{input}]"

# 直接注册（满足 Protocol）
manager.register(ThirdPartyProcessor())
print(f"\n第三方插件已注册: {manager.list_plugins()}")
```

**运行输出**：
```
=== 类型安全的插件系统 ===

✅ 注册插件: uppercase
✅ 注册插件: length
✅ 注册插件: reverse

已注册插件: ['uppercase', 'length', 'reverse']

Uppercase: HELLO
Length: 5

=== 适配为 Runnable ===

Reverse (via Runnable): olleh

=== 组合到 LCEL 链 ===

Chain result: OLLEH

✅ 注册插件: third_party

第三方插件已注册: ['uppercase', 'length', 'reverse', 'third_party']
```

---

## 常见陷阱

### 1. Protocol 方法签名必须完全匹配

```python
from typing import Protocol

class Processor(Protocol):
    def process(self, data: str) -> str:
        ...

# ❌ 签名不匹配（参数名不同）
class BadProcessor:
    def process(self, text: str) -> str:  # 参数名是 text，不是 data
        return text

# ⚠️ 类型检查器可能报错（取决于配置）
processor: Processor = BadProcessor()

# ✅ 签名完全匹配
class GoodProcessor:
    def process(self, data: str) -> str:
        return data
```

### 2. runtime_checkable 的局限性

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Adder(Protocol):
    def add(self, a: int, b: int) -> int:
        ...

class BadAdder:
    def add(self, a: str, b: str) -> str:  # 签名错误
        return a + b

# ⚠️ 运行时检查只看方法名，不看签名
print(isinstance(BadAdder(), Adder))  # True（错误！）

# 但类型检查器会报错
adder: Adder = BadAdder()  # 类型错误
```

### 3. Protocol 不能有实现

```python
from typing import Protocol

# ❌ Protocol 不能有实现
class Bad(Protocol):
    def process(self, data: str) -> str:
        return data.upper()  # 错误：Protocol 不能有实现

# ✅ 使用 ABC 如果需要共享实现
from abc import ABC, abstractmethod

class Good(ABC):
    def process(self, data: str) -> str:
        return self._do_process(data.strip())

    @abstractmethod
    def _do_process(self, data: str) -> str:
        pass
```

---

## 学习检查清单

- [ ] 理解 Protocol 的作用（结构类型 vs 名义类型）
- [ ] 掌握 Protocol 的定义和使用
- [ ] 了解泛型 Protocol
- [ ] 理解协变和逆变在 Protocol 中的应用
- [ ] 知道何时使用 Protocol vs ABC
- [ ] 了解 LangChain Runnable Protocol 的设计
- [ ] 能够创建类型安全的插件系统
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 4**：Runnable 泛型设计 - 深入 LangChain 的 Runnable[Input, Output]
- **核心概念 2**：TypeVar 与泛型 - 复习泛型知识
- **核心概念 8**：类型守卫 - 学习运行时类型检查

---

## 参考资源

1. [Interfaces in Python for 2025: Best Practices](https://python.plainenglish.io/interfaces-in-python-for-2025-best-practices-and-modern-approaches-2c9c1272f37f) - 2025 最佳实践
2. [Advanced Python Type Hinting: Generics, Protocols](https://dhirendrabiswal.com/advanced-python-type-hinting-generics-protocols-and-structural-subtyping-explained) - 2025
3. [Real Python - Python Protocol](https://realpython.com/python-protocol/) - 详细教程
4. [PEP 544 – Protocols: Structural subtyping](https://peps.python.org/pep-0544/) - Protocol 规范
5. [Python typing 官方文档 - Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
6. [LangChain Runnable 源码](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/base.py) - Runnable Protocol 实现
