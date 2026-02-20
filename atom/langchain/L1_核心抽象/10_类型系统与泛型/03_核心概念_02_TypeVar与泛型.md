# 核心概念 2：TypeVar 与泛型

## 一句话定义

**TypeVar 是 Python 中定义类型变量的工具，配合 Generic 可以创建泛型函数和类，让代码在保持类型安全的同时支持多种类型，是 LangChain Runnable[Input, Output] 的核心实现机制。**

---

## 为什么需要泛型？

### 问题：类型安全 vs 代码复用

假设我们要实现一个"盒子"类，可以存储任意类型的值：

```python
# 方案1：使用 Any（失去类型安全）
from typing import Any

class Box:
    def __init__(self, value: Any):
        self.value = value

    def get(self) -> Any:
        return self.value

# 问题：类型检查器无法推断类型
int_box = Box(42)
result: int = int_box.get()  # ⚠️ 类型检查器认为 result 是 Any，不是 int
result.upper()  # ❌ 运行时才报错：AttributeError
```

```python
# 方案2：为每种类型创建类（代码重复）
class IntBox:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value

class StrBox:
    def __init__(self, value: str):
        self.value = value

    def get(self) -> str:
        return self.value

# 问题：代码重复，维护困难
```

**解决方案：泛型（Generic）**

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

# ✅ 类型安全 + 代码复用
int_box: Box[int] = Box(42)
result: int = int_box.get()  # 类型检查器知道 result 是 int
result.upper()  # ❌ 类型检查器立即报错：int has no attribute 'upper'

str_box: Box[str] = Box("hello")
text: str = str_box.get()  # 类型检查器知道 text 是 str
text.upper()  # ✅ 类型正确
```

---

## TypeVar 基础

### 1. 定义 TypeVar

```python
from typing import TypeVar

# 基础定义
T = TypeVar('T')  # 可以是任意类型

# 命名约定（2025-2026 最佳实践）
T = TypeVar('T')           # 通用类型变量
Input = TypeVar('Input')   # 输入类型
Output = TypeVar('Output') # 输出类型
K = TypeVar('K')           # 键类型
V = TypeVar('V')           # 值类型
```

**命名规则**：
- 单字母：`T`, `U`, `V`（通用场景）
- 描述性名称：`Input`, `Output`, `Key`, `Value`（特定场景）
- 协变后缀：`T_co`（covariant）
- 逆变后缀：`T_contra`（contravariant）

### 2. 泛型函数

```python
from typing import TypeVar

T = TypeVar('T')

# 泛型函数：保留类型信息
def first(items: list[T]) -> T:
    return items[0]

# 类型推断
numbers = [1, 2, 3]
result: int = first(numbers)  # ✅ 推断为 int

words = ["hello", "world"]
result: str = first(words)  # ✅ 推断为 str

# 对比：使用 Any 会丢失类型信息
from typing import Any

def first_any(items: list[Any]) -> Any:
    return items[0]

result = first_any(numbers)  # ⚠️ 类型检查器认为是 Any
```

### 3. 多个 TypeVar

```python
from typing import TypeVar

K = TypeVar('K')
V = TypeVar('V')

def get_first_item(d: dict[K, V]) -> tuple[K, V]:
    key = next(iter(d))
    return key, d[key]

# 类型推断
data: dict[str, int] = {"age": 30}
key, value = get_first_item(data)  # key: str, value: int
```

---

## TypeVar 约束

### 1. bound 参数（上界约束）

```python
from typing import TypeVar

# 约束 T 必须是 str 或其子类
T = TypeVar('T', bound=str)

def repeat(value: T, times: int) -> list[T]:
    return [value] * times

# ✅ 可以使用 str
result: list[str] = repeat("hello", 3)

# ❌ 不能使用 int（不是 str 的子类）
result = repeat(42, 3)  # 类型错误

# 实际应用：约束为 BaseMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

MessageT = TypeVar('MessageT', bound=BaseMessage)

def get_content(message: MessageT) -> str:
    return message.content

# ✅ 可以使用任何 BaseMessage 子类
human_msg = HumanMessage(content="Hello")
ai_msg = AIMessage(content="Hi")
get_content(human_msg)  # ✅
get_content(ai_msg)     # ✅
```

**2025-2026 最佳实践**：
- ✅ 优先使用 `bound` 而非 `constraints`
- ✅ `bound` 更灵活，支持子类
- ✅ `bound` 可以访问基类的方法和属性

### 2. constraints 参数（多选一约束）

```python
from typing import TypeVar

# 约束 T 只能是 int 或 str（不能是子类）
T = TypeVar('T', int, str)

def double(value: T) -> T:
    if isinstance(value, int):
        return value * 2  # type: ignore
    return value * 2  # type: ignore

# ✅ 可以使用 int 或 str
double(21)      # 42
double("hi")    # "hihi"

# ❌ 不能使用其他类型
double(3.14)    # 类型错误
```

**何时使用 constraints**：
- 需要精确限制为几种类型
- 不需要支持子类
- 通常 `bound` 更好

---

## Generic 类

### 1. 单类型参数

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def is_empty(self) -> bool:
        return len(self._items) == 0

# 使用
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)
value: int = int_stack.pop()  # 类型安全

str_stack: Stack[str] = Stack()
str_stack.push("hello")
text: str = str_stack.pop()
```

### 2. 多类型参数

```python
from typing import TypeVar, Generic

K = TypeVar('K')
V = TypeVar('V')

class Cache(Generic[K, V]):
    def __init__(self):
        self._data: dict[K, V] = {}

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get(self, key: K) -> V | None:
        return self._data.get(key)

# 使用
cache: Cache[str, int] = Cache()
cache.set("age", 30)
age: int | None = cache.get("age")
```

### 3. 继承泛型类

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T):
        self.value = value

# 方式1：保持泛型
class Box(Container[T]):
    def get(self) -> T:
        return self.value

# 方式2：具体化类型
class IntBox(Container[int]):
    def double(self) -> int:
        return self.value * 2

# 使用
box: Box[str] = Box("hello")
int_box: IntBox = IntBox(42)
```

---

## 协变（Covariant）与逆变（Contravariant）

### 1. 不变（Invariant）- 默认行为

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value

# 不变：Box[子类] 不能赋值给 Box[父类]
class Animal: pass
class Dog(Animal): pass

dog_box: Box[Dog] = Box(Dog())
animal_box: Box[Animal] = dog_box  # ❌ 类型错误（不变）
```

**为什么不变**？因为 `set` 方法可能破坏类型安全：

```python
# 如果允许 Box[Dog] -> Box[Animal]
animal_box: Box[Animal] = dog_box  # 假设允许
animal_box.set(Animal())  # 现在 dog_box 里有 Animal！
dog: Dog = dog_box.get()  # ❌ 运行时错误
```

### 2. 协变（Covariant）- 只读

```python
from typing import TypeVar, Generic

T_co = TypeVar('T_co', covariant=True)

class ReadOnlyBox(Generic[T_co]):
    def __init__(self, value: T_co):
        self._value = value

    def get(self) -> T_co:
        return self._value

    # ❌ 不能有 set 方法（协变类型只能作为返回值）

# 协变：ReadOnlyBox[子类] 可以赋值给 ReadOnlyBox[父类]
class Animal: pass
class Dog(Animal): pass

dog_box: ReadOnlyBox[Dog] = ReadOnlyBox(Dog())
animal_box: ReadOnlyBox[Animal] = dog_box  # ✅ 类型正确（协变）

# 安全：只能读取，不能写入
animal: Animal = animal_box.get()  # ✅ Dog 是 Animal
```

**协变规则**：
- `T_co` 只能出现在返回值位置
- 不能出现在参数位置
- 适用于只读容器（如 `tuple`, `Sequence`）

### 3. 逆变（Contravariant）- 只写

```python
from typing import TypeVar, Generic

T_contra = TypeVar('T_contra', contravariant=True)

class Sink(Generic[T_contra]):
    def put(self, item: T_contra) -> None:
        print(f"Received: {item}")

    # ❌ 不能有返回 T_contra 的方法（逆变类型只能作为参数）

# 逆变：Sink[父类] 可以赋值给 Sink[子类]
class Animal: pass
class Dog(Animal): pass

animal_sink: Sink[Animal] = Sink()
dog_sink: Sink[Dog] = animal_sink  # ✅ 类型正确（逆变）

# 安全：可以接受 Dog（因为 animal_sink 可以接受任何 Animal）
dog_sink.put(Dog())  # ✅
```

**逆变规则**：
- `T_contra` 只能出现在参数位置
- 不能出现在返回值位置
- 适用于只写容器（如回调函数）

### 4. 实际应用：Callable

```python
from typing import Callable

# Callable 的参数是逆变的，返回值是协变的
# Callable[[T_contra], T_co]

class Animal:
    def speak(self) -> str:
        return "..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# 函数类型
def process_animal(animal: Animal) -> Animal:
    return animal

def process_dog(dog: Dog) -> Dog:
    return dog

# 逆变：参数可以更宽泛
handler: Callable[[Dog], Animal] = process_animal  # ✅
# process_animal 接受 Animal，所以也能接受 Dog

# 协变：返回值可以更具体
handler: Callable[[Animal], Dog] = process_dog  # ❌ 类型错误
# process_dog 只接受 Dog，不能接受所有 Animal
```

---

## 在 LangChain 中的应用

### 1. Runnable[Input, Output]

```python
from typing import TypeVar, Generic
from langchain_core.runnables import Runnable

# LangChain 的 Runnable 定义（简化版）
Input = TypeVar('Input')
Output = TypeVar('Output')

class Runnable(Generic[Input, Output]):
    def invoke(self, input: Input) -> Output:
        ...

    def batch(self, inputs: list[Input]) -> list[Output]:
        ...

    def stream(self, input: Input):
        ...

# 使用示例
from langchain_core.runnables import RunnableLambda

def to_upper(text: str) -> str:
    return text.upper()

# RunnableLambda 自动推断为 Runnable[str, str]
runnable: Runnable[str, str] = RunnableLambda(to_upper)

# 类型安全
result: str = runnable.invoke("hello")  # ✅
result: int = runnable.invoke("hello")  # ❌ 类型错误
```

### 2. 自定义泛型 Runnable

```python
from typing import TypeVar, Generic
from langchain_core.runnables import Runnable

Input = TypeVar('Input')
Output = TypeVar('Output')

class TransformRunnable(Runnable[Input, Output], Generic[Input, Output]):
    def __init__(self, transform_fn: Callable[[Input], Output]):
        self.transform_fn = transform_fn

    def invoke(self, input: Input) -> Output:
        return self.transform_fn(input)

# 使用
def length(text: str) -> int:
    return len(text)

# 类型推断
length_runnable: TransformRunnable[str, int] = TransformRunnable(length)
result: int = length_runnable.invoke("hello")  # 5
```

### 3. 链式组合的类型推断

```python
from langchain_core.runnables import RunnableLambda

# 定义类型明确的函数
def format_prompt(topic: str) -> str:
    return f"Tell me about {topic}"

def count_words(text: str) -> int:
    return len(text.split())

# 构建链
formatter: Runnable[str, str] = RunnableLambda(format_prompt)
counter: Runnable[str, int] = RunnableLambda(count_words)

# 组合：Runnable[str, str] | Runnable[str, int] = Runnable[str, int]
chain: Runnable[str, int] = formatter | counter

# 类型检查器验证整个链
result: int = chain.invoke("Python")  # ✅
result: str = chain.invoke("Python")  # ❌ 类型错误
```

### 4. with_types 覆盖类型

```python
from langchain_core.runnables import RunnableLambda

# 动态函数（类型不明确）
def process(x):
    return x.upper()

# 创建 Runnable（类型推断为 Any）
runnable = RunnableLambda(process)

# 使用 with_types 显式指定类型
typed_runnable: Runnable[str, str] = runnable.with_types(
    input_type=str,
    output_type=str
)

# 现在有类型检查
result: str = typed_runnable.invoke("hello")  # ✅
```

---

## 2025-2026 最佳实践

### 1. 优先使用 bound 而非 constraints

```python
from typing import TypeVar

# ❌ 不推荐：constraints 限制太死
T = TypeVar('T', int, str, float)

# ✅ 推荐：bound 更灵活
from numbers import Number
T = TypeVar('T', bound=Number)
```

### 2. 清晰的命名

```python
# ❌ 不清晰
T = TypeVar('T')
U = TypeVar('U')

def transform(x: T, y: U) -> T:
    ...

# ✅ 清晰
Input = TypeVar('Input')
Output = TypeVar('Output')

def transform(input: Input, config: Output) -> Input:
    ...
```

### 3. 协变逆变命名约定

```python
from typing import TypeVar

# 2025-2026 标准命名
T_co = TypeVar('T_co', covariant=True)      # 协变
T_contra = TypeVar('T_contra', contravariant=True)  # 逆变
T = TypeVar('T')  # 不变（默认）
```

### 4. 从简单开始

```python
# ❌ 过度设计
from typing import TypeVar, Generic, Protocol

T = TypeVar('T', bound='Comparable')

class Comparable(Protocol):
    def __lt__(self, other: 'Comparable') -> bool: ...

class Container(Generic[T]):
    ...

# ✅ 从简单开始
T = TypeVar('T')

class Container(Generic[T]):
    ...

# 需要时再添加约束
```

---

## 实战示例：类型安全的数据处理管道

```python
"""
类型安全的数据处理管道
演示：TypeVar + Generic 在实际项目中的应用
"""

from typing import TypeVar, Generic, Callable
from langchain_core.runnables import Runnable, RunnableLambda

# ===== 1. 定义泛型处理器 =====
Input = TypeVar('Input')
Output = TypeVar('Output')

class Processor(Generic[Input, Output]):
    """泛型数据处理器"""

    def __init__(self, process_fn: Callable[[Input], Output]):
        self.process_fn = process_fn

    def process(self, data: Input) -> Output:
        return self.process_fn(data)

    def to_runnable(self) -> Runnable[Input, Output]:
        """转换为 Runnable"""
        return RunnableLambda(self.process_fn)

# ===== 2. 创建具体处理器 =====

# 文本处理器：str -> str
text_processor: Processor[str, str] = Processor(
    lambda text: text.strip().lower()
)

# 长度计算器：str -> int
length_calculator: Processor[str, int] = Processor(
    lambda text: len(text)
)

# 格式化器：int -> str
formatter: Processor[int, str] = Processor(
    lambda num: f"Length: {num}"
)

# ===== 3. 组合处理器 =====

# 方式1：手动组合
def pipeline(text: str) -> str:
    cleaned = text_processor.process(text)
    length = length_calculator.process(cleaned)
    result = formatter.process(length)
    return result

# 方式2：使用 Runnable 组合
chain: Runnable[str, str] = (
    text_processor.to_runnable()
    | length_calculator.to_runnable()
    | formatter.to_runnable()
)

# ===== 4. 类型安全验证 =====
print("=== 类型安全的数据处理管道 ===")

# ✅ 类型正确
input_text = "  Hello World  "
result: str = chain.invoke(input_text)
print(f"输入: '{input_text}'")
print(f"输出: {result}")

# ❌ 类型错误（编译时发现）
# result: int = chain.invoke(input_text)  # 类型检查器报错

# ===== 5. 泛型工具函数 =====
T = TypeVar('T')
U = TypeVar('U')

def compose(
    first: Processor[T, U],
    second: Processor[U, Output]
) -> Processor[T, Output]:
    """组合两个处理器"""
    return Processor(lambda x: second.process(first.process(x)))

# 使用组合
combined: Processor[str, int] = compose(text_processor, length_calculator)
length: int = combined.process("  Hello  ")
print(f"\n组合处理器结果: {length}")

# ===== 6. 批处理支持 =====
class BatchProcessor(Generic[Input, Output]):
    """支持批处理的泛型处理器"""

    def __init__(self, process_fn: Callable[[Input], Output]):
        self.process_fn = process_fn

    def process_batch(self, items: list[Input]) -> list[Output]:
        return [self.process_fn(item) for item in items]

# 使用批处理
batch_processor: BatchProcessor[str, int] = BatchProcessor(len)
texts = ["hello", "world", "python"]
lengths: list[int] = batch_processor.process_batch(texts)
print(f"\n批处理结果: {lengths}")
```

**运行输出**：
```
=== 类型安全的数据处理管道 ===
输入: '  Hello World  '
输出: Length: 11

组合处理器结果: 5

批处理结果: [5, 5, 6]
```

---

## 常见陷阱

### 1. TypeVar 作用域

```python
from typing import TypeVar

# ❌ 错误：每次调用都创建新的 TypeVar
def bad_function():
    T = TypeVar('T')  # 不要在函数内定义
    def inner(x: T) -> T:
        return x
    return inner

# ✅ 正确：在模块级别定义
T = TypeVar('T')

def good_function(x: T) -> T:
    return x
```

### 2. TypeVar 与 Generic 的关系

```python
from typing import TypeVar, Generic

T = TypeVar('T')

# ❌ 错误：忘记继承 Generic
class Box:
    def __init__(self, value: T):  # T 未定义
        self.value = value

# ✅ 正确：继承 Generic[T]
class Box(Generic[T]):
    def __init__(self, value: T):
        self.value = value
```

### 3. 协变逆变的误用

```python
from typing import TypeVar, Generic

# ❌ 错误：协变类型出现在参数位置
T_co = TypeVar('T_co', covariant=True)

class Bad(Generic[T_co]):
    def set(self, value: T_co) -> None:  # 类型检查器报错
        ...

# ✅ 正确：协变类型只出现在返回值位置
class Good(Generic[T_co]):
    def get(self) -> T_co:
        ...
```

---

## 学习检查清单

- [ ] 理解泛型的作用（类型安全 + 代码复用）
- [ ] 掌握 TypeVar 的定义和使用
- [ ] 了解 bound 和 constraints 的区别
- [ ] 理解协变、逆变、不变的概念
- [ ] 掌握 Generic 类的创建
- [ ] 了解 LangChain Runnable[Input, Output] 的设计
- [ ] 能够创建类型安全的泛型组件
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 3**：Protocol 协议 - 学习结构类型系统
- **核心概念 4**：Runnable 泛型设计 - 深入 LangChain 类型系统
- **核心概念 10**：高级类型技巧 - 学习 ParamSpec、Self 等高级特性

---

## 参考资源

1. [How to Build Generic Types with TypeVar in Python](https://oneuptime.com/blog/post/2026-01-30-python-typevar-generics/view) - 2026 最佳实践
2. [Mastering Python Generics: The Power of TypeVar](https://medium.com/@tihomir.manushev/mastering-python-generics-the-power-of-typevar-26978f688acd) - 2025
3. [Python typing 官方文档 - Generic](https://docs.python.org/3/library/typing.html#typing.Generic)
4. [PEP 484 – Type Hints](https://peps.python.org/pep-0484/) - TypeVar 规范
5. [Real Python - Python Type Checking](https://realpython.com/python-type-checking/) - 泛型教程
6. [LangChain Runnables 源码](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/base.py) - Runnable[Input, Output] 实现
