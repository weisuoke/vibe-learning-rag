# 核心概念 6：Callable 类型

## 一句话定义

**Callable 是 Python 中表示可调用对象（函数、方法、lambda）的类型注解，通过 `Callable[[参数类型...], 返回类型]` 语法定义函数签名，是 LangChain RunnableLambda 和高阶函数的类型基础。**

---

## 为什么需要 Callable 类型？

### 问题：函数作为参数的类型安全

在 Python 中，函数是一等公民，可以作为参数传递：

```python
# 没有类型注解（不安全）
def apply_operation(x, y, operation):
    return operation(x, y)

def add(a, b):
    return a + b

result = apply_operation(5, 3, add)  # ⚠️ 类型不明确
result = apply_operation(5, 3, "not a function")  # ❌ 运行时才报错
```

**Callable 的解决方案**：

```python
from typing import Callable

# 使用 Callable 类型注解
def apply_operation(
    x: int,
    y: int,
    operation: Callable[[int, int], int]
) -> int:
    return operation(x, y)

def add(a: int, b: int) -> int:
    return a + b

# ✅ 类型安全
result: int = apply_operation(5, 3, add)

# ❌ 类型检查器立即报错
result = apply_operation(5, 3, "not a function")
# 类型错误：str is not callable
```

---

## Callable 基础语法

### 1. 基本语法

```python
from typing import Callable

# Callable[[参数类型...], 返回类型]

# 无参数函数
def get_value() -> int:
    return 42

getter: Callable[[], int] = get_value

# 单参数函数
def double(x: int) -> int:
    return x * 2

doubler: Callable[[int], int] = double

# 多参数函数
def add(a: int, b: int) -> int:
    return a + b

adder: Callable[[int, int], int] = add

# 无返回值
def log_message(message: str) -> None:
    print(message)

logger: Callable[[str], None] = log_message
```

### 2. Lambda 表达式

```python
from typing import Callable

# Lambda 也是 Callable
square: Callable[[int], int] = lambda x: x ** 2
greet: Callable[[str], str] = lambda name: f"Hello, {name}"

# 使用
result = square(5)  # 25
message = greet("Alice")  # "Hello, Alice"
```

### 3. 方法和类方法

```python
from typing import Callable

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    @staticmethod
    def multiply(a: int, b: int) -> int:
        return a * b

    @classmethod
    def from_string(cls, s: str) -> 'Calculator':
        return cls()

# 实例方法（不包含 self）
calc = Calculator()
add_method: Callable[[int, int], int] = calc.add

# 静态方法
multiply_method: Callable[[int, int], int] = Calculator.multiply

# 类方法（不包含 cls）
from_string_method: Callable[[str], Calculator] = Calculator.from_string
```

---

## 高阶函数与 Callable

### 1. 接受函数作为参数

```python
from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def map_list(items: list[T], fn: Callable[[T], U]) -> list[U]:
    """对列表中的每个元素应用函数"""
    return [fn(item) for item in items]

# 使用
numbers = [1, 2, 3, 4, 5]
doubled = map_list(numbers, lambda x: x * 2)  # [2, 4, 6, 8, 10]
strings = map_list(numbers, str)  # ["1", "2", "3", "4", "5"]
```

### 2. 返回函数

```python
from typing import Callable

def make_multiplier(factor: int) -> Callable[[int], int]:
    """创建一个乘法函数"""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

# 使用
double = make_multiplier(2)
triple = make_multiplier(3)

result = double(5)  # 10
result = triple(5)  # 15
```

### 3. 装饰器

```python
from typing import Callable, TypeVar, Any
from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])

def log_calls(func: F) -> F:
    """装饰器：记录函数调用"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper  # type: ignore

# 使用
@log_calls
def add(a: int, b: int) -> int:
    return a + b

result = add(5, 3)
# 输出：
# Calling add
# Result: 8
```

---

## ParamSpec：保留函数签名

### 1. 问题：装饰器丢失签名信息

```python
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def decorator(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args: Any, **kwargs: Any) -> T:
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@decorator
def add(a: int, b: int) -> int:
    return a + b

# ⚠️ 类型检查器不知道 add 的参数类型
result = add(5, 3)  # 类型推断为 Any
result = add("hello", "world")  # 类型检查器不会报错
```

### 2. ParamSpec 解决方案

```python
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def decorator(func: Callable[P, T]) -> Callable[P, T]:
    """保留函数签名的装饰器"""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@decorator
def add(a: int, b: int) -> int:
    return a + b

# ✅ 类型检查器知道 add 的参数类型
result: int = add(5, 3)  # ✅ 类型正确
result = add("hello", "world")  # ❌ 类型错误
```

### 3. ParamSpec 的实际应用

```python
from typing import Callable, ParamSpec, TypeVar, Any
from functools import wraps
import time

P = ParamSpec('P')
T = TypeVar('T')

def timing_decorator(func: Callable[P, T]) -> Callable[P, T]:
    """测量函数执行时间"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timing_decorator
def process_data(data: list[int], multiplier: int) -> list[int]:
    return [x * multiplier for x in data]

# ✅ 类型安全
result: list[int] = process_data([1, 2, 3], 2)
```

---

## Concatenate：添加额外参数

### 1. 基本用法

```python
from typing import Callable, ParamSpec, TypeVar, Concatenate

P = ParamSpec('P')
T = TypeVar('T')

def add_logging(
    func: Callable[P, T]
) -> Callable[Concatenate[str, P], T]:
    """添加日志参数的装饰器"""
    def wrapper(log_level: str, *args: P.args, **kwargs: P.kwargs) -> T:
        print(f"[{log_level}] Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@add_logging
def add(a: int, b: int) -> int:
    return a + b

# 使用：需要额外的 log_level 参数
result = add("INFO", 5, 3)  # [INFO] Calling add
```

### 2. 实际应用：依赖注入

```python
from typing import Callable, ParamSpec, TypeVar, Concatenate

P = ParamSpec('P')
T = TypeVar('T')

class Database:
    def query(self, sql: str) -> list[dict]:
        return [{"id": 1, "name": "Alice"}]

def inject_db(
    func: Callable[Concatenate[Database, P], T]
) -> Callable[P, T]:
    """注入数据库依赖"""
    db = Database()
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(db, *args, **kwargs)
    return wrapper

@inject_db
def get_user(db: Database, user_id: int) -> dict:
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")[0]

# 使用：不需要传递 db 参数
user = get_user(1)  # Database 自动注入
```

---

## 在 LangChain 中的应用

### 1. RunnableLambda

```python
from typing import Callable
from langchain_core.runnables import RunnableLambda, Runnable

# RunnableLambda 接受 Callable
def to_upper(text: str) -> str:
    return text.upper()

# 创建 Runnable
runnable: Runnable[str, str] = RunnableLambda(to_upper)

# 也可以使用 lambda
runnable: Runnable[str, str] = RunnableLambda(lambda x: x.upper())
```

### 2. 自定义转换函数

```python
from typing import Callable
from langchain_core.runnables import RunnableLambda

# 定义转换函数类型
Transform = Callable[[str], str]

# 创建多个转换
transforms: list[Transform] = [
    lambda x: x.strip(),
    lambda x: x.lower(),
    lambda x: x.replace(" ", "_")
]

# 组合转换
def compose_transforms(transforms: list[Transform]) -> Transform:
    def combined(text: str) -> str:
        result = text
        for transform in transforms:
            result = transform(result)
        return result
    return combined

# 使用
combined_transform = compose_transforms(transforms)
runnable = RunnableLambda(combined_transform)
result = runnable.invoke("  Hello World  ")  # "hello_world"
```

### 3. 回调函数

```python
from typing import Callable, Any
from langchain_core.runnables import Runnable

# 定义回调类型
OnStart = Callable[[str], None]
OnEnd = Callable[[str, Any], None]
OnError = Callable[[str, Exception], None]

class CallbackRunnable(Runnable[str, str]):
    """支持回调的 Runnable"""

    def __init__(
        self,
        process_fn: Callable[[str], str],
        on_start: OnStart | None = None,
        on_end: OnEnd | None = None,
        on_error: OnError | None = None
    ):
        self.process_fn = process_fn
        self.on_start = on_start
        self.on_end = on_end
        self.on_error = on_error

    def invoke(self, input: str, config: Any = None) -> str:
        if self.on_start:
            self.on_start(input)

        try:
            result = self.process_fn(input)
            if self.on_end:
                self.on_end(input, result)
            return result
        except Exception as e:
            if self.on_error:
                self.on_error(input, e)
            raise

# 使用
runnable = CallbackRunnable(
    process_fn=lambda x: x.upper(),
    on_start=lambda x: print(f"Processing: {x}"),
    on_end=lambda x, r: print(f"Result: {r}"),
    on_error=lambda x, e: print(f"Error: {e}")
)
```

---

## 2025-2026 最佳实践

### 1. 使用 ParamSpec 保留签名

```python
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')

# ✅ 推荐：使用 ParamSpec
def decorator(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper

# ❌ 不推荐：使用 ...
def decorator(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper
```

### 2. 明确参数和返回类型

```python
# ❌ 不明确
callback: Callable = lambda x: x * 2

# ✅ 明确
callback: Callable[[int], int] = lambda x: x * 2
```

### 3. 使用 Protocol 定义复杂签名

```python
from typing import Protocol

# 复杂的 Callable 签名可以用 Protocol
class Processor(Protocol):
    def __call__(self, data: str, *, verbose: bool = False) -> dict[str, Any]:
        ...

# 使用
def process_with_callback(data: str, callback: Processor) -> dict[str, Any]:
    return callback(data, verbose=True)
```

### 4. 避免过度使用 Callable[..., Any]

```python
# ❌ 失去类型安全
def apply(fn: Callable[..., Any], *args: Any) -> Any:
    return fn(*args)

# ✅ 使用泛型
from typing import TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def apply(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return fn(*args, **kwargs)
```

---

## 实战示例：类型安全的管道构建器

```python
"""
类型安全的管道构建器
演示：Callable 在实际项目中的应用
"""

from typing import Callable, TypeVar, Generic
from langchain_core.runnables import Runnable, RunnableLambda

# ===== 1. 定义类型 =====
Input = TypeVar('Input')
Output = TypeVar('Output')
Intermediate = TypeVar('Intermediate')

# 转换函数类型
Transform = Callable[[Input], Output]

# ===== 2. 管道构建器 =====

class Pipeline(Generic[Input, Output]):
    """类型安全的管道构建器"""

    def __init__(self, transform: Transform[Input, Output]):
        self.transform = transform

    def __call__(self, input: Input) -> Output:
        return self.transform(input)

    def then(
        self,
        next_transform: Transform[Output, Intermediate]
    ) -> 'Pipeline[Input, Intermediate]':
        """链接下一个转换"""
        def combined(input: Input) -> Intermediate:
            intermediate = self.transform(input)
            return next_transform(intermediate)
        return Pipeline(combined)

    def to_runnable(self) -> Runnable[Input, Output]:
        """转换为 Runnable"""
        return RunnableLambda(self.transform)

# ===== 3. 定义转换函数 =====

# 文本处理
def clean_text(text: str) -> str:
    """清理文本"""
    return text.strip().lower()

def extract_words(text: str) -> list[str]:
    """提取单词"""
    return text.split()

def count_words(words: list[str]) -> int:
    """计数单词"""
    return len(words)

def format_count(count: int) -> str:
    """格式化计数"""
    return f"Word count: {count}"

# ===== 4. 构建管道 =====

# 方式1：使用 Pipeline 类
text_pipeline: Pipeline[str, str] = (
    Pipeline(clean_text)
    .then(extract_words)
    .then(count_words)
    .then(format_count)
)

# 方式2：手动组合
def manual_pipeline(text: str) -> str:
    cleaned = clean_text(text)
    words = extract_words(cleaned)
    count = count_words(words)
    return format_count(count)

# 方式3：使用 Runnable
runnable_pipeline: Runnable[str, str] = (
    RunnableLambda(clean_text)
    | RunnableLambda(extract_words)
    | RunnableLambda(count_words)
    | RunnableLambda(format_count)
)

# ===== 5. 使用管道 =====

print("=== 类型安全的管道构建器 ===\n")

input_text = "  Hello World Python  "

# 使用 Pipeline
result1 = text_pipeline(input_text)
print(f"Pipeline result: {result1}")

# 使用手动组合
result2 = manual_pipeline(input_text)
print(f"Manual result: {result2}")

# 使用 Runnable
result3 = runnable_pipeline.invoke(input_text)
print(f"Runnable result: {result3}")

# ===== 6. 高阶函数：过滤器 =====

def create_filter(
    predicate: Callable[[str], bool]
) -> Callable[[list[str]], list[str]]:
    """创建过滤器"""
    def filter_fn(items: list[str]) -> list[str]:
        return [item for item in items if predicate(item)]
    return filter_fn

# 使用过滤器
long_words_filter = create_filter(lambda word: len(word) > 5)

filtered_pipeline: Pipeline[str, list[str]] = (
    Pipeline(clean_text)
    .then(extract_words)
    .then(long_words_filter)
)

result = filtered_pipeline(input_text)
print(f"\nFiltered words: {result}")

# ===== 7. 装饰器：添加日志 =====

from typing import ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')

def log_transform(
    func: Callable[P, T]
) -> Callable[P, T]:
    """记录转换过程"""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"  → {func.__name__}({args[0] if args else ''})")
        result = func(*args, **kwargs)
        print(f"  ← {result}")
        return result
    return wrapper

# 使用装饰器
@log_transform
def logged_clean(text: str) -> str:
    return text.strip().lower()

@log_transform
def logged_extract(text: str) -> list[str]:
    return text.split()

print("\n=== 带日志的管道 ===\n")
logged_pipeline = Pipeline(logged_clean).then(logged_extract)
result = logged_pipeline("  Hello World  ")
```

**运行输出**：
```
=== 类型安全的管道构建器 ===

Pipeline result: Word count: 3
Manual result: Word count: 3
Runnable result: Word count: 3

Filtered words: ['python']

=== 带日志的管道 ===

  → logged_clean(  Hello World  )
  ← hello world
  → logged_extract(hello world)
  ← ['hello', 'world']
```

---

## 常见陷阱

### 1. 忘记参数类型

```python
# ❌ 不完整
callback: Callable = lambda x: x * 2

# ✅ 完整
callback: Callable[[int], int] = lambda x: x * 2
```

### 2. 装饰器丢失签名

```python
# ❌ 丢失签名
def decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# ✅ 保留签名
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')

def decorator(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper
```

### 3. 过度使用 Callable[..., Any]

```python
# ❌ 失去类型安全
def apply(fn: Callable[..., Any]) -> Any:
    return fn()

# ✅ 使用具体类型
def apply(fn: Callable[[], int]) -> int:
    return fn()
```

---

## 学习检查清单

- [ ] 理解 Callable 的作用和语法
- [ ] 掌握高阶函数的类型注解
- [ ] 了解 ParamSpec 保留函数签名
- [ ] 掌握 Concatenate 添加额外参数
- [ ] 理解装饰器的类型注解
- [ ] 了解 LangChain 中 Callable 的应用
- [ ] 能够创建类型安全的管道
- [ ] 遵循 2025-2026 最佳实践

---

## 下一步学习

- **核心概念 7**：Literal 与 TypedDict - 学习精确类型和结构化字典
- **核心概念 2**：TypeVar 与泛型 - 复习泛型知识
- **核心概念 10**：高级类型技巧 - 学习更多高级特性

---

## 参考资源

1. [Python typing 官方文档 - Callable](https://docs.python.org/3/library/typing.html#typing.Callable)
2. [PEP 612 – Parameter Specification Variables](https://peps.python.org/pep-0612/) - ParamSpec 规范
3. [A crash course on Python function signatures and typing](https://chipx86.blog/2025/07/12/a-crash-course-on-python-function-signatures-and-typing) - 2025
4. [Typing Best Practices](https://typing.python.org/en/latest/reference/best_practices.html) - 官方最佳实践
5. [LangChain RunnableLambda 文档](https://reference.langchain.com/python/langchain_core/runnables/base.html#langchain_core.runnables.base.RunnableLambda)
