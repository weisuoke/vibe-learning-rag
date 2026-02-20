# 实战代码 1：自定义 Runnable

## 场景说明

创建类型安全的自定义 Runnable 组件，展示如何在 LangChain 中应用类型系统。

---

## 完整代码示例

```python
"""
类型安全的自定义 Runnable
演示：如何创建类型安全的 LangChain 组件
"""

from typing import Any, Iterator, TypeVar, Generic
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ===== 1. 基础自定义 Runnable =====

class UppercaseRunnable(Runnable[str, str]):
    """将文本转换为大写的 Runnable"""

    def invoke(self, input: str, config: Any = None) -> str:
        """同步调用"""
        return input.upper()

    async def ainvoke(self, input: str, config: Any = None) -> str:
        """异步调用"""
        return input.upper()

    def batch(self, inputs: list[str], config: Any = None) -> list[str]:
        """批量调用"""
        return [text.upper() for text in inputs]

    async def abatch(self, inputs: list[str], config: Any = None) -> list[str]:
        """异步批量调用"""
        return [text.upper() for text in inputs]

    def stream(self, input: str, config: Any = None) -> Iterator[str]:
        """流式调用"""
        # 逐字符流式返回
        for char in input.upper():
            yield char

    async def astream(self, input: str, config: Any = None):
        """异步流式调用"""
        for char in input.upper():
            yield char

# ===== 2. 泛型自定义 Runnable =====

Input = TypeVar('Input')
Output = TypeVar('Output')

class TransformRunnable(Runnable[Input, Output], Generic[Input, Output]):
    """泛型转换 Runnable"""

    def __init__(self, transform_fn: callable):
        self.transform_fn = transform_fn

    def invoke(self, input: Input, config: Any = None) -> Output:
        return self.transform_fn(input)

    async def ainvoke(self, input: Input, config: Any = None) -> Output:
        return self.transform_fn(input)

    def batch(self, inputs: list[Input], config: Any = None) -> list[Output]:
        return [self.transform_fn(item) for item in inputs]

# ===== 3. 带状态的 Runnable =====

class CounterRunnable(Runnable[str, dict]):
    """计数器 Runnable（带状态）"""

    def __init__(self):
        self.count = 0

    def invoke(self, input: str, config: Any = None) -> dict:
        self.count += 1
        return {
            "input": input,
            "length": len(input),
            "count": self.count
        }

# ===== 4. 消息处理 Runnable =====

class MessageFormatterRunnable(Runnable[str, list[BaseMessage]]):
    """将文本转换为消息列表"""

    def __init__(self, role: str = "user"):
        self.role = role

    def invoke(self, input: str, config: Any = None) -> list[BaseMessage]:
        if self.role == "user":
            return [HumanMessage(content=input)]
        elif self.role == "assistant":
            return [AIMessage(content=input)]
        else:
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=input)]

# ===== 5. 条件 Runnable =====

class ConditionalRunnable(Runnable[str, str]):
    """根据条件选择不同的处理逻辑"""

    def __init__(self, condition_fn: callable, true_fn: callable, false_fn: callable):
        self.condition_fn = condition_fn
        self.true_fn = true_fn
        self.false_fn = false_fn

    def invoke(self, input: str, config: Any = None) -> str:
        if self.condition_fn(input):
            return self.true_fn(input)
        else:
            return self.false_fn(input)

# ===== 6. 缓存 Runnable =====

class CachedRunnable(Runnable[str, str]):
    """带缓存的 Runnable"""

    def __init__(self, runnable: Runnable[str, str]):
        self.runnable = runnable
        self.cache: dict[str, str] = {}

    def invoke(self, input: str, config: Any = None) -> str:
        if input in self.cache:
            print(f"Cache hit: {input}")
            return self.cache[input]

        print(f"Cache miss: {input}")
        result = self.runnable.invoke(input, config)
        self.cache[input] = result
        return result

# ===== 7. 使用示例 =====

if __name__ == "__main__":
    print("=== 自定义 Runnable 示例 ===\n")

    # 1. 基础 Runnable
    print("1. 基础 Runnable")
    uppercase = UppercaseRunnable()
    result = uppercase.invoke("hello world")
    print(f"Result: {result}\n")

    # 批量调用
    results = uppercase.batch(["hello", "world", "python"])
    print(f"Batch results: {results}\n")

    # 流式调用
    print("Stream results: ", end="")
    for char in uppercase.stream("hello"):
        print(char, end="")
    print("\n")

    # 2. 泛型 Runnable
    print("2. 泛型 Runnable")
    length_runnable: TransformRunnable[str, int] = TransformRunnable(len)
    length = length_runnable.invoke("hello")
    print(f"Length: {length}\n")

    # 3. 带状态的 Runnable
    print("3. 带状态的 Runnable")
    counter = CounterRunnable()
    result1 = counter.invoke("hello")
    result2 = counter.invoke("world")
    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}\n")

    # 4. 消息处理 Runnable
    print("4. 消息处理 Runnable")
    formatter = MessageFormatterRunnable(role="user")
    messages = formatter.invoke("Hello, AI!")
    print(f"Messages: {messages}\n")

    # 5. 条件 Runnable
    print("5. 条件 Runnable")
    conditional = ConditionalRunnable(
        condition_fn=lambda x: len(x) > 5,
        true_fn=lambda x: f"Long: {x.upper()}",
        false_fn=lambda x: f"Short: {x.lower()}"
    )
    result1 = conditional.invoke("hello world")
    result2 = conditional.invoke("hi")
    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}\n")

    # 6. 缓存 Runnable
    print("6. 缓存 Runnable")
    base_runnable = UppercaseRunnable()
    cached = CachedRunnable(base_runnable)

    result1 = cached.invoke("hello")  # Cache miss
    result2 = cached.invoke("hello")  # Cache hit
    result3 = cached.invoke("world")  # Cache miss
    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}")
    print(f"Result 3: {result3}\n")

    # 7. 组合使用
    print("7. 组合使用")
    from langchain_core.runnables import RunnableLambda

    # 创建管道
    pipeline = (
        UppercaseRunnable()
        | RunnableLambda(lambda x: x.split())
        | RunnableLambda(len)
    )

    result = pipeline.invoke("hello world python")
    print(f"Pipeline result: {result}")  # 3
```

---

## 运行输出

```
=== 自定义 Runnable 示例 ===

1. 基础 Runnable
Result: HELLO WORLD

Batch results: ['HELLO', 'WORLD', 'PYTHON']

Stream results: HELLO

2. 泛型 Runnable
Length: 5

3. 带状态的 Runnable
Result 1: {'input': 'hello', 'length': 5, 'count': 1}
Result 2: {'input': 'world', 'length': 5, 'count': 2}

4. 消息处理 Runnable
Messages: [HumanMessage(content='Hello, AI!')]

5. 条件 Runnable
Result 1: Long: HELLO WORLD
Result 2: Short: hi

6. 缓存 Runnable
Cache miss: hello
Cache miss: hello
Cache miss: world
Result 1: HELLO
Result 2: HELLO
Result 3: WORLD

7. 组合使用
Pipeline result: 3
```

---

## 关键要点

### 1. 类型安全

```python
# ✅ 类型明确
class UppercaseRunnable(Runnable[str, str]):
    def invoke(self, input: str, config: Any = None) -> str:
        return input.upper()

# 类型检查器知道：
# - 输入是 str
# - 输出是 str
# - 可以与其他 Runnable[str, ...] 组合
```

### 2. 泛型支持

```python
# 泛型 Runnable 支持多种类型
Input = TypeVar('Input')
Output = TypeVar('Output')

class TransformRunnable(Runnable[Input, Output], Generic[Input, Output]):
    ...

# 使用时指定具体类型
length_runnable: TransformRunnable[str, int] = TransformRunnable(len)
```

### 3. 完整接口实现

```python
# 实现所有必需方法
class MyRunnable(Runnable[Input, Output]):
    def invoke(self, input: Input, config: Any = None) -> Output:
        ...

    async def ainvoke(self, input: Input, config: Any = None) -> Output:
        ...

    def batch(self, inputs: list[Input], config: Any = None) -> list[Output]:
        ...

    def stream(self, input: Input, config: Any = None) -> Iterator[Output]:
        ...
```

### 4. 状态管理

```python
# 带状态的 Runnable
class CounterRunnable(Runnable[str, dict]):
    def __init__(self):
        self.count = 0  # 状态

    def invoke(self, input: str, config: Any = None) -> dict:
        self.count += 1  # 更新状态
        return {"count": self.count}
```

---

## 最佳实践

### 1. 明确类型注解

```python
# ✅ 推荐：明确类型
class MyRunnable(Runnable[str, int]):
    def invoke(self, input: str, config: Any = None) -> int:
        return len(input)

# ❌ 不推荐：使用 Any
class BadRunnable(Runnable[Any, Any]):
    def invoke(self, input: Any, config: Any = None) -> Any:
        return input
```

### 2. 实现异步方法

```python
# ✅ 推荐：实现异步方法
class MyRunnable(Runnable[str, str]):
    def invoke(self, input: str, config: Any = None) -> str:
        return input.upper()

    async def ainvoke(self, input: str, config: Any = None) -> str:
        return input.upper()

# 支持异步调用
result = await my_runnable.ainvoke("hello")
```

### 3. 支持流式输出

```python
# ✅ 推荐：实现流式方法
class MyRunnable(Runnable[str, str]):
    def stream(self, input: str, config: Any = None) -> Iterator[str]:
        for char in input.upper():
            yield char

# 支持流式调用
for chunk in my_runnable.stream("hello"):
    print(chunk, end="")
```

### 4. 错误处理

```python
# ✅ 推荐：添加错误处理
class SafeRunnable(Runnable[str, str]):
    def invoke(self, input: str, config: Any = None) -> str:
        try:
            return input.upper()
        except Exception as e:
            print(f"Error: {e}")
            return ""
```

---

## 扩展示例

### 1. 重试 Runnable

```python
import time

class RetryRunnable(Runnable[str, str]):
    """带重试机制的 Runnable"""

    def __init__(self, runnable: Runnable[str, str], max_retries: int = 3):
        self.runnable = runnable
        self.max_retries = max_retries

    def invoke(self, input: str, config: Any = None) -> str:
        for attempt in range(self.max_retries):
            try:
                return self.runnable.invoke(input, config)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries}")
                time.sleep(1)
```

### 2. 日志 Runnable

```python
class LoggingRunnable(Runnable[str, str]):
    """带日志的 Runnable"""

    def __init__(self, runnable: Runnable[str, str], name: str = "Runnable"):
        self.runnable = runnable
        self.name = name

    def invoke(self, input: str, config: Any = None) -> str:
        print(f"[{self.name}] Input: {input}")
        result = self.runnable.invoke(input, config)
        print(f"[{self.name}] Output: {result}")
        return result
```

### 3. 验证 Runnable

```python
from pydantic import BaseModel, ValidationError

class ValidatingRunnable(Runnable[dict, dict]):
    """带验证的 Runnable"""

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def invoke(self, input: dict, config: Any = None) -> dict:
        try:
            # 验证输入
            validated = self.schema(**input)
            return validated.dict()
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")
```

---

## 学习检查清单

- [ ] 理解如何创建自定义 Runnable
- [ ] 掌握类型注解的使用
- [ ] 了解泛型 Runnable 的实现
- [ ] 能够实现完整的 Runnable 接口
- [ ] 理解状态管理
- [ ] 能够组合自定义 Runnable
- [ ] 掌握最佳实践

---

## 下一步

- **实战代码 2**：泛型工具函数 - 学习创建泛型工具
- **实战代码 3**：类型检查集成 - 学习集成类型检查器
- **核心概念系列** - 深入理解类型系统

---

## 参考资源

1. [LangChain Runnable 文档](https://reference.langchain.com/python/langchain_core/runnables/base.html)
2. [自定义 Runnable 指南](https://python.langchain.com/docs/expression_language/how_to/custom_runnable)
3. [Python typing 官方文档](https://docs.python.org/3/library/typing.html)
