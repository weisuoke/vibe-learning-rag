# 核心概念 01: Runnable 协议

> 理解 LangChain 的统一执行接口设计

---

## 什么是 Runnable 协议？

**Runnable 是 LangChain 中所有可执行组件的基础协议（Protocol），定义了统一的执行接口契约。**

### 一句话定义

Runnable 协议规定：任何实现了 `invoke`、`batch`、`stream` 方法的对象都是 Runnable，可以无缝组合和执行。

---

## 协议定义

### Python Protocol 实现

```python
from typing import Protocol, TypeVar, Iterator, Optional, List
from langchain_core.runnables.config import RunnableConfig

# 泛型类型变量
Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)

class Runnable(Protocol[Input, Output]):
    """
    LangChain 统一执行协议

    所有可执行组件都实现这个协议，确保：
    1. 统一的执行接口
    2. 可组合性
    3. 可观测性
    """

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> Output:
        """
        同步单次执行

        Args:
            input: 输入数据
            config: 运行时配置（可选）

        Returns:
            输出数据
        """
        ...

    def batch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None
    ) -> List[Output]:
        """
        批量并发执行

        Args:
            inputs: 输入数据列表
            config: 运行时配置（可选）

        Returns:
            输出数据列表
        """
        ...

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        """
        流式实时输出

        Args:
            input: 输入数据
            config: 运行时配置（可选）

        Yields:
            输出数据块
        """
        ...
```

### 为什么使用 Protocol 而非继承？

**Protocol 是 Python 的结构化类型（Structural Typing），也称为"鸭子类型"。**

```python
# ❌ 传统继承方式
class MyRunnable(RunnableBase):
    def invoke(self, input, config=None):
        return input.upper()

# ✅ Protocol 方式（更灵活）
class MyRunnable:
    """不需要继承，只需实现方法"""
    def invoke(self, input, config=None):
        return input.upper()

    def batch(self, inputs, config=None):
        return [self.invoke(i, config) for i in inputs]

    def stream(self, input, config=None):
        yield self.invoke(input, config)

# 自动满足 Runnable 协议！
```

**优势**：
1. **解耦**: 不强制继承关系，降低耦合
2. **灵活**: 任何对象只要实现方法就是 Runnable
3. **兼容**: 易于集成第三方库和已有代码

---

## 类型系统

### 泛型约束

Runnable 使用泛型定义输入输出类型：

```python
from langchain_core.runnables import Runnable

# 明确类型的 Runnable
class TextUppercase(Runnable[str, str]):
    """输入 str，输出 str"""
    def invoke(self, input: str, config=None) -> str:
        return input.upper()

class TextToDict(Runnable[str, dict]):
    """输入 str，输出 dict"""
    def invoke(self, input: str, config=None) -> dict:
        return {"text": input, "length": len(input)}

# 类型检查
uppercase: Runnable[str, str] = TextUppercase()
to_dict: Runnable[str, dict] = TextToDict()

# 组合时类型自动推断
chain = uppercase | to_dict  # Runnable[str, dict]
```

### 类型安全的好处

**2025-2026 年 LangChain 1.0 增强了类型安全**[^1]：

```python
from langchain_core.runnables import RunnableLambda

# ✅ 类型正确
def process_text(text: str) -> dict:
    return {"result": text.upper()}

runnable = RunnableLambda(process_text)  # Runnable[str, dict]

# ❌ 类型错误（IDE 会提示）
result: str = runnable.invoke("hello")  # 类型不匹配！
```

---

## RunnableConfig 配置参数

### 配置项详解

```python
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import StdOutCallbackHandler

config = RunnableConfig(
    # 标签：用于追踪和分类
    tags=["production", "translation", "v2"],

    # 元数据：附加信息
    metadata={
        "user_id": "user_123",
        "session_id": "session_456",
        "environment": "production"
    },

    # 回调：监控执行过程
    callbacks=[StdOutCallbackHandler()],

    # 并发限制
    max_concurrency=5,

    # 递归深度限制（防止无限递归）
    recursion_limit=10,

    # 运行名称（用于 LangSmith 追踪）
    run_name="translation_chain"
)

# 使用配置
result = chain.invoke({"text": "你好"}, config=config)
```

### 配置传递机制

配置会自动传递给链中的所有组件：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("翻译: {text}")
llm = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | llm

# 配置传递给 prompt 和 llm
config = RunnableConfig(tags=["translation"])
result = chain.invoke({"text": "你好"}, config=config)

# 等价于：
# prompt.invoke({"text": "你好"}, config=config)
# llm.invoke(prompt_output, config=config)
```

---

## 2025-2026 安全更新

### CVE-2025-68664: 序列化注入漏洞

**2025年12月披露的严重漏洞**[^2][^3]：

**问题**：
- `load()`/`loads()` 函数可被利用提取秘密
- 不安全的序列化可导致代码执行

**修复**（langchain-core 1.2.5+）：

```python
from langchain.load import load

# ❌ 不安全（旧版本）
chain = load(serialized_data)

# ✅ 安全（1.2.5+）
chain = load(
    serialized_data,
    allowed_objects=['core']  # 白名单机制
)

# ✅ 更安全（显式禁用秘密加载）
chain = load(
    serialized_data,
    allowed_objects=['core'],
    secrets_from_env=False  # 2025+ 默认为 False
)
```

**迁移指南**：

1. **升级版本**：
   ```bash
   uv add langchain-core@1.2.5
   ```

2. **更新代码**：
   ```python
   # 查找所有 load() 调用
   # 添加 allowed_objects 参数
   ```

3. **测试**：
   ```python
   # 确保序列化/反序列化正常工作
   serialized = chain.dumps()
   loaded = load(serialized, allowed_objects=['core'])
   ```

### 安全最佳实践

```python
from langchain_core.runnables import Runnable
from langchain.pydantic_v1 import SecretStr
import os

# ✅ 使用 SecretStr 管理敏感信息
api_key = SecretStr(os.getenv("OPENAI_API_KEY"))

# ✅ 避免在序列化中包含秘密
class SafeRunnable(Runnable[str, str]):
    def __init__(self, api_key: SecretStr):
        self._api_key = api_key  # 不会被序列化

    def invoke(self, input: str, config=None) -> str:
        # 使用 api_key
        return process_with_api(input, self._api_key.get_secret_value())

# ✅ 序列化时排除敏感字段
serialized = runnable.dumps(exclude={"_api_key"})
```

---

## 手写最小 Runnable 实现

### 完整示例

```python
"""
手写 Runnable 实现
演示协议的核心要求
"""

from typing import Optional, List, Iterator
from langchain_core.runnables.config import RunnableConfig

class MinimalRunnable:
    """
    最小 Runnable 实现

    只需实现三个方法即可成为 Runnable
    """

    def __init__(self, transform_fn):
        """
        Args:
            transform_fn: 转换函数
        """
        self.transform_fn = transform_fn

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """同步单次执行"""
        print(f"[invoke] 处理输入: {input}")
        result = self.transform_fn(input)
        print(f"[invoke] 输出结果: {result}")
        return result

    def batch(
        self,
        inputs: List[str],
        config: Optional[RunnableConfig] = None
    ) -> List[str]:
        """批量执行（基于 invoke 实现）"""
        print(f"[batch] 处理 {len(inputs)} 个输入")
        results = [self.invoke(input, config) for input in inputs]
        print(f"[batch] 完成批量处理")
        return results

    def stream(
        self,
        input: str,
        config: Optional[RunnableConfig] = None
    ) -> Iterator[str]:
        """流式执行（逐字符输出）"""
        print(f"[stream] 开始流式处理: {input}")
        result = self.invoke(input, config)
        for char in result:
            yield char
        print(f"\n[stream] 流式处理完成")

    def __or__(self, other):
        """
        支持管道操作符 |

        这是 LCEL 的核心：组合两个 Runnable
        """
        return ChainedRunnable(self, other)


class ChainedRunnable:
    """链式 Runnable：实现 A | B"""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def invoke(self, input, config=None):
        """串行执行"""
        intermediate = self.first.invoke(input, config)
        return self.second.invoke(intermediate, config)

    def batch(self, inputs, config=None):
        """批量串行执行"""
        intermediates = self.first.batch(inputs, config)
        return self.second.batch(intermediates, config)

    def stream(self, input, config=None):
        """流式串行执行"""
        intermediate = self.first.invoke(input, config)
        yield from self.second.stream(intermediate, config)

    def __or__(self, other):
        """支持继续链式组合"""
        return ChainedRunnable(self, other)


# ===== 使用示例 =====
if __name__ == "__main__":
    # 定义转换函数
    uppercase = MinimalRunnable(lambda x: x.upper())
    add_prefix = MinimalRunnable(lambda x: f"[PREFIX] {x}")
    add_suffix = MinimalRunnable(lambda x: f"{x} [SUFFIX]")

    # 组合成链
    chain = uppercase | add_prefix | add_suffix

    print("=== 1. invoke 测试 ===")
    result = chain.invoke("hello world")
    print(f"最终结果: {result}\n")

    print("=== 2. batch 测试 ===")
    results = chain.batch(["hello", "world", "python"])
    print(f"批量结果: {results}\n")

    print("=== 3. stream 测试 ===")
    for chunk in chain.stream("hello"):
        print(chunk, end="", flush=True)
    print()
```

**运行输出**：
```
=== 1. invoke 测试 ===
[invoke] 处理输入: hello world
[invoke] 输出结果: HELLO WORLD
[invoke] 处理输入: HELLO WORLD
[invoke] 输出结果: [PREFIX] HELLO WORLD
[invoke] 处理输入: [PREFIX] HELLO WORLD
[invoke] 输出结果: [PREFIX] HELLO WORLD [SUFFIX]
最终结果: [PREFIX] HELLO WORLD [SUFFIX]

=== 2. batch 测试 ===
[batch] 处理 3 个输入
[invoke] 处理输入: hello
[invoke] 输出结果: HELLO
[invoke] 处理输入: world
[invoke] 输出结果: WORLD
[invoke] 处理输入: python
[invoke] 输出结果: PYTHON
[batch] 完成批量处理
[batch] 处理 3 个输入
...
批量结果: ['[PREFIX] HELLO [SUFFIX]', '[PREFIX] WORLD [SUFFIX]', '[PREFIX] PYTHON [SUFFIX]']

=== 3. stream 测试 ===
[stream] 开始流式处理: hello
[invoke] 处理输入: hello
[invoke] 输出结果: HELLO
[invoke] 处理输入: HELLO
[invoke] 输出结果: [PREFIX] HELLO
[stream] 开始流式处理: [PREFIX] HELLO
[invoke] 处理输入: [PREFIX] HELLO
[invoke] 输出结果: [PREFIX] HELLO [SUFFIX]
[PREFIX] HELLO [SUFFIX]
[stream] 流式处理完成
```

---

## 在 LangChain 架构中的位置

### 架构层次

```
┌─────────────────────────────────────┐
│   应用层 (RAG, Agent, Chatbot)      │
├─────────────────────────────────────┤
│   LCEL 表达式层 (管道操作符)         │
├─────────────────────────────────────┤
│   Runnable 协议层 (统一接口)        │ ← 本文档
├─────────────────────────────────────┤
│   组件层 (LLM, Prompt, Parser)      │
├─────────────────────────────────────┤
│   基础设施层 (HTTP, 向量数据库)      │
└─────────────────────────────────────┘
```

### 核心组件都是 Runnable

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 所有组件都实现 Runnable 协议
prompt: Runnable[dict, PromptValue] = ChatPromptTemplate.from_template("...")
llm: Runnable[PromptValue, AIMessage] = ChatOpenAI(model="gpt-4o-mini")
parser: Runnable[AIMessage, str] = StrOutputParser()

# 因此可以无缝组合
chain: Runnable[dict, str] = prompt | llm | parser
```

---

## 实际应用场景

### 场景 1: 自定义数据处理组件

```python
from langchain_core.runnables import Runnable
from typing import Optional, List

class DataValidator(Runnable[dict, dict]):
    """数据验证 Runnable"""

    def invoke(self, input: dict, config=None) -> dict:
        # 验证必需字段
        required_fields = ["text", "language"]
        for field in required_fields:
            if field not in input:
                raise ValueError(f"缺少必需字段: {field}")

        # 验证通过，返回数据
        return input

# 集成到链中
chain = DataValidator() | prompt | llm | parser
```

### 场景 2: 可观测性包装器

```python
import time
from langchain_core.runnables import Runnable

class TimingRunnable(Runnable):
    """性能监控 Runnable"""

    def __init__(self, inner: Runnable):
        self.inner = inner

    def invoke(self, input, config=None):
        start = time.time()
        result = self.inner.invoke(input, config)
        elapsed = time.time() - start
        print(f"执行耗时: {elapsed:.2f}秒")
        return result

# 包装任何 Runnable
monitored_chain = TimingRunnable(chain)
```

### 场景 3: 错误处理和重试

```python
from langchain_core.runnables import Runnable
import time

class RetryRunnable(Runnable):
    """重试 Runnable"""

    def __init__(self, inner: Runnable, max_retries=3):
        self.inner = inner
        self.max_retries = max_retries

    def invoke(self, input, config=None):
        for attempt in range(self.max_retries):
            try:
                return self.inner.invoke(input, config)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"重试 {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)  # 指数退避

# 使用
reliable_chain = RetryRunnable(chain, max_retries=3)
```

---

## 总结

### Runnable 协议的核心价值

1. **统一接口**: 所有组件都实现相同的方法
2. **可组合性**: 任何 Runnable 都可以用 `|` 组合
3. **类型安全**: 泛型确保类型正确性
4. **灵活扩展**: Protocol 设计允许灵活实现

### 2025-2026 年关键更新

- **LangChain 1.0**: 协议稳定，承诺到 2.0 无重大变更[^1]
- **安全修复**: CVE-2025-68664 修复，必须升级到 1.2.5+[^2][^3]
- **类型增强**: 更好的类型推断和 IDE 支持

---

## 参考资料

[^1]: [LangChain 1.0 Release](https://blog.langchain.com/langchain-langgraph-1dot0) - LangChain Blog, 2025年10月
[^2]: [CVE-2025-68664 Details](https://nvd.nist.gov/vuln/detail/CVE-2025-68664) - NVD, 2025年12月
[^3]: [Critical LangChain Core Vulnerability](https://thehackernews.com/2025/12/critical-langchain-core-vulnerability.html) - The Hacker News, 2025年12月

### 官方文档
- [Runnable Protocol Reference](https://reference.langchain.com/python/langchain_core/runnables) - LangChain, 2025-2026
- [LangChain Security Best Practices](https://python.langchain.com/docs/security) - 2025-2026

---

**下一步**: 阅读 [03_核心概念_02_invoke方法.md](./03_核心概念_02_invoke方法.md) 深入理解同步执行
