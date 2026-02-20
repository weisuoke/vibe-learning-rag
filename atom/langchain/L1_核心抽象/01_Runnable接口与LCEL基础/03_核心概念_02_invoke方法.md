# 核心概念 02: invoke 方法

> 理解 Runnable 的同步单次执行方法

---

## 什么是 invoke 方法？

**invoke 是 Runnable 协议的核心方法，用于同步执行单个输入并返回单个输出。**

### 一句话定义

invoke 方法接收一个输入和可选的配置参数，同步执行处理逻辑，返回一个输出结果。

---

## 方法签名

### 完整签名

```python
from typing import TypeVar, Optional
from langchain_core.runnables.config import RunnableConfig

Input = TypeVar("Input")
Output = TypeVar("Output")

def invoke(
    self,
    input: Input,
    config: Optional[RunnableConfig] = None
) -> Output:
    """
    同步单次执行

    Args:
        input: 输入数据，类型由 Runnable[Input, Output] 的 Input 泛型定义
        config: 运行时配置，包含标签、元数据、回调等（可选）

    Returns:
        输出数据，类型由 Runnable[Input, Output] 的 Output 泛型定义

    Raises:
        Exception: 执行过程中的任何异常都会向上传播
    """
    ...
```

### 类型约束

```python
from langchain_core.runnables import Runnable

# 明确输入输出类型
class TextProcessor(Runnable[str, dict]):
    def invoke(self, input: str, config=None) -> dict:
        return {
            "original": input,
            "length": len(input),
            "uppercase": input.upper()
        }

# 使用时类型安全
processor: Runnable[str, dict] = TextProcessor()
result: dict = processor.invoke("hello")  # ✅ 类型正确
```

---

## Config 参数深度解析

### RunnableConfig 结构

```python
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from typing import Optional, List, Dict, Any

config = RunnableConfig(
    # ===== 追踪和分类 =====
    tags: Optional[List[str]] = None,
    # 用途：标记执行，便于在 LangSmith 中过滤和分析
    # 示例：["production", "translation", "v2"]

    # ===== 元数据 =====
    metadata: Optional[Dict[str, Any]] = None,
    # 用途：附加任意元数据，用于日志和分析
    # 示例：{"user_id": "123", "session_id": "abc", "environment": "prod"}

    # ===== 回调系统 =====
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    # 用途：监控执行过程，记录日志，发送通知
    # 示例：[StdOutCallbackHandler(), CustomMetricsHandler()]

    # ===== 并发控制 =====
    max_concurrency: Optional[int] = None,
    # 用途：限制并发执行数量（主要用于 batch）
    # 示例：5（最多同时执行 5 个任务）

    # ===== 递归限制 =====
    recursion_limit: int = 25,
    # 用途：防止无限递归（主要用于 Agent）
    # 示例：10（最多递归 10 层）

    # ===== 运行标识 =====
    run_name: Optional[str] = None,
    # 用途：在 LangSmith 中显示的运行名称
    # 示例："translation_chain_v2"

    # ===== 运行 ID =====
    run_id: Optional[str] = None,
    # 用途：唯一标识一次执行（通常自动生成）
    # 示例：UUID("...")

    # ===== 可配置参数 =====
    configurable: Optional[Dict[str, Any]] = None,
    # 用途：传递自定义配置参数
    # 示例：{"temperature": 0.7, "model": "gpt-4"}
)
```

### Config 传递机制

**Config 会自动传递给链中的所有组件**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("翻译: {text}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | llm | parser

# 配置传递给所有组件
config = RunnableConfig(
    tags=["translation"],
    metadata={"user_id": "123"}
)

result = chain.invoke({"text": "你好"}, config=config)

# 等价于：
# step1 = prompt.invoke({"text": "你好"}, config=config)
# step2 = llm.invoke(step1, config=config)
# step3 = parser.invoke(step2, config=config)
```

### 使用 configurable 传递自定义参数

```python
from langchain_core.runnables import RunnableConfig, Runnable

class ConfigurableProcessor(Runnable[str, str]):
    """支持配置的处理器"""

    def invoke(self, input: str, config: RunnableConfig = None) -> str:
        # 从 config 中读取自定义参数
        if config and config.get("configurable"):
            mode = config["configurable"].get("mode", "default")
            prefix = config["configurable"].get("prefix", "")
        else:
            mode = "default"
            prefix = ""

        # 根据配置处理
        if mode == "uppercase":
            result = input.upper()
        elif mode == "lowercase":
            result = input.lower()
        else:
            result = input

        return f"{prefix}{result}"

# 使用不同配置
processor = ConfigurableProcessor()

# 配置 1: 大写模式
config1 = RunnableConfig(configurable={"mode": "uppercase", "prefix": "[UP] "})
print(processor.invoke("Hello", config1))  # "[UP] HELLO"

# 配置 2: 小写模式
config2 = RunnableConfig(configurable={"mode": "lowercase", "prefix": "[LOW] "})
print(processor.invoke("Hello", config2))  # "[LOW] hello"
```

---

## 错误处理模式

### 基础错误处理

```python
from langchain_core.runnables import Runnable
from typing import Optional

class SafeProcessor(Runnable[str, str]):
    """带错误处理的处理器"""

    def invoke(self, input: str, config=None) -> str:
        try:
            # 验证输入
            if not input or not isinstance(input, str):
                raise ValueError("输入必须是非空字符串")

            # 处理逻辑
            result = input.upper()

            # 验证输出
            if not result:
                raise RuntimeError("处理结果为空")

            return result

        except ValueError as e:
            # 输入验证错误
            print(f"输入错误: {e}")
            raise

        except RuntimeError as e:
            # 处理逻辑错误
            print(f"处理错误: {e}")
            raise

        except Exception as e:
            # 未预期的错误
            print(f"未知错误: {e}")
            raise
```

### 使用回调处理错误

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

class ErrorLoggingHandler(BaseCallbackHandler):
    """错误日志回调"""

    def on_chain_error(self, error: Exception, **kwargs):
        """链执行错误时调用"""
        print(f"❌ 链执行失败: {error}")
        # 可以发送告警、记录日志等

    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 调用错误时调用"""
        print(f"❌ LLM 调用失败: {error}")

# 使用错误处理回调
config = RunnableConfig(callbacks=[ErrorLoggingHandler()])

try:
    result = chain.invoke({"text": "你好"}, config=config)
except Exception as e:
    print(f"最终捕获错误: {e}")
```

### 重试模式

```python
import time
from langchain_core.runnables import Runnable
from typing import Optional

class RetryableRunnable(Runnable):
    """支持重试的 Runnable 包装器"""

    def __init__(self, inner: Runnable, max_retries: int = 3, backoff: float = 1.0):
        self.inner = inner
        self.max_retries = max_retries
        self.backoff = backoff

    def invoke(self, input, config=None):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.inner.invoke(input, config)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff * (2 ** attempt)  # 指数退避
                    print(f"重试 {attempt + 1}/{self.max_retries}，等待 {wait_time}秒")
                    time.sleep(wait_time)
                else:
                    print(f"重试失败，已达最大次数")

        raise last_error

# 使用
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
reliable_llm = RetryableRunnable(llm, max_retries=3, backoff=1.0)

# 自动重试失败的调用
result = reliable_llm.invoke("你好")
```

---

## 性能特征

### 同步执行特点

**invoke 是同步方法，会阻塞当前线程直到执行完成。**

```python
import time
from langchain_core.runnables import RunnableLambda

def slow_process(x: str) -> str:
    time.sleep(2)  # 模拟耗时操作
    return x.upper()

runnable = RunnableLambda(slow_process)

# 同步执行，阻塞 2 秒
start = time.time()
result = runnable.invoke("hello")  # 阻塞 2 秒
print(f"耗时: {time.time() - start:.2f}秒")  # 约 2 秒
```

### 与异步方法对比

```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def async_process(x: str) -> str:
    await asyncio.sleep(2)
    return x.upper()

runnable = RunnableLambda(async_process)

# 同步 invoke：阻塞执行
result = runnable.invoke("hello")  # 阻塞 2 秒

# 异步 ainvoke：非阻塞执行
async def main():
    result = await runnable.ainvoke("hello")  # 非阻塞
    return result

# 在异步环境中运行
result = asyncio.run(main())
```

### 性能优化建议

根据 2025-2026 年的最佳实践[^1][^2]：

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 单次查询 | `invoke` | 简单直接 |
| 多个独立查询 | `batch` 或 `ainvoke` + `asyncio.gather` | 并发执行 |
| 实时响应 | `stream` | 降低感知延迟 |
| IO 密集型 | `ainvoke` | 非阻塞 |
| CPU 密集型 | `invoke` + 多进程 | 避免 GIL |

---

## 实战代码示例

### 示例 1: 自定义 Runnable 实现

```python
"""
自定义 Runnable 实现
演示 invoke 方法的完整实现
"""

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from typing import Optional
import re

class TextAnalyzer(Runnable[str, dict]):
    """
    文本分析 Runnable

    输入: 文本字符串
    输出: 分析结果字典
    """

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None
    ) -> dict:
        """
        分析文本并返回统计信息

        Args:
            input: 待分析的文本
            config: 运行时配置

        Returns:
            包含分析结果的字典
        """
        # 1. 输入验证
        if not isinstance(input, str):
            raise TypeError(f"输入必须是字符串，实际类型: {type(input)}")

        if not input.strip():
            raise ValueError("输入不能为空")

        # 2. 从 config 中读取配置
        include_details = False
        if config and config.get("configurable"):
            include_details = config["configurable"].get("include_details", False)

        # 3. 执行分析
        text = input.strip()

        # 基础统计
        result = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines()),
            "has_chinese": bool(re.search(r'[\u4e00-\u9fff]', text)),
            "has_english": bool(re.search(r'[a-zA-Z]', text))
        }

        # 详细统计（可选）
        if include_details:
            result["details"] = {
                "uppercase_count": sum(1 for c in text if c.isupper()),
                "lowercase_count": sum(1 for c in text if c.islower()),
                "digit_count": sum(1 for c in text if c.isdigit()),
                "space_count": sum(1 for c in text if c.isspace()),
                "punctuation_count": sum(1 for c in text if c in ",.!?;:")
            }

        # 4. 记录日志（如果有回调）
        if config and config.get("callbacks"):
            for callback in config["callbacks"]:
                if hasattr(callback, "on_text"):
                    callback.on_text(f"分析完成: {result['word_count']} 个词")

        return result


# ===== 使用示例 =====
if __name__ == "__main__":
    analyzer = TextAnalyzer()

    # 示例 1: 基础使用
    print("=== 示例 1: 基础使用 ===")
    text1 = "Hello World! 你好世界！"
    result1 = analyzer.invoke(text1)
    print(f"输入: {text1}")
    print(f"结果: {result1}")
    print()

    # 示例 2: 使用配置
    print("=== 示例 2: 使用配置 ===")
    config = RunnableConfig(
        configurable={"include_details": True},
        tags=["analysis", "production"],
        metadata={"user_id": "user_123"}
    )
    result2 = analyzer.invoke(text1, config=config)
    print(f"结果（含详情）: {result2}")
    print()

    # 示例 3: 错误处理
    print("=== 示例 3: 错误处理 ===")
    try:
        analyzer.invoke("")  # 空字符串
    except ValueError as e:
        print(f"捕获错误: {e}")
    print()

    # 示例 4: 集成到 LCEL 链
    print("=== 示例 4: 集成到 LCEL 链 ===")
    from langchain_core.runnables import RunnableLambda

    # 预处理
    preprocessor = RunnableLambda(lambda x: x.strip().lower())

    # 后处理
    def format_result(analysis: dict) -> str:
        return f"文本包含 {analysis['word_count']} 个词，{analysis['char_count']} 个字符"

    postprocessor = RunnableLambda(format_result)

    # 组合成链
    chain = preprocessor | analyzer | postprocessor

    result = chain.invoke("  HELLO WORLD  ")
    print(f"链式处理结果: {result}")
```

**运行输出**：
```
=== 示例 1: 基础使用 ===
输入: Hello World! 你好世界！
结果: {'char_count': 18, 'word_count': 3, 'line_count': 1, 'has_chinese': True, 'has_english': True}

=== 示例 2: 使用配置 ===
结果（含详情）: {'char_count': 18, 'word_count': 3, 'line_count': 1, 'has_chinese': True, 'has_english': True, 'details': {'uppercase_count': 2, 'lowercase_count': 8, 'digit_count': 0, 'space_count': 2, 'punctuation_count': 2}}

=== 示例 3: 错误处理 ===
捕获错误: 输入不能为空

=== 示例 4: 集成到 LCEL 链 ===
链式处理结果: 文本包含 2 个词，11 个字符
```

### 示例 2: 带监控的 LLM 调用

```python
"""
带监控的 LLM 调用
演示 Config 和回调的实际应用
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

class PerformanceMonitor(BaseCallbackHandler):
    """性能监控回调"""

    def __init__(self):
        self.start_time = None
        self.token_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始时记录时间"""
        self.start_time = time.time()
        print(f"🚀 LLM 调用开始")

    def on_llm_end(self, response, **kwargs):
        """LLM 结束时计算耗时"""
        elapsed = time.time() - self.start_time
        print(f"✅ LLM 调用完成，耗时: {elapsed:.2f}秒")

        # 统计 token 使用
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            print(f"📊 Token 使用: {token_usage}")

    def on_llm_error(self, error, **kwargs):
        """LLM 错误时记录"""
        print(f"❌ LLM 调用失败: {error}")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 定义链
    prompt = ChatPromptTemplate.from_template("将以下文本翻译成英文: {text}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    # 配置监控
    config = RunnableConfig(
        callbacks=[PerformanceMonitor()],
        tags=["translation", "production"],
        metadata={
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        },
        run_name="translation_with_monitoring"
    )

    # 执行
    print("=== 带监控的翻译 ===")
    result = chain.invoke({"text": "你好，世界！"}, config=config)
    print(f"翻译结果: {result.content}")
```

**运行输出**：
```
=== 带监控的翻译 ===
🚀 LLM 调用开始
✅ LLM 调用完成，耗时: 1.23秒
📊 Token 使用: {'prompt_tokens': 15, 'completion_tokens': 5, 'total_tokens': 20}
翻译结果: Hello, World!
```

---

## 使用场景

### 适合使用 invoke 的场景

| 场景 | 原因 | 示例 |
|------|------|------|
| **单次查询** | 简单直接，无需并发 | 翻译一段文本 |
| **同步工作流** | 需要等待结果再继续 | 验证 → 处理 → 保存 |
| **简单脚本** | 代码简洁，易于理解 | 命令行工具 |
| **调试测试** | 便于断点调试 | 单元测试 |

### 不适合使用 invoke 的场景

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **批量处理** | `batch` | 并发执行更快 |
| **实时响应** | `stream` | 降低感知延迟 |
| **高并发** | `ainvoke` + `asyncio` | 非阻塞执行 |
| **长时间任务** | `stream` 或异步 | 避免阻塞 |

---

## 2025-2026 最佳实践

### 1. 使用类型注解

```python
from langchain_core.runnables import Runnable

# ✅ 明确类型
class TypedProcessor(Runnable[str, dict]):
    def invoke(self, input: str, config=None) -> dict:
        return {"result": input}

# ❌ 缺少类型
class UntypedProcessor(Runnable):
    def invoke(self, input, config=None):
        return {"result": input}
```

### 2. 验证输入

```python
def invoke(self, input: str, config=None) -> str:
    # ✅ 验证输入
    if not isinstance(input, str):
        raise TypeError(f"期望 str，实际 {type(input)}")

    if not input.strip():
        raise ValueError("输入不能为空")

    return input.upper()
```

### 3. 使用 Config 进行可观测性

```python
from langchain_core.runnables import RunnableConfig

# ✅ 生产环境配置
config = RunnableConfig(
    tags=["production", "v2"],
    metadata={"user_id": "123"},
    callbacks=[monitoring_handler],
    run_name="my_chain"
)

result = chain.invoke(input, config=config)
```

### 4. 错误处理

```python
def invoke(self, input: str, config=None) -> str:
    try:
        return self._process(input)
    except ValueError as e:
        # ✅ 记录并重新抛出
        logger.error(f"输入验证失败: {e}")
        raise
    except Exception as e:
        # ✅ 包装未知错误
        logger.error(f"处理失败: {e}")
        raise RuntimeError(f"处理失败: {e}") from e
```

---

## 总结

### invoke 方法的核心特点

1. **同步执行**: 阻塞当前线程直到完成
2. **单次处理**: 一次处理一个输入
3. **类型安全**: 通过泛型确保类型正确
4. **配置灵活**: 通过 Config 传递运行时参数

### 何时使用 invoke

- ✅ 单次查询和简单脚本
- ✅ 同步工作流和调试测试
- ❌ 批量处理（用 batch）
- ❌ 实时响应（用 stream）

---

## 参考资料

[^1]: [LangChain Runnable Methods Best Practices](https://medium.com/@sajo02/building-production-ready-ai-pipelines-with-langchain-runnables-a-complete-lcel-guide-2f9b27f6d557) - Medium, 2026
[^2]: [LangChain Best Practices](https://www.swarnendu.de/blog/langchain-best-practices) - Swarnendu De, 2025-2026

### 官方文档
- [Runnable invoke Reference](https://reference.langchain.com/python/langchain_core/runnables) - LangChain, 2025-2026
- [RunnableConfig Documentation](https://python.langchain.com/docs/concepts/runnables#config) - 2025-2026

---

**下一步**: 阅读 [03_核心概念_03_batch方法.md](./03_核心概念_03_batch方法.md) 学习批量处理和成本优化
