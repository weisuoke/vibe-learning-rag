# 03_核心概念_02_RunnableLambda自定义函数

> **学习目标**：深入理解 RunnableLambda 的函数包装机制、同步/异步支持、错误处理和类型系统

---

## 概述

RunnableLambda 是 LangChain LCEL 中的**适配器**（Adapter），它将任意 Python 函数转换为标准的 Runnable 接口，使得自定义逻辑可以无缝集成到 LCEL 链中。

**核心价值**：
- 将普通函数包装为 Runnable
- 自动检测同步/异步函数
- 提供统一的调用接口
- 支持错误处理和重试机制

---

## 函数包装机制

### 适配器模式

**设计模式**：Adapter Pattern

```
普通函数（不兼容的接口）
    ↓
RunnableLambda（适配器）
    ↓
Runnable 接口（统一接口）
```

**核心思想**：
- 不修改原函数
- 提供统一接口
- 保持函数行为不变

### 基础用法

```python
from langchain_core.runnables import RunnableLambda

# 定义普通函数
def process_text(x):
    return x.upper()

# 包装为 Runnable
lambda_runnable = RunnableLambda(process_text)

# 使用
result = lambda_runnable.invoke("hello")  # 输出: "HELLO"

# 可以链式组合
chain = lambda_runnable | other_runnable
```

---

## 同步与异步支持

### 自动检测机制

RunnableLambda 会自动检测函数是同步还是异步：

```python
import asyncio
from langchain_core.runnables import RunnableLambda

# 同步函数
def sync_func(x):
    return x.upper()

# 异步函数
async def async_func(x):
    await asyncio.sleep(0.1)
    return x.upper()

# 自动检测
sync_runnable = RunnableLambda(sync_func)    # 检测为同步
async_runnable = RunnableLambda(async_func)  # 检测为异步
```

### 调用方式

```python
# 同步调用
result = sync_runnable.invoke("hello")

# 异步调用
result = await async_runnable.ainvoke("hello")

# 批处理
results = sync_runnable.batch(["hello", "world"])
results = await async_runnable.abatch(["hello", "world"])
```

### 性能对比

**同步函数在异步链中的问题**：

```python
import time
import asyncio

# ❌ 同步函数阻塞
def slow_sync(x):
    time.sleep(1)  # 阻塞整个事件循环
    return x

# 并发测试
async def test_sync():
    chain = RunnableLambda(slow_sync)
    tasks = [chain.ainvoke(f"req{i}") for i in range(10)]
    start = time.time()
    await asyncio.gather(*tasks)
    print(f"同步版本耗时: {time.time() - start:.2f}秒")
    # 输出: 10.00秒（串行执行）

# ✅ 异步函数不阻塞
async def fast_async(x):
    await asyncio.sleep(1)  # 不阻塞事件循环
    return x

async def test_async():
    chain = RunnableLambda(fast_async)
    tasks = [chain.ainvoke(f"req{i}") for i in range(10)]
    start = time.time()
    await asyncio.gather(*tasks)
    print(f"异步版本耗时: {time.time() - start:.2f}秒")
    # 输出: 1.00秒（并行执行）
```

**性能提升**：10 倍

---

## 错误处理与重试

### 方式 1：函数内部处理

```python
import logging
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

def safe_func(x):
    """带错误处理的函数"""
    try:
        # 可能失败的操作
        result = risky_operation(x)
        return result
    except ValueError as e:
        logger.error(f"值错误: {e}")
        return {"error": "invalid_value", "fallback": "default"}
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return {"error": "unknown", "fallback": "default"}

# 使用
chain = RunnableLambda(safe_func) | llm
```

### 方式 2：使用 with_retry()

```python
import httpx
from langchain_core.runnables import RunnableLambda

async def risky_api_call(x):
    """可能失败的 API 调用"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()

# 添加重试机制
safe_call = RunnableLambda(risky_api_call).with_retry(
    stop_after_attempt=3,              # 最多重试 3 次
    wait_exponential_multiplier=1,     # 指数退避：1s, 2s, 4s
    wait_exponential_max=10,           # 最大等待 10 秒
    retry_if_exception_type=(          # 只重试特定异常
        httpx.TimeoutException,
        httpx.HTTPStatusError
    )
)

# 使用
chain = safe_call | llm
```

### 方式 3：使用 with_fallbacks()

```python
from langchain_core.runnables import RunnableLambda

# 主函数
async def primary_api(x):
    # 调用主 API
    return await call_primary_api(x)

# 降级函数
async def fallback_api(x):
    # 调用备用 API
    return await call_fallback_api(x)

# 组合：重试 + 降级
chain = (
    RunnableLambda(primary_api)
    .with_retry(stop_after_attempt=3)
    .with_fallbacks([RunnableLambda(fallback_api)])
)
```

### 重试策略对比

```python
import random

# 模拟不稳定的 API（30% 失败率）
async def unstable_api(x):
    if random.random() < 0.3:
        raise Exception("API failure")
    return "success"

# 测试：无重试 vs 有重试
async def benchmark():
    # 无重试
    no_retry = RunnableLambda(unstable_api)
    success = sum(1 for _ in range(100) if try_invoke(no_retry))
    print(f"无重试成功率: {success}%")  # 约 70%

    # 有重试（3 次）
    with_retry = RunnableLambda(unstable_api).with_retry(stop_after_attempt=3)
    success = sum(1 for _ in range(100) if try_invoke(with_retry))
    print(f"有重试成功率: {success}%")  # 约 97%
```

---

## 类型系统

### 类型签名

```python
from typing import TypeVar, Callable

Input = TypeVar("Input")
Output = TypeVar("Output")

class RunnableLambda(Runnable[Input, Output]):
    """
    类型签名：Runnable[Input, Output]
    输入输出类型由函数决定
    """

    def __init__(self, func: Callable[[Input], Output]):
        self.func = func
```

### 类型推断

```python
# 类型系统自动推断
def process(x: str) -> int:
    return len(x)

lambda_runnable: Runnable[str, int] = RunnableLambda(process)

# 链式组合的类型检查
chain = (
    RunnableLambda(lambda x: x.upper())  # str → str
    | RunnableLambda(lambda x: len(x))   # str → int
    | RunnableLambda(lambda x: x * 2)    # int → int
)

# 类型系统知道：
# - 输入类型：str
# - 输出类型：int
```

---

## 实现原理

### 简化的实现

```python
import asyncio
from typing import Callable, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

class RunnableLambda(Runnable[Input, Output]):
    """将函数包装为 Runnable"""

    def __init__(self, func: Callable[[Input], Output]):
        self.func = func
        # 自动检测是否为异步函数
        self.is_async = asyncio.iscoroutinefunction(func)

    def invoke(self, input: Input) -> Output:
        """同步调用"""
        if self.is_async:
            raise RuntimeError("Use ainvoke for async functions")
        return self.func(input)

    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""
        if self.is_async:
            return await self.func(input)
        else:
            # 同步函数在异步上下文中调用
            return self.func(input)

    def batch(self, inputs: list[Input]) -> list[Output]:
        """批量调用"""
        return [self.invoke(input) for input in inputs]

    async def abatch(self, inputs: list[Input]) -> list[Output]:
        """异步批量调用"""
        if self.is_async:
            return await asyncio.gather(*[self.ainvoke(input) for input in inputs])
        else:
            return [self.invoke(input) for input in inputs]
```

### 关键设计决策

**1. 为什么自动检测同步/异步？**
- 用户体验：不需要手动指定
- 减少错误：自动适配正确的调用方式
- 灵活性：同一个 Runnable 可以在同步和异步上下文中使用

**2. 为什么提供 invoke 和 ainvoke？**
- 统一接口：所有 Runnable 都有这两个方法
- 类型安全：编译时检查调用方式
- 性能优化：异步函数在异步上下文中不阻塞

---

## 常见使用模式

### 模式 1：数据预处理

```python
from langchain_core.runnables import RunnableLambda

def preprocess(input_dict):
    """预处理输入数据"""
    query = input_dict["query"]
    # 清理和标准化
    cleaned = query.lower().strip()
    return {"query": cleaned, "original": query}

# 集成到链中
chain = (
    RunnableLambda(preprocess)
    | RunnablePassthrough.assign(context=retriever)
    | llm
)
```

### 模式 2：数据后处理

```python
def postprocess(llm_output):
    """后处理 LLM 输出"""
    # 提取答案
    answer = llm_output.content.split("\n")[0]
    # 添加元数据
    return {
        "answer": answer,
        "confidence": calculate_confidence(llm_output),
        "sources": extract_sources(llm_output)
    }

chain = (
    prompt
    | llm
    | RunnableLambda(postprocess)
)
```

### 模式 3：条件路由

```python
def route_query(input_dict):
    """根据查询类型路由"""
    query = input_dict["query"]

    if "代码" in query:
        return {"type": "code", "query": query}
    elif "文档" in query:
        return {"type": "docs", "query": query}
    else:
        return {"type": "general", "query": query}

chain = (
    RunnableLambda(route_query)
    | RunnableBranch(
        (lambda x: x["type"] == "code", code_chain),
        (lambda x: x["type"] == "docs", docs_chain),
        general_chain
    )
)
```

### 模式 4：外部 API 集成

```python
import httpx

async def call_external_api(input_dict):
    """调用外部 API"""
    query = input_dict["query"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/process",
            json={"query": query}
        )
        response.raise_for_status()
        return {
            **input_dict,
            "api_result": response.json()
        }

# 添加重试
safe_api = RunnableLambda(call_external_api).with_retry(
    stop_after_attempt=3
)

chain = safe_api | llm
```

---

## 性能优化

### 1. 使用异步函数

```python
# ❌ 同步版本（阻塞）
import requests

def sync_fetch(url):
    return requests.get(url).json()

# ✅ 异步版本（不阻塞）
import httpx

async def async_fetch(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# 性能提升：10x+
```

### 2. 批处理优化

```python
# 批量处理
results = await chain.abatch([
    {"query": "q1"},
    {"query": "q2"},
    {"query": "q3"}
])

# 比逐个调用快 3-5 倍
```

### 3. 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_compute(query: str):
    # 昂贵的计算
    return result

chain = RunnableLambda(expensive_compute) | llm
```

---

## 2025-2026 最新特性

### 1. 智能重试（LangChain v0.3.18+）

```python
from langchain_core.runnables import RunnableLambda

# 自动识别可重试的错误
smart_chain = RunnableLambda(risky_func).with_retry(
    strategy="smart",  # 智能策略
    stop_after_attempt=3
)

# 智能策略会：
# 1. 自动识别临时性错误（网络超时、5xx 错误）
# 2. 跳过永久性错误（4xx 错误、逻辑错误）
# 3. 动态调整重试间隔
```

### 2. 类型推断增强

```python
# 自动类型推断
def process(x: str) -> int:
    return len(x)

lambda_runnable = RunnableLambda(process)
# 类型系统自动识别: Runnable[str, int]
```

### 3. 错误传播改进

```python
# 更清晰的错误堆栈
chain = RunnableLambda(risky_func).with_retry(
    stop_after_attempt=3
)
# 错误信息包含完整调用链路和重试历史
```

---

## 常见陷阱

### 陷阱 1：同步函数阻塞异步链

```python
# ❌ 错误
import time

def slow_sync(x):
    time.sleep(5)  # 阻塞！
    return x

chain = RunnableLambda(slow_sync) | llm
await chain.ainvoke("input")  # 会阻塞 5 秒

# ✅ 正确
import asyncio

async def fast_async(x):
    await asyncio.sleep(5)  # 不阻塞
    return x

chain = RunnableLambda(fast_async) | llm
await chain.ainvoke("input")  # 不会阻塞其他任务
```

### 陷阱 2：忘记错误处理

```python
# ❌ 错误
def risky_func(x):
    return x["key"]  # 如果 key 不存在会崩溃

# ✅ 正确
def safe_func(x):
    try:
        return x.get("key", "default")
    except Exception as e:
        logger.error(f"Error: {e}")
        return "fallback"
```

### 陷阱 3：重试所有错误

```python
# ❌ 错误：重试所有错误（包括 4xx）
chain = RunnableLambda(api_call).with_retry(
    stop_after_attempt=3
)

# ✅ 正确：只重试临时性错误
chain = RunnableLambda(api_call).with_retry(
    stop_after_attempt=3,
    retry_if_exception_type=(
        httpx.TimeoutException,
        httpx.NetworkError
    )
)
```

---

## 最佳实践

### 1. 分层错误处理

```python
# 函数内部：处理预期错误
def safe_func(x):
    try:
        return process(x)
    except ValueError:
        return fallback_value

# with_retry：处理临时性错误
safe_chain = RunnableLambda(safe_func).with_retry(
    stop_after_attempt=3
)

# with_fallbacks：处理完全失败
final_chain = safe_chain.with_fallbacks([
    RunnableLambda(ultimate_fallback)
])
```

### 2. 添加日志和监控

```python
import logging
import time

logger = logging.getLogger(__name__)

def monitored_func(x):
    start = time.time()
    try:
        result = process(x)
        logger.info(f"Success, time: {time.time() - start:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise
```

### 3. 使用类型注解

```python
from typing import Dict, List

def process(input_dict: Dict[str, str]) -> List[str]:
    """
    处理输入字典

    Args:
        input_dict: 包含 query 字段的字典

    Returns:
        处理后的字符串列表
    """
    return input_dict["query"].split()

chain = RunnableLambda(process)
```

---

## 总结

### 核心要点

1. **函数包装**：RunnableLambda 将任意函数转换为 Runnable
2. **自动检测**：自动识别同步/异步函数
3. **错误处理**：支持 with_retry() 和 with_fallbacks()
4. **类型系统**：完整的类型推断和检查

### 最佳实践

1. **使用异步**：在异步链中使用异步函数
2. **分层错误处理**：函数内部 + with_retry + with_fallbacks
3. **添加日志**：记录关键信息和性能指标
4. **类型注解**：提高代码可维护性

### 常见陷阱

1. ❌ 同步函数阻塞异步链
2. ❌ 忘记错误处理
3. ❌ 重试所有错误（包括永久性错误）

---

**下一步**：
- 学习 `03_核心概念_03_数据转换与组合.md` 了解组合模式
- 查看 `07_实战代码_02_Lambda自定义处理.md` 获取实战示例

---

**参考资料**：
- [RunnableLambda 官方文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.RunnableLambda.html) (2025)
- [RunnableRetry 官方文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.retry.RunnableRetry.html) (2025)
- [LangChain Runnable 重试机制深入分析](https://dev.to/jamesli/in-depth-analysis-of-langchain-runnable-components-flexible-configuration-error-handling-and-9dn) (2025)

---

**版本**：v1.0
**最后更新**：2026-02-18
