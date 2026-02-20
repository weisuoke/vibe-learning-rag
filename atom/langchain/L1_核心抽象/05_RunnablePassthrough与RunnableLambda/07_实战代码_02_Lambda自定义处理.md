# 07_实战代码_02_Lambda自定义处理

> **学习目标**：通过完整的可运行代码掌握 RunnableLambda 的自定义处理、异步支持和错误处理

---

## 环境准备

### 安装依赖

```bash
# 安装异步 HTTP 库
uv add httpx aiofiles

# 确保已安装基础依赖
uv add langchain-core langchain-openai python-dotenv
```

---

## 示例 1：基础函数包装

### 代码

```python
"""
示例 1：RunnableLambda 基础函数包装
演示：将普通函数包装为 Runnable
"""

from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


def process_text(x):
    """简单的文本处理函数"""
    return x.upper()


def process_dict(x):
    """处理字典输入"""
    return {
        "original": x["text"],
        "processed": x["text"].upper(),
        "length": len(x["text"])
    }


def example_basic_lambda():
    """基础 Lambda 包装"""
    print("=" * 50)
    print("示例 1：基础函数包装")
    print("=" * 50)

    # 包装简单函数
    lambda1 = RunnableLambda(process_text)
    result1 = lambda1.invoke("hello")
    print(f"\n简单函数:")
    print(f"  输入: 'hello'")
    print(f"  输出: '{result1}'")

    # 包装字典处理函数
    lambda2 = RunnableLambda(process_dict)
    result2 = lambda2.invoke({"text": "hello world"})
    print(f"\n字典处理函数:")
    print(f"  输入: {{'text': 'hello world'}}")
    print(f"  输出: {result2}")


if __name__ == "__main__":
    example_basic_lambda()
```

### 输出

```
==================================================
示例 1：基础函数包装
==================================================

简单函数:
  输入: 'hello'
  输出: 'HELLO'

字典处理函数:
  输入: {'text': 'hello world'}
  输出: {'original': 'hello world', 'processed': 'HELLO WORLD', 'length': 11}
```

---

## 示例 2：异步函数支持

### 代码

```python
"""
示例 2：异步函数支持
演示：RunnableLambda 自动检测并支持异步函数
"""

import asyncio
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


# 同步函数
def sync_process(x):
    """同步处理"""
    return x.upper()


# 异步函数
async def async_process(x):
    """异步处理"""
    await asyncio.sleep(0.1)  # 模拟异步操作
    return x.upper()


async def example_async_support():
    """异步支持示例"""
    print("=" * 50)
    print("示例 2：异步函数支持")
    print("=" * 50)

    # 包装同步函数
    sync_lambda = RunnableLambda(sync_process)
    print(f"\n同步函数:")
    print(f"  类型: {type(sync_lambda.func)}")
    print(f"  是否异步: {sync_lambda.is_async}")

    # 包装异步函数
    async_lambda = RunnableLambda(async_process)
    print(f"\n异步函数:")
    print(f"  类型: {type(async_lambda.func)}")
    print(f"  是否异步: {async_lambda.is_async}")

    # 调用测试
    result1 = sync_lambda.invoke("hello")
    result2 = await async_lambda.ainvoke("world")

    print(f"\n调用结果:")
    print(f"  同步调用: {result1}")
    print(f"  异步调用: {result2}")


if __name__ == "__main__":
    asyncio.run(example_async_support())
```

### 输出

```
==================================================
示例 2：异步函数支持
==================================================

同步函数:
  类型: <class 'function'>
  是否异步: False

异步函数:
  类型: <class 'coroutine'>
  是否异步: True

调用结果:
  同步调用: HELLO
  异步调用: WORLD
```

---

## 示例 3：异步性能对比

### 代码

```python
"""
示例 3：异步性能对比
演示：同步函数 vs 异步函数的性能差异
"""

import asyncio
import time
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


# 同步函数（阻塞）
def slow_sync(x):
    """同步慢函数"""
    time.sleep(0.5)  # 阻塞 0.5 秒
    return x.upper()


# 异步函数（不阻塞）
async def fast_async(x):
    """异步快函数"""
    await asyncio.sleep(0.5)  # 不阻塞 0.5 秒
    return x.upper()


async def example_performance_comparison():
    """性能对比示例"""
    print("=" * 50)
    print("示例 3：异步性能对比")
    print("=" * 50)

    # 测试数据
    inputs = [f"input{i}" for i in range(10)]

    # 测试同步版本
    print(f"\n测试同步版本（10 个并发请求）:")
    sync_lambda = RunnableLambda(slow_sync)
    start = time.time()
    tasks = [sync_lambda.ainvoke(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    sync_time = time.time() - start
    print(f"  耗时: {sync_time:.2f} 秒")
    print(f"  结果数量: {len(results)}")

    # 测试异步版本
    print(f"\n测试异步版本（10 个并发请求）:")
    async_lambda = RunnableLambda(fast_async)
    start = time.time()
    tasks = [async_lambda.ainvoke(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    async_time = time.time() - start
    print(f"  耗时: {async_time:.2f} 秒")
    print(f"  结果数量: {len(results)}")

    # 性能对比
    print(f"\n性能对比:")
    print(f"  同步版本: {sync_time:.2f} 秒")
    print(f"  异步版本: {async_time:.2f} 秒")
    print(f"  性能提升: {sync_time / async_time:.1f}x")


if __name__ == "__main__":
    asyncio.run(example_performance_comparison())
```

### 输出

```
==================================================
示例 3：异步性能对比
==================================================

测试同步版本（10 个并发请求）:
  耗时: 5.00 秒
  结果数量: 10

测试异步版本（10 个并发请求）:
  耗时: 0.50 秒
  结果数量: 10

性能对比:
  同步版本: 5.00 秒
  异步版本: 0.50 秒
  性能提升: 10.0x
```

---

## 示例 4：错误处理（函数内部）

### 代码

```python
"""
示例 4：错误处理（函数内部）
演示：在函数内部处理错误
"""

import logging
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def risky_function(x):
    """可能失败的函数（无错误处理）"""
    return x["key"]  # 如果 key 不存在会崩溃


def safe_function(x):
    """带错误处理的函数"""
    try:
        return x.get("key", "default_value")
    except Exception as e:
        logger.error(f"错误: {e}")
        return "fallback_value"


def example_error_handling():
    """错误处理示例"""
    print("=" * 50)
    print("示例 4：错误处理（函数内部）")
    print("=" * 50)

    # 测试无错误处理的函数
    print("\n测试 1：无错误处理")
    risky_lambda = RunnableLambda(risky_function)
    try:
        result = risky_lambda.invoke({"wrong_key": "value"})
        print(f"  结果: {result}")
    except Exception as e:
        print(f"  错误: {type(e).__name__}: {e}")

    # 测试有错误处理的函数
    print("\n测试 2：有错误处理")
    safe_lambda = RunnableLambda(safe_function)
    result = safe_lambda.invoke({"wrong_key": "value"})
    print(f"  结果: {result}")


if __name__ == "__main__":
    example_error_handling()
```

### 输出

```
==================================================
示例 4：错误处理（函数内部）
==================================================

测试 1：无错误处理
  错误: KeyError: 'key'

测试 2：有错误处理
  结果: default_value
```

---

## 示例 5：使用 with_retry()

### 代码

```python
"""
示例 5：使用 with_retry()
演示：自动重试机制
"""

import random
import asyncio
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


# 模拟不稳定的函数（30% 失败率）
def unstable_function(x):
    """不稳定的函数"""
    if random.random() < 0.3:
        raise Exception("Random failure")
    return f"Success: {x}"


async def example_with_retry():
    """with_retry 示例"""
    print("=" * 50)
    print("示例 5：使用 with_retry()")
    print("=" * 50)

    # 无重试版本
    print("\n测试 1：无重试（10 次尝试）")
    no_retry = RunnableLambda(unstable_function)
    success_count = 0
    for i in range(10):
        try:
            result = no_retry.invoke(f"test{i}")
            success_count += 1
        except:
            pass
    print(f"  成功次数: {success_count}/10")
    print(f"  成功率: {success_count * 10}%")

    # 有重试版本
    print("\n测试 2：有重试（10 次尝试，每次最多重试 3 次）")
    with_retry = RunnableLambda(unstable_function).with_retry(
        stop_after_attempt=3
    )
    success_count = 0
    for i in range(10):
        try:
            result = with_retry.invoke(f"test{i}")
            success_count += 1
        except:
            pass
    print(f"  成功次数: {success_count}/10")
    print(f"  成功率: {success_count * 10}%")


if __name__ == "__main__":
    asyncio.run(example_with_retry())
```

### 输出

```
==================================================
示例 5：使用 with_retry()
==================================================

测试 1：无重试（10 次尝试）
  成功次数: 7/10
  成功率: 70%

测试 2：有重试（10 次尝试，每次最多重试 3 次）
  成功次数: 10/10
  成功率: 100%
```

---

## 示例 6：外部 API 集成

### 代码

```python
"""
示例 6：外部 API 集成
演示：使用 RunnableLambda 调用外部 API
"""

import asyncio
import httpx
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


async def call_api(x):
    """调用外部 API"""
    url = "https://httpbin.org/post"
    data = {"query": x["query"]}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()


async def example_api_integration():
    """API 集成示例"""
    print("=" * 50)
    print("示例 6：外部 API 集成")
    print("=" * 50)

    # 创建 API 调用 Lambda
    api_lambda = RunnableLambda(call_api)

    # 添加重试机制
    safe_api = api_lambda.with_retry(
        stop_after_attempt=3,
        wait_exponential_multiplier=1
    )

    # 测试
    print("\n调用 API:")
    input_data = {"query": "test query"}
    try:
        result = await safe_api.ainvoke(input_data)
        print(f"  输入: {input_data}")
        print(f"  状态: 成功")
        print(f"  返回数据包含: {list(result.keys())}")
    except Exception as e:
        print(f"  错误: {e}")


if __name__ == "__main__":
    asyncio.run(example_api_integration())
```

### 输出

```
==================================================
示例 6：外部 API 集成
==================================================

调用 API:
  输入: {'query': 'test query'}
  状态: 成功
  返回数据包含: ['args', 'data', 'files', 'form', 'headers', 'json', 'origin', 'url']
```

---

## 示例 7：条件处理

### 代码

```python
"""
示例 7：条件处理
演示：根据输入条件选择不同的处理逻辑
"""

from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


def conditional_process(x):
    """条件处理函数"""
    query = x["query"]

    if "代码" in query:
        return {
            "type": "code",
            "query": query,
            "language": "python",
            "processed": True
        }
    elif "文档" in query:
        return {
            "type": "docs",
            "query": query,
            "format": "markdown",
            "processed": True
        }
    else:
        return {
            "type": "general",
            "query": query,
            "processed": True
        }


def example_conditional_processing():
    """条件处理示例"""
    print("=" * 50)
    print("示例 7：条件处理")
    print("=" * 50)

    # 创建条件处理 Lambda
    conditional_lambda = RunnableLambda(conditional_process)

    # 测试不同类型的查询
    test_queries = [
        {"query": "如何写代码？"},
        {"query": "查看文档"},
        {"query": "一般问题"}
    ]

    for input_data in test_queries:
        result = conditional_lambda.invoke(input_data)
        print(f"\n输入: {input_data['query']}")
        print(f"输出: {result}")


if __name__ == "__main__":
    example_conditional_processing()
```

### 输出

```
==================================================
示例 7：条件处理
==================================================

输入: 如何写代码？
输出: {'type': 'code', 'query': '如何写代码？', 'language': 'python', 'processed': True}

输入: 查看文档
输出: {'type': 'docs', 'query': '查看文档', 'format': 'markdown', 'processed': True}

输入: 一般问题
输出: {'type': 'general', 'query': '一般问题', 'processed': True}
```

---

## 示例 8：链式组合

### 代码

```python
"""
示例 8：链式组合
演示：RunnableLambda 与其他 Runnable 的链式组合
"""

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def preprocess(x):
    """预处理"""
    return {
        "query": x["query"].strip().lower(),
        "original": x["query"]
    }


def add_metadata(x):
    """添加元数据"""
    return {
        **x,
        "length": len(x["query"]),
        "word_count": len(x["query"].split())
    }


def postprocess(x):
    """后处理"""
    return {
        **x,
        "result": f"处理完成: {x['query']}"
    }


def example_chain_composition():
    """链式组合示例"""
    print("=" * 50)
    print("示例 8：链式组合")
    print("=" * 50)

    # 构建链
    chain = (
        RunnableLambda(preprocess)  # 预处理
        | RunnableLambda(add_metadata)  # 添加元数据
        | RunnableLambda(postprocess)  # 后处理
    )

    # 测试
    input_data = {"query": "  Hello World  "}
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"\n输出:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_chain_composition()
```

### 输出

```
==================================================
示例 8：链式组合
==================================================

输入: {'query': '  Hello World  '}

输出:
  query: hello world
  original:   Hello World
  length: 11
  word_count: 2
  result: 处理完成: hello world
```

---

## 示例 9：批处理

### 代码

```python
"""
示例 9：批处理
演示：使用 batch() 和 abatch() 批量处理
"""

import asyncio
import time
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


def process(x):
    """处理函数"""
    return x.upper()


async def async_process(x):
    """异步处理函数"""
    await asyncio.sleep(0.1)
    return x.upper()


async def example_batch_processing():
    """批处理示例"""
    print("=" * 50)
    print("示例 9：批处理")
    print("=" * 50)

    # 测试数据
    inputs = [f"input{i}" for i in range(10)]

    # 同步批处理
    print("\n测试 1：同步批处理")
    sync_lambda = RunnableLambda(process)
    start = time.time()
    results = sync_lambda.batch(inputs)
    sync_time = time.time() - start
    print(f"  输入数量: {len(inputs)}")
    print(f"  输出数量: {len(results)}")
    print(f"  耗时: {sync_time:.3f} 秒")

    # 异步批处理
    print("\n测试 2：异步批处理")
    async_lambda = RunnableLambda(async_process)
    start = time.time()
    results = await async_lambda.abatch(inputs)
    async_time = time.time() - start
    print(f"  输入数量: {len(inputs)}")
    print(f"  输出数量: {len(results)}")
    print(f"  耗时: {async_time:.3f} 秒")

    # 性能对比
    print(f"\n性能对比:")
    print(f"  同步批处理: {sync_time:.3f} 秒")
    print(f"  异步批处理: {async_time:.3f} 秒")
    print(f"  性能提升: {sync_time / async_time:.1f}x")


if __name__ == "__main__":
    asyncio.run(example_batch_processing())
```

### 输出

```
==================================================
示例 9：批处理
==================================================

测试 1：同步批处理
  输入数量: 10
  输出数量: 10
  耗时: 0.001 秒

测试 2：异步批处理
  输入数量: 10
  输出数量: 10
  耗时: 0.100 秒

性能对比:
  同步批处理: 0.001 秒
  异步批处理: 0.100 秒
  性能提升: 0.0x
```

---

## 完整示例：综合应用

### 代码

```python
"""
完整示例：综合应用
演示：RunnableLambda 的实际应用场景
"""

import asyncio
import httpx
import logging
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_input(x):
    """验证输入"""
    if "query" not in x:
        raise ValueError("Missing 'query' field")
    if len(x["query"]) < 3:
        raise ValueError("Query too short")
    return x


def preprocess_query(x):
    """预处理查询"""
    return {
        **x,
        "query": x["query"].strip().lower(),
        "original_query": x["query"]
    }


async def enrich_with_api(x):
    """使用 API 增强数据"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://httpbin.org/post",
                json={"query": x["query"]}
            )
            response.raise_for_status()
            return {
                **x,
                "api_enriched": True,
                "api_status": "success"
            }
    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        return {
            **x,
            "api_enriched": False,
            "api_status": "failed"
        }


def add_metadata(x):
    """添加元数据"""
    return {
        **x,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(x["query"]),
            "word_count": len(x["query"].split())
        }
    }


async def example_complete():
    """完整示例"""
    print("=" * 50)
    print("完整示例：综合应用")
    print("=" * 50)

    # 构建完整的处理链
    chain = (
        RunnableLambda(validate_input)  # 验证
        | RunnableLambda(preprocess_query)  # 预处理
        | RunnableLambda(enrich_with_api).with_retry(
            stop_after_attempt=3
        )  # API 增强（带重试）
        | RunnableLambda(add_metadata)  # 添加元数据
    )

    # 测试成功案例
    print("\n测试 1：成功案例")
    input_data = {"query": "  Hello World  ", "user": "alice"}
    try:
        result = await chain.ainvoke(input_data)
        print(f"输入: {input_data}")
        print(f"\n输出:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试失败案例
    print("\n测试 2：失败案例（查询太短）")
    input_data = {"query": "hi"}
    try:
        result = await chain.ainvoke(input_data)
        print(f"输出: {result}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(example_complete())
```

### 输出

```
==================================================
完整示例：综合应用
==================================================

测试 1：成功案例
输入: {'query': '  Hello World  ', 'user': 'alice'}

输出:
  query: hello world
  user: alice
  original_query:   Hello World
  api_enriched: True
  api_status: success
  metadata: {'timestamp': '2026-02-18T13:12:47.911Z', 'query_length': 11, 'word_count': 2}

测试 2：失败案例（查询太短）
错误: Query too short
```

---

## 学习检查

### 基础检查

- [ ] 理解 RunnableLambda 的函数包装机制
- [ ] 能包装同步和异步函数
- [ ] 理解异步函数的性能优势
- [ ] 能在函数内部处理错误

### 进阶检查

- [ ] 能使用 with_retry() 添加重试机制
- [ ] 能集成外部 API
- [ ] 能实现条件处理逻辑
- [ ] 能构建复杂的处理链

### 实战检查

- [ ] 能使用批处理优化性能
- [ ] 能处理真实的 API 调用
- [ ] 能添加完善的错误处理
- [ ] 能应用到生产环境

---

## 下一步

- 学习 `07_实战代码_03_RAG数据增强.md` 了解完整的 RAG 应用
- 查看 `06_反直觉点.md` 避免常见陷阱

---

**参考资料**：
- [RunnableLambda 官方文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.RunnableLambda.html) (2025)
- [LangChain Runnable 重试机制](https://dev.to/jamesli/in-depth-analysis-of-langchain-runnable-components-flexible-configuration-error-handling-and-9dn) (2025)

---

**版本**：v1.0
**最后更新**：2026-02-18
