# 核心概念 03: batch 方法

> 理解 Runnable 的批量并发执行方法

---

## 什么是 batch 方法？

**batch 是 Runnable 协议的批量执行方法，用于并发处理多个输入并返回对应的输出列表。**

### 一句话定义

batch 方法接收一个输入列表和可选的配置参数，并发执行处理逻辑，返回一个输出列表。

---

## 方法签名

```python
from typing import TypeVar, Optional, List
from langchain_core.runnables.config import RunnableConfig

Input = TypeVar("Input")
Output = TypeVar("Output")

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
        输出数据列表，顺序与输入对应

    注意:
        - 默认使用线程池并行执行
        - 保证输出顺序与输入顺序一致
        - 单个任务失败会导致整个 batch 失败
    """
    ...
```

---

## 并发执行机制

### 默认实现：线程池

**LangChain 默认使用线程池并行执行 batch**[^1]：

```python
from langchain_core.runnables import RunnableLambda
import time

def slow_process(x: str) -> str:
    time.sleep(1)  # 模拟耗时操作
    return x.upper()

runnable = RunnableLambda(slow_process)

# 顺序执行（使用 invoke）
start = time.time()
results = [runnable.invoke(x) for x in ["a", "b", "c"]]
print(f"顺序执行耗时: {time.time() - start:.2f}秒")  # 约 3 秒

# 并发执行（使用 batch）
start = time.time()
results = runnable.batch(["a", "b", "c"])
print(f"并发执行耗时: {time.time() - start:.2f}秒")  # 约 1 秒
```

### 并发控制

```python
from langchain_core.runnables import RunnableConfig

# 限制并发数量
config = RunnableConfig(max_concurrency=2)

# 最多同时执行 2 个任务
results = runnable.batch(inputs, config=config)
```

---

## 成本优化：langasync 集成

### 2025-2026 年重大突破：50% 成本降低

**langasync 通过批处理 API 实现 50% 成本节省**[^2][^3]：

```python
from langasync import wrap_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 原始 LCEL 链
prompt = ChatPromptTemplate.from_template("分类: {text}")
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

# 包装为批处理模式
async_chain = wrap_chain(chain, batch_size=10)

# 批量执行（成本降低 50%）
inputs = [{"text": f"文本{i}"} for i in range(100)]
results = await async_chain.abatch(inputs)
```

### 成本对比

| 方法 | 成本 | 延迟 | 适用场景 |
|------|------|------|----------|
| **invoke 循环** | 100% | 实时 | 实时查询 |
| **batch** | 100% | 实时 | 实时批量 |
| **langasync + Batch API** | 50% | 延迟（分钟级） | 离线批量 |

### 适用场景

**✅ 推荐使用 langasync**：
- 批量评估和测试
- 数据标注任务
- 离线分析和报告
- 非实时批量处理

**❌ 不推荐使用 langasync**：
- 实时对话应用
- 需要即时响应的场景
- 单次查询

---

## 性能优化

### 批量大小选择

```python
# 小批量：低延迟，低吞吐
results = runnable.batch(inputs[:10])

# 大批量：高延迟，高吞吐
results = runnable.batch(inputs[:1000])

# 推荐：分批处理
batch_size = 50
for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i+batch_size]
    results = runnable.batch(batch)
    process_results(results)
```

### 异步批处理

```python
import asyncio
from langchain_core.runnables import RunnableLambda

async def async_process(x: str) -> str:
    await asyncio.sleep(1)
    return x.upper()

runnable = RunnableLambda(async_process)

# 异步批处理（更高效）
results = await runnable.abatch(["a", "b", "c"])
```

---

## 实战代码示例

### 示例 1: 批量文档分类

```python
"""
批量文档分类
演示 batch 方法的实际应用
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import time

# 定义分类链
prompt = ChatPromptTemplate.from_template(
    "将以下文本分类为：技术、商业、娱乐、体育\n\n文本: {text}\n\n分类:"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = prompt | llm | parser

# 测试数据
documents = [
    "Python 3.13 发布了新特性",
    "苹果公司发布财报",
    "电影《流浪地球3》上映",
    "NBA 总决赛开始",
    "机器学习算法优化",
]

# ===== 方式 1: 顺序执行 =====
print("=== 顺序执行 ===")
start = time.time()
results_sequential = []
for doc in documents:
    result = chain.invoke({"text": doc})
    results_sequential.append(result)
elapsed_sequential = time.time() - start
print(f"耗时: {elapsed_sequential:.2f}秒")
print(f"结果: {results_sequential}\n")

# ===== 方式 2: 批量执行 =====
print("=== 批量执行 ===")
start = time.time()
inputs = [{"text": doc} for doc in documents]
results_batch = chain.batch(inputs)
elapsed_batch = time.time() - start
print(f"耗时: {elapsed_batch:.2f}秒")
print(f"结果: {results_batch}")
print(f"加速比: {elapsed_sequential / elapsed_batch:.2f}x\n")

# ===== 方式 3: 带并发控制 =====
print("=== 带并发控制 ===")
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(max_concurrency=2)
start = time.time()
results_controlled = chain.batch(inputs, config=config)
elapsed_controlled = time.time() - start
print(f"耗时: {elapsed_controlled:.2f}秒")
print(f"结果: {results_controlled}")
```

### 示例 2: 成本追踪

```python
"""
批量处理成本追踪
演示如何监控 token 使用和成本
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

class CostTracker(BaseCallbackHandler):
    """成本追踪回调"""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束时统计 token"""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += token_usage.get('prompt_tokens', 0)
            self.completion_tokens += token_usage.get('completion_tokens', 0)
            self.total_tokens += token_usage.get('total_tokens', 0)

    def get_cost(self, model="gpt-4o-mini"):
        """计算成本（美元）"""
        # gpt-4o-mini 价格（2026年）
        prompt_cost = self.prompt_tokens * 0.00015 / 1000
        completion_cost = self.completion_tokens * 0.0006 / 1000
        return prompt_cost + completion_cost

    def report(self):
        """生成报告"""
        print(f"📊 Token 使用统计:")
        print(f"  - Prompt tokens: {self.prompt_tokens}")
        print(f"  - Completion tokens: {self.completion_tokens}")
        print(f"  - Total tokens: {self.total_tokens}")
        print(f"💰 预估成本: ${self.get_cost():.4f}")


# 使用成本追踪
tracker = CostTracker()
config = RunnableConfig(callbacks=[tracker])

# 批量处理
inputs = [{"text": f"文本{i}"} for i in range(100)]
results = chain.batch(inputs, config=config)

# 查看成本
tracker.report()
```

---

## 错误处理

### 单个任务失败

```python
from langchain_core.runnables import RunnableLambda

def risky_process(x: str) -> str:
    if x == "error":
        raise ValueError("处理失败")
    return x.upper()

runnable = RunnableLambda(risky_process)

# ❌ 单个失败导致整个 batch 失败
try:
    results = runnable.batch(["a", "error", "c"])
except ValueError as e:
    print(f"Batch 失败: {e}")

# ✅ 使用 try-except 包装
def safe_process(x: str) -> str:
    try:
        return risky_process(x)
    except Exception as e:
        return f"ERROR: {e}"

safe_runnable = RunnableLambda(safe_process)
results = safe_runnable.batch(["a", "error", "c"])
print(results)  # ["A", "ERROR: 处理失败", "C"]
```

---

## 2025-2026 最佳实践

### 1. 使用 batch 而非循环

```python
# ❌ 不推荐
results = [chain.invoke(input) for input in inputs]

# ✅ 推荐
results = chain.batch(inputs)
```

### 2. 合理设置并发数

```python
# ✅ 根据 API 限制设置
config = RunnableConfig(max_concurrency=5)
results = chain.batch(inputs, config=config)
```

### 3. 监控成本

```python
# ✅ 使用回调追踪成本
tracker = CostTracker()
config = RunnableConfig(callbacks=[tracker])
results = chain.batch(inputs, config=config)
tracker.report()
```

### 4. 考虑 langasync

```python
# ✅ 非实时场景使用 langasync
if not real_time_required:
    async_chain = wrap_chain(chain, batch_size=10)
    results = await async_chain.abatch(inputs)  # 50% 成本节省
```

---

## 总结

### batch 方法的核心价值

1. **并发执行**: 自动并行处理多个输入
2. **成本优化**: 结合 langasync 降低 50% 成本
3. **顺序保证**: 输出顺序与输入一致
4. **简单易用**: 无需手动管理线程池

### 何时使用 batch

- ✅ 批量处理多个独立任务
- ✅ 评估和测试场景
- ✅ 成本敏感的离线任务
- ❌ 单次查询（用 invoke）
- ❌ 实时流式输出（用 stream）

---

## 参考资料

[^1]: [Runnable batch Reference](https://reference.langchain.com/python/langchain_core/runnables) - LangChain, 2025-2026
[^2]: [langasync GitHub](https://github.com/langasync/langasync) - 50% Cost Savings, 2025-2026
[^3]: [LangChain Batch Processing Cost Optimization](https://medium.com/@vinodkrane/langchain-in-production-performance-security-and-cost-optimization-d5e0b44a26fd) - Medium, 2025

---

**下一步**: 阅读 [03_核心概念_04_stream方法.md](./03_核心概念_04_stream方法.md) 学习流式输出
