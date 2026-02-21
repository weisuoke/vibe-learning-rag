# 核心概念_01_batch方法详解

> 深入理解 batch() 和 abatch() 方法的实现原理和使用场景

---

## batch() 方法概述

batch() 是 LCEL 中最重要的批处理方法，用于并行执行多个独立任务。

### 方法签名

```python
def batch(
    self,
    inputs: List[Input],
    config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any
) -> List[Output]:
    """批量执行多个输入"""
```

### 核心参数

1. **inputs**: 输入列表
2. **config**: 配置对象（包含 max_concurrency）
3. **return_exceptions**: 是否返回异常而非抛出

---

## batch() 实现原理

### 1. ThreadPoolExecutor 执行引擎

```python
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig

def batch_implementation(self, inputs, config=None):
    """batch() 的简化实现"""
    # 1. 解析配置
    max_concurrency = 10  # 默认值
    if config and "max_concurrency" in config:
        max_concurrency = config["max_concurrency"]

    # 2. 创建线程池
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # 3. 提交所有任务
        futures = [
            executor.submit(self.invoke, inp, config)
            for inp in inputs
        ]

        # 4. 收集结果（保持顺序）
        results = [future.result() for future in futures]

    return results
```

**关键点**：
- 使用 ThreadPoolExecutor 管理线程池
- max_workers 由 max_concurrency 决定
- 结果按输入顺序返回

**参考来源**：[LangChain Runnables 文档](https://reference.langchain.com/python/langchain_core/runnables/)

---

### 2. 为什么使用 ThreadPoolExecutor？

**IO 密集型任务的特点**：
```python
import time

def llm_call():
    # 实际执行时间分解
    network_time = 1.8  # 网络传输（90%）
    compute_time = 0.2  # 计算处理（10%）
    return network_time + compute_time  # 总计 2 秒
```

**GIL 与 IO 操作**：
- Python 的 GIL（全局解释器锁）限制 CPU 密集型并行
- IO 操作会释放 GIL，允许其他线程执行
- LLM API 调用是 IO 密集型，适合多线程

**性能对比**：
```python
# 串行执行
total_time = n * 2  # 100 个任务 = 200 秒

# ThreadPoolExecutor（10 线程）
total_time = (n / 10) * 2  # 100 个任务 = 20 秒
```

---

### 3. 结果顺序保证

batch() 保证结果顺序与输入顺序一致：

```python
def batch_with_order(self, inputs, config=None):
    """保证顺序的批处理实现"""
    max_concurrency = config.get("max_concurrency", 10)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # 提交任务时记录索引
        futures = {
            executor.submit(self.invoke, inp, config): i
            for i, inp in enumerate(inputs)
        }

        # 按索引排序结果
        results = [None] * len(inputs)
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()

    return results
```

**为什么需要保证顺序**：
- 用户期望结果与输入对应
- 便于后续处理和调试
- 符合函数式编程习惯

---

## abatch() 方法详解

### 1. 异步实现

```python
import asyncio

async def abatch_implementation(self, inputs, config=None):
    """abatch() 的简化实现"""
    max_concurrency = config.get("max_concurrency", 10)

    # 创建信号量限制并发
    semaphore = asyncio.Semaphore(max_concurrency)

    async def invoke_with_semaphore(inp):
        async with semaphore:
            return await self.ainvoke(inp, config)

    # 并发执行所有任务
    tasks = [invoke_with_semaphore(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)

    return results
```

**关键差异**：
- 使用 asyncio.Semaphore 而非 ThreadPoolExecutor
- 使用协程而非线程
- 使用 asyncio.gather 收集结果

**参考来源**：[LangChain 并发编程指南](https://medium.com/@oscar.f.agreda/unlocking-pythons-potential-concurrency-with-langchain-and-beyond-876149aaf475)

---

### 2. batch() vs abatch() 性能对比

**低并发场景（10 个任务）**：
```python
# batch()
with ThreadPoolExecutor(max_workers=10) as executor:
    # 线程创建开销：约 10ms
    # 线程切换开销：约 1-10μs × 切换次数
    # 总开销：约 20-50ms

# abatch()
async with asyncio.Semaphore(10):
    # 协程创建开销：约 1ms
    # 协程切换开销：约 0.1-1μs × 切换次数
    # 总开销：约 5-10ms
```

**高并发场景（1000 个任务）**：
```python
# batch()
# 线程开销：1000 × 1-8MB = 1-8GB 内存
# 线程切换：频繁切换导致 CPU 浪费
# 性能：受限于线程数量

# abatch()
# 协程开销：1000 × 1-2KB = 1-2MB 内存
# 协程切换：几乎无开销
# 性能：可以支持更高并发
```

**结论**：
- 低并发（< 50）：batch() 和 abatch() 性能相近
- 高并发（> 100）：abatch() 性能优势明显

---

## batch_as_completed() 方法（2025 新特性）

### 1. 方法签名

```python
def batch_as_completed(
    self,
    inputs: List[Input],
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Iterator[Tuple[int, Output]]:
    """按完成顺序返回结果"""
```

### 2. 实现原理

```python
from concurrent.futures import as_completed

def batch_as_completed_implementation(self, inputs, config=None):
    """batch_as_completed() 的简化实现"""
    max_concurrency = config.get("max_concurrency", 10)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # 提交任务并记录索引
        future_to_index = {
            executor.submit(self.invoke, inp, config): i
            for i, inp in enumerate(inputs)
        }

        # 按完成顺序返回
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            yield idx, result
```

### 3. 使用场景

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("生成{length}个字的故事")
chain = prompt | llm

inputs = [
    {"length": 100},  # 慢
    {"length": 10},   # 快
    {"length": 50},   # 中
]

# 按完成顺序处理
for idx, result in chain.batch_as_completed(inputs):
    print(f"任务 {idx} 完成: {len(result.content)} 字符")
    # 可以立即处理结果，不用等待所有任务完成
```

**优势**：
- 实时反馈进度
- 可以提前处理已完成的结果
- 提升用户体验

**参考来源**：[LangChain Models 文档](https://docs.langchain.com/oss/python/langchain/models)

---

## 配置传递机制

### 1. RunnableConfig 结构

```python
from typing import TypedDict, Optional

class RunnableConfig(TypedDict, total=False):
    """Runnable 配置"""
    max_concurrency: Optional[int]  # 最大并发数
    recursion_limit: Optional[int]  # 递归限制
    tags: Optional[List[str]]       # 标签
    metadata: Optional[Dict]        # 元数据
    callbacks: Optional[List]       # 回调函数
```

### 2. 配置继承

```python
# 全局配置
global_config = {"max_concurrency": 10, "tags": ["production"]}

# 局部配置（会覆盖全局配置）
local_config = {"max_concurrency": 5}

# 合并配置
merged_config = {**global_config, **local_config}
# 结果：{"max_concurrency": 5, "tags": ["production"]}
```

### 3. 配置传递示例

```python
from langchain_core.runnables import RunnableConfig

# 创建链
chain = prompt | llm | parser

# 方式1：直接传递配置
results = chain.batch(
    inputs,
    config={"max_concurrency": 10}
)

# 方式2：使用 RunnableConfig 对象
config = RunnableConfig(max_concurrency=10, tags=["batch"])
results = chain.batch(inputs, config=config)

# 方式3：为每个输入指定不同配置
configs = [
    {"max_concurrency": 5} if i < 50 else {"max_concurrency": 10}
    for i in range(len(inputs))
]
results = chain.batch(inputs, config=configs)
```

---

## 错误处理

### 1. return_exceptions 参数

```python
# 默认行为：抛出异常
try:
    results = chain.batch(inputs)
except Exception as e:
    print(f"批处理失败: {e}")

# 返回异常而非抛出
results = chain.batch(inputs, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"任务 {i} 失败: {result}")
    else:
        print(f"任务 {i} 成功: {result}")
```

### 2. 部分失败处理

```python
def batch_with_fallback(chain, inputs, config=None):
    """处理部分失败的批处理"""
    results = []

    for inp in inputs:
        try:
            result = chain.invoke(inp, config)
            results.append(result)
        except Exception as e:
            print(f"任务失败: {e}")
            results.append(None)  # 或者使用默认值

    return results
```

---

## 性能优化技巧

### 1. 选择合适的并发数

```python
import time

def find_optimal_concurrency(chain, inputs, test_range=range(1, 51, 5)):
    """测试找到最优并发数"""
    results = {}

    for conc in test_range:
        start = time.time()
        chain.batch(inputs[:20], config={"max_concurrency": conc})
        duration = time.time() - start
        results[conc] = duration

    optimal = min(results, key=results.get)
    print(f"最优并发数: {optimal}")
    return optimal
```

### 2. 批处理分片

```python
def chunked_batch(chain, inputs, chunk_size=100, max_concurrency=10):
    """分片批处理，避免一次性处理过多任务"""
    all_results = []

    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i + chunk_size]
        chunk_results = chain.batch(
            chunk,
            config={"max_concurrency": max_concurrency}
        )
        all_results.extend(chunk_results)

        # 分片间休息，避免限流
        if i + chunk_size < len(inputs):
            time.sleep(1)

    return all_results
```

### 3. 动态调整并发数

```python
class AdaptiveBatch:
    def __init__(self, chain, initial_concurrency=10):
        self.chain = chain
        self.concurrency = initial_concurrency
        self.success_count = 0
        self.failure_count = 0

    def batch(self, inputs):
        """自适应批处理"""
        try:
            results = self.chain.batch(
                inputs,
                config={"max_concurrency": self.concurrency}
            )
            self.success_count += len(inputs)
            self.adjust_concurrency(success=True)
            return results
        except Exception as e:
            self.failure_count += len(inputs)
            self.adjust_concurrency(success=False)
            raise

    def adjust_concurrency(self, success):
        """根据成功率调整并发数"""
        if success and self.success_count > 100:
            self.concurrency = min(self.concurrency + 2, 50)
        elif not success:
            self.concurrency = max(self.concurrency - 2, 5)
```

---

## 实战案例

### 案例1：批量评估

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建评估链
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "评估以下回答的质量（1-10分）：\n问题：{question}\n回答：{answer}"
)
chain = prompt | llm

# 准备评估数据
eval_data = [
    {"question": "什么是 AI？", "answer": "AI 是人工智能..."},
    {"question": "什么是 ML？", "answer": "ML 是机器学习..."},
    # ... 更多数据
]

# 批量评估
results = chain.batch(eval_data, config={"max_concurrency": 10})
```

### 案例2：批量翻译

```python
# 创建翻译链
prompt = ChatPromptTemplate.from_template("将以下文本翻译成英文：{text}")
chain = prompt | llm

# 批量翻译
texts = ["你好", "世界", "人工智能", ...]
inputs = [{"text": text} for text in texts]
translations = chain.batch(inputs, config={"max_concurrency": 20})
```

### 案例3：批量摘要

```python
# 创建摘要链
prompt = ChatPromptTemplate.from_template("总结以下文本（50字以内）：{text}")
chain = prompt | llm

# 批量摘要
documents = ["长文档1...", "长文档2...", ...]
inputs = [{"text": doc} for doc in documents]
summaries = chain.batch(inputs, config={"max_concurrency": 15})
```

---

## 总结

batch() 方法的核心要点：

1. **实现原理**：
   - 使用 ThreadPoolExecutor 管理线程池
   - max_concurrency 控制并发数
   - 结果按输入顺序返回

2. **性能特点**：
   - 适合 IO 密集型任务
   - 5-10 倍性能提升
   - 低并发场景最优

3. **abatch() 优势**：
   - 高并发场景性能更好
   - 内存占用更小
   - 需要异步环境

4. **batch_as_completed()**：
   - 按完成顺序返回结果
   - 实时反馈进度
   - 2025 新特性

5. **最佳实践**：
   - 根据场景选择并发数
   - 使用分片处理大批量任务
   - 实现错误处理和重试
   - 监控性能和成本

---

## 参考来源

1. [LangChain Runnables 文档](https://reference.langchain.com/python/langchain_core/runnables/) - batch 和 abatch API
2. [LangChain 并发编程指南](https://medium.com/@oscar.f.agreda/unlocking-pythons-potential-concurrency-with-langchain-and-beyond-876149aaf475) - ThreadPoolExecutor 详解
3. [LangChain Models 文档](https://docs.langchain.com/oss/python/langchain/models) - batch_as_completed
4. [LangChain 最佳实践](https://www.swarnendu.de/blog/langchain-best-practices/) - 并发与性能优化

---

**下一步**：阅读 `03_核心概念_02_并发控制机制.md` 深入理解 max_concurrency 和线程池管理
