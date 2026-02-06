# 核心概念3：Map-Reduce 策略

**Map-Reduce 是将长文档分解为独立片段并行处理，最后聚合结果的策略，适合可以分解为独立子问题的场景。**

---

## 什么是 Map-Reduce？

### 直觉理解

想象你要统计一本 1000 页书中每个词出现的次数：

```
方案1：一个人从头到尾数（串行处理）
- 时间：1000 分钟

方案2：10 个人分工，每人数 100 页，最后汇总（并行处理）
- Map 阶段：10 个人同时数，每人 100 分钟
- Reduce 阶段：汇总 10 个人的结果，10 分钟
- 总时间：110 分钟（快了 9 倍）
```

**Map-Reduce 就是这样的过程**：将大任务分解为小任务，并行处理，最后聚合结果。

### 形式化定义

```
Map-Reduce = Map（映射）+ Reduce（归约）

Map 阶段：
- 将文档分成 N 个片段
- 对每个片段独立执行相同的操作
- 得到 N 个中间结果

Reduce 阶段：
- 将 N 个中间结果聚合
- 得到最终结果
```

---

## 为什么需要 Map-Reduce？

### 问题：某些问题可以分解

假设用户问："这篇论文每章的核心观点是什么？"

```python
# 传统方案：串行处理
for chapter in chapters:
    summary = llm(f"总结这一章的核心观点：{chapter}")
    summaries.append(summary)

# 问题：
# - 如果有 10 章，需要调用 LLM 10 次（串行）
# - 每次调用 5 秒，总共 50 秒
```

**Map-Reduce 方案**：

```python
# Map 阶段：并行处理每章
results = parallel_map(
    func=lambda chapter: llm(f"总结这一章的核心观点：{chapter}"),
    data=chapters
)

# Reduce 阶段：聚合结果
final_summary = llm(f"汇总以下各章的核心观点：{results}")

# 优势：
# - 10 章并行处理，只需 5 秒（而不是 50 秒）
# - Reduce 阶段再花 5 秒
# - 总共 10 秒（快了 5 倍）
```

---

## Map-Reduce 的核心原理

### 原理1：任务分解

```
原始问题："总结这篇 500 页论文的核心观点"

分解为子问题：
- 子问题1："总结第 1-50 页的核心观点"
- 子问题2："总结第 51-100 页的核心观点"
- ...
- 子问题10："总结第 451-500 页的核心观点"

关键特性：
- 每个子问题独立（不依赖其他子问题的结果）
- 每个子问题可以并行处理
```

### 原理2：并行处理（Map）

```python
def map_phase(documents, process_func):
    """Map 阶段：并行处理"""

    # 方式1：多线程（适合 I/O 密集型，如 API 调用）
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_func, documents))

    # 方式2：多进程（适合 CPU 密集型）
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_func, documents))

    # 方式3：异步（适合大量 API 调用）
    async def async_map():
        tasks = [process_func(doc) for doc in documents]
        return await asyncio.gather(*tasks)

    return results
```

### 原理3：结果聚合（Reduce）

```python
def reduce_phase(intermediate_results, aggregate_func):
    """Reduce 阶段：聚合结果"""

    # 方式1：简单拼接
    final_result = "\n\n".join(intermediate_results)

    # 方式2：LLM 总结
    final_result = llm(f"汇总以下内容：\n\n{chr(10).join(intermediate_results)}")

    # 方式3：递归 Reduce（如果中间结果太多）
    if len(intermediate_results) > 10:
        # 先分组 Reduce
        groups = [intermediate_results[i:i+10] for i in range(0, len(intermediate_results), 10)]
        group_results = [aggregate_func(group) for group in groups]
        # 再对分组结果 Reduce
        final_result = aggregate_func(group_results)
    else:
        final_result = aggregate_func(intermediate_results)

    return final_result
```

---

## Python 手写实现

### 完整实现：Map-Reduce 系统

```python
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

class MapReduce:
    """Map-Reduce 处理器"""

    def __init__(self, max_workers: int = 10):
        """
        Args:
            max_workers: 最大并行数
        """
        self.max_workers = max_workers

    def process(
        self,
        documents: List[str],
        map_func: Callable[[str], Any],
        reduce_func: Callable[[List[Any]], Any]
    ) -> Any:
        """
        执行 Map-Reduce

        Args:
            documents: 文档列表
            map_func: Map 函数（处理单个文档）
            reduce_func: Reduce 函数（聚合结果）

        Returns:
            最终结果
        """
        # Map 阶段：并行处理
        print(f"=== Map 阶段：处理 {len(documents)} 个文档 ===")
        map_results = self._map_parallel(documents, map_func)

        # Reduce 阶段：聚合结果
        print(f"=== Reduce 阶段：聚合 {len(map_results)} 个结果 ===")
        final_result = reduce_func(map_results)

        return final_result

    def _map_parallel(self, documents: List[str], map_func: Callable) -> List[Any]:
        """并行执行 Map"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_doc = {
                executor.submit(map_func, doc): i
                for i, doc in enumerate(documents)
            }

            # 收集结果（保持顺序）
            results = [None] * len(documents)
            for future in as_completed(future_to_doc):
                idx = future_to_doc[future]
                try:
                    result = future.result()
                    results[idx] = result
                    print(f"  完成文档 {idx + 1}/{len(documents)}")
                except Exception as e:
                    print(f"  文档 {idx + 1} 处理失败: {e}")
                    results[idx] = None

        # 过滤掉失败的结果
        return [r for r in results if r is not None]

    async def process_async(
        self,
        documents: List[str],
        map_func: Callable[[str], Any],
        reduce_func: Callable[[List[Any]], Any]
    ) -> Any:
        """
        异步执行 Map-Reduce（适合大量 API 调用）
        """
        # Map 阶段：异步并行处理
        print(f"=== Map 阶段（异步）：处理 {len(documents)} 个文档 ===")
        tasks = [map_func(doc) for doc in documents]
        map_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤掉失败的结果
        map_results = [r for r in map_results if not isinstance(r, Exception)]

        # Reduce 阶段：聚合结果
        print(f"=== Reduce 阶段：聚合 {len(map_results)} 个结果 ===")
        final_result = reduce_func(map_results)

        return final_result


# ===== 使用示例 =====

import time

# 模拟 LLM 调用（实际应使用 OpenAI API）
def mock_llm(prompt: str, delay: float = 0.5) -> str:
    """模拟 LLM 调用"""
    time.sleep(delay)  # 模拟 API 延迟
    # 简化版：返回前 50 字
    return prompt[:50] + "..."

# 准备长文档（分成多个章节）
chapters = [
    "第1章：神经网络基础。神经网络是由大量神经元组成的计算模型...",
    "第2章：卷积神经网络。CNN 专门用于处理图像数据...",
    "第3章：循环神经网络。RNN 用于处理序列数据...",
    "第4章：生成对抗网络。GAN 由生成器和判别器组成...",
    "第5章：强化学习。强化学习通过与环境交互学习最优策略...",
    "第6章：迁移学习。迁移学习利用预训练模型加速训练...",
    "第7章：模型压缩。模型压缩通过剪枝、量化等技术减小模型大小...",
    "第8章：联邦学习。联邦学习在保护隐私的前提下训练模型...",
    "第9章：可解释性。可解释性研究如何理解模型的决策过程...",
    "第10章：未来展望。深度学习的未来发展方向..."
]

# 定义 Map 函数：总结每章
def map_summarize_chapter(chapter: str) -> str:
    """总结单个章节"""
    prompt = f"请用一句话总结这一章的核心观点：\n\n{chapter}"
    return mock_llm(prompt)

# 定义 Reduce 函数：汇总所有章节摘要
def reduce_aggregate_summaries(summaries: List[str]) -> str:
    """汇总所有章节摘要"""
    combined = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])
    prompt = f"请汇总以下各章的核心观点：\n\n{combined}"
    return mock_llm(prompt, delay=1.0)

# 执行 Map-Reduce
print("=== 开始 Map-Reduce 处理 ===\n")
start_time = time.time()

mr = MapReduce(max_workers=5)
result = mr.process(
    documents=chapters,
    map_func=map_summarize_chapter,
    reduce_func=reduce_aggregate_summaries
)

end_time = time.time()

print(f"\n=== 处理完成 ===")
print(f"总耗时: {end_time - start_time:.2f} 秒")
print(f"\n最终结果:\n{result}")

# 对比：串行处理
print("\n\n=== 对比：串行处理 ===\n")
start_time = time.time()

summaries = []
for i, chapter in enumerate(chapters):
    summary = map_summarize_chapter(chapter)
    summaries.append(summary)
    print(f"  完成章节 {i + 1}/{len(chapters)}")

result_serial = reduce_aggregate_summaries(summaries)

end_time = time.time()

print(f"\n=== 处理完成 ===")
print(f"总耗时: {end_time - start_time:.2f} 秒")
print(f"\n最终结果:\n{result_serial}")
```

**运行输出示例：**

```
=== 开始 Map-Reduce 处理 ===

=== Map 阶段：处理 10 个文档 ===
  完成文档 1/10
  完成文档 2/10
  完成文档 3/10
  完成文档 4/10
  完成文档 5/10
  完成文档 6/10
  完成文档 7/10
  完成文档 8/10
  完成文档 9/10
  完成文档 10/10
=== Reduce 阶段：聚合 10 个结果 ===

=== 处理完成 ===
总耗时: 2.05 秒

最终结果:
请汇总以下各章的核心观点：

1. 请用一句话总结这一章的核心观点：

第1章：神经网络基础。神经网络是由大量神经元组成的计算模型......


=== 对比：串行处理 ===

  完成章节 1/10
  完成章节 2/10
  完成章节 3/10
  完成章节 4/10
  完成章节 5/10
  完成章节 6/10
  完成章节 7/10
  完成章节 8/10
  完成章节 9/10
  完成章节 10/10

=== 处理完成 ===
总耗时: 6.01 秒
```

**性能对比**：
- Map-Reduce（并行）：2.05 秒
- 串行处理：6.01 秒
- **加速比：2.9 倍**

---

## Map-Reduce 的优缺点

### 优点

| 优点 | 说明 |
|------|------|
| ✅ 并行处理，速度快 | 充分利用多核 CPU 或并发 API 调用 |
| ✅ 突破 Context Window 限制 | 每个片段独立处理，不受限制 |
| ✅ 适合可分解问题 | 如"每章的核心观点"、"每段的关键词" |
| ✅ 容错性好 | 单个片段失败不影响其他片段 |

### 缺点

| 缺点 | 说明 |
|------|------|
| ❌ 不适合需要全局理解的问题 | 如"论文的整体逻辑结构" |
| ❌ Reduce 阶段可能成为瓶颈 | 如果中间结果太多，聚合困难 |
| ❌ 信息损失 | 每个片段独立处理，丢失片段间的关联 |
| ❌ 成本高 | 并行调用 LLM，API 费用增加 |

---

## 在 RAG 开发中的应用

### 应用场景1：多文档问答

```python
# 场景：用户问"对比 3 篇论文的方法"
#
# Map-Reduce 流程：
# Map 阶段：并行处理 3 篇论文，提取每篇的方法
# Reduce 阶段：对比 3 篇论文的方法

def compare_papers(papers, question):
    """对比多篇论文"""

    # Map 函数：提取单篇论文的方法
    def extract_method(paper):
        prompt = f"提取这篇论文的方法：\n\n{paper}"
        return llm(prompt)

    # Reduce 函数：对比所有方法
    def compare_methods(methods):
        combined = "\n\n".join([f"论文{i+1}的方法：\n{m}" for i, m in enumerate(methods)])
        prompt = f"对比以下论文的方法：\n\n{combined}\n\n问题：{question}"
        return llm(prompt)

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(papers, extract_method, compare_methods)
```

### 应用场景2：长文档摘要

```python
# 场景：用户问"总结这篇 500 页论文的核心观点"
#
# Map-Reduce 流程：
# Map 阶段：并行总结每章
# Reduce 阶段：汇总所有章节摘要

def summarize_long_document(document):
    """总结长文档"""

    # 分章节
    chapters = split_into_chapters(document)

    # Map 函数：总结单章
    def summarize_chapter(chapter):
        prompt = f"总结这一章的核心观点：\n\n{chapter}"
        return llm(prompt)

    # Reduce 函数：汇总所有章节摘要
    def aggregate_summaries(summaries):
        combined = "\n".join(summaries)
        prompt = f"汇总以下各章的核心观点：\n\n{combined}"
        return llm(prompt)

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(chapters, summarize_chapter, aggregate_summaries)
```

### 应用场景3：批量问答

```python
# 场景：用户有多个问题，需要在同一文档中查找答案
#
# Map-Reduce 流程：
# Map 阶段：并行回答每个问题
# Reduce 阶段：汇总所有答案

def batch_qa(document, questions):
    """批量问答"""

    # Map 函数：回答单个问题
    def answer_question(question):
        prompt = f"根据以下文档回答问题：\n\n文档：{document}\n\n问题：{question}"
        return llm(prompt)

    # Reduce 函数：汇总所有答案
    def aggregate_answers(answers):
        result = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            result.append(f"问题{i+1}：{q}\n答案：{a}")
        return "\n\n".join(result)

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(questions, answer_question, aggregate_answers)
```

---

## Map-Reduce vs 其他策略

| 维度 | Map-Reduce | 分层索引 | 摘要链 |
|------|-----------|---------|--------|
| **核心思想** | 并行处理，聚合结果 | 逐层检索，快速定位 | 递归总结，压缩信息 |
| **适用场景** | 可分解为独立子问题 | 需要快速定位 | 需要全文理解 |
| **处理方式** | 并行 | 串行（逐层） | 串行（递归） |
| **速度** | 快（并行） | 中（逐层检索） | 慢（多次总结） |
| **信息保留** | 中（片段间关联丢失） | 高（保留结构） | 中（有损压缩） |
| **成本** | 高（并行调用 API） | 中 | 高（多次调用 LLM） |

---

## 实战技巧

### 技巧1：递归 Reduce

```python
def recursive_reduce(results, reduce_func, max_batch_size=10):
    """递归 Reduce（处理大量中间结果）"""

    if len(results) <= max_batch_size:
        # 结果数量少，直接 Reduce
        return reduce_func(results)

    # 结果数量多，分批 Reduce
    batches = [results[i:i+max_batch_size] for i in range(0, len(results), max_batch_size)]
    batch_results = [reduce_func(batch) for batch in batches]

    # 递归 Reduce
    return recursive_reduce(batch_results, reduce_func, max_batch_size)
```

### 技巧2：动态调整并行度

```python
def adaptive_map_reduce(documents, map_func, reduce_func):
    """自适应并行度"""

    # 根据文档数量调整并行度
    if len(documents) < 5:
        max_workers = len(documents)  # 文档少，不需要太多并行
    elif len(documents) < 20:
        max_workers = 5
    else:
        max_workers = 10  # 文档多，充分并行

    mr = MapReduce(max_workers=max_workers)
    return mr.process(documents, map_func, reduce_func)
```

### 技巧3：混合策略（Map-Reduce + 摘要链）

```python
def hybrid_map_reduce_summary(document):
    """混合策略：先 Map-Reduce 处理章节，再用摘要链压缩"""

    # 第1步：分章节
    chapters = split_into_chapters(document)

    # 第2步：Map-Reduce 总结每章
    def summarize_chapter(chapter):
        # 如果章节太长，先用摘要链压缩
        if len(chapter) > 10000:
            chain = SummaryChain(summarize_func=llm_summarize)
            chain.build(chapter)
            return chain.get_full_summary()
        else:
            return llm(f"总结这一章：{chapter}")

    mr = MapReduce()
    chapter_summaries = mr._map_parallel(chapters, summarize_chapter)

    # 第3步：Reduce 汇总
    final_summary = llm(f"汇总以下各章摘要：\n\n{chr(10).join(chapter_summaries)}")

    return final_summary
```

### 技巧4：错误处理与重试

```python
def map_with_retry(documents, map_func, max_retries=3):
    """带重试的 Map"""

    results = []

    for doc in documents:
        for attempt in range(max_retries):
            try:
                result = map_func(doc)
                results.append(result)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"文档处理失败（已重试 {max_retries} 次）: {e}")
                    results.append(None)
                else:
                    print(f"文档处理失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)  # 等待 1 秒后重试

    return [r for r in results if r is not None]
```

---

## 高级应用：异步 Map-Reduce

```python
import asyncio
from typing import List

class AsyncMapReduce:
    """异步 Map-Reduce（适合大量 API 调用）"""

    async def process(
        self,
        documents: List[str],
        map_func,  # async function
        reduce_func
    ):
        """异步执行 Map-Reduce"""

        # Map 阶段：异步并行处理
        print(f"=== Map 阶段（异步）：处理 {len(documents)} 个文档 ===")
        tasks = [map_func(doc) for doc in documents]
        map_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤掉失败的结果
        map_results = [r for r in map_results if not isinstance(r, Exception)]
        print(f"  成功处理 {len(map_results)}/{len(documents)} 个文档")

        # Reduce 阶段：聚合结果
        print(f"=== Reduce 阶段：聚合结果 ===")
        final_result = reduce_func(map_results)

        return final_result


# 使用示例
async def async_llm(prompt: str) -> str:
    """异步 LLM 调用"""
    await asyncio.sleep(0.5)  # 模拟 API 延迟
    return prompt[:50] + "..."

async def main():
    chapters = ["第1章...", "第2章...", "第3章..."]

    async def map_summarize(chapter):
        return await async_llm(f"总结：{chapter}")

    def reduce_aggregate(summaries):
        return "\n".join(summaries)

    mr = AsyncMapReduce()
    result = await mr.process(chapters, map_summarize, reduce_aggregate)
    print(f"\n最终结果:\n{result}")

# 运行
# asyncio.run(main())
```

---

## 总结

**Map-Reduce 的核心**：
1. **任务分解**：将大任务分解为独立的小任务
2. **并行处理**：同时处理多个小任务（Map）
3. **结果聚合**：将所有结果汇总（Reduce）

**适用场景**：
- ✅ 可以分解为独立子问题的场景（如"每章的核心观点"）
- ✅ 多文档问答（每篇文档独立处理）
- ✅ 批量问答（每个问题独立回答）
- ✅ 需要加速处理的场景

**不适用场景**：
- ❌ 需要全局理解的问题（如"论文的整体逻辑结构"）
- ❌ 片段间有强依赖的场景
- ❌ 成本敏感的场景（并行调用 API 费用高）

**性能优势**：
- 并行处理可以获得接近线性的加速比
- 10 个文档并行处理，理论上可以快 10 倍
- 实际加速比取决于并行度和 Reduce 阶段的开销

---

**下一步：** [06_最小可用](./06_最小可用.md) - 掌握 20% 核心知识解决 80% 问题
