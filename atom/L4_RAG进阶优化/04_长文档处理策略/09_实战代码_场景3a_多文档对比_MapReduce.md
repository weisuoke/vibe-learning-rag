# 实战代码 - 场景3a：多文档对比（Map-Reduce 基础版）

完整可运行的多文档对比示例，展示如何使用 Map-Reduce 并行处理多篇论文。

---

## 场景描述

**任务**：对比 3 篇 AI 研究论文的方法，找出它们的异同点

**挑战**：
- 3 篇论文共 150 页，无法一次性处理
- 每篇论文的处理是独立的（可以并行）
- 需要最后汇总对比结果

**解决方案**：
- 使用 Map-Reduce 并行处理每篇论文
- Map 阶段：提取每篇论文的方法描述
- Reduce 阶段：对比所有论文的方法

---

## 完整代码实现

```python
"""
多文档对比系统（Map-Reduce 基础版）
演示：使用 Map-Reduce 并行处理多篇论文
"""

import time
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# ===== 1. 数据结构定义 =====

@dataclass
class Paper:
    """论文"""
    id: str
    title: str
    content: str
    method_section: str  # 方法章节

# ===== 2. Map-Reduce 实现 =====

class MapReduce:
    """Map-Reduce 处理器"""

    def __init__(self, max_workers: int = 5):
        """
        Args:
            max_workers: 最大并行数
        """
        self.max_workers = max_workers

    def process(
        self,
        documents: List[any],
        map_func: Callable,
        reduce_func: Callable
    ) -> any:
        """
        执行 Map-Reduce

        Args:
            documents: 文档列表
            map_func: Map 函数（处理单个文档）
            reduce_func: Reduce 函数（聚合结果）

        Returns:
            最终结果
        """
        print(f"\n=== Map 阶段：处理 {len(documents)} 个文档 ===")
        start_time = time.time()

        # Map 阶段：并行处理
        map_results = self._map_parallel(documents, map_func)

        map_time = time.time() - start_time
        print(f"Map 阶段耗时: {map_time:.2f} 秒")

        # Reduce 阶段：聚合结果
        print(f"\n=== Reduce 阶段：聚合 {len(map_results)} 个结果 ===")
        reduce_start = time.time()

        final_result = reduce_func(map_results)

        reduce_time = time.time() - reduce_start
        print(f"Reduce 阶段耗时: {reduce_time:.2f} 秒")

        total_time = time.time() - start_time
        print(f"\n总耗时: {total_time:.2f} 秒")

        return final_result

    def _map_parallel(self, documents: List[any], map_func: Callable) -> List[any]:
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
                    print(f"  ✓ 完成文档 {idx + 1}/{len(documents)}")
                except Exception as e:
                    print(f"  ✗ 文档 {idx + 1} 处理失败: {e}")
                    results[idx] = None

        # 过滤掉失败的结果
        return [r for r in results if r is not None]

# ===== 3. 论文对比系统 =====

class PaperComparisonSystem:
    """论文对比系统"""

    def __init__(self, llm_func):
        self.llm_func = llm_func
        self.map_reduce = MapReduce(max_workers=3)

    def compare_papers(self, papers: List[Paper], aspect: str = "方法") -> str:
        """
        对比多篇论文

        Args:
            papers: 论文列表
            aspect: 对比的方面（方法、实验、结论等）

        Returns:
            对比结果
        """
        print(f"\n{'='*60}")
        print(f"对比 {len(papers)} 篇论文的{aspect}")
        print(f"{'='*60}")

        # 定义 Map 函数：提取每篇论文的指定方面
        def map_extract_aspect(paper: Paper) -> Dict:
            """提取单篇论文的指定方面"""
            print(f"  处理论文: {paper.title}")

            if aspect == "方法":
                content = paper.method_section
            else:
                content = paper.content

            prompt = f"""请提取这篇论文的{aspect}：

论文标题：{paper.title}

内容：
{content}

请用 2-3 段话总结{aspect}的核心内容。"""

            result = self.llm_func(prompt)

            return {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "aspect": aspect,
                "content": result
            }

        # 定义 Reduce 函数：对比所有论文
        def reduce_compare(results: List[Dict]) -> str:
            """对比所有论文的结果"""
            print(f"  汇总 {len(results)} 篇论文的{aspect}")

            # 构建对比内容
            comparison_text = []
            for i, result in enumerate(results):
                comparison_text.append(
                    f"论文{i+1}：{result['paper_title']}\n{result['content']}"
                )

            combined = "\n\n---\n\n".join(comparison_text)

            prompt = f"""请对比以下论文的{aspect}，找出它们的异同点：

{combined}

请按以下格式回答：
1. 共同点：
2. 不同点：
3. 各自的优势："""

            return self.llm_func(prompt)

        # 执行 Map-Reduce
        result = self.map_reduce.process(
            documents=papers,
            map_func=map_extract_aspect,
            reduce_func=reduce_compare
        )

        return result

# ===== 4. 模拟函数 =====

def mock_llm(prompt: str, delay: float = 1.0) -> str:
    """模拟 LLM 函数"""
    time.sleep(delay)  # 模拟 API 延迟

    if "提取这篇论文的方法" in prompt:
        if "Efficient Attention" in prompt:
            return """这篇论文提出了 Efficient Attention 机制，包括三个核心技术：

1. 分层注意力：先在局部计算注意力，再在全局聚合
2. 序列压缩：使用可学习的压缩函数，将长序列压缩为短序列
3. 动态稀疏化：只计算重要的注意力权重

通过这三个技术，将复杂度从 O(n^2) 降低到 O(n log n)。"""

        elif "Sparse Transformer" in prompt:
            return """这篇论文提出了 Sparse Transformer，使用稀疏注意力模式：

1. 局部注意力：每个 token 只关注邻近的 k 个 token
2. 跨步注意力：每隔 k 个 token 关注一次
3. 全局注意力：少数特殊 token 关注所有 token

通过稀疏化，将复杂度从 O(n^2) 降低到 O(n√n)。"""

        elif "Linformer" in prompt:
            return """这篇论文提出了 Linformer，使用低秩分解：

1. 投影矩阵：将序列长度 n 投影到固定长度 k
2. 线性注意力：在投影后的空间计算注意力
3. 参数共享：多个注意力头共享投影矩阵

通过低秩分解，将复杂度从 O(n^2) 降低到 O(n)。"""

    elif "对比以下论文的方法" in prompt:
        return """对比结果：

1. 共同点：
   - 都致力于降低 Transformer 的计算复杂度
   - 都保持了模型的表达能力
   - 都在长序列任务上取得了良好效果

2. 不同点：
   - Efficient Attention 使用分层 + 压缩 + 稀疏化的组合策略
   - Sparse Transformer 使用固定的稀疏注意力模式
   - Linformer 使用低秩分解的数学方法

3. 各自的优势：
   - Efficient Attention：灵活性高，可以根据任务调整策略
   - Sparse Transformer：实现简单，易于理解和部署
   - Linformer：理论保证强，复杂度最低（O(n)）"""

    return "模拟 LLM 回答"

# ===== 5. 准备测试数据 =====

# 模拟 3 篇论文
papers = [
    Paper(
        id="paper1",
        title="Efficient Attention: A New Approach for Long Sequence Processing",
        content="...",
        method_section="""我们提出的 Efficient Attention 机制基于以下核心思想：
(1) 使用分层注意力，先在局部计算注意力，再在全局聚合；
(2) 使用可学习的压缩函数，将长序列压缩为短序列；
(3) 使用动态稀疏化，只计算重要的注意力权重。
通过这三个技术，我们将复杂度从 O(n^2) 降低到 O(n log n)。"""
    ),
    Paper(
        id="paper2",
        title="Sparse Transformer: Efficient Attention with Sparse Patterns",
        content="...",
        method_section="""我们提出的 Sparse Transformer 使用稀疏注意力模式：
(1) 局部注意力：每个 token 只关注邻近的 k 个 token；
(2) 跨步注意力：每隔 k 个 token 关注一次；
(3) 全局注意力：少数特殊 token（如 CLS）关注所有 token。
通过稀疏化，我们将复杂度从 O(n^2) 降低到 O(n√n)。"""
    ),
    Paper(
        id="paper3",
        title="Linformer: Self-Attention with Linear Complexity",
        content="...",
        method_section="""我们提出的 Linformer 使用低秩分解：
(1) 投影矩阵：将序列长度 n 投影到固定长度 k；
(2) 线性注意力：在投影后的空间计算注意力；
(3) 参数共享：多个注意力头共享投影矩阵。
通过低秩分解，我们将复杂度从 O(n^2) 降低到 O(n)。"""
    )
]

# ===== 6. 运行示例 =====

if __name__ == "__main__":
    print("=" * 60)
    print("多文档对比系统 - Map-Reduce 基础版")
    print("=" * 60)

    # 初始化对比系统
    comparison_system = PaperComparisonSystem(llm_func=mock_llm)

    # 对比论文的方法
    result = comparison_system.compare_papers(papers, aspect="方法")

    print(f"\n{'='*60}")
    print("对比结果：")
    print(f"{'='*60}")
    print(result)

    print(f"\n{'='*60}")
    print("示例完成")
    print(f"{'='*60}")

    # 对比：串行处理
    print(f"\n\n{'='*60}")
    print("对比：串行处理")
    print(f"{'='*60}")

    start_time = time.time()

    # 串行提取每篇论文的方法
    results = []
    for i, paper in enumerate(papers):
        print(f"  处理论文 {i+1}/{len(papers)}: {paper.title}")
        prompt = f"请提取这篇论文的方法：\n\n{paper.method_section}"
        result = mock_llm(prompt)
        results.append({
            "paper_title": paper.title,
            "content": result
        })

    # 对比
    comparison_text = [f"{r['paper_title']}\n{r['content']}" for r in results]
    combined = "\n\n---\n\n".join(comparison_text)
    prompt = f"请对比以下论文的方法：\n\n{combined}"
    final_result = mock_llm(prompt)

    serial_time = time.time() - start_time

    print(f"\n串行处理总耗时: {serial_time:.2f} 秒")
    print(f"\n加速比: {serial_time / 4.0:.2f}x")  # 假设并行耗时 4 秒
```

---

## 运行输出示例

```
============================================================
多文档对比系统 - Map-Reduce 基础版
============================================================

============================================================
对比 3 篇论文的方法
============================================================

=== Map 阶段：处理 3 个文档 ===
  处理论文: Efficient Attention: A New Approach for Long Sequence Processing
  处理论文: Sparse Transformer: Efficient Attention with Sparse Patterns
  处理论文: Linformer: Self-Attention with Linear Complexity
  ✓ 完成文档 1/3
  ✓ 完成文档 2/3
  ✓ 完成文档 3/3
Map 阶段耗时: 1.02 秒

=== Reduce 阶段：聚合 3 个结果 ===
  汇总 3 篇论文的方法
Reduce 阶段耗时: 1.01 秒

总耗时: 2.03 秒

============================================================
对比结果：
============================================================
对比结果：

1. 共同点：
   - 都致力于降低 Transformer 的计算复杂度
   - 都保持了模型的表达能力
   - 都在长序列任务上取得了良好效果

2. 不同点：
   - Efficient Attention 使用分层 + 压缩 + 稀疏化的组合策略
   - Sparse Transformer 使用固定的稀疏注意力模式
   - Linformer 使用低秩分解的数学方法

3. 各自的优势：
   - Efficient Attention：灵活性高，可以根据任务调整策略
   - Sparse Transformer：实现简单，易于理解和部署
   - Linformer：理论保证强，复杂度最低（O(n)）

============================================================
示例完成
============================================================


============================================================
对比：串行处理
============================================================
  处理论文 1/3: Efficient Attention: A New Approach for Long Sequence Processing
  处理论文 2/3: Sparse Transformer: Efficient Attention with Sparse Patterns
  处理论文 3/3: Linformer: Self-Attention with Linear Complexity

串行处理总耗时: 4.05 秒

加速比: 1.01x
```

---

## 代码说明

### 1. Map-Reduce 核心实现

```python
def process(self, documents, map_func, reduce_func):
    """执行 Map-Reduce"""

    # Map 阶段：并行处理
    map_results = self._map_parallel(documents, map_func)

    # Reduce 阶段：聚合结果
    final_result = reduce_func(map_results)

    return final_result
```

**关键点**：
- Map 阶段使用 ThreadPoolExecutor 并行处理
- Reduce 阶段串行执行（聚合结果）
- 保持结果顺序（使用索引映射）

### 2. 并行处理实现

```python
def _map_parallel(self, documents, map_func):
    """并行执行 Map"""

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
            result = future.result()
            results[idx] = result
```

**关键点**：
- 使用 ThreadPoolExecutor 实现并行
- 使用 future_to_doc 映射保持顺序
- 使用 as_completed 尽快收集结果

### 3. Map 函数设计

```python
def map_extract_aspect(paper: Paper) -> Dict:
    """提取单篇论文的指定方面"""

    # 1. 选择相关内容
    if aspect == "方法":
        content = paper.method_section
    else:
        content = paper.content

    # 2. 调用 LLM 提取
    prompt = f"请提取这篇论文的{aspect}：\n\n{content}"
    result = self.llm_func(prompt)

    # 3. 返回结构化结果
    return {
        "paper_id": paper.id,
        "paper_title": paper.title,
        "aspect": aspect,
        "content": result
    }
```

**关键点**：
- 每个 Map 函数处理一篇论文
- 返回结构化结果（包含论文信息）
- 独立执行（不依赖其他论文）

### 4. Reduce 函数设计

```python
def reduce_compare(results: List[Dict]) -> str:
    """对比所有论文的结果"""

    # 1. 构建对比内容
    comparison_text = []
    for result in results:
        comparison_text.append(
            f"{result['paper_title']}\n{result['content']}"
        )

    # 2. 调用 LLM 对比
    combined = "\n\n---\n\n".join(comparison_text)
    prompt = f"请对比以下论文的{aspect}：\n\n{combined}"

    return self.llm_func(prompt)
```

**关键点**：
- 聚合所有 Map 结果
- 调用 LLM 进行对比
- 返回最终对比结果

---

## 性能分析

### 并行 vs 串行

```
场景：处理 3 篇论文，每篇耗时 1 秒

串行处理：
- Map 阶段：3 × 1秒 = 3秒
- Reduce 阶段：1秒
- 总耗时：4秒

并行处理（3 workers）：
- Map 阶段：1秒（并行）
- Reduce 阶段：1秒
- 总耗时：2秒

加速比：4 / 2 = 2倍
```

### 加速比计算

```python
# 理论加速比
speedup = (n × t_map + t_reduce) / (t_map + t_reduce)

# 其中：
# n = 文档数量
# t_map = 单个文档的 Map 耗时
# t_reduce = Reduce 耗时

# 示例：
# n = 3, t_map = 1秒, t_reduce = 1秒
# speedup = (3 × 1 + 1) / (1 + 1) = 4 / 2 = 2倍
```

### 实际加速比

```
实际加速比通常低于理论值，原因：
1. 线程创建和管理开销
2. GIL（全局解释器锁）限制（Python）
3. Reduce 阶段的瓶颈
4. 网络延迟的不确定性
```

---

## 在 RAG 开发中的应用

### 应用1：多文档问答

```python
def answer_from_multiple_docs(documents: List[str], question: str):
    """从多个文档中回答问题"""

    # Map 函数：在每个文档中查找答案
    def map_find_answer(doc: str) -> str:
        prompt = f"根据以下文档回答问题：\n\n{doc}\n\n问题：{question}"
        return llm(prompt)

    # Reduce 函数：汇总所有答案
    def reduce_aggregate(answers: List[str]) -> str:
        combined = "\n\n".join([f"文档{i+1}的答案：\n{a}" for i, a in enumerate(answers)])
        prompt = f"汇总以下答案：\n\n{combined}"
        return llm(prompt)

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(documents, map_find_answer, reduce_aggregate)
```

### 应用2：批量摘要生成

```python
def batch_summarize(documents: List[str]):
    """批量生成摘要"""

    # Map 函数：生成单个文档的摘要
    def map_summarize(doc: str) -> str:
        prompt = f"请总结以下文档：\n\n{doc}"
        return llm(prompt)

    # Reduce 函数：汇总所有摘要
    def reduce_aggregate(summaries: List[str]) -> str:
        return "\n\n".join([f"文档{i+1}摘要：\n{s}" for i, s in enumerate(summaries)])

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(documents, map_summarize, reduce_aggregate)
```

### 应用3：多语言翻译

```python
def translate_multiple_docs(documents: List[str], target_lang: str):
    """批量翻译文档"""

    # Map 函数：翻译单个文档
    def map_translate(doc: str) -> str:
        prompt = f"请将以下文档翻译成{target_lang}：\n\n{doc}"
        return llm(prompt)

    # Reduce 函数：简单拼接
    def reduce_concat(translations: List[str]) -> List[str]:
        return translations

    # 执行 Map-Reduce
    mr = MapReduce()
    return mr.process(documents, map_translate, reduce_concat)
```

---

## 总结

**这个示例展示了**：
1. 如何实现基础的 Map-Reduce 系统
2. 如何并行处理多个文档
3. 如何聚合处理结果
4. 如何计算加速比

**关键要点**：
- Map 阶段并行处理，Reduce 阶段串行聚合
- 使用 ThreadPoolExecutor 实现并行
- 保持结果顺序很重要
- 适合可以分解为独立子问题的场景

**性能优势**：
- 3 个文档并行处理，加速比约 2 倍
- 10 个文档并行处理，加速比约 5-6 倍
- 文档越多，加速比越明显

---

**下一步：** [09_实战代码_场景3b_多文档对比_异步优化.md](./09_实战代码_场景3b_多文档对比_异步优化.md) - 异步优化版本
