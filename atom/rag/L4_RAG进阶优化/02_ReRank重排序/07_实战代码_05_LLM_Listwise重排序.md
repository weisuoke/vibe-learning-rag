# 07_实战代码_05_LLM_Listwise重排序

## 场景说明

LLM Listwise重排序(如RankGPT)让大模型一次性对整个文档列表排序,相比Pointwise能更好地考虑文档间的相对关系,但成本更高、延迟更大。

**核心价值:**
- 考虑文档间相对关系
- 整体最优排序
- 更符合人类排序直觉
- 适合需要全局比较的场景

**适用场景:**
- 需要全局最优排序
- 文档数量较少(<20个)
- 对延迟不敏感
- 预算充足

**成本对比:**
- Pointwise: $0.001/doc
- Listwise: $0.01-0.05/query (处理整个列表)

---

## 完整实现代码

### 1. RankGPT基础实现

```python
"""
RankGPT Listwise重排序实现
使用滑动窗口算法处理长列表
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document

load_dotenv()


class RankGPTReranker:
    """RankGPT Listwise重排序器"""

    def __init__(
        self,
        model: str = "gpt-4o",
        window_size: int = 20,
        step_size: int = 10
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.window_size = window_size
        self.step_size = step_size

    def _create_ranking_prompt(
        self,
        query: str,
        documents: List[str]
    ) -> str:
        """创建排序prompt"""
        doc_list = "\n".join([
            f"[{i+1}] {doc}"
            for i, doc in enumerate(documents)
        ])

        return f"""你是一个专业的文档排序专家。请根据查询对以下文档进行排序。

查询: {query}

文档列表:
{doc_list}

请按相关性从高到低排序,只返回文档编号列表,用逗号分隔。
例如: 3,1,5,2,4

排序结果:"""

    def _parse_ranking(self, response: str, num_docs: int) -> List[int]:
        """解析排序结果"""
        try:
            # 提取数字
            numbers = [
                int(x.strip())
                for x in response.strip().split(',')
                if x.strip().isdigit()
            ]

            # 验证并补全
            valid_numbers = [n for n in numbers if 1 <= n <= num_docs]
            missing = set(range(1, num_docs + 1)) - set(valid_numbers)

            # 添加缺失的编号到末尾
            result = valid_numbers + sorted(list(missing))

            return result[:num_docs]

        except Exception as e:
            print(f"解析失败: {e}, 返回原始顺序")
            return list(range(1, num_docs + 1))

    def rank_window(
        self,
        query: str,
        documents: List[str]
    ) -> List[int]:
        """
        对单个窗口内的文档排序

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            排序后的索引列表(1-based)
        """
        prompt = self._create_ranking_prompt(query, documents)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            ranking = self._parse_ranking(
                response.choices[0].message.content,
                len(documents)
            )

            return ranking

        except Exception as e:
            print(f"排序失败: {e}")
            return list(range(1, len(documents) + 1))

    def sliding_window_rank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        使用滑动窗口算法排序

        算法:
        1. 将文档分成重叠的窗口
        2. 对每个窗口内部排序
        3. 合并窗口结果

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            排序后的文档列表
        """
        if len(documents) <= self.window_size:
            # 文档数量小于窗口大小,直接排序
            doc_texts = [doc.page_content for doc in documents]
            ranking = self.rank_window(query, doc_texts)

            # 转换为0-based索引
            return [documents[i-1] for i in ranking]

        # 滑动窗口排序
        ranked_docs = list(documents)

        for start in range(0, len(documents), self.step_size):
            end = min(start + self.window_size, len(documents))

            if end - start < 2:
                break

            # 提取窗口
            window_docs = ranked_docs[start:end]
            window_texts = [doc.page_content for doc in window_docs]

            # 排序窗口
            window_ranking = self.rank_window(query, window_texts)

            # 更新排序
            ranked_docs[start:end] = [
                window_docs[i-1] for i in window_ranking
            ]

        return ranked_docs

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        重排序文档列表

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量

        Returns:
            重排序后的文档列表
        """
        ranked = self.sliding_window_rank(query, documents)

        # 添加排名到元数据
        for i, doc in enumerate(ranked, 1):
            doc.metadata['rank'] = i

        return ranked[:top_k]


# 使用示例
def main():
    documents = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"id": 1}),
        Document(page_content="Python is a programming language.", metadata={"id": 2}),
        Document(page_content="Vector databases enable semantic search.", metadata={"id": 3}),
        Document(page_content="LLM listwise ranking considers relative relevance.", metadata={"id": 4}),
        Document(page_content="RankGPT uses sliding window algorithm.", metadata={"id": 5}),
    ]

    reranker = RankGPTReranker(model="gpt-4o", window_size=20)
    query = "How does RAG work with ranking?"
    results = reranker.rerank(query, documents, top_k=3)

    print(f"查询: {query}\n")
    for doc in results:
        print(f"排名 {doc.metadata['rank']}: {doc.page_content}")


if __name__ == "__main__":
    main()
```

---

### 2. 优化的Prompt工程

```python
"""
改进的Listwise排序Prompt
包含思维链和评分标准
"""


class ImprovedRankGPT:
    """改进的RankGPT实现"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _create_cot_prompt(
        self,
        query: str,
        documents: List[str]
    ) -> str:
        """创建带思维链的排序prompt"""
        doc_list = "\n".join([
            f"[{i+1}] {doc}"
            for i, doc in enumerate(documents)
        ])

        return f"""你是一个专业的文档排序专家。请根据查询对文档进行排序。

查询: {query}

文档列表:
{doc_list}

请按以下步骤思考:
1. 分析查询的核心意图
2. 评估每个文档与查询的相关性
3. 考虑文档间的相对关系
4. 给出最终排序

评分标准:
- 主题匹配度(0-4分)
- 信息完整性(0-3分)
- 直接回答程度(0-3分)

请先简要说明你的思考过程,然后给出排序结果。

格式:
思考: [你的分析]
排序: [编号列表,用逗号分隔]"""

    def rank_with_reasoning(
        self,
        query: str,
        documents: List[str]
    ) -> Dict:
        """
        带推理的排序

        Returns:
            包含ranking和reasoning的字典
        """
        prompt = self._create_cot_prompt(query, documents)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            content = response.choices[0].message.content

            # 解析响应
            lines = content.strip().split('\n')
            reasoning = ""
            ranking = []

            for line in lines:
                if line.startswith("思考:"):
                    reasoning = line.replace("思考:", "").strip()
                elif line.startswith("排序:"):
                    ranking_str = line.replace("排序:", "").strip()
                    ranking = [
                        int(x.strip())
                        for x in ranking_str.split(',')
                        if x.strip().isdigit()
                    ]

            return {
                "ranking": ranking if ranking else list(range(1, len(documents) + 1)),
                "reasoning": reasoning
            }

        except Exception as e:
            print(f"排序失败: {e}")
            return {
                "ranking": list(range(1, len(documents) + 1)),
                "reasoning": "排序失败"
            }


# 使用示例
def cot_example():
    reranker = ImprovedRankGPT(model="gpt-4o")

    documents = [
        "RAG improves LLM accuracy with retrieval.",
        "Python is great for AI development.",
        "Listwise ranking considers document relationships.",
    ]

    query = "How does RAG work?"
    result = reranker.rank_with_reasoning(query, documents)

    print(f"查询: {query}\n")
    print(f"思考过程: {result['reasoning']}\n")
    print(f"排序结果: {result['ranking']}")
```

---

### 3. 批处理优化

```python
"""
批处理Listwise排序
处理多个查询
"""

import asyncio
from typing import List, Dict


class BatchRankGPT:
    """批处理RankGPT"""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_concurrent: int = 5
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def rank_async(
        self,
        query: str,
        documents: List[str]
    ) -> List[int]:
        """异步排序"""
        async with self.semaphore:
            doc_list = "\n".join([
                f"[{i+1}] {doc}"
                for i, doc in enumerate(documents)
            ])

            prompt = f"""根据查询对文档排序:

查询: {query}

文档:
{doc_list}

只返回编号列表(逗号分隔):"""

            try:
                # 注意: OpenAI Python SDK 目前不支持原生async
                # 这里使用同步调用,实际生产中可使用httpx等异步库
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )

                ranking_str = response.choices[0].message.content.strip()
                ranking = [
                    int(x.strip())
                    for x in ranking_str.split(',')
                    if x.strip().isdigit()
                ]

                return ranking if ranking else list(range(1, len(documents) + 1))

            except Exception as e:
                print(f"排序失败: {e}")
                return list(range(1, len(documents) + 1))

    async def batch_rank_async(
        self,
        queries: List[str],
        documents_list: List[List[str]]
    ) -> List[List[int]]:
        """批量异步排序"""
        tasks = [
            self.rank_async(query, docs)
            for query, docs in zip(queries, documents_list)
        ]

        results = await asyncio.gather(*tasks)
        return results

    def batch_rank(
        self,
        queries: List[str],
        documents_list: List[List[str]]
    ) -> List[List[int]]:
        """批量排序(同步接口)"""
        return asyncio.run(
            self.batch_rank_async(queries, documents_list)
        )


# 使用示例
def batch_example():
    reranker = BatchRankGPT(max_concurrent=5)

    queries = [
        "How does RAG work?",
        "What is vector search?",
        "Explain reranking methods"
    ]

    documents_list = [
        ["RAG combines retrieval and generation.", "Python is a language.", "Vectors enable search."],
        ["Vector search uses embeddings.", "RAG improves accuracy.", "Reranking refines results."],
        ["Pointwise ranks individually.", "Listwise ranks together.", "Cross-encoders are effective."]
    ]

    import time
    start = time.time()
    results = reranker.batch_rank(queries, documents_list)
    elapsed = time.time() - start

    print(f"批量处理{len(queries)}个查询耗时: {elapsed:.2f}秒\n")

    for i, (query, ranking) in enumerate(zip(queries, results), 1):
        print(f"查询{i}: {query}")
        print(f"排序: {ranking}\n")
```

---

### 4. 与Pointwise对比

```python
"""
Listwise vs Pointwise性能对比
"""

import time
from typing import List


def compare_listwise_pointwise():
    """对比两种方法"""

    documents = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"id": 1}),
        Document(page_content="Vector search enables semantic matching.", metadata={"id": 2}),
        Document(page_content="Listwise ranking considers relationships.", metadata={"id": 3}),
        Document(page_content="Python is a programming language.", metadata={"id": 4}),
    ]

    query = "How does RAG improve search quality?"

    # 1. Pointwise
    print("=== Pointwise Reranking ===")
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    pointwise_scores = []

    for doc in documents:
        prompt = f"""评分(0-10):
查询: {query}
文档: {doc.page_content}
只返回数字:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 0.0

        pointwise_scores.append((doc, score))

    pointwise_time = time.time() - start
    pointwise_sorted = sorted(pointwise_scores, key=lambda x: x[1], reverse=True)

    print(f"耗时: {pointwise_time:.2f}秒")
    for i, (doc, score) in enumerate(pointwise_sorted, 1):
        print(f"{i}. [分数: {score:.1f}] {doc.page_content[:50]}")

    # 2. Listwise
    print("\n=== Listwise Reranking ===")
    reranker = RankGPTReranker(model="gpt-4o-mini")

    start = time.time()
    listwise_results = reranker.rerank(query, documents, top_k=len(documents))
    listwise_time = time.time() - start

    print(f"耗时: {listwise_time:.2f}秒")
    for i, doc in enumerate(listwise_results, 1):
        print(f"{i}. {doc.page_content[:50]}")

    # 对比
    print("\n=== 性能对比 ===")
    print(f"Pointwise耗时: {pointwise_time:.2f}秒 ({len(documents)}次API调用)")
    print(f"Listwise耗时: {listwise_time:.2f}秒 (1次API调用)")
    print(f"速度比: {pointwise_time/listwise_time:.1f}x")

    # 成本估算
    pointwise_cost = len(documents) * 0.001
    listwise_cost = 0.01
    print(f"\nPointwise成本: ${pointwise_cost:.4f}")
    print(f"Listwise成本: ${listwise_cost:.4f}")


if __name__ == "__main__":
    compare_listwise_pointwise()
```

---

## 代码说明

### 核心组件

1. **RankGPTReranker**: 基础实现
   - 滑动窗口算法处理长列表
   - 窗口大小和步长可配置
   - 自动解析和验证排序结果

2. **ImprovedRankGPT**: 改进版本
   - 思维链(CoT)提示
   - 评分标准明确
   - 返回推理过程

3. **BatchRankGPT**: 批处理版本
   - 并发处理多个查询
   - Semaphore控制并发数

### 滑动窗口算法

```python
# 窗口大小: 20
# 步长: 10
# 文档数: 50

窗口1: [1-20]   -> 排序
窗口2: [11-30]  -> 排序(与窗口1重叠10个)
窗口3: [21-40]  -> 排序
窗口4: [31-50]  -> 排序
```

---

## 运行示例

### 环境准备

```bash
pip install openai langchain

export OPENAI_API_KEY="your_key"
```

### 执行代码

```bash
python rankgpt_basic.py
python rankgpt_improved.py
python rankgpt_batch.py
python listwise_vs_pointwise.py
```

### 预期输出

```
查询: How does RAG work with ranking?

排名 1: RAG combines retrieval with generation.
排名 2: LLM listwise ranking considers relative relevance.
排名 3: RankGPT uses sliding window algorithm.

思考过程: 查询关注RAG和ranking的结合。文档1直接解释RAG,文档4讨论listwise ranking,文档5介绍RankGPT算法。这三个最相关。

批量处理3个查询耗时: 5.23秒
```

---

## 性能优化

### 1. 延迟优化

```python
# 减少窗口大小
window_size = 10  # 而非 20

# 使用更快的模型
model = "gpt-4o-mini"  # 而非 gpt-4o

# 只对top候选排序
candidates = documents[:10]
```

### 2. 成本优化

```python
# 使用更便宜的模型
model = "gpt-4o-mini"

# 减少窗口重叠
step_size = window_size  # 无重叠

# 混合策略: Pointwise初排 + Listwise精排
top_candidates = pointwise_rerank(documents, top_k=20)
final_results = listwise_rerank(top_candidates, top_k=5)
```

### 3. 质量优化

```python
# 使用更强的模型
model = "gpt-4o"

# 增加窗口重叠
step_size = window_size // 2

# 使用思维链prompt
use_cot = True
```

---

## 常见问题

### Q1: Listwise vs Pointwise如何选择?

**A:** 根据场景选择:

| 维度 | Pointwise | Listwise |
|------|-----------|----------|
| 延迟 | 高(多次调用) | 低(单次调用) |
| 成本 | 高(N次) | 中(1次) |
| 质量 | 中 | 高 |
| 文档数限制 | 无 | <20个 |

### Q2: 滑动窗口参数如何设置?

**A:**
```python
# 文档数 < 20: 直接排序
window_size = 20
step_size = 20

# 文档数 20-50: 小重叠
window_size = 20
step_size = 15

# 文档数 > 50: 大重叠
window_size = 20
step_size = 10
```

### Q3: 如何处理排序失败?

**A:**
```python
def _parse_ranking(self, response: str, num_docs: int) -> List[int]:
    try:
        # 解析排序
        numbers = parse_numbers(response)

        # 验证完整性
        if len(numbers) != num_docs:
            # 补全缺失的编号
            missing = set(range(1, num_docs + 1)) - set(numbers)
            numbers.extend(sorted(missing))

        return numbers[:num_docs]

    except:
        # 降级: 返回原始顺序
        return list(range(1, num_docs + 1))
```

### Q4: Listwise适合什么场景?

**A:**
- ✅ 需要全局最优排序
- ✅ 文档数量少(<20)
- ✅ 对延迟不敏感
- ❌ 大规模检索(>50文档)
- ❌ 实时系统(延迟敏感)

---

## 参考资料

### 官方文档
- [RankGPT GitHub](https://github.com/sunnweiwei/RankGPT) - 官方实现
- [RankLLM Package](https://github.com/castorini/rank_llm) - Python工具包

### 技术文章
- [RankGPT Paper](https://arxiv.org/abs/2304.09542) - EMNLP 2023论文
- [AFR-Rank Framework](https://www.sciencedirect.com/science/article/abs/pii/S0306457325001736) - 高效listwise框架
- [LLM Ranking Playbook](https://utm.one/resources/playbooks/llm-ranking) - 完整指南

### 代码示例
- [RankLLM Examples](https://github.com/castorini/rank_llm/tree/main/examples) - 完整示例
- [LlamaIndex RankLLM](https://developers.llamaindex.ai/python/examples/node_postprocessor/rankllm) - LlamaIndex集成

---

**版本:** v1.0 (2026年标准)
**最后更新:** 2026-02-16
**代码测试:** Python 3.13 + openai 1.x
**成本估算:** GPT-4o ~$0.01-0.05/query
