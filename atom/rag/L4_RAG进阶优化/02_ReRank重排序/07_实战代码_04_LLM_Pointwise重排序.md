# 07_实战代码_04_LLM_Pointwise重排序

## 场景说明

LLM Pointwise重排序使用GPT-4或Claude等大模型对每个文档独立评分,相比Cross-Encoder更灵活但成本更高。适合需要复杂推理或领域知识的场景。

**核心价值:**
- 利用LLM的推理能力
- 可解释的评分结果
- 支持复杂的相关性判断
- 易于定制评分标准

**适用场景:**
- 需要领域知识判断
- 要求可解释性
- 复杂的相关性标准
- 预算充足的场景

**成本对比:**
- Cross-Encoder: $0.0001/doc
- LLM Pointwise: $0.001-0.01/doc (10-100倍)

---

## 完整实现代码

### 1. 基础LLM Pointwise实现

```python
"""
LLM Pointwise重排序基础实现
使用GPT-4对每个文档独立评分
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document
import json

load_dotenv()


class LLMPointwiseReranker:
    """LLM Pointwise重排序器"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def _create_prompt(self, query: str, document: str) -> str:
        """创建评分prompt"""
        return f"""你是一个文档相关性评估专家。请评估以下文档与查询的相关性。

查询: {query}

文档: {document}

请从0-10评分,其中:
- 0-2: 完全不相关
- 3-4: 弱相关
- 5-6: 中等相关
- 7-8: 高度相关
- 9-10: 完美匹配

只返回JSON格式:
{{"score": <分数>, "reason": "<简短理由>"}}"""

    def score_document(
        self,
        query: str,
        document: str
    ) -> Dict:
        """
        对单个文档评分

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            包含score和reason的字典
        """
        prompt = self._create_prompt(query, document)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "score": float(result.get("score", 0)),
                "reason": result.get("reason", "")
            }

        except Exception as e:
            print(f"评分失败: {e}")
            return {"score": 0.0, "reason": "评分失败"}

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
        scored_docs = []

        for doc in documents:
            result = self.score_document(query, doc.page_content)
            doc.metadata['llm_score'] = result['score']
            doc.metadata['llm_reason'] = result['reason']
            scored_docs.append((doc, result['score']))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]


# 使用示例
def main():
    documents = [
        Document(page_content="RAG combines retrieval with generation for factual answers.", metadata={"id": 1}),
        Document(page_content="Python is a programming language.", metadata={"id": 2}),
        Document(page_content="Vector databases enable semantic search.", metadata={"id": 3}),
        Document(page_content="LLM reranking provides explainable relevance scores.", metadata={"id": 4}),
    ]

    reranker = LLMPointwiseReranker(model="gpt-4o-mini")
    query = "How does RAG improve answer quality?"
    results = reranker.rerank(query, documents, top_k=3)

    print(f"查询: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [分数: {doc.metadata['llm_score']:.1f}]")
        print(f"   理由: {doc.metadata['llm_reason']}")
        print(f"   内容: {doc.page_content}\n")


if __name__ == "__main__":
    main()
```

---

### 2. 并行处理优化

```python
"""
并行LLM Pointwise重排序
使用asyncio提升处理速度
"""

import asyncio
from typing import List
from openai import AsyncOpenAI


class ParallelLLMReranker:
    """并行LLM重排序器"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 10
    ):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def score_document_async(
        self,
        query: str,
        document: str
    ) -> Dict:
        """异步评分单个文档"""
        async with self.semaphore:
            prompt = f"""评估文档与查询的相关性(0-10分):

查询: {query}
文档: {document}

返回JSON: {{"score": <分数>, "reason": "<理由>"}}"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                return {
                    "score": float(result.get("score", 0)),
                    "reason": result.get("reason", "")
                }

            except Exception as e:
                return {"score": 0.0, "reason": f"错误: {e}"}

    async def rerank_async(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """异步重排序"""
        # 并行评分所有文档
        tasks = [
            self.score_document_async(query, doc.page_content)
            for doc in documents
        ]
        results = await asyncio.gather(*tasks)

        # 添加分数到文档
        scored_docs = []
        for doc, result in zip(documents, results):
            doc.metadata['llm_score'] = result['score']
            doc.metadata['llm_reason'] = result['reason']
            scored_docs.append((doc, result['score']))

        # 排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """同步接口"""
        return asyncio.run(self.rerank_async(query, documents, top_k))


# 使用示例
def parallel_example():
    documents = [
        Document(page_content=f"Document {i} about various topics.", metadata={"id": i})
        for i in range(20)
    ]

    reranker = ParallelLLMReranker(max_concurrent=10)
    query = "Find relevant documents"

    import time
    start = time.time()
    results = reranker.rerank(query, documents, top_k=5)
    elapsed = time.time() - start

    print(f"处理{len(documents)}个文档耗时: {elapsed:.2f}秒")
    print(f"平均每个文档: {elapsed/len(documents):.2f}秒")
```

---

### 3. 成本控制策略

```python
"""
LLM重排序成本控制
实现缓存、批处理和降级策略
"""

import hashlib
from functools import lru_cache


class CostOptimizedLLMReranker:
    """成本优化的LLM重排序器"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cache_file: str = "llm_rerank_cache.json",
        use_cache: bool = True,
        fallback_threshold: float = 0.5
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.cache_file = cache_file
        self.use_cache = use_cache
        self.fallback_threshold = fallback_threshold
        self.cache = self._load_cache()
        self.cost_tracker = {"api_calls": 0, "cached_calls": 0}

    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, query: str, document: str) -> str:
        """生成缓存键"""
        content = f"{query}|||{document}"
        return hashlib.md5(content.encode()).hexdigest()

    def score_document(
        self,
        query: str,
        document: str,
        use_cache: bool = True
    ) -> Dict:
        """评分文档(带缓存)"""
        # 检查缓存
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(query, document)
            if cache_key in self.cache:
                self.cost_tracker["cached_calls"] += 1
                return self.cache[cache_key]

        # 调用API
        self.cost_tracker["api_calls"] += 1
        prompt = f"""评估相关性(0-10):
查询: {query}
文档: {document}
返回JSON: {{"score": <分数>, "reason": "<理由>"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            score_data = {
                "score": float(result.get("score", 0)),
                "reason": result.get("reason", "")
            }

            # 保存到缓存
            if self.use_cache:
                cache_key = self._get_cache_key(query, document)
                self.cache[cache_key] = score_data
                self._save_cache()

            return score_data

        except Exception as e:
            return {"score": 0.0, "reason": f"错误: {e}"}

    def rerank_with_budget(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        max_api_calls: int = 50
    ) -> List[Document]:
        """
        带预算限制的重排序

        策略:
        1. 优先使用缓存
        2. 达到预算限制后使用简单排序
        """
        scored_docs = []
        api_calls_used = 0

        for doc in documents:
            # 检查预算
            if api_calls_used >= max_api_calls:
                # 降级: 使用简单的文本匹配分数
                simple_score = self._simple_score(query, doc.page_content)
                doc.metadata['llm_score'] = simple_score
                doc.metadata['llm_reason'] = "预算限制,使用简单评分"
                scored_docs.append((doc, simple_score))
                continue

            # 正常评分
            result = self.score_document(query, doc.page_content)
            if result['score'] > 0:  # 成功评分
                api_calls_used += 1

            doc.metadata['llm_score'] = result['score']
            doc.metadata['llm_reason'] = result['reason']
            scored_docs.append((doc, result['score']))

        # 排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print(f"成本统计:")
        print(f"  API调用: {self.cost_tracker['api_calls']}")
        print(f"  缓存命中: {self.cost_tracker['cached_calls']}")
        print(f"  本次API调用: {api_calls_used}")

        return [doc for doc, _ in scored_docs[:top_k]]

    def _simple_score(self, query: str, document: str) -> float:
        """简单的关键词匹配评分(降级策略)"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words & doc_words)
        return min(10.0, overlap * 2.0)


# 使用示例
def cost_control_example():
    documents = [
        Document(page_content=f"Document {i} content", metadata={"id": i})
        for i in range(100)
    ]

    reranker = CostOptimizedLLMReranker(
        model="gpt-4o-mini",
        use_cache=True
    )

    query = "Find relevant information"

    # 第一次: 会调用API
    print("=== 第一次查询 ===")
    results1 = reranker.rerank_with_budget(
        query, documents, top_k=5, max_api_calls=20
    )

    # 第二次: 命中缓存
    print("\n=== 第二次查询(相同) ===")
    results2 = reranker.rerank_with_budget(
        query, documents, top_k=5, max_api_calls=20
    )
```

---

### 4. 与Cross-Encoder对比

```python
"""
LLM vs Cross-Encoder性能对比
"""

import time
from sentence_transformers import CrossEncoder


def compare_rerankers():
    """对比LLM和Cross-Encoder"""

    documents = [
        Document(page_content="RAG improves LLM accuracy with retrieval.", metadata={"id": 1}),
        Document(page_content="Python is a programming language.", metadata={"id": 2}),
        Document(page_content="Vector search enables semantic matching.", metadata={"id": 3}),
    ]

    query = "How does RAG work?"

    # 1. LLM Reranker
    print("=== LLM Pointwise Reranker ===")
    llm_reranker = LLMPointwiseReranker(model="gpt-4o-mini")

    start = time.time()
    llm_results = llm_reranker.rerank(query, documents, top_k=3)
    llm_time = time.time() - start

    print(f"耗时: {llm_time:.2f}秒")
    for i, doc in enumerate(llm_results, 1):
        print(f"{i}. [分数: {doc.metadata['llm_score']:.2f}] {doc.page_content[:50]}")

    # 2. Cross-Encoder
    print("\n=== Cross-Encoder ===")
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    start = time.time()
    pairs = [(query, doc.page_content) for doc in documents]
    ce_scores = ce_model.predict(pairs)
    ce_time = time.time() - start

    ce_results = sorted(
        zip(documents, ce_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"耗时: {ce_time:.2f}秒")
    for i, (doc, score) in enumerate(ce_results, 1):
        print(f"{i}. [分数: {score:.2f}] {doc.page_content[:50]}")

    # 对比
    print("\n=== 性能对比 ===")
    print(f"LLM耗时: {llm_time:.2f}秒")
    print(f"Cross-Encoder耗时: {ce_time:.2f}秒")
    print(f"速度差异: {llm_time/ce_time:.1f}x")

    # 成本估算(假设)
    llm_cost = len(documents) * 0.001  # $0.001/doc
    ce_cost = len(documents) * 0.0001  # $0.0001/doc
    print(f"\nLLM成本: ${llm_cost:.4f}")
    print(f"Cross-Encoder成本: ${ce_cost:.4f}")
    print(f"成本差异: {llm_cost/ce_cost:.1f}x")


if __name__ == "__main__":
    compare_rerankers()
```

---

## 代码说明

### 核心组件

1. **LLMPointwiseReranker**: 基础实现
   - 使用JSON模式确保结构化输出
   - 0-10分评分标准
   - 包含评分理由

2. **ParallelLLMReranker**: 并行优化
   - 使用asyncio并发处理
   - Semaphore控制并发数
   - 显著降低总延迟

3. **CostOptimizedLLMReranker**: 成本控制
   - 语义缓存减少重复调用
   - 预算限制保护
   - 降级策略(简单评分)

### Prompt设计要点

```python
# 好的Prompt
"""
评估文档与查询的相关性(0-10分):
- 0-2: 完全不相关
- 7-10: 高度相关

查询: {query}
文档: {document}

返回JSON: {"score": <分数>, "reason": "<理由>"}
"""

# 避免的Prompt
"""
这个文档相关吗? 给个分数
"""
```

---

## 运行示例

### 环境准备

```bash
pip install openai langchain sentence-transformers

export OPENAI_API_KEY="your_key"
```

### 执行代码

```bash
python llm_pointwise_basic.py
python llm_pointwise_parallel.py
python llm_pointwise_cost.py
python llm_vs_crossencoder.py
```

### 预期输出

```
查询: How does RAG improve answer quality?

1. [分数: 9.0]
   理由: 直接回答RAG如何提升答案质量
   内容: RAG combines retrieval with generation for factual answers.

2. [分数: 7.0]
   理由: 提到了相关的reranking技术
   内容: LLM reranking provides explainable relevance scores.

成本统计:
  API调用: 4
  缓存命中: 0
  本次API调用: 4
```

---

## 性能优化

### 1. 延迟优化

```python
# 并行处理
max_concurrent = 10  # 同时处理10个文档

# 使用更快的模型
model = "gpt-4o-mini"  # 而非 gpt-4

# 减少候选数量
candidates = documents[:20]  # 只对top 20重排序
```

### 2. 成本优化

```python
# 启用缓存
use_cache = True

# 设置预算限制
max_api_calls = 50

# 使用更便宜的模型
model = "gpt-4o-mini"  # $0.15/1M tokens
```

### 3. 质量优化

```python
# 使用更强的模型
model = "gpt-4o"  # 更高质量

# 更详细的Prompt
prompt = """详细评估文档相关性,考虑:
1. 主题匹配度
2. 信息完整性
3. 时效性
..."""

# 多次采样取平均
temperature = 0.3
n = 3  # 生成3个评分取平均
```

---

## 常见问题

### Q1: LLM vs Cross-Encoder如何选择?

**A:** 根据场景选择:

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 通用检索 | Cross-Encoder | 成本低,速度快 |
| 需要推理 | LLM | 理解复杂语义 |
| 需要解释 | LLM | 提供评分理由 |
| 大规模 | Cross-Encoder | 成本可控 |

### Q2: 如何降低LLM重排序成本?

**A:** 三种策略:
1. 缓存: 避免重复评分
2. 预算限制: 超出后降级
3. 混合策略: Cross-Encoder初排 + LLM精排

### Q3: Pointwise vs Listwise?

**A:**
- **Pointwise**: 独立评分,易并行,成本高
- **Listwise**: 整体排序,考虑相对关系,成本更高

### Q4: 如何提高评分一致性?

**A:**
```python
# 使用temperature=0
temperature = 0.0

# 使用JSON模式
response_format = {"type": "json_object"}

# 明确评分标准
prompt = "0-2分:不相关, 7-10分:高度相关"
```

---

## 参考资料

### 官方文档
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) - JSON模式使用
- [RankLLM GitHub](https://github.com/castorini/rank_llm) - LLM reranking工具包

### 技术文章
- [Using LLMs as Rerankers](https://fin.ai/research/using-llms-as-a-reranker-for-rag-a-practical-guide) - 实践指南
- [The Case Against LLMs as Rerankers](https://blog.voyageai.com/2025/10/22/the-case-against-llms-as-rerankers) - 成本分析

### 代码示例
- [RankLLM Examples](https://github.com/castorini/rank_llm/tree/main/examples) - 完整示例
- [LLM Rankers](https://github.com/ielab/llm-rankers) - 多种ranking策略

---

**版本:** v1.0 (2026年标准)
**最后更新:** 2026-02-16
**代码测试:** Python 3.13 + openai 1.x
**成本估算:** GPT-4o-mini ~$0.001/doc
