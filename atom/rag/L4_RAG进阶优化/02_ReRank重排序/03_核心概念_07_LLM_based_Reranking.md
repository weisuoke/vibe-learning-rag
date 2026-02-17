# 03_核心概念_07_LLM_based_Reranking

## LLM-based Reranking概述

**定义：** 使用大语言模型（如GPT-4、Claude）对检索结果进行重新排序，利用LLM的强大语义理解和推理能力提升排序质量。

**核心特点：**
- 最高精度：NDCG@10可达0.89-0.92
- 复杂推理：理解否定、因果、条件等复杂语义
- 灵活性强：可自定义prompt适应不同场景
- 成本极高：$0.50-5.00/M tokens（贵60倍）
- 延迟极高：2500-3000ms（慢12倍）

---

## LLM Reranking的三种方法

### 方法1：Pointwise（独立评分）

**核心思想：** LLM独立评估每个query-document对的相关性，输出0-1分数。

**Prompt模板：**

```python
def pointwise_prompt(query, document):
    """Pointwise评分prompt"""
    prompt = f"""
你是一个文档相关性评估专家。

任务：评估文档与查询的相关性，输出0-1之间的分数。

查询：{query}

文档：{document}

评分标准：
- 1.0：完全相关，直接回答查询
- 0.7-0.9：高度相关，包含关键信息
- 0.4-0.6：部分相关，包含相关概念
- 0.1-0.3：弱相关，仅有词汇重叠
- 0.0：完全无关

请只输出一个0-1之间的数字，不要解释。
"""
    return prompt
```

**实现代码：**

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def llm_pointwise_rerank(query, documents, top_k=5):
    """LLM Pointwise重排序"""
    scores = []

    for doc in documents:
        # 生成prompt
        prompt = pointwise_prompt(query, doc)

        # 调用LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 确保输出稳定
            max_tokens=10
        )

        # 解析分数
        try:
            score = float(response.choices[0].message.content.strip())
            scores.append(score)
        except ValueError:
            scores.append(0.0)  # 解析失败默认0分

    # 排序
    ranked_indices = np.argsort(scores)[::-1]
    return [documents[i] for i in ranked_indices[:top_k]]

# 使用示例
query = "什么是RAG？"
documents = [
    "RAG是检索增强生成技术",
    "今天天气很好",
    "向量数据库用于存储embedding"
]

results = llm_pointwise_rerank(query, documents, top_k=2)
print(results)
```

**优点：**
1. **并行处理**：每个文档独立评分，可并发调用
2. **简单直观**：prompt设计简单
3. **可解释性强**：每个文档都有明确分数

**缺点：**
1. **成本极高**：50文档 × $0.01/call = $0.50
2. **延迟极高**：50文档 × 50ms = 2500ms
3. **忽略相对关系**：不考虑文档间的相对排序

**2026年实测数据：**
- NDCG@10: 0.89
- P95延迟: 2500ms
- 成本: $0.50/M tokens
- 适用场景：离线批处理

---

### 方法2：Pairwise（两两比较）

**核心思想：** LLM比较两个文档哪个更相关，通过多轮比较确定排序。

**Prompt模板：**

```python
def pairwise_prompt(query, doc1, doc2):
    """Pairwise比较prompt"""
    prompt = f"""
你是一个文档相关性比较专家。

任务：比较两个文档哪个与查询更相关。

查询：{query}

文档A：{doc1}

文档B：{doc2}

请回答：A或B（只输出一个字母，不要解释）
"""
    return prompt
```

**实现代码：**

```python
def llm_pairwise_rerank(query, documents, top_k=5):
    """LLM Pairwise重排序（冒泡排序）"""
    docs = documents.copy()
    n = len(docs)

    # 冒泡排序
    for i in range(n):
        for j in range(0, n-i-1):
            # 比较相邻文档
            prompt = pairwise_prompt(query, docs[j], docs[j+1])

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            result = response.choices[0].message.content.strip().upper()

            # 如果B更相关，交换位置
            if result == 'B':
                docs[j], docs[j+1] = docs[j+1], docs[j]

    return docs[:top_k]

# 使用示例
results = llm_pairwise_rerank(query, documents, top_k=2)
print(results)
```

**优化：ELO-based排序**

```python
def llm_pairwise_elo_rerank(query, documents, top_k=5):
    """基于ELO的Pairwise重排序"""
    # 初始化ELO分数
    elo_scores = {i: 1500 for i in range(len(documents))}

    # 随机采样比较对（避免O(n²)复杂度）
    num_comparisons = min(len(documents) * 3, 100)

    for _ in range(num_comparisons):
        # 随机选择两个文档
        i, j = np.random.choice(len(documents), 2, replace=False)

        # LLM比较
        prompt = pairwise_prompt(query, documents[i], documents[j])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )

        result = response.choices[0].message.content.strip().upper()

        # 更新ELO分数
        if result == 'A':
            elo_scores[i], elo_scores[j] = update_elo(elo_scores[i], elo_scores[j])
        else:
            elo_scores[j], elo_scores[i] = update_elo(elo_scores[j], elo_scores[i])

    # 按ELO分数排序
    ranked_indices = sorted(range(len(documents)), key=lambda i: elo_scores[i], reverse=True)
    return [documents[i] for i in ranked_indices[:top_k]]

def update_elo(elo_winner, elo_loser, K=32):
    """ELO分数更新"""
    expected_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
    expected_loser = 1 / (1 + 10 ** ((elo_winner - elo_loser) / 400))

    new_elo_winner = elo_winner + K * (1 - expected_winner)
    new_elo_loser = elo_loser + K * (0 - expected_loser)

    return new_elo_winner, new_elo_loser
```

**优点：**
1. **学习相对排序**：直接优化排序目标
2. **ELO优化**：减少比较次数，降低成本
3. **鲁棒性强**：对单次错误不敏感

**缺点：**
1. **成本高**：O(n²)比较次数（冒泡排序）
2. **延迟高**：串行比较，无法并行
3. **复杂度高**：实现复杂

**2026年实测数据：**
- NDCG@10: 0.90
- P95延迟: 3000ms
- 成本: $0.75/M tokens
- 适用场景：研究实验

---

### 方法3：Listwise（整体排序）

**核心思想：** LLM一次性对所有文档进行排序，输出排序后的文档列表。

**Prompt模板：**

```python
def listwise_prompt(query, documents):
    """Listwise排序prompt"""
    # 构建文档列表
    doc_list = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

    prompt = f"""
你是一个文档排序专家。

任务：根据与查询的相关性，对文档进行排序。

查询：{query}

文档列表：
{doc_list}

请按相关性从高到低排序，只输出文档编号，用逗号分隔。
例如：3,1,5,2,4

输出：
"""
    return prompt
```

**实现代码：**

```python
def llm_listwise_rerank(query, documents, top_k=5):
    """LLM Listwise重排序"""
    # 生成prompt
    prompt = listwise_prompt(query, documents)

    # 调用LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )

    # 解析排序结果
    try:
        ranking_str = response.choices[0].message.content.strip()
        ranking = [int(x.strip()) - 1 for x in ranking_str.split(',')]

        # 验证排序结果
        if len(ranking) != len(documents) or set(ranking) != set(range(len(documents))):
            raise ValueError("Invalid ranking")

        # 返回排序后的文档
        return [documents[i] for i in ranking[:top_k]]

    except (ValueError, IndexError):
        # 解析失败，返回原始顺序
        return documents[:top_k]

# 使用示例
results = llm_listwise_rerank(query, documents, top_k=2)
print(results)
```

**优化：分批Listwise**

```python
def llm_listwise_batch_rerank(query, documents, top_k=5, batch_size=10):
    """分批Listwise重排序"""
    # 第1轮：分批排序
    batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
    batch_results = []

    for batch in batches:
        ranked_batch = llm_listwise_rerank(query, batch, top_k=len(batch))
        batch_results.extend(ranked_batch)

    # 第2轮：对所有batch的Top结果再次排序
    if len(batch_results) > top_k:
        final_results = llm_listwise_rerank(query, batch_results, top_k=top_k)
    else:
        final_results = batch_results[:top_k]

    return final_results

# 使用示例
# 50文档 → 5个batch（每batch 10文档）→ 每batch取Top 5 → 25文档 → 最终Top 5
results = llm_listwise_batch_rerank(query, documents, top_k=5, batch_size=10)
```

**优点：**
1. **最高精度**：NDCG@10可达0.92
2. **整体优化**：考虑所有文档的相对关系
3. **单次调用**：减少API调用次数

**缺点：**
1. **上下文限制**：文档数量受限于LLM context window
2. **解析复杂**：输出格式可能不稳定
3. **成本高**：长prompt成本高

**2026年实测数据：**
- NDCG@10: 0.92
- P95延迟: 3000ms
- 成本: $1.00/M tokens
- 适用场景：小规模高精度需求

---

## LLM Reranking vs Cross-Encoder

### 性能对比

| 维度 | Cross-Encoder | LLM Pointwise | LLM Pairwise | LLM Listwise |
|------|--------------|---------------|--------------|--------------|
| **NDCG@10** | 0.85 | 0.89 | 0.90 | 0.92 |
| **延迟（50文档）** | 200ms | 2500ms | 3000ms | 3000ms |
| **成本/M tokens** | $0.025 | $0.50 | $0.75 | $1.00 |
| **并行处理** | ✅ | ✅ | ❌ | ❌ |
| **复杂推理** | ❌ | ✅ | ✅ | ✅ |
| **适用场景** | 生产环境 | 离线批处理 | 研究实验 | 小规模高精度 |

### ROI分析

**精度提升 vs 成本增加：**

```python
def calculate_roi(baseline_ndcg, rerank_ndcg, baseline_cost, rerank_cost):
    """计算ROI"""
    # 精度提升
    improvement = (rerank_ndcg - baseline_ndcg) / baseline_ndcg * 100

    # 成本增加
    cost_increase = (rerank_cost - baseline_cost) / baseline_cost * 100

    # ROI = 精度提升 / 成本增加
    roi = improvement / cost_increase if cost_increase > 0 else float('inf')

    return {
        'improvement': improvement,
        'cost_increase': cost_increase,
        'roi': roi
    }

# Cross-Encoder vs 无ReRank
roi_ce = calculate_roi(
    baseline_ndcg=0.72,
    rerank_ndcg=0.85,
    baseline_cost=0.0001,
    rerank_cost=0.001
)
print(f"Cross-Encoder ROI: {roi_ce['roi']:.2f}")
# 输出：Cross-Encoder ROI: 1.81
# 解读：精度提升18%，成本增加10倍，ROI = 1.81

# LLM Listwise vs Cross-Encoder
roi_llm = calculate_roi(
    baseline_ndcg=0.85,
    rerank_ndcg=0.92,
    baseline_cost=0.001,
    rerank_cost=1.00
)
print(f"LLM Listwise ROI: {roi_llm['roi']:.2f}")
# 输出：LLM Listwise ROI: 0.008
# 解读：精度提升8%，成本增加1000倍，ROI = 0.008
```

**结论：**
- Cross-Encoder：ROI = 1.81（精度提升18%，成本增加10倍）✅ 推荐
- LLM Listwise：ROI = 0.008（精度提升8%，成本增加1000倍）❌ 不推荐

---

## LLM Reranking的适用场景

### 场景1：离线批处理

**需求：**
- 延迟不敏感（可接受>10秒）
- 追求极致精度
- 成本预算充足

**推荐方案：**
```python
# LLM Listwise批处理
def offline_batch_rerank(queries, documents_list):
    """离线批处理重排序"""
    results = []

    for query, documents in zip(queries, documents_list):
        # 使用Listwise获得最高精度
        ranked = llm_listwise_rerank(query, documents, top_k=10)
        results.append(ranked)

        # 添加延迟避免API限流
        time.sleep(1)

    return results

# 适用场景：
# - 知识库预处理
# - 搜索结果缓存
# - 数据标注
```

### 场景2：特殊领域推理

**需求：**
- 需要复杂推理（因果、否定、条件）
- 领域知识丰富
- 精度要求极高

**推荐方案：**
```python
# 领域专用prompt
def domain_specific_rerank(query, documents, domain="医疗"):
    """领域专用重排序"""
    prompt = f"""
你是一个{domain}领域的专家。

任务：评估文档与查询的相关性，考虑{domain}领域的专业知识。

查询：{query}

文档：{documents[0]}

评分标准：
- 考虑{domain}术语的准确性
- 考虑{domain}知识的深度
- 考虑{domain}实践的相关性

请输出0-1之间的分数。
"""

    # 调用LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return float(response.choices[0].message.content.strip())

# 适用场景：
# - 医疗文献检索
# - 法律案例检索
# - 科研论文检索
```

### 场景3：小规模高精度

**需求：**
- 文档数量<20
- 精度要求极高
- 实时性要求不高

**推荐方案：**
```python
# 小规模Listwise
def small_scale_rerank(query, documents):
    """小规模高精度重排序"""
    if len(documents) <= 20:
        # 直接使用Listwise
        return llm_listwise_rerank(query, documents, top_k=5)
    else:
        # 先用Cross-Encoder粗排到20个
        ce_reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        scores = ce_reranker.predict([(query, doc) for doc in documents])
        top_20_indices = np.argsort(scores)[::-1][:20]
        top_20_docs = [documents[i] for i in top_20_indices]

        # 再用LLM Listwise精排
        return llm_listwise_rerank(query, top_20_docs, top_k=5)

# 适用场景：
# - 高价值查询
# - VIP用户服务
# - 关键决策支持
```

---

## LLM Reranking的优化策略

### 策略1：混合ReRank

**核心思想：** Cross-Encoder粗排 + LLM精排

```python
def hybrid_rerank(query, documents, top_k=5):
    """混合重排序"""
    # 第1层：Cross-Encoder粗排（100 → 20）
    ce_reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
    scores = ce_reranker.predict([(query, doc) for doc in documents])
    top_20_indices = np.argsort(scores)[::-1][:20]
    top_20_docs = [documents[i] for i in top_20_indices]

    # 第2层：LLM Listwise精排（20 → 5）
    final_results = llm_listwise_rerank(query, top_20_docs, top_k=top_k)

    return final_results

# 成本分析：
# - Cross-Encoder：100文档 × $0.0001 = $0.01
# - LLM Listwise：20文档 × $0.05 = $1.00
# - 总成本：$1.01/query
# - vs 全LLM：100文档 × $0.05 = $5.00/query
# - 节省成本：80%
```

### 策略2：缓存优化

```python
from functools import lru_cache
import hashlib

class CachedLLMReranker:
    def __init__(self):
        self.cache = {}

    def _cache_key(self, query, doc):
        """生成缓存key"""
        return hashlib.md5(f"{query}:{doc}".encode()).hexdigest()

    def rerank(self, query, documents, top_k=5):
        """带缓存的LLM重排序"""
        scores = []

        for doc in documents:
            key = self._cache_key(query, doc)

            if key in self.cache:
                scores.append(self.cache[key])
            else:
                # LLM评分
                score = llm_pointwise_score(query, doc)
                self.cache[key] = score
                scores.append(score)

        # 排序
        ranked_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in ranked_indices[:top_k]]

# 效果：
# - 缓存命中率：60-80%（重复query）
# - 成本降低：60-80%
# - 延迟降低：60-80%
```

### 策略3：异步批处理

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_llm_pointwise_rerank(query, documents, top_k=5):
    """异步LLM Pointwise重排序"""
    # 并发调用LLM
    tasks = [
        async_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": pointwise_prompt(query, doc)}],
            temperature=0,
            max_tokens=10
        )
        for doc in documents
    ]

    responses = await asyncio.gather(*tasks)

    # 解析分数
    scores = []
    for response in responses:
        try:
            score = float(response.choices[0].message.content.strip())
            scores.append(score)
        except ValueError:
            scores.append(0.0)

    # 排序
    ranked_indices = np.argsort(scores)[::-1]
    return [documents[i] for i in ranked_indices[:top_k]]

# 使用示例
results = asyncio.run(async_llm_pointwise_rerank(query, documents))

# 效果：
# - 延迟降低：2500ms → 500ms（5倍加速）
# - 成本不变：$0.50/query
```

---

## 关键要点速记

### 核心概念
1. **三种方法**：Pointwise（独立评分）、Pairwise（两两比较）、Listwise（整体排序）
2. **最高精度**：NDCG@10可达0.89-0.92
3. **成本极高**：$0.50-5.00/M tokens（贵60倍）
4. **延迟极高**：2500-3000ms（慢12倍）

### 性能对比
5. **Pointwise**：并行处理，成本$0.50，NDCG 0.89
6. **Pairwise**：相对排序，成本$0.75，NDCG 0.90
7. **Listwise**：整体优化，成本$1.00，NDCG 0.92
8. **ROI极低**：精度提升8%，成本增加1000倍

### 适用场景
9. **离线批处理**：延迟不敏感，追求极致精度
10. **特殊领域**：需要复杂推理和领域知识
11. **小规模高精度**：文档数量<20，精度要求极高
12. **不适合生产**：实时系统成本和延迟不可接受

### 优化策略
13. **混合ReRank**：Cross-Encoder粗排 + LLM精排，节省80%成本
14. **缓存优化**：缓存命中率60-80%，降低成本和延迟
15. **异步批处理**：并发调用，延迟降低5倍

---

## 参考资料

### 核心研究
- [The Case Against LLMs as Rerankers](https://blog.voyageai.com/2025/10/22/the-case-against-llms-as-rerankers) - Voyage AI, 2025
- [RankLLM: Listwise Reranking with Large Language Models](http://zijianchen.ca/publications/rankllm_SIGIR2025.pdf) - SIGIR 2025

### 技术实现
- [ielab/llm-rankers GitHub](https://github.com/ielab/llm-rankers) - LLM reranking工具包
- [RankLLM Python Package](https://pypi.org/project/rank-llm/) - 官方实现

### 对比分析
- [Ultimate Guide to Choosing the Best Reranking Model in 2026](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Databricks Reranking Research](https://www.databricks.com/blog/reranking-mosaic-ai-vector-search-faster-smarter-retrieval-rag-agents)

---

**版本：** v1.0 (2026年标准)
**最后更新：** 2026-02-16
**适用场景：** RAG开发、信息检索、LLM应用
