# 核心概念5：Top-K选择与MMR

> 结果数量控制与多样性优化，提升用户体验的关键技术

---

## 一句话定义

**Top-K是控制检索返回结果数量的策略，MMR (Maximal Marginal Relevance) 是在相关性和多样性之间平衡的算法，两者结合能避免重复结果，提升RAG系统的用户体验。**

---

## Top-K原理

### 什么是Top-K？

**Top-K**：返回相似度最高的前K个结果

```python
"""
Top-K检索示例
"""
# 假设有100个文档，相似度分数如下
scores = [0.95, 0.92, 0.88, 0.85, 0.82, ...]  # 100个分数

# Top-5：返回前5个最相关的
top_5 = scores[:5]  # [0.95, 0.92, 0.88, 0.85, 0.82]

# Top-10：返回前10个
top_10 = scores[:10]
```

### K值选择

| K值 | 适用场景 | 优点 | 缺点 |
|-----|----------|------|------|
| **3-5** | 精准问答 | 结果聚焦，相关性高 | 可能漏掉相关信息 |
| **5-10** | 通用检索 | 平衡相关性和覆盖面 | 适中 |
| **10-20** | 探索性搜索 | 提供更多选择 | 可能包含不太相关的结果 |
| **20-50** | ReRank前 | 先召回更多候选 | 需要后续精排 |

**参考**: [RAG Evaluation Guide 2025 - Maxim AI](https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025)

---

## MMR算法

### MMR原理

**MMR (Maximal Marginal Relevance)** 在相关性和多样性之间平衡：

```
MMR = arg max [λ × Sim1(Di, Q) - (1-λ) × max Sim2(Di, Dj)]
              Di∈R\S                    Dj∈S

其中：
- Di: 候选文档
- Q: 查询
- S: 已选文档集合
- R: 候选文档集合
- λ: 平衡参数（0-1）
- Sim1: 查询-文档相似度
- Sim2: 文档-文档相似度
```

**直觉理解**：
- 第一项：与查询的相关性（越大越好）
- 第二项：与已选文档的相似度（越小越好，避免重复）
- λ：控制相关性和多样性的权衡

**参考**: [Diversifying search results with MMR - Elastic 2025](https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results)

### MMR实现

```python
"""
MMR算法实现
"""
import numpy as np

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    lambda_param: float = 0.7,
    k: int = 5,
    fetch_k: int = 20
) -> list:
    """
    MMR算法

    Args:
        query_embedding: 查询向量
        doc_embeddings: 文档向量列表
        lambda_param: 平衡参数（0-1）
        k: 最终返回数量
        fetch_k: 候选文档数量

    Returns:
        选中的文档索引列表
    """
    # 1. 计算查询-文档相似度
    query_doc_sim = np.dot(doc_embeddings, query_embedding)

    # 2. 选择Top-fetch_k作为候选
    candidate_indices = query_doc_sim.argsort()[-fetch_k:][::-1]

    # 3. 初始化
    selected = []
    selected_embeddings = []

    # 4. 迭代选择
    while len(selected) < k and len(candidate_indices) > 0:
        # 计算每个候选的MMR得分
        mmr_scores = []

        for idx in candidate_indices:
            if idx in selected:
                continue

            # 相关性得分
            relevance = query_doc_sim[idx]

            # 多样性得分（与已选文档的最大相似度）
            if len(selected_embeddings) > 0:
                doc_doc_sim = np.dot(
                    selected_embeddings,
                    doc_embeddings[idx]
                )
                diversity = -np.max(doc_doc_sim)  # 负号：越不相似越好
            else:
                diversity = 0

            # MMR得分
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr_score))

        # 选择MMR得分最高的
        if mmr_scores:
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
            selected_embeddings.append(doc_embeddings[best_idx])

            # 从候选中移除
            candidate_indices = [i for i in candidate_indices if i != best_idx]

    return selected


# 示例使用
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 文档
documents = [
    "Python是一种编程语言",
    "Python是一门编程语言",  # 与第1个高度相似
    "JavaScript用于Web开发",
    "JS是Web开发语言",       # 与第3个高度相似
    "机器学习是AI的核心"
]

# 向量化
doc_embeddings = model.encode(documents)
query = "编程语言"
query_embedding = model.encode([query])[0]

# MMR检索
selected_indices = maximal_marginal_relevance(
    query_embedding=query_embedding,
    doc_embeddings=doc_embeddings,
    lambda_param=0.7,
    k=3,
    fetch_k=5
)

print("=== MMR检索结果 ===")
for i, idx in enumerate(selected_indices, 1):
    print(f"{i}. {documents[idx]}")

# 输出:
# 1. Python是一种编程语言  ← 最相关
# 2. JavaScript用于Web开发  ← 多样性（不同主题）
# 3. 机器学习是AI的核心    ← 多样性（不同主题）
# 注意：没有选择"Python是一门编程语言"（与第1个重复）
```

---

## Lambda参数调优

### Lambda值的影响

```python
"""
Lambda参数对比实验
"""
def compare_lambda_values(query_embedding, doc_embeddings, documents):
    """
    对比不同lambda值的效果
    """
    lambda_values = [1.0, 0.7, 0.5, 0.3, 0.0]

    for lambda_val in lambda_values:
        print(f"\n=== Lambda = {lambda_val} ===")

        selected = maximal_marginal_relevance(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            lambda_param=lambda_val,
            k=3,
            fetch_k=5
        )

        for i, idx in enumerate(selected, 1):
            print(f"{i}. {documents[idx]}")

# 运行对比
compare_lambda_values(query_embedding, doc_embeddings, documents)

# 输出示例:
# === Lambda = 1.0 ===（只考虑相关性）
# 1. Python是一种编程语言
# 2. Python是一门编程语言  ← 重复
# 3. JavaScript用于Web开发
#
# === Lambda = 0.7 ===（推荐：平衡）
# 1. Python是一种编程语言
# 2. JavaScript用于Web开发  ← 多样性
# 3. 机器学习是AI的核心    ← 多样性
#
# === Lambda = 0.5 ===（强调多样性）
# 1. Python是一种编程语言
# 2. 机器学习是AI的核心    ← 优先多样性
# 3. JavaScript用于Web开发
#
# === Lambda = 0.0 ===（只考虑多样性）
# 1. Python是一种编程语言
# 2. 机器学习是AI的核心    ← 可能不太相关
# 3. JavaScript用于Web开发
```

### Lambda选择建议

| Lambda值 | 相关性 | 多样性 | 适用场景 |
|----------|--------|--------|----------|
| **1.0** | 100% | 0% | 精准问答（只要最相关的） |
| **0.7-0.8** | 70-80% | 20-30% | 通用场景（推荐） |
| **0.5** | 50% | 50% | 探索性搜索 |
| **0.3** | 30% | 70% | 发现新内容 |
| **0.0** | 0% | 100% | 纯多样性（不推荐） |

**参考**: [Native MMR Support - OpenSearch](https://opensearch.org/blog/improving-vector-search-diversity-through-native-mmr)

---

## LangChain实现

### 使用max_marginal_relevance_search

```python
"""
LangChain MMR检索
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings
)

# MMR检索
results = vectorstore.max_marginal_relevance_search(
    query="编程语言",
    k=3,              # 最终返回3个结果
    fetch_k=10,       # 先检索10个候选
    lambda_mult=0.7   # Lambda参数
)

print("=== MMR检索结果 ===")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
```

### 配置MMR参数

```python
"""
不同场景的MMR配置
"""
# 场景1：精准问答
results_precise = vectorstore.max_marginal_relevance_search(
    query="什么是RAG系统？",
    k=3,
    fetch_k=10,
    lambda_mult=0.9  # 高相关性
)

# 场景2：通用检索（推荐）
results_general = vectorstore.max_marginal_relevance_search(
    query="如何学习编程",
    k=5,
    fetch_k=20,
    lambda_mult=0.7  # 平衡
)

# 场景3：探索性搜索
results_explore = vectorstore.max_marginal_relevance_search(
    query="人工智能",
    k=10,
    fetch_k=50,
    lambda_mult=0.5  # 强调多样性
)
```

---

## 在RAG中的应用

### 应用1：避免重复结果

```python
"""
避免重复结果示例
"""
# 场景：用户查询"Python教程"

# 不使用MMR（Top-K）
top_k_results = vectorstore.similarity_search("Python教程", k=5)
print("=== Top-K结果（可能重复）===")
for doc in top_k_results:
    print(f"- {doc.page_content[:50]}...")

# 输出:
# - Python入门教程第一章...
# - Python入门教程第二章...  ← 重复主题
# - Python基础教程...        ← 重复主题
# - Python编程指南...        ← 重复主题
# - Python快速上手...        ← 重复主题

# 使用MMR
mmr_results = vectorstore.max_marginal_relevance_search(
    "Python教程",
    k=5,
    fetch_k=20,
    lambda_mult=0.7
)
print("\n=== MMR结果（多样性）===")
for doc in mmr_results:
    print(f"- {doc.page_content[:50]}...")

# 输出:
# - Python入门教程...        ← 基础
# - Python进阶技巧...        ← 进阶
# - Python实战项目...        ← 实战
# - Python性能优化...        ← 优化
# - Python最佳实践...        ← 最佳实践
```

### 应用2：多主题覆盖

```python
"""
多主题覆盖示例
"""
# 场景：企业知识库检索

documents = [
    "Python异步编程详解",
    "Python协程使用指南",      # 与第1个相似
    "FastAPI框架入门",
    "FastAPI高级特性",         # 与第3个相似
    "Docker容器化部署",
    "Kubernetes集群管理",      # 与第5个相似
    "PostgreSQL数据库优化",
    "Redis缓存策略"
]

# 查询："后端开发技术"

# MMR检索（多样性）
results = vectorstore.max_marginal_relevance_search(
    "后端开发技术",
    k=4,
    fetch_k=8,
    lambda_mult=0.6  # 强调多样性
)

# 输出:
# 1. Python异步编程详解     ← 编程语言
# 2. FastAPI框架入门        ← Web框架
# 3. Docker容器化部署       ← 部署
# 4. PostgreSQL数据库优化   ← 数据库
# ✅ 覆盖了4个不同主题
```

### 应用3：长文档分块去重

```python
"""
长文档分块去重
"""
# 场景：长文档被分成多个chunk，相邻chunk可能高度相似

chunks = [
    "第一章：Python基础。Python是一种高级编程语言...",
    "Python是一种高级编程语言，由Guido创建...",  # 与第1个重复
    "第二章：数据类型。Python有多种数据类型...",
    "Python的数据类型包括整数、浮点数...",      # 与第3个重复
    "第三章：控制流。if语句用于条件判断..."
]

# 使用MMR避免返回重复chunk
results = vectorstore.max_marginal_relevance_search(
    "Python基础知识",
    k=3,
    fetch_k=5,
    lambda_mult=0.7
)

# 输出:
# 1. 第一章：Python基础...   ← 最相关
# 2. 第二章：数据类型...     ← 不同主题
# 3. 第三章：控制流...       ← 不同主题
# ✅ 避免了重复chunk
```

---

## 2025-2026最佳实践

### 1. 默认使用MMR

```python
"""
2025-2026标准：默认使用MMR
"""
# ❌ 2024年做法：只用Top-K
results = vectorstore.similarity_search(query, k=5)

# ✅ 2025-2026标准：使用MMR
results = vectorstore.max_marginal_relevance_search(
    query=query,
    k=5,
    fetch_k=20,
    lambda_mult=0.7
)
```

### 2. 根据场景调整参数

```python
"""
场景化MMR配置
"""
mmr_configs = {
    "精准问答": {
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.9
    },
    "通用检索": {
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    },
    "探索性搜索": {
        "k": 10,
        "fetch_k": 50,
        "lambda_mult": 0.5
    }
}

def get_mmr_config(scenario):
    """根据场景获取MMR配置"""
    return mmr_configs.get(scenario, mmr_configs["通用检索"])

# 使用
config = get_mmr_config("通用检索")
results = vectorstore.max_marginal_relevance_search(
    query=query,
    **config
)
```

### 3. 监控多样性指标

```python
"""
监控MMR效果
"""
def evaluate_diversity(results):
    """
    评估结果多样性

    Args:
        results: 检索结果列表
    """
    # 计算结果之间的平均相似度
    embeddings = [model.encode(doc.page_content) for doc in results]

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(sim)

    avg_sim = np.mean(similarities)

    print(f"结果数量: {len(results)}")
    print(f"平均相似度: {avg_sim:.3f}")

    if avg_sim > 0.8:
        print("⚠️ 警告: 结果相似度过高，缺乏多样性")
    elif avg_sim < 0.3:
        print("⚠️ 警告: 结果相似度过低，可能不够相关")
    else:
        print("✅ 多样性良好")

# 使用
evaluate_diversity(results)
```

### 4. OpenSearch 3.3原生支持

```python
"""
OpenSearch 3.3原生MMR支持
"""
# OpenSearch 3.3开始原生支持MMR
# 无需手动实现，性能更优

from opensearchpy import OpenSearch

client = OpenSearch(...)

# 使用原生MMR
response = client.search(
    index="documents",
    body={
        "query": {
            "neural": {
                "embedding": {
                    "query_text": "编程语言",
                    "model_id": "text-embedding-model",
                    "k": 5
                }
            }
        },
        "ext": {
            "mmr": {
                "enabled": True,
                "lambda": 0.7,
                "fetch_k": 20
            }
        }
    }
)
```

**参考**: [Native MMR Support - OpenSearch](https://opensearch.org/blog/improving-vector-search-diversity-through-native-mmr)

---

## 常见问题

### Q1: 什么时候用Top-K，什么时候用MMR？

**A**:
- **Top-K**: 精准问答，只需要最相关的结果
- **MMR**: 通用检索，需要多样性避免重复
- **推荐**: 默认使用MMR（lambda=0.7），除非明确只需要最相关的

### Q2: fetch_k如何选择？

**A**:
- fetch_k应该是k的2-5倍
- k=5 → fetch_k=10-25
- k=10 → fetch_k=20-50
- fetch_k越大，多样性选择空间越大，但计算成本越高

### Q3: MMR会降低相关性吗？

**A**:
- 会略微降低，但提升用户体验
- lambda=0.7时，相关性降低<10%，但多样性显著提升
- 实际应用中，用户更喜欢多样化的结果

### Q4: MMR计算成本高吗？

**A**:
- 相比Top-K，增加约20-30%计算时间
- 主要开销在文档-文档相似度计算
- 可以通过缓存和优化减少开销
- 对用户体验的提升远大于性能开销

---

## 总结

**Top-K与MMR的核心价值**：
1. ✅ Top-K：控制结果数量，聚焦最相关的
2. ✅ MMR：平衡相关性和多样性，避免重复
3. ✅ 提升用户体验：多样化结果更受欢迎

**核心要点**：
- Top-K适合精准问答，MMR适合通用检索
- Lambda=0.7是推荐值（70%相关性 + 30%多样性）
- fetch_k应该是k的2-5倍
- OpenSearch 3.3开始原生支持MMR

**2025-2026标准实践**：
- 默认使用MMR而非Top-K
- 根据场景调整lambda参数
- 监控多样性指标
- 使用原生MMR支持（如OpenSearch 3.3）

**下一步**：学习【核心概念6：HNSW索引】，理解层次化图结构和参数调优。
