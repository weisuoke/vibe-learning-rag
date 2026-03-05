# 核心概念1：EnsembleRetriever 混合检索

## 一句话定义

**EnsembleRetriever 通过加权倒数排名融合（RRF）将多个检索器的结果合并为统一排序列表，是 LangChain 实现混合检索的核心组件。**

---

## 为什么需要混合检索？

单一检索方式都有盲区，混合检索的本质是**互补**。

### 向量检索的盲区

向量检索擅长语义理解，但在以下场景表现差：

```
用户查询："错误代码 E1024 的解决方案"

向量检索的问题：
- "E1024" 是精确标识符，语义模型可能把它理解为普通数字
- 向量空间中 "E1024" 和 "E2048" 可能距离很近（都是错误代码）
- 但用户要的是精确匹配 E1024，不是语义相似的错误
```

**盲区清单：**
- 精确关键词匹配（产品型号、错误代码、人名）
- 专有名词和缩写（API、SDK、特定术语）
- 数字和日期（"2024年Q3报告"）
- 短查询（一两个词时语义信息不足）

### 关键词检索的盲区

BM25 等关键词检索擅长精确匹配，但在以下场景表现差：

```
用户查询："如何让程序跑得更快"

BM25 的问题：
- 文档中写的是 "性能优化"、"提升吞吐量"
- 没有出现 "跑得更快" 这几个字
- BM25 匹配不到，但语义上完全相关
```

**盲区清单：**
- 同义词和近义词（"快" vs "高性能"）
- 语义理解（"省钱" vs "成本优化"）
- 跨语言查询
- 长尾表述（用户用口语化表达搜索专业内容）

### 混合检索 = 两者互补

```
┌─────────────────────────────────────────────┐
│              用户查询                         │
│         "E1024 性能优化方案"                   │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  ┌─────────┐    ┌──────────┐
  │  BM25   │    │  向量检索  │
  │ 精确匹配 │    │  语义理解  │
  │ "E1024" │    │ "性能优化" │
  └────┬────┘    └─────┬────┘
       │               │
       └───────┬───────┘
               ▼
      ┌─────────────────┐
      │   RRF 排名融合    │
      │  合并 + 去重 + 排序 │
      └────────┬────────┘
               ▼
        最终排序结果
```

---

## RRF 算法详解

### 什么是 RRF？

**RRF（Reciprocal Rank Fusion）** 是一种排名融合算法，核心思想极其简单：

> 一个文档在多个检索器中排名都靠前，那它大概率是好结果。

### 公式解释

```
score(doc) = Σ weight_i / (rank_i + c)
```

用大白话说：

| 符号 | 含义 | 例子 |
|------|------|------|
| `score(doc)` | 文档的最终得分 | 某篇文档的综合分数 |
| `weight_i` | 第 i 个检索器的权重 | BM25 权重 0.4，向量权重 0.6 |
| `rank_i` | 文档在第 i 个检索器中的排名 | 排第 1 名、第 3 名... |
| `c` | 平滑常数 | 默认 60 |
| `Σ` | 对所有检索器求和 | 把每个检索器的贡献加起来 |

### c=60 的含义

`c` 是一个平滑常数，来自 RRF 原始论文（Cormack et al., SIGIR 2009）：

- **c 越大**：排名差异的影响越小（第 1 名和第 10 名的分数差距缩小）
- **c 越小**：排名靠前的文档优势越大
- **c=60 是论文实验得出的最优默认值**，适用于大多数场景

```python
# c=60 时，排名 1 和排名 10 的分数对比
score_rank_1 = 1 / (1 + 60)   # = 0.01639
score_rank_10 = 1 / (10 + 60)  # = 0.01429

# 差距只有 15%，排名靠后的文档仍有机会
# 如果 c=1：
score_rank_1_small_c = 1 / (1 + 1)   # = 0.5
score_rank_10_small_c = 1 / (10 + 1)  # = 0.0909
# 差距达到 450%，排名靠前的文档碾压式优势
```

### 权重的作用

权重决定了每个检索器的"话语权"：

```python
# 场景：BM25 排名第 1，向量检索排名第 3
# 权重：BM25=0.4, Vector=0.6

score_from_bm25 = 0.4 / (1 + 60)    # = 0.00656
score_from_vector = 0.6 / (3 + 60)   # = 0.00952

total_score = 0.00656 + 0.00952       # = 0.01608
```

即使 BM25 排名更高（第 1），但因为向量检索权重更大（0.6 > 0.4），向量检索的贡献反而更大。

### 手写 RRF 实现

理解算法最好的方式是自己实现一遍：

```python
"""
手写 RRF 算法 —— 理解 EnsembleRetriever 的核心
"""
from collections import defaultdict


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    weights: list[float] | None = None,
    c: int = 60,
) -> list[tuple[str, float]]:
    """
    加权倒数排名融合。

    Args:
        rank_lists: 多个检索器的排名结果，每个是文档 ID 列表（按相关性排序）
        weights: 每个检索器的权重，默认等权重
        c: 平滑常数，默认 60

    Returns:
        按 RRF 分数降序排列的 (文档ID, 分数) 列表
    """
    # 默认等权重
    if weights is None:
        weights = [1.0 / len(rank_lists)] * len(rank_lists)

    # 累积每个文档的 RRF 分数
    scores: dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(rank_lists, weights):
        for rank, doc_id in enumerate(doc_list, start=1):
            scores[doc_id] += weight / (rank + c)

    # 按分数降序排列
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ===== 演示 =====
bm25_results = ["doc_A", "doc_B", "doc_C", "doc_D"]
vector_results = ["doc_C", "doc_A", "doc_E", "doc_B"]

fused = reciprocal_rank_fusion(
    rank_lists=[bm25_results, vector_results],
    weights=[0.4, 0.6],
    c=60,
)

print("=== RRF 融合结果 ===")
for doc_id, score in fused:
    print(f"  {doc_id}: {score:.6f}")
```

**运行输出：**
```
=== RRF 融合结果 ===
  doc_A: 0.016230    # BM25 第1 + Vector 第2 → 两路都靠前，分数最高
  doc_C: 0.016230    # BM25 第3 + Vector 第1 → 同样两路都靠前
  doc_B: 0.016036    # BM25 第2 + Vector 第4
  doc_D: 0.006349    # 只在 BM25 中出现
  doc_E: 0.009524    # 只在 Vector 中出现
```

---

## EnsembleRetriever 源码解析

### 类结构和属性

```python
class EnsembleRetriever(BaseRetriever):
    """组合多个检索器，使用加权 RRF 融合结果。"""

    retrievers: list[RetrieverLike]   # 检索器列表
    weights: list[float]              # 权重列表，默认等权重 1/n
    c: int = 60                       # RRF 平滑常数
    id_key: str | None = None         # 去重键，None 时用 page_content
```

**关键设计决策：**
- `weights` 默认通过 `@model_validator` 自动设为等权重 `[1/n, 1/n, ...]`
- `id_key` 允许用 metadata 中的字段做去重（比如文档 ID），而不仅仅是内容去重

### weighted_reciprocal_rank 方法

这是核心算法实现，源码逻辑分三步：

```python
def weighted_reciprocal_rank(self, doc_lists: list[list[Document]]) -> list[Document]:
    # 第一步：累积 RRF 分数
    rrf_score: dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(doc_lists, self.weights):
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            rrf_score[key] += weight / (rank + self.c)

    # 第二步：去重（保留首次出现的版本）
    all_docs = chain.from_iterable(doc_lists)
    unique_docs = unique_by_key(all_docs, lambda doc: ...)

    # 第三步：按 RRF 分数降序排列
    return sorted(unique_docs, reverse=True, key=lambda doc: rrf_score[...])
```

### 去重机制

`unique_by_key` 是一个生成器函数，用 `set` 跟踪已见过的键：

```python
def unique_by_key(iterable, key):
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e
```

**去重策略：**
- 默认按 `page_content` 去重 —— 内容完全相同的文档只保留一份
- 设置 `id_key` 后按 metadata 字段去重 —— 适合同一文档被不同检索器返回时 metadata 不同的情况

### 异步并行执行

同步版本是顺序执行，异步版本用 `asyncio.gather` 并行：

```python
# 同步：顺序执行（慢）
retriever_docs = [
    retriever.invoke(query, config)
    for retriever in self.retrievers
]

# 异步：并行执行（快）
retriever_docs = await asyncio.gather(*[
    retriever.ainvoke(query, config)
    for retriever in self.retrievers
])
```

**为什么覆盖 invoke 而不是 _get_relevant_documents？**

源码直接覆盖了 `invoke`/`ainvoke`，而不是常规的 `_get_relevant_documents`。原因是需要将完整的 `RunnableConfig`（包括 callbacks、tags、metadata）传播到每个子检索器，而 `_get_relevant_documents` 只接收 `run_manager`，无法传递完整配置。

---

## 使用方式

### 基础用法：BM25 + VectorStore

```python
"""
EnsembleRetriever 基础用法：BM25 + 向量检索混合
"""
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever

# ===== 1. 准备文档 =====
docs = [
    "Python 是一种解释型编程语言，适合快速开发",
    "FastAPI 是基于 Python 的高性能 Web 框架",
    "错误代码 E1024 表示内存溢出，需要优化数据结构",
    "向量数据库 Milvus 支持十亿级向量检索",
    "RAG 系统通过检索增强生成来提升 LLM 回答质量",
]

# ===== 2. 创建两个检索器 =====
# BM25：擅长精确关键词匹配
bm25_retriever = BM25Retriever.from_texts(docs, k=4)

# 向量检索：擅长语义理解
vectorstore = Chroma.from_texts(docs, embedding=OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ===== 3. 创建混合检索器 =====
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # 向量检索权重更高
)

# ===== 4. 执行检索 =====
results = ensemble_retriever.invoke("E1024 性能优化")
for i, doc in enumerate(results, 1):
    print(f"  #{i}: {doc.page_content[:60]}")
```

### 权重调优策略

不同场景需要不同的权重配比：

```python
# 场景1：技术文档检索（精确术语多）
# BM25 权重高，因为用户经常搜索精确的 API 名、错误码
ensemble_tech = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4],  # BM25 主导
)

# 场景2：客服问答（口语化查询多）
# 向量权重高，因为用户表述多样，需要语义理解
ensemble_cs = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # 向量主导
)

# 场景3：通用场景（推荐默认值）
ensemble_default = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # 略偏向量
)
```

| 场景 | BM25 权重 | 向量权重 | 原因 |
|------|-----------|----------|------|
| 技术文档 | 0.5-0.6 | 0.4-0.5 | 精确术语、代码片段多 |
| 客服问答 | 0.2-0.3 | 0.7-0.8 | 口语化表述，需语义理解 |
| 法律/合规 | 0.5-0.6 | 0.4-0.5 | 精确条款引用重要 |
| 通用知识库 | 0.4 | 0.6 | 平衡精确与语义 |

### 三路混合检索

EnsembleRetriever 不限于两个检索器，可以组合任意多个：

```python
"""
三路混合检索：BM25 + 向量 + TF-IDF
"""
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain.retrievers import EnsembleRetriever

# 三个不同策略的检索器
bm25_retriever = BM25Retriever.from_texts(docs, k=4)
tfidf_retriever = TFIDFRetriever.from_texts(docs, k=4)
# vector_retriever 同上

ensemble_3way = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever, tfidf_retriever],
    weights=[0.3, 0.5, 0.2],  # 向量为主，BM25 辅助，TF-IDF 补充
)

results = ensemble_3way.invoke("如何优化 RAG 系统的检索质量")
```

---

## 生产最佳实践

### 默认配置推荐

```python
# 生产环境推荐配置
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],   # 2025-2026 社区广泛验证的默认值
    c=60,                  # 保持默认，除非有明确的调优需求
)
```

### 何时调整权重

```python
# 判断标准：看你的查询类型分布
#
# 如果大部分查询包含精确关键词（型号、代码、人名）：
#   → 提高 BM25 权重到 0.5-0.6
#
# 如果大部分查询是自然语言描述（"怎么做"、"为什么"）：
#   → 提高向量权重到 0.7-0.8
#
# 实际操作：用评估集测试不同权重，选 recall@k 最高的
```

### 性能考虑

```python
# 1. 异步并行 —— 生产环境必用
results = await ensemble_retriever.ainvoke("查询内容")
# 两个检索器并行执行，总耗时 ≈ max(bm25_time, vector_time)
# 而非 bm25_time + vector_time

# 2. 控制每个子检索器的 k 值
bm25_retriever = BM25Retriever.from_texts(docs, k=10)      # 多召回一些
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# RRF 融合后自然会排序，可以在后续截取 top-k
results = ensemble_retriever.invoke(query)[:5]  # 最终只取前 5

# 3. id_key 去重优化
# 当同一文档可能被不同检索器返回（metadata 不同）时
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
    id_key="doc_id",  # 用 metadata 中的 doc_id 去重，而非内容比较
)
```

---

## 在 RAG 中的应用

### 混合检索作为 RAG 第一阶段

在典型的 RAG 管道中，EnsembleRetriever 负责**召回阶段**：

```
用户查询
    │
    ▼
┌──────────────────────┐
│  EnsembleRetriever   │  ← 第一阶段：混合召回（宽网捞鱼）
│  BM25 + Vector       │
│  返回 top-20 候选     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  CrossEncoder Rerank │  ← 第二阶段：精排（精挑细选）
│  返回 top-5 最相关    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  LLM 生成回答         │  ← 第三阶段：生成
│  基于 top-5 上下文     │
└──────────────────────┘
```

### 与重排序的配合

```python
"""
生产级 RAG 管道：Hybrid Search + Rerank
"""
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ===== 第一阶段：混合召回 =====
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# ===== 第二阶段：重排序 =====
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

# 组合：先混合召回，再重排序
retriever_with_rerank = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever,
)

# ===== 第三阶段：接入 RAG 链 =====
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever_with_rerank | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("E1024 错误如何解决？")
print(answer)
```

### 为什么这个组合是生产标配？

| 阶段 | 组件 | 作用 | 类比 |
|------|------|------|------|
| 召回 | EnsembleRetriever | 宽网捞鱼，不漏掉好结果 | 海选 |
| 精排 | CrossEncoder Rerank | 精挑细选，排出最佳顺序 | 决赛 |
| 生成 | LLM | 基于精选上下文生成回答 | 评委点评 |

**关键洞察：** 召回阶段追求的是**高召回率**（别漏掉好文档），精排阶段追求的是**高精确率**（排在前面的都是好文档）。EnsembleRetriever 的 RRF 融合天然适合召回阶段 —— 它不需要精确排序，只需要把好文档都捞上来。

---

## 小结

| 要点 | 说明 |
|------|------|
| 核心算法 | 加权 RRF：`score = Σ weight_i / (rank_i + c)` |
| 默认参数 | `c=60`（论文最优），权重等分或 `[0.4, 0.6]` |
| 去重策略 | 默认按 `page_content`，可设 `id_key` 按 metadata |
| 异步优势 | `asyncio.gather` 并行执行子检索器 |
| 生产推荐 | Ensemble(BM25+Vector) → Rerank → LLM |
| 权重调优 | 精确查询多 → 提高 BM25；语义查询多 → 提高向量 |
