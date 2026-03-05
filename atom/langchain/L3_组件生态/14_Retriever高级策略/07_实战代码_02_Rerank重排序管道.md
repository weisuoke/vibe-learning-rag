# Retriever高级策略 - 实战代码：Rerank 重排序管道

> **完整可运行的重排序示例**：在基础检索器之上添加重排序层，显著提升检索精度

---

## 场景描述

基础检索器（向量检索或混合检索）返回的 top-K 结果中，排序往往不够精准。重排序（Reranking）是一个轻量级的后处理阶段：先用快速检索召回候选文档（如 top-20），再用更精确的模型对候选集重新打分排序，取 top-3~5 返回。

本文档演示三种方案：
- **方案A**：FlashrankRerank — 轻量级，适合快速上手
- **方案B**：CrossEncoder 本地重排序 — 精度更高，完全本地化
- **方案C**：DocumentCompressorPipeline 组合管道 — 生产级多阶段管道

---

## 环境准备

### 安装依赖

```bash
# 使用 uv 安装
uv add langchain langchain-openai langchain-chroma langchain-community \
      flashrank sentence-transformers

# 或使用 pip
pip install langchain langchain-openai langchain-chroma langchain-community \
            flashrank sentence-transformers
```

### 配置环境变量

```bash
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

---

## 完整代码

### 公共部分：准备文档和基础检索器

```python
"""
公共部分：创建测试文档和基础向量检索器
后续三个方案都复用这个基础检索器
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 模拟技术文档知识库（故意包含相关度不同的文档）
documents = [
    Document(
        page_content="LangChain 的 LCEL（LangChain Expression Language）是一种声明式语法，"
                     "用于将 Runnable 组件链式组合。核心操作符是管道符 |，"
                     "支持 invoke、stream、batch 三种调用模式。",
        metadata={"topic": "lcel", "relevance": "high"}
    ),
    Document(
        page_content="LangChain 支持多种向量存储后端，包括 Chroma、FAISS、Milvus、Pinecone 等。"
                     "每种后端有不同的性能特点和部署要求。",
        metadata={"topic": "vectorstore", "relevance": "medium"}
    ),
    Document(
        page_content="LCEL 表达式可以通过 RunnablePassthrough 和 RunnableParallel "
                     "实现复杂的数据流控制，包括分支、合并和条件路由。",
        metadata={"topic": "lcel", "relevance": "high"}
    ),
    Document(
        page_content="Prompt Template 是 LangChain 中管理提示词的核心组件，"
                     "支持变量插值、Few-shot 示例和消息格式化。",
        metadata={"topic": "prompt", "relevance": "low"}
    ),
    Document(
        page_content="LangChain 的 Retriever 接口定义了统一的文档检索协议，"
                     "所有检索器都实现 invoke(query) -> list[Document] 方法。",
        metadata={"topic": "retriever", "relevance": "medium"}
    ),
    Document(
        page_content="LCEL 链可以通过 .with_config() 方法注入运行时配置，"
                     "包括回调、标签和元数据，便于追踪和调试。",
        metadata={"topic": "lcel", "relevance": "medium"}
    ),
    Document(
        page_content="LangChain Agent 使用 LLM 作为推理引擎，"
                     "动态决定调用哪些工具来完成任务。支持 ReAct 和 Plan-and-Execute 模式。",
        metadata={"topic": "agent", "relevance": "low"}
    ),
    Document(
        page_content="OutputParser 负责将 LLM 的原始文本输出解析为结构化数据，"
                     "常用的有 JsonOutputParser、PydanticOutputParser 等。",
        metadata={"topic": "output_parser", "relevance": "low"}
    ),
]

# 创建基础向量检索器（故意设 k=6，召回较多候选）
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    collection_name="rerank_demo",
)

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}  # 召回 6 个候选，后续 rerank 取 top-3
)

print("基础检索器创建完成")
print(f"文档总数: {len(documents)}")
print(f"基础召回数: k=6")

# 测试基础检索（无 rerank）
query = "LCEL 表达式语法怎么用"
base_results = base_retriever.invoke(query)
print(f"\n基础检索结果 (query='{query}'):")
for i, doc in enumerate(base_results, 1):
    print(f"  {i}. [{doc.metadata['topic']}] (相关度:{doc.metadata['relevance']}) "
          f"{doc.page_content[:45]}...")
```

---

### 方案A: FlashrankRerank（推荐入门）

```python
"""
方案A: FlashrankRerank
- 轻量级重排序模型，首次运行自动下载（约 100MB）
- 不需要 GPU，CPU 即可运行
- 适合快速原型和中小规模场景
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# 创建 FlashrankRerank 压缩器
flashrank_compressor = FlashrankRerank(
    top_n=3,       # 重排序后保留 top-3
    model="ms-marco-MiniLM-L-12-v2",  # 默认模型，平衡速度和精度
)

# 包装为 ContextualCompressionRetriever
flashrank_retriever = ContextualCompressionRetriever(
    base_compressor=flashrank_compressor,
    base_retriever=base_retriever,
)

# 测试
query = "LCEL 表达式语法怎么用"
print(f"=== 方案A: FlashrankRerank ===")
print(f"查询: '{query}'")

reranked_results = flashrank_retriever.invoke(query)
print(f"重排序后返回 {len(reranked_results)} 个文档:")
for i, doc in enumerate(reranked_results, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"  {i}. [score={score}] [{doc.metadata['topic']}] "
          f"{doc.page_content[:45]}...")
```

---

### 方案B: CrossEncoder 本地重排序

```python
"""
方案B: CrossEncoder 本地重排序
- 使用 sentence-transformers 的 CrossEncoder 模型
- 精度通常高于 FlashRank
- 完全本地运行，无 API 调用
- 适合对精度要求高、有 GPU 资源的场景
"""

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载 CrossEncoder 模型（首次运行自动下载）
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    # 其他可选模型:
    # "cross-encoder/ms-marco-MiniLM-L-12-v2"  (更大更准)
    # "BAAI/bge-reranker-base"                  (中文友好)
    # "BAAI/bge-reranker-v2-m3"                 (多语言)
)

# 创建 CrossEncoder 重排序器
cross_encoder_compressor = CrossEncoderReranker(
    model=cross_encoder_model,
    top_n=3,
)

# 包装为 ContextualCompressionRetriever
cross_encoder_retriever = ContextualCompressionRetriever(
    base_compressor=cross_encoder_compressor,
    base_retriever=base_retriever,
)

# 测试
query = "LCEL 表达式语法怎么用"
print(f"=== 方案B: CrossEncoder ===")
print(f"查询: '{query}'")

reranked_results = cross_encoder_retriever.invoke(query)
print(f"重排序后返回 {len(reranked_results)} 个文档:")
for i, doc in enumerate(reranked_results, 1):
    print(f"  {i}. [{doc.metadata['topic']}] (原始标注:{doc.metadata['relevance']}) "
          f"{doc.page_content[:45]}...")
```

---

### 方案C: DocumentCompressorPipeline 组合管道

```python
"""
方案C: DocumentCompressorPipeline 组合管道
将多个处理步骤串联：过滤 → 去重 → 重排序
这是生产环境推荐的模式
"""

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_openai import OpenAIEmbeddings

# 阶段1: 去除冗余文档（基于 embedding 相似度去重）
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.95,  # 相似度 > 0.95 视为重复
)

# 阶段2: 重排序
reranker = FlashrankRerank(top_n=3)

# 组合管道：去重 → 重排序
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, reranker]
)

# 包装为检索器
pipeline_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

# 测试
query = "LCEL 表达式语法怎么用"
print(f"=== 方案C: Pipeline (去重 + 重排序) ===")
print(f"查询: '{query}'")

pipeline_results = pipeline_retriever.invoke(query)
print(f"管道处理后返回 {len(pipeline_results)} 个文档:")
for i, doc in enumerate(pipeline_results, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"  {i}. [score={score}] [{doc.metadata['topic']}] "
          f"{doc.page_content[:45]}...")
```

---

### 效果对比

```python
"""
效果对比：基础检索 vs 三种重排序方案
"""

import time

query = "LCEL 表达式语法怎么用"
print(f"查询: '{query}'")
print(f"{'='*70}")

# 基础检索
start = time.time()
base_docs = base_retriever.invoke(query)
base_time = time.time() - start
print(f"\n[基础检索] 耗时: {base_time:.3f}s, 返回: {len(base_docs)} 个文档")
for i, doc in enumerate(base_docs, 1):
    print(f"  {i}. [{doc.metadata['topic']}] ({doc.metadata['relevance']})")

# FlashRank
start = time.time()
flash_docs = flashrank_retriever.invoke(query)
flash_time = time.time() - start
print(f"\n[FlashRank] 耗时: {flash_time:.3f}s, 返回: {len(flash_docs)} 个文档")
for i, doc in enumerate(flash_docs, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"  {i}. [{doc.metadata['topic']}] ({doc.metadata['relevance']}) score={score}")

# CrossEncoder
start = time.time()
ce_docs = cross_encoder_retriever.invoke(query)
ce_time = time.time() - start
print(f"\n[CrossEncoder] 耗时: {ce_time:.3f}s, 返回: {len(ce_docs)} 个文档")
for i, doc in enumerate(ce_docs, 1):
    print(f"  {i}. [{doc.metadata['topic']}] ({doc.metadata['relevance']})")

# Pipeline
start = time.time()
pipe_docs = pipeline_retriever.invoke(query)
pipe_time = time.time() - start
print(f"\n[Pipeline] 耗时: {pipe_time:.3f}s, 返回: {len(pipe_docs)} 个文档")
for i, doc in enumerate(pipe_docs, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"  {i}. [{doc.metadata['topic']}] ({doc.metadata['relevance']}) score={score}")

# 汇总
print(f"\n{'='*70}")
print("耗时对比:")
print(f"  基础检索:    {base_time:.3f}s")
print(f"  FlashRank:   {flash_time:.3f}s (+{flash_time-base_time:.3f}s)")
print(f"  CrossEncoder:{ce_time:.3f}s (+{ce_time-base_time:.3f}s)")
print(f"  Pipeline:    {pipe_time:.3f}s (+{pipe_time-base_time:.3f}s)")
```

---

## 运行输出示例

```
基础检索器创建完成
文档总数: 8
基础召回数: k=6

基础检索结果 (query='LCEL 表达式语法怎么用'):
  1. [lcel] (相关度:high) LangChain 的 LCEL（LangChain Expression L...
  2. [lcel] (相关度:high) LCEL 表达式可以通过 RunnablePassthrough 和 Ru...
  3. [lcel] (相关度:medium) LCEL 链可以通过 .with_config() 方法注入运行时配...
  4. [retriever] (相关度:medium) LangChain 的 Retriever 接口定义了统一的文档检索...
  5. [prompt] (相关度:low) Prompt Template 是 LangChain 中管理提示词的核心...
  6. [vectorstore] (相关度:medium) LangChain 支持多种向量存储后端，包括 Chroma、FAI...

=== 方案A: FlashrankRerank ===
查询: 'LCEL 表达式语法怎么用'
重排序后返回 3 个文档:
  1. [score=0.9847] [lcel] LangChain 的 LCEL（LangChain Expression L...
  2. [score=0.9523] [lcel] LCEL 表达式可以通过 RunnablePassthrough 和 Ru...
  3. [score=0.8134] [lcel] LCEL 链可以通过 .with_config() 方法注入运行时配...

=== 方案B: CrossEncoder ===
查询: 'LCEL 表达式语法怎么用'
重排序后返回 3 个文档:
  1. [lcel] (原始标注:high) LangChain 的 LCEL（LangChain Expression L...
  2. [lcel] (原始标注:high) LCEL 表达式可以通过 RunnablePassthrough 和 Ru...
  3. [lcel] (原始标注:medium) LCEL 链可以通过 .with_config() 方法注入运行时配...

======================================================================
耗时对比:
  基础检索:    0.156s
  FlashRank:   0.198s (+0.042s)
  CrossEncoder:0.312s (+0.156s)
  Pipeline:    0.245s (+0.089s)
```

---

## 关键参数说明

| 参数 | 组件 | 默认值 | 说明 |
|------|------|--------|------|
| `top_n` | FlashrankRerank | `3` | 重排序后保留的文档数 |
| `model` | FlashrankRerank | `ms-marco-MiniLM-L-12-v2` | 重排序模型 |
| `top_n` | CrossEncoderReranker | `3` | 重排序后保留的文档数 |
| `model_name` | HuggingFaceCrossEncoder | - | CrossEncoder 模型名 |
| `similarity_threshold` | EmbeddingsRedundantFilter | `0.95` | 去重相似度阈值 |
| base_retriever `k` | VectorStoreRetriever | `4` | 基础召回数，建议设为 top_n 的 3-5 倍 |

### 基础召回数 vs 重排序 top_n 的关系

```
基础召回 k=20  →  Rerank top_n=5  →  最终返回 5 个文档

为什么 k 要远大于 top_n？
- 基础检索是"粗筛"，速度快但排序不够精准
- Rerank 是"精排"，在候选集中重新打分
- 候选集越大，Rerank 越有可能找到真正相关的文档
- 但 k 太大会增加 Rerank 延迟（线性增长）

经验值：
- 快速原型: k=10, top_n=3
- 生产环境: k=20, top_n=5
- 高精度场景: k=50, top_n=5
```

---

## 生产建议

### 1. 模型选择指南

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 快速原型 | FlashrankRerank | 零配置，CPU 友好 |
| 中文场景 | CrossEncoder + `BAAI/bge-reranker-v2-m3` | 中文效果好 |
| 英文场景 | CrossEncoder + `cross-encoder/ms-marco-MiniLM-L-12-v2` | 精度高 |
| 有预算 | Cohere Rerank API | 效果最好，但有 API 成本 |
| 多语言 | CrossEncoder + `BAAI/bge-reranker-v2-m3` | 多语言支持 |

### 2. 与混合检索组合（推荐生产管道）

```python
"""
生产推荐管道: EnsembleRetriever + Rerank
Hybrid Search → Rerank → Top-K
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 第一层：混合检索（粗筛）
bm25 = BM25Retriever.from_documents(documents, k=10)
vector = vectorstore.as_retriever(search_kwargs={"k": 10})
ensemble = EnsembleRetriever(
    retrievers=[vector, bm25],
    weights=[0.6, 0.4],
)

# 第二层：重排序（精排）
reranker = FlashrankRerank(top_n=5)
production_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble,
)

# 使用
results = production_retriever.invoke("LCEL 怎么用")
for i, doc in enumerate(results, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"{i}. [score={score}] {doc.page_content[:50]}...")
```

### 3. 性能优化要点

- **控制基础召回数**：k=20 是性价比最高的起点，k>50 收益递减
- **模型缓存**：CrossEncoder 模型加载一次后复用，避免重复初始化
- **异步调用**：生产环境使用 `ainvoke` 避免阻塞
- **批量处理**：多个查询用 `batch` 或 `abatch` 并行处理

---

**版本**：v1.0
**最后更新**：2026-02-27
**数据来源**：LangChain 源码分析 + Context7 官方文档 + 社区最佳实践
