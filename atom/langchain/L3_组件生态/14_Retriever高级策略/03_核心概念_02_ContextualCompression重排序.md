# Retriever高级策略 - 核心概念：ContextualCompression 重排序

> **两阶段检索范式**：ContextualCompressionRetriever 通过装饰器模式将重排序/压缩逻辑叠加在基础检索器之上，实现"先粗筛、再精排"的两阶段检索。

---

## 概述

向量检索的核心矛盾：召回率和精确率难以兼得。检索 Top-4 精确但可能漏掉好结果，检索 Top-20 召回高但噪声多。ContextualCompressionRetriever 用装饰器模式优雅地解决了这个问题——让基础检索器负责高召回粗筛，让压缩器/重排序器负责高精确精排。

这就是信息检索领域经典的 **两阶段检索范式（Two-Stage Retrieval）**：

```
用户查询 → [第一阶段: 粗筛 k=20] → [第二阶段: 精排 top_n=5] → 最终结果
```

---

## 两阶段检索：为什么比直接精确检索更好

### 直接检索的困境

```python
# 方案 A：直接检索少量文档 → 精确但可能漏掉好结果
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 方案 B：检索大量文档 → 召回高但噪声多，塞进 prompt 浪费 token
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
```

### 两阶段检索的优势

```
第一阶段（粗筛）          第二阶段（精排）
┌─────────────────┐     ┌─────────────────┐
│ 向量相似度检索    │     │ 交叉编码器打分    │
│ 速度快，成本低    │ ──→ │ 精度高，理解深    │
│ 召回 20 篇文档   │     │ 精选 5 篇文档    │
└─────────────────┘     └─────────────────┘
```

**类比**：招聘流程。HR 先从 1000 份简历中筛出 20 份（关键词匹配，快但粗），再由技术面试官从 20 份中选 5 个（深度评估，慢但准）。你不会让面试官看 1000 份简历，也不会让 HR 只挑 5 份。

---

## Retriever vs Compressor 架构分离

LangChain 将检索过程拆分为两个独立的关注点：

| 角色 | 接口 | 职责 | 输入 | 输出 |
|------|------|------|------|------|
| Retriever | `BaseRetriever` | 搜索逻辑 | query | `list[Document]` |
| Compressor | `BaseDocumentCompressor` | 后处理 | query + docs | `list[Document]`（子集） |

**为什么要分离？**

1. **单一职责**：Retriever 专注"从哪找"，Compressor 专注"怎么筛"
2. **自由组合**：任意 Retriever + 任意 Compressor，排列组合
3. **独立演进**：换重排序模型不影响检索逻辑，换向量库不影响重排序

```python
# 同一个 Compressor 可以搭配不同 Retriever
compressor = CrossEncoderReranker(model=model, top_n=5)

# 搭配向量检索
ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vector_retriever)

# 搭配混合检索
ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

# 搭配 BM25 检索
ContextualCompressionRetriever(base_compressor=compressor, base_retriever=bm25_retriever)
```

---

## ContextualCompressionRetriever 源码解析

### 类定义

```python
class ContextualCompressionRetriever(BaseRetriever):
    """装饰器模式：包装基础检索器，通过文档压缩器后处理结果"""

    base_compressor: BaseDocumentCompressor
    """压缩/重排序器"""

    base_retriever: RetrieverLike
    """底层检索器（BaseRetriever 或任何实现了 invoke 的对象）"""
```

只有两个属性，极其简洁。这就是装饰器模式的精髓——不改变原有对象，只在外面包一层。

### 核心方法

```python
def _get_relevant_documents(
    self,
    query: str,
    *,
    run_manager: CallbackManagerForRetrieverRun,
) -> list[Document]:
    # 第一阶段：调用基础检索器
    docs = self.base_retriever.invoke(
        query, config={"callbacks": run_manager.get_child()}
    )

    # 第二阶段：调用压缩器/重排序器
    compressed_docs = self.base_compressor.compress_documents(
        docs, query, callbacks=run_manager.get_child()
    )

    return list(compressed_docs)
```

**关键设计点**：

1. **回调传播**：`run_manager.get_child()` 确保 LangSmith 追踪链路完整
2. **接口统一**：返回值仍然是 `list[Document]`，对外透明
3. **装饰器透明性**：ContextualCompressionRetriever 本身也是 BaseRetriever，可以被再次装饰

### 执行流程图

```
用户调用 compression_retriever.invoke("RAG 是什么？")
    │
    ▼
ContextualCompressionRetriever._get_relevant_documents()
    │
    ├──→ base_retriever.invoke("RAG 是什么？")
    │       │
    │       ▼
    │    VectorStore.similarity_search()  →  返回 20 篇文档
    │
    ├──→ base_compressor.compress_documents(20篇文档, "RAG 是什么？")
    │       │
    │       ▼
    │    CrossEncoder 对每篇文档打分  →  按分数排序  →  取 Top-5
    │
    ▼
返回 5 篇高质量文档
```

---

## 重排序方案对比

LangChain 生态提供三种主流重排序方案，适用于不同场景。

### 方案一：CohereRerank（远程 API）

```python
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# 创建 Cohere 重排序器
compressor = CohereRerank(
    model="rerank-english-v3.0",  # 最新模型
    top_n=5,                       # 返回 Top-5
)

# 包装基础检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

# 使用
docs = compression_retriever.invoke("什么是 RAG？")
```

**注意**：`langchain.retrievers.document_compressors.CohereRerank` 已弃用，请使用 `langchain_cohere.CohereRerank`。

**源码行为**：
- 深拷贝文档（不修改原始文档）
- 将 `relevance_score` 注入到文档的 `metadata` 中
- 默认 `top_n=3`

**适用场景**：
- 无 GPU 环境
- 对精度要求极高（Cohere 模型效果业界领先）
- 可接受 API 调用延迟和成本

### 方案二：CrossEncoderReranker（本地模型）

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载本地 CrossEncoder 模型
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# 创建重排序器
compressor = CrossEncoderReranker(
    model=model,
    top_n=5,
)

# 包装
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

docs = compression_retriever.invoke("什么是 RAG？")
```

**源码解析（仅 51 行）**：

```python
class CrossEncoderReranker(BaseDocumentCompressor):
    model: BaseCrossEncoder
    top_n: int = 3

    def compress_documents(self, documents, query, callbacks=None):
        # 1. 构建 (query, doc) 对
        scores = self.model.score(
            [(query, doc.page_content) for doc in documents]
        )
        # 2. 按分数降序排列
        docs_with_scores = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        # 3. 取 Top-N
        return [doc for doc, _ in docs_with_scores[:self.top_n]]
```

**CrossEncoder vs Bi-Encoder 的区别**：

```
Bi-Encoder（向量检索用的）：
  query  → Encoder → 向量 ─┐
                            ├→ 余弦相似度
  doc    → Encoder → 向量 ─┘
  特点：query 和 doc 独立编码，速度快，但交互不够深

CrossEncoder（重排序用的）：
  [query, doc] → Encoder → 相关性分数
  特点：query 和 doc 联合编码，交互更深，精度更高，但速度慢
```

**推荐模型**：
- `BAAI/bge-reranker-base`：中文友好，效果好
- `cross-encoder/ms-marco-MiniLM-L-6-v2`：英文场景，速度快
- `BAAI/bge-reranker-v2-m3`：多语言，效果最好

### 方案三：FlashrankRerank（轻量级本地）

```python
from langchain_community.document_compressors import FlashrankRerank

# 创建 FlashRank 重排序器（CPU 友好）
compressor = FlashrankRerank(
    top_n=5,
    model="ms-marco-MultiBERT-L-12",  # 默认模型
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

docs = compression_retriever.invoke("什么是 RAG？")
```

**适用场景**：
- 纯 CPU 环境（无 GPU）
- 对延迟敏感
- 资源受限的部署环境

### 三种方案对比表

| 维度 | CohereRerank | CrossEncoderReranker | FlashrankRerank |
|------|-------------|---------------------|-----------------|
| 运行位置 | 远程 API | 本地 | 本地 |
| GPU 需求 | 无 | 推荐有 | 无（CPU 友好） |
| 精度 | 最高 | 高 | 中等 |
| 延迟 | 高（网络） | 中 | 低 |
| 成本 | 按调用计费 | 免费 | 免费 |
| 隐私 | 数据发送到 API | 数据本地 | 数据本地 |
| 中文支持 | 好 | 取决于模型 | 一般 |
| 安装依赖 | `langchain_cohere` | `sentence-transformers` | `flashrank` |

**选型建议**：
- 生产环境 + 预算充足 → CohereRerank
- 生产环境 + 隐私敏感 → CrossEncoderReranker + `bge-reranker-base`
- 快速原型 / CPU 环境 → FlashrankRerank

---

## DocumentCompressorPipeline：管道模式

当你需要组合多个压缩/转换步骤时，使用 `DocumentCompressorPipeline`。

### 核心设计

```python
class DocumentCompressorPipeline(BaseDocumentCompressor):
    """将多个转换器/压缩器链接成顺序管道"""

    transformers: list[BaseDocumentTransformer | BaseDocumentCompressor]
```

### 多态分发机制

Pipeline 内部根据组件类型自动选择调用方式：

```python
def compress_documents(self, documents, query, callbacks=None):
    for _transformer in self.transformers:
        if isinstance(_transformer, BaseDocumentCompressor):
            # Compressor：需要 query 参数
            documents = _transformer.compress_documents(
                documents, query, callbacks=callbacks
            )
        elif isinstance(_transformer, BaseDocumentTransformer):
            # Transformer：不需要 query，只做文档变换
            documents = _transformer.transform_documents(documents)
        else:
            raise ValueError(f"Unknown transformer type: {type(_transformer)}")
    return documents
```

**两种组件的区别**：
- `BaseDocumentCompressor`：需要 query（如重排序，需要知道查询才能打分）
- `BaseDocumentTransformer`：不需要 query（如文本分割、冗余过滤）

### 实际组合示例

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import OpenAIEmbeddings

# 步骤 1：去除冗余文档（Transformer，不需要 query）
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.95,  # 相似度 > 0.95 视为冗余
)

# 步骤 2：重排序（Compressor，需要 query）
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
    top_n=5,
)

# 组合成管道
pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, reranker]
)

# 包装成 Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

**执行流程**：
```
20 篇文档 → [去冗余: 20→15篇] → [重排序: 15→5篇] → 5 篇高质量文档
```

---

## 完整实战示例

### 场景：构建带重排序的 RAG 管道

```python
"""
完整示例：ContextualCompressionRetriever + CrossEncoder 重排序
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 1. 准备数据 ==========
docs = [
    Document(page_content="RAG 是检索增强生成的缩写，结合检索和生成两个阶段", metadata={"source": "intro"}),
    Document(page_content="向量数据库存储文档的 Embedding 表示，支持相似度检索", metadata={"source": "vector"}),
    Document(page_content="Chunking 是将长文档分割成小块的过程，影响检索质量", metadata={"source": "chunk"}),
    Document(page_content="CrossEncoder 通过联合编码 query 和 document 来计算相关性分数", metadata={"source": "rerank"}),
    Document(page_content="BM25 是基于词频的经典检索算法，擅长精确关键词匹配", metadata={"source": "bm25"}),
    Document(page_content="Prompt Engineering 是设计提示词以引导 LLM 生成高质量回答的技术", metadata={"source": "prompt"}),
    Document(page_content="LangChain 的 LCEL 表达式语言用于构建可组合的 AI 管道", metadata={"source": "lcel"}),
    Document(page_content="Temperature 参数控制 LLM 输出的随机性，0 表示确定性输出", metadata={"source": "llm"}),
]

# ========== 2. 创建向量存储 ==========
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# ========== 3. 创建两阶段检索器 ==========
# 第一阶段：向量检索，高召回
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 第二阶段：CrossEncoder 重排序，高精确
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# ========== 4. 构建 RAG 管道 ==========
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# ========== 5. 使用 ==========
answer = rag_chain.invoke("RAG 的核心流程是什么？")
print(answer)
```

### 对比实验：有无重排序的效果差异

```python
"""
对比实验：直接检索 vs 重排序检索
"""

query = "CrossEncoder 重排序的工作原理"

# 方案 A：直接检索 Top-3
direct_docs = vectorstore.as_retriever(
    search_kwargs={"k": 3}
).invoke(query)

print("=== 直接检索 Top-3 ===")
for i, doc in enumerate(direct_docs):
    print(f"  [{i+1}] {doc.page_content[:50]}...")

# 方案 B：检索 Top-6 → 重排序 → Top-3
reranked_docs = compression_retriever.invoke(query)

print("\n=== 重排序后 Top-3 ===")
for i, doc in enumerate(reranked_docs):
    print(f"  [{i+1}] {doc.page_content[:50]}...")

# 通常你会发现：重排序后，最相关的文档排名更靠前
```

---

## 生产最佳实践

### 推荐管道配置

```python
# 生产级推荐配置
# Retriever(k=20) → Rerank(top_n=5) → LLM

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}  # 粗筛：高召回
)

compressor = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
    top_n=5,  # 精排：高精确
)

production_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

### 参数调优指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 粗筛 k | 15-30 | 太小漏结果，太大浪费重排序时间 |
| 精排 top_n | 3-5 | 取决于 Context Window 和 prompt 设计 |
| 粗筛/精排比 | 4:1 ~ 6:1 | 经验值，k=20 → top_n=5 |

### 成本 vs 效果权衡

```
无重排序:     Retriever(k=5)  → LLM
              成本低，精度一般

本地重排序:   Retriever(k=20) → CrossEncoder → Top-5 → LLM
              成本低，精度高，需要模型加载时间

API 重排序:   Retriever(k=20) → CohereRerank → Top-5 → LLM
              成本中等，精度最高，有网络延迟
```

### 常见陷阱

1. **粗筛 k 太小**：重排序只能从已检索的文档中选，粗筛漏掉的文档无法被找回
2. **忘记传播回调**：自定义实现时，确保 `callbacks` 参数正确传递，否则 LangSmith 追踪断裂
3. **混淆 Compressor 和 Retriever**：Compressor 不能独立使用，必须搭配 Retriever

---

## 总结

ContextualCompressionRetriever 的核心价值：

1. **装饰器模式**：不修改原有 Retriever，只在外面包一层后处理
2. **两阶段检索**：粗筛（高召回）+ 精排（高精确），兼得鱼和熊掌
3. **自由组合**：任意 Retriever + 任意 Compressor，通过 Pipeline 还能链接多个步骤
4. **生产标配**：2025-2026 年，Hybrid Search + Reranking 已成为 RAG 生产环境的默认管道

记住这个公式：**好的 RAG 检索 = 高召回粗筛 + 高精确精排**。

---

**版本**: v1.0
**最后更新**: 2026-02-27
**数据来源**: LangChain 源码分析 + Context7 官方文档 + 社区最佳实践
