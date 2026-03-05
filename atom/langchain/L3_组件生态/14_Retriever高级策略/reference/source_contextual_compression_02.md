---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/retrievers/contextual_compression.py
  - libs/langchain/langchain_classic/retrievers/document_compressors/base.py
  - libs/langchain/langchain_classic/retrievers/document_compressors/cohere_rerank.py
  - libs/langchain/langchain_classic/retrievers/document_compressors/cross_encoder_rerank.py
analyzed_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
---

# 源码分析：ContextualCompression 与 Reranking

## 分析的文件
- `contextual_compression.py` - 上下文压缩检索器
- `document_compressors/base.py` - 压缩器管道
- `document_compressors/cohere_rerank.py` - Cohere 重排序
- `document_compressors/cross_encoder_rerank.py` - CrossEncoder 重排序

## 关键发现

### ContextualCompressionRetriever(BaseRetriever)
装饰器模式：包装基础检索器，通过文档压缩器后处理结果。

**属性：**
- `base_compressor`: BaseDocumentCompressor - 压缩/重排序器
- `base_retriever`: RetrieverLike - 底层检索器

**流程：** Query → base_retriever.invoke() → base_compressor.compress_documents() → 返回压缩后的文档

### DocumentCompressorPipeline(BaseDocumentCompressor)
管道模式：将多个转换器/压缩器链接成顺序管道。

**属性：**
- `transformers`: list[BaseDocumentTransformer | BaseDocumentCompressor]

**多态分发：**
- BaseDocumentCompressor → compress_documents(docs, query, callbacks)
- BaseDocumentTransformer → transform_documents(docs)（无 query）

### CohereRerank(BaseDocumentCompressor) [已弃用]
适配器模式：将 Cohere SDK 包装在 BaseDocumentCompressor 接口后面。
- model: "rerank-english-v2.0"（默认）
- top_n: 3（默认）
- 深拷贝文档，注入 relevance_score 到 metadata

### CrossEncoderReranker(BaseDocumentCompressor)
最简单的重排序器（51行）：
- model: BaseCrossEncoder（必需）
- top_n: 3（默认）
- 创建 (query, doc.page_content) 对 → model.score() → 按分数降序排列
- 不修改文档 metadata

### Retriever vs Compressor 架构区别
- **Retriever**：接收查询字符串，返回文档。拥有搜索逻辑。
- **Compressor**：接收查询 + 已检索文档，返回过滤/重排后的子集。是后处理阶段。
