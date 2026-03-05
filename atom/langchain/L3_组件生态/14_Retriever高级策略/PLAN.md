# 14_Retriever高级策略 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_ensemble_retriever_01.md - EnsembleRetriever 混合检索核心实现
- ✓ reference/source_contextual_compression_02.md - ContextualCompression 与 Reranking
- ✓ reference/source_multi_query_03.md - MultiQueryRetriever 多查询检索器
- ✓ reference/source_parent_document_04.md - MultiVectorRetriever 与 ParentDocumentRetriever
- ✓ reference/source_self_query_05.md - SelfQueryRetriever 自查询检索器

### Context7 官方文档
- ✓ reference/context7_langchain_retriever_01.md - LangChain Retriever 高级策略官方文档

### 网络搜索
- ✓ reference/search_hybrid_reranking_01.md - 混合检索与重排序最佳实践 2025-2026

### 待抓取链接（将由第三方工具自动保存到 reference/）
- [ ] https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking
- [ ] https://thedataguy.pro/blog/2025/05/evaluating-advanced-rag-retrievers

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_01_EnsembleRetriever混合检索.md - RRF 融合 BM25+向量检索 [来源: 源码/Context7/网络]
- [ ] 03_核心概念_02_ContextualCompression重排序.md - 上下文压缩与重排序管道 [来源: 源码/Context7/网络]
- [ ] 03_核心概念_03_MultiQueryRetriever查询扩展.md - LLM 驱动的多查询检索 [来源: 源码/Context7]
- [ ] 03_核心概念_04_ParentDocumentRetriever父文档检索.md - 小到大检索策略 [来源: 源码/Context7]
- [ ] 03_核心概念_05_SelfQueryRetriever自查询.md - LLM 结构化查询生成 [来源: 源码/Context7]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_01_BM25向量混合检索.md - EnsembleRetriever + BM25 + ChromaDB [来源: 源码/Context7/网络]
- [ ] 07_实战代码_02_Rerank重排序管道.md - ContextualCompression + CrossEncoder/FlashRank [来源: 源码/Context7/网络]
- [ ] 07_实战代码_03_MultiQuery查询扩展.md - LLM 多查询生成与检索 [来源: 源码/Context7]
- [ ] 07_实战代码_04_ParentDocument父文档检索.md - 两层/三层文档检索 [来源: 源码/Context7]
- [ ] 07_实战代码_05_生产级组合策略.md - Hybrid + Rerank + LCEL 完整 RAG 管道 [来源: 源码/Context7/网络]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [ ] 阶段二：补充调研（针对需要更多资料的部分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）
