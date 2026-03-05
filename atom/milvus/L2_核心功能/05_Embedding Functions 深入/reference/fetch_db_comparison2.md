---
type: fetched_content
source: https://www.reddit.com/r/LocalLLaMA/comments/1e63m16/vector_database_pgvector_vs_milvus_vs_weaviate/
title: Vector database : pgvector vs milvus vs weaviate.
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date: 2024-07-18
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: low
word_count: 3600
---

# Vector database : pgvector vs milvus vs weaviate.

## 元信息
- **来源**：https://www.reddit.com/r/LocalLLaMA/comments/1e63m16/vector_database_pgvector_vs_milvus_vs_weaviate/
- **作者**：
- **发布日期**：2024-07-18
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子给出对 pgvector、Milvus、Weaviate 的优缺点（部分内容疑似来自 Google/LLM 总结），评论区大量讨论实际生产经验：pgvector+pgvectorscale、Postgres FTS 做 hybrid、Milvus（含 Milvus Lite）学习曲线与稳定性、Weaviate 的模块化与 hybrid、以及 LanceDB/Qdrant/Elastic 等替代选项。

---

## 正文内容

Any recommendation? From google

## Weaviate

Strengths:

- Focus on semantic search: Weaviate excels at understanding the meaning behind search queries, going beyond simple keyword matching. It uses GraphQL for querying, making it flexible and powerful.
- Built-in modules: Offers modules for text, image, and other data types, simplifying integration.
- Strong community: Active and growing community with good documentation and support.

Weaknesses:

- Performance: Can be slower than Milvus for very large datasets and high-throughput scenarios.
- Maturity: Relatively newer compared to Milvus, so some features might be less mature.

## Milvus

Strengths:

- High performance: Designed for speed and scalability, handling massive datasets and high query volumes efficiently.
- Mature ecosystem: Well-established with a large community, extensive documentation, and integrations with various tools.
- Flexibility: Supports various indexing algorithms and distance metrics, allowing for customization.

Weaknesses:

- Less focus on semantic search: Primarily focused on vector similarity search, lacking Weaviate's semantic understanding capabilities.
- Steeper learning curve: Can be more complex to set up and configure compared to Weaviate.

## pgvector

Strengths:

- Simplicity: Leverages the familiarity and power of PostgreSQL, making it easy to integrate into existing PostgreSQL workflows.
- Cost-effective: Utilizes existing PostgreSQL infrastructure, potentially reducing costs compared to standalone vector databases.
- Good performance: Offers decent performance for moderate-sized datasets.

Weaknesses:

- Scalability: May struggle with very large datasets and high query loads compared to Milvus.
- Limited features: Fewer features and customization options compared to Weaviate and Milvus.

...（评论区抓取到大量内容，包括 pgvector 的 metadata filtering、Milvus 官方回复、LanceDB/Elastic/Qdrant/Weaviate 经验等；线程很长，此处略）

---

## 关键信息提取

### 技术要点
- pgvector 的优势常来自“与现有 Postgres 生态整合 + SQL/metadata filtering + 运维简化”；但规模化与 ANN 索引策略需结合业务评测。
- Milvus 强调生产级可扩展、索引多样与生态（Milvus Lite 降门槛）；也有用户反馈大批量写入稳定性问题。
- Weaviate 以模块化、hybrid（BM25+向量）与插件化 embedding/rerank 为卖点。

### 代码示例
- 无

### 相关链接
- https://www.reddit.com/r/LocalLLaMA/comments/1e63m16/vector_database_pgvector_vs_milvus_vs_weaviate/

---

## 抓取质量评估
- **完整性**：部分（评论线程很长；当前保留正文与部分代表性评论）
- **可用性**：中
- **时效性**：较新
