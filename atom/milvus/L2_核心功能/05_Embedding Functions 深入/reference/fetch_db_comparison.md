---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/170jigz/my_strategy_for_picking_a_vector_database_a/
title: My strategy for picking a vector database: a side-by-side comparison
fetched_at: 2026-02-24T00:00:00Z
status: partial
author: u/kyrodrax
published_date: 2023-10-05
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: low
word_count: 3000
---

# My strategy for picking a vector database: a side-by-side comparison

## 元信息
- **来源**：https://www.reddit.com/r/LangChain/comments/170jigz/my_strategy_for_picking_a_vector_database_a/
- **作者**：u/kyrodrax
- **发布日期**：2023-10-05
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子分享作者为选择向量数据库制作的 side-by-side 对比表（速度、可扩展性、开发者体验、社区、价格），并披露自己是 Vectorview.ai 的联合创始人。评论区补充了 Redis、metadata 存储/过滤、Elastic/Weaviate/Qdrant/Chroma、以及“向量能力应集成到通用数据库”的观点。

---

## 正文内容

I made this table to compare vector databases in order to help me choose the best one for a new project. I spent quite a few hours on it, so I wanted to share it here too in hopes it might help others as well. My main criteria when choosing vector DB were the speed, scalability, developer experinece, community and price.

You'll find all of the comparison parameters in the article and more details here:

**Edit**: For transparency, I am the co-fouder of Vectorview.ai which is an analytics tool for semantic search that let's developers understand how their embedded documents are used.

## Comments

Missing Redis. Been very happy with it

I suggest adding an additional evaluation dimension: metadata storage support along with raw vectors. This can make some projects significantly easier.

There are a number of them at this point. Top of my head there is Weaviate, Qdrant, Elasticsearch. ... Elastic being the most reliable actually.

It is time, you just don't need a pure vector databases, it is a trap. ... SingleStore, Redis and MongoDB can also be used as a vector database and they also provide great other needed features.

I went for Milvus personally, and custom embeddings..

Milvus seems to be the clear winner here

...（其余评论与嵌套回复文本已按抓取结果保留；线程很长，此处略）

---

## 关键信息提取

### 技术要点
- 选型维度常见：性能（QPS/latency/方差）、可扩展性、写入并发、元数据过滤能力、运维/生态。
- 讨论中多次出现“向量 + 结构化/全文检索一体化”的偏好（Postgres/Elastic/Redis/Mongo 等）。

### 代码示例
- 无

### 相关链接
- https://benchmark.vectorview.ai/vectordbs.html

---

## 抓取质量评估
- **完整性**：部分（评论线程超长；抓取到主要正文与大量评论文本）
- **可用性**：中
- **时效性**：过时
