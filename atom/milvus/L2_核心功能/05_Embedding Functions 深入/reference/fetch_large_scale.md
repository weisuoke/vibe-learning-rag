---
type: fetched_content
source: https://www.reddit.com/r/Rag/comments/1qcgoo8/need_help_embedding_250m_vectors_chunks_at_1024/
title: need help embedding 250M vectors / chunks at 1024 dims, should I self host embedder (BGE-M3) and self host Qdrant OR use voyage-3.5 or 4?
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date: 2026-01-14
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 2600
---

# need help embedding 250M vectors / chunks at 1024 dims, should I self host embedder (BGE-M3) and self host Qdrant OR use voyage-3.5 or 4?

## 元信息
- **来源**：https://www.reddit.com/r/Rag/comments/1qcgoo8/need_help_embedding_250m_vectors_chunks_at_1024/
- **作者**：
- **发布日期**：2026-01-14
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子讨论法律检索 RAG：1.5TB 语料、预计 2.5 亿向量（1024 dims），在自建 embedding+向量库（如 BGE-M3 + Qdrant/Milvus）与使用 Voyage 等商业 API 之间权衡，包含成本估算、吞吐与延迟、chunk 参数、以及社区关于评测、维度、网络瓶颈与自托管经验的建议。

---

## 正文内容

hey redditors, I am building a legal research RAG tool for law firms, just research and nothing else.

I have around 1.5TB of legal precedence data, parsed them all using 64 core Azure VM, using PyMuPDF + Layout + Pro. Using custom scripts and getting around 30 - 150 files / second parse speed.

Voyage-3-large surpassed voyage-law-2 and now gemini 001 embedder is ranked #2 (MTEB ranking). Domain specific models are now overthrown by general embedders.

I have around 250 million vectors to embed, and even using voyage-3.5 (0.06$/mill token), the cost is around $3k dollars.

Using Qdrant cloud will be another $500.

Question I need help with:

1) Should I self host embedder and vectorDB? (for chunking as well retrival later on)

2) Bear one time cost of it and be hastle free?

Feel free to DM me for the parsing and chunking and embedding scripts. Using BM25 + RRF + Hybrid search + Rerank using voyage-rank2.5, CRAG + Web Search.

Current latency woth 2048 dims on test dataset of 400k legal text vectors is 5 seconds.

Chunking by characters and not token.

| Metric | Value |
| --- | --- |
| **Avg parsed file size** | 68.5 KB |
| **Sample text length** | 2,521 chars (small doc) |
| **Total PDFs** | 16,428,832 |
| **Chunk size** | 4,096 chars (~1,024 tokens) |
| **Chunk overlap** | 512 chars (~128 tokens) |
| **Min chunk size** | 256 chars |

## Comments

- Do you have an eval set that would allow you to test recall etc. on a subset of the corpus? ...
- ...（评论区完整文本抓取到的部分已保留；此处为节省重复显示，略）

---

## 关键信息提取

### 技术要点
- 规模化 embedding 的关键约束：总 token/请求速率、批处理、网络/服务稳定性、以及向量维度对存储/延迟的影响。
- 社区建议包括：先做 eval/小样本评测；降低维度；排查 400k 向量 5s 延迟的瓶颈（可能是网络/管线调用）；自托管 Milvus 在合适机器上可低于 0.5s（评论中有自述）。
- 自托管与云服务的 trade-off：一次性硬件/运维 vs 直接付费与可预测性。

### 代码示例
- 无

### 相关链接
- https://www.reddit.com/r/Rag/comments/1qcgoo8/need_help_embedding_250m_vectors_chunks_at_1024/

---

## 抓取质量评估
- **完整性**：部分（Reddit 动态加载/工具限制，评论与作者信息可能不全）
- **可用性**：中
- **时效性**：最新
