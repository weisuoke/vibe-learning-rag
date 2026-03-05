---
type: fetched_content
source: https://www.reddit.com/r/Rag/comments/1hdd3u2/which_embedding_model_should_i_use_need_help/
title: Which embedding model should I use??? NEED HELP!!!
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: medium
word_count: 1600
---

# Which embedding model should I use??? NEED HELP!!!

## 元信息
- **来源**：https://www.reddit.com/r/Rag/comments/1hdd3u2/which_embedding_model_should_i_use_need_help/
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子提问：POC 场景下使用 AllMiniLM v6 embedding，文档多或上下文长时无法生成 embedding，询问是否有支持更大上下文的 embedding 模型（免费优先）。回复集中强调：不要把整篇长文一次性 embed，应先做 chunking；可参考 MTEB leaderboard；OpenAI 支持更长 token，但更通用的办法是小 chunk + 向量库检索；也有人提到 VoyageAI 的免费层、以及通过 RAG evaluation 对 chunking 策略做对比。

---

## 正文内容

I am currently using AllminiLM v6 as the embedding model for my RAG Application. When I tried with more no. of documents or documents with large context, the embedding was not created. It is for POC and I don't have the budget to go with any paid services.

Is there any other embedding model that supports large context?

Paid or free.... but free is more preferred..!!

## 评论区

- Uhm dude. Just break the context up into smaller chunks, embed each chunk, then compute the average embedding among them.
- Look at MTEB leaderboard for which models perform best. You can’t embed entire multipage documents and expect reasonable results. ... Look at semchunk for chunking
- ...（其余抓取到的嵌套评论文本保留，略）

---

## 关键信息提取

### 技术要点
- “大上下文 embedding”不是主要解法；chunking + 检索才是 RAG 的通用模式。
- 选择模型建议结合 MTEB 与自身数据 eval；chunk size/overlap 需要实验。
- 商业 API 可能有更长 token 上限，但成本与速率限制需要权衡。

### 代码示例
- 无

### 相关链接
- https://huggingface.co/spaces/mteb/leaderboard
- https://github.com/FullStackRetrieval-com/semchunk

---

## 抓取质量评估
- **完整性**：部分（Reddit 动态加载/工具限制）
- **可用性**：中
- **时效性**：较新
