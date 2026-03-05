---
type: fetched_content
source: https://www.reddit.com/r/vectordatabase/comments/15deqny/milvus_creating_vector_embedding_with_langchain/
title: [MILVUS] Creating vector embedding with LangChain and OpenAI
fetched_at: 2026-02-24T00:00:00Z
status: partial
author: "[deleted]"
published_date: 2023-07-30
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: medium
word_count: 420
---

# [MILVUS] Creating vector embedding with LangChain and OpenAI

## 元信息
- **来源**：https://www.reddit.com/r/vectordatabase/comments/15deqny/milvus_creating_vector_embedding_with_langchain/
- **作者**：[deleted]
- **发布日期**：2023-07-30
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子询问：用 LangChain + OpenAI 生成 PDF embeddings 写入 Milvus 后，是否每次问答都要重复从 PDF 生成 embeddings；以及如何实现长期使用（持久化/复用）与延迟优化。评论指出 embedding 可持久化存储在 Milvus（磁盘），后续直接做向量检索并把命中 chunk 提供给 LangChain，无需反复 ingestion。

---

## 正文内容

I just started using Milvus a few days ago so this will most likely be a naive question, but I would still appreciate if anyone could help me towards getting an answer.

I've been following this

Let's say I have a .pdf with data, I used LangChain to generate the embeddings and successfully saved everything inside just like it is shown in the link above. My question is whether I need to create these embeddings from given .pdf all the time so I could create question-answer system? It was shown how it is done for one time use, but not how could I do this over a longer period? So am I supposed to constantly do the steps that were provided there? It also takes about 15 seconds to get an answer so that's a huge limitation as well and I would like to improve that. I was wondering if it is possible that I can just do normal vector search after creating the vector embeddings from .pdf for the first time since the link provided above didn't discuss anything of that.

Thanks to everyone who is willing to help (and sorry for this basic question but I couldn't find a definitive answer regarding that). Have a nice day!

## 评论

### [deleted] (0 points, ~2 years ago)

you should be able to persistently store the embeddings in your Milvus instance ie on disk, it'll be in a file under `<something something>/.milvus.io/`

not understanding the pdf repeated ingestion questions or the question about q/a, why are you ingesting it multiple times?

### [deleted] (0 points, ~2 years ago)

Any reason you are using Milvus. Could leverage weaviate, pinecone or qdrant that have cloud versions tha could persist the vectors for you so you generate once and then store there.

### [deleted] (1 point, ~2 years ago)

I believe I understand what you are asking because I had a similar question. My (somewhat limited) understanding is right now that you are grabbing the .pdf and creating a vector (a numerical representation of the text in that pdf) and using the vector to feed Langchain to ask a question based on that vector information (the .pdf)

Milvus allows you to store that vector so that the vector (just a list of numbers representing the text inside the .pdf file) can later be queried (without recalculation from the original document) based on similarity searches.

So the change is that instead of reading in the .pdf and creating the vector the next time you want to use that .pdf in your question, you simply query Milvus for the vector information you stored there the first time, and you feed langchain the stored vector to ask your question. Milvus will provide you with the ability to skip the 2 steps that are taking so much time - reading the .pdf file and creating the vectors from the text by just giving you the vector directly.

I hope that helps clarify.

---

## 关键信息提取

### 技术要点
- embeddings 持久化：一次 ingestion 后存入 Milvus，后续问答只需 query embedding + vector search。
- 低延迟关键：避免重复解析 PDF 与重复 embedding；按 chunk 存储并检索最相关 chunk。

### 代码示例
- 无

### 相关链接
- https://milvus.io/docs/integrate_with_langchain.md

---

## 抓取质量评估
- **完整性**：部分（页面显示评论数可能更多；抓取到 3 条顶级评论）
- **可用性**：中
- **时效性**：过时
