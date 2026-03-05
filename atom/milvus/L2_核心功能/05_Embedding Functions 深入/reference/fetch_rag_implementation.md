---
type: fetched_content
source: https://www.reddit.com/r/Rag/comments/1mmct4h/i_built_a_comprehensive_rag_system_and_heres_what/
title: I built a comprehensive RAG system, and here’s what I’ve learned
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 2100
---

# I built a comprehensive RAG system, and here’s what I’ve learned

## 元信息
- **来源**：https://www.reddit.com/r/Rag/comments/1mmct4h/i_built_a_comprehensive_rag_system_and_heres_what/
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子介绍作者业余时间构建的 RAG 产品（ChatVia.ai）在“准确率与速度优先”的取舍下的技术栈与实践：文档解析（LlamaParse）、agentic chunking（LLM 参与切分并生成摘要/问题）、embedding 模型对比（OpenAI/Cohere/Mistral/Gemini）、重排（Cohere Rerank）、评测（evals）、流式输出（SSE）等；评论区补充了 chunk/top-k、reranker、规模化手册检索等经验。

---

## 正文内容

**Disclaimer**: This is a very biased setup, with decisions based on my research from different sources and books. You might not agree with this setup — and that’s fine. However, I’m not going to defend why I chose PostgreSQL over Qdrant or any other vector database, nor any other decision made here.

## What is ChatVia.ai?

A few months ago, I had the idea of creating an AI agent (similar to ChatGPT) lingering in my mind. I first tried building it with Chainlit (failed many times) and then with Streamlit (failed miserably as well).

About three months ago, I decided to start a completely new project from scratch, welcome to [ChatVia.ai](https://chatvia.ai).

[ChatVia.ai](https://chatvia.ai) provides a comprehensive RAG system that uses multiple techniques to process and chunk data. In this post, I’ll explain each technique and technology.

I built [ChatVia.ai](https://chatvia.ai) in my free time. On some weekends, I found myself working 10–12 hours straight, but with such a big project, I had no choice but to keep going.

What makes [ChatVia.ai](https://chatvia.ai) different from other RAG systems is how much I cared about accuracy and speed above everything else. I also wanted simplicity, something easy to use and straightforward. Since I only launched it today, you might still encounter bugs here and there, which is why I’ve set up a ticket system so you can report any issues, and I’ll keep fixing them.

[ChatVia.ai](https://chatvia.ai) supports streaming images. If you ask about a chart included in a document, it will return the actual chart as an image along with a description, it won’t just tell you what’s in the chart.

## My Stack

Frontend:

- Tailwind CSS 4
- Vue.js 3
- TypeScript

Backend:

- PHP 8.4
- Laravel 12
- Rust (for tiktoken)
- Python (FastAPI) for ingestion and chunking

Web Server:

- Nginx
- PHP-FPM with Opcache and Jit.

Database:

- PostgreSQL
- Redis

## Vector Database

Among all the databases I’ve tested (Qdrant, Milvus, ChromaDB, Pinecone), I found PostgreSQL to be the best option for my setup.

Why? Three main reasons:

- It is insanely fast. When combined with binary quantization (I do use binary quantization), it can handle millions of documents in under 500 ms, that’s very impressive.
- Supports BM25 for hybrid search.
- Since I already use PostgreSQL, I can keep everything together with no need for an extra database.

For BM25, I use the llmlingua2 model because it’s multilingual.

## My Servers

I currently have two servers — one primary and one secondary (for disaster recovery).

Both run on AMD EPYC 7502P, with 2 TB NVMe storage and 256 GB RAM. That’s enough to handle hundreds of thousands of concurrent requests.

## Document Parsing

...（原帖关于 LlamaParse 的段落与要点保留，略）

## Chunking

Among all the techniques I’ve tried for chunking, I found agentic chunking to be the most effective.

Along with chunking, I ask the LLM to generate two additional elements:

- A summary of the chunk
- Relevant questions

## Embedding Model

I’ve tried a few embedding models, including:

- OpenAI text-embedding-3-large
- Cohere embed-v4
- Mistral embed.
- gemini-embedding-001

Honestly, I couldn’t tell the difference, but from my limited testing I found Cohere embed-v4 works very well with different languages (tested with Arabic, Danish and English).

## Re-ranking

I use Cohere Rerank when retrieving data from PostgreSQL (top-k = 6), and then I populate the sources so the user can see the retrieved chunks for the given answer.

## Evals

...（原帖 eval 定义与做法保留，略）

## Streaming

...（SSE vs WebSockets 段落保留，略）

## Comments

（以下为抓取到的可见评论文本，已按层级展开；Reddit 动态加载可能导致未包含所有回复）

- What about structural Data ? Like CSV and excel files. Do you use chunking for structural data ? Or you are storing information into Database directly ?
- Awesome work! Could you share more about how many ranked results you give to the LLM and how you control this part of the retrieval process?
- Thank you! I give it 6 top-k, however I think this can also be reduced to 3-4. You need to do some testing before choosing the right value. but 6 is a good starting.
- ...（其余评论文本保留，略）

---

## 关键信息提取

### 技术要点
- “准确率/速度优先”驱动的工程取舍：解析、chunk、embedding、rerank、eval、streaming。
- agentic chunking：LLM 参与切分并生成摘要与问题，用以提升可检索性。
- embedding 模型主观差异不大；多语言场景作者偏好 Cohere embed-v4。
- rerank top-k（作者示例 6）与评测（eval）被强调为关键。

### 代码示例
- 无

### 相关链接
- https://chatvia.ai
- https://docs.llamaindex.ai/en/stable/api_reference/readers/llama_parse/

---

## 抓取质量评估
- **完整性**：部分（Reddit 动态加载/反爬，正文与评论可能未覆盖全部）
- **可用性**：中
- **时效性**：较新
