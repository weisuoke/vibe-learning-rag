---
type: fetched_content
source: https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md
title: We Benchmarked 20+ Embedding APIs with Milvus: 7 Insights That Will Surprise You
fetched_at: 2026-02-24T00:00:00Z
status: success
author: Jeremy Zhu
published_date: 2025-05-22
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: article
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 1900
---

# We Benchmarked 20+ Embedding APIs with Milvus: 7 Insights That Will Surprise You

## 元信息
- **来源**：https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md
- **作者**：Jeremy Zhu
- **发布日期**：2025-05-22
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
文章用 Milvus 的 TextEmbedding Function 在北美与亚洲环境下，对 20+ embedding provider/模型做端到端延迟测试，总结出网络地理位置影响远大于模型结构、不同 provider 的 batch/token trade-off 差异、API 可靠性方差风险、CPU 本地推理可竞争、以及 Milvus 自身引入的额外开销很小等 7 点结论，并给出 RAG 生产优化建议。

---

## 正文内容

# We Benchmarked 20+ Embedding APIs with Milvus: 7 Insights That Will Surprise You

**May 22, 2025**  
**Jeremy Zhu**

Probably every AI developer has built a RAG system that works perfectly… in their local environment.

You’ve nailed the retrieval accuracy, optimized your vector database, and your demo runs like butter. Then you deploy to production and suddenly:

- Your 200ms local queries take 3 seconds for actual users
- Colleagues in different regions report completely different performance
- The embedding provider you chose for “best accuracy” becomes your biggest bottleneck

What happened? Here’s the performance killer no one benchmarks: **embedding API latency**.

While MTEB rankings obsess over recall scores and model sizes, they ignore the metric your users feel—how long they wait before seeing any response. We tested every major embedding provider across real-world conditions and discovered extreme latency differences they’ll make you question your entire provider selection strategy.

Spoiler: The most popular embedding APIs aren’t the fastest. Geography matters more than model architecture. And sometimes a $20/month CPU beats a $200/month API call.

## Why Embedding API Latency Is the Hidden Bottleneck in RAG

...（正文已按“激进清洗”去除导航/推荐块；保留核心段落、列表与表格）

## Measuring Real-World Embedding API Latency with Milvus

Milvus offers a new `TextEmbedding` Function interface. Using this, we benchmarked embedding APIs from providers such as OpenAI, Cohere, AWS Bedrock, Google Vertex AI, Voyage AI, AliCloud Dashscope, SiliconFlow, and TEI.

| Provider | Model | Dimensions |
|---|---|---|
| OpenAI | text-embedding-ada-002 | 1536 |
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| AWS Bedrock | amazon.titan-embed-text-v2:0 | 1024 |
| Google Vertex AI | text-embedding-005 | 768 |
| Google Vertex AI | text-multilingual-embedding-002 | 768 |
| VoyageAI | voyage-3-large | 1024 |
| VoyageAI | voyage-3 | 1024 |
| VoyageAI | voyage-3-lite | 512 |
| VoyageAI | voyage-code-3 | 1024 |
| Cohere | embed-english-v3.0 | 1024 |
| Cohere | embed-multilingual-v3.0 | 1024 |
| Cohere | embed-english-light-v3.0 | 384 |
| Cohere | embed-multilingual-light-v3.0 | 384 |
| Aliyun Dashscope | text-embedding-v1 | 1536 |
| Aliyun Dashscope | text-embedding-v2 | 1536 |
| Aliyun Dashscope | text-embedding-v3 | 1024 |
| Siliconflow | BAAI/bge-large-zh-v1.5 | 1024 |
| Siliconflow | BAAI/bge-large-en-v1.5 | 1024 |
| Siliconflow | netease-youdao/bce-embedding-base_v1 | 768 |
| Siliconflow | BAAI/bge-m3 | 1024 |
| Siliconflow | Pro/BAAI/bge-m3 | 1024 |
| TEI | BAAI/bge-base-en-v1.5 | 768 |

## 7 Key Findings from Our Benchmarking Results

### 1. Global Network Effects Are More Significant Than You Think

Cross-region calls often increase latency by **3–4×**; for some providers the degradation can be far worse.

### 2. Model Performance Rankings Reveal Surprising Results

North America-based: Cohere > Vertex AI > VoyageAI > OpenAI > Bedrock  
Asia-based: SiliconFlow > Dashscope

### 3. Model Size Impact Varies Dramatically by Provider

Some providers show minimal latency gap across model sizes; others show clear scaling effects.

### 4. Token Length and Batch Size Create Complex Trade-offs

Batch=1→10 usually increases latency 2×–5× while improving throughput; behavior differs by provider.

### 5. API Reliability Introduces Production Risk

Latency variance (especially for some providers) creates unpredictability; implement retries/timeouts/circuit breakers.

### 6. Local Inference Can Be Surprisingly Competitive

CPU-based TEI with mid-size models can be competitive; local inference can beat expensive APIs in some scenarios.

### 7. Milvus Overhead Is Negligible

Milvus adds ~20–40ms overhead in tests, typically <5% of total time versus embedding API latency.

## Tips: How to Optimize Your RAG Embedding Performance

- Localize testing in your deployment region
- Geo-match providers
- Optimize batch/chunk configs per provider
- Cache frequent query embeddings
- Consider local inference when cost/latency/privacy demands it

## Conclusion

Embedding API latency can dominate user-perceived latency; geography and provider infrastructure matter as much as model accuracy.

---

## 关键信息提取

### 技术要点
- embedding API 延迟是 RAG 生产链路的关键瓶颈，跨区域网络会显著放大延迟。
- batch size 与 token length 的影响高度依赖 provider 的后端 batching 策略。
- API 延迟方差与稳定性是生产风险，需要工程化治理。
- 本地推理（如 TEI）在某些场景下性价比/时延具有竞争力。
- Milvus TextEmbedding Function 额外开销很小。

### 代码示例
- 文中主要为 cURL/notebook 链接与参数示例（此处不重复粘贴）。

### 相关链接
- https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md
- https://github.com/zhuwenxing/text-embedding-bench

---

## 抓取质量评估
- **完整性**：完整
- **可用性**：高
- **时效性**：较新
