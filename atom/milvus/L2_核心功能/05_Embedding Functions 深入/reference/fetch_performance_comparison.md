---
type: fetched_content
source: https://www.reddit.com/r/singularity/comments/1gyu5ud/help_fastest_reliable_embedding_model_for_300gb/
title: [Help] Fastest reliable embedding model for 300GB corpus? (OpenAI too slow, BGE unreliable)
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date: 2024-11-25
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: medium
word_count: 2600
---

# [Help] Fastest reliable embedding model for 300GB corpus? (OpenAI too slow, BGE unreliable)

## 元信息
- **来源**：https://www.reddit.com/r/singularity/comments/1gyu5ud/help_fastest_reliable_embedding_model_for_300gb/
- **作者**：
- **发布日期**：2024-11-25
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
帖子讨论对 300GB 纯文本做 embedding 的模型/方案选择：抱怨 OpenAI text-embedding-3-small 的“未注明限额/排队限制”导致周期过长，考虑开源 BGE/GTE、Nomic、Azure OpenAI、以及配额/批处理策略。高赞回复强调：评估任务规模（token 量）、用 MTEB 对比、GPU 本地推理吞吐、以及把 embedding 结果高效存储（parquet + 压缩 + 对象存储）。

---

## 正文内容

I have 300 GB of plaintext I would like to generate embeddings for. Currently using text-embedding-3-small from OpenAI but it has undocumented rate limit. This means it’ll take 6 months which is too long a wait for me. Any other model you would recommend?

Options I’ve already looked into:

- bge models - supposed to be as good as OpenAI models on MTEB benchmark but is said to underperform in production
- voyage ai - too expensive. I’m ready to pay $0.01/1M tokens, their price is too much
- llama3 based embedding models on huggingface - haven’t yet figured out which ones have good performance in production.

Would love advice on same?

## Comments

- 300GB is 3x the size of Wikipedia as a whole as far as I know. Holy Shit!

- Hey, this is one of my areas of specialty ... You’re going to want to check out MTEB ... You’re 100% going to need a GPU ...

- ...（其余抓取到的评论与回复文本保留，略）

---

## 关键信息提取

### 技术要点
- 大规模 embedding 的瓶颈往往是配额/吞吐与队列，而不是“模型是否能一次吃下长文”。
- 评估方法：MTEB + 自己数据的小比例试验（先 5%）+ 看吞吐/延迟/成本。
- 工程建议：GPU 本地推理；embedding 存储用列式压缩（parquet）与对象存储；必要时降低维度以降成本。

### 代码示例
```python
import litserve as ls
from sentence_transformers import SentenceTransformer

class EmbeddingsAPI(ls.LitAPI):
    def setup(self, device):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def predict(self, inputs):
        embeddings = self.model.encode(inputs)
        return embeddings

if __name__ == "__main__":
    api = EmbeddingsAPI()
    server = ls.LitServer(api, spec=ls.OpenAIEmbeddingSpec())
    server.run(port=8000)
```

### 相关链接
- https://www.reddit.com/r/singularity/comments/1gyu5ud/help_fastest_reliable_embedding_model_for_300gb/

---

## 抓取质量评估
- **完整性**：部分（Reddit 动态加载/工具限制；抓取到主要正文与大量评论文本）
- **可用性**：中
- **时效性**：较新
