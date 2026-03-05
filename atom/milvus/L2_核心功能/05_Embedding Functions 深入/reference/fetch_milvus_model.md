---
type: fetched_content
source: https://github.com/milvus-io/milvus-model
title: GitHub - milvus-io/milvus-model
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: code
fetch_tool: Grok-mcp web-fetch
priority: medium
word_count: 650
---

# milvus-io/milvus-model

## 元信息
- **来源**：https://github.com/milvus-io/milvus-model
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
`milvus-model` 是 PyMilvus 的可选依赖（`pymilvus[model]`），用于集成常见 embedding 与 reranker 模型（OpenAI、Voyage AI、Cohere、SentenceTransformers、Hugging Face TEI 等），提供统一调用方式以便在 Milvus 场景下进行向量化与重排。

---

## 正文内容

# Milvus Model Lib

The `milvus-model` library provides the integration with common embedding and reranker models for Milvus, a high performance open-source vector database built for AI applications. `milvus-model` lib is included as a dependency in `pymilvus`, the Python SDK of Milvus.

`milvus-model` supports embedding and reranker models from service providers like OpenAI, Voyage AI, Cohere, and open-source models through SentenceTransformers or Hugging Face Text Embeddings Inference (TEI).

`milvus-model` supports Python 3.8 and above.

## Installation

If you use `pymilvus`, you can install `milvus-model` through its alias `pymilvus[model]`:

```bash
pip install pymilvus[model]
# or pip install "pymilvus[model]" for zsh.
```

You can also install it directly:

```bash
pip install pymilvus.model
```

To upgrade milvus-model to the latest version, use:

```bash
pip install pymilvus.model --upgrade
```

If milvus-model was initially installed as part of the PyMilvus optional components, you should also upgrade PyMilvus to ensure compatibility:

```bash
pip install pymilvus[model] --upgrade
```

If you need to install a specific version:

```bash
pip install pymilvus.model==0.3.0
```

See more: https://milvus.io/docs/embeddings.md

---

## 关键信息提取

### 技术要点
- 通过 `pymilvus[model]` 安装可获取 embedding/reranker 集成能力。
- 支持服务商与开源模型（SentenceTransformers/TEI）。

### 代码示例
- 安装命令见上。

### 相关链接
- https://github.com/milvus-io/milvus-model
- https://milvus.io/docs/embeddings.md

---

## 抓取质量评估
- **完整性**：部分（按说明应抓 README + 关键文档；当前以 README 为主）
- **可用性**：中
- **时效性**：较新
