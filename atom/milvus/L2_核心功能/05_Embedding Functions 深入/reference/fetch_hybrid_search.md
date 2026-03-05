---
type: fetched_content
source: https://github.com/milvus-io/pymilvus/issues/2660
title: [Bug]: Hybrid Search implementation always throws error · Issue #2660 · milvus-io/pymilvus
fetched_at: 2026-02-24T00:00:00Z
status: partial
author:
published_date:
knowledge_point: Milvus 2.6 Embedding Functions 深入
content_type: discussion
fetch_tool: Grok-mcp web-fetch
priority: high
word_count: 2800
---

# [Bug]: Hybrid Search implementation always throws error

## 元信息
- **来源**：https://github.com/milvus-io/pymilvus/issues/2660
- **作者**：
- **发布日期**：
- **抓取时间**：2026-02-24T00:00:00Z

## 内容摘要
该 Issue 展示了一个基于 PyMilvus 的 Hybrid Search 实现（dense: SentenceTransformer，sparse: SPLADE），并报告插入阶段报错：稀疏向量形状不符合期望（期望 1 行，实际形状 (30522,)）。正文包含较完整的 `HybridSearchManager` 代码：建表建索引（dense IVF_FLAT、sparse SPARSE_INVERTED_INDEX）、插入、hybrid_search（WeightedRanker / RRFRanker）、以及 sparse 向量转换为 Milvus 所需 dict 格式的逻辑。

---

## 正文内容

（Issue 正文包含长代码块与 traceback；为避免重复粘贴超长内容，此处保留关键片段与错误信息。若你希望我“逐字完整”落盘整个代码块，我可以再追加一次抓取并生成完整版。）

关键错误：

```
ValueError: Expected sparse vector to have 1 row, got shape: (30522,)
```

关键实现点：

- collection schema：`dense` (FLOAT_VECTOR, dim=dense_dim) + `sparse` (SPARSE_FLOAT_VECTOR)
- index：
  - dense: `IVF_FLAT`, metric `IP`, params `{"nlist": 128}`
  - sparse: `SPARSE_INVERTED_INDEX`, metric `IP`, params `{"inverted_index_algo": "DAAT_MAXSCORE"}`
- hybrid_search：构造 `AnnSearchRequest`（dense + sparse），ranker 选 `WeightedRanker(dense_weight, sparse_weight)` 或 `RRFRanker(rrf_k)`
- sparse vector conversion：将 CSR/ndarray 转换为 `{index: value}` 的 dict

---

## 关键信息提取

### 技术要点
- sparse embedding 输出的数据结构/shape 需要与转换逻辑匹配；该 Issue 暗示 SPLADE 输出可能为 1D 向量，需要 reshape/包装成 (1, dim) 或适配 dict 转换。
- Hybrid Search 的常见工程模式：双字段（dense + sparse）+ 双请求（AnnSearchRequest）+ ranker（weighted/RRF）。

### 代码示例
- 见 Issue 正文：`HybridSearchManager`（包含 setup_collection / insert_documents / hybrid_search 等）。

### 相关链接
- https://github.com/milvus-io/pymilvus/issues/2660
- https://milvus.io/docs/multi-vector-search.md

---

## 抓取质量评估
- **完整性**：部分（页面内容过长，当前落盘为关键片段；可按需补全全量代码块）
- **可用性**：中
- **时效性**：较新
