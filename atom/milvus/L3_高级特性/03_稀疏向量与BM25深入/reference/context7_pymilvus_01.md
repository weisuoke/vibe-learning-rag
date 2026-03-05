---
type: context7_documentation
library: pymilvus
version: master (2026-01-22)
fetched_at: 2026-02-25
knowledge_point: 03_稀疏向量与BM25深入
context7_query: sparse vector BM25 index SPARSE_INVERTED_INDEX SPARSE_WAND
---

# Context7 文档：PyMilvus 稀疏向量与 BM25

## 文档来源
- 库名称：Milvus PyMilvus
- 版本：master (2026-01-22)
- 官方文档链接：https://context7.com/milvus-io/pymilvus
- Trust Score：9.8
- Benchmark Score：90.3
- Total Snippets：198

## 关键信息提取

### 1. BM25 全文搜索

**核心特性**：
- 自动分词并生成稀疏向量
- 使用 `Function` 和 `FunctionType.BM25` 自动生成稀疏向量
- 索引类型：`SPARSE_INVERTED_INDEX`
- 度量类型：`BM25`

**BM25 参数**：
- `bm25_k1`：默认 1.2（词频饱和参数）
- `bm25_b`：默认 0.75（文档长度归一化参数）

**代码示例**：
```python
# 创建 BM25 函数
bm25_function = Function(
    name="bm25_fn",
    input_field_names=["content"],
    output_field_names="sparse_vector",
    function_type=FunctionType.BM25
)
schema.add_function(bm25_function)

# 创建索引
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={"bm25_k1": 1.2, "bm25_b": 0.75}
)
```

---

### 2. 稀疏向量索引

**索引类型**：
- `SPARSE_INVERTED_INDEX`：倒排索引

**度量类型**：
- `IP`（Inner Product）：内积

**关键参数**：
- `drop_ratio_build`：构建时丢弃低值条目（例如 0.2）
- `drop_ratio_search`：搜索时丢弃低值条目（例如 0.2）

**数据格式**：
- 稀疏向量使用字典格式：`{dimension_index: value}`
- 例如：`{0: 0.5, 100: 0.8, 500: 0.3}`

**代码示例**：
```python
# 创建稀疏向量索引
index_params.add_index(
    field_name="sparse_embedding",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={"drop_ratio_build": 0.2}
)

# 搜索参数
search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
```

---

### 3. 混合检索与权重调整

**Ranker 类型**：

#### RRFRanker（Reciprocal Rank Fusion）
- 公式：`RRF(d) = Σ 1/(k + rank_i(d))`
- 参数：`k`（默认 60）
- 适用场景：不确定各向量字段的相对重要性

```python
ranker=RRFRanker(k=60)
```

#### WeightedRanker（加权融合）
- 公式：`Score = w1 * score1 + w2 * score2`
- 参数：权重列表（例如 `[0.7, 0.3]`）
- 适用场景：明确知道各向量字段的重要性

```python
ranker=WeightedRanker(0.7, 0.3)  # 70% text, 30% image
```

**混合检索流程**：
1. 创建多个 `AnnSearchRequest`（每个向量字段一个）
2. 使用 `hybrid_search` 方法
3. 指定 Ranker 策略
4. 返回融合后的结果

**代码示例**：
```python
# 创建搜索请求
req1 = AnnSearchRequest(
    data=[text_query],
    anns_field="text_embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=20
)
req2 = AnnSearchRequest(
    data=[image_query],
    anns_field="image_embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=20
)

# 混合检索
results = client.hybrid_search(
    collection_name="multimodal",
    reqs=[req1, req2],
    ranker=WeightedRanker(0.7, 0.3),
    limit=5,
    output_fields=["title"]
)
```

---

## 核心概念总结

### 稀疏向量特点
- **稀疏存储**：只存储非零元素
- **字典格式**：`{index: value}`
- **高维度**：通常 10000+ 维度
- **低密度**：通常 1-5% 非零元素

### BM25 算法
- **自动分词**：使用 Analyzer
- **自动生成稀疏向量**：通过 Function
- **参数可调**：`bm25_k1` 和 `bm25_b`
- **4x 快于 Elasticsearch**

### 混合检索策略
- **RRF**：适用于不确定权重的场景
- **WeightedRanker**：适用于明确权重的场景
- **多向量字段**：支持多模态检索

---

## 应用场景

### 1. 全文搜索
- 使用 BM25 进行关键词匹配
- 自动分词和稀疏向量生成
- 适用于文档检索、问答系统

### 2. 混合检索
- 向量检索（语义相似度）+ BM25（关键词匹配）
- 提升检索准确率
- 适用于 RAG 系统

### 3. 多模态检索
- 文本 + 图像混合检索
- 使用 WeightedRanker 调整权重
- 适用于电商、内容推荐

---

## 性能优化

### drop_ratio 参数
- **drop_ratio_build**：减少索引大小
- **drop_ratio_search**：加速搜索
- **权衡**：精度 vs 性能

### 索引选择
- **SPARSE_INVERTED_INDEX**：适用于大多数场景
- **度量类型**：BM25（全文搜索）或 IP（稀疏向量）

---

## 下一步调研方向

1. **BM25 参数调优**：如何选择 `bm25_k1` 和 `bm25_b`
2. **drop_ratio 调优**：如何平衡精度和性能
3. **混合检索权重**：如何确定最优权重
4. **Analyzer 配置**：不同语言的分词器配置
5. **性能对比**：BM25 vs Elasticsearch
