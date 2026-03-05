# 核心概念：HNSW索引

## 概述

HNSW（Hierarchical Navigable Small World）是Milvus中最流行的生产环境索引，采用多层图结构实现高效的近似最近邻搜索。本文档详细介绍HNSW索引的原理、配置和最佳实践。

---

## 什么是HNSW索引？

**HNSW索引 = 多层图 + 贪婪搜索 = O(log n)复杂度**

HNSW是基于图的索引算法，通过构建多层导航图实现快速的向量检索。

### 核心特征

1. **多层图结构**：类似跳表，构建多层图加速搜索
2. **高准确率**：95-99%召回率，接近精确搜索
3. **快速查询**：O(log n)时间复杂度
4. **生产标准**：业界最常用的向量索引

---

## 工作原理

### 多层图结构

```
Layer 2 (top):    [Entry] -----> [Node A]
                     |              |
Layer 1:          [Node B] ---> [Node C] ---> [Node D]
                     |              |              |
Layer 0 (base):   [All vectors with dense connections]
```

**关键思想：**
- 顶层稀疏，用于快速跳跃
- 底层密集，用于精确搜索
- 每层通过贪婪搜索找到局部最优

### 搜索过程

```
1. 从顶层入口点开始
2. 在当前层贪婪搜索最近邻
3. 找到局部最优后，下降到下一层
4. 重复2-3，直到底层
5. 返回底层的top-k结果
```

**时间复杂度：** O(log n)

### 可视化示例

```
查询向量: q

Layer 2: Entry → A (距离q最近)
         ↓
Layer 1: A → C → D (贪婪搜索)
         ↓
Layer 0: D → [邻居节点] → 返回top-k
```

---

## 配置方法

### 创建HNSW索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="HNSW index demo")

# 创建Collection
collection = Collection("hnsw_demo", schema)

# 创建HNSW索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",  # 或 "IP", "COSINE"
    "params": {
        "M": 16,              # 每个节点最多M个连接
        "efConstruction": 200  # 构建时的搜索质量
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)
```

### 参数说明

| 参数 | 说明 | 范围 | 推荐值 | 影响 |
|------|------|------|--------|------|
| `M` | 每个节点最多连接数 | 4-64 | 16-32 | M越大：准确率↑，内存↑，构建慢 |
| `efConstruction` | 构建时搜索质量 | 8-512 | 200-400 | 越大：索引质量↑，构建慢 |
| `ef` (查询参数) | 查询时搜索质量 | top_k-32768 | 100-200 | 越大：准确率↑，查询慢 |

**关键点：**
- `M` 和 `efConstruction` 是构建参数（一次性）
- `ef` 是查询参数（每次查询可调整）
- `ef` 必须 >= `top_k`（返回结果数）

---

## 使用步骤

### 完整示例

```python
import numpy as np
from pymilvus import Collection, connections, utility

# 1. 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 创建Collection（如果不存在）
if not utility.has_collection("hnsw_demo"):
    from pymilvus import FieldSchema, CollectionSchema, DataType

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields)
    collection = Collection("hnsw_demo", schema)
else:
    collection = Collection("hnsw_demo")

# 3. 创建HNSW索引
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200}
    }
)

# 4. 插入数据
vectors = np.random.rand(1000000, 128).astype(np.float32).tolist()
collection.insert([vectors])

# 5. 加载Collection
collection.load()

# 6. 查询（调整ef参数）
query_vectors = np.random.rand(1, 128).astype(np.float32).tolist()
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10
)

# 7. 输出结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance}")

# 8. 清理
collection.release()
```

---

## 性能特征

### 查询延迟

| 数据规模 | 查询延迟 (ef=100) | 说明 |
|---------|------------------|------|
| 1M向量 | ~1-5ms | 优秀 |
| 10M向量 | ~2-10ms | 很好 |
| 100M向量 | ~5-20ms | 可接受 |

**特点：** 延迟随数据规模呈对数增长（sub-linear）

### 内存占用

```
内存 = n × (d × 4 + M × 2 × 8) bytes

示例（1M向量，128维，M=16）：
= 1M × (128 × 4 + 16 × 2 × 8)
= 1M × (512 + 256)
= 768 MB

对比FLAT：512 MB
内存开销：1.5x
```

### 准确率vs速度权衡

```
ef=50:  92%准确率, 最快
ef=100: 96%准确率, 平衡 ✅
ef=200: 98%准确率, 较慢
ef=400: 99%准确率, 最慢
```

**关键认知：** 准确率提升呈对数曲线，边际递减

---

## 参数调优指南

### M参数选择

```python
# 低内存配置
M = 8  # 内存: 1x, 准确率: 95%

# 平衡配置（推荐）
M = 16  # 内存: 2x, 准确率: 97%

# 高质量配置
M = 32  # 内存: 4x, 准确率: 98%

# 极致配置
M = 64  # 内存: 8x, 准确率: 98.5%（边际递减）
```

**建议：** M=16-32是最佳平衡点

### efConstruction参数选择

```python
# 快速构建（开发测试）
efConstruction = 100

# 平衡配置（推荐）
efConstruction = 200

# 高质量（生产环境）
efConstruction = 400

# 极致质量（关键应用）
efConstruction = 500
```

**建议：** 生产环境使用200-400

### ef参数选择（查询时）

```python
# 快速查询
search_params = {"ef": 50}  # 90-95%准确率

# 平衡查询（推荐）
search_params = {"ef": 100}  # 95-97%准确率

# 高准确率
search_params = {"ef": 200}  # 97-99%准确率

# 极致准确率
search_params = {"ef": 400}  # 99%+准确率
```

**建议：** 根据业务需求动态调整ef

### 调优流程

```
1. 先用默认值（M=16, efConstruction=200, ef=100）
2. 如果准确率不够 → 增加ef
3. 如果查询太慢 → 减少ef
4. 如果内存不够 → 减少M或切换到IVF_SQ8
5. 如果准确率仍不够 → 增加M和efConstruction
```

---

## 限制与约束

### 性能限制

1. **内存开销**：比FLAT高1.5倍（图结构开销）
2. **构建时间**：比IVF慢（需要构建多层图）
3. **更新代价**：不支持高效的向量删除和更新

### 使用限制

1. **数据规模**：推荐10M-100M向量
2. **更新频率**：不适合频繁更新（需要重建索引）
3. **内存要求**：需要充足内存（1.5-2x数据大小）

### 适用边界

```
✅ 推荐使用：
- 生产环境RAG系统
- 高QPS场景（> 100）
- 高准确率要求（> 95%）
- 数据相对稳定

⚠️ 谨慎使用：
- 内存受限环境
- 频繁更新场景
- 超大规模（> 100M）

❌ 不推荐使用：
- 内存极度受限
- 实时更新需求
- 成本敏感场景
```

---

## RAG应用场景

### 场景1：生产环境RAG系统

```python
# 典型配置：10M文档，高QPS，高准确率

collection.create_index("embeddings", {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # RAG常用余弦相似度
    "params": {
        "M": 16,              # 平衡内存和准确率
        "efConstruction": 200  # 保证索引质量
    }
})

# 查询时动态调整ef
search_params = {"ef": 100}  # 96%准确率，5ms延迟

# 适用：
# - 文档问答系统
# - 知识库检索
# - 智能客服
```

### 场景2：实时搜索系统

```python
# 低延迟配置：优先速度

collection.create_index("embeddings", {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
})

# 查询时使用较小的ef
search_params = {"ef": 50}  # 92%准确率，2ms延迟

# 适用：
# - 实时推荐
# - 图像搜索
# - 语义搜索
```

### 场景3：高准确率场景

```python
# 高质量配置：优先准确率

collection.create_index("embeddings", {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 32,              # 更多连接
        "efConstruction": 400  # 更高质量
    }
})

# 查询时使用较大的ef
search_params = {"ef": 200}  # 98%准确率，10ms延迟

# 适用：
# - 医疗诊断
# - 法律检索
# - 科研文献
```

---

## 性能优化技巧

### 技巧1：动态调整ef

```python
# 根据业务场景动态调整

def search_with_adaptive_ef(query, accuracy_requirement):
    if accuracy_requirement == "high":
        ef = 200  # 98%准确率
    elif accuracy_requirement == "medium":
        ef = 100  # 96%准确率
    else:
        ef = 50   # 92%准确率

    return collection.search(
        data=query,
        anns_field="embeddings",
        param={"params": {"ef": ef}},
        limit=10
    )
```

### 技巧2：批量查询

```python
# 批量查询提升吞吐量

# 单次查询（慢）
for query in queries:
    results = collection.search([query], "embeddings", {"ef": 100}, limit=10)

# 批量查询（快2-3倍）
results = collection.search(queries, "embeddings", {"ef": 100}, limit=10)
```

### 技巧3：连接池

```python
# 高QPS场景使用连接池

from pymilvus import connections

# 创建连接池
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    pool_size=10  # 连接池大小
)

# 复用连接
collection = Collection("hnsw_demo")
```

### 技巧4：索引预加载

```python
# 启动时预加载索引到内存

collection.load()  # 预加载

# 避免首次查询延迟
# 首次查询：50ms（加载索引）
# 后续查询：5ms（已加载）
```

---

## 常见问题

### Q1：HNSW为什么比IVF快？

**A：** HNSW是O(log n)，IVF是O(nprobe * n/nlist)。

**详细解释：**
- HNSW通过多层图优化搜索路径
- IVF只是减少搜索范围（聚类）
- 大数据集时，HNSW优势明显

**性能对比：**
```
10M向量：
- HNSW: 5ms
- IVF_FLAT (nprobe=16): 15ms
```

### Q2：HNSW内存占用为什么高？

**A：** 需要存储图结构（节点连接信息）。

**内存组成：**
```
向量数据：n × d × 4 bytes
图结构：n × M × 2 × 8 bytes

示例（1M向量，128维，M=16）：
向量：512 MB
图结构：256 MB
总计：768 MB（1.5x）
```

### Q3：HNSW支持更新吗？

**A：** 支持插入，但删除和更新效率低。

**原因：**
- 插入：可以增量添加到图中
- 删除：需要重建图连接（代价高）
- 更新：等同于删除+插入

**建议：**
- 数据相对稳定：使用HNSW
- 频繁更新：使用IVF_FLAT或定期重建

### Q4：何时应该重建HNSW索引？

**A：** 当插入大量新数据后（> 10%）。

**原因：**
- 新数据可能破坏图结构
- 准确率和性能下降

**监控指标：**
```python
# 监控准确率
if current_recall < 95%:
    rebuild_index()

# 监控延迟
if p99_latency > 20ms:
    rebuild_index()
```

---

## 最佳实践

### 实践1：生产环境配置

```python
# 推荐配置

index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # RAG常用
    "params": {
        "M": 16,              # 平衡
        "efConstruction": 200  # 保证质量
    }
}

search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 100}  # 平衡
}
```

### 实践2：监控关键指标

```python
# 监控脚本

import time

def monitor_hnsw_performance():
    # 1. 查询延迟
    start = time.time()
    results = collection.search(query, "embeddings", search_params, limit=10)
    latency = (time.time() - start) * 1000

    # 2. 准确率（与FLAT对比）
    flat_results = collection_flat.search(query, "embeddings", {}, limit=10)
    recall = calculate_recall(results, flat_results)

    # 3. 内存使用
    memory_usage = collection.get_stats()["memory_size"]

    print(f"Latency: {latency}ms, Recall: {recall}%, Memory: {memory_usage}MB")
```

### 实践3：定期重建索引

```python
# 定期重建策略

def rebuild_index_if_needed(collection):
    # 检查插入比例
    total_entities = collection.num_entities
    new_entities = get_new_entities_count()

    if new_entities / total_entities > 0.1:  # 超过10%
        print("Rebuilding index...")
        collection.drop_index()
        collection.create_index("embeddings", index_params)
        collection.load()
```

### 实践4：A/B测试参数

```python
# A/B测试不同ef值

def ab_test_ef():
    ef_values = [50, 100, 200]

    for ef in ef_values:
        search_params = {"params": {"ef": ef}}

        # 测试延迟
        latency = benchmark_latency(search_params)

        # 测试准确率
        recall = benchmark_recall(search_params)

        print(f"ef={ef}: Latency={latency}ms, Recall={recall}%")
```

---

## 对比其他索引

### vs FLAT

| 维度 | FLAT | HNSW |
|------|------|------|
| 准确率 | 100% | 95-99% |
| 速度 | 慢（O(n)） | 很快（O(log n)） |
| 内存 | 1x | 1.5x |
| 适用规模 | < 1M | 10M-100M |
| 构建时间 | 无 | 较长 |

**选择建议：**
- 开发测试：FLAT
- 生产环境：HNSW

### vs IVF_FLAT

| 维度 | IVF_FLAT | HNSW |
|------|----------|------|
| 准确率 | 95-99% | 95-99% |
| 速度 | 中等 | 很快 |
| 内存 | 1x | 1.5x |
| 过滤性能 | 好 | 一般 |
| 更新支持 | 好 | 差 |

**选择建议：**
- 带过滤条件：IVF_FLAT
- 纯向量搜索：HNSW

### vs GPU_CAGRA

| 维度 | HNSW | GPU_CAGRA |
|------|------|-----------|
| 速度 | 快 | 极快（10x） |
| 硬件 | CPU | GPU |
| 成本 | 低 | 高 |
| 适用规模 | 10M-100M | 100M-1B |

**选择建议：**
- 无GPU：HNSW
- 有GPU + 大规模：GPU_CAGRA

---

## 总结

### 核心要点

1. **HNSW = 多层图 + 贪婪搜索**：O(log n)复杂度
2. **生产标准**：业界最常用的向量索引
3. **高准确率**：95-99%召回率
4. **参数调优**：M（构建）、efConstruction（构建）、ef（查询）

### 使用场景

✅ **推荐使用：**
- 生产环境RAG系统
- 高QPS场景（> 100）
- 高准确率要求（> 95%）
- 数据相对稳定（低更新频率）

❌ **不推荐使用：**
- 内存极度受限
- 频繁更新需求
- 超大规模（> 100M，考虑GPU_CAGRA）
- 成本敏感（考虑IVF_SQ8或RaBitQ）

### 关键参数

```python
# 推荐配置
M = 16                # 平衡内存和准确率
efConstruction = 200  # 保证索引质量
ef = 100              # 平衡速度和准确率
```

### 下一步

- **GPU加速**：学习 [03_核心概念_7_GPU_CAGRA索引.md](./03_核心概念_7_GPU_CAGRA索引.md)
- **成本优化**：学习 [03_核心概念_8_RaBitQ量化.md](./03_核心概念_8_RaBitQ量化.md)
- **实战应用**：实践 [07_实战代码_场景3_HNSW高性能检索.md](./07_实战代码_场景3_HNSW高性能检索.md)

---

**记住：HNSW是生产环境的标准选择。理解其原理和参数调优，才能构建高性能的向量检索系统。**
