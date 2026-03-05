# 核心概念：FLAT索引

## 概述

FLAT索引是Milvus中最简单的向量索引类型，采用暴力搜索（brute-force）算法，保证100%召回率。本文档详细介绍FLAT索引的原理、配置、使用场景和最佳实践。

---

## 什么是FLAT索引？

**FLAT索引 = 无索引的精确搜索**

FLAT索引实际上不是一个真正的"索引"，而是将所有向量按原样存储，查询时遍历所有向量计算距离。

### 核心特征

1. **100%召回率**：保证找到真实最近邻
2. **暴力搜索**：与数据集中每个向量比较
3. **无预处理**：无需构建索引，插入即可用
4. **O(n)复杂度**：查询时间随数据量线性增长

---

## 工作原理

### 算法流程

```
1. 存储阶段：
   - 向量按原样存储（FP32格式）
   - 无任何预处理或转换

2. 查询阶段：
   - 遍历所有向量
   - 计算查询向量与每个向量的距离
   - 排序并返回top-k结果

3. 时间复杂度：
   - 插入：O(1)
   - 查询：O(n * d)
     - n：向量数量
     - d：向量维度
```

### 可视化

```
数据集：[v1, v2, v3, ..., vn]
查询向量：q

FLAT搜索过程：
q → 计算distance(q, v1)
q → 计算distance(q, v2)
q → 计算distance(q, v3)
...
q → 计算distance(q, vn)

排序所有距离 → 返回top-k
```

---

## 配置方法

### 创建FLAT索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="FLAT index demo")

# 创建Collection
collection = Collection("flat_demo", schema)

# 创建FLAT索引
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",  # 或 "IP", "COSINE"
    "params": {}  # 无需额外参数
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)
```

### 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `index_type` | 索引类型 | "FLAT" | - |
| `metric_type` | 距离度量 | "L2", "IP", "COSINE" | "L2" |
| `params` | 索引参数 | {} | {} |

**关键点：**
- FLAT索引**无需任何构建参数**
- `params`字段为空字典
- 支持所有距离度量类型

---

## 使用步骤

### 完整示例

```python
import numpy as np
from pymilvus import Collection, connections, utility

# 1. 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 创建Collection（如果不存在）
if not utility.has_collection("flat_demo"):
    from pymilvus import FieldSchema, CollectionSchema, DataType

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields)
    collection = Collection("flat_demo", schema)
else:
    collection = Collection("flat_demo")

# 3. 创建FLAT索引
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {}
    }
)

# 4. 插入数据
vectors = np.random.rand(10000, 128).astype(np.float32).tolist()
collection.insert([vectors])

# 5. 加载Collection
collection.load()

# 6. 查询
query_vectors = np.random.rand(1, 128).astype(np.float32).tolist()
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2"},
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

| 数据规模 | 查询延迟 | 说明 |
|---------|---------|------|
| 1K向量 | ~0.5ms | 可接受 |
| 10K向量 | ~5ms | 可接受 |
| 100K向量 | ~50ms | 边界 |
| 1M向量 | ~500ms | 不推荐 |
| 10M向量 | ~5s | 不可用 |

### 内存占用

```
内存 = n_vectors × dimension × 4 bytes

示例：
- 10K × 128维 = 5MB
- 100K × 128维 = 50MB
- 1M × 128维 = 512MB
- 10M × 128维 = 5GB
```

### 吞吐量

```
QPS = 1000ms / 查询延迟

示例（单线程）：
- 10K向量：QPS = 1000/5 = 200
- 100K向量：QPS = 1000/50 = 20
- 1M向量：QPS = 1000/500 = 2
```

---

## 限制与约束

### 性能限制

1. **线性增长**：查询时间随数据量线性增长
2. **不可扩展**：无法通过索引优化提升性能
3. **高延迟**：大数据集时延迟不可接受

### 使用限制

1. **数据规模**：推荐 < 1M向量
2. **实时性**：不适合高QPS场景
3. **生产环境**：不推荐用于生产系统

### 适用边界

```
✅ 推荐使用：
- 数据量 < 100K
- 开发测试阶段
- 准确率验证
- 性能基准测试

⚠️ 谨慎使用：
- 数据量 100K-1M
- 低QPS场景（< 10）

❌ 不推荐使用：
- 数据量 > 1M
- 高QPS场景（> 100）
- 生产环境
```

---

## RAG应用场景

### 场景1：原型验证

```python
# 快速验证RAG系统可行性
# 无需调参，直接使用

collection.create_index("embeddings", {
    "index_type": "FLAT",
    "metric_type": "L2"
})

# 适用：
# - 文档数量 < 10K
# - 快速原型开发
# - 概念验证（POC）
```

### 场景2：准确率基准

```python
# 建立100%准确率基准
# 用于验证其他索引的召回率

flat_results = collection_flat.search(query, "embeddings", {}, limit=10)
hnsw_results = collection_hnsw.search(query, "embeddings", {}, limit=10)

# 计算HNSW召回率
recall = calculate_recall(flat_results, hnsw_results)
print(f"HNSW Recall: {recall}%")
```

### 场景3：小规模知识库

```python
# 小型企业知识库
# 文档数量 < 10K

# 示例：公司内部文档检索
# - 100个PDF文档
# - 每个文档分10个chunk
# - 总计1000个向量
# - FLAT索引完全够用
```

---

## 性能优化技巧

### 技巧1：批量查询

```python
# 单次查询（慢）
for query in queries:
    results = collection.search([query], "embeddings", {}, limit=10)

# 批量查询（快）
results = collection.search(queries, "embeddings", {}, limit=10)

# 性能提升：2-3倍
```

### 技巧2：降维

```python
# 使用PCA降维
from sklearn.decomposition import PCA

# 原始：1536维（OpenAI ada-002）
# 降维：128维
pca = PCA(n_components=128)
reduced_vectors = pca.fit_transform(original_vectors)

# 性能提升：12倍（1536/128）
```

### 技巧3：提前过滤

```python
# 先用标量过滤减少搜索范围
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2"},
    limit=10,
    expr="category == 'tech'"  # 提前过滤
)

# 如果过滤掉90%数据，性能提升10倍
```

### 技巧4：考虑升级

```python
# 当数据量增长时，及时切换索引

if num_vectors < 100K:
    index_type = "FLAT"
elif num_vectors < 10M:
    index_type = "IVF_FLAT"
else:
    index_type = "HNSW"
```

---

## 常见问题

### Q1：FLAT索引为什么这么慢？

**A：** FLAT是暴力搜索，时间复杂度O(n)。对于1M向量，需要计算1M次距离。

**解决方案：**
- 数据量 < 100K：可接受
- 数据量 > 100K：切换到IVF_FLAT或HNSW

### Q2：FLAT索引能否加速？

**A：** 无法通过索引优化加速，但可以：
1. 使用批量查询
2. 降低向量维度
3. 提前过滤数据
4. 使用多线程

### Q3：FLAT索引支持更新吗？

**A：** 支持。FLAT索引支持实时插入、删除、更新，无需重建索引。

```python
# 插入
collection.insert([new_vectors])

# 删除
collection.delete(expr="id in [1, 2, 3]")

# 无需重建索引
```

### Q4：何时应该从FLAT切换到其他索引？

**A：** 当出现以下情况时：
1. 查询延迟 > 100ms
2. 数据量 > 100K
3. QPS > 10
4. 准备上生产环境

---

## 最佳实践

### 实践1：用于开发阶段

```python
# 开发阶段使用FLAT
# 简单、无需调参

if ENV == "development":
    index_type = "FLAT"
else:
    index_type = "HNSW"
```

### 实践2：建立性能基准

```python
# 使用FLAT建立性能基准

# 1. FLAT索引（基准）
flat_latency = benchmark_flat()

# 2. HNSW索引
hnsw_latency = benchmark_hnsw()

# 3. 计算提升
speedup = flat_latency / hnsw_latency
print(f"HNSW is {speedup}x faster")
```

### 实践3：验证准确率

```python
# 使用FLAT验证其他索引的准确率

def calculate_recall(flat_results, approx_results):
    flat_ids = set([hit.id for hit in flat_results[0]])
    approx_ids = set([hit.id for hit in approx_results[0]])

    intersection = flat_ids & approx_ids
    recall = len(intersection) / len(flat_ids)

    return recall * 100

# FLAT作为ground truth
flat_results = collection_flat.search(query, "embeddings", {}, limit=10)
hnsw_results = collection_hnsw.search(query, "embeddings", {}, limit=10)

recall = calculate_recall(flat_results, hnsw_results)
print(f"HNSW Recall: {recall}%")
```

### 实践4：及时升级

```python
# 监控数据量，及时升级索引

def recommend_index(num_vectors):
    if num_vectors < 100_000:
        return "FLAT"
    elif num_vectors < 10_000_000:
        return "IVF_FLAT"
    else:
        return "HNSW"

# 定期检查
current_count = collection.num_entities
recommended = recommend_index(current_count)

if current_index != recommended:
    print(f"建议升级到 {recommended} 索引")
```

---

## 对比其他索引

### vs IVF_FLAT

| 维度 | FLAT | IVF_FLAT |
|------|------|----------|
| 准确率 | 100% | 95-99% |
| 速度 | 慢（O(n)） | 快（O(nprobe*n/nlist)） |
| 内存 | 1x | 1x |
| 适用规模 | < 1M | 1M-10M |
| 参数调优 | 无需 | 需要（nlist, nprobe） |

**选择建议：**
- 数据量 < 100K：FLAT
- 数据量 > 100K：IVF_FLAT

### vs HNSW

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

---

## 总结

### 核心要点

1. **FLAT = 暴力搜索**：遍历所有向量，100%准确
2. **O(n)复杂度**：查询时间随数据量线性增长
3. **无需参数**：最简单的索引类型
4. **适合小数据集**：< 100K向量

### 使用场景

✅ **推荐使用：**
- 开发测试阶段
- 准确率验证
- 小规模数据集（< 100K）
- 性能基准测试

❌ **不推荐使用：**
- 生产环境
- 大规模数据集（> 1M）
- 高QPS场景
- 实时搜索系统

### 下一步

- **升级索引**：学习 [03_核心概念_2_IVF_FLAT索引.md](./03_核心概念_2_IVF_FLAT索引.md)
- **生产标准**：学习 [03_核心概念_5_HNSW索引.md](./03_核心概念_5_HNSW索引.md)
- **实战应用**：实践 [07_实战代码_场景1_FLAT索引实战.md](./07_实战代码_场景1_FLAT索引实战.md)

---

**记住：FLAT索引是最简单但也是最慢的索引。它的价值在于提供100%准确率的基准，而不是用于生产环境。**
