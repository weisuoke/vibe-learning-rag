# 核心概念：IVF_PQ索引

## 概述

IVF_PQ（Inverted File with Product Quantization）是IVF系列中压缩比最高的索引，通过乘积量化实现8-32倍内存节省。本文档详细介绍IVF_PQ的原理、配置和最佳实践。

---

## 什么是IVF_PQ索引？

**IVF_PQ = IVF_FLAT + 乘积量化 = 8-32x内存节省 + 5-10%准确率损失**

IVF_PQ通过将向量分割成子向量并分别量化，实现极致的内存压缩。

### 核心特征

1. **乘积量化**：向量分割 + 子向量量化
2. **极致压缩**：8-32倍内存节省
3. **快速搜索**：非对称距离计算
4. **准确率损失**：5-10%准确率损失

---

## 工作原理

### 乘积量化

```
原始向量（128维）：
[v1, v2, v3, ..., v128]

分割成m个子向量（m=16）：
子向量1: [v1, v2, ..., v8]    (8维)
子向量2: [v9, v10, ..., v16]  (8维)
...
子向量16: [v121, ..., v128]   (8维)

量化每个子向量：
子向量1 → 码字ID: 42  (1字节)
子向量2 → 码字ID: 137 (1字节)
...
子向量16 → 码字ID: 89 (1字节)

最终表示：[42, 137, ..., 89] (16字节)
压缩比：512字节 → 16字节 = 32x
```

### 码本训练

```
训练阶段：
1. 将所有向量分割成子向量
2. 对每组子向量进行k-means聚类
3. 每个聚类中心称为"码字"
4. 每个子向量用最近的码字ID表示

示例（m=16, nbits=8）：
- 16个子向量组
- 每组256个码字（2^8）
- 总共16×256=4096个码字
```

### 非对称距离计算

```
查询时：
1. 查询向量保持FP32（不量化）
2. 预计算查询子向量到所有码字的距离
3. 使用查找表快速计算距离

优势：
- 查询向量不量化，保持精度
- 数据库向量量化，节省内存
- 查找表加速距离计算
```

---

## 配置方法

### 创建IVF_PQ索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="IVF_PQ demo")

# 创建Collection
collection = Collection("ivf_pq_demo", schema)

# 创建IVF_PQ索引
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 1024,  # 聚类数
        "m": 16,        # 子量化器数量
        "nbits": 8      # 每个子量化器的位数
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)
```

### 参数说明

| 参数 | 说明 | 范围 | 推荐值 | 约束 |
|------|------|------|--------|------|
| `nlist` | 聚类数 | 1-65536 | sqrt(n) to 4*sqrt(n) | - |
| `m` | 子量化器数量 | 1-64 | 8-32 | dimension % m == 0 |
| `nbits` | 每个子量化器位数 | 1-16 | 8 | 2^nbits个码字 |
| `nprobe` | 搜索聚类数（查询） | 1-nlist | 8-128 | - |

**关键约束：**
```python
# dimension必须能被m整除
assert dimension % m == 0

# 示例
128维向量：m可以是8, 16, 32, 64
256维向量：m可以是8, 16, 32, 64, 128
```

---

## 使用步骤

### 完整示例

```python
import numpy as np
from pymilvus import Collection, connections, utility

# 1. 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 创建Collection
if not utility.has_collection("ivf_pq_demo"):
    from pymilvus import FieldSchema, CollectionSchema, DataType

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields)
    collection = Collection("ivf_pq_demo", schema)
else:
    collection = Collection("ivf_pq_demo")

# 3. 创建IVF_PQ索引
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "IVF_PQ",
        "metric_type": "L2",
        "params": {
            "nlist": 1024,
            "m": 16,      # 128/16 = 8维子向量
            "nbits": 8    # 256个码字
        }
    }
)

# 4. 插入数据
vectors = np.random.rand(10000000, 128).astype(np.float32).tolist()
collection.insert([vectors])

# 5. 加载Collection
collection.load()

# 6. 查询
query_vectors = np.random.rand(1, 128).astype(np.float32).tolist()
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"nprobe": 16}},
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

### 内存节省

```
原始数据（IVF_FLAT）：
10M向量 × 128维 × 4字节 = 5.12 GB

IVF_PQ（m=16, nbits=8）：
10M向量 × 16字节 = 160 MB

内存节省：5.12 GB → 160 MB（97%节省）
压缩比：32x
```

### 查询速度

| 数据规模 | IVF_FLAT | IVF_PQ | 加速比 |
|---------|----------|--------|--------|
| 1M向量 | 10ms | 4ms | 2.5x |
| 10M向量 | 30ms | 12ms | 2.5x |
| 100M向量 | 100ms | 40ms | 2.5x |

**特点：** IVF_PQ比IVF_FLAT快2-3倍

### 准确率对比

```
IVF_FLAT:  96-98%准确率（基准）
IVF_SQ8:   94-97%准确率（-1-2%）
IVF_PQ:    86-93%准确率（-5-10%）

准确率损失：中等（5-10%）
```

### 参数影响

```
m参数影响：
m=8:  压缩16x, 准确率90-95%
m=16: 压缩32x, 准确率85-93%
m=32: 压缩64x, 准确率80-90%

nbits参数影响：
nbits=4:  64个码字, 准确率80-85%
nbits=8:  256个码字, 准确率85-93%
nbits=16: 65536个码字, 准确率90-95%
```

---

## 限制与约束

### 技术限制

1. **维度约束**：dimension % m == 0
2. **准确率损失**：5-10%准确率损失
3. **训练开销**：需要训练码本

### 使用限制

1. **数据规模**：推荐 > 10M向量
2. **准确率要求**：不适合高准确率场景（>95%）
3. **维度要求**：维度必须能被m整除

### 适用边界

```
✅ 推荐使用：
- 超大规模（> 10M向量）
- 内存极度受限
- 准确率要求85-93%
- 成本敏感场景

⚠️ 谨慎使用：
- 中等规模（1M-10M）
- 准确率要求>95%
- 维度不能被m整除

❌ 不推荐使用：
- 小规模（< 1M）
- 高准确率要求（>95%）
- 实时更新需求
```

---

## RAG应用场景

### 场景1：大规模推荐系统

```python
# 1亿商品，内存受限

collection.create_index("embeddings", {
    "index_type": "IVF_PQ",
    "metric_type": "IP",  # 内积，适合推荐
    "params": {
        "nlist": 4096,
        "m": 16,
        "nbits": 8
    }
})

# 适用：
# - 电商推荐
# - 内容推荐
# - 广告推荐
```

### 场景2：成本优化RAG

```python
# 5000万文档，极致成本优化

collection.create_index("embeddings", {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {
        "nlist": 8192,
        "m": 32,      # 更高压缩
        "nbits": 8
    }
})

# 成本节省：
# IVF_FLAT: 25.6 GB → $500/月
# IVF_PQ: 800 MB → $20/月
# 节省：96%
```

### 场景3：多租户系统

```python
# 1000租户，每个租户10万文档

# 传统方案（IVF_FLAT）：
# 1000 × 100K × 512字节 = 51.2 GB

# IVF_PQ方案：
# 1000 × 100K × 16字节 = 1.6 GB

# 成本节省：97%
```

---

## 性能优化技巧

### 技巧1：选择合适的m

```python
# 低压缩（高准确率）
m = 8   # 16x压缩, 90-95%准确率

# 平衡压缩
m = 16  # 32x压缩, 85-93%准确率

# 高压缩（低准确率）
m = 32  # 64x压缩, 80-90%准确率

# 选择原则：
# - 准确率要求>90%：m=8
# - 准确率要求85-90%：m=16
# - 准确率要求<85%：m=32
```

### 技巧2：调整nbits

```python
# 快速但低准确率
nbits = 4   # 64个码字

# 平衡（推荐）
nbits = 8   # 256个码字

# 高准确率但慢
nbits = 16  # 65536个码字
```

### 技巧3：监控准确率

```python
def monitor_pq_accuracy():
    # 1. 使用IVF_FLAT作为基准
    flat_results = collection_flat.search(query, "embeddings", {}, limit=10)

    # 2. 使用IVF_PQ查询
    pq_results = collection_pq.search(query, "embeddings", {"nprobe": 16}, limit=10)

    # 3. 计算召回率
    recall = calculate_recall(flat_results, pq_results)

    print(f"IVF_PQ Recall: {recall}%")

    # 4. 如果召回率<85%，调整参数
    if recall < 85:
        print("Warning: Low recall, consider:")
        print("  - Decrease m (less compression)")
        print("  - Increase nbits (more码字)")
        print("  - Increase nprobe (more clusters)")
```

### 技巧4：批量查询

```python
# 批量查询提升吞吐量

# 差：单次查询
for query in queries:
    results = collection.search([query], "embeddings", {"nprobe": 16}, limit=10)

# 好：批量查询
results = collection.search(queries, "embeddings", {"nprobe": 16}, limit=10)

# 吞吐量提升：3-4倍
```

---

## 常见问题

### Q1：IVF_PQ为什么准确率损失较大？

**A：** 乘积量化损失信息较多。

**详细解释：**
```
IVF_SQ8：
- 整个向量一起量化
- 保留向量整体结构
- 准确率损失：1-2%

IVF_PQ：
- 向量分割成子向量
- 每个子向量独立量化
- 丢失子向量间的关系
- 准确率损失：5-10%
```

### Q2：如何选择m和nbits？

**A：** 根据准确率和内存需求权衡。

**选择策略：**
```
准确率优先：
- m=8, nbits=8
- 压缩16x, 准确率90-95%

平衡：
- m=16, nbits=8
- 压缩32x, 准确率85-93%

内存优先：
- m=32, nbits=4
- 压缩128x, 准确率75-85%
```

### Q3：IVF_PQ适合RAG系统吗？

**A：** 适合大规模、成本敏感的RAG系统。

**适用场景：**
```
✅ 适合：
- 超大规模（> 10M文档）
- 成本敏感
- 准确率要求85-93%
- 推荐系统

❌ 不适合：
- 高准确率要求（>95%）
- 小规模（< 1M文档）
- 实时更新频繁
```

### Q4：何时应该从IVF_PQ切换到其他索引？

**A：** 当准确率不足或数据规模较小时。

**决策因素：**
```
准确率不足：
- 当前准确率 < 85%
- 业务要求 > 90%
→ 切换到IVF_SQ8或IVF_FLAT

数据规模较小：
- 数据量 < 10M
- 内存充足
→ 切换到IVF_FLAT或HNSW

成本充足：
- 内存预算增加
- 可以承受更高成本
→ 切换到HNSW或GPU_CAGRA
```

---

## 最佳实践

### 实践1：生产环境配置

```python
# 推荐配置

index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 4096,   # 大规模数据
        "m": 16,         # 平衡压缩
        "nbits": 8       # 平衡准确率
    }
}

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 32}  # 提升准确率
}
```

### 实践2：A/B测试

```python
# A/B测试不同m值

def ab_test_m_values():
    m_values = [8, 16, 32]

    for m in m_values:
        collection = Collection(f"test_pq_m{m}")
        collection.create_index("embeddings", {
            "index_type": "IVF_PQ",
            "metric_type": "L2",
            "params": {"nlist": 1024, "m": m, "nbits": 8}
        })

        # 测试性能
        memory = collection.get_stats()["memory_size"]
        latency = benchmark_latency(collection)
        recall = benchmark_recall(collection)

        print(f"m={m}:")
        print(f"  Memory: {memory} MB")
        print(f"  Latency: {latency} ms")
        print(f"  Recall: {recall}%")
```

### 实践3：渐进式部署

```python
# 从IVF_FLAT迁移到IVF_PQ

# 阶段1：小规模测试（100K向量）
test_collection = Collection("test_ivf_pq")
test_collection.create_index("embeddings", ivf_pq_params)
benchmark_performance(test_collection)

# 阶段2：验证准确率
if recall > 85:
    # 阶段3：灰度切换（10%流量）
    route_10_percent_traffic_to_pq()
    monitor_metrics()

    # 阶段4：全量迁移（100%流量）
    if metrics_ok:
        migrate_all_traffic_to_pq()
        decommission_flat()
```

---

## 对比其他索引

### vs IVF_FLAT

| 维度 | IVF_FLAT | IVF_PQ |
|------|----------|--------|
| 准确率 | 95-99% | 85-93% |
| 速度 | 中等 | 快（2-3x） |
| 内存 | 1x | 0.03x |
| 压缩比 | 1x | 32x |

### vs IVF_SQ8

| 维度 | IVF_SQ8 | IVF_PQ |
|------|---------|--------|
| 准确率 | 93-98% | 85-93% |
| 速度 | 快 | 更快 |
| 内存 | 0.25x | 0.03x |
| 压缩比 | 4x | 32x |

### vs RaBitQ

| 维度 | IVF_PQ | RaBitQ |
|------|--------|--------|
| 准确率 | 85-93% | 94-98% |
| 速度 | 快 | 快 |
| 内存 | 0.03x | 0.03x |
| 适用场景 | 百万级 | 十亿级 |

---

## 总结

### 核心要点

1. **IVF_PQ = IVF_FLAT + 乘积量化**：8-32x内存节省
2. **准确率损失**：5-10%准确率损失
3. **极致压缩**：最高压缩比的IVF索引
4. **成本优化**：96-97%成本节省

### 使用场景

✅ **推荐使用：**
- 超大规模（> 10M向量）
- 内存极度受限
- 成本敏感场景
- 准确率要求85-93%

❌ **不推荐使用：**
- 高准确率要求（>95%）
- 小规模（< 1M）
- 实时更新频繁
- 维度不能被m整除

### 关键参数

```python
# 推荐配置
nlist = sqrt(n) to 4*sqrt(n)  # 聚类数
m = 16                        # 子量化器数量
nbits = 8                     # 每个子量化器位数
nprobe = 16-32                # 搜索聚类数
```

### 下一步

- **高性能**：学习 [03_核心概念_5_HNSW索引.md](./03_核心概念_5_HNSW索引.md)
- **成本优化**：学习 [03_核心概念_8_RaBitQ量化.md](./03_核心概念_8_RaBitQ量化.md)
- **实战应用**：实践 [07_实战代码_场景2_IVF系列索引对比.md](./07_实战代码_场景2_IVF系列索引对比.md)

---

**记住：IVF_PQ是极致压缩的选择。32x压缩 + 5-10%准确率损失 = 超大规模向量检索的成本优化方案。**
