# 核心概念：SCANN索引

## 概述

SCANN（Scalable Nearest Neighbors）是Google开发的向量索引算法，通过分数感知量化实现速度和准确率的平衡。本文档介绍SCANN的原理、配置和最佳实践。

---

## 什么是SCANN索引？

**SCANN = IVF + 分数感知量化 + 4-bit FastScan = 平衡性能**

SCANN是Google Research开发的向量检索算法，在IVF_PQ的基础上通过分数感知量化提升准确率。

### 核心特征

1. **分数感知量化**：改进的量化算法，提升准确率
2. **4-bit FastScan**：快速距离计算
3. **平衡性能**：速度和准确率的良好权衡
4. **Google技术**：基于Google Research

---

## 工作原理

### 分数感知量化

```
传统PQ量化：
- 最小化重构误差
- 不考虑查询分布

SCANN量化：
- 最小化检索误差
- 考虑查询分布
- 分数感知：优化top-k准确率
```

### 4-bit FastScan

```
传统PQ：8-bit量化
SCANN：4-bit量化

优势：
- 更快的距离计算
- 更少的内存占用
- 通过分数感知补偿精度损失
```

---

## 配置方法

### 创建SCANN索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="SCANN demo")

# 创建Collection
collection = Collection("scann_demo", schema)

# 创建SCANN索引
index_params = {
    "index_type": "SCANN",
    "metric_type": "L2",
    "params": {
        "nlist": 1024,
        "with_raw_data": True  # 存储原始向量用于精排
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `nlist` | 聚类数 | 1024-4096 |
| `with_raw_data` | 存储原始向量 | True（提升准确率） |
| `nprobe` | 搜索聚类数（查询） | 16-32 |

---

## 性能特征

### 查询速度

```
SCANN vs IVF_PQ：
- QPS：显著更高
- 延迟：更低
- 准确率：更高
```

### 准确率对比

```
IVF_PQ:   85-93%准确率
SCANN:    90-98%准确率

提升：5-10%
```

### 内存占用

```
与IVF_PQ相似：
- 4-bit量化
- 可选存储原始向量（with_raw_data=True）
```

---

## 使用场景

### 推荐使用

```
✅ 中大规模（10M-100M向量）
✅ 平衡性能需求
✅ 准确率要求90-98%
✅ 内存适中
```

### 不推荐使用

```
❌ 小规模（< 1M）
❌ 100%准确率要求
❌ 内存极度受限
```

---

## 对比其他索引

### vs IVF_PQ

| 维度 | IVF_PQ | SCANN |
|------|--------|-------|
| 准确率 | 85-93% | 90-98% |
| 速度 | 快 | 更快 |
| 内存 | 0.03x | 0.03-0.5x |
| 技术 | 乘积量化 | 分数感知量化 |

### vs HNSW

| 维度 | HNSW | SCANN |
|------|------|-------|
| 准确率 | 95-99% | 90-98% |
| 速度 | 很快 | 快 |
| 内存 | 1.5x | 0.03-0.5x |
| 适用场景 | 高性能 | 平衡性能 |

---

## 总结

### 核心要点

1. **SCANN = IVF + 分数感知量化**：平衡性能
2. **Google技术**：基于Google Research
3. **准确率提升**：比IVF_PQ高5-10%
4. **适合中大规模**：10M-100M向量

### 使用场景

✅ **推荐使用：**
- 中大规模（10M-100M向量）
- 平衡性能需求
- 准确率要求90-98%

❌ **不推荐使用：**
- 小规模（< 1M）
- 100%准确率要求
- 内存极度受限

### 关键参数

```python
# 推荐配置
nlist = 1024-4096
with_raw_data = True
nprobe = 16-32
```

### 下一步

- **高性能**：学习 [03_核心概念_5_HNSW索引.md](./03_核心概念_5_HNSW索引.md)
- **索引选型**：学习 [03_核心概念_9_索引选型决策树.md](./03_核心概念_9_索引选型决策树.md)

---

**记住：SCANN是平衡性能的选择。分数感知量化 + 4-bit FastScan = Google的向量检索技术。**
