# 核心概念：GPU_CAGRA索引

## 概述

GPU_CAGRA（CUDA-Accelerated Graph-based ANN）是Milvus 2.6中引入的GPU加速向量索引，利用NVIDIA GPU的并行计算能力实现极致性能。本文档详细介绍GPU_CAGRA的原理、配置和2026年最新的混合模式特性。

---

## 什么是GPU_CAGRA索引？

**GPU_CAGRA = GPU并行计算 + 图索引 = 10x速度提升**

GPU_CAGRA是基于图的索引算法，通过NVIDIA CUDA加速实现超高性能的向量检索。

### 核心特征

1. **极快构建**：比CPU HNSW快12-15倍
2. **极快查询**：比CPU HNSW快10倍
3. **混合模式**：GPU构建 + CPU查询（2026特性）
4. **高准确率**：95-99%召回率，与HNSW相当

---

## 工作原理

### GPU并行计算

```
CPU HNSW:
Thread 1: 处理向量1
Thread 2: 处理向量2
...
Thread 16: 处理向量16

GPU CAGRA:
GPU Core 1-1000: 并行处理1000个向量
GPU Core 1001-2000: 并行处理下1000个向量
...
总计: 数千个核心同时工作
```

**关键优势：** GPU有数千个核心，可以并行处理大量向量

### 图索引结构

```
与HNSW类似的多层图结构：
- 层次化导航
- 贪婪搜索
- 小世界网络

区别：
- GPU优化的数据结构
- CUDA加速的距离计算
- 并行图遍历
```

### 混合模式（Milvus 2.6.1+）

```
构建阶段（GPU）:
1. 使用GPU快速构建高质量图
2. 12-15x faster than CPU HNSW
3. 一次性成本

查询阶段（CPU）:
1. 将索引适配到CPU
2. 使用CPU执行查询
3. 降低成本，提高可扩展性

优势：
✅ GPU构建速度
✅ CPU查询成本
✅ 最佳性价比
```

---

## 配置方法

### 创建GPU_CAGRA索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="GPU CAGRA demo")

# 创建Collection
collection = Collection("gpu_cagra_demo", schema)

# 创建GPU_CAGRA索引
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",  # 或 "IP", "COSINE"
    "params": {
        "intermediate_graph_degree": 128,  # 构建时图度数
        "graph_degree": 64,                # 最终图度数
        "build_algo": "IVF_PQ"             # 构建算法
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
| `intermediate_graph_degree` | 构建时图度数 | 64-512 | 128 | 越大：质量↑，构建慢 |
| `graph_degree` | 最终图度数 | 32-128 | 64 | 越大：准确率↑，内存↑ |
| `build_algo` | 构建算法 | IVF_PQ, NN_DESCENT | IVF_PQ | 影响构建速度和质量 |

### 混合模式配置（2026特性）

```python
# 启用混合模式：GPU构建 + CPU查询
collection.load(
    replica_number=1,
    _resource_groups=["gpu"],
    adapt_for_cpu=True  # 关键参数：启用CPU查询
)

# 查询时使用CPU
search_params = {
    "metric_type": "L2",
    "params": {
        "itopk_size": 128,      # 内部top-k大小
        "search_width": 4,      # 搜索宽度
        "min_iterations": 0,    # 最小迭代次数
        "max_iterations": 0     # 最大迭代次数（0=自动）
    }
}
```

---

## 使用步骤

### 完整示例（纯GPU模式）

```python
import numpy as np
from pymilvus import Collection, connections, utility

# 1. 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 创建Collection
if not utility.has_collection("gpu_cagra_demo"):
    from pymilvus import FieldSchema, CollectionSchema, DataType

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields)
    collection = Collection("gpu_cagra_demo", schema)
else:
    collection = Collection("gpu_cagra_demo")

# 3. 创建GPU_CAGRA索引
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "GPU_CAGRA",
        "metric_type": "L2",
        "params": {
            "intermediate_graph_degree": 128,
            "graph_degree": 64
        }
    }
)

# 4. 插入数据
vectors = np.random.rand(10000000, 128).astype(np.float32).tolist()
collection.insert([vectors])

# 5. 加载Collection（GPU模式）
collection.load(_resource_groups=["gpu"])

# 6. 查询
query_vectors = np.random.rand(100, 128).astype(np.float32).tolist()
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"itopk_size": 128}},
    limit=10
)

# 7. 输出结果
for i, hits in enumerate(results):
    print(f"Query {i}:")
    for hit in hits:
        print(f"  ID: {hit.id}, Distance: {hit.distance}")

# 8. 清理
collection.release()
```

### 混合模式示例（2026特性）

```python
# GPU构建 + CPU查询

# 1. 创建索引（GPU构建）
collection.create_index(
    field_name="embeddings",
    index_params={
        "index_type": "GPU_CAGRA",
        "metric_type": "L2",
        "params": {
            "intermediate_graph_degree": 128,
            "graph_degree": 64
        }
    }
)

# 2. 加载Collection（启用CPU查询）
collection.load(
    replica_number=1,
    _resource_groups=["gpu"],
    adapt_for_cpu=True  # 关键：启用混合模式
)

# 3. 查询（使用CPU）
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"itopk_size": 128}},
    limit=10
)

# 优势：
# - GPU构建：12-15x faster
# - CPU查询：降低成本，提高可扩展性
# - 最佳性价比
```

---

## 性能特征

### 构建速度

| 数据规模 | CPU HNSW | GPU CAGRA | 加速比 |
|---------|----------|-----------|--------|
| 1M向量 | 10分钟 | 40秒 | 15x |
| 10M向量 | 100分钟 | 7分钟 | 14x |
| 100M向量 | 1000分钟 | 70分钟 | 14x |

**特点：** 大规模数据集时，GPU优势更明显

### 查询速度

| 数据规模 | CPU HNSW | GPU CAGRA (纯GPU) | 加速比 |
|---------|----------|-------------------|--------|
| 1M向量 | 5ms | 0.5ms | 10x |
| 10M向量 | 10ms | 1ms | 10x |
| 100M向量 | 20ms | 2ms | 10x |

**注意：** 批量查询时GPU优势更明显（batch > 10）

### 混合模式性能

```
纯GPU模式：
- 构建：最快（12-15x）
- 查询：最快（10x）
- 成本：最高（需要GPU）

混合模式：
- 构建：最快（12-15x）
- 查询：略慢于纯GPU，但仍快于CPU HNSW
- 成本：中等（GPU构建，CPU查询）

CPU HNSW：
- 构建：慢
- 查询：慢
- 成本：最低（仅需CPU）
```

### 内存占用

```
GPU内存 = n × (d × 4 + graph_degree × 8) bytes

示例（10M向量，128维，graph_degree=64）：
= 10M × (128 × 4 + 64 × 8)
= 10M × (512 + 512)
= 10.24 GB

要求：
- NVIDIA GPU with >= 12GB memory
- CUDA 11.0+
```

---

## 硬件要求

### GPU要求

```
✅ 推荐配置：
- GPU: NVIDIA A100, V100, RTX 3090/4090
- Memory: >= 16GB
- CUDA: 11.0+
- Driver: Latest NVIDIA drivers

⚠️ 最低配置：
- GPU: NVIDIA GTX 1080 Ti
- Memory: >= 8GB
- CUDA: 11.0+

❌ 不支持：
- AMD GPU
- Intel GPU
- CPU-only环境
```

### 软件要求

```
- Milvus: 2.6.0+
- pymilvus: 2.6.0+
- CUDA Toolkit: 11.0+
- NVIDIA Driver: 450.80.02+
```

---

## 限制与约束

### 硬件限制

1. **必须有GPU**：无GPU环境无法使用
2. **GPU内存限制**：索引必须完全加载到GPU内存
3. **NVIDIA专属**：仅支持NVIDIA GPU + CUDA

### 性能限制

1. **小批量查询**：batch < 10时，GPU启动开销大
2. **数据传输**：CPU-GPU数据传输有开销
3. **单次查询**：不如CPU HNSW（GPU启动开销）

### 适用边界

```
✅ 推荐使用：
- 大规模数据集（> 10M向量）
- 高吞吐量场景（QPS > 1000）
- 批量查询（batch > 10）
- 有GPU资源

⚠️ 谨慎使用：
- 中等规模（1M-10M）
- 单次查询（batch = 1）
- GPU资源有限

❌ 不推荐使用：
- 小规模数据集（< 1M）
- 无GPU环境
- 成本敏感场景
```

---

## RAG应用场景

### 场景1：大规模RAG系统

```python
# 10M+文档，高QPS，实时搜索

collection.create_index("embeddings", {
    "index_type": "GPU_CAGRA",
    "metric_type": "COSINE",
    "params": {
        "intermediate_graph_degree": 128,
        "graph_degree": 64
    }
})

# 混合模式：GPU构建 + CPU查询
collection.load(adapt_for_cpu=True)

# 适用：
# - 大规模知识库（> 10M文档）
# - 高并发查询（QPS > 1000）
# - 实时响应要求（< 10ms）
```

### 场景2：批量文档检索

```python
# 批量查询优化

# 单次查询（慢）
for query in queries:
    results = collection.search([query], "embeddings", params, limit=10)

# 批量查询（快10倍）
results = collection.search(queries, "embeddings", params, limit=10)

# 适用：
# - 批量文档处理
# - 离线分析
# - 数据预处理
```

### 场景3：实时推荐系统

```python
# 实时推荐配置

collection.create_index("embeddings", {
    "index_type": "GPU_CAGRA",
    "metric_type": "IP",  # 内积，适合推荐
    "params": {
        "intermediate_graph_degree": 128,
        "graph_degree": 64
    }
})

# 纯GPU模式：最快响应
collection.load(_resource_groups=["gpu"])

# 适用：
# - 实时推荐
# - 个性化搜索
# - 内容发现
```

---

## 性能优化技巧

### 技巧1：批量查询

```python
# 批量查询提升吞吐量

# 差：单次查询
for query in queries:
    results = collection.search([query], "embeddings", params, limit=10)
# 延迟：5ms × 100 = 500ms

# 好：批量查询
results = collection.search(queries, "embeddings", params, limit=10)
# 延迟：50ms（快10倍）
```

### 技巧2：混合模式

```python
# 成本优化：GPU构建 + CPU查询

# 构建阶段（GPU）
collection.create_index("embeddings", gpu_cagra_params)

# 查询阶段（CPU）
collection.load(adapt_for_cpu=True)

# 优势：
# - 构建快12-15倍
# - 查询成本低
# - 可扩展性好
```

### 技巧3：GPU内存管理

```python
# 监控GPU内存

import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 查询前检查内存
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used / 1024**3:.2f} GB / {info.total / 1024**3:.2f} GB")

# 如果内存不足，使用混合模式
if info.free < required_memory:
    collection.load(adapt_for_cpu=True)
```

### 技巧4：参数调优

```python
# 平衡配置
params = {
    "intermediate_graph_degree": 128,  # 平衡
    "graph_degree": 64                 # 平衡
}

# 高质量配置
params = {
    "intermediate_graph_degree": 256,  # 更高质量
    "graph_degree": 128                # 更高准确率
}

# 快速构建配置
params = {
    "intermediate_graph_degree": 64,   # 更快构建
    "graph_degree": 32                 # 更少内存
}
```

---

## 常见问题

### Q1：GPU_CAGRA为什么这么快？

**A：** GPU有数千个核心，可以并行处理大量向量。

**详细解释：**
- CPU HNSW：16核心，串行处理
- GPU CAGRA：数千核心，并行处理
- 构建速度：12-15x faster
- 查询速度：10x faster（批量查询）

### Q2：混合模式如何工作？

**A：** GPU构建高质量图，CPU执行查询。

**工作流程：**
```
1. 构建阶段（GPU）：
   - 使用GPU快速构建图
   - 12-15x faster than CPU

2. 适配阶段：
   - 将GPU图适配到CPU
   - adapt_for_cpu=True

3. 查询阶段（CPU）：
   - 使用CPU执行查询
   - 降低成本，提高可扩展性
```

### Q3：GPU_CAGRA适合小数据集吗？

**A：** 不适合。小数据集时，GPU启动开销大于收益。

**性能对比：**
```
1M向量，单次查询：
- CPU HNSW: 5ms
- GPU CAGRA: 8ms（慢60%，因为GPU启动开销）

10M向量，批量查询（batch=100）：
- CPU HNSW: 500ms
- GPU CAGRA: 50ms（快10倍）
```

**建议：**
- < 1M向量：使用FLAT或HNSW
- 1M-10M向量：使用HNSW
- > 10M向量：使用GPU_CAGRA

### Q4：如何选择纯GPU还是混合模式？

**A：** 根据成本和性能需求选择。

**决策树：**
```
需要极致性能？
  ├─ 是 → 纯GPU模式
  └─ 否 → 需要降低成本？
       ├─ 是 → 混合模式
       └─ 否 → CPU HNSW
```

---

## 最佳实践

### 实践1：混合模式部署

```python
# 推荐：GPU构建 + CPU查询

# 1. 构建阶段（GPU）
collection.create_index("embeddings", {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",
    "params": {
        "intermediate_graph_degree": 128,
        "graph_degree": 64
    }
})

# 2. 加载阶段（启用CPU查询）
collection.load(adapt_for_cpu=True)

# 3. 查询阶段（CPU）
results = collection.search(queries, "embeddings", params, limit=10)

# 优势：
# - 构建快12-15倍
# - 查询成本低
# - 可扩展性好
```

### 实践2：批量查询优化

```python
# 批量查询提升吞吐量

# 收集查询
query_batch = []
for user_query in user_queries:
    query_batch.append(user_query)

    # 每100个查询批量执行
    if len(query_batch) >= 100:
        results = collection.search(
            query_batch,
            "embeddings",
            params,
            limit=10
        )
        process_results(results)
        query_batch = []
```

### 实践3：GPU内存监控

```python
# 监控GPU内存使用

import pynvml

def monitor_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    used_gb = info.used / 1024**3
    total_gb = info.total / 1024**3
    usage_percent = (info.used / info.total) * 100

    print(f"GPU Memory: {used_gb:.2f} GB / {total_gb:.2f} GB ({usage_percent:.1f}%)")

    if usage_percent > 90:
        print("Warning: GPU memory usage high!")

# 定期监控
monitor_gpu_memory()
```

### 实践4：渐进式部署

```python
# 渐进式部署策略

# 阶段1：小规模测试（1M向量）
test_collection = Collection("test_gpu_cagra")
test_collection.create_index("embeddings", gpu_cagra_params)
benchmark_performance(test_collection)

# 阶段2：中规模验证（10M向量）
if test_passed:
    medium_collection = Collection("medium_gpu_cagra")
    medium_collection.create_index("embeddings", gpu_cagra_params)
    benchmark_performance(medium_collection)

# 阶段3：全规模部署（100M向量）
if medium_passed:
    production_collection = Collection("production_gpu_cagra")
    production_collection.create_index("embeddings", gpu_cagra_params)
    production_collection.load(adapt_for_cpu=True)
```

---

## 对比其他索引

### vs CPU HNSW

| 维度 | CPU HNSW | GPU CAGRA |
|------|----------|-----------|
| 构建速度 | 慢 | 快12-15倍 |
| 查询速度 | 中等 | 快10倍（批量） |
| 硬件要求 | CPU | GPU |
| 成本 | 低 | 高 |
| 适用规模 | 10M-100M | 100M-1B |

**选择建议：**
- 无GPU：CPU HNSW
- 有GPU + 大规模：GPU_CAGRA

### vs RaBitQ

| 维度 | GPU_CAGRA | RaBitQ |
|------|-----------|--------|
| 性能 | 极快 | 快 |
| 内存 | 1.5x | 0.03x |
| 成本 | 高（GPU） | 低（CPU） |
| 适用场景 | 速度优先 | 成本优先 |

**选择建议：**
- 速度优先：GPU_CAGRA
- 成本优先：RaBitQ

---

## 总结

### 核心要点

1. **GPU_CAGRA = GPU并行 + 图索引**：10x速度提升
2. **混合模式**：GPU构建 + CPU查询（2026特性）
3. **适用场景**：大规模数据集（> 10M向量）
4. **硬件要求**：NVIDIA GPU + CUDA

### 使用场景

✅ **推荐使用：**
- 大规模数据集（> 10M向量）
- 高吞吐量场景（QPS > 1000）
- 批量查询（batch > 10）
- 有GPU资源

❌ **不推荐使用：**
- 小规模数据集（< 1M）
- 单次查询（batch = 1）
- 无GPU环境
- 成本敏感场景

### 关键参数

```python
# 推荐配置
intermediate_graph_degree = 128  # 构建质量
graph_degree = 64                # 最终质量
adapt_for_cpu = True             # 混合模式
```

### 下一步

- **成本优化**：学习 [03_核心概念_8_RaBitQ量化.md](./03_核心概念_8_RaBitQ量化.md)
- **索引选型**：学习 [03_核心概念_9_索引选型决策树.md](./03_核心概念_9_索引选型决策树.md)
- **实战应用**：实践 [07_实战代码_场景4_GPU_CAGRA加速.md](./07_实战代码_场景4_GPU_CAGRA加速.md)

---

**记住：GPU_CAGRA是速度之王，但需要GPU资源。混合模式是最佳性价比选择。**
