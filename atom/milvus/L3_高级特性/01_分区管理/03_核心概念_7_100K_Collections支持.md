# 核心概念 7：100K Collections 支持

> Milvus 2.6 支持大规模多租户场景的技术实现

---

## 概述

**100K Collections 支持**是 Milvus 2.6 的核心特性之一，指的是 Milvus 能够在单个集群中支持大规模的 Collection 数量，从而实现百万级租户的多租户场景。

**核心价值**：
- 支持百万级租户
- 热冷存储机制
- 成本优化策略
- 灵活的数据管理

---

## 1. 100K Collections 的定义

### 1.1 什么是 100K Collections？

**来源**：Context7 官方文档

**定义**：100K Collections 是指 Milvus 2.6 支持大规模多租户场景，可以处理数百到数百万租户。这不是指单个 Collection 可以有 100K 个分区，而是指整个 Milvus 集群可以支持大量的 Collection 或租户。

**核心特性**：
- **大规模支持**：单集群处理数百到数百万租户
- **热冷存储**：频繁访问的热数据存储在内存或 SSD 中，较少访问的冷数据保存在较慢、成本效益高的存储中
- **成本优化**：通过热冷存储显著降低成本，同时保持关键任务的高性能
- **灵活策略**：支持 Database、Collection、Partition 或 Partition Key 级别的隔离

### 1.2 与分区的关系

**100K Collections 不等于 100K 个分区**：
- **100K Collections**：指整个 Milvus 集群可以支持大量的 Collection
- **1,024 个分区**：指单个 Collection 最多可以有 1,024 个手动分区
- **Partition Key**：通过 Partition Key 自动分区，可以在单个 Collection 中支持数百万租户

**示例**：
```python
# 场景 1：100K Collections（Collection-level 多租户）
# 为每个租户创建独立的 Collection
for tenant_id in range(100000):
    collection = Collection(f"tenant_{tenant_id}_collection", schema)

# 场景 2：Partition Key（Partition Key-level 多租户）
# 单个 Collection 支持数百万租户
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True)
    ]
)
collection = Collection("multi_tenant_collection", schema)
# 支持数百万个 tenant_id 值
```

---

## 2. 热冷存储机制

### 2.1 热冷存储的定义

**来源**：Context7 官方文档

**定义**：Milvus 通过热冷存储增强成本效益：
- **热数据**：频繁访问的数据可以存储在内存或 SSD 中以获得更好的性能
- **冷数据**：较少访问的数据保存在较慢、成本效益高的存储中

**核心价值**：
- 显著降低成本
- 保持关键任务的高性能
- 灵活的数据管理

### 2.2 热冷存储的实现

**手动分区的热冷存储**：
```python
from pymilvus import Collection
from datetime import datetime, timedelta

collection = Collection("time_series_data")

# 创建按月份的分区
for i in range(12):
    month = (datetime.now() - timedelta(days=30*i)).strftime("%Y_%m")
    collection.create_partition(month)

# 只加载最近 3 个月的数据（热数据）
hot_partitions = [
    (datetime.now() - timedelta(days=30*i)).strftime("%Y_%m")
    for i in range(3)
]
collection.load(partition_names=hot_partitions)

# 冷数据保存在磁盘上，不加载到内存
# 如果需要查询冷数据，临时加载
cold_month = "2025_12"
collection.load(partition_names=[cold_month])
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    partition_names=[cold_month]
)
# 查询完成后释放冷数据
collection.release(partition_names=[cold_month])
```

**Collection-level 的热冷存储**：
```python
from pymilvus import Collection, utility

# 场景：100 个租户，每个租户一个 Collection
# 只加载活跃租户的 Collection（热数据）

# 加载活跃租户的 Collection
active_tenants = [f"tenant_{i}_collection" for i in range(1, 11)]
for tenant_collection in active_tenants:
    collection = Collection(tenant_collection)
    collection.load()

# 释放不活跃租户的 Collection（冷数据）
inactive_tenants = [f"tenant_{i}_collection" for i in range(11, 101)]
for tenant_collection in inactive_tenants:
    collection = Collection(tenant_collection)
    collection.release()
```

### 2.3 热冷存储的成本优化

**成本对比**：
```python
# 场景：100 个租户，每个租户 100 万条数据，总数据量 1 亿条

# 策略 1：全部加载到内存（热数据）
# 内存占用：1 亿条数据 × 128 维 × 4 字节 = 51.2 GB
# 成本：高

# 策略 2：只加载活跃租户（热数据）
# 假设只有 10% 的租户活跃
# 内存占用：1000 万条数据 × 128 维 × 4 字节 = 5.12 GB
# 成本：降低 90%

# 策略 3：按需加载（热冷混合）
# 内存占用：根据实际访问情况动态调整
# 成本：最优
```

---

## 3. 多租户策略与 100K Collections

### 3.1 四种多租户策略

**来源**：Context7 官方文档

| 策略 | 可扩展性 | 适用场景 | 与 100K Collections 的关系 |
|------|----------|----------|---------------------------|
| Database-level | 低 | 少量租户（< 10） | 不适用 |
| Collection-level | 中 | 中等数量租户（< 100K） | 适用（100K Collections） |
| Partition-level | 中 | 中等数量租户（< 1,024） | 不适用 |
| Partition Key-level | 高 | 大量租户（百万级） | 适用（单 Collection 支持百万租户） |

### 3.2 Collection-level 多租户（100K Collections）

**场景**：中等数量租户（< 100K），每个租户需要独立的 Schema

**实现**：
```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 为每个租户创建独立的 Collection
for tenant_id in range(100000):
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024)
        ]
    )
    collection = Collection(f"tenant_{tenant_id}_collection", schema)

    # 创建索引
    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="vector", index_params=index_params)

# 只加载活跃租户的 Collection
active_tenants = [f"tenant_{i}_collection" for i in range(1, 11)]
for tenant_collection in active_tenants:
    collection = Collection(tenant_collection)
    collection.load()
```

**优点**：
- ✅ 强隔离（每个租户独立 Collection）
- ✅ Schema 灵活性（每个租户可以有独立 Schema）
- ✅ 支持 RBAC

**缺点**：
- ❌ 管理开销大（需要管理大量 Collection）
- ❌ 资源开销较大（每个 Collection 需要独立资源）

### 3.3 Partition Key-level 多租户（推荐）

**场景**：大量租户（百万级），所有租户共享相同的 Schema

**实现**：
```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 创建单个 Collection，使用 Partition Key 支持百万租户
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048)
    ]
)

collection = Collection(
    name="multi_tenant_collection",
    schema=schema,
    properties={
        "partition.num": 200,  # 200 个物理分区
        "partitionkey.isolation": True
    }
)

# 创建索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="vector", index_params=index_params)

# 加载 Collection
collection.load()

# 支持数百万个 tenant_id 值
```

**优点**：
- ✅ 最高可扩展性（支持数百万租户）
- ✅ 自动分区管理，无需手动创建分区
- ✅ 管理开销小（只需管理一个 Collection）

**缺点**：
- ❌ 数据隔离较弱（多个租户可能共享物理分区）
- ❌ 所有租户必须共享相同的 Schema
- ❌ 不支持 RBAC

---

## 4. 成本优化策略

### 4.1 内存优化

**策略 1：只加载活跃租户**
```python
# 场景：100K 个租户，只有 1% 活跃

# 只加载活跃租户的 Collection
active_tenants = [f"tenant_{i}_collection" for i in range(1, 1001)]
for tenant_collection in active_tenants:
    collection = Collection(tenant_collection)
    collection.load()

# 内存节省：99%
```

**策略 2：按需加载**
```python
def query_tenant_data(tenant_id, query_vector):
    tenant_collection = f"tenant_{tenant_id}_collection"
    collection = Collection(tenant_collection)

    # 检查是否已加载
    if not collection.is_loaded:
        collection.load()

    # 查询数据
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10
    )

    # 查询完成后可以选择释放（可选）
    # collection.release()

    return results
```

### 4.2 存储优化

**策略 1：使用 Partition Key 自动分区**
```python
# 使用 Partition Key 自动分区，减少 Collection 数量
# 从 100K Collections 减少到 1 个 Collection
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048)
    ]
)
collection = Collection("multi_tenant_collection", schema)

# 管理开销降低：100K Collections → 1 Collection
```

**策略 2：定期清理冷数据**
```python
from datetime import datetime, timedelta

# 删除 6 个月前的冷数据
cutoff_date = datetime.now() - timedelta(days=180)

# 删除冷数据分区
for i in range(6, 12):
    month = (datetime.now() - timedelta(days=30*i)).strftime("%Y_%m")
    if collection.has_partition(month):
        collection.drop_partition(month)

# 存储节省：50%
```

---

## 5. 实际应用案例

### 5.1 SaaS 应用（100K 租户）

**场景**：SaaS 应用，有 100K 个租户，每个租户需要独立的 Schema

**方案 1：Collection-level 多租户（100K Collections）**
```python
# 为每个租户创建独立的 Collection
for tenant_id in range(100000):
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            # 每个租户可以有不同的字段
            FieldSchema(name=f"custom_field_{tenant_id}", dtype=DataType.VARCHAR, max_length=512)
        ]
    )
    collection = Collection(f"tenant_{tenant_id}_collection", schema)

# 只加载活跃租户的 Collection（1% 活跃）
active_tenants = [f"tenant_{i}_collection" for i in range(1, 1001)]
for tenant_collection in active_tenants:
    collection = Collection(tenant_collection)
    collection.load()
```

**方案 2：Partition Key-level 多租户（推荐）**
```python
# 使用 Partition Key 自动分区
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048)
    ]
)

collection = Collection(
    name="multi_tenant_collection",
    schema=schema,
    properties={
        "partition.num": 200,
        "partitionkey.isolation": True
    }
)

# 加载 Collection
collection.load()

# 支持 100K 个租户
```

### 5.2 时间序列数据（热冷存储）

**场景**：时间序列数据，按月份分区，只查询最近的数据

**方案**：
```python
from datetime import datetime, timedelta

collection = Collection("time_series_data")

# 创建按月份的分区
for i in range(12):
    month = (datetime.now() - timedelta(days=30*i)).strftime("%Y_%m")
    collection.create_partition(month)

# 只加载最近 3 个月的数据（热数据）
hot_partitions = [
    (datetime.now() - timedelta(days=30*i)).strftime("%Y_%m")
    for i in range(3)
]
collection.load(partition_names=hot_partitions)

# 冷数据保存在磁盘上，不加载到内存
# 内存节省：75%
```

---

## 6. 与其他多租户策略的对比

| 维度 | Collection-level (100K Collections) | Partition Key-level |
|------|-------------------------------------|---------------------|
| **可扩展性** | 中（< 100K 租户） | 高（百万级租户） |
| **数据隔离** | 强（独立 Collection） | 弱（逻辑隔离） |
| **Schema 灵活性** | 高（每个租户独立 Schema） | 低（共享 Schema） |
| **管理开销** | 高（需要管理大量 Collection） | 低（只需管理一个 Collection） |
| **内存优化** | 支持（可独立加载/释放） | 不支持（全部加载） |
| **RBAC 支持** | 是 | 否 |
| **适用场景** | 中等数量租户，需要独立 Schema | 大量租户，共享 Schema |

---

## 7. 最佳实践

### 7.1 选择建议

**根据租户数量选择**：
- **< 100 个租户**：Collection-level 或 Database-level
- **100-100K 个租户**：Collection-level（100K Collections）
- **> 100K 个租户**：Partition Key-level（推荐）

**根据 Schema 需求选择**：
- **每个租户需要独立 Schema**：Collection-level（100K Collections）
- **所有租户共享 Schema**：Partition Key-level

### 7.2 成本优化建议

**内存优化**：
- 只加载活跃租户的 Collection
- 按需加载冷数据
- 定期释放不活跃租户的 Collection

**存储优化**：
- 使用 Partition Key 自动分区，减少 Collection 数量
- 定期清理冷数据
- 使用压缩算法（如 RaBitQ 量化）

### 7.3 监控和告警

**监控指标**：
- Collection 数量
- 加载的 Collection 数量
- 内存使用情况
- 活跃租户数量

**告警策略**：
- Collection 数量超过阈值（如 50K）
- 内存使用超过阈值（如 80%）
- 活跃租户数量异常

---

## 核心要点总结

1. **100K Collections 支持**：Milvus 2.6 支持大规模多租户场景，可以处理数百到数百万租户
2. **热冷存储机制**：频繁访问的热数据存储在内存或 SSD 中，较少访问的冷数据保存在较慢、成本效益高的存储中
3. **成本优化**：通过热冷存储显著降低成本，同时保持关键任务的高性能
4. **多租户策略**：Collection-level（100K Collections）适用于中等数量租户，Partition Key-level 适用于大量租户
5. **最佳实践**：根据租户数量和 Schema 需求选择合适的策略

---

## 参考来源

- **Context7**：100K Collections 支持、热冷存储机制、多租户策略对比

---

**下一步**：阅读 **04_最小可用.md**，快速掌握分区管理的核心知识。
