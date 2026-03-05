# 核心概念 1：Partition 基础概念

> Partition 的定义、数据结构和生命周期

---

## 概述

**Partition 是 Collection 的子集**，用于数据隔离和性能优化。理解 Partition 的基础概念是掌握分区管理的第一步。

---

## 1. Partition 的定义

### 1.1 形式化定义

```
Collection C = {entity₁, entity₂, ..., entityₙ}
Partition P ⊆ C

特性：
- Partition 是 Collection 的子集
- 所有 Partition 的并集 = Collection
- Partition 之间可以不相交（推荐）
```

### 1.2 核心特征

**与 Collection 的关系**：
- Partition 继承 Collection 的 Schema
- Partition 共享 Collection 的索引配置
- Partition 可以独立加载和释放

**数据隔离**：
- 物理隔离：手动分区（Partition-level）
- 逻辑隔离：Partition Key（Partition Key-level）

---

## 2. Partition 数据结构

### 2.1 客户端视角

**来源**：client/entity/collection.go

```go
// Partition represent partition meta in Milvus
type Partition struct {
	ID     int64  // partition id
	Name   string // partition name
	Loaded bool   // partition loaded
}
```

**字段说明**：
- `ID`：分区的唯一标识符（系统生成）
- `Name`：分区名称（用户可读，用户指定）
- `Loaded`：分区是否已加载到内存

**Python 对应**：
```python
from pymilvus import Collection

collection = Collection("my_collection")

# 获取分区信息
partitions = collection.partitions
for partition in partitions:
    print(f"Partition: {partition.name}")
    print(f"  - Loaded: {partition.is_loaded}")
```

### 2.2 内部视角

**来源**：internal/metastore/model/partition.go

```go
type Partition struct {
	PartitionID               int64
	PartitionName             string
	PartitionCreatedTimestamp uint64
	CollectionID              int64
	State                     pb.PartitionState
}
```

**字段说明**：
- `PartitionID`：分区的唯一标识符
- `PartitionName`：分区名称
- `PartitionCreatedTimestamp`：分区创建时间戳
- `CollectionID`：所属 Collection 的 ID
- `State`：分区状态（`PartitionState_PartitionCreated` 等）

**状态管理**：
```go
func (p *Partition) Available() bool {
	return p.State == pb.PartitionState_PartitionCreated
}
```

---

## 3. Partition 生命周期

### 3.1 创建（Create）

**手动创建**：
```python
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 创建分区
collection.create_partition("partition_a")
```

**自动创建（Partition Key）**：
```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

# 定义 Schema，标记 Partition Key
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True)  # 自动分区
    ]
)

# 创建 Collection（Milvus 自动创建 16 个物理分区）
collection = Collection("my_collection", schema)
```

### 3.2 加载（Load）

**加载分区到内存**：
```python
# 加载特定分区
collection.load(partition_names=["partition_a"])

# 检查加载状态
state = collection.get_load_state(partition_name="partition_a")
print(state)  # LoadState.Loaded
```

**为什么需要加载？**
- Milvus 是内存数据库，数据需要加载到内存才能检索
- 只加载需要的分区，节省内存

### 3.3 使用（Use）

**在检索时指定分区**：
```python
# 搜索特定分区
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    partition_names=["partition_a"]  # 只搜索 partition_a
)
```

### 3.4 释放（Release）

**从内存中释放分区**：
```python
# 释放分区
collection.release(partition_names=["partition_a"])

# 检查加载状态
state = collection.get_load_state(partition_name="partition_a")
print(state)  # LoadState.NotLoad
```

### 3.5 删除（Drop）

**删除分区及其数据**：
```python
# 删除分区
collection.drop_partition("partition_a")

# 检查分区是否存在
has_partition = collection.has_partition("partition_a")
print(has_partition)  # False
```

---

## 4. 默认分区（_default）

### 4.1 自动创建

**来源**：Context7 官方文档

**核心信息**：
- 创建 Collection 时，Milvus 自动创建一个名为 `_default` 的分区
- 如果不添加其他分区，所有插入的实体都会进入默认分区
- 所有搜索和查询也会在默认分区内进行

**示例**：
```python
from pymilvus import Collection

collection = Collection("my_collection")

# 列出所有分区
partitions = collection.partitions
print([p.name for p in partitions])  # ['_default']
```

### 4.2 使用默认分区

**插入数据到默认分区**：
```python
# 不指定分区，数据进入默认分区
data = [
    [1, 2, 3],  # id
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # vector
]
collection.insert(data)
```

**搜索默认分区**：
```python
# 不指定分区，搜索默认分区
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10
)
```

---

## 5. 分区数量限制

### 5.1 手动分区限制

**来源**：Context7 官方文档

**核心限制**：
- 一个 Collection 最多可以有 **1,024 个分区**（手动分区）
- 包括默认分区 `_default`

**示例**：
```python
# 创建多个分区
for i in range(1, 1024):  # 最多 1023 个（加上 _default 共 1024 个）
    collection.create_partition(f"partition_{i}")
```

### 5.2 Partition Key 限制

**来源**：Context7 官方文档

**核心限制**：
- Partition Key 自动路由到 **16 个物理分区**（默认）
- 可以配置为 100-200 个物理分区（推荐）
- 支持数百万租户（逻辑隔离）

**示例**：
```python
# Partition Key 自动分区（无需手动创建分区）
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True)
    ]
)
collection = Collection("my_collection", schema)

# Milvus 自动创建 16 个物理分区
# 支持数百万个 tenant_id 值（逻辑隔离）
```

---

## 6. Partition 与 Collection 的关系

### 6.1 Schema 继承

**Partition 继承 Collection 的 Schema**：
```python
# Collection Schema
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64)
    ]
)
collection = Collection("my_collection", schema)

# 创建分区（继承 Schema）
collection.create_partition("partition_a")

# 插入数据到分区（必须符合 Schema）
data = [
    [1, 2, 3],  # id
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # vector
    ["cat1", "cat2", "cat3"]  # category
]
collection.insert(data, partition_name="partition_a")
```

### 6.2 索引共享

**Partition 共享 Collection 的索引配置**：
```python
# 创建索引（应用于所有分区）
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="vector", index_params=index_params)

# 所有分区共享相同的索引配置
```

### 6.3 独立加载

**Partition 可以独立加载和释放**：
```python
# 只加载 partition_a
collection.load(partition_names=["partition_a"])

# 搜索 partition_a（其他分区未加载）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    partition_names=["partition_a"]
)

# 释放 partition_a，加载 partition_b
collection.release(partition_names=["partition_a"])
collection.load(partition_names=["partition_b"])
```

---

## 7. 实际应用场景

### 7.1 多租户隔离

**场景**：SaaS 应用，每个租户独立的数据隔离

**方案 1：手动分区**（中等数量租户）
```python
# 为每个租户创建分区
collection.create_partition("tenant_a")
collection.create_partition("tenant_b")

# 插入租户 A 的数据
collection.insert(data_a, partition_name="tenant_a")

# 搜索租户 A 的数据
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    partition_names=["tenant_a"]
)
```

**方案 2：Partition Key**（大规模多租户）
```python
# 使用 Partition Key 自动分区
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True)
    ]
)
collection = Collection("my_collection", schema)

# 插入数据（自动路由到相应分区）
data = [
    [1, 2, 3],
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    ["tenant_a", "tenant_a", "tenant_b"]  # tenant_id
]
collection.insert(data)

# 搜索租户 A 的数据（自动只搜索相关分区）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr='tenant_id == "tenant_a"'  # 自动路由
)
```

### 7.2 时间序列数据

**场景**：按时间分区，只查询最近的数据

**方案**：
```python
# 按月份创建分区
collection.create_partition("2026_01")
collection.create_partition("2026_02")

# 插入数据到相应分区
collection.insert(data_jan, partition_name="2026_01")
collection.insert(data_feb, partition_name="2026_02")

# 只加载最近的分区
collection.load(partition_names=["2026_02"])

# 搜索最近的数据
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    partition_names=["2026_02"]
)
```

---

## 核心要点总结

1. **Partition 是 Collection 的子集**，用于数据隔离和性能优化
2. **两种数据结构**：客户端视角（ID, Name, Loaded）和内部视角（PartitionID, PartitionName, PartitionCreatedTimestamp, CollectionID, State）
3. **生命周期**：创建 → 加载 → 使用 → 释放 → 删除
4. **默认分区**：Milvus 自动创建 `_default` 分区
5. **数量限制**：手动分区最多 1,024 个，Partition Key 支持百万租户
6. **与 Collection 的关系**：继承 Schema、共享索引、独立加载

---

## 参考来源

- **源码**：client/entity/collection.go, internal/metastore/model/partition.go
- **Context7**：Partition 定义、默认分区、数量限制
- **网络**：GitHub 最佳实践

---

**下一步**：阅读 **03_核心概念_2_Partition_Key自动分区.md**，深入理解 Partition Key 的自动分区机制。
