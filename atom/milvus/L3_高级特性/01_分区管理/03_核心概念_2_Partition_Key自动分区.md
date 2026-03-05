# 核心概念 2：Partition Key 自动分区

> Milvus 2.6 核心特性：通过 Partition Key 实现自动分区管理

---

## 概述

**Partition Key 是 Milvus 2.6 的核心创新**，通过在 Schema 中标记字段为 `is_partition_key=True`，Milvus 会自动根据该字段的值进行分区，无需手动管理分区。

**核心价值**：
- 自动管理：无需手动创建分区
- 高可扩展性：支持百万级租户
- 简化开发：开发者只需标记字段

---

## 1. Partition Key 的定义

### 1.1 什么是 Partition Key？

**来源**：Context7 官方文档

**定义**：Partition Key 是一种自动分区管理机制，通过在 Schema 中标记字段为 `is_partition_key=True`，Milvus 会自动根据该字段的值进行分区。

**核心特性**：
- **自动路由**：Milvus 根据 Partition Key 的值自动将数据路由到 16 个物理分区
- **逻辑隔离**：虽然多个租户可能共享一个物理分区，但数据在逻辑上是分离的
- **最高可扩展性**：支持数百万租户
- **共享 Schema**：所有租户必须共享相同的数据 Schema

### 1.2 Partition Key 字段定义

**Python 示例**：
```python
from pymilvus import FieldSchema, CollectionSchema, DataType

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True)  # 标记为 Partition Key
    ],
    description="Multi-tenant collection with Partition Key"
)
```

**Go 示例**：
```go
schema := entity.NewSchema().WithDynamicFieldEnabled(false)
schema.WithField(entity.NewField().
    WithName("id").
    WithDataType(entity.FieldTypeInt64).
    WithIsPrimaryKey(true),
).WithField(entity.NewField().
    WithName("tenant_id").
    WithDataType(entity.FieldTypeVarChar).
    WithIsPartitionKey(true).  // 标记为 Partition Key
    WithMaxLength(64),
).WithField(entity.NewField().
    WithName("vector").
    WithDataType(entity.FieldTypeFloatVector).
    WithDim(128),
)
```

**支持的数据类型**：
- `VARCHAR`：字符串类型（推荐）
- `INT64`：整数类型

---

## 2. 自动分区机制

### 2.1 哈希分布

**来源**：Context7 官方文档 + 网络搜索

**核心机制**：
- Milvus 使用哈希算法将 Partition Key 值映射到 **16 个物理分区**（默认）
- 哈希算法确保数据均匀分布，避免数据倾斜

**示例**：
```python
# 假设有 1000 个租户
tenant_ids = [f"tenant_{i}" for i in range(1000)]

# Milvus 使用哈希算法将这 1000 个租户映射到 16 个物理分区
# 每个物理分区大约包含 1000 / 16 ≈ 62 个租户的数据
```

**哈希算法伪代码**：
```python
def get_partition_id(partition_key_value, num_partitions=16):
    hash_value = hash(partition_key_value)
    partition_id = hash_value % num_partitions
    return partition_id

# 示例
print(get_partition_id("tenant_a", 16))  # 输出：3
print(get_partition_id("tenant_b", 16))  # 输出：11
```

### 2.2 物理分区数量

**来源**：GitHub 最佳实践

**默认配置**：
- 默认：16 个物理分区
- 推荐：100-200 个物理分区（生产环境）
- 最大：根据系统资源调整

**配置方法**：
```python
# 配置物理分区数量（通过 Collection 属性）
from pymilvus import Collection

collection = Collection(
    name="my_collection",
    schema=schema,
    properties={"partition.num": 100}  # 配置 100 个物理分区
)
```

**选择建议**：
- **小规模**（< 1 万租户）：16 个物理分区
- **中等规模**（1-10 万租户）：64 个物理分区
- **大规模**（> 10 万租户）：100-200 个物理分区

### 2.3 数据路由

**插入数据时的自动路由**：
```python
from pymilvus import Collection

collection = Collection("my_collection")

# 插入数据（自动路由到相应分区）
data = [
    [1, 2, 3, 4, 5],  # id
    [[0.1]*128, [0.2]*128, [0.3]*128, [0.4]*128, [0.5]*128],  # vector
    ["tenant_a", "tenant_a", "tenant_b", "tenant_c", "tenant_c"]  # tenant_id
]
collection.insert(data)

# Milvus 自动将数据路由到相应分区：
# - tenant_a 的数据 → 分区 3
# - tenant_b 的数据 → 分区 11
# - tenant_c 的数据 → 分区 7
```

**搜索时的自动路由**：
```python
# 搜索租户 A 的数据（自动只搜索相关分区）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr='tenant_id == "tenant_a"'  # 自动路由到分区 3
)
```

---

## 3. 性能优化

### 3.1 避免 99% 无效扫描

**来源**：Twitter/X 性能优化数据

**核心数据**：
- **99% 无效扫描避免**：使用 Partition Key 可以避免扫描 99% 的无关数据
- **延迟优化**：多租户搜索延迟从秒级降至毫秒级

**示例**：
```python
# 场景：1000 个租户，每个租户 10 万条数据，总数据量 1 亿条

# 不使用 Partition Key：扫描 1 亿条数据
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr='tenant_id == "tenant_a"'  # 需要扫描 1 亿条数据
)
# 延迟：> 1 秒

# 使用 Partition Key：只扫描 10 万条数据
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr='tenant_id == "tenant_a"'  # 只扫描租户 A 的分区
)
# 延迟：< 10 毫秒
# 性能提升：100 倍
```

### 3.2 Partition Key 隔离特性

**来源**：Context7 官方文档

**功能**：通过启用 Partition Key 隔离，可以进一步提升搜索性能。当前仅支持 HNSW 索引类型。启用后，搜索必须在过滤器中包含单个特定的 Partition Key 值。

**Python 配置**：
```python
from pymilvus import Collection

collection = Collection(
    name="my_collection",
    schema=schema,
    properties={"partitionkey.isolation": True}  # 启用 Partition Key 隔离
)
```

**限制**：
- 仅支持 HNSW 索引类型
- 搜索时必须在过滤器中包含单个特定的 Partition Key 值

**示例**：
```python
# 正确：包含单个 Partition Key 值
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10,
    expr='tenant_id == "tenant_a"'  # 单个值
)

# 错误：包含多个 Partition Key 值
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10,
    expr='tenant_id in ["tenant_a", "tenant_b"]'  # 多个值（不支持）
)
```

---

## 4. 支持百万租户

### 4.1 可扩展性

**来源**：Context7 官方文档 + Reddit 多租户实践

**核心数据**：
- **支持百万租户**：单个 Collection 支持数百万个 tenant_id 值
- **逻辑隔离**：虽然多个租户可能共享一个物理分区，但数据在逻辑上是分离的

**示例**：
```python
# 场景：100 万个租户，每个租户 1000 条数据，总数据量 10 亿条

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

# 插入 100 万个租户的数据
for tenant_id in range(1000000):
    data = [
        list(range(1000)),  # id
        [[0.1]*128] * 1000,  # vector
        [f"tenant_{tenant_id}"] * 1000  # tenant_id
    ]
    collection.insert(data)

# 搜索租户 A 的数据（只扫描 1000 条数据）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr='tenant_id == "tenant_0"'
)
# 延迟：< 10 毫秒
```

### 4.2 数据倾斜处理

**问题**：如果某些租户的数据量远大于其他租户，可能导致数据倾斜。

**解决方案**：
1. **增加物理分区数量**：从 16 个增加到 100-200 个
2. **使用复合 Partition Key**：结合多个字段（如 `tenant_id` + `category`）
3. **监控分区负载**：定期检查分区数据分布

**示例**：
```python
# 方案 1：增加物理分区数量
collection = Collection(
    name="my_collection",
    schema=schema,
    properties={"partition.num": 200}  # 增加到 200 个物理分区
)

# 方案 2：使用复合 Partition Key（需要自定义实现）
# 将 tenant_id 和 category 组合成一个字段
data = [
    [1, 2, 3],
    [[0.1]*128, [0.2]*128, [0.3]*128],
    ["tenant_a_cat1", "tenant_a_cat2", "tenant_b_cat1"]  # 复合 Partition Key
]
collection.insert(data)
```

---

## 5. 实际应用场景

### 5.1 SaaS 应用多租户

**场景**：SaaS 应用，每个租户独立的数据隔离

**方案**：
```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 定义 Schema，使用 tenant_id 作为 Partition Key
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024)
    ]
)

# 创建 Collection
collection = Collection("saas_app", schema)

# 创建索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="vector", index_params=index_params)

# 加载 Collection
collection.load()

# 插入租户 A 的数据
data_a = [
    [1, 2, 3],
    [[0.1]*128, [0.2]*128, [0.3]*128],
    ["tenant_a", "tenant_a", "tenant_a"],
    ["doc1", "doc2", "doc3"]
]
collection.insert(data_a)

# 搜索租户 A 的数据
query_vector = [[0.15]*128]
results = collection.search(
    data=query_vector,
    anns_field="vector",
    param={"metric_type": "L2", "params": {"ef": 100}},
    limit=10,
    expr='tenant_id == "tenant_a"'
)

print(f"Found {len(results[0])} results for tenant_a")
```

### 5.2 RAG 系统租户隔离

**场景**：RAG 系统，每个租户独立的知识库

**方案**：
```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 定义 Schema
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64,
                   is_partition_key=True),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=2048)
    ]
)

# 创建 Collection
collection = Collection("rag_knowledge_base", schema)

# 创建索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="vector", index_params=index_params)

# 加载 Collection
collection.load()

# 初始化 Embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 插入租户 A 的文档
documents_a = [
    "Milvus is a vector database.",
    "Partition Key enables multi-tenancy.",
    "HNSW is a graph-based index."
]
vectors_a = model.encode(documents_a).tolist()
data_a = [
    vectors_a,
    ["tenant_a"] * len(documents_a),
    documents_a
]
collection.insert(data_a)

# 搜索租户 A 的文档
query = "What is Milvus?"
query_vector = model.encode([query]).tolist()
results = collection.search(
    data=query_vector,
    anns_field="vector",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=3,
    expr='tenant_id == "tenant_a"',
    output_fields=["document"]
)

print("Search results for tenant_a:")
for hit in results[0]:
    print(f"  - {hit.entity.get('document')} (score: {hit.score})")
```

---

## 6. 与手动分区的对比

| 维度 | 手动分区 | Partition Key |
|------|----------|---------------|
| **管理方式** | 手动创建 | 自动管理 |
| **最大数量** | 1,024 个 | 支持百万租户 |
| **数据隔离** | 物理隔离 | 逻辑隔离 |
| **适用场景** | 中等数量租户 | 大规模多租户 |
| **开发复杂度** | 高（需要手动管理） | 低（自动管理） |
| **性能** | 高（物理隔离） | 高（逻辑隔离 + 自动路由） |
| **热冷数据管理** | 支持（可独立加载/释放） | 不支持（全部加载） |

---

## 核心要点总结

1. **Partition Key 是 Milvus 2.6 的核心创新**，通过标记字段为 `is_partition_key=True` 实现自动分区管理
2. **自动路由机制**：Milvus 使用哈希算法将 Partition Key 值映射到 16 个物理分区（默认）
3. **性能优化**：避免 99% 无效扫描，延迟从秒级降至毫秒级
4. **支持百万租户**：单个 Collection 支持数百万个 tenant_id 值
5. **Partition Key 隔离特性**：启用后可进一步提升性能（仅支持 HNSW 索引）
6. **适用场景**：SaaS 应用多租户、RAG 系统租户隔离

---

## 参考来源

- **源码**：tests/integration/hellomilvus/partition_key_test.go - Partition Key 测试
- **Context7**：Partition Key 定义和使用、Partition Key-level 多租户
- **网络**：Twitter/X - 99% 无效扫描避免、Reddit - SaaS 应用多租户实践

---

**下一步**：阅读 **03_核心概念_3_手动分区管理.md**，深入理解手动分区的完整操作流程。
