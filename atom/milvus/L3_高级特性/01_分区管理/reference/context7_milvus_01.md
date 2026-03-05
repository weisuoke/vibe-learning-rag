---
type: context7_documentation
library: milvus
version: v2.6.x
fetched_at: 2026-02-25
knowledge_point: 01_分区管理
context7_query: partition management, partition key, partition operations
---

# Context7 文档：Milvus 分区管理

## 文档来源
- 库名称：Milvus Documentation
- 版本：v2.6.x
- 官方文档链接：https://github.com/milvus-io/milvus-docs/blob/v2.6.x/

## 关键信息提取

### 1. Partition Key（Milvus 2.6 核心特性）

**定义**：Partition Key 是一种自动分区管理机制，通过在 Schema 中标记字段为 `is_partition_key=True`，Milvus 会自动根据该字段的值进行分区。

**Python 示例**：
```python
from pymilvus import FieldSchema, CollectionSchema, DataType

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
        FieldSchema(name="my_varchar", dtype=DataType.VARCHAR, max_length=512, is_partition_key=True)
    ],
    description="My collection description"
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
    WithName("my_varchar").
    WithDataType(entity.FieldTypeVarChar).
    WithIsPartitionKey(true).
    WithMaxLength(512),
).WithField(entity.NewField().
    WithName("vector").
    WithDataType(entity.FieldTypeFloatVector).
    WithDim(5),
)
```

**Java 示例**：
```java
schema.addField(AddFieldReq.builder()
        .fieldName("my_varchar")
        .dataType(DataType.VarChar)
        .maxLength(512)
        .isPartitionKey(true)
        .build());
```

**JavaScript 示例**：
```javascript
const fields = [
    {
        name: "my_varchar",
        data_type: DataType.VarChar,
        max_length: 512,
        is_partition_key: true
    }
]
```

### 2. Partition 管理操作

#### 2.1 创建分区（CreatePartition）

**Python**：
```python
# 手动创建分区
client.create_partition(
    collection_name="my_collection",
    partition_name="partitionA"
)
```

**Java**：
```java
CreatePartitionReq createPartitionReq = CreatePartitionReq.builder()
        .collectionName("my_collection")
        .partitionName("partitionA")
        .build();

client.createPartition(createPartitionReq);
```

**JavaScript**：
```javascript
await client.createPartition({
    collection_name: "my_collection",
    partition_name: "partitionA"
})
```

**Go**：
```go
err = client.CreatePartition(ctx, milvusclient.NewCreatePartitionOption("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}
```

#### 2.2 列出分区（ListPartitions）

**Python**：
```python
partitions = client.list_partitions(collection_name="my_collection")
print(partitions)
# Output: ["_default", "partitionA"]
```

**Java**：
```java
ListPartitionsReq listPartitionsReq = ListPartitionsReq.builder()
        .collectionName("my_collection")
        .build();

List<String> partitionNames = client.listPartitions(listPartitionsReq);
System.out.println(partitionNames);
// Output: [_default, partitionA]
```

**JavaScript**：
```javascript
res = await client.listPartitions({
    collection_name: "my_collection"
})
console.log(res)
// Output: ["_default", "partitionA"]
```

**Go**：
```go
partitionNames, err := client.ListPartitions(ctx, milvusclient.NewListPartitionOption("my_collection"))
if err != nil {
    fmt.Println(err.Error())
}
fmt.Println(partitionNames)
// Output: ["_default", "partitionA"]
```

#### 2.3 检查分区是否存在（HasPartition）

**Python**：
```python
has_partition = client.has_partition(
    collection_name="my_collection",
    partition_name="partitionA"
)
print(has_partition)
# Output: True
```

**Java**：
```java
HasPartitionReq hasPartitionReq = HasPartitionReq.builder()
        .collectionName("my_collection")
        .partitionName("partitionA")
        .build();

Boolean hasPartitionRes = client.hasPartition(hasPartitionReq);
System.out.println(hasPartitionRes);
// Output: true
```

**JavaScript**：
```javascript
res = await client.hasPartition({
    collection_name: "my_collection",
    partition_name: "partitionA"
})
console.log(res.value)
// Output: true
```

**Go**：
```go
result, err := client.HasPartition(ctx, milvusclient.NewHasPartitionOption("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}
fmt.Println(result)
// Output: true
```

#### 2.4 加载分区（LoadPartitions）

**Python**：
```python
client.load_partitions(
    collection_name="my_collection",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="my_collection",
    partition_name="partitionA"
)
print(res)
# Output: {"state": "<LoadState: Loaded>"}
```

**Java**：
```java
LoadPartitionsReq loadPartitionsReq = LoadPartitionsReq.builder()
        .collectionName("my_collection")
        .partitionNames(Collections.singletonList("partitionA"))
        .build();

client.loadPartitions(loadPartitionsReq);

GetLoadStateReq getLoadStateReq = GetLoadStateReq.builder()
        .collectionName("my_collection")
        .partitionName("partitionA")
        .build();

Boolean getLoadStateRes = client.getLoadState(getLoadStateReq);
System.out.println(getLoadStateRes);
```

**JavaScript**：
```javascript
await client.loadPartitions({
    collection_name: "my_collection",
    partition_names: ["partitionA"]
})

res = await client.getLoadState({
    collection_name: "my_collection",
    partition_name: "partitionA"
})
console.log(res)
// Output: LoadStateLoaded
```

**Go**：
```go
err = client.LoadPartitions(ctx, milvusclient.NewLoadPartitionsOption("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}

state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}
fmt.Println(state)
```

#### 2.5 释放分区（ReleasePartitions）

**Python**：
```python
client.release_partitions(
    collection_name="my_collection",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="my_collection",
    partition_name="partitionA"
)
print(res)
# Output: {"state": "<LoadState: NotLoaded>"}
```

**Java**：
```java
ReleasePartitionsReq releasePartitionsReq = ReleasePartitionsReq.builder()
        .collectionName("my_collection")
        .partitionNames(Collections.singletonList("partitionA"))
        .build();

client.releasePartitions(releasePartitionsReq);

GetLoadStateReq getLoadStateReq = GetLoadStateReq.builder()
        .collectionName("my_collection")
        .partitionName("partitionA")
        .build();

Boolean getLoadStateRes = client.getLoadState(getLoadStateReq);
System.out.println(getLoadStateRes);
// Output: False
```

**JavaScript**：
```javascript
await client.releasePartitions({
    collection_name: "my_collection",
    partition_names: ["partitionA"]
})

res = await client.getLoadState({
    collection_name: "my_collection",
    partition_name: "partitionA"
})
console.log(res)
// Output: LoadStateNotLoaded
```

**Go**：
```go
err = client.ReleasePartitions(ctx, milvusclient.NewReleasePartitionsOptions("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}

state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption("my_collection", "partitionA"))
if err != nil {
    fmt.Println(err.Error())
}
fmt.Println(state)
```

### 3. Partition Key 隔离特性（Milvus 2.6 新特性）

**功能**：通过启用 Partition Key 隔离，可以提升搜索性能。当前仅支持 HNSW 索引类型。启用后，搜索必须在过滤器中包含单个特定的 Partition Key 值。

**Python**：
```python
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    properties={"partitionkey.isolation": True}
)
```

**Java**：
```java
Map<String, String> properties = new HashMap<>();
properties.put("partitionkey.isolation", "true");

CreateCollectionReq createCollectionReq = CreateCollectionReq.builder()
        .collectionName("my_collection")
        .collectionSchema(schema)
        .properties(properties)
        .build();
client.createCollection(createCollectionReq);
```

**Go**：
```go
err = client.CreateCollection(ctx,
    milvusclient.NewCreateCollectionOption("my_collection", schema).
        WithProperty("partitionkey.isolation", true))
if err != nil {
    fmt.Println(err.Error())
}
```

**JavaScript**：
```javascript
res = await client.alterCollection({
    collection_name: "my_collection",
    properties: {
        "partitionkey.isolation": true
    }
})
```

### 4. Partition 最佳实践

#### 4.1 Partition 搜索和加载

**来源**：Product FAQ

**关键信息**：
- 当指定分区进行搜索时，Milvus 只搜索指定的分区，Collection 大小不会影响查询性能
- 是否需要加载整个 Collection 取决于搜索需要什么数据
- 所有可能出现在搜索结果中的分区都必须在搜索前加载
- 如果只想搜索特定分区，调用 `load_partition()` 加载目标分区，然后在 `search()` 方法中指定分区
- 如果想搜索所有分区，调用 `load_collection()` 加载整个 Collection（包括所有分区）
- 如果在搜索前未加载 Collection 或特定分区，Milvus 会返回错误

#### 4.2 Partition 定义

**来源**：Manage Partitions

**关键信息**：
- Partition 是 Collection 的子集
- 每个 Partition 与其父 Collection 共享相同的数据结构，但只包含 Collection 中的一部分数据
- 创建 Collection 时，Milvus 也会在 Collection 中创建一个名为 **_default** 的分区
- 如果不添加其他分区，所有插入 Collection 的实体都会进入默认分区，所有搜索和查询也会在默认分区内进行
- 可以添加更多分区，并根据特定标准将实体插入其中
- 然后可以将搜索和查询限制在特定分区内，提升搜索性能
- 一个 Collection 最多可以有 **1,024 个分区**

#### 4.3 Partition 在 Schema 设计中的应用

**来源**：Schema Design > Partitioning

**关键信息**：
- 为了加速搜索，可以选择性地启用分区
- 通过指定特定的标量字段进行分区，并在搜索期间基于该字段指定过滤条件，可以有效地将搜索范围限制在相关分区内
- 这种方法通过减少搜索域显著提升检索操作的效率

### 5. Partition 与 Collection 的关系

**来源**：Manage Collections

**关键信息**：
- Partitions 是 Collection 的子集，与其父 Collection 共享相同的字段集
- 每个 Partition 包含一部分实体
- 通过将实体分配到不同的分区，可以创建实体组
- 可以在特定分区中进行搜索和查询，让 Milvus 忽略其他分区中的实体，提升搜索效率

## 核心概念总结

### 1. Partition 的两种使用方式

#### 方式 1：手动分区管理
- 使用 `create_partition()` 手动创建分区
- 插入数据时指定分区名称
- 搜索时指定分区名称

#### 方式 2：Partition Key 自动分区（Milvus 2.6 推荐）
- 在 Schema 中标记字段为 `is_partition_key=True`
- Milvus 自动根据该字段的值进行分区
- 无需手动创建分区，无需在插入和搜索时指定分区名称

### 2. Partition 的核心操作

1. **创建**：`create_partition()`
2. **列出**：`list_partitions()`
3. **检查**：`has_partition()`
4. **加载**：`load_partitions()`
5. **释放**：`release_partitions()`
6. **删除**：`drop_partition()`（未在示例中展示）

### 3. Partition 的性能优化

1. **Partition Key 隔离**：启用 `partitionkey.isolation` 属性，提升搜索性能（仅支持 HNSW 索引）
2. **分区搜索**：在搜索时指定分区，减少搜索范围
3. **分区加载**：只加载需要搜索的分区，节省内存

### 4. Partition 的限制

1. **最大分区数**：一个 Collection 最多可以有 **1,024 个分区**
2. **Partition Key 隔离限制**：当前仅支持 HNSW 索引类型
3. **加载要求**：搜索前必须加载相关分区，否则会返回错误

## 需要进一步调研的技术点

1. **Partition Key 的哈希算法**：如何将 Partition Key 值映射到具体的分区？
2. **Partition 数量对性能的影响**：分区数量与检索性能的关系？
3. **100K Collections 支持**：Milvus 2.6 如何支持 100K collections？与分区的关系？
4. **Partition 的存储机制**：分区数据在磁盘上如何组织？
5. **Partition Key 隔离的实现原理**：为什么只支持 HNSW 索引？
