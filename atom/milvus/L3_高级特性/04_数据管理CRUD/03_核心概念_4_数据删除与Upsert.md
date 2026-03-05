# 核心概念 4：数据删除与 Upsert

> 本文档详细讲解 Milvus 2.6 中的 Delete 和 Upsert 操作

---

## 概述

Delete 和 Upsert 是 Milvus 中的数据修改操作。Delete 用于删除不需要的数据，Upsert 用于插入或更新数据。理解这两个操作的原理和最佳实践对于管理 Milvus 数据至关重要。

---

## 1. Delete 操作

### 1.1 Delete 的定义

**Delete 是基于主键或标量过滤表达式删除实体的操作。**

### 1.2 Delete 的核心特征

1. **按主键删除**：通过主键 ID 列表删除指定实体
2. **按表达式删除**：通过标量字段过滤表达式批量删除
3. **逻辑删除**：删除操作是逻辑删除，不会立即释放存储空间
4. **需要 Compaction**：需要执行 Compaction 操作才能回收存储空间

### 1.3 Delete 的基本用法

#### 按主键删除

```python
from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")

# 按主键 ID 删除
result = client.delete(
    collection_name="my_collection",
    ids=[1, 2, 3]  # 主键 ID 列表
)

print(f"Deleted count: {result['delete_count']}")
```

#### 按表达式删除

```python
# 按过滤表达式删除
result = client.delete(
    collection_name="my_collection",
    filter="status == 'archived' and created_at < '2025-01-01'"
)

print(f"Deleted count: {result['delete_count']}")
```

### 1.4 Delete 的实现原理

**Delete 流程**：
```
用户请求
   ↓
Delete 操作（标记为删除）
   ↓
写入 WAL（Write-Ahead Log）
   ↓
更新 Bloom Filter（标记删除）
   ↓
返回删除数量
   ↓
Compaction（后台异步执行，回收空间）
```

**关键点**：
- Delete 是逻辑删除，不会立即释放存储空间
- 删除的数据在 Compaction 前仍然占用存储空间
- Search 和 Query 会自动过滤已删除的数据

### 1.5 Delete 的常见场景

#### 场景 1：删除过期数据

```python
# 删除过期数据
result = client.delete(
    collection_name="documents",
    filter="expired_at < '2026-01-01'"
)

print(f"Deleted expired documents: {result['delete_count']}")
```

#### 场景 2：删除测试数据

```python
# 删除测试数据
result = client.delete(
    collection_name="my_collection",
    filter="is_test == true"
)

print(f"Deleted test data: {result['delete_count']}")
```

#### 场景 3：删除全部数据

```python
# 方法 1：按表达式删除全部（适合保留 Collection 结构）
result = client.delete(
    collection_name="my_collection",
    filter="id >= 0"  # 删除所有 ID >= 0 的数据
)

# 方法 2：Drop Collection 后重新创建（适合完全清空）
client.drop_collection("my_collection")
client.create_collection("my_collection", schema=schema)
```

### 1.6 Delete 的注意事项

**1. 删除不会立即释放存储空间**

```python
# 删除数据
client.delete(collection_name="my_collection", ids=[1, 2, 3])

# 需要执行 Compaction 才能回收空间
from pymilvus import Collection
collection = Collection("my_collection")
collection.compact()  # 触发 Compaction
```

**2. 删除操作的一致性**

```python
# 使用 Strong Consistency 确保删除立即生效
result = client.delete(
    collection_name="my_collection",
    ids=[1, 2, 3]
)

# 立即查询（使用 Strong Consistency）
results = client.query(
    collection_name="my_collection",
    ids=[1, 2, 3],
    consistency_level="Strong"
)

print(f"Results after delete: {len(results)}")  # 应该为 0
```

**3. 批量删除的性能**

```python
# 批量删除比单条删除更高效
batch_ids = list(range(1, 10001))  # 10000 个 ID
result = client.delete(
    collection_name="my_collection",
    ids=batch_ids
)
```

---

## 2. Upsert 操作

### 2.1 Upsert 的定义

**Upsert 是插入或更新操作，如果主键存在则更新，不存在则插入。**

### 2.2 Upsert 的核心特征

1. **自动判断**：基于主键自动判断插入或更新
2. **原子操作**：Upsert 是原子操作，保证数据一致性
3. **部分更新**：Milvus 2.5+ 支持部分更新（仅更新指定字段）
4. **性能开销**：Upsert 性能开销比 Insert 大（需要先查询）

### 2.3 Upsert 的基本用法

#### 基础 Upsert

```python
import numpy as np

# Upsert 数据
data = [
    {"id": 1, "vector": np.random.rand(128).tolist(), "name": "Product A", "price": 99.99},
    {"id": 2, "vector": np.random.rand(128).tolist(), "name": "Product B", "price": 199.99},
    {"id": 3, "vector": np.random.rand(128).tolist(), "name": "Product C", "price": 299.99}
]

result = client.upsert(
    collection_name="my_collection",
    data=data
)

print(f"Upserted count: {result['upsert_count']}")
```

#### 部分更新（Milvus 2.5+）

```python
# 仅更新标量字段，不更新向量
data = [
    {"id": 1, "name": "Product A (Updated)", "price": 89.99},  # 不包含 vector 字段
    {"id": 2, "name": "Product B (Updated)", "price": 179.99}
]

result = client.upsert(
    collection_name="my_collection",
    data=data
)

print(f"Upserted count: {result['upsert_count']}")
```

### 2.4 Upsert 的实现原理

**Upsert 流程**：
```
用户请求
   ↓
查询主键是否存在
   ↓
存在 → 删除旧数据 → 插入新数据
   ↓
不存在 → 直接插入
   ↓
返回 Upsert 数量
```

**关键点**：
- Upsert 内部实现为 Delete + Insert
- 性能开销比 Insert 大（需要先查询）
- 支持部分更新（Milvus 2.5+）

### 2.5 Upsert 的常见场景

#### 场景 1：数据增量同步

```python
# 从外部数据源同步数据
external_data = fetch_data_from_external_source()

# Upsert 到 Milvus
for batch in batch_data(external_data, batch_size=1000):
    result = client.upsert(
        collection_name="my_collection",
        data=batch
    )
    print(f"Upserted {result['upsert_count']} records")
```

#### 场景 2：元数据更新

```python
# 仅更新元数据，不更新向量
metadata_updates = [
    {"id": 1, "status": "active", "updated_at": "2026-02-25"},
    {"id": 2, "status": "inactive", "updated_at": "2026-02-25"}
]

result = client.upsert(
    collection_name="my_collection",
    data=metadata_updates
)

print(f"Updated metadata for {result['upsert_count']} records")
```

#### 场景 3：文档版本控制

```python
# 更新文档版本
def update_document_version(doc_id, new_content, new_vector):
    data = [{
        "id": doc_id,
        "vector": new_vector,
        "content": new_content,
        "version": get_current_version(doc_id) + 1,
        "updated_at": datetime.now().isoformat()
    }]

    result = client.upsert(
        collection_name="documents",
        data=data
    )

    return result['upsert_count']
```

### 2.6 Upsert 的注意事项

**1. Upsert 性能开销**

```python
# Upsert 性能比 Insert 慢（需要先查询）
import time

# Insert 性能测试
start = time.time()
client.insert(collection_name="my_collection", data=data)
insert_time = time.time() - start

# Upsert 性能测试
start = time.time()
client.upsert(collection_name="my_collection", data=data)
upsert_time = time.time() - start

print(f"Insert time: {insert_time:.2f}s")
print(f"Upsert time: {upsert_time:.2f}s")
print(f"Upsert is {upsert_time / insert_time:.2f}x slower")
```

**2. Upsert 的已知 Bug（Milvus 2.5.x）**

根据 GitHub Issue #38947 和 #43315：
- Upsert 可能未完全移除旧数据
- 多次 Upsert + Delete 后 Search 可能返回错误结果
- 需要执行 Compaction 来解决

**解决方案**：
```python
from pymilvus import Collection

# 执行 Upsert
client.upsert(collection_name="my_collection", data=data)

# 执行 Compaction 确保数据一致性
collection = Collection("my_collection")
collection.compact()

# 等待 Compaction 完成
collection.wait_for_compaction_completed()
```

**3. 部分更新的限制**

```python
# ✅ 正确：仅更新标量字段
data = [{"id": 1, "name": "Updated Name", "price": 99.99}]
client.upsert(collection_name="my_collection", data=data)

# ❌ 错误：不能仅更新向量字段（必须同时提供标量字段）
data = [{"id": 1, "vector": new_vector}]  # 可能导致标量字段丢失
```

---

## 3. Delete vs Upsert 对比

| 特性 | Delete | Upsert |
|------|--------|--------|
| **操作类型** | 删除数据 | 插入或更新数据 |
| **性能** | 快速 | 较慢（需要先查询） |
| **存储空间** | 逻辑删除，需要 Compaction | 可能产生碎片 |
| **应用场景** | 数据清理、过期数据删除 | 数据同步、元数据更新 |
| **一致性** | 支持多种一致性级别 | 支持多种一致性级别 |
| **批量操作** | 支持 | 支持 |

---

## 4. 源码实现（Go 客户端）

根据 `client/milvusclient/write.go` 的源码分析：

### Delete 实现

```go
func (c *Client) Delete(ctx context.Context, option DeleteOption, callOptions ...grpc.CallOption) (DeleteResult, error) {
    startTime := time.Now()
    collectionName := option.CollectionName()
    result := DeleteResult{}

    // 获取 Collection 信息
    collection, err := c.getCollection(ctx, option.CollectionName())
    if err != nil {
        return result, err
    }

    // 构建 Delete 请求
    req, err := option.DeleteRequest(collection)
    if err != nil {
        return result, err
    }

    // 调用 gRPC 服务
    err = c.callService(func(milvusService milvuspb.MilvusServiceClient) error {
        resp, err := milvusService.Delete(ctx, req, callOptions...)
        err = merr.CheckRPCCall(resp, err)
        if err != nil {
            return err
        }

        result.DeleteCount = resp.GetDeleteCnt()
        return nil
    })

    // 记录操作
    c.recordOperation("Delete", collectionName, startTime, err)
    return result, err
}
```

### Upsert 实现

```go
func (c *Client) Upsert(ctx context.Context, option UpsertOption, callOptions ...grpc.CallOption) (UpsertResult, error) {
    startTime := time.Now()
    collectionName := option.CollectionName()
    result := UpsertResult{}

    // 使用 retryIfSchemaError 机制处理 schema 不匹配
    err := c.retryIfSchemaError(ctx, collectionName, func(ctx context.Context) (uint64, error) {
        collection, err := c.getCollection(ctx, option.CollectionName())
        if err != nil {
            return math.MaxUint64, err
        }

        req, err := option.UpsertRequest(collection)
        if err != nil {
            return collection.UpdateTimestamp, merr.WrapErrCollectionSchemaMisMatch(err)
        }

        return collection.UpdateTimestamp, c.callService(func(milvusService milvuspb.MilvusServiceClient) error {
            resp, err := milvusService.Upsert(ctx, req, callOptions...)
            err = merr.CheckRPCCall(resp, err)
            if err != nil {
                return err
            }

            result.UpsertCount = resp.GetUpsertCnt()
            result.IDs, err = column.IDColumns(collection.Schema, resp.GetIDs(), 0, -1)
            return err
        })
    })

    // 记录操作
    c.recordOperation("Upsert", collectionName, startTime, err)
    return result, err
}
```

**关键点**：
- Delete 直接调用 gRPC 服务，返回删除数量
- Upsert 使用 `retryIfSchemaError` 机制处理 schema 不匹配
- Upsert 返回 Upsert 数量和 ID 列

---

## 5. 最佳实践

### 5.1 Delete 最佳实践

**1. 批量删除优于单条删除**

```python
# ✅ 推荐：批量删除
batch_ids = list(range(1, 10001))
client.delete(collection_name="my_collection", ids=batch_ids)

# ❌ 不推荐：单条删除
for id in range(1, 10001):
    client.delete(collection_name="my_collection", ids=[id])
```

**2. 定期执行 Compaction**

```python
from pymilvus import Collection

collection = Collection("my_collection")

# 定期执行 Compaction（如每天一次）
collection.compact()
collection.wait_for_compaction_completed()

print("Compaction completed")
```

**3. 使用 Partition 管理数据生命周期**

```python
# 按时间分区
client.create_partition(collection_name="my_collection", partition_name="2026_01")
client.create_partition(collection_name="my_collection", partition_name="2026_02")

# 删除整个分区（比按表达式删除更高效）
client.drop_partition(collection_name="my_collection", partition_name="2026_01")
```

### 5.2 Upsert 最佳实践

**1. 仅在必要时使用 Upsert**

```python
# 如果确定数据不存在，使用 Insert（性能更好）
client.insert(collection_name="my_collection", data=new_data)

# 如果不确定数据是否存在，使用 Upsert
client.upsert(collection_name="my_collection", data=data)
```

**2. 利用部分更新优化性能**

```python
# ✅ 推荐：仅更新需要修改的字段
data = [{"id": 1, "status": "active"}]  # 不包含 vector 字段
client.upsert(collection_name="my_collection", data=data)

# ❌ 不推荐：更新所有字段（包括向量）
data = [{"id": 1, "vector": old_vector, "status": "active"}]
client.upsert(collection_name="my_collection", data=data)
```

**3. 处理 Upsert 的已知 Bug**

```python
from pymilvus import Collection

# 执行 Upsert
client.upsert(collection_name="my_collection", data=data)

# 执行 Compaction 确保数据一致性
collection = Collection("my_collection")
collection.compact()
collection.wait_for_compaction_completed()
```

---

## 6. 常见问题

### 6.1 Delete 后数据仍然可见

**原因**：一致性级别设置不当

**解决方案**：
```python
# 使用 Strong Consistency
client.delete(collection_name="my_collection", ids=[1, 2, 3])

results = client.query(
    collection_name="my_collection",
    ids=[1, 2, 3],
    consistency_level="Strong"  # 确保读取最新数据
)

print(f"Results: {len(results)}")  # 应该为 0
```

### 6.2 Upsert 后数据不一致

**原因**：Upsert 的已知 Bug（Milvus 2.5.x）

**解决方案**：
```python
from pymilvus import Collection

# 执行 Upsert
client.upsert(collection_name="my_collection", data=data)

# 执行 Compaction
collection = Collection("my_collection")
collection.compact()
collection.wait_for_compaction_completed()
```

### 6.3 Delete 不释放存储空间

**原因**：Delete 是逻辑删除，需要 Compaction

**解决方案**：
```python
from pymilvus import Collection

# 删除数据
client.delete(collection_name="my_collection", filter="status == 'archived'")

# 执行 Compaction 回收空间
collection = Collection("my_collection")
collection.compact()
collection.wait_for_compaction_completed()
```

---

## 7. 总结

**Delete 和 Upsert 是 Milvus 中的数据修改操作，具有以下特点：**

**Delete**：
1. **逻辑删除**：不会立即释放存储空间
2. **需要 Compaction**：需要执行 Compaction 才能回收空间
3. **支持批量操作**：批量删除性能远优于单条删除
4. **一致性保证**：支持多种一致性级别

**Upsert**：
1. **自动判断**：基于主键自动判断插入或更新
2. **部分更新**：Milvus 2.5+ 支持部分更新
3. **性能开销**：Upsert 性能开销比 Insert 大
4. **已知 Bug**：Milvus 2.5.x 存在已知 Bug，需要执行 Compaction

**最佳实践**：
- 批量操作优于单条操作
- 定期执行 Compaction
- 使用 Partition 管理数据生命周期
- 仅在必要时使用 Upsert
- 利用部分更新优化性能

---

## 参考资料

**源码分析**：
- `client/milvusclient/write.go` - Delete 和 Upsert 操作实现

**官方文档**：
- PyMilvus Delete API - Context7 文档
- PyMilvus Upsert API - Context7 文档

**社区讨论**：
- GitHub Issue #38947 - Upsert 未完全移除旧数据 Bug
- GitHub Issue #43315 - 多次 Upsert Delete 后 Search 错误
- GitHub Discussion #37282 - Upsert 仅修改元数据讨论
- Reddit - 嵌入向量更新策略
- Reddit - 删除操作实用方法

---

**版本**：v1.0
**创建时间**：2026-02-25
**适用版本**：Milvus 2.6+
