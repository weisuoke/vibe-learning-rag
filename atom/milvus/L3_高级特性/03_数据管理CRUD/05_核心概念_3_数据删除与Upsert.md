# 核心概念 3: 数据删除与Upsert

> 深入理解 Milvus 的删除机制和 Upsert 操作，掌握数据更新的最佳实践

---

## 概述

Milvus 的数据删除和更新机制与传统数据库有很大不同：
- **删除是软删除**：只标记删除，需要 Compaction 才能释放空间
- **没有原生 Update**：需要使用 Delete + Insert 或 Upsert
- **Upsert 是原子操作**：存在则更新，不存在则插入

**本章内容**：
1. Delete：软删除机制
2. Compaction：空间回收
3. Upsert：原子更新
4. 数据更新策略
5. 性能优化
6. 在 RAG 系统中的应用

---

## 1. Delete：软删除机制

### 1.1 什么是软删除？

**定义**：软删除是指只标记数据为已删除，不立即释放存储空间。

**执行流程**：
```
delete() → 标记删除 → flush() → 持久化标记 → Compaction → 真正释放空间
```

### 1.2 基本用法

```python
from pymilvus import Collection

collection = Collection("my_collection")

# 按 ID 删除
collection.delete(expr="id in [1, 2, 3]")

# 按条件删除
collection.delete(expr="price < 10")

# 持久化删除
collection.flush()
```

### 1.3 删除表达式

**支持的表达式**：

```python
# 删除单个 ID
collection.delete(expr="id == 1")

# 删除多个 ID
collection.delete(expr="id in [1, 2, 3, 4, 5]")

# 按范围删除
collection.delete(expr="price > 100 and price < 200")

# 按条件删除
collection.delete(expr="category == 'obsolete'")

# 复杂条件
collection.delete(expr="(price < 10 or stock == 0) and category != 'important'")
```

### 1.4 删除的限制

**限制 1: 不能按向量相似度删除**

```python
# 错误：不能按向量相似度删除
# collection.delete(vector=[0.1, 0.2, 0.3])  # 不支持

# 正确：先 Search 找到 ID，再删除
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="embedding",
    limit=10
)
ids_to_delete = [hit.id for hit in results[0]]
collection.delete(expr=f"id in {ids_to_delete}")
```

**限制 2: 删除后数据仍占用空间**

```python
# 删除数据
collection.delete(expr="id in [1, 2, 3]")
collection.flush()

# 此时数据仍占用空间
print(collection.num_entities)  # 数量减少了
# 但磁盘空间并未释放

# 需要 Compaction 才能释放空间
collection.compact()
collection.wait_for_compaction_completed()
```

### 1.5 删除的性能特征

**性能模型**：
```
Delete 时间 = 扫描时间 + 标记时间
```

**性能数据**：
- 小批量删除（< 1000 条）：< 100ms
- 中批量删除（1000-10000 条）：100ms - 1s
- 大批量删除（> 10000 条）：> 1s

**优化策略**：
1. 批量删除（一次删除多条）
2. 使用分区（只删除指定分区）
3. 定期 Compaction（释放空间）

---

## 2. Compaction：空间回收

### 2.1 什么是 Compaction？

**定义**：Compaction 是合并 Segment 并清理已删除数据的过程。

**作用**：
1. 清理已删除的数据
2. 合并小 Segment
3. 释放存储空间
4. 优化查询性能

### 2.2 Compaction 的类型

**类型 1: 自动 Compaction**

Milvus 会自动触发 Compaction，触发条件：
- 删除比例 > 10%
- Segment 大小 > 512MB
- 定期触发（每小时）

**类型 2: 手动 Compaction**

```python
# 手动触发 Compaction
collection.compact()

# 等待 Compaction 完成
collection.wait_for_compaction_completed()

# 获取 Compaction 状态
state = collection.get_compaction_state()
print(f"Compaction 状态: {state}")
```

### 2.3 Compaction 的执行流程

```
1. 扫描 Segment
   ↓
2. 识别需要合并的 Segment
   ↓
3. 读取 Segment 数据
   ↓
4. 过滤已删除的数据
   ↓
5. 合并数据到新 Segment
   ↓
6. 删除旧 Segment
   ↓
7. 释放存储空间
```

### 2.4 Compaction 的性能影响

**性能特征**：
- Compaction 是后台任务，不阻塞前台操作
- Compaction 会占用 CPU 和磁盘 I/O
- Compaction 期间查询性能可能下降

**优化策略**：
1. 在低峰期执行 Compaction
2. 控制 Compaction 频率
3. 监控 Compaction 进度

### 2.5 定期 Compaction

```python
import schedule
import time

def compact_collection(collection):
    """执行 Compaction"""
    print("开始 Compaction...")
    collection.compact()
    collection.wait_for_compaction_completed()
    print("Compaction 完成")

# 每天凌晨 2 点执行
schedule.every().day.at("02:00").do(compact_collection, collection)

# 运行调度器
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 3. Upsert：原子更新

### 3.1 什么是 Upsert？

**定义**：Upsert = Update + Insert，存在则更新，不存在则插入。

**特点**：
- 原子操作（不会出现中间状态）
- 自动判断是更新还是插入
- 简化更新逻辑

### 3.2 基本用法

```python
import numpy as np

# 准备数据
data = [
    [1, 2, 3],  # id
    ["text_1", "text_2", "text_3"],  # text
    [np.random.rand(128).tolist() for _ in range(3)]  # embedding
]

# Upsert（存在则更新，不存在则插入）
collection.upsert(data)
collection.flush()
```

### 3.3 Upsert vs Delete + Insert

**传统方式：Delete + Insert**

```python
# 检查是否存在
existing = collection.query(expr="id in [1, 2, 3]")

# 删除旧数据
if existing:
    collection.delete(expr="id in [1, 2, 3]")

# 插入新数据
collection.insert(data)
collection.flush()
```

**Upsert 方式**

```python
# 一步完成
collection.upsert(data)
collection.flush()
```

**对比**：

| 特性 | Delete + Insert | Upsert |
|------|----------------|--------|
| **操作步骤** | 3 步 | 1 步 |
| **原子性** | 否 | 是 |
| **性能** | 慢 | 快 |
| **代码复杂度** | 高 | 低 |

### 3.4 Upsert 的使用场景

**场景 1: 数据更新**

```python
# 更新文档的 Embedding
def update_document_embedding(collection, doc_id, new_text):
    from sentence_transformers import SentenceTransformer

    # 生成新 Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    new_embedding = model.encode([new_text])[0].tolist()

    # Upsert
    data = [[doc_id], [new_text], [new_embedding]]
    collection.upsert(data)
    collection.flush()
```

**场景 2: 数据同步**

```python
# 从其他系统同步数据
def sync_from_database(collection, db_records):
    """从数据库同步数据到 Milvus"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 批量处理
    batch_size = 1000
    for i in range(0, len(db_records), batch_size):
        batch = db_records[i:i + batch_size]

        # 准备数据
        ids = [r["id"] for r in batch]
        texts = [r["text"] for r in batch]
        embeddings = model.encode(texts).tolist()

        # Upsert（存在则更新，不存在则插入）
        collection.upsert([ids, texts, embeddings])

    collection.flush()
```

**场景 3: 增量导入**

```python
# 增量导入新数据
def incremental_import(collection, new_documents):
    """增量导入文档"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 生成 Embedding
    texts = [doc["text"] for doc in new_documents]
    embeddings = model.encode(texts).tolist()

    # 准备数据
    ids = [doc["id"] for doc in new_documents]
    data = [ids, texts, embeddings]

    # Upsert（自动处理新增和更新）
    collection.upsert(data)
    collection.flush()
```

### 3.5 Upsert 的性能优化

**优化 1: 批量 Upsert**

```python
# 错误：单条 Upsert
for record in records:
    collection.upsert([[record["id"]], [record["text"]], [record["embedding"]]])
    collection.flush()

# 正确：批量 Upsert
ids = [r["id"] for r in records]
texts = [r["text"] for r in records]
embeddings = [r["embedding"] for r in records]
collection.upsert([ids, texts, embeddings])
collection.flush()
```

**优化 2: 减少 flush() 调用**

```python
# 错误：频繁 flush
for batch in batches:
    collection.upsert(batch)
    collection.flush()

# 正确：批量 flush
for batch in batches:
    collection.upsert(batch)
collection.flush()
```

---

## 4. 数据更新策略

### 4.1 更新策略对比

| 策略 | 实现方式 | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|---------|
| **Upsert** | `collection.upsert()` | 简单、原子 | 需要提供所有字段 | 常规更新 |
| **Delete + Insert** | `delete()` + `insert()` | 灵活 | 非原子、复杂 | 复杂更新 |
| **重建 Collection** | `drop()` + `create()` | 彻底 | 停机时间长 | 大规模重构 |

### 4.2 部分字段更新

**问题**：Milvus 不支持只更新部分字段。

```python
# 错误：不支持部分字段更新
# collection.update({"id": 1, "text": "new text"})  # 不存在

# 正确：必须提供所有字段
# 1. 先查询获取原始数据
result = collection.query(expr="id == 1", output_fields=["id", "text", "embedding"])
original = result[0]

# 2. 修改需要更新的字段
original["text"] = "new text"

# 3. Upsert 所有字段
collection.upsert([[original["id"]], [original["text"]], [original["embedding"]]])
collection.flush()
```

**封装为工具函数**：

```python
def update_fields(collection, id, updates):
    """更新指定字段"""
    # 1. 查询原始数据
    results = collection.query(
        expr=f"id == {id}",
        output_fields=["*"]
    )

    if not results:
        raise ValueError(f"ID {id} 不存在")

    original = results[0]

    # 2. 更新字段
    for field, value in updates.items():
        original[field] = value

    # 3. Upsert
    data = [[original[field] for field in schema_fields]]
    collection.upsert(data)
    collection.flush()

# 使用
update_fields(collection, id=1, updates={"text": "new text"})
```

### 4.3 批量更新

```python
def batch_update(collection, updates):
    """批量更新数据"""
    # updates = [{"id": 1, "text": "new text 1"}, {"id": 2, "text": "new text 2"}]

    # 1. 查询所有需要更新的数据
    ids = [u["id"] for u in updates]
    results = collection.query(
        expr=f"id in {ids}",
        output_fields=["*"]
    )

    # 2. 构建 ID 到数据的映射
    id_to_data = {r["id"]: r for r in results}

    # 3. 更新数据
    for update in updates:
        id = update["id"]
        if id in id_to_data:
            for field, value in update.items():
                if field != "id":
                    id_to_data[id][field] = value

    # 4. 准备 Upsert 数据
    updated_data = list(id_to_data.values())
    ids = [d["id"] for d in updated_data]
    texts = [d["text"] for d in updated_data]
    embeddings = [d["embedding"] for d in updated_data]

    # 5. Upsert
    collection.upsert([ids, texts, embeddings])
    collection.flush()

# 使用
batch_update(collection, [
    {"id": 1, "text": "new text 1"},
    {"id": 2, "text": "new text 2"}
])
```

### 4.4 条件更新

```python
def conditional_update(collection, condition, updates):
    """条件更新"""
    # 1. 查询符合条件的数据
    results = collection.query(
        expr=condition,
        output_fields=["*"]
    )

    if not results:
        print("没有符合条件的数据")
        return

    # 2. 更新数据
    for result in results:
        for field, value in updates.items():
            result[field] = value

    # 3. 准备 Upsert 数据
    ids = [r["id"] for r in results]
    texts = [r["text"] for r in results]
    embeddings = [r["embedding"] for r in results]

    # 4. Upsert
    collection.upsert([ids, texts, embeddings])
    collection.flush()

    print(f"已更新 {len(results)} 条数据")

# 使用
conditional_update(
    collection,
    condition="category == 'tech' and price < 100",
    updates={"category": "tech_sale"}
)
```

---

## 5. 性能优化

### 5.1 删除性能优化

**优化 1: 批量删除**

```python
# 错误：单条删除
for id in ids:
    collection.delete(expr=f"id == {id}")
    collection.flush()

# 正确：批量删除
collection.delete(expr=f"id in {ids}")
collection.flush()
```

**优化 2: 使用分区**

```python
# 不使用分区：扫描所有数据
collection.delete(expr="category == 'obsolete'")

# 使用分区：只扫描指定分区
collection.delete(
    expr="category == 'obsolete'",
    partition_name="partition_2023"
)
```

**优化 3: 定期 Compaction**

```python
# 删除大量数据后，立即 Compaction
collection.delete(expr="id in [1, 2, 3, ..., 10000]")
collection.flush()
collection.compact()
collection.wait_for_compaction_completed()
```

### 5.2 Upsert 性能优化

**优化 1: 批量 Upsert**

```python
# 批量大小：1000-10000
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    collection.upsert(batch)
collection.flush()
```

**优化 2: 并行 Upsert**

```python
from concurrent.futures import ThreadPoolExecutor

def upsert_batch(collection, batch):
    collection.upsert(batch)

def parallel_upsert(collection, batches, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(upsert_batch, collection, batch) for batch in batches]
        for future in futures:
            future.result()
    collection.flush()
```

### 5.3 Compaction 优化

**优化 1: 控制 Compaction 频率**

```python
# 配置 Compaction 策略
# 在 milvus.yaml 中配置
# dataCoord:
#   compaction:
#     enableAutoCompaction: true
#     minSegmentToMerge: 3
#     maxSegmentToMerge: 10
```

**优化 2: 监控 Compaction 进度**

```python
def monitor_compaction(collection):
    """监控 Compaction 进度"""
    import time

    collection.compact()

    while True:
        state = collection.get_compaction_state()
        print(f"Compaction 状态: {state}")

        if state.state == 3:  # Completed
            break

        time.sleep(5)

    print("Compaction 完成")
```

---

## 6. 在 RAG 系统中的应用

### 6.1 文档更新

```python
def update_document(collection, doc_id, new_text):
    """更新文档"""
    from sentence_transformers import SentenceTransformer

    # 生成新 Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    new_embedding = model.encode([new_text])[0].tolist()

    # Upsert
    data = [[doc_id], [new_text], [new_embedding]]
    collection.upsert(data)
    collection.flush()

    print(f"文档 {doc_id} 已更新")
```

### 6.2 文档删除

```python
def delete_documents(collection, doc_ids):
    """删除文档"""
    # 批量删除
    collection.delete(expr=f"id in {doc_ids}")
    collection.flush()

    print(f"已删除 {len(doc_ids)} 个文档")

    # 定期 Compaction
    collection.compact()
    collection.wait_for_compaction_completed()
```

### 6.3 知识库同步

```python
def sync_knowledge_base(collection, documents):
    """同步知识库"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 批量处理
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # 生成 Embedding
        texts = [doc["text"] for doc in batch]
        embeddings = model.encode(texts).tolist()

        # 准备数据
        ids = [doc["id"] for doc in batch]
        data = [ids, texts, embeddings]

        # Upsert（自动处理新增和更新）
        collection.upsert(data)

        print(f"已同步 {i + len(batch)}/{len(documents)} 个文档")

    collection.flush()
    print("知识库同步完成")
```

### 6.4 过期数据清理

```python
def cleanup_expired_documents(collection, days=30):
    """清理过期文档"""
    import time

    # 计算过期时间戳
    expired_timestamp = int(time.time()) - days * 24 * 60 * 60

    # 删除过期文档
    collection.delete(expr=f"timestamp < {expired_timestamp}")
    collection.flush()

    # Compaction 释放空间
    collection.compact()
    collection.wait_for_compaction_completed()

    print(f"已清理 {days} 天前的过期文档")
```

---

## 总结

### 核心要点

1. **Delete 是软删除**：只标记删除，需要 Compaction 释放空间
2. **Compaction 是必须的**：定期执行 Compaction 释放空间
3. **Upsert 是原子操作**：存在则更新，不存在则插入
4. **没有部分字段更新**：必须提供所有字段
5. **批量操作提升性能**：批量删除、批量 Upsert

### 最佳实践

1. **使用 Upsert 简化更新**：`collection.upsert(data)`
2. **批量删除提升性能**：`collection.delete(expr=f"id in {ids}")`
3. **定期 Compaction 释放空间**：每天凌晨执行
4. **封装更新工具函数**：简化部分字段更新
5. **监控 Compaction 进度**：确保空间释放

### 性能优化清单

- [ ] 使用批量删除（一次删除多条）
- [ ] 使用批量 Upsert（batch_size >= 1000）
- [ ] 批量 flush（不要每次操作都 flush）
- [ ] 使用分区优化删除性能
- [ ] 定期执行 Compaction（每天凌晨）
- [ ] 监控 Compaction 进度
- [ ] 封装更新工具函数

---

**下一步**: 学习 [09_实战代码_场景1_基础CRUD操作.md](./09_实战代码_场景1_基础CRUD操作.md) 实践完整的 CRUD 操作
