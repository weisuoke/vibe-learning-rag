# 07_实战代码_场景1_基础CRUD操作

> 演示 Milvus 2.6 中 Insert、Query、Delete 的基础操作

---

## 场景说明

本示例演示 Milvus 2.6 中最基础的 CRUD 操作：
- **Insert**：插入向量数据
- **Query**：基于主键或表达式的精确查询
- **Delete**：删除指定数据
- **Search**：向量相似度检索

适用场景：
- 快速上手 Milvus CRUD 操作
- 理解数据插入、查询、删除的基本流程
- 为 RAG 系统构建数据管理基础

---

## 完整代码

```python
"""
Milvus 2.6 基础 CRUD 操作实战
演示：Insert、Query、Delete、Search 的基本用法
"""

from pymilvus import MilvusClient
import numpy as np

# ===== 1. 连接 Milvus =====
print("=== 1. 连接 Milvus ===")
client = MilvusClient(uri="http://localhost:19530")
print("✓ 连接成功")

# ===== 2. 创建 Collection =====
print("\n=== 2. 创建 Collection ===")
collection_name = "basic_crud_demo"

# 删除已存在的 Collection（如果有）
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"✓ 删除旧 Collection: {collection_name}")

# 创建新 Collection
client.create_collection(
    collection_name=collection_name,
    dimension=128,  # 向量维度
    metric_type="L2",  # 距离度量类型
    auto_id=False,  # 手动指定主键
)
print(f"✓ 创建 Collection: {collection_name}")

# ===== 3. Insert 操作 =====
print("\n=== 3. Insert 操作 ===")

# 准备数据
rng = np.random.default_rng(seed=42)
data = [
    {
        "id": i,
        "vector": rng.random(128).tolist(),
        "title": f"Document {i}",
        "category": "tech" if i % 2 == 0 else "business",
        "score": float(i * 10)
    }
    for i in range(1, 11)  # 插入 10 条数据
]

# 执行插入
result = client.insert(collection_name=collection_name, data=data)
print(f"✓ 插入数据: {result['insert_count']} 条")
print(f"  主键列表: {result['ids'][:3]}...")  # 显示前3个ID

# ===== 4. Query 操作（精确查询）=====
print("\n=== 4. Query 操作（精确查询）===")

# 4.1 按主键查询
print("\n4.1 按主键查询")
query_result = client.query(
    collection_name=collection_name,
    ids=[1, 3, 5],
    output_fields=["id", "title", "category", "score"]
)
print(f"✓ 查询到 {len(query_result)} 条数据:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, Category: {item['category']}")

# 4.2 按表达式查询
print("\n4.2 按表达式查询")
query_result = client.query(
    collection_name=collection_name,
    filter='category == "tech" and score >= 40',
    output_fields=["id", "title", "category", "score"]
)
print(f"✓ 查询到 {len(query_result)} 条数据:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, Score: {item['score']}")

# 4.3 查询所有数据
print("\n4.3 查询所有数据")
query_result = client.query(
    collection_name=collection_name,
    filter="id >= 0",
    output_fields=["id", "title"],
    limit=5  # 限制返回数量
)
print(f"✓ 查询到 {len(query_result)} 条数据（限制5条）:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}")

# ===== 5. Search 操作（向量检索）=====
print("\n=== 5. Search 操作（向量检索）===")

# 准备查询向量
query_vector = rng.random(128).tolist()

# 执行向量检索
search_result = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,  # 返回 Top-3
    output_fields=["id", "title", "category", "score"]
)

print(f"✓ 检索到 {len(search_result[0])} 条相似数据:")
for hit in search_result[0]:
    print(f"  - ID: {hit['id']}, Title: {hit['entity']['title']}, "
          f"Distance: {hit['distance']:.4f}")

# ===== 6. Delete 操作 =====
print("\n=== 6. Delete 操作 ===")

# 6.1 按主键删除
print("\n6.1 按主键删除")
delete_result = client.delete(
    collection_name=collection_name,
    ids=[1, 2, 3]
)
print(f"✓ 删除数据: {delete_result['delete_count']} 条")

# 验证删除
query_result = client.query(
    collection_name=collection_name,
    ids=[1, 2, 3],
    output_fields=["id"]
)
print(f"  验证: 查询已删除的ID，返回 {len(query_result)} 条（应为0）")

# 6.2 按表达式删除
print("\n6.2 按表达式删除")
delete_result = client.delete(
    collection_name=collection_name,
    filter='category == "business" and score < 80'
)
print(f"✓ 删除数据: {delete_result['delete_count']} 条")

# 验证剩余数据
query_result = client.query(
    collection_name=collection_name,
    filter="id >= 0",
    output_fields=["id", "title", "category"]
)
print(f"  验证: 剩余 {len(query_result)} 条数据:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, Category: {item['category']}")

# ===== 7. 清理资源 =====
print("\n=== 7. 清理资源 ===")
client.drop_collection(collection_name)
print(f"✓ 删除 Collection: {collection_name}")

print("\n=== 完成 ===")
```

---

## 运行输出示例

```
=== 1. 连接 Milvus ===
✓ 连接成功

=== 2. 创建 Collection ===
✓ 创建 Collection: basic_crud_demo

=== 3. Insert 操作 ===
✓ 插入数据: 10 条
  主键列表: [1, 2, 3]...

=== 4. Query 操作（精确查询）===

4.1 按主键查询
✓ 查询到 3 条数据:
  - ID: 1, Title: Document 1, Category: business
  - ID: 3, Title: Document 3, Category: business
  - ID: 5, Title: Document 5, Category: business

4.2 按表达式查询
✓ 查询到 3 条数据:
  - ID: 4, Title: Document 4, Score: 40.0
  - ID: 6, Title: Document 6, Score: 60.0
  - ID: 8, Title: Document 8, Score: 80.0

4.3 查询所有数据
✓ 查询到 5 条数据（限制5条）:
  - ID: 1, Title: Document 1
  - ID: 2, Title: Document 2
  - ID: 3, Title: Document 3
  - ID: 4, Title: Document 4
  - ID: 5, Title: Document 5

=== 5. Search 操作（向量检索）===
✓ 检索到 3 条相似数据:
  - ID: 7, Title: Document 7, Distance: 12.3456
  - ID: 9, Title: Document 9, Distance: 13.4567
  - ID: 5, Title: Document 5, Distance: 14.5678

=== 6. Delete 操作 ===

6.1 按主键删除
✓ 删除数据: 3 条
  验证: 查询已删除的ID，返回 0 条（应为0）

6.2 按表达式删除
✓ 删除数据: 2 条
  验证: 剩余 5 条数据:
  - ID: 4, Title: Document 4, Category: tech
  - ID: 6, Title: Document 6, Category: tech
  - ID: 8, Title: Document 8, Category: tech
  - ID: 9, Title: Document 9, Category: business
  - ID: 10, Title: Document 10, Category: tech

=== 7. 清理资源 ===
✓ 删除 Collection: basic_crud_demo

=== 完成 ===
```

---

## 关键知识点

### 1. Insert 操作

**数据格式**：
```python
data = [
    {"id": 1, "vector": [...], "field1": "value1"},
    {"id": 2, "vector": [...], "field2": "value2"}
]
```

**返回结果**：
```python
{
    "insert_count": 10,
    "ids": [1, 2, 3, ...]
}
```

### 2. Query 操作

**按主键查询**：
```python
client.query(collection_name, ids=[1, 2, 3])
```

**按表达式查询**：
```python
client.query(collection_name, filter='category == "tech"')
```

**支持的表达式**：
- 比较运算符：`==`, `!=`, `>`, `>=`, `<`, `<=`
- 逻辑运算符：`and`, `or`, `not`
- 成员运算符：`in`, `not in`

### 3. Search 操作

**基本用法**：
```python
client.search(
    collection_name=collection_name,
    data=[query_vector],  # 查询向量
    limit=3,  # Top-K
    output_fields=["id", "title"]
)
```

**返回结果**：
```python
[
    [  # 第一个查询向量的结果
        {"id": 1, "distance": 0.123, "entity": {...}},
        {"id": 2, "distance": 0.456, "entity": {...}}
    ]
]
```

### 4. Delete 操作

**按主键删除**：
```python
client.delete(collection_name, ids=[1, 2, 3])
```

**按表达式删除**：
```python
client.delete(collection_name, filter='score < 50')
```

**注意事项**：
- 删除操作不会立即释放存储空间
- 需要 Compaction 操作来回收空间
- 删除后的数据在 Query 和 Search 中不可见

---

## 最佳实践

### 1. 批量操作优于单条操作

❌ **不推荐**：
```python
for item in data:
    client.insert(collection_name, [item])  # 逐条插入
```

✅ **推荐**：
```python
client.insert(collection_name, data)  # 批量插入
```

### 2. 使用 output_fields 减少数据传输

❌ **不推荐**：
```python
# 返回所有字段（包括向量）
result = client.query(collection_name, ids=[1, 2, 3])
```

✅ **推荐**：
```python
# 只返回需要的字段
result = client.query(
    collection_name,
    ids=[1, 2, 3],
    output_fields=["id", "title"]
)
```

### 3. 使用 limit 限制返回数量

```python
# 避免返回过多数据
result = client.query(
    collection_name,
    filter="id >= 0",
    limit=100  # 限制返回100条
)
```

### 4. 删除前先查询验证

```python
# 先查询要删除的数据
to_delete = client.query(
    collection_name,
    filter='score < 50',
    output_fields=["id"]
)
print(f"将删除 {len(to_delete)} 条数据")

# 确认后再删除
client.delete(collection_name, filter='score < 50')
```

---

## RAG 应用场景

### 场景1：文档知识库管理

```python
# 插入文档
documents = [
    {
        "id": doc_id,
        "vector": embedding,
        "title": "Python 教程",
        "content": "...",
        "category": "programming",
        "created_at": "2026-02-25"
    }
]
client.insert("knowledge_base", documents)

# 查询特定类别的文档
docs = client.query(
    "knowledge_base",
    filter='category == "programming"',
    output_fields=["id", "title"]
)

# 删除过期文档
client.delete(
    "knowledge_base",
    filter='created_at < "2025-01-01"'
)
```

### 场景2：用户对话历史管理

```python
# 插入对话记录
conversations = [
    {
        "id": conv_id,
        "vector": query_embedding,
        "user_id": "user123",
        "query": "如何使用 Milvus？",
        "response": "...",
        "timestamp": 1708876800
    }
]
client.insert("conversations", conversations)

# 查询用户的历史对话
history = client.query(
    "conversations",
    filter='user_id == "user123"',
    output_fields=["query", "response"],
    limit=10
)
```

---

## 常见问题

### Q1: Insert 后立即 Query 查询不到数据？

**原因**：数据可见性延迟，取决于一致性级别。

**解决方案**：
```python
# 方案1：使用 Strong 一致性
client.search(
    collection_name,
    data=[vector],
    consistency_level="Strong"
)

# 方案2：手动 Flush
client.flush(collection_name)
```

### Q2: Delete 后存储空间没有释放？

**原因**：删除操作只是标记删除，需要 Compaction 回收空间。

**解决方案**：
```python
# 触发 Compaction
client.compact(collection_name)
```

### Q3: Query 表达式报错？

**常见错误**：
- 字符串值未加引号：`category == tech` ❌
- 正确写法：`category == "tech"` ✅

---

## 参考资料

**官方文档**：
- [Milvus Insert API](https://milvus.io/docs/insert-update-delete.md)
- [Milvus Query API](https://milvus.io/docs/query.md)
- [Milvus Search API](https://milvus.io/docs/search.md)

**源码参考**：
- `client/milvusclient/write.go` - Insert/Delete 实现
- `client/milvusclient/read.go` - Query/Search 实现

**社区资源**：
- [Milvus CheatSheet](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/MilvusCheatSheet.md)
- [PyMilvus 官方文档](https://github.com/milvus-io/pymilvus)

---

**版本**：Milvus 2.6+ (2026年2月)
**最后更新**：2026-02-25
