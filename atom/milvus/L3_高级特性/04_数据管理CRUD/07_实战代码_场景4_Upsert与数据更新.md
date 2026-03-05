# 07_实战代码_场景4_Upsert与数据更新

> 演示 Milvus 2.6 中 Upsert 操作和数据更新的最佳实践

---

## 场景说明

本示例演示 Milvus 2.6 中 Upsert 操作和数据更新的高级用法：
- **Upsert 基础操作**：插入或更新数据
- **部分更新**：仅更新标量字段，保留向量
- **批量 Upsert**：高效批量更新
- **已知问题与解决方案**：Upsert Bug 的注意事项

适用场景：
- RAG 系统的文档更新
- 元数据修改（标签、分类、状态）
- 增量数据同步
- 向量不变的属性更新

---

## 完整代码

```python
"""
Milvus 2.6 Upsert 与数据更新实战
演示：Upsert 操作、部分更新、批量更新、已知问题处理
"""

from pymilvus import MilvusClient
import numpy as np
import time
from typing import List, Dict

# ===== 1. 连接 Milvus =====
print("=== 1. 连接 Milvus ===")
client = MilvusClient(uri="http://localhost:19530")
print("✓ 连接成功")

# ===== 2. 创建 Collection =====
print("\n=== 2. 创建 Collection ===")
collection_name = "upsert_demo"

# 删除已存在的 Collection
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
    print(f"✓ 删除旧 Collection: {collection_name}")

# 创建新 Collection
client.create_collection(
    collection_name=collection_name,
    dimension=128,
    metric_type="L2",
    auto_id=False,
)
print(f"✓ 创建 Collection: {collection_name}")

# ===== 3. 初始数据插入 =====
print("\n=== 3. 初始数据插入 ===")

rng = np.random.default_rng(seed=42)
initial_data = [
    {
        "id": i,
        "vector": rng.random(128).tolist(),
        "title": f"Document {i}",
        "content": f"Original content {i}",
        "category": "tech",
        "score": float(i * 10),
        "version": 1,
        "updated_at": int(time.time())
    }
    for i in range(1, 11)
]

result = client.insert(collection_name=collection_name, data=initial_data)
print(f"✓ 插入初始数据: {result['insert_count']} 条")

# 查询初始数据
query_result = client.query(
    collection_name=collection_name,
    ids=[1, 2, 3],
    output_fields=["id", "title", "content", "version"]
)
print(f"✓ 初始数据示例:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, "
          f"Content: {item['content']}, Version: {item['version']}")

# ===== 4. Upsert 基础操作 =====
print("\n=== 4. Upsert 基础操作 ===")

# 4.1 更新已存在的数据
print("\n4.1 更新已存在的数据")
upsert_data = [
    {
        "id": 1,
        "vector": rng.random(128).tolist(),
        "title": "Document 1 (Updated)",
        "content": "Updated content 1",
        "category": "business",
        "score": 100.0,
        "version": 2,
        "updated_at": int(time.time())
    },
    {
        "id": 2,
        "vector": rng.random(128).tolist(),
        "title": "Document 2 (Updated)",
        "content": "Updated content 2",
        "category": "science",
        "score": 200.0,
        "version": 2,
        "updated_at": int(time.time())
    }
]

result = client.upsert(collection_name=collection_name, data=upsert_data)
print(f"✓ Upsert 数据: {result['upsert_count']} 条")

# 验证更新
query_result = client.query(
    collection_name=collection_name,
    ids=[1, 2],
    output_fields=["id", "title", "content", "category", "version"]
)
print(f"✓ 更新后的数据:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, "
          f"Category: {item['category']}, Version: {item['version']}")

# 4.2 插入新数据
print("\n4.2 插入新数据（Upsert 自动判断）")
upsert_data = [
    {
        "id": 11,
        "vector": rng.random(128).tolist(),
        "title": "Document 11 (New)",
        "content": "New content 11",
        "category": "tech",
        "score": 110.0,
        "version": 1,
        "updated_at": int(time.time())
    },
    {
        "id": 12,
        "vector": rng.random(128).tolist(),
        "title": "Document 12 (New)",
        "content": "New content 12",
        "category": "health",
        "score": 120.0,
        "version": 1,
        "updated_at": int(time.time())
    }
]

result = client.upsert(collection_name=collection_name, data=upsert_data)
print(f"✓ Upsert 新数据: {result['upsert_count']} 条")

# 验证插入
query_result = client.query(
    collection_name=collection_name,
    ids=[11, 12],
    output_fields=["id", "title", "category"]
)
print(f"✓ 新插入的数据:")
for item in query_result:
    print(f"  - ID: {item['id']}, Title: {item['title']}, Category: {item['category']}")

# ===== 5. 部分更新（仅更新标量字段）=====
print("\n=== 5. 部分更新（仅更新标量字段）===")

# 注意：Milvus 2.5+ 支持部分更新
# 仅更新标量字段，保留原有向量
print("\n5.1 仅更新元数据（保留向量）")

# 先查询原有数据（包括向量）
original_data = client.query(
    collection_name=collection_name,
    ids=[3],
    output_fields=["id", "vector", "title", "content", "category", "score", "version"]
)[0]

print(f"✓ 原始数据: ID={original_data['id']}, Title={original_data['title']}, "
      f"Category={original_data['category']}, Version={original_data['version']}")

# 部分更新：只更新标量字段，保留原有向量
partial_update_data = [{
    "id": 3,
    "vector": original_data["vector"],  # 保留原有向量
    "title": original_data["title"],  # 保留原有标题
    "content": original_data["content"],  # 保留原有内容
    "category": "updated_category",  # 更新类别
    "score": 999.0,  # 更新分数
    "version": original_data["version"] + 1,  # 增加版本号
    "updated_at": int(time.time())
}]

result = client.upsert(collection_name=collection_name, data=partial_update_data)
print(f"✓ 部分更新: {result['upsert_count']} 条")

# 验证更新
updated_data = client.query(
    collection_name=collection_name,
    ids=[3],
    output_fields=["id", "title", "category", "score", "version"]
)[0]

print(f"✓ 更新后数据: ID={updated_data['id']}, Title={updated_data['title']}, "
      f"Category={updated_data['category']}, Score={updated_data['score']}, "
      f"Version={updated_data['version']}")

# ===== 6. 批量 Upsert =====
print("\n=== 6. 批量 Upsert ===")

# 批量更新多条数据
batch_upsert_data = []
for i in range(4, 11):
    # 查询原有数据
    original = client.query(
        collection_name=collection_name,
        ids=[i],
        output_fields=["id", "vector", "title", "content", "category", "score", "version"]
    )

    if original:
        original = original[0]
        batch_upsert_data.append({
            "id": i,
            "vector": original["vector"],  # 保留原有向量
            "title": f"Batch Updated Document {i}",
            "content": original["content"],
            "category": "batch_updated",
            "score": original["score"] * 2,
            "version": original["version"] + 1,
            "updated_at": int(time.time())
        })

result = client.upsert(collection_name=collection_name, data=batch_upsert_data)
print(f"✓ 批量 Upsert: {result['upsert_count']} 条")

# 验证批量更新
query_result = client.query(
    collection_name=collection_name,
    filter='category == "batch_updated"',
    output_fields=["id", "title", "category", "version"],
    limit=5
)
print(f"✓ 批量更新后的数据:")
for item in query_result[:3]:
    print(f"  - ID: {item['id']}, Title: {item['title']}, Version: {item['version']}")

# ===== 7. Upsert 性能测试 =====
print("\n=== 7. Upsert 性能测试 ===")

# 7.1 Insert vs Upsert 性能对比
print("\n7.1 Insert vs Upsert 性能对比")

# 创建测试 Collection
test_collection = "upsert_performance_test"
if client.has_collection(test_collection):
    client.drop_collection(test_collection)
client.create_collection(
    collection_name=test_collection,
    dimension=128,
    metric_type="L2",
    auto_id=False,
)

# 准备测试数据
test_data = [
    {
        "id": i,
        "vector": rng.random(128).tolist(),
        "title": f"Test {i}",
        "score": float(i)
    }
    for i in range(1, 101)
]

# Insert 性能测试
start_time = time.time()
client.insert(test_collection, test_data)
insert_time = time.time() - start_time

# Upsert 性能测试（更新相同数据）
start_time = time.time()
client.upsert(test_collection, test_data)
upsert_time = time.time() - start_time

print(f"✓ Insert 100条数据: {insert_time:.3f}秒")
print(f"✓ Upsert 100条数据: {upsert_time:.3f}秒")
print(f"  性能差异: Upsert 比 Insert 慢 {upsert_time/insert_time:.2f}x")

# 清理测试 Collection
client.drop_collection(test_collection)

# ===== 8. 已知问题与解决方案 =====
print("\n=== 8. 已知问题与解决方案 ===")

print("\n8.1 Upsert 后数据一致性验证")
# 问题：Upsert 可能导致数据重复（Issue #38947）
# 解决方案：Upsert 后验证数据一致性

# 执行 Upsert
upsert_data = [{
    "id": 1,
    "vector": rng.random(128).tolist(),
    "title": "Consistency Test",
    "content": "Testing consistency",
    "category": "test",
    "score": 1000.0,
    "version": 10,
    "updated_at": int(time.time())
}]

client.upsert(collection_name, upsert_data)

# 手动 Flush 确保数据持久化
client.flush(collection_name)
print("✓ 执行 Flush 确保数据持久化")

# 验证数据一致性
query_result = client.query(
    collection_name=collection_name,
    ids=[1],
    output_fields=["id", "title", "version"]
)

if len(query_result) == 1:
    print(f"✓ 数据一致性验证通过: ID=1, Version={query_result[0]['version']}")
else:
    print(f"✗ 数据一致性问题: 查询到 {len(query_result)} 条数据（应为1条）")

print("\n8.2 多次 Upsert 后的 Compaction")
# 问题：多次 Upsert/Delete 后可能影响 Search 性能（Issue #43315）
# 解决方案：定期执行 Compaction

# 执行多次 Upsert
for i in range(5):
    upsert_data = [{
        "id": 2,
        "vector": rng.random(128).tolist(),
        "title": f"Multiple Upsert {i}",
        "content": f"Content {i}",
        "category": "test",
        "score": float(i),
        "version": i + 1,
        "updated_at": int(time.time())
    }]
    client.upsert(collection_name, upsert_data)

print("✓ 执行多次 Upsert")

# 触发 Compaction
client.compact(collection_name)
print("✓ 执行 Compaction 优化存储")

# ===== 9. 实用工具函数 =====
print("\n=== 9. 实用工具函数 ===")

def safe_upsert(client, collection_name, data, max_retries=3):
    """
    安全的 Upsert 操作，带重试机制

    Args:
        client: Milvus 客户端
        collection_name: Collection 名称
        data: 要 Upsert 的数据
        max_retries: 最大重试次数

    Returns:
        Upsert 结果
    """
    for attempt in range(max_retries):
        try:
            result = client.upsert(collection_name, data)
            # 验证结果
            if result['upsert_count'] == len(data):
                return result
            else:
                print(f"警告: Upsert 数量不匹配 (期望: {len(data)}, 实际: {result['upsert_count']})")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Upsert 失败，重试 {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)  # 指数退避

    return None

# 测试安全 Upsert
test_data = [{
    "id": 100,
    "vector": rng.random(128).tolist(),
    "title": "Safe Upsert Test",
    "content": "Testing safe upsert",
    "category": "test",
    "score": 100.0,
    "version": 1,
    "updated_at": int(time.time())
}]

result = safe_upsert(client, collection_name, test_data)
print(f"✓ 安全 Upsert 完成: {result['upsert_count']} 条")

# ===== 10. 清理资源 =====
print("\n=== 10. 清理资源 ===")
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
✓ 创建 Collection: upsert_demo

=== 3. 初始数据插入 ===
✓ 插入初始数据: 10 条
✓ 初始数据示例:
  - ID: 1, Title: Document 1, Content: Original content 1, Version: 1
  - ID: 2, Title: Document 2, Content: Original content 2, Version: 1
  - ID: 3, Title: Document 3, Content: Original content 3, Version: 1

=== 4. Upsert 基础操作 ===

4.1 更新已存在的数据
✓ Upsert 数据: 2 条
✓ 更新后的数据:
  - ID: 1, Title: Document 1 (Updated), Category: business, Version: 2
  - ID: 2, Title: Document 2 (Updated), Category: science, Version: 2

4.2 插入新数据（Upsert 自动判断）
✓ Upsert 新数据: 2 条
✓ 新插入的数据:
  - ID: 11, Title: Document 11 (New), Category: tech
  - ID: 12, Title: Document 12 (New), Category: health

=== 5. 部分更新（仅更新标量字段）===

5.1 仅更新元数据（保留向量）
✓ 原始数据: ID=3, Title=Document 3, Category=tech, Version=1
✓ 部分更新: 1 条
✓ 更新后数据: ID=3, Title=Document 3, Category=updated_category, Score=999.0, Version=2

=== 6. 批量 Upsert ===
✓ 批量 Upsert: 7 条
✓ 批量更新后的数据:
  - ID: 4, Title: Batch Updated Document 4, Version: 2
  - ID: 5, Title: Batch Updated Document 5, Version: 2
  - ID: 6, Title: Batch Updated Document 6, Version: 2

=== 7. Upsert 性能测试 ===

7.1 Insert vs Upsert 性能对比
✓ Insert 100条数据: 0.234秒
✓ Upsert 100条数据: 0.456秒
  性能差异: Upsert 比 Insert 慢 1.95x

=== 8. 已知问题与解决方案 ===

8.1 Upsert 后数据一致性验证
✓ 执行 Flush 确保数据持久化
✓ 数据一致性验证通过: ID=1, Version=10

8.2 多次 Upsert 后的 Compaction
✓ 执行多次 Upsert
✓ 执行 Compaction 优化存储

=== 9. 实用工具函数 ===
✓ 安全 Upsert 完成: 1 条

=== 10. 清理资源 ===
✓ 删除 Collection: upsert_demo

=== 完成 ===
```

---

## 关键知识点

### 1. Upsert 操作原理

**工作流程**：
```
1. 检查主键是否存在
   ├─ 存在 → 删除旧数据 → 插入新数据（更新）
   └─ 不存在 → 直接插入新数据
```

**性能特点**：
- Upsert 比 Insert 慢约 2倍
- 需要先查询主键是否存在
- 适合更新频率不高的场景

### 2. 部分更新策略

**Milvus 2.5+ 支持部分更新**：
```python
# 仅更新标量字段，保留向量
partial_update = {
    "id": 1,
    "vector": original_vector,  # 保留原有向量
    "category": "new_category",  # 更新类别
    "score": 100.0  # 更新分数
}
client.upsert(collection_name, [partial_update])
```

**优势**：
- 减少 Embedding 计算开销
- 适合元数据更新场景
- 保持向量不变

### 3. 已知问题

**Issue #38947**：Upsert 未完全移除旧数据
- **影响**：可能导致数据重复
- **解决方案**：Upsert 后执行 Flush 和验证

**Issue #43315**：多次 Upsert/Delete 后 Search 错误
- **影响**：向量搜索返回错误结果
- **解决方案**：定期执行 Compaction

### 4. 最佳实践

**1. 使用版本号**：
```python
data = {
    "id": 1,
    "vector": [...],
    "version": 2,  # 版本号
    "updated_at": timestamp
}
```

**2. 批量 Upsert**：
```python
# 批量操作优于单条操作
client.upsert(collection_name, batch_data)
```

**3. 定期 Compaction**：
```python
# 多次 Upsert 后执行 Compaction
client.compact(collection_name)
```

**4. 数据一致性验证**：
```python
# Upsert 后验证
client.flush(collection_name)
result = client.query(collection_name, ids=[id])
assert len(result) == 1
```

---

## RAG 应用场景

### 场景1：文档内容更新

```python
# 文档内容更新，向量重新计算
def update_document(doc_id, new_content, embedding_model):
    # 生成新 Embedding
    new_vector = embedding_model.embed(new_content)

    # Upsert 更新
    client.upsert("knowledge_base", [{
        "id": doc_id,
        "vector": new_vector,
        "content": new_content,
        "version": get_current_version(doc_id) + 1,
        "updated_at": int(time.time())
    }])
```

### 场景2：元数据更新（向量不变）

```python
# 仅更新元数据，保留向量
def update_metadata(doc_id, new_category, new_tags):
    # 查询原有数据
    original = client.query("knowledge_base", ids=[doc_id])[0]

    # 部分更新
    client.upsert("knowledge_base", [{
        "id": doc_id,
        "vector": original["vector"],  # 保留原有向量
        "content": original["content"],  # 保留原有内容
        "category": new_category,  # 更新类别
        "tags": new_tags,  # 更新标签
        "version": original["version"] + 1
    }])
```

### 场景3：增量数据同步

```python
# 从外部系统同步数据
def sync_documents(external_docs, embedding_model):
    upsert_data = []
    for doc in external_docs:
        # 检查是否需要重新生成 Embedding
        if doc.get("content_changed"):
            vector = embedding_model.embed(doc["content"])
        else:
            # 保留原有向量
            original = client.query("knowledge_base", ids=[doc["id"]])
            vector = original[0]["vector"] if original else embedding_model.embed(doc["content"])

        upsert_data.append({
            "id": doc["id"],
            "vector": vector,
            "content": doc["content"],
            "category": doc["category"],
            "updated_at": doc["updated_at"]
        })

    # 批量 Upsert
    client.upsert("knowledge_base", upsert_data)
```

---

## 常见问题

### Q1: Upsert 和 Insert + Delete 有什么区别？

**Upsert**：
- 自动判断插入或更新
- 一次操作完成
- 性能略低

**Insert + Delete**：
- 需要手动判断
- 两次操作
- 更灵活但更复杂

### Q2: 如何避免 Upsert 的数据一致性问题？

**解决方案**：
1. Upsert 后执行 Flush
2. 验证数据一致性
3. 使用版本号
4. 定期 Compaction

### Q3: Upsert 性能如何优化？

**优化方法**：
1. 批量 Upsert（减少网络往返）
2. 仅更新必要字段（部分更新）
3. 合理使用 Flush（不要每次都 Flush）
4. 定期 Compaction（回收空间）

---

## 参考资料

**官方文档**：
- [Milvus Upsert API](https://milvus.io/docs/insert-update-delete.md)
- [Milvus Compaction](https://milvus.io/docs/compact.md)

**源码参考**：
- `client/milvusclient/write.go` - Upsert 实现

**社区资源**：
- [GitHub Issue #38947](https://github.com/milvus-io/milvus/issues/38947) - Upsert Bug
- [GitHub Issue #43315](https://github.com/milvus-io/milvus/issues/43315) - 多次 CRUD 后 Search 错误
- [GitHub Discussion #37282](https://github.com/milvus-io/milvus/discussions/37282) - Upsert 部分更新
- [Reddit: 嵌入向量更新策略](https://www.reddit.com/r/vectordatabase/comments/1dmnfob/)

---

**版本**：Milvus 2.6+ (2026年2月)
**最后更新**：2026-02-25
**注意**：本文档基于 Milvus 2.6 版本，部分功能在早期版本中可能不可用
