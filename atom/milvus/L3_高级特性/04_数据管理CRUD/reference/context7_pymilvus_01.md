---
type: context7_documentation
library: pymilvus
version: master (2026-01-22)
fetched_at: 2026-02-25
knowledge_point: 04_数据管理CRUD
context7_query: CRUD operations Insert Delete Upsert Query Search APIs
---

# Context7 文档：PyMilvus

## 文档来源
- 库名称：pymilvus
- 版本：master (2026-01-22)
- 官方文档链接：https://github.com/milvus-io/pymilvus
- Context7 Library ID：`/milvus-io/pymilvus`
- Total Snippets：198
- Trust Score：9.8
- Benchmark Score：90.3

## 关键信息提取

### 1. Upsert API

**端点**：`POST /collection/{collection_name}/upsert`

**描述**：Upserts data into the collection. If an entity with the same primary key already exists, it will be updated; otherwise, it will be inserted.

**参数**：
- **Path Parameters**：
  - `collection_name` (string) - Required - The name of the collection to upsert data into.
- **Request Body**：
  - `data` (list[object]) - Required - A list of entities to upsert. Each object represents an entity with field values.

**请求示例**：
```json
{
  "data": [
    {"primary_field": "id1", "vector_field": [0.1, 0.2], "other_field": "value1"},
    {"primary_field": "id2", "vector_field": [0.3, 0.4], "other_field": "value2"}
  ]
}
```

**响应**：
- **Success Response (200)**：
  - `upsert_count` (integer) - The number of entities successfully upserted.

**响应示例**：
```json
{
  "upsert_count": 2
}
```

### 2. Query API

**端点**：`POST /collection/{collection_name}/query`

**描述**：Queries the collection with a set of criteria, returning entities that match the specified conditions.

**参数**：
- **Path Parameters**：
  - `collection_name` (string) - Required - The name of the collection to query.
- **Query Parameters**：
  - `expr` (string) - Required - The expression to filter query results.
  - `output_fields` (list[string]) - Optional - A list of field names to include in the output.
  - `limit` (integer) - Optional - The maximum number of results to return.

**响应**：
- **Success Response (200)**：
  - `results` (list[object]) - A list of entities matching the query criteria.

**响应示例**：
```json
{
  "results": [
    {"primary_field": "id1", "vector_field": [0.1, 0.2], "other_field": "value1"},
    {"primary_field": "id3", "vector_field": [0.5, 0.6], "other_field": "value3"}
  ]
}
```

### 3. Search API

**端点**：`POST /collection/{collection_name}/search`

**描述**：Performs a vector similarity search on the collection with optional boolean expression as filters.

**参数**：
- **Path Parameters**：
  - `collection_name` (string) - Required - The name of the collection to search within.
- **Query Parameters**：
  - `limit` (integer) - Required - The maximum number of results to return.
  - `expr` (string) - Optional - A boolean expression to filter search results.
  - `output_fields` (list[string]) - Optional - A list of field names to include in the output.
  - `consistency_level` (string) - Optional - The consistency level for the search.
- **Request Body**：
  - `data` (list[list[float]]) - Required - The vector(s) to search for.
  - `search_params` (object) - Required - Parameters specific to the search algorithm.
    - `metric_type` (string) - Required - The metric type used for similarity calculation.
    - `params` (object) - Required - Algorithm-specific parameters (e.g., ef for ANNS).

**请求示例**：
```json
{
  "data": [[0.1, 0.2, 0.3]],
  "search_params": {
    "metric_type": "L2",
    "params": {"ef": 10}
  }
}
```

**响应**：
- **Success Response (200)**：
  - `results` (object) - Contains the search results.
    - `ids` (list[string]) - The IDs of the matching entities.
    - `scores` (list[float]) - The similarity scores for the matching entities.
    - `entities` (object) - The output fields for the matching entities.

**响应示例**：
```json
{
  "results": {
    "ids": ["id1", "id2"],
    "scores": [0.9, 0.8],
    "entities": {
      "output_field1": ["value1", "value2"]
    }
  }
}
```

### 4. Python 代码示例

#### Upsert 操作
```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")
client.create_collection("products", dimension=64)

rng = np.random.default_rng(seed=42)

# Initial insert
client.insert("products", [
    {"id": 1, "vector": rng.random(64).tolist(), "name": "Widget", "price": 9.99},
    {"id": 2, "vector": rng.random(64).tolist(), "name": "Gadget", "price": 19.99}
])

# Upsert - updates existing record (id=1) and inserts new (id=3)
result = client.upsert("products", [
    {"id": 1, "vector": rng.random(64).tolist(), "name": "Widget Pro", "price": 14.99},
    {"id": 3, "vector": rng.random(64).tolist(), "name": "Doohickey", "price": 29.99}
])
print(f"Upserted: {result}")  # {'upsert_count': 2}

# Verify updates
results = client.query("products", ids=[1, 2, 3])
for r in results:
    print(r)

client.drop_collection("products")
```

**关键特性**：
- Upsert 支持部分更新（partial updates）
- 允许修改特定字段while preserving others
- 基于主键自动判断插入或更新

#### Delete 操作
```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")
client.create_collection("inventory", dimension=32)

rng = np.random.default_rng(seed=42)
items = [
    {"id": i, "vector": rng.random(32).tolist(), "status": "active" if i % 3 != 0 else "archived"}
    for i in range(100)
]
client.insert("inventory", items)

# Delete by primary key IDs
result = client.delete("inventory", ids=[1, 2, 3])
print(f"Deleted: {result}")  # {'delete_count': 3}

# Delete by filter expression
result = client.delete("inventory", filter="status == 'archived'")
print(f"Deleted archived items: {result}")

# Verify deletion
remaining = client.query("inventory", filter="id < 10")
print(f"Remaining records with id < 10: {len(remaining)}")

client.drop_collection("inventory")
```

**关键特性**：
- 支持按主键 ID 删除
- 支持按过滤表达式批量删除
- 返回删除数量

### 5. API 设计模式

#### MilvusClient 简化接口
从代码示例中可以看到，PyMilvus 提供了简化的 `MilvusClient` 接口：

```python
client = MilvusClient("http://localhost:19530")

# 简化的 CRUD 操作
client.insert(collection_name, data)
client.upsert(collection_name, data)
client.delete(collection_name, ids=[], filter="")
client.query(collection_name, ids=[], filter="")
client.search(collection_name, data, search_params)
```

**优势**：
- 更简洁的 API 设计
- 自动处理连接管理
- 统一的错误处理
- 支持链式调用

#### 数据格式
所有操作都使用统一的数据格式：
```python
data = [
    {"id": 1, "vector": [0.1, 0.2], "field1": "value1"},
    {"id": 2, "vector": [0.3, 0.4], "field2": "value2"}
]
```

**特点**：
- 字典列表格式
- 自动类型推断
- 支持动态字段

### 6. 与 Go 客户端的对比

| 特性 | Go 客户端 | Python 客户端 |
|------|-----------|---------------|
| **API 风格** | Option 模式 | 简化接口 |
| **数据格式** | Column-based | Row-based (字典列表) |
| **类型安全** | 编译时检查 | 运行时检查 |
| **错误处理** | 显式错误返回 | 异常机制 |
| **性能** | 更高 | 略低但更易用 |

### 7. 最佳实践

#### Upsert 使用场景
- 数据更新频繁的场景
- 需要保证数据唯一性
- 增量数据同步

#### Delete 使用场景
- 数据清理和维护
- 合规性要求（数据删除）
- 批量数据管理

#### Query vs Search
- **Query**：精确查询，基于主键或过滤表达式
- **Search**：相似度检索，基于向量距离

## 总结

PyMilvus 的 CRUD 操作具有以下特点：

1. **简化的 API 设计**：`MilvusClient` 提供了更简洁的接口
2. **统一的数据格式**：使用字典列表格式，易于理解和使用
3. **完整的操作支持**：Insert、Upsert、Delete、Query、Search
4. **灵活的删除方式**：支持按 ID 和按表达式删除
5. **Upsert 支持部分更新**：可以只更新特定字段
6. **Python 友好**：符合 Python 开发习惯，易于集成
