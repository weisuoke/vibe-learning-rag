# PyMilvus 标量过滤官方文档

> 来源：Context7 - /milvus-io/pymilvus

---

## 1. Query Data with Filters

**来源**：https://context7.com/milvus-io/pymilvus/llms.txt

### 功能说明

演示如何使用主键 ID、过滤表达式和分页从 Milvus 集合中查询数据。

### 支持的操作符

- `>=`：大于等于
- `<`：小于
- `==`：等于
- `in`：IN 操作符

### 代码示例

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")
client.create_collection("users", dimension=64)

rng = np.random.default_rng(seed=42)
users = [
    {"id": i, "vector": rng.random(64).tolist(),
     "name": f"User_{i}", "age": 20 + (i % 50), "active": i % 2 == 0, "score": i * 1.5}
    for i in range(100)
]
client.insert("users", users)

# 1. 通过主键 ID 查询
results = client.query("users", ids=[1, 5, 10])
print("Query by IDs:")
for r in results:
    print(f"  {r}")

# 2. 使用过滤表达式查询
results = client.query(
    "users",
    filter="age >= 30 and age < 40 and active == true",
    output_fields=["name", "age", "score"]
)
print(f"\nActive users aged 30-39: {len(results)} found")

# 3. 使用 IN 操作符的复杂过滤
results = client.query(
    "users",
    filter="id in [1, 2, 3, 4, 5] and score > 5.0",
    output_fields=["*"]  # 所有字段
)

# 4. 使用 limit 和 offset 进行分页
results = client.query(
    "users",
    filter="active == true",
    output_fields=["name", "age"],
    limit=10,
    offset=20
)

client.drop_collection("users")
```

### 关键特性

1. **主键查询**：通过 `ids` 参数直接查询
2. **过滤表达式**：使用 `filter` 参数进行条件过滤
3. **字段选择**：使用 `output_fields` 指定返回字段
4. **分页支持**：使用 `limit` 和 `offset` 进行分页

---

## 2. Vector Similarity Search with Filters

**来源**：https://context7.com/milvus-io/pymilvus/llms.txt

### 功能说明

演示如何在向量相似度搜索中使用过滤表达式。

### 代码示例

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")
client.create_collection("articles", dimension=128, metric_type="L2")

rng = np.random.default_rng(seed=42)

# 插入示例数据
articles = [
    {"id": i, "vector": rng.random(128).tolist(), "category": f"cat_{i % 5}", "views": i * 100}
    for i in range(1000)
]
client.insert("articles", articles)

# 1. 基础向量搜索
query_vector = rng.random(128).tolist()
results = client.search(
    collection_name="articles",
    data=[query_vector],
    limit=5,
    output_fields=["category", "views"]
)

print("Search results:")
for hits in results:
    for hit in hits:
        print(f"  ID: {hit['id']}, Distance: {hit['distance']:.4f}, "
              f"Category: {hit['entity']['category']}, Views: {hit['entity']['views']}")

# 2. 带过滤表达式的搜索
results = client.search(
    collection_name="articles",
    data=[query_vector],
    filter="category == 'cat_2' and views > 500",
    limit=3,
    output_fields=["category", "views"],
    search_params={"metric_type": "L2", "params": {"nprobe": 16}}
)

# 3. 多向量搜索（批量）
query_vectors = [rng.random(128).tolist() for _ in range(3)]
results = client.search(
    collection_name="articles",
    data=query_vectors,
    limit=2
)
print(f"\nBatch search returned {len(results)} result sets")

client.drop_collection("articles")
```

### 关键特性

1. **向量搜索 + 过滤**：在 `search()` 方法中使用 `filter` 参数
2. **字段返回**：使用 `output_fields` 指定返回字段
3. **搜索参数**：使用 `search_params` 指定搜索算法参数
4. **批量搜索**：支持多个查询向量的批量搜索

---

## 3. Collection Search API

**来源**：https://github.com/milvus-io/pymilvus/blob/master/docs/source/api/collection.rst

### API 描述

在集合上执行向量相似度搜索，支持使用布尔表达式进行过滤。

### 端点

```
POST /collection/{collection_name}/search
```

### 参数

#### 路径参数
- **collection_name** (string) - 必需 - 要搜索的集合名称

#### 查询参数
- **limit** (integer) - 必需 - 返回的最大结果数
- **expr** (string) - 可选 - 用于过滤搜索结果的布尔表达式
- **output_fields** (list[string]) - 可选 - 要包含在输出中的字段名称列表
- **consistency_level** (string) - 可选 - 搜索的一致性级别

#### 请求体
- **data** (list[list[float]]) - 必需 - 要搜索的向量
- **search_params** (object) - 必需 - 搜索算法特定参数
  - **metric_type** (string) - 必需 - 用于相似度计算的度量类型
  - **params** (object) - 必需 - 算法特定参数（如 ANNS 的 ef）

### 请求示例

```json
{
  "data": [[0.1, 0.2, 0.3]],
  "search_params": {
    "metric_type": "L2",
    "params": {"ef": 10}
  }
}
```

### 响应示例

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

---

## 4. Collection Query API

**来源**：https://github.com/milvus-io/pymilvus/blob/master/docs/source/api/collection.rst

### API 描述

根据一组条件查询集合，通常通过主键或其他字段进行过滤。

### 端点

```
POST /collection/{collection_name}/query
```

### 参数

#### 路径参数
- **collection_name** (string) - 必需 - 要查询的集合名称

#### 查询参数
- **expr** (string) - 必需 - 用于过滤查询结果的表达式
- **output_fields** (list[string]) - 可选 - 要包含在输出中的字段名称列表
- **limit** (integer) - 可选 - 返回的最大结果数

### 响应示例

```json
{
  "results": [
    {"primary_field": "id1", "vector_field": [0.1, 0.2], "other_field": "value1"},
    {"primary_field": "id3", "vector_field": [0.5, 0.6], "other_field": "value3"}
  ]
}
```

---

## 5. Hybrid Search with Reranking

**来源**：https://context7.com/milvus-io/pymilvus/llms.txt

### 功能说明

介绍 Milvus 的混合搜索功能，允许跨多个向量字段进行搜索并使用融合或重排序策略。

### 代码示例

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker, WeightedRanker
import numpy as np

client = MilvusClient("http://localhost:19530")

# 混合搜索的设置将在这里进行，
# 包括创建具有多个向量字段的集合，
# 插入数据，然后定义带有重排序策略的搜索请求。
```

### 关键组件

1. **AnnSearchRequest**：定义单个向量字段的搜索请求
2. **RRFRanker**：Reciprocal Rank Fusion 重排序器
3. **WeightedRanker**：加权重排序器

---

## 总结

### 标量过滤语法

```python
# 比较操作符
"age >= 30"
"age < 40"
"active == true"

# 逻辑操作符
"age >= 30 and age < 40"
"category == 'cat_2' and views > 500"

# IN 操作符
"id in [1, 2, 3, 4, 5]"

# 复合条件
"age >= 30 and age < 40 and active == true"
"id in [1, 2, 3, 4, 5] and score > 5.0"
```

### 使用场景

1. **纯标量查询**：使用 `client.query()` 方法
2. **向量搜索 + 标量过滤**：使用 `client.search()` 方法的 `filter` 参数
3. **混合搜索**：使用 `AnnSearchRequest` 和 `Ranker`

### 性能优化

1. **Pre-filtering**：在向量搜索前应用过滤条件
2. **Post-filtering**：在向量搜索后应用过滤条件
3. **索引优化**：为常用的标量字段创建索引
