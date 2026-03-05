# 实战代码 场景4：JSON字段过滤

> **目标**：掌握 JSON 字段的嵌套访问、数组过滤和 JSON Path Index

---

## 概述

本文档提供 JSON 字段过滤的完整实战代码，包括嵌套访问、数组操作和性能优化。

**学习目标**：
- 掌握 JSON Path 语法
- 理解 JSON Path Index
- 实现复杂 JSON 查询
- 优化 JSON 查询性能

---

## 环境准备

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import random

client = MilvusClient("http://localhost:19530")
model = SentenceTransformer('all-MiniLM-L6-v2')
```

---

## 场景1：基础 JSON 字段查询

### 数据准备

```python
# 创建集合
collection_name = "products_json"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

schema = client.create_schema(auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=384)
schema.add_field("metadata", DataType.JSON)

client.create_collection(collection_name=collection_name, schema=schema)

# 准备数据
data = []
for i in range(10000):
    text = f"Product {i}"
    vector = model.encode(text).tolist()
    data.append({
        "vector": vector,
        "metadata": {
            "product": {
                "category": random.choice(["electronics", "books", "clothing"]),
                "brand": random.choice(["Apple", "Samsung", "Nike"]),
                "model": f"Model-{i % 100}"
            },
            "pricing": {
                "base": random.uniform(100, 1000),
                "final": random.uniform(80, 900),
                "currency": "USD"
            },
            "tags": random.sample(["hot", "new", "sale", "limited", "premium"], k=random.randint(1, 3)),
            "stock": random.randint(0, 100)
        }
    })

client.insert(collection_name, data)
print(f"插入 {len(data)} 条数据")
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

### 基础查询

```python
query_vector = model.encode("Product query").tolist()

# 查询1：单层访问
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="metadata['stock'] > 50",
    limit=10,
    output_fields=["metadata"]
)

print("查询结果（stock > 50）：")
for hits in results:
    for hit in hits:
        print(f"  Stock: {hit['entity']['metadata']['stock']}")

# 查询2：嵌套访问
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="metadata['product']['category'] == 'electronics'",
    limit=10,
    output_fields=["metadata"]
)

print("\n查询结果（category == 'electronics'）：")
for hits in results:
    for hit in hits:
        print(f"  Category: {hit['entity']['metadata']['product']['category']}")
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

---

## 场景2：JSON Path Index

### 创建索引

```python
# 准备索引参数
index_params = client.prepare_index_params()

# 向量索引
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# JSON Path Index - category
index_params.add_index(
    field_name="metadata",
    index_name="category_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['category']",
        "json_cast_type": "varchar"
    }
)

# JSON Path Index - brand
index_params.add_index(
    field_name="metadata",
    index_name="brand_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['brand']",
        "json_cast_type": "varchar"
    }
)

# JSON Path Index - price
index_params.add_index(
    field_name="metadata",
    index_name="price_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['pricing']['final']",
        "json_cast_type": "double"
    }
)

# JSON Path Index - stock
index_params.add_index(
    field_name="metadata",
    index_name="stock_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['stock']",
        "json_cast_type": "int64"
    }
)

# 创建索引
client.create_index(collection_name, index_params)
print("JSON Path Index 创建完成")
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

### 性能对比

```python
import time

# 创建无索引集合
collection_no_index = "products_json_no_index"
if client.has_collection(collection_no_index):
    client.drop_collection(collection_no_index)

client.create_collection(collection_name=collection_no_index, schema=schema)
client.insert(collection_no_index, data)

# 只创建向量索引
index_params_no_json = client.prepare_index_params()
index_params_no_json.add_index("vector", "AUTOINDEX", "COSINE")
client.create_index(collection_no_index, index_params_no_json)

# 测试查询
filter_expr = "metadata['product']['category'] == 'electronics' AND metadata['pricing']['final'] < 500"

# 有索引
start = time.time()
client.search(collection_name, [query_vector], filter=filter_expr, limit=10)
time_with_index = (time.time() - start) * 1000

# 无索引
start = time.time()
client.search(collection_no_index, [query_vector], filter=filter_expr, limit=10)
time_no_index = (time.time() - start) * 1000

print("\n=== JSON Path Index 性能对比 ===")
print(f"有索引: {time_with_index:.2f} ms")
print(f"无索引: {time_no_index:.2f} ms")
print(f"性能提升: {time_no_index / time_with_index:.2f}x")
```

[来源: reference/search_web_results.md | Milvus 2.6 官方博客]

---

## 场景3：数组操作

### 数组查询

```python
# 查询1：ARRAY_CONTAINS
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="ARRAY_CONTAINS(metadata['tags'], 'hot')",
    limit=10,
    output_fields=["metadata"]
)

print("\n=== ARRAY_CONTAINS 查询 ===")
for hits in results:
    for hit in hits:
        print(f"  Tags: {hit['entity']['metadata']['tags']}")

# 查询2：ARRAY_LENGTH
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="ARRAY_LENGTH(metadata['tags']) >= 2",
    limit=10,
    output_fields=["metadata"]
)

print("\n=== ARRAY_LENGTH 查询 ===")
for hits in results:
    for hit in hits:
        tags = hit['entity']['metadata']['tags']
        print(f"  Tags ({len(tags)}): {tags}")

# 查询3：多个标签
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="ARRAY_CONTAINS(metadata['tags'], 'hot') AND ARRAY_CONTAINS(metadata['tags'], 'new')",
    limit=10,
    output_fields=["metadata"]
)

print("\n=== 多标签查询 ===")
for hits in results:
    for hit in hits:
        print(f"  Tags: {hit['entity']['metadata']['tags']}")
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

### 数组索引

```python
# 为数组字段创建索引
index_params_array = client.prepare_index_params()
index_params_array.add_index("vector", "AUTOINDEX", "COSINE")
index_params_array.add_index(
    field_name="metadata",
    index_name="tags_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['tags']",
        "json_cast_type": "array_varchar"
    }
)

# 重建索引
client.drop_index(collection_name, "tags_index")
client.create_index(collection_name, index_params_array)
```

---

## 场景4：复杂 JSON 查询

### 多层嵌套查询

```python
# 查询：电子产品，价格<500，有库存，包含"hot"标签
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="""
        metadata['product']['category'] == 'electronics'
        AND metadata['pricing']['final'] < 500
        AND metadata['stock'] > 10
        AND ARRAY_CONTAINS(metadata['tags'], 'hot')
    """,
    limit=10,
    output_fields=["metadata"]
)

print("\n=== 复杂 JSON 查询 ===")
for hits in results:
    for hit in hits:
        meta = hit['entity']['metadata']
        print(f"  Category: {meta['product']['category']}")
        print(f"  Price: {meta['pricing']['final']:.2f}")
        print(f"  Stock: {meta['stock']}")
        print(f"  Tags: {meta['tags']}\n")
```

### 类型转换

```python
# 创建带类型转换的索引
index_params_cast = client.prepare_index_params()
index_params_cast.add_index("vector", "AUTOINDEX", "COSINE")
index_params_cast.add_index(
    field_name="metadata",
    index_name="price_cast_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['pricing']['final']",
        "json_cast_type": "double",
        "json_cast_function": "STRING_TO_DOUBLE"  # 如果存储为字符串
    }
)
```

[来源: reference/source_test_files.md | test_milvus_client_json_path_index.py]

---

## 场景5：NULL 值处理

### NULL 查询

```python
# 插入包含 NULL 的数据
data_with_null = []
for i in range(100):
    text = f"Product {i}"
    vector = model.encode(text).tolist()
    data_with_null.append({
        "vector": vector,
        "metadata": {
            "product": {
                "category": "electronics",
                "brand": None if i % 3 == 0 else "Apple"
            },
            "pricing": {
                "final": None if i % 5 == 0 else random.uniform(100, 1000)
            }
        }
    })

client.insert(collection_name, data_with_null)

# 查询 NULL 值
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="metadata['product']['brand'] IS NULL",
    limit=10,
    output_fields=["metadata"]
)

print("\n=== NULL 值查询 ===")
for hits in results:
    for hit in hits:
        brand = hit['entity']['metadata']['product']['brand']
        print(f"  Brand: {brand}")

# 查询非 NULL 值
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="metadata['product']['brand'] IS NOT NULL",
    limit=10,
    output_fields=["metadata"]
)
```

---

## 完整示例代码

```python
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import random
import time

def main():
    # 初始化
    client = MilvusClient("http://localhost:19530")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 创建集合
    collection_name = "json_filter_demo"
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=384)
    schema.add_field("metadata", DataType.JSON)

    client.create_collection(collection_name=collection_name, schema=schema)

    # 准备数据
    data = []
    for i in range(1000):
        text = f"Product {i}"
        vector = model.encode(text).tolist()
        data.append({
            "vector": vector,
            "metadata": {
                "product": {
                    "category": random.choice(["electronics", "books"]),
                    "brand": random.choice(["Apple", "Samsung"])
                },
                "pricing": {
                    "final": random.uniform(100, 1000)
                },
                "tags": random.sample(["hot", "new", "sale"], k=2),
                "stock": random.randint(0, 100)
            }
        })

    client.insert(collection_name, data)

    # 创建 JSON Path Index
    index_params = client.prepare_index_params()
    index_params.add_index("vector", "AUTOINDEX", "COSINE")
    index_params.add_index(
        field_name="metadata",
        index_name="category_index",
        index_type="INVERTED",
        params={
            "json_path": "metadata['product']['category']",
            "json_cast_type": "varchar"
        }
    )
    index_params.add_index(
        field_name="metadata",
        index_name="price_index",
        index_type="INVERTED",
        params={
            "json_path": "metadata['pricing']['final']",
            "json_cast_type": "double"
        }
    )

    client.create_index(collection_name, index_params)

    query_vector = model.encode("Product query").tolist()

    # 测试1：嵌套访问
    print("=== 嵌套访问 ===")
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        filter="metadata['product']['category'] == 'electronics'",
        limit=5,
        output_fields=["metadata"]
    )
    for hits in results:
        for hit in hits:
            print(f"Category: {hit['entity']['metadata']['product']['category']}")

    # 测试2：数组操作
    print("\n=== 数组操作 ===")
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        filter="ARRAY_CONTAINS(metadata['tags'], 'hot')",
        limit=5,
        output_fields=["metadata"]
    )
    for hits in results:
        for hit in hits:
            print(f"Tags: {hit['entity']['metadata']['tags']}")

    # 测试3：复合查询
    print("\n=== 复合查询 ===")
    start = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        filter="""
            metadata['product']['category'] == 'electronics'
            AND metadata['pricing']['final'] < 500
            AND metadata['stock'] > 10
        """,
        limit=5
    )
    elapsed = (time.time() - start) * 1000
    print(f"查询延迟: {elapsed:.2f} ms")

if __name__ == "__main__":
    main()
```

---

## 常见问题

### Q1: JSON Path 语法错误

**问题**：
```python
# 错误：使用点号语法
filter = "metadata.product.category == 'electronics'"
# SyntaxError
```

**解决方案**：
```python
# 正确：使用方括号语法
filter = "metadata['product']['category'] == 'electronics'"
```

### Q2: 类型不匹配

**问题**：
```python
# 错误：字符串比较数字
filter = "metadata['price'] > 100"  # price 存储为字符串
```

**解决方案**：
```python
# 方案1：创建索引时指定类型转换
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    params={
        "json_path": "metadata['price']",
        "json_cast_type": "double",
        "json_cast_function": "STRING_TO_DOUBLE"
    }
)

# 方案2：存储时使用正确类型
data = {"metadata": {"price": 100.0}}  # 数字类型
```

### Q3: 数组查询失败

**问题**：
```python
# 错误：直接比较数组
filter = "metadata['tags'] == 'hot'"  # tags 是数组
```

**解决方案**：
```python
# 正确：使用 ARRAY_CONTAINS
filter = "ARRAY_CONTAINS(metadata['tags'], 'hot')"
```

### Q4: 性能慢

**问题**：
```python
# 查询很慢（> 1000ms）
filter = "metadata['product']['category'] == 'electronics'"
```

**解决方案**：
```python
# 创建 JSON Path Index
index_params.add_index(
    field_name="metadata",
    index_name="category_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['category']",
        "json_cast_type": "varchar"
    }
)
client.create_index(collection_name, index_params)
```

---

## 优化建议

### 1. 为常用路径创建索引

```python
# 分析查询模式，为高频路径创建索引
index_params.add_index(
    field_name="metadata",
    index_name="category_index",
    index_type="INVERTED",
    params={
        "json_path": "metadata['product']['category']",
        "json_cast_type": "varchar"
    }
)
```

### 2. 正确指定类型

```python
# 数值类型
"json_cast_type": "double"  # 或 "int64"

# 字符串类型
"json_cast_type": "varchar"

# 数组类型
"json_cast_type": "array_varchar"  # 或 "array_int64"
```

### 3. 避免深层嵌套

```python
# 不推荐：过深嵌套
"metadata['level1']['level2']['level3']['level4']['value']"

# 推荐：扁平化结构
"metadata['value']"
```

---

## 类比理解

### 前端开发类比

**JSON Path = JavaScript 对象访问**：
```javascript
// JavaScript
const category = product.metadata.product.category;

// Milvus
"metadata['product']['category']"
```

**ARRAY_CONTAINS = Array.includes()**：
```javascript
// JavaScript
const hasTags = product.tags.includes('hot');

// Milvus
"ARRAY_CONTAINS(metadata['tags'], 'hot')"
```

### 日常生活类比

**JSON Path = 地址导航**：
```
地址：北京市 → 朝阳区 → 建国路 → 1号
JSON Path：metadata['city']['district']['street']['number']

都是层层访问，逐级深入
```

**JSON Path Index = 快递分拣**：
```
无索引：逐个包裹查找（慢）
有索引：按地址分类存放（快）

JSON Path Index：按路径分类索引（快）
```

---

## 一句话总结

JSON 字段过滤通过方括号语法访问嵌套路径，为常用路径创建 JSON Path Index 可实现 100 倍性能提升，ARRAY_CONTAINS 用于数组查询，正确指定 json_cast_type 避免类型错误。

---

**下一步**：学习 [07_实战代码_场景5_JSON_Path_Index性能对比.md](./07_实战代码_场景5_JSON_Path_Index性能对比.md)，深入理解性能优化
