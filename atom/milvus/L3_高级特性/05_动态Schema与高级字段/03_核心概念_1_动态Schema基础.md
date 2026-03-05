# 核心概念1：动态Schema基础

> 本文档详细讲解 Milvus 2.6 动态 Schema 的核心机制、$meta 字段原理和实际应用场景

---

## 一句话定义

**动态 Schema 是 Milvus 允许在不修改 Collection Schema 的情况下插入未定义字段的机制，通过 `EnableDynamicField` 启用，所有动态字段自动存储在隐式的 `$meta` 字段中。**

---

## 核心机制

### 1. EnableDynamicField 启用机制

**定义**：通过在 Collection Schema 中设置 `enable_dynamic_field=True` 来启用动态字段支持。

**源码实现**（来源：reference/source_动态Schema_01.md）：

```go
schema.EnableDynamicField = true
```

**Python API**（来源：reference/context7_pymilvus_01.md）：

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 方式1：使用 MilvusClient 快速创建（默认启用动态字段）
client.create_collection(
    collection_name="documents",
    dimension=768,
    metric_type="COSINE"
)
# 注意：MilvusClient.create_collection() 默认启用动态字段

# 方式2：显式启用动态字段
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 显式启用
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

# 创建 Collection
client.create_collection(
    collection_name="documents",
    schema=schema,
    index_params=index_params
)
```

**关键特性**：
- 启用后，可以插入任意未在 Schema 中定义的字段
- 动态字段不需要预先声明类型
- 支持多种数据类型（字符串、数字、数组、JSON 等）

---

### 2. $meta 字段的隐式存储

**定义**：`$meta` 是 Milvus 内部自动创建的特殊字段，用于存储所有动态字段的数据。

**存储机制**（来源：reference/source_动态Schema_01.md）：

```
固定字段：直接存储在对应的列中
动态字段：统一存储在 $meta 字段中（JSON 格式）

示例数据结构：
{
  "id": 1,                    // 固定字段
  "vector": [0.1, 0.2, ...],  // 固定字段
  "title": "Document 1",      // 固定字段
  "$meta": {                  // 动态字段容器
    "author": "Alice",
    "tags": ["AI", "ML"],
    "created_at": 1640000000
  }
}
```

**关键特性**：
- `$meta` 字段对用户透明，无需手动操作
- 查询时可以直接访问动态字段，无需通过 `$meta` 前缀
- 动态字段的类型在插入时自动推断

**类型推断规则**：

| Python 类型 | Milvus 动态字段类型 |
|------------|-------------------|
| `str` | VARCHAR |
| `int` | INT64 |
| `float` | DOUBLE |
| `bool` | BOOL |
| `list` | ARRAY |
| `dict` | JSON |

---

### 3. 动态字段的数据类型支持

**支持的数据类型**（来源：reference/context7_milvus_02.md）：

```python
# 标量类型
data = {
    "id": 1,
    "vector": [0.1] * 768,
    # 动态字段
    "title": "Document 1",           # VARCHAR
    "page_count": 100,               # INT64
    "price": 29.99,                  # DOUBLE
    "is_published": True,            # BOOL
}

# 数组类型
data = {
    "id": 2,
    "vector": [0.2] * 768,
    # 动态字段
    "tags": ["AI", "ML", "NLP"],     # ARRAY<VARCHAR>
    "scores": [0.9, 0.8, 0.95],      # ARRAY<DOUBLE>
}

# JSON 类型
data = {
    "id": 3,
    "vector": [0.3] * 768,
    # 动态字段
    "metadata": {                    # JSON
        "author": "Bob",
        "department": "Engineering",
        "permissions": {
            "read": ["user1", "user2"],
            "write": ["user1"]
        }
    }
}
```

**类型限制**：
- 动态字段不能是向量类型（FLOAT_VECTOR, INT8_VECTOR 等）
- 动态字段不能设置为主键
- 动态字段不能创建索引（但可以用于过滤）

---

## 与固定 Schema 的对比

### 对比表

| 特性 | 固定 Schema | 动态 Schema |
|------|------------|------------|
| **字段定义** | 必须预先定义 | 可以动态添加 |
| **类型检查** | 严格类型检查 | 自动类型推断 |
| **索引支持** | 支持所有索引类型 | 动态字段不支持索引 |
| **查询性能** | 高（有索引支持） | 较低（无索引） |
| **灵活性** | 低 | 高 |
| **适用场景** | 结构化数据 | 半结构化数据 |
| **Schema 变更** | 需要 AddCollectionField | 无需变更 |

### 代码对比

**固定 Schema**：

```python
# 必须预先定义所有字段
schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="tags", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=128)

# 插入数据时必须包含所有字段
data = [{
    "id": 1,
    "vector": [0.1] * 768,
    "title": "Document 1",
    "author": "Alice",
    "tags": ["AI", "ML"]
}]

# 如果需要添加新字段，必须使用 AddCollectionField
client.add_collection_field(
    collection_name="documents",
    field_name="category",
    data_type=DataType.VARCHAR,
    max_length=100,
    nullable=True
)
```

**动态 Schema**：

```python
# 只定义核心字段
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

# 插入数据时可以包含任意字段
data = [{
    "id": 1,
    "vector": [0.1] * 768,
    "title": "Document 1",      # 动态字段
    "author": "Alice",          # 动态字段
    "tags": ["AI", "ML"]        # 动态字段
}]

# 后续可以插入不同的动态字段
data = [{
    "id": 2,
    "vector": [0.2] * 768,
    "title": "Document 2",
    "category": "Tech",         # 新的动态字段
    "published_date": "2026-01-01"  # 新的动态字段
}]
```

---

## 使用场景

### 场景1：元数据不确定的 RAG 系统

**问题**：不同来源的文档有不同的元数据字段。

**解决方案**：使用动态 Schema 灵活存储元数据。

```python
# PDF 文档
pdf_doc = {
    "id": 1,
    "vector": embedding,
    "text": "Document content...",
    "source": "pdf",
    "page_number": 5,
    "total_pages": 100
}

# 网页文档
web_doc = {
    "id": 2,
    "vector": embedding,
    "text": "Web content...",
    "source": "web",
    "url": "https://example.com",
    "crawled_at": "2026-01-01"
}

# 邮件文档
email_doc = {
    "id": 3,
    "vector": embedding,
    "text": "Email content...",
    "source": "email",
    "sender": "alice@example.com",
    "subject": "Meeting notes"
}

# 所有文档可以插入同一个 Collection
client.insert(collection_name="documents", data=[pdf_doc, web_doc, email_doc])
```

**优势**：
- 无需为每种文档类型创建单独的 Collection
- 新增文档类型时无需修改 Schema
- 查询时可以按来源过滤

---

### 场景2：多租户系统

**问题**：不同租户需要不同的自定义字段。

**解决方案**（来源：reference/search_RAG多租户_03.md）：

```python
# 租户 A 的数据
tenant_a_data = {
    "id": 1,
    "vector": embedding,
    "text": "Content...",
    "tenant_id": "tenant_a",
    "custom_field_a": "Value A",
    "priority": "high"
}

# 租户 B 的数据
tenant_b_data = {
    "id": 2,
    "vector": embedding,
    "text": "Content...",
    "tenant_id": "tenant_b",
    "custom_field_b": "Value B",
    "department": "Engineering"
}

# 查询租户 A 的数据
results = client.search(
    collection_name="multi_tenant",
    data=[query_vector],
    filter='tenant_id == "tenant_a"',
    output_fields=["text", "custom_field_a", "priority"]
)
```

**优势**：
- 每个租户可以有自己的自定义字段
- 无需为每个租户创建单独的 Collection
- 灵活支持租户需求变化

---

### 场景3：快速迭代的项目

**问题**：产品需求频繁变化，需要经常添加新字段。

**解决方案**：

```python
# 初始版本
v1_data = {
    "id": 1,
    "vector": embedding,
    "title": "Document 1"
}

# 迭代1：添加作者字段
v2_data = {
    "id": 2,
    "vector": embedding,
    "title": "Document 2",
    "author": "Alice"
}

# 迭代2：添加标签和分类
v3_data = {
    "id": 3,
    "vector": embedding,
    "title": "Document 3",
    "author": "Bob",
    "tags": ["AI", "ML"],
    "category": "Tech"
}

# 所有版本的数据可以共存
client.insert(collection_name="documents", data=[v1_data, v2_data, v3_data])
```

**优势**：
- 无需停机修改 Schema
- 旧数据和新数据可以共存
- 支持渐进式迁移

---

### 场景4：A/B 测试和实验

**问题**：需要为不同实验组存储不同的元数据。

**解决方案**：

```python
# 实验组 A
experiment_a = {
    "id": 1,
    "vector": embedding,
    "text": "Content...",
    "experiment_id": "exp_a",
    "variant": "control",
    "feature_flag_1": True
}

# 实验组 B
experiment_b = {
    "id": 2,
    "vector": embedding,
    "text": "Content...",
    "experiment_id": "exp_b",
    "variant": "treatment",
    "feature_flag_2": True,
    "new_algorithm": "v2"
}

# 查询特定实验组
results = client.search(
    collection_name="experiments",
    data=[query_vector],
    filter='experiment_id == "exp_a" and variant == "control"'
)
```

---

## 动态字段的查询和过滤

### 基础查询

```python
# 查询动态字段
results = client.search(
    collection_name="documents",
    data=[query_vector],
    output_fields=["title", "author", "tags"],  # 包含动态字段
    limit=10
)

# 访问结果
for hit in results[0]:
    print(f"Title: {hit['title']}")      # 动态字段
    print(f"Author: {hit['author']}")    # 动态字段
    print(f"Tags: {hit['tags']}")        # 动态字段
```

### 过滤动态字段

```python
# 字符串过滤
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='author == "Alice"',  # 动态字段过滤
    limit=10
)

# 数值过滤
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='page_count > 50',
    limit=10
)

# 数组过滤
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='array_contains(tags, "AI")',
    limit=10
)

# 复合过滤
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='author == "Alice" and page_count > 50',
    limit=10
)
```

---

## 动态字段的限制

### 限制1：不支持索引

**问题**：动态字段无法创建索引，查询性能较低。

**影响**（来源：reference/search_动态Schema_01.md）：
- 过滤动态字段时需要全表扫描
- 大数据量下性能显著下降

**解决方案**：
- 将高频查询字段定义为固定字段
- 为固定字段创建标量索引

```python
# 不推荐：所有字段都是动态的
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
# author 是动态字段，无法创建索引

# 推荐：高频查询字段定义为固定字段
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)  # 固定字段

# 为 author 创建索引
index_params.add_index(field_name="author", index_type="INVERTED")
```

---

### 限制2：group_by 问题

**问题**（来源：reference/search_动态Schema_01.md，Issue #47438）：
- 动态字段的 `group_by` 对缺失字段处理错误，视为有效组
- 可能导致聚合查询结果不准确

**示例**：

```python
# 数据
data = [
    {"id": 1, "vector": [0.1]*768, "category": "A"},
    {"id": 2, "vector": [0.2]*768, "category": "B"},
    {"id": 3, "vector": [0.3]*768}  # 缺少 category 字段
]

# group_by 查询
results = client.search(
    collection_name="documents",
    data=[query_vector],
    group_by_field="category",  # 动态字段
    limit=10
)

# 问题：id=3 的记录会被视为一个独立的组（category=null）
# 预期：应该被过滤或单独处理
```

**解决方案**：
- 在应用层过滤缺失字段
- 或使用固定字段进行 group_by

---

### 限制3：不能是向量类型

**问题**：动态字段不能是向量类型。

**错误示例**：

```python
# ❌ 错误：动态字段不能是向量
data = {
    "id": 1,
    "vector": [0.1] * 768,
    "extra_vector": [0.2] * 768  # 错误：动态字段不能是向量
}
```

**正确做法**：

```python
# ✅ 正确：向量字段必须在 Schema 中定义
schema.add_field(field_name="vector1", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="vector2", datatype=DataType.FLOAT_VECTOR, dim=768)

data = {
    "id": 1,
    "vector1": [0.1] * 768,
    "vector2": [0.2] * 768
}
```

---

## 最佳实践

### 实践1：核心字段固定，业务字段动态

```python
# 核心字段（固定）
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="created_at", datatype=DataType.INT64)

# 业务字段（动态）
# 根据实际需求动态添加
```

**优势**：
- 核心字段有索引支持，查询性能高
- 业务字段灵活，支持快速迭代

---

### 实践2：为高频过滤字段创建固定字段

```python
# 高频过滤字段（固定）
schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)

# 创建索引
index_params.add_index(field_name="tenant_id", index_type="INVERTED")
index_params.add_index(field_name="category", index_type="INVERTED")
index_params.add_index(field_name="status", index_type="INVERTED")
```

---

### 实践3：使用 JSON 字段存储复杂元数据

```python
# 固定字段
schema.add_field(field_name="metadata", datatype=DataType.JSON)

# 插入数据
data = {
    "id": 1,
    "vector": [0.1] * 768,
    "metadata": {
        "author": "Alice",
        "tags": ["AI", "ML"],
        "permissions": {
            "read": ["user1", "user2"],
            "write": ["user1"]
        }
    }
}

# 查询 JSON 字段
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='metadata["author"] == "Alice"',
    limit=10
)
```

**优势**：
- JSON 字段支持嵌套结构
- 可以使用 JSON Path Index（Milvus 2.6+）优化查询

---

## 在 Milvus 2.6 中的应用

### 2026 年生产环境标准配置

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 创建 Schema（启用动态字段）
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 2026 年标准配置
)

# 核心字段（固定）
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

# 高频过滤字段（固定）
schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="created_at", datatype=DataType.INT64)

# 复杂元数据（固定 JSON 字段）
schema.add_field(field_name="metadata", datatype=DataType.JSON)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="tenant_id", index_type="INVERTED")
index_params.add_index(field_name="created_at", index_type="INVERTED")

# 创建 Collection
client.create_collection(
    collection_name="documents",
    schema=schema,
    index_params=index_params
)
```

---

## 参考资料

### 源码分析
- reference/source_动态Schema_01.md - EnableDynamicField 机制和 $meta 字段实现

### 官方文档
- reference/context7_pymilvus_01.md - PyMilvus API 文档
- reference/context7_milvus_02.md - Milvus 官方文档

### 生产经验
- reference/search_动态Schema_01.md - 动态 Schema 生产环境问题和最佳实践
- reference/search_RAG多租户_03.md - RAG 多租户架构和元数据管理

---

## 总结

动态 Schema 是 Milvus 2.6 提供的灵活数据模型，通过 `EnableDynamicField` 启用，所有动态字段自动存储在 `$meta` 字段中。适用于元数据不确定、多租户系统、快速迭代等场景。

**核心要点**：
1. 通过 `enable_dynamic_field=True` 启用
2. 动态字段存储在隐式的 `$meta` 字段中
3. 支持多种数据类型（标量、数组、JSON）
4. 不支持索引，查询性能较低
5. 最佳实践：核心字段固定，业务字段动态

**2026 年生产标准**：
- 默认启用动态字段
- 核心字段和高频过滤字段定义为固定字段
- 使用 JSON 字段存储复杂元数据
- 为固定字段创建索引优化性能
