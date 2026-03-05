---
type: context7_documentation
library: pymilvus
version: master (2026-01-22)
fetched_at: 2026-02-25
knowledge_point: 05_动态Schema与高级字段
context7_query: AddCollectionField API usage nullable default values
---

# Context7 文档：PyMilvus

## 文档来源
- 库名称：pymilvus
- 版本：master (2026-01-22)
- Context7 ID：/milvus-io/pymilvus
- 官方文档链接：https://github.com/milvus-io/pymilvus

## 关键信息提取

### 1. FieldSchema 构造函数

**用途**：定义 Collection 中单个字段的 schema

**参数**：
- `name` (str) - **必需** - 字段名称
- `dtype` (str) - **必需** - 数据类型（如 'int64', 'float_vector'）
- `is_primary` (bool) - **可选** - 是否为主键，默认 False
- `auto_id` (bool) - **可选** - 是否自动生成 ID，默认 True（仅当 `is_primary=True` 时适用）
- `dim` (int) - **向量类型必需** - 向量维度
- `description` (str) - **可选** - 字段描述

**示例**：
```python
from pymilvus import FieldSchema, DataType

# 主键字段
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

# 向量字段
vector_field = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=128
)
```

---

### 2. CollectionSchema 构造函数

**用途**：定义 Collection 的完整 schema

**参数**：
- `fields` (list[FieldSchema]) - **必需** - FieldSchema 对象列表
- `description` (str) - **可选** - Collection 描述
- `auto_id` (bool) - **可选** - 是否自动生成 ID，默认 True

**属性**：
- `fields` - 字段列表
- `description` - 描述文本
- `primary_field()` - 主键字段名称
- `auto_id()` - 是否自动生成 ID

**示例**：
```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]

schema = CollectionSchema(
    fields=fields,
    description="My first collection"
)

# 访问属性
print(schema.primary_field())  # "id"
print(schema.auto_id())        # True
```

---

### 3. 快速创建 Collection（MilvusClient）

**用途**：使用最少配置快速创建 Collection

**特性**：
- 自动设置 schema、索引并加载到内存
- **动态字段默认启用**
- 支持自定义主键类型和向量字段名

**基础用法**：
```python
from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")

# 最简单的创建方式
# 自动创建：id (INT64 主键), vector (FLOAT_VECTOR)
client.create_collection(
    collection_name="quick_demo",
    dimension=128,
    metric_type="COSINE"  # 选项：COSINE, L2, IP
)
```

**高级配置**：
```python
# 自定义主键和向量字段
client.create_collection(
    collection_name="articles",
    dimension=768,
    primary_field_name="article_id",  # 自定义主键名
    id_type="string",                 # 字符串主键
    vector_field_name="embedding",    # 自定义向量字段名
    metric_type="L2",
    auto_id=True,
    consistency_level="Strong"
)
```

**验证和清理**：
```python
# 验证 Collection 是否存在
print(client.has_collection("quick_demo"))  # True

# 列出所有 Collection
print(client.list_collections())  # ['quick_demo', 'articles']

# 查看 Collection 详情
print(client.describe_collection("quick_demo"))

# 删除 Collection
client.drop_collection("quick_demo")
```

---

### 4. 字段默认值支持

**从 CollectionSchema 文档中发现**：

字段可以设置默认值，当插入或更新数据时，如果该字段为空，将使用默认值。

**示例**：
```python
from pymilvus import FieldSchema, DataType

# VARCHAR 字段带默认值
book_name = FieldSchema(
    name="book_name",
    dtype=DataType.VARCHAR,
    max_length=200,
    default_value="Unknown"  # 默认值
)

# INT64 字段带默认值
word_count = FieldSchema(
    name="word_count",
    dtype=DataType.INT64,
    default_value=9999  # 默认值
)
```

**重要说明**：
- `default_value` 的数据类型必须与 `dtype` 一致
- 默认值在数据插入或 upsert 时生效

---

## 关键发现总结

### 1. 动态字段默认启用
使用 `MilvusClient.create_collection()` 创建的 Collection **默认启用动态字段**，无需额外配置。

### 2. 字段定义灵活性
- 支持多种数据类型（INT64, VARCHAR, FLOAT_VECTOR 等）
- 支持自定义主键名称和类型
- 支持向量字段自定义名称

### 3. 默认值机制
- 字段可以设置默认值
- 默认值类型必须与字段类型一致
- 适用于插入和 upsert 操作

### 4. 快速开发模式
`MilvusClient` 提供了简化的 API，适合快速原型开发和简单场景。

---

## 与源码分析的对应关系

### 源码中的 `AddCollectionField`
从源码分析中，我们看到 `AddCollectionField` 是通过 DDL 回调实现的，支持：
- Nullable 字段
- 默认值设置
- Schema 版本管理

### pymilvus 客户端 API
pymilvus 提供了两种创建 Collection 的方式：
1. **MilvusClient**：简化 API，适合快速开发
2. **Collection + CollectionSchema**：完整 API，适合复杂场景

---

## 待确认问题

### 1. AddCollectionField API 在 pymilvus 中的位置
从 Context7 查询结果中，没有直接找到 `AddCollectionField` 的 API 文档。这可能意味着：
- 该 API 是较新的功能，文档尚未完善
- 该 API 在 `MilvusClient` 或 `Collection` 类中
- 需要进一步查询 Milvus 2.6 的官方文档

### 2. Nullable 字段的 API
从源码分析中看到 `WithNullable(true)` 的用法，但在 Context7 查询结果中没有找到对应的 Python API 文档。

### 3. 动态字段的详细配置
虽然知道 `enable_dynamic_field=True` 可以启用动态字段，但具体的配置选项和限制需要进一步查询。

---

## 下一步调研方向

1. **查询 Milvus 2.6 官方文档**：
   - AddCollectionField API 的完整文档
   - Nullable 字段的 Python API
   - 动态字段的详细配置和限制

2. **查询社区资料**：
   - AddCollectionField 的实际使用案例
   - Nullable 字段的最佳实践
   - 动态字段的性能影响
