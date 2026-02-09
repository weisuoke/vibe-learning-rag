# 核心概念 1：Schema 定义

## 什么是 Schema？

**Schema 是 Collection 的结构定义，规定了数据的字段类型、约束条件和索引配置。**

就像建筑图纸定义了房子的结构，Schema 定义了 Collection 的数据结构。

## Schema 的组成部分

一个完整的 Schema 包含以下核心元素：

### 1. 字段列表（Fields）

定义 Collection 中有哪些字段，每个字段的类型和属性。

### 2. 主键字段（Primary Key）

唯一标识每条数据的字段，类似于数据库表的主键。

### 3. 向量字段（Vector Field）

存储向量数据的字段，是向量检索的核心。

### 4. 标量字段（Scalar Fields）

存储元数据的字段，如文本、数字、布尔值等。

## Schema 定义示例

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# ===== 1. 定义字段 =====

# 主键字段：唯一标识
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=False,  # 手动指定 ID
    description="文档唯一标识"
)

# 向量字段：存储 Embedding
vector_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768,  # 向量维度（必须指定）
    description="文档的向量表示"
)

# 标量字段：文档标题
title_field = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=200,  # VARCHAR 必须指定最大长度
    description="文档标题"
)

# 标量字段：文档分类
category_field = FieldSchema(
    name="category",
    dtype=DataType.VARCHAR,
    max_length=50,
    description="文档分类"
)

# 标量字段：创建时间
timestamp_field = FieldSchema(
    name="created_at",
    dtype=DataType.INT64,
    description="创建时间戳"
)

# ===== 2. 创建 Schema =====

schema = CollectionSchema(
    fields=[id_field, vector_field, title_field, category_field, timestamp_field],
    description="技术文档 Collection",
    enable_dynamic_field=False  # 是否允许动态添加字段
)

print("Schema 创建成功！")
print(f"字段数量: {len(schema.fields)}")
print(f"主键字段: {schema.primary_field.name}")
print(f"向量字段: {[f.name for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR]}")
```

**输出：**
```
Schema 创建成功！
字段数量: 5
主键字段: id
向量字段: ['embedding']
```

## Schema 的关键属性

### 1. 字段类型（DataType）

Milvus 支持多种数据类型：

**标量类型：**
- `INT8`, `INT16`, `INT32`, `INT64`：整数
- `FLOAT`, `DOUBLE`：浮点数
- `VARCHAR`：字符串（需要指定 `max_length`）
- `BOOL`：布尔值
- `JSON`：JSON 对象（Milvus 2.2+）

**向量类型：**
- `FLOAT_VECTOR`：浮点向量（最常用）
- `BINARY_VECTOR`：二进制向量

### 2. 主键约束

```python
# 自动生成主键
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True  # Milvus 自动生成 ID
)

# 手动指定主键
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=False  # 插入数据时必须提供 ID
)
```

**注意：**
- 每个 Collection 必须有且只有一个主键字段
- 主键字段只能是 `INT64` 或 `VARCHAR` 类型
- 主键值必须唯一

### 3. 向量维度

```python
# 向量字段必须指定维度
vector_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768  # 必须与 Embedding 模型的输出维度一致
)
```

**常见 Embedding 模型的维度：**
- OpenAI `text-embedding-3-small`: 1536
- OpenAI `text-embedding-ada-002`: 1536
- `sentence-transformers/all-MiniLM-L6-v2`: 384
- `sentence-transformers/all-mpnet-base-v2`: 768

### 4. 动态字段（Dynamic Field）

```python
# 启用动态字段
schema = CollectionSchema(
    fields=[id_field, vector_field],
    enable_dynamic_field=True  # 允许插入时添加额外字段
)
```

**动态字段的作用：**
- 允许在插入数据时添加 Schema 中未定义的字段
- 适用于元数据不固定的场景
- 动态字段会被存储在一个特殊的 JSON 字段中

## Schema 设计最佳实践

### 1. 字段命名规范

```python
# ✅ 推荐：使用小写字母和下划线
"document_id", "content_vector", "created_at"

# ❌ 避免：使用特殊字符或空格
"document-id", "content vector", "created@time"
```

### 2. 向量维度选择

```python
# 根据 Embedding 模型选择维度
# 不要随意修改，必须与模型输出一致

# 示例：使用 OpenAI Embedding
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="测试文本"
)

embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")  # 1536

# Schema 中的维度必须匹配
vector_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=1536  # 与模型输出一致
)
```

### 3. VARCHAR 长度设置

```python
# 根据实际数据设置合理的长度
title_field = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=200  # 标题通常不超过 200 字符
)

content_field = FieldSchema(
    name="content",
    dtype=DataType.VARCHAR,
    max_length=5000  # 内容可能较长
)
```

**注意：**
- `max_length` 是字符数，不是字节数
- 设置过大会浪费存储空间
- 设置过小会导致数据截断

### 4. 是否启用动态字段

```python
# 场景1：元数据固定 → 不启用动态字段
schema = CollectionSchema(
    fields=[id_field, vector_field, title_field],
    enable_dynamic_field=False  # 严格的 Schema 约束
)

# 场景2：元数据不固定 → 启用动态字段
schema = CollectionSchema(
    fields=[id_field, vector_field],
    enable_dynamic_field=True  # 灵活的 Schema
)
```

## Schema 的不可变性

**重要：Schema 一旦创建，大部分属性不可修改！**

```python
# ❌ 不能修改的属性：
# - 字段类型
# - 向量维度
# - 主键字段
# - 字段名称

# ✅ 可以修改的属性：
# - 字段描述（通过 alter_collection）
# - 添加新字段（Milvus 2.3+，如果启用了动态 Schema）
```

**如果需要修改 Schema：**
1. 创建新的 Collection（新 Schema）
2. 将旧 Collection 的数据迁移到新 Collection
3. 删除旧 Collection

## 在 RAG 中的应用

### 场景1：文档问答系统

```python
# 文档 Collection Schema
doc_schema = CollectionSchema(
    fields=[
        FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="created_at", dtype=DataType.INT64)
    ],
    description="文档问答系统"
)
```

### 场景2：多模态检索

```python
# 图文混合 Collection Schema
multimodal_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)
    ],
    description="图文混合检索系统"
)
```

### 场景3：用户画像

```python
# 用户画像 Collection Schema
user_schema = CollectionSchema(
    fields=[
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
        FieldSchema(name="profile_vector", dtype=DataType.FLOAT_VECTOR, dim=256),
        FieldSchema(name="age", dtype=DataType.INT32),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="interests", dtype=DataType.JSON)  # 使用 JSON 存储复杂数据
    ],
    description="用户画像系统",
    enable_dynamic_field=True  # 允许添加额外的用户属性
)
```

## 总结

**Schema 是 Collection 的蓝图：**
1. 定义了数据的结构和类型
2. 必须在创建 Collection 前定义
3. 大部分属性不可修改
4. 向量维度必须与 Embedding 模型一致
5. 合理设计 Schema 是构建高效向量检索系统的基础

**下一步：** 了解如何使用 Schema 创建 Collection（核心概念 2）
