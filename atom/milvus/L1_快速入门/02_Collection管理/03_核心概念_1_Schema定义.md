# Collection管理 - 核心概念1：Schema定义

> Schema 是 Collection 的结构定义，决定了数据的组织方式和约束规则

---

## 什么是 Schema？

**Schema（模式）** 是 Collection 的结构定义，类似于数据库表的表结构，定义了：
- 有哪些字段（Fields）
- 每个字段的类型（DataType）
- 字段的约束（主键、维度、长度等）

**核心作用：**
1. **类型安全**：确保插入的数据符合预定义的类型
2. **结构一致**：所有记录都有相同的字段结构
3. **检索优化**：为向量字段创建索引，加速检索

---

## Schema 的三大组成部分

### 1. 主键字段（Primary Key）

**定义：** 唯一标识每条记录的字段

```python
from pymilvus import FieldSchema, DataType

# 主键字段定义
pk_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True  # 自动生成 ID
)
```

**特性：**
- 必须唯一
- 支持类型：INT64、VARCHAR
- 可以自动生成（auto_id=True）
- 不可修改

**RAG 应用场景：**
- 文档 ID：唯一标识每个文档片段
- 用户 ID：多租户场景下的用户标识

---

### 2. 向量字段（Vector Field）

**定义：** 存储向量数据的字段，是向量检索的核心

```python
# FLOAT_VECTOR（标准精度）
vector_field = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=768  # 向量维度
)

# FLOAT16_VECTOR（低精度，节省 50% 存储）
vector_field_f16 = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT16_VECTOR,
    dim=768
)

# BFLOAT16_VECTOR（超大规模场景）
vector_field_bf16 = FieldSchema(
    name="vector",
    dtype=DataType.BFLOAT16_VECTOR,
    dim=768
)
```

**向量类型对比：**

| 类型 | 字节/维度 | 768维存储 | 精度 | 适用场景 |
|------|----------|----------|------|---------|
| FLOAT_VECTOR | 4 | 3072 字节 | 100% | 高精度场景（金融、医疗） |
| FLOAT16_VECTOR | 2 | 1536 字节 | 99%+ | 一般场景（文档检索、推荐） |
| BFLOAT16_VECTOR | 2 | 1536 字节 | 98%+ | 超大规模（>100M 向量） |

**关键约束：**
- 维度（dim）必须固定，不可修改
- 同一字段的所有向量维度必须一致
- 一个 Collection 可以有多个向量字段（多向量检索）

**RAG 应用场景：**
- 文本向量：存储文档的 Embedding
- 图像向量：多模态检索
- 稀疏向量：BM25 混合检索

---

### 3. 标量字段（Scalar Field）

**定义：** 存储元数据的字段，用于过滤和展示

```python
# VARCHAR 字段（文本）
text_field = FieldSchema(
    name="text",
    dtype=DataType.VARCHAR,
    max_length=512
)

# INT64 字段（整数）
timestamp_field = FieldSchema(
    name="timestamp",
    dtype=DataType.INT64
)

# JSON 字段（灵活元数据）
metadata_field = FieldSchema(
    name="metadata",
    dtype=DataType.JSON
)

# ARRAY 字段（多值数据，Milvus 2.6）
tags_field = FieldSchema(
    name="tags",
    dtype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=100
)
```

**标量类型速查：**

| 类型 | 用途 | 示例 |
|------|------|------|
| INT8/16/32/64 | 整数 | 时间戳、计数器 |
| FLOAT/DOUBLE | 浮点数 | 评分、权重 |
| VARCHAR | 文本 | 文档内容、标签 |
| BOOL | 布尔值 | 是否已读、是否公开 |
| JSON | 复杂元数据 | `{"author": "张三", "tags": ["AI"]}` |
| ARRAY | 数组（2.6） | 标签列表 `["AI", "ML", "DL"]` |

**RAG 应用场景：**
- text：存储原始文档内容
- source：文档来源（用于溯源）
- timestamp：文档创建时间（用于时间过滤）
- category：文档分类（用于 Partition 预过滤）
- metadata：灵活的附加信息（JSON）

---

## Schema 定义的完整流程

### 步骤1：定义字段

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

# 定义所有字段
fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    
    # 文本内容
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    
    # 向量表示
    FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    
    # 元数据
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="timestamp", dtype=DataType.INT64),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]
```

### 步骤2：创建 Schema

```python
# 创建 Schema
schema = CollectionSchema(
    fields=fields,
    description="RAG document collection with metadata"
)

# 查看 Schema 信息
print(f"字段数量: {len(schema.fields)}")
for field in schema.fields:
    print(f"  - {field.name}: {field.dtype}")
```

### 步骤3：创建 Collection

```python
from pymilvus import Collection

# 使用 Schema 创建 Collection
collection = Collection(
    name="rag_documents",
    schema=schema
)

print(f"✅ Collection '{collection.name}' 创建成功")
```

---

## Milvus 2.6 新特性：Dynamic Schema

### 什么是 Dynamic Schema？

**Dynamic Schema** 允许在 Collection 创建后动态添加字段，无需重建 Collection。

**传统方案的痛点：**
```python
# 传统方案：需求变化时
# 1. 导出所有数据
# 2. 删除旧 Collection
# 3. 创建新 Collection（新 Schema）
# 4. 重新导入数据
# 5. 重新创建索引
# 时间成本：数小时到数天
```

**Milvus 2.6 方案：**
```python
# 动态添加字段（秒级生效）
collection = Collection("rag_documents")

# 添加新字段
new_field = FieldSchema(
    name="category",
    dtype=DataType.VARCHAR,
    max_length=64
)
collection.add_field(new_field)

print("✅ 新字段 'category' 添加成功")
```

### Dynamic Schema 的能力边界

**✅ 可以做的：**
1. 添加新的标量字段（VARCHAR、INT、JSON、ARRAY）
2. 添加新的向量字段（支持多向量检索）
3. 无需停机，秒级生效

**❌ 不能做的：**
1. 修改已有字段的类型
2. 修改向量维度
3. 删除字段
4. 修改主键定义

**示例：**
```python
from pymilvus import Collection, FieldSchema, DataType

collection = Collection("rag_documents")

# ✅ 正确：添加新字段
collection.add_field(
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64)
)

# ✅ 正确：添加新向量字段
collection.add_field(
    FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512)
)

# ❌ 错误：修改已有字段类型（不支持）
# collection.alter_field("text", dtype=DataType.JSON)

# ❌ 错误：修改向量维度（不支持）
# collection.alter_field("vector", dim=1024)

# ❌ 错误：删除字段（不支持）
# collection.drop_field("source")
```

### Dynamic Schema 的应用场景

**场景1：敏捷开发**
```python
# 初始 Schema（MVP 版本）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
collection = Collection("docs", CollectionSchema(fields))

# 业务迭代1：添加文档来源
collection.add_field(
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128)
)

# 业务迭代2：添加文档分类
collection.add_field(
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64)
)

# 业务迭代3：添加元数据
collection.add_field(
    FieldSchema(name="metadata", dtype=DataType.JSON)
)
```

**场景2：A/B 测试**
```python
# 添加实验字段
collection.add_field(
    FieldSchema(name="experiment_group", dtype=DataType.VARCHAR, max_length=32)
)

# 测试完成后，字段保留（无法删除，但可以不使用）
```

**场景3：多模态扩展**
```python
# 初始：仅文本向量
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
collection = Collection("multimodal", CollectionSchema(fields))

# 扩展：添加图像向量
collection.add_field(
    FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512)
)

# 扩展：添加音频向量
collection.add_field(
    FieldSchema(name="audio_vector", dtype=DataType.FLOAT_VECTOR, dim=256)
)
```

---

## Schema 设计最佳实践

### 1. 向量类型选择

**决策树：**
```
需要高精度？（金融、医疗）
├─ 是 → FLOAT_VECTOR (32-bit)
└─ 否 → 数据量大？（>10M 向量）
    ├─ 是 → BFLOAT16_VECTOR (16-bit, 节省 50%)
    └─ 否 → FLOAT16_VECTOR (16-bit, 节省 50%)
```

**成本对比：**
```python
# 场景：100M 向量，768 维

# FLOAT_VECTOR
# 存储：100M * 768 * 4 = 307 GB
# 成本：$$$

# FLOAT16_VECTOR
# 存储：100M * 768 * 2 = 154 GB（节省 50%）
# 成本：$$（推荐）

# BFLOAT16_VECTOR
# 存储：100M * 768 * 2 = 154 GB（节省 50%）
# 成本：$$（超大规模）
```

### 2. 主键设计

**推荐：使用 auto_id**
```python
# ✅ 推荐：自动生成 ID
FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

# ❌ 不推荐：手动管理 ID（容易冲突）
FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
```

**VARCHAR 主键：**
```python
# 适用场景：已有唯一标识（如 UUID、文档 ID）
FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True)
```

### 3. 文本字段长度

**根据 Chunking 策略确定：**
```python
# 小块（适合精确检索）
FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256)

# 中块（平衡检索和上下文）
FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)

# 大块（保留更多上下文）
FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048)
```

### 4. 元数据存储

**使用 JSON 字段存储灵活元数据：**
```python
# ✅ 推荐：使用 JSON
FieldSchema(name="metadata", dtype=DataType.JSON)

# 插入数据
data = {
    "metadata": {
        "author": "张三",
        "tags": ["AI", "ML"],
        "created_at": "2026-02-21",
        "version": 1
    }
}

# ❌ 不推荐：为每个元数据创建单独字段（不灵活）
FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=64)
FieldSchema(name="tag1", dtype=DataType.VARCHAR, max_length=32)
FieldSchema(name="tag2", dtype=DataType.VARCHAR, max_length=32)
```

### 5. Partition 设计

**按常用过滤字段创建 Partition：**
```python
# 场景：按文档分类检索
# ✅ 推荐：使用 Partition
collection.create_partition("tech")
collection.create_partition("business")
collection.create_partition("news")

# 插入时指定 Partition
collection.insert(data, partition_name="tech")

# 检索时指定 Partition（避免标量过滤）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "COSINE"},
    limit=10,
    partition_names=["tech"]  # 只在 tech Partition 中检索
)

# ❌ 不推荐：使用标量过滤（性能差）
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "COSINE"},
    limit=10,
    expr="category == 'tech'"  # 标量过滤，性能下降
)
```

---

## RAG 应用的 Schema 设计模板

### 模板1：基础文档问答

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    
    # 文档内容
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    
    # 向量表示（使用 FLOAT16 节省成本）
    FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    
    # 元数据
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="timestamp", dtype=DataType.INT64)
]

schema = CollectionSchema(fields, description="Basic RAG document collection")
```

### 模板2：多模态检索

```python
fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    
    # 文本内容
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    
    # 文本向量
    FieldSchema(name="text_vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    
    # 图像向量
    FieldSchema(name="image_vector", dtype=DataType.FLOAT16_VECTOR, dim=512),
    
    # 元数据
    FieldSchema(name="metadata", dtype=DataType.JSON)
]

schema = CollectionSchema(fields, description="Multi-modal RAG collection")
```

### 模板3：企业知识库

```python
fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    
    # 文档内容
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    
    # 向量表示
    FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=1536),
    
    # 文档分类（用于 Partition）
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    
    # 标签列表（ARRAY 类型）
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50),
    
    # 元数据（JSON）
    FieldSchema(name="metadata", dtype=DataType.JSON)
]

schema = CollectionSchema(fields, description="Enterprise knowledge base")
```

---

## 学习检查

完成本节学习后，你应该能够：

- [ ] 理解 Schema 的三大组成部分（主键、向量、标量）
- [ ] 选择合适的向量类型（FLOAT vs FLOAT16 vs BFLOAT16）
- [ ] 使用 Dynamic Schema 动态添加字段
- [ ] 设计合理的 RAG Collection Schema
- [ ] 理解 Dynamic Schema 的能力边界
- [ ] 应用 Schema 设计最佳实践

---

## 下一步

- **CRUD 操作**：[03_核心概念_2_Collection_CRUD操作](./03_核心概念_2_Collection_CRUD操作.md)
- **字段类型详解**：[03_核心概念_3_字段类型详解](./03_核心概念_3_字段类型详解.md)
- **实战代码**：[07_实战代码_场景1_基础Collection创建](./07_实战代码_场景1_基础Collection创建.md)
- **返回导航**：[00_概览](./00_概览.md)
