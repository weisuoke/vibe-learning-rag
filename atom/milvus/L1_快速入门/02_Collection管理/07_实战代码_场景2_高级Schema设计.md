# Collection管理 - 实战代码场景2：高级Schema设计

> 多模态文档检索系统：文本向量 + 图像向量 + JSON元数据 + ARRAY标签

---

## 场景描述

**应用场景：** 多模态文档检索系统

**需求：**
- 同时存储文本向量和图像向量
- 使用 JSON 字段存储灵活元数据
- 使用 ARRAY 字段存储标签列表
- 支持多向量检索
- 使用 Milvus 2.6 新特性

**技术栈：**
- Milvus 2.6
- pymilvus 2.6+
- Python 3.9+

---

## 完整代码实现

```python
"""
Milvus 2.6 高级 Schema 设计 - 多模态文档检索
演示：多向量字段 + JSON元数据 + ARRAY标签 + 混合检索
"""

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import numpy as np
from typing import List, Dict
import json
import time

# ===== 1. 连接到 Milvus =====
print("=" * 70)
print("步骤1: 连接到 Milvus 2.6")
print("=" * 70)

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
print("✅ 已连接到 Milvus")

# ===== 2. 定义高级 Schema =====
print("\n" + "=" * 70)
print("步骤2: 定义多模态 Collection Schema")
print("=" * 70)

fields = [
    # 主键
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    
    # 文档标题
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=256
    ),
    
    # 文本向量（FLOAT16，节省存储）
    FieldSchema(
        name="text_vector",
        dtype=DataType.FLOAT16_VECTOR,
        dim=768,
        description="文本内容的向量表示"
    ),
    
    # 图像向量（FLOAT16）
    FieldSchema(
        name="image_vector",
        dtype=DataType.FLOAT16_VECTOR,
        dim=512,
        description="文档图像的向量表示"
    ),
    
    # 标签列表（ARRAY 类型，Milvus 2.6）
    FieldSchema(
        name="tags",
        dtype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=50,
        description="文档标签列表"
    ),
    
    # 元数据（JSON 类型）
    FieldSchema(
        name="metadata",
        dtype=DataType.JSON,
        description="灵活的文档元数据"
    )
]

# 创建 Schema
schema = CollectionSchema(
    fields=fields,
    description="多模态文档检索 Collection"
)

print(f"✅ 高级 Schema 定义完成")
print(f"   - 字段数量: {len(fields)}")
print(f"   - 向量字段: 2 个（text_vector + image_vector）")
print(f"   - 向量类型: FLOAT16_VECTOR（节省 50% 存储）")
print(f"   - 特殊字段: ARRAY（标签）+ JSON（元数据）")

# ===== 3. 创建 Collection =====
print("\n" + "=" * 70)
print("步骤3: 创建多模态 Collection")
print("=" * 70)

collection_name = "multimodal_docs"

if utility.has_collection(collection_name):
    print(f"⚠️  Collection '{collection_name}' 已存在，删除旧的")
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)
print(f"✅ Collection '{collection_name}' 创建成功")

# ===== 4. 准备多模态数据 =====
print("\n" + "=" * 70)
print("步骤4: 准备多模态示例数据")
print("=" * 70)

# 模拟多模态文档数据
documents = [
    {
        "title": "Milvus 2.6 架构设计",
        "tags": ["Milvus", "架构", "向量数据库"],
        "metadata": {
            "author": "张三",
            "department": "技术部",
            "created_at": "2026-02-21",
            "page_count": 50,
            "has_images": True,
            "language": "zh-CN"
        }
    },
    {
        "title": "RAG 系统实战指南",
        "tags": ["RAG", "LLM", "实战"],
        "metadata": {
            "author": "李四",
            "department": "AI研究院",
            "created_at": "2026-02-20",
            "page_count": 80,
            "has_images": True,
            "language": "zh-CN"
        }
    },
    {
        "title": "向量检索性能优化",
        "tags": ["性能优化", "向量检索", "HNSW"],
        "metadata": {
            "author": "王五",
            "department": "技术部",
            "created_at": "2026-02-19",
            "page_count": 30,
            "has_images": False,
            "language": "zh-CN"
        }
    },
    {
        "title": "多模态AI应用开发",
        "tags": ["多模态", "AI", "应用开发"],
        "metadata": {
            "author": "赵六",
            "department": "AI研究院",
            "created_at": "2026-02-18",
            "page_count": 100,
            "has_images": True,
            "language": "zh-CN"
        }
    },
    {
        "title": "FLOAT16向量存储优化",
        "tags": ["FLOAT16", "存储优化", "成本"],
        "metadata": {
            "author": "钱七",
            "department": "技术部",
            "created_at": "2026-02-17",
            "page_count": 25,
            "has_images": False,
            "language": "zh-CN"
        }
    }
]

# 生成模拟向量
def generate_vector(text: str, dim: int) -> List[float]:
    """生成模拟向量"""
    np.random.seed(hash(text) % (2**32))
    return np.random.rand(dim).tolist()

# 准备插入数据
titles = [doc["title"] for doc in documents]
text_vectors = [generate_vector(doc["title"], 768) for doc in documents]
image_vectors = [generate_vector(doc["title"] + "_image", 512) for doc in documents]
tags_list = [doc["tags"] for doc in documents]
metadata_list = [doc["metadata"] for doc in documents]

print(f"✅ 准备了 {len(documents)} 条多模态文档数据")
print(f"   - 文本向量维度: {len(text_vectors[0])}")
print(f"   - 图像向量维度: {len(image_vectors[0])}")
print(f"   - 标签示例: {tags_list[0]}")
print(f"   - 元数据示例: {json.dumps(metadata_list[0], ensure_ascii=False, indent=2)}")

# ===== 5. 插入数据 =====
print("\n" + "=" * 70)
print("步骤5: 插入多模态数据")
print("=" * 70)

insert_result = collection.insert([
    titles,
    text_vectors,
    image_vectors,
    tags_list,
    metadata_list
])

print(f"✅ 数据插入成功")
print(f"   - 插入记录数: {len(insert_result.primary_keys)}")

collection.flush()
print(f"✅ 数据已刷新到磁盘")

# ===== 6. 为多个向量字段创建索引 =====
print("\n" + "=" * 70)
print("步骤6: 为多个向量字段创建索引")
print("=" * 70)

# 文本向量索引
text_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 256}
}

collection.create_index(
    field_name="text_vector",
    index_params=text_index_params
)
print(f"✅ 文本向量索引创建成功")

# 图像向量索引
image_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 256}
}

collection.create_index(
    field_name="image_vector",
    index_params=image_index_params
)
print(f"✅ 图像向量索引创建成功")

# ===== 7. 加载 Collection =====
print("\n" + "=" * 70)
print("步骤7: 加载 Collection 到内存")
print("=" * 70)

collection.load()
print(f"✅ Collection 已加载到内存")

# ===== 8. 场景1：文本向量检索 =====
print("\n" + "=" * 70)
print("步骤8: 场景1 - 文本向量检索")
print("=" * 70)

query_text = "如何优化向量数据库的性能？"
print(f"查询: {query_text}")

query_vector = generate_vector(query_text, 768)

results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=3,
    output_fields=["title", "tags", "metadata"]
)

print(f"\n✅ 文本检索完成，Top-3 结果:")
for i, hit in enumerate(results[0], 1):
    print(f"\n结果 {i}:")
    print(f"  - 相似度: {hit.distance:.4f}")
    print(f"  - 标题: {hit.entity.get('title')}")
    print(f"  - 标签: {hit.entity.get('tags')}")
    metadata = hit.entity.get('metadata')
    print(f"  - 作者: {metadata.get('author')}")
    print(f"  - 部门: {metadata.get('department')}")

# ===== 9. 场景2：图像向量检索 =====
print("\n" + "=" * 70)
print("步骤9: 场景2 - 图像向量检索")
print("=" * 70)

query_image = "architecture_diagram.png"
print(f"查询图像: {query_image}")

query_image_vector = generate_vector(query_image, 512)

results = collection.search(
    data=[query_image_vector],
    anns_field="image_vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=3,
    output_fields=["title", "tags", "metadata"]
)

print(f"\n✅ 图像检索完成，Top-3 结果:")
for i, hit in enumerate(results[0], 1):
    print(f"\n结果 {i}:")
    print(f"  - 相似度: {hit.distance:.4f}")
    print(f"  - 标题: {hit.entity.get('title')}")
    metadata = hit.entity.get('metadata')
    print(f"  - 包含图像: {metadata.get('has_images')}")

# ===== 10. 场景3：标量过滤检索 =====
print("\n" + "=" * 70)
print("步骤10: 场景3 - 标量过滤检索")
print("=" * 70)

# 过滤条件：只检索技术部的文档
print("过滤条件: metadata['department'] == '技术部'")

results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=3,
    expr="metadata['department'] == '技术部'",
    output_fields=["title", "metadata"]
)

print(f"\n✅ 过滤检索完成，结果:")
for i, hit in enumerate(results[0], 1):
    print(f"\n结果 {i}:")
    print(f"  - 标题: {hit.entity.get('title')}")
    metadata = hit.entity.get('metadata')
    print(f"  - 部门: {metadata.get('department')}")

# ===== 11. 场景4：ARRAY 标签过滤 =====
print("\n" + "=" * 70)
print("步骤11: 场景4 - ARRAY 标签过滤")
print("=" * 70)

# 过滤条件：包含"性能优化"标签的文档
print("过滤条件: ARRAY_CONTAINS(tags, '性能优化')")

results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=3,
    expr="ARRAY_CONTAINS(tags, '性能优化')",
    output_fields=["title", "tags"]
)

print(f"\n✅ 标签过滤完成，结果:")
for i, hit in enumerate(results[0], 1):
    print(f"\n结果 {i}:")
    print(f"  - 标题: {hit.entity.get('title')}")
    print(f"  - 标签: {hit.entity.get('tags')}")

# ===== 12. 查看 Collection 统计 =====
print("\n" + "=" * 70)
print("步骤12: 查看 Collection 统计信息")
print("=" * 70)

print(f"Collection 名称: {collection.name}")
print(f"记录数: {collection.num_entities}")
print(f"\nSchema 字段:")
for field in collection.schema.fields:
    print(f"  - {field.name}: {field.dtype}")

print("\n" + "=" * 70)
print("🎉 多模态检索演示完成！")
print("=" * 70)
```

---

## 关键特性详解

### 1. 多向量字段设计

```python
# 文本向量
FieldSchema(
    name="text_vector",
    dtype=DataType.FLOAT16_VECTOR,
    dim=768
)

# 图像向量
FieldSchema(
    name="image_vector",
    dtype=DataType.FLOAT16_VECTOR,
    dim=512
)
```

**优势：**
- 支持多模态检索（文本 + 图像）
- 每个向量字段独立索引
- 可以根据场景选择检索字段

### 2. ARRAY 字段（Milvus 2.6）

```python
FieldSchema(
    name="tags",
    dtype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_capacity=50
)

# 插入数据
tags = ["Milvus", "架构", "向量数据库"]

# 过滤查询
expr="ARRAY_CONTAINS(tags, '性能优化')"
```

**使用场景：**
- 文档标签
- 分类列表
- 关键词列表

### 3. JSON 字段

```python
FieldSchema(
    name="metadata",
    dtype=DataType.JSON
)

# 插入数据
metadata = {
    "author": "张三",
    "department": "技术部",
    "created_at": "2026-02-21",
    "page_count": 50
}

# JSON Path 过滤
expr="metadata['department'] == '技术部'"
```

**优势：**
- 灵活存储复杂元数据
- 支持嵌套结构
- 支持 JSON Path 查询

---

## 实际应用场景

### 场景1：企业文档管理系统

```python
# Schema 设计
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    FieldSchema(name="image_vector", dtype=DataType.FLOAT16_VECTOR, dim=512),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]

# 元数据示例
metadata = {
    "author": "张三",
    "department": "技术部",
    "document_type": "技术文档",
    "security_level": "内部",
    "created_at": "2026-02-21",
    "updated_at": "2026-02-21",
    "version": "1.0"
}
```

### 场景2：电商商品检索

```python
# Schema 设计
fields = [
    FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="product_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    FieldSchema(name="image_vector", dtype=DataType.FLOAT16_VECTOR, dim=512),
    FieldSchema(name="categories", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20),
    FieldSchema(name="product_info", dtype=DataType.JSON)
]

# 元数据示例
product_info = {
    "brand": "Apple",
    "price": 8999.00,
    "stock": 100,
    "rating": 4.8,
    "sales_count": 5000,
    "attributes": {
        "color": "银色",
        "storage": "256GB",
        "screen_size": "6.1英寸"
    }
}
```

### 场景3：学术论文检索

```python
# Schema 设计
fields = [
    FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="abstract_vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
    FieldSchema(name="figure_vector", dtype=DataType.FLOAT16_VECTOR, dim=512),
    FieldSchema(name="keywords", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=30),
    FieldSchema(name="paper_metadata", dtype=DataType.JSON)
]

# 元数据示例
paper_metadata = {
    "authors": ["张三", "李四", "王五"],
    "institution": "清华大学",
    "publication_date": "2026-02-21",
    "journal": "Nature",
    "citations": 150,
    "doi": "10.1038/s41586-026-12345-6"
}
```

---

## 性能优化建议

### 1. 向量维度选择

| 模态 | 推荐维度 | 模型示例 |
|------|---------|---------|
| 文本 | 768 | text-embedding-3-small |
| 文本 | 1536 | text-embedding-3-large |
| 图像 | 512 | CLIP ViT-B/32 |
| 图像 | 768 | CLIP ViT-L/14 |

### 2. 存储成本优化

```python
# 场景：100M 文档，文本向量 768 维 + 图像向量 512 维

# FLOAT_VECTOR
# 文本：100M * 768 * 4 = 307 GB
# 图像：100M * 512 * 4 = 205 GB
# 总计：512 GB

# FLOAT16_VECTOR（推荐）
# 文本：100M * 768 * 2 = 154 GB
# 图像：100M * 512 * 2 = 102 GB
# 总计：256 GB（节省 50%）
```

### 3. 标量过滤优化

```python
# ❌ 不推荐：复杂的 JSON 嵌套过滤
expr="metadata['attributes']['color'] == '银色' and metadata['price'] < 10000"

# ✅ 推荐：使用 Partition 预过滤
collection.create_partition("electronics")
collection.create_partition("clothing")

# 检索时指定 Partition
results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    limit=10,
    partition_names=["electronics"]
)
```

---

## 常见问题

### Q1: 多向量字段会影响性能吗？

**A:** 每个向量字段独立索引和检索，不会相互影响。但会增加存储和内存占用。

### Q2: ARRAY 字段的最大容量如何选择？

**A:** 根据实际需求设置，建议不超过 100。过大会影响性能。

### Q3: JSON 字段可以创建索引吗？

**A:** Milvus 2.6 支持 JSON Path Index，可以为 JSON 字段的特定路径创建索引。

```python
# 创建 JSON Path Index
collection.create_index(
    field_name="metadata",
    index_params={
        "index_type": "JSON_PATH_INDEX",
        "params": {
            "json_path": "$.department",
            "json_cast_type": "VARCHAR"
        }
    }
)
```

---

## 下一步

- **生命周期管理**：[07_实战代码_场景3_Collection生命周期管理](./07_实战代码_场景3_Collection生命周期管理.md)
- **返回导航**：[00_概览](./00_概览.md)
