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
import time
import json

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
        max_length=200
    ),

    # 文本向量（FLOAT16，节省内存）
    FieldSchema(
        name="text_vector",
        dtype=DataType.FLOAT16_VECTOR,
        dim=768
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
        # ARRAY<VARCHAR> 需要声明每个字符串元素的最大长度，否则 create_collection 会报错
        max_length=100,
        max_capacity=50,
        description="文档的标签列表"
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

collection_name="multimodal_docs"

if utility.has_collection(collection_name):
    print(f"⚠️ Collection '{collection_name}' 已存在，正在删除...")
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)
print(f"✅ Collection '{collection_name}' 创建成功")

# ===== 4. 准备多模态数据 =====
print("\n" + "=" * 70)
print("步骤4: 准备多模态示例数据")
print("=" * 70)

# 模拟多模态数据
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
def generate_vector(text: str, dim: int) -> np.ndarray:
    """生成模拟向量"""
    np.random.seed(hash(text) % (2**32))
    return np.random.rand(dim).astype(np.float16)

# 准备插入数据
titles = [doc["title"] for doc in documents]
text_vectors = [generate_vector(doc["title"], 768) for doc in documents]
image_vectors= [generate_vector(doc["title"] + "_image", 512) for doc in documents]
tags_list = [doc["tags"] for doc in documents]
metadata_list = [doc["metadata"] for doc in documents]

print(f"✅ 准备了 {len(documents)} 条多模态文档数据")
print(f"   - 文本向量维度: {len(text_vectors[0])}")
print(f"   - 图像向量维度: {len(image_vectors[0])}")
print(f"   - 标签示例: {tags_list[0]}")
print(f"   - 元数据示例: {json.dumps(metadata_list[0], ensure_ascii=False, indent=2)}")

# # ===== 5. 插入数据 =====
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
print(f"查询 {query_text}")

query_vector = generate_vector(query_text, 768)

results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "ef": 64},
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
print(f"查询图像 {query_image}")

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
    param={"metric_type": "COSINE", "ef": 64},
    limit=3,
    output_fields=["title", "metadata"],
    expr="metadata['department'] == '技术部'"
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

# 过滤条件：包含“性能优化”标签的文档
print("过滤条件: ARRAY_CONTAINS(tags, '性能优化')")

results = collection.search(
    data=[query_vector],
    anns_field="text_vector",
    param={"metric_type": "COSINE", "ef": 64},
    limit=3,
    output_fields=["title", "tags"],
    expr="ARRAY_CONTAINS(tags, '性能优化')"
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