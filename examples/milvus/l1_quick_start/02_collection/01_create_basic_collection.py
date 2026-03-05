"""
Milvus 2.6 基础 Collection 创建 - 文档问答系统
演示：Schema 定义 - Collection 创建 - 数据插入 - 索引创建 - 检索
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

# ===== 1. 连接到 Milvus =====
print("=" * 60)
print("步骤1: 连接到 Milvus 2.6")
print("=" * 60)

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("✅ 已连接到 Milvus")

# ===== 2. 定义 Schema =====
print("\n" + "=" * 60)
print("步骤2: 定义 Collection Schema")
print("=" * 60)

fields = [
    # 主键（自动生成）
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        description="文档片段的唯一标识"
    ),

    # 文档内容
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=512,
        description="文档片段文本内容"
    ),

    # 向量标识（使用 FLOAT16 节省 50% 存储）
    FieldSchema(
        name="vector",
        dtype=DataType.FLOAT16_VECTOR,
        dim=768,
        description="文本的向量表示"
    ),

    # 文档来源
    FieldSchema(
        name="source",
        dtype=DataType.VARCHAR,
        max_length=128,
        description="文档来源"
    ),

    # 时间戳
    FieldSchema(
        name="timestamp",
        dtype=DataType.INT64,
        description="创建时间戳"
    )
]

# 创建 Schema
schema = CollectionSchema(
    fields=fields,
    description="文档问答系统 Collection"
)

print(f"✅ Schema 定义完成")
print(f"   - 字段数量: {len(fields)}")
print(f"   - 向量维度: 768")
print(f"   - 向量类型: FLOAT16_VECTOR（节省 50% 存储）")

# ===== 3. 创建 Collection =====
print("\n" + "=" * 60)
print("步骤3: 创建 Collection")
print("=" * 60)

collection_name = "doc_qa_collection"

# 检查是否已存在
if utility.has_collection(collection_name):
    print(f"⚠️  Collection '{collection_name}' 已存在，删除旧的")
    utility.drop_collection(collection_name)

# 创建 Collection
collection = Collection(
    name=collection_name,
    schema=schema
)

print(f"✅ Collection '{collection_name}' 创建成功")

# ===== 4. 准备示例数据 =====
print("\n" + "=" * 60)
print("步骤4: 准备示例数据")
print("=" * 60)

# 模拟文档数据
documents = [
    {
        "text": "Milvus 是一个开源的向量数据库，专为 AI 应用设计。",
        "source": "milvus_intro.pdf",
        "timestamp": int(time.time())
    },
    {
        "text": "Milvus 2.6 支持 100K collections，适合大规模多租户场景。",
        "source": "milvus_features.pdf",
        "timestamp": int(time.time())
    },
    {
        "text": "FLOAT16_VECTOR 可以节省 50% 的存储空间，精度损失小于 1%。",
        "source": "milvus_optimization.pdf",
        "timestamp": int(time.time())
    },
    {
        "text": "Dynamic Schema 允许在运行时动态添加字段，无需重建 Collection。",
        "source": "milvus_schema.pdf",
        "timestamp": int(time.time())
    },
    {
        "text": "RAG 系统使用 Milvus 存储文档向量，实现语义检索。",
        "source": "rag_guide.pdf",
        "timestamp": int(time.time())
    }
]

# 生成模拟向量 （实际应用中应使用真实的 Embedding 模型）
def generate_mock_vector(text: str, dim: int = 768) -> np.ndarray:
    """生成模拟向量（实际应用中使用 Embedding 模型）

    pymilvus 对 FLOAT16_VECTOR 期望的数据类型为 `np.ndarray(dtype=float16)`。
    """
    np.random.seed(hash(text) % (2**32))  # 保持同一文本生成相同向量
    return np.random.rand(dim).astype(np.float16)

# 准备插入数据
texts = [doc["text"] for doc in documents]
vectors = [generate_mock_vector(text) for text in texts]
sources = [doc["source"] for doc in documents]
timestamps = [doc["timestamp"] for doc in documents]

print(f"✅ 准备了 {len(documents)} 条文档数据")
print(f"   - 文本示例: {texts[0][:50]}...")
print(f"   - 向量维度: {len(vectors[0])}")

# ===== 5. 插入数据 =====
print("\n" + "=" * 60)
print("步骤5: 插入数据到 Collection")
print("=" * 60)

# 插入数据
insert_result = collection.insert([
    texts,
    vectors,
    sources,
    timestamps
])

print(f"✅ 数据插入成功")
print(f"   - 插入记录数: {len(insert_result.primary_keys)}")
print(f"   - 主键示例: {insert_result.primary_keys[:3]}")

# 刷新数据（确保数据持久化）
collection.flush()
print(f"✅ 数据已刷新到磁盘")

# ===== 6. 创建索引 =====
print("\n" + "=" * 60)
print("步骤6: 为向量字段创建索引")
print("=" * 60)

# 定义索引参数
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 256
    }
}

# 创建索引
collection.create_index(
    field_name="vector",
    index_params=index_params
)

print(f"✅ 索引创建成功")
print(f"   - 索引类型: HNSW")
print(f"   - 度量类型: COSINE")
print(f"   - 参数: M=16, efConstruction=256")

# ===== 7. 加载 Collection =====
print("\n" + "=" * 60)
print("步骤7: 加载 Collection 到内存")
print("=" * 60)

collection.load()
print(f"✅ Collection 已加载到内存")

# ===== 8. 执行检索 =====
print("\n" + "=" * 60)
print("步骤8: 执行语义检索")
print("=" * 60)

# 查询文本
query_text = "如何优化 Milvus 的存储空间？"
print(f"查询: {query_text}")

# 生成查询向量（实际应用中使用相同的 Embedding 模型）
query_vector = generate_mock_vector(query_text)

# 执行检索
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}
}

results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param=search_params,
    limit=3,
    output_fields=["text", "source", "timestamp"]
)

print(f"\n✅ 检索完成，返回 Top-{len(results[0])} 结果:")
print("-" * 60)

for i, hit in enumerate(results[0], 1):
    print(f"\n结果 {i}:")
    print(f"  - 相似度: {hit.distance:.4f}")
    print(f"  - 文本: {hit.entity.get('text')}")
    print(f"  - 来源: {hit.entity.get('source')}")
    print(f"  - 时间戳: {hit.entity.get('timestamp')}")

    # ===== 9. 查看 Collection 统计信息 =====
print("\n" + "=" * 60)
print("步骤9: 查看 Collection 统计信息")
print("=" * 60)

print(f"Collection 名称: {collection.name}")
print(f"记录数: {collection.num_entities}")
print(f"加载状态: {'已加载' if collection.is_loaded else '未加载'}")

# 查看 Schema
print(f"\nSchema 字段:")
for field in collection.schema.fields:
    print(f"  - {field.name}: {field.dtype}")

# ===== 10. 清理资源 =====
print("\n" + "=" * 60)
print("步骤10: 清理资源（可选）")
print("=" * 60)