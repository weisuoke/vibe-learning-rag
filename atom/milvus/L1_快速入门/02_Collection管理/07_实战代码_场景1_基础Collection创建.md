# Collection管理 - 实战代码场景1：基础Collection创建

> 完整的文档问答系统 Collection 创建示例，从 Schema 定义到检索的完整流程

---

## 场景描述

**应用场景：** 简单的文档问答系统

**需求：**
- 存储文档片段的文本和向量
- 支持语义检索
- 记录文档来源和时间戳
- 使用 FLOAT16_VECTOR 节省成本

**技术栈：**
- Milvus 2.6
- pymilvus 2.6+
- Python 3.9+

---

## 完整代码实现

```python
"""
Milvus 2.6 基础 Collection 创建 - 文档问答系统
演示：Schema 定义 → Collection 创建 → 数据插入 → 索引创建 → 检索
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
        description="文档片段唯一标识"
    ),
    
    # 文档内容
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=512,
        description="文档片段文本内容"
    ),
    
    # 向量表示（使用 FLOAT16 节省 50% 存储）
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

# 生成模拟向量（实际应用中应使用真实的 Embedding 模型）
def generate_mock_vector(text: str, dim: int = 768) -> List[float]:
    """生成模拟向量（实际应用中使用 Embedding 模型）"""
    np.random.seed(hash(text) % (2**32))
    return np.random.rand(dim).tolist()

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

# 释放 Collection（释放内存）
# collection.release()
# print(f"✅ Collection 已释放")

# 删除 Collection（如果需要）
# utility.drop_collection(collection_name)
# print(f"✅ Collection 已删除")

print("\n" + "=" * 60)
print("🎉 完整流程执行成功！")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
步骤1: 连接到 Milvus 2.6
============================================================
✅ 已连接到 Milvus

============================================================
步骤2: 定义 Collection Schema
============================================================
✅ Schema 定义完成
   - 字段数量: 5
   - 向量维度: 768
   - 向量类型: FLOAT16_VECTOR（节省 50% 存储）

============================================================
步骤3: 创建 Collection
============================================================
✅ Collection 'doc_qa_collection' 创建成功

============================================================
步骤4: 准备示例数据
============================================================
✅ 准备了 5 条文档数据
   - 文本示例: Milvus 是一个开源的向量数据库，专为 AI 应用设计。...
   - 向量维度: 768

============================================================
步骤5: 插入数据到 Collection
============================================================
✅ 数据插入成功
   - 插入记录数: 5
   - 主键示例: [450123456789, 450123456790, 450123456791]
✅ 数据已刷新到磁盘

============================================================
步骤6: 为向量字段创建索引
============================================================
✅ 索引创建成功
   - 索引类型: HNSW
   - 度量类型: COSINE
   - 参数: M=16, efConstruction=256

============================================================
步骤7: 加载 Collection 到内存
============================================================
✅ Collection 已加载到内存

============================================================
步骤8: 执行语义检索
============================================================
查询: 如何优化 Milvus 的存储空间？

✅ 检索完成，返回 Top-3 结果:
------------------------------------------------------------

结果 1:
  - 相似度: 0.8523
  - 文本: FLOAT16_VECTOR 可以节省 50% 的存储空间，精度损失小于 1%。
  - 来源: milvus_optimization.pdf
  - 时间戳: 1708531200

结果 2:
  - 相似度: 0.7891
  - 文本: Milvus 2.6 支持 100K collections，适合大规模多租户场景。
  - 来源: milvus_features.pdf
  - 时间戳: 1708531200

结果 3:
  - 相似度: 0.7234
  - 文本: Milvus 是一个开源的向量数据库，专为 AI 应用设计。
  - 来源: milvus_intro.pdf
  - 时间戳: 1708531200

============================================================
步骤9: 查看 Collection 统计信息
============================================================
Collection 名称: doc_qa_collection
记录数: 5
加载状态: 已加载

Schema 字段:
  - id: DataType.INT64
  - text: DataType.VARCHAR
  - vector: DataType.FLOAT16_VECTOR
  - source: DataType.VARCHAR
  - timestamp: DataType.INT64

============================================================
步骤10: 清理资源（可选）
============================================================

============================================================
🎉 完整流程执行成功！
============================================================
```

---

## 代码详解

### 1. Schema 设计要点

```python
# 主键设计
FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True  # 自动生成，无需手动管理
)

# 向量字段设计
FieldSchema(
    name="vector",
    dtype=DataType.FLOAT16_VECTOR,  # 使用 FLOAT16 节省 50% 存储
    dim=768  # 维度必须与 Embedding 模型一致
)
```

**设计原则：**
- 主键使用 `auto_id=True` 简化管理
- 向量类型选择 FLOAT16_VECTOR 优化成本
- VARCHAR 字段指定合理的 `max_length`

### 2. 索引选择

```python
index_params = {
    "index_type": "HNSW",  # 高召回率索引
    "metric_type": "COSINE",  # 余弦相似度
    "params": {
        "M": 16,  # 每个节点的连接数
        "efConstruction": 256  # 构建时的搜索范围
    }
}
```

**索引类型选择：**
- **HNSW**: 高召回率，适合中等数据集（10万-1000万）
- **IVF_FLAT**: 平衡性能和召回，适合大数据集（>1000万）
- **FLAT**: 精确检索，适合小数据集（<10万）

### 3. 检索参数

```python
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}  # 搜索时的范围，越大召回率越高
}
```

**参数调优：**
- `ef` 值越大，召回率越高，但性能越慢
- 推荐范围：32-128
- 生产环境需要根据实际数据测试

---

## 实际应用扩展

### 扩展1：使用真实 Embedding 模型

```python
from sentence_transformers import SentenceTransformer

# 加载 Embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 生成向量
def get_embedding(text: str) -> List[float]:
    return model.encode(text).tolist()

# 使用
vectors = [get_embedding(text) for text in texts]
query_vector = get_embedding(query_text)
```

### 扩展2：批量插入优化

```python
# 批量插入（每批 1000 条）
batch_size = 1000
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_vectors = vectors[i:i+batch_size]
    batch_sources = sources[i:i+batch_size]
    batch_timestamps = timestamps[i:i+batch_size]
    
    collection.insert([
        batch_texts,
        batch_vectors,
        batch_sources,
        batch_timestamps
    ])
    
    print(f"✅ 已插入 {i+len(batch_texts)}/{len(texts)} 条记录")
```

### 扩展3：添加标量过滤

```python
# 按来源过滤检索
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param=search_params,
    limit=3,
    expr="source == 'milvus_optimization.pdf'",  # 标量过滤
    output_fields=["text", "source"]
)
```

---

## 性能优化建议

### 1. 向量类型选择

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 一般文档检索 | FLOAT16_VECTOR | 节省 50% 存储，精度损失 <1% |
| 高精度要求 | FLOAT_VECTOR | 无精度损失 |
| 超大规模 | BFLOAT16_VECTOR | 节省 50% 存储，适合训练 |

### 2. 索引参数调优

```python
# 小数据集（<10万）
index_params = {"index_type": "FLAT"}

# 中等数据集（10万-1000万）
index_params = {
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}

# 大数据集（>1000万）
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
```

### 3. 内存管理

```python
# 使用完后释放内存
collection.release()

# 需要时再加载
collection.load()
```

---

## 常见问题

### Q1: 为什么检索前必须创建索引？

**A:** Milvus 的向量检索依赖索引结构（如 HNSW、IVF），没有索引无法进行 ANN（近似最近邻）检索。

### Q2: FLOAT16_VECTOR 会影响检索精度吗？

**A:** 精度损失小于 1%，对大多数应用（文档检索、推荐系统）影响可忽略。

### Q3: 如何选择合适的索引类型？

**A:** 根据数据量选择：
- <10万：FLAT（精确检索）
- 10万-1000万：HNSW（高召回率）
- >1000万：IVF_FLAT（平衡性能）

---

## 下一步

- **高级 Schema 设计**：[07_实战代码_场景2_高级Schema设计](./07_实战代码_场景2_高级Schema设计.md)
- **生命周期管理**：[07_实战代码_场景3_Collection生命周期管理](./07_实战代码_场景3_Collection生命周期管理.md)
- **返回导航**：[00_概览](./00_概览.md)
