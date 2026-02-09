# 实战代码 - 场景1：基础 Collection 创建与管理

## 场景描述

演示如何创建一个基础的 Collection，包括：
- 定义 Schema
- 创建 Collection
- 插入数据
- 创建索引
- 执行检索

## 完整代码

```python
"""
场景1：基础 Collection 创建与管理
演示：从零开始创建一个文档检索 Collection
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import random

# ===== 1. 连接到 Milvus =====
print("=" * 50)
print("步骤1：连接到 Milvus")
print("=" * 50)

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
print("✅ 已连接到 Milvus")

# ===== 2. 定义 Schema =====
print("\n" + "=" * 50)
print("步骤2：定义 Collection Schema")
print("=" * 50)

# 定义字段
fields = [
    # 主键字段
    FieldSchema(
        name="doc_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False,
        description="文档唯一标识"
    ),

    # 向量字段
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=128,  # 向量维度
        description="文档的向量表示"
    ),

    # 标量字段：文档标题
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=200,
        description="文档标题"
    ),

    # 标量字段：文档分类
    FieldSchema(
        name="category",
        dtype=DataType.VARCHAR,
        max_length=50,
        description="文档分类"
    ),

    # 标量字段：创建时间
    FieldSchema(
        name="created_at",
        dtype=DataType.INT64,
        description="创建时间戳"
    )
]

# 创建 Schema
schema = CollectionSchema(
    fields=fields,
    description="文档检索 Collection",
    enable_dynamic_field=False
)

print("✅ Schema 定义完成")
print(f"   - 字段数量: {len(schema.fields)}")
print(f"   - 主键字段: {schema.primary_field.name}")

# ===== 3. 创建 Collection =====
print("\n" + "=" * 50)
print("步骤3：创建 Collection")
print("=" * 50)

collection_name = "documents"

# 检查是否已存在
if utility.has_collection(collection_name):
    print(f"⚠️  Collection '{collection_name}' 已存在，先删除")
    utility.drop_collection(collection_name)

# 创建 Collection
collection = Collection(
    name=collection_name,
    schema=schema,
    using="default"
)

print(f"✅ Collection '{collection_name}' 创建成功")

# ===== 4. 插入数据 =====
print("\n" + "=" * 50)
print("步骤4：插入数据")
print("=" * 50)

# 生成示例数据
num_entities = 100
data = []

categories = ["技术", "产品", "设计", "运营", "市场"]

for i in range(num_entities):
    data.append({
        "doc_id": i,
        "embedding": [random.random() for _ in range(128)],
        "title": f"文档标题 {i}",
        "category": random.choice(categories),
        "created_at": 1700000000 + i * 1000
    })

# 插入数据
insert_result = collection.insert(data)
print(f"✅ 插入了 {len(data)} 条数据")
print(f"   - 插入的 ID 范围: {insert_result.primary_keys[0]} - {insert_result.primary_keys[-1]}")

# 刷新数据（确保数据持久化）
collection.flush()
print("✅ 数据已刷新到磁盘")

# 查看数据量
print(f"   - Collection 中的数据量: {collection.num_entities}")

# ===== 5. 创建索引 =====
print("\n" + "=" * 50)
print("步骤5：创建索引")
print("=" * 50)

# 定义索引参数
index_params = {
    "index_type": "IVF_FLAT",  # 索引类型
    "metric_type": "L2",       # 距离度量
    "params": {"nlist": 128}   # 索引参数
}

# 创建索引
collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print("✅ 索引创建成功")
print(f"   - 索引类型: {index_params['index_type']}")
print(f"   - 距离度量: {index_params['metric_type']}")

# ===== 6. 加载 Collection =====
print("\n" + "=" * 50)
print("步骤6：加载 Collection 到内存")
print("=" * 50)

collection.load()
print("✅ Collection 已加载到内存")

# 检查加载状态
from pymilvus import utility
load_state = utility.load_state(collection_name)
print(f"   - 加载状态: {load_state}")

# ===== 7. 执行检索 =====
print("\n" + "=" * 50)
print("步骤7：执行向量检索")
print("=" * 50)

# 生成查询向量
query_vector = [[random.random() for _ in range(128)]]

# 定义检索参数
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

# 执行检索
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["title", "category", "created_at"]
)

print(f"✅ 检索完成，返回 Top-{len(results[0])} 结果：")
print()

for i, hit in enumerate(results[0], 1):
    print(f"结果 {i}:")
    print(f"  - ID: {hit.id}")
    print(f"  - 距离: {hit.distance:.4f}")
    print(f"  - 标题: {hit.entity.get('title')}")
    print(f"  - 分类: {hit.entity.get('category')}")
    print(f"  - 创建时间: {hit.entity.get('created_at')}")
    print()

# ===== 8. 标量查询 =====
print("=" * 50)
print("步骤8：执行标量查询")
print("=" * 50)

# 查询特定分类的文档
query_expr = 'category == "技术"'

query_results = collection.query(
    expr=query_expr,
    output_fields=["doc_id", "title", "category"],
    limit=5
)

print(f"✅ 查询完成，找到 {len(query_results)} 条结果：")
print()

for result in query_results:
    print(f"  - ID: {result['doc_id']}, 标题: {result['title']}, 分类: {result['category']}")

# ===== 9. 查看 Collection 信息 =====
print("\n" + "=" * 50)
print("步骤9：查看 Collection 信息")
print("=" * 50)

print(f"Collection 名称: {collection.name}")
print(f"Collection 描述: {collection.description}")
print(f"数据量: {collection.num_entities}")
print(f"是否为空: {collection.is_empty}")

print("\n字段列表:")
for field in collection.schema.fields:
    print(f"  - {field.name} ({field.dtype})")
    if field.is_primary:
        print(f"    [主键]")
    if field.dtype == DataType.FLOAT_VECTOR:
        print(f"    维度: {field.params.get('dim')}")
    if field.dtype == DataType.VARCHAR:
        print(f"    最大长度: {field.params.get('max_length')}")

# ===== 10. 释放 Collection =====
print("\n" + "=" * 50)
print("步骤10：释放 Collection")
print("=" * 50)

collection.release()
print("✅ Collection 已从内存释放")

# ===== 11. 清理（可选）=====
print("\n" + "=" * 50)
print("步骤11：清理资源（可选）")
print("=" * 50)

# 如果需要删除 Collection，取消下面的注释
# utility.drop_collection(collection_name)
# print(f"✅ Collection '{collection_name}' 已删除")

print("\n" + "=" * 50)
print("🎉 完成！")
print("=" * 50)
```

## 运行输出示例

```
==================================================
步骤1：连接到 Milvus
==================================================
✅ 已连接到 Milvus

==================================================
步骤2：定义 Collection Schema
==================================================
✅ Schema 定义完成
   - 字段数量: 5
   - 主键字段: doc_id

==================================================
步骤3：创建 Collection
==================================================
✅ Collection 'documents' 创建成功

==================================================
步骤4：插入数据
==================================================
✅ 插入了 100 条数据
   - 插入的 ID 范围: 0 - 99
✅ 数据已刷新到磁盘
   - Collection 中的数据量: 100

==================================================
步骤5：创建索引
==================================================
✅ 索引创建成功
   - 索引类型: IVF_FLAT
   - 距离度量: L2

==================================================
步骤6：加载 Collection 到内存
==================================================
✅ Collection 已加载到内存
   - 加载状态: LoadState.Loaded

==================================================
步骤7：执行向量检索
==================================================
✅ 检索完成，返回 Top-5 结果：

结果 1:
  - ID: 42
  - 距离: 12.3456
  - 标题: 文档标题 42
  - 分类: 技术
  - 创建时间: 1700042000

结果 2:
  - ID: 15
  - 距离: 13.7890
  - 标题: 文档标题 15
  - 分类: 产品
  - 创建时间: 1700015000

...

==================================================
步骤8：执行标量查询
==================================================
✅ 查询完成，找到 5 条结果：

  - ID: 5, 标题: 文档标题 5, 分类: 技术
  - ID: 12, 标题: 文档标题 12, 分类: 技术
  - ID: 23, 标题: 文档标题 23, 分类: 技术
  - ID: 34, 标题: 文档标题 34, 分类: 技术
  - ID: 45, 标题: 文档标题 45, 分类: 技术

==================================================
步骤9：查看 Collection 信息
==================================================
Collection 名称: documents
Collection 描述: 文档检索 Collection
数据量: 100
是否为空: False

字段列表:
  - doc_id (DataType.INT64)
    [主键]
  - embedding (DataType.FLOAT_VECTOR)
    维度: 128
  - title (DataType.VARCHAR)
    最大长度: 200
  - category (DataType.VARCHAR)
    最大长度: 50
  - created_at (DataType.INT64)

==================================================
步骤10：释放 Collection
==================================================
✅ Collection 已从内存释放

==================================================
步骤11：清理资源（可选）
==================================================

==================================================
🎉 完成！
==================================================
```

## 关键要点

1. **完整流程**：从连接到检索的完整流程
2. **Schema 设计**：包含主键、向量、标量字段
3. **索引创建**：必须在检索前创建索引
4. **加载到内存**：必须在检索前加载
5. **两种查询**：向量检索 + 标量查询

## 下一步

- 场景2：高级 Schema 设计
- 场景3：多 Collection 管理
- 场景4：Collection 生命周期管理
