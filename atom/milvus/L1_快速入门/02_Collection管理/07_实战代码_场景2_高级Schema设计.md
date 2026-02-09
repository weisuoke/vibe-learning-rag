# 实战代码 - 场景2：高级 Schema 设计

## 场景描述

演示高级 Schema 设计技巧，包括：
- 多向量字段（Multi-vector）
- JSON 字段
- 动态 Schema
- VARCHAR 主键

## 完整代码

```python
"""
场景2：高级 Schema 设计
演示：多向量、JSON 字段、动态 Schema 等高级特性
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
import json

# ===== 1. 连接到 Milvus =====
print("=" * 60)
print("场景2：高级 Schema 设计")
print("=" * 60)

connections.connect(host="localhost", port="19530")
print("✅ 已连接到 Milvus\n")

# ===== 2. 示例1：多向量字段 Collection =====
print("=" * 60)
print("示例1：多向量字段 - 图文混合检索")
print("=" * 60)

# 定义包含多个向量字段的 Schema
multimodal_fields = [
    # VARCHAR 主键
    FieldSchema(
        name="item_id",
        dtype=DataType.VARCHAR,
        max_length=50,
        is_primary=True,
        description="商品唯一标识"
    ),

    # 文本向量
    FieldSchema(
        name="text_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=768,
        description="商品描述的文本向量"
    ),

    # 图片向量
    FieldSchema(
        name="image_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=512,
        description="商品图片的图像向量"
    ),

    # 标量字段
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=200,
        description="商品标题"
    ),

    FieldSchema(
        name="price",
        dtype=DataType.FLOAT,
        description="商品价格"
    ),

    FieldSchema(
        name="category",
        dtype=DataType.VARCHAR,
        max_length=50,
        description="商品分类"
    )
]

multimodal_schema = CollectionSchema(
    fields=multimodal_fields,
    description="图文混合检索 Collection"
)

# 创建 Collection
if utility.has_collection("multimodal_products"):
    utility.drop_collection("multimodal_products")

multimodal_collection = Collection(
    name="multimodal_products",
    schema=multimodal_schema
)

print("✅ 多向量 Collection 创建成功")
print(f"   - 向量字段数量: 2 (text_vector, image_vector)")
print(f"   - 主键类型: VARCHAR")

# 插入数据
products = [
    {
        "item_id": f"PROD_{i:04d}",
        "text_vector": [random.random() for _ in range(768)],
        "image_vector": [random.random() for _ in range(512)],
        "title": f"商品 {i}",
        "price": random.uniform(10.0, 1000.0),
        "category": random.choice(["电子产品", "服装", "食品", "图书"])
    }
    for i in range(50)
]

multimodal_collection.insert(products)
multimodal_collection.flush()
print(f"✅ 插入了 {len(products)} 条数据")

# 为两个向量字段分别创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

# 为 text_vector 创建索引
multimodal_collection.create_index(
    field_name="text_vector",
    index_params=index_params
)
print("✅ text_vector 索引创建成功")

# 为 image_vector 创建索引
multimodal_collection.create_index(
    field_name="image_vector",
    index_params=index_params
)
print("✅ image_vector 索引创建成功")

# 加载 Collection
multimodal_collection.load()
print("✅ Collection 已加载\n")

# 使用 text_vector 检索
print("--- 使用文本向量检索 ---")
query_text_vector = [[random.random() for _ in range(768)]]

text_results = multimodal_collection.search(
    data=query_text_vector,
    anns_field="text_vector",  # 指定使用 text_vector
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["title", "price", "category"]
)

for i, hit in enumerate(text_results[0], 1):
    print(f"  {i}. {hit.entity.get('title')} - ¥{hit.entity.get('price'):.2f} ({hit.entity.get('category')})")

# 使用 image_vector 检索
print("\n--- 使用图像向量检索 ---")
query_image_vector = [[random.random() for _ in range(512)]]

image_results = multimodal_collection.search(
    data=query_image_vector,
    anns_field="image_vector",  # 指定使用 image_vector
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["title", "price", "category"]
)

for i, hit in enumerate(image_results[0], 1):
    print(f"  {i}. {hit.entity.get('title')} - ¥{hit.entity.get('price'):.2f} ({hit.entity.get('category')})")

multimodal_collection.release()
print("\n✅ 示例1完成\n")

# ===== 3. 示例2：JSON 字段 Collection =====
print("=" * 60)
print("示例2：JSON 字段 - 灵活的元数据存储")
print("=" * 60)

# 定义包含 JSON 字段的 Schema
json_fields = [
    FieldSchema(
        name="doc_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
        description="文档ID（自动生成）"
    ),

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=128,
        description="文档向量"
    ),

    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=200,
        description="文档标题"
    ),

    # JSON 字段：存储复杂的元数据
    FieldSchema(
        name="metadata",
        dtype=DataType.JSON,
        description="文档元数据（JSON 格式）"
    )
]

json_schema = CollectionSchema(
    fields=json_fields,
    description="包含 JSON 字段的 Collection"
)

# 创建 Collection
if utility.has_collection("documents_with_json"):
    utility.drop_collection("documents_with_json")

json_collection = Collection(
    name="documents_with_json",
    schema=json_schema
)

print("✅ JSON 字段 Collection 创建成功")

# 插入包含 JSON 数据的记录
documents = [
    {
        "embedding": [random.random() for _ in range(128)],
        "title": f"技术文档 {i}",
        "metadata": {
            "author": f"作者{i}",
            "tags": ["Python", "AI", "RAG"],
            "stats": {
                "views": random.randint(100, 10000),
                "likes": random.randint(10, 1000)
            },
            "published": True,
            "version": "1.0"
        }
    }
    for i in range(20)
]

json_collection.insert(documents)
json_collection.flush()
print(f"✅ 插入了 {len(documents)} 条数据")

# 创建索引
json_collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
)
print("✅ 索引创建成功")

# 加载 Collection
json_collection.load()
print("✅ Collection 已加载\n")

# 查询 JSON 字段
print("--- 查询 JSON 字段 ---")

# 查询：author == "作者5"
results = json_collection.query(
    expr='metadata["author"] == "作者5"',
    output_fields=["doc_id", "title", "metadata"],
    limit=5
)

print(f"查询条件: metadata['author'] == '作者5'")
for result in results:
    print(f"  - ID: {result['doc_id']}")
    print(f"    标题: {result['title']}")
    print(f"    作者: {result['metadata']['author']}")
    print(f"    标签: {result['metadata']['tags']}")
    print(f"    浏览量: {result['metadata']['stats']['views']}")
    print()

json_collection.release()
print("✅ 示例2完成\n")

# ===== 4. 示例3：动态 Schema Collection =====
print("=" * 60)
print("示例3：动态 Schema - 灵活添加字段")
print("=" * 60)

# 定义启用动态字段的 Schema
dynamic_fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False
    ),

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=128
    ),

    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=200
    )
]

dynamic_schema = CollectionSchema(
    fields=dynamic_fields,
    description="动态 Schema Collection",
    enable_dynamic_field=True  # 启用动态字段
)

# 创建 Collection
if utility.has_collection("dynamic_collection"):
    utility.drop_collection("dynamic_collection")

dynamic_collection = Collection(
    name="dynamic_collection",
    schema=dynamic_schema
)

print("✅ 动态 Schema Collection 创建成功")
print("   - 启用动态字段: True")

# 插入数据时可以添加额外字段
dynamic_data = [
    {
        "id": 1,
        "embedding": [random.random() for _ in range(128)],
        "title": "文档1",
        # 动态字段（Schema 中未定义）
        "author": "张三",
        "category": "技术",
        "views": 1000
    },
    {
        "id": 2,
        "embedding": [random.random() for _ in range(128)],
        "title": "文档2",
        # 不同的动态字段
        "author": "李四",
        "tags": ["Python", "AI"],
        "published_date": "2024-01-01"
    },
    {
        "id": 3,
        "embedding": [random.random() for _ in range(128)],
        "title": "文档3",
        # 又是不同的动态字段
        "department": "研发部",
        "priority": "high"
    }
]

dynamic_collection.insert(dynamic_data)
dynamic_collection.flush()
print(f"✅ 插入了 {len(dynamic_data)} 条数据（包含动态字段）")

# 创建索引
dynamic_collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
)

dynamic_collection.load()
print("✅ Collection 已加载\n")

# 查询动态字段
print("--- 查询动态字段 ---")

# 查询所有数据
results = dynamic_collection.query(
    expr="id >= 0",
    output_fields=["*"],  # 返回所有字段（包括动态字段）
    limit=10
)

for result in results:
    print(f"ID: {result['id']}, 标题: {result['title']}")
    # 打印动态字段
    for key, value in result.items():
        if key not in ["id", "title", "embedding"]:
            print(f"  - {key}: {value}")
    print()

dynamic_collection.release()
print("✅ 示例3完成\n")

# ===== 5. 示例4：复杂 Schema 设计 =====
print("=" * 60)
print("示例4：复杂 Schema - 综合应用")
print("=" * 60)

# 定义一个复杂的 Schema（用户画像系统）
complex_fields = [
    # VARCHAR 主键
    FieldSchema(
        name="user_id",
        dtype=DataType.VARCHAR,
        max_length=50,
        is_primary=True,
        description="用户ID"
    ),

    # 多个向量字段
    FieldSchema(
        name="behavior_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=256,
        description="用户行为向量"
    ),

    FieldSchema(
        name="interest_vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=256,
        description="用户兴趣向量"
    ),

    # 标量字段
    FieldSchema(
        name="age",
        dtype=DataType.INT8,
        description="用户年龄"
    ),

    FieldSchema(
        name="gender",
        dtype=DataType.VARCHAR,
        max_length=10,
        description="用户性别"
    ),

    FieldSchema(
        name="vip_level",
        dtype=DataType.INT8,
        description="VIP 等级"
    ),

    FieldSchema(
        name="total_spent",
        dtype=DataType.DOUBLE,
        description="总消费金额"
    ),

    FieldSchema(
        name="is_active",
        dtype=DataType.BOOL,
        description="是否活跃"
    ),

    # JSON 字段
    FieldSchema(
        name="preferences",
        dtype=DataType.JSON,
        description="用户偏好设置"
    ),

    FieldSchema(
        name="last_login",
        dtype=DataType.INT64,
        description="最后登录时间戳"
    )
]

complex_schema = CollectionSchema(
    fields=complex_fields,
    description="用户画像系统",
    enable_dynamic_field=True  # 启用动态字段
)

# 创建 Collection
if utility.has_collection("user_profiles"):
    utility.drop_collection("user_profiles")

complex_collection = Collection(
    name="user_profiles",
    schema=complex_schema
)

print("✅ 复杂 Schema Collection 创建成功")
print(f"   - 字段数量: {len(complex_schema.fields)}")
print(f"   - 向量字段: 2 (behavior_vector, interest_vector)")
print(f"   - JSON 字段: 1 (preferences)")
print(f"   - 动态字段: 启用")

# 插入复杂数据
users = [
    {
        "user_id": f"USER_{i:05d}",
        "behavior_vector": [random.random() for _ in range(256)],
        "interest_vector": [random.random() for _ in range(256)],
        "age": random.randint(18, 60),
        "gender": random.choice(["男", "女"]),
        "vip_level": random.randint(0, 5),
        "total_spent": random.uniform(0, 10000),
        "is_active": random.choice([True, False]),
        "preferences": {
            "language": "zh-CN",
            "theme": random.choice(["light", "dark"]),
            "notifications": True,
            "categories": random.sample(["电子产品", "服装", "食品", "图书", "运动"], 3)
        },
        "last_login": 1700000000 + i * 1000,
        # 动态字段
        "registration_date": f"2024-{random.randint(1, 12):02d}-01",
        "referral_code": f"REF{random.randint(1000, 9999)}"
    }
    for i in range(30)
]

complex_collection.insert(users)
complex_collection.flush()
print(f"✅ 插入了 {len(users)} 条用户数据")

# 为两个向量字段创建索引
for vector_field in ["behavior_vector", "interest_vector"]:
    complex_collection.create_index(
        field_name=vector_field,
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
    )
    print(f"✅ {vector_field} 索引创建成功")

complex_collection.load()
print("✅ Collection 已加载\n")

# 复杂查询
print("--- 复杂查询示例 ---")

# 查询：VIP 用户 + 活跃 + 消费金额 > 5000
results = complex_collection.query(
    expr='vip_level >= 3 and is_active == true and total_spent > 5000',
    output_fields=["user_id", "age", "gender", "vip_level", "total_spent", "preferences"],
    limit=5
)

print("查询条件: VIP >= 3 且活跃且消费 > 5000")
for result in results:
    print(f"  - 用户: {result['user_id']}")
    print(f"    年龄: {result['age']}, 性别: {result['gender']}")
    print(f"    VIP等级: {result['vip_level']}, 消费: ¥{result['total_spent']:.2f}")
    print(f"    偏好主题: {result['preferences']['theme']}")
    print()

complex_collection.release()
print("✅ 示例4完成\n")

# ===== 6. 总结 =====
print("=" * 60)
print("总结：高级 Schema 设计")
print("=" * 60)

print("""
✅ 示例1：多向量字段
   - 支持多个向量字段（文本 + 图像）
   - 可以针对不同向量字段进行检索
   - 适用于多模态检索场景

✅ 示例2：JSON 字段
   - 存储复杂的嵌套数据结构
   - 支持 JSON 路径查询
   - 适用于元数据不固定的场景

✅ 示例3：动态 Schema
   - 允许插入时添加额外字段
   - 灵活应对需求变化
   - 适用于快速迭代的项目

✅ 示例4：复杂 Schema
   - 综合应用多种字段类型
   - VARCHAR 主键 + 多向量 + JSON + 动态字段
   - 适用于复杂的业务场景

关键要点：
1. 多向量字段：每个向量字段需要单独创建索引
2. JSON 字段：灵活但查询性能较低
3. 动态 Schema：方便但需要权衡性能
4. VARCHAR 主键：适用于需要字符串ID的场景
""")

print("🎉 场景2完成！")
```

## 运行输出示例

```
============================================================
场景2：高级 Schema 设计
============================================================
✅ 已连接到 Milvus

============================================================
示例1：多向量字段 - 图文混合检索
============================================================
✅ 多向量 Collection 创建成功
   - 向量字段数量: 2 (text_vector, image_vector)
   - 主键类型: VARCHAR
✅ 插入了 50 条数据
✅ text_vector 索引创建成功
✅ image_vector 索引创建成功
✅ Collection 已加载

--- 使用文本向量检索 ---
  1. 商品 23 - ¥456.78 (电子产品)
  2. 商品 45 - ¥123.45 (服装)
  3. 商品 12 - ¥789.01 (食品)

--- 使用图像向量检索 ---
  1. 商品 34 - ¥234.56 (图书)
  2. 商品 8 - ¥567.89 (电子产品)
  3. 商品 19 - ¥345.67 (服装)

✅ 示例1完成

============================================================
示例2：JSON 字段 - 灵活的元数据存储
============================================================
✅ JSON 字段 Collection 创建成功
✅ 插入了 20 条数据
✅ 索引创建成功
✅ Collection 已加载

--- 查询 JSON 字段 ---
查询条件: metadata['author'] == '作者5'
  - ID: 5
    标题: 技术文档 5
    作者: 作者5
    标签: ['Python', 'AI', 'RAG']
    浏览量: 5432

✅ 示例2完成

============================================================
示例3：动态 Schema - 灵活添加字段
============================================================
✅ 动态 Schema Collection 创建成功
   - 启用动态字段: True
✅ 插入了 3 条数据（包含动态字段）
✅ Collection 已加载

--- 查询动态字段 ---
ID: 1, 标题: 文档1
  - author: 张三
  - category: 技术
  - views: 1000

ID: 2, 标题: 文档2
  - author: 李四
  - tags: ['Python', 'AI']
  - published_date: 2024-01-01

ID: 3, 标题: 文档3
  - department: 研发部
  - priority: high

✅ 示例3完成

============================================================
示例4：复杂 Schema - 综合应用
============================================================
✅ 复杂 Schema Collection 创建成功
   - 字段数量: 10
   - 向量字段: 2 (behavior_vector, interest_vector)
   - JSON 字段: 1 (preferences)
   - 动态字段: 启用
✅ 插入了 30 条用户数据
✅ behavior_vector 索引创建成功
✅ interest_vector 索引创建成功
✅ Collection 已加载

--- 复杂查询示例 ---
查询条件: VIP >= 3 且活跃且消费 > 5000
  - 用户: USER_00012
    年龄: 35, 性别: 男
    VIP等级: 4, 消费: ¥7234.56
    偏好主题: dark

  - 用户: USER_00023
    年龄: 28, 性别: 女
    VIP等级: 5, 消费: ¥8901.23
    偏好主题: light

✅ 示例4完成

============================================================
总结：高级 Schema 设计
============================================================

✅ 示例1：多向量字段
   - 支持多个向量字段（文本 + 图像）
   - 可以针对不同向量字段进行检索
   - 适用于多模态检索场景

✅ 示例2：JSON 字段
   - 存储复杂的嵌套数据结构
   - 支持 JSON 路径查询
   - 适用于元数据不固定的场景

✅ 示例3：动态 Schema
   - 允许插入时添加额外字段
   - 灵活应对需求变化
   - 适用于快速迭代的项目

✅ 示例4：复杂 Schema
   - 综合应用多种字段类型
   - VARCHAR 主键 + 多向量 + JSON + 动态字段
   - 适用于复杂的业务场景

关键要点：
1. 多向量字段：每个向量字段需要单独创建索引
2. JSON 字段：灵活但查询性能较低
3. 动态 Schema：方便但需要权衡性能
4. VARCHAR 主键：适用于需要字符串ID的场景

🎉 场景2完成！
```

## 关键要点

1. **多向量字段**：一个 Collection 可以包含多个向量字段
2. **JSON 字段**：存储复杂的嵌套数据结构
3. **动态 Schema**：允许灵活添加字段
4. **VARCHAR 主键**：适用于字符串ID场景
5. **综合应用**：根据业务需求组合使用各种特性
