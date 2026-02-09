# 核心概念 2：Collection CRUD 操作

## Collection 的生命周期

Collection 的生命周期包含以下核心操作：

```
创建 (Create) → 加载 (Load) → 使用 (Use) → 释放 (Release) → 删除 (Drop)
```

## 1. 创建 Collection (Create)

### 基本创建

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

# 连接到 Milvus
connections.connect(alias="default", host="localhost", port="19530")

# ===== 1. 定义 Schema =====
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200)

schema = CollectionSchema(
    fields=[id_field, vector_field, title_field],
    description="文档 Collection"
)

# ===== 2. 创建 Collection =====
collection = Collection(
    name="documents",
    schema=schema,
    using="default"  # 使用哪个连接（默认可省略）
)

print(f"Collection '{collection.name}' 创建成功！")
print(f"字段数量: {len(collection.schema.fields)}")
```

**输出：**
```
Collection 'documents' 创建成功！
字段数量: 3
```

### 创建时的常见参数

```python
collection = Collection(
    name="documents",           # Collection 名称（必需）
    schema=schema,              # Schema 定义（必需）
    using="default",            # 连接别名（可选）
    shards_num=2,               # 分片数量（默认 2）
    consistency_level="Strong"  # 一致性级别（默认 Bounded）
)
```

**分片数量（shards_num）：**
- 控制数据分布的并行度
- 更多分片 = 更高的写入吞吐量
- 建议：小数据集用 2，大数据集用 4-8

**一致性级别（consistency_level）：**
- `Strong`：强一致性（最慢，最准确）
- `Bounded`：有界一致性（默认，平衡）
- `Eventually`：最终一致性（最快，可能有延迟）

## 2. 查看 Collection (Read)

### 检查 Collection 是否存在

```python
from pymilvus import utility

# 检查 Collection 是否存在
exists = utility.has_collection("documents")
print(f"Collection 'documents' 存在: {exists}")

# 列出所有 Collection
collections = utility.list_collections()
print(f"所有 Collection: {collections}")
```

**输出：**
```
Collection 'documents' 存在: True
所有 Collection: ['documents', 'images', 'users']
```

### 获取 Collection 信息

```python
# 方式1：通过 Collection 对象
collection = Collection("documents")
print(f"Collection 名称: {collection.name}")
print(f"Collection 描述: {collection.description}")
print(f"字段数量: {len(collection.schema.fields)}")
print(f"主键字段: {collection.schema.primary_field.name}")

# 方式2：通过 utility
from pymilvus import utility

# 获取 Collection 统计信息
stats = utility.get_collection_stats("documents")
print(f"Collection 统计: {stats}")
```

### 查看 Collection 的 Schema

```python
collection = Collection("documents")

# 遍历所有字段
for field in collection.schema.fields:
    print(f"字段名: {field.name}")
    print(f"  类型: {field.dtype}")
    print(f"  是否主键: {field.is_primary}")
    if field.dtype == DataType.FLOAT_VECTOR:
        print(f"  向量维度: {field.params.get('dim')}")
    if field.dtype == DataType.VARCHAR:
        print(f"  最大长度: {field.params.get('max_length')}")
    print()
```

**输出：**
```
字段名: id
  类型: DataType.INT64
  是否主键: True

字段名: embedding
  类型: DataType.FLOAT_VECTOR
  是否主键: False
  向量维度: 768

字段名: title
  类型: DataType.VARCHAR
  是否主键: False
  最大长度: 200
```

### 查看 Collection 的加载状态

```python
from pymilvus import utility

# 检查 Collection 是否已加载到内存
is_loaded = utility.load_state("documents")
print(f"Collection 加载状态: {is_loaded}")

# 可能的状态：
# - LoadState.NotLoad: 未加载
# - LoadState.Loading: 加载中
# - LoadState.Loaded: 已加载
```

## 3. 加载与释放 Collection

### 为什么需要加载？

**Milvus 的数据存储机制：**
- 数据持久化存储在磁盘上
- 检索时需要将数据加载到内存
- 加载后才能执行 Search 和 Query 操作

**类比：**
- 就像图书馆的书架（磁盘）和阅览桌（内存）
- 要读书，必须先从书架拿到阅览桌
- 读完后可以放回书架，释放阅览桌空间

### 加载 Collection

```python
# ===== 加载整个 Collection =====
collection = Collection("documents")
collection.load()
print("Collection 已加载到内存")

# 检查加载状态
from pymilvus import utility
state = utility.load_state("documents")
print(f"加载状态: {state}")
```

### 释放 Collection

```python
# ===== 释放 Collection（从内存中卸载）=====
collection.release()
print("Collection 已从内存释放")

# 释放后无法执行检索操作
# 需要重新加载才能检索
```

**何时释放？**
- 内存不足时
- 长时间不使用的 Collection
- 需要切换到其他 Collection 时

## 4. 删除 Collection (Delete)

### 删除 Collection

```python
from pymilvus import utility

# 方式1：通过 Collection 对象
collection = Collection("documents")
collection.drop()
print("Collection 已删除")

# 方式2：通过 utility
utility.drop_collection("documents")
print("Collection 已删除")
```

**⚠️ 警告：**
- 删除操作不可逆！
- 所有数据和索引都会被永久删除
- 删除前请确认备份

### 安全删除模式

```python
from pymilvus import utility, Collection

def safe_drop_collection(collection_name: str):
    """安全删除 Collection（带确认）"""

    # 1. 检查 Collection 是否存在
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' 不存在")
        return

    # 2. 获取 Collection 信息
    collection = Collection(collection_name)
    num_entities = collection.num_entities

    # 3. 确认删除
    print(f"警告：即将删除 Collection '{collection_name}'")
    print(f"  - 数据量: {num_entities} 条")
    print(f"  - 字段数: {len(collection.schema.fields)}")

    confirm = input("确认删除？(yes/no): ")
    if confirm.lower() == "yes":
        collection.drop()
        print(f"Collection '{collection_name}' 已删除")
    else:
        print("取消删除")

# 使用
safe_drop_collection("documents")
```

## 5. Collection 的常用属性和方法

### 常用属性

```python
collection = Collection("documents")

# 基本信息
print(f"名称: {collection.name}")
print(f"描述: {collection.description}")
print(f"Schema: {collection.schema}")

# 数据统计
print(f"数据量: {collection.num_entities}")
print(f"是否为空: {collection.is_empty}")

# 索引信息
print(f"索引: {collection.indexes}")
```

### 常用方法

```python
# 数据操作
collection.insert(data)          # 插入数据
collection.delete(expr)          # 删除数据
collection.search(...)           # 向量检索
collection.query(expr)           # 标量查询

# 索引操作
collection.create_index(...)     # 创建索引
collection.drop_index()          # 删除索引
collection.has_index()           # 检查索引是否存在

# 生命周期管理
collection.load()                # 加载到内存
collection.release()             # 从内存释放
collection.drop()                # 删除 Collection

# 分区操作（高级）
collection.create_partition(...)  # 创建分区
collection.drop_partition(...)    # 删除分区
collection.has_partition(...)     # 检查分区是否存在
```

## 6. Collection 操作的最佳实践

### 1. 命名规范

```python
# ✅ 推荐：使用小写字母和下划线
"user_profiles", "product_embeddings", "document_vectors"

# ❌ 避免：使用特殊字符、空格、中文
"user-profiles", "product embeddings", "文档向量"

# ✅ 推荐：有意义的名称
"tech_docs", "customer_reviews", "image_features"

# ❌ 避免：无意义的名称
"collection1", "test", "data"
```

### 2. 创建前检查

```python
from pymilvus import utility, Collection

def create_collection_safe(name: str, schema: CollectionSchema):
    """安全创建 Collection（避免重复）"""

    # 检查是否已存在
    if utility.has_collection(name):
        print(f"Collection '{name}' 已存在")

        # 选择：返回现有 Collection 或删除重建
        choice = input("1. 使用现有 Collection\n2. 删除并重建\n选择: ")

        if choice == "1":
            return Collection(name)
        elif choice == "2":
            utility.drop_collection(name)
            print(f"已删除旧 Collection '{name}'")
        else:
            return None

    # 创建新 Collection
    collection = Collection(name=name, schema=schema)
    print(f"Collection '{name}' 创建成功")
    return collection
```

### 3. 资源管理

```python
# ✅ 推荐：使用完后释放内存
collection = Collection("documents")
collection.load()

# ... 执行检索操作 ...

collection.release()  # 释放内存

# ✅ 推荐：使用上下文管理器（如果支持）
# 注意：pymilvus 目前不直接支持 with 语句，需要手动管理
```

### 4. 错误处理

```python
from pymilvus import Collection, utility
from pymilvus.exceptions import CollectionNotExistException

def get_collection_safe(name: str):
    """安全获取 Collection（带错误处理）"""
    try:
        if not utility.has_collection(name):
            raise CollectionNotExistException(f"Collection '{name}' 不存在")

        collection = Collection(name)
        return collection

    except CollectionNotExistException as e:
        print(f"错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# 使用
collection = get_collection_safe("documents")
if collection:
    print(f"成功获取 Collection: {collection.name}")
```

## 7. 在 RAG 中的应用

### 场景1：多租户知识库

```python
# 为每个租户创建独立的 Collection
def create_tenant_collection(tenant_id: str, schema: CollectionSchema):
    """为租户创建专属 Collection"""
    collection_name = f"tenant_{tenant_id}_docs"

    if utility.has_collection(collection_name):
        print(f"租户 {tenant_id} 的 Collection 已存在")
        return Collection(collection_name)

    collection = Collection(name=collection_name, schema=schema)
    print(f"为租户 {tenant_id} 创建 Collection: {collection_name}")
    return collection

# 使用
tenant_schema = CollectionSchema(fields=[...])
tenant_a_collection = create_tenant_collection("tenant_a", tenant_schema)
tenant_b_collection = create_tenant_collection("tenant_b", tenant_schema)
```

### 场景2：按时间归档

```python
from datetime import datetime

def create_monthly_collection(base_name: str, schema: CollectionSchema):
    """按月创建 Collection（用于归档）"""
    month = datetime.now().strftime("%Y%m")
    collection_name = f"{base_name}_{month}"

    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"创建月度 Collection: {collection_name}")
        return collection

    return Collection(collection_name)

# 使用
current_collection = create_monthly_collection("documents", schema)
```

### 场景3：开发/测试/生产环境隔离

```python
import os

def get_collection_by_env(base_name: str, schema: CollectionSchema):
    """根据环境获取 Collection"""
    env = os.getenv("ENV", "dev")  # dev, test, prod
    collection_name = f"{env}_{base_name}"

    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"创建 {env} 环境 Collection: {collection_name}")
        return collection

    return Collection(collection_name)

# 使用
# 开发环境: dev_documents
# 测试环境: test_documents
# 生产环境: prod_documents
collection = get_collection_by_env("documents", schema)
```

## 总结

**Collection CRUD 操作的核心要点：**

1. **创建（Create）**：定义 Schema → 创建 Collection
2. **查看（Read）**：检查存在 → 获取信息 → 查看 Schema
3. **加载（Load）**：加载到内存 → 执行检索 → 释放内存
4. **删除（Drop）**：确认备份 → 安全删除

**关键原则：**
- 创建前检查是否存在
- 检索前必须加载到内存
- 删除前务必确认备份
- 合理命名，便于管理

**下一步：** 了解 Collection 中的字段类型（核心概念 3）
