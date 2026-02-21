# Collection管理 - 核心概念2：Collection CRUD操作

> 掌握 Collection 的创建、读取、更新、删除操作，理解 Milvus 2.6 的 100K collections 能力

---

## CRUD 操作概览

**CRUD** 是数据库操作的四大基本操作：
- **C**reate（创建）：创建新的 Collection
- **R**ead（读取）：查看 Collection 信息
- **U**pdate（更新）：修改 Collection（Milvus 2.6 支持重命名）
- **D**elete（删除）：删除 Collection

**Milvus 2.6 的特殊性：**
- 支持 100K Collections（10 万个 Collection）
- 支持 rename_collection()（重命名操作）
- 轻量级设计，按需加载（Lazy Loading）

---

## Create 操作：创建 Collection

### 基本创建流程

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Document collection")

# 3. 创建 Collection
collection = Collection(name="my_documents", schema=schema)

print(f"✅ Collection '{collection.name}' 创建成功")
```

### 创建前检查是否存在

```python
from pymilvus import utility

collection_name = "my_documents"

# 检查是否已存在
if utility.has_collection(collection_name):
    print(f"⚠️  Collection '{collection_name}' 已存在")
    # 选项1：使用现有 Collection
    collection = Collection(collection_name)
    # 选项2：删除后重建
    # utility.drop_collection(collection_name)
    # collection = Collection(collection_name, schema)
else:
    # 创建新 Collection
    collection = Collection(collection_name, schema)
    print(f"✅ Collection '{collection_name}' 创建成功")
```

### 创建时的常见错误

**错误1：Collection 名称不合法**
```python
# ❌ 错误：包含特殊字符
collection = Collection("my-documents", schema)  # 不支持连字符

# ❌ 错误：名称过长
collection = Collection("a" * 300, schema)  # 超过 255 字符

# ✅ 正确：只包含字母、数字、下划线
collection = Collection("my_documents_2026", schema)
```

**错误2：Schema 定义不完整**
```python
# ❌ 错误：缺少主键
fields = [
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields)
collection = Collection("docs", schema)  # 报错：缺少主键

# ✅ 正确：必须有主键
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
```

**错误3：向量维度为 0**
```python
# ❌ 错误：维度为 0
FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=0)

# ✅ 正确：维度必须 > 0
FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
```

---

## Read 操作：查看 Collection 信息

### 1. 检查 Collection 是否存在

```python
from pymilvus import utility

# 检查单个 Collection
if utility.has_collection("my_documents"):
    print("✅ Collection 存在")
else:
    print("❌ Collection 不存在")

# 批量检查
collection_names = ["docs1", "docs2", "docs3"]
for name in collection_names:
    exists = utility.has_collection(name)
    print(f"{name}: {'✅ 存在' if exists else '❌ 不存在'}")
```

### 2. 列出所有 Collection

```python
# 列出所有 Collection
collections = utility.list_collections()
print(f"共有 {len(collections)} 个 Collection:")
for name in collections:
    print(f"  - {name}")

# Milvus 2.6：支持 100K Collections
# 如果有大量 Collection，可以分页处理
if len(collections) > 1000:
    print(f"⚠️  检测到 {len(collections)} 个 Collection，建议使用命名规范管理")
```

### 3. 查看 Collection 详细信息

```python
from pymilvus import Collection

collection = Collection("my_documents")

# 基本信息
print(f"Collection 名称: {collection.name}")
print(f"描述: {collection.schema.description}")
print(f"记录数: {collection.num_entities}")

# Schema 信息
print(f"\nSchema 字段:")
for field in collection.schema.fields:
    print(f"  - {field.name}:")
    print(f"      类型: {field.dtype}")
    if field.is_primary:
        print(f"      主键: True")
    if hasattr(field, 'dim'):
        print(f"      维度: {field.dim}")
    if hasattr(field, 'max_length'):
        print(f"      最大长度: {field.max_length}")

# 加载状态
print(f"\n加载状态: {'已加载' if collection.is_loaded else '未加载'}")

# 索引信息
indexes = collection.indexes
if indexes:
    print(f"\n索引信息:")
    for index in indexes:
        print(f"  - 字段: {index.field_name}")
        print(f"    类型: {index.params.get('index_type')}")
        print(f"    度量: {index.params.get('metric_type')}")
```

### 4. 使用 describe() 方法

```python
# describe() 返回详细的 Collection 信息
info = collection.describe()
print(f"Collection 信息:")
print(f"  - 名称: {info['collection_name']}")
print(f"  - 描述: {info['description']}")
print(f"  - 字段数: {len(info['fields'])}")
print(f"  - Shard 数: {info['shards_num']}")
print(f"  - 一致性级别: {info['consistency_level']}")
```

---

## Update 操作：修改 Collection

### Milvus 2.6 新特性：rename_collection()

**传统方案的痛点：**
```python
# 传统方案：重命名需要重建
# 1. 导出数据
# 2. 删除旧 Collection
# 3. 创建新 Collection
# 4. 重新导入数据
# 时间成本：数小时
```

**Milvus 2.6 方案：**
```python
from pymilvus import utility

# 重命名 Collection（秒级完成）
old_name = "my_documents"
new_name = "rag_documents_2026"

utility.rename_collection(old_name, new_name)
print(f"✅ Collection 已重命名: {old_name} → {new_name}")

# 验证
print(f"旧名称存在: {utility.has_collection(old_name)}")  # False
print(f"新名称存在: {utility.has_collection(new_name)}")  # True
```

**重命名的注意事项：**
```python
# ❌ 错误：新名称已存在
utility.rename_collection("docs1", "docs2")  # 如果 docs2 已存在，会报错

# ✅ 正确：先检查
if not utility.has_collection(new_name):
    utility.rename_collection(old_name, new_name)
else:
    print(f"⚠️  目标名称 '{new_name}' 已存在")

# ❌ 错误：旧名称不存在
utility.rename_collection("non_existent", "new_name")  # 报错

# ✅ 正确：先检查
if utility.has_collection(old_name):
    utility.rename_collection(old_name, new_name)
else:
    print(f"⚠️  源名称 '{old_name}' 不存在")
```

### Dynamic Schema：动态添加字段

```python
from pymilvus import Collection, FieldSchema, DataType

collection = Collection("my_documents")

# 添加新字段
new_field = FieldSchema(
    name="category",
    dtype=DataType.VARCHAR,
    max_length=64
)
collection.add_field(new_field)
print("✅ 新字段 'category' 添加成功")

# 验证
for field in collection.schema.fields:
    print(f"  - {field.name}: {field.dtype}")
```

**注意：** Dynamic Schema 只能添加字段，不能修改或删除字段。详见 [03_核心概念_1_Schema定义](./03_核心概念_1_Schema定义.md)。

---

## Delete 操作：删除 Collection

### 基本删除操作

```python
from pymilvus import utility

collection_name = "my_documents"

# 删除 Collection
utility.drop_collection(collection_name)
print(f"✅ Collection '{collection_name}' 已删除")

# 验证
print(f"Collection 存在: {utility.has_collection(collection_name)}")  # False
```

### 删除前的安全检查

```python
# 安全删除流程
def safe_drop_collection(collection_name):
    # 1. 检查是否存在
    if not utility.has_collection(collection_name):
        print(f"⚠️  Collection '{collection_name}' 不存在")
        return False
    
    # 2. 获取 Collection 信息
    collection = Collection(collection_name)
    num_entities = collection.num_entities
    
    # 3. 确认删除（生产环境建议添加用户确认）
    print(f"⚠️  即将删除 Collection '{collection_name}'")
    print(f"    记录数: {num_entities}")
    print(f"    此操作不可逆！")
    
    # 4. 执行删除
    utility.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' 已删除")
    return True

# 使用
safe_drop_collection("my_documents")
```

### 批量删除

```python
# 批量删除 Collection（谨慎使用）
def batch_drop_collections(pattern):
    """
    批量删除匹配模式的 Collection
    pattern: 名称模式，如 "test_*"
    """
    all_collections = utility.list_collections()
    
    # 筛选匹配的 Collection
    import fnmatch
    matched = [name for name in all_collections if fnmatch.fnmatch(name, pattern)]
    
    if not matched:
        print(f"⚠️  没有匹配 '{pattern}' 的 Collection")
        return
    
    print(f"⚠️  即将删除 {len(matched)} 个 Collection:")
    for name in matched:
        print(f"    - {name}")
    
    # 执行删除
    for name in matched:
        utility.drop_collection(name)
        print(f"✅ 已删除: {name}")

# 示例：删除所有测试 Collection
# batch_drop_collections("test_*")
```

### 删除后的空间释放

```python
# 注意：删除 Collection 后，磁盘空间不会立即释放
# 需要触发 Compaction

from pymilvus import Collection

# 1. 删除 Collection
utility.drop_collection("my_documents")

# 2. 触发 Compaction（如果需要立即释放空间）
# 注意：Compaction 是全局操作，影响所有 Collection
# 通常由 Milvus 自动管理，无需手动触发

# 如果确实需要手动触发，可以通过其他 Collection 触发
# collection = Collection("other_collection")
# collection.compact()
```

---

## Milvus 2.6 特性：100K Collections

### 什么是 100K Collections？

**100K Collections** 意味着单个 Milvus 实例可以支持 **10 万个 Collection**。

**技术实现：**
1. **元数据存储**：Collection 元数据存储在 etcd（分布式键值存储）
2. **轻量级设计**：Collection 本身非常轻量，不占用大量内存
3. **按需加载**：只在需要检索时才加载 Collection 到内存（Lazy Loading）

### 多租户架构示例

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

connections.connect("default", host="localhost", port="19530")

# 定义通用 Schema
def create_tenant_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=768)
    ]
    return CollectionSchema(fields, description="Tenant document collection")

# 为每个租户创建独立 Collection
def create_tenant_collection(tenant_id):
    collection_name = f"tenant_{tenant_id}"
    
    # 检查是否已存在
    if utility.has_collection(collection_name):
        print(f"⚠️  租户 {tenant_id} 的 Collection 已存在")
        return Collection(collection_name)
    
    # 创建 Collection
    schema = create_tenant_schema()
    collection = Collection(collection_name, schema)
    print(f"✅ 租户 {tenant_id} 的 Collection 创建成功")
    return collection

# 批量创建租户 Collection
def batch_create_tenants(tenant_ids):
    for tenant_id in tenant_ids:
        create_tenant_collection(tenant_id)

# 示例：创建 1000 个租户的 Collection
tenant_ids = range(1, 1001)
batch_create_tenants(tenant_ids)

print(f"\n✅ 共创建 {len(list(tenant_ids))} 个租户 Collection")
print(f"总 Collection 数: {len(utility.list_collections())}")
```

### 按需加载策略

```python
# Collection 工厂模式 + Lazy Loading
class CollectionManager:
    def __init__(self):
        self.loaded_collections = {}  # 缓存已加载的 Collection
    
    def get_collection(self, tenant_id):
        """获取租户的 Collection（按需加载）"""
        collection_name = f"tenant_{tenant_id}"
        
        # 1. 检查缓存
        if collection_name in self.loaded_collections:
            return self.loaded_collections[collection_name]
        
        # 2. 检查是否存在
        if not utility.has_collection(collection_name):
            raise ValueError(f"租户 {tenant_id} 的 Collection 不存在")
        
        # 3. 加载 Collection
        collection = Collection(collection_name)
        
        # 4. 检查是否已加载到内存
        if not collection.is_loaded:
            collection.load()
            print(f"✅ 租户 {tenant_id} 的 Collection 已加载到内存")
        
        # 5. 缓存
        self.loaded_collections[collection_name] = collection
        return collection
    
    def release_collection(self, tenant_id):
        """释放租户的 Collection（释放内存）"""
        collection_name = f"tenant_{tenant_id}"
        
        if collection_name in self.loaded_collections:
            collection = self.loaded_collections[collection_name]
            collection.release()
            del self.loaded_collections[collection_name]
            print(f"✅ 租户 {tenant_id} 的 Collection 已释放")
    
    def cleanup_inactive(self, max_loaded=100):
        """清理不活跃的 Collection（保持内存占用在合理范围）"""
        if len(self.loaded_collections) > max_loaded:
            # 简单策略：释放最早加载的 Collection
            # 生产环境可以使用 LRU 策略
            to_release = list(self.loaded_collections.keys())[:len(self.loaded_collections) - max_loaded]
            for name in to_release:
                tenant_id = name.replace("tenant_", "")
                self.release_collection(tenant_id)

# 使用示例
manager = CollectionManager()

# 租户 1 检索
collection = manager.get_collection(tenant_id=1)
results = collection.search(...)

# 租户 2 检索
collection = manager.get_collection(tenant_id=2)
results = collection.search(...)

# 定期清理
manager.cleanup_inactive(max_loaded=100)
```

### 性能考虑

**100K Collections 的性能影响：**

| 指标 | 传统方案（共享 Collection） | Milvus 2.6（独立 Collection） |
|------|---------------------------|------------------------------|
| 数据隔离 | 逻辑隔离（tenant_id 过滤） | 物理隔离 |
| 检索性能 | 随租户数增加而下降 | 不受其他租户影响 |
| 内存占用 | 所有租户数据都在内存 | 按需加载（Lazy Loading） |
| 管理复杂度 | 简单 | 中等（需要 Collection 管理器） |
| 安全性 | 中等（依赖应用层过滤） | 高（物理隔离） |

**最佳实践：**
1. 使用 Collection 管理器统一管理
2. 实现 Lazy Loading 策略
3. 定期清理不活跃的 Collection
4. 监控内存占用，控制同时加载的 Collection 数量
5. 使用命名规范（如 `tenant_{id}`）便于管理

---

## Collection 生命周期管理

### 完整生命周期

```
创建 → 加载 → 使用 → 释放 → 删除
  ↓      ↓      ↓      ↓      ↓
Schema  Load  Search Release Drop
```

### 生命周期管理示例

```python
from pymilvus import connections, Collection, utility

class CollectionLifecycle:
    def __init__(self, collection_name, schema):
        self.collection_name = collection_name
        self.schema = schema
        self.collection = None
    
    def create(self):
        """创建 Collection"""
        if utility.has_collection(self.collection_name):
            print(f"⚠️  Collection '{self.collection_name}' 已存在")
            self.collection = Collection(self.collection_name)
        else:
            self.collection = Collection(self.collection_name, self.schema)
            print(f"✅ Collection '{self.collection_name}' 创建成功")
        return self
    
    def load(self):
        """加载到内存"""
        if not self.collection.is_loaded:
            self.collection.load()
            print(f"✅ Collection '{self.collection_name}' 已加载")
        return self
    
    def insert_data(self, data):
        """插入数据"""
        self.collection.insert(data)
        print(f"✅ 已插入 {len(data[0])} 条记录")
        return self
    
    def create_index(self, field_name, index_params):
        """创建索引"""
        self.collection.create_index(field_name, index_params)
        print(f"✅ 已为字段 '{field_name}' 创建索引")
        return self
    
    def search(self, query_vectors, **kwargs):
        """检索"""
        return self.collection.search(query_vectors, **kwargs)
    
    def release(self):
        """释放内存"""
        self.collection.release()
        print(f"✅ Collection '{self.collection_name}' 已释放")
        return self
    
    def drop(self):
        """删除 Collection"""
        utility.drop_collection(self.collection_name)
        print(f"✅ Collection '{self.collection_name}' 已删除")
        self.collection = None

# 使用示例
lifecycle = CollectionLifecycle("my_docs", schema)
lifecycle.create().load().insert_data(data).create_index("vector", index_params)

# 检索
results = lifecycle.search(query_vectors, anns_field="vector", limit=10)

# 清理
lifecycle.release().drop()
```

---

## 学习检查

完成本节学习后，你应该能够：

- [ ] 创建 Collection 并处理常见错误
- [ ] 检查 Collection 是否存在
- [ ] 列出所有 Collection
- [ ] 查看 Collection 的详细信息
- [ ] 使用 rename_collection() 重命名 Collection
- [ ] 安全地删除 Collection
- [ ] 理解 100K Collections 的技术实现
- [ ] 实现多租户架构的 Collection 管理
- [ ] 使用 Lazy Loading 策略优化内存占用
- [ ] 管理 Collection 的完整生命周期

---

## 下一步

- **字段类型详解**：[03_核心概念_3_字段类型详解](./03_核心概念_3_字段类型详解.md)
- **实战代码**：[07_实战代码_场景3_Collection生命周期管理](./07_实战代码_场景3_Collection生命周期管理.md)
- **返回导航**：[00_概览](./00_概览.md)
