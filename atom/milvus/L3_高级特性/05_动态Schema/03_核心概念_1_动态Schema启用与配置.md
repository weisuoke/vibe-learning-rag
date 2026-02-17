# 核心概念1：动态Schema启用与配置

> 深入理解如何启用和配置Milvus动态Schema

---

## 概述

动态Schema的启用和配置是使用动态字段的第一步。本文将详细讲解：
1. 如何启用动态Schema
2. Schema设计原则
3. 固定字段与动态字段的选择策略
4. 配置最佳实践

---

## 1. 启用动态Schema

### 1.1 基本启用方式

**核心参数**：`enable_dynamic_field=True`

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# 连接Milvus
connections.connect(host="localhost", port="19530")

# 定义固定字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]

# 创建Schema并启用动态字段
schema = CollectionSchema(
    fields=fields,
    description="Collection with dynamic schema enabled",
    enable_dynamic_field=True  # 关键参数
)

# 创建Collection
collection = Collection(name="dynamic_demo", schema=schema)
print(f"✅ 动态Schema已启用")
```

**关键点**：
- `enable_dynamic_field=True` 是唯一的开关
- 必须在创建Collection时设置，后续无法修改
- 默认值是 `False`

---

### 1.2 验证动态Schema是否启用

```python
# 方法1：查看Schema信息
schema_info = collection.schema
print(f"动态字段启用状态: {schema_info.enable_dynamic_field}")

# 方法2：尝试插入动态字段
try:
    data = [
        {
            "text": "Test document",
            "embedding": [0.1] * 768,
            "dynamic_field": "test"  # 动态字段
        }
    ]
    collection.insert(data)
    print("✅ 动态Schema已启用（插入成功）")
except Exception as e:
    print(f"❌ 动态Schema未启用: {e}")
```

---

### 1.3 动态Schema的限制

**限制1：创建后无法修改**

```python
# ❌ 错误：无法在创建后修改enable_dynamic_field
collection = Collection(name="test", schema=schema)
# 无法执行: collection.schema.enable_dynamic_field = True
# 只能删除Collection重新创建
```

**限制2：必须有固定字段**

```python
# ❌ 错误：至少需要主键和向量字段
fields = []  # 空字段列表
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
# 报错：至少需要主键字段和向量字段
```

**限制3：动态字段不能与固定字段同名**

```python
# ❌ 错误：动态字段与固定字段同名
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
collection = Collection(name="test", schema=schema)

# 插入数据时，动态字段不能与固定字段同名
data = [
    {
        "text": "Document",  # 固定字段
        "embedding": [0.1] * 768,
        "text": "Another value"  # ❌ 错误：与固定字段同名
    }
]
```

---

## 2. Schema设计原则

### 2.1 固定字段的选择原则

**原则1：核心业务字段必须固定**

```python
# ✅ 正确：核心字段固定
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),  # 核心字段
    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50)  # 核心字段
]
```

**原则2：高频查询字段必须固定**

```python
# 分析查询频率
query_frequency = {
    "author": 1000,    # 高频 → 固定字段
    "category": 500,   # 中频 → 固定字段
    "tags": 10,        # 低频 → 动态字段
    "metadata": 5      # 低频 → 动态字段
}

# ✅ 正确：高频字段固定
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),    # 高频
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)    # 中频
]
# tags和metadata作为动态字段
```

**原则3：需要索引的字段必须固定**

```python
# ✅ 正确：需要索引的字段固定
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),  # 需要索引
    FieldSchema(name="created_at", dtype=DataType.INT64)  # 需要范围查询
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
collection = Collection(name="test", schema=schema)

# 为固定字段创建索引
collection.create_index(
    field_name="author",
    index_params={"index_type": "TRIE"}
)
```

**原则4：需要复杂查询的字段必须固定**

```python
# ✅ 正确：需要范围查询、排序的字段固定
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="priority", dtype=DataType.INT64),  # 需要范围查询
    FieldSchema(name="created_at", dtype=DataType.INT64)  # 需要排序
]

# 支持复杂查询
results = collection.query(
    expr='priority > 3 and priority < 10',  # 范围查询
    output_fields=["*"]
)
```

---

### 2.2 动态字段的选择原则

**原则1：低频查询字段使用动态**

```python
# ✅ 正确：低频字段动态
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ],
    enable_dynamic_field=True
)

# 插入数据时添加低频字段
data = [
    {
        "text": "Document",
        "embedding": [0.1] * 768,
        "last_modified": "2024-01-01",  # 低频字段
        "version": "1.0"  # 低频字段
    }
]
```

**原则2：实验性字段使用动态**

```python
# ✅ 正确：实验性字段动态
data = [
    {
        "text": "Document",
        "embedding": [0.1] * 768,
        "experimental_score": 0.85,  # 实验性字段，可能会删除
        "test_flag": True  # 实验性字段
    }
]
```

**原则3：租户特有字段使用动态**

```python
# ✅ 正确：租户特有字段动态
# 租户A的数据
tenant_a_data = {
    "text": "Product",
    "embedding": [0.1] * 768,
    "price": 99.99,  # 租户A特有
    "stock": 100  # 租户A特有
}

# 租户B的数据
tenant_b_data = {
    "text": "Article",
    "embedding": [0.2] * 768,
    "author": "Bob",  # 租户B特有
    "publish_date": "2024-01-01"  # 租户B特有
}
```

**原则4：元数据字段使用动态**

```python
# ✅ 正确：元数据字段动态
data = [
    {
        "text": "Document",
        "embedding": [0.1] * 768,
        "metadata": {  # 元数据
            "source": "web",
            "crawl_date": "2024-01-01",
            "language": "en"
        }
    }
]
```

---

### 2.3 混合Schema设计模式

**模式1：核心固定 + 扩展动态**

```python
# 适用场景：大部分字段已知，少量字段需要灵活扩展
schema = CollectionSchema(
    fields=[
        # 核心固定字段
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="created_at", dtype=DataType.INT64)
    ],
    enable_dynamic_field=True  # 扩展字段动态
)

# 插入数据
data = [
    {
        # 固定字段
        "text": "Document",
        "embedding": [0.1] * 768,
        "author": "Alice",
        "category": "tech",
        "created_at": 1704067200,
        # 动态字段
        "tags": ["AI", "ML"],
        "rating": 4.5
    }
]
```

**模式2：最小固定 + 最大动态**

```python
# 适用场景：快速原型，字段不确定
schema = CollectionSchema(
    fields=[
        # 最小固定字段
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ],
    enable_dynamic_field=True  # 所有业务字段都动态
)

# 插入数据
data = [
    {
        "embedding": [0.1] * 768,
        # 所有业务字段都是动态的
        "text": "Document",
        "author": "Alice",
        "category": "tech",
        "tags": ["AI", "ML"]
    }
]
```

**模式3：分层固定 + 分层动态**

```python
# 适用场景：多租户系统
schema = CollectionSchema(
    fields=[
        # 通用固定字段
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50)
    ],
    enable_dynamic_field=True  # 租户特有字段动态
)

# 租户A的数据
tenant_a_data = {
    "tenant_id": "tenant_a",
    "doc_type": "product",
    "embedding": [0.1] * 768,
    # 租户A特有字段（动态）
    "price": 99.99,
    "stock": 100
}

# 租户B的数据
tenant_b_data = {
    "tenant_id": "tenant_b",
    "doc_type": "article",
    "embedding": [0.2] * 768,
    # 租户B特有字段（动态）
    "author": "Bob",
    "publish_date": "2024-01-01"
}
```

---

## 3. 配置最佳实践

### 3.1 字段命名规范

**规范1：使用有意义的字段名**

```python
# ✅ 正确：有意义的字段名
data = {
    "text": "Document content",
    "embedding": [0.1] * 768,
    "author_name": "Alice",  # 清晰
    "created_timestamp": 1704067200,  # 清晰
    "document_category": "tech"  # 清晰
}

# ❌ 错误：无意义的字段名
data = {
    "text": "Document content",
    "embedding": [0.1] * 768,
    "f1": "Alice",  # 不清晰
    "t": 1704067200,  # 不清晰
    "c": "tech"  # 不清晰
}
```

**规范2：使用一致的命名风格**

```python
# ✅ 正确：统一使用snake_case
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "author_name": "Alice",
    "created_at": 1704067200,
    "doc_category": "tech"
}

# ❌ 错误：混合使用不同风格
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "authorName": "Alice",  # camelCase
    "created_at": 1704067200,  # snake_case
    "DocCategory": "tech"  # PascalCase
}
```

**规范3：避免使用保留字**

```python
# ❌ 错误：使用保留字
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "type": "article",  # 可能与系统保留字冲突
    "class": "A"  # 可能与系统保留字冲突
}

# ✅ 正确：使用明确的字段名
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "doc_type": "article",
    "doc_class": "A"
}
```

---

### 3.2 类型一致性保证

**策略1：在应用层验证类型**

```python
def validate_dynamic_fields(data, field_types):
    """验证动态字段的类型一致性"""
    for field_name, expected_type in field_types.items():
        if field_name in data:
            actual_type = type(data[field_name])
            if actual_type != expected_type:
                raise TypeError(
                    f"Field '{field_name}' type mismatch: "
                    f"expected {expected_type}, got {actual_type}"
                )

# 定义动态字段的类型约束
dynamic_field_types = {
    "author": str,
    "priority": int,
    "rating": float,
    "tags": list,
    "is_public": bool
}

# 插入前验证
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "author": "Alice",
    "priority": 5,
    "rating": 4.5
}

validate_dynamic_fields(data, dynamic_field_types)
collection.insert([data])
```

**策略2：使用类型转换**

```python
def normalize_dynamic_fields(data, field_types):
    """标准化动态字段的类型"""
    for field_name, target_type in field_types.items():
        if field_name in data:
            try:
                data[field_name] = target_type(data[field_name])
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Cannot convert field '{field_name}' to {target_type}: {e}"
                )
    return data

# 使用示例
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "priority": "5",  # 字符串
    "rating": "4.5"  # 字符串
}

# 标准化类型
data = normalize_dynamic_fields(data, {"priority": int, "rating": float})
# 现在priority是int，rating是float
collection.insert([data])
```

**策略3：记录字段类型映射**

```python
import json

class DynamicSchemaManager:
    def __init__(self, collection):
        self.collection = collection
        self.field_types = {}  # 记录字段类型

    def insert(self, data):
        """插入数据并记录字段类型"""
        for field_name, value in data.items():
            if field_name not in self.collection.schema.fields:
                # 动态字段
                value_type = type(value).__name__
                if field_name in self.field_types:
                    # 检查类型一致性
                    if self.field_types[field_name] != value_type:
                        raise TypeError(
                            f"Field '{field_name}' type mismatch: "
                            f"expected {self.field_types[field_name]}, got {value_type}"
                        )
                else:
                    # 记录类型
                    self.field_types[field_name] = value_type

        # 插入数据
        self.collection.insert([data])

    def save_field_types(self, filepath):
        """保存字段类型映射"""
        with open(filepath, 'w') as f:
            json.dump(self.field_types, f, indent=2)

    def load_field_types(self, filepath):
        """加载字段类型映射"""
        with open(filepath, 'r') as f:
            self.field_types = json.load(f)

# 使用示例
manager = DynamicSchemaManager(collection)
manager.insert({"text": "doc1", "embedding": [0.1]*768, "priority": 5})
manager.insert({"text": "doc2", "embedding": [0.2]*768, "priority": 3})
manager.save_field_types("field_types.json")
```

---

### 3.3 性能优化配置

**优化1：限制动态字段数量**

```python
# ✅ 推荐：限制动态字段数量（< 20个）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    # 动态字段（控制在20个以内）
    "field1": "value1",
    "field2": "value2",
    # ...
    "field15": "value15"
}

# ❌ 不推荐：过多动态字段（> 50个）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    # 动态字段（过多）
    "field1": "value1",
    # ...
    "field100": "value100"  # 性能下降
}
```

**优化2：控制动态字段值的大小**

```python
# ✅ 推荐：动态字段值较小（< 1KB）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "metadata": "short metadata"  # 小值
}

# ❌ 不推荐：动态字段值过大（> 10KB）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "large_metadata": "x" * 100000  # 大值，性能下降
}
```

**优化3：避免嵌套过深**

```python
# ✅ 推荐：嵌套层级较浅（< 3层）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "metadata": {
        "source": "web",
        "details": {
            "url": "http://..."
        }
    }
}

# ❌ 不推荐：嵌套过深（> 5层）
data = {
    "text": "Document",
    "embedding": [0.1] * 768,
    "metadata": {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "level5": "value"
                    }
                }
            }
        }
    }
}
```

---

## 4. 完整配置示例

### 4.1 RAG系统配置

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# 连接Milvus
connections.connect(host="localhost", port="19530")

# RAG系统Schema设计
fields = [
    # 核心固定字段
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),

    # 高频查询字段（固定）
    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="created_at", dtype=DataType.INT64)
]

schema = CollectionSchema(
    fields=fields,
    description="RAG system with dynamic schema",
    enable_dynamic_field=True  # 低频字段使用动态
)

collection = Collection(name="rag_documents", schema=schema)

# 创建索引
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
)

collection.create_index(
    field_name="author",
    index_params={"index_type": "TRIE"}
)

# 插入数据
data = [
    {
        # 固定字段
        "text": "Introduction to vector databases",
        "embedding": [0.1] * 768,
        "doc_type": "article",
        "author": "Alice",
        "category": "database",
        "created_at": 1704067200,
        # 动态字段
        "tags": ["vector", "database", "AI"],
        "rating": 4.5,
        "views": 1000,
        "source_url": "http://example.com"
    }
]

collection.insert(data)
collection.load()

print("✅ RAG系统配置完成")
```

---

## 总结

### 核心要点

1. **启用方式**：`enable_dynamic_field=True`，创建时设置，后续无法修改
2. **固定字段选择**：核心字段、高频查询、需要索引、需要复杂查询
3. **动态字段选择**：低频查询、实验性、租户特有、元数据
4. **设计模式**：核心固定+扩展动态、最小固定+最大动态、分层固定+分层动态
5. **最佳实践**：命名规范、类型一致性、性能优化

### 决策流程

```
需要添加新字段？
    ↓
是否核心业务字段？
    ↓ 是
    固定字段
    ↓ 否
是否高频查询（> 10%）？
    ↓ 是
    固定字段
    ↓ 否
是否需要索引？
    ↓ 是
    固定字段
    ↓ 否
是否需要复杂查询？
    ↓ 是
    固定字段
    ↓ 否
    动态字段
```

---

**记住**：动态Schema的配置是灵活性和性能的权衡，根据实际需求选择合适的策略。
