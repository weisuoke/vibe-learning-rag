---
type: context7_documentation
library: milvus
version: v2.6.x (2025-11-17)
fetched_at: 2026-02-25
knowledge_point: 05_动态Schema与高级字段
context7_query: nullable fields default values add field alter collection spatial data types
---

# Context7 文档：Milvus 官方文档

## 文档来源
- 库名称：Milvus
- 版本：v2.6.x (2025-11-17)
- Context7 ID：/milvus-io/milvus-docs, /websites/milvus_io
- 官方文档链接：https://milvus.io/docs

## 关键信息提取

### 1. AddCollectionField API（添加字段到已存在的 Collection）

**用途**：向已存在的 Milvus Collection 添加新的标量字段

**核心特性**：
- **必须是 nullable 字段**：添加的字段必须设置 `nullable=True`
- **不能是向量字段**：只能添加标量字段（INT64, VARCHAR, JSON 等）
- **立即可用**：字段添加后几乎立即可用，延迟极小
- **已存在实体的值**：已存在的实体该字段值为 NULL（或默认值）

**Python API**：
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 添加 nullable 字段
client.add_collection_field(
    collection_name="product_catalog",
    field_name="created_timestamp",  # 新字段名称
    data_type=DataType.INT64,        # 数据类型（必须是标量类型）
    nullable=True                    # 必须为 True
)
```

**Java API**：
```java
import io.milvus.v2.service.collection.request.AddCollectionFieldReq;

client.addCollectionField(AddCollectionFieldReq.builder()
        .collectionName("product_catalog")
        .fieldName("created_timestamp")
        .dataType(DataType.Int64)
        .isNullable(true)
        .build());
```

**Node.js API**：
```javascript
await client.addCollectionField({
    collection_name: 'product_catalog',
    field: {
        name: 'created_timestamp',
        dataType: 'Int64',
        nullable: true
     }
});
```

**REST API**：
```bash
curl -X POST "http://localhost:19530/v2/vectordb/collections/fields/add" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "collectionName": "product_catalog",
    "schema": {
      "fieldName": "created_timestamp",
      "dataType": "Int64",
      "nullable": true
    }
  }'
```

---

### 2. 添加字段时设置默认值

**用途**：为新添加的字段设置默认值，确保已存在的实体和新实体都使用该默认值

**核心特性**：
- **已存在实体**：自动更新为默认值
- **新实体**：如果未提供值，使用默认值
- **数据一致性**：确保所有实体的该字段都有值

**Python 示例**：
```python
client.add_collection_field(
    collection_name="product_catalog",
    field_name="priority_level",     # 新字段名称
    data_type=DataType.VARCHAR,      # 字符串类型
    max_length=20,                   # 最大长度
    nullable=True,                   # 必须为 True
    default_value="standard"         # 默认值
)
```

**Java 示例**：
```java
client.addCollectionField(AddCollectionFieldReq.builder()
        .collectionName("product_catalog")
        .fieldName("priority_level")
        .dataType(DataType.VarChar)
        .maxLength(20)
        .isNullable(true)
        .build());
```

**Node.js 示例**：
```javascript
await client.addCollectionField({
    collection_name: 'product_catalog',
    field: {
        name: 'priority_level',
        dataType: 'VarChar',
        nullable: true,
        default_value: 'standard',
     }
});
```

---

### 3. 创建 Collection 时设置默认值

**用途**：在创建 Collection 时为字段设置默认值

**Python 示例**：
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 创建 Schema
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_schema=True,
)

# 添加字段并设置默认值
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="age", datatype=DataType.INT64, default_value=18)
schema.add_field(field_name="status", datatype=DataType.VARCHAR, default_value="active", max_length=10)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")

# 创建 Collection
client.create_collection(collection_name="my_collection", schema=schema, index_params=index_params)
```

**Java 示例**：
```java
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;

CreateCollectionReq.CollectionSchema schema = client.createSchema();
schema.setEnableDynamicField(true);

schema.addField(AddFieldReq.builder()
        .fieldName("id")
        .dataType(DataType.Int64)
        .isPrimaryKey(true)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("vector")
        .dataType(DataType.FloatVector)
        .dimension(5)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("age")
        .dataType(DataType.Int64)
        .defaultValue(18L)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("status")
        .dataType(DataType.VarChar)
        .maxLength(10)
        .defaultValue("active")
        .build());

List<IndexParam> indexes = new ArrayList<>();
indexes.add(IndexParam.builder()
        .fieldName("vector")
        .indexType(IndexParam.IndexType.AUTOINDEX)
        .metricType(IndexParam.MetricType.L2)
        .build());

CreateCollectionReq requestCreate = CreateCollectionReq.builder()
        .collectionName("my_collection")
        .collectionSchema(schema)
        .indexParams(indexes)
        .build();
client.createCollection(requestCreate);
```

**Go 示例**：
```go
import (
    "context"
    "github.com/milvus-io/milvus/client/v2/entity"
    "github.com/milvus-io/milvus/client/v2/index"
    "github.com/milvus-io/milvus/client/v2/milvusclient"
)

schema := entity.NewSchema()
schema.WithField(entity.NewField().
    WithName("id").
    WithDataType(entity.FieldTypeInt64).
    WithIsPrimaryKey(true),
).WithField(entity.NewField().
    WithName("vector").
    WithDataType(entity.FieldTypeFloatVector).
    WithDim(5),
).WithField(entity.NewField().
    WithName("age").
    WithDataType(entity.FieldTypeInt64).
    WithDefaultValueLong(18),
).WithField(entity.NewField().
    WithName("status").
    WithDataType(entity.FieldTypeVarChar).
    WithMaxLength(10).
    WithDefaultValueString("active"),
)

indexOption := milvusclient.NewCreateIndexOption("my_collection", "vector",
    index.NewAutoIndex(index.MetricType(entity.L2)))

err = client.CreateCollection(ctx,
    milvusclient.NewCreateCollectionOption("my_collection", schema).
        WithIndexOptions(indexOption))
```

---

### 4. 空间数据类型支持

**从 Milvus 官方文档中发现**：

Milvus 支持多种高级数据类型：
- **Sparse Vectors**（稀疏向量）
- **Binary Vectors**（二进制向量）
- **JSON Support**（JSON 支持）
- **Array Support**（数组支持）
- **Text**（文本，开发中）
- **Geolocation**（地理位置，开发中）

**当前状态**：
- Geolocation（地理位置）数据类型正在开发中（under development）
- 从源码分析中看到 `GISFunctionFilterExpr` 支持 WKT 格式和 9 种 GIS 操作

**GIS 操作支持**（从源码分析）：
- `Equals` - 相等
- `Touches` - 接触
- `Overlaps` - 重叠
- `Crosses` - 交叉
- `Contains` - 包含
- `Intersects` - 相交
- `Within` - 在内部
- `DWithin` - 距离内
- `STIsValid` - 验证有效性

---

### 5. 向量类型支持

**从 Milvus 官方文档中发现**：

Milvus 支持多种向量类型及其对应的度量类型：

| 字段类型 | 维度范围 | 支持的度量类型 | 默认度量类型 |
|---------|---------|---------------|-------------|
| `FLOAT_VECTOR` | 2-32,768 | `COSINE`, `L2`, `IP` | `COSINE` |
| `FLOAT16_VECTOR` | 2-32,768 | `COSINE`, `L2`, `IP` | `COSINE` |
| `BFLOAT16_VECTOR` | 2-32,768 | `COSINE`, `L2`, `IP` | `COSINE` |
| `INT8_VECTOR` | 2-32,768 | `COSINE`, `L2`, `IP` | `COSINE` |
| `SPARSE_FLOAT_VECTOR` | 无需指定维度 | `IP`, `BM25`（仅用于全文搜索） | `IP` |
| `BINARY_VECTOR` | 8-32,768*8 | `HAMMING`, `JACCARD`, `MHJACCARD` | `HAMMING` |

**重要说明**：
- **SPARSE_FLOAT_VECTOR**：使用 `BM25` 度量类型仅用于全文搜索
- **BINARY_VECTOR**：维度值（dim）必须是 8 的倍数

**向量类型精度说明**：
- **Binary vectors**：存储二进制数据（0 和 1 序列），用于图像处理和信息检索
- **Float32 vectors**：默认存储类型，精度约 7 位小数。即使输入 Float64 值，也会以 Float32 精度存储，可能导致精度损失
- **Float16 vectors**：降低精度和内存使用。适合带宽和存储受限的应用
- **BFloat16 vectors**：平衡范围和效率，常用于深度学习，减少计算需求而不显著影响精度

---

### 6. 完整的 Schema 定义示例

**从 Milvus 官方文档中发现**：

创建包含多种数据类型的 Collection Schema：

```python
from pymilvus import MilvusClient, DataType

# 创建 Schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)

DIM = 512

# 添加各种字段类型
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="bool", datatype=DataType.BOOL)
schema.add_field(field_name="int8", datatype=DataType.INT8)
schema.add_field(field_name="int16", datatype=DataType.INT16)
schema.add_field(field_name="int32", datatype=DataType.INT32)
schema.add_field(field_name="int64", datatype=DataType.INT64)
schema.add_field(field_name="float", datatype=DataType.FLOAT)
schema.add_field(field_name="double", datatype=DataType.DOUBLE)
schema.add_field(field_name="varchar", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="json", datatype=DataType.JSON)
schema.add_field(field_name="array_str", datatype=DataType.ARRAY, max_capacity=100, element_type=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="array_int", datatype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64)
schema.add_field(field_name="float_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field(field_name="binary_vector", datatype=DataType.BINARY_VECTOR, dim=DIM)
schema.add_field(field_name="float16_vector", datatype=DataType.FLOAT16_VECTOR, dim=DIM)
```

**支持的数据类型**：
- **标量类型**：BOOL, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, VARCHAR
- **复杂类型**：JSON, ARRAY
- **向量类型**：FLOAT_VECTOR, BINARY_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR, INT8_VECTOR, SPARSE_FLOAT_VECTOR

---

## 关键发现总结

### 1. AddCollectionField 的限制
- **必须是 nullable**：添加的字段必须设置 `nullable=True`
- **不能是向量字段**：只能添加标量字段
- **立即可用**：字段添加后几乎立即可用

### 2. 默认值机制
- **创建时设置**：可以在创建 Collection 时为字段设置默认值
- **添加时设置**：可以在添加字段时设置默认值
- **数据一致性**：确保所有实体的该字段都有值

### 3. 向量类型丰富
- 支持 6 种向量类型：FLOAT, FLOAT16, BFLOAT16, INT8, BINARY, SPARSE
- 每种向量类型有对应的度量类型
- INT8 向量支持 COSINE, L2, IP 度量类型

### 4. 空间数据类型
- Geolocation 数据类型正在开发中
- 从源码看已支持 GIS 函数和 WKT 格式
- 支持 9 种 GIS 操作

### 5. 动态 Schema 默认启用
- 使用 `MilvusClient.create_collection()` 创建的 Collection 默认启用动态字段
- 可以通过 `enable_dynamic_field=True` 显式启用

---

## 与源码分析的对应关系

### 1. AddCollectionField 实现
- **源码**：`ddl_callbacks_alter_collection_add_field.go` 实现了字段添加的核心逻辑
- **API**：pymilvus 提供了 `add_collection_field()` 方法
- **限制**：必须是 nullable 字段，不能是向量字段

### 2. 向量类型定义
- **源码**：`plan.proto` 定义了 `VectorType` 枚举
- **API**：pymilvus 支持所有向量类型的创建和操作
- **度量类型**：每种向量类型有对应的度量类型

### 3. 空间数据类型
- **源码**：`plan.proto` 定义了 `GISFunctionFilterExpr`
- **状态**：Geolocation 数据类型正在开发中
- **功能**：已支持 WKT 格式和 9 种 GIS 操作

---

## 待确认问题

### 1. 空间数据类型的完整 API
从官方文档看，Geolocation 数据类型正在开发中（under development），但从源码分析中看到已经有 `GISFunctionFilterExpr` 的实现。需要确认：
- 当前版本是否已支持空间数据类型？
- 如何在 pymilvus 中使用空间数据类型？
- WKT 格式的具体使用方法？

### 2. Int8 向量的量化方法
从官方文档看，Int8 向量支持 COSINE, L2, IP 度量类型，但没有说明：
- 如何将 Float32 向量量化为 Int8 向量？
- 量化过程中的精度损失是多少？
- 是否有推荐的量化方法？

### 3. 动态字段的性能影响
从官方文档看，动态字段默认启用，但没有说明：
- 动态字段的性能开销是多少？
- 何时应该禁用动态字段？
- 动态字段的最佳实践是什么？

---

## 下一步调研方向

### 1. 使用 Grok-mcp 搜索社区资料
- Int8 向量的量化方法和实践案例
- 空间数据类型的实际使用案例
- AddCollectionField 的生产环境使用经验
- 动态 Schema 的性能影响和最佳实践

### 2. 补充源码分析
- 查找 Int8 向量的量化实现
- 查找空间数据类型的完整实现
- 查找动态字段的性能优化代码
