---
type: source_code_analysis
source: sourcecode/milvus
analyzed_files:
  - tests/integration/import/dynamic_field_test.go
  - tests/go_client/testcases/add_field_test.go
  - internal/rootcoord/ddl_callbacks_alter_collection_add_field.go
  - pkg/proto/plan.proto
analyzed_at: 2026-02-25
knowledge_point: 05_动态Schema与高级字段
---

# 源码分析：动态Schema与字段管理

## 分析的文件

### 1. `tests/integration/import/dynamic_field_test.go`
- **功能**：动态字段的集成测试
- **关键特性**：BulkInsert 场景下的动态字段支持

### 2. `tests/go_client/testcases/add_field_test.go`
- **功能**：AddCollectionField API 测试
- **关键特性**：字段添加、Nullable、默认值设置

### 3. `internal/rootcoord/ddl_callbacks_alter_collection_add_field.go`
- **功能**：字段添加的核心实现
- **关键特性**：DDL 回调、Schema 版本管理、字段验证

### 4. `pkg/proto/plan.proto`
- **功能**：查询计划的 Proto 定义
- **关键特性**：向量类型定义、列信息、GIS 函数支持

---

## 关键发现

### 1. 动态 Schema 启用机制

**从 `dynamic_field_test.go` 中发现**：

```go
schema.EnableDynamicField = true
```

- 动态 Schema 通过 `EnableDynamicField` 字段启用
- 启用后，可以插入未在 Schema 中定义的字段
- 动态字段存储在特殊的 `$meta` 字段中（隐式）

**关键代码片段**：

```go
schema := integration.ConstructSchema(collectionName, dim, true, &schemapb.FieldSchema{
    FieldID:      100,
    Name:         integration.Int64Field,
    IsPrimaryKey: true,
    DataType:     schemapb.DataType_Int64,
    AutoID:       true,
}, &schemapb.FieldSchema{
    FieldID:  101,
    Name:     integration.FloatVecField,
    DataType: schemapb.DataType_FloatVector,
    TypeParams: []*commonpb.KeyValuePair{
        {
            Key:   common.DimKey,
            Value: fmt.Sprintf("%d", dim),
        },
    },
})
schema.EnableDynamicField = true  // 启用动态字段
```

---

### 2. 字段添加机制（AddCollectionField）

**从 `add_field_test.go` 中发现**：

```go
// 创建新字段
newField := entity.NewField().
    WithName(common.DefaultNewField).
    WithDataType(entity.FieldTypeInt64).
    WithNullable(true)

// 设置默认值（可选）
if tc.defaultValue != -1 {
    newField.WithDefaultValueLong(tc.defaultValue)
}

// 添加字段到 Collection
err = mc.AddCollectionField(ctx, client.NewAddCollectionFieldOption(collName, newField))
```

**关键特性**：
- 支持 `WithNullable(true)` - 字段可为空
- 支持 `WithDefaultValueLong()` - 设置默认值
- 添加字段后立即可用于查询和搜索
- 已存在的数据，新字段值为 NULL 或默认值

**测试用例**：

```go
testCases := []struct {
    name         string
    defaultValue int64
    filter       string
}{
    {
        name:         "defaultValueNone",
        defaultValue: -1,
        filter:       fmt.Sprintf("%s is null", common.DefaultNewField),
    },
    {
        name:         "defaultValue100",
        defaultValue: 100,
        filter:       fmt.Sprintf("%s == 100", common.DefaultNewField),
    },
}
```

---

### 3. 字段添加的内部实现

**从 `ddl_callbacks_alter_collection_add_field.go` 中发现**：

**核心流程**：

```go
func (c *Core) broadcastAlterCollectionForAddField(ctx context.Context, req *milvuspb.AddCollectionFieldRequest) error {
    // 1. 检查 Collection 是否存在
    coll, err := c.meta.GetCollectionByName(ctx, req.GetDbName(), req.GetCollectionName(), typeutil.MaxTimestamp)

    // 2. 解析并验证字段 Schema
    fieldSchema := &schemapb.FieldSchema{}
    if err = proto.Unmarshal(req.Schema, fieldSchema); err != nil {
        return errors.Wrap(err, "failed to unmarshal field schema")
    }
    if err := checkFieldSchema([]*schemapb.FieldSchema{fieldSchema}); err != nil {
        return errors.Wrap(err, "failed to check field schema")
    }

    // 3. 检查字段是否已存在
    for _, field := range coll.Fields {
        if field.Name == fieldSchema.Name {
            return merr.WrapErrParameterInvalidMsg("field already exists, name: %s", fieldSchema.Name)
        }
    }

    // 4. 分配新的 field ID
    fieldSchema.FieldID = nextFieldID(coll)

    // 5. 构建新的 Collection Schema（版本号 +1）
    schema := &schemapb.CollectionSchema{
        Name:               coll.Name,
        Description:        coll.Description,
        AutoID:             coll.AutoID,
        Fields:             model.MarshalFieldModels(coll.Fields),
        StructArrayFields:  model.MarshalStructArrayFieldModels(coll.StructArrayFields),
        Functions:          model.MarshalFunctionModels(coll.Functions),
        EnableDynamicField: coll.EnableDynamicField,  // 保留动态字段设置
        Properties:         coll.Properties,
        Version:            coll.SchemaVersion + 1,    // Schema 版本号递增
    }
    schema.Fields = append(schema.Fields, fieldSchema)

    // 6. 广播 AlterCollection 消息
    msg := message.NewAlterCollectionMessageBuilderV2().
        WithHeader(&messagespb.AlterCollectionMessageHeader{
            DbId:         coll.DBID,
            CollectionId: coll.CollectionID,
            UpdateMask: &fieldmaskpb.FieldMask{
                Paths: []string{message.FieldMaskCollectionSchema},
            },
            CacheExpirations: cacheExpirations,
        }).
        WithBody(&messagespb.AlterCollectionMessageBody{
            Updates: &messagespb.AlterCollectionMessageUpdates{
                Schema: schema,
            },
        }).
        WithBroadcast(channels).
        MustBuildBroadcast()

    if _, err := broadcaster.Broadcast(ctx, msg); err != nil {
        return err
    }

    return nil
}
```

**关键特性**：
- **Schema 版本管理**：每次添加字段，Schema 版本号递增
- **字段 ID 分配**：自动分配唯一的 field ID
- **字段验证**：添加前验证字段 Schema 的合法性
- **幂等性检查**：检查字段是否已存在（TODO: 完整的幂等性检查）
- **消息广播**：通过 Streaming Node 广播 AlterCollection 消息
- **Timestamptz 支持**：特殊处理 Timestamptz 数据类型的默认值

---

### 4. 向量类型定义（Int8、Float16、BFloat16）

**从 `plan.proto` 中发现**：

```protobuf
enum VectorType {
  BinaryVector = 0;
  FloatVector = 1;
  Float16Vector = 2;      // 半精度浮点向量
  BFloat16Vector = 3;     // Brain Float16 向量
  SparseFloatVector = 4;  // 稀疏向量
  Int8Vector = 5;         // 8位整数向量（内存优化）
  EmbListFloatVector = 6;
  EmbListFloat16Vector = 7;
  EmbListBFloat16Vector = 8;
  EmbListInt8Vector = 9;
  EmbListBinaryVector = 10;
}
```

**关键发现**：
- **Int8Vector (5)**：使用 8 位整数表示向量，内存占用仅为 Float32 的 1/4
- **Float16Vector (2)**：半精度浮点向量，内存占用为 Float32 的 1/2
- **BFloat16Vector (3)**：Brain Float16 向量，保留 Float32 的指数范围，但精度降低
- **EmbList 系列**：支持多向量场景（Embedding List）

**应用场景**：
- Int8Vector：大规模向量存储，内存受限场景
- Float16Vector：平衡精度和内存的场景
- BFloat16Vector：深度学习模型输出，兼容性好

---

### 5. 列信息与嵌套结构支持

**从 `plan.proto` 中发现**：

```protobuf
message ColumnInfo {
  int64 field_id = 1;
  schema.DataType data_type = 2;
  bool is_primary_key = 3;
  bool is_autoID = 4;
  repeated string nested_path = 5;      // 嵌套路径支持
  bool is_partition_key = 6;
  schema.DataType element_type = 7;
  bool is_clustering_key = 8;
  bool nullable = 9;                    // 可为空
  bool is_element_level = 10;
}
```

**关键特性**：
- **nested_path**：支持嵌套结构的路径访问（如 JSON 字段的嵌套访问）
- **nullable**：字段可为空（Milvus 2.6 新特性）
- **element_type**：数组元素类型（支持数组字段）
- **is_partition_key**：分区键标识
- **is_clustering_key**：聚簇键标识

---

### 6. 空间数据类型支持（GIS Functions）

**从 `plan.proto` 中发现**：

```protobuf
message GISFunctionFilterExpr {
  ColumnInfo column_info = 1;
  string wkt_string = 2;  // Well-Known Text 格式
  enum GISOp {
    Invalid = 0;
    Equals = 1;      // 相等
    Touches = 2;     // 接触
    Overlaps = 3;    // 重叠
    Crosses = 4;     // 交叉
    Contains = 5;    // 包含
    Intersects = 6;  // 相交
    Within = 7;      // 在内部
    DWithin = 8;     // 距离内
    STIsValid = 9;   // 验证有效性
  }
  GISOp op = 3;
  double distance = 4;  // DWithin 操作的距离参数
}
```

**关键特性**：
- **WKT 格式**：使用 Well-Known Text 格式表示空间数据
- **GIS 操作**：支持 9 种空间关系操作
- **距离查询**：DWithin 操作支持距离参数

**应用场景**：
- 地理位置检索（经纬度）
- 地理围栏查询
- 空间关系分析

---

## 技术依赖识别

### Python 客户端依赖
- `pymilvus >= 2.6.0` - 支持 AddCollectionField、动态 Schema
- `numpy` - 向量数据处理
- `pandas` - 数据处理（可选）

### 向量类型转换
- Int8 向量需要量化处理（Float32 → Int8）
- Float16/BFloat16 向量需要精度转换

### 空间数据处理
- 可能需要 `shapely` 库处理 WKT 格式
- 可能需要 `geopy` 库处理地理位置计算

---

## 下一步调研方向

### 需要通过 Context7 查询的官方文档
1. **pymilvus 2.6+ API 文档**：
   - `AddCollectionField` API 详细用法
   - 动态 Schema 的完整配置选项
   - Int8/Float16/BFloat16 向量的使用方法

2. **Milvus 官方文档**：
   - 动态 Schema 的最佳实践
   - 字段添加的限制和注意事项
   - 空间数据类型的使用指南

### 需要通过 Grok-mcp 搜索的社区资料
1. **动态 Schema 实践案例**：
   - RAG 系统中的元数据扩展案例
   - 多租户系统的字段管理策略
   - 渐进式 Schema 演化的实践

2. **Int8 向量实践**：
   - 量化方法和精度损失分析
   - 性能对比测试结果
   - 生产环境使用经验

3. **空间数据类型应用**：
   - 地理位置检索的实际案例
   - GIS 函数的性能表现
   - 与传统 GIS 数据库的对比

---

## 总结

### 核心发现
1. **动态 Schema**：通过 `EnableDynamicField = true` 启用，支持插入未定义字段
2. **字段添加**：`AddCollectionField` API 支持 Nullable 和默认值，Schema 版本自动管理
3. **Int8 向量**：内存占用仅为 Float32 的 1/4，适合大规模存储
4. **空间数据**：支持 WKT 格式和 9 种 GIS 操作
5. **嵌套结构**：通过 `nested_path` 支持 JSON 字段的嵌套访问

### 技术亮点
- Schema 版本管理机制
- 消息广播架构（Streaming Node）
- 多种向量类型支持（Int8/Float16/BFloat16）
- GIS 函数集成

### 待确认问题
1. 动态字段的性能开销是多少？
2. Int8 向量的量化方法和精度损失？
3. 空间数据类型的索引支持情况？
4. 嵌套结构的深度限制？
