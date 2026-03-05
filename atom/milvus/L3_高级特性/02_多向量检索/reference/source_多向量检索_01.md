---
type: source_code_analysis
source: sourcecode/milvus
analyzed_files:
  - tests/integration/hellomilvus/hybridsearch_test.go
  - tests/python_client/milvus_client_v2/test_milvus_client_hybrid_search_v2.py
  - tests/integration/import/multi_vector_test.go
analyzed_at: 2026-02-25
knowledge_point: 02_多向量检索
---

# 源码分析：Milvus 多向量检索实现机制

## 分析的文件

### 1. tests/integration/hellomilvus/hybridsearch_test.go
**文件说明**：Go 语言的混合检索集成测试，展示了如何在一个 Collection 中定义多个向量字段并进行检索。

### 2. tests/python_client/milvus_client_v2/test_milvus_client_hybrid_search_v2.py
**文件说明**：Python SDK 的混合检索测试，展示了 AnnSearchRequest、RRFRanker、WeightedRanker 的使用方式。

### 3. tests/integration/import/multi_vector_test.go
**文件说明**：多向量字段的批量导入测试，展示了不同类型向量字段的定义和导入。

## 关键发现

### 1. 多向量字段定义

**Go SDK 示例**（hybridsearch_test.go:35-40）：
```go
schema := integration.ConstructSchema(collectionName, dim, true,
    &schemapb.FieldSchema{Name: integration.Int64Field, DataType: schemapb.DataType_Int64, IsPrimaryKey: true, AutoID: true},
    &schemapb.FieldSchema{Name: integration.FloatVecField, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "128"}}},
    &schemapb.FieldSchema{Name: integration.BinVecField, DataType: schemapb.DataType_BinaryVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "128"}}},
    &schemapb.FieldSchema{Name: integration.SparseFloatVecField, DataType: schemapb.DataType_SparseFloatVector},
)
```

**关键点**：
- 一个 Collection 可以包含多个向量字段
- 支持的向量类型：FloatVector、BinaryVector、SparseFloatVector、BFloat16Vector
- 每个向量字段可以有不同的维度

**Python SDK 示例**（test_milvus_client_hybrid_search_v2.py:94-98）：
```python
collection_schema.add_field(self.float_vector_field_name1, DataType.FLOAT_VECTOR, dim=self.float_vector_dim)
collection_schema.add_field(self.float_vector_field_name2, DataType.FLOAT_VECTOR, dim=self.float_vector_dim)
collection_schema.add_field(self.sparse_vector_field_name1, DataType.SPARSE_FLOAT_VECTOR)
collection_schema.add_field(self.sparse_vector_field_name2, DataType.SPARSE_FLOAT_VECTOR)
```

### 2. 索引创建要求

**关键约束**（hybridsearch_test.go:98-148）：
- **所有向量字段都必须创建索引**才能加载 Collection
- 如果只为部分向量字段创建索引，LoadCollection 会失败

**代码证据**：
```go
// load without index on vector fields
loadStatus, err := c.MilvusClient.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{
    DbName:         dbName,
    CollectionName: collectionName,
})
s.NoError(err)
s.Error(merr.Error(loadStatus))  // 预期失败
```

**索引创建示例**：
```go
// create index for float vector
createIndexStatus, err := c.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
    CollectionName: collectionName,
    FieldName:      integration.FloatVecField,
    IndexName:      "_default_float",
    ExtraParams:    integration.ConstructIndexParam(dim, integration.IndexFaissIvfFlat, metric.L2),
})

// create index for binary vector
createIndexStatus, err = c.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
    CollectionName: collectionName,
    FieldName:      integration.BinVecField,
    IndexName:      "_default_binary",
    ExtraParams:    integration.ConstructIndexParam(dim, integration.IndexFaissBinIvfFlat, metric.JACCARD),
})
```

### 3. BM25 函数集成

**Python SDK 示例**（test_milvus_client_hybrid_search_v2.py:106-121）：
```python
bm25_function1 = Function(
    name=self.sparse_vector_field_name1,
    function_type=FunctionType.BM25,
    input_field_names=[self.text_field_name1],
    output_field_names=self.sparse_vector_field_name1,
    params={},
)
collection_schema.add_function(bm25_function1)
```

**关键点**：
- BM25 函数可以自动将文本字段转换为稀疏向量
- 输入字段需要启用 `enable_analyzer=True`
- 输出字段类型为 `SPARSE_FLOAT_VECTOR`

### 4. 混合检索 API

**Python SDK 使用的关键类**（test_milvus_client_hybrid_search_v2.py:3）：
```python
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
```

**关键组件**：
- `AnnSearchRequest`：定义单个向量字段的检索请求
- `RRFRanker`：Reciprocal Rank Fusion 排序器
- `WeightedRanker`：加权排序器

### 5. 多向量数据插入

**Go SDK 示例**（hybridsearch_test.go:63-73）：
```go
fVecColumn := integration.NewFloatVectorFieldData(integration.FloatVecField, rowNum, dim)
bVecColumn := integration.NewBinaryVectorFieldData(integration.BinVecField, rowNum, dim)
sparseVecColumn := integration.NewSparseFloatVectorFieldData(integration.SparseFloatVecField, rowNum)

insertResult, err := c.MilvusClient.Insert(ctx, &milvuspb.InsertRequest{
    DbName:         dbName,
    CollectionName: collectionName,
    FieldsData:     []*schemapb.FieldData{fVecColumn, bVecColumn, sparseVecColumn},
    HashKeys:       hashKeys,
    NumRows:        uint32(rowNum),
})
```

**关键点**：
- 插入数据时需要为所有向量字段提供数据
- 不同类型的向量字段使用不同的数据生成函数

### 6. 不同向量类型支持

**从 multi_vector_test.go 中发现**（multi_vector_test.go:55-81）：
```go
schema := integration.ConstructSchema(collectionName, 0, true, &schemapb.FieldSchema{
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
            Value: fmt.Sprintf("%d", dim1),  // dim1 = 64
        },
    },
}, &schemapb.FieldSchema{
    FieldID:  102,
    Name:     integration.BFloat16VecField,
    DataType: schemapb.DataType_BFloat16Vector,
    TypeParams: []*commonpb.KeyValuePair{
        {
            Key:   common.DimKey,
            Value: fmt.Sprintf("%d", dim2),  // dim2 = 32
        },
    },
})
```

**关键点**：
- 不同向量字段可以有不同的维度（dim1=64, dim2=32）
- 支持 BFloat16Vector 类型（内存优化）

## 核心技术要点总结

### 1. Schema 设计
- 一个 Collection 可以包含多个向量字段
- 每个向量字段独立配置维度和类型
- 支持的向量类型：FloatVector、BinaryVector、SparseFloatVector、BFloat16Vector

### 2. 索引要求
- **所有向量字段都必须创建索引**
- 不同类型的向量字段使用不同的索引类型
- 索引创建完成后才能加载 Collection

### 3. 数据插入
- 插入时需要为所有向量字段提供数据
- 支持批量插入多向量数据

### 4. 混合检索
- 使用 `AnnSearchRequest` 定义多个检索请求
- 使用 `RRFRanker` 或 `WeightedRanker` 融合结果
- 支持 BM25 函数自动生成稀疏向量

### 5. 应用场景
- 多模态检索（图片+文本）
- 稠密向量+稀疏向量混合检索
- 多语言检索（中文+英文）
- 多粒度检索（标题+内容）

## 下一步调研方向

1. **Context7 官方文档**：
   - Milvus 多向量检索 API 详细文档
   - AnnSearchRequest、RRFRanker、WeightedRanker 使用指南
   - BM25 函数配置和参数

2. **网络搜索**：
   - 多模态检索实践案例（CLIP 模型集成）
   - RRF vs Weighted Ranker 性能对比
   - 生产环境中的多向量检索优化策略

3. **需要深入理解的技术点**：
   - RRF（Reciprocal Rank Fusion）算法原理
   - 加权融合策略的权重调整方法
   - 多向量检索的性能优化技巧
