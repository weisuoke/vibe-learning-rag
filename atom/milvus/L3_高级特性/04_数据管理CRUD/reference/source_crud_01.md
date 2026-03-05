---
type: source_code_analysis
source: sourcecode/milvus
analyzed_files:
  - client/milvusclient/write.go
  - client/milvusclient/read.go
  - tests/go_client/testcases/insert_test.go
analyzed_at: 2026-02-25
knowledge_point: 04_数据管理CRUD
---

# 源码分析：Milvus CRUD 操作实现

## 分析的文件

- `client/milvusclient/write.go` - 写操作实现（Insert/Delete/Upsert）
- `client/milvusclient/read.go` - 读操作实现（Search/Query）
- `tests/go_client/testcases/insert_test.go` - Insert 操作测试用例

## 关键发现

### 1. 写操作架构（write.go）

#### Insert 操作
```go
func (c *Client) Insert(ctx context.Context, option InsertOption, callOptions ...grpc.CallOption) (InsertResult, error)

type InsertResult struct {
    InsertCount int64
    IDs         column.Column
}
```

**核心特性**：
- 使用 `retryIfSchemaError` 机制处理 schema 不匹配
- 支持自动写回主键（WriteBackPKs）
- 返回插入数量和生成的 ID 列
- 支持 AutoID 和手动指定 ID 两种模式

#### Delete 操作
```go
func (c *Client) Delete(ctx context.Context, option DeleteOption, callOptions ...grpc.CallOption) (DeleteResult, error)

type DeleteResult struct {
    DeleteCount int64
}
```

**核心特性**：
- 直接调用 gRPC 服务
- 返回删除数量
- 支持基于主键和表达式的删除

#### Upsert 操作
```go
func (c *Client) Upsert(ctx context.Context, option UpsertOption, callOptions ...grpc.CallOption) (UpsertResult, error)

type UpsertResult struct {
    UpsertCount int64
    IDs         column.Column
}
```

**核心特性**：
- 使用 `retryIfSchemaError` 机制处理 schema 不匹配
- 返回 Upsert 数量和 ID 列
- 自动处理插入或更新逻辑

### 2. 读操作架构（read.go）

#### Search 操作
```go
func (c *Client) Search(ctx context.Context, option SearchOption, callOptions ...grpc.CallOption) ([]ResultSet, error)
```

**核心特性**：
- 支持多查询（返回 `[]ResultSet`）
- 支持 wildcard 输出字段（`*`）
- 支持动态字段（dynamic field）
- 支持 group-by 值
- 自动处理结果分页

**结果处理**：
```go
type ResultSet struct {
    sch          *entity.Schema
    ResultCount  int
    Scores       []float32
    IDs          column.Column
    Fields       []column.Column
    GroupByValue column.Column
    Recall       float32
    Err          error
}
```

#### Query 操作
```go
func (c *Client) Query(ctx context.Context, option QueryOption, callOptions ...grpc.CallOption) (ResultSet, error)
```

**核心特性**：
- 返回单个 `ResultSet`
- 基于主键或表达式的精确查询
- 支持输出字段选择

### 3. 测试用例分析（insert_test.go）

#### 基础插入测试
```go
func TestInsertDefault(t *testing.T) {
    // 测试 AutoID = false 和 true 两种模式
    for _, autoID := range [2]bool{false, true} {
        // 创建 Collection
        cp := hp.NewCreateCollectionParams(hp.Int64Vec)
        _, schema := hp.CollPrepare.CreateCollection(ctx, t, mc, cp,
            hp.TNewFieldsOption().TWithAutoID(autoID),
            hp.TNewSchemaOption())

        // 插入数据
        insertOpt := client.NewColumnBasedInsertOption(schema.CollectionName).WithColumns(vecColumn)
        if !autoID {
            insertOpt.WithColumns(pkColumn)
        }
        insertRes, err := mc.Insert(ctx, insertOpt)
    }
}
```

**测试覆盖**：
- AutoID 模式测试
- 分区插入测试
- VarChar 主键测试
- 所有字段类型插入测试
- 动态字段插入测试

### 4. 关键设计模式

#### Schema 不匹配重试机制
```go
err := c.retryIfSchemaError(ctx, collectionName, func(ctx context.Context) (uint64, error) {
    collection, err := c.getCollection(ctx, option.CollectionName())
    if err != nil {
        return math.MaxUint64, err
    }
    req, err := option.InsertRequest(collection)
    if err != nil {
        // 返回 schema mismatch err 以触发重试
        return collection.UpdateTimestamp, merr.WrapErrCollectionSchemaMisMatch(err)
    }
    // 执行实际操作
})
```

**优势**：
- 自动处理 schema 变更
- 避免因 schema 缓存导致的错误
- 提高系统鲁棒性

#### 操作记录机制
```go
defer func() {
    c.recordOperation("Insert", collectionName, startTime, err)
}()
```

**用途**：
- 性能监控
- 错误追踪
- 操作审计

## 代码片段

### Insert 完整实现
```go
func (c *Client) Insert(ctx context.Context, option InsertOption, callOptions ...grpc.CallOption) (InsertResult, error) {
    startTime := time.Now()
    collectionName := option.CollectionName()
    result := InsertResult{}
    err := c.retryIfSchemaError(ctx, collectionName, func(ctx context.Context) (uint64, error) {
        collection, err := c.getCollection(ctx, option.CollectionName())
        if err != nil {
            return math.MaxUint64, err
        }
        req, err := option.InsertRequest(collection)
        if err != nil {
            return collection.UpdateTimestamp, merr.WrapErrCollectionSchemaMisMatch(err)
        }

        return collection.UpdateTimestamp, c.callService(func(milvusService milvuspb.MilvusServiceClient) error {
            resp, err := milvusService.Insert(ctx, req, callOptions...)
            err = merr.CheckRPCCall(resp, err)
            if err != nil {
                return err
            }

            result.InsertCount = resp.GetInsertCnt()
            result.IDs, err = column.IDColumns(collection.Schema, resp.GetIDs(), 0, -1)
            if err != nil {
                return err
            }

            return option.WriteBackPKs(collection.Schema, result.IDs)
        })
    })
    c.recordOperation("Insert", collectionName, startTime, err)
    return result, err
}
```

### Search 结果处理
```go
func (c *Client) handleSearchResult(schema *entity.Schema, outputFields []string, nq int, resp *milvuspb.SearchResults) ([]ResultSet, error) {
    sr := make([]ResultSet, 0, nq)
    results := resp.GetResults()
    offset := 0
    fieldDataList := results.GetFieldsData()
    gb := results.GetGroupByFieldValue()

    for i := 0; i < int(results.GetNumQueries()); i++ {
        var rc int
        entry := ResultSet{sch: schema}

        rc = int(results.GetTopks()[i])
        entry.ResultCount = rc
        entry.Scores = results.GetScores()[offset : offset+rc]

        // 设置 recall（如果返回）
        if i < len(results.Recalls) {
            entry.Recall = results.Recalls[i]
        }

        // 解析 ID 列
        entry.IDs, entry.Err = column.IDColumns(schema, results.GetIds(), offset, offset+rc)

        // 解析 group-by 值
        if gb != nil {
            entry.GroupByValue, entry.Err = column.FieldDataColumn(gb, offset, offset+rc)
        }

        // 解析输出字段
        entry.Fields, entry.Err = c.parseSearchResult(schema, outputFields, fieldDataList, i, offset, offset+rc)

        offset += rc
        sr = append(sr, entry)
    }
    return sr, nil
}
```

## 总结

Milvus 2.6 的 CRUD 操作实现具有以下特点：

1. **统一的 Option 模式**：所有操作都使用 Option 接口，提供灵活的参数配置
2. **自动重试机制**：Insert 和 Upsert 支持 schema 不匹配时的自动重试
3. **完整的结果返回**：所有操作都返回详细的结果信息（数量、ID、字段等）
4. **性能监控**：内置操作记录机制，便于性能分析和问题排查
5. **类型安全**：使用 column.Column 接口处理不同类型的数据
6. **动态字段支持**：Search 和 Query 支持动态字段和 wildcard 输出
