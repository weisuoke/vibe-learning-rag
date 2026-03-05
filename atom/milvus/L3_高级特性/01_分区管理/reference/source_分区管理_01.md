---
type: source_code_analysis
source: sourcecode/milvus
analyzed_files:
  - client/entity/collection.go
  - internal/metastore/model/partition.go
  - tests/integration/hellomilvus/partition_key_test.go
analyzed_at: 2026-02-25
knowledge_point: 01_分区管理
---

# 源码分析：Partition 核心数据结构

## 分析的文件

- `client/entity/collection.go:41-46` - 客户端 Partition 模型
- `internal/metastore/model/partition.go:7-77` - 内部 Partition 模型
- `tests/integration/hellomilvus/partition_key_test.go:38-150` - Partition Key 测试

## 关键发现

### 1. Partition 数据结构（客户端视角）

**文件位置**: `client/entity/collection.go:41-46`

```go
// Partition represent partition meta in Milvus
type Partition struct {
	ID     int64  // partition id
	Name   string // partition name
	Loaded bool   // partition loaded
}
```

**核心字段**:
- `ID`: 分区的唯一标识符
- `Name`: 分区名称（用户可读）
- `Loaded`: 分区是否已加载到内存

**设计理念**: 客户端模型非常简洁，只包含用户关心的核心信息。

### 2. Partition 数据结构（内部视角）

**文件位置**: `internal/metastore/model/partition.go:7-13`

```go
type Partition struct {
	PartitionID               int64
	PartitionName             string
	PartitionCreatedTimestamp uint64
	CollectionID              int64
	State                     pb.PartitionState
}
```

**核心字段**:
- `PartitionID`: 分区的唯一标识符
- `PartitionName`: 分区名称
- `PartitionCreatedTimestamp`: 分区创建时间戳
- `CollectionID`: 所属 Collection 的 ID
- `State`: 分区状态（`pb.PartitionState` 枚举）

**关键方法**:

```go
func (p *Partition) Available() bool {
	return p.State == pb.PartitionState_PartitionCreated
}
```

**设计理念**: 内部模型包含更多元数据，用于系统管理和状态跟踪。

### 3. Partition Key 特性

**文件位置**: `tests/integration/hellomilvus/partition_key_test.go:38-150`

**核心发现**:

1. **Partition Key 字段定义**:
```go
schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
	FieldID:        102,
	Name:           "pid",
	Description:    "",
	DataType:       schemapb.DataType_Int64,
	TypeParams:     nil,
	IndexParams:    nil,
	IsPartitionKey: true,  // 关键标记
})
```

2. **自动分区管理**:
   - 当字段标记为 `IsPartitionKey: true` 时，Milvus 会自动根据该字段的值进行分区
   - 测试中插入了三批数据，分别使用 `pid=1`, `pid=10`, `pid=100`
   - Milvus 会自动将这些数据分配到不同的分区

3. **使用场景**:
   - 多租户系统（每个租户一个 Partition Key 值）
   - 时间序列数据（按时间戳分区）
   - 业务隔离（按业务类型分区）

### 4. Partition 状态管理

**文件位置**: `internal/metastore/model/partition.go:15-17`

```go
func (p *Partition) Available() bool {
	return p.State == pb.PartitionState_PartitionCreated
}
```

**状态枚举** (推断自代码):
- `PartitionState_PartitionCreated`: 分区已创建并可用
- 其他状态（需要进一步查看 protobuf 定义）

### 5. Partition 克隆与比较

**文件位置**: `internal/metastore/model/partition.go:19-56`

```go
func (p *Partition) Clone() *Partition {
	return &Partition{
		PartitionID:               p.PartitionID,
		PartitionName:             p.PartitionName,
		PartitionCreatedTimestamp: p.PartitionCreatedTimestamp,
		CollectionID:              p.CollectionID,
		State:                     p.State,
	}
}

func (p *Partition) Equal(other Partition) bool {
	return p.PartitionName == other.PartitionName
}
```

**设计理念**:
- 提供深拷贝功能，避免并发修改问题
- 相等性比较基于 `PartitionName`，而非 `PartitionID`

## 核心概念总结

### 1. Partition 的本质
- **定义**: Partition 是 Collection 内部的逻辑数据分区
- **目的**: 提升检索效率、实现数据隔离、支持多租户

### 2. Partition Key（Milvus 2.6 核心特性）
- **自动分区**: 无需手动创建分区，Milvus 根据 Partition Key 字段自动管理
- **简化开发**: 开发者只需标记字段，无需关心分区逻辑
- **性能优化**: 检索时可以指定 Partition Key 值，只扫描相关分区

### 3. Partition 生命周期
1. **创建**: 通过 API 创建或通过 Partition Key 自动创建
2. **加载**: 将分区数据加载到内存（`Loaded` 字段）
3. **使用**: 在检索时指定分区
4. **释放**: 从内存中释放分区数据
5. **删除**: 删除分区及其数据

### 4. Partition 与 Collection 的关系
- 一个 Collection 可以包含多个 Partition
- 每个 Partition 属于一个 Collection（`CollectionID` 字段）
- Partition 继承 Collection 的 Schema 定义

## 需要进一步调研的技术点

1. **Partition 数量限制**: Milvus 2.6 支持 100K collections，Partition 数量限制是多少？
2. **Partition Key 的哈希算法**: 如何将 Partition Key 值映射到具体的分区？
3. **Partition 的性能影响**: 分区数量对检索性能的影响？
4. **Partition 的最佳实践**: 什么时候应该使用 Partition？什么时候应该使用 Partition Key？
5. **Partition 的存储机制**: 分区数据在磁盘上如何组织？

## 代码片段

### 创建带 Partition Key 的 Collection

```go
schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
	FieldID:        102,
	Name:           "tenant_id",
	DataType:       schemapb.DataType_Int64,
	IsPartitionKey: true,
})
```

### 插入数据（自动分区）

```go
insertResult, err := c.MilvusClient.Insert(ctx, &milvuspb.InsertRequest{
	DbName:         dbName,
	CollectionName: collectionName,
	FieldsData:     []*schemapb.FieldData{pkColumn, fVecColumn, partitionKeyColumn},
	NumRows:        uint32(rowNum),
})
```

Milvus 会根据 `partitionKeyColumn` 的值自动将数据分配到相应的分区。
