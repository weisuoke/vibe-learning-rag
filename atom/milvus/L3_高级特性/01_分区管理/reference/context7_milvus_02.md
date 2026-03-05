---
type: context7_documentation
library: milvus
version: v2.6.x
fetched_at: 2026-02-25
knowledge_point: 01_分区管理
context7_query: multi-tenancy, partition key, 100K collections, scalability
---

# Context7 文档：Milvus 多租户与分区策略

## 文档来源
- 库名称：Milvus Documentation
- 版本：v2.6.x
- 官方文档链接：https://github.com/milvus-io/milvus-docs/blob/v2.6.x/

## 关键信息提取

### 1. 多租户策略概览

Milvus 提供多种多租户策略，每种策略在数据隔离、灵活性和可扩展性方面有不同的权衡。

#### 策略对比

| 策略 | 数据隔离 | 可扩展性 | Schema 灵活性 | RBAC 支持 | 适用场景 |
|------|----------|----------|---------------|-----------|----------|
| Database-level | 强 | 低 | 高 | 是 | 少量租户，需要强隔离 |
| Collection-level | 强 | 中 | 高 | 是 | 中等数量租户，需要独立 Schema |
| Partition-level | 中 | 中 | 低 | 否 | 中等数量租户，共享 Schema |
| Partition Key-level | 弱 | 高 | 低 | 否 | 大量租户（百万级），共享 Schema |

### 2. Partition Key-level 多租户（Milvus 2.6 推荐）

**定义**：所有租户共享一个 Collection 和 Schema，但每个租户的数据会根据 Partition Key 值自动路由到 16 个物理隔离的分区中。

**核心特性**：
- **自动分区**：Milvus 自动将数据路由到 16 个物理分区
- **逻辑隔离**：虽然多个租户可能共享一个物理分区，但数据在逻辑上是分离的
- **最高可扩展性**：支持数百万租户
- **共享 Schema**：所有租户必须共享相同的数据 Schema
- **无 RBAC 支持**：不支持 Partition Key 级别的 RBAC
- **灵活查询**：可以单独查询租户，也可以跨多个分区进行聚合查询

**Python 示例**：
```python
from langchain_core.documents import Document

docs = [
    Document(page_content="i worked at kensho", metadata={"namespace": "harrison"}),
    Document(page_content="i worked at facebook", metadata={"namespace": "ankush"}),
]
vectorstore = Milvus.from_documents(
    docs,
    embeddings,
    collection_name="partitioned_collection",
    connection_args={"uri": URI},
    partition_key_field="namespace",  # 使用 "namespace" 字段作为 Partition Key
)
```

**适用场景**：
- SaaS 应用，需要支持大量租户
- 数据 Schema 统一的多租户系统
- 需要跨租户聚合查询或分析的场景

**优点**：
- ✅ 最高可扩展性（支持数百万租户）
- ✅ 自动分区管理，无需手动创建分区
- ✅ 支持跨租户聚合查询

**缺点**：
- ❌ 数据隔离较弱（多个租户可能共享物理分区）
- ❌ 所有租户必须共享相同的 Schema
- ❌ 不支持 RBAC

### 3. Partition-level 多租户

**定义**：每个租户分配到一个手动创建的分区，一个 Collection 最多可以有 1,024 个分区。

**核心特性**：
- **物理隔离**：每个租户的数据在物理上由分区分离
- **手动管理**：需要手动创建分区
- **共享 Schema**：所有租户必须共享相同的数据 Schema
- **无 RBAC 支持**：不支持 Partition 级别的 RBAC
- **灵活查询**：可以单独查询租户，也可以跨多个分区进行聚合查询
- **热冷数据管理**：可以灵活加载或释放特定租户的数据

**适用场景**：
- 中等数量租户（最多 1,024 个）
- 需要物理隔离的场景
- 需要热冷数据管理的场景

**优点**：
- ✅ 物理隔离（每个租户独立分区）
- ✅ 支持热冷数据管理（可以加载/释放特定分区）
- ✅ 支持跨租户聚合查询

**缺点**：
- ❌ 可扩展性有限（最多 1,024 个分区）
- ❌ 需要手动创建分区
- ❌ 所有租户必须共享相同的 Schema
- ❌ 不支持 RBAC

### 4. Collection-level 多租户

**定义**：每个租户分配到一个独立的 Collection。

**核心特性**：
- **强隔离**：每个租户的数据在物理上完全隔离
- **Schema 灵活性**：每个租户可以有独立的 Schema
- **RBAC 支持**：支持 Collection 级别的 RBAC
- **独立管理**：每个租户可以独立管理索引、加载、释放等操作

**适用场景**：
- 中等数量租户
- 需要强隔离的场景
- 每个租户需要独立 Schema 的场景

**优点**：
- ✅ 强隔离（每个租户独立 Collection）
- ✅ Schema 灵活性（每个租户可以有独立 Schema）
- ✅ 支持 RBAC

**缺点**：
- ❌ 可扩展性有限（Collection 数量有限制）
- ❌ 资源开销较大（每个 Collection 需要独立资源）

### 5. Database-level 多租户

**定义**：每个租户分配到一个独立的 Database。

**核心特性**：
- **最强隔离**：每个租户的数据在物理上完全隔离
- **最高灵活性**：每个租户可以有多个 Collection 和独立 Schema
- **RBAC 支持**：支持 Database 级别的 RBAC
- **独立管理**：每个租户可以独立管理所有资源

**适用场景**：
- 少量租户
- 需要最强隔离的场景
- 每个租户需要多个 Collection 的场景

**优点**：
- ✅ 最强隔离（每个租户独立 Database）
- ✅ 最高灵活性（每个租户可以有多个 Collection）
- ✅ 支持 RBAC

**缺点**：
- ❌ 可扩展性最低（Database 数量有限制）
- ❌ 资源开销最大（每个 Database 需要独立资源）

### 6. 选择多租户策略的建议

**来源**：Choosing the right multi-tenancy strategy

**核心建议**：
- **可扩展性优先级**：Partition Key-level > Partition-level > Collection-level > Database-level
- **大量租户场景**：如果应用预计有大量租户，推荐使用 Partition Key-level 策略
- **中等租户场景**：如果需要物理隔离和热冷数据管理，推荐使用 Partition-level 策略
- **少量租户场景**：如果需要强隔离和 Schema 灵活性，推荐使用 Collection-level 或 Database-level 策略

### 7. Partition 的基本限制

**来源**：Manage Partitions

**核心限制**：
- 一个 Collection 最多可以有 **1,024 个分区**
- 创建 Collection 时，Milvus 会自动创建一个名为 **_default** 的分区
- 如果不添加其他分区，所有插入的实体都会进入默认分区
- 可以添加更多分区，并根据特定标准将实体插入其中
- 可以将搜索和查询限制在特定分区内，提升搜索性能

### 8. Partition Key 的自动分区机制

**核心机制**：
- 当字段标记为 `is_partition_key=True` 时，Milvus 会自动根据该字段的值进行分区
- Milvus 会自动将数据路由到 **16 个物理分区**
- 虽然多个租户可能共享一个物理分区，但数据在逻辑上是分离的
- 这种机制支持数百万租户的可扩展性

### 9. 100K Collections 支持（Milvus 2.6 新特性）

**来源**：README.md

**核心信息**：
- Milvus 支持多租户，通过 Database、Collection、Partition 或 Partition Key 级别的隔离
- 灵活的策略允许单个集群处理 **数百到数百万租户**
- 确保优化的搜索性能和灵活的访问控制
- Milvus 通过热冷存储增强成本效益：
  - 频繁访问的热数据可以存储在内存或 SSD 中以获得更好的性能
  - 较少访问的冷数据保存在较慢、成本效益高的存储中
- 这种机制可以显著降低成本，同时保持关键任务的高性能

**关键点**：
- **100K Collections** 是指 Milvus 2.6 支持大规模多租户场景，可以处理数百到数百万租户
- 这不是指单个 Collection 可以有 100K 个分区，而是指整个 Milvus 集群可以支持大量的 Collection 或租户
- 通过 Partition Key-level 策略，可以在单个 Collection 中支持数百万租户

## 核心概念总结

### 1. Partition 的两种使用方式

#### 方式 1：手动分区管理（Partition-level）
- 使用 `create_partition()` 手动创建分区
- 插入数据时指定分区名称
- 搜索时指定分区名称
- 最多支持 1,024 个分区
- 适用于中等数量租户

#### 方式 2：Partition Key 自动分区（Partition Key-level，Milvus 2.6 推荐）
- 在 Schema 中标记字段为 `is_partition_key=True`
- Milvus 自动根据该字段的值进行分区（自动路由到 16 个物理分区）
- 无需手动创建分区，无需在插入和搜索时指定分区名称
- 支持数百万租户
- 适用于大规模多租户场景

### 2. 多租户策略选择

**可扩展性排序**：
1. **Partition Key-level**：支持数百万租户（最高）
2. **Partition-level**：支持最多 1,024 个租户
3. **Collection-level**：支持中等数量租户
4. **Database-level**：支持少量租户（最低）

**数据隔离排序**：
1. **Database-level**：最强隔离
2. **Collection-level**：强隔离
3. **Partition-level**：中等隔离
4. **Partition Key-level**：弱隔离（最低）

**Schema 灵活性排序**：
1. **Database-level**：最高灵活性
2. **Collection-level**：高灵活性
3. **Partition-level**：低灵活性（共享 Schema）
4. **Partition Key-level**：低灵活性（共享 Schema）

### 3. Partition 的性能优化

1. **Partition Key 隔离**：启用 `partitionkey.isolation` 属性，提升搜索性能（仅支持 HNSW 索引）
2. **分区搜索**：在搜索时指定分区，减少搜索范围
3. **分区加载**：只加载需要搜索的分区，节省内存
4. **热冷数据管理**：可以灵活加载或释放特定租户的数据（Partition-level）

### 4. Partition 的限制

1. **最大分区数**：一个 Collection 最多可以有 **1,024 个分区**（Partition-level）
2. **Partition Key 物理分区数**：Partition Key 自动路由到 **16 个物理分区**（Partition Key-level）
3. **Partition Key 隔离限制**：当前仅支持 HNSW 索引类型
4. **加载要求**：搜索前必须加载相关分区，否则会返回错误
5. **RBAC 限制**：Partition-level 和 Partition Key-level 不支持 RBAC

## 需要进一步调研的技术点

1. **Partition Key 的哈希算法**：如何将 Partition Key 值映射到 16 个物理分区？
2. **Partition 数量对性能的影响**：分区数量与检索性能的关系？
3. **100K Collections 的实现机制**：Milvus 2.6 如何支持大规模 Collection？
4. **Partition 的存储机制**：分区数据在磁盘上如何组织？
5. **Partition Key 隔离的实现原理**：为什么只支持 HNSW 索引？
6. **热冷存储的实现机制**：如何实现热冷数据的自动管理？
