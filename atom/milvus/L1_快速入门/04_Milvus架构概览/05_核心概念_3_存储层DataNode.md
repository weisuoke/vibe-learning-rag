# 核心概念 3：存储层 DataNode

> **定位**：Milvus 架构概览 - 存储层核心组件
> **角色**：数据持久化与索引构建的执行者

---

## 1. 什么是 DataNode + IndexNode

### 1.1 DataNode：数据持久化执行者

**核心定义**：
- DataNode 是 Milvus 存储层的核心组件，负责将数据持久化到对象存储
- 类比：数据库的存储引擎，负责将内存中的数据写入磁盘

**在架构中的位置**：
```
Client → Proxy → DataNode → Object Storage (MinIO/S3)
                    ↓
                   WAL (Pulsar/Kafka)
```

### 1.2 IndexNode：索引构建执行者

**核心定义**：
- IndexNode 负责为向量数据构建索引，提升检索性能
- 类比：数据库的索引构建器，为数据创建高效的查询结构

**在架构中的位置**：
```
IndexCoord → IndexNode → Object Storage
                ↓
           Index Files (IVF, HNSW, etc.)
```

### 1.3 两者的协作关系

```
写入流程：
Client → Proxy → DataNode → Object Storage
                    ↓
                 Segment 封存
                    ↓
         IndexCoord 调度 IndexNode
                    ↓
              构建索引并存储
```

---

## 2. DataNode 核心职责

### 2.1 写入数据到对象存储

**职责说明**：
- 将 Proxy 转发的数据写入对象存储（MinIO/S3）
- 数据以 Segment 为单位组织

**存储格式**：
```
Object Storage:
├── collection_1/
│   ├── segment_1/
│   │   ├── insert_log_1.binlog    # 插入数据
│   │   ├── insert_log_2.binlog
│   │   └── delta_log_1.binlog     # 删除标记
│   └── segment_2/
│       └── ...
```

**在 RAG 中的意义**：
- 文档向量持久化存储，确保数据不丢失
- 支持大规模数据存储（TB 级别）

### 2.2 管理 WAL（Write-Ahead Log）

**职责说明**：
- 先写 WAL（Pulsar/Kafka），再写对象存储
- 保证数据持久性和故障恢复

**WAL 流程**：
```
1. 数据到达 DataNode
2. 写入 WAL (Pulsar/Kafka)  ← 先写日志
3. 返回成功给 Proxy
4. 异步写入 Object Storage  ← 后写存储
```

**为什么需要 WAL**：
- **持久性**：即使 DataNode 崩溃，数据也不会丢失
- **恢复**：重启后可以从 WAL 恢复未完成的写入
- **解耦**：写入和持久化异步进行，提升性能

### 2.3 数据 Compaction（合并）

**职责说明**：
- 将多个小 Segment 合并成大 Segment
- 清理删除标记，回收存储空间

**Compaction 类型**：

1. **Minor Compaction**：
   - 合并小文件，减少文件数量
   - 频率：高（每隔几分钟）

2. **Major Compaction**：
   - 清理删除标记，回收空间
   - 频率：低（每隔几小时）

**示例**：
```
Before Compaction:
├── segment_1 (100MB, 10% deleted)
├── segment_2 (50MB, 5% deleted)
└── segment_3 (80MB, 20% deleted)

After Compaction:
└── segment_merged (200MB, 0% deleted)  ← 合并 + 清理
```

**在 RAG 中的意义**：
- 优化存储空间，降低成本
- 提升查询性能（减少文件扫描）

### 2.4 Segment 封存（Sealing）

**职责说明**：
- 将 Growing Segment 转换为 Sealed Segment
- Sealed Segment 不可再写入，可以构建索引

**Segment 状态转换**：
```
Growing Segment (可写)
    ↓ (达到大小阈值或时间阈值)
Sealed Segment (只读)
    ↓ (IndexNode 构建索引)
Indexed Segment (可高效查询)
```

**封存触发条件**：
- Segment 大小达到阈值（如 512MB）
- Segment 存在时间达到阈值（如 10 分钟）
- 手动触发 Flush

---

## 3. IndexNode 核心职责

### 3.1 构建向量索引

**职责说明**：
- 为 Sealed Segment 构建向量索引
- 支持多种索引类型：IVF、HNSW、DiskANN 等

**索引类型对比**：

| 索引类型 | 适用场景 | 构建速度 | 查询速度 | 内存占用 |
|---------|---------|---------|---------|---------|
| FLAT | 小数据集 | 快 | 慢 | 低 |
| IVF_FLAT | 中等数据集 | 中 | 中 | 中 |
| IVF_PQ | 大数据集 | 中 | 快 | 低 |
| HNSW | 高性能查询 | 慢 | 极快 | 高 |
| DiskANN | 超大数据集 | 慢 | 快 | 极低 |

**在 RAG 中的选择**：
- **小规模（< 100 万）**：HNSW（追求速度）
- **中规模（100 万 - 1000 万）**：IVF_FLAT
- **大规模（> 1000 万）**：IVF_PQ 或 DiskANN

### 3.2 索引优化

**职责说明**：
- 根据数据分布调整索引参数
- 平衡构建时间、查询性能和内存占用

**优化策略**：
```python
# IVF 索引参数优化
nlist = int(sqrt(num_vectors))  # 聚类中心数量
nprobe = nlist * 0.1            # 查询时探测的聚类数

# HNSW 索引参数优化
M = 16                          # 每个节点的连接数
efConstruction = 200            # 构建时的搜索深度
```

### 3.3 后台任务执行

**职责说明**：
- 索引构建在后台异步执行
- 不阻塞数据写入和查询

**执行流程**：
```
1. DataNode 封存 Segment
2. IndexCoord 检测到新的 Sealed Segment
3. IndexCoord 调度 IndexNode 构建索引
4. IndexNode 构建索引并写入 Object Storage
5. 更新元数据，标记 Segment 为 Indexed
```

**在 RAG 中的意义**：
- 新插入的文档可以立即查询（虽然性能较低）
- 索引构建完成后，查询性能大幅提升

---

## 4. DataCoord 的作用

### 4.1 管理 DataNode 生命周期

**职责**：
- 监控 DataNode 健康状态
- 分配数据写入任务
- 处理 DataNode 故障

**示例**：
```
DataCoord:
├── DataNode-1 (健康, 负载 60%)
├── DataNode-2 (健康, 负载 40%)
└── DataNode-3 (故障, 正在恢复)

决策：将新数据分配给 DataNode-2
```

### 4.2 调度 Compaction 任务

**职责**：
- 监控 Segment 状态
- 决定何时触发 Compaction
- 分配 Compaction 任务给 DataNode

**调度策略**：
```python
# 触发 Minor Compaction 的条件
if segment_count > 10 and total_size < 1GB:
    trigger_minor_compaction()

# 触发 Major Compaction 的条件
if deleted_ratio > 0.2:
    trigger_major_compaction()
```

### 4.3 分配 Segment

**职责**：
- 为新数据分配 Segment ID
- 管理 Segment 的状态转换
- 协调 Segment 的分布

---

## 5. IndexCoord 的作用

### 5.1 管理 IndexNode 生命周期

**职责**：
- 监控 IndexNode 健康状态
- 分配索引构建任务
- 处理 IndexNode 故障

### 5.2 调度索引构建任务

**职责**：
- 检测新的 Sealed Segment
- 决定索引构建优先级
- 分配任务给 IndexNode

**调度策略**：
```python
# 优先级排序
priority = segment_size * query_frequency

# 分配给负载最低的 IndexNode
target_node = min(index_nodes, key=lambda n: n.load)
```

---

## 6. 架构图

```
┌─────────────────────────────────────────────────────────┐
│                      存储层架构                          │
└─────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │  DataCoord   │
                    └──────┬───────┘
                           │ 调度
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │DataNode1│  │DataNode2│  │DataNode3│
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └────────────┼────────────┘
                          ↓
                 ┌─────────────────┐
                 │ Object Storage  │
                 │  (MinIO / S3)   │
                 └─────────────────┘

                    ┌──────────────┐
                    │  IndexCoord  │
                    └──────┬───────┘
                           │ 调度
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │IndexNode1│ │IndexNode2│ │IndexNode3│
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └────────────┼────────────┘
                          ↓
                 ┌─────────────────┐
                 │ Object Storage  │
                 │  (Index Files)  │
                 └─────────────────┘

                    ┌──────────────┐
                    │     WAL      │
                    │(Pulsar/Kafka)│
                    └──────────────┘
```

---

## 7. 存储架构

### 7.1 对象存储

**支持的存储后端**：
- **MinIO**：本地部署，开源
- **S3**：AWS 云存储
- **OSS**：阿里云对象存储
- **GCS**：Google 云存储

**为什么使用对象存储**：
- **成本低**：比本地磁盘便宜
- **可扩展**：无限容量
- **高可用**：自动备份和容错

### 7.2 WAL（Write-Ahead Log）

**支持的消息队列**：
- **Pulsar**：推荐，支持多租户
- **Kafka**：常用，生态成熟

**WAL 的作用**：
- 保证数据持久性
- 支持故障恢复
- 解耦写入和存储

### 7.3 元数据存储

**使用 etcd**：
- 存储 Collection、Segment、Index 等元数据
- 提供分布式一致性
- 支持 Watch 机制（监听变化）

---

## 8. 数据流

### 8.1 写入流程

```
1. Client 发送插入请求
   ↓
2. Proxy 接收并转发给 DataNode
   ↓
3. DataNode 写入 WAL (Pulsar/Kafka)
   ↓
4. DataNode 返回成功给 Proxy
   ↓
5. Proxy 返回成功给 Client
   ↓
6. DataNode 异步写入 Object Storage
   ↓
7. DataNode 更新元数据到 etcd
```

### 8.2 Compaction 流程

```
1. DataCoord 监控 Segment 状态
   ↓
2. 检测到需要 Compaction 的 Segment
   ↓
3. DataCoord 调度 DataNode 执行 Compaction
   ↓
4. DataNode 读取多个 Segment
   ↓
5. DataNode 合并数据并清理删除标记
   ↓
6. DataNode 写入新的 Segment 到 Object Storage
   ↓
7. DataNode 更新元数据，删除旧 Segment
```

### 8.3 索引构建流程

```
1. DataNode 封存 Segment (Growing → Sealed)
   ↓
2. IndexCoord 检测到新的 Sealed Segment
   ↓
3. IndexCoord 调度 IndexNode 构建索引
   ↓
4. IndexNode 从 Object Storage 读取 Segment
   ↓
5. IndexNode 构建索引（IVF, HNSW, etc.）
   ↓
6. IndexNode 写入索引文件到 Object Storage
   ↓
7. IndexNode 更新元数据，标记 Segment 为 Indexed
```

---

## 9. 代码示例：监控数据持久化

### 9.1 监控 Segment 状态

```python
from pymilvus import connections, utility

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

# 获取 Collection 信息
collection_name = "rag_documents"
stats = utility.get_query_segment_info(collection_name)

print(f"Collection: {collection_name}")
print(f"Total Segments: {len(stats)}")
print()

# 遍历所有 Segment
for segment in stats:
    print(f"Segment ID: {segment.segmentID}")
    print(f"  State: {segment.state}")  # Growing, Sealed, Flushed
    print(f"  Rows: {segment.num_rows}")
    print(f"  Index: {segment.index_name}")
    print(f"  Node ID: {segment.nodeID}")
    print()
```

**输出示例**：
```
Collection: rag_documents
Total Segments: 5

Segment ID: 448900612345678901
  State: Sealed
  Rows: 524288
  Index: HNSW
  Node ID: 1

Segment ID: 448900612345678902
  State: Growing
  Rows: 10240
  Index: None
  Node ID: 2
```

### 9.2 手动触发 Flush

```python
from pymilvus import Collection

# 获取 Collection
collection = Collection("rag_documents")

# 手动触发 Flush（封存 Growing Segment）
collection.flush()
print("Flush completed. Growing segments are now sealed.")

# 等待索引构建完成
collection.wait_for_index_building_complete()
print("Index building completed.")
```

### 9.3 监控 Compaction 状态

```python
from pymilvus import Collection
import time

collection = Collection("rag_documents")

# 手动触发 Compaction
compaction_id = collection.compact()
print(f"Compaction started: {compaction_id}")

# 监控 Compaction 进度
while True:
    state = collection.get_compaction_state(compaction_id)
    print(f"Compaction state: {state}")

    if state.state == 3:  # Completed
        print("Compaction completed!")
        break

    time.sleep(5)

# 获取 Compaction 计划
plans = collection.get_compaction_plans(compaction_id)
print(f"Compaction plans: {plans}")
```

### 9.4 查询存储统计信息

```python
from pymilvus import Collection

collection = Collection("rag_documents")

# 获取存储统计信息
stats = collection.get_stats()
print(f"Row count: {stats['row_count']}")
print(f"Data size: {stats['data_size']} bytes")

# 计算存储成本（假设 S3 每 GB $0.023/月）
data_size_gb = int(stats['data_size']) / (1024 ** 3)
monthly_cost = data_size_gb * 0.023
print(f"Estimated monthly storage cost: ${monthly_cost:.2f}")
```

---

## 10. 在 RAG 中的应用

### 10.1 数据持久化：确保文档向量不丢失

**场景**：
- 用户上传了 10 万份文档，生成了 100 万个向量
- 需要确保这些向量持久化存储，不会因为服务重启而丢失

**DataNode 的作用**：
- 将向量数据写入对象存储（S3/MinIO）
- 通过 WAL 保证数据持久性
- 即使 Milvus 重启，数据也不会丢失

**代码示例**：
```python
from pymilvus import Collection

collection = Collection("rag_documents")

# 插入数据
collection.insert([
    [1, 2, 3],  # IDs
    [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],  # Vectors
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"]  # Metadata
])

# 手动 Flush，确保数据持久化
collection.flush()
print("Data persisted to object storage.")
```

### 10.2 索引优化：提升检索性能

**场景**：
- RAG 系统需要在 100 万个文档向量中检索 Top-K 相似文档
- 没有索引时，查询需要 10 秒；有索引后，查询只需 0.1 秒

**IndexNode 的作用**：
- 为向量数据构建 HNSW 索引
- 查询性能提升 100 倍

**代码示例**：
```python
from pymilvus import Collection

collection = Collection("rag_documents")

# 创建 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 等待索引构建完成
collection.wait_for_index_building_complete()
print("Index built. Query performance improved 100x.")
```

### 10.3 存储成本：对象存储更便宜

**场景**：
- RAG 系统存储了 1 TB 的向量数据
- 本地 SSD：$100/TB/月
- S3 对象存储：$23/TB/月

**DataNode 的作用**：
- 将数据存储到 S3，降低存储成本 77%

**成本对比**：
```python
# 计算存储成本
data_size_tb = 1  # 1 TB

local_ssd_cost = data_size_tb * 100  # $100/TB/月
s3_cost = data_size_tb * 23          # $23/TB/月

savings = local_ssd_cost - s3_cost
savings_percent = (savings / local_ssd_cost) * 100

print(f"Local SSD cost: ${local_ssd_cost}/month")
print(f"S3 cost: ${s3_cost}/month")
print(f"Savings: ${savings}/month ({savings_percent:.0f}%)")
```

**输出**：
```
Local SSD cost: $100/month
S3 cost: $23/month
Savings: $77/month (77%)
```

---

## 11. 性能考虑

### 11.1 DataNode 是 I/O 密集型

**特点**：
- 大量读写对象存储
- 网络带宽是瓶颈

**优化策略**：
- 使用高带宽网络（10 Gbps+）
- 启用对象存储的多线程上传
- 使用 SSD 作为本地缓存

### 11.2 IndexNode 是 CPU 密集型

**特点**：
- 索引构建需要大量计算
- CPU 是瓶颈

**优化策略**：
- 使用高性能 CPU（多核）
- 并行构建多个索引
- 使用 GPU 加速（部分索引类型）

### 11.3 Compaction 策略：平衡存储效率和查询性能

**策略 1：激进 Compaction**
- 优点：存储空间小，查询性能高
- 缺点：Compaction 开销大，影响写入性能

**策略 2：保守 Compaction**
- 优点：写入性能高
- 缺点：存储空间大，查询性能低

**推荐策略**：
```python
# 根据业务场景调整
if write_heavy:
    # 写入密集型：保守 Compaction
    compaction_interval = 3600  # 每小时一次
else:
    # 查询密集型：激进 Compaction
    compaction_interval = 600   # 每 10 分钟一次
```

---

## 12. 总结

### 12.1 DataNode 核心要点

1. **数据持久化**：将数据写入对象存储
2. **WAL 管理**：保证数据持久性和故障恢复
3. **Compaction**：合并小文件，优化存储
4. **Segment 封存**：将 Growing Segment 转换为 Sealed Segment

### 12.2 IndexNode 核心要点

1. **索引构建**：为向量数据构建索引（IVF, HNSW, etc.）
2. **索引优化**：调整参数，平衡性能和成本
3. **后台任务**：异步执行，不阻塞写入和查询

### 12.3 在 RAG 中的价值

1. **数据安全**：对象存储 + WAL 保证数据不丢失
2. **性能提升**：索引构建提升查询性能 100 倍
3. **成本优化**：对象存储降低存储成本 77%

### 12.4 关键设计思想

1. **存储计算分离**：数据存储在对象存储，计算在 DataNode/IndexNode
2. **异步处理**：写入、Compaction、索引构建都是异步的
3. **可扩展性**：通过增加 DataNode/IndexNode 实现水平扩展

---

**下一步**：学习 Milvus 的数据插入与查询实战，将架构知识应用到实际代码中。
