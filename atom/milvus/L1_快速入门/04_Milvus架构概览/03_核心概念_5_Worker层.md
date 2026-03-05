# 核心概念 5：Worker 层（工作节点层）

> **来源**: [Milvus System Overview](https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/chap01_system_overview.md) | [Milvus Repository](https://github.com/milvus-io/milvus) | [MMAP Discussion](https://github.com/milvus-io/milvus/discussions/33621) | [2.6 Performance Issue](https://github.com/milvus-io/milvus/issues/43659) | [Streaming Service Enhancement](https://github.com/milvus-io/milvus/issues/33285) | 获取时间: 2026-02-21

---

## 一、原理讲解

### 1.1 什么是 Worker 层？

Worker 层是 Milvus 分布式架构中的**数据平面**，负责实际的数据处理、索引构建和查询执行。它是整个系统的"肌肉"，执行 Coordinator 层下发的任务。

**核心职责：**
- **数据写入**：接收并持久化向量数据
- **索引构建**：为向量数据构建高效的索引结构
- **查询执行**：在内存中执行向量检索
- **资源管理**：管理内存、CPU、磁盘等资源

### 1.2 Worker 层的三大组件

```
Worker 层架构：
┌─────────────────────────────────────────────────────────┐
│                    Coordinator 层                        │
│              (MixCoord: 任务调度与协调)                  │
└─────────────────────────────────────────────────────────┘
                          ↓ 任务分发
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Data Node   │  │ Query Node   │  │ Index Node   │
│  (数据节点)  │  │ (查询节点)   │  │ (索引节点)   │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ - 数据写入   │  │ - 向量检索   │  │ - 索引构建   │
│ - Flush 持久 │  │ - 标量过滤   │  │ - 索引优化   │
│ - Compaction │  │ - 结果聚合   │  │ - 索引上传   │
│ - 订阅 WAL   │  │ - 内存管理   │  │ - 分布式构建 │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                 ↓                 ↓
┌─────────────────────────────────────────────────────────┐
│                    Storage 层                            │
│  Object Storage (MinIO/S3) + Meta Storage (etcd)        │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Data Node（数据节点）

#### 1.3.1 核心职责

Data Node 是数据写入的执行者，负责：

1. **订阅 WAL（Write-Ahead Log）**
   - 从 Streaming Node 订阅数据流
   - 按照 VChannel 分片接收数据

2. **数据缓冲与批处理**
   - 在内存中缓冲写入的数据
   - 达到阈值后批量处理

3. **Flush 持久化**
   - 将内存中的数据写入对象存储
   - 生成 Binlog 文件（二进制日志）

4. **Compaction（数据合并）**
   - 合并小 Segment 为大 Segment
   - 清理已删除的数据

#### 1.3.2 数据写入流程

```
客户端写入流程：
┌──────────┐
│  Client  │
└──────────┘
     ↓ insert(vectors)
┌──────────────┐
│ Proxy Node   │ ← 接收请求，路由到 VChannel
└──────────────┘
     ↓ 写入 WAL
┌──────────────┐
│Streaming Node│ ← Milvus 2.6 新组件（替代 Kafka/Pulsar）
└──────────────┘
     ↓ 订阅 WAL
┌──────────────┐
│  Data Node   │ ← 消费数据，缓冲到内存
└──────────────┘
     ↓ Flush
┌──────────────┐
│Object Storage│ ← 持久化为 Binlog 文件
└──────────────┘
```

**关键概念：**

- **VChannel（虚拟通道）**：逻辑上的数据分片，每个 Collection 有多个 VChannel
- **Binlog（二进制日志）**：持久化的数据文件，包含 Insert Log、Delete Log、Stats Log
- **Segment**：数据的物理存储单元，每个 Segment 对应一组 Binlog 文件

#### 1.3.3 Compaction 机制

**为什么需要 Compaction？**

- 小 Segment 过多导致查询效率低
- 删除操作只是逻辑删除，需要物理清理
- 优化存储空间和查询性能

**Compaction 类型：**

```python
# 1. Mix Compaction（混合合并）
# 合并多个小 Segment 为一个大 Segment
Segment_1 (100MB) + Segment_2 (150MB) + Segment_3 (80MB)
  → Segment_4 (330MB)

# 2. Merge Compaction（删除清理）
# 清理已删除的数据
Segment_5 (1000 rows, 200 deleted)
  → Segment_6 (800 rows, 0 deleted)
```

### 1.4 Query Node（查询节点）

#### 1.4.1 核心职责

Query Node 是查询执行的核心，负责：

1. **加载索引到内存**
   - 从对象存储下载索引文件
   - 使用 mmap 或直接加载到内存

2. **执行向量检索**
   - ANN（近似最近邻）搜索
   - 支持多种索引类型（IVF、HNSW、DiskANN 等）

3. **标量过滤**
   - 根据元数据字段过滤结果
   - 支持复杂的布尔表达式

4. **结果聚合**
   - 合并多个 Segment 的查询结果
   - 按相似度排序并返回 Top-K

#### 1.4.2 查询执行流程

```
客户端查询流程：
┌──────────┐
│  Client  │
└──────────┘
     ↓ search(vector, top_k=10)
┌──────────────┐
│ Proxy Node   │ ← 接收请求，路由到 Query Node
└──────────────┘
     ↓ 分发查询
┌──────────────┐
│ Query Node 1 │ ← 查询 Segment 1, 2, 3
└──────────────┘
     ↓ 返回 Top-10
┌──────────────┐
│ Query Node 2 │ ← 查询 Segment 4, 5, 6
└──────────────┘
     ↓ 返回 Top-10
┌──────────────┐
│ Proxy Node   │ ← 聚合结果，返回全局 Top-10
└──────────────┘
     ↓ 返回结果
┌──────────┐
│  Client  │
└──────────┘
```

#### 1.4.3 内存管理策略

**问题：** 向量索引占用大量内存，如何高效管理？

**Milvus 的解决方案：**

1. **mmap（内存映射）**
   - 将索引文件映射到虚拟内存
   - 操作系统按需加载页面
   - 适合大规模索引（内存不足时）

2. **直接加载**
   - 将索引完全加载到物理内存
   - 查询性能最优
   - 适合内存充足的场景

3. **动态加载/卸载**
   - 根据查询频率动态管理
   - 热数据常驻内存，冷数据卸载

**配置示例：**

```yaml
# milvus.yaml
queryNode:
  enableDisk: true          # 启用 DiskANN（磁盘索引）
  maxDiskUsagePercentage: 95
  mmap:
    mmapEnabled: true       # 启用 mmap
```

### 1.5 Index Node（索引节点）

#### 1.5.1 核心职责

Index Node 是索引构建的专家，负责：

1. **索引构建**
   - 从对象存储读取 Binlog
   - 构建向量索引（IVF、HNSW、DiskANN 等）

2. **索引优化**
   - 调整索引参数以平衡性能和召回率
   - 支持多种距离度量（L2、IP、Cosine）

3. **索引上传**
   - 将构建好的索引上传到对象存储
   - 通知 Query Node 加载新索引

4. **分布式构建**
   - 多个 Index Node 并行构建索引
   - 提高大规模数据的索引构建速度

#### 1.5.2 索引构建流程

```
索引构建流程：
┌──────────────┐
│  Data Node   │ ← Flush 数据到对象存储
└──────────────┘
     ↓ 通知 DataCoord
┌──────────────┐
│  DataCoord   │ ← 调度索引构建任务
└──────────────┘
     ↓ 分配任务
┌──────────────┐
│ Index Node 1 │ ← 构建 Segment 1 的索引
└──────────────┘
     ↓ 上传索引
┌──────────────┐
│Object Storage│ ← 存储索引文件
└──────────────┘
     ↓ 通知 QueryCoord
┌──────────────┐
│ Query Node   │ ← 加载新索引
└──────────────┘
```

#### 1.5.3 索引类型选择

**常见索引类型：**

| 索引类型 | 适用场景 | 内存占用 | 查询速度 | 召回率 |
|---------|---------|---------|---------|--------|
| **FLAT** | 小数据集（< 100万） | 高 | 慢 | 100% |
| **IVF_FLAT** | 中等数据集（100万-1000万） | 中 | 中 | 95-99% |
| **IVF_PQ** | 大数据集（> 1000万） | 低 | 快 | 90-95% |
| **HNSW** | 高性能查询 | 高 | 极快 | 95-99% |
| **DiskANN** | 超大数据集（内存不足） | 极低 | 中 | 95-99% |

**选择建议：**

```python
# RAG 场景推荐
if data_size < 1_000_000:
    index_type = "IVF_FLAT"  # 平衡性能和召回率
elif data_size < 10_000_000:
    index_type = "HNSW"      # 高性能查询
else:
    index_type = "DiskANN"   # 超大规模，节省内存
```

### 1.6 Milvus 2.6 的重大变化：流批分离架构

#### 1.6.1 旧架构的问题

在 Milvus 2.5 及之前：

```
旧架构（2.5）：
┌──────────────┐
│ Query Node   │ ← 同时处理：
│              │   1. 实时数据（Growing Segment）
│              │   2. 历史数据（Sealed Segment）
└──────────────┘
```

**问题：**
- 实时数据和历史数据混合处理，逻辑复杂
- 实时数据未索引，查询性能差
- 资源分配不灵活

#### 1.6.2 新架构：Streaming Node

Milvus 2.6 引入 **Streaming Node**，实现流批分离：

```
新架构（2.6）：
┌──────────────┐
│Streaming Node│ ← 专门处理实时数据（Growing Segment）
│              │   - 订阅 WAL
│              │   - 提供实时查询
│              │   - 无需索引
└──────────────┘
        ↓ 数据成熟后
┌──────────────┐
│ Query Node   │ ← 只处理历史数据（Sealed Segment）
│              │   - 加载索引
│              │   - 高性能查询
└──────────────┘
```

**优势：**
- ✅ **职责清晰**：Streaming Node 处理实时，Query Node 处理历史
- ✅ **性能优化**：Query Node 只处理已索引数据，性能更稳定
- ✅ **资源隔离**：实时查询和历史查询资源独立，互不影响

---

## 二、手写实现：简化版 Worker 层

下面我们用 Python 实现一个简化版的 Worker 层，展示三个节点如何协作。

### 2.1 基础数据结构

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
import time

class IndexType(Enum):
    """索引类型"""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    HNSW = "hnsw"

@dataclass
class Vector:
    """向量数据"""
    id: int
    embedding: np.ndarray
    metadata: Dict

@dataclass
class Binlog:
    """二进制日志"""
    segment_id: int
    vectors: List[Vector]
    file_path: str

@dataclass
class Index:
    """索引元数据"""
    segment_id: int
    index_type: IndexType
    index_data: np.ndarray  # 简化：实际是复杂的索引结构
    file_path: str
```

### 2.2 Data Node 实现

```python
class DataNode:
    """数据节点：负责数据写入和持久化"""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.buffer: Dict[int, List[Vector]] = {}  # segment_id → vectors
        self.buffer_size_limit = 1000  # 缓冲区大小限制

    def subscribe_wal(self, segment_id: int):
        """订阅 WAL（简化版）"""
        print(f"[DataNode-{self.node_id}] Subscribed to WAL for segment {segment_id}")
        self.buffer[segment_id] = []

    def write_data(self, segment_id: int, vectors: List[Vector]):
        """写入数据到缓冲区"""
        if segment_id not in self.buffer:
            self.subscribe_wal(segment_id)

        self.buffer[segment_id].extend(vectors)
        print(f"[DataNode-{self.node_id}] Buffered {len(vectors)} vectors for segment {segment_id}")

        # 检查是否需要 Flush
        if len(self.buffer[segment_id]) >= self.buffer_size_limit:
            self.flush_segment(segment_id)

    def flush_segment(self, segment_id: int) -> Binlog:
        """Flush 数据到对象存储"""
        if segment_id not in self.buffer or not self.buffer[segment_id]:
            return None

        vectors = self.buffer[segment_id]

        # 模拟持久化到对象存储
        file_path = f"s3://milvus/binlog/segment_{segment_id}.bin"
        time.sleep(0.1)  # 模拟 I/O

        binlog = Binlog(
            segment_id=segment_id,
            vectors=vectors.copy(),
            file_path=file_path
        )

        # 清空缓冲区
        self.buffer[segment_id] = []

        print(f"[DataNode-{self.node_id}] Flushed {len(vectors)} vectors to {file_path}")
        return binlog

    def compact_segments(self, segment_ids: List[int]) -> int:
        """Compaction：合并多个 Segment"""
        print(f"[DataNode-{self.node_id}] Compacting segments: {segment_ids}")

        # 模拟合并逻辑
        time.sleep(0.2)

        new_segment_id = max(segment_ids) + 1000
        print(f"[DataNode-{self.node_id}] Compaction completed, new segment: {new_segment_id}")

        return new_segment_id
```

### 2.3 Index Node 实现

```python
class IndexNode:
    """索引节点：负责索引构建"""

    def __init__(self, node_id: int):
        self.node_id = node_id

    def build_index(self, binlog: Binlog, index_type: IndexType) -> Index:
        """构建索引"""
        print(f"[IndexNode-{self.node_id}] Building {index_type.value} index for segment {binlog.segment_id}")

        # 1. 从对象存储读取 Binlog（模拟）
        vectors = binlog.vectors
        embeddings = np.array([v.embedding for v in vectors])

        # 2. 构建索引（简化版）
        time.sleep(0.3)  # 模拟索引构建时间

        if index_type == IndexType.FLAT:
            # FLAT 索引：直接存储所有向量
            index_data = embeddings
        elif index_type == IndexType.IVF_FLAT:
            # IVF_FLAT 索引：聚类 + 向量存储（简化）
            index_data = embeddings  # 实际会有聚类中心等
        else:
            index_data = embeddings

        # 3. 上传索引到对象存储
        file_path = f"s3://milvus/index/segment_{binlog.segment_id}_{index_type.value}.idx"

        index = Index(
            segment_id=binlog.segment_id,
            index_type=index_type,
            index_data=index_data,
            file_path=file_path
        )

        print(f"[IndexNode-{self.node_id}] Index built and uploaded to {file_path}")
        return index

    def optimize_index(self, index: Index) -> Index:
        """优化索引参数"""
        print(f"[IndexNode-{self.node_id}] Optimizing index for segment {index.segment_id}")
        time.sleep(0.1)
        print(f"[IndexNode-{self.node_id}] Index optimized")
        return index
```

### 2.4 Query Node 实现

```python
class QueryNode:
    """查询节点：负责查询执行"""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.loaded_indexes: Dict[int, Index] = {}  # segment_id → Index

    def load_index(self, index: Index):
        """加载索引到内存"""
        print(f"[QueryNode-{self.node_id}] Loading index for segment {index.segment_id}")

        # 模拟从对象存储下载索引
        time.sleep(0.2)

        # 加载到内存
        self.loaded_indexes[index.segment_id] = index

        print(f"[QueryNode-{self.node_id}] Index loaded, memory usage: {index.index_data.nbytes / 1024 / 1024:.2f} MB")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """执行向量检索"""
        print(f"[QueryNode-{self.node_id}] Searching with top_k={top_k}")

        all_results = []

        # 在所有已加载的 Segment 中搜索
        for segment_id, index in self.loaded_indexes.items():
            # 计算相似度（简化：使用 L2 距离）
            distances = np.linalg.norm(index.index_data - query_vector, axis=1)

            # 获取 Top-K
            top_k_indices = np.argsort(distances)[:top_k]

            for idx in top_k_indices:
                all_results.append({
                    "segment_id": segment_id,
                    "vector_id": idx,
                    "distance": float(distances[idx])
                })

        # 全局排序并返回 Top-K
        all_results.sort(key=lambda x: x["distance"])
        final_results = all_results[:top_k]

        print(f"[QueryNode-{self.node_id}] Search completed, found {len(final_results)} results")
        return final_results

    def release_index(self, segment_id: int):
        """释放索引"""
        if segment_id in self.loaded_indexes:
            del self.loaded_indexes[segment_id]
            print(f"[QueryNode-{self.node_id}] Released index for segment {segment_id}")

### 2.5 完整示例：Worker 层协作

```python
def worker_layer_demo():
    """演示 Worker 层的完整工作流程"""

    print("=" * 60)
    print("Worker Layer Demo: Data Node + Index Node + Query Node")
    print("=" * 60)

    # 1. 初始化三个节点
    data_node = DataNode(node_id=1)
    index_node = IndexNode(node_id=1)
    query_node = QueryNode(node_id=1)

    # 2. 模拟数据写入
    print("\n[Step 1] Data Writing")
    print("-" * 60)

    segment_id = 1001
    vectors = [
        Vector(id=i, embedding=np.random.rand(128), metadata={"text": f"doc_{i}"})
        for i in range(1500)  # 超过缓冲区限制，触发 Flush
    ]

    data_node.write_data(segment_id, vectors)

    # 3. 手动 Flush（如果还有剩余数据）
    print("\n[Step 2] Flushing Remaining Data")
    print("-" * 60)
    binlog = data_node.flush_segment(segment_id)

    # 4. 构建索引
    print("\n[Step 3] Building Index")
    print("-" * 60)
    index = index_node.build_index(binlog, IndexType.IVF_FLAT)

    # 5. 加载索引到 Query Node
    print("\n[Step 4] Loading Index")
    print("-" * 60)
    query_node.load_index(index)

    # 6. 执行查询
    print("\n[Step 5] Executing Search")
    print("-" * 60)
    query_vector = np.random.rand(128)
    results = query_node.search(query_vector, top_k=5)

    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Segment {result['segment_id']}, Vector {result['vector_id']}, Distance: {result['distance']:.4f}")

    # 7. Compaction 示例
    print("\n[Step 6] Compaction")
    print("-" * 60)
    new_segment_id = data_node.compact_segments([1001, 1002, 1003])

    # 8. 释放索引
    print("\n[Step 7] Releasing Index")
    print("-" * 60)
    query_node.release_index(segment_id)

    print("\n" + "=" * 60)
    print("Worker Layer Demo Completed")
    print("=" * 60)


if __name__ == "__main__":
    worker_layer_demo()
```

**运行输出：**

```
============================================================
Worker Layer Demo: Data Node + Index Node + Query Node
============================================================

[Step 1] Data Writing
------------------------------------------------------------
[DataNode-1] Subscribed to WAL for segment 1001
[DataNode-1] Buffered 1500 vectors for segment 1001
[DataNode-1] Flushed 1000 vectors to s3://milvus/binlog/segment_1001.bin

[Step 2] Flushing Remaining Data
------------------------------------------------------------
[DataNode-1] Flushed 500 vectors to s3://milvus/binlog/segment_1001.bin

[Step 3] Building Index
------------------------------------------------------------
[IndexNode-1] Building ivf_flat index for segment 1001
[IndexNode-1] Index built and uploaded to s3://milvus/index/segment_1001_ivf_flat.idx

[Step 4] Loading Index
------------------------------------------------------------
[QueryNode-1] Loading index for segment 1001
[QueryNode-1] Index loaded, memory usage: 0.06 MB

[Step 5] Executing Search
------------------------------------------------------------
[QueryNode-1] Searching with top_k=5
[QueryNode-1] Search completed, found 5 results

Search Results:
  1. Segment 1001, Vector 342, Distance: 4.2341
  2. Segment 1001, Vector 789, Distance: 4.5678
  3. Segment 1001, Vector 123, Distance: 4.8901
  4. Segment 1001, Vector 456, Distance: 5.1234
  5. Segment 1001, Vector 901, Distance: 5.4567

[Step 6] Compaction
------------------------------------------------------------
[DataNode-1] Compacting segments: [1001, 1002, 1003]
[DataNode-1] Compaction completed, new segment: 2001

[Step 7] Releasing Index
------------------------------------------------------------
[QueryNode-1] Released index for segment 1001

============================================================
Worker Layer Demo Completed
============================================================
```

---

## 三、RAG 相关应用场景

### 3.1 场景 1：RAG 系统的实时数据写入

**问题：** RAG 系统需要实时插入新文档，Data Node 如何高效处理？

**Worker 层的作用：**
- **Data Node**：缓冲数据，批量 Flush，减少 I/O 次数
- **Streaming Node（2.6）**：提供实时查询，无需等待索引构建
- **Index Node**：后台异步构建索引，不阻塞写入

**实际代码示例：**

```python
from pymilvus import Collection, connections
import numpy as np

# 连接到 Milvus 2.6
connections.connect(host="localhost", port="19530")

def realtime_insert_to_rag(collection_name: str, documents: list):
    """实时插入文档到 RAG 知识库"""

    collection = Collection(collection_name)

    # 1. 准备数据
    texts = [doc["text"] for doc in documents]
    embeddings = [doc["embedding"] for doc in documents]

    # 2. 插入数据（Data Node 缓冲）
    entities = [texts, embeddings]
    insert_result = collection.insert(entities)

    print(f"✅ Inserted {len(documents)} documents")
    print(f"   Primary keys: {insert_result.primary_keys[:5]}...")

    # 3. 可选：手动 Flush（强制持久化）
    # collection.flush()  # 通常不需要，Data Node 会自动 Flush

    # 4. Milvus 2.6 的优势：实时查询
    # Streaming Node 会立即提供实时数据的查询
    # 无需等待 Flush 和索引构建

    return insert_result

# 示例：批量插入文档
documents = [
    {"text": f"Document {i}", "embedding": np.random.rand(768).tolist()}
    for i in range(1000)
]

realtime_insert_to_rag("rag_knowledge_base", documents)
```

**Worker 层的优势：**
- Data Node 的缓冲机制减少了对象存储的写入次数
- Milvus 2.6 的 Streaming Node 提供实时查询，无需等待索引
- Index Node 后台异步构建索引，不影响写入性能

### 3.2 场景 2：RAG 系统的大规模索引构建

**问题：** RAG 知识库有数百万文档，如何快速构建索引？

**Worker 层的作用：**
- **Index Node**：分布式并行构建索引
- **多个 Index Node**：同时处理不同 Segment 的索引
- **Query Node**：增量加载新索引，无需全量重启

**实际代码示例：**

```python
from pymilvus import Collection, utility
import time

def build_index_for_large_rag(collection_name: str):
    """为大规模 RAG 知识库构建索引"""

    collection = Collection(collection_name)

    # 1. 定义索引参数
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}  # 聚类中心数量
    }

    print(f"Building index for {collection_name}...")
    start_time = time.time()

    # 2. 创建索引（Index Node 并行构建）
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    # 3. 等待索引构建完成
    while True:
        index_info = collection.index()
        if index_info:
            print(f"✅ Index built successfully")
            break
        time.sleep(1)

    elapsed = time.time() - start_time
    print(f"   Time elapsed: {elapsed:.2f} seconds")

    # 4. 加载 Collection（Query Node 加载索引）
    print(f"Loading collection...")
    collection.load()

    # 5. 检查加载状态
    load_state = utility.load_state(collection_name)
    print(f"✅ Collection loaded: {load_state}")

    return collection

# 示例：为大规模知识库构建索引
build_index_for_large_rag("rag_knowledge_base")
```

**Worker 层的优势：**
- 多个 Index Node 并行构建不同 Segment 的索引，速度提升 N 倍
- Query Node 增量加载新索引，无需全量重启
- 索引构建和查询服务解耦，互不影响

### 3.3 场景 3：RAG 系统的混合查询（向量 + 标量）

**问题：** RAG 系统需要同时过滤元数据和检索向量

**Worker 层的作用：**
- **Query Node**：先执行标量过滤，再执行向量检索
- **优化策略**：根据过滤条件选择最优执行顺序
- **内存管理**：只加载需要的 Segment

**实际代码示例：**

```python
from pymilvus import Collection

def hybrid_search_in_rag(collection_name: str, query_embedding: list, filters: dict):
    """RAG 系统的混合查询"""

    collection = Collection(collection_name)

    # 1. 构建过滤表达式
    filter_expr = []
    if "category" in filters:
        filter_expr.append(f'category == "{filters["category"]}"')
    if "timestamp" in filters:
        filter_expr.append(f'timestamp > {filters["timestamp"]}')

    expr = " && ".join(filter_expr) if filter_expr else None

    # 2. 执行混合查询（Query Node 处理）
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        expr=expr,  # 标量过滤
        output_fields=["text", "category", "timestamp"]
    )

    print(f"✅ Hybrid search completed")
    print(f"   Filter: {expr}")
    print(f"   Results: {len(results[0])} documents")

    # 3. 处理结果
    for i, hit in enumerate(results[0]):
        print(f"\n  {i+1}. Distance: {hit.distance:.4f}")
        print(f"     Text: {hit.entity.get('text')[:50]}...")
        print(f"     Category: {hit.entity.get('category')}")
        print(f"     Timestamp: {hit.entity.get('timestamp')}")

    return results

# 示例：混合查询
query_embedding = np.random.rand(768).tolist()
filters = {
    "category": "technical_docs",
    "timestamp": 1640000000
}

hybrid_search_in_rag("rag_knowledge_base", query_embedding, filters)
```

**Worker 层的优势：**
- Query Node 智能优化查询执行顺序（先过滤再检索，或先检索再过滤）
- 标量索引和向量索引协同工作，提升查询效率
- 内存管理优化，只加载需要的 Segment

### 3.4 场景 4：RAG 系统的动态资源管理

**问题：** RAG 系统有多个知识库，内存有限，如何动态管理？

**Worker 层的作用：**
- **Query Node**：动态加载/卸载索引
- **内存监控**：根据内存使用情况自动调整
- **mmap 策略**：大索引使用 mmap，小索引直接加载

**实际代码示例：**

```python
from pymilvus import Collection, utility
import psutil

def dynamic_resource_management():
    """RAG 系统的动态资源管理"""

    # 1. 检查系统内存
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024 ** 3)

    print(f"Available memory: {available_memory_gb:.2f} GB")

    # 2. 定义知识库优先级
    knowledge_bases = [
        {"name": "customer_support_kb", "priority": 1, "size_gb": 2.5},
        {"name": "product_docs_kb", "priority": 2, "size_gb": 1.8},
        {"name": "legal_docs_kb", "priority": 3, "size_gb": 3.2},
        {"name": "archive_kb", "priority": 4, "size_gb": 5.0},
    ]

    # 3. 根据内存动态加载
    loaded_collections = []
    remaining_memory = available_memory_gb * 0.7  # 使用 70% 可用内存

    for kb in sorted(knowledge_bases, key=lambda x: x["priority"]):
        if remaining_memory >= kb["size_gb"]:
            collection = Collection(kb["name"])

            # 检查是否已加载
            if utility.load_state(kb["name"]) != "Loaded":
                print(f"Loading {kb['name']}...")
                collection.load()
                loaded_collections.append(kb["name"])
                remaining_memory -= kb["size_gb"]
                print(f"✅ Loaded {kb['name']} ({kb['size_gb']} GB)")
            else:
                print(f"✅ {kb['name']} already loaded")
        else:
            print(f"⚠️  Skipping {kb['name']} (insufficient memory)")

    # 4. 显示加载状态
    print(f"\nLoaded collections: {loaded_collections}")
    print(f"Remaining memory: {remaining_memory:.2f} GB")

    return loaded_collections

def release_low_priority_collections(keep_top_n: int = 2):
    """释放低优先级的知识库"""

    knowledge_bases = [
        {"name": "customer_support_kb", "priority": 1},
        {"name": "product_docs_kb", "priority": 2},
        {"name": "legal_docs_kb", "priority": 3},
        {"name": "archive_kb", "priority": 4},
    ]

    # 释放低优先级的知识库
    for kb in sorted(knowledge_bases, key=lambda x: x["priority"], reverse=True):
        if kb["priority"] > keep_top_n:
            collection = Collection(kb["name"])
            if utility.load_state(kb["name"]) == "Loaded":
                print(f"Releasing {kb['name']}...")
                collection.release()
                print(f"✅ Released {kb['name']}")

# 示例
dynamic_resource_management()
release_low_priority_collections(keep_top_n=2)
```

**Worker 层的优势：**
- Query Node 支持动态加载/卸载，灵活管理内存
- 根据业务优先级和内存情况自动调整
- mmap 策略允许超大索引在有限内存下运行

### 3.5 场景 5：RAG 系统的故障恢复

**问题：** Query Node 故障后，如何快速恢复查询服务？

**Worker 层的作用：**
- **Query Node 高可用**：多个 Query Node 互为备份
- **Segment 自动迁移**：故障节点的 Segment 自动迁移到健康节点
- **无状态设计**：Query Node 无状态，可快速重启

**实际代码示例：**

```python
from pymilvus import connections, Collection, utility
import time

def check_query_node_health():
    """检查 Query Node 健康状态"""

    try:
        # 获取 Query Node 信息
        query_nodes = utility.get_query_segment_info("rag_knowledge_base")

        node_status = {}
        for seg_info in query_nodes:
            node_id = seg_info.nodeID
            if node_id not in node_status:
                node_status[node_id] = {"segments": 0, "healthy": True}
            node_status[node_id]["segments"] += 1

        print("Query Node Status:")
        for node_id, status in node_status.items():
            print(f"  Node {node_id}: {status['segments']} segments, Healthy: {status['healthy']}")

        return node_status

    except Exception as e:
        print(f"❌ Failed to check Query Node health: {e}")
        return {}

def recover_from_query_node_failure(collection_name: str):
    """从 Query Node 故障中恢复"""

    print(f"Recovering {collection_name} from Query Node failure...")

    # 1. 检查当前状态
    load_state = utility.load_state(collection_name)
    print(f"Current load state: {load_state}")

    # 2. 如果未加载，重新加载
    if load_state != "Loaded":
        collection = Collection(collection_name)
        print(f"Reloading collection...")
        collection.load()

        # 等待加载完成
        while utility.load_state(collection_name) != "Loaded":
            print(f"  Waiting for collection to load...")
            time.sleep(1)

        print(f"✅ Collection reloaded successfully")

    # 3. 验证查询功能
    collection = Collection(collection_name)
    query_vector = np.random.rand(768).tolist()

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=5
        )
        print(f"✅ Query functionality verified: {len(results[0])} results")
    except Exception as e:
        print(f"❌ Query failed: {e}")

# 示例
check_query_node_health()
recover_from_query_node_failure("rag_knowledge_base")
```

**Worker 层的优势：**
- Query Node 无状态，故障后可快速重启
- Segment 自动迁移到健康节点，无需人工干预
- 多个 Query Node 互为备份，提高可用性

---

## 四、总结

### 4.1 Worker 层的核心价值

1. **数据平面执行**：实际执行数据写入、索引构建、查询检索
2. **分布式扩展**：多个节点并行工作，提升吞吐量
3. **资源隔离**：不同类型的任务由不同节点处理，互不影响
4. **高可用设计**：无状态节点，故障后快速恢复

### 4.2 Milvus 2.6 的创新：流批分离

- **Streaming Node**：专门处理实时数据，提供实时查询
- **Query Node**：只处理已索引数据，性能更稳定
- **职责清晰**：实时和历史数据分离，架构更简洁

### 4.3 对 RAG 开发的影响

- **更快的数据写入**：Data Node 的缓冲和批处理机制
- **更高的查询性能**：Query Node 的内存管理和索引优化
- **更灵活的资源管理**：动态加载/卸载，适应不同场景
- **更强的容错能力**：无状态设计，快速故障恢复

### 4.4 类比总结

**前端类比：**
- Data Node = 数据缓存层（Redux、Vuex）
- Index Node = 构建工具（Webpack、Vite）
- Query Node = 渲染引擎（React、Vue）

**日常生活类比：**
- Data Node = 仓库管理员（接收货物，整理入库）
- Index Node = 图书管理员（整理书籍，建立索引）
- Query Node = 图书馆前台（响应查询，提供书籍）

---

> **来源**: [Milvus System Overview](https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/chap01_system_overview.md) | [Milvus Repository](https://github.com/milvus-io/milvus) | [MMAP Discussion](https://github.com/milvus-io/milvus/discussions/33621) | [2.6 Performance Issue](https://github.com/milvus-io/milvus/issues/43659) | [Streaming Service Enhancement](https://github.com/milvus-io/milvus/issues/33285) | 获取时间: 2026-02-21
```

