# 核心概念 6：Storage 层（存储层）

> **来源**: [CollectionCreate Timeout Bug](https://github.com/milvus-io/milvus/issues/45007) | [AWS Multipart Upload Error](https://github.com/milvus-io/milvus/issues/44853) | [Streamingnode MinIO Recovery](https://github.com/milvus-io/milvus/issues/43597) | [External S3 Configuration](https://github.com/milvus-io/milvus/discussions/46881) | 获取时间: 2026-02-21

---

## 一、原理讲解

### 1.1 什么是 Storage 层？

Storage 层是 Milvus 分布式架构中的**持久化层**，负责所有数据的长期存储。它是整个系统的"硬盘"，确保数据的可靠性和持久性。

**核心职责：**
- **对象存储**：存储向量数据、索引文件、日志文件
- **元数据存储**：存储集群状态、Collection 信息、Segment 元数据
- **数据可靠性**：保证数据不丢失，支持备份和恢复
- **存储与计算分离**：Worker 层无状态，数据全部在 Storage 层

### 1.2 Storage 层的两大组件

```
Storage 层架构：
┌─────────────────────────────────────────────────────────┐
│                  Coordinator 层 + Worker 层              │
│              (无状态，数据全部存储在 Storage 层)          │
└─────────────────────────────────────────────────────────┘
                          ↓ 读写数据
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────────┐          ┌──────────────────────┐
│   Object Storage     │          │   Meta Storage       │
│   (对象存储)         │          │   (元数据存储)       │
├──────────────────────┤          ├──────────────────────┤
│ - MinIO / S3 / GCS   │          │ - etcd               │
│ - 向量数据 (Binlog)  │          │ - Collection Schema  │
│ - 索引文件 (Index)   │          │ - Segment 元数据     │
│ - WAL 日志 (2.6)     │          │ - 集群状态           │
│ - 支持多副本         │          │ - 分布式一致性       │
└──────────────────────┘          └──────────────────────┘
```

### 1.3 Object Storage（对象存储）

#### 1.3.1 为什么选择对象存储？

**传统方案的问题：**
- **本地磁盘**：扩展性差，故障风险高，运维复杂
- **分布式文件系统（HDFS）**：部署复杂，成本高

**对象存储的优势：**
- ✅ **无限扩展**：存储容量几乎无限
- ✅ **高可靠性**：多副本自动备份，数据不丢失
- ✅ **低成本**：按需付费，冷数据成本极低
- ✅ **云原生**：与云平台深度集成
- ✅ **存储与计算分离**：Worker 节点无状态，可随意扩缩容

#### 1.3.2 支持的对象存储类型

| 存储类型 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| **MinIO** | 本地部署、开发测试 | 开源免费、兼容 S3 API | 需要自己运维 |
| **AWS S3** | AWS 云环境 | 高可靠、全球分布 | 成本较高 |
| **Google Cloud Storage** | GCP 云环境 | 高性能、低延迟 | 仅限 GCP |
| **Azure Blob Storage** | Azure 云环境 | 与 Azure 集成好 | 仅限 Azure |
| **阿里云 OSS** | 阿里云环境 | 国内访问快 | 仅限阿里云 |

#### 1.3.3 对象存储中的数据类型

```
对象存储目录结构：
s3://milvus-bucket/
├── binlog/                    # 向量数据（Binlog）
│   ├── insert_log/
│   │   ├── collection_1/
│   │   │   ├── partition_0/
│   │   │   │   ├── segment_1001/
│   │   │   │   │   ├── field_0.binlog
│   │   │   │   │   ├── field_1.binlog
│   │   │   │   │   └── ...
│   ├── delete_log/            # 删除日志
│   └── stats_log/             # 统计信息
├── index/                     # 索引文件
│   ├── collection_1/
│   │   ├── partition_0/
│   │   │   ├── segment_1001/
│   │   │   │   ├── field_1.ivf_flat
│   │   │   │   └── ...
└── wal/                       # WAL 日志（Milvus 2.6）
    ├── collection_1/
    │   ├── vchannel_0.wal
    │   └── vchannel_1.wal
```

**关键文件类型：**

1. **Insert Log（插入日志）**
   - 存储实际的向量数据和标量字段
   - 按字段分别存储（列式存储）
   - 使用 Parquet 或自定义二进制格式

2. **Delete Log（删除日志）**
   - 存储已删除的主键列表
   - 逻辑删除，不立即清理数据
   - Compaction 时物理删除

3. **Stats Log（统计日志）**
   - 存储 Segment 的统计信息
   - 用于查询优化和负载均衡

4. **Index Files（索引文件）**
   - 存储构建好的向量索引
   - 不同索引类型有不同的文件格式
   - Query Node 下载后加载到内存

5. **WAL Files（Milvus 2.6 新增）**
   - 存储 Write-Ahead Log
   - 替代 Kafka/Pulsar，实现零磁盘架构
   - 直接写入对象存储

#### 1.3.4 Milvus 2.6 的创新：Multipart Upload

**问题：** 大文件上传到对象存储速度慢，容易失败

**Milvus 2.6 的解决方案：Multipart Upload（分片上传）**

```
传统上传（2.5 及之前）：
┌──────────┐
│ 1GB 文件 │ ──────────────────────────> S3
└──────────┘
  单次上传，失败需重传整个文件

Multipart Upload（2.6）：
┌──────────┐
│ 1GB 文件 │
└──────────┘
     ↓ 分片
┌────┬────┬────┬────┐
│100M│100M│100M│100M│ ──并行上传──> S3
└────┴────┴────┴────┘
  失败只需重传单个分片
```

**优势：**
- ✅ **并行上传**：多个分片同时上传，速度提升 5-10 倍
- ✅ **断点续传**：失败后只需重传失败的分片
- ✅ **大文件支持**：支持 TB 级别的单文件上传

**配置示例：**

```yaml
# milvus.yaml
minio:
  address: localhost:9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  useSSL: false
  bucketName: milvus-bucket

  # Multipart Upload 配置（2.6 新增）
  multipartUpload:
    enabled: true
    partSize: 104857600  # 100MB per part
    maxParts: 10000      # 最多 10000 个分片
```

### 1.4 Meta Storage（元数据存储）

#### 1.4.1 为什么需要独立的元数据存储？

**元数据的特点：**
- **小而频繁**：元数据量小，但读写频繁
- **强一致性**：需要分布式一致性保证
- **低延迟**：元数据访问延迟直接影响系统性能

**对象存储不适合元数据：**
- 对象存储针对大文件优化，小文件性能差
- 不支持事务和强一致性
- 延迟较高（通常 10-100ms）

**etcd 的优势：**
- ✅ **强一致性**：基于 Raft 协议，保证分布式一致性
- ✅ **低延迟**：内存存储，延迟通常 < 10ms
- ✅ **Watch 机制**：支持实时监听元数据变化
- ✅ **事务支持**：支持原子操作和事务

#### 1.4.2 etcd 中存储的元数据

```
etcd 键值结构：
/milvus/
├── root-coord/
│   ├── collections/
│   │   ├── collection_1
│   │   │   ├── schema          # Collection Schema
│   │   │   ├── partitions      # Partition 列表
│   │   │   └── state           # Collection 状态
│   ├── id-allocator/           # 全局 ID 分配器
│   └── timestamp/              # 时间戳服务
├── data-coord/
│   ├── segments/
│   │   ├── segment_1001
│   │   │   ├── state           # Segment 状态
│   │   │   ├── binlog_paths    # Binlog 文件路径
│   │   │   ├── num_rows        # 行数
│   │   │   └── size            # 大小
│   ├── channels/               # VChannel 分配
│   └── compaction/             # Compaction 任务
└── query-coord/
    ├── replicas/               # 副本分配
    ├── segments/               # Segment 加载状态
    └── nodes/                  # QueryNode 状态
```

**关键元数据类型：**

1. **Collection Schema**
   - 字段定义（名称、类型、维度）
   - 索引配置
   - 分片数量

2. **Segment 元数据**
   - Segment 状态（Growing/Sealed/Flushed/Indexed）
   - Binlog 文件路径
   - 行数、大小、时间戳

3. **集群状态**
   - 节点健康状态
   - 资源使用情况
   - 任务分配

4. **ID 分配**
   - Collection ID
   - Segment ID
   - 时间戳

#### 1.4.3 etcd 的高可用配置

**单节点 etcd（开发测试）：**

```yaml
# milvus.yaml
etcd:
  endpoints:
    - localhost:2379
  rootPath: by-dev  # etcd 根路径
```

**etcd 集群（生产环境）：**

```yaml
# milvus.yaml
etcd:
  endpoints:
    - etcd-1:2379
    - etcd-2:2379
    - etcd-3:2379
  rootPath: by-prod

  # 高可用配置
  dialTimeout: 5s
  keepAliveTime: 10s
  keepAliveTimeout: 20s
```

**etcd 集群部署建议：**
- 至少 3 个节点（支持 1 个节点故障）
- 5 个节点（支持 2 个节点故障）
- 奇数个节点（避免脑裂）

### 1.5 存储与计算分离的优势

```
传统架构（存储与计算耦合）：
┌──────────────────────────────┐
│      Milvus Node             │
│  ┌────────────┬────────────┐ │
│  │   计算     │   存储     │ │
│  │  (查询)    │  (磁盘)    │ │
│  └────────────┴────────────┘ │
└──────────────────────────────┘
  扩展困难：计算和存储必须同时扩展
  故障风险：磁盘故障导致数据丢失

Milvus 架构（存储与计算分离）：
┌──────────────────────────────┐
│      Worker 层 (无状态)       │
│  ┌────────────┐              │
│  │   计算     │              │
│  │  (查询)    │              │
│  └────────────┘              │
└──────────────────────────────┘
         ↓ 读写数据
┌──────────────────────────────┐
│      Storage 层 (有状态)      │
│  ┌────────────┬────────────┐ │
│  │ Object     │   etcd     │ │
│  │ Storage    │            │ │
│  └────────────┴────────────┘ │
└──────────────────────────────┘
  灵活扩展：计算和存储独立扩展
  高可靠：数据多副本，节点故障不丢数据
```

**优势总结：**

1. **弹性扩展**
   - Worker 节点可随意增减
   - 存储容量独立扩展
   - 按需付费，成本优化

2. **高可用**
   - Worker 节点故障，数据不丢失
   - 新节点启动后自动从 Storage 层恢复
   - 无需复杂的数据迁移

3. **简化运维**
   - Worker 节点无状态，部署简单
   - 升级时可以滚动更新
   - 备份和恢复只需操作 Storage 层

4. **成本优化**
   - 冷数据存储在低成本对象存储
   - 热数据缓存在 Worker 节点内存
   - 存储和计算资源独立优化

---

## 二、手写实现：简化版 Storage 层

下面我们用 Python 实现一个简化版的 Storage 层，展示对象存储和元数据存储如何协作。

### 2.1 基础数据结构

```python
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum
import json
import os
import pickle

class SegmentState(Enum):
    """Segment 状态"""
    GROWING = "growing"
    SEALED = "sealed"
    FLUSHED = "flushed"
    INDEXED = "indexed"

@dataclass
class SegmentMeta:
    """Segment 元数据"""
    segment_id: int
    collection_id: int
    state: SegmentState
    num_rows: int
    size_bytes: int
    binlog_paths: List[str]
    index_path: Optional[str] = None

@dataclass
class CollectionSchema:
    """Collection Schema"""
    collection_id: int
    collection_name: str
    fields: List[Dict]
    shards_num: int
```

### 2.2 Object Storage 实现

```python
class ObjectStorage:
    """对象存储：模拟 MinIO/S3"""

    def __init__(self, base_path: str = "/tmp/milvus_storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        print(f"[ObjectStorage] Initialized at {base_path}")

    def put_object(self, key: str, data: bytes) -> str:
        """上传对象"""
        file_path = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(data)

        size_mb = len(data) / (1024 * 1024)
        print(f"[ObjectStorage] PUT {key} ({size_mb:.2f} MB)")
        return f"s3://milvus/{key}"

    def get_object(self, key: str) -> Optional[bytes]:
        """下载对象"""
        file_path = os.path.join(self.base_path, key)

        if not os.path.exists(file_path):
            print(f"[ObjectStorage] GET {key} - NOT FOUND")
            return None

        with open(file_path, 'rb') as f:
            data = f.read()

        size_mb = len(data) / (1024 * 1024)
        print(f"[ObjectStorage] GET {key} ({size_mb:.2f} MB)")
        return data

    def delete_object(self, key: str):
        """删除对象"""
        file_path = os.path.join(self.base_path, key)

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[ObjectStorage] DELETE {key}")
        else:
            print(f"[ObjectStorage] DELETE {key} - NOT FOUND")

    def list_objects(self, prefix: str) -> List[str]:
        """列出对象"""
        prefix_path = os.path.join(self.base_path, prefix)
        objects = []

        if os.path.exists(prefix_path):
            for root, dirs, files in os.walk(prefix_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_path)
                    objects.append(rel_path)

        print(f"[ObjectStorage] LIST {prefix} - {len(objects)} objects")
        return objects

    def multipart_upload(self, key: str, data: bytes, part_size: int = 5 * 1024 * 1024) -> str:
        """分片上传（Milvus 2.6 新特性）"""
        total_size = len(data)
        num_parts = (total_size + part_size - 1) // part_size

        print(f"[ObjectStorage] Multipart upload: {key}")
        print(f"  Total size: {total_size / (1024 * 1024):.2f} MB")
        print(f"  Part size: {part_size / (1024 * 1024):.2f} MB")
        print(f"  Number of parts: {num_parts}")

        # 模拟分片上传
        for i in range(num_parts):
            start = i * part_size
            end = min(start + part_size, total_size)
            part_data = data[start:end]
            print(f"  Uploading part {i+1}/{num_parts} ({len(part_data) / (1024 * 1024):.2f} MB)")

        # 最终合并（实际实现中由对象存储完成）
        return self.put_object(key, data)
```

### 2.3 Meta Storage 实现

```python
class MetaStorage:
    """元数据存储：模拟 etcd"""

    def __init__(self, base_path: str = "/tmp/milvus_meta"):
        self.base_path = base_path
        self.data: Dict[str, str] = {}
        self.watchers: Dict[str, List] = {}
        os.makedirs(base_path, exist_ok=True)
        self._load_from_disk()
        print(f"[MetaStorage] Initialized at {base_path}")

    def put(self, key: str, value: str):
        """存储键值对"""
        self.data[key] = value
        self._save_to_disk()
        print(f"[MetaStorage] PUT {key}")

        # 触发 Watch 回调
        self._trigger_watchers(key, value)

    def get(self, key: str) -> Optional[str]:
        """获取值"""
        value = self.data.get(key)
        if value:
            print(f"[MetaStorage] GET {key} - FOUND")
        else:
            print(f"[MetaStorage] GET {key} - NOT FOUND")
        return value

    def delete(self, key: str):
        """删除键"""
        if key in self.data:
            del self.data[key]
            self._save_to_disk()
            print(f"[MetaStorage] DELETE {key}")
        else:
            print(f"[MetaStorage] DELETE {key} - NOT FOUND")

    def list_keys(self, prefix: str) -> List[str]:
        """列出键"""
        keys = [k for k in self.data.keys() if k.startswith(prefix)]
        print(f"[MetaStorage] LIST {prefix} - {len(keys)} keys")
        return keys

    def watch(self, key: str, callback):
        """监听键变化"""
        if key not in self.watchers:
            self.watchers[key] = []
        self.watchers[key].append(callback)
        print(f"[MetaStorage] WATCH {key}")

    def _trigger_watchers(self, key: str, value: str):
        """触发 Watch 回调"""
        if key in self.watchers:
            for callback in self.watchers[key]:
                callback(key, value)

    def _save_to_disk(self):
        """持久化到磁盘"""
        file_path = os.path.join(self.base_path, "meta.json")
        with open(file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def _load_from_disk(self):
        """从磁盘加载"""
        file_path = os.path.join(self.base_path, "meta.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            print(f"[MetaStorage] Loaded {len(self.data)} keys from disk")

### 2.4 Storage Manager（统一管理）

```python
class StorageManager:
    """存储管理器：统一管理对象存储和元数据存储"""

    def __init__(self):
        self.object_storage = ObjectStorage()
        self.meta_storage = MetaStorage()
        print("=" * 60)
        print("StorageManager initialized")
        print("=" * 60)

    def save_segment(self, segment_meta: SegmentMeta, vectors: List) -> str:
        """保存 Segment 数据"""
        print(f"\n[StorageManager] Saving segment {segment_meta.segment_id}")

        # 1. 保存向量数据到对象存储
        binlog_key = f"binlog/collection_{segment_meta.collection_id}/segment_{segment_meta.segment_id}.bin"
        binlog_data = pickle.dumps(vectors)

        # 使用 Multipart Upload（如果数据较大）
        if len(binlog_data) > 10 * 1024 * 1024:  # > 10MB
            binlog_path = self.object_storage.multipart_upload(binlog_key, binlog_data)
        else:
            binlog_path = self.object_storage.put_object(binlog_key, binlog_data)

        segment_meta.binlog_paths = [binlog_path]

        # 2. 保存元数据到 etcd
        meta_key = f"/milvus/segments/{segment_meta.segment_id}"
        meta_value = json.dumps(asdict(segment_meta), default=str)
        self.meta_storage.put(meta_key, meta_value)

        print(f"[StorageManager] Segment {segment_meta.segment_id} saved successfully")
        return binlog_path

    def load_segment(self, segment_id: int) -> tuple:
        """加载 Segment 数据"""
        print(f"\n[StorageManager] Loading segment {segment_id}")

        # 1. 从 etcd 读取元数据
        meta_key = f"/milvus/segments/{segment_id}"
        meta_value = self.meta_storage.get(meta_key)

        if not meta_value:
            print(f"[StorageManager] Segment {segment_id} not found")
            return None, None

        segment_meta = SegmentMeta(**json.loads(meta_value))

        # 2. 从对象存储读取向量数据
        binlog_key = segment_meta.binlog_paths[0].replace("s3://milvus/", "")
        binlog_data = self.object_storage.get_object(binlog_key)

        if not binlog_data:
            print(f"[StorageManager] Binlog not found for segment {segment_id}")
            return segment_meta, None

        vectors = pickle.loads(binlog_data)

        print(f"[StorageManager] Segment {segment_id} loaded successfully")
        return segment_meta, vectors

    def delete_segment(self, segment_id: int):
        """删除 Segment"""
        print(f"\n[StorageManager] Deleting segment {segment_id}")

        # 1. 从 etcd 读取元数据
        meta_key = f"/milvus/segments/{segment_id}"
        meta_value = self.meta_storage.get(meta_key)

        if meta_value:
            segment_meta = SegmentMeta(**json.loads(meta_value))

            # 2. 删除对象存储中的数据
            for binlog_path in segment_meta.binlog_paths:
                binlog_key = binlog_path.replace("s3://milvus/", "")
                self.object_storage.delete_object(binlog_key)

            if segment_meta.index_path:
                index_key = segment_meta.index_path.replace("s3://milvus/", "")
                self.object_storage.delete_object(index_key)

        # 3. 删除 etcd 中的元数据
        self.meta_storage.delete(meta_key)

        print(f"[StorageManager] Segment {segment_id} deleted successfully")

### 2.5 完整示例

```python
import numpy as np

def storage_layer_demo():
    """演示 Storage 层的完整工作流程"""

    print("=" * 60)
    print("Storage Layer Demo")
    print("=" * 60)

    # 1. 初始化 Storage Manager
    storage_mgr = StorageManager()

    # 2. 创建测试数据
    segment_id = 1001
    collection_id = 1
    vectors = [np.random.rand(128) for _ in range(1000)]

    segment_meta = SegmentMeta(
        segment_id=segment_id,
        collection_id=collection_id,
        state=SegmentState.FLUSHED,
        num_rows=len(vectors),
        size_bytes=len(vectors) * 128 * 4,  # float32
        binlog_paths=[]
    )

    # 3. 保存 Segment
    print("\n[Step 1] Saving Segment")
    print("-" * 60)
    storage_mgr.save_segment(segment_meta, vectors)

    # 4. 加载 Segment
    print("\n[Step 2] Loading Segment")
    print("-" * 60)
    loaded_meta, loaded_vectors = storage_mgr.load_segment(segment_id)

    if loaded_meta and loaded_vectors:
        print(f"\nLoaded Segment Info:")
        print(f"  Segment ID: {loaded_meta.segment_id}")
        print(f"  State: {loaded_meta.state.value}")
        print(f"  Num rows: {loaded_meta.num_rows}")
        print(f"  Size: {loaded_meta.size_bytes / (1024 * 1024):.2f} MB")
        print(f"  Vectors loaded: {len(loaded_vectors)}")

    # 5. 列出所有 Segment
    print("\n[Step 3] Listing All Segments")
    print("-" * 60)
    segment_keys = storage_mgr.meta_storage.list_keys("/milvus/segments/")
    print(f"Total segments: {len(segment_keys)}")

    # 6. 删除 Segment
    print("\n[Step 4] Deleting Segment")
    print("-" * 60)
    storage_mgr.delete_segment(segment_id)

    print("\n" + "=" * 60)
    print("Storage Layer Demo Completed")
    print("=" * 60)


if __name__ == "__main__":
    storage_layer_demo()
```

---

## 三、RAG 相关应用场景

### 3.1 场景 1：RAG 知识库的持久化存储

**问题：** RAG 系统需要持久化存储大量文档向量

**Storage 层的作用：**
- **Object Storage**：存储向量数据和索引文件
- **Meta Storage**：存储 Collection Schema 和 Segment 元数据
- **存储与计算分离**：Worker 节点故障不影响数据

**实际代码示例：**

```python
from pymilvus import connections, Collection, utility

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

def check_storage_usage(collection_name: str):
    """检查 RAG 知识库的存储使用情况"""

    collection = Collection(collection_name)

    # 获取 Collection 统计信息
    stats = collection.num_entities
    print(f"Collection: {collection_name}")
    print(f"  Total entities: {stats}")

    # 获取 Segment 信息
    segments = utility.get_query_segment_info(collection_name)

    total_size = 0
    for seg in segments:
        total_size += seg.mem_size

    print(f"  Total memory size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    print(f"  Number of segments: {len(segments)}")

    # 估算对象存储使用量
    # 假设索引大小约为原始数据的 1.5 倍
    estimated_storage = total_size * 1.5
    print(f"  Estimated object storage: {estimated_storage / (1024 * 1024 * 1024):.2f} GB")

# 示例
check_storage_usage("rag_knowledge_base")
```

### 3.2 场景 2：RAG 系统的备份与恢复

**问题：** RAG 系统需要定期备份，故障后快速恢复

**Storage 层的作用：**
- **对象存储备份**：使用对象存储的快照或复制功能
- **元数据备份**：导出 etcd 数据
- **快速恢复**：从备份恢复数据

**实际代码示例：**

```python
import subprocess
import datetime

def backup_rag_system():
    """备份 RAG 系统"""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 备份对象存储（使用 MinIO 客户端）
    print(f"Backing up object storage...")
    subprocess.run([
        "mc", "mirror",
        "myminio/milvus-bucket",
        f"myminio/milvus-backup-{timestamp}"
    ])

    # 2. 备份 etcd 元数据
    print(f"Backing up etcd metadata...")
    subprocess.run([
        "etcdctl", "snapshot", "save",
        f"/backup/etcd-snapshot-{timestamp}.db"
    ])

    print(f"✅ Backup completed: {timestamp}")

def restore_rag_system(backup_timestamp: str):
    """恢复 RAG 系统"""

    # 1. 恢复对象存储
    print(f"Restoring object storage...")
    subprocess.run([
        "mc", "mirror",
        f"myminio/milvus-backup-{backup_timestamp}",
        "myminio/milvus-bucket"
    ])

    # 2. 恢复 etcd 元数据
    print(f"Restoring etcd metadata...")
    subprocess.run([
        "etcdctl", "snapshot", "restore",
        f"/backup/etcd-snapshot-{backup_timestamp}.db"
    ])

    print(f"✅ Restore completed from: {backup_timestamp}")

# 示例
backup_rag_system()
```

### 3.3 场景 3：RAG 系统的冷热数据分层

**问题：** RAG 系统有大量历史数据，如何优化存储成本？

**Storage 层的作用：**
- **热数据**：频繁访问的数据，使用高性能存储
- **冷数据**：历史数据，使用低成本存储
- **自动分层**：根据访问频率自动迁移

**实际代码示例：**

```python
from pymilvus import Collection
import time

def archive_old_data(collection_name: str, days_old: int = 90):
    """归档旧数据到冷存储"""

    collection = Collection(collection_name)

    # 1. 查询旧数据
    cutoff_timestamp = int(time.time()) - (days_old * 24 * 3600)

    expr = f"timestamp < {cutoff_timestamp}"
    old_data = collection.query(expr=expr, output_fields=["id", "timestamp"])

    print(f"Found {len(old_data)} old records (> {days_old} days)")

    # 2. 导出到冷存储（例如 S3 Glacier）
    # 实际实现中，可以使用 Milvus 的 Bulk Insert/Export 功能
    # 这里简化为示例

    print(f"Exporting to cold storage...")
    # export_to_cold_storage(old_data)

    # 3. 从热存储删除
    # collection.delete(expr=expr)

    print(f"✅ Archived {len(old_data)} records to cold storage")

# 示例
archive_old_data("rag_knowledge_base", days_old=90)
```

### 3.4 场景 4：RAG 系统的多租户隔离

**问题：** 多个 RAG 应用共享 Milvus，如何隔离存储？

**Storage 层的作用：**
- **对象存储隔离**：不同租户使用不同的 Bucket 或前缀
- **元数据隔离**：etcd 使用不同的 rootPath
- **资源配额**：限制每个租户的存储使用量

**实际代码示例：**

```python
# 配置多租户存储隔离
# milvus_tenant_a.yaml
minio:
  bucketName: milvus-tenant-a
etcd:
  rootPath: tenant-a

# milvus_tenant_b.yaml
minio:
  bucketName: milvus-tenant-b
etcd:
  rootPath: tenant-b
```

---

## 四、总结

### 4.1 Storage 层的核心价值

1. **数据持久化**：保证数据不丢失，支持备份和恢复
2. **存储与计算分离**：Worker 节点无状态，灵活扩展
3. **高可靠性**：对象存储多副本，etcd 分布式一致性
4. **成本优化**：冷热数据分层，按需付费

### 4.2 Milvus 2.6 的创新：零磁盘架构

- **WAL 直接写入对象存储**：无需本地磁盘
- **Multipart Upload**：大文件并行上传，速度提升 5-10 倍
- **简化运维**：无需管理本地磁盘，降低运维成本

### 4.3 对 RAG 开发的影响

- **更高的可靠性**：数据多副本，故障不丢失
- **更低的成本**：冷热数据分层，优化存储成本
- **更简单的运维**：存储与计算分离，部署更简单
- **更好的扩展性**：存储容量几乎无限

### 4.4 类比总结

**前端类比：**
- Object Storage = CDN（存储静态资源）
- Meta Storage = Redis（存储元数据）

**日常生活类比：**
- Object Storage = 仓库（存储货物）
- Meta Storage = 档案室（存储文件索引）

---

> **来源**: [CollectionCreate Timeout Bug](https://github.com/milvus-io/milvus/issues/45007) | [AWS Multipart Upload Error](https://github.com/milvus-io/milvus/issues/44853) | [Streamingnode MinIO Recovery](https://github.com/milvus-io/milvus/issues/43597) | [External S3 Configuration](https://github.com/milvus-io/milvus/discussions/46881) | 获取时间: 2026-02-21
```

