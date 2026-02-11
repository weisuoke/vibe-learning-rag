# 核心概念 03：Segment 管理机制

## 什么是 Segment？

**Segment 是 Milvus 存储数据的基本单位，类似于数据库的"数据页"或文件系统的"文件块"，负责管理数据版本、生命周期和查询路由。**

---

## 一句话定义

**Segment 是 Milvus 中存储向量数据的最小逻辑单元，支持版本管理、并发控制和增量更新。**

---

## Segment 的核心原理

### 1. 为什么需要 Segment？

**问题：如何高效管理海量向量数据？**

```python
# 场景：存储 1 亿条向量数据（每条 128 维）
# 数据量：1 亿 × 128 × 4 字节 = 51.2 GB

# 方案1：单个大文件（不可行）
# 问题：
# 1. 无法并行查询（只能顺序扫描）
# 2. 无法增量更新（修改需要重写整个文件）
# 3. 无法版本管理（无法支持 MVCC）
# 4. 内存占用过大（无法全部加载到内存）

# 方案2：分成多个 Segment（Milvus 的方案）
# 优势：
# 1. 并行查询（多个 Segment 可以并行扫描）
# 2. 增量更新（新数据写入新 Segment）
# 3. 版本管理（每个 Segment 有独立版本）
# 4. 内存友好（按需加载 Segment）
```

**解决方案：Segment 机制**

```python
# Milvus 的 Segment 设计
# 1 亿条向量 → 100 个 Segment（每个 100 万条）

# 优势：
# - 并行查询：100 个 Segment 可以并行扫描
# - 增量更新：新数据写入新 Segment，不影响旧 Segment
# - 版本管理：每个 Segment 有独立的时间戳和版本号
# - 内存友好：只加载需要的 Segment 到内存
```

---

## Segment 的生命周期

### 完整生命周期图

```
1. Growing Segment（增长中）
    ↓
    数据不断写入
    ↓
2. Sealed Segment（已封闭）
    ↓
    达到大小阈值或时间阈值
    ↓
3. Flushing（刷新中）
    ↓
    数据从内存刷新到磁盘
    ↓
4. Flushed Segment（已刷新）
    ↓
    数据持久化到磁盘
    ↓
5. Indexed Segment（已索引）
    ↓
    构建向量索引
    ↓
6. Compacted Segment（已合并）
    ↓
    通过 Compaction 合并优化
    ↓
7. Dropped Segment（已删除）
    ↓
    不再使用，等待垃圾回收
```

### 详细状态转换

```python
# Segment 状态机
class SegmentState:
    GROWING = "Growing"        # 正在写入
    SEALED = "Sealed"          # 已封闭，不再写入
    FLUSHING = "Flushing"      # 正在刷新到磁盘
    FLUSHED = "Flushed"        # 已刷新到磁盘
    INDEXED = "Indexed"        # 已构建索引
    COMPACTED = "Compacted"    # 已合并
    DROPPED = "Dropped"        # 已删除

# 状态转换条件
class SegmentTransition:
    def growing_to_sealed(self, segment):
        """Growing → Sealed"""
        # 条件1：达到大小阈值（如 512MB）
        if segment.size >= 512 * 1024 * 1024:
            return True

        # 条件2：达到时间阈值（如 10 分钟）
        if time.time() - segment.create_time >= 600:
            return True

        # 条件3：手动触发 flush
        if segment.manual_flush_requested:
            return True

        return False

    def sealed_to_flushing(self, segment):
        """Sealed → Flushing"""
        # 自动触发：Sealed 后立即开始刷新
        return segment.state == SegmentState.SEALED

    def flushing_to_flushed(self, segment):
        """Flushing → Flushed"""
        # 条件：数据已完全写入磁盘
        return segment.flush_completed

    def flushed_to_indexed(self, segment):
        """Flushed → Indexed"""
        # 条件：索引构建完成
        return segment.index_built

    def indexed_to_compacted(self, segment):
        """Indexed → Compacted"""
        # 条件：参与 Compaction 并合并
        return segment.compaction_completed

    def any_to_dropped(self, segment):
        """Any → Dropped"""
        # 条件：Segment 被删除或过期
        return segment.marked_for_deletion
```

---

## Segment 的类型

### 1. Growing Segment（增长中）

**特点：**
- 存储在内存中
- 可以写入新数据
- 查询时需要扫描（没有索引）

**工作流程：**
```python
# Growing Segment 的写入流程
class GrowingSegment:
    def __init__(self):
        self.data = []  # 内存中的数据
        self.max_size = 512 * 1024 * 1024  # 512MB
        self.state = SegmentState.GROWING

    def insert(self, rows):
        """插入数据"""
        # 1. 检查是否已满
        if self._is_full():
            raise Exception("Segment is full, cannot insert")

        # 2. 写入内存
        self.data.extend(rows)

        # 3. 检查是否需要封闭
        if self._should_seal():
            self._seal()

    def _is_full(self):
        """检查是否已满"""
        current_size = len(self.data) * self._row_size()
        return current_size >= self.max_size

    def _should_seal(self):
        """检查是否需要封闭"""
        # 条件1：达到大小阈值
        if self._is_full():
            return True

        # 条件2：达到时间阈值（10 分钟）
        if time.time() - self.create_time >= 600:
            return True

        return False

    def _seal(self):
        """封闭 Segment"""
        self.state = SegmentState.SEALED
        print(f"Segment {self.id} sealed")

    def search(self, query):
        """查询数据（暴力扫描）"""
        # Growing Segment 没有索引，需要暴力扫描
        results = []
        for row in self.data:
            distance = self._calculate_distance(query, row.vector)
            results.append((row, distance))

        # 排序并返回 Top-K
        results.sort(key=lambda x: x[1])
        return results[:10]
```

**在 RAG 中的应用：**
```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("realtime_kb")

# 实时插入数据（写入 Growing Segment）
for i in range(1000):
    collection.insert([{
        "id": i,
        "vector": [0.1, 0.2, 0.3],
        "text": f"文档 {i}"
    }])

# 查询时会扫描 Growing Segment（暴力扫描，较慢）
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10
)

# 注意：Growing Segment 的查询性能较低
# 建议定期 flush，将数据刷新到 Sealed Segment
```

---

### 2. Sealed Segment（已封闭）

**特点：**
- 不再接受新数据
- 准备刷新到磁盘
- 可能仍在内存中

**工作流程：**
```python
# Sealed Segment 的刷新流程
class SealedSegment:
    def __init__(self, growing_segment):
        self.data = growing_segment.data
        self.state = SegmentState.SEALED

    def flush(self):
        """刷新到磁盘"""
        # 1. 标记为刷新中
        self.state = SegmentState.FLUSHING

        # 2. 写入磁盘
        self._write_to_disk()

        # 3. 标记为已刷新
        self.state = SegmentState.FLUSHED

        print(f"Segment {self.id} flushed to disk")

    def _write_to_disk(self):
        """写入磁盘"""
        # 1. 序列化数据
        serialized_data = self._serialize(self.data)

        # 2. 写入文件
        with open(f"/var/lib/milvus/data/segment_{self.id}.dat", "wb") as f:
            f.write(serialized_data)

        # 3. 同步到磁盘
        f.flush()
        os.fsync(f.fileno())

    def search(self, query):
        """查询数据"""
        # Sealed Segment 可能还在内存中
        if self.data is not None:
            # 内存查询（快）
            return self._search_in_memory(query)
        else:
            # 磁盘查询（慢）
            return self._search_on_disk(query)
```

---

### 3. Flushed Segment（已刷新）

**特点：**
- 数据已持久化到磁盘
- 可以构建索引
- 查询时需要加载到内存

**工作流程：**
```python
# Flushed Segment 的索引构建流程
class FlushedSegment:
    def __init__(self, sealed_segment):
        self.file_path = sealed_segment.file_path
        self.state = SegmentState.FLUSHED
        self.index = None

    def build_index(self):
        """构建索引"""
        # 1. 加载数据
        data = self._load_from_disk()

        # 2. 构建索引
        self.index = self._build_index(data)

        # 3. 保存索引
        self._save_index(self.index)

        # 4. 标记为已索引
        self.state = SegmentState.INDEXED

        print(f"Segment {self.id} indexed")

    def _build_index(self, data):
        """构建索引（如 IVF_FLAT）"""
        import faiss

        # 1. 提取向量
        vectors = np.array([row.vector for row in data])

        # 2. 构建索引
        dimension = vectors.shape[1]
        nlist = 100  # 聚类中心数量
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension),
            dimension,
            nlist
        )

        # 3. 训练索引
        index.train(vectors)

        # 4. 添加向量
        index.add(vectors)

        return index

    def search(self, query):
        """查询数据（使用索引）"""
        # 1. 加载索引
        if self.index is None:
            self.index = self._load_index()

        # 2. 使用索引查询
        distances, indices = self.index.search(query, k=10)

        # 3. 返回结果
        return [(self.data[i], distances[0][j]) for j, i in enumerate(indices[0])]
```

---

## Segment 的版本管理（MVCC）

### MVCC 原理

**MVCC (Multi-Version Concurrency Control)** 允许多个版本的数据同时存在，支持并发读写。

```python
# Segment 的 MVCC 实现
class SegmentWithMVCC:
    def __init__(self):
        self.versions = []  # 多个版本
        self.current_version = 0

    def insert(self, rows):
        """插入数据（创建新版本）"""
        # 1. 创建新版本
        new_version = self.current_version + 1

        # 2. 写入数据
        new_segment = Segment(
            id=self.id,
            version=new_version,
            data=rows,
            timestamp=time.time()
        )

        # 3. 添加到版本列表
        self.versions.append(new_segment)

        # 4. 更新当前版本
        self.current_version = new_version

    def search(self, query, timestamp=None):
        """查询数据（指定版本）"""
        # 1. 选择版本
        if timestamp is None:
            # 使用最新版本
            segment = self.versions[-1]
        else:
            # 使用指定时间戳的版本
            segment = self._find_version_by_timestamp(timestamp)

        # 2. 查询数据
        return segment.search(query)

    def _find_version_by_timestamp(self, timestamp):
        """根据时间戳查找版本"""
        for segment in reversed(self.versions):
            if segment.timestamp <= timestamp:
                return segment

        # 如果没有找到，返回最早的版本
        return self.versions[0]

# 使用示例
segment = SegmentWithMVCC()

# 插入数据（版本 1）
segment.insert([{"id": 1, "vector": [0.1, 0.2, 0.3]}])
timestamp_v1 = time.time()

# 插入数据（版本 2）
time.sleep(1)
segment.insert([{"id": 2, "vector": [0.4, 0.5, 0.6]}])
timestamp_v2 = time.time()

# 查询最新版本
results = segment.search(query)  # 返回版本 2 的数据

# 查询历史版本（时间旅行）
results = segment.search(query, timestamp=timestamp_v1)  # 返回版本 1 的数据
```

### 时间旅行查询

```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("versioned_kb")

# 插入数据（版本 1）
collection.insert([{"id": 1, "vector": [0.1, 0.2, 0.3]}])
timestamp_v1 = int(time.time() * 1000)  # 毫秒时间戳

# 等待 1 秒
time.sleep(1)

# 插入数据（版本 2）
collection.insert([{"id": 2, "vector": [0.4, 0.5, 0.6]}])
timestamp_v2 = int(time.time() * 1000)

# 查询最新版本
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10
)
print(f"最新版本结果: {len(results[0])} 条")  # 2 条

# 时间旅行查询（查询版本 1）
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10,
    travel_timestamp=timestamp_v1  # 指定时间戳
)
print(f"版本 1 结果: {len(results[0])} 条")  # 1 条
```

---

## Segment 的查询路由

### 查询路由策略

```python
# Segment 查询路由
class SegmentRouter:
    def __init__(self, collection):
        self.collection = collection

    def route_query(self, query, filters=None):
        """路由查询到合适的 Segment"""
        # 1. 获取所有 Segment
        all_segments = self._get_all_segments()

        # 2. 根据过滤条件筛选 Segment
        candidate_segments = self._filter_segments(all_segments, filters)

        # 3. 并行查询所有候选 Segment
        results = self._parallel_search(candidate_segments, query)

        # 4. 合并结果
        merged_results = self._merge_results(results)

        # 5. 返回 Top-K
        return merged_results[:10]

    def _filter_segments(self, segments, filters):
        """根据过滤条件筛选 Segment"""
        if filters is None:
            return segments

        candidate_segments = []
        for segment in segments:
            # 检查 Segment 的元数据是否匹配过滤条件
            if self._segment_matches_filter(segment, filters):
                candidate_segments.append(segment)

        return candidate_segments

    def _segment_matches_filter(self, segment, filters):
        """检查 Segment 是否匹配过滤条件"""
        # 示例：根据时间范围过滤
        if "timestamp_range" in filters:
            start, end = filters["timestamp_range"]
            if segment.min_timestamp > end or segment.max_timestamp < start:
                return False

        # 示例：根据分区过滤
        if "partition" in filters:
            if segment.partition != filters["partition"]:
                return False

        return True

    def _parallel_search(self, segments, query):
        """并行查询多个 Segment"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(segment.search, query) for segment in segments]
            results = [future.result() for future in futures]

        return results

    def _merge_results(self, results):
        """合并多个 Segment 的查询结果"""
        # 1. 合并所有结果
        all_results = []
        for result in results:
            all_results.extend(result)

        # 2. 按距离排序
        all_results.sort(key=lambda x: x[1])

        # 3. 去重（如果有重复）
        seen_ids = set()
        unique_results = []
        for row, distance in all_results:
            if row.id not in seen_ids:
                unique_results.append((row, distance))
                seen_ids.add(row.id)

        return unique_results
```

---

## Segment 的元数据管理

### Segment 元数据结构

```python
# Segment 元数据
class SegmentMetadata:
    def __init__(self):
        self.id = None                  # Segment ID
        self.collection_id = None       # Collection ID
        self.partition_id = None        # Partition ID
        self.state = None               # 状态
        self.num_rows = 0               # 行数
        self.size = 0                   # 大小（字节）
        self.create_time = None         # 创建时间
        self.flush_time = None          # 刷新时间
        self.index_build_time = None    # 索引构建时间
        self.min_timestamp = None       # 最小时间戳
        self.max_timestamp = None       # 最大时间戳
        self.deleted_rows = 0           # 已删除行数
        self.index_name = None          # 索引名称
        self.index_params = None        # 索引参数

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "partition_id": self.partition_id,
            "state": self.state,
            "num_rows": self.num_rows,
            "size": self.size,
            "create_time": self.create_time,
            "flush_time": self.flush_time,
            "index_build_time": self.index_build_time,
            "min_timestamp": self.min_timestamp,
            "max_timestamp": self.max_timestamp,
            "deleted_rows": self.deleted_rows,
            "index_name": self.index_name,
            "index_params": self.index_params,
        }
```

### 查询 Segment 元数据

```python
from pymilvus import utility, connections

connections.connect("default", host="localhost", port="19530")

# 查询 Segment 信息
segments = utility.get_query_segment_info("my_collection")

for segment in segments:
    print(f"Segment ID: {segment.segmentID}")
    print(f"  Collection ID: {segment.collectionID}")
    print(f"  Partition ID: {segment.partitionID}")
    print(f"  状态: {segment.state}")
    print(f"  行数: {segment.num_rows}")
    print(f"  索引名称: {segment.index_name}")
    print(f"  索引 ID: {segment.indexID}")
    print()

# 输出示例：
"""
Segment ID: 123456789
  Collection ID: 1
  Partition ID: 0
  状态: Sealed
  行数: 1000000
  索引名称: IVF_FLAT
  索引 ID: 987654321
"""
```

---

## 在 RAG 系统中的应用

### 场景1：增量更新知识库

```python
from pymilvus import Collection, connections, utility

connections.connect("default", host="localhost", port="19530")
collection = Collection("incremental_kb")

# 初始导入：100 万条文档
print("初始导入...")
initial_docs = load_initial_documents()
collection.insert(initial_docs)
collection.flush()  # 刷新到 Sealed Segment

# 查看 Segment 信息
segments = utility.get_query_segment_info("incremental_kb")
print(f"初始 Segment 数量: {len(segments)}")

# 每天增量更新：1 万条新文档
for day in range(30):
    print(f"第 {day+1} 天：增量更新...")
    new_docs = load_new_documents(day)
    collection.insert(new_docs)

    # 每天刷新一次
    collection.flush()

# 30 天后：31 个 Segment（1 个初始 + 30 个增量）
segments = utility.get_query_segment_info("incremental_kb")
print(f"30 天后 Segment 数量: {len(segments)}")

# 查询时会并行扫描所有 Segment
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10
)
```

### 场景2：时间旅行查询

```python
from pymilvus import Collection, connections
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("time_travel_kb")

# 版本 1：初始数据
print("插入版本 1...")
collection.insert([{"id": i, "vector": [0.1, 0.2, 0.3], "text": f"文档 {i}"} for i in range(1000)])
timestamp_v1 = int(time.time() * 1000)
print(f"版本 1 时间戳: {timestamp_v1}")

# 等待 10 秒
time.sleep(10)

# 版本 2：更新数据
print("插入版本 2...")
collection.delete(expr="id < 500")  # 删除前 500 条
collection.insert([{"id": i, "vector": [0.4, 0.5, 0.6], "text": f"新文档 {i}"} for i in range(1000, 1500)])
timestamp_v2 = int(time.time() * 1000)
print(f"版本 2 时间戳: {timestamp_v2}")

# 查询最新版本
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10
)
print(f"最新版本结果: {len(results[0])} 条")

# 时间旅行查询（查询版本 1）
results = collection.search(
    data=[[0.1, 0.2, 0.3]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=10,
    travel_timestamp=timestamp_v1
)
print(f"版本 1 结果: {len(results[0])} 条")
```

---

## 核心要点总结

1. **Segment 是存储的基本单位**：类似于数据库的"数据页"
2. **生命周期管理**：Growing → Sealed → Flushed → Indexed → Compacted
3. **MVCC 支持并发**：多个版本同时存在，支持时间旅行查询
4. **查询路由优化**：根据过滤条件筛选 Segment，减少扫描范围
5. **增量更新友好**：新数据写入新 Segment，不影响旧数据
6. **元数据管理**：记录 Segment 的状态、大小、时间戳等信息

Segment 是 Milvus 数据管理的核心机制，支持高效的并发读写和版本管理！
