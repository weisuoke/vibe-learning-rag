# 核心概念 02：Compaction 机制

## 什么是 Compaction？

**Compaction 是一种"数据整理和优化"机制，通过合并小文件、删除无效数据、重建索引来提升存储效率和查询性能。**

---

## 一句话定义

**Compaction 是数据库定期合并和清理数据碎片，优化存储布局和查询性能的后台任务。**

---

## Compaction 的核心原理

### 1. 为什么需要 Compaction？

**问题：数据碎片化的困境**

```python
# 场景：知识库每天更新
# 第 1 天：插入 10 万条数据
collection.insert(data_day1)  # 创建 Segment 1（100MB）

# 第 2 天：插入 10 万条数据
collection.insert(data_day2)  # 创建 Segment 2（100MB）

# 第 3 天：删除 5 万条旧数据
collection.delete(expr="id < 50000")  # Segment 1 中的数据被标记删除

# 第 4 天：插入 5 万条数据
collection.insert(data_day4)  # 创建 Segment 3（50MB）

# 问题：
# 1. Segment 1 中有 50% 的数据已删除，但仍占用空间
# 2. Segment 3 太小，查询时需要打开更多文件
# 3. 索引结构不再最优，查询性能下降
```

**解决方案：Compaction 机制**

```python
# 运行 Compaction
utility.do_compact(collection_name="knowledge_base")

# Compaction 做了什么：
# 1. 合并 Segment 1 + Segment 3 → Segment 4（75MB，只包含有效数据）
# 2. 删除 Segment 1 和 Segment 3
# 3. 重建 Segment 4 的索引
# 4. 优化存储布局

# 结果：
# - 存储空间：从 250MB 减少到 175MB（节省 30%）
# - Segment 数量：从 3 个减少到 2 个
# - 查询性能：提升 30%（不再扫描无效数据）
```

---

## Compaction 的三种类型

### 类型 1：Minor Compaction（小规模合并）

**目标：** 合并小 Segment 为大 Segment

**触发条件：**
- 小 Segment 数量 > 阈值（如 10 个）
- Segment 大小 < 阈值（如 64MB）

**运行频率：** 高（每小时）

**工作流程：**
```python
# Minor Compaction 示例
def minor_compaction():
    # 1. 选择小 Segment
    small_segments = [
        Segment(id=1, size=10MB, rows=10000),
        Segment(id=2, size=15MB, rows=15000),
        Segment(id=3, size=20MB, rows=20000),
    ]

    # 2. 合并为大 Segment
    merged_segment = merge_segments(small_segments)
    # Segment(id=4, size=45MB, rows=45000)

    # 3. 删除旧 Segment
    for segment in small_segments:
        delete_segment(segment)

    # 4. 返回新 Segment
    return merged_segment

# 效果：
# - Segment 数量：从 3 个减少到 1 个
# - 查询性能：提升 20%（减少文件打开次数）
```

**在 RAG 中的应用：**
```python
# 场景：实时知识库，每小时导入新文档
from pymilvus import Collection, utility

collection = Collection("realtime_kb")

# 每小时导入 1 万条文档
for hour in range(24):
    documents = load_documents_for_hour(hour)
    collection.insert(documents)
    print(f"第 {hour} 小时：插入 {len(documents)} 条文档")

# 24 小时后：24 个小 Segment（每个 10MB）
segments = utility.get_query_segment_info("realtime_kb")
print(f"Segment 数量: {len(segments)}")  # 24 个

# 运行 Minor Compaction
utility.do_compact(collection_name="realtime_kb")

# Compaction 后：2-3 个大 Segment（每个 100MB）
segments = utility.get_query_segment_info("realtime_kb")
print(f"Segment 数量: {len(segments)}")  # 2-3 个
```

---

### 类型 2：Major Compaction（大规模合并）

**目标：** 删除标记为删除的数据，重建索引

**触发条件：**
- 删除数据比例 > 阈值（如 20%）
- 索引碎片化严重

**运行频率：** 中（每天）

**工作流程：**
```python
# Major Compaction 示例
def major_compaction():
    # 1. 选择需要清理的 Segment
    segments_to_compact = [
        Segment(id=1, size=100MB, rows=100000, deleted_rows=30000),  # 30% 已删除
        Segment(id=2, size=100MB, rows=100000, deleted_rows=50000),  # 50% 已删除
    ]

    # 2. 读取有效数据
    valid_data = []
    for segment in segments_to_compact:
        for row in segment.read_all():
            if not row.is_deleted:
                valid_data.append(row)

    # 3. 创建新 Segment
    new_segment = create_segment(valid_data)
    # Segment(id=3, size=120MB, rows=120000, deleted_rows=0)

    # 4. 重建索引
    new_segment.build_index()

    # 5. 删除旧 Segment
    for segment in segments_to_compact:
        delete_segment(segment)

    return new_segment

# 效果：
# - 存储空间：从 200MB 减少到 120MB（节省 40%）
# - 查询性能：提升 40%（不再扫描无效数据）
```

**在 RAG 中的应用：**
```python
# 场景：知识库每天更新，删除过时文档
from pymilvus import Collection, utility
import time

collection = Collection("daily_update_kb")

# 初始状态：100 万条文档
print("初始状态:")
print(f"  数据量: 1,000,000 条")
print(f"  存储空间: 1GB")

# 30 天后：每天删除 10% 旧数据，插入 10% 新数据
for day in range(30):
    # 删除旧数据
    cutoff_timestamp = time.time() - 30 * 24 * 3600
    collection.delete(expr=f"timestamp < {cutoff_timestamp}")

    # 插入新数据
    new_documents = load_new_documents(day)
    collection.insert(new_documents)

# 30 天后的状态（未运行 Compaction）
print("\n30 天后（未运行 Compaction）:")
print(f"  数据量: 1,000,000 条（不变）")
print(f"  存储空间: 1.5GB（增加 50%，因为删除的数据未清理）")
print(f"  查询耗时: 150ms（增加 50%，因为需要扫描无效数据）")

# 运行 Major Compaction
print("\n运行 Major Compaction...")
compaction_id = utility.do_compact(collection_name="daily_update_kb")
utility.wait_for_compaction_completed(compaction_id)

# Compaction 后的状态
print("\nCompaction 后:")
print(f"  数据量: 1,000,000 条（不变）")
print(f"  存储空间: 1GB（回收 0.5GB）")
print(f"  查询耗时: 100ms（恢复到初始水平）")
```

---

### 类型 3：Full Compaction（全量合并）

**目标：** 重新组织所有数据，优化存储布局

**触发条件：**
- 手动触发
- 定期运行（如每周）

**运行频率：** 低（每周）

**工作流程：**
```python
# Full Compaction 示例
def full_compaction():
    # 1. 读取所有 Segment
    all_segments = get_all_segments()

    # 2. 读取所有有效数据
    all_data = []
    for segment in all_segments:
        for row in segment.read_all():
            if not row.is_deleted:
                all_data.append(row)

    # 3. 按最优方式重新组织数据
    # - 按时间排序
    # - 按 ID 范围分区
    # - 优化数据布局
    all_data.sort(key=lambda x: x.timestamp)

    # 4. 创建新 Segment（最优大小）
    new_segments = []
    optimal_size = 512 * 1024 * 1024  # 512MB
    for i in range(0, len(all_data), optimal_size):
        segment_data = all_data[i:i+optimal_size]
        new_segment = create_segment(segment_data)
        new_segment.build_index()
        new_segments.append(new_segment)

    # 5. 删除所有旧 Segment
    for segment in all_segments:
        delete_segment(segment)

    return new_segments

# 效果：
# - 存储空间：最优化
# - 查询性能：最优化
# - Segment 数量：最优化
```

**在 RAG 中的应用：**
```python
# 场景：季度性能优化
from pymilvus import Collection, utility

collection = Collection("enterprise_kb")

# 季度末：运行 Full Compaction
print("运行 Full Compaction（季度性能优化）...")
compaction_id = utility.do_compact(
    collection_name="enterprise_kb",
    compaction_type="full"  # 全量合并
)
utility.wait_for_compaction_completed(compaction_id)

print("Full Compaction 完成！")
print("效果：")
print("  - 存储空间：优化 30%-50%")
print("  - 查询性能：提升 30%-50%")
print("  - Segment 数量：减少 80%-90%")
print("  - 索引结构：重建为最优状态")
```

---

## Compaction 的工作流程

### 详细流程图

```
1. 触发 Compaction
    ↓
2. 选择需要合并的 Segment
    ↓
3. 创建 Compaction 任务
    ↓
4. 读取 Segment 数据
    ↓
5. 过滤已删除数据
    ↓
6. 合并数据
    ↓
7. 创建新 Segment
    ↓
8. 重建索引
    ↓
9. 标记旧 Segment 为"待删除"
    ↓
10. 等待所有查询完成
    ↓
11. 删除旧 Segment
    ↓
12. 更新元数据
    ↓
13. Compaction 完成
```

### 并发控制（MVCC）

```python
# Compaction 使用 MVCC 保证查询不受影响
class CompactionWithMVCC:
    def compact(self, segments):
        # 1. 创建新版本的 Segment
        new_segment = self._merge_segments(segments)
        new_segment.version = self._get_next_version()

        # 2. 旧版本仍然可用（查询可以继续使用）
        for segment in segments:
            segment.mark_as_old_version()

        # 3. 新查询使用新版本
        self._update_active_segments(new_segment)

        # 4. 等待所有使用旧版本的查询完成
        self._wait_for_old_queries_to_complete(segments)

        # 5. 删除旧版本
        for segment in segments:
            self._delete_segment(segment)

# 查询线程的视角
def query_thread():
    # 获取当前版本的 Segment
    segments = get_active_segments()

    # 执行查询（即使 Compaction 正在运行）
    results = search(segments, query)

    # 查询完成后，释放 Segment 引用
    release_segments(segments)

# Compaction 线程的视角
def compaction_thread():
    # 选择需要合并的 Segment
    segments = select_segments_to_compact()

    # 创建新 Segment（不影响查询）
    new_segment = merge_segments(segments)

    # 等待所有查询完成
    wait_for_queries_to_complete(segments)

    # 删除旧 Segment
    delete_segments(segments)
```

---

## Compaction 的触发策略

### 1. 自动触发

```python
# Milvus 自动触发 Compaction 的条件
class AutoCompactionTrigger:
    def __init__(self):
        # 小 Segment 数量阈值
        self.small_segment_threshold = 10

        # 删除数据比例阈值
        self.deleted_ratio_threshold = 0.2

        # Segment 大小阈值
        self.small_segment_size = 64 * 1024 * 1024  # 64MB

    def should_trigger_minor_compaction(self, collection):
        # 统计小 Segment 数量
        small_segments = [
            s for s in collection.segments
            if s.size < self.small_segment_size
        ]

        # 如果小 Segment 数量超过阈值，触发 Minor Compaction
        return len(small_segments) > self.small_segment_threshold

    def should_trigger_major_compaction(self, collection):
        # 统计删除数据比例
        total_rows = sum(s.num_rows for s in collection.segments)
        deleted_rows = sum(s.deleted_rows for s in collection.segments)
        deleted_ratio = deleted_rows / total_rows if total_rows > 0 else 0

        # 如果删除数据比例超过阈值，触发 Major Compaction
        return deleted_ratio > self.deleted_ratio_threshold

# Milvus 后台定期检查
def background_compaction_checker():
    while True:
        for collection in get_all_collections():
            trigger = AutoCompactionTrigger()

            if trigger.should_trigger_minor_compaction(collection):
                print(f"自动触发 Minor Compaction: {collection.name}")
                do_compact(collection.name, compaction_type="minor")

            if trigger.should_trigger_major_compaction(collection):
                print(f"自动触发 Major Compaction: {collection.name}")
                do_compact(collection.name, compaction_type="major")

        time.sleep(3600)  # 每小时检查一次
```

### 2. 手动触发

```python
from pymilvus import utility

# 手动触发 Compaction
compaction_id = utility.do_compact(collection_name="my_collection")

# 查询 Compaction 状态
state = utility.get_compaction_state(compaction_id)
print(f"Compaction 状态: {state}")

# 等待 Compaction 完成
utility.wait_for_compaction_completed(compaction_id)
print("Compaction 完成！")
```

### 3. 定时触发

```python
import schedule
import time
from pymilvus import utility

# 定时触发 Compaction
def scheduled_compaction():
    print("运行定时 Compaction...")
    compaction_id = utility.do_compact(collection_name="my_collection")
    utility.wait_for_compaction_completed(compaction_id)
    print("Compaction 完成！")

# 每天凌晨 2 点运行
schedule.every().day.at("02:00").do(scheduled_compaction)

# 每周日凌晨 3 点运行 Full Compaction
def scheduled_full_compaction():
    print("运行 Full Compaction...")
    compaction_id = utility.do_compact(
        collection_name="my_collection",
        compaction_type="full"
    )
    utility.wait_for_compaction_completed(compaction_id)
    print("Full Compaction 完成！")

schedule.every().sunday.at("03:00").do(scheduled_full_compaction)

# 运行调度器
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Compaction 的性能优化

### 1. 并行 Compaction

```python
# 并行运行多个 Compaction 任务
import concurrent.futures
from pymilvus import utility

def compact_collection(collection_name):
    compaction_id = utility.do_compact(collection_name=collection_name)
    utility.wait_for_compaction_completed(compaction_id)
    return f"{collection_name} Compaction 完成"

# 并行 Compaction 多个 Collection
collections = ["kb1", "kb2", "kb3", "kb4", "kb5"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(compact_collection, col) for col in collections]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())

# 输出：
# kb1 Compaction 完成
# kb2 Compaction 完成
# kb3 Compaction 完成
# kb4 Compaction 完成
# kb5 Compaction 完成
```

### 2. 增量 Compaction

```python
# 增量 Compaction：只合并最近的 Segment
def incremental_compaction(collection_name, days=7):
    """
    只合并最近 N 天的 Segment
    """
    from pymilvus import utility
    import time

    # 获取所有 Segment
    segments = utility.get_query_segment_info(collection_name)

    # 筛选最近 N 天的 Segment
    cutoff_timestamp = time.time() - days * 24 * 3600
    recent_segments = [
        s for s in segments
        if s.timestamp > cutoff_timestamp
    ]

    print(f"找到 {len(recent_segments)} 个最近 {days} 天的 Segment")

    # 只合并这些 Segment
    # （Milvus 会自动选择需要合并的 Segment）
    compaction_id = utility.do_compact(collection_name=collection_name)
    utility.wait_for_compaction_completed(compaction_id)

    print("增量 Compaction 完成！")

# 使用示例
incremental_compaction("my_collection", days=7)
```

### 3. 分区 Compaction

```python
# 分区 Compaction：只合并特定分区
from pymilvus import Collection, utility

collection = Collection("partitioned_kb")

# 只合并特定分区
partition_name = "2024_Q1"
print(f"运行分区 Compaction: {partition_name}")

# Milvus 会自动处理分区级别的 Compaction
compaction_id = utility.do_compact(collection_name="partitioned_kb")
utility.wait_for_compaction_completed(compaction_id)

print(f"分区 {partition_name} Compaction 完成！")
```

---

## 在 Milvus 中的实现

### Milvus Compaction 配置

```yaml
# Milvus 配置文件（milvus.yaml）
dataCoord:
  # 启用自动 Compaction
  enableCompaction: true

  # Minor Compaction 配置
  compaction:
    # 小 Segment 数量阈值
    minSegmentToMerge: 10

    # Segment 大小阈值（64MB）
    maxSegmentSize: 67108864

  # Major Compaction 配置
  majorCompaction:
    # 删除数据比例阈值
    deletedRatioThreshold: 0.2

    # 运行间隔（24 小时）
    interval: 86400

  # Compaction 并发数
  maxParallelCompactionTaskNum: 10
```

### 监控 Compaction 状态

```python
from pymilvus import utility

# 查询 Compaction 状态
def monitor_compaction(compaction_id):
    while True:
        state = utility.get_compaction_state(compaction_id)

        print(f"Compaction ID: {compaction_id}")
        print(f"  状态: {state.state}")
        print(f"  进度: {state.executing_plan_no}/{state.total_plan_no}")
        print(f"  完成时间: {state.completed_time}")

        if state.state == "Completed":
            print("Compaction 完成！")
            break

        time.sleep(5)

# 使用示例
compaction_id = utility.do_compact(collection_name="my_collection")
monitor_compaction(compaction_id)
```

---

## 在 RAG 系统中的应用

### 最佳实践

```python
from pymilvus import Collection, utility
import schedule
import time

# 1. 自动 Compaction（推荐）
# Milvus 会自动运行 Compaction，无需手动配置

# 2. 定期手动 Compaction（可选）
def daily_compaction():
    """每天凌晨运行 Compaction"""
    collections = ["kb1", "kb2", "kb3"]
    for collection_name in collections:
        print(f"运行 Compaction: {collection_name}")
        compaction_id = utility.do_compact(collection_name=collection_name)
        utility.wait_for_compaction_completed(compaction_id)
        print(f"{collection_name} Compaction 完成")

schedule.every().day.at("02:00").do(daily_compaction)

# 3. 周末 Full Compaction（推荐）
def weekly_full_compaction():
    """每周日运行 Full Compaction"""
    collections = ["kb1", "kb2", "kb3"]
    for collection_name in collections:
        print(f"运行 Full Compaction: {collection_name}")
        compaction_id = utility.do_compact(
            collection_name=collection_name,
            compaction_type="full"
        )
        utility.wait_for_compaction_completed(compaction_id)
        print(f"{collection_name} Full Compaction 完成")

schedule.every().sunday.at("03:00").do(weekly_full_compaction)

# 运行调度器
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 核心要点总结

1. **Compaction 是长期运行的必需品**：没有 Compaction，系统会逐渐变慢
2. **三种类型各有用途**：Minor（合并小文件）、Major（删除无效数据）、Full（全量优化）
3. **MVCC 保证并发安全**：Compaction 不会阻塞查询
4. **自动触发是默认选项**：Milvus 会自动运行 Compaction
5. **定期手动 Compaction 是最佳实践**：每周运行一次 Full Compaction
6. **监控 Compaction 状态**：确保 Compaction 正常运行

Compaction 是 Milvus 性能优化的核心机制，确保长期稳定运行！
