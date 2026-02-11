# 核心概念 3: Partition 性能优化

## 什么是 Partition 性能优化？

**Partition 性能优化**是指通过合理的分区策略、参数配置、加载策略等手段，最大化 Milvus 检索性能和资源利用率的过程。

### 核心定义

```python
# Partition 性能优化的核心
# 1. Partition Pruning: 跳过无关分区
# 2. 并行检索: 多分区并行搜索
# 3. 内存优化: 热数据常驻，冷数据按需加载
# 4. 分区大小: 控制每个分区的数据量
```

**类比理解**:
- **前端类比**: 就像 React 的虚拟滚动，只渲染可见区域，跳过不可见部分
- **日常类比**: 就像图书馆只开放常用书架，冷门书架按需开放

---

## 1. Partition Pruning（分区裁剪）

### 1.1 什么是 Partition Pruning？

**Partition Pruning** 是 Milvus 的核心优化机制，通过指定分区名，跳过无关分区，只检索相关分区。

```python
# 无 Partition Pruning：检索所有分区
results = collection.search(
    query_vector,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5
    # 没有指定 partition_names，检索所有分区
)
# 耗时: 500ms（检索 1000万条数据）

# 有 Partition Pruning：只检索指定分区
results = collection.search(
    query_vector,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    partition_names=["partition_2024_02"]  # 只检索2月的数据
)
# 耗时: 50ms（只检索 100万条数据）
# 性能提升 10倍！
```

**工作原理**:
```
1. 用户指定 partition_names=["partition_2024_02"]
2. Milvus 只加载 partition_2024_02 的索引
3. 只在 partition_2024_02 的数据上执行检索
4. 跳过其他 11 个分区（partition_2024_01, 03, 04, ...）
5. 返回结果
```

---

### 1.2 Partition Pruning 性能测试

```python
import time
import numpy as np

def benchmark_partition_pruning():
    """测试 Partition Pruning 的性能提升"""

    query_vector = np.random.rand(1, 128).tolist()
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    # 测试1：无 Partition Pruning（检索所有分区）
    start = time.time()
    results_all = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=10
    )
    time_all = (time.time() - start) * 1000

    # 测试2：有 Partition Pruning（只检索1个分区）
    start = time.time()
    results_single = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=10,
        partition_names=["partition_2024_02"]
    )
    time_single = (time.time() - start) * 1000

    # 测试3：有 Partition Pruning（检索3个分区）
    start = time.time()
    results_multi = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=10,
        partition_names=["partition_2024_01", "partition_2024_02", "partition_2024_03"]
    )
    time_multi = (time.time() - start) * 1000

    # 输出结果
    print("Partition Pruning 性能测试:")
    print(f"检索所有分区 (12个): {time_all:.2f}ms")
    print(f"检索单个分区 (1个): {time_single:.2f}ms, 提升 {time_all/time_single:.1f}x")
    print(f"检索多个分区 (3个): {time_multi:.2f}ms, 提升 {time_all/time_multi:.1f}x")

# 运行测试
benchmark_partition_pruning()
```

**预期输出**:
```
Partition Pruning 性能测试:
检索所有分区 (12个): 500.00ms
检索单个分区 (1个): 50.00ms, 提升 10.0x
检索多个分区 (3个): 150.00ms, 提升 3.3x
```

---

### 1.3 最大化 Partition Pruning 效果

**策略1：精确指定分区**

```python
# ❌ 不好：检索所有分区
results = collection.search(query_vector, limit=5)

# ✅ 好：精确指定需要的分区
results = collection.search(
    query_vector,
    limit=5,
    partition_names=["partition_2024_02"]  # 只检索2月
)
```

**策略2：根据查询条件动态确定分区**

```python
def search_with_smart_pruning(query, time_range=None, category=None):
    """根据查询条件智能确定分区"""
    partition_names = []

    # 根据时间范围确定分区
    if time_range:
        start_date, end_date = time_range
        current = start_date
        while current <= end_date:
            partition_name = f"partition_{current.strftime('%Y_%m')}"
            if collection.has_partition(partition_name):
                partition_names.append(partition_name)
            current = datetime(current.year, current.month + 1, 1) if current.month < 12 else datetime(current.year + 1, 1, 1)

    # 根据类别确定分区
    elif category:
        partition_names = [f"category_{category}"]

    # 如果没有条件，检索最近3个月
    else:
        for i in range(3):
            date = datetime.now() - timedelta(days=30*i)
            partition_name = f"partition_{date.strftime('%Y_%m')}"
            if collection.has_partition(partition_name):
                partition_names.append(partition_name)

    # 执行检索
    query_embedding = get_embedding(query)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        partition_names=partition_names
    )

    return results, partition_names

# 使用示例
results, partitions = search_with_smart_pruning(
    query="最新的技术文档",
    time_range=(datetime(2024, 1, 1), datetime(2024, 3, 31))
)
print(f"检索了 {len(partitions)} 个分区: {partitions}")
```

---

## 2. 并行检索优化

### 2.1 多分区并行检索

Milvus 会自动并行检索多个分区，但需要合理控制分区数量。

```python
# 并行检索多个分区
partition_names = ["partition_2024_01", "partition_2024_02", "partition_2024_03"]

results = collection.search(
    query_vector,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=10,
    partition_names=partition_names  # Milvus 会并行检索这3个分区
)
```

**并行检索原理**:
```
1. Milvus 为每个分区分配一个检索任务
2. 多个任务并行执行（利用多核 CPU）
3. 每个任务返回 Top-K 结果
4. 合并所有结果，重新排序，返回全局 Top-K
```

---

### 2.2 并行度控制

```python
# 并行度 vs 性能的权衡

# 场景1：检索1个分区
# - 并行度: 1
# - 耗时: 50ms

# 场景2：检索3个分区
# - 并行度: 3
# - 耗时: 60ms（不是 150ms，因为并行）

# 场景3：检索10个分区
# - 并行度: 10
# - 耗时: 100ms（并行度过高，调度开销增加）

# 推荐：检索分区数量控制在 1-5 个
```

---

## 3. 内存优化策略

### 3.1 冷热数据分离

```python
class HotColdPartitionManager:
    """冷热数据分离管理器"""

    def __init__(self, collection, hot_partition_count=3):
        self.collection = collection
        self.hot_partition_count = hot_partition_count
        self.hot_partitions = []

    def load_hot_partitions(self):
        """加载热数据分区"""
        # 获取最近N个月的分区作为热数据
        for i in range(self.hot_partition_count):
            date = datetime.now() - timedelta(days=30*i)
            partition_name = f"partition_{date.strftime('%Y_%m')}"

            if self.collection.has_partition(partition_name):
                partition = self.collection.partition(partition_name)
                if not partition.is_loaded:
                    partition.load()
                    print(f"✓ 加载热数据分区: {partition_name}")
                self.hot_partitions.append(partition_name)

        print(f"热数据分区已加载: {self.hot_partitions}")

    def search_hot_data(self, query_vector):
        """检索热数据（快速）"""
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=10,
            partition_names=self.hot_partitions
        )
        return results

    def search_cold_data(self, query_vector, partition_name):
        """检索冷数据（按需加载）"""
        partition = self.collection.partition(partition_name)

        # 临时加载冷数据分区
        if not partition.is_loaded:
            partition.load()
            print(f"临时加载冷数据分区: {partition_name}")

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=10,
            partition_names=[partition_name]
        )

        # 用完释放
        partition.release()
        print(f"释放冷数据分区: {partition_name}")

        return results

# 使用示例
manager = HotColdPartitionManager(collection, hot_partition_count=3)
manager.load_hot_partitions()

# 检索热数据（快速）
query_vector = np.random.rand(128).tolist()
results = manager.search_hot_data(query_vector)

# 检索冷数据（按需加载）
results = manager.search_cold_data(query_vector, "partition_2023_01")
```

**内存优化效果**:
```
无优化：加载全部12个月分区 → 占用 12GB 内存
有优化：只加载最近3个月 → 占用 3GB 内存（节省 75%）
```

---

### 3.2 LRU 缓存策略

```python
from collections import OrderedDict

class LRUPartitionCache:
    """LRU 分区缓存管理器"""

    def __init__(self, collection, max_loaded=5):
        self.collection = collection
        self.max_loaded = max_loaded
        self.cache = OrderedDict()  # 保持插入顺序

    def load_partition(self, partition_name):
        """加载分区（LRU策略）"""
        # 如果已在缓存中，移到最前面
        if partition_name in self.cache:
            self.cache.move_to_end(partition_name)
            return

        # 如果缓存已满，释放最久未用的分区
        if len(self.cache) >= self.max_loaded:
            oldest_name, _ = self.cache.popitem(last=False)
            oldest_partition = self.collection.partition(oldest_name)
            oldest_partition.release()
            print(f"释放最久未用的分区: {oldest_name}")

        # 加载新分区
        partition = self.collection.partition(partition_name)
        partition.load()
        self.cache[partition_name] = partition
        print(f"加载分区: {partition_name}")

    def search(self, query_vector, partition_names):
        """检索（自动管理缓存）"""
        # 确保所有分区都已加载
        for name in partition_names:
            self.load_partition(name)

        # 执行检索
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=10,
            partition_names=partition_names
        )

        return results

# 使用示例
cache = LRUPartitionCache(collection, max_loaded=3)

# 检索会自动管理缓存
query_vector = np.random.rand(128).tolist()
results = cache.search(query_vector, ["partition_2024_02"])
results = cache.search(query_vector, ["partition_2024_01"])
results = cache.search(query_vector, ["partition_2023_12"])
results = cache.search(query_vector, ["partition_2023_11"])  # 会释放 partition_2024_02
```

---

## 4. 分区大小优化

### 4.1 最优分区大小

```python
# 推荐的分区大小
OPTIMAL_PARTITION_SIZE = {
    "min": 100_000,        # 最小 10万条
    "optimal": 1_000_000,  # 最优 100万条
    "max": 10_000_000      # 最大 1000万条
}

def calculate_optimal_partition_count(total_entities):
    """计算最优分区数量"""
    optimal_size = OPTIMAL_PARTITION_SIZE["optimal"]
    count = max(1, total_entities // optimal_size)

    # 限制分区数量在合理范围
    if count > 200:
        count = 200
        print(f"警告：分区数量过多，限制为 200 个")

    return count

# 示例
total = 50_000_000
count = calculate_optimal_partition_count(total)
print(f"推荐创建 {count} 个分区，每个分区约 {total//count:,} 条数据")
```

**分区大小 vs 性能**:
```
分区太小（< 10万条）：
- 优点：检索快
- 缺点：分区数量多，管理复杂，元数据开销大

分区太大（> 1000万条）：
- 优点：分区数量少，管理简单
- 缺点：检索慢，Partition Pruning 效果差

最优大小（100万条）：
- 平衡性能和管理复杂度
```

---

### 4.2 动态分区调整

```python
def rebalance_partitions():
    """重新平衡分区（定期维护）"""

    # 1. 检查所有分区的大小
    partition_stats = []
    for partition in collection.partitions:
        stats = {
            "name": partition.name,
            "size": partition.num_entities
        }
        partition_stats.append(stats)

    # 2. 识别需要拆分的大分区
    large_partitions = [p for p in partition_stats if p["size"] > 10_000_000]

    # 3. 识别需要合并的小分区
    small_partitions = [p for p in partition_stats if p["size"] < 100_000]

    print(f"需要拆分的大分区: {len(large_partitions)} 个")
    print(f"需要合并的小分区: {len(small_partitions)} 个")

    # 4. 执行重新平衡（实际实现需要数据迁移）
    # ...

# 定期执行（如每月一次）
rebalance_partitions()
```

---

## 5. 查询优化策略

### 5.1 查询模式分析

```python
class QueryPatternAnalyzer:
    """查询模式分析器"""

    def __init__(self):
        self.query_log = []

    def log_query(self, partition_names, latency_ms):
        """记录查询"""
        self.query_log.append({
            "partitions": partition_names,
            "latency": latency_ms,
            "timestamp": datetime.now()
        })

    def analyze(self):
        """分析查询模式"""
        if not self.query_log:
            return

        # 统计最常访问的分区
        partition_access_count = {}
        for log in self.query_log:
            for partition in log["partitions"]:
                partition_access_count[partition] = partition_access_count.get(partition, 0) + 1

        # 排序
        sorted_partitions = sorted(partition_access_count.items(), key=lambda x: x[1], reverse=True)

        print("最常访问的分区:")
        for partition, count in sorted_partitions[:10]:
            print(f"  {partition}: {count} 次")

        # 统计平均延迟
        avg_latency = sum(log["latency"] for log in self.query_log) / len(self.query_log)
        print(f"\n平均查询延迟: {avg_latency:.2f}ms")

        # 推荐优化策略
        hot_partitions = [p for p, _ in sorted_partitions[:5]]
        print(f"\n推荐常驻内存的热数据分区: {hot_partitions}")

# 使用示例
analyzer = QueryPatternAnalyzer()

# 记录查询
analyzer.log_query(["partition_2024_02"], 50)
analyzer.log_query(["partition_2024_01"], 55)
analyzer.log_query(["partition_2024_02"], 48)

# 分析
analyzer.analyze()
```

---

### 5.2 查询缓存

```python
from functools import lru_cache
import hashlib
import json

class QueryCache:
    """查询结果缓存"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    def _generate_key(self, query_vector, partition_names, limit):
        """生成缓存键"""
        data = {
            "vector": query_vector,
            "partitions": sorted(partition_names),
            "limit": limit
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get(self, query_vector, partition_names, limit):
        """获取缓存"""
        key = self._generate_key(query_vector, partition_names, limit)
        return self.cache.get(key)

    def set(self, query_vector, partition_names, limit, results):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 简单的 FIFO 策略
            self.cache.pop(next(iter(self.cache)))

        key = self._generate_key(query_vector, partition_names, limit)
        self.cache[key] = results

    def search_with_cache(self, query_vector, partition_names, limit=10):
        """带缓存的检索"""
        # 尝试从缓存获取
        cached_results = self.get(query_vector, partition_names, limit)
        if cached_results:
            print("✓ 命中缓存")
            return cached_results

        # 执行检索
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            partition_names=partition_names
        )

        # 缓存结果
        self.set(query_vector, partition_names, limit, results)
        print("✓ 缓存结果")

        return results

# 使用示例
cache = QueryCache(max_size=100)

query_vector = np.random.rand(128).tolist()
results = cache.search_with_cache(query_vector, ["partition_2024_02"], limit=10)
results = cache.search_with_cache(query_vector, ["partition_2024_02"], limit=10)  # 命中缓存
```

---

## 6. 性能监控

### 6.1 性能指标收集

```python
class PartitionPerformanceMonitor:
    """分区性能监控器"""

    def __init__(self):
        self.metrics = []

    def measure_search(self, partition_names, query_func):
        """测量检索性能"""
        start = time.time()
        results = query_func()
        latency = (time.time() - start) * 1000

        metric = {
            "partitions": partition_names,
            "partition_count": len(partition_names),
            "latency_ms": latency,
            "result_count": len(results[0]) if results else 0,
            "timestamp": datetime.now()
        }
        self.metrics.append(metric)

        return results, metric

    def report(self):
        """生成性能报告"""
        if not self.metrics:
            return

        print("=== 分区性能报告 ===\n")

        # 按分区数量分组统计
        by_partition_count = {}
        for m in self.metrics:
            count = m["partition_count"]
            if count not in by_partition_count:
                by_partition_count[count] = []
            by_partition_count[count].append(m["latency_ms"])

        print("按分区数量统计:")
        for count in sorted(by_partition_count.keys()):
            latencies = by_partition_count[count]
            avg_latency = sum(latencies) / len(latencies)
            print(f"  {count} 个分区: 平均延迟 {avg_latency:.2f}ms")

        # 最慢的查询
        slowest = max(self.metrics, key=lambda x: x["latency_ms"])
        print(f"\n最慢的查询:")
        print(f"  分区: {slowest['partitions']}")
        print(f"  延迟: {slowest['latency_ms']:.2f}ms")

# 使用示例
monitor = PartitionPerformanceMonitor()

query_vector = np.random.rand(128).tolist()

# 测量单分区检索
results, metric = monitor.measure_search(
    ["partition_2024_02"],
    lambda: collection.search(
        [query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        partition_names=["partition_2024_02"]
    )
)
print(f"单分区检索: {metric['latency_ms']:.2f}ms")

# 生成报告
monitor.report()
```

---

## 7. 在实际应用中的使用

### 7.1 RAG 系统的性能优化

```python
class OptimizedRAGSystem:
    """优化的 RAG 系统"""

    def __init__(self, collection):
        self.collection = collection
        self.hot_manager = HotColdPartitionManager(collection, hot_partition_count=3)
        self.query_cache = QueryCache(max_size=100)

        # 加载热数据分区
        self.hot_manager.load_hot_partitions()

    def search(self, query, time_range_months=3):
        """优化的检索"""
        # 1. 生成查询向量
        query_embedding = get_embedding(query)

        # 2. 确定检索分区（最近N个月）
        partition_names = []
        for i in range(time_range_months):
            date = datetime.now() - timedelta(days=30*i)
            partition_name = f"partition_{date.strftime('%Y_%m')}"
            if self.collection.has_partition(partition_name):
                partition_names.append(partition_name)

        # 3. 带缓存的检索
        results = self.query_cache.search_with_cache(
            query_embedding,
            partition_names,
            limit=10
        )

        return results

# 使用示例
rag_system = OptimizedRAGSystem(collection)
results = rag_system.search("最新的技术文档", time_range_months=3)
```

---

## 总结

### 核心要点

1. **Partition Pruning**: 通过指定分区跳过无关数据，性能提升 3-10 倍
2. **并行检索**: Milvus 自动并行检索多个分区，控制分区数量在 1-5 个
3. **内存优化**: 热数据常驻内存，冷数据按需加载，节省 50-80% 内存
4. **分区大小**: 推荐每个分区 100万条数据，平衡性能和管理复杂度
5. **查询优化**: 分析查询模式，使用缓存，监控性能指标

### 最佳实践

- ✅ 始终明确指定 partition_names
- ✅ 根据查询模式设计分区策略
- ✅ 控制分区数量在 10-200 个
- ✅ 只加载常用分区到内存
- ✅ 定期分析查询模式和性能指标

### 性能提升总结

| 优化策略 | 性能提升 | 内存节省 |
|---------|---------|---------|
| Partition Pruning | 3-10x | - |
| 冷热数据分离 | - | 50-80% |
| 查询缓存 | 10-100x（命中时） | - |
| 分区大小优化 | 1.5-2x | - |
| 组合优化 | 10-50x | 50-80% |

### 下一步

学习 **实战代码场景**，通过实际案例掌握 Partition 的使用和优化技巧。
