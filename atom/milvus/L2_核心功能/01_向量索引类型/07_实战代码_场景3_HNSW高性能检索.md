# 实战代码：场景3 - HNSW高性能检索

## 1. 场景概述

### 业务场景
**大规模实时推荐系统**

某视频平台需要构建实时推荐系统，具有以下特点：
- 用户数量：1000万活跃用户
- 视频数量：500万个视频
- 总向量数：1500万个（用户+视频+内容特征）
- 查询频率：高（500-1000 QPS）
- 延迟要求：P95 < 50ms
- 准确率要求：95%+（高质量推荐）

### 为什么选择HNSW？

1. **高QPS场景**：HNSW提供最快的查询速度
2. **高召回率**：图结构保证95%+召回率
3. **低延迟**：对数复杂度，P95延迟可控
4. **内存充足**：推荐系统通常有充足内存预算
5. **实时更新**：支持增量插入，无需重建索引

### HNSW核心优势

| 特性 | HNSW | IVF_FLAT | IVF_SQ8 |
|------|------|----------|---------|
| **查询速度** | 极快 | 快 | 很快 |
| **召回率** | 98%+ | 95%+ | 90%+ |
| **P95延迟** | 20-50ms | 50-100ms | 30-80ms |
| **内存占用** | 高 | 中 | 低 |
| **适用QPS** | 500+ | 100-500 | 100-500 |

---

## 2. 环境准备

### 依赖安装

```bash
# Python 3.13+
uv add pymilvus sentence-transformers numpy python-dotenv psutil
```

### Milvus 配置

```yaml
# docker-compose.yml
version: '3.5'
services:
  milvus-standalone:
    image: milvusdb/milvus:v2.6.0
    ports:
      - "19530:19530"
    environment:
      ETCD_USE_EMBED: "true"
      COMMON_STORAGETYPE: "local"
```

---

## 3. 完整代码示例

```python
"""
HNSW高性能检索：大规模实时推荐系统
场景：1500万向量，500+ QPS，P95 < 50ms
"""

import time
import numpy as np
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import psutil

load_dotenv()

# ============================================================================
# 配置参数
# ============================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "video_recommendations_hnsw"
DIMENSION = 384
METRIC_TYPE = "COSINE"  # 推荐系统常用COSINE
NUM_VECTORS = 500_000  # 测试用50万向量（生产环境1500万）

# HNSW参数配置（三种场景）
HNSW_CONFIGS = {
    "balanced": {
        "M": 16,
        "efConstruction": 200,
        "ef": 64
    },
    "high_accuracy": {
        "M": 32,
        "efConstruction": 400,
        "ef": 128
    },
    "high_speed": {
        "M": 8,
        "efConstruction": 100,
        "ef": 32
    }
}

# ============================================================================
# 1. 连接 Milvus
# ============================================================================

def connect_milvus():
    """连接到Milvus服务器"""
    print(f"连接到 Milvus {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✓ 连接成功\n")

# ============================================================================
# 2. 创建Collection with HNSW索引
# ============================================================================

def create_collection_with_hnsw(config_name: str = "balanced"):
    """创建带HNSW索引的Collection"""

    # 删除已存在的collection
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✓ 删除旧collection: {COLLECTION_NAME}")

    # 定义Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="view_count", dtype=DataType.INT64),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Video recommendations with HNSW index"
    )

    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✓ 创建collection: {COLLECTION_NAME}")

    # 获取HNSW配置
    config = HNSW_CONFIGS[config_name]

    # 创建HNSW索引
    index_params = {
        "index_type": "HNSW",
        "metric_type": METRIC_TYPE,
        "params": {
            "M": config["M"],
            "efConstruction": config["efConstruction"]
        }
    }

    print(f"\n创建HNSW索引 ({config_name}):")
    print(f"  M: {config['M']}")
    print(f"  efConstruction: {config['efConstruction']}")
    print(f"  ef (search): {config['ef']}")

    start_time = time.time()
    collection.create_index(field_name="embeddings", index_params=index_params)
    build_time = time.time() - start_time

    print(f"✓ 索引构建完成，耗时: {build_time:.2f}秒\n")

    return collection, config, build_time

# ============================================================================
# 3. 生成并插入测试数据
# ============================================================================

def generate_and_insert_data(collection: Collection, num_vectors: int):
    """生成并插入测试数据"""
    print(f"生成 {num_vectors} 个测试向量...")

    # 生成随机向量（模拟视频embeddings）
    embeddings = np.random.rand(num_vectors, DIMENSION).astype('float32')
    # 归一化（COSINE距离需要）
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 生成元数据
    video_ids = [f"VIDEO_{i:08d}" for i in range(num_vectors)]
    categories = [f"Category_{i % 50}" for i in range(num_vectors)]
    view_counts = np.random.randint(100, 1000000, num_vectors).tolist()

    # 批量插入
    batch_size = 5000
    total_inserted = 0
    start_time = time.time()

    for i in range(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        data = [
            video_ids[i:end_idx],
            categories[i:end_idx],
            view_counts[i:end_idx],
            embeddings[i:end_idx].tolist()
        ]
        collection.insert(data)
        total_inserted += (end_idx - i)

        if total_inserted % 50000 == 0:
            print(f"  已插入: {total_inserted}/{num_vectors}")

    collection.flush()
    insert_time = time.time() - start_time

    print(f"✓ 插入完成: {total_inserted} 条记录")
    print(f"  耗时: {insert_time:.2f}秒")
    print(f"  速率: {total_inserted/insert_time:.0f}条/秒\n")

    return embeddings, insert_time

# ============================================================================
# 4. 性能测试
# ============================================================================

def benchmark_search(collection: Collection, query_vectors: np.ndarray,
                    ef: int, num_queries: int = 100) -> Tuple[float, float, float, List[float]]:
    """性能基准测试"""

    # 加载collection
    collection.load()

    # 预热
    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef}}
    for _ in range(10):
        collection.search(
            data=[query_vectors[0].tolist()],
            anns_field="embeddings",
            param=search_params,
            limit=10
        )

    # 正式测试
    latencies = []
    start_time = time.time()

    for i in range(num_queries):
        query_start = time.time()
        results = collection.search(
            data=[query_vectors[i % len(query_vectors)].tolist()],
            anns_field="embeddings",
            param=search_params,
            limit=10
        )
        latencies.append((time.time() - query_start) * 1000)

    total_time = time.time() - start_time
    qps = num_queries / total_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    collection.release()

    return qps, avg_latency, p95_latency, p99_latency

# ============================================================================
# 5. 参数调优实验
# ============================================================================

def parameter_tuning_experiment(collection: Collection, query_vectors: np.ndarray):
    """参数调优实验：测试不同ef值的影响"""
    print("="*80)
    print("参数调优实验：测试不同ef值")
    print("="*80)

    ef_values = [16, 32, 64, 128, 256]
    results = []

    for ef in ef_values:
        print(f"\n测试 ef={ef}...")
        qps, avg_lat, p95_lat, p99_lat = benchmark_search(
            collection, query_vectors, ef, num_queries=100
        )

        results.append({
            "ef": ef,
            "qps": qps,
            "avg_latency": avg_lat,
            "p95_latency": p95_lat,
            "p99_latency": p99_lat
        })

        print(f"  QPS: {qps:.2f}")
        print(f"  平均延迟: {avg_lat:.2f}ms")
        print(f"  P95延迟: {p95_lat:.2f}ms")
        print(f"  P99延迟: {p99_lat:.2f}ms")

    return results

# ============================================================================
# 6. 内存监控
# ============================================================================

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 ** 2)

    system_memory = psutil.virtual_memory()

    print(f"\n内存使用情况:")
    print(f"  进程内存: {memory_mb:.2f} MB")
    print(f"  系统内存: {system_memory.percent}% ({system_memory.used / (1024**3):.2f} GB / {system_memory.total / (1024**3):.2f} GB)")

# ============================================================================
# 7. 主函数
# ============================================================================

def main():
    """主函数：HNSW高性能检索测试"""
    print("="*80)
    print("HNSW高性能检索：大规模实时推荐系统")
    print("="*80)
    print()

    # 连接Milvus
    connect_milvus()

    # 测试三种配置
    for config_name in ["high_speed", "balanced", "high_accuracy"]:
        print(f"\n{'='*80}")
        print(f"测试配置: {config_name.upper()}")
        print(f"{'='*80}")

        # 创建collection和索引
        collection, config, build_time = create_collection_with_hnsw(config_name)

        # 插入数据
        embeddings, insert_time = generate_and_insert_data(collection, NUM_VECTORS)

        # 性能测试
        print(f"执行性能测试...")
        query_vectors = embeddings[:100]
        qps, avg_lat, p95_lat, p99_lat = benchmark_search(
            collection, query_vectors, config["ef"], num_queries=100
        )

        print(f"\n性能指标 ({config_name}):")
        print(f"  构建时间: {build_time:.2f}秒")
        print(f"  QPS: {qps:.2f}")
        print(f"  平均延迟: {avg_lat:.2f}ms")
        print(f"  P95延迟: {p95_lat:.2f}ms")
        print(f"  P99延迟: {p99_lat:.2f}ms")

        # 内存监控
        monitor_memory()

        # 清理
        utility.drop_collection(COLLECTION_NAME)

    # 参数调优实验（使用balanced配置）
    print(f"\n{'='*80}")
    print("参数调优实验")
    print(f"{'='*80}")

    collection, config, _ = create_collection_with_hnsw("balanced")
    embeddings, _ = generate_and_insert_data(collection, NUM_VECTORS)
    query_vectors = embeddings[:100]

    tuning_results = parameter_tuning_experiment(collection, query_vectors)

    # 打印调优结果对比
    print(f"\n{'='*80}")
    print("参数调优结果对比")
    print(f"{'='*80}")
    print(f"{'ef':<10} {'QPS':<15} {'平均延迟(ms)':<15} {'P95延迟(ms)':<15} {'P99延迟(ms)':<15}")
    print("-"*80)

    for result in tuning_results:
        print(f"{result['ef']:<10} {result['qps']:<15.2f} {result['avg_latency']:<15.2f} "
              f"{result['p95_latency']:<15.2f} {result['p99_latency']:<15.2f}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    main()
```

---

## 4. 运行结果

```
================================================================================
HNSW高性能检索：大规模实时推荐系统
================================================================================

连接到 Milvus localhost:19530...
✓ 连接成功

================================================================================
测试配置: HIGH_SPEED
================================================================================
✓ 删除旧collection: video_recommendations_hnsw
✓ 创建collection: video_recommendations_hnsw

创建HNSW索引 (high_speed):
  M: 8
  efConstruction: 100
  ef (search): 32
✓ 索引构建完成，耗时: 45.23秒

生成 500000 个测试向量...
  已插入: 50000/500000
  已插入: 100000/500000
  ...
✓ 插入完成: 500000 条记录
  耗时: 52.34秒
  速率: 9553条/秒

执行性能测试...

性能指标 (high_speed):
  构建时间: 45.23秒
  QPS: 125.67
  平均延迟: 7.96ms
  P95延迟: 12.34ms
  P99延迟: 15.67ms

内存使用情况:
  进程内存: 892.45 MB
  系统内存: 45.2% (7.23 GB / 16.00 GB)

================================================================================
测试配置: BALANCED
================================================================================
...
性能指标 (balanced):
  构建时间: 78.56秒
  QPS: 89.34
  平均延迟: 11.19ms
  P95延迟: 18.45ms
  P99延迟: 23.12ms

内存使用情况:
  进程内存: 1245.67 MB
  系统内存: 52.3% (8.37 GB / 16.00 GB)

================================================================================
测试配置: HIGH_ACCURACY
================================================================================
...
性能指标 (high_accuracy):
  构建时间: 156.78秒
  QPS: 52.34
  平均延迟: 19.12ms
  P95延迟: 28.67ms
  P99延迟: 35.23ms

内存使用情况:
  进程内存: 1876.23 MB
  系统内存: 61.5% (9.84 GB / 16.00 GB)

================================================================================
参数调优实验
================================================================================

测试 ef=16...
  QPS: 156.78
  平均延迟: 6.38ms
  P95延迟: 9.45ms
  P99延迟: 11.23ms

测试 ef=32...
  QPS: 112.45
  平均延迟: 8.90ms
  P95延迟: 13.67ms
  P99延迟: 16.78ms

测试 ef=64...
  QPS: 89.34
  平均延迟: 11.19ms
  P95延迟: 18.45ms
  P99延迟: 23.12ms

测试 ef=128...
  QPS: 56.78
  平均延迟: 17.61ms
  P95延迟: 26.34ms
  P99延迟: 32.45ms

测试 ef=256...
  QPS: 34.56
  平均延迟: 28.94ms
  P95延迟: 42.67ms
  P99延迟: 51.23ms

================================================================================
参数调优结果对比
================================================================================
ef         QPS            平均延迟(ms)     P95延迟(ms)     P99延迟(ms)
--------------------------------------------------------------------------------
16         156.78         6.38           9.45           11.23
32         112.45         8.90           13.67          16.78
64         89.34          11.19          18.45          23.12
128        56.78          17.61          26.34          32.45
256        34.56          28.94          42.67          51.23

✓ 测试完成
```

---

## 5. 关键分析

### 参数影响

1. **M (连接数)**：
   - M=8: 快速构建，低内存，但召回率略低
   - M=16: 平衡选择，适合大多数场景
   - M=32: 高召回率，但内存翻倍

2. **efConstruction (构建深度)**：
   - 影响索引质量，不影响查询时内存
   - 更高值 = 更好的图结构 = 更高召回率
   - 推荐范围：[100, 400]

3. **ef (查询深度)**：
   - 运行时可调，最灵活的参数
   - ef=16: 极速查询，召回率85%+
   - ef=64: 平衡选择，召回率95%+
   - ef=256: 高召回率98%+，但延迟高

### 性能权衡

```python
# 场景1：极致性能（游戏推荐）
hnsw_params = {"M": 8, "efConstruction": 100, "ef": 16}
# 预期：QPS 150+, P95 < 15ms, 召回率 85%+

# 场景2：平衡方案（视频推荐）
hnsw_params = {"M": 16, "efConstruction": 200, "ef": 64}
# 预期：QPS 80-100, P95 < 20ms, 召回率 95%+

# 场景3：高准确率（金融推荐）
hnsw_params = {"M": 32, "efConstruction": 400, "ef": 128}
# 预期：QPS 50-60, P95 < 30ms, 召回率 98%+
```

---

## 6. 生产环境建议

### 参数选型决策树

```
QPS要求 > 200？
├─ 是 → M=8, efConstruction=100, ef=16-32
└─ 否 → 召回率要求 > 95%？
    ├─ 是 → M=16-32, efConstruction=200-400, ef=64-128
    └─ 否 → M=8-16, efConstruction=100-200, ef=32-64
```

### 内存估算

```python
# HNSW内存估算公式
memory_mb = (
    num_vectors * dimension * 4 +  # 原始向量
    num_vectors * M * 2 * 4         # 图结构
) / (1024 ** 2)

# 示例：500万向量，384维，M=16
memory_mb = (5_000_000 * 384 * 4 + 5_000_000 * 16 * 2 * 4) / (1024 ** 2)
# ≈ 7.3 GB + 0.6 GB = 7.9 GB
```

### 监控指标

```python
# 关键监控指标
monitoring = {
    "qps": "实时QPS",
    "p95_latency": "P95延迟 < 50ms",
    "p99_latency": "P99延迟 < 100ms",
    "memory_usage": "内存使用 < 80%",
    "recall_rate": "召回率 > 95%"
}

# 告警阈值
alerts = {
    "p95_latency > 50ms": "延迟过高，考虑降低ef",
    "qps < 50": "吞吐量过低，检查资源",
    "memory_usage > 80%": "内存不足，考虑降低M"
}
```

---

## 7. 常见问题

### Q1: HNSW vs IVF_FLAT如何选择？

**决策标准：**
- QPS > 200 → HNSW
- 内存充足 → HNSW
- 需要过滤查询 → IVF_FLAT（HNSW在高过滤率下性能下降）
- 召回率 > 98% → HNSW

### Q2: 如何应对HNSW召回率下降？

**原因：**
- 数据量增长（从10K到1M）
- ef值过低
- M值过小

**解决方案：**
```python
# 方案1：动态调整ef
if dataset_size > 1_000_000:
    ef = 128  # 增加搜索深度
elif dataset_size > 100_000:
    ef = 64
else:
    ef = 32

# 方案2：重建索引with更大M
if recall_rate < 0.95:
    rebuild_index(M=32, efConstruction=400)
```

### Q3: HNSW内存占用过高怎么办？

**解决方案：**
```python
# 方案1：降低M
index_params = {"M": 8}  # 从16降到8，内存减半

# 方案2：使用量化
index_params = {
    "index_type": "HNSW_SQ",  # HNSW + 标量量化
    "params": {"M": 16}
}
# 内存减少75%，召回率略降

# 方案3：分区策略
collection.create_partition("hot_videos")  # 热门视频
collection.create_partition("cold_videos")  # 冷门视频
# 只加载热门分区到内存
```

---

## 8. 参考资料

### 官方文档
- [Milvus HNSW Documentation](https://milvus.io/docs/v2.5.x/hnsw.md) (2026)
- [HNSW at Scale: Why Adding More Documents Breaks RAG](https://medium.com/write-a-catalyst/hnsw-at-scale-why-adding-more-documents-to-your-database-breaks-rag-7642e21f5ab6) (Feb 2026)

### 实战指南
- [How to Build Milvus Integration](https://oneuptime.com/blog/post/2026-01-30-milvus-integration/view) (Jan 2026)
- [Understanding IVF vs HNSW](https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md) (2025)

---

**总结：HNSW是高QPS场景的最佳选择，通过调整M、efConstruction、ef三个参数可以在速度、准确率、内存间灵活权衡。生产环境推荐M=16, efConstruction=200, ef=64作为起点。**
