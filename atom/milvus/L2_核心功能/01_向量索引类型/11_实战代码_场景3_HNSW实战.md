# 实战代码 - 场景3：HNSW 索引实战

**场景描述：** 大规模高性能检索 - 适用于 > 100万向量，高 QPS 要求

**实际案例：** 大型知识库（100万篇文档）

---

## 场景概述

### 适用场景
- 大型知识库（> 100万篇文档）
- 实时搜索引擎（高 QPS 要求）
- 图像/视频检索系统
- 需要低延迟（< 20ms）的场景

### 性能预期
- 向量数量：> 1,000,000
- 查询延迟：5-15ms
- 召回率：90-95%（可调）
- 内存占用：高（1.5-2x 向量大小）

---

## 完整代码示例

```python
"""
场景3：HNSW 索引实战 - 大规模高性能检索
演示：大型知识库（100万篇文档）

环境要求：
- pymilvus
- numpy
"""

import time
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from typing import List, Dict

# ===== 1. 环境准备 =====
print("=" * 60)
print("场景3：HNSW 索引实战 - 大规模高性能检索")
print("=" * 60)

connections.connect(alias="default", host="localhost", port="19530")
print("✅ 已连接到 Milvus")

# ===== 2. 创建 Collection =====
print("\n步骤1：创建 Collection")
print("-" * 60)

collection_name = "large_knowledge_base_hnsw"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="doc_id", dtype=DataType.INT64),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="大型知识库 - HNSW 索引")
collection = Collection(collection_name, schema)

print(f"✅ 已创建 Collection: {collection_name}")

# ===== 3. 生成并插入测试数据 =====
print("\n步骤2：生成并插入测试数据")
print("-" * 60)

num_docs = 100000  # 使用10万作为演示（实际可扩展到100万+）
dim = 768
batch_size = 10000

print(f"插入 {num_docs} 个向量（分批插入）...")

start_time = time.time()
for i in range(0, num_docs, batch_size):
    batch_doc_ids = list(range(i, min(i + batch_size, num_docs)))
    batch_titles = [f"文档_{j:06d}" for j in batch_doc_ids]
    batch_embeddings = np.random.rand(len(batch_doc_ids), dim).tolist()

    collection.insert([batch_doc_ids, batch_titles, batch_embeddings])
    print(f"  已插入 {i + len(batch_doc_ids)}/{num_docs}")

insert_time = time.time() - start_time
collection.flush()

print(f"✅ 插入完成: {insert_time:.2f}秒, {num_docs/insert_time:.0f}条/秒")

# ===== 4. 创建 HNSW 索引 =====
print("\n步骤3：创建 HNSW 索引")
print("-" * 60)

index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,              # 每层最大连接数
        "efConstruction": 200  # 构建时搜索范围
    }
}

print(f"索引参数: M={index_params['params']['M']}, "
      f"efConstruction={index_params['params']['efConstruction']}")

start_time = time.time()
collection.create_index(field_name="embedding", index_params=index_params)
index_time = time.time() - start_time

print(f"✅ HNSW 索引创建完成: {index_time:.2f}秒")
print(f"   说明: HNSW 无需训练，构建速度快")

# ===== 5. 加载 Collection =====
print("\n步骤4：加载 Collection")
print("-" * 60)

start_time = time.time()
collection.load()
load_time = time.time() - start_time

print(f"✅ 加载完成: {load_time:.2f}秒")
print(f"   向量数量: {collection.num_entities}")

# ===== 6. 参数调优实验 =====
print("\n步骤5：参数调优 - 测试不同 ef 值")
print("-" * 60)

def benchmark_hnsw(collection, ef_values: List[int], num_queries: int = 50):
    """测试不同 ef 值的性能"""
    results = {}
    test_queries = np.random.rand(num_queries, dim).tolist()

    for ef in ef_values:
        search_params = {"metric_type": "L2", "params": {"ef": ef}}
        latencies = []

        for query in test_queries:
            start = time.time()
            collection.search(
                data=[query],
                anns_field="embedding",
                param=search_params,
                limit=10
            )
            latencies.append((time.time() - start) * 1000)

        results[ef] = {
            "avg": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }

    return results

ef_values = [32, 64, 128, 256]
print(f"测试不同 ef 值（{len(ef_values)}组测试）...")

results = benchmark_hnsw(collection, ef_values)

print("\n性能对比:")
print(f"{'ef':<6} {'平均延迟':<12} {'P50':<10} {'P95':<10} {'P99':<10}")
print("-" * 60)
for ef, metrics in results.items():
    print(f"{ef:<6} {metrics['avg']:>8.2f}ms {metrics['p50']:>8.2f}ms "
          f"{metrics['p95']:>8.2f}ms {metrics['p99']:>8.2f}ms")

# ===== 7. 内存占用分析 =====
print("\n步骤6：内存占用分析")
print("-" * 60)

vector_size_gb = num_docs * dim * 4 / (1024**3)
hnsw_overhead = vector_size_gb * 0.5  # HNSW 约50%开销
total_memory = vector_size_gb + hnsw_overhead

print(f"向量数据: {vector_size_gb:.2f} GB")
print(f"HNSW 开销: {hnsw_overhead:.2f} GB (约50%)")
print(f"总内存: {total_memory:.2f} GB")

# ===== 8. 增量插入测试 =====
print("\n步骤7：增量插入测试")
print("-" * 60)

print("HNSW 支持增量插入，无需重建索引")

new_docs = 1000
new_doc_ids = list(range(num_docs, num_docs + new_docs))
new_titles = [f"新文档_{i:06d}" for i in new_doc_ids]
new_embeddings = np.random.rand(new_docs, dim).tolist()

start_time = time.time()
collection.insert([new_doc_ids, new_titles, new_embeddings])
collection.flush()
insert_time = (time.time() - start_time) * 1000

print(f"✅ 增量插入 {new_docs} 个向量: {insert_time:.2f}ms")
print(f"   无需重建索引，立即可用")

# ===== 9. 高并发测试 =====
print("\n步骤8：高并发模拟")
print("-" * 60)

def simulate_concurrent_queries(collection, num_queries: int, ef: int = 64):
    """模拟并发查询"""
    queries = np.random.rand(num_queries, dim).tolist()
    search_params = {"metric_type": "L2", "params": {"ef": ef}}

    start = time.time()
    for query in queries:
        collection.search(
            data=[query],
            anns_field="embedding",
            param=search_params,
            limit=10
        )
    total_time = time.time() - start

    return {
        "total_time": total_time,
        "qps": num_queries / total_time,
        "avg_latency": (total_time / num_queries) * 1000
    }

num_queries = 100
print(f"模拟 {num_queries} 个连续查询...")

perf = simulate_concurrent_queries(collection, num_queries, ef=64)

print(f"✅ 性能指标:")
print(f"   QPS: {perf['qps']:.0f} 查询/秒")
print(f"   平均延迟: {perf['avg_latency']:.2f}ms")

# ===== 10. RAG 应用示例 =====
print("\n步骤9：RAG 应用示例")
print("-" * 60)

def rag_search_hnsw(collection, query_text: str, ef: int = 64, top_k: int = 5):
    """HNSW 索引的 RAG 检索"""
    print(f"\n用户问题: {query_text}")
    print("-" * 60)

    query_vector = np.random.rand(1, dim).tolist()
    search_params = {"metric_type": "L2", "params": {"ef": ef}}

    start = time.time()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["doc_id", "title"]
    )
    latency = (time.time() - start) * 1000

    print(f"检索耗时: {latency:.2f}ms (ef={ef})")
    print(f"检索结果 (Top {top_k}):")

    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. {hit.entity.get('title')} (距离: {hit.distance:.4f})")

    return results

rag_search_hnsw(collection, "如何优化大规模向量检索？", ef=64, top_k=5)

# ===== 11. 动态 ef 策略 =====
print("\n步骤10：动态 ef 策略")
print("-" * 60)

def adaptive_ef_search(collection, query, latency_budget_ms: float = 20):
    """根据延迟预算动态调整 ef"""
    ef_options = [32, 64, 128, 256]

    for ef in ef_options:
        search_params = {"metric_type": "L2", "params": {"ef": ef}}

        start = time.time()
        results = collection.search(
            data=[query],
            anns_field="embedding",
            param=search_params,
            limit=10
        )
        latency = (time.time() - start) * 1000

        if latency <= latency_budget_ms:
            return results, ef, latency

    # 如果都超时，返回最小 ef
    return results, ef_options[0], latency

query = np.random.rand(dim).tolist()
results, chosen_ef, latency = adaptive_ef_search(collection, query, latency_budget_ms=15)

print(f"延迟预算: 15ms")
print(f"选择 ef={chosen_ef}, 实际延迟={latency:.2f}ms")

# ===== 12. 总结 =====
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print(f"""
HNSW 索引特点：
✅ 优点：
   - 查询速度快（< 15ms）
   - 对数时间复杂度 O(log n)
   - 支持增量插入
   - 无需训练

❌ 缺点：
   - 内存占用高（1.5-2x）
   - 召回率略低（90-95%）
   - 参数调优较复杂

性能参考（本次测试）：
   - 数据规模: {num_docs + new_docs} 向量
   - 推荐参数: M=16, efConstruction=200, ef=64
   - 平均延迟: {results[64]['avg']:.2f}ms
   - P95 延迟: {results[64]['p95']:.2f}ms
   - QPS: {perf['qps']:.0f} 查询/秒
   - 内存占用: {total_memory:.2f}GB

适用场景：
✅ 大型知识库（> 100万文档）
✅ 实时搜索引擎
✅ 需要低延迟（< 20ms）
✅ 需要频繁增量插入

参数建议：
   - M: 16（平衡性能和内存）
   - efConstruction: 200（构建质量）
   - ef: 64-128（根据召回率要求）

下一步：
   - 学习索引选型决策
   - 参考：12_实战代码_场景4_索引选型.md
""")

# ===== 13. 清理 =====
collection.release()
connections.disconnect("default")
print("\n✅ 场景3 完成！")
```

---

## 关键要点

### 1. HNSW 参数选择

**M（每层最大连接数）：**
- 推荐值：16
- 范围：4-64
- 影响：越大召回率越高，但内存占用越大

**efConstruction（构建时搜索范围）：**
- 推荐值：200
- 范围：100-500
- 影响：越大索引质量越高，但构建越慢

**ef（搜索时候选集大小）：**
- 推荐值：64-128
- 范围：32-256
- 影响：越大召回率越高，但查询越慢

### 2. 性能特征

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 查询延迟 | 5-15ms | P95 延迟 |
| QPS | 500-2000 | 单机性能 |
| 召回率 | 90-95% | 可通过 ef 调整 |
| 内存占用 | 1.5-2x | 向量大小的1.5-2倍 |

### 3. 与 IVF_FLAT 对比

| 维度 | HNSW | IVF_FLAT |
|------|------|----------|
| 查询速度 | 快（5-15ms） | 中（30-50ms） |
| 召回率 | 90-95% | 95-98% |
| 内存占用 | 高（1.5-2x） | 低（1.1x） |
| 增量插入 | 支持 | 需重建 |
| 训练 | 无需 | 需要 |

### 4. 实际应用建议

**选择 HNSW 的场景：**
- 数据规模 > 100万
- 查询延迟要求 < 20ms
- 需要频繁增量插入
- 内存充足

**优化技巧：**
- 根据延迟预算动态调整 ef
- 使用 M=16 平衡性能和内存
- 高召回率场景使用 ef=128

---

**下一步：** [12_实战代码_场景4_索引选型.md](./12_实战代码_场景4_索引选型.md) - 综合对比与选型
