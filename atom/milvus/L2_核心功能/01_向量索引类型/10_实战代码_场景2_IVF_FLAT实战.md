# 实战代码 - 场景2：IVF_FLAT 索引实战

**场景描述：** 中等规模快速检索 - 适用于 10万-100万向量，平衡精度和速度

**实际案例：** 中型企业知识库（5万篇文档）

---

## 场景概述

### 适用场景
- 中型企业知识库（1万-10万篇文档）
- 电商商品推荐（10万-100万商品）
- 新闻推荐系统（历史新闻库）
- 需要平衡召回率和速度的场景

### 性能预期
- 向量数量：10,000 - 1,000,000
- 查询延迟：20-50ms
- 召回率：95-98%（可调）
- 内存占用：中等

---

## 完整代码示例

```python
"""
场景2：IVF_FLAT 索引实战 - 中等规模快速检索
演示：中型企业知识库（5万篇文档）

环境要求：
- pymilvus
- numpy
"""

import time
import numpy as np
import math
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from typing import List, Tuple, Dict

# ===== 1. 环境准备 =====
print("=" * 60)
print("场景2：IVF_FLAT 索引实战 - 中等规模快速检索")
print("=" * 60)

# 连接到 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
print("✅ 已连接到 Milvus")

# ===== 2. 创建 Collection =====
print("\n" + "=" * 60)
print("步骤1：创建 Collection")
print("=" * 60)

collection_name = "medium_knowledge_base_ivf"

# 删除已存在的 Collection（如果有）
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"已删除旧的 Collection: {collection_name}")

# 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="doc_id", dtype=DataType.INT64),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="中型知识库 - IVF_FLAT 索引")

# 创建 Collection
collection = Collection(collection_name, schema)
print(f"✅ 已创建 Collection: {collection_name}")
print(f"   - 向量维度: 768")
print(f"   - 字段: id, doc_id, category, title, embedding")

# ===== 3. 生成测试数据 =====
print("\n" + "=" * 60)
print("步骤2：生成测试数据")
print("=" * 60)

def generate_test_data(num_docs: int, dim: int) -> Tuple[List, List, List, List]:
    """生成测试数据"""
    categories = ["技术", "产品", "运营", "市场", "财务"]

    doc_ids = list(range(num_docs))
    category_list = [categories[i % len(categories)] for i in range(num_docs)]
    titles = [f"{category_list[i]}_文档_{i:05d}" for i in range(num_docs)]

    # 生成随机向量（实际应用中应该用真实的 embedding 模型）
    embeddings = np.random.rand(num_docs, dim).tolist()

    return doc_ids, category_list, titles, embeddings

# 生成 50,000 篇文档
num_docs = 50000
dim = 768
print(f"正在生成 {num_docs} 篇文档的测试数据...")
doc_ids, categories, titles, embeddings = generate_test_data(num_docs, dim)

print(f"✅ 已生成 {num_docs} 篇文档的测试数据")
print(f"   - 向量维度: {dim}")
print(f"   - 类别: {set(categories)}")
print(f"   - 示例标题: {titles[:3]}")

# ===== 4. 批量插入数据 =====
print("\n" + "=" * 60)
print("步骤3：批量插入数据")
print("=" * 60)

# 分批插入（每批 5000 条）
batch_size = 5000
num_batches = (num_docs + batch_size - 1) // batch_size

print(f"分 {num_batches} 批插入，每批 {batch_size} 条")

start_time = time.time()
total_inserted = 0

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_docs)

    batch_doc_ids = doc_ids[start_idx:end_idx]
    batch_categories = categories[start_idx:end_idx]
    batch_titles = titles[start_idx:end_idx]
    batch_embeddings = embeddings[start_idx:end_idx]

    insert_result = collection.insert([
        batch_doc_ids,
        batch_categories,
        batch_titles,
        batch_embeddings
    ])

    total_inserted += len(insert_result.primary_keys)
    print(f"   批次 {i+1}/{num_batches}: 已插入 {len(insert_result.primary_keys)} 条")

insert_time = time.time() - start_time

print(f"\n✅ 已插入 {total_inserted} 条数据")
print(f"   - 插入耗时: {insert_time:.2f} 秒")
print(f"   - 插入速度: {total_inserted / insert_time:.0f} 条/秒")

# 刷新数据
collection.flush()
print("✅ 数据已刷新到磁盘")

# ===== 5. 计算最优 nlist =====
print("\n" + "=" * 60)
print("步骤4：计算最优 nlist")
print("=" * 60)

def calculate_optimal_nlist(num_vectors: int) -> Dict:
    """计算最优 nlist"""
    sqrt_n = math.sqrt(num_vectors)

    min_nlist = int(2 * sqrt_n)
    max_nlist = int(4 * sqrt_n)
    recommended = int(3 * sqrt_n)

    # 调整到 2 的幂次（便于内存对齐）
    recommended_power2 = 2 ** int(math.log2(recommended))

    return {
        "min": min_nlist,
        "max": max_nlist,
        "recommended": recommended,
        "recommended_power2": recommended_power2
    }

nlist_options = calculate_optimal_nlist(num_docs)

print(f"向量数量: {num_docs}")
print(f"   - 最小 nlist: {nlist_options['min']}")
print(f"   - 最大 nlist: {nlist_options['max']}")
print(f"   - 推荐 nlist: {nlist_options['recommended']}")
print(f"   - 推荐 nlist (2的幂): {nlist_options['recommended_power2']}")

# 使用推荐值
nlist = nlist_options['recommended_power2']
print(f"\n✅ 选择 nlist = {nlist}")

# ===== 6. 创建 IVF_FLAT 索引 =====
print("\n" + "=" * 60)
print("步骤5：创建 IVF_FLAT 索引")
print("=" * 60)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": nlist}
}

print(f"索引参数:")
print(f"   - index_type: IVF_FLAT")
print(f"   - metric_type: L2")
print(f"   - nlist: {nlist}")

start_time = time.time()
collection.create_index(
    field_name="embedding",
    index_params=index_params
)
index_time = time.time() - start_time

print(f"\n✅ 已创建 IVF_FLAT 索引")
print(f"   - 构建耗时: {index_time:.2f} 秒")
print(f"   - 说明: IVF_FLAT 需要训练 K-means 聚类，耗时较长")

# ===== 7. 加载 Collection =====
print("\n" + "=" * 60)
print("步骤6：加载 Collection 到内存")
print("=" * 60)

start_time = time.time()
collection.load()
load_time = time.time() - start_time

print(f"✅ Collection 已加载到内存")
print(f"   - 加载耗时: {load_time:.2f} 秒")
print(f"   - 向量数量: {collection.num_entities}")

# ===== 8. 参数调优实验 =====
print("\n" + "=" * 60)
print("步骤7：参数调优实验 - 测试不同 nprobe")
print("=" * 60)

def measure_performance(collection, query_vectors, nprobe: int, num_queries: int = 20):
    """测量性能指标"""
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": nprobe}
    }

    latencies = []
    for query in query_vectors[:num_queries]:
        start = time.time()
        collection.search(
            data=[query],
            anns_field="embedding",
            param=search_params,
            limit=10
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.percentile(latencies, 50),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99)
    }

# 生成测试查询
test_queries = np.random.rand(50, dim).tolist()

# 测试不同 nprobe 值
nprobe_values = [8, 16, 32, 64, 128]
results = {}

print(f"测试不同 nprobe 值（每个测试 20 次查询）...")
print("-" * 60)

for nprobe in nprobe_values:
    metrics = measure_performance(collection, test_queries, nprobe)
    results[nprobe] = metrics

    print(f"nprobe={nprobe:3d}: "
          f"平均延迟={metrics['avg_latency']:5.1f}ms, "
          f"P95={metrics['p95_latency']:5.1f}ms")

# ===== 9. 召回率测试 =====
print("\n" + "=" * 60)
print("步骤8：召回率测试")
print("=" * 60)

def calculate_recall(collection, query_vectors, nprobe: int, num_queries: int = 10):
    """
    计算召回率
    使用 FLAT 索引的结果作为 ground truth
    """
    # 创建临时 FLAT 索引作为 ground truth
    temp_name = "temp_flat_gt"
    if utility.has_collection(temp_name):
        utility.drop_collection(temp_name)

    # 复制 schema
    temp_collection = Collection(temp_name, schema)

    # 插入相同数据（只插入少量用于测试）
    sample_size = 10000
    temp_collection.insert([
        doc_ids[:sample_size],
        categories[:sample_size],
        titles[:sample_size],
        embeddings[:sample_size]
    ])
    temp_collection.flush()

    # 创建 FLAT 索引
    temp_collection.create_index("embedding", {
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {}
    })
    temp_collection.load()

    # 计算召回率
    recalls = []
    for query in query_vectors[:num_queries]:
        # Ground truth (FLAT)
        gt_results = temp_collection.search(
            data=[query],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {}},
            limit=10
        )
        gt_ids = set([hit.id for hit in gt_results[0]])

        # IVF_FLAT 结果
        ivf_results = temp_collection.search(
            data=[query],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": nprobe}},
            limit=10
        )
        ivf_ids = set([hit.id for hit in ivf_results[0]])

        # 计算召回率
        recall = len(gt_ids & ivf_ids) / len(gt_ids)
        recalls.append(recall)

    # 清理
    temp_collection.release()
    utility.drop_collection(temp_name)

    return np.mean(recalls)

print("计算不同 nprobe 的召回率（基于 10,000 个向量的子集）...")
print("-" * 60)

recall_results = {}
for nprobe in [8, 16, 32, 64]:
    recall = calculate_recall(collection, test_queries, nprobe, num_queries=10)
    recall_results[nprobe] = recall
    print(f"nprobe={nprobe:3d}: 召回率={recall:.2%}")

# ===== 10. 选择最优 nprobe =====
print("\n" + "=" * 60)
print("步骤9：选择最优 nprobe")
print("=" * 60)

def choose_optimal_nprobe(results, recall_results, target_recall=0.95, max_latency=50):
    """选择满足要求的最小 nprobe"""
    print(f"目标要求:")
    print(f"   - 召回率 >= {target_recall:.0%}")
    print(f"   - P95 延迟 <= {max_latency}ms")
    print()

    for nprobe in sorted(results.keys()):
        latency = results[nprobe]['p95_latency']
        recall = recall_results.get(nprobe, 0)

        meets_recall = recall >= target_recall
        meets_latency = latency <= max_latency

        status = "✅" if (meets_recall and meets_latency) else "❌"

        print(f"{status} nprobe={nprobe:3d}: "
              f"召回率={recall:.2%} {'✅' if meets_recall else '❌'}, "
              f"P95延迟={latency:.1f}ms {'✅' if meets_latency else '❌'}")

        if meets_recall and meets_latency:
            return nprobe

    # 如果都不满足，返回召回率最高的
    return max(recall_results.keys(), key=lambda k: recall_results[k])

optimal_nprobe = choose_optimal_nprobe(results, recall_results, target_recall=0.95, max_latency=50)

print(f"\n✅ 最优 nprobe = {optimal_nprobe}")
print(f"   - 召回率: {recall_results[optimal_nprobe]:.2%}")
print(f"   - P95 延迟: {results[optimal_nprobe]['p95_latency']:.1f}ms")

# ===== 11. 性能对比：IVF_FLAT vs FLAT =====
print("\n" + "=" * 60)
print("步骤10：性能对比 - IVF_FLAT vs FLAT")
print("=" * 60)

# 创建 FLAT 索引进行对比
flat_name = "temp_flat_comparison"
if utility.has_collection(flat_name):
    utility.drop_collection(flat_name)

flat_collection = Collection(flat_name, schema)

# 插入相同数据（使用子集）
sample_size = 10000
flat_collection.insert([
    doc_ids[:sample_size],
    categories[:sample_size],
    titles[:sample_size],
    embeddings[:sample_size]
])
flat_collection.flush()

# 创建 FLAT 索引
flat_collection.create_index("embedding", {
    "index_type": "FLAT",
    "metric_type": "L2",
    "params": {}
})
flat_collection.load()

# 测试 FLAT 性能
flat_metrics = measure_performance(flat_collection, test_queries, nprobe=0, num_queries=20)

# 清理
flat_collection.release()
utility.drop_collection(flat_name)

# 对比结果
print(f"性能对比（基于 {sample_size} 个向量）:")
print("-" * 60)
print(f"{'索引类型':<15} {'平均延迟':<12} {'P95延迟':<12} {'召回率':<10}")
print("-" * 60)
print(f"{'FLAT':<15} {flat_metrics['avg_latency']:>8.1f}ms {flat_metrics['p95_latency']:>8.1f}ms {'100.0%':>10}")
print(f"{'IVF_FLAT':<15} {results[optimal_nprobe]['avg_latency']:>8.1f}ms {results[optimal_nprobe]['p95_latency']:>8.1f}ms {recall_results[optimal_nprobe]:>9.1%}")

speedup = flat_metrics['avg_latency'] / results[optimal_nprobe]['avg_latency']
print(f"\n✅ IVF_FLAT 比 FLAT 快 {speedup:.1f}x")

# ===== 12. RAG 应用示例 =====
print("\n" + "=" * 60)
print("步骤11：RAG 应用示例")
print("=" * 60)

def rag_search_with_filter(collection, query_text: str, category: str = None, top_k: int = 5):
    """
    RAG 应用中的检索（支持类别过滤）
    """
    print(f"\n用户问题: {query_text}")
    if category:
        print(f"过滤条件: 类别={category}")
    print("-" * 60)

    # 模拟：将问题转换为向量
    query_vector = np.random.rand(1, dim).tolist()

    # 构建过滤表达式
    expr = f'category == "{category}"' if category else None

    # 检索
    start = time.time()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": optimal_nprobe}},
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "category", "title"]
    )
    latency = (time.time() - start) * 1000

    print(f"检索耗时: {latency:.2f} ms")
    print(f"\n检索到 {len(results[0])} 个相关文档:")

    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. [{hit.entity.get('category')}] {hit.entity.get('title')} "
              f"(距离: {hit.distance:.4f})")

    return results

# 示例查询
rag_search_with_filter(collection, "如何优化系统性能？", category=None, top_k=5)
rag_search_with_filter(collection, "产品定价策略", category="产品", top_k=3)
rag_search_with_filter(collection, "市场推广方案", category="市场", top_k=3)

# ===== 13. 高级技巧：动态 nprobe =====
print("\n" + "=" * 60)
print("步骤12：高级技巧 - 动态 nprobe")
print("=" * 60)

def adaptive_search(collection, query, query_type: str = "balanced"):
    """
    根据查询类型动态调整 nprobe
    """
    nprobe_map = {
        "fast": 8,      # 快速查询，低延迟
        "balanced": 32,  # 平衡查询
        "accurate": 64   # 精确查询，高召回率
    }

    nprobe = nprobe_map.get(query_type, 32)

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": nprobe}
    }

    start = time.time()
    results = collection.search(
        data=[query],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["title"]
    )
    latency = (time.time() - start) * 1000

    return results, latency, nprobe

print("动态 nprobe 示例:")
print("-" * 60)

query = np.random.rand(dim).tolist()

for query_type in ["fast", "balanced", "accurate"]:
    results, latency, nprobe = adaptive_search(collection, query, query_type)
    print(f"{query_type:>10}: nprobe={nprobe:2d}, 延迟={latency:5.1f}ms")

# ===== 14. 总结 =====
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print(f"""
IVF_FLAT 索引特点：
✅ 优点：
   - 平衡精度和速度
   - 召回率可调（95-98%）
   - 适合中等规模数据
   - 内存占用适中

❌ 缺点：
   - 需要训练（K-means 聚类）
   - 参数调优较复杂
   - 不支持增量插入（需重建索引）

适用场景：
✅ 中型企业知识库（1万-10万篇文档）
✅ 电商商品推荐（10万-100万商品）
✅ 需要平衡召回率和速度的场景

性能参考（本次测试）：
   - 数据规模: {num_docs} 向量
   - 最优 nlist: {nlist}
   - 最优 nprobe: {optimal_nprobe}
   - 平均延迟: {results[optimal_nprobe]['avg_latency']:.1f} ms
   - P95 延迟: {results[optimal_nprobe]['p95_latency']:.1f} ms
   - 召回率: {recall_results[optimal_nprobe]:.1%}

参数调优建议：
   - nlist = 3 * sqrt(n) ≈ {nlist}
   - nprobe 从 16 开始，根据召回率要求调整
   - 召回率 > 95% 时，nprobe ≥ 32
   - 延迟敏感时，nprobe ≤ 16

下一步：
   - 数据规模 > 100万时，考虑使用 HNSW
   - 参考：11_实战代码_场景3_HNSW实战.md
""")

# ===== 15. 清理资源 =====
print("\n" + "=" * 60)
print("清理资源")
print("=" * 60)

collection.release()
print("✅ 已释放 Collection")

connections.disconnect("default")
print("✅ 已断开 Milvus 连接")

print("\n" + "=" * 60)
print("场景2 完成！")
print("=" * 60)
```

---

## 关键要点

### 1. nlist 计算公式
```python
nlist = 3 * sqrt(num_vectors)
```
- 50,000 向量 → nlist ≈ 671 → 调整到 512 或 1024
- 100,000 向量 → nlist ≈ 949 → 调整到 1024
- 500,000 向量 → nlist ≈ 2121 → 调整到 2048

### 2. nprobe 调优策略
- **起始值**: 16
- **高召回率** (> 95%): nprobe = 32-64
- **低延迟** (< 30ms): nprobe = 8-16
- **平衡**: nprobe = 16-32

### 3. 召回率 vs 延迟权衡
| nprobe | 召回率 | 延迟 | 适用场景 |
|--------|--------|------|---------|
| 8 | 91-93% | 低 | 快速查询 |
| 16 | 94-95% | 中低 | 平衡场景 |
| 32 | 96-97% | 中 | 高召回率 |
| 64 | 97-98% | 中高 | 精确查询 |

### 4. 与 FLAT 的对比
- **速度**: IVF_FLAT 快 3-10 倍
- **召回率**: FLAT 100%, IVF_FLAT 95-98%
- **适用规模**: FLAT < 10万, IVF_FLAT 10万-100万

---

**下一步：** [11_实战代码_场景3_HNSW实战.md](./11_实战代码_场景3_HNSW实战.md) - 大规模高性能检索
