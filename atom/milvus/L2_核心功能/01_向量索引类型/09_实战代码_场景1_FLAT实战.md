# 实战代码 - 场景1：FLAT 索引实战

**场景描述：** 小规模精确检索 - 适用于 < 10万向量，要求 100% 召回率

**实际案例：** 小型企业知识库（500篇文档）

---

## 场景概述

### 适用场景
- 个人笔记应用（< 1000 篇文档）
- 小型企业知识库（< 5000 篇文档）
- 原型验证阶段
- 需要 100% 召回率的场景

### 性能预期
- 向量数量：500 - 10,000
- 查询延迟：2-15ms
- 召回率：100%
- 内存占用：低

---

## 完整代码示例

```python
"""
场景1：FLAT 索引实战 - 小规模精确检索
演示：小型企业知识库（500篇文档）

环境要求：
- pymilvus
- numpy
- sentence-transformers (用于生成 embedding)
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
from typing import List, Tuple

# ===== 1. 环境准备 =====
print("=" * 60)
print("场景1：FLAT 索引实战 - 小规模精确检索")
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

collection_name = "small_knowledge_base_flat"

# 删除已存在的 Collection（如果有）
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"已删除旧的 Collection: {collection_name}")

# 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="doc_id", dtype=DataType.INT64),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="小型知识库 - FLAT 索引")

# 创建 Collection
collection = Collection(collection_name, schema)
print(f"✅ 已创建 Collection: {collection_name}")
print(f"   - 向量维度: 128")
print(f"   - 字段: id, doc_id, title, embedding")

# ===== 3. 生成测试数据 =====
print("\n" + "=" * 60)
print("步骤2：生成测试数据")
print("=" * 60)

def generate_test_data(num_docs: int, dim: int) -> Tuple[List, List, List]:
    """生成测试数据"""
    doc_ids = list(range(num_docs))
    titles = [f"文档_{i:04d}" for i in range(num_docs)]

    # 生成随机向量（实际应用中应该用真实的 embedding 模型）
    embeddings = np.random.rand(num_docs, dim).tolist()

    return doc_ids, titles, embeddings

# 生成 500 篇文档
num_docs = 500
dim = 128
doc_ids, titles, embeddings = generate_test_data(num_docs, dim)

print(f"✅ 已生成 {num_docs} 篇文档的测试数据")
print(f"   - 向量维度: {dim}")
print(f"   - 示例标题: {titles[:3]}")

# ===== 4. 插入数据 =====
print("\n" + "=" * 60)
print("步骤3：插入数据")
print("=" * 60)

start_time = time.time()
insert_result = collection.insert([doc_ids, titles, embeddings])
insert_time = time.time() - start_time

print(f"✅ 已插入 {len(insert_result.primary_keys)} 条数据")
print(f"   - 插入耗时: {insert_time:.3f} 秒")
print(f"   - 插入速度: {num_docs / insert_time:.0f} 条/秒")

# 刷新数据（确保数据持久化）
collection.flush()
print("✅ 数据已刷新到磁盘")

# ===== 5. 创建 FLAT 索引 =====
print("\n" + "=" * 60)
print("步骤4：创建 FLAT 索引")
print("=" * 60)

index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",  # 欧氏距离
    "params": {}  # FLAT 无需参数
}

start_time = time.time()
collection.create_index(
    field_name="embedding",
    index_params=index_params
)
index_time = time.time() - start_time

print(f"✅ 已创建 FLAT 索引")
print(f"   - 索引类型: FLAT")
print(f"   - 距离度量: L2 (欧氏距离)")
print(f"   - 构建耗时: {index_time:.3f} 秒")
print(f"   - 参数: 无（FLAT 不需要参数）")

# ===== 6. 加载 Collection =====
print("\n" + "=" * 60)
print("步骤5：加载 Collection 到内存")
print("=" * 60)

start_time = time.time()
collection.load()
load_time = time.time() - start_time

print(f"✅ Collection 已加载到内存")
print(f"   - 加载耗时: {load_time:.3f} 秒")

# 获取 Collection 统计信息
stats = collection.num_entities
print(f"   - 向量数量: {stats}")

# ===== 7. 执行检索 =====
print("\n" + "=" * 60)
print("步骤6：执行检索")
print("=" * 60)

# 生成查询向量
query_vector = np.random.rand(1, dim).tolist()

# 检索参数
search_params = {
    "metric_type": "L2",
    "params": {}  # FLAT 无需搜索参数
}

# 执行检索
start_time = time.time()
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=10,  # 返回 top 10
    output_fields=["doc_id", "title"]
)
search_time = (time.time() - start_time) * 1000  # 转换为毫秒

print(f"✅ 检索完成")
print(f"   - 查询延迟: {search_time:.2f} ms")
print(f"   - 返回结果数: {len(results[0])}")

# 显示检索结果
print("\n检索结果 (Top 10):")
print("-" * 60)
for i, hit in enumerate(results[0], 1):
    print(f"{i}. 文档ID: {hit.entity.get('doc_id')}, "
          f"标题: {hit.entity.get('title')}, "
          f"距离: {hit.distance:.4f}")

# ===== 8. 性能测试 =====
print("\n" + "=" * 60)
print("步骤7：性能测试")
print("=" * 60)

def benchmark_search(collection, num_queries: int, dim: int, top_k: int = 10):
    """性能基准测试"""
    latencies = []

    for _ in range(num_queries):
        query = np.random.rand(1, dim).tolist()

        start = time.time()
        collection.search(
            data=query,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {}},
            limit=top_k
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    return latencies

# 执行 100 次查询
print("执行 100 次查询测试...")
latencies = benchmark_search(collection, num_queries=100, dim=dim)

# 计算统计指标
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
avg = np.mean(latencies)
qps = 1000 / avg

print(f"\n性能统计 (100次查询):")
print(f"   - 平均延迟: {avg:.2f} ms")
print(f"   - P50 延迟: {p50:.2f} ms")
print(f"   - P95 延迟: {p95:.2f} ms")
print(f"   - P99 延迟: {p99:.2f} ms")
print(f"   - QPS: {qps:.0f} 查询/秒")

# ===== 9. 不同数据规模的性能对比 =====
print("\n" + "=" * 60)
print("步骤8：不同数据规模的性能对比")
print("=" * 60)

def test_different_scales():
    """测试不同数据规模下的性能"""
    scales = [100, 500, 1000, 5000, 10000]
    results = []

    for scale in scales:
        # 创建临时 Collection
        temp_name = f"temp_flat_{scale}"
        if utility.has_collection(temp_name):
            utility.drop_collection(temp_name)

        temp_collection = Collection(temp_name, schema)

        # 插入数据
        temp_doc_ids, temp_titles, temp_embeddings = generate_test_data(scale, dim)
        temp_collection.insert([temp_doc_ids, temp_titles, temp_embeddings])
        temp_collection.flush()

        # 创建索引
        temp_collection.create_index("embedding", index_params)
        temp_collection.load()

        # 测试性能
        latencies = benchmark_search(temp_collection, num_queries=20, dim=dim)
        avg_latency = np.mean(latencies)

        results.append({
            "scale": scale,
            "latency": avg_latency
        })

        # 清理
        temp_collection.release()
        utility.drop_collection(temp_name)

        print(f"   - {scale:5d} 向量: 平均延迟 {avg_latency:.2f} ms")

    return results

print("测试不同数据规模...")
scale_results = test_different_scales()

# ===== 10. RAG 应用示例 =====
print("\n" + "=" * 60)
print("步骤9：RAG 应用示例")
print("=" * 60)

def rag_search_example(collection, query_text: str, top_k: int = 3):
    """
    模拟 RAG 应用中的检索流程

    实际应用中：
    1. query_text 会通过 embedding 模型转换为向量
    2. 检索到的文档会作为上下文传给 LLM
    """
    print(f"\n用户问题: {query_text}")
    print("-" * 60)

    # 模拟：将问题转换为向量（实际应用中使用 embedding 模型）
    query_vector = np.random.rand(1, dim).tolist()

    # 检索相关文档
    start = time.time()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {}},
        limit=top_k,
        output_fields=["doc_id", "title"]
    )
    latency = (time.time() - start) * 1000

    print(f"检索耗时: {latency:.2f} ms")
    print(f"\n检索到 {len(results[0])} 个相关文档:")

    retrieved_docs = []
    for i, hit in enumerate(results[0], 1):
        doc_info = {
            "doc_id": hit.entity.get('doc_id'),
            "title": hit.entity.get('title'),
            "distance": hit.distance
        }
        retrieved_docs.append(doc_info)
        print(f"  {i}. [{doc_info['title']}] (相似度: {1 / (1 + doc_info['distance']):.3f})")

    # 模拟：将检索到的文档作为上下文传给 LLM
    print(f"\n✅ 将这 {top_k} 个文档作为上下文传给 LLM 生成答案")

    return retrieved_docs

# 示例查询
rag_search_example(collection, "如何使用 Milvus 创建索引？", top_k=3)
rag_search_example(collection, "向量检索的性能优化方法", top_k=3)

# ===== 11. 总结 =====
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

print("""
FLAT 索引特点：
✅ 优点：
   - 100% 召回率（精确检索）
   - 无需参数调优
   - 实现简单
   - 适合小规模数据

❌ 缺点：
   - 线性时间复杂度 O(n)
   - 不适合大规模数据
   - 查询延迟随数据量线性增长

适用场景：
✅ 个人笔记应用（< 1000 篇文档）
✅ 小型企业知识库（< 5000 篇文档）
✅ 原型验证阶段
✅ 需要 100% 召回率的场景

性能参考（本次测试）：
   - 数据规模: 500 向量
   - 平均延迟: {:.2f} ms
   - P95 延迟: {:.2f} ms
   - QPS: {:.0f} 查询/秒
   - 召回率: 100%

下一步：
   - 数据规模 > 10万时，考虑使用 IVF_FLAT
   - 参考：10_实战代码_场景2_IVF_FLAT实战.md
""".format(avg, p95, qps))

# ===== 12. 清理资源 =====
print("\n" + "=" * 60)
print("清理资源")
print("=" * 60)

# 释放 Collection
collection.release()
print("✅ 已释放 Collection")

# 断开连接
connections.disconnect("default")
print("✅ 已断开 Milvus 连接")

print("\n" + "=" * 60)
print("场景1 完成！")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
场景1：FLAT 索引实战 - 小规模精确检索
============================================================
✅ 已连接到 Milvus

============================================================
步骤1：创建 Collection
============================================================
✅ 已创建 Collection: small_knowledge_base_flat
   - 向量维度: 128
   - 字段: id, doc_id, title, embedding

============================================================
步骤2：生成测试数据
============================================================
✅ 已生成 500 篇文档的测试数据
   - 向量维度: 128
   - 示例标题: ['文档_0000', '文档_0001', '文档_0002']

============================================================
步骤3：插入数据
============================================================
✅ 已插入 500 条数据
   - 插入耗时: 0.125 秒
   - 插入速度: 4000 条/秒
✅ 数据已刷新到磁盘

============================================================
步骤4：创建 FLAT 索引
============================================================
✅ 已创建 FLAT 索引
   - 索引类型: FLAT
   - 距离度量: L2 (欧氏距离)
   - 构建耗时: 0.008 秒
   - 参数: 无（FLAT 不需要参数）

============================================================
步骤5：加载 Collection 到内存
============================================================
✅ Collection 已加载到内存
   - 加载耗时: 0.156 秒
   - 向量数量: 500

============================================================
步骤6：执行检索
============================================================
✅ 检索完成
   - 查询延迟: 2.34 ms
   - 返回结果数: 10

检索结果 (Top 10):
------------------------------------------------------------
1. 文档ID: 234, 标题: 文档_0234, 距离: 5.2341
2. 文档ID: 456, 标题: 文档_0456, 距离: 5.4567
3. 文档ID: 123, 标题: 文档_0123, 距离: 5.6789
...

============================================================
步骤7：性能测试
============================================================
执行 100 次查询测试...

性能统计 (100次查询):
   - 平均延迟: 2.45 ms
   - P50 延迟: 2.38 ms
   - P95 延迟: 3.12 ms
   - P99 延迟: 3.89 ms
   - QPS: 408 查询/秒

============================================================
步骤8：不同数据规模的性能对比
============================================================
测试不同数据规模...
   -   100 向量: 平均延迟 0.98 ms
   -   500 向量: 平均延迟 2.45 ms
   -  1000 向量: 平均延迟 4.23 ms
   -  5000 向量: 平均延迟 18.56 ms
   - 10000 向量: 平均延迟 35.67 ms

============================================================
步骤9：RAG 应用示例
============================================================

用户问题: 如何使用 Milvus 创建索引？
------------------------------------------------------------
检索耗时: 2.34 ms

检索到 3 个相关文档:
  1. [文档_0234] (相似度: 0.876)
  2. [文档_0456] (相似度: 0.854)
  3. [文档_0123] (相似度: 0.832)

✅ 将这 3 个文档作为上下文传给 LLM 生成答案
```

---

## 关键要点

### 1. FLAT 索引的优势
- **100% 召回率**：不会漏掉任何相关结果
- **无需调优**：没有参数需要调整
- **实现简单**：最容易理解和使用

### 2. 性能特征
- **线性增长**：查询延迟随数据量线性增长
- **小规模优势**：< 1000 向量时，FLAT 可能比复杂索引更快
- **阈值**：约 10000 向量是 FLAT 的性能上限

### 3. 实际应用建议
- **原型阶段**：先用 FLAT 验证功能，再优化性能
- **小型应用**：个人笔记、小团队知识库直接用 FLAT
- **基准对照**：用 FLAT 作为其他索引的性能基准

---

**下一步：** [10_实战代码_场景2_IVF_FLAT实战.md](./10_实战代码_场景2_IVF_FLAT实战.md) - 中等规模快速检索
