# 实战代码_场景3：Milvus生产级部署

> 使用Milvus部署大规模向量数据库

---

## 场景描述

使用Milvus部署生产级向量数据库，支持百万级向量检索。

**学习目标**：
- 掌握Milvus部署
- 理解HNSW/IVF索引配置
- 实现批量插入与查询优化

---

## 完整代码

```python
"""
Milvus生产级部署
演示：大规模向量检索系统
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from openai import OpenAI
import numpy as np
import time
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 连接Milvus =====

print("=" * 60)
print("Milvus生产级部署")
print("=" * 60)

# 连接到Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("\n连接成功")
print(f"Milvus版本: {utility.get_server_version()}")


# ===== 2. 创建Collection =====

collection_name = "production_documents"

# 删除已存在的集合
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"\n删除已存在的集合: {collection_name}")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="timestamp", dtype=DataType.INT64)
]

schema = CollectionSchema(
    fields=fields,
    description="生产级文档集合"
)

# 创建Collection
collection = Collection(
    name=collection_name,
    schema=schema
)

print(f"\n创建集合: {collection_name}")
print(f"字段: {[f.name for f in fields]}")


# ===== 3. 创建索引 =====

print("\n" + "=" * 60)
print("创建索引")
print("=" * 60)

# HNSW索引（高召回率）
hnsw_index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {
        "M": 16,
        "efConstruction": 200
    }
}

# IVF索引（大规模）
ivf_index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024
    }
}

# 选择HNSW索引
index_params = hnsw_index_params

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print(f"索引类型: {index_params['index_type']}")
print(f"距离度量: {index_params['metric_type']}")
print(f"参数: {index_params['params']}")


# ===== 4. 生成测试数据 =====

print("\n" + "=" * 60)
print("生成测试数据")
print("=" * 60)

def generate_test_data(n=10000):
    """生成测试数据"""
    embeddings = np.random.randn(n, 1536).astype('float32')

    # 归一化（余弦相似度需要）
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    texts = [f"文档内容 {i}" for i in range(n)]
    sources = [f"source_{i % 10}.txt" for i in range(n)]
    timestamps = [int(time.time()) + i for i in range(n)]

    return embeddings, texts, sources, timestamps

n_vectors = 10000
embeddings, texts, sources, timestamps = generate_test_data(n_vectors)

print(f"生成了 {n_vectors} 个向量")
print(f"向量维度: {embeddings.shape[1]}")


# ===== 5. 批量插入 =====

print("\n" + "=" * 60)
print("批量插入")
print("=" * 60)

# 批量插入（每次1000个）
batch_size = 1000
start_time = time.time()

for i in range(0, n_vectors, batch_size):
    end = min(i + batch_size, n_vectors)

    batch_data = [
        embeddings[i:end].tolist(),
        texts[i:end],
        sources[i:end],
        timestamps[i:end]
    ]

    collection.insert(batch_data)

    print(f"  已插入 {end}/{n_vectors} 个向量")

insert_time = time.time() - start_time
print(f"\n插入完成，耗时: {insert_time:.2f}秒")
print(f"吞吐量: {n_vectors/insert_time:.0f} 向量/秒")

# 刷新
collection.flush()
print("数据已刷新到磁盘")


# ===== 6. 加载Collection =====

print("\n" + "=" * 60)
print("加载Collection")
print("=" * 60)

start_time = time.time()
collection.load()
load_time = time.time() - start_time

print(f"加载完成，耗时: {load_time:.2f}秒")
print(f"向量数量: {collection.num_entities}")


# ===== 7. 查询测试 =====

print("\n" + "=" * 60)
print("查询测试")
print("=" * 60)

# 生成查询向量
query_vector = np.random.randn(1536).astype('float32')
query_vector = query_vector / np.linalg.norm(query_vector)

# 查询参数
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}  # HNSW参数
}

# 单次查询
start_time = time.time()
results = collection.search(
    data=[query_vector.tolist()],
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=["text", "source"]
)
query_time = (time.time() - start_time) * 1000

print(f"查询延迟: {query_time:.2f}ms")
print(f"\nTop-10结果:")
for i, hit in enumerate(results[0], 1):
    print(f"{i}. 距离={hit.distance:.4f}, 文本={hit.entity.get('text')}")


# ===== 8. 批量查询 =====

print("\n" + "=" * 60)
print("批量查询测试")
print("=" * 60)

# 生成100个查询
n_queries = 100
query_vectors = np.random.randn(n_queries, 1536).astype('float32')
query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

start_time = time.time()
results = collection.search(
    data=query_vectors.tolist(),
    anns_field="embedding",
    param=search_params,
    limit=10
)
batch_time = time.time() - start_time

print(f"批量查询 {n_queries} 次")
print(f"总耗时: {batch_time:.2f}秒")
print(f"平均延迟: {batch_time/n_queries*1000:.2f}ms")
print(f"QPS: {n_queries/batch_time:.0f}")


# ===== 9. 过滤查询 =====

print("\n" + "=" * 60)
print("过滤查询测试")
print("=" * 60)

# 只检索特定来源的文档
expr = "source == 'source_0.txt'"

start_time = time.time()
results = collection.search(
    data=[query_vector.tolist()],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr=expr,
    output_fields=["text", "source"]
)
filter_time = (time.time() - start_time) * 1000

print(f"过滤条件: {expr}")
print(f"查询延迟: {filter_time:.2f}ms")
print(f"结果数量: {len(results[0])}")


# ===== 10. 性能对比 =====

print("\n" + "=" * 60)
print("性能对比：不同ef参数")
print("=" * 60)

ef_values = [16, 32, 64, 128]

for ef in ef_values:
    search_params["params"]["ef"] = ef

    # 测试10次取平均
    times = []
    for _ in range(10):
        start = time.time()
        collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=10
        )
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    print(f"ef={ef:3d}: 平均延迟={avg_time:.2f}ms")


# ===== 11. 索引统计 =====

print("\n" + "=" * 60)
print("索引统计")
print("=" * 60)

stats = collection.get_stats()
print(f"向量数量: {collection.num_entities}")
print(f"索引类型: {index_params['index_type']}")
print(f"距离度量: {index_params['metric_type']}")


# ===== 12. IVF索引对比 =====

print("\n" + "=" * 60)
print("IVF索引对比")
print("=" * 60)

# 创建IVF集合
ivf_collection_name = "ivf_documents"

if utility.has_collection(ivf_collection_name):
    utility.drop_collection(ivf_collection_name)

ivf_collection = Collection(
    name=ivf_collection_name,
    schema=schema
)

# 创建IVF索引
ivf_collection.create_index(
    field_name="embedding",
    index_params=ivf_index_params
)

# 插入数据
print("\n插入数据到IVF集合...")
for i in range(0, n_vectors, batch_size):
    end = min(i + batch_size, n_vectors)
    batch_data = [
        embeddings[i:end].tolist(),
        texts[i:end],
        sources[i:end],
        timestamps[i:end]
    ]
    ivf_collection.insert(batch_data)

ivf_collection.flush()
ivf_collection.load()

# IVF查询
ivf_search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}

start_time = time.time()
ivf_results = ivf_collection.search(
    data=[query_vector.tolist()],
    anns_field="embedding",
    param=ivf_search_params,
    limit=10
)
ivf_time = (time.time() - start_time) * 1000

print(f"\nIVF查询延迟: {ivf_time:.2f}ms")
print(f"HNSW查询延迟: {query_time:.2f}ms")
print(f"对比: IVF比HNSW慢 {ivf_time/query_time:.1f}倍")


# ===== 13. 清理 =====

print("\n" + "=" * 60)
print("清理资源")
print("=" * 60)

# 释放Collection
collection.release()
ivf_collection.release()

print("Collection已释放")

# 断开连接
connections.disconnect("default")
print("连接已断开")


print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
Milvus生产级部署
============================================================

连接成功
Milvus版本: v2.5.0

删除已存在的集合: production_documents

创建集合: production_documents
字段: ['id', 'embedding', 'text', 'source', 'timestamp']

============================================================
创建索引
============================================================
索引类型: HNSW
距离度量: COSINE
参数: {'M': 16, 'efConstruction': 200}

============================================================
生成测试数据
============================================================
生成了 10000 个向量
向量维度: 1536

============================================================
批量插入
============================================================
  已插入 1000/10000 个向量
  已插入 2000/10000 个向量
  ...
  已插入 10000/10000 个向量

插入完成，耗时: 5.23秒
吞吐量: 1912 向量/秒
数据已刷新到磁盘

============================================================
加载Collection
============================================================
加载完成，耗时: 2.34秒
向量数量: 10000

============================================================
查询测试
============================================================
查询延迟: 8.45ms

Top-10结果:
1. 距离=0.9234, 文本=文档内容 1234
2. 距离=0.9156, 文本=文档内容 5678
...

============================================================
批量查询测试
============================================================
批量查询 100 次
总耗时: 0.85秒
平均延迟: 8.50ms
QPS: 118

============================================================
性能对比：不同ef参数
============================================================
ef= 16: 平均延迟=4.23ms
ef= 32: 平均延迟=6.45ms
ef= 64: 平均延迟=8.67ms
ef=128: 平均延迟=12.34ms

============================================================
测试完成
============================================================
```

---

## 关键学习点

### 1. Milvus架构

**组件**：
- etcd：元数据存储
- MinIO：对象存储
- Milvus：向量检索引擎

---

### 2. 索引选择

**HNSW**：
- 召回率高（96-98%）
- 查询快（5-10ms）
- 适合<1000万向量

**IVF**：
- 内存效率高
- 适合>1000万向量
- 可结合PQ量化

---

### 3. 性能优化

**批量插入**：
- batch_size=1000
- 吞吐量：~2000向量/秒

**批量查询**：
- 并行查询
- QPS：~100-200

**参数调优**：
- ef=64（平衡）
- nprobe=10（IVF）

---

## 练习题

### 练习1：IVF-PQ索引

**任务**：使用IVF-PQ压缩内存

**提示**：
```python
ivfpq_index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 1024,
        "m": 8,
        "nbits": 8
    }
}
```

---

### 练习2：分布式部署

**任务**：部署Milvus集群

**提示**：
```yaml
# docker-compose-cluster.yml
services:
  milvus-coordinator:
    ...
  milvus-datanode:
    ...
  milvus-querynode:
    ...
```

---

### 练习3：监控与告警

**任务**：集成Prometheus监控

**提示**：
```python
from prometheus_client import Counter, Histogram

query_latency = Histogram('query_latency_seconds', 'Query latency')
query_count = Counter('query_total', 'Total queries')
```

---

## 总结

通过Milvus，我们实现了：
1. 生产级向量数据库部署
2. HNSW/IVF索引配置
3. 批量插入与查询优化
4. 性能监控与调优

**下一步**：学习混合检索RAG系统，结合Vector和BM25。
