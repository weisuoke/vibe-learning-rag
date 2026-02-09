# 核心概念 2：查询层 QueryNode

## 1. 什么是 QueryNode

**QueryNode** 是 Milvus 的查询执行引擎，负责执行向量检索和标量查询。

### 在架构中的位置

```
用户请求
   ↓
Proxy (接收查询)
   ↓
QueryCoord (调度)
   ↓
QueryNode (执行查询) ← 从对象存储加载 Segment
   ↓
返回结果
```

**核心特点：**
- 无状态计算节点
- 从对象存储加载数据到内存
- 执行 ANN (近似最近邻) 搜索
- 支持水平扩展

---

## 2. 核心职责

### 2.1 Load Segment - 加载数据段

**作用：** 从对象存储加载 Segment 到内存

```python
# QueryNode 加载 Segment 的过程
# 1. 接收 QueryCoord 的加载指令
# 2. 从 MinIO/S3 下载 Segment 文件
# 3. 加载到内存并构建索引
# 4. 报告加载完成状态
```

**两种 Segment 类型：**
- **Sealed Segment**: 已封存的不可变数据（历史数据）
- **Growing Segment**: 正在写入的可变数据（实时数据）

### 2.2 Execute ANN Search - 执行向量检索

**作用：** 执行近似最近邻搜索

```python
# ANN 搜索流程
query_vector = [0.1, 0.2, ..., 0.768]  # 查询向量

# QueryNode 执行：
# 1. 在内存中的索引上搜索
# 2. 使用 HNSW/IVF 等算法
# 3. 返回 Top-K 最相似的向量
```

**支持的索引类型：**
- FLAT: 暴力搜索（精确）
- IVF_FLAT: 倒排索引
- HNSW: 层次化图索引
- DISKANN: 磁盘索引

### 2.3 Scalar Filtering - 标量过滤

**作用：** 根据标量字段过滤结果

```python
# 标量过滤示例
# 先向量检索，再标量过滤
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    expr="age > 25 and city == 'Beijing'"  # 标量过滤条件
)
```

**过滤策略：**
- Post-filtering: 先向量检索，后标量过滤（默认）
- Pre-filtering: 先标量过滤，后向量检索（需要索引支持）

### 2.4 Result Merging - 结果合并

**作用：** 合并多个 Segment 的查询结果

```python
# 结果合并流程
# 1. 每个 Segment 返回 Top-K 结果
# 2. QueryNode 合并所有结果
# 3. 重新排序，返回全局 Top-K
```

**合并策略：**
- 按距离排序
- 去重（如果有重复 ID）
- 返回最终 Top-K

---

## 3. QueryCoord 的作用

**QueryCoord** 是查询层的协调者，负责管理 QueryNode。

### 3.1 核心职责

| 职责 | 说明 |
|------|------|
| **Segment 分配** | 决定哪个 QueryNode 加载哪些 Segment |
| **负载均衡** | 平衡各 QueryNode 的内存和查询负载 |
| **故障恢复** | QueryNode 故障时重新分配 Segment |
| **查询路由** | 告诉 Proxy 哪些 QueryNode 有所需数据 |

### 3.2 工作流程

```
QueryCoord 工作流程：

1. 监听 Collection 加载请求
   ↓
2. 查询 Segment 元数据（从 etcd）
   ↓
3. 分配 Segment 到 QueryNode
   ↓
4. 监控加载进度
   ↓
5. 更新 Segment 分布信息
   ↓
6. 响应 Proxy 的查询路由请求
```

---

## 4. 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        QueryCoord                            │
│  - Segment 分配                                              │
│  - 负载均衡                                                  │
│  - 查询路由                                                  │
└────────────┬────────────────────────────────────────────────┘
             │ 分配 Segment
             ↓
┌────────────────────────────────────────────────────────────┐
│                     QueryNode 集群                          │
├──────────────────┬──────────────────┬──────────────────────┤
│  QueryNode 1     │  QueryNode 2     │  QueryNode 3         │
│                  │                  │                      │
│  Segment 1,2,3   │  Segment 4,5,6   │  Segment 7,8,9       │
│                  │                  │                      │
│  [内存数据]      │  [内存数据]      │  [内存数据]          │
│  [索引]          │  [索引]          │  [索引]              │
└────────┬─────────┴────────┬─────────┴────────┬─────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            ↓
                    从对象存储加载 Segment
                            ↓
                ┌───────────────────────┐
                │   MinIO / S3          │
                │   (对象存储)          │
                └───────────────────────┘
```

---

## 5. Segment 加载机制

### 5.1 Sealed Segment (封存段)

**特点：**
- 不可变数据
- 已构建索引
- 存储在对象存储中

**加载流程：**
```python
# 1. QueryCoord 分配 Sealed Segment
# 2. QueryNode 从对象存储下载
# 3. 加载到内存
# 4. 准备好接受查询
```

### 5.2 Growing Segment (增长段)

**特点：**
- 可变数据（正在写入）
- 未构建索引（使用暴力搜索）
- 存储在 DataNode 内存中

**加载流程：**
```python
# 1. QueryNode 订阅消息队列
# 2. 实时接收新插入的数据
# 3. 在内存中维护 Growing Segment
# 4. 查询时同时搜索 Sealed + Growing
```

### 5.3 加载策略

**On-demand Loading (按需加载)：**
```python
# 查询时才加载 Collection
collection.load()  # 触发加载

# 优点：节省内存
# 缺点：首次查询慢
```

**Pre-loading (预加载)：**
```python
# 启动时自动加载
# 在 Milvus 配置中设置

# 优点：查询快
# 缺点：占用内存
```

---

## 6. 代码示例：监控 QueryNode 状态

### 示例 1：查看 Segment 分布

```python
from pymilvus import connections, utility

# 连接 Milvus
connections.connect(host="localhost", port="19530")

# 查看 Collection 的 Segment 信息
collection_name = "my_collection"
segments = utility.get_query_segment_info(collection_name)

print(f"Collection: {collection_name}")
print(f"Total Segments: {len(segments)}")
print("\nSegment 分布:")

for seg in segments:
    print(f"  Segment ID: {seg.segmentID}")
    print(f"    QueryNode ID: {seg.nodeID}")
    print(f"    State: {seg.state}")
    print(f"    Num Rows: {seg.num_rows}")
    print(f"    Memory Size: {seg.mem_size / 1024 / 1024:.2f} MB")
    print()
```

**输出示例：**
```
Collection: my_collection
Total Segments: 6

Segment 分布:
  Segment ID: 448900612345678901
    QueryNode ID: 1
    State: Sealed
    Num Rows: 10000
    Memory Size: 45.23 MB

  Segment ID: 448900612345678902
    QueryNode ID: 2
    State: Sealed
    Num Rows: 10000
    Memory Size: 45.23 MB
...
```

### 示例 2：监控查询性能

```python
import time
from pymilvus import Collection

collection = Collection("my_collection")

# 执行查询并监控性能
query_vector = [[0.1] * 768]

start_time = time.time()
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10
)
end_time = time.time()

print(f"查询耗时: {(end_time - start_time) * 1000:.2f} ms")
print(f"返回结果数: {len(results[0])}")

# 查看查询统计
for i, hit in enumerate(results[0]):
    print(f"  Top-{i+1}: ID={hit.id}, Distance={hit.distance:.4f}")
```

### 示例 3：负载均衡测试

```python
import concurrent.futures
import time

def query_task(task_id):
    """单个查询任务"""
    collection = Collection("my_collection")
    query_vector = [[0.1] * 768]

    start = time.time()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10
    )
    latency = (time.time() - start) * 1000

    return task_id, latency

# 并发查询测试
num_queries = 100
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(query_task, i) for i in range(num_queries)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# 统计结果
latencies = [r[1] for r in results]
print(f"总查询数: {num_queries}")
print(f"平均延迟: {sum(latencies) / len(latencies):.2f} ms")
print(f"P50 延迟: {sorted(latencies)[len(latencies)//2]:.2f} ms")
print(f"P99 延迟: {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
```

---

## 7. 在 RAG 中的应用

### 7.1 高并发检索

**场景：** 多用户同时查询知识库

```python
# RAG 系统中的并发查询
class RAGSystem:
    def __init__(self):
        self.collection = Collection("knowledge_base")
        self.collection.load()  # 预加载到 QueryNode

    def retrieve(self, query: str, top_k: int = 5):
        """检索相关文档"""
        # 1. 生成查询向量
        query_vector = self.embed_model.encode(query)

        # 2. 向量检索（由 QueryNode 执行）
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["text", "source"]
        )

        return results[0]

# 多用户并发访问
rag = RAGSystem()

# 用户 1 查询
results1 = rag.retrieve("什么是向量数据库？")

# 用户 2 查询（同时进行）
results2 = rag.retrieve("如何优化检索性能？")

# QueryNode 自动负载均衡
```

### 7.2 内存管理

**场景：** 大规模知识库的内存优化

```python
# 按需加载策略
class SmartRAG:
    def __init__(self):
        self.collections = {
            "tech_docs": Collection("tech_docs"),
            "business_docs": Collection("business_docs"),
            "legal_docs": Collection("legal_docs")
        }

    def retrieve(self, query: str, domain: str):
        """根据领域动态加载 Collection"""
        collection = self.collections[domain]

        # 检查是否已加载
        if not collection.is_loaded:
            print(f"Loading {domain} to QueryNode...")
            collection.load()

        # 执行查询
        query_vector = self.embed_model.encode(query)
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2"},
            limit=5
        )

        return results[0]

# 使用
rag = SmartRAG()
results = rag.retrieve("合同条款", domain="legal_docs")
# 只加载 legal_docs，节省内存
```

### 7.3 性能优化

**场景：** 优化 RAG 检索延迟

```python
# 优化策略
class OptimizedRAG:
    def __init__(self):
        self.collection = Collection("knowledge_base")

        # 1. 预加载到 QueryNode
        self.collection.load()

        # 2. 创建高性能索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("embedding", index_params)

    def retrieve_fast(self, query: str):
        """快速检索"""
        query_vector = self.embed_model.encode(query)

        # 使用优化的搜索参数
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"ef": 32}  # 降低 ef 提升速度
            },
            limit=5
        )

        return results[0]

# 性能对比
rag = OptimizedRAG()

# 快速检索（牺牲少量精度）
start = time.time()
results = rag.retrieve_fast("向量数据库")
print(f"Fast retrieval: {(time.time() - start) * 1000:.2f} ms")
```

---

## 8. 性能考虑

### 8.1 QueryNode 是计算密集型

**特点：**
- CPU 密集：向量计算、距离计算
- 内存密集：加载 Segment 和索引
- I/O 密集：从对象存储加载数据

**优化建议：**
```python
# 1. 使用高性能 CPU
# 2. 配置足够内存
# 3. 使用 SSD 存储
# 4. 启用 SIMD 指令集
```

### 8.2 内存需求

**计算公式：**
```
QueryNode 内存 = Segment 数据 + 索引 + 查询缓存

示例：
- 1M 向量 × 768 维 × 4 字节 = 3 GB (原始数据)
- HNSW 索引 ≈ 1.5 × 原始数据 = 4.5 GB
- 查询缓存 ≈ 1 GB
- 总计 ≈ 8.5 GB
```

**内存优化：**
```python
# 1. 使用量化索引（减少内存）
index_params = {
    "index_type": "IVF_PQ",  # Product Quantization
    "params": {"nlist": 1024, "m": 8}
}

# 2. 分批加载 Partition
collection.load(partition_names=["partition_2024"])

# 3. 释放不用的 Collection
collection.release()
```

### 8.3 水平扩展

**扩展策略：**
```bash
# 增加 QueryNode 数量
# 1. 启动新的 QueryNode 实例
docker run -d milvus/milvus:latest querynode

# 2. QueryCoord 自动重新分配 Segment
# 3. 负载自动均衡
```

**扩展效果：**
```
1 QueryNode:  100 QPS
2 QueryNode:  180 QPS (1.8x)
4 QueryNode:  320 QPS (3.2x)
8 QueryNode:  560 QPS (5.6x)

# 注意：扩展效率受网络和协调开销影响
```

---

## 9. 常见问题

### Q1: QueryNode 和 DataNode 的区别？

**回答：**
- **DataNode**: 负责数据写入和持久化
- **QueryNode**: 负责数据查询和检索
- 分离读写，互不影响

### Q2: 如何监控 QueryNode 性能？

**回答：**
```python
# 使用 Milvus Metrics API
import requests

response = requests.get("http://localhost:9091/metrics")
metrics = response.text

# 关键指标：
# - milvus_querynode_search_latency
# - milvus_querynode_memory_usage
# - milvus_querynode_segment_num
```

### Q3: QueryNode 故障如何恢复？

**回答：**
```
1. QueryCoord 检测到 QueryNode 故障
2. 将故障节点的 Segment 重新分配给其他 QueryNode
3. 其他 QueryNode 从对象存储加载 Segment
4. 恢复查询服务（自动完成）
```

### Q4: 如何优化 QueryNode 内存使用？

**回答：**
```python
# 1. 使用 Partition 分批加载
collection.load(partition_names=["recent_data"])

# 2. 使用量化索引
index_params = {"index_type": "IVF_SQ8"}  # Scalar Quantization

# 3. 定期释放不用的 Collection
collection.release()

# 4. 配置内存限制
# 在 milvus.yaml 中设置 queryNode.cache.memoryLimit
```

---

## 10. 总结

**QueryNode 核心要点：**

1. **执行引擎**: 负责向量检索和标量查询
2. **无状态设计**: 从对象存储加载数据，支持水平扩展
3. **四大职责**: Load Segment、ANN Search、Scalar Filtering、Result Merging
4. **QueryCoord 协调**: 负责 Segment 分配和负载均衡
5. **两种 Segment**: Sealed (历史) + Growing (实时)
6. **RAG 应用**: 高并发检索、内存管理、性能优化
7. **性能优化**: 计算密集型，需要足够 CPU 和内存
8. **水平扩展**: 增加 QueryNode 提升查询吞吐量

**记住：** QueryNode 是 Milvus 的查询引擎，负责执行向量检索。理解 QueryNode 的工作原理，对优化 RAG 系统的检索性能至关重要。

---

**下一步：** 学习核心概念 3 - 数据层 DataNode，了解数据写入和持久化机制。
