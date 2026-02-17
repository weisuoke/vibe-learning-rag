# 核心概念08：DiskANN与SPANN生产算法

> 了解Microsoft的超大规模向量检索算法

---

## 概述

**DiskANN**和**SPANN**是Microsoft Research开发的生产级向量检索算法，支持数十亿规模。

**核心特点**：
- DiskANN：10亿向量<100ms
- SPANN：数百亿向量，Bing生产部署
- 存储：SSD而非内存

---

## 1. DiskANN算法

### 1.1 核心思想

**问题**：内存无法容纳数十亿向量
**解决**：将向量存储在SSD，优化磁盘访问

**Vamana图算法**：
```
1. 构建导航图（类似HNSW）
2. 优化磁盘访问模式
3. 使用SSD而非内存
```

---

### 1.2 架构设计

**三层存储**：
```
内存：
- 图结构（邻接表）
- 聚类中心
- 缓存热点向量

SSD：
- 完整向量数据
- 按图结构组织
- 优化顺序读取

查询流程：
1. 在内存图中导航
2. 按需从SSD加载向量
3. 缓存热点数据
```

---

### 1.3 性能指标

**10亿向量测试**：
```
数据规模：10亿向量，768维
内存占用：<10GB（只存图结构）
SSD占用：~3TB
查询延迟：<100ms
召回率：>95%
```

---

## 2. SPANN算法

### 2.1 核心思想

**分布式HNSW**：
```
1. 将向量分片到多个节点
2. 每个节点运行HNSW
3. 协调器聚合结果
```

**架构**：
```
┌─────────────────────────────────────┐
│         Query Coordinator           │
└─────────────────────────────────────┘
          │
          ├─────────┬─────────┬─────────┐
          │         │         │         │
     ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐ ┌──▼─────┐
     │ HNSW   │ │ HNSW   │ │ HNSW   │ │ HNSW   │
     │ Shard1 │ │ Shard2 │ │ Shard3 │ │ Shard4 │
     └────────┘ └────────┘ └────────┘ └────────┘
```

---

### 2.2 分片策略

**基于聚类的分片**：
```python
# 1. 训练全局聚类
kmeans = KMeans(n_clusters=1000)
kmeans.fit(sample_vectors)

# 2. 分配向量到分片
for vector in vectors:
    cluster_id = kmeans.predict(vector)
    shard_id = cluster_id % num_shards
    shards[shard_id].add(vector)

# 3. 每个分片构建HNSW
for shard in shards:
    shard.build_hnsw()
```

---

### 2.3 查询流程

```python
def spann_search(query, k=10):
    """SPANN查询"""

    # 1. 找到最相关的分片
    cluster_id = kmeans.predict(query)
    relevant_shards = get_shards_for_cluster(cluster_id)

    # 2. 并行查询多个分片
    results = []
    for shard in relevant_shards:
        shard_results = shard.search(query, k=k*2)
        results.extend(shard_results)

    # 3. 全局排序
    results.sort(key=lambda x: x.distance)
    return results[:k]
```

---

## 3. 生产部署

### 3.1 Azure Cosmos DB集成

**DiskANN in Cosmos DB**：
```python
from azure.cosmos import CosmosClient

client = CosmosClient(url, key)
database = client.get_database_client("mydb")
container = database.get_container_client("vectors")

# 创建DiskANN索引
container.create_index({
    "indexingMode": "consistent",
    "vectorIndexes": [{
        "path": "/embedding",
        "type": "diskANN",
        "params": {
            "maxDegree": 64,
            "searchListSize": 100
        }
    }]
})

# 查询
results = container.query_items(
    query="SELECT * FROM c ORDER BY VectorDistance(c.embedding, @query)",
    parameters=[{"name": "@query", "value": query_vector}],
    max_item_count=10
)
```

---

### 3.2 SQL Server 2025集成

**DiskANN公测**：
```sql
-- 创建向量表
CREATE TABLE documents (
    id INT PRIMARY KEY,
    content NVARCHAR(MAX),
    embedding VECTOR(768)
);

-- 创建DiskANN索引
CREATE VECTOR INDEX doc_diskann_idx ON documents(embedding)
WITH (TYPE = DISKANN, MAX_DEGREE = 64);

-- 向量检索
SELECT TOP 10 id, content,
       VECTOR_DISTANCE(embedding, @query_vector) as distance
FROM documents
ORDER BY distance;
```

---

### 3.3 Bing生产部署

**SPANN in Bing**：
- 规模：数百亿向量
- 延迟：<50ms（P95）
- 召回率：>95%
- 架构：分布式HNSW

---

## 4. 与HNSW/IVF对比

### 4.1 性能对比

| 算法 | 规模 | 内存 | 查询延迟 | 召回率 |
|------|------|------|---------|--------|
| **HNSW** | <1000万 | 3GB | 5ms | 98% |
| **IVF-PQ** | <1亿 | 5GB | 50ms | 90% |
| **DiskANN** | >10亿 | 10GB | 100ms | 95% |
| **SPANN** | >100亿 | 分布式 | 50ms | 95% |

---

### 4.2 适用场景

**HNSW**：
- 中小规模
- 内存充足
- 高召回率

**IVF-PQ**：
- 大规模
- 内存受限
- 可接受召回率

**DiskANN**：
- 超大规模（>10亿）
- SSD可用
- 单机部署

**SPANN**：
- 超大规模（>100亿）
- 分布式集群
- 生产级稳定性

---

## 5. 实现示例

### 5.1 DiskANN概念实现

```python
import numpy as np
import os

class SimpleDiskANN:
    """简化的DiskANN实现"""

    def __init__(self, index_dir="./diskann_index"):
        self.index_dir = index_dir
        self.graph = {}  # 内存中的图结构
        self.vector_file = os.path.join(index_dir, "vectors.dat")

        os.makedirs(index_dir, exist_ok=True)

    def build(self, vectors, M=16):
        """构建索引"""
        n = len(vectors)

        # 1. 将向量写入磁盘
        with open(self.vector_file, 'wb') as f:
            for vector in vectors:
                f.write(vector.tobytes())

        # 2. 构建图结构（内存）
        for i in range(n):
            # 找到最近的M个邻居
            neighbors = self._find_neighbors(vectors[i], vectors, M, exclude=[i])
            self.graph[i] = neighbors

        print(f"索引构建完成: {n}个向量")

    def _find_neighbors(self, query, vectors, k, exclude=[]):
        """找到k个最近邻"""
        distances = []
        for i, vector in enumerate(vectors):
            if i in exclude:
                continue
            dist = np.linalg.norm(query - vector)
            distances.append((dist, i))

        distances.sort()
        return [idx for _, idx in distances[:k]]

    def search(self, query, k=10):
        """搜索"""
        # 1. 从入口点开始导航
        current = 0
        visited = set()

        # 2. 贪心搜索
        while True:
            visited.add(current)

            # 从磁盘加载当前向量
            current_vector = self._load_vector(current)
            current_dist = np.linalg.norm(query - current_vector)

            # 检查邻居
            found_better = False
            for neighbor in self.graph[current]:
                if neighbor in visited:
                    continue

                # 从磁盘加载邻居向量
                neighbor_vector = self._load_vector(neighbor)
                neighbor_dist = np.linalg.norm(query - neighbor_vector)

                if neighbor_dist < current_dist:
                    current = neighbor
                    current_dist = neighbor_dist
                    found_better = True
                    break

            if not found_better:
                break

        # 3. 收集k个最近邻
        candidates = [current]
        for neighbor in self.graph[current]:
            candidates.append(neighbor)

        # 计算距离
        results = []
        for idx in candidates[:k]:
            vector = self._load_vector(idx)
            dist = np.linalg.norm(query - vector)
            results.append((dist, idx))

        results.sort()
        return results[:k]

    def _load_vector(self, idx):
        """从磁盘加载向量"""
        with open(self.vector_file, 'rb') as f:
            # 假设768维float32
            f.seek(idx * 768 * 4)
            data = f.read(768 * 4)
            vector = np.frombuffer(data, dtype=np.float32)
        return vector

# 使用示例
vectors = np.random.randn(10000, 768).astype('float32')

diskann = SimpleDiskANN()
diskann.build(vectors, M=16)

query = np.random.randn(768).astype('float32')
results = diskann.search(query, k=10)

print("Top-10结果:")
for dist, idx in results:
    print(f"向量{idx}: 距离{dist:.3f}")
```

---

## 6. 优化技巧

### 6.1 SSD优化

**顺序读取优化**：
```python
# 按图结构组织向量
# 邻居向量在磁盘上相邻
# 减少随机读取
```

**预取策略**：
```python
# 预取下一层邻居
# 减少磁盘访问延迟
```

---

### 6.2 缓存策略

**LRU缓存**：
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def load_vector_cached(idx):
    """缓存热点向量"""
    return load_vector(idx)
```

---

## 7. 2025-2026最新进展

### 7.1 Azure Cosmos DB集成（2025）

**来源**：Microsoft Azure Blog 2025

**特性**：
- DiskANN原生支持
- 自动索引管理
- 分布式部署

---

### 7.2 SQL Server 2025公测

**来源**：Microsoft SQL Server 2025

**特性**：
- DiskANN公测
- SQL查询支持
- 企业级稳定性

---

## 8. 在RAG系统中的应用

### 8.1 适用场景

**超大规模知识库**：
- 数十亿文档
- 单机部署
- SSD可用

**企业数据库集成**：
- Azure Cosmos DB
- SQL Server 2025
- 事务一致性

---

### 8.2 部署建议

```python
# 场景1：10亿文档（DiskANN）
# 硬件：
# - CPU：16核
# - 内存：64GB
# - SSD：4TB NVMe

# 场景2：100亿文档（SPANN）
# 硬件：
# - 10个节点
# - 每节点：16核，64GB内存
# - 分布式SSD存储
```

---

## 总结

### 核心要点

1. **DiskANN**：SSD上的亿级向量检索
2. **SPANN**：分布式HNSW，数百亿规模
3. **生产部署**：Azure Cosmos DB、SQL Server 2025
4. **性能**：10亿向量<100ms

### 选择建议

- <1000万 → HNSW
- 1000万-1亿 → IVF-PQ
- >10亿 → DiskANN
- >100亿 → SPANN

### 下一步

学习 `03_核心概念_09_向量数据库选型对比.md`，了解数据库选择。
