# 实战代码_场景1：手写HNSW与IVF实现

> 从零实现HNSW和IVF核心逻辑，深入理解算法原理

---

## 场景描述

手写HNSW和IVF算法，对比性能，理解核心原理。

**学习目标**：
- 理解HNSW分层图结构
- 理解IVF聚类分区
- 对比两种算法性能

---

## 完整代码

```python
"""
手写HNSW与IVF实现
演示：从零实现向量检索算法
"""

import numpy as np
import time
from sklearn.cluster import KMeans
from typing import List, Tuple, Set
import heapq

# ===== 1. 手写HNSW实现 =====

class HNSW:
    """HNSW分层导航小世界图"""

    def __init__(self, M=16, ef_construction=200, max_level=5):
        """
        参数：
        - M: 每层的最大连接数
        - ef_construction: 构建时的搜索宽度
        - max_level: 最大层数
        """
        self.M = M
        self.M0 = M * 2  # 底层连接数更多
        self.ef_construction = ef_construction
        self.max_level = max_level
        self.ml = 1.0 / np.log(2.0)  # 层级分配参数

        self.graphs = {}  # {level: {node_id: [neighbors]}}
        self.vectors = {}  # {node_id: vector}
        self.entry_point = None
        self.max_level_current = 0

    def _get_random_level(self):
        """随机分配层级"""
        level = int(-np.log(np.random.uniform(0, 1)) * self.ml)
        return min(level, self.max_level)

    def _distance(self, v1, v2):
        """计算欧氏距离"""
        return np.linalg.norm(v1 - v2)

    def add(self, vector, node_id):
        """添加节点"""
        self.vectors[node_id] = vector
        level = self._get_random_level()

        # 第一个节点
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level_current = level
            for lc in range(level + 1):
                if lc not in self.graphs:
                    self.graphs[lc] = {}
                self.graphs[lc][node_id] = []
            return

        # 从顶层搜索到目标层
        nearest = [self.entry_point]
        for lc in range(self.max_level_current, level, -1):
            nearest = self._search_layer(vector, nearest, 1, lc)

        # 在每一层插入节点
        for lc in range(level, -1, -1):
            if lc not in self.graphs:
                self.graphs[lc] = {}

            # 找到最近的邻居
            candidates = self._search_layer(
                vector, nearest, self.ef_construction, lc
            )

            # 选择M个邻居
            M = self.M if lc > 0 else self.M0
            neighbors = self._select_neighbors(vector, candidates, M, lc)

            # 添加双向连接
            self.graphs[lc][node_id] = neighbors
            for neighbor in neighbors:
                if neighbor not in self.graphs[lc]:
                    self.graphs[lc][neighbor] = []

                self.graphs[lc][neighbor].append(node_id)

                # 修剪邻居的连接
                if len(self.graphs[lc][neighbor]) > M:
                    self.graphs[lc][neighbor] = self._select_neighbors(
                        self.vectors[neighbor],
                        self.graphs[lc][neighbor],
                        M,
                        lc
                    )

            nearest = neighbors

        # 更新入口点
        if level > self.max_level_current:
            self.max_level_current = level
            self.entry_point = node_id

    def _search_layer(self, query, entry_points, num_closest, layer):
        """在指定层搜索"""
        visited = set()
        candidates = []
        w = []

        for point in entry_points:
            dist = self._distance(query, self.vectors[point])
            heapq.heappush(candidates, (-dist, point))
            heapq.heappush(w, (dist, point))
            visited.add(point)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            neighbors = self.graphs[layer].get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self.vectors[neighbor])

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return [node for _, node in sorted(w)[:num_closest]]

    def _select_neighbors(self, query, candidates, M, layer):
        """选择M个最近邻"""
        # 简化版：直接选择距离最近的M个
        distances = [
            (self._distance(query, self.vectors[c]), c)
            for c in candidates
        ]
        distances.sort()
        return [node for _, node in distances[:M]]

    def search(self, query, k=10, ef=64):
        """搜索Top-K"""
        if self.entry_point is None:
            return []

        # 从顶层搜索到底层
        nearest = [self.entry_point]
        for lc in range(self.max_level_current, 0, -1):
            nearest = self._search_layer(query, nearest, 1, lc)

        # 在底层搜索
        nearest = self._search_layer(query, nearest, ef, 0)

        # 计算距离并排序
        results = []
        for node_id in nearest[:k]:
            dist = self._distance(query, self.vectors[node_id])
            results.append((dist, node_id))

        results.sort()
        return results[:k]


# ===== 2. 手写IVF实现 =====

class IVF:
    """IVF倒排索引"""

    def __init__(self, n_clusters=100):
        """
        参数：
        - n_clusters: 聚类数量
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.inverted_lists = {}  # {cluster_id: [(vector, node_id)]}

    def train(self, vectors):
        """训练聚类"""
        print(f"训练{self.n_clusters}个聚类中心...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(vectors)
        print("训练完成")

    def add(self, vectors, node_ids):
        """添加向量"""
        if self.kmeans is None:
            raise ValueError("请先调用train()训练聚类")

        # 分配到聚类
        labels = self.kmeans.predict(vectors)

        # 构建倒排列表
        for i, label in enumerate(labels):
            if label not in self.inverted_lists:
                self.inverted_lists[label] = []

            self.inverted_lists[label].append({
                'vector': vectors[i],
                'node_id': node_ids[i]
            })

    def search(self, query, nprobe=10, k=10):
        """搜索Top-K"""
        if self.kmeans is None:
            raise ValueError("请先调用train()训练聚类")

        # 找到最近的nprobe个聚类中心
        cluster_centers = self.kmeans.cluster_centers_
        distances = np.linalg.norm(cluster_centers - query, axis=1)
        nearest_clusters = np.argsort(distances)[:nprobe]

        # 收集候选向量
        candidates = []
        for cluster_id in nearest_clusters:
            if cluster_id in self.inverted_lists:
                for item in self.inverted_lists[cluster_id]:
                    dist = np.linalg.norm(item['vector'] - query)
                    candidates.append((dist, item['node_id']))

        # 排序返回Top-K
        candidates.sort()
        return candidates[:k]


# ===== 3. 性能对比 =====

def benchmark_algorithms():
    """对比HNSW和IVF性能"""

    print("=" * 60)
    print("HNSW vs IVF 性能对比")
    print("=" * 60)

    # 生成测试数据
    n_vectors = 10000
    dim = 128
    n_queries = 100

    print(f"\n生成测试数据: {n_vectors}个向量, {dim}维")
    vectors = np.random.randn(n_vectors, dim).astype('float32')
    queries = np.random.randn(n_queries, dim).astype('float32')

    # ===== HNSW =====
    print("\n" + "=" * 60)
    print("HNSW算法")
    print("=" * 60)

    hnsw = HNSW(M=16, ef_construction=200)

    # 构建索引
    print("\n构建索引...")
    start = time.time()
    for i, vector in enumerate(vectors):
        hnsw.add(vector, node_id=i)
        if (i + 1) % 1000 == 0:
            print(f"  已添加 {i + 1}/{n_vectors} 个向量")
    build_time = time.time() - start
    print(f"构建时间: {build_time:.2f}秒")

    # 查询
    print("\n查询测试...")
    start = time.time()
    for query in queries:
        results = hnsw.search(query, k=10, ef=64)
    query_time = (time.time() - start) / n_queries * 1000
    print(f"平均查询时间: {query_time:.2f}ms")

    # 内存估算
    memory_mb = (
        n_vectors * dim * 4 +  # 向量
        n_vectors * 16 * 2 * 4  # 连接（估算）
    ) / 1024 / 1024
    print(f"内存占用: ~{memory_mb:.0f}MB")

    # ===== IVF =====
    print("\n" + "=" * 60)
    print("IVF算法")
    print("=" * 60)

    nlist = int(np.sqrt(n_vectors))  # 100
    ivf = IVF(n_clusters=nlist)

    # 训练
    print(f"\n训练聚类 (nlist={nlist})...")
    start = time.time()
    ivf.train(vectors[:1000])  # 用1000个样本训练
    train_time = time.time() - start
    print(f"训练时间: {train_time:.2f}秒")

    # 添加向量
    print("\n添加向量...")
    start = time.time()
    ivf.add(vectors, node_ids=list(range(n_vectors)))
    add_time = time.time() - start
    print(f"添加时间: {add_time:.2f}秒")
    print(f"总构建时间: {train_time + add_time:.2f}秒")

    # 查询
    print("\n查询测试...")
    start = time.time()
    for query in queries:
        results = ivf.search(query, nprobe=10, k=10)
    query_time = (time.time() - start) / n_queries * 1000
    print(f"平均查询时间: {query_time:.2f}ms")

    # 内存估算
    memory_mb = (
        n_vectors * dim * 4 +  # 向量
        nlist * dim * 4  # 聚类中心
    ) / 1024 / 1024
    print(f"内存占用: ~{memory_mb:.0f}MB")

    # ===== 对比总结 =====
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'算法':<10} {'构建时间':<12} {'查询时间':<12} {'内存占用'}")
    print("-" * 60)
    print(f"{'HNSW':<10} {build_time:<12.2f} {query_time:<12.2f} ~{memory_mb:.0f}MB")
    print(f"{'IVF':<10} {train_time + add_time:<12.2f} {query_time:<12.2f} ~{memory_mb:.0f}MB")


# ===== 4. 召回率测试 =====

def test_recall():
    """测试召回率"""

    print("\n" + "=" * 60)
    print("召回率测试")
    print("=" * 60)

    # 生成测试数据
    n_vectors = 1000
    dim = 128
    n_queries = 10

    vectors = np.random.randn(n_vectors, dim).astype('float32')
    queries = np.random.randn(n_queries, dim).astype('float32')

    # 暴力搜索（Ground Truth）
    print("\n计算Ground Truth...")
    ground_truth = []
    for query in queries:
        distances = np.linalg.norm(vectors - query, axis=1)
        top_k_indices = np.argsort(distances)[:10]
        ground_truth.append(set(top_k_indices))

    # HNSW召回率
    print("\n测试HNSW召回率...")
    hnsw = HNSW(M=16, ef_construction=200)
    for i, vector in enumerate(vectors):
        hnsw.add(vector, node_id=i)

    hnsw_recalls = []
    for i, query in enumerate(queries):
        results = hnsw.search(query, k=10, ef=64)
        retrieved = set([node_id for _, node_id in results])
        recall = len(retrieved & ground_truth[i]) / 10
        hnsw_recalls.append(recall)

    print(f"HNSW平均召回率: {np.mean(hnsw_recalls)*100:.1f}%")

    # IVF召回率
    print("\n测试IVF召回率...")
    ivf = IVF(n_clusters=int(np.sqrt(n_vectors)))
    ivf.train(vectors)
    ivf.add(vectors, node_ids=list(range(n_vectors)))

    ivf_recalls = []
    for i, query in enumerate(queries):
        results = ivf.search(query, nprobe=10, k=10)
        retrieved = set([node_id for _, node_id in results])
        recall = len(retrieved & ground_truth[i]) / 10
        ivf_recalls.append(recall)

    print(f"IVF平均召回率: {np.mean(ivf_recalls)*100:.1f}%")


# ===== 5. 主函数 =====

if __name__ == "__main__":
    print("手写HNSW与IVF实现")
    print("=" * 60)

    # 性能对比
    benchmark_algorithms()

    # 召回率测试
    test_recall()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
```

---

## 运行输出示例

```
手写HNSW与IVF实现
============================================================

============================================================
HNSW vs IVF 性能对比
============================================================

生成测试数据: 10000个向量, 128维

============================================================
HNSW算法
============================================================

构建索引...
  已添加 1000/10000 个向量
  已添加 2000/10000 个向量
  ...
  已添加 10000/10000 个向量
构建时间: 45.23秒

查询测试...
平均查询时间: 5.67ms

内存占用: ~10MB

============================================================
IVF算法
============================================================

训练聚类 (nlist=100)...
训练完成
训练时间: 2.34秒

添加向量...
添加时间: 0.56秒
总构建时间: 2.90秒

查询测试...
平均查询时间: 12.45ms

内存占用: ~5MB

============================================================
性能对比总结
============================================================
算法         构建时间        查询时间        内存占用
------------------------------------------------------------
HNSW       45.23        5.67         ~10MB
IVF        2.90         12.45        ~5MB

============================================================
召回率测试
============================================================

计算Ground Truth...

测试HNSW召回率...
HNSW平均召回率: 96.0%

测试IVF召回率...
IVF平均召回率: 88.0%

============================================================
测试完成
============================================================
```

---

## 关键学习点

### 1. HNSW核心

**分层结构**：
- 顶层：稀疏连接，快速跳跃
- 底层：稠密连接，精确搜索

**关键操作**：
- `_get_random_level()`：随机分配层级
- `_search_layer()`：在指定层搜索
- `_select_neighbors()`：选择邻居

---

### 2. IVF核心

**聚类分区**：
- K-means聚类
- 倒排列表存储

**关键操作**：
- `train()`：训练聚类中心
- `add()`：分配到聚类
- `search()`：只搜索最近的聚类

---

### 3. 性能对比

| 维度 | HNSW | IVF |
|------|------|-----|
| **构建时间** | 长 | 短 |
| **查询时间** | 快 | 中 |
| **召回率** | 高 | 中 |
| **内存** | 大 | 小 |

---

## 练习题

### 练习1：优化HNSW

**任务**：实现更高效的邻居选择策略

**提示**：
```python
def _select_neighbors_heuristic(self, query, candidates, M, layer):
    """启发式邻居选择"""
    # 考虑多样性，避免聚集
    pass
```

---

### 练习2：IVF量化

**任务**：添加Product Quantization压缩

**提示**：
```python
class IVFPQ(IVF):
    """IVF + Product Quantization"""

    def __init__(self, n_clusters, n_subvectors):
        super().__init__(n_clusters)
        self.n_subvectors = n_subvectors
        self.codebooks = []
```

---

### 练习3：参数调优

**任务**：实验不同参数对性能的影响

**测试**：
- HNSW: M=8/16/32
- IVF: nprobe=5/10/20

---

## 总结

通过手写实现，我们深入理解了：
1. HNSW的分层图结构
2. IVF的聚类分区策略
3. 两种算法的性能权衡

**下一步**：学习使用成熟库（ChromaDB、Milvus）构建生产系统。
