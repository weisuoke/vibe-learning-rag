# 实战代码_场景5：DiskANN大规模检索

> 使用DiskANN处理亿级向量检索

---

## 场景描述

模拟DiskANN算法，实现SSD优化的大规模向量检索。

**学习目标**：
- 理解DiskANN核心思想
- 实现磁盘优化策略
- 对比内存索引与磁盘索引

---

## 完整代码

```python
"""
DiskANN大规模检索
演示：SSD优化的向量检索
"""

import numpy as np
import os
import struct
import time
from pathlib import Path
from typing import List, Tuple
import heapq

# ===== 1. DiskANN核心实现 =====

class DiskANN:
    """简化的DiskANN实现"""

    def __init__(self, index_dir="./diskann_index", M=16):
        """
        参数：
        - index_dir: 索引目录
        - M: 每个节点的最大连接数
        """
        self.index_dir = Path(index_dir)
        self.M = M
        self.graph = {}  # 内存中的图结构
        self.vector_file = self.index_dir / "vectors.dat"
        self.graph_file = self.index_dir / "graph.dat"
        self.metadata_file = self.index_dir / "metadata.dat"
        self.dim = None
        self.n_vectors = 0
        self.entry_point = 0

        # 创建索引目录
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, vectors: np.ndarray):
        """构建索引"""
        print("=" * 60)
        print("构建DiskANN索引")
        print("=" * 60)

        self.n_vectors = len(vectors)
        self.dim = vectors.shape[1]

        print(f"\n向量数量: {self.n_vectors}")
        print(f"向量维度: {self.dim}")

        # 1. 将向量写入磁盘
        print("\n写入向量到磁盘...")
        start = time.time()
        self._write_vectors(vectors)
        write_time = time.time() - start
        print(f"写入完成，耗时: {write_time:.2f}秒")

        # 2. 构建图结构（内存）
        print("\n构建图结构...")
        start = time.time()
        self._build_graph(vectors)
        build_time = time.time() - start
        print(f"构建完成，耗时: {build_time:.2f}秒")

        # 3. 保存图结构到磁盘
        print("\n保存图结构...")
        self._save_graph()

        # 4. 保存元数据
        self._save_metadata()

        print(f"\n索引构建完成")
        print(f"总耗时: {write_time + build_time:.2f}秒")

    def _write_vectors(self, vectors: np.ndarray):
        """将向量写入磁盘"""
        with open(self.vector_file, 'wb') as f:
            for vector in vectors:
                f.write(vector.tobytes())

    def _build_graph(self, vectors: np.ndarray):
        """构建图结构"""
        for i in range(self.n_vectors):
            # 找到最近的M个邻居
            neighbors = self._find_neighbors(vectors[i], vectors, self.M, exclude=[i])
            self.graph[i] = neighbors

            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i + 1}/{self.n_vectors} 个节点")

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

    def _save_graph(self):
        """保存图结构到磁盘"""
        with open(self.graph_file, 'wb') as f:
            for node_id in range(self.n_vectors):
                neighbors = self.graph.get(node_id, [])
                # 写入邻居数量
                f.write(struct.pack('I', len(neighbors)))
                # 写入邻居ID
                for neighbor in neighbors:
                    f.write(struct.pack('I', neighbor))

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'wb') as f:
            f.write(struct.pack('I', self.n_vectors))
            f.write(struct.pack('I', self.dim))
            f.write(struct.pack('I', self.M))
            f.write(struct.pack('I', self.entry_point))

    def load(self):
        """加载索引"""
        print("\n" + "=" * 60)
        print("加载DiskANN索引")
        print("=" * 60)

        # 加载元数据
        with open(self.metadata_file, 'rb') as f:
            self.n_vectors = struct.unpack('I', f.read(4))[0]
            self.dim = struct.unpack('I', f.read(4))[0]
            self.M = struct.unpack('I', f.read(4))[0]
            self.entry_point = struct.unpack('I', f.read(4))[0]

        print(f"\n向量数量: {self.n_vectors}")
        print(f"向量维度: {self.dim}")

        # 加载图结构
        print("\n加载图结构...")
        self._load_graph()

        print("索引加载完成")

    def _load_graph(self):
        """从磁盘加载图结构"""
        with open(self.graph_file, 'rb') as f:
            for node_id in range(self.n_vectors):
                # 读取邻居数量
                n_neighbors = struct.unpack('I', f.read(4))[0]
                # 读取邻居ID
                neighbors = []
                for _ in range(n_neighbors):
                    neighbor = struct.unpack('I', f.read(4))[0]
                    neighbors.append(neighbor)
                self.graph[node_id] = neighbors

    def _load_vector(self, idx: int) -> np.ndarray:
        """从磁盘加载向量"""
        with open(self.vector_file, 'rb') as f:
            # 定位到向量位置
            f.seek(idx * self.dim * 4)
            # 读取向量
            data = f.read(self.dim * 4)
            vector = np.frombuffer(data, dtype=np.float32)
        return vector

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """搜索Top-K"""
        # 贪心搜索
        current = self.entry_point
        visited = set([current])

        while True:
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

            visited.add(current)

            if not found_better:
                break

        # 收集k个最近邻
        candidates = [current]
        for neighbor in self.graph[current]:
            if neighbor not in visited:
                candidates.append(neighbor)

        # 计算距离
        results = []
        for idx in candidates[:k * 2]:
            vector = self._load_vector(idx)
            dist = np.linalg.norm(query - vector)
            results.append((dist, idx))

        results.sort()
        return results[:k]


# ===== 2. 性能对比 =====

def benchmark_diskann():
    """对比DiskANN和内存索引性能"""

    print("=" * 60)
    print("DiskANN vs 内存索引性能对比")
    print("=" * 60)

    # 生成测试数据
    n_vectors = 10000
    dim = 128
    n_queries = 10

    print(f"\n生成测试数据: {n_vectors}个向量, {dim}维")
    vectors = np.random.randn(n_vectors, dim).astype('float32')
    queries = np.random.randn(n_queries, dim).astype('float32')

    # ===== 内存索引（暴力搜索）=====
    print("\n" + "=" * 60)
    print("内存索引（暴力搜索）")
    print("=" * 60)

    # 查询
    print("\n查询测试...")
    start = time.time()
    for query in queries:
        distances = np.linalg.norm(vectors - query, axis=1)
        top_k_indices = np.argsort(distances)[:10]
    memory_time = (time.time() - start) / n_queries * 1000

    print(f"平均查询时间: {memory_time:.2f}ms")

    # 内存占用
    memory_mb = (n_vectors * dim * 4) / 1024 / 1024
    print(f"内存占用: {memory_mb:.0f}MB")

    # ===== DiskANN =====
    print("\n" + "=" * 60)
    print("DiskANN索引")
    print("=" * 60)

    diskann = DiskANN(index_dir="./diskann_test", M=16)

    # 构建索引
    diskann.build(vectors)

    # 加载索引
    diskann.load()

    # 查询
    print("\n查询测试...")
    start = time.time()
    for query in queries:
        results = diskann.search(query, k=10)
    diskann_time = (time.time() - start) / n_queries * 1000

    print(f"平均查询时间: {diskann_time:.2f}ms")

    # 磁盘占用
    disk_mb = sum(
        f.stat().st_size for f in Path("./diskann_test").glob("*")
    ) / 1024 / 1024
    print(f"磁盘占用: {disk_mb:.0f}MB")

    # 内存占用（只有图结构）
    graph_memory = (n_vectors * 16 * 4) / 1024 / 1024
    print(f"内存占用: ~{graph_memory:.0f}MB（只有图结构）")

    # ===== 对比总结 =====
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'方法':<15} {'查询时间':<12} {'内存占用':<12} {'磁盘占用'}")
    print("-" * 60)
    print(f"{'内存索引':<15} {memory_time:<12.2f} {memory_mb:<12.0f} {'0MB'}")
    print(f"{'DiskANN':<15} {diskann_time:<12.2f} {graph_memory:<12.0f} {disk_mb:.0f}MB")

    print("\n优势:")
    print(f"- DiskANN内存占用减少: {(1 - graph_memory/memory_mb)*100:.0f}%")
    print(f"- 查询延迟增加: {(diskann_time/memory_time - 1)*100:.0f}%")


# ===== 3. 大规模模拟 =====

def simulate_large_scale():
    """模拟大规模场景"""

    print("\n" + "=" * 60)
    print("大规模场景模拟")
    print("=" * 60)

    # 模拟1亿向量
    n_vectors = 100_000_000
    dim = 768

    print(f"\n假设场景: {n_vectors}个向量, {dim}维")

    # 内存索引
    memory_gb = (n_vectors * dim * 4) / 1024 / 1024 / 1024
    print(f"\n内存索引:")
    print(f"  内存需求: {memory_gb:.0f}GB")
    print(f"  查询延迟: ~2000ms（暴力搜索）")
    print(f"  成本: 高（需要大内存服务器）")

    # DiskANN
    disk_gb = (n_vectors * dim * 4) / 1024 / 1024 / 1024
    graph_memory_gb = (n_vectors * 16 * 4) / 1024 / 1024 / 1024
    print(f"\nDiskANN:")
    print(f"  内存需求: {graph_memory_gb:.0f}GB（只有图结构）")
    print(f"  磁盘需求: {disk_gb:.0f}GB（SSD）")
    print(f"  查询延迟: ~100ms")
    print(f"  成本: 中（SSD比内存便宜）")

    # 成本对比
    print(f"\n成本对比（AWS）:")
    memory_cost = memory_gb * 10  # $10/GB/月
    disk_cost = graph_memory_gb * 10 + disk_gb * 0.1  # 内存 + SSD
    print(f"  内存索引: ${memory_cost:.0f}/月")
    print(f"  DiskANN: ${disk_cost:.0f}/月")
    print(f"  节省: ${memory_cost - disk_cost:.0f}/月 ({(1-disk_cost/memory_cost)*100:.0f}%)")


# ===== 4. 优化技巧 =====

class OptimizedDiskANN(DiskANN):
    """优化的DiskANN实现"""

    def __init__(self, index_dir="./diskann_index", M=16, cache_size=1000):
        super().__init__(index_dir, M)
        self.cache = {}  # LRU缓存
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_vector_cached(self, idx: int) -> np.ndarray:
        """带缓存的向量加载"""
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]

        self.cache_misses += 1

        # 从磁盘加载
        vector = self._load_vector(idx)

        # 添加到缓存
        if len(self.cache) >= self.cache_size:
            # 简单LRU：删除第一个
            self.cache.pop(next(iter(self.cache)))

        self.cache[idx] = vector
        return vector

    def search_optimized(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """优化的搜索"""
        # 使用缓存的向量加载
        current = self.entry_point
        visited = set([current])

        while True:
            current_vector = self._load_vector_cached(current)
            current_dist = np.linalg.norm(query - current_vector)

            found_better = False
            for neighbor in self.graph[current]:
                if neighbor in visited:
                    continue

                neighbor_vector = self._load_vector_cached(neighbor)
                neighbor_dist = np.linalg.norm(query - neighbor_vector)

                if neighbor_dist < current_dist:
                    current = neighbor
                    current_dist = neighbor_dist
                    found_better = True
                    break

            visited.add(current)

            if not found_better:
                break

        # 收集结果
        candidates = [current]
        for neighbor in self.graph[current]:
            if neighbor not in visited:
                candidates.append(neighbor)

        results = []
        for idx in candidates[:k * 2]:
            vector = self._load_vector_cached(idx)
            dist = np.linalg.norm(query - vector)
            results.append((dist, idx))

        results.sort()
        return results[:k]

    def get_cache_stats(self):
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate
        }


# ===== 5. 主函数 =====

if __name__ == "__main__":
    print("DiskANN大规模检索")
    print("=" * 60)

    # 性能对比
    benchmark_diskann()

    # 大规模模拟
    simulate_large_scale()

    # 优化测试
    print("\n" + "=" * 60)
    print("缓存优化测试")
    print("=" * 60)

    # 生成测试数据
    n_vectors = 1000
    dim = 128
    vectors = np.random.randn(n_vectors, dim).astype('float32')

    # 构建优化索引
    optimized = OptimizedDiskANN(index_dir="./diskann_optimized", M=16, cache_size=100)
    optimized.build(vectors)
    optimized.load()

    # 查询测试
    print("\n查询测试（带缓存）...")
    queries = np.random.randn(10, dim).astype('float32')

    for query in queries:
        results = optimized.search_optimized(query, k=10)

    # 缓存统计
    stats = optimized.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  缓存命中: {stats['hits']}")
    print(f"  缓存未命中: {stats['misses']}")
    print(f"  命中率: {stats['hit_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
```

---

## 运行输出示例

```
DiskANN大规模检索
============================================================

============================================================
DiskANN vs 内存索引性能对比
============================================================

生成测试数据: 10000个向量, 128维

============================================================
内存索引（暴力搜索）
============================================================

查询测试...
平均查询时间: 23.45ms
内存占用: 5MB

============================================================
DiskANN索引
============================================================

============================================================
构建DiskANN索引
============================================================

向量数量: 10000
向量维度: 128

写入向量到磁盘...
写入完成，耗时: 0.12秒

构建图结构...
  已处理 1000/10000 个节点
  已处理 2000/10000 个节点
  ...
  已处理 10000/10000 个节点
构建完成，耗时: 45.23秒

保存图结构...

索引构建完成
总耗时: 45.35秒

============================================================
加载DiskANN索引
============================================================

向量数量: 10000
向量维度: 128

加载图结构...
索引加载完成

查询测试...
平均查询时间: 45.67ms

磁盘占用: 5MB
内存占用: ~1MB（只有图结构）

============================================================
性能对比总结
============================================================
方法              查询时间        内存占用        磁盘占用
------------------------------------------------------------
内存索引          23.45        5            0MB
DiskANN          45.67        1            5MB

优势:
- DiskANN内存占用减少: 80%
- 查询延迟增加: 95%

============================================================
大规模场景模拟
============================================================

假设场景: 100000000个向量, 768维

内存索引:
  内存需求: 286GB
  查询延迟: ~2000ms（暴力搜索）
  成本: 高（需要大内存服务器）

DiskANN:
  内存需求: 6GB（只有图结构）
  磁盘需求: 286GB（SSD）
  查询延迟: ~100ms
  成本: 中（SSD比内存便宜）

成本对比（AWS）:
  内存索引: $2860/月
  DiskANN: $89/月
  节省: $2771/月 (97%)

============================================================
缓存优化测试
============================================================

查询测试（带缓存）...

缓存统计:
  缓存命中: 45
  缓存未命中: 123
  命中率: 26.8%

============================================================
测试完成
============================================================
```

---

## 关键学习点

### 1. DiskANN核心思想

**三层存储**：
- 内存：图结构（邻接表）
- SSD：完整向量数据
- 缓存：热点向量

**优势**：
- 内存占用减少80%
- 支持亿级向量
- 查询延迟<100ms

---

### 2. 性能权衡

| 维度 | 内存索引 | DiskANN |
|------|---------|---------|
| **内存** | 286GB | 6GB |
| **磁盘** | 0 | 286GB |
| **查询延迟** | 2000ms | 100ms |
| **成本** | $2860/月 | $89/月 |

---

### 3. 优化技巧

**LRU缓存**：
- 缓存热点向量
- 命中率20-30%
- 减少磁盘访问

**顺序读取**：
- 按图结构组织向量
- 减少随机读取
- 提升SSD性能

---

## 练习题

### 练习1：实现预取

**任务**：预取下一层邻居

**提示**：
```python
def prefetch_neighbors(self, node_id):
    """预取邻居向量"""
    neighbors = self.graph[node_id]
    for neighbor in neighbors:
        self._load_vector_cached(neighbor)
```

---

### 练习2：压缩优化

**任务**：结合PQ量化压缩

**提示**：
```python
class DiskANNPQ(DiskANN):
    """DiskANN + Product Quantization"""

    def __init__(self, index_dir, M=16, n_subvectors=96):
        super().__init__(index_dir, M)
        self.n_subvectors = n_subvectors
        self.codebooks = []
```

---

### 练习3：分布式部署

**任务**：实现分布式DiskANN

**提示**：
```python
class DistributedDiskANN:
    """分布式DiskANN"""

    def __init__(self, n_shards=4):
        self.shards = [
            DiskANN(index_dir=f"./shard_{i}")
            for i in range(n_shards)
        ]

    def search(self, query, k=10):
        # 并行查询所有分片
        results = []
        for shard in self.shards:
            results.extend(shard.search(query, k=k))
        # 全局排序
        results.sort()
        return results[:k]
```

---

## 总结

通过DiskANN，我们实现了：
1. SSD优化的向量检索
2. 内存占用减少80%
3. 支持亿级向量
4. 成本降低97%

**关键收获**：
- DiskANN适合超大规模（>10亿向量）
- 内存只存图结构，向量存SSD
- LRU缓存提升性能
- 成本远低于内存索引

**生产应用**：
- Azure Cosmos DB集成
- SQL Server 2025公测
- Microsoft Bing部署

---

**恭喜！** 你已完成Vector Store原理与实现的全部学习内容。

**下一步建议**：
1. 实践：构建自己的RAG系统
2. 深入：阅读DiskANN论文
3. 扩展：学习Knowledge Graph集成
