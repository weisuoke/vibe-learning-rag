# 核心概念：HNSW 索引

深入理解 HNSW 索引的原理、实现和应用。

---

## 概念定义

**HNSW（Hierarchical Navigable Small World）索引是基于图结构的向量索引，通过构建多层导航图实现对数时间复杂度的近似最近邻搜索。**

**核心特点：**
- 多层图结构（层级导航）
- 贪心搜索策略
- 时间复杂度 O(log n)
- 适合大规模数据（> 100万向量）

---

## 算法原理

### 核心思想：层级导航

```
高速公路（稀疏，长距离）→ 省道（中等密度）→ 县道（稠密，短距离）

Layer 2: A ←→ E ←→ J (快速跳跃)
         ↓     ↓     ↓
Layer 1: A ←→ C ←→ E ←→ G ←→ J (逐步接近)
         ↓     ↓     ↓     ↓     ↓
Layer 0: A-B-C-D-E-F-G-H-I-J (精确定位)
```

### 小世界网络特性

**小世界网络：**
- 任意两个节点之间的距离很短（通常 < 6 跳）
- 局部聚类系数高（邻居之间也相互连接）

**HNSW 利用这个特性：**
- 顶层：稀疏连接，快速跳跃
- 底层：稠密连接，精确定位
- 搜索路径：O(log n) 跳

### 三个阶段

**阶段1：构建（增量）**
```
for each new vector:
    1. 随机决定层数 level
    2. 从顶层开始搜索插入位置
    3. 在每层建立连接（最多 M 个）
    4. 更新邻居的连接
```

**阶段2：插入**
```
1. 确定节点层数（指数分布）
2. 从入口点开始导航
3. 在每层找到最近邻
4. 建立双向连接
5. 修剪连接（保持 M 个最近邻）
```

**阶段3：检索**
```
1. 从顶层入口点开始
2. 在当前层贪心搜索
3. 找到局部最优后下降到下一层
4. 重复直到底层
5. 返回底层的最近邻
```

### 可视化

```
构建过程：

插入节点 A (level=2):
Layer 2: A
Layer 1: A
Layer 0: A

插入节点 B (level=0):
Layer 2: A
Layer 1: A
Layer 0: A ←→ B

插入节点 C (level=1):
Layer 2: A
Layer 1: A ←→ C
Layer 0: A ←→ B ←→ C

插入节点 D (level=0):
Layer 2: A
Layer 1: A ←→ C
Layer 0: A ←→ B ←→ C ←→ D

...

检索过程：

查询 Q，从 A 开始

Layer 2: A → E (E 更近)
         ↓
Layer 1: E → G (G 更近)
         ↓
Layer 0: G → H (H 更近) → I (I 更近) → 找到！
```

### 时间复杂度分析

**构建时间：**
```
T_build = O(n × log n × M × d)

其中：
- n: 向量数量
- M: 每层最大连接数
- d: 向量维度
```

**查询时间：**
```
T_query = O(log n × M × d)

推导：
- 层数：O(log n)
- 每层搜索：O(M) 个邻居
- 距离计算：O(d)
```

**实际性能：**
```
示例：100万向量，M=16，d=768
- 理论：log(1000000) × 16 × 768 ≈ 245,760 次计算
- 实际：约 10-15ms（比理论更快，因为贪心策略）
```

### 空间复杂度

```
S = S_vectors + S_graph
  = n×d×4 + n×M×avg_layers×4
  ≈ n×d×4 × (1 + M×avg_layers/d)
  ≈ n×d×4 × 1.5-2.0  (约 50-100% 额外开销)

其中：
- avg_layers ≈ 1/(1-p)，p 通常为 0.5
- avg_layers ≈ 2
```

---

## 手写实现（简化版）

### 核心数据结构

```python
import numpy as np
import random
from typing import List, Tuple, Set

class HNSWNode:
    """HNSW 节点"""
    def __init__(self, vector: np.ndarray, vec_id: int, level: int):
        self.vector = vector
        self.id = vec_id
        self.level = level
        # 每层的邻居列表
        self.neighbors = [[] for _ in range(level + 1)]

class SimpleHNSW:
    """
    HNSW 索引的简化实现
    演示核心原理
    """

    def __init__(self, M: int = 16, ef_construction: int = 200, ml: float = 1.0):
        """
        初始化索引

        参数：
        - M: 每层最大连接数
        - ef_construction: 构建时的搜索范围
        - ml: 层数分布参数
        """
        self.M = M
        self.M_max = M
        self.M_max0 = M * 2  # 底层可以有更多连接
        self.ef_construction = ef_construction
        self.ml = ml
        self.entry_point = None  # 入口节点
        self.nodes = []  # 所有节点

    def _get_random_level(self) -> int:
        """
        随机决定节点的层数
        使用指数分布
        """
        level = 0
        while random.random() < 0.5 and level < 16:
            level += 1
        return level

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算欧氏距离"""
        return np.linalg.norm(v1 - v2)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[HNSWNode],
        num_closest: int,
        layer: int
    ) -> List[HNSWNode]:
        """
        在指定层搜索最近邻

        参数：
        - query: 查询向量
        - entry_points: 入口节点列表
        - num_closest: 返回多少个最近邻
        - layer: 层数

        返回：
        - 最近邻节点列表
        """
        visited = set()
        candidates = []
        w = []

        # 初始化
        for point in entry_points:
            dist = self._distance(query, point.vector)
            candidates.append((-dist, point))
            w.append((dist, point))
            visited.add(point.id)

        import heapq
        heapq.heapify(candidates)
        heapq.heapify(w)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            # 检查邻居
            for neighbor in current.neighbors[layer]:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    dist = self._distance(query, neighbor.vector)

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return [node for _, node in sorted(w)]

    def _get_neighbors(
        self,
        candidates: List[Tuple[float, HNSWNode]],
        M: int
    ) -> List[HNSWNode]:
        """
        选择 M 个最近邻
        使用启发式策略
        """
        # 简化版：直接选择距离最近的 M 个
        candidates.sort(key=lambda x: x[0])
        return [node for _, node in candidates[:M]]

    def add(self, vector: np.ndarray, vec_id: int):
        """
        添加向量

        参数：
        - vector: 向量
        - vec_id: ID
        """
        # 1. 确定层数
        level = self._get_random_level()
        node = HNSWNode(vector, vec_id, level)

        if self.entry_point is None:
            # 第一个节点
            self.entry_point = node
            self.nodes.append(node)
            return

        # 2. 从顶层开始搜索
        nearest = [self.entry_point]
        for lc in range(self.entry_point.level, level, -1):
            nearest = self._search_layer(vector, nearest, 1, lc)

        # 3. 在每层建立连接
        for lc in range(min(level, self.entry_point.level), -1, -1):
            candidates = self._search_layer(
                vector,
                nearest,
                self.ef_construction,
                lc
            )

            # 选择邻居
            M = self.M_max0 if lc == 0 else self.M_max
            neighbors = self._get_neighbors(
                [(self._distance(vector, c.vector), c) for c in candidates],
                M
            )

            # 建立双向连接
            node.neighbors[lc] = neighbors
            for neighbor in neighbors:
                neighbor.neighbors[lc].append(node)

                # 修剪邻居的连接
                if len(neighbor.neighbors[lc]) > M:
                    neighbor.neighbors[lc] = self._get_neighbors(
                        [(self._distance(neighbor.vector, n.vector), n)
                         for n in neighbor.neighbors[lc]],
                        M
                    )

            nearest = candidates

        # 4. 更新入口点
        if level > self.entry_point.level:
            self.entry_point = node

        self.nodes.append(node)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        ef: int = 64
    ) -> List[Tuple[int, float]]:
        """
        搜索最近邻

        参数：
        - query: 查询向量
        - top_k: 返回结果数
        - ef: 搜索时的候选集大小

        返回：
        - [(id, distance), ...]
        """
        if self.entry_point is None:
            return []

        # 从顶层开始
        nearest = [self.entry_point]

        # 逐层下降
        for lc in range(self.entry_point.level, 0, -1):
            nearest = self._search_layer(query, nearest, 1, lc)

        # 在底层搜索
        nearest = self._search_layer(query, nearest, ef, 0)

        # 返回 top_k
        results = [
            (node.id, self._distance(query, node.vector))
            for node in nearest[:top_k]
        ]
        results.sort(key=lambda x: x[1])

        return results

    def get_stats(self):
        """获取索引统计信息"""
        if not self.nodes:
            return {}

        levels = [node.level for node in self.nodes]
        connections = [
            sum(len(node.neighbors[i]) for i in range(node.level + 1))
            for node in self.nodes
        ]

        return {
            "num_nodes": len(self.nodes),
            "max_level": max(levels),
            "avg_level": np.mean(levels),
            "avg_connections": np.mean(connections),
            "entry_point_level": self.entry_point.level if self.entry_point else 0
        }


# 使用示例
if __name__ == "__main__":
    import time

    # 参数
    dim = 128
    num_vectors = 10000
    M = 16
    ef_construction = 200

    # 生成测试数据
    print("生成测试数据...")
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)

    # 创建索引
    index = SimpleHNSW(M=M, ef_construction=ef_construction)

    # 插入
    print(f"插入 {num_vectors} 个向量...")
    start = time.time()
    for i, vector in enumerate(vectors):
        index.add(vector, i)
        if (i + 1) % 1000 == 0:
            print(f"  已插入 {i + 1}/{num_vectors}")
    build_time = time.time() - start

    print(f"\n构建耗时: {build_time:.2f}秒")

    # 统计信息
    stats = index.get_stats()
    print(f"\n索引统计:")
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  最大层数: {stats['max_level']}")
    print(f"  平均层数: {stats['avg_level']:.2f}")
    print(f"  平均连接数: {stats['avg_connections']:.1f}")

    # 搜索
    query = np.random.rand(dim).astype(np.float32)

    for ef in [32, 64, 128]:
        start = time.time()
        results = index.search(query, top_k=10, ef=ef)
        search_time = (time.time() - start) * 1000

        print(f"\n搜索结果 (ef={ef}):")
        print(f"  查询延迟: {search_time:.2f}ms")
        print(f"  Top 3:")
        for i, (vec_id, dist) in enumerate(results[:3], 1):
            print(f"    {i}. ID={vec_id}, 距离={dist:.4f}")
```

---

## Milvus 中的使用

### 创建 HNSW 索引

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# 1. 连接
connections.connect("default", host="localhost", port="19530")

# 2. 创建 Collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields)
collection = Collection("hnsw_example", schema)

# 3. 插入数据
num_vectors = 100000
vectors = np.random.rand(num_vectors, 768).tolist()
collection.insert([vectors])
collection.flush()

# 4. 创建 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,              # 每层最大连接数
        "efConstruction": 200  # 构建时搜索范围
    }
}
collection.create_index("embedding", index_params)

# 5. 加载
collection.load()

# 6. 搜索
query = np.random.rand(1, 768).tolist()
search_params = {
    "metric_type": "L2",
    "params": {"ef": 64}  # 搜索时候选集大小
}
results = collection.search(
    data=query,
    anns_field="embedding",
    param=search_params,
    limit=10
)

for hit in results[0]:
    print(f"ID: {hit.id}, 距离: {hit.distance:.4f}")
```

### 参数详解

**M（构建参数）：**
- **含义**：每层最大连接数
- **推荐值**：16
- **影响**：
  - 越大：召回率越高，但内存占用越大
  - 越小：内存占用越小，但召回率越低
- **选择建议**：
  - 通用场景：M=16
  - 高召回率：M=32
  - 内存受限：M=8

**efConstruction（构建参数）：**
- **含义**：构建时的搜索范围
- **推荐值**：200
- **影响**：
  - 越大：索引质量越高，但构建越慢
  - 越小：构建越快，但索引质量越低
- **选择建议**：
  - 标准：efConstruction=200
  - 高质量：efConstruction=500
  - 快速构建：efConstruction=100

**ef（搜索参数）：**
- **含义**：搜索时的候选集大小
- **推荐值**：64-128
- **影响**：
  - 越大：召回率越高，但速度越慢
  - 越小：速度越快，但召回率越低
- **调优策略**：
  ```python
  # 高召回率（> 95%）
  ef = 128

  # 平衡
  ef = 64

  # 低延迟
  ef = 32
  ```

---

## RAG 应用场景

### 场景1：大型知识库

**需求：**
- 100万篇文档
- 高并发（QPS > 500）
- 延迟 < 20ms

**方案：**
```python
# 参数选择
M = 16
efConstruction = 200
ef = 64

# 预期性能
# - 召回率：93%
# - P95 延迟：12ms
# - QPS：800+
# - 内存占用：~6GB
```

### 场景2：实时内容推荐

**需求：**
- 频繁增量插入
- 实时更新
- 低延迟

**方案：**
```python
# HNSW 支持增量插入
# 无需重建索引

# 插入性能
# - 单次插入：< 1ms
# - 批量插入（1000个）：< 1秒
```

---

## 性能分析

### 查询延迟

**实测数据（768维，M=16）：**

| 向量数量 | ef=32 | ef=64 | ef=128 | ef=256 |
|---------|-------|-------|--------|--------|
| 10万 | 3ms | 5ms | 8ms | 15ms |
| 50万 | 5ms | 8ms | 12ms | 20ms |
| 100万 | 6ms | 10ms | 15ms | 25ms |
| 500万 | 8ms | 12ms | 18ms | 30ms |

**关键发现：**
- 延迟增长缓慢（对数增长）
- 100万 vs 10万：延迟只增加 2倍
- 适合大规模数据

### 召回率

**实测数据：**

| ef | 召回率 |
|----|--------|
| 32 | 88-90% |
| 64 | 91-93% |
| 128 | 93-95% |
| 256 | 95-96% |

### 内存占用

```
内存 = 向量数据 + 图结构
     ≈ n×d×4 × (1 + M×2/d)
     ≈ n×d×4 × 1.5-2.0

示例：
- 100万向量，768维，M=16
- 向量数据：1000000 × 768 × 4 = 2.86GB
- 图结构：约 1.5GB
- 总内存：约 4.3GB
```

---

## 优点与缺点

### 优点

1. **查询速度快**
   - O(log n) 复杂度
   - P95 延迟 < 15ms

2. **支持增量插入**
   - 无需重建索引
   - 适合实时更新

3. **无需训练**
   - 增量构建
   - 立即可用

4. **可扩展性好**
   - 适合大规模数据
   - 延迟增长缓慢

### 缺点

1. **内存占用高**
   - 1.5-2x 向量大小
   - 不适合内存受限场景

2. **召回率略低**
   - 90-95%
   - 不如 FLAT 和 IVF_FLAT

3. **参数调优复杂**
   - M, efConstruction, ef
   - 需要测试验证

4. **构建时间较长**
   - 虽然无需训练
   - 但增量插入累计时间长

---

## 总结

**HNSW 的本质：**
- 基于图的层级导航
- 贪心搜索策略
- O(log n) 复杂度

**核心参数：**
- M: 16
- efConstruction: 200
- ef: 64-128

**适用场景：**
- 大规模数据（> 100万向量）
- 需要低延迟（< 20ms）
- 需要频繁增量插入
- 内存充足

**何时选择 HNSW：**
- 数据规模 > 100万
- QPS 要求 > 500
- 延迟要求 < 20ms
- 可接受 90-95% 召回率

---

**完成！** 你已经掌握了三种核心索引类型的原理和应用。

**返回：** [00_概览.md](./00_概览.md) 查看完整学习路径
