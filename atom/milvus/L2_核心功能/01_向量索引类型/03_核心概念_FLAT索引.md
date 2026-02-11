# 核心概念：FLAT 索引

深入理解 FLAT 索引的原理、实现和应用。

---

## 概念定义

**FLAT 索引是最简单的向量索引类型，本质上是"不使用索引"的暴力检索方法，通过遍历所有向量计算距离来找到最近邻。**

**核心特点：**
- 100% 召回率（精确检索）
- 无需参数调优
- 时间复杂度 O(n×d)
- 适合小规模数据（< 10万向量）

---

## 算法原理

### 工作流程

```
1. 存储阶段：
   向量按顺序存储在数组中
   [v1, v2, v3, ..., vn]

2. 检索阶段：
   for each vector in all_vectors:
       distance = calculate_distance(query, vector)
       candidates.append((vector_id, distance))

   return sort(candidates)[:k]
```

**可视化：**

```
查询向量: Q

所有向量:
v1 -----> 计算距离 d1
v2 -----> 计算距离 d2
v3 -----> 计算距离 d3
...
vn -----> 计算距离 dn

排序: d1, d2, d3, ..., dn
返回: Top K 个最小距离
```

### 时间复杂度分析

**查询时间：**
```
T(n, d, k) = O(n × d) + O(n × log n)
           = O(n × d)  (主导项)

其中：
- n: 向量数量
- d: 向量维度
- k: 返回结果数
```

**具体计算：**
```python
# 假设：100万向量，768维
n = 1000000
d = 768

# 距离计算次数
distance_calculations = n * d = 768,000,000

# 假设：每秒10亿次浮点运算
operations_per_second = 1e9

# 查询时间
query_time = distance_calculations / operations_per_second
           = 0.768 秒
```

### 空间复杂度

**存储空间：**
```
S(n, d) = n × d × sizeof(float)
        = n × d × 4 bytes

示例：
- 10万向量，768维
- 空间 = 100000 × 768 × 4 = 307,200,000 bytes ≈ 293 MB
```

**无额外开销：**
- FLAT 不需要额外的索引结构
- 只存储原始向量数据

### 为什么叫 FLAT

**FLAT = 平铺（Flat）存储**

```
向量在内存中平铺排列，无任何组织结构：

内存布局：
[v1_dim1, v1_dim2, ..., v1_dim768,
 v2_dim1, v2_dim2, ..., v2_dim768,
 ...
 vn_dim1, vn_dim2, ..., vn_dim768]

没有：
- 聚类结构（IVF）
- 图结构（HNSW）
- 树结构（KD-Tree）
```

---

## 手写实现

### 基础版本

```python
import numpy as np
from typing import List, Tuple

class SimpleFlatIndex:
    """
    FLAT 索引的简单实现
    演示核心原理
    """

    def __init__(self, dim: int):
        """
        初始化索引

        参数：
        - dim: 向量维度
        """
        self.dim = dim
        self.vectors = []  # 存储所有向量
        self.ids = []      # 存储对应的ID

    def add(self, vectors: List[List[float]], ids: List[int]):
        """
        添加向量

        参数：
        - vectors: 向量列表
        - ids: ID列表
        """
        assert len(vectors) == len(ids), "向量数量和ID数量必须相同"

        for vector, vec_id in zip(vectors, ids):
            assert len(vector) == self.dim, f"向量维度必须是 {self.dim}"
            self.vectors.append(vector)
            self.ids.append(vec_id)

    def search(self, query: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索最近邻

        参数：
        - query: 查询向量
        - top_k: 返回结果数

        返回：
        - [(id, distance), ...] 按距离排序
        """
        assert len(query) == self.dim, f"查询向量维度必须是 {self.dim}"

        # 计算所有距离
        distances = []
        for i, vector in enumerate(self.vectors):
            # 欧氏距离
            dist = np.linalg.norm(np.array(query) - np.array(vector))
            distances.append((self.ids[i], dist))

        # 排序并返回 top_k
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def size(self) -> int:
        """返回索引中的向量数量"""
        return len(self.vectors)


# 使用示例
if __name__ == "__main__":
    # 创建索引
    dim = 128
    index = SimpleFlatIndex(dim)

    # 添加向量
    vectors = np.random.rand(1000, dim).tolist()
    ids = list(range(1000))
    index.add(vectors, ids)

    # 搜索
    query = np.random.rand(dim).tolist()
    results = index.search(query, top_k=5)

    print(f"索引大小: {index.size()}")
    print(f"Top 5 结果:")
    for vec_id, dist in results:
        print(f"  ID: {vec_id}, 距离: {dist:.4f}")
```

### 优化版本（向量化计算）

```python
import numpy as np
from typing import List, Tuple

class OptimizedFlatIndex:
    """
    FLAT 索引的优化实现
    使用 NumPy 向量化计算加速
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None  # NumPy 数组
        self.ids = None      # NumPy 数组
        self.size = 0

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """
        添加向量（批量）

        参数：
        - vectors: shape (n, dim)
        - ids: shape (n,)
        """
        assert vectors.shape[1] == self.dim

        if self.vectors is None:
            self.vectors = vectors
            self.ids = ids
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, ids])

        self.size = len(self.ids)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索最近邻（向量化计算）

        参数：
        - query: shape (dim,)
        - top_k: 返回结果数

        返回：
        - [(id, distance), ...]
        """
        assert query.shape[0] == self.dim

        # 向量化计算所有距离（一次性）
        # distances.shape = (n,)
        distances = np.linalg.norm(self.vectors - query, axis=1)

        # 找到 top_k 个最小距离的索引
        top_k_indices = np.argpartition(distances, top_k)[:top_k]

        # 排序
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        # 返回结果
        results = [
            (int(self.ids[i]), float(distances[i]))
            for i in top_k_indices
        ]

        return results

    def batch_search(self, queries: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        批量搜索（进一步优化）

        参数：
        - queries: shape (num_queries, dim)
        - top_k: 返回结果数

        返回：
        - [[(id, distance), ...], ...] 每个查询的结果
        """
        assert queries.shape[1] == self.dim

        # 批量计算距离矩阵
        # distances.shape = (num_queries, n)
        distances = np.linalg.norm(
            self.vectors[np.newaxis, :, :] - queries[:, np.newaxis, :],
            axis=2
        )

        results = []
        for i in range(len(queries)):
            query_distances = distances[i]
            top_k_indices = np.argpartition(query_distances, top_k)[:top_k]
            top_k_indices = top_k_indices[np.argsort(query_distances[top_k_indices])]

            query_results = [
                (int(self.ids[j]), float(query_distances[j]))
                for j in top_k_indices
            ]
            results.append(query_results)

        return results


# 性能对比
if __name__ == "__main__":
    import time

    dim = 768
    num_vectors = 10000

    # 生成测试数据
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors)
    query = np.random.rand(dim).astype(np.float32)

    # 测试基础版本
    simple_index = SimpleFlatIndex(dim)
    simple_index.add(vectors.tolist(), ids.tolist())

    start = time.time()
    simple_results = simple_index.search(query.tolist(), top_k=10)
    simple_time = (time.time() - start) * 1000

    # 测试优化版本
    optimized_index = OptimizedFlatIndex(dim)
    optimized_index.add(vectors, ids)

    start = time.time()
    optimized_results = optimized_index.search(query, top_k=10)
    optimized_time = (time.time() - start) * 1000

    print(f"基础版本: {simple_time:.2f}ms")
    print(f"优化版本: {optimized_time:.2f}ms")
    print(f"加速比: {simple_time / optimized_time:.1f}x")
```

**性能对比：**
- 基础版本：~150ms（Python 循环）
- 优化版本：~15ms（NumPy 向量化）
- 加速比：10x

---

## Milvus 中的使用

### 创建 FLAT 索引

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 1. 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="FLAT 索引示例")

# 3. 创建 Collection
collection = Collection("flat_example", schema)

# 4. 插入数据
import numpy as np
vectors = np.random.rand(10000, 768).tolist()
collection.insert([vectors])
collection.flush()

# 5. 创建 FLAT 索引
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",  # 欧氏距离
    "params": {}  # FLAT 无需参数
}
collection.create_index(field_name="embedding", index_params=index_params)

# 6. 加载 Collection
collection.load()

# 7. 搜索
query_vector = np.random.rand(1, 768).tolist()
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param={"metric_type": "L2"},
    limit=10
)

# 8. 输出结果
for hit in results[0]:
    print(f"ID: {hit.id}, 距离: {hit.distance:.4f}")
```

### 参数说明

**FLAT 索引参数：**
```python
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",  # 或 "IP", "COSINE"
    "params": {}  # 无需参数
}
```

**距离度量：**
- `L2`: 欧氏距离（默认）
- `IP`: 内积（适合归一化向量）
- `COSINE`: 余弦相似度

**搜索参数：**
```python
search_params = {
    "metric_type": "L2",
    "params": {}  # 无需参数
}
```

### 适用场景

**✅ 推荐使用 FLAT 的场景：**

1. **小规模数据（< 10万向量）**
   ```python
   # 个人笔记应用
   num_docs = 500
   # FLAT 查询延迟：< 5ms
   ```

2. **需要 100% 召回率**
   ```python
   # 医疗诊断系统
   # 不能漏掉任何相关结果
   ```

3. **原型验证阶段**
   ```python
   # 快速验证 RAG 系统功能
   # 无需调优参数
   ```

4. **基准测试**
   ```python
   # 作为其他索引的性能对照
   # 100% 召回率作为 ground truth
   ```

**❌ 不推荐使用 FLAT 的场景：**

1. **大规模数据（> 10万向量）**
   - 查询延迟过高（> 100ms）
   - 推荐使用 IVF_FLAT 或 HNSW

2. **高 QPS 要求**
   - FLAT 无法支持高并发
   - 推荐使用 HNSW

3. **实时系统**
   - 延迟不可预测
   - 推荐使用 HNSW

---

## RAG 应用场景

### 场景1：个人笔记应用

**需求：**
- 500篇笔记
- 实时搜索
- 100% 召回率

**方案：**
```python
# 使用 FLAT 索引
index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {}}

# 性能：
# - 查询延迟：< 5ms
# - 召回率：100%
# - 内存占用：~1.5GB（500 × 768 × 4 bytes）
```

### 场景2：小型企业知识库

**需求：**
- 5000篇文档
- 10-50人使用
- 高精度要求

**方案：**
```python
# 使用 FLAT 索引
# 或者 IVF_FLAT（如果需要更低延迟）

# FLAT 性能：
# - 查询延迟：~15ms
# - 召回率：100%

# IVF_FLAT 性能：
# - 查询延迟：~8ms
# - 召回率：96%
```

### 场景3：原型验证

**需求：**
- 快速验证 RAG 系统
- 无需调优
- 功能优先

**方案：**
```python
# 第一阶段：使用 FLAT 验证功能
index_params = {"index_type": "FLAT", "metric_type": "L2", "params": {}}

# 第二阶段：数据增长后切换到 IVF_FLAT
# 第三阶段：大规模部署时切换到 HNSW
```

---

## 性能分析

### 查询延迟

**实测数据（768维向量）：**

| 向量数量 | 查询延迟 (P50) | 查询延迟 (P95) |
|---------|---------------|---------------|
| 100 | 0.5ms | 0.8ms |
| 1,000 | 2ms | 3ms |
| 10,000 | 15ms | 20ms |
| 100,000 | 150ms | 180ms |
| 1,000,000 | 1500ms | 1800ms |

**延迟增长：**
- 线性增长：延迟 ∝ 向量数量
- 10倍数据 → 10倍延迟

### 内存占用

**计算公式：**
```
内存 = 向量数量 × 维度 × 4 bytes

示例：
- 10万向量，768维
- 内存 = 100000 × 768 × 4 = 307MB
```

**无额外开销：**
- FLAT 只存储原始向量
- 没有索引结构的额外开销

### QPS（每秒查询数）

**单线程 QPS：**
```
QPS = 1000ms / 平均延迟(ms)

示例：
- 1000向量：QPS = 1000 / 2 = 500
- 10000向量：QPS = 1000 / 15 = 67
- 100000向量：QPS = 1000 / 150 = 7
```

**多线程 QPS：**
- 可以通过多线程提升 QPS
- 但延迟不会降低

---

## 优点与缺点

### 优点

**1. 100% 召回率**
- 精确检索，不会漏掉任何结果
- 适合高精度要求的场景

**2. 无需参数调优**
- 没有 nlist、nprobe、M、ef 等参数
- 开箱即用

**3. 实现简单**
- 代码简单，易于理解和维护
- 不容易出错

**4. 适合小规模数据**
- < 10万向量时，FLAT 可能比复杂索引更快
- 无索引构建开销

**5. 作为基准**
- 其他索引的性能对照
- 召回率的 ground truth

### 缺点

**1. 线性时间复杂度**
- O(n×d)，无法扩展到大规模数据
- 延迟随数据量线性增长

**2. 不适合大规模数据**
- > 10万向量时，延迟过高
- > 100万向量时，几乎不可用

**3. 无法支持高 QPS**
- 单次查询耗时长
- 并发能力有限

**4. 延迟不可预测**
- 延迟随数据量变化
- 难以保证 SLA

---

## 与其他索引对比

### FLAT vs IVF_FLAT

| 维度 | FLAT | IVF_FLAT |
|------|------|----------|
| 时间复杂度 | O(n×d) | O(nprobe×n/nlist×d) |
| 召回率 | 100% | 95-98% |
| 查询延迟 | 线性增长 | 次线性增长 |
| 参数调优 | 无需 | 需要（nlist, nprobe） |
| 适用规模 | < 10万 | 10万-100万 |
| 构建时间 | 无 | 需要训练 |

**何时从 FLAT 切换到 IVF_FLAT？**
- 向量数量 > 10万
- 查询延迟 > 50ms
- 可接受 2-5% 召回率损失

### FLAT vs HNSW

| 维度 | FLAT | HNSW |
|------|------|------|
| 时间复杂度 | O(n×d) | O(log n×d) |
| 召回率 | 100% | 90-95% |
| 查询延迟 | 高 | 低 |
| 内存占用 | 低 | 高（1.5-2x） |
| 适用规模 | < 10万 | > 100万 |
| 增量插入 | 支持 | 支持 |

**何时从 FLAT 切换到 HNSW？**
- 向量数量 > 100万
- 查询延迟要求 < 20ms
- 可接受 5-10% 召回率损失
- 内存充足

---

## 总结

**FLAT 索引的本质：**
- 最简单的向量检索方法
- 暴力遍历所有向量
- 100% 召回率

**核心特点：**
- 时间复杂度：O(n×d)
- 空间复杂度：O(n×d)
- 无需参数调优

**适用场景：**
- 小规模数据（< 10万向量）
- 需要 100% 召回率
- 原型验证阶段
- 基准测试

**何时切换到其他索引：**
- 数据规模 > 10万 → IVF_FLAT
- 数据规模 > 100万 → HNSW
- 查询延迟 > 50ms → IVF_FLAT 或 HNSW

---

**下一步：** [04_核心概念_IVF_FLAT索引.md](./04_核心概念_IVF_FLAT索引.md) - 学习聚类检索
