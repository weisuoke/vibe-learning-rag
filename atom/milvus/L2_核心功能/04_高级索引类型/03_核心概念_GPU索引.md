# 核心概念：GPU 索引

> GPU 索引通过并行计算加速向量检索，是实时推荐和高并发场景的核心技术

---

## 概念定义

**GPU 索引**是利用 GPU（图形处理器）的大规模并行计算能力，加速向量距离计算和检索过程的索引类型。

**核心特点**：
- 并行计算：数千个核心同时工作
- 高带宽：显存带宽远超 CPU 内存
- 批处理优化：适合高并发场景

---

## 一、GPU 索引原理

### 1.1 硬件架构对比

#### CPU 架构（顺序执行）

```
CPU（如 Intel Xeon）
├─ 核心数：16-64 个
├─ 每核心：复杂，支持分支预测、乱序执行
├─ 内存带宽：~100 GB/s（DDR4）
└─ 适合：复杂逻辑、分支密集的任务

单核心处理流程：
向量1 → 计算距离 → 向量2 → 计算距离 → 向量3 → ...
```

#### GPU 架构（并行执行）

```
GPU（如 NVIDIA A100）
├─ CUDA 核心数：6912 个
├─ 每核心：简单，只做浮点运算
├─ 显存带宽：1600 GB/s（HBM2）
└─ 适合：数据并行、计算密集的任务

并行处理流程：
核心1 → 向量1 → 计算距离
核心2 → 向量2 → 计算距离
核心3 → 向量3 → 计算距离
...
核心6912 → 向量6912 → 计算距离
（同时进行）
```

### 1.2 为什么 GPU 适合向量检索？

**原因1：向量距离计算天然可并行**

```python
# 向量内积（L2距离的核心）
def dot_product(query, doc):
    result = 0
    for i in range(len(query)):
        result += query[i] * doc[i]  # 每个维度独立计算
    return result

# GPU 并行化
# 768 个线程同时计算 768 个维度的乘法
# 然后归约求和（O(log N) 时间）
```

**原因2：向量检索是计算密集型任务**

```python
# 1 亿个 768 维向量的检索
N = 100_000_000
D = 768

# 每次查询的浮点运算次数
operations = N * D * 2  # 乘法 + 加法
print(f"运算次数: {operations:,}")  # 153,600,000,000 次

# CPU（单核 10 GFLOPS）
cpu_time = operations / 10_000_000_000
print(f"CPU 时间: {cpu_time:.2f} 秒")  # 15.36 秒

# GPU（A100, 19.5 TFLOPS）
gpu_time = operations / 19_500_000_000_000
print(f"GPU 时间: {gpu_time * 1000:.2f} 毫秒")  # 7.88 毫秒

# 加速比
speedup = cpu_time / gpu_time
print(f"加速比: {speedup:.0f}x")  # 1950x
```

**原因3：高内存带宽**

```python
# 读取 1 亿个 768 维向量
data_size = 100_000_000 * 768 * 4  # float32
data_gb = data_size / (1024 ** 3)
print(f"数据量: {data_gb:.2f} GB")  # 286 GB

# CPU 内存带宽（DDR4）
cpu_bandwidth = 100  # GB/s
cpu_read_time = data_gb / cpu_bandwidth
print(f"CPU 读取时间: {cpu_read_time:.2f} 秒")  # 2.86 秒

# GPU 显存带宽（HBM2）
gpu_bandwidth = 1600  # GB/s
gpu_read_time = data_gb / gpu_bandwidth
print(f"GPU 读取时间: {gpu_read_time:.3f} 秒")  # 0.179 秒

# 带宽优势
bandwidth_ratio = gpu_bandwidth / cpu_bandwidth
print(f"带宽优势: {bandwidth_ratio}x")  # 16x
```

### 1.3 GPU 索引类型

#### GPU_IVF_FLAT

```python
# 特点：GPU 加速 + IVF 聚类 + 无压缩
index_params = {
    "index_type": "GPU_IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 1024  # 聚类中心数量
    }
}

# 优点：最快，精度高（无量化损失）
# 缺点：显存占用大（需要存储所有向量）
# 适用：实时检索，显存充足
```

#### GPU_IVF_PQ

```python
# 特点：GPU 加速 + IVF 聚类 + PQ 量化
index_params = {
    "index_type": "GPU_IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 1024,
        "m": 96,      # 子向量数量
        "nbits": 8    # 编码位数
    }
}

# 优点：快 + 省显存（压缩 8-32 倍）
# 缺点：精度略降（2-5%）
# 适用：大规模检索，显存有限
```

---

## 二、手写实现：GPU 索引模拟

### 2.1 CPU 顺序计算（基准）

```python
import numpy as np
import time

def cpu_search(query, vectors, k=10):
    """CPU 顺序计算向量距离"""
    N, D = vectors.shape
    distances = np.zeros(N)

    # 顺序计算每个向量的距离
    for i in range(N):
        distances[i] = np.linalg.norm(query - vectors[i])

    # 返回 Top-K
    top_k_indices = np.argsort(distances)[:k]
    return top_k_indices, distances[top_k_indices]

# 测试
N = 100_000  # 10 万个向量
D = 768
vectors = np.random.randn(N, D).astype(np.float32)
query = np.random.randn(D).astype(np.float32)

start = time.time()
indices, distances = cpu_search(query, vectors, k=10)
cpu_time = time.time() - start
print(f"CPU 时间: {cpu_time * 1000:.2f} ms")
# 输出: CPU 时间: ~500 ms
```

### 2.2 GPU 并行计算（NumPy 向量化模拟）

```python
def gpu_search_simulated(query, vectors, k=10):
    """GPU 并行计算（NumPy 向量化模拟）"""
    # NumPy 的向量化操作会利用 SIMD 指令
    # 模拟 GPU 的并行计算
    distances = np.linalg.norm(vectors - query, axis=1)

    # 返回 Top-K
    top_k_indices = np.argsort(distances)[:k]
    return top_k_indices, distances[top_k_indices]

# 测试
start = time.time()
indices, distances = gpu_search_simulated(query, vectors, k=10)
vectorized_time = time.time() - start
print(f"向量化时间: {vectorized_time * 1000:.2f} ms")
# 输出: 向量化时间: ~50 ms

speedup = cpu_time / vectorized_time
print(f"加速比: {speedup:.1f}x")
# 输出: 加速比: ~10x
```

### 2.3 GPU IVF 索引实现

```python
from sklearn.cluster import KMeans

class GPU_IVF_Index:
    """简化的 GPU IVF 索引实现"""

    def __init__(self, nlist=100):
        self.nlist = nlist
        self.centers = None
        self.buckets = None

    def build(self, vectors):
        """构建索引：聚类 + 分桶"""
        print(f"聚类中心数: {self.nlist}")

        # 1. K-Means 聚类
        kmeans = KMeans(n_clusters=self.nlist, random_state=42)
        labels = kmeans.fit_predict(vectors)
        self.centers = kmeans.cluster_centers_

        # 2. 分桶：将向量分配到最近的聚类中心
        self.buckets = {i: [] for i in range(self.nlist)}
        for i, label in enumerate(labels):
            self.buckets[label].append(i)

        # 统计每个桶的大小
        bucket_sizes = [len(self.buckets[i]) for i in range(self.nlist)]
        print(f"平均桶大小: {np.mean(bucket_sizes):.0f}")
        print(f"最大桶大小: {np.max(bucket_sizes)}")
        print(f"最小桶大小: {np.min(bucket_sizes)}")

    def search(self, query, vectors, nprobe=10, k=10):
        """搜索：找到最近的 nprobe 个桶，然后在桶内搜索"""
        # 1. 找到最近的 nprobe 个聚类中心
        center_distances = np.linalg.norm(self.centers - query, axis=1)
        nearest_centers = np.argsort(center_distances)[:nprobe]

        # 2. 收集候选向量
        candidates = []
        for center_id in nearest_centers:
            candidates.extend(self.buckets[center_id])

        print(f"候选向量数: {len(candidates)} / {len(vectors)}")

        # 3. 在候选向量中搜索（GPU 并行计算）
        candidate_vectors = vectors[candidates]
        distances = np.linalg.norm(candidate_vectors - query, axis=1)

        # 4. 返回 Top-K
        top_k_local = np.argsort(distances)[:k]
        top_k_global = [candidates[i] for i in top_k_local]

        return top_k_global, distances[top_k_local]

# 测试
index = GPU_IVF_Index(nlist=100)
index.build(vectors)

start = time.time()
indices, distances = index.search(query, vectors, nprobe=10, k=10)
ivf_time = time.time() - start
print(f"IVF 时间: {ivf_time * 1000:.2f} ms")
# 输出: IVF 时间: ~5 ms

speedup = cpu_time / ivf_time
print(f"相比 CPU 加速比: {speedup:.0f}x")
# 输出: 相比 CPU 加速比: ~100x
```

### 2.4 批处理优化

```python
def gpu_batch_search(queries, vectors, k=10):
    """GPU 批处理搜索（多个查询同时处理）"""
    batch_size = len(queries)
    N, D = vectors.shape

    # 批量计算距离矩阵
    # queries: (batch_size, D)
    # vectors: (N, D)
    # distances: (batch_size, N)
    distances = np.linalg.norm(
        queries[:, np.newaxis, :] - vectors[np.newaxis, :, :],
        axis=2
    )

    # 批量返回 Top-K
    top_k_indices = np.argsort(distances, axis=1)[:, :k]
    top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)

    return top_k_indices, top_k_distances

# 测试批处理
batch_queries = np.random.randn(100, D).astype(np.float32)

start = time.time()
batch_indices, batch_distances = gpu_batch_search(batch_queries, vectors, k=10)
batch_time = time.time() - start
print(f"批处理时间: {batch_time * 1000:.2f} ms")
print(f"平均每个查询: {batch_time / 100 * 1000:.2f} ms")
# 输出: 平均每个查询: ~1 ms

# 批处理加速比
single_time = vectorized_time * 100
batch_speedup = single_time / batch_time
print(f"批处理加速比: {batch_speedup:.1f}x")
# 输出: 批处理加速比: ~50x
```

---

## 三、RAG 应用场景

### 3.1 实时推荐系统

**场景**：电商平台的商品推荐
- 用户数：1 亿
- 商品数：1000 万
- QPS：1000（高峰期）
- 延迟要求：< 50ms

**解决方案**：GPU_IVF_FLAT

```python
from pymilvus import Collection, connections

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 创建 Collection
collection = Collection("product_recommendations")

# 创建 GPU 索引
index_params = {
    "index_type": "GPU_IVF_FLAT",
    "metric_type": "IP",  # 内积（推荐场景常用）
    "params": {"nlist": 2048}
}
collection.create_index("embedding", index_params)
collection.load()

# 批处理搜索（充分利用 GPU）
def recommend_batch(user_embeddings, top_k=10):
    """批量推荐"""
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    results = collection.search(
        data=user_embeddings,  # batch_size=100
        anns_field="embedding",
        param=search_params,
        limit=top_k
    )
    return results

# 性能测试
import time
batch_size = 100
user_embeddings = [[0.1] * 768 for _ in range(batch_size)]

start = time.time()
results = recommend_batch(user_embeddings, top_k=10)
elapsed = time.time() - start

print(f"批处理时间: {elapsed * 1000:.2f} ms")
print(f"平均每个用户: {elapsed / batch_size * 1000:.2f} ms")
print(f"QPS: {batch_size / elapsed:.0f}")
# 输出: QPS: ~2000（满足需求）
```

### 3.2 文档问答系统

**场景**：企业知识库问答
- 文档数：5000 万
- 向量维度：768（BERT）
- QPS：100
- 延迟要求：< 100ms

**解决方案**：GPU_IVF_PQ（平衡性能与成本）

```python
# 创建 GPU_IVF_PQ 索引
index_params = {
    "index_type": "GPU_IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 4096,  # 聚类中心数
        "m": 96,        # 子向量数（768 / 8 = 96）
        "nbits": 8      # 编码位数
    }
}

collection = Collection("knowledge_base")
collection.create_index("embedding", index_params)
collection.load()

# 搜索
def search_documents(query_embedding, top_k=10):
    """搜索相关文档"""
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 64}  # 搜索 64 个聚类
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "title"]
    )
    return results[0]

# 性能分析
query = [0.1] * 768
start = time.time()
docs = search_documents(query, top_k=10)
elapsed = time.time() - start

print(f"查询延迟: {elapsed * 1000:.2f} ms")
# 输出: 查询延迟: ~15 ms（满足需求）

# 存储成本分析
N = 50_000_000
D = 768

# 原始存储
original_gb = N * D * 4 / (1024 ** 3)
print(f"原始存储: {original_gb:.2f} GB")  # 143 GB

# PQ 压缩后
m = 96
pq_gb = N * m * 1 / (1024 ** 3)
print(f"PQ 存储: {pq_gb:.2f} GB")  # 18 GB

# 成本节省
savings = (original_gb - pq_gb) / original_gb * 100
print(f"存储节省: {savings:.1f}%")  # 87.5%
```

### 3.3 图像搜索

**场景**：图片搜索引擎
- 图片数：1 亿
- 向量维度：2048（ResNet）
- QPS：500
- 延迟要求：< 100ms

**解决方案**：GPU_IVF_FLAT + 批处理

```python
# 创建索引
index_params = {
    "index_type": "GPU_IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 8192}
}

collection = Collection("image_search")
collection.create_index("embedding", index_params)
collection.load()

# 批处理搜索
def search_images_batch(query_embeddings, top_k=20):
    """批量图片搜索"""
    search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
    results = collection.search(
        data=query_embeddings,  # batch_size=50
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_url", "title"]
    )
    return results

# 性能测试
batch_size = 50
query_embeddings = [[0.1] * 2048 for _ in range(batch_size)]

start = time.time()
results = search_images_batch(query_embeddings, top_k=20)
elapsed = time.time() - start

print(f"批处理时间: {elapsed * 1000:.2f} ms")
print(f"平均每个查询: {elapsed / batch_size * 1000:.2f} ms")
print(f"QPS: {batch_size / elapsed:.0f}")
# 输出: QPS: ~1000（满足需求）
```

---

## 四、性能优化技巧

### 4.1 参数调优

```python
# nlist 选择（聚类中心数量）
N = 50_000_000  # 向量数量
nlist = int(4 * np.sqrt(N))  # 经验公式
print(f"推荐 nlist: {nlist}")  # 28284

# nprobe 选择（搜索的聚类数）
nprobe = max(16, int(nlist * 0.05))  # 搜索 5% 的聚类
print(f"推荐 nprobe: {nprobe}")  # 1414

# 召回率 vs 延迟权衡
nprobe_values = [16, 32, 64, 128, 256]
for nprobe in nprobe_values:
    # 测试不同 nprobe 的召回率和延迟
    # 找到满足召回率要求的最小 nprobe
    pass
```

### 4.2 批处理优化

```python
# 单个查询（慢）
for query in queries:
    result = collection.search([query], ...)  # 每次都有开销

# 批处理（快）
results = collection.search(queries, ...)  # 一次处理多个查询

# 最优 batch size
# - 太小：GPU 利用率低
# - 太大：延迟增加
# 推荐：50-200
```

### 4.3 显存管理

```python
# 检查显存使用
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# 显存不足时的策略
# 1. 使用 GPU_IVF_PQ（压缩）
# 2. 减少 nlist（减少聚类中心）
# 3. 分批加载数据
```

---

## 五、GPU 索引的局限性

### 5.1 不适合的场景

```python
# 场景1：小批量查询（batch_size < 10）
# GPU 启动开销大，CPU 可能更快

# 场景2：低维向量（< 128 维）
# GPU 并行优势不明显

# 场景3：数据不在 GPU 上
# 数据传输时间可能超过计算时间
```

### 5.2 成本考虑

```python
# GPU 成本（NVIDIA A100）
gpu_cost_per_hour = 3.0  # 美元/小时

# CPU 成本（16核）
cpu_cost_per_hour = 0.5  # 美元/小时

# 成本效益分析
# 如果 QPS < 50，CPU 可能更经济
# 如果 QPS > 100，GPU 更经济
```

---

## 总结

### GPU 索引的核心价值

1. **并行计算**：数千个核心同时工作，加速 10-100 倍
2. **高带宽**：显存带宽是 CPU 内存的 16 倍
3. **批处理**：适合高并发场景，充分利用 GPU

### 适用场景

- ✅ 实时推荐（QPS > 100）
- ✅ 图像搜索（高维向量）
- ✅ 文档问答（大规模检索）
- ❌ 小批量查询（batch_size < 10）
- ❌ 低维向量（< 128 维）

### 下一步

- **实战练习**：[09_实战代码_场景1_GPU索引实战](./09_实战代码_场景1_GPU索引实战.md)
- **对比学习**：[04_核心概念_量化索引](./04_核心概念_量化索引.md)
- **综合应用**：[12_实战代码_场景4_RAG性能优化](./12_实战代码_场景4_RAG性能优化.md)
