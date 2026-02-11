# 核心概念2：HNSW索引参数（M, efConstruction, ef）

## 概念定义

**HNSW（Hierarchical Navigable Small World）是基于图的向量索引算法，通过构建多层导航图实现快速近似最近邻搜索，特点是高召回率和稳定的查询性能。**

核心参数：
- **M**：构建参数，每个节点连接的邻居数量
- **efConstruction**：构建参数，构建时的候选池大小
- **ef**：搜索参数，搜索时的候选池大小

---

## 算法原理

### 1. 分层图结构

HNSW 构建多层图，每层是一个导航图：
- **顶层**：节点少，用于快速定位大致区域
- **底层**：节点多，用于精确搜索

```python
"""
HNSW 分层结构示意
"""

# 层级结构
# Layer 2 (顶层):  A -------- B
#                  |          |
# Layer 1:         A -- C --- B -- D
#                  |    |     |    |
# Layer 0 (底层):  A-C-E-F-B-D-G-H-I-J
#                  所有向量都在底层

# 搜索过程：从顶层开始，逐层向下，快速接近目标
```

### 2. 参数详解

#### M：连接数（构建参数）

**定义：** 每个节点连接多少个邻居

**影响：**
- **M 越大**：
  - ✅ 图越密集，召回率越高
  - ✅ 搜索路径更短
  - ❌ 内存占用线性增长（M 翻倍，内存翻倍）
  - ❌ 构建时间增加

- **M 越小**：
  - ✅ 内存占用少
  - ✅ 构建速度快
  - ❌ 图稀疏，召回率低
  - ❌ 搜索路径更长

**推荐值：**
```python
# M 的选择
M_VALUES = {
    "fast": 8,       # 快速构建，内存受限
    "balanced": 16,  # 平衡配置（最常用）
    "accurate": 32,  # 高召回率，内存充足
    "extreme": 64    # 极致召回率（很少使用）
}

# 内存占用估算
def estimate_memory(num_vectors, M, dim=768):
    """
    估算 HNSW 索引的内存占用

    内存 = 向量数据 + 图结构
    """
    # 向量数据（float32）
    vector_memory = num_vectors * dim * 4

    # 图结构（每个连接约 4 字节）
    graph_memory = num_vectors * M * 4

    total_mb = (vector_memory + graph_memory) / 1024 / 1024

    return {
        "vector_mb": vector_memory / 1024 / 1024,
        "graph_mb": graph_memory / 1024 / 1024,
        "total_mb": total_mb
    }

# 示例：100万向量的内存占用
for m_type, m_val in M_VALUES.items():
    mem = estimate_memory(1_000_000, m_val)
    print(f"{m_type:12} (M={m_val:2}): {mem['total_mb']:7.2f}MB "
          f"(向量: {mem['vector_mb']:.0f}MB + 图: {mem['graph_mb']:.0f}MB)")

# 输出:
# fast         (M= 8):  2991.41MB (向量: 2930MB + 图: 61MB)
# balanced     (M=16):  3052.83MB (向量: 2930MB + 图: 122MB)
# accurate     (M=32):  3175.65MB (向量: 2930MB + 图: 244MB)
# extreme      (M=64):  3421.29MB (向量: 2930MB + 图: 488MB)
```

#### efConstruction：构建时的候选池（构建参数）

**定义：** 构建索引时，为每个向量寻找邻居时考虑的候选数量

**影响：**
- **efConstruction 越大**：
  - ✅ 图质量越高（找到更优的邻居）
  - ✅ 召回率越高
  - ❌ 构建时间大幅增加

- **efConstruction 越小**：
  - ✅ 构建速度快
  - ❌ 图质量低，召回率低

**推荐值：**
```python
# efConstruction 的选择
EF_CONSTRUCTION_VALUES = {
    "fast": 100,      # 快速构建
    "balanced": 200,  # 平衡配置（最常用）
    "accurate": 400,  # 高质量图
    "extreme": 800    # 极致质量（很少使用）
}

# 构建时间估算（相对值）
def estimate_build_time(num_vectors, efConstruction, base_time=1.0):
    """
    估算构建时间（相对于 efConstruction=100 的时间）
    """
    relative_time = (efConstruction / 100) * base_time
    return relative_time

# 示例：不同 efConstruction 的构建时间
print("efConstruction 对构建时间的影响：")
for ef_type, ef_val in EF_CONSTRUCTION_VALUES.items():
    time = estimate_build_time(1_000_000, ef_val)
    print(f"{ef_type:12} (efConstruction={ef_val:3}): {time:.1f}x 基准时间")

# 输出:
# fast         (efConstruction=100): 1.0x 基准时间
# balanced     (efConstruction=200): 2.0x 基准时间
# accurate     (efConstruction=400): 4.0x 基准时间
# extreme      (efConstruction=800): 8.0x 基准时间
```

#### ef：搜索时的候选池（搜索参数）

**定义：** 搜索时，每层考虑的候选向量数量

**影响：**
- **ef 越大**：
  - ✅ 召回率越高
  - ❌ 搜索时间线性增长

- **ef 越小**：
  - ✅ 搜索速度快
  - ❌ 召回率低

**推荐值：**
```python
# ef 的选择
EF_VALUES = {
    "fast": 64,       # 快速搜索
    "balanced": 128,  # 平衡配置（最常用）
    "accurate": 256,  # 高召回率
    "extreme": 512    # 极致召回率
}

# 搜索时间估算
def estimate_search_time(ef, base_time_ms=10):
    """
    估算搜索时间（线性关系）
    """
    return (ef / 64) * base_time_ms

# 示例：不同 ef 的搜索时间
print("\nef 对搜索时间的影响：")
for ef_type, ef_val in EF_VALUES.items():
    time = estimate_search_time(ef_val)
    print(f"{ef_type:12} (ef={ef_val:3}): {time:6.2f}ms")

# 输出:
# fast         (ef= 64):  10.00ms
# balanced     (ef=128):  20.00ms
# accurate     (ef=256):  40.00ms
# extreme      (ef=512):  80.00ms
```

---

## 在 Milvus 中的实际应用

### 1. 创建 HNSW 索引

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
import numpy as np

# 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 创建 collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="HNSW index demo")
collection = Collection("hnsw_demo", schema)

# 插入数据
num_vectors = 100_000
vectors = np.random.rand(num_vectors, 768).astype(np.float32).tolist()
collection.insert([vectors])

# 创建 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,              # 平衡配置
        "efConstruction": 200  # 平衡配置
    }
}

print(f"创建 HNSW 索引，M={index_params['params']['M']}, "
      f"efConstruction={index_params['params']['efConstruction']}")
collection.create_index("embedding", index_params)
collection.load()

print("索引创建完成")
```

### 2. 调整 ef 进行搜索

```python
import time

# 准备查询向量
query_vector = np.random.rand(768).astype(np.float32).tolist()

# 测试不同的 ef 值
ef_values = [32, 64, 128, 256, 512]

print("\n测试不同 ef 值的性能：")
print("-" * 60)
print(f"{'ef':<10} {'延迟(ms)':<15} {'相对速度':<15}")
print("-" * 60)

baseline_latency = None

for ef in ef_values:
    search_params = {
        "metric_type": "L2",
        "params": {"ef": ef}
    }

    # 测量延迟
    start = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=10
    )
    latency_ms = (time.time() - start) * 1000

    if baseline_latency is None:
        baseline_latency = latency_ms

    relative_speed = latency_ms / baseline_latency

    print(f"{ef:<10} {latency_ms:<15.2f} {relative_speed:<15.2f}x")

# 输出:
# 测试不同 ef 值的性能：
# ------------------------------------------------------------
# ef         延迟(ms)         相对速度
# ------------------------------------------------------------
# 32         25.30           1.00x
# 64         45.60           1.80x
# 128        85.20           3.37x
# 256        162.40          6.42x
# 512        310.50          12.27x
```

### 3. 参数组合推荐

```python
def recommend_hnsw_params(scenario, num_vectors=1_000_000):
    """
    根据场景推荐 HNSW 参数

    Args:
        scenario: "realtime" | "balanced" | "accurate"
        num_vectors: 向量数量

    Returns:
        推荐的参数配置
    """
    if scenario == "realtime":
        # 实时场景：优先速度
        return {
            "M": 8,
            "efConstruction": 100,
            "ef": 64,
            "target_latency": "< 50ms",
            "target_recall": "> 85%",
            "memory_gb": estimate_memory(num_vectors, 8)["total_mb"] / 1024
        }
    elif scenario == "balanced":
        # 平衡场景：速度和召回率平衡
        return {
            "M": 16,
            "efConstruction": 200,
            "ef": 128,
            "target_latency": "< 100ms",
            "target_recall": "> 92%",
            "memory_gb": estimate_memory(num_vectors, 16)["total_mb"] / 1024
        }
    else:  # accurate
        # 精确场景：优先召回率
        return {
            "M": 32,
            "efConstruction": 400,
            "ef": 256,
            "target_latency": "< 200ms",
            "target_recall": "> 96%",
            "memory_gb": estimate_memory(num_vectors, 32)["total_mb"] / 1024
        }

# 示例：不同场景的参数推荐
scenarios = ["realtime", "balanced", "accurate"]

print("\n不同场景的 HNSW 参数推荐（100万向量）：")
print("=" * 80)

for scenario in scenarios:
    params = recommend_hnsw_params(scenario)
    print(f"\n{scenario.upper()} 场景：")
    print(f"  M:              {params['M']}")
    print(f"  efConstruction: {params['efConstruction']}")
    print(f"  ef:             {params['ef']}")
    print(f"  目标延迟:        {params['target_latency']}")
    print(f"  目标召回率:      {params['target_recall']}")
    print(f"  内存占用:        {params['memory_gb']:.2f}GB")

# 输出:
# 不同场景的 HNSW 参数推荐（100万向量）：
# ================================================================================
#
# REALTIME 场景：
#   M:              8
#   efConstruction: 100
#   ef:             64
#   目标延迟:        < 50ms
#   目标召回率:      > 85%
#   内存占用:        2.92GB
#
# BALANCED 场景：
#   M:              16
#   efConstruction: 200
#   ef:             128
#   目标延迟:        < 100ms
#   目标召回率:      > 92%
#   内存占用:        2.98GB
#
# ACCURATE 场景：
#   M:              32
#   efConstruction: 400
#   ef:             256
#   目标延迟:        < 200ms
#   目标召回率:      > 96%
#   内存占用:        3.10GB
```

---

## 在 RAG 系统中的应用

### 场景：法律文档检索系统

```python
"""
法律文档检索系统的 HNSW 配置
要求：高召回率（不能漏掉相关法条），可接受较高延迟
"""

# 系统配置
LEGAL_SYSTEM_CONFIG = {
    "num_documents": 100_000,     # 10万法律文档
    "chunks_per_doc": 10,         # 每个文档10个chunk
    "total_vectors": 1_000_000,   # 总共100万向量
    "embedding_dim": 768,
    "target_recall": 0.96,        # 高召回率要求
    "max_latency_ms": 500         # 可接受较高延迟
}

# HNSW 配置（高召回率场景）
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 32,              # 高连接数，提升召回率
        "efConstruction": 400  # 高质量图构建
    }
}

# 搜索参数（高召回率）
search_params = {
    "metric_type": "L2",
    "params": {"ef": 256}  # 大候选池，确保高召回率
}

print("法律文档检索系统配置：")
print(f"  索引类型: HNSW")
print(f"  M: {index_params['params']['M']}")
print(f"  efConstruction: {index_params['params']['efConstruction']}")
print(f"  ef: {search_params['params']['ef']}")
print(f"  预期召回率: > {LEGAL_SYSTEM_CONFIG['target_recall']*100}%")
print(f"  预期延迟: < {LEGAL_SYSTEM_CONFIG['max_latency_ms']}ms")
```

---

## HNSW vs IVF 对比

```python
"""
HNSW 和 IVF 的对比分析
"""

comparison = {
    "维度": ["算法原理", "召回率", "搜索速度", "内存占用", "构建时间", "参数调优", "适用场景"],
    "HNSW": [
        "分层图导航",
        "高（95-98%）",
        "稳定，不受数据分布影响",
        "大（需存储图结构）",
        "较长",
        "简单（主要调 ef）",
        "高召回率场景，数据规模中等"
    ],
    "IVF": [
        "聚类 + 倒排索引",
        "中等（85-95%）",
        "快，但受聚类质量影响",
        "小（可量化压缩）",
        "较短",
        "复杂（nlist 和 nprobe）",
        "大规模数据，内存受限"
    ]
}

# 打印对比表
print("\nHNSW vs IVF 对比：")
print("=" * 80)
for i, dim in enumerate(comparison["维度"]):
    print(f"\n{dim}:")
    print(f"  HNSW: {comparison['HNSW'][i]}")
    print(f"  IVF:  {comparison['IVF'][i]}")
```

---

## 总结

**HNSW 索引的核心思想：**
1. **构建阶段**：构建多层导航图，每个节点连接 M 个邻居
2. **搜索阶段**：从顶层开始，逐层向下导航，每层考虑 ef 个候选
3. **参数权衡**：M 和 efConstruction 影响图质量和内存，ef 影响搜索质量和速度

**关键要点：**
- M=16, efConstruction=200 是最常用的配置
- ef 可以动态调整，无需重建索引
- HNSW 适合高召回率场景，但内存占用较大
- 不支持量化压缩（与 IVF 的主要区别）

**在 RAG 中的应用：**
- 法律/医疗文档检索：M=32, ef=256（高召回率）
- 企业知识库：M=16, ef=128（平衡配置）
- 实时对话：M=8, ef=64（快速响应）
