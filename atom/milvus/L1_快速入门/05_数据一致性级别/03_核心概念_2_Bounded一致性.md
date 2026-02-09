# 核心概念 2：Bounded 一致性（有界一致性）

## 一句话定义

**Bounded 一致性允许读取操作容忍一定时间范围内的数据延迟，在性能和准确性之间取得平衡。**

---

## 详细解释

### 工作原理

Bounded 一致性通过设置可容忍的时间窗口来平衡性能和准确性：

```
写入操作
   ↓
1. 数据写入主节点
   ↓
2. 异步同步到副本节点
   ↓
3. 不等待所有节点确认
   ↓
读取操作（容忍 N 秒延迟）
   ↓
返回"不早于 N 秒前"的数据
```

**关键特性：**
- 使用 **guarantee_timestamp** 参数控制时间窗口
- 读取操作返回"不早于指定时间"的数据
- 平衡了 Strong 和 Eventually 的优缺点

---

### 技术实现（Milvus 视角）

**基本用法：**

```python
from pymilvus import Collection, connections
import time

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 使用 Bounded 一致性查询（容忍 10 秒延迟）
results = collection.search(
    data=[[0.1, 0.2, 0.3, ...]],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 10)  # 容忍 10 秒延迟
)

# 保证：返回的数据不早于 10 秒前
```

**完整示例：**

```python
from pymilvus import Collection, connections
import numpy as np
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("test_collection")

# 1. 插入新数据
print("=== 插入数据 ===")
new_id = 200
new_embedding = np.random.rand(128).tolist()
new_text = "Bounded 一致性测试文档"

collection.insert([[new_embedding], [new_text], [new_id]])
collection.flush()  # 确保数据持久化
write_time = time.time()
print(f"✅ 插入数据 ID: {new_id}")
print(f"写入时间: {write_time}")

# 2. 立即查询（Bounded，容忍 5 秒）
print("\n=== 立即查询（Bounded，容忍 5 秒）===")
results = collection.query(
    expr=f"id == {new_id}",
    output_fields=["id", "text"],
    consistency_level="Bounded",
    guarantee_timestamp=int(write_time - 5)  # 容忍 5 秒
)

if results:
    print(f"✅ 查询成功: {results[0]}")
    print("Bounded 可能立即可见（取决于同步速度）")
else:
    print("⚠️ 查询为空（同步速度 > 5 秒）")

# 3. 等待后查询
print("\n=== 5 秒后查询 ===")
time.sleep(5)
results_after = collection.query(
    expr=f"id == {new_id}",
    output_fields=["id", "text"],
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 5)
)

if results_after:
    print(f"✅ 查询成功: {results_after[0]}")
    print("等待后一定能查到")
```

---

### 性能特征

**延迟：**
- P50: ~80ms
- P95: ~112ms
- P99: ~145ms
- **比 Strong 快 46%**

**吞吐量：**
- ~240 QPS
- **比 Strong 高 85%**

**资源消耗：**
- CPU: 中（部分节点协调）
- 网络: 中（减少跨节点通信）
- 内存: 中（维护部分状态）

**性能对比：**

| 指标 | Strong | Bounded | Eventually |
|------|--------|---------|------------|
| P50 延迟 | 150ms | **80ms** | 40ms |
| 吞吐量 | 130 QPS | **240 QPS** | 475 QPS |
| 资源消耗 | 高 | **中** | 低 |
| 相对 Strong | 基准 | **+85% 吞吐** | +265% 吞吐 |

---

## 适用场景

### 场景1：知识库定期更新

**需求：**
知识库每天凌晨更新，白天查询可容忍几秒延迟

**示例：**
```
时间线：
T0: 凌晨 2:00，批量更新知识库
T1: 上午 9:00，用户开始查询
T2: 查询时容忍 10 秒延迟

特点：
- 更新不频繁（每天一次）
- 用户不期望实时性
- 可以容忍短暂延迟
```

**代码实现：**

```python
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("knowledge_base")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. 凌晨批量更新知识库
def batch_update_knowledge_base(documents: list):
    """批量更新知识库"""
    embeddings = [model.encode(doc).tolist() for doc in documents]
    collection.insert([embeddings, documents])
    collection.flush()
    print(f"✅ 更新了 {len(documents)} 个文档")

# 2. 白天用户查询（Bounded）
def search_knowledge_base(query: str, tolerance_seconds: int = 10):
    """查询知识库，容忍一定延迟"""
    query_embedding = model.encode(query).tolist()

    # 使用 Bounded，容忍 10 秒延迟
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        consistency_level="Bounded",
        guarantee_timestamp=int(time.time() - tolerance_seconds)
    )

    return results

# 使用示例
# 凌晨更新
batch_update_knowledge_base([
    "Milvus 2.4 新特性",
    "向量数据库最佳实践",
    "RAG 系统优化指南"
])

# 白天查询（容忍 10 秒）
results = search_knowledge_base("Milvus 新特性", tolerance_seconds=10)
print(f"找到 {len(results[0])} 个结果")
```

---

### 场景2：多租户 RAG 系统

**需求：**
不同租户的数据更新频率不同，使用 Bounded 平衡性能

**示例：**
```
租户 A：高频更新（每小时）→ 容忍 5 秒
租户 B：中频更新（每天）→ 容忍 30 秒
租户 C：低频更新（每周）→ 容忍 60 秒

策略：根据租户的更新频率动态调整时间窗口
```

**代码实现：**

```python
from pymilvus import Collection, connections
import time

connections.connect("default", host="localhost", port="19530")

# 租户配置
TENANT_CONFIG = {
    "tenant_a": {"update_freq": "hourly", "tolerance": 5},
    "tenant_b": {"update_freq": "daily", "tolerance": 30},
    "tenant_c": {"update_freq": "weekly", "tolerance": 60}
}

def search_multi_tenant(tenant_id: str, query_embedding: list):
    """多租户查询，根据租户配置调整时间窗口"""
    # 获取租户配置
    config = TENANT_CONFIG.get(tenant_id, {"tolerance": 10})
    tolerance = config["tolerance"]

    # 获取租户的 Collection
    collection = Collection(f"kb_{tenant_id}")

    # 使用 Bounded，根据租户配置调整时间窗口
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        consistency_level="Bounded",
        guarantee_timestamp=int(time.time() - tolerance)
    )

    print(f"租户 {tenant_id}: 容忍 {tolerance} 秒延迟")
    return results

# 使用示例
query_embedding = [0.1, 0.2, 0.3, ...]

# 租户 A：高频更新，容忍 5 秒
results_a = search_multi_tenant("tenant_a", query_embedding)

# 租户 B：中频更新，容忍 30 秒
results_b = search_multi_tenant("tenant_b", query_embedding)

# 租户 C：低频更新，容忍 60 秒
results_c = search_multi_tenant("tenant_c", query_embedding)
```

---

### 场景3：推荐系统

**需求：**
用户行为数据实时写入，推荐结果可容忍短暂延迟

**示例：**
```
用户行为：
- 浏览商品 A
- 点击商品 B
- 加入购物车 C

推荐系统：
- 实时更新用户画像
- 推荐结果可容忍 5-10 秒延迟
- 使用 Bounded 平衡性能
```

**代码实现：**

```python
from pymilvus import Collection, connections
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("user_behavior")

def update_user_behavior(user_id: int, behavior_embedding: list):
    """更新用户行为"""
    collection.insert([[behavior_embedding], [user_id]])
    collection.flush()
    print(f"✅ 更新用户 {user_id} 的行为")

def recommend_items(user_id: int, tolerance_seconds: int = 10):
    """推荐商品，容忍一定延迟"""
    # 查询用户最近的行为
    results = collection.query(
        expr=f"user_id == {user_id}",
        output_fields=["behavior_embedding"],
        consistency_level="Bounded",
        guarantee_timestamp=int(time.time() - tolerance_seconds),
        limit=10
    )

    # 基于行为推荐商品
    # ...（推荐逻辑）

    return results

# 使用示例
user_id = 12345
behavior_embedding = [0.1, 0.2, 0.3, ...]

# 更新用户行为
update_user_behavior(user_id, behavior_embedding)

# 推荐商品（容忍 10 秒）
recommendations = recommend_items(user_id, tolerance_seconds=10)
```

---

## 权衡分析

### 优势

✅ **性能较好**
- 延迟比 Strong 低 46%
- 吞吐量比 Strong 高 85%
- 资源消耗适中

✅ **准确性可控**
- 延迟有上界（可配置）
- 不会无限延迟
- 行为可预测

✅ **适合大多数场景**
- 60% 的场景推荐使用
- 平衡性能和准确性
- 易于调优

---

### 劣势

❌ **可能读到旧数据**
- 在时间窗口内可能读到旧数据
- 需要根据场景调整时间窗口
- 行为不如 Strong 直观

❌ **需要调优时间窗口**
- 时间窗口设置需要经验
- 不同场景需要不同配置
- 调优成本高于 Strong/Eventually

❌ **复杂度高于 Eventually**
- 需要设置 guarantee_timestamp
- 理解成本高于 Eventually
- 代码略复杂

---

## 时间窗口选择指南

### 时间窗口的含义

**guarantee_timestamp** 参数的含义：

```python
import time

# 当前时间
current_time = time.time()

# 容忍 10 秒延迟
guarantee_timestamp = int(current_time - 10)

# 含义：返回"不早于 10 秒前"的数据
# 即：如果数据在 10 秒前写入，一定能查到
```

### 时间窗口选择表

| 场景 | 推荐窗口 | 理由 | 示例 |
|------|---------|------|------|
| 实时聊天 | 1-2 秒 | 用户期望快速响应 | 对话上下文 |
| 文档检索 | 5-10 秒 | 平衡性能和准确性 | 知识库查询 |
| 推荐系统 | 10-30 秒 | 容忍一定延迟 | 商品推荐 |
| 批量分析 | 30-60 秒 | 性能优先 | 数据分析 |

### 时间窗口设置示例

```python
from pymilvus import Collection
import time

collection = Collection("my_collection")

# 场景1：实时聊天（容忍 2 秒）
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 2)  # 2 秒
)

# 场景2：文档检索（容忍 10 秒）
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 10)  # 10 秒
)

# 场景3：批量分析（容忍 60 秒）
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 60)  # 60 秒
)
```

---

## 在 RAG 系统中的应用

### 典型流程

```
1. 知识库批量更新（夜间）
   ↓
2. 白天用户查询
   ↓
3. 使用 Bounded（容忍 10 秒）
   ↓
4. 性能提升 40%，准确性损失可忽略
```

### 完整代码示例

```python
"""
RAG 场景：知识库更新后的渐进可见
演示 Bounded 一致性的完整应用
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer
import time

# ===== 1. 初始化 =====
print("=== 初始化 RAG 系统 ===")
connections.connect("default", host="localhost", port="19530")

# 创建 Collection
collection_name = "rag_bounded_demo"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="update_time", dtype=DataType.INT64)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=collection_name, schema=schema)

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

# 加载 Embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ 初始化完成\n")

# ===== 2. 批量更新知识库 =====
print("=== 批量更新知识库 ===")
documents = [
    "Milvus 2.4 支持动态 Schema",
    "向量数据库性能优化技巧",
    "RAG 系统最佳实践"
]

embeddings = [model.encode(doc).tolist() for doc in documents]
update_times = [int(time.time())] * len(documents)

collection.insert([embeddings, documents, update_times])
collection.flush()
print(f"✅ 更新了 {len(documents)} 个文档\n")

# ===== 3. 用户查询（Bounded）=====
print("=== 用户查询（Bounded，容忍 10 秒）===")
query = "Milvus 动态 Schema"
query_embedding = model.encode(query).tolist()

# 使用 Bounded，容忍 10 秒延迟
start_time = time.time()
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text", "update_time"],
    consistency_level="Bounded",
    guarantee_timestamp=int(time.time() - 10)  # 容忍 10 秒
)
query_time = (time.time() - start_time) * 1000

print(f"查询延迟: {query_time:.2f}ms")
print(f"找到 {len(results[0])} 个结果\n")

# 显示结果
for i, hit in enumerate(results[0]):
    print(f"结果 {i+1}:")
    print(f"  相似度: {hit.score:.4f}")
    print(f"  内容: {hit.entity.get('text')}")
    print(f"  更新时间: {hit.entity.get('update_time')}")
    print()

# ===== 4. 性能对比 =====
print("=== 性能对比 ===")

# Bounded 查询
bounded_latencies = []
for _ in range(10):
    start = time.time()
    collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        consistency_level="Bounded",
        guarantee_timestamp=int(time.time() - 10)
    )
    bounded_latencies.append((time.time() - start) * 1000)

avg_bounded = sum(bounded_latencies) / len(bounded_latencies)
print(f"Bounded 平均延迟: {avg_bounded:.2f}ms")

# Strong 查询（对比）
strong_latencies = []
for _ in range(10):
    start = time.time()
    collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        consistency_level="Strong"
    )
    strong_latencies.append((time.time() - start) * 1000)

avg_strong = sum(strong_latencies) / len(strong_latencies)
print(f"Strong 平均延迟: {avg_strong:.2f}ms")

improvement = ((avg_strong - avg_bounded) / avg_strong * 100)
print(f"\n✅ Bounded 比 Strong 快 {improvement:.1f}%")

# ===== 5. 清理 =====
collection.release()
utility.drop_collection(collection_name)
connections.disconnect("default")
print("\n✅ 演示完成")
```

---

## 与其他级别对比

| 维度 | Strong | Bounded | Eventually |
|------|--------|---------|------------|
| **数据新鲜度** | ✅✅✅ 最新 | ✅✅ 较新（有界） | ✅ 可能旧 |
| **查询延迟** | ❌❌❌ ~150ms | ⚠️ **~80ms** | ✅✅✅ ~40ms |
| **吞吐量** | ❌❌❌ ~130 QPS | ⚠️ **~240 QPS** | ✅✅✅ ~475 QPS |
| **资源消耗** | ❌❌❌ 高 | ⚠️ **中** | ✅✅✅ 低 |
| **适用场景** | 实时问答 | **一般检索** | 批量分析 |
| **RAG 推荐** | 文档刚上传 | **日常查询** | 历史数据 |
| **使用占比** | 20% | **60%** | 20% |

---

## 记忆要点

### 核心特征

🎯 **核心**：容忍有限延迟，平衡性能
- 设置可容忍的时间窗口
- 返回"不早于 N 秒前"的数据
- 平衡 Strong 和 Eventually

⚡ **代价**：可能读到旧数据（有上界）
- 延迟适中（~80ms）
- 吞吐量较高（~240 QPS）
- 需要调优时间窗口

🔧 **使用**：`consistency_level="Bounded"`
- 配合 `guarantee_timestamp` 设置时间窗口
- 推荐用于 60% 的场景
- 是大多数场景的最佳选择

### 适用场景速记

✅ **推荐用 Bounded：**
- 日常检索（最常见）
- 知识库定期更新
- 多租户 RAG 系统
- 推荐系统
- 80% 的查询场景

❌ **不要用 Bounded：**
- 文档刚上传（用 Strong）
- 批量分析（用 Eventually）
- 需要绝对最新数据（用 Strong）
- 完全不在乎延迟（用 Eventually）

### 一句话总结

**Bounded 一致性是 60% 场景的最佳选择，通过容忍有限延迟实现性能和准确性的平衡。**

---

## 下一步学习

完成 Bounded 一致性后，建议：

1. **对比学习**
   - 对比 Strong 一致性（准确但慢）
   - 对比 Eventually 一致性（快但可能旧）
   - 理解三者的权衡

2. **实践练习**
   - 运行完整代码示例
   - 调整时间窗口参数
   - 在自己的 RAG 项目中应用

3. **深入理解**
   - 阅读"反直觉点"（避免误区）
   - 阅读"实战代码"（动手实践）
   - 阅读"面试必问"（深入原理）
