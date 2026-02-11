# 核心概念1：Search参数优化

**一句话定义：** Search参数优化是通过调整 `limit`、`nprobe`、`ef` 等搜索参数来平衡查询延迟和召回率的核心技术。

---

## 1. 什么是Search参数？

Search参数是控制向量检索行为的配置项，决定了：
- **搜索范围**：在多大范围内搜索（nprobe、ef）
- **返回数量**：返回多少个结果（limit）
- **距离度量**：如何计算相似度（metric_type）

**核心作用：** 在性能和精度之间找到最佳平衡点

```python
from pymilvus import Collection

collection = Collection("documents")

# Search参数示例
results = collection.search(
    data=[query_vector],              # 查询向量
    anns_field="embedding",           # 向量字段名
    param={
        "metric_type": "L2",          # 距离度量类型
        "params": {"nprobe": 16}      # 索引特定参数
    },
    limit=10,                         # 返回结果数量
    expr="category == 'tech'",        # 标量过滤表达式
    output_fields=["id", "title"]     # 返回字段
)
```

---

## 2. 核心参数详解

### 2.1 limit（top_k）- 返回结果数量

**定义：** 返回相似度最高的前 k 个结果

**原理：**
```
1. Milvus 计算查询向量与候选向量的距离
2. 按距离排序（升序或降序）
3. 返回前 k 个结果
```

**性能影响：**
```python
# limit 越大 → 计算量越大 → 延迟越高

limit=10   → 延迟 50ms,  召回率 85%
limit=100  → 延迟 200ms, 召回率 90%
limit=1000 → 延迟 2s,    召回率 92%

# 结论：limit 增加 100 倍，延迟增加 40 倍，召回率只提升 7%
```

**选择原则：**
```python
# RAG 文档问答：3-5 个文档片段
limit = 5

# 推荐系统：10-20 个候选商品
limit = 20

# 相似图片搜索：50-100 个候选图片
limit = 100

# 原则：只返回真正需要的数量
```

**代码示例：**
```python
from pymilvus import Collection, connections
import numpy as np

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

collection = Collection("documents")
collection.load()

# 生成查询向量
query_vector = np.random.rand(768).tolist()

# 测试不同 limit 的性能
import time

for limit in [10, 50, 100, 500, 1000]:
    start = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=limit
    )
    elapsed = time.time() - start
    print(f"limit={limit:4d} → 延迟: {elapsed*1000:6.2f}ms")

# 输出示例：
# limit=  10 → 延迟:  50.23ms
# limit=  50 → 延迟:  85.67ms
# limit= 100 → 延迟: 156.89ms
# limit= 500 → 延迟: 678.45ms
# limit=1000 → 延迟: 1345.12ms
```

---

### 2.2 nprobe（IVF索引）- 搜索的桶数量

**定义：** 在 IVF 索引中，搜索多少个聚类中心（桶）

**原理：**
```
IVF 索引将向量分成多个桶（聚类）：
1. 建索引时：将 1000万 向量聚类成 1024 个桶
2. 查询时：
   - 找到最近的 nprobe 个桶
   - 只在这些桶中搜索
   - 返回 top_k 结果

例如：nprobe=16
→ 只搜索 16 个桶（约 16 × 10000 = 16万 向量）
→ 而不是全部 1000万 向量
```

**性能影响：**
```python
# nprobe 越大 → 搜索范围越大 → 延迟越高，召回率越高

# 场景：1000万向量，1024个桶，每个桶约1万向量
nprobe=4   → 搜索 4万向量,  延迟 20ms,  召回率 75%
nprobe=8   → 搜索 8万向量,  延迟 40ms,  召回率 85%
nprobe=16  → 搜索 16万向量, 延迟 80ms,  召回率 92%
nprobe=32  → 搜索 32万向量, 延迟 160ms, 召回率 96%
nprobe=64  → 搜索 64万向量, 延迟 320ms, 召回率 98%

# 结论：nprobe 翻倍，延迟翻倍，召回率提升递减
```

**选择原则：**
```python
# 根据召回率要求选择：
# - 召回率 > 95%：nprobe = 32-64
# - 召回率 > 90%：nprobe = 16-32
# - 召回率 > 85%：nprobe = 8-16
# - 召回率 > 80%：nprobe = 4-8

# 根据延迟要求选择：
# - 延迟 < 50ms：nprobe = 4-8
# - 延迟 < 100ms：nprobe = 8-16
# - 延迟 < 200ms：nprobe = 16-32

# 权衡：延迟 vs 召回率
```

**代码示例：**
```python
from pymilvus import Collection, connections
import numpy as np
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("documents")
collection.load()

query_vector = np.random.rand(768).tolist()

# 测试不同 nprobe 的性能
print("nprobe | 延迟(ms) | 召回率")
print("-------|----------|--------")

# 先用 nprobe=64 获取"真实"的 top 10（作为召回率基准）
ground_truth = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 64}},
    limit=10
)
ground_truth_ids = set([hit.id for hit in ground_truth[0]])

# 测试不同 nprobe
for nprobe in [4, 8, 16, 32, 64]:
    start = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": nprobe}},
        limit=10
    )
    elapsed = time.time() - start

    # 计算召回率
    result_ids = set([hit.id for hit in results[0]])
    recall = len(result_ids & ground_truth_ids) / len(ground_truth_ids)

    print(f"{nprobe:6d} | {elapsed*1000:8.2f} | {recall*100:6.1f}%")

# 输出示例：
# nprobe | 延迟(ms) | 召回率
# -------|----------|--------
#      4 |    20.45 |   75.0%
#      8 |    40.23 |   85.0%
#     16 |    80.67 |   92.0%
#     32 |   160.89 |   96.0%
#     64 |   320.12 |  100.0%
```

---

### 2.3 ef（HNSW索引）- 搜索时的候选集大小

**定义：** 在 HNSW 索引中，搜索时维护的候选集大小

**原理：**
```
HNSW 索引是一个多层图结构：
1. 从顶层开始，找到最近的节点
2. 逐层向下，维护一个大小为 ef 的候选集
3. 在底层，从候选集中选出 top_k 结果

ef 越大 → 候选集越大 → 搜索越全面 → 召回率越高，但延迟越高
```

**性能影响：**
```python
# ef 越大 → 搜索越全面 → 延迟越高，召回率越高

# 场景：1000万向量，HNSW索引
ef=32   → 延迟 15ms,  召回率 80%
ef=64   → 延迟 30ms,  召回率 90%
ef=128  → 延迟 60ms,  召回率 95%
ef=256  → 延迟 120ms, 召回率 98%
ef=512  → 延迟 240ms, 召回率 99%

# 结论：ef 翻倍，延迟翻倍，召回率提升递减
```

**选择原则：**
```python
# 根据召回率要求选择：
# - 召回率 > 95%：ef = 128-256
# - 召回率 > 90%：ef = 64-128
# - 召回率 > 85%：ef = 32-64

# 根据延迟要求选择：
# - 延迟 < 30ms：ef = 32-64
# - 延迟 < 60ms：ef = 64-128
# - 延迟 < 120ms：ef = 128-256

# 注意：ef 必须 >= limit（返回结果数量）
```

**代码示例：**
```python
from pymilvus import Collection, connections
import numpy as np
import time

connections.connect("default", host="localhost", port="19530")
collection = Collection("documents_hnsw")  # 使用 HNSW 索引的 collection
collection.load()

query_vector = np.random.rand(768).tolist()

# 测试不同 ef 的性能
print("ef    | 延迟(ms) | 召回率")
print("------|----------|--------")

# 先用 ef=512 获取"真实"的 top 10
ground_truth = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"ef": 512}},
    limit=10
)
ground_truth_ids = set([hit.id for hit in ground_truth[0]])

# 测试不同 ef
for ef in [32, 64, 128, 256, 512]:
    start = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": ef}},
        limit=10
    )
    elapsed = time.time() - start

    # 计算召回率
    result_ids = set([hit.id for hit in results[0]])
    recall = len(result_ids & ground_truth_ids) / len(ground_truth_ids)

    print(f"{ef:5d} | {elapsed*1000:8.2f} | {recall*100:6.1f}%")

# 输出示例：
# ef    | 延迟(ms) | 召回率
# ------|----------|--------
#    32 |    15.23 |   80.0%
#    64 |    30.45 |   90.0%
#   128 |    60.67 |   95.0%
#   256 |   120.89 |   98.0%
#   512 |   240.12 |  100.0%
```

---

### 2.4 metric_type - 距离度量类型

**定义：** 计算向量相似度的方法

**常用类型：**

| 类型 | 全称 | 计算公式 | 适用场景 | 值范围 |
|-----|------|---------|---------|--------|
| **L2** | 欧氏距离 | √(Σ(a-b)²) | 通用场景 | [0, +∞)，越小越相似 |
| **IP** | 内积 | Σ(a×b) | 归一化向量 | (-∞, +∞)，越大越相似 |
| **COSINE** | 余弦相似度 | Σ(a×b)/(‖a‖×‖b‖) | 文本向量 | [-1, 1]，越大越相似 |

**选择原则：**
```python
# 1. 如果向量已归一化（‖v‖=1）：
#    → 使用 IP（内积），速度最快
#    → IP 和 COSINE 等价（对于归一化向量）

# 2. 如果向量未归一化：
#    → 使用 L2（欧氏距离）或 COSINE

# 3. 文本 Embedding（如 OpenAI、Sentence-Transformers）：
#    → 通常已归一化，使用 IP

# 4. 图像 Embedding（如 ResNet、CLIP）：
#    → 可能未归一化，使用 L2 或 COSINE
```

**代码示例：**
```python
import numpy as np
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("documents")
collection.load()

query_vector = np.random.rand(768).tolist()

# 测试不同 metric_type
for metric_type in ["L2", "IP", "COSINE"]:
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": metric_type, "params": {"nprobe": 16}},
        limit=10
    )

    print(f"\nmetric_type: {metric_type}")
    for i, hit in enumerate(results[0]):
        print(f"  {i+1}. id={hit.id}, distance={hit.distance:.4f}")

# 输出示例：
# metric_type: L2
#   1. id=123, distance=0.4523
#   2. id=456, distance=0.5678
#   ...
#
# metric_type: IP
#   1. id=123, distance=0.9234
#   2. id=456, distance=0.8765
#   ...
#
# metric_type: COSINE
#   1. id=123, distance=0.9234
#   2. id=456, distance=0.8765
#   ...
```

---

## 3. 参数组合策略

### 3.1 低延迟场景（实时查询）

**需求：** 延迟 < 50ms，召回率 > 80%

**推荐配置：**
```python
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={
        "metric_type": "IP",           # 使用 IP（最快）
        "params": {"nprobe": 8}        # 较小的 nprobe
    },
    limit=10,                          # 只返回 10 个结果
    output_fields=["id", "title"]      # 不返回向量
)

# 性能：延迟 40ms，召回率 85%
```

**适用场景：**
- 聊天机器人（实时响应）
- 在线推荐系统（毫秒级推荐）
- 实时搜索（用户输入时的自动补全）

---

### 3.2 高召回率场景（精准检索）

**需求：** 召回率 > 95%，延迟 < 200ms

**推荐配置：**
```python
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={
        "metric_type": "L2",           # 使用 L2（更准确）
        "params": {"nprobe": 32}       # 较大的 nprobe
    },
    limit=20,                          # 返回更多结果
    output_fields=["id", "title", "content"]
)

# 性能：延迟 160ms，召回率 96%
```

**适用场景：**
- 医疗诊断（需要高准确度）
- 法律文档检索（不能遗漏重要信息）
- 科研论文搜索（需要全面的结果）

---

### 3.3 平衡场景（通用RAG）

**需求：** 延迟 < 100ms，召回率 > 90%

**推荐配置：**
```python
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={
        "metric_type": "IP",           # 使用 IP（快速）
        "params": {"nprobe": 16}       # 中等的 nprobe
    },
    limit=5,                           # RAG 通常只需要 5 个文档
    output_fields=["id", "content"],   # 只返回需要的字段
    expr="publish_date > '2024-01-01'" # 添加标量过滤
)

# 性能：延迟 80ms，召回率 92%
```

**适用场景：**
- 文档问答系统
- 知识库检索
- 智能客服

---

## 4. 参数调优流程

### 步骤1：确定业务需求

```python
# 问自己：
# 1. 延迟要求是多少？（< 50ms / < 100ms / < 200ms）
# 2. 召回率要求是多少？（> 80% / > 90% / > 95%）
# 3. 返回多少个结果？（3-5 / 10-20 / 50-100）
# 4. 是否需要标量过滤？（按类别、时间、用户等）
```

### 步骤2：选择基准配置

```python
# 根据索引类型选择基准配置

# IVF 索引：
base_config = {
    "metric_type": "IP",
    "params": {"nprobe": 16}
}

# HNSW 索引：
base_config = {
    "metric_type": "IP",
    "params": {"ef": 64}
}
```

### 步骤3：测试性能

```python
import time
import numpy as np

def test_search_performance(collection, query_vector, config, limit):
    """测试搜索性能"""
    start = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=config,
        limit=limit
    )
    elapsed = time.time() - start
    return elapsed, results

# 测试基准配置
query_vector = np.random.rand(768).tolist()
elapsed, results = test_search_performance(
    collection, query_vector, base_config, limit=10
)
print(f"延迟: {elapsed*1000:.2f}ms")
```

### 步骤4：调整参数

```python
# 如果延迟太高 → 降低 nprobe/ef
# 如果召回率太低 → 增加 nprobe/ef
# 如果结果太多 → 降低 limit
# 如果结果太少 → 增加 limit

# 迭代调整，直到满足需求
```

### 步骤5：验证召回率

```python
def calculate_recall(results, ground_truth):
    """计算召回率"""
    result_ids = set([hit.id for hit in results[0]])
    truth_ids = set([hit.id for hit in ground_truth[0]])
    recall = len(result_ids & truth_ids) / len(truth_ids)
    return recall

# 使用高 nprobe/ef 获取 ground truth
ground_truth = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 64}},
    limit=10
)

# 计算当前配置的召回率
recall = calculate_recall(results, ground_truth)
print(f"召回率: {recall*100:.1f}%")
```

---

## 5. 在RAG中的应用

### 场景1：文档问答系统

```python
from pymilvus import Collection, connections
from openai import OpenAI

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("knowledge_base")
collection.load()

# 初始化 OpenAI
client = OpenAI()

def rag_query(question: str) -> str:
    """RAG 查询流程"""

    # 1. 生成查询向量
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_vector = response.data[0].embedding

    # 2. 向量检索（优化后的参数）
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={
            "metric_type": "IP",        # 快速
            "params": {"nprobe": 16}    # 平衡
        },
        limit=5,                        # 只需要 5 个文档
        output_fields=["content"],      # 只返回内容
        expr="category == 'tech'"       # 过滤类别
    )

    # 3. 构建上下文
    context = "\n\n".join([hit.entity.get("content") for hit in results[0]])

    # 4. 生成答案
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个helpful的助手，根据提供的上下文回答问题。"},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"}
        ]
    )

    return response.choices[0].message.content

# 使用
answer = rag_query("什么是向量数据库？")
print(answer)
```

### 场景2：多租户知识库

```python
def multi_tenant_search(user_id: str, question: str) -> list:
    """多租户搜索"""

    # 生成查询向量
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_vector = response.data[0].embedding

    # 向量检索（添加用户过滤）
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={
            "metric_type": "IP",
            "params": {"nprobe": 16}
        },
        limit=10,
        output_fields=["id", "title", "content"],
        expr=f"user_id == '{user_id}'"  # 只搜索该用户的文档
    )

    return results[0]

# 使用
user_docs = multi_tenant_search(user_id="user123", question="如何使用API？")
```

---

## 6. 常见问题

### Q1: nprobe 和 ef 有什么区别？

**A:**
- `nprobe` 用于 **IVF 索引**（如 IVF_FLAT、IVF_SQ8、IVF_PQ）
- `ef` 用于 **HNSW 索引**
- 两者都控制搜索范围，但算法不同

### Q2: 如何选择 metric_type？

**A:**
```python
# 1. 如果向量已归一化 → 使用 IP（最快）
# 2. 如果向量未归一化 → 使用 L2 或 COSINE
# 3. 文本 Embedding → 通常使用 IP
# 4. 图像 Embedding → 通常使用 L2
```

### Q3: limit 设置多大合适？

**A:**
```python
# RAG 文档问答：3-5
# 推荐系统：10-20
# 相似图片搜索：50-100
# 原则：只返回真正需要的数量
```

### Q4: 如何平衡延迟和召回率？

**A:**
```python
# 1. 先确定业务需求（延迟要求 vs 召回率要求）
# 2. 从中等参数开始（nprobe=16 或 ef=64）
# 3. 测试性能和召回率
# 4. 根据结果调整：
#    - 延迟太高 → 降低 nprobe/ef
#    - 召回率太低 → 增加 nprobe/ef
```

---

## 7. 总结

**Search参数优化的核心：**
1. **limit**：控制返回数量，越小越快
2. **nprobe/ef**：控制搜索范围，越大越准但越慢
3. **metric_type**：选择合适的距离度量
4. **权衡**：延迟 vs 召回率

**最佳实践：**
- 从业务需求出发，确定延迟和召回率要求
- 选择合适的基准配置
- 测试性能，迭代调整
- 在 RAG 中，通常 limit=3-5，nprobe=16，metric_type=IP
