# 核心概念 1：Strong 一致性（强一致性）

## 一句话定义

**Strong 一致性保证读取操作总是返回最新写入的数据，即使需要等待所有节点同步完成。**

---

## 详细解释

### 工作原理

Strong 一致性通过以下机制保证数据的最新性：

```
写入操作
   ↓
1. 数据写入主节点
   ↓
2. 同步到所有副本节点（等待确认）
   ↓
3. 所有节点确认完成
   ↓
读取操作（等待同步完成）
   ↓
返回最新数据
```

**关键特性：**
- 使用 **timestamp 机制**保证顺序
- 读取操作必须等待所有节点确认
- 保证线性一致性（Linearizability）

---

### 技术实现（Milvus 视角）

**基本用法：**

```python
from pymilvus import Collection, connections

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 使用 Strong 一致性查询
results = collection.search(
    data=[[0.1, 0.2, 0.3, ...]],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    consistency_level="Strong"  # 强一致性
)

# 保证：返回的结果包含所有已写入的数据
```

**完整示例：**

```python
from pymilvus import Collection, connections
import numpy as np

connections.connect("default", host="localhost", port="19530")
collection = Collection("test_collection")

# 1. 插入新数据
print("=== 插入数据 ===")
new_id = 100
new_embedding = np.random.rand(128).tolist()
new_text = "Strong 一致性测试文档"

collection.insert([[new_embedding], [new_text], [new_id]])
collection.flush()  # 确保数据持久化
print(f"✅ 插入数据 ID: {new_id}")

# 2. 立即查询（Strong 一致性）
print("\n=== 立即查询（Strong）===")
results = collection.query(
    expr=f"id == {new_id}",
    output_fields=["id", "text"],
    consistency_level="Strong"
)

if results:
    print(f"✅ 查询成功: {results[0]}")
    print("Strong 一致性保证立即可见")
else:
    print("❌ 查询失败（不应该发生）")
```

---

### 性能特征

**延迟：**
- P50: ~150ms
- P95: ~200ms
- P99: ~250ms
- **最高延迟**（需要等待所有节点）

**吞吐量：**
- ~130 QPS
- **最低吞吐量**（串行化写入）

**资源消耗：**
- CPU: 高（需要协调所有节点）
- 网络: 高（跨节点通信）
- 内存: 高（维护全局状态）

**性能对比：**

| 指标 | Strong | Bounded | Eventually |
|------|--------|---------|------------|
| P50 延迟 | 150ms | 80ms | 40ms |
| 吞吐量 | 130 QPS | 240 QPS | 475 QPS |
| 资源消耗 | 高 | 中 | 低 |

---

## 适用场景

### 场景1：实时文档问答

**需求：**
用户刚上传文档，立即提问，必须检索到新文档

**示例：**
```
时间线：
T0: 用户上传文档 "Milvus 2.4 新特性"
T1: 文档向量化，插入 Milvus
T2: 用户提问 "Milvus 2.4 有什么新特性？"
T3: 系统检索（必须用 Strong）

要求：T3 必须检索到 T1 插入的数据
```

**代码实现：**

```python
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

connections.connect("default", host="localhost", port="19530")
collection = Collection("knowledge_base")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. 用户上传文档
def upload_document(text: str):
    embedding = model.encode(text).tolist()
    collection.insert([[embedding], [text]])
    collection.flush()  # 确保持久化
    print("✅ 文档上传成功")

# 2. 用户立即提问
def immediate_query(query: str):
    query_embedding = model.encode(query).tolist()

    # 使用 Strong 保证能检索到刚上传的文档
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        consistency_level="Strong"  # 必须用 Strong
    )

    return results

# 使用示例
upload_document("Milvus 2.4 支持动态 Schema")
results = immediate_query("Milvus 2.4 新特性")
print(f"找到 {len(results[0])} 个结果")
```

---

### 场景2：金融/医疗 RAG

**需求：**
监管要求查询结果必须基于最新数据，不能有延迟

**示例：**
```
金融场景：
- 用户查询最新的交易记录
- 必须返回最新的数据（监管要求）
- 不能容忍任何延迟

医疗场景：
- 医生查询患者最新的检查报告
- 必须是最新数据（关乎生命安全）
- 不能有任何数据延迟
```

**代码实现：**

```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("financial_records")

def query_latest_transaction(user_id: int):
    """查询用户最新交易记录（金融场景）"""
    results = collection.query(
        expr=f"user_id == {user_id}",
        output_fields=["transaction_id", "amount", "timestamp"],
        consistency_level="Strong",  # 金融场景必须用 Strong
        limit=10
    )

    # 保证：返回的是最新数据
    return results

# 医疗场景
collection_medical = Collection("medical_reports")

def query_latest_report(patient_id: int):
    """查询患者最新检查报告（医疗场景）"""
    results = collection_medical.query(
        expr=f"patient_id == {patient_id}",
        output_fields=["report_id", "diagnosis", "timestamp"],
        consistency_level="Strong",  # 医疗场景必须用 Strong
        limit=5
    )

    return results
```

---

### 场景3：A/B 测试

**需求：**
需要精确控制哪些用户看到新版本知识库

**示例：**
```
A/B 测试场景：
- 50% 用户使用新版本知识库
- 50% 用户使用旧版本知识库
- 必须精确控制（不能有延迟导致的混乱）

要求：
- 用户分组后，立即生效
- 不能有延迟导致用户看到错误版本
```

**代码实现：**

```python
from pymilvus import Collection, connections
import random

connections.connect("default", host="localhost", port="19530")
collection_v1 = Collection("knowledge_base_v1")
collection_v2 = Collection("knowledge_base_v2")

def ab_test_query(user_id: int, query_embedding: list):
    """A/B 测试查询"""
    # 根据用户 ID 分组
    group = "A" if user_id % 2 == 0 else "B"

    if group == "A":
        # A 组：使用新版本
        collection = collection_v2
        print(f"用户 {user_id} 分配到 A 组（新版本）")
    else:
        # B 组：使用旧版本
        collection = collection_v1
        print(f"用户 {user_id} 分配到 B 组（旧版本）")

    # 使用 Strong 保证立即生效
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        consistency_level="Strong"  # 保证分组立即生效
    )

    return results, group
```

---

## 权衡分析

### 优势

✅ **数据准确性最高**
- 保证读取最新数据
- 无脏读风险
- 符合线性一致性

✅ **符合直觉**
- 写入即可见
- 行为可预测
- 易于理解和调试

✅ **适合关键场景**
- 金融/医疗等对准确性要求极高的场景
- 实时性要求高的场景
- 监管合规场景

---

### 劣势

❌ **查询延迟高**
- 延迟 ~150ms（比 Bounded 慢 46%）
- 需要等待所有节点同步
- 影响用户体验

❌ **吞吐量低**
- 吞吐量 ~130 QPS（比 Bounded 低 46%）
- 串行化写入
- 系统容量受限

❌ **资源消耗大**
- CPU 消耗高（协调所有节点）
- 网络消耗高（跨节点通信）
- 内存消耗高（维护全局状态）

❌ **可扩展性差**
- 节点越多，延迟越高
- 难以水平扩展
- 成本高

---

## 在 RAG 系统中的应用

### 典型流程

```
1. 用户上传新文档
   ↓
2. 文档切块 + Embedding
   ↓
3. 插入 Milvus (Strong 一致性)
   ↓
4. 用户立即提问
   ↓
5. 检索时使用 Strong 级别
   ↓
6. 保证能检索到刚插入的文档
```

### 完整代码示例

```python
"""
RAG 场景：文档上传后立即可查询
演示 Strong 一致性的完整应用
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
collection_name = "rag_strong_demo"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
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

# ===== 2. 上传文档 =====
print("=== 用户上传文档 ===")
document = """
Milvus 2.4 引入了动态 Schema 功能，允许在不重建 Collection 的情况下添加新字段。
这大大提升了系统的灵活性，特别适合需要频繁调整数据结构的场景。
"""

# 生成 Embedding
embedding = model.encode(document).tolist()

# 插入 Milvus
insert_result = collection.insert([[embedding], [document]])
collection.flush()  # 确保数据持久化

doc_id = insert_result.primary_keys[0]
print(f"✅ 文档上传成功，ID: {doc_id}\n")

# ===== 3. 立即查询 =====
print("=== 用户立即提问 ===")
query = "Milvus 2.4 的动态 Schema 是什么？"
query_embedding = model.encode(query).tolist()

# 使用 Strong 一致性
start_time = time.time()
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"],
    consistency_level="Strong"  # 保证能检索到刚上传的文档
)
query_time = (time.time() - start_time) * 1000

print(f"查询延迟: {query_time:.2f}ms")
print(f"找到 {len(results[0])} 个结果\n")

# 显示结果
for i, hit in enumerate(results[0]):
    print(f"结果 {i+1}:")
    print(f"  相似度: {hit.score:.4f}")
    print(f"  内容: {hit.entity.get('text')[:100]}...")
    print()

# ===== 4. 验证 =====
print("=== 验证 Strong 一致性 ===")
# 查询刚插入的文档
verify_results = collection.query(
    expr=f"id == {doc_id}",
    output_fields=["id", "text"],
    consistency_level="Strong"
)

if verify_results:
    print("✅ Strong 一致性验证成功")
    print(f"   刚插入的文档立即可查询")
else:
    print("❌ 验证失败（不应该发生）")

# ===== 5. 清理 =====
collection.release()
utility.drop_collection(collection_name)
connections.disconnect("default")
print("\n✅ 演示完成")
```

**运行输出示例：**
```
=== 初始化 RAG 系统 ===
✅ 初始化完成

=== 用户上传文档 ===
✅ 文档上传成功，ID: 448979873564958720

=== 用户立即提问 ===
查询延迟: 152.34ms
找到 1 个结果

结果 1:
  相似度: 0.8765
  内容: Milvus 2.4 引入了动态 Schema 功能，允许在不重建 Collection 的情况下添加新字段...

=== 验证 Strong 一致性 ===
✅ Strong 一致性验证成功
   刚插入的文档立即可查询

✅ 演示完成
```

---

## 与其他级别对比

| 维度 | Strong | Bounded | Eventually |
|------|--------|---------|------------|
| **数据新鲜度** | ✅✅✅ 最新 | ✅✅ 较新 | ✅ 可能旧 |
| **查询延迟** | ❌❌❌ ~150ms | ⚠️ ~80ms | ✅✅✅ ~40ms |
| **吞吐量** | ❌❌❌ ~130 QPS | ⚠️ ~240 QPS | ✅✅✅ ~475 QPS |
| **资源消耗** | ❌❌❌ 高 | ⚠️ 中 | ✅✅✅ 低 |
| **适用场景** | 实时问答 | 一般检索 | 批量分析 |
| **RAG 推荐** | 文档刚上传 | 日常查询 | 历史数据 |
| **使用占比** | 20% | 60% | 20% |

---

## 记忆要点

### 核心特征

🎯 **核心**：读取最新数据，不惜代价
- 等待所有节点同步完成
- 保证线性一致性
- 写入即可见

⚡ **代价**：性能换准确性
- 延迟最高（~150ms）
- 吞吐量最低（~130 QPS）
- 资源消耗最大

🔧 **使用**：`consistency_level="Strong"`
- 配合 `flush()` 确保数据持久化
- 仅在必要时使用（20% 场景）
- 不要盲目使用

### 适用场景速记

✅ **必须用 Strong：**
- 文档刚上传，立即提问
- 金融/医疗等关键场景
- A/B 测试（精确控制）
- 实时聊天（对话上下文）

❌ **不要用 Strong：**
- 日常检索（用 Bounded）
- 批量分析（用 Eventually）
- 历史数据查询（用 Eventually）
- 高并发场景（用 Bounded/Eventually）

### 一句话总结

**Strong 一致性是用性能换准确性的选择，适合 20% 的关键场景，不要盲目使用。**

---

## 下一步学习

完成 Strong 一致性后，建议：

1. **对比学习**
   - 阅读 Bounded 一致性（平衡选择）
   - 阅读 Eventually 一致性（性能优先）
   - 理解三者的权衡

2. **实践练习**
   - 运行完整代码示例
   - 对比三种级别的性能差异
   - 在自己的 RAG 项目中应用

3. **深入理解**
   - 阅读"反直觉点"（避免误区）
   - 阅读"实战代码"（动手实践）
   - 阅读"面试必问"（深入原理）
