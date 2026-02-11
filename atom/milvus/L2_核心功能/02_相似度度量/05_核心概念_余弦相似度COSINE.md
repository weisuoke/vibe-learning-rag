# 核心概念：余弦相似度COSINE

## 一句话定义

**余弦相似度是测量两个向量方向一致性的度量方式，完全忽略向量幅度，值越大表示方向越一致、越相似。**

---

## 1. 数学原理

### 1.1 公式定义

对于两个n维向量 x = [x₁, x₂, ..., xₙ] 和 y = [y₁, y₂, ..., yₙ]：

```
COSINE(x, y) = (x · y) / (‖x‖ × ‖y‖)
             = Σ(xᵢ × yᵢ) / (√Σxᵢ² × √Σyᵢ²)
             = cos(θ)
```

其中θ是两向量的夹角。

### 1.2 手写实现

```python
import numpy as np

def cosine_similarity(x, y):
    """
    计算余弦相似度

    参数:
        x: 向量1
        y: 向量2

    返回:
        余弦相似度值（-1到1之间）
    """
    # 步骤1：计算点积
    dot_product = np.dot(x, y)

    # 步骤2：计算范数
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    # 步骤3：计算余弦相似度
    cosine = dot_product / (norm_x * norm_y)

    return cosine

# 示例
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 4.0])

similarity = cosine_similarity(x, y)
print(f"余弦相似度: {similarity:.4f}")  # 0.9926
```

---

## 2. 几何意义

### 2.1 角度解释

余弦相似度 = 两向量夹角的余弦值

```
cos(0°) = 1.0    # 方向完全相同
cos(45°) = 0.707  # 方向偏离45度
cos(90°) = 0.0    # 方向垂直
cos(180°) = -1.0  # 方向完全相反
```

**可视化：**
```
      y
     /
    /  θ
   /______ x

COSINE(x, y) = cos(θ)
```

### 2.2 忽略幅度

**关键特性：** 向量长度不影响余弦相似度

```python
# 两个向量
x = np.array([1, 0])
y = np.array([100, 0])  # y是x的100倍

# 余弦相似度
cosine = cosine_similarity(x, y)
print(f"余弦相似度: {cosine:.4f}")  # 1.0000（完全相同）

# 方向一致，长度不影响结果
```

---

## 3. 在Milvus中的使用

### 3.1 配置COSINE度量

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields=fields, description="COSINE度量示例")

# 创建Collection
collection = Collection(name="cosine_collection", schema=schema)

# 创建索引，指定COSINE度量
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",  # 使用余弦相似度
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

### 3.2 搜索时使用COSINE

```python
# 加载Collection
collection.load()

# 查询向量
query_vector = [[0.1, 0.2, 0.3, ...]]  # 768维

# 搜索参数
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}

# 执行搜索
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=["id"]
)

# 结果按COSINE值从大到小排序
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, COSINE: {hit.distance:.4f}")
```

### 3.3 COSINE的排序规则

**重要：COSINE值越大越相似**

```python
# 示例结果
# ID: 123, COSINE: 0.9876  ← 最相似（接近1）
# ID: 456, COSINE: 0.8765
# ID: 789, COSINE: 0.7654
# ...
# ID: 999, COSINE: 0.1234  ← 最不相似（接近0或负数）
```

---

## 4. RAG应用场景

### 4.1 场景1：语义搜索（最常用）

**适用情况：**
- Embedding模型输出归一化向量
- 只关注语义方向，不关注幅度
- 文档问答、知识库检索

**示例：文档语义搜索**

```python
import numpy as np
from pymilvus import connections, Collection

# 连接Milvus
connections.connect(host="localhost", port="19530")

# 模拟OpenAI Embedding（归一化）
def openai_embedding(text, dim=768):
    """模拟OpenAI Embedding（归一化向量）"""
    np.random.seed(hash(text) % 2**32)
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)  # 归一化

# 文档库
documents = [
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的重要分支",
    "神经网络是深度学习的基础",
]

# 生成Embedding
embeddings = [openai_embedding(doc) for doc in documents]

# 插入Milvus
collection = Collection("semantic_search")
collection.insert([documents, embeddings])

# 用户查询
query = "什么是机器学习"
query_emb = openai_embedding(query)

# 使用COSINE检索
results = collection.search(
    data=[query_emb],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)

# 结果按语义相似度排序
for hit in results[0]:
    print(f"[相似度: {hit.distance:.4f}] {hit.entity.get('text')}")
```

### 4.2 场景2：多语言语义匹配

**适用情况：**
- 跨语言语义搜索
- 多语言Embedding模型
- 只关注语义，不关注语言

**示例：中英文语义匹配**

```python
# 多语言Embedding
def multilingual_embedding(text, dim=768):
    """模拟多语言Embedding"""
    np.random.seed(hash(text) % 2**32)
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)

# 中英文文档
documents = {
    "en": ["Machine learning is a core technology of AI"],
    "zh": ["机器学习是人工智能的核心技术"],
}

# 生成Embedding
embeddings = {
    lang: [multilingual_embedding(doc) for doc in docs]
    for lang, docs in documents.items()
}

# 中文查询，匹配英文文档
query_zh = "什么是机器学习"
query_emb = multilingual_embedding(query_zh)

# COSINE能正确匹配跨语言语义
en_doc_emb = embeddings["en"][0]
similarity = np.dot(query_emb, en_doc_emb)
print(f"中英文语义相似度: {similarity:.4f}")  # 高相似度
```

### 4.3 场景3：去重与聚类

**适用情况：**
- 文档去重
- 内容聚类
- 相似内容检测

**示例：文档去重**

```python
# 文档去重阈值
DUPLICATE_THRESHOLD = 0.95

def is_duplicate(doc1_emb, doc2_emb, threshold=DUPLICATE_THRESHOLD):
    """判断两个文档是否重复"""
    similarity = np.dot(doc1_emb, doc2_emb)
    return similarity > threshold

# 文档库
documents = [
    "机器学习是AI的核心",
    "机器学习是人工智能的核心",  # 与第1个相似
    "深度学习是机器学习的分支",
]

embeddings = [openai_embedding(doc) for doc in documents]

# 检测重复
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if is_duplicate(embeddings[i], embeddings[j]):
            print(f"重复文档: {i} 和 {j}")
            print(f"  文档{i}: {documents[i]}")
            print(f"  文档{j}: {documents[j]}")
```

---

## 5. 完整实战代码

```python
"""
余弦相似度检索系统完整示例
演示：RAG系统中的语义搜索
"""

import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

# ===== 1. 连接Milvus =====
print("=== 连接Milvus ===")
connections.connect(alias="default", host="localhost", port="19530")
print("✓ 连接成功")

# ===== 2. 创建Collection =====
print("\n=== 创建Collection ===")

collection_name = "cosine_semantic_search"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields=fields, description="COSINE语义搜索")
collection = Collection(name=collection_name, schema=schema)
print(f"✓ 创建Collection: {collection_name}")

# ===== 3. 插入文档 =====
print("\n=== 插入文档 ===")

documents = [
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的重要分支",
    "神经网络是深度学习的基础",
    "自然语言处理研究计算机理解人类语言",
    "计算机视觉让机器能够理解图像",
    "强化学习通过奖励机制训练智能体",
    "迁移学习利用已有知识解决新问题",
    "生成对抗网络可以生成逼真的图像",
]

# 生成归一化Embedding
def generate_normalized_embedding(text, dim=128):
    """生成归一化Embedding"""
    np.random.seed(hash(text) % 2**32)
    vec = np.random.randn(dim)
    return (vec / np.linalg.norm(vec)).tolist()

embeddings = [generate_normalized_embedding(doc) for doc in documents]

# 验证归一化
norms = [np.linalg.norm(emb) for emb in embeddings]
print(f"向量范数（应该都接近1.0）: {norms[0]:.4f}")

entities = [documents, embeddings]
insert_result = collection.insert(entities)
print(f"✓ 插入 {len(documents)} 个文档")

# ===== 4. 创建COSINE索引 =====
print("\n=== 创建COSINE索引 ===")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 16}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("✓ 创建COSINE索引")

# ===== 5. 加载Collection =====
collection.load()
print("✓ Collection已加载")

# ===== 6. 语义搜索 =====
print("\n=== 语义搜索 ===")

queries = [
    "什么是机器学习",
    "深度学习的基础是什么",
    "如何让计算机理解图片",
]

for query_text in queries:
    print(f"\n查询: {query_text}")
    query_emb = [generate_normalized_embedding(query_text)]

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_emb,
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text"]
    )

    print("Top-3 结果:")
    for rank, hit in enumerate(results[0], 1):
        print(f"  {rank}. [相似度: {hit.distance:.4f}] {hit.entity.get('text')}")

# ===== 7. 对比不同度量 =====
print("\n=== 对比：COSINE vs L2 ===")

# 重建L2索引
collection.release()
collection.drop_index()

l2_index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 16}
}
collection.create_index(field_name="embedding", index_params=l2_index_params)
collection.load()

# 同样的查询
query_text = "什么是机器学习"
query_emb = [generate_normalized_embedding(query_text)]

l2_results = collection.search(
    data=query_emb,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)

print(f"\n查询: {query_text}")
print("\n用L2的结果:")
for rank, hit in enumerate(l2_results[0], 1):
    print(f"  {rank}. [L2距离: {hit.distance:.4f}] {hit.entity.get('text')}")

print("\n分析:")
print("- COSINE: 只看语义方向，更适合语义搜索")
print("- L2: 对归一化向量，排序与COSINE相同但数值不同")

# ===== 8. 清理 =====
collection.release()
print("\n✓ 完成")
```

---

## 6. COSINE的特点

### 6.1 优点

✅ **忽略幅度**
- 只关注方向，不受向量长度影响
- 适合语义搜索

✅ **值域直观**
- 值域：[-1, 1]
- 1表示完全相同，-1表示完全相反

✅ **归一化友好**
- 最适合归一化向量
- 与大多数Embedding模型兼容

✅ **语义清晰**
- 直接对应向量夹角
- 符合"相似"的直觉

### 6.2 缺点

❌ **计算稍慢**
- 需要计算范数
- 比IP慢10-20%

❌ **丢失幅度信息**
- 如果幅度有意义，COSINE会丢失
- 不适合需要考虑"强度"的场景

---

## 7. 何时使用COSINE

### 7.1 推荐使用的场景

✅ **语义搜索（首选）**
```python
# RAG系统、文档问答、知识库检索
metric_type = "COSINE"
```

✅ **归一化向量**
```python
# OpenAI、sentence-transformers等
if np.isclose(np.linalg.norm(embedding), 1.0):
    metric_type = "COSINE"
```

✅ **只关注方向的场景**
- 文本相似度
- 语义匹配
- 内容去重

✅ **不确定时的默认选择**
- 适用于90%的场景
- 最安全的选择

### 7.2 不推荐使用的场景

❌ **需要考虑幅度**
- 推荐系统（用IP）
- 幅度表示重要性的场景

❌ **物理距离概念**
- 地理坐标（用L2）
- 物理测量值

---

## 8. COSINE vs 其他度量

### 8.1 COSINE vs L2（归一化向量）

```python
# 归一化向量
x = np.array([0.6, 0.8])
y = np.array([0.8, 0.6])

# COSINE
cosine = np.dot(x, y)  # 0.96

# L2
l2 = np.linalg.norm(x - y)  # 0.2828

# 关系：L2² = 2(1 - COSINE)
print(f"L2² = {l2**2:.4f}")  # 0.0800
print(f"2(1-COSINE) = {2*(1-cosine):.4f}")  # 0.0800
```

**结论：** 归一化向量的L2和COSINE排序相同，但COSINE语义更清晰。

### 8.2 COSINE vs IP（归一化向量）

```python
# 归一化向量
x_norm = x / np.linalg.norm(x)
y_norm = y / np.linalg.norm(y)

# COSINE
cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# IP（归一化后）
ip = np.dot(x_norm, y_norm)

# 结果相同
assert np.isclose(cosine, ip)
```

**结论：** 归一化向量的IP = COSINE，但IP更快。

---

## 9. 性能优化

### 9.1 使用IP代替COSINE

```python
# 方案1：直接用COSINE
cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 方案2：归一化 + IP（更快10-20%）
x_norm = x / np.linalg.norm(x)
y_norm = y / np.linalg.norm(y)
ip = np.dot(x_norm, y_norm)  # 等价于COSINE

# 在Milvus中
# 1. 插入前归一化向量
# 2. 使用IP度量
# 3. 效果与COSINE相同，性能更好
```

### 9.2 批量归一化

```python
def normalize_batch(vectors):
    """批量归一化向量"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# 批量处理
vectors = np.random.randn(1000, 768)
normalized = normalize_batch(vectors)

# 验证
norms = np.linalg.norm(normalized, axis=1)
assert np.allclose(norms, 1.0)
```

---

## 10. 总结

### 核心要点

1. **COSINE = 方向一致性（忽略幅度）**
2. **值域：[-1, 1]，越大越相似**
3. **最适合语义搜索**
4. **归一化向量的标准选择**
5. **不确定时的默认选择**

### 记忆口诀

**COSINE测角度，只看方向不看长度，语义搜索首选，归一化向量标配。**

---

**下一步：** [09_实战代码_场景1_L2距离实战.md](./09_实战代码_场景1_L2距离实战.md) - 动手实践
