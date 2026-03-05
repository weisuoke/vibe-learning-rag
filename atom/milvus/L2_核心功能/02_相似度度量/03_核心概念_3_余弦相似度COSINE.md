# 核心概念3：余弦相似度COSINE（Cosine Similarity）

> 余弦相似度测量向量之间的夹角，只关注方向而忽略长度，是文本语义检索的首选度量

---

## 一句话定义

**余弦相似度（Cosine Similarity）通过计算两个向量夹角的余弦值来度量相似性，值越大表示越相似，特别适合归一化向量和文本语义检索场景。**

---

## 数学定义

### 公式

对于两个n维向量 **A = (a₀, a₁, ..., aₙ₋₁)** 和 **B = (b₀, b₁, ..., bₙ₋₁)**：

```
COSINE(A, B) = (A·B) / (||A|| × ||B||)
             = Σ(aᵢ×bᵢ) / (√Σ(aᵢ²) × √Σ(bᵢ²))
```

### 几何意义

余弦相似度测量的是两个向量之间的夹角：

```
COSINE(A, B) = cos(θ)
```

其中 `θ` 是两个向量之间的夹角。

**物理意义**：
- **COSINE = 1**：夹角0°，向量同向（完全相似）
- **COSINE = 0**：夹角90°，向量正交（无关）
- **COSINE = -1**：夹角180°，向量反向（完全相反）

---

## 核心特征

### 1. 值域特征

- **值域**：[-1, 1]
- **相似性**：值越大，越相似
- **1表示**：完全同向（最相似）
- **0表示**：正交（无关）
- **-1表示**：完全反向（最不相似）

### 2. 只关注方向

**关键洞察**：余弦相似度忽略向量长度，只看方向

```python
import numpy as np

# 两个向量：方向相同，长度不同
vec_a = np.array([1.0, 2.0, 3.0])
vec_b = np.array([2.0, 4.0, 6.0])  # vec_a的2倍

# 余弦相似度
cosine = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
print(f"余弦相似度: {cosine:.4f}")  # 1.0000（完全相似）

# 即使长度不同，方向相同就认为相似
```

### 3. 与归一化的关系

**归一化后**：COSINE = IP（内积）

```python
# 归一化向量
vec_a_norm = vec_a / np.linalg.norm(vec_a)
vec_b_norm = vec_b / np.linalg.norm(vec_b)

# 余弦相似度
cosine = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# 内积（归一化后）
ip = np.dot(vec_a_norm, vec_b_norm)

print(f"COSINE: {cosine:.6f}")  # 1.000000
print(f"IP: {ip:.6f}")          # 1.000000
# 完全相同！
```

**来源**：[Milvus Official Docs](https://milvus.io/docs/metric.md)

> "If you use IP to calculate similarities between embeddings, you must normalize your embeddings. After normalization, the inner product equals cosine similarity."

---

## 详细解释

### 工作原理（步骤拆解）

**步骤1：对齐向量**

```
向量A: [1, 2, 3]
向量B: [2, 3, 4]
```

**步骤2：计算点积（分子）**

```
点积: 1×2 + 2×3 + 3×4 = 2 + 6 + 12 = 20
```

**步骤3：计算向量长度（分母）**

```
||A|| = √(1² + 2² + 3²) = √14 ≈ 3.742
||B|| = √(2² + 3² + 4²) = √29 ≈ 5.385
```

**步骤4：计算余弦相似度**

```
COSINE = 20 / (3.742 × 5.385) ≈ 0.9746
```

### Python实现

```python
import numpy as np

def cosine_similarity(a, b):
    """
    计算余弦相似度

    参数:
        a: numpy数组，向量A
        b: numpy数组，向量B

    返回:
        float: 余弦相似度
    """
    # 方法1：手动计算
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = dot_product / (norm_a * norm_b)

    return cosine

# 方法2：使用sklearn
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# 示例
vec_a = np.array([1.0, 2.0, 3.0])
vec_b = np.array([2.0, 3.0, 4.0])

print(f"余弦相似度（手动）: {cosine_similarity(vec_a, vec_b):.4f}")  # 0.9746

# sklearn需要2D数组
vec_a_2d = vec_a.reshape(1, -1)
vec_b_2d = vec_b.reshape(1, -1)
print(f"余弦相似度（sklearn）: {sklearn_cosine(vec_a_2d, vec_b_2d)[0][0]:.4f}")  # 0.9746
```

---

## Milvus中的余弦相似度

### 创建Collection时指定

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="余弦相似度示例")
collection = Collection(name="cosine_demo", schema=schema)

# 创建索引（指定COSINE度量）
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # ← 指定余弦相似度（默认值）
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)

print("✅ Collection创建成功，使用余弦相似度度量")
```

### 检索时使用

```python
# 加载Collection
collection.load()

# 准备查询向量
query_embedding = np.array([[0.1, 0.2, 0.3, ...]])  # 128维

# 检索参数
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 100}
}

# 执行检索
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=["id"]
)

# 输出结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 余弦相似度: {hit.distance:.4f}")
```

---

## 何时使用余弦相似度

### ✅ 适用场景

#### 1. 文本语义检索（最常见）

```python
# 场景：文档问答系统
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 文档Embedding（已归一化）
doc_embedding = model.encode("Milvus是一个开源向量数据库")

# 检查归一化
norm = np.linalg.norm(doc_embedding)
print(f"向量长度: {norm:.6f}")  # 接近1.0

# ✅ 使用COSINE（推荐）
```

**原因**：
- 文本Embedding通常已归一化
- 只关注语义方向，不关注文档长度
- COSINE是NLP应用的标准度量

**来源**：[Zilliz Blog](https://zilliz.com/blog/similarity-metrics-for-vector-search)

> "Cosine similarity is primarily used in NLP applications. The main thing that cosine similarity measures is the difference in semantic orientation."

---

#### 2. 高维稀疏数据

```python
# 场景：TF-IDF文档向量
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Milvus向量数据库",
    "向量检索系统",
    "语义搜索引擎"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# TF-IDF向量通常稀疏且高维
# ✅ 使用COSINE（推荐）
```

**原因**：
- 高维稀疏数据中，大部分维度为0
- COSINE对稀疏数据计算高效
- 只关注非零维度的方向

---

#### 3. 归一化向量（默认选择）

```python
# 场景：任何归一化Embedding模型
embedding = model.encode("文本")

# 检查是否归一化
if abs(np.linalg.norm(embedding) - 1.0) < 1e-6:
    print("已归一化 → 使用COSINE")
    # ✅ COSINE是最佳选择
```

**原因**：
- 归一化后，向量长度都是1
- COSINE简化为点积，计算更快
- Milvus默认度量就是COSINE

**来源**：[IBM watsonx.data Guide (2025)](https://community.ibm.com/community/user/blogs/divya13/2025/01/08/in-memory-indexes-for-floating-point-embeddings)

> "**Cosine Similarity (COSINE):** Compares the angle between two vectors. Ideal for normalized vectors. **Default metric in Milvus.**"

---

### ❌ 不适用场景

#### 1. 向量长度有业务含义

```python
# 场景：用户行为强度
user_behavior_1 = np.array([5, 10, 3])   # 低活跃度
user_behavior_2 = np.array([50, 100, 30]) # 高活跃度

# 余弦相似度
cosine = np.dot(user_behavior_1, user_behavior_2) / (
    np.linalg.norm(user_behavior_1) * np.linalg.norm(user_behavior_2)
)
print(f"COSINE: {cosine:.4f}")  # 1.0000（认为相似）

# ❌ COSINE忽略了活跃度差异
# ✅ 应使用IP或L2
```

**原因**：COSINE只看方向，忽略长度（活跃度）差异

---

#### 2. 图像像素强度比较

```python
# 场景：图像Embedding（像素强度）
image_1 = np.array([10, 20, 30, ...])   # 低亮度
image_2 = np.array([100, 200, 300, ...]) # 高亮度

# 余弦相似度
cosine = cosine_similarity(image_1, image_2)
print(f"COSINE: {cosine:.4f}")  # 1.0000（认为相似）

# ❌ COSINE忽略了亮度差异
# ✅ 应使用L2
```

**来源**：[Zilliz Blog](https://zilliz.com/blog/similarity-metrics-for-vector-search)

> "It's probably not suitable when you have data where the magnitude of the vectors is important and should be taken into account when determining similarity. For example, it is not appropriate for comparing the similarity of image embeddings based on pixel intensities."

---

## 与其他度量的对比

### COSINE vs L2

| 维度 | 余弦相似度COSINE | L2距离 |
|-----|-----------------|--------|
| **考虑因素** | 只考虑方向 | 方向 + 长度 |
| **归一化向量** | 推荐 ✅ | 不推荐 |
| **未归一化向量** | 可用 | 推荐 ✅ |
| **值域** | [-1, 1] | [0, +∞) |
| **相似性** | 越大越相似 | 越小越相似 |
| **计算成本** | 中 | 中 |
| **文本检索** | 推荐 ✅ | 不推荐 |

**关系公式**（归一化向量）：

```
L2² = 2(1 - COSINE)
```

---

### COSINE vs IP

| 维度 | 余弦相似度COSINE | 内积IP |
|-----|-----------------|--------|
| **考虑因素** | 只考虑方向 | 方向 + 长度 |
| **归一化向量** | 等价于IP ✅ | 推荐 ✅ |
| **未归一化向量** | 推荐 ✅ | 可用 |
| **值域** | [-1, 1] | (-∞, +∞) |
| **相似性** | 越大越相似 | 越大越相似 |
| **计算成本** | 中（需归一化） | 低 |

**关键区别**：
- 归一化向量：COSINE = IP（结果相同）
- 未归一化向量：COSINE忽略长度，IP考虑长度

---

## RAG应用场景

### 场景1：文档语义检索

**需求**：根据用户问题检索相关文档

```python
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 加载Embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 用户问题
query = "如何优化Milvus检索性能？"
query_embedding = model.encode(query)

# 使用COSINE检索
collection = Collection("documents")
collection.load()

search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
results = collection.search(
    data=[query_embedding],
    anns_field="doc_embedding",
    param=search_params,
    limit=10
)

# 输出相关文档
for hit in results[0]:
    print(f"文档ID: {hit.id}, 相似度: {hit.distance:.4f}")
```

**为什么使用COSINE**：
- 文本Embedding已归一化
- 只关注语义方向，不关注文档长度
- COSINE是文本检索的标准度量

---

### 场景2：多语言语义搜索

**需求**：跨语言文档检索

```python
# 多语言Embedding模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 中文查询
query_zh = "向量数据库"
query_embedding = model.encode(query_zh)

# 检索英文文档
search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
results = collection.search(
    data=[query_embedding],
    anns_field="doc_embedding",
    param=search_params,
    limit=10
)

# 输出跨语言匹配结果
for hit in results[0]:
    print(f"文档ID: {hit.id}, 语义相似度: {hit.distance:.4f}")
```

**为什么使用COSINE**：
- 多语言Embedding在同一语义空间
- COSINE捕捉跨语言语义方向
- 忽略语言表达长度差异

---

## 2025-2026最佳实践

### 1. COSINE是默认首选

**决策流程**：

```python
def select_metric():
    """度量选择决策树"""

    # 1. 检查数据类型
    if is_text_embedding():
        return "COSINE"  # 文本检索首选 ✅

    # 2. 检查是否归一化
    if is_normalized():
        return "COSINE"  # 归一化向量首选 ✅

    # 3. 检查是否关注长度
    if magnitude_matters():
        return "IP"  # 长度有意义用IP
    else:
        return "COSINE"  # 只关注方向用COSINE ✅
```

**来源**：[Milvus Official Docs](https://milvus.io/docs/metric.md)

> "**Default Metric Type**: COSINE (for FLOAT_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR, INT8_VECTOR)"

---

### 2. 归一化提升效率

```python
# 推荐：索引前归一化
embeddings = model.encode(documents)
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# 归一化后：COSINE简化为点积
# 计算效率提升
```

**来源**：[Milvus AI Quick Reference](https://milvus.io/ai-quick-reference/what-is-the-relationship-between-vector-normalization-and-the-choice-of-metric-ie-when-and-why-should-vectors-be-normalized-before-indexing)

> "Normalization is required when using cosine similarity and optional but beneficial for Euclidean distance. Normalization simplifies the calculation to a dot product, improving computational efficiency."

---

### 3. 80%场景使用COSINE

**2026年标准**：

```python
# 默认配置
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # 默认选择
    "params": {"M": 16, "efConstruction": 200}
}

# 除非有特殊需求，否则使用COSINE
```

**来源**：[IBM watsonx.data Guide (2025)](https://community.ibm.com/community/user/blogs/divya13/2025/01/08/in-memory-indexes-for-floating-point-embeddings)

> "**Pro Tip:** Use `COSINE` for normalized vectors unless your use case benefits specifically from another metric."

---

## 常见误区

### 误区1：COSINE总是比L2好 ❌

**错误观点**："COSINE是默认度量，应该总是使用COSINE"

**正确理解**：
- COSINE适合归一化向量和文本检索
- 未归一化向量且长度有意义时，应使用L2或IP
- 度量选择取决于数据特征和业务需求

---

### 误区2：归一化后COSINE=L2 ❌

**错误观点**："向量归一化后，COSINE和L2结果一样"

**正确理解**：
- 归一化后：**COSINE = IP**（内积），不是L2
- 归一化后：L2² = 2(1 - COSINE)（有单调关系）
- COSINE和L2排序结果相同，但分数不同

---

### 误区3：COSINE不能用于未归一化向量 ❌

**错误观点**："COSINE只能用于归一化向量"

**正确理解**：
- COSINE可以用于未归一化向量
- 未归一化时，COSINE自动忽略长度差异
- 归一化只是提升计算效率，不是必需的

---

## 学习检查清单

完成本节后，你应该能够：

- [ ] 理解余弦相似度的数学定义和几何意义
- [ ] 知道COSINE只关注方向，忽略长度
- [ ] 理解COSINE与归一化的关系
- [ ] 能够判断何时使用余弦相似度
- [ ] 知道COSINE是Milvus的默认度量
- [ ] 理解COSINE与L2、IP的区别
- [ ] 能够在Milvus中配置COSINE度量
- [ ] 知道2025-2026的最佳实践
- [ ] 避开常见的COSINE误区

---

## 参考资源

### 官方文档

- [Milvus Metric Types](https://milvus.io/docs/metric.md) - 官方度量类型文档
- [Milvus AI Quick Reference - Normalization](https://milvus.io/ai-quick-reference/what-is-the-relationship-between-vector-normalization-and-the-choice-of-metric-ie-when-and-why-should-vectors-be-normalized-before-indexing) - 归一化与度量关系

### 技术博客

- [Zilliz Blog - Similarity Metrics](https://zilliz.com/blog/similarity-metrics-for-vector-search) - 详细解释余弦相似度
- [IBM watsonx.data Guide (2025)](https://community.ibm.com/community/user/blogs/divya13/2025/01/08/in-memory-indexes-for-floating-point-embeddings) - 2025年最佳实践

---

## 下一步

- **汉明距离**：[03_核心概念_4_汉明距离HAMMING](./03_核心概念_4_汉明距离HAMMING.md) - 学习二值向量度量
- **实战代码**：[07_实战代码_场景1_基础度量对比](./07_实战代码_场景1_基础度量对比.md) - 对比L2/IP/COSINE
- **内积IP**：[03_核心概念_2_内积IP](./03_核心概念_2_内积IP.md) - 回顾内积度量

---

**返回：** [00_概览](./00_概览.md)
