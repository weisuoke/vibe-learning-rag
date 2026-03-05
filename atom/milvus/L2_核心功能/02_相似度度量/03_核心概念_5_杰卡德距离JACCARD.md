# 核心概念5：杰卡德距离JACCARD（Jaccard Distance）

> 杰卡德距离测量集合之间的不相似度，基于交集与并集的比例，是二值向量集合相似度的标准度量

---

## 一句话定义

**杰卡德距离（Jaccard Distance）通过计算1减去两个集合交集与并集的比值来度量不相似度，值越小表示越相似，特别适合文档相似度、标签匹配和化学结构比对。**

---

## 数学定义

### 杰卡德相似系数

对于两个集合 **A** 和 **B**：

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

其中：
- `|A ∩ B|` 是交集的元素个数
- `|A ∪ B|` 是并集的元素个数

### 杰卡德距离

```
d_J(A, B) = 1 - J(A, B)
         = (|A ∪ B| - |A ∩ B|) / |A ∪ B|
```

**几何意义**：
- **J(A,B) = 1**：两个集合完全相同
- **J(A,B) = 0**：两个集合完全不同（无交集）
- **d_J(A,B) = 0**：完全相似
- **d_J(A,B) = 1**：完全不相似

---

## 核心特征

### 1. 值域特征

- **值域**：[0, 1]
- **相似性**：值越小，越相似
- **0表示**：两个集合完全相同
- **1表示**：两个集合完全不同

### 2. 等价于Tanimoto系数

**关键洞察**：对于二值变量，JACCARD距离等价于Tanimoto系数

```python
import numpy as np

# 二值向量
vec_a = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
vec_b = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)

# JACCARD = Tanimoto
intersection = np.sum(vec_a & vec_b)  # 2
union = np.sum(vec_a | vec_b)         # 6

jaccard_similarity = intersection / union
print(f"Jaccard相似度: {jaccard_similarity:.4f}")  # 0.3333

jaccard_distance = 1 - jaccard_similarity
print(f"Jaccard距离: {jaccard_distance:.4f}")      # 0.6667
```

**来源**：[Milvus Official Docs](https://milvus.io/docs/metric.md)

> "For binary variables, JACCARD distance is equivalent to the Tanimoto coefficient."

### 3. 集合视角

**JACCARD关注集合重叠**：

```
集合A: {1, 2, 3, 4}
集合B: {3, 4, 5, 6}

交集: {3, 4} → |A ∩ B| = 2
并集: {1, 2, 3, 4, 5, 6} → |A ∪ B| = 6

Jaccard相似度: 2/6 = 0.333
Jaccard距离: 1 - 0.333 = 0.667
```

---

## 详细解释

### 工作原理（步骤拆解）

**步骤1：表示为集合**

```
文档A: "Milvus 向量 数据库"
文档B: "向量 数据库 检索"

集合A: {Milvus, 向量, 数据库}
集合B: {向量, 数据库, 检索}
```

**步骤2：计算交集**

```
A ∩ B = {向量, 数据库}
|A ∩ B| = 2
```

**步骤3：计算并集**

```
A ∪ B = {Milvus, 向量, 数据库, 检索}
|A ∪ B| = 4
```

**步骤4：计算Jaccard相似度**

```
J(A, B) = 2 / 4 = 0.5
```

**步骤5：计算Jaccard距离**

```
d_J(A, B) = 1 - 0.5 = 0.5
```

### Python实现

```python
import numpy as np

def jaccard_similarity(a, b):
    """
    计算Jaccard相似度（集合）

    参数:
        a: set，集合A
        b: set，集合B

    返回:
        float: Jaccard相似度
    """
    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union

def jaccard_distance(a, b):
    """计算Jaccard距离"""
    return 1 - jaccard_similarity(a, b)

# 方法2：二值向量实现
def jaccard_binary(vec_a, vec_b):
    """
    计算二值向量的Jaccard距离

    参数:
        vec_a: numpy数组，二值向量A（0/1）
        vec_b: numpy数组，二值向量B（0/1）

    返回:
        float: Jaccard距离
    """
    intersection = np.sum(vec_a & vec_b)
    union = np.sum(vec_a | vec_b)

    if union == 0:
        return 0.0

    jaccard_sim = intersection / union
    return 1 - jaccard_sim

# 示例1：集合
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

print(f"Jaccard相似度: {jaccard_similarity(set_a, set_b):.4f}")  # 0.3333
print(f"Jaccard距离: {jaccard_distance(set_a, set_b):.4f}")      # 0.6667

# 示例2：二值向量
vec_a = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
vec_b = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)

print(f"Jaccard距离（二值）: {jaccard_binary(vec_a, vec_b):.4f}")  # 0.6667
```

---

## Milvus中的杰卡德距离

### 创建Collection时指定

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 定义Schema（二值向量）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128)
    # 注意：dim必须是8的倍数
]
schema = CollectionSchema(fields, description="Jaccard距离示例")
collection = Collection(name="jaccard_demo", schema=schema)

# 创建索引（指定JACCARD度量）
index_params = {
    "index_type": "BIN_IVF_FLAT",  # 二值向量索引
    "metric_type": "JACCARD",      # ← 指定Jaccard距离
    "params": {"nlist": 1024}
}
collection.create_index(field_name="binary_vector", index_params=index_params)

print("✅ Collection创建成功，使用Jaccard距离度量")
```

### 检索时使用

```python
# 加载Collection
collection.load()

# 准备查询向量（二值）
query_bool = [1, 0, 1, 0, 1, 0, 1, 0] * 16  # 128位
query_bytes = convert_bool_list_to_bytes(query_bool)

# 检索参数
search_params = {
    "metric_type": "JACCARD",
    "params": {"nprobe": 10}
}

# 执行检索
results = collection.search(
    data=[query_bytes],
    anns_field="binary_vector",
    param=search_params,
    limit=10,
    output_fields=["id"]
)

# 输出结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Jaccard距离: {hit.distance:.4f}")
```

---

## 何时使用杰卡德距离

### ✅ 适用场景

#### 1. 文档相似度（词袋模型）

```python
# 场景：基于词汇重叠的文档相似度
def document_to_set(text):
    """将文档转换为词汇集合"""
    return set(text.lower().split())

doc1 = "Milvus向量数据库支持多种相似度度量"
doc2 = "向量数据库Milvus提供高性能检索"

set1 = document_to_set(doc1)
set2 = document_to_set(doc2)

jaccard_sim = jaccard_similarity(set1, set2)
print(f"文档相似度: {jaccard_sim:.4f}")

# 判断相似度
if jaccard_sim > 0.5:
    print("文档相似")
else:
    print("文档不同")
```

**原因**：
- 文档表示为词汇集合
- Jaccard直接度量词汇重叠
- 适合简单的文档去重

---

#### 2. 标签匹配（用户兴趣）

```python
# 场景：用户兴趣标签匹配
user1_tags = {"AI", "机器学习", "向量数据库", "Python"}
user2_tags = {"向量数据库", "Python", "深度学习", "NLP"}

jaccard_sim = jaccard_similarity(user1_tags, user2_tags)
print(f"兴趣相似度: {jaccard_sim:.4f}")  # 0.3333

# 推荐相似用户
if jaccard_sim > 0.3:
    print("推荐关注该用户")
```

**原因**：
- 标签表示为集合
- Jaccard度量标签重叠
- 适合基于标签的推荐

---

#### 3. 化学结构相似度（分子指纹）

```python
# 场景：化学分子指纹比对
# 分子指纹表示为二值向量

# 分子A的指纹（128位）
molecule_a = np.array([1, 0, 1, 1, 0, 1, 0, 0] * 16, dtype=np.uint8)

# 分子B的指纹
molecule_b = np.array([1, 0, 1, 0, 0, 1, 1, 0] * 16, dtype=np.uint8)

# 使用Jaccard距离
jaccard_dist = jaccard_binary(molecule_a, molecule_b)
print(f"分子相似度: {1 - jaccard_dist:.4f}")

# 判断结构相似
if jaccard_dist < 0.3:
    print("化学结构相似")
```

**原因**：
- 分子指纹是二值向量
- Jaccard度量结构重叠
- 化学信息学标准度量

**来源**：[Milvus Official Docs](https://milvus.io/docs/metric.md)

> "JACCARD distance coefficient measures the similarity between two sample sets and is defined as the cardinality of the intersection of the defined sets divided by the cardinality of the union of them."

---

### ❌ 不适用场景

#### 1. 浮点向量检索

```python
# 场景：文本语义检索（float32 embedding）
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("文本示例")

print(f"Embedding类型: {embedding.dtype}")  # float32

# ❌ 不能使用JACCARD
# ✅ 应使用COSINE或IP
```

**原因**：JACCARD只适用于二值向量（集合），不适用于浮点向量

---

#### 2. 需要位级精确比较

```python
# 场景：图像哈希精确匹配
# 需要位级差异而非集合重叠

image_hash_1 = np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
image_hash_2 = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)

# HAMMING距离：2（2位不同）
hamming = np.sum(image_hash_1 != image_hash_2)

# JACCARD距离：0.4（集合重叠）
jaccard = jaccard_binary(image_hash_1, image_hash_2)

# ❌ JACCARD不适合位级比较
# ✅ 应使用HAMMING
```

**原因**：HAMMING关注位级差异，JACCARD关注集合重叠

---

## 与其他度量的对比

### JACCARD vs HAMMING

| 维度 | 杰卡德距离JACCARD | 汉明距离HAMMING |
|-----|------------------|----------------|
| **适用向量** | 二值向量（集合） | 二值向量（位） |
| **计算方式** | 1 - (交集/并集) | 不同位数量 |
| **值域** | [0, 1] | [0, dim] |
| **相似性** | 越小越相似 | 越小越相似 |
| **计算成本** | 中（集合运算） | 低（XOR+POPCNT） |
| **应用场景** | 集合相似度 | 位级比较 |

**示例对比**：

```python
# 两个二值向量
vec_a = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
vec_b = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)

# HAMMING距离
hamming = np.sum(vec_a != vec_b)
print(f"HAMMING距离: {hamming}")  # 4（4位不同）

# JACCARD距离
jaccard = jaccard_binary(vec_a, vec_b)
print(f"JACCARD距离: {jaccard:.4f}")  # 0.6667（集合重叠）

# 结论：HAMMING关注位级差异，JACCARD关注集合重叠
```

---

### JACCARD vs COSINE

| 维度 | 杰卡德距离JACCARD | 余弦相似度COSINE |
|-----|------------------|-----------------|
| **适用向量** | 二值向量 | 浮点向量 |
| **考虑因素** | 集合重叠 | 方向相似度 |
| **值域** | [0, 1] | [-1, 1] |
| **相似性** | 越小越相似 | 越大越相似 |
| **计算成本** | 中 | 中 |
| **语义理解** | 无（集合） | 有（方向） |

**选择建议**：
- 集合相似度 → JACCARD
- 语义相似度 → COSINE

---

## RAG应用场景

### 场景1：文档去重系统

**需求**：检测重复或近似重复的文档

```python
from pymilvus import connections, Collection

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 文档转二值向量（词袋模型）
def document_to_binary(text, vocabulary):
    """
    将文档转换为二值向量

    参数:
        text: str，文档文本
        vocabulary: list，词汇表

    返回:
        bytes: 二值向量（字节格式）
    """
    words = set(text.lower().split())
    binary = [1 if word in words else 0 for word in vocabulary]

    # 填充到8的倍数
    while len(binary) % 8 != 0:
        binary.append(0)

    return convert_bool_list_to_bytes(binary)

# 词汇表（128个词）
vocabulary = ["milvus", "向量", "数据库", "检索", ...]  # 128个词

# 查询文档
query_doc = "Milvus向量数据库支持多种检索方式"
query_binary = document_to_binary(query_doc, vocabulary)

# 使用JACCARD检索相似文档
collection = Collection("documents")
collection.load()

search_params = {"metric_type": "JACCARD", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_binary],
    anns_field="doc_vector",
    param=search_params,
    limit=10
)

# 输出重复文档
for hit in results[0]:
    if hit.distance < 0.3:  # Jaccard距离阈值
        print(f"重复文档ID: {hit.id}, Jaccard距离: {hit.distance:.4f}")
```

**为什么使用JACCARD**：
- 文档表示为词汇集合
- Jaccard直接度量词汇重叠
- 适合简单的文档去重

---

### 场景2：用户兴趣匹配

**需求**：根据标签匹配相似用户

```python
# 用户标签转二值向量
def tags_to_binary(tags, all_tags):
    """
    将标签集合转换为二值向量

    参数:
        tags: set，用户标签
        all_tags: list，所有标签列表

    返回:
        bytes: 二值向量
    """
    binary = [1 if tag in tags else 0 for tag in all_tags]

    # 填充到8的倍数
    while len(binary) % 8 != 0:
        binary.append(0)

    return convert_bool_list_to_bytes(binary)

# 所有标签（128个）
all_tags = ["AI", "机器学习", "向量数据库", "Python", ...]  # 128个标签

# 查询用户标签
query_tags = {"AI", "机器学习", "向量数据库", "Python"}
query_binary = tags_to_binary(query_tags, all_tags)

# 使用JACCARD检索相似用户
search_params = {"metric_type": "JACCARD", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_binary],
    anns_field="user_tags",
    param=search_params,
    limit=10
)

# 输出相似用户
for hit in results[0]:
    print(f"相似用户ID: {hit.id}, 标签重叠度: {1 - hit.distance:.4f}")
```

**为什么使用JACCARD**：
- 标签表示为集合
- Jaccard度量标签重叠
- 适合基于标签的推荐

---

## 2025-2026最佳实践

### 1. JACCARD适合集合相似度

**决策流程**：

```python
def should_use_jaccard(data_type, use_case):
    """判断是否使用JACCARD"""

    # 1. 数据类型
    if data_type == "binary_vector":
        # 2. 使用场景
        if use_case in ["set_similarity", "tag_matching", "document_dedup"]:
            return True  # 推荐JACCARD ✅

    # 3. 其他场景
    if data_type == "float_vector":
        return False  # 使用COSINE或IP

    if use_case == "bit_level_comparison":
        return False  # 使用HAMMING

    return False
```

---

### 2. 维度必须是8的倍数

```python
# ✅ 正确：128维（8的倍数）
fields = [
    FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128)
]

# ❌ 错误：100维（不是8的倍数）
# 会导致错误
```

**来源**：[Milvus Official Docs - Binary Vector](https://milvus.io/docs/binary-vector.md)

> "Note that `dim` must be a multiple of 8 as binary vectors must be converted into a byte array when inserting."

---

### 3. 索引选择建议

```python
# 小数据集（<100K）：BIN_FLAT
index_params_small = {
    "index_type": "BIN_FLAT",  # 精确搜索
    "metric_type": "JACCARD"
}

# 大数据集（>2GB）：BIN_IVF_FLAT
index_params_large = {
    "index_type": "BIN_IVF_FLAT",  # 近似搜索
    "metric_type": "JACCARD",
    "params": {"nlist": 1024}
}

# 自动选择：AUTOINDEX
index_params_auto = {
    "index_type": "AUTOINDEX",
    "metric_type": "JACCARD"
}
```

---

## 常见误区

### 误区1：JACCARD可以用于浮点向量 ❌

**错误观点**："JACCARD是通用距离度量"

**正确理解**：
- JACCARD只适用于二值向量（集合）
- 浮点向量应使用L2、IP或COSINE
- 需要先进行二值化才能使用JACCARD

---

### 误区2：JACCARD和HAMMING结果相同 ❌

**错误观点**："JACCARD和HAMMING都是二值向量度量，结果一样"

**正确理解**：
- JACCARD关注集合重叠（交集/并集）
- HAMMING关注位级差异（XOR）
- 两者结果不同，适用场景不同

---

### 误区3：JACCARD总是比HAMMING慢 ❌

**错误观点**："JACCARD计算复杂，总是比HAMMING慢"

**正确理解**：
- JACCARD需要集合运算（AND/OR）
- HAMMING需要XOR + POPCNT
- 实际性能取决于实现和硬件
- 通常HAMMING更快，但差异不大

---

## 学习检查清单

完成本节后，你应该能够：

- [ ] 理解杰卡德距离的数学定义和集合意义
- [ ] 知道JACCARD等价于Tanimoto系数
- [ ] 理解JACCARD与HAMMING的区别
- [ ] 能够判断何时使用杰卡德距离
- [ ] 知道维度必须是8的倍数
- [ ] 能够在Milvus中配置JACCARD度量
- [ ] 理解JACCARD的应用场景
- [ ] 知道2025-2026的最佳实践
- [ ] 避开常见的JACCARD误区

---

## 参考资源

### 官方文档

- [Milvus Metric Types](https://milvus.io/docs/metric.md) - 官方度量类型文档
- [Milvus Binary Vector](https://milvus.io/docs/binary-vector.md) - 官方二值向量文档

### 技术博客

- [Zilliz Blog - Similarity Metrics](https://zilliz.com/blog/similarity-metrics-for-vector-search) - 详细解释相似度度量

---

## 下一步

- **子结构距离**：[03_核心概念_6_子结构距离SUBSTRUCTURE](./03_核心概念_6_子结构距离SUBSTRUCTURE.md) - 学习化学结构度量
- **实战代码**：[07_实战代码_场景2_二值向量度量](./07_实战代码_场景2_二值向量度量.md) - 二值向量实战
- **汉明距离**：[03_核心概念_4_汉明距离HAMMING](./03_核心概念_4_汉明距离HAMMING.md) - 回顾位级比较

---

**返回：** [00_概览](./00_概览.md)
