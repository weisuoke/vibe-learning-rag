# 核心概念4：汉明距离HAMMING（Hamming Distance）

> 汉明距离测量二值向量中不同比特位的数量，是二值向量检索的核心度量，提供极致的存储效率和计算速度

---

## 一句话定义

**汉明距离（Hamming Distance）计算两个等长二值向量中不同比特位的数量，值越小表示越相似，特别适合图像哈希、文本去重和资源受限环境。**

---

## 数学定义

### 公式

对于两个n维二值向量 **A = (a₀, a₁, ..., aₙ₋₁)** 和 **B = (b₀, b₁, ..., bₙ₋₁)**（每个元素为0或1）：

```
HAMMING(A, B) = Σ(aᵢ ≠ bᵢ)
              = count of differing bits
```

### 位运算实现

```
HAMMING(A, B) = POPCNT(A ⊕ B)
```

其中：
- `⊕` 是XOR（异或）运算
- `POPCNT` 是计算1的个数（population count）

**示例：**

```
向量A: 11011001
向量B: 10011101

A ⊕ B: 01000100  (XOR结果)
POPCNT: 2        (有2个1)

HAMMING(A, B) = 2
```

---

## 核心特征

### 1. 值域特征

- **值域**：[0, dim(vector)]
- **相似性**：值越小，越相似
- **0表示**：两个向量完全相同
- **dim表示**：两个向量完全不同

### 2. 极致存储效率

**关键洞察**：二值向量提供32x存储压缩

```python
import numpy as np

# 128维向量存储对比
float32_size = 128 * 4  # 512 bytes
binary_size = 128 // 8  # 16 bytes

compression_ratio = float32_size / binary_size
print(f"压缩比: {compression_ratio}x")  # 32x
```

**实际案例**：
- 1亿个128维float32向量：**~512 GB**
- 1亿个128维binary向量：**~16 GB**
- **节省496 GB存储空间**

**来源**：[Milvus AI Quick Reference](https://milvus.io/ai-quick-reference/how-does-using-a-binary-embedding-eg-sign-of-components-only-or-learned-binary-codes-drastically-cut-down-storage-and-what-kind-of-search-algorithms-support-such-binary-vectors)

> "A 128-dimensional vector stored as 32-bit floats requires **512 bytes**, but its binary equivalent uses just **16 bytes** (128 bits). This makes binary embeddings **32x** more efficient in raw bit terms."

### 3. 极快计算速度

**硬件加速**：POPCNT指令

```python
# XOR + POPCNT 比浮点运算快数个数量级
def hamming_distance_fast(a, b):
    """使用位运算计算汉明距离"""
    xor_result = a ^ b  # XOR运算（硬件级别）
    return bin(xor_result).count('1')  # POPCNT
```

**性能提升**：
- QPS提升：**5-20x**（相比float32向量）
- 延迟降低：**数个数量级**
- 吞吐量提升：**显著**

---

## 详细解释

### 工作原理（步骤拆解）

**步骤1：对齐二值向量**

```
向量A: [1, 1, 0, 1, 1, 0, 0, 1]
向量B: [1, 0, 0, 1, 1, 1, 0, 1]
```

**步骤2：逐位比较（XOR）**

```
A ⊕ B: [0, 1, 0, 0, 0, 1, 0, 0]
       ↑  ↑              ↑
       相同 不同          不同
```

**步骤3：计数不同位（POPCNT）**

```
不同位数量: 2
HAMMING距离: 2
```

### Python实现

```python
import numpy as np

def hamming_distance(a, b):
    """
    计算汉明距离（二值向量）

    参数:
        a: numpy数组，二值向量A（0/1）
        b: numpy数组，二值向量B（0/1）

    返回:
        int: 汉明距离
    """
    # 方法1：逐位比较
    diff = a != b
    hamming = np.sum(diff)

    return hamming

# 方法2：使用XOR
def hamming_distance_xor(a, b):
    """使用XOR计算汉明距离"""
    xor_result = np.bitwise_xor(a, b)
    return np.sum(xor_result)

# 方法3：字节数组（Milvus格式）
def hamming_distance_bytes(a_bytes, b_bytes):
    """
    计算字节数组的汉明距离

    参数:
        a_bytes: bytes，二值向量A（字节格式）
        b_bytes: bytes，二值向量B（字节格式）

    返回:
        int: 汉明距离
    """
    hamming = 0
    for byte_a, byte_b in zip(a_bytes, b_bytes):
        xor_byte = byte_a ^ byte_b
        hamming += bin(xor_byte).count('1')
    return hamming

# 示例
vec_a = np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
vec_b = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)

print(f"汉明距离（方法1）: {hamming_distance(vec_a, vec_b)}")      # 2
print(f"汉明距离（方法2）: {hamming_distance_xor(vec_a, vec_b)}")  # 2
```

---

## Milvus中的汉明距离

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
schema = CollectionSchema(fields, description="汉明距离示例")
collection = Collection(name="hamming_demo", schema=schema)

# 创建索引（指定HAMMING度量）
index_params = {
    "index_type": "BIN_IVF_FLAT",  # 二值向量索引
    "metric_type": "HAMMING",      # ← 指定汉明距离（默认值）
    "params": {"nlist": 1024}
}
collection.create_index(field_name="binary_vector", index_params=index_params)

print("✅ Collection创建成功，使用汉明距离度量")
```

### 二值向量数据准备

```python
def convert_bool_list_to_bytes(bool_list):
    """
    将布尔列表转换为字节数组（Milvus格式）

    参数:
        bool_list: list，布尔值列表（0/1）

    返回:
        bytes: 字节数组
    """
    if len(bool_list) % 8 != 0:
        raise ValueError("长度必须是8的倍数")

    byte_array = bytearray(len(bool_list) // 8)
    for i, bit in enumerate(bool_list):
        if bit == 1:
            index = i // 8
            shift = i % 8
            byte_array[index] |= (1 << shift)
    return bytes(byte_array)

# 示例：128维二值向量
bool_vector = [1, 0, 1, 1, 0, 0, 1, 0] * 16  # 128位
binary_bytes = convert_bool_list_to_bytes(bool_vector)

print(f"布尔列表长度: {len(bool_vector)}")      # 128
print(f"字节数组长度: {len(binary_bytes)}")     # 16
```

### 插入数据

```python
# 准备数据
bool_vectors = [
    [1, 0, 0, 1, 1, 0, 1, 1] * 16,  # 128位
    [0, 1, 0, 1, 0, 1, 0, 0] * 16,
]

data = [
    {"binary_vector": convert_bool_list_to_bytes(vec)}
    for vec in bool_vectors
]

# 插入数据
collection.insert(data)
print("✅ 数据插入成功")
```

### 检索时使用

```python
# 加载Collection
collection.load()

# 准备查询向量
query_bool = [1, 0, 1, 0, 1, 0, 1, 0] * 16
query_bytes = convert_bool_list_to_bytes(query_bool)

# 检索参数
search_params = {
    "metric_type": "HAMMING",
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
        print(f"ID: {hit.id}, 汉明距离: {hit.distance}")
```

---

## 何时使用汉明距离

### ✅ 适用场景

#### 1. 图像哈希去重

```python
# 场景：图像感知哈希（pHash）
import imagehash
from PIL import Image

# 计算图像哈希
image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")

hash1 = imagehash.phash(image1)  # 64位二值哈希
hash2 = imagehash.phash(image2)

# 汉明距离
hamming = hash1 - hash2  # imagehash内部使用汉明距离
print(f"汉明距离: {hamming}")

# 判断相似度
if hamming < 10:
    print("图像相似（可能是重复）")
else:
    print("图像不同")
```

**原因**：
- 图像哈希生成二值向量
- 汉明距离快速比较
- 适合大规模图像去重

**来源**：[Milvus AI Quick Reference](https://milvus.io/ai-quick-reference/how-does-using-a-binary-embedding-eg-sign-of-components-only-or-learned-binary-codes-drastically-cut-down-storage-and-what-kind-of-search-algorithms-support-such-binary-vectors)

> "Practical applications include: Image retrieval (e.g., mapping CNN features to binary codes), Text search (hashing word embeddings)."

---

#### 2. 文本MinHash去重

```python
# 场景：文本相似度检测（MinHash）
from datasketch import MinHash

def text_to_minhash(text, num_perm=128):
    """将文本转换为MinHash二值向量"""
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

text1 = "Milvus向量数据库支持汉明距离"
text2 = "Milvus向量数据库支持汉明距离检索"

mh1 = text_to_minhash(text1)
mh2 = text_to_minhash(text2)

# 估计Jaccard相似度（基于汉明距离）
jaccard = mh1.jaccard(mh2)
print(f"Jaccard相似度: {jaccard:.4f}")
```

**原因**：
- MinHash生成二值签名
- 汉明距离估计Jaccard相似度
- 适合文本去重和相似度检测

---

#### 3. 资源受限环境

```python
# 场景：移动设备或嵌入式系统
# 内存和计算资源有限

# 二值向量：极低内存占用
binary_vectors = np.random.randint(0, 2, (1000000, 128), dtype=np.uint8)
memory_usage = binary_vectors.nbytes / (1024**2)
print(f"内存占用: {memory_usage:.2f} MB")  # ~15 MB

# 对比：float32向量
float_vectors = np.random.rand(1000000, 128).astype(np.float32)
memory_usage_float = float_vectors.nbytes / (1024**2)
print(f"float32内存占用: {memory_usage_float:.2f} MB")  # ~488 MB
```

**原因**：
- 32x存储压缩
- 极快计算速度
- 适合边缘设备和嵌入式系统

---

### ❌ 不适用场景

#### 1. 浮点向量检索

```python
# 场景：文本语义检索（float32 embedding）
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("文本示例")

print(f"Embedding类型: {embedding.dtype}")  # float32
print(f"向量长度: {len(embedding)}")         # 384

# ❌ 不能直接使用HAMMING
# ✅ 应使用COSINE或IP
```

**原因**：HAMMING只适用于二值向量（0/1），不适用于浮点向量

---

#### 2. 需要高精度语义理解

```python
# 场景：复杂语义相似度
# 二值量化会损失精度

# 原始float32向量
vec_float = np.array([0.1, 0.2, 0.3, 0.4])

# 二值化（简单阈值）
vec_binary = (vec_float > 0.25).astype(np.uint8)
print(f"二值化: {vec_binary}")  # [0, 0, 1, 1]

# ❌ 损失了细微的语义差异
# ✅ 应使用COSINE保留精度
```

**来源**：[Zilliz Binary Quantization Guide](https://zilliz.com/learn/enhancing-efficiency-in-vector-searches-with-binary-quantization-and-milvus)

> "While binary embeddings sacrifice some precision, their storage and speed benefits make them indispensable for real-time systems where scalability and latency are critical."

---

## 与其他度量的对比

### HAMMING vs JACCARD

| 维度 | 汉明距离HAMMING | 杰卡德距离JACCARD |
|-----|----------------|------------------|
| **适用向量** | 二值向量 | 二值向量（集合） |
| **计算方式** | 不同位数量 | 1 - (交集/并集) |
| **值域** | [0, dim] | [0, 1] |
| **相似性** | 越小越相似 | 越小越相似 |
| **计算成本** | 极低（XOR+POPCNT） | 低 |
| **应用场景** | 图像哈希、位级比较 | 集合相似度 |

**关系**：
- HAMMING关注位级差异
- JACCARD关注集合相似度
- 两者都适用于二值向量

---

### HAMMING vs COSINE

| 维度 | 汉明距离HAMMING | 余弦相似度COSINE |
|-----|----------------|-----------------|
| **适用向量** | 二值向量 | 浮点向量 |
| **考虑因素** | 位级差异 | 方向相似度 |
| **值域** | [0, dim] | [-1, 1] |
| **相似性** | 越小越相似 | 越大越相似 |
| **计算成本** | 极低 | 中 |
| **存储效率** | 32x压缩 | 无压缩 |

**选择建议**：
- 需要极致效率 → HAMMING
- 需要高精度语义 → COSINE

---

## RAG应用场景

### 场景1：大规模图像去重系统

**需求**：检测数十亿图像中的重复内容

```python
from pymilvus import connections, Collection
import imagehash
from PIL import Image

# 连接Milvus
connections.connect("default", host="localhost", port="19530")

# 创建图像哈希Collection
collection = Collection("image_dedup")
collection.load()

# 计算图像哈希
def compute_image_hash(image_path):
    """计算图像感知哈希"""
    image = Image.open(image_path)
    phash = imagehash.phash(image, hash_size=8)  # 64位哈希

    # 转换为二值向量
    hash_bits = [int(bit) for bit in str(phash)]
    # 填充到128位（8的倍数）
    hash_bits += [0] * (128 - len(hash_bits))

    return convert_bool_list_to_bytes(hash_bits)

# 查询图像
query_image = "query.jpg"
query_hash = compute_image_hash(query_image)

# 使用HAMMING检索相似图像
search_params = {"metric_type": "HAMMING", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_hash],
    anns_field="image_hash",
    param=search_params,
    limit=10
)

# 输出重复图像
for hit in results[0]:
    if hit.distance < 10:  # 汉明距离阈值
        print(f"重复图像ID: {hit.id}, 汉明距离: {hit.distance}")
```

**为什么使用HAMMING**：
- 图像哈希生成二值向量
- 数十亿图像需要极致存储效率
- 汉明距离提供快速比较

---

### 场景2：文本指纹去重

**需求**：检测重复或近似重复的文档

```python
from datasketch import MinHash

# 文档MinHash指纹
def document_to_binary(text, num_perm=128):
    """将文档转换为MinHash二值指纹"""
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))

    # 提取MinHash签名（二值）
    signature = m.hashvalues
    # 转换为二值向量（简化示例）
    binary = [(h % 2) for h in signature]

    return convert_bool_list_to_bytes(binary)

# 查询文档
query_doc = "Milvus向量数据库支持多种相似度度量"
query_binary = document_to_binary(query_doc)

# 使用HAMMING检索相似文档
search_params = {"metric_type": "HAMMING", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_binary],
    anns_field="doc_fingerprint",
    param=search_params,
    limit=10
)

# 输出相似文档
for hit in results[0]:
    print(f"文档ID: {hit.id}, 汉明距离: {hit.distance}")
```

**为什么使用HAMMING**：
- MinHash生成二值签名
- 汉明距离估计Jaccard相似度
- 适合大规模文本去重

---

## 2025-2026最佳实践

### 1. 二值量化是生产标准

**决策流程**：

```python
def should_use_binary_quantization(dataset_size, memory_budget, recall_requirement):
    """判断是否使用二值量化"""

    # 1. 数据集规模
    if dataset_size > 100_000_000:  # 1亿+
        if memory_budget == "limited":
            return True  # 推荐二值量化 ✅

    # 2. 召回率要求
    if recall_requirement < 0.95:  # 召回率要求<95%
        return True  # 可接受精度损失 ✅

    # 3. 延迟敏感
    if latency_critical:
        return True  # 需要极快速度 ✅

    return False  # 使用float32
```

**来源**：[Zilliz Binary Quantization Guide](https://zilliz.com/learn/enhancing-efficiency-in-vector-searches-with-binary-quantization-and-milvus)

> "Binary quantization represents a transformative approach to managing and searching vector data within Milvus, offering significant enhancements in both efficiency and performance."

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

**原因**：二值向量存储为字节数组，每8位打包为1字节

**来源**：[Milvus Official Docs - Binary Vector](https://milvus.io/docs/binary-vector.md)

> "**Note that dim must be a multiple of 8** as binary vectors must be converted into a byte array when inserting."

---

### 3. 索引选择建议

```python
# 小数据集（<100K）：BIN_FLAT
index_params_small = {
    "index_type": "BIN_FLAT",  # 精确搜索
    "metric_type": "HAMMING"
}

# 大数据集（>2GB）：BIN_IVF_FLAT
index_params_large = {
    "index_type": "BIN_IVF_FLAT",  # 近似搜索
    "metric_type": "HAMMING",
    "params": {"nlist": 1024}
}

# 自动选择：AUTOINDEX
index_params_auto = {
    "index_type": "AUTOINDEX",
    "metric_type": "HAMMING"
}
```

---

## 常见误区

### 误区1：HAMMING可以用于浮点向量 ❌

**错误观点**："HAMMING是通用距离度量"

**正确理解**：
- HAMMING只适用于二值向量（0/1）
- 浮点向量应使用L2、IP或COSINE
- 需要先进行二值量化才能使用HAMMING

---

### 误区2：二值量化不损失精度 ❌

**错误观点**："二值量化和float32效果一样"

**正确理解**：
- 二值量化会损失精度
- 召回率通常下降5-10%
- 需要在效率和精度之间权衡

**来源**：[Zilliz Binary Quantization Guide](https://zilliz.com/learn/enhancing-efficiency-in-vector-searches-with-binary-quantization-and-milvus)

> "Recall rate: With reasonable parameters: **90%+**, some scenarios approach floating-point baseline."

---

### 误区3：HAMMING总是最快的 ❌

**错误观点**："HAMMING比所有度量都快"

**正确理解**：
- HAMMING快是因为二值向量小
- 如果数据已经是float32，转换成本可能抵消收益
- 需要考虑端到端性能

---

## 学习检查清单

完成本节后，你应该能够：

- [ ] 理解汉明距离的数学定义和位运算实现
- [ ] 知道二值向量提供32x存储压缩
- [ ] 理解XOR和POPCNT的作用
- [ ] 能够判断何时使用汉明距离
- [ ] 知道维度必须是8的倍数
- [ ] 能够在Milvus中配置HAMMING度量
- [ ] 理解二值量化的权衡
- [ ] 知道2025-2026的最佳实践
- [ ] 避开常见的HAMMING误区

---

## 参考资源

### 官方文档

- [Milvus Binary Vector](https://milvus.io/docs/binary-vector.md) - 官方二值向量文档
- [Milvus Metric Types](https://milvus.io/docs/metric.md) - 官方度量类型文档

### 技术博客

- [Milvus AI Quick Reference - Binary Embedding](https://milvus.io/ai-quick-reference/how-does-using-a-binary-embedding-eg-sign-of-components-only-or-learned-binary-codes-drastically-cut-down-storage-and-what-kind-of-search-algorithms-support-such-binary-vectors) - 二值嵌入存储效率
- [Zilliz - Binary Quantization](https://zilliz.com/learn/enhancing-efficiency-in-vector-searches-with-binary-quantization-and-milvus) - 二值量化指南

---

## 下一步

- **杰卡德距离**：[03_核心概念_5_杰卡德距离JACCARD](./03_核心概念_5_杰卡德距离JACCARD.md) - 学习集合相似度
- **实战代码**：[07_实战代码_场景2_二值向量度量](./07_实战代码_场景2_二值向量度量.md) - 二值向量实战
- **余弦相似度**：[03_核心概念_3_余弦相似度COSINE](./03_核心概念_3_余弦相似度COSINE.md) - 回顾浮点向量度量

---

**返回：** [00_概览](./00_概览.md)
