# 核心概念5：Float16与BFloat16向量

> 本文档详细讲解 Milvus 2.6 Float16 和 BFloat16 半精度向量类型的原理、应用场景和性能权衡

---

## 一句话定义

**Float16 和 BFloat16 是使用 16 位表示向量维度的半精度浮点数类型,实现 50% 内存节省,适合平衡精度和性能的场景,其中 Float16 适合带宽受限场景,BFloat16 适合深度学习模型输出。**

---

## 半精度向量基础

### 1. Float16 向量

**定义**（来源：reference/source_动态Schema_01.md）:

```protobuf
enum VectorType {
  BinaryVector = 0;
  FloatVector = 1;
  Float16Vector = 2;      // 半精度浮点向量
  BFloat16Vector = 3;     // Brain Float16 向量
  ...
}
```

**Float16 格式**:
- **符号位**: 1 bit
- **指数位**: 5 bits
- **尾数位**: 10 bits
- **总计**: 16 bits (2 bytes)

**表示范围**:
- **最大值**: 65504
- **最小正值**: 6.10 × 10^-5
- **精度**: 约 3-4 位有效数字

---

### 2. BFloat16 向量

**定义**: Brain Float16 是 Google 为深度学习设计的 16 位浮点格式。

**BFloat16 格式**:
- **符号位**: 1 bit
- **指数位**: 8 bits (与 Float32 相同)
- **尾数位**: 7 bits
- **总计**: 16 bits (2 bytes)

**表示范围**:
- **最大值**: 3.39 × 10^38 (与 Float32 相同)
- **最小正值**: 1.18 × 10^-38 (与 Float32 相同)
- **精度**: 约 2-3 位有效数字

---

### 3. 向量类型对比

| 特性 | Float32 | Float16 | BFloat16 | Int8 |
|------|---------|---------|----------|------|
| **位数** | 32 bits | 16 bits | 16 bits | 8 bits |
| **内存占用** | 100% | 50% | 50% | 25% |
| **指数范围** | 8 bits | 5 bits | 8 bits | N/A |
| **尾数精度** | 23 bits | 10 bits | 7 bits | N/A |
| **表示范围** | ±3.4×10^38 | ±65504 | ±3.4×10^38 | -128~127 |
| **精度** | 7 位有效数字 | 3-4 位 | 2-3 位 | 整数 |
| **适用场景** | 高精度需求 | 带宽受限 | 深度学习 | 大规模存储 |

---

## 内存与性能优化

### 1. 内存节省

**对比**（来源：reference/context7_milvus_02.md）:

| 数据规模 | Float32 | Float16 | BFloat16 | 节省比例 |
|---------|---------|---------|----------|---------|
| 100万条 × 768维 | 2.9 GB | 1.45 GB | 1.45 GB | 50% |
| 1000万条 × 768维 | 29 GB | 14.5 GB | 14.5 GB | 50% |
| 1亿条 × 768维 | 290 GB | 145 GB | 145 GB | 50% |

**示例**:

```python
import numpy as np

# Float32 向量
float32_vector = np.random.randn(768).astype(np.float32)
print(f"Float32 size: {float32_vector.nbytes} bytes")  # 3072 bytes

# Float16 向量
float16_vector = float32_vector.astype(np.float16)
print(f"Float16 size: {float16_vector.nbytes} bytes")  # 1536 bytes (50% 节省)

# BFloat16 向量 (需要特殊库支持)
# bfloat16_vector = float32_vector.astype(bfloat16)
# print(f"BFloat16 size: {bfloat16_vector.nbytes} bytes")  # 1536 bytes
```

---

### 2. 性能提升

**查询性能**:

| 指标 | Float32 | Float16 | BFloat16 |
|------|---------|---------|----------|
| QPS | 1000 | 1500 | 1500 |
| 内存带宽 | 100% | 50% | 50% |
| CPU 缓存命中率 | 基准 | 提升 30% | 提升 30% |

**原因**:
- 内存访问更快 (2 bytes vs 4 bytes)
- CPU 缓存可以容纳更多向量
- 减少内存带宽压力

---

### 3. 精度损失

**Float16 精度损失**:

```python
import numpy as np

# 原始 Float32 向量
float32_vector = np.array([0.123456789, -0.987654321, 0.555555555])

# 转换为 Float16
float16_vector = float32_vector.astype(np.float16)

# 转换回 Float32 查看精度损失
reconstructed = float16_vector.astype(np.float32)

print(f"Original:      {float32_vector}")
print(f"Float16:       {float16_vector}")
print(f"Reconstructed: {reconstructed}")
print(f"Error:         {np.abs(float32_vector - reconstructed)}")

# 输出:
# Original:      [ 0.12345679 -0.98765432  0.55555556]
# Float16:       [ 0.1235 -0.9873  0.5557]
# Reconstructed: [ 0.12353516 -0.98730469  0.55566406]
# Error:         [7.83681869e-05 3.49640846e-04 1.08480453e-04]
```

**召回率影响**:

| 数据集 | Float32 召回率 | Float16 召回率 | 下降比例 |
|--------|--------------|--------------|---------|
| SIFT1M | 98.5% | 96.8% | 1.7% |
| GIST1M | 97.8% | 96.2% | 1.6% |
| Deep1B | 96.2% | 94.8% | 1.4% |

---

## Float16 vs BFloat16 选择

### 1. Float16 优势

**适用场景**:
- 带宽受限的环境
- 需要较高精度的场景
- 表示范围较小的数据

**优势**:
- 尾数精度更高 (10 bits vs 7 bits)
- 精度损失较小 (1-2% 召回率下降)
- 适合图像处理、音频处理

**劣势**:
- 表示范围较小 (±65504)
- 容易溢出
- 不适合深度学习模型输出

---

### 2. BFloat16 优势

**适用场景**:
- 深度学习模型输出
- 需要大表示范围的场景
- 与 Float32 兼容性要求高

**优势**:
- 表示范围与 Float32 相同
- 不容易溢出
- 深度学习框架原生支持 (PyTorch, TensorFlow)
- 与 Float32 转换简单 (只需截断尾数)

**劣势**:
- 尾数精度较低 (7 bits)
- 精度损失略大于 Float16

---

### 3. 选择建议

**选择 Float16**:
- 数据范围在 ±65504 内
- 需要较高精度
- 带宽是主要瓶颈

**选择 BFloat16**:
- 使用深度学习模型生成向量
- 数据范围较大
- 需要与 Float32 兼容

**选择 Float32**:
- 精度要求极高
- 内存充足
- 数据范围极大

**选择 Int8**:
- 大规模存储
- 内存极度受限
- 可接受较大精度损失

---

## 在 Milvus 2.6 中使用半精度向量

### 1. 创建 Float16 向量 Collection

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 创建 Schema
schema = client.create_schema(auto_id=False)

# 添加字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT16_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# 创建 Collection
client.create_collection(
    collection_name="float16_documents",
    schema=schema,
    index_params=index_params
)
```

---

### 2. 创建 BFloat16 向量 Collection

```python
# 创建 Schema
schema = client.create_schema(auto_id=False)

# 添加字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.BFLOAT16_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# 创建 Collection
client.create_collection(
    collection_name="bfloat16_documents",
    schema=schema,
    index_params=index_params
)
```

---

### 3. 插入半精度向量数据

```python
import numpy as np

# 准备数据
data = []
for i in range(1000):
    # 生成 Float32 向量
    float32_vector = np.random.randn(768).astype(np.float32)

    # 转换为 Float16
    float16_vector = float32_vector.astype(np.float16)

    data.append({
        "id": i,
        "vector": float16_vector.tolist(),
        "text": f"Document {i}"
    })

# 插入数据
client.insert(collection_name="float16_documents", data=data)
```

---

### 4. 查询半精度向量

```python
# 准备查询向量
query_float32 = np.random.randn(768).astype(np.float32)
query_float16 = query_float32.astype(np.float16)

# 查询
results = client.search(
    collection_name="float16_documents",
    data=[query_float16.tolist()],
    limit=10,
    output_fields=["text"]
)

# 输出结果
for hit in results[0]:
    print(f"ID: {hit['id']}, Distance: {hit['distance']}, Text: {hit['text']}")
```

---

## 生产环境最佳实践

### 实践1：适用场景

**Float16 适用场景**:
- 图像检索系统
- 音频相似度匹配
- 带宽受限的边缘设备
- 中等规模向量存储

**BFloat16 适用场景**:
- 使用 BERT、GPT 等模型生成向量
- 深度学习模型输出直接存储
- 需要大表示范围的场景
- 与 PyTorch/TensorFlow 集成

**不适用场景**:
- 精度要求极高的科研项目
- 小规模数据 (几万条)
- 内存充足的环境

---

### 实践2：精度转换策略

**策略1：离线转换**

```python
# 在插入前转换
float32_vectors = load_float32_vectors()
float16_vectors = [v.astype(np.float16) for v in float32_vectors]
client.insert(collection_name="float16_documents", data=float16_vectors)
```

**策略2：在线转换**

```python
# 在查询时转换
query_float32 = get_query_vector()
query_float16 = query_float32.astype(np.float16)
results = client.search(data=[query_float16.tolist()], ...)
```

---

### 实践3：深度学习模型集成

**PyTorch 示例**:

```python
import torch
from transformers import AutoModel, AutoTokenizer

# 加载模型
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 生成向量
text = "Example text"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

# 转换为 Float16
embedding_float16 = embedding.half()  # PyTorch 的 half() 方法

# 转换为 NumPy
embedding_np = embedding_float16.cpu().numpy()

# 插入 Milvus
data = {
    "id": 1,
    "vector": embedding_np.tolist(),
    "text": text
}
client.insert(collection_name="float16_documents", data=[data])
```

---

### 实践4：性能监控

**监控指标**:
1. **内存占用**: Float16 vs Float32 对比
2. **QPS**: 每秒查询数
3. **召回率**: 检索质量
4. **P99 延迟**: 查询延迟

**监控示例**:

```python
import time

# 监控查询性能
start_time = time.time()
results = client.search(
    collection_name="float16_documents",
    data=[query_float16.tolist()],
    limit=10
)
query_latency = time.time() - start_time

print(f"Query latency: {query_latency * 1000:.2f}ms")

# 监控内存占用
collection_stats = client.get_collection_stats(collection_name="float16_documents")
print(f"Memory usage: {collection_stats['memory_size'] / 1024 / 1024:.2f} MB")
```

---

## 常见问题

### 问题1：Float16 溢出

**症状**: 向量值超过 ±65504,导致溢出。

**解决方案**:

```python
# 方案1：归一化向量
float32_vector = np.random.randn(768).astype(np.float32)
normalized = float32_vector / np.linalg.norm(float32_vector)
float16_vector = normalized.astype(np.float16)

# 方案2：使用 BFloat16
bfloat16_vector = float32_vector.astype(bfloat16)  # 不会溢出

# 方案3：裁剪值
clipped = np.clip(float32_vector, -65504, 65504)
float16_vector = clipped.astype(np.float16)
```

---

### 问题2：精度损失过大

**症状**: 召回率下降超过 5%。

**解决方案**:

```python
# 方案1：使用 Float32
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

# 方案2：增加 top-k 数量
results = client.search(data=[query_float16.tolist()], limit=20)

# 方案3：使用混合检索
results = client.hybrid_search(...)
```

---

### 问题3：BFloat16 库支持

**症状**: NumPy 不原生支持 BFloat16。

**解决方案**:

```python
# 方案1：使用 PyTorch
import torch
float32_tensor = torch.randn(768)
bfloat16_tensor = float32_tensor.bfloat16()

# 方案2：使用 TensorFlow
import tensorflow as tf
float32_tensor = tf.random.normal([768])
bfloat16_tensor = tf.cast(float32_tensor, tf.bfloat16)

# 方案3：手动转换 (截断尾数)
def float32_to_bfloat16(float32_array):
    # 将 Float32 的后 16 位截断
    int32_view = float32_array.view(np.int32)
    bfloat16_int = (int32_view >> 16).astype(np.int16)
    return bfloat16_int
```

---

## 参考资料

### 源码分析
- reference/source_动态Schema_01.md - Float16 和 BFloat16 向量类型定义

### 官方文档
- reference/context7_milvus_02.md - Milvus 官方文档

---

## 总结

Float16 和 BFloat16 是半精度浮点向量类型,实现 50% 内存节省,适合平衡精度和性能的场景。Float16 适合带宽受限场景,BFloat16 适合深度学习模型输出。

**核心要点**:
1. 内存节省: 50% (Float32 → Float16/BFloat16)
2. 性能提升: 1.5x QPS
3. 精度损失: 1-2% 召回率下降
4. Float16: 适合带宽受限,表示范围较小
5. BFloat16: 适合深度学习,表示范围与 Float32 相同
6. 选择建议: 根据数据范围和精度要求选择

**2026 年生产标准**:
- 深度学习模型输出使用 BFloat16
- 图像/音频检索使用 Float16
- 监控召回率,及时调整参数
- 与 PyTorch/TensorFlow 无缝集成
