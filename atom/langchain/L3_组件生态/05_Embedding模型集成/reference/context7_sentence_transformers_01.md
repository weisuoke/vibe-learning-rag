---
type: context7_documentation
library: sentence-transformers
version: latest
fetched_at: 2026-02-25
knowledge_point: Embedding模型集成
context7_query: SentenceTransformer model usage, encode method, batch encoding, model selection for embeddings
---

# Context7 文档：Sentence Transformers

## 文档来源
- 库名称：sentence-transformers
- Context7 ID：/huggingface/sentence-transformers
- 官方文档链接：https://github.com/huggingface/sentence-transformers

## 关键信息提取

### 1. 基础使用模式

**加载模型**：
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

**生成嵌入**：
```python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

### 2. encode() 方法的高级特性

**返回格式控制**：
```python
# 返回 numpy 数组（默认）
embeddings = model.encode(sentences)

# 返回 PyTorch tensor
embeddings_tensor = model.encode(sentences, convert_to_tensor=True)
```

**归一化**：
```python
# 归一化嵌入（用于点积相似度）
embeddings_normalized = model.encode(sentences, normalize_embeddings=True)
```

**维度截断（Matryoshka 模型）**：
```python
# 截断维度
embeddings_truncated = model.encode(sentences, truncate_dim=128)
print(embeddings_truncated.shape)  # (3, 128)
```

**量化**：
```python
# 二进制量化（节省存储空间）
embeddings_binary = model.encode(sentences, precision="binary")
print(embeddings_binary.dtype)  # uint8
```

### 3. 批量处理

**批量编码**：
```python
large_corpus = ["sentence " + str(i) for i in range(10000)]
corpus_embeddings = model.encode(
    large_corpus,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    device="cuda"
)
```

### 4. 自定义 Prompt

**查询 Prompt**：
```python
query_embedding = model.encode(
    "What is semantic search?",
    prompt="query: "
)
```

### 5. 信息检索专用方法

**文档和查询分离**：
- `encode_query()` - 编码查询
- `encode_document()` - 编码文档

### 6. 相似度计算

**使用 similarity() 方法**：
```python
similarity_scores = model.similarity(embeddings1, embeddings2)
```

## 与 LangChain 集成的关联

1. **模型选择**：`all-MiniLM-L6-v2` 是常用的轻量级模型
2. **批量处理**：支持 `batch_size` 参数
3. **归一化**：与 LangChain 的向量存储兼容
4. **设备选择**：支持 CPU/GPU
5. **进度条**：用户友好的批量处理

## 性能优化建议

1. **批量大小**：根据 GPU 内存调整 `batch_size`
2. **归一化**：使用 `normalize_embeddings=True` 提高检索效率
3. **量化**：使用 `precision="binary"` 节省存储空间
4. **设备选择**：使用 `device="cuda"` 加速计算
