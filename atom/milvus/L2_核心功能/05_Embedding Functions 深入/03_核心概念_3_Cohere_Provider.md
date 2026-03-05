# 核心概念：Cohere Provider

> Cohere Provider是Milvus支持的多语言embedding提供商，支持Int8量化和灵活的截断策略

---

## 提供商概述

**Cohere Provider**是Milvus支持的专注于多语言场景的embedding提供商，特别适合需要处理多种语言的RAG应用。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 96 | 第三高的批量大小 |
| **输出类型** | Float32/Int8 | 支持量化输出 |
| **维度控制** | ✅ 支持 | embed-v3系列支持自定义维度 |
| **延迟** | ~250ms | 北美环境单次API调用延迟 |
| **成本** | $0.10/M | 中等成本 |
| **多语言** | ⭐⭐⭐⭐⭐ | 100+语言支持 |

### 核心优势

1. **多语言支持**：支持100+语言，包括中文、日文、韩文等
2. **Int8量化**：支持Int8输出，存储空间减少75%
3. **灵活截断**：支持START/END/NONE三种截断策略
4. **输入类型区分**：区分文档和查询的embedding类型

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

cohere_ef = Function(
    name="cohere_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "cohere",                      # 必需：提供商名称
        "model_name": "embed-english-v3.0",        # 必需：模型名称
        "api_key": "your-cohere-api-key"           # 必需：API密钥
    }
)
```

### 可选参数

```python
params = {
    "provider": "cohere",
    "model_name": "embed-english-v3.0",
    "api_key": "your-cohere-api-key",

    # 可选参数
    "dim": 1024,                                   # 输出维度（embed-v3系列支持）
    "truncate": "END",                             # 截断策略：NONE/START/END
    "output_type": "float",                        # 输出类型：float/int8
    "url": "https://api.cohere.com/v2/embed"       # 自定义端点
}
```

### 参数说明

#### 1. model_name（模型名称）

**支持的模型**：

| 模型 | 默认维度 | 语言支持 | 成本($/M tokens) | 适用场景 |
|------|----------|----------|------------------|---------|
| **embed-english-v3.0** | 1024 | 英文 | 0.10 | 英文RAG |
| **embed-multilingual-v3.0** | 1024 | 100+语言 | 0.10 | 多语言RAG |
| **embed-english-light-v3.0** | 384 | 英文 | 0.10 | 轻量级应用 |
| **embed-multilingual-light-v3.0** | 384 | 100+语言 | 0.10 | 轻量级多语言 |
| **embed-english-v2.0** | 4096 | 英文 | 0.10 | 旧版（不推荐） |

**选择建议**：
- 英文场景：使用`embed-english-v3.0`
- 多语言场景：使用`embed-multilingual-v3.0`
- 轻量级应用：使用`embed-*-light-v3.0`
- 避免使用：v2.0系列（已过时，不支持input_type）

#### 2. truncate（截断策略）

**Cohere特有的截断策略**

```python
# 策略1：END（默认）- 截断末尾
params = {"truncate": "END"}
# 文本："This is a very long document..."
# 超过最大长度时，保留开头，截断末尾

# 策略2：START - 截断开头
params = {"truncate": "START"}
# 文本："This is a very long document..."
# 超过最大长度时，保留末尾，截断开头

# 策略3：NONE - 不截断，报错
params = {"truncate": "NONE"}
# 文本超过最大长度时，API返回错误
```

**截断策略对比**：

| 策略 | 行为 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **END** | 保留开头 | 文档摘要、标题重要 | 保留关键信息 | 可能丢失结论 |
| **START** | 保留末尾 | 结论重要、时间序列 | 保留最新信息 | 可能丢失背景 |
| **NONE** | 报错 | 严格控制输入长度 | 不丢失信息 | 需要手动处理 |

**实战示例**：

```python
# 场景1：新闻文章（标题和开头重要）
params = {"truncate": "END"}  # 保留标题和开头

# 场景2：聊天记录（最新消息重要）
params = {"truncate": "START"}  # 保留最新消息

# 场景3：法律文档（不能丢失任何信息）
params = {"truncate": "NONE"}  # 报错，手动分块处理
```

#### 3. output_type（输出类型）

**Cohere支持Int8量化输出**

```python
# Float32输出（默认）
params = {"output_type": "float"}
# 输出：[-0.123, 0.456, -0.789, ...]
# 存储：4字节/维度

# Int8输出（量化）
params = {"output_type": "int8"}
# 输出：[-12, 45, -78, ...]
# 存储：1字节/维度
```

**Int8量化对比**：

| 维度 | Float32 | Int8 | 存储节省 | 质量损失 |
|------|---------|------|----------|---------|
| 1024 | 4KB | 1KB | 75% | <3% |
| 384 | 1.5KB | 384B | 75% | <3% |

**何时使用Int8**：
- ✅ 大规模数据（百万级以上）
- ✅ 存储成本敏感
- ✅ 查询速度要求高
- ❌ 对质量要求极高的场景

#### 4. dim（维度控制）

**仅embed-v3系列支持**

```python
# embed-english-v3.0: 默认1024维
params = {"model_name": "embed-english-v3.0", "dim": 1024}  # 默认

# 降低维度
params = {"model_name": "embed-english-v3.0", "dim": 512}   # 节省50%存储
params = {"model_name": "embed-english-v3.0", "dim": 256}   # 节省75%存储

# 支持的维度值
# embed-english-v3.0: 任意 ≤ 1024
# embed-multilingual-v3.0: 任意 ≤ 1024
# embed-*-light-v3.0: 任意 ≤ 384
```

---

## Cohere特有功能

### 1. 输入类型区分（Input Type）

**Cohere区分文档和查询的embedding类型**

```python
# Milvus自动处理输入类型
# InsertMode → input_type = "search_document"
# SearchMode → input_type = "search_query"

# 插入文档
collection.insert([{"text": "This is a document"}])
# Cohere API调用：input_type = "search_document"

# 查询
collection.search(data=["query text"], anns_field="vector")
# Cohere API调用：input_type = "search_query"
```

**为什么需要区分**：
- **文档embedding**：捕获完整语义和细节
- **查询embedding**：捕获意图和关键词
- **不对称embedding**：提升检索质量

**注意**：v2.0模型不支持input_type参数

### 2. 多语言支持

**Cohere支持100+语言**

```python
# 多语言模型
params = {"model_name": "embed-multilingual-v3.0"}

# 支持的语言（部分）
texts = [
    "Hello world",                    # 英文
    "你好世界",                       # 中文
    "こんにちは世界",                 # 日文
    "안녕하세요 세계",                # 韩文
    "Hola mundo",                     # 西班牙文
    "Bonjour le monde",               # 法文
    "Hallo Welt",                     # 德文
    "Привет мир",                     # 俄文
    "مرحبا بالعالم",                  # 阿拉伯文
    "नमस्ते दुनिया"                   # 印地文
]

collection.insert([{"text": t} for t in texts])
# Cohere自动处理所有语言
```

### 3. Embed Jobs API（批量处理）

**Cohere提供专门的批量embedding API**

```python
# 注意：Milvus Embedding Functions不直接支持Embed Jobs API
# 但可以通过Cohere SDK使用

from cohere import Client

client = Client(api_key="your-api-key")

# 创建embed job（适合大规模数据）
job = client.embed_jobs.create(
    model="embed-english-v3.0",
    dataset_id="your-dataset-id",
    input_type="search_document"
)

# 查询job状态
status = client.embed_jobs.get(job.id)

# 获取结果
if status.status == "complete":
    embeddings = client.embed_jobs.get_embeddings(job.id)
```

**Embed Jobs适用场景**：
- 100K+文档的初始化
- 定期批量更新
- 离线处理

---

## 完整配置示例

### 示例1：基础多语言配置

```python
from pymilvus import MilvusClient, Function, DataType, FunctionType
import os

# 设置环境变量
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

# 创建客户端
client = MilvusClient("http://localhost:19530")

# 定义Cohere embedding函数
cohere_ef = Function(
    name="cohere_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "cohere",
        "model_name": "embed-multilingual-v3.0",
        "dim": 512,
        "truncate": "END"
    }
)

# 创建schema
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=2000)
schema.add_field("language", DataType.VARCHAR, max_length=50)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=512)
schema.add_function(cohere_ef)

# 创建collection
client.create_collection("multilingual_docs", schema=schema)

# 插入多语言数据
client.insert("multilingual_docs", [
    {"text": "Artificial intelligence is transforming the world", "language": "en"},
    {"text": "人工智能正在改变世界", "language": "zh"},
    {"text": "人工知能が世界を変えている", "language": "ja"}
])

# 多语言查询
results = client.search(
    "multilingual_docs",
    data=["AI technology"],
    anns_field="vector",
    limit=5
)
```

### 示例2：Int8量化配置

```python
# 使用Int8量化节省存储
cohere_ef = Function(
    name="cohere_ef",
    params={
        "provider": "cohere",
        "model_name": "embed-english-v3.0",
        "output_type": "int8",  # Int8量化
        "dim": 1024
    }
)

# 注意：schema中的向量类型仍然是FLOAT_VECTOR
# Milvus会自动处理Int8到Float的转换
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)

# 存储对比
# Float32: 1M向量 × 1024维 × 4字节 = 4GB
# Int8: 1M向量 × 1024维 × 1字节 = 1GB
# 节省：75%
```

### 示例3：截断策略配置

```python
# 场景：处理长文档，保留开头
cohere_ef = Function(
    name="cohere_ef",
    params={
        "provider": "cohere",
        "model_name": "embed-english-v3.0",
        "truncate": "END",  # 保留开头，截断末尾
        "dim": 512
    }
)

# 插入长文档
long_text = "This is a very long document..." * 1000  # 超过最大长度
collection.insert([{"text": long_text}])
# Cohere自动截断末尾，保留开头
```

---

## 常见问题与解决方案

### 问题1：v2.0模型不支持input_type

**错误信息**：
```
MilvusException: input_type not supported for v2.0 models
```

**解决方案**：
```python
# ❌ 错误：使用v2.0模型
params = {"model_name": "embed-english-v2.0"}

# ✅ 正确：使用v3.0模型
params = {"model_name": "embed-english-v3.0"}
```

### 问题2：文本超过最大长度

**错误信息**：
```
MilvusException: text exceeds maximum length
```

**解决方案**：
```python
# 方案1：使用END截断（推荐）
params = {"truncate": "END"}

# 方案2：手动分块
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# 使用
long_text = "very long text..."
chunks = chunk_text(long_text)
for chunk in chunks:
    collection.insert([{"text": chunk}])
```

### 问题3：Int8量化后质量下降

**问题描述**：使用Int8后检索质量明显下降

**解决方案**：
```python
# 1. 评估质量损失
def evaluate_quality(collection, queries, ground_truth):
    # Float32结果
    float_results = collection.search(queries, output_type="float")

    # Int8结果
    int8_results = collection.search(queries, output_type="int8")

    # 计算Recall@K
    recall_float = calculate_recall(float_results, ground_truth)
    recall_int8 = calculate_recall(int8_results, ground_truth)

    print(f"Float32 Recall: {recall_float}")
    print(f"Int8 Recall: {recall_int8}")
    print(f"Quality Loss: {(recall_float - recall_int8) / recall_float * 100}%")

# 2. 如果质量损失>5%，考虑使用Float32
if quality_loss > 0.05:
    params = {"output_type": "float"}
```

### 问题4：多语言混合查询

**问题描述**：查询语言与文档语言不同

**解决方案**：
```python
# Cohere的多语言模型支持跨语言检索
params = {"model_name": "embed-multilingual-v3.0"}

# 插入中文文档
collection.insert([{"text": "人工智能正在改变世界"}])

# 使用英文查询
results = collection.search(data=["artificial intelligence"], anns_field="vector")
# Cohere自动处理跨语言语义匹配
```

---

## 最佳实践

### 1. 模型选择

```python
# 场景1：纯英文RAG
params = {"model_name": "embed-english-v3.0"}

# 场景2：多语言RAG
params = {"model_name": "embed-multilingual-v3.0"}

# 场景3：轻量级应用（移动端、边缘设备）
params = {"model_name": "embed-english-light-v3.0"}

# 场景4：大规模存储（百万级以上）
params = {
    "model_name": "embed-english-v3.0",
    "output_type": "int8",  # 节省75%存储
    "dim": 512              # 进一步降低维度
}
```

### 2. 截断策略选择

```python
# 场景1：新闻文章、博客（标题和开头重要）
params = {"truncate": "END"}

# 场景2：聊天记录、日志（最新信息重要）
params = {"truncate": "START"}

# 场景3：法律文档、合同（不能丢失信息）
params = {"truncate": "NONE"}
# 配合手动分块
```

### 3. 性能优化

```python
# 1. 使用较小的模型
params = {"model_name": "embed-english-light-v3.0"}  # 384维 vs 1024维

# 2. 降低维度
params = {"dim": 512}  # 而非1024

# 3. 使用Int8量化
params = {"output_type": "int8"}

# 4. 批量处理
# Milvus自动批处理，MaxBatch=96
```

### 4. 成本控制

```python
# 1. 使用light模型（维度更小）
params = {"model_name": "embed-english-light-v3.0"}

# 2. 降低维度
params = {"dim": 256}  # 最小维度

# 3. 缓存embedding结果
embedding_cache = {}

def get_embedding_with_cache(text):
    if text in embedding_cache:
        return embedding_cache[text]

    result = collection.insert([{"text": text}])
    embedding_cache[text] = result
    return result
```

---

## 与其他提供商对比

### 性能对比

| 维度 | Cohere | OpenAI | VoyageAI |
|------|--------|--------|----------|
| **MaxBatch** | 96 | 128 | 128 |
| **延迟** | 250ms | 300ms | 200ms |
| **成本** | $0.10/M | $0.02/M | $0.12/M |
| **多语言** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Int8支持** | ✅ | ❌ | ✅ |
| **截断策略** | ✅ | ❌ | ❌ |

### 选择建议

**选择Cohere的场景**：
- ✅ 多语言RAG应用
- ✅ 需要Int8量化
- ✅ 需要灵活的截断策略
- ✅ 跨语言语义检索

**选择其他提供商的场景**：
- OpenAI：纯英文、追求最低成本
- VoyageAI：追求最快速度、大规模RAG

---

## 参考资源

### 官方文档
- Cohere Embed API: https://docs.cohere.com/reference/embed
- Cohere Embed Jobs API: https://docs.cohere.com/docs/embed-jobs-api
- Milvus Embedding Functions: https://milvus.io/docs/embeddings.md

### 源代码
- Milvus Cohere Provider: `sourcecode/milvus/internal/util/function/embedding/cohere_embedding_provider.go`

---

**版本信息**：
- Milvus版本：2.6+
- Cohere API版本：v2
- 文档版本：v1.0
- 最后更新：2026-02-24
