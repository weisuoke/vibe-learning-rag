# 核心概念 1：Embedding Functions（自动向量化）

> **Milvus 2.6 核心特性**：Embedding Functions 是 2026 年 Milvus 的标准向量化方案，将文本自动转换为向量，无需外部 API 调用。

---

## 什么是 Embedding Functions？

**Embedding Functions** 是 Milvus 2.6 引入的内置向量化能力，通过 **Function 模块**实现。它允许你直接插入原始文本数据，Milvus 会自动调用配置的 Embedding 提供商（如 OpenAI、Cohere、Bedrock 等）生成向量并存储。

### 核心价值

在 Milvus 2.6 之前，向量化流程是这样的：

```
用户代码 → 调用 OpenAI API → 获取向量 → 插入 Milvus
```

有了 Embedding Functions 后，流程变成：

```
用户代码 → 插入原始文本到 Milvus → Milvus 自动向量化并存储
```

**关键优势：**
- **简化开发**：无需管理外部 Embedding 服务
- **统一管理**：向量化逻辑与数据存储在同一系统
- **减少网络调用**：向量化在 Milvus 内部完成
- **降低维护成本**：不需要维护独立的 Embedding 服务

---

## Embedding Functions 的工作原理

### 1. Data-in, Data-out 工作流

Embedding Functions 基于 **Data-in, Data-out** 模式：

```
┌─────────────────────────────────────────────────────────────┐
│                    Milvus 2.6 内部流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 用户插入原始文本                                           │
│     ↓                                                         │
│  2. Function 模块自动调用 Embedding Provider                  │
│     ↓                                                         │
│  3. 获取向量并存储到 Vector Field                             │
│     ↓                                                         │
│  4. 用户查询时，自动向量化查询文本                             │
│     ↓                                                         │
│  5. 执行相似度检索                                            │
│     ↓                                                         │
│  6. 返回结果                                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Function 模块架构

**Function 模块**是 Milvus 2.6 的核心框架，负责数据转换和向量生成：

- **输入字段（Input Field）**：存储原始文本的 Scalar Field
- **输出字段（Output Field）**：存储生成向量的 Vector Field
- **Provider 配置**：指定使用哪个 Embedding 提供商
- **Model 配置**：指定使用哪个 Embedding 模型

---

## 支持的 Embedding 提供商

Milvus 2.6 支持多种主流 Embedding 提供商：

| Provider | 典型模型 | 向量类型 | 认证方式 |
|----------|----------|----------|----------|
| **OpenAI** | text-embedding-3-small<br>text-embedding-3-large | FLOAT_VECTOR | API Key |
| **Azure OpenAI** | Deployment-based | FLOAT_VECTOR | API Key |
| **Cohere** | embed-english-v3.0 | FLOAT_VECTOR<br>INT8_VECTOR | API Key |
| **AWS Bedrock** | amazon.titan-embed-text-v2 | FLOAT_VECTOR | AK/SK Pair |
| **Google Vertex AI** | text-embedding-005 | FLOAT_VECTOR | GCP Service Account JSON |
| **Voyage AI** | voyage-3<br>voyage-lite-02 | FLOAT_VECTOR<br>INT8_VECTOR | API Key |
| **Hugging Face TEI** | 任何 TEI 模型 | FLOAT_VECTOR | Optional API Key |

### 选择建议

- **OpenAI**：最常用，性能稳定，适合大多数场景
- **Cohere**：支持多语言，适合国际化应用
- **AWS Bedrock**：适合 AWS 生态，企业级安全
- **Hugging Face TEI**：适合自托管，成本可控

---

## 配置 Embedding Functions

### 步骤 1：配置凭证

在 `milvus.yaml` 中配置 API 密钥：

```yaml
# milvus.yaml
credential:
  # OpenAI API Key
  apikey_openai:
    apikey: "sk-your-openai-api-key"

  # Cohere API Key
  apikey_cohere:
    apikey: "your-cohere-api-key"

  # AWS Bedrock AK/SK
  aksk_bedrock:
    access_key_id: "your-access-key"
    secret_access_key: "your-secret-key"
```

### 步骤 2：配置 Provider

在 `milvus.yaml` 中配置 Embedding 提供商：

```yaml
function:
  textEmbedding:
    providers:
      openai:
        credential: apikey_openai  # 引用上面定义的凭证
        # url: https://api.openai.com/v1/embeddings  # 可选：自定义 URL

      cohere:
        credential: apikey_cohere
        enable: true
        url: "https://api.cohere.com/v2/embed"

      bedrock:
        credential: aksk_bedrock
        region: us-east-2
```

**重要提示：**
- 凭证名称（如 `apikey_openai`）可以自定义
- `milvus.yaml` 中的配置优先级高于环境变量
- 修改配置后需要重启 Milvus

---

## 使用 Embedding Functions

### 完整示例：使用 OpenAI Embedding Function

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

# 1. 连接 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 2. 创建 Schema
schema = client.create_schema()

# 添加主键字段
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)

# 添加文本字段（存储原始文本）
schema.add_field("document", DataType.VARCHAR, max_length=9000)

# 添加向量字段（存储生成的向量）
# 重要：dim 必须与 Embedding 模型的输出维度一致
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)

# 3. 定义 Embedding Function
text_embedding_function = Function(
    name="openai_embedding",                    # 唯一标识符
    function_type=FunctionType.TEXTEMBEDDING,   # 函数类型
    input_field_names=["document"],             # 输入字段（原始文本）
    output_field_names=["dense"],               # 输出字段（向量）
    params={
        "provider": "openai",                   # Embedding 提供商
        "model_name": "text-embedding-3-small", # Embedding 模型
        # 可选参数：
        # "credential": "apikey_openai",        # 凭证标签（如果有多个）
        # "dim": "1536",                        # 缩短向量维度
        # "user": "user123"                     # 用户标识（用于 API 追踪）
    }
)

# 4. 将 Embedding Function 添加到 Schema
schema.add_function(text_embedding_function)

# 5. 配置索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# 6. 创建 Collection
client.create_collection(
    collection_name='demo',
    schema=schema,
    index_params=index_params
)

# 7. 插入数据（直接插入原始文本，无需手动向量化）
client.insert('demo', [
    {'id': 1, 'document': 'Milvus simplifies semantic search through embeddings.'},
    {'id': 2, 'document': 'Vector embeddings convert text into searchable numeric data.'},
    {'id': 3, 'document': 'Semantic search helps users find relevant information quickly.'},
])

print("✅ 数据插入成功！Milvus 已自动生成向量。")

# 8. 查询（直接使用原始文本查询，无需手动向量化）
results = client.search(
    collection_name='demo',
    data=['How does Milvus help with semantic search?'],  # 原始文本查询
    anns_field='dense',
    limit=3,
    output_fields=['document'],
)

print("\n🔍 查询结果：")
for i, result in enumerate(results[0]):
    print(f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['document']}")
```

**输出示例：**

```
✅ 数据插入成功！Milvus 已自动生成向量。

🔍 查询结果：
1. Score: 0.8821, Content: Milvus simplifies semantic search through embeddings.
2. Score: 0.7543, Content: Vector embeddings convert text into searchable numeric data.
3. Score: 0.6234, Content: Semantic search helps users find relevant information quickly.
```

---

## 传统方式对比：手动向量化

### 传统方式：手动调用 OpenAI API

在 Milvus 2.6 之前，你需要手动调用 Embedding API：

```python
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import os

# 1. 连接 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 2. 创建 Schema（没有 Embedding Function）
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("document", DataType.VARCHAR, max_length=9000)
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)

# 3. 配置索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# 4. 创建 Collection
client.create_collection(
    collection_name='demo_traditional',
    schema=schema,
    index_params=index_params
)

# 5. 手动调用 OpenAI API 生成向量
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

documents = [
    'Milvus simplifies semantic search through embeddings.',
    'Vector embeddings convert text into searchable numeric data.',
    'Semantic search helps users find relevant information quickly.',
]

# 手动生成向量
response = openai_client.embeddings.create(
    input=documents,
    model="text-embedding-3-small"
)

embeddings = [item.embedding for item in response.data]

# 6. 插入数据（需要手动提供向量）
entities = []
for i, doc in enumerate(documents):
    entities.append({
        'id': i + 1,
        'document': doc,
        'dense': embeddings[i]  # 手动提供向量
    })

client.insert('demo_traditional', entities)
print("✅ 数据插入成功！")

# 7. 查询（需要手动向量化查询文本）
query_text = 'How does Milvus help with semantic search?'

# 手动生成查询向量
query_response = openai_client.embeddings.create(
    input=[query_text],
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

# 使用向量查询
results = client.search(
    collection_name='demo_traditional',
    data=[query_embedding],  # 手动提供查询向量
    anns_field='dense',
    limit=3,
    output_fields=['document'],
)

print("\n🔍 查询结果：")
for i, result in enumerate(results[0]):
    print(f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['document']}")
```

---

## 对比总结：Embedding Functions vs 传统方式

### 1. 代码复杂度对比

| 维度 | Embedding Functions (2026) | 传统方式 |
|------|----------------------------|----------|
| **代码行数** | ~50 行 | ~80 行 |
| **外部依赖** | 无需 `openai` 库 | 需要 `openai` 库 |
| **API 调用** | 0 次（Milvus 内部处理） | 2 次（插入 + 查询） |
| **错误处理** | Milvus 统一处理 | 需要手动处理 API 错误 |
| **配置管理** | `milvus.yaml` 统一配置 | 代码中分散配置 |

### 2. 工作流对比

**Embedding Functions 工作流（3 步）：**

```
1. 创建 Collection（配置 Embedding Function）
   ↓
2. 插入原始文本（自动向量化 + 自动索引）
   ↓
3. 查询（自动向量化查询文本 + 检索）
```

**传统方式工作流（5 步）：**

```
1. 创建 Collection
   ↓
2. 手动调用 Embedding API（插入数据）
   ↓
3. 插入向量
   ↓
4. 手动调用 Embedding API（查询）
   ↓
5. 查询
```

### 3. 性能对比

| 维度 | Embedding Functions | 传统方式 |
|------|---------------------|----------|
| **网络延迟** | 低（Milvus 内部优化） | 高（多次外部 API 调用） |
| **批量插入** | 自动批处理 | 需要手动批处理 |
| **错误重试** | 内置重试机制 | 需要手动实现 |
| **并发控制** | Milvus 自动管理 | 需要手动管理 |

### 4. 维护成本对比

| 维度 | Embedding Functions | 传统方式 |
|------|---------------------|----------|
| **API Key 管理** | `milvus.yaml` 统一管理 | 代码中分散管理 |
| **模型切换** | 修改配置即可 | 需要修改代码 |
| **版本升级** | Milvus 统一升级 | 需要手动升级 SDK |
| **监控日志** | Milvus 统一监控 | 需要自建监控 |

---

## 何时使用 Embedding Functions？

### ✅ 推荐使用 Embedding Functions 的场景

1. **新项目开发**：2026 年的标准方案，简化开发流程
2. **RAG 应用**：自动向量化文档和查询，专注业务逻辑
3. **多模型切换**：需要频繁切换 Embedding 模型
4. **团队协作**：统一向量化逻辑，降低维护成本
5. **生产环境**：需要稳定的向量化服务

### ⚠️ 考虑传统方式的场景

1. **自定义 Embedding 逻辑**：需要对向量进行特殊处理（如降维、归一化）
2. **离线向量化**：已有预计算的向量，直接插入即可
3. **特殊 Embedding 模型**：使用 Milvus 不支持的 Embedding 提供商
4. **精细控制**：需要对每次 API 调用进行精细控制（如超时、重试策略）

---

## 常见问题

### 1. Embedding Functions 会影响性能吗？

**不会。** Embedding Functions 在 Milvus 内部进行了优化：
- **批处理**：自动将多个请求合并为批量请求
- **缓存**：相同文本的向量会被缓存
- **并发控制**：自动管理并发请求，避免 API 限流

### 2. 如何切换 Embedding 模型？

修改 `milvus.yaml` 中的 `model_name` 参数，然后重启 Milvus：

```yaml
function:
  textEmbedding:
    providers:
      openai:
        credential: apikey_openai
        model_name: "text-embedding-3-large"  # 切换到更大的模型
```

### 3. 可以同时使用多个 Embedding 提供商吗？

**可以。** 在同一个 Collection 中，你可以为不同的 Vector Field 配置不同的 Embedding Function：

```python
# 为文本配置 OpenAI Embedding
text_embedding_function = Function(
    name="openai_text_embedding",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["text_vector"],
    params={"provider": "openai", "model_name": "text-embedding-3-small"}
)

# 为图片配置 Cohere Embedding
image_embedding_function = Function(
    name="cohere_image_embedding",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["image_description"],
    output_field_names=["image_vector"],
    params={"provider": "cohere", "model_name": "embed-english-v3.0"}
)

schema.add_function(text_embedding_function)
schema.add_function(image_embedding_function)
```

### 4. Embedding Functions 支持自定义模型吗？

**支持。** 使用 **Hugging Face TEI** 提供商，你可以部署自己的 Embedding 模型：

```yaml
function:
  textEmbedding:
    providers:
      tei:
        enable: true
        url: "http://your-tei-server:8080"  # 自定义 TEI 服务地址
```

---

## 最佳实践

### 1. 选择合适的 Embedding 模型

- **小型应用**：`text-embedding-3-small`（1536 维，成本低）
- **高精度应用**：`text-embedding-3-large`（3072 维，精度高）
- **多语言应用**：Cohere `embed-multilingual-v3.0`

### 2. 配置合理的向量维度

OpenAI 的 `text-embedding-3-*` 模型支持缩短向量维度：

```python
params={
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "dim": "768"  # 从 1536 缩短到 768，降低存储成本
}
```

**权衡：**
- **维度越高**：精度越高，但存储和计算成本越高
- **维度越低**：成本越低，但精度会下降

### 3. 使用环境变量管理敏感信息

对于 Docker Compose 部署，使用环境变量管理 API Key：

```yaml
# docker-compose.yaml
services:
  standalone:
    environment:
      MILVUSAI_OPENAI_API_KEY: ${OPENAI_API_KEY}
```

### 4. 监控 Embedding 调用

在生产环境中，监控 Embedding API 的调用情况：
- **调用次数**：避免超出 API 限额
- **响应时间**：检测 API 性能问题
- **错误率**：及时发现配置问题

---

## 总结

**Embedding Functions** 是 Milvus 2.6 的核心特性，它将向量化从外部服务变成了数据库的内置能力。通过 Embedding Functions，你可以：

1. **简化开发**：无需手动调用 Embedding API
2. **降低维护成本**：统一管理向量化逻辑
3. **提升性能**：减少网络调用，优化批处理
4. **增强可靠性**：内置错误处理和重试机制

**核心原则：**
- **2026 年新项目优先使用 Embedding Functions**
- **传统方式仅用于特殊场景**
- **选择合适的 Embedding 提供商和模型**
- **在 `milvus.yaml` 中统一管理配置**

**下一步：** 学习如何使用 Embedding Functions 进行数据插入操作（Insert、Upsert、Bulk Insert）。
