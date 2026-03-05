# 核心概念 1: Embedding Functions

> **Milvus 2.6 核心特性：自动向量化的 Data-in, Data-out 模式**

---

## 概述

Embedding Functions 是 Milvus 2.6 引入的革命性特性，它将嵌入生成（Embedding Generation）从用户代码中解放出来，内置到数据库层。这个特性实现了真正的 **Data-in, Data-out** 模式：输入原始文本，输出原始文本，向量化过程对用户完全透明。

### 核心价值

**传统方式的痛点**:
```python
# 用户需要管理嵌入逻辑
embeddings = openai_client.embeddings.create(...)  # 手动调用
milvus_client.insert(embeddings)                   # 插入向量
```

**Milvus 2.6 的解决方案**:
```python
# Milvus 自动处理嵌入
milvus_client.insert({"document": "原始文本"})  # 直接插入文本
```

---

## 什么是 Embedding Functions？

### 定义

Embedding Functions 是 Milvus 2.6 中的 **Function 模块**的一部分，它允许你通过配置的方式指定嵌入模型提供商（如 OpenAI、Cohere、AWS Bedrock 等），Milvus 会自动调用这些外部服务将原始文本转换为向量嵌入。

### 核心机制

```
用户代码                    Milvus (Function 模块)              外部嵌入服务
   |                              |                                  |
   |--插入原始文本--------------->|                                  |
   |                              |--调用嵌入API------------------->|
   |                              |<-返回向量------------------------|
   |                              |--存储向量到collection           |
   |<-返回成功--------------------|                                  |
```

### 关键特点

1. **自动化**: 无需手动调用嵌入 API
2. **一致性**: 插入和查询自动使用同一模型
3. **透明性**: 向量化过程对用户透明
4. **多提供商**: 支持 9+ 主流嵌入服务提供商

---

## Data-in, Data-out 模式

### 传统流程（5步）

```python
# 步骤1: 准备数据
docs = ["Milvus is a vector database"]

# 步骤2: 手动生成嵌入
from openai import OpenAI
client = OpenAI()
embeddings = []
for doc in docs:
    emb = client.embeddings.create(
        input=doc,
        model="text-embedding-3-small"
    )
    embeddings.append(emb.data[0].embedding)

# 步骤3: 插入向量
milvus_client.insert({
    "id": 1,
    "vector": embeddings[0],
    "text": docs[0]
})

# 步骤4: 查询时再次生成嵌入
query = "What is Milvus?"
query_emb = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
).data[0].embedding

# 步骤5: 搜索
results = milvus_client.search(data=[query_emb])
```

### Milvus 2.6 流程（3步）

```python
# 步骤1: 配置 Embedding Function（创建 collection 时一次性配置）
schema.add_function(
    Function(
        name="openai_embedding",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=["document"],
        output_field_names=["dense"],
        params={
            "provider": "openai",
            "model_name": "text-embedding-3-small"
        }
    )
)

# 步骤2: 直接插入原始文本（自动嵌入）
milvus_client.insert({"document": "Milvus is a vector database"})

# 步骤3: 直接用文本查询（自动嵌入）
results = milvus_client.search(data=["What is Milvus?"])
```

**效率提升**:
- 代码量减少 **75%**
- 步骤减少 **40%**（5步 → 3步）
- 维护成本降低 **80%**

---

## 支持的嵌入服务提供商

Milvus 2.6 支持 9+ 主流嵌入服务提供商：

| 提供商 | 典型模型 | 向量类型 | 认证方式 | 适用场景 |
|--------|----------|----------|----------|----------|
| **OpenAI** | text-embedding-3-small<br>text-embedding-3-large | FLOAT_VECTOR | API Key | 通用场景，质量高 |
| **Azure OpenAI** | 基于部署的模型 | FLOAT_VECTOR | API Key | 企业级，Azure 生态 |
| **Cohere** | embed-english-v3.0<br>embed-multilingual-v3.0 | FLOAT_VECTOR<br>INT8_VECTOR | API Key | 多语言支持 |
| **AWS Bedrock** | amazon.titan-embed-text-v2 | FLOAT_VECTOR | AK/SK | AWS 生态集成 |
| **Google Vertex AI** | text-embedding-005 | FLOAT_VECTOR | GCP Service Account | GCP 生态集成 |
| **Voyage AI** | voyage-3<br>voyage-lite-02 | FLOAT_VECTOR<br>INT8_VECTOR | API Key | 高性能检索 |
| **Jina AI** | jina-embeddings-v3 | FLOAT_VECTOR | API Key | 多模态支持 |
| **DashScope** | text-embedding-v3 | FLOAT_VECTOR | API Key | 阿里云生态 |
| **SiliconFlow** | BAAI/bge-large-zh-v1.5 | FLOAT_VECTOR | API Key | 中文优化 |
| **Hugging Face TEI** | 任何 TEI 服务的模型 | FLOAT_VECTOR | Optional API Key | 自托管模型 |

### 提供商选择建议

**通用场景**:
- **OpenAI**: 质量最高，成本适中，推荐用于生产环境
- **Cohere**: 多语言支持好，适合国际化应用

**云生态集成**:
- **AWS Bedrock**: 已在 AWS 上的应用
- **Azure OpenAI**: 已在 Azure 上的应用
- **Vertex AI**: 已在 GCP 上的应用

**成本优化**:
- **Voyage AI**: 性价比高，支持 INT8 量化
- **SiliconFlow**: 中文场景成本低

**自托管**:
- **Hugging Face TEI**: 完全控制，无外部依赖

---

## 工作原理

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Milvus 2.6 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  用户代码                                                     │
│    │                                                          │
│    │ 1. insert({"document": "原始文本"})                     │
│    ▼                                                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Function 模块                           │       │
│  │  ┌────────────────────────────────────────┐     │       │
│  │  │  1. 检测到 document 字段                │     │       │
│  │  │  2. 查找配置的 Embedding Function       │     │       │
│  │  │  3. 调用外部嵌入服务 API                │     │  ────┼──> OpenAI/Cohere/...
│  │  │  4. 接收向量结果                        │     │  <───┼──  返回向量
│  │  │  5. 存储到 dense 字段                   │     │       │
│  │  └────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────┘       │
│    │                                                          │
│    │ 2. 向量存储到 Collection                                │
│    ▼                                                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │           向量存储引擎                            │       │
│  │  - 索引构建                                       │       │
│  │  - 向量存储                                       │       │
│  │  - 元数据管理                                     │       │
│  └──────────────────────────────────────────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 六个关键步骤

1. **输入原始数据**: 用户插入原始文本到 Milvus
2. **自动生成嵌入**: Function 模块自动调用配置的嵌入服务
3. **存储向量**: 生成的向量存储到指定的向量字段
4. **提交查询**: 用户提交原始文本查询
5. **语义搜索**: Milvus 自动嵌入查询文本并执行相似度搜索
6. **返回结果**: 返回最相关的原始数据

---

## 配置方法

### 方式1: milvus.yaml 配置（推荐）

**步骤1: 配置凭证**

编辑 `milvus.yaml` 文件：

```yaml
# 凭证配置
credential:
  # AWS Bedrock 或其他使用 AK/SK 的服务
  aksk1:
    access_key_id: <YOUR_ACCESS_KEY>
    secret_access_key: <YOUR_SECRET_KEY>

  # OpenAI, Cohere, Voyage 等使用 API Key 的服务
  apikey1:
    apikey: <YOUR_API_KEY>

  # Google Vertex AI 使用 Service Account
  gcp1:
    credential_json: <BASE64_ENCODED_JSON>
```

**步骤2: 配置提供商**

```yaml
function:
  textEmbedding:
    providers:
      # OpenAI 配置
      openai:
        credential: apikey1
        # url: https://api.openai.com/v1  # 可选：自定义端点

      # AWS Bedrock 配置
      bedrock:
        credential: aksk1
        region: us-east-2

      # Google Vertex AI 配置
      vertexai:
        credential: gcp1
        # url: https://us-central1-aiplatform.googleapis.com

      # Cohere 配置
      cohere:
        credential: apikey1
        enable: true
```

**步骤3: 重启 Milvus**

```bash
docker-compose restart
```

### 方式2: 环境变量配置

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Cohere
export COHERE_API_KEY="..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-2"
```

---

## 使用步骤

### 完整示例

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

# 1. 初始化客户端
client = MilvusClient(uri="http://localhost:19530")

# 2. 创建 Schema
schema = client.create_schema()

# 添加字段
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("document", DataType.VARCHAR, max_length=5000)  # 原始文本
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)       # 向量字段

# 3. 配置 Embedding Function（核心步骤）
embedding_function = Function(
    name="openai_embedding",                      # 函数名称（唯一标识）
    function_type=FunctionType.TEXTEMBEDDING,     # 函数类型：文本嵌入
    input_field_names=["document"],               # 输入字段：原始文本
    output_field_names=["dense"],                 # 输出字段：向量
    params={
        "provider": "openai",                     # 提供商
        "model_name": "text-embedding-3-small",   # 模型名称
        # 可选参数：
        # "dim": 1536,                            # 向量维度（某些模型支持调整）
        # "credential": "apikey1",                # 凭证标签（覆盖默认）
    }
)

# 将 Embedding Function 添加到 Schema
schema.add_function(embedding_function)

# 4. 配置索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense",
    index_type="AUTOINDEX",      # 自动选择最优索引
    metric_type="COSINE"          # 余弦相似度
)

# 5. 创建 Collection
collection_name = "my_rag_collection"
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

# 6. 插入数据（直接插入原始文本）
documents = [
    {"document": "Milvus is an open-source vector database."},
    {"document": "Milvus 2.6 introduces Embedding Functions."},
    {"document": "RAG applications benefit from automatic vectorization."},
]

client.insert(collection_name=collection_name, data=documents)
print(f"✅ 插入了 {len(documents)} 条文档")

# 7. 查询（直接用文本查询）
query = "What is Milvus 2.6's new feature?"

results = client.search(
    collection_name=collection_name,
    data=[query],                 # 直接传入文本，不是向量！
    anns_field="dense",
    limit=3,
    output_fields=["document"]
)

# 8. 显示结果
for i, hit in enumerate(results[0], 1):
    print(f"{i}. [相似度: {hit['distance']:.4f}] {hit['entity']['document']}")
```

### 参数详解

**Function 参数**:

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | str | ✅ | 函数的唯一标识符 |
| `function_type` | FunctionType | ✅ | 函数类型，使用 `FunctionType.TEXTEMBEDDING` |
| `input_field_names` | List[str] | ✅ | 输入字段列表（原始文本字段） |
| `output_field_names` | List[str] | ✅ | 输出字段列表（向量字段） |
| `params` | Dict | ✅ | 提供商特定的配置参数 |

**params 参数**:

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `provider` | str | ✅ | 嵌入服务提供商（如 "openai", "cohere"） |
| `model_name` | str | ✅ | 模型名称（如 "text-embedding-3-small"） |
| `credential` | str | ❌ | 凭证标签（覆盖 milvus.yaml 中的默认配置） |
| `dim` | int | ❌ | 向量维度（某些模型支持调整，如 OpenAI） |
| `user` | str | ❌ | 用户标识（用于 API 跟踪） |

---

## 限制与约束

### 字段类型限制

**输入字段**:
- ✅ 必须是 `VARCHAR` 类型
- ❌ 不支持动态字段（dynamic fields）
- ❌ 不能为 `NULL`（必须有值）

**输出字段**:
- ✅ 支持 `FLOAT_VECTOR`
- ✅ 支持 `INT8_VECTOR`（某些提供商）
- ❌ 不支持 `BINARY_VECTOR`
- ❌ 不支持 `FLOAT16_VECTOR`
- ❌ 不支持 `BFLOAT16_VECTOR`

### 使用限制

1. **NULL 值处理**: 如果输入字段为 NULL，Function 模块会抛出错误
2. **动态字段**: Function 模块不处理动态字段（$meta 中的字段）
3. **Schema 定义**: 输入和输出字段必须在 Schema 中显式定义

### 示例：错误处理

```python
# ❌ 错误：输入字段为 NULL
client.insert(collection_name, [
    {"document": None}  # 会抛出错误
])

# ✅ 正确：提供有效值
client.insert(collection_name, [
    {"document": "Valid text"}
])

# ❌ 错误：使用动态字段
schema.enable_dynamic_field = True
client.insert(collection_name, [
    {"document": "text", "extra_field": "value"}  # extra_field 不会被嵌入
])
```

---

## RAG 应用场景

### 场景1: 文档问答系统

```python
# 插入文档
documents = [
    {"document": "Milvus is a vector database for AI applications."},
    {"document": "RAG combines retrieval and generation for better answers."},
    {"document": "Embedding Functions simplify RAG development."},
]
client.insert(collection_name, documents)

# 用户提问
question = "How does RAG work?"
results = client.search(
    collection_name,
    data=[question],
    anns_field="dense",
    limit=3
)

# 将检索结果传给 LLM
context = "\n".join([hit['entity']['document'] for hit in results[0]])
llm_response = call_llm(question, context)
```

### 场景2: 知识库检索

```python
# 插入知识库文章
articles = [
    {"document": "Article 1: Introduction to Vector Databases..."},
    {"document": "Article 2: Building RAG Applications..."},
    {"document": "Article 3: Optimizing Search Performance..."},
]
client.insert(collection_name, articles)

# 语义搜索
query = "How to optimize vector search?"
results = client.search(collection_name, data=[query], limit=5)
```

### 场景3: 智能客服

```python
# 插入常见问题
faqs = [
    {"document": "Q: How to reset password? A: Click 'Forgot Password'..."},
    {"document": "Q: How to upgrade plan? A: Go to Settings > Billing..."},
]
client.insert(collection_name, faqs)

# 用户咨询
user_query = "I can't log in"
results = client.search(collection_name, data=[user_query], limit=1)
# 返回最相关的 FAQ
```

---

## 性能优化

### 批量插入优化

```python
# ❌ 低效：逐条插入
for doc in documents:
    client.insert(collection_name, [doc])  # 每次调用一次嵌入 API

# ✅ 高效：批量插入
client.insert(collection_name, documents)  # Milvus 批量调用嵌入 API
```

### 成本优化

**1. 选择合适的模型**:
```python
# 高质量但成本高
params={"provider": "openai", "model_name": "text-embedding-3-large"}

# 平衡质量和成本
params={"provider": "openai", "model_name": "text-embedding-3-small"}

# 成本最低
params={"provider": "voyage", "model_name": "voyage-lite-02"}
```

**2. 使用向量量化**:
```python
# 使用 INT8 向量（某些提供商支持）
schema.add_field("dense", DataType.INT8_VECTOR, dim=1536)
params={"provider": "cohere", "model_name": "embed-english-v3.0"}
```

**3. 调整向量维度**:
```python
# OpenAI 支持调整维度
params={
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "dim": 512  # 从 1536 降到 512，减少存储和计算成本
}
```

---

## 常见问题

### Q1: Embedding Functions 与传统方式的性能对比？

**A**: 性能相当或更好：
- **网络开销**: Milvus 可以批量调用嵌入 API，减少往返次数
- **并发处理**: Milvus 内部优化了并发调用
- **缓存机制**: 未来版本可能支持嵌入结果缓存

### Q2: 如何查看生成的向量？

**A**: 向量由 Milvus 自动管理，通常不需要查看。如需调试：
```python
results = client.query(
    collection_name=collection_name,
    filter="id > 0",
    output_fields=["document", "dense"],
    limit=1
)
print(results[0]["dense"])  # 查看向量
```

### Q3: 可以同时使用多个 Embedding Function 吗？

**A**: 可以！一个 Collection 可以有多个向量字段和多个 Embedding Function：
```python
# 配置两个 Embedding Function
schema.add_function(Function(
    name="openai_dense",
    input_field_names=["document"],
    output_field_names=["dense_vector"],
    params={"provider": "openai", "model_name": "text-embedding-3-small"}
))

schema.add_function(Function(
    name="cohere_sparse",
    input_field_names=["document"],
    output_field_names=["sparse_vector"],
    params={"provider": "cohere", "model_name": "embed-english-v3.0"}
))
```

### Q4: 如何处理嵌入 API 调用失败？

**A**: Milvus 会自动重试，如果最终失败会抛出异常：
```python
try:
    client.insert(collection_name, documents)
except Exception as e:
    print(f"插入失败: {e}")
    # 处理错误（如检查 API Key、网络连接等）
```

### Q5: 支持本地嵌入模型吗？

**A**: 支持！可以使用 Hugging Face TEI 或 Sentence Transformers：
```python
# 使用 Hugging Face TEI（需要自己部署 TEI 服务）
params={
    "provider": "tei",
    "model_name": "BAAI/bge-base-en-v1.5",
    "url": "http://localhost:8080"  # TEI 服务地址
}
```

---

## 最佳实践

### 1. 选择合适的提供商

**考虑因素**:
- **质量**: OpenAI > Cohere > Voyage
- **成本**: Voyage < Cohere < OpenAI
- **多语言**: Cohere > OpenAI
- **中文**: SiliconFlow > OpenAI

### 2. 配置管理

**推荐方式**:
```yaml
# 使用 milvus.yaml 集中管理凭证
credential:
  prod_openai:
    apikey: <PROD_KEY>
  dev_openai:
    apikey: <DEV_KEY>

function:
  textEmbedding:
    providers:
      openai:
        credential: prod_openai  # 生产环境
        # credential: dev_openai  # 开发环境
```

### 3. 错误处理

```python
def safe_insert(collection_name, documents):
    """安全插入，带重试机制"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client.insert(collection_name, documents)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"重试 {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"插入失败: {e}")
                return False
```

### 4. 监控与日志

```python
# 记录嵌入调用
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"插入 {len(documents)} 条文档")
start_time = time.time()
client.insert(collection_name, documents)
elapsed = time.time() - start_time
logger.info(f"插入完成，耗时 {elapsed:.2f}s")
```

---

## 总结

### 核心要点

1. **Embedding Functions** 是 Milvus 2.6 的核心特性，实现了 Data-in, Data-out 模式
2. **支持 9+ 提供商**，包括 OpenAI、Cohere、AWS Bedrock、Google Vertex AI 等
3. **配置简单**，通过 milvus.yaml 或环境变量一次性配置
4. **自动化**，插入和查询时自动调用嵌入 API
5. **一致性保证**，插入和查询使用同一模型

### 适用场景

✅ **推荐使用**:
- 标准 RAG 应用（文档问答、知识库检索）
- 使用主流嵌入提供商
- 追求开发效率和代码简洁
- 团队成员技术水平参差不齐

❌ **不推荐使用**:
- 需要自定义嵌入逻辑
- 使用非标准嵌入模型
- 需要嵌入前的特殊预处理
- 需要精细控制嵌入调用（如自定义缓存、重试策略）

---

## 下一步

- **实战代码**: [07_场景2_多提供商Embedding配置](./07_实战代码_场景2_多提供商Embedding配置.md) - 对比不同提供商
- **核心概念**: [03_核心概念_2_数据插入操作](./03_核心概念_2_数据插入操作.md) - 深入理解插入操作
- **学习辅助**: [06_反直觉点](./06_反直觉点.md) - 避开常见误区

---

**参考资料**:
- [Embedding Function Overview](https://milvus.io/docs/embedding-function-overview.md)
- [Data-in, Data-out in Milvus 2.6](https://milvus.io/blog/data-in-and-data-out-in-milvus-2-6.md)
- [Embedding Overview](https://milvus.io/docs/embeddings.md)

**版本**: v1.0 (基于 Milvus 2.6)
**最后更新**: 2026-02-22
