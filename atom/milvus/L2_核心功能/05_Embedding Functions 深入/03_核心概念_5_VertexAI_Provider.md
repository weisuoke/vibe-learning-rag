# 核心概念:VertexAI Provider

> Google VertexAI Provider是Milvus Embedding Functions中支持任务类型定制的企业级提供商

---

## 提供商概述

**VertexAI Provider**是Milvus支持的10个embedding提供商中唯一支持任务类型(task_type)定制的选择,特别适合需要针对不同检索场景优化的RAG应用。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 128 | 最高批量大小(与OpenAI、VoyageAI并列第一) |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | text-embedding-004支持自定义维度 |
| **任务类型** | ✅ 支持 | DOC_RETRIEVAL、CODE_RETRIEVAL、STS |
| **延迟** | ~200ms | 美国中部环境单次API调用延迟 |
| **成本** | $0.025/M | 统一定价 |
| **可靠性** | ⭐⭐⭐⭐⭐ | Google Cloud企业级SLA |

**来源**: `reference/source_architecture.md:154-183`, `reference/context7_vertexai.md:1-185`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

vertexai_ef = Function(
    name="vertexai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "vertexai",                    # 必需:提供商名称
        "model_name": "text-embedding-004",        # 必需:模型名称
        "project_id": "your-gcp-project-id",       # 必需:GCP项目ID
        # 认证方式1:服务账号JSON(推荐)
        "credential": {
            "type": "service_account",
            "project_id": "your-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...",
            "client_email": "service-account@project.iam.gserviceaccount.com"
        }
    }
)
```

### 可选参数

```python
params = {
    "provider": "vertexai",
    "model_name": "text-embedding-004",
    "project_id": "your-gcp-project-id",

    # 可选参数
    "location": "us-central1",                     # GCP区域(默认:us-central1)
    "dim": 512,                                    # 输出维度(仅text-embedding-004支持)
    "task_type": "DOC_RETRIEVAL",                  # 任务类型(默认:DOC_RETRIEVAL)

    # 认证方式2:环境变量
    # 设置环境变量 GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
}
```

**来源**: `reference/source_architecture.md:158-180`

---

## 参数说明

### 1. model_name(模型名称)

**支持的模型**:

| 模型 | 默认维度 | 成本($/M tokens) | 适用场景 |
|------|----------|------------------|---------|
| **text-embedding-004** | 768 | 0.025 | 通用场景(推荐) |
| **text-multilingual-embedding-002** | 768 | 0.025 | 多语言场景 |
| **textembedding-gecko@003** | 768 | 0.025 | 旧版模型 |

**选择建议**:
- 95%场景:使用`text-embedding-004`(最新模型)
- 多语言需求:使用`text-multilingual-embedding-002`
- 避免使用:旧版gecko模型

**来源**: `reference/context7_vertexai.md:1-40`

### 2. task_type(任务类型)

**VertexAI独有特性**:根据不同任务类型优化embedding

```python
# 文档检索场景(默认)
params = {"task_type": "DOC_RETRIEVAL"}
# InsertMode → "RETRIEVAL_DOCUMENT"
# SearchMode → "RETRIEVAL_QUERY"

# 代码检索场景
params = {"task_type": "CODE_RETRIEVAL"}
# InsertMode → "RETRIEVAL_DOCUMENT"
# SearchMode → "CODE_RETRIEVAL_QUERY"

# 语义相似度场景
params = {"task_type": "STS"}
# 两种模式都使用 "SEMANTIC_SIMILARITY"
```

**任务类型对比**:

| Task Type | Insert模式 | Search模式 | 适用场景 |
|-----------|-----------|-----------|---------|
| **DOC_RETRIEVAL** | RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | 文档问答、知识库检索 |
| **CODE_RETRIEVAL** | RETRIEVAL_DOCUMENT | CODE_RETRIEVAL_QUERY | 代码搜索、API文档检索 |
| **STS** | SEMANTIC_SIMILARITY | SEMANTIC_SIMILARITY | 文本去重、相似度计算 |

**性能提升**:
- DOC_RETRIEVAL:检索准确率提升15-20%
- CODE_RETRIEVAL:代码检索准确率提升25-30%
- STS:相似度计算更稳定

**来源**: `reference/source_architecture.md:170-176`, `reference/context7_vertexai.md:166-185`

### 3. dim(维度控制)

**仅text-embedding-004支持**

```python
# 原始维度
params = {"model_name": "text-embedding-004", "dim": 768}  # 默认

# 降低维度(推荐)
params = {"model_name": "text-embedding-004", "dim": 256}  # 节省67%存储

# 支持的维度值
# text-embedding-004: 任意 ≤ 768
```

**维度对比**:

| 维度 | 存储大小 | 计算速度 | 质量损失 | 推荐场景 |
|------|----------|----------|----------|---------|
| 768 | 3KB | 1x | 0% | 质量优先 |
| 512 | 2KB | 1.5x | <3% | 平衡 |
| 256 | 1KB | 3x | <5% | 性能优先(推荐) |
| 128 | 0.5KB | 6x | ~10% | 极致性能 |

**来源**: `reference/context7_vertexai.md:1-40`

### 4. location(GCP区域)

**支持的区域**:

```python
# 美国区域(推荐)
params = {"location": "us-central1"}  # 默认,延迟最低

# 其他区域
params = {"location": "europe-west1"}  # 欧洲
params = {"location": "asia-southeast1"}  # 亚洲
```

**区域选择建议**:
- 北美用户:`us-central1`(延迟~200ms)
- 欧洲用户:`europe-west1`(延迟~150ms)
- 亚洲用户:`asia-southeast1`(延迟~180ms)

**来源**: `reference/source_architecture.md:177-180`

### 5. 认证配置

**方式1:服务账号JSON(推荐)**

```python
params = {
    "credential": {
        "type": "service_account",
        "project_id": "your-project-id",
        "private_key_id": "key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
        "client_email": "service-account@project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}
```

**方式2:环境变量**

```bash
# 设置环境变量
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

```python
# Python代码中无需传递credential参数
params = {
    "provider": "vertexai",
    "model_name": "text-embedding-004",
    "project_id": "your-gcp-project-id"
}
```

**认证优先级**:
1. Function参数中的`credential`(最高优先级)
2. Milvus YAML配置文件
3. 环境变量`GOOGLE_APPLICATION_CREDENTIALS`(最低优先级)

**来源**: `reference/source_architecture.md:346-358`

---

## 完整配置示例

### 示例1:文档检索RAG系统

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=256)
]

# 定义VertexAI Embedding Function
vertexai_ef = Function(
    name="vertexai_doc_retrieval",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "vertexai",
        "model_name": "text-embedding-004",
        "project_id": "my-rag-project",
        "location": "us-central1",
        "dim": 256,                          # 降低维度节省存储
        "task_type": "DOC_RETRIEVAL",        # 文档检索优化
        "credential": {
            "type": "service_account",
            "project_id": "my-rag-project",
            "private_key": "...",
            "client_email": "rag-service@my-rag-project.iam.gserviceaccount.com"
        }
    }
)

schema = CollectionSchema(fields=fields, functions=[vertexai_ef])
```

**来源**: `reference/source_architecture.md:154-183`

### 示例2:代码检索系统

```python
# 代码检索专用配置
vertexai_code_ef = Function(
    name="vertexai_code_retrieval",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["code_snippet"],
    output_field_names=["code_vector"],
    params={
        "provider": "vertexai",
        "model_name": "text-embedding-004",
        "project_id": "code-search-project",
        "task_type": "CODE_RETRIEVAL",       # 代码检索优化
        "dim": 512                           # 代码检索建议使用更高维度
    }
)
```

**来源**: `reference/context7_vertexai.md:166-185`

### 示例3:多语言场景

```python
# 多语言embedding配置
vertexai_multilingual_ef = Function(
    name="vertexai_multilingual",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "vertexai",
        "model_name": "text-multilingual-embedding-002",  # 多语言模型
        "project_id": "multilingual-project",
        "task_type": "DOC_RETRIEVAL",
        "dim": 768                           # 多语言建议使用完整维度
    }
)
```

---

## 最佳实践

### 1. 任务类型选择策略

```python
# 场景1:文档问答系统
task_type = "DOC_RETRIEVAL"
# 适用:技术文档、知识库、FAQ、新闻文章

# 场景2:代码搜索引擎
task_type = "CODE_RETRIEVAL"
# 适用:GitHub代码搜索、API文档检索、代码片段推荐

# 场景3:文本去重/聚类
task_type = "STS"
# 适用:重复内容检测、文本聚类、相似度排序
```

**来源**: `reference/context7_vertexai.md:166-185`

### 2. 维度优化策略

```python
# 小规模数据(<10万条)
dim = 768  # 使用完整维度,质量优先

# 中规模数据(10万-100万条)
dim = 512  # 平衡质量与性能

# 大规模数据(>100万条)
dim = 256  # 性能优先,质量损失<5%
```

### 3. 区域选择策略

```python
# 根据用户地理位置选择
if user_location == "North America":
    location = "us-central1"
elif user_location == "Europe":
    location = "europe-west1"
elif user_location == "Asia":
    location = "asia-southeast1"
```

### 4. 批量处理优化

```python
# VertexAI支持最大批量128
# 建议批量大小:64-128之间

from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 批量插入(推荐批量大小:128)
texts = ["text1", "text2", ..., "text128"]  # 128条
client.insert(
    collection_name="doc_collection",
    data=[{"text": t} for t in texts]
)
```

**来源**: `reference/source_architecture.md:391-405`

---

## 常见陷阱与解决方案

### 陷阱1:任务类型不匹配

**错误示例**:

```python
# 代码检索场景使用了DOC_RETRIEVAL
params = {
    "task_type": "DOC_RETRIEVAL",  # ❌ 错误:代码检索应使用CODE_RETRIEVAL
}
```

**正确做法**:

```python
# 代码检索场景使用CODE_RETRIEVAL
params = {
    "task_type": "CODE_RETRIEVAL",  # ✅ 正确
}
```

**影响**:任务类型不匹配会导致检索准确率下降20-30%

**来源**: `reference/context7_vertexai.md:166-185`

### 陷阱2:认证配置错误

**错误示例**:

```python
# 服务账号JSON格式错误
params = {
    "credential": {
        "type": "service_account",
        "project_id": "my-project",
        # ❌ 缺少private_key和client_email
    }
}
```

**正确做法**:

```python
# 完整的服务账号JSON
params = {
    "credential": {
        "type": "service_account",
        "project_id": "my-project",
        "private_key": "-----BEGIN PRIVATE KEY-----\n...",  # ✅ 必需
        "client_email": "service@project.iam.gserviceaccount.com"  # ✅ 必需
    }
}
```

**错误信息**:
```
Error: Invalid service account credentials
```

### 陷阱3:区域与项目不匹配

**错误示例**:

```python
# 项目在us-central1,但指定了europe-west1
params = {
    "project_id": "us-project",
    "location": "europe-west1",  # ❌ 错误:区域不匹配
}
```

**正确做法**:

```python
# 确保区域与项目配置一致
params = {
    "project_id": "us-project",
    "location": "us-central1",  # ✅ 正确
}
```

**影响**:区域不匹配会导致API调用失败或延迟增加

### 陷阱4:维度设置超出范围

**错误示例**:

```python
# text-embedding-004最大维度768
params = {
    "model_name": "text-embedding-004",
    "dim": 1024,  # ❌ 错误:超出最大维度
}
```

**正确做法**:

```python
# 维度不超过模型最大值
params = {
    "model_name": "text-embedding-004",
    "dim": 768,  # ✅ 正确:不超过最大维度
}
```

**错误信息**:
```
Error: Dimension 1024 exceeds maximum 768 for model text-embedding-004
```

**来源**: `reference/source_architecture.md:467-479`

---

## 性能对比

### VertexAI vs OpenAI vs VoyageAI

| 指标 | VertexAI | OpenAI | VoyageAI |
|------|----------|--------|----------|
| **MaxBatch** | 128 | 128 | 128 |
| **延迟(单次)** | ~200ms | ~300ms | ~150ms |
| **成本($/M)** | 0.025 | 0.02-0.13 | 0.12 |
| **任务类型** | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| **Int8输出** | ❌ 不支持 | ❌ 不支持 | ✅ 支持 |
| **企业SLA** | ✅ 支持 | ✅ 支持 | ⚠️ 有限 |

**选择建议**:
- **需要任务类型优化**:选择VertexAI(独有特性)
- **成本敏感**:选择OpenAI text-embedding-3-small
- **延迟敏感**:选择VoyageAI
- **企业级可靠性**:选择VertexAI或OpenAI

**来源**: `reference/source_architecture.md:389-405`, `reference/search_github.md:1-140`

---

## 生产环境建议

### 1. 服务账号权限配置

```bash
# 创建服务账号
gcloud iam service-accounts create milvus-embedding \
    --display-name="Milvus Embedding Service Account"

# 授予必要权限
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:milvus-embedding@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# 生成密钥文件
gcloud iam service-accounts keys create service-account.json \
    --iam-account=milvus-embedding@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 2. 错误处理与重试

```python
import time
from pymilvus import MilvusClient, MilvusException

def insert_with_retry(client, collection_name, data, max_retries=3):
    """带重试的插入操作"""
    for attempt in range(max_retries):
        try:
            result = client.insert(
                collection_name=collection_name,
                data=data
            )
            return result
        except MilvusException as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                # 配额或速率限制错误,等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

### 3. 成本优化

```python
# 策略1:使用较低维度
params = {"dim": 256}  # 相比768维度,成本降低67%

# 策略2:批量处理
batch_size = 128  # 使用最大批量,减少API调用次数

# 策略3:缓存常用embedding
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_embedding(text):
    # 缓存常用文本的embedding
    pass
```

### 4. 监控与告警

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 监控embedding生成
def monitor_embedding_generation(client, collection_name, data):
    start_time = time.time()
    try:
        result = client.insert(collection_name=collection_name, data=data)
        duration = time.time() - start_time

        # 记录性能指标
        logger.info(f"Embedding generated: {len(data)} texts in {duration:.2f}s")

        # 告警:延迟过高
        if duration > 5.0:
            logger.warning(f"High latency detected: {duration:.2f}s")

        return result
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
```

**来源**: `reference/search_github.md:99-110`

---

## 社区最佳实践

### 1. RAG系统集成

```python
from pymilvus import MilvusClient, Function, FunctionType

# 初始化客户端
client = MilvusClient(uri="http://localhost:19530")

# 创建collection with VertexAI
client.create_collection(
    collection_name="rag_docs",
    dimension=256,
    function=Function(
        name="vertexai_rag",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=["text"],
        output_field_names=["vector"],
        params={
            "provider": "vertexai",
            "model_name": "text-embedding-004",
            "project_id": "rag-project",
            "task_type": "DOC_RETRIEVAL",
            "dim": 256
        }
    )
)

# 插入文档
docs = [
    {"text": "Milvus is a vector database..."},
    {"text": "RAG combines retrieval and generation..."}
]
client.insert(collection_name="rag_docs", data=docs)

# 检索
results = client.search(
    collection_name="rag_docs",
    data=["What is Milvus?"],
    limit=5
)
```

**来源**: `reference/search_github.md:49-58`

### 2. 代码搜索引擎

```python
# 代码检索专用配置
code_search_ef = Function(
    name="vertexai_code",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["code"],
    output_field_names=["code_vector"],
    params={
        "provider": "vertexai",
        "model_name": "text-embedding-004",
        "project_id": "code-search",
        "task_type": "CODE_RETRIEVAL",  # 代码检索优化
        "dim": 512
    }
)

# 插入代码片段
code_snippets = [
    {"code": "def hello_world():\n    print('Hello, World!')"},
    {"code": "class Vector:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"}
]
client.insert(collection_name="code_repo", data=code_snippets)
```

**来源**: `reference/context7_vertexai.md:166-185`

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:154-183`
   - VertexAI Provider实现细节
   - 任务类型映射逻辑
   - 批量处理机制

2. **官方文档**: `reference/context7_vertexai.md:1-185`
   - VertexAI Embeddings API参数
   - 任务类型说明
   - 最佳实践建议

3. **社区实践**: `reference/search_github.md:1-140`
   - 生产环境部署案例
   - 性能优化经验
   - 常见问题解决方案

---

## 总结

**VertexAI Provider核心优势**:
1. **任务类型优化**:DOC_RETRIEVAL、CODE_RETRIEVAL、STS三种模式
2. **企业级可靠性**:Google Cloud SLA保障
3. **高批量支持**:MaxBatch=128,与OpenAI并列第一
4. **灵活维度控制**:支持256-768维度自定义

**适用场景**:
- 需要针对不同检索场景优化的RAG系统
- 代码搜索引擎
- 企业级应用(需要SLA保障)
- 已使用Google Cloud生态的项目

**不适用场景**:
- 需要Int8量化输出(选择VoyageAI或Cohere)
- 极致延迟要求(选择VoyageAI)
- 成本极度敏感(选择OpenAI text-embedding-3-small)

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
