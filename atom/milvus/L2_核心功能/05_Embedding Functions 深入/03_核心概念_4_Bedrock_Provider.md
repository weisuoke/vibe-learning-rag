# 核心概念：AWS Bedrock Provider

> AWS Bedrock Provider是Milvus支持的AWS生态embedding提供商，适合已使用AWS服务的企业用户

---

## 提供商概述

**AWS Bedrock Provider**是Milvus支持的AWS托管embedding服务，通过Amazon Bedrock提供多种embedding模型。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 1 | ⚠️ 不支持批处理（最低） |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | 部分模型支持 |
| **延迟** | ~500ms | 较高延迟 |
| **成本** | 按AWS定价 | 按使用量计费 |
| **AWS集成** | ⭐⭐⭐⭐⭐ | 深度集成AWS生态 |

### 核心限制

**⚠️ 重要警告：Bedrock不支持批处理**

```python
# Bedrock MaxBatch = 1
# 这意味着每条数据需要单独调用API

# 插入10000条数据
collection.insert([{"text": f"doc{i}"} for i in range(10000)])

# Bedrock处理：
# - 10000次API调用（每条数据一次）
# - 总延迟：10000 × 500ms = 5000秒 = 83分钟
# - 对比OpenAI（128条/批）：79次API调用 = 23.7秒
# - 性能差距：211倍！
```

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

bedrock_ef = Function(
    name="bedrock_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "bedrock",                           # 必需：提供商名称
        "model_name": "amazon.titan-embed-text-v1",      # 必需：模型名称
        "region": "us-east-1",                           # 必需：AWS区域
        "aws_access_key_id": "AKIA...",                  # 必需：AWS访问密钥
        "aws_secret_access_key": "xxx"                   # 必需：AWS密钥
    }
)
```

### 可选参数

```python
params = {
    "provider": "bedrock",
    "model_name": "amazon.titan-embed-text-v1",
    "region": "us-east-1",
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "xxx",

    # 可选参数
    "dim": 512,                                          # 输出维度（部分模型支持）
    "normalize": True                                    # 是否归一化（默认True）
}
```

### 参数说明

#### 1. model_name（模型名称）

**支持的Bedrock模型**：

| 模型 | 默认维度 | 成本 | 适用场景 |
|------|----------|------|---------|
| **amazon.titan-embed-text-v1** | 1536 | 按量 | 通用文本embedding |
| **amazon.titan-embed-text-v2** | 1024 | 按量 | 新版Titan模型 |
| **cohere.embed-english-v3** | 1024 | 按量 | 英文场景 |
| **cohere.embed-multilingual-v3** | 1024 | 按量 | 多语言场景 |

**选择建议**：
- AWS原生：使用`amazon.titan-embed-text-v1`或`v2`
- 多语言：使用`cohere.embed-multilingual-v3`
- 注意：所有模型都不支持批处理（MaxBatch=1）

#### 2. region（AWS区域）

**Bedrock可用区域**：

```python
# 美国区域
params = {"region": "us-east-1"}      # 美国东部（弗吉尼亚）
params = {"region": "us-west-2"}      # 美国西部（俄勒冈）

# 欧洲区域
params = {"region": "eu-west-1"}      # 欧洲（爱尔兰）
params = {"region": "eu-central-1"}   # 欧洲（法兰克福）

# 亚太区域
params = {"region": "ap-southeast-1"} # 亚太（新加坡）
params = {"region": "ap-northeast-1"} # 亚太（东京）
```

**区域选择建议**：
- 选择距离用户最近的区域
- 确认该区域支持Bedrock服务
- 查看区域可用性：https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html

#### 3. AWS凭证配置

**三种配置方式**（优先级从高到低）：

```python
# 方式1：函数参数（最高优先级）
params = {
    "aws_access_key_id": "AKIA...",
    "aws_secret_access_key": "xxx"
}

# 方式2：环境变量（中等优先级）
import os
os.environ["AWS_ACCESS_KEY_ID"] = "AKIA..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"
params = {}  # Milvus自动读取

# 方式3：AWS配置文件（最低优先级）
# ~/.aws/credentials
# [default]
# aws_access_key_id = AKIA...
# aws_secret_access_key = xxx
```

**安全建议**：
- 生产环境：使用IAM角色（EC2/ECS/Lambda）
- 开发环境：使用AWS配置文件
- 避免：硬编码在代码中

#### 4. normalize（归一化）

**Bedrock特有参数**：

```python
# 归一化（默认）
params = {"normalize": True}
# 输出向量的L2范数 = 1
# 适合：余弦相似度检索

# 不归一化
params = {"normalize": False}
# 输出向量保持原始值
# 适合：L2距离检索
```

---

## Bedrock特有限制

### 限制1：不支持批处理（MaxBatch=1）

**这是Bedrock最大的限制**

```python
# 性能对比：插入10000条数据

# OpenAI（MaxBatch=128）
# - API调用次数：79次
# - 总延迟：79 × 300ms = 23.7秒

# Bedrock（MaxBatch=1）
# - API调用次数：10000次
# - 总延迟：10000 × 500ms = 5000秒 = 83分钟
# - 性能差距：211倍！
```

**影响**：
- 大规模数据插入极慢
- API调用成本高
- 不适合实时场景

**缓解策略**：
```python
# 策略1：减少数据量
# 只插入必要的数据

# 策略2：离线批处理
# 在非高峰时段处理

# 策略3：使用其他提供商
# 对于大规模场景，考虑OpenAI/VoyageAI
```

### 限制2：较高延迟

**Bedrock延迟较高**：

```python
# 延迟对比（单次API调用）
latency = {
    "voyageai": 200,   # ms
    "cohere": 250,     # ms
    "openai": 300,     # ms
    "vertexai": 400,   # ms
    "bedrock": 500,    # ms（最慢）
}

# 加上不支持批处理，总延迟更高
```

### 限制3：AWS账户要求

**使用Bedrock需要**：
- AWS账户
- 启用Bedrock服务
- 配置IAM权限
- 可能需要申请模型访问权限

```python
# 检查Bedrock访问权限
import boto3

client = boto3.client('bedrock', region_name='us-east-1')

try:
    # 列出可用模型
    response = client.list_foundation_models()
    print("Bedrock可用")
except Exception as e:
    print(f"Bedrock不可用：{e}")
```

---

## 完整配置示例

### 示例1：基础配置

```python
from pymilvus import MilvusClient, Function, DataType, FunctionType
import os

# 设置AWS凭证
os.environ["AWS_ACCESS_KEY_ID"] = "AKIA..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"

# 创建客户端
client = MilvusClient("http://localhost:19530")

# 定义Bedrock embedding函数
bedrock_ef = Function(
    name="bedrock_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "bedrock",
        "model_name": "amazon.titan-embed-text-v1",
        "region": "us-east-1"
    }
)

# 创建schema
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=1000)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1536)
schema.add_function(bedrock_ef)

# 创建collection
client.create_collection("bedrock_demo", schema=schema)

# ⚠️ 注意：插入大量数据会很慢
# 建议：小批量插入或使用其他提供商
client.insert("bedrock_demo", [
    {"text": "AWS Bedrock is a managed AI service"},
    {"text": "Milvus supports AWS Bedrock integration"}
])

# 查询
results = client.search(
    "bedrock_demo",
    data=["What is AWS Bedrock?"],
    anns_field="vector",
    limit=5
)
```

### 示例2：使用IAM角色（推荐）

```python
# 在EC2/ECS/Lambda上运行时，使用IAM角色
# 无需配置AWS凭证

bedrock_ef = Function(
    name="bedrock_ef",
    params={
        "provider": "bedrock",
        "model_name": "amazon.titan-embed-text-v1",
        "region": "us-east-1"
        # 不需要aws_access_key_id和aws_secret_access_key
        # Milvus自动使用IAM角色
    }
)

# IAM角色需要的权限
# {
#   "Version": "2012-10-17",
#   "Statement": [
#     {
#       "Effect": "Allow",
#       "Action": [
#         "bedrock:InvokeModel"
#       ],
#       "Resource": "*"
#     }
#   ]
# }
```

### 示例3：多区域配置

```python
# 配置多个区域的Bedrock资源
bedrock_us_ef = Function(
    name="bedrock_us_ef",
    params={
        "provider": "bedrock",
        "model_name": "amazon.titan-embed-text-v1",
        "region": "us-east-1"
    }
)

bedrock_eu_ef = Function(
    name="bedrock_eu_ef",
    params={
        "provider": "bedrock",
        "model_name": "amazon.titan-embed-text-v1",
        "region": "eu-west-1"
    }
)

# 根据用户地理位置选择合适的区域
def get_embedding_function(user_location):
    if user_location in ["US", "CA", "MX"]:
        return bedrock_us_ef
    elif user_location in ["EU", "UK", "FR"]:
        return bedrock_eu_ef
    else:
        return bedrock_us_ef  # 默认

# 使用
ef = get_embedding_function("US")
schema.add_function(ef)
```

---

## 常见问题与解决方案

### 问题1：权限不足

**错误信息**：
```
MilvusException: AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```

**解决方案**：
```python
# 1. 检查IAM权限
# 确保IAM用户/角色有bedrock:InvokeModel权限

# 2. 添加IAM策略
# {
#   "Effect": "Allow",
#   "Action": ["bedrock:InvokeModel"],
#   "Resource": "*"
# }

# 3. 检查模型访问权限
# AWS Console → Bedrock → Model access
# 确保已启用所需的模型
```

### 问题2：区域不支持

**错误信息**：
```
MilvusException: Bedrock is not available in this region
```

**解决方案**：
```python
# 1. 检查Bedrock可用区域
# https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html

# 2. 使用支持的区域
supported_regions = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-southeast-1",
    "ap-northeast-1"
]

params = {"region": "us-east-1"}  # 使用支持的区域
```

### 问题3：性能太慢

**问题描述**：插入大量数据时速度极慢

**解决方案**：
```python
# 问题根源：Bedrock不支持批处理（MaxBatch=1）

# 方案1：减少数据量
# 只插入必要的数据，过滤冗余数据

# 方案2：分批处理 + 进度显示
from tqdm import tqdm

texts = [f"document {i}" for i in range(10000)]
batch_size = 100

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    collection.insert([{"text": t} for t in batch])
    # 每100条显示一次进度

# 方案3：切换到其他提供商
# 对于大规模场景，强烈建议使用OpenAI/VoyageAI
params = {"provider": "openai"}  # 性能提升211倍
```

### 问题4：成本过高

**问题描述**：API调用次数太多导致成本高

**解决方案**：
```python
# 问题根源：Bedrock不支持批处理，API调用次数多

# 方案1：缓存embedding结果
embedding_cache = {}

def get_embedding_with_cache(text):
    if text in embedding_cache:
        return embedding_cache[text]

    result = collection.insert([{"text": text}])
    embedding_cache[text] = result
    return result

# 方案2：去重
texts = list(set(texts))  # 去除重复文本

# 方案3：使用其他提供商
# Bedrock的API调用次数是OpenAI的126倍
# 即使OpenAI单价更高，总成本可能更低
```

---

## 最佳实践

### 1. 何时使用Bedrock？

**适合使用Bedrock的场景**：
- ✅ 已深度使用AWS生态
- ✅ 需要AWS合规性和安全性
- ✅ 数据量较小（<1000条）
- ✅ 对延迟不敏感
- ✅ 需要AWS统一账单管理

**不适合使用Bedrock的场景**：
- ❌ 大规模数据（>10000条）
- ❌ 实时场景
- ❌ 对性能要求高
- ❌ 成本敏感

### 2. 性能优化

```python
# 优化1：使用IAM角色（减少认证开销）
# 在EC2/ECS/Lambda上运行，使用IAM角色

# 优化2：选择就近区域
params = {"region": "us-east-1"}  # 选择距离用户最近的区域

# 优化3：并行处理（有限效果）
from concurrent.futures import ThreadPoolExecutor

def insert_batch(batch):
    collection.insert([{"text": t} for t in batch])

with ThreadPoolExecutor(max_workers=4) as executor:
    batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
    executor.map(insert_batch, batches)
# 注意：由于MaxBatch=1，并行效果有限
```

### 3. 成本控制

```python
# 策略1：减少数据量
# 只插入必要的数据

# 策略2：缓存embedding
# 避免重复计算相同文本的embedding

# 策略3：监控使用情况
# AWS Console → Bedrock → Usage
# 设置预算警报

# 策略4：考虑其他提供商
# 对于大规模场景，OpenAI/VoyageAI可能更经济
```

### 4. AWS集成

```python
# 集成1：使用AWS Secrets Manager存储凭证
import boto3

secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
secret = secrets_client.get_secret_value(SecretId='milvus/bedrock')
api_key = secret['SecretString']

# 集成2：使用AWS CloudWatch监控
# 记录API调用次数、延迟等指标

# 集成3：使用AWS VPC Endpoint
# 通过私有网络访问Bedrock，提升安全性
```

---

## 与其他提供商对比

### 性能对比

| 维度 | Bedrock | OpenAI | VoyageAI |
|------|---------|--------|----------|
| **MaxBatch** | 1 ⚠️ | 128 | 128 |
| **延迟** | 500ms | 300ms | 200ms |
| **10K数据插入** | 83分钟 | 24秒 | 16秒 |
| **性能差距** | 1x | 211x | 316x |
| **AWS集成** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |

### 选择建议

**选择Bedrock的场景**：
- ✅ 深度使用AWS生态
- ✅ 数据量小（<1000条）
- ✅ 需要AWS合规性

**选择其他提供商的场景**：
- OpenAI：大规模场景、追求性能
- VoyageAI：最高性能要求
- Cohere：多语言场景

---

## 参考资源

### 官方文档
- AWS Bedrock: https://docs.aws.amazon.com/bedrock/
- Bedrock Embeddings: https://docs.aws.amazon.com/bedrock/latest/userguide/embeddings.html
- Milvus Embedding Functions: https://milvus.io/docs/embeddings.md

### 源代码
- Milvus Bedrock Provider: `sourcecode/milvus/internal/util/function/embedding/bedrock_embedding_provider.go`

---

**版本信息**：
- Milvus版本：2.6+
- AWS Bedrock API版本：2023-09-30
- 文档版本：v1.0
- 最后更新：2026-02-24

---

**重要提醒**：Bedrock不支持批处理（MaxBatch=1），不适合大规模场景。对于大规模RAG应用，强烈建议使用OpenAI或VoyageAI。
