# 核心概念：Azure OpenAI Provider

> Azure OpenAI Provider是OpenAI在Azure云平台上的企业级实现，提供与OpenAI相同的embedding能力，但具有更强的企业级特性

---

## 提供商概述

**Azure OpenAI Provider**是Milvus支持的OpenAI在Azure云平台上的变体，适合已经使用Azure生态的企业用户。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 128 | 与OpenAI相同的最高批量大小 |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | text-embedding-3系列支持自定义维度 |
| **延迟** | ~300-400ms | 取决于Azure区域 |
| **成本** | 按Azure定价 | 通常与OpenAI相近 |
| **可靠性** | ⭐⭐⭐⭐⭐ | 企业级SLA保障 |

### 与OpenAI Provider的区别

| 维度 | OpenAI | Azure OpenAI |
|------|--------|--------------|
| **认证方式** | API Key | API Key + Resource Name |
| **端点URL** | api.openai.com | {resource}.openai.azure.com |
| **环境变量** | OPENAI_API_KEY | AZURE_OPENAI_API_KEY |
| **部署模式** | 公有云 | Azure私有部署 |
| **企业特性** | 基础 | 高级（VNet、Private Link等） |
| **合规性** | 标准 | 企业级（HIPAA、SOC2等） |
| **成本控制** | 按量付费 | Azure订阅管理 |

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

azure_openai_ef = Function(
    name="azure_openai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "azure_openai",                    # 必需：提供商名称
        "model_name": "text-embedding-3-small",        # 必需：模型名称
        "api_key": "your-azure-api-key",               # 必需：Azure API密钥
        "resource_name": "your-resource-name"          # 必需：Azure资源名称
    }
)
```

### 可选参数

```python
params = {
    "provider": "azure_openai",
    "model_name": "text-embedding-3-small",
    "api_key": "your-azure-api-key",
    "resource_name": "your-resource-name",

    # 可选参数
    "dim": 512,                                        # 输出维度
    "user": "user_123",                                # 用户标识
    "api_version": "2023-05-15",                       # API版本
    "deployment_name": "my-embedding-deployment"       # 部署名称（如果与model_name不同）
}
```

### 参数说明

#### 1. resource_name（Azure资源名称）

**这是Azure OpenAI特有的参数**

```python
# Azure OpenAI端点格式
# https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/embeddings?api-version={api_version}

params = {
    "resource_name": "my-openai-resource",  # 你的Azure OpenAI资源名称
}

# 完整URL示例
# https://my-openai-resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15
```

**如何获取resource_name**：
1. 登录Azure Portal
2. 进入你的Azure OpenAI资源
3. 在"Keys and Endpoint"页面查看"Endpoint"
4. 从URL中提取resource_name：`https://{resource_name}.openai.azure.com`

#### 2. api_key（Azure API密钥）

**Azure OpenAI使用不同的环境变量**

```python
# 方式1：函数参数（最高优先级）
params = {"api_key": "your-azure-api-key"}

# 方式2：环境变量（推荐）
import os
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
params = {}  # Milvus自动读取

# 方式3：YAML配置
# 在milvus.yaml中配置
```

**如何获取API密钥**：
1. 登录Azure Portal
2. 进入你的Azure OpenAI资源
3. 在"Keys and Endpoint"页面查看"KEY 1"或"KEY 2"

#### 3. deployment_name（部署名称）

**Azure OpenAI使用部署概念**

```python
# 如果部署名称与模型名称相同
params = {
    "model_name": "text-embedding-3-small",
    # deployment_name默认使用model_name
}

# 如果部署名称与模型名称不同
params = {
    "model_name": "text-embedding-3-small",
    "deployment_name": "my-custom-embedding-deployment"
}
```

**部署名称说明**：
- Azure OpenAI要求先创建"部署"才能使用模型
- 部署名称可以与模型名称相同或不同
- 如果不指定deployment_name，Milvus使用model_name作为部署名称

#### 4. api_version（API版本）

**Azure OpenAI API版本控制**

```python
# 默认版本
params = {"api_version": "2023-05-15"}  # 默认值

# 使用最新版本
params = {"api_version": "2024-02-01"}  # 最新稳定版

# 使用预览版本
params = {"api_version": "2024-03-01-preview"}  # 预览版
```

**版本选择建议**：
- 生产环境：使用稳定版本（如2023-05-15）
- 测试环境：可以使用最新版本或预览版
- 查看最新版本：https://learn.microsoft.com/en-us/azure/ai-services/openai/reference

---

## Azure OpenAI特有功能

### 1. 虚拟网络（VNet）集成

**场景**：企业内网环境，需要通过VNet访问Azure OpenAI

```python
# 配置VNet端点
params = {
    "provider": "azure_openai",
    "resource_name": "my-openai-resource",
    "api_key": "your-api-key",
    # VNet端点会自动使用，无需额外配置
}

# Azure Portal配置：
# 1. 进入Azure OpenAI资源
# 2. 选择"Networking"
# 3. 配置"Private endpoint connections"
```

### 2. 托管身份（Managed Identity）

**场景**：使用Azure托管身份进行认证，无需管理API密钥

```python
# 使用托管身份（需要Azure环境支持）
from azure.identity import DefaultAzureCredential

# 获取token
credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

# 配置Milvus（使用token而非api_key）
params = {
    "provider": "azure_openai",
    "resource_name": "my-openai-resource",
    "api_key": token.token,  # 使用token
}

# 注意：token会过期，需要定期刷新
```

### 3. 内容过滤

**Azure OpenAI提供内置的内容过滤功能**

```python
# Azure OpenAI会自动过滤不当内容
# 如果输入包含不当内容，API会返回错误

try:
    collection.insert([{"text": "inappropriate content"}])
except Exception as e:
    if "content_filter" in str(e):
        print("内容被过滤")
        # 处理内容过滤错误
```

### 4. 数据驻留

**场景**：确保数据存储在特定地理区域

```python
# 选择特定区域的Azure OpenAI资源
params = {
    "resource_name": "my-openai-eastus",  # 美国东部
    # 或
    "resource_name": "my-openai-westeurope",  # 西欧
}

# 数据会存储在对应区域
```

---

## 完整配置示例

### 示例1：基础配置

```python
from pymilvus import MilvusClient, Function, DataType, FunctionType
import os

# 设置环境变量
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"

# 创建客户端
client = MilvusClient("http://localhost:19530")

# 定义Azure OpenAI embedding函数
azure_openai_ef = Function(
    name="azure_openai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "azure_openai",
        "model_name": "text-embedding-3-small",
        "resource_name": "my-openai-resource",
        "dim": 512
    }
)

# 创建schema
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("text", DataType.VARCHAR, max_length=1000)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=512)
schema.add_function(azure_openai_ef)

# 创建collection
client.create_collection("azure_openai_demo", schema=schema)

# 插入数据
client.insert("azure_openai_demo", [
    {"text": "Azure OpenAI is Microsoft's cloud-based AI service"},
    {"text": "Milvus supports Azure OpenAI for enterprise deployments"}
])

# 查询
results = client.search(
    "azure_openai_demo",
    data=["What is Azure OpenAI?"],
    anns_field="vector",
    limit=5
)

print(results)
```

### 示例2：多区域配置

```python
# 配置多个Azure OpenAI资源（不同区域）
azure_eastus_ef = Function(
    name="azure_eastus_ef",
    params={
        "provider": "azure_openai",
        "resource_name": "my-openai-eastus",
        "model_name": "text-embedding-3-small"
    }
)

azure_westeurope_ef = Function(
    name="azure_westeurope_ef",
    params={
        "provider": "azure_openai",
        "resource_name": "my-openai-westeurope",
        "model_name": "text-embedding-3-small"
    }
)

# 根据用户地理位置选择合适的资源
def get_embedding_function(user_location):
    if user_location in ["US", "CA", "MX"]:
        return azure_eastus_ef
    elif user_location in ["EU", "UK", "FR"]:
        return azure_westeurope_ef
    else:
        return azure_eastus_ef  # 默认

# 使用
ef = get_embedding_function("US")
schema.add_function(ef)
```

### 示例3：使用自定义部署名称

```python
# 场景：Azure部署名称与模型名称不同
azure_openai_ef = Function(
    name="azure_openai_ef",
    params={
        "provider": "azure_openai",
        "model_name": "text-embedding-3-small",
        "deployment_name": "prod-embedding-v1",  # 自定义部署名称
        "resource_name": "my-openai-resource"
    }
)
```

---

## 常见问题与解决方案

### 问题1：资源名称错误

**错误信息**：
```
MilvusException: resource not found
```

**解决方案**：
```python
# 1. 检查resource_name是否正确
import requests

resource_name = "my-openai-resource"
url = f"https://{resource_name}.openai.azure.com"

try:
    response = requests.get(url, timeout=5)
    print(f"资源存在：{resource_name}")
except Exception as e:
    print(f"资源不存在或无法访问：{e}")

# 2. 确认resource_name与Azure Portal中的一致
# Azure Portal → Azure OpenAI → Keys and Endpoint → Endpoint
```

### 问题2：部署不存在

**错误信息**：
```
MilvusException: deployment not found
```

**解决方案**：
```python
# 1. 检查部署是否存在
# Azure Portal → Azure OpenAI → Model deployments

# 2. 确认deployment_name正确
params = {
    "model_name": "text-embedding-3-small",
    "deployment_name": "text-embedding-3-small",  # 必须与Azure中的部署名称一致
}

# 3. 如果部署名称与模型名称不同，必须指定deployment_name
params = {
    "model_name": "text-embedding-3-small",
    "deployment_name": "my-custom-deployment",  # 使用实际的部署名称
}
```

### 问题3：API版本不兼容

**错误信息**：
```
MilvusException: api version not supported
```

**解决方案**：
```python
# 1. 使用稳定的API版本
params = {"api_version": "2023-05-15"}  # 推荐

# 2. 查看支持的API版本
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference

# 3. 避免使用过旧或过新的版本
# 过旧：可能缺少新特性
# 过新：可能不稳定
```

### 问题4：配额限制

**错误信息**：
```
RateLimitError: Requests to the Embeddings_Create Operation under Azure OpenAI API have exceeded call rate limit
```

**解决方案**：
```python
# 1. 检查Azure OpenAI配额
# Azure Portal → Azure OpenAI → Quotas

# 2. 增加配额（如果需要）
# Azure Portal → Azure OpenAI → Quotas → Request quota increase

# 3. 实现重试机制
import time
from openai import RateLimitError

def insert_with_retry(collection, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            collection.insert(data)
            break
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"配额限制，等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise

# 4. 减小批量大小
# 在milvus.yaml中配置
proxy:
  embedding:
    batchFactor: 0.5  # 减小批量大小
```

---

## 最佳实践

### 1. 企业级安全配置

```python
# 使用Azure Key Vault存储API密钥
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# 从Key Vault获取API密钥
credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://my-keyvault.vault.azure.net/", credential=credential)
api_key = client.get_secret("azure-openai-api-key").value

# 配置Milvus
params = {
    "provider": "azure_openai",
    "api_key": api_key,
    "resource_name": "my-openai-resource"
}
```

### 2. 多区域高可用

```python
# 配置主备Azure OpenAI资源
primary_ef = Function(
    name="primary_ef",
    params={
        "provider": "azure_openai",
        "resource_name": "my-openai-eastus",
        "model_name": "text-embedding-3-small"
    }
)

backup_ef = Function(
    name="backup_ef",
    params={
        "provider": "azure_openai",
        "resource_name": "my-openai-westus",
        "model_name": "text-embedding-3-small"
    }
)

# 实现故障转移
def insert_with_failover(collection, data):
    try:
        # 尝试使用主资源
        collection.insert(data)
    except Exception as e:
        print(f"主资源失败：{e}，切换到备份资源")
        # 切换到备份资源
        # 需要重新创建collection或更新embedding函数
        pass
```

### 3. 成本优化

```python
# 1. 使用较小的模型
params = {"model_name": "text-embedding-3-small"}  # 而非3-large

# 2. 降低维度
params = {"dim": 512}  # 而非1536

# 3. 监控使用情况
# Azure Portal → Azure OpenAI → Metrics
# 监控：Total Tokens, Total Calls, Latency

# 4. 设置预算警报
# Azure Portal → Cost Management → Budgets
```

### 4. 合规性配置

```python
# 1. 启用审计日志
# Azure Portal → Azure OpenAI → Diagnostic settings

# 2. 配置数据驻留
# 选择符合合规要求的区域
params = {"resource_name": "my-openai-germanywestcentral"}  # 德国区域

# 3. 启用Private Link
# Azure Portal → Azure OpenAI → Networking → Private endpoint connections

# 4. 配置内容过滤
# Azure Portal → Azure OpenAI → Content filters
```

---

## 与OpenAI Provider对比

### 何时选择Azure OpenAI？

**选择Azure OpenAI的场景**：
- ✅ 已经使用Azure生态
- ✅ 需要企业级SLA保障
- ✅ 需要VNet/Private Link等企业特性
- ✅ 需要符合特定合规要求（HIPAA、SOC2等）
- ✅ 需要数据驻留在特定地理区域
- ✅ 需要Azure订阅统一管理成本

**选择OpenAI的场景**：
- ✅ 不使用Azure生态
- ✅ 追求最简单的配置
- ✅ 需要最新的模型和特性
- ✅ 个人或小团队项目

### 性能对比

| 维度 | OpenAI | Azure OpenAI |
|------|--------|--------------|
| **延迟** | ~300ms | ~300-400ms（取决于区域） |
| **可靠性** | 99.9% | 99.95%（企业级SLA） |
| **批量大小** | 128 | 128 |
| **配额管理** | 按账户 | 按Azure订阅 |
| **成本** | 按量付费 | 按量付费（Azure定价） |

---

## 参考资源

### 官方文档
- Azure OpenAI Service: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- Azure OpenAI Embeddings: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings
- Milvus Embedding Functions: https://milvus.io/docs/embeddings.md

### 源代码
- Milvus OpenAI Provider: `sourcecode/milvus/internal/util/function/embedding/openai_embedding_provider.go`

---

**版本信息**：
- Milvus版本：2.6+
- Azure OpenAI API版本：2023-05-15+
- 文档版本：v1.0
- 最后更新：2026-02-24
