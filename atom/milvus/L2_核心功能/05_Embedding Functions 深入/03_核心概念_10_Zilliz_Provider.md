# 核心概念:Zilliz Provider

> Zilliz Cloud Pipelines Provider是Milvus Embedding Functions中唯一的云原生集成提供商

---

## 提供商概述

**Zilliz Provider**是Milvus支持的10个embedding提供商中唯一与Zilliz Cloud深度集成的选择,提供无缝的云原生embedding体验。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 64 | 中等批量大小 |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | 由模型决定 |
| **输入类型** | ✅ 支持 | document vs query优化 |
| **延迟** | ~100ms | 云原生低延迟 |
| **成本** | 按用量计费 | Zilliz Cloud定价 |
| **可靠性** | ⭐⭐⭐⭐⭐ | 云原生高可用 |

**来源**: `reference/source_architecture.md:214-239`, `reference/search_github.md:1-140`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

zilliz_ef = Function(
    name="zilliz_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "zilliz",                      # 必需:提供商名称
        "model_deployment_id": "pipeline-xxx"      # 必需:Pipeline部署ID
    }
)
```

### 可选参数

```python
params = {
    "provider": "zilliz",
    "model_deployment_id": "pipeline-xxx",

    # 可选参数:模型特定参数
    "model_params": {
        "temperature": 0.7,
        "max_tokens": 512
    }
}
```

**来源**: `reference/source_architecture.md:218-236`

---

## 参数说明

### 1. model_deployment_id(Pipeline部署ID)

**Zilliz Cloud Pipelines部署标识**

```python
# 从Zilliz Cloud控制台获取Pipeline ID
params = {
    "model_deployment_id": "pipeline-abc123def456"
}
```

**获取Pipeline ID**:
1. 登录Zilliz Cloud控制台
2. 进入Pipelines页面
3. 创建或选择Embedding Pipeline
4. 复制Pipeline部署ID

**来源**: `reference/source_architecture.md:219-220`

### 2. 输入类型优化(自动)

**Zilliz根据模式自动优化**

```python
# Milvus自动设置input_type
# InsertMode → input_type = "document"
# SearchMode → input_type = "query"
```

**内部实现**:

```go
// 源码:zilliz_embedding_provider.go:93-96
if mode == models.InsertMode {
    inputType = "document"  // 文档入库优化
} else {
    inputType = "query"     // 查询检索优化
}
```

**性能提升**:
- 文档embedding:优化存储表示
- 查询embedding:优化检索匹配
- 准确率提升:10-15%

**来源**: `reference/source_architecture.md:228-231`

### 3. model_params(模型参数)

**灵活的模型参数传递**

```python
# 传递模型特定参数
params = {
    "model_deployment_id": "pipeline-xxx",
    "model_params": {
        "temperature": 0.7,      # 温度参数
        "max_tokens": 512,       # 最大token数
        "top_p": 0.9            # Top-p采样
    }
}
```

**参数说明**:
- `model_params`是一个灵活的字典
- 参数会直接传递给Pipeline
- 具体支持的参数取决于Pipeline配置

**来源**: `reference/source_architecture.md:221-227`

### 4. 认证管理(自动)

**Zilliz Provider使用内部认证**

```python
# 无需手动配置API密钥
# 认证信息由Milvus自动管理
params = {
    "provider": "zilliz",
    "model_deployment_id": "pipeline-xxx"
    # 无需api_key参数
}
```

**认证机制**:
- 使用`extraInfo.ClusterID`和`extraInfo.DBName`
- 由Milvus内部管理
- 无需用户手动配置

**来源**: `reference/source_architecture.md:232-234`

---

## 完整配置示例

### 示例1:基础Zilliz Cloud集成(推荐)

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# 定义Zilliz Embedding Function
zilliz_ef = Function(
    name="zilliz_cloud",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "zilliz",
        "model_deployment_id": "pipeline-abc123def456"  # 从控制台获取
    }
)

schema = CollectionSchema(fields=fields, functions=[zilliz_ef])
```

**适用场景**:
- Zilliz Cloud用户
- 云原生部署
- 无需管理API密钥

**来源**: `reference/source_architecture.md:214-239`

### 示例2:自定义模型参数

```python
# 传递自定义模型参数
zilliz_custom_ef = Function(
    name="zilliz_custom",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "zilliz",
        "model_deployment_id": "pipeline-abc123def456",
        "model_params": {
            "temperature": 0.5,      # 降低随机性
            "max_tokens": 1024       # 增加token限制
        }
    }
)
```

### 示例3:多Pipeline配置

```python
# 使用不同的Pipeline处理不同类型的数据
zilliz_doc_ef = Function(
    name="zilliz_doc",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["doc_text"],
    output_field_names=["doc_vector"],
    params={
        "provider": "zilliz",
        "model_deployment_id": "pipeline-doc-xxx"  # 文档专用Pipeline
    }
)

zilliz_code_ef = Function(
    name="zilliz_code",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["code_text"],
    output_field_names=["code_vector"],
    params={
        "provider": "zilliz",
        "model_deployment_id": "pipeline-code-xxx"  # 代码专用Pipeline
    }
)
```

---

## 最佳实践

### 1. Pipeline选择策略

```python
# 场景1:通用文档检索
model_deployment_id = "pipeline-general-xxx"
# 理由:通用Pipeline适合大多数场景

# 场景2:代码检索
model_deployment_id = "pipeline-code-xxx"
# 理由:代码专用Pipeline优化代码语义

# 场景3:多语言场景
model_deployment_id = "pipeline-multilingual-xxx"
# 理由:多语言Pipeline支持多种语言
```

### 2. 批量处理优化

```python
# Zilliz批量大小:64
batch_size = 64

from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 批量插入
texts = ["文本1", "文本2", ..., "文本64"]  # 64条
client.insert(
    collection_name="docs",
    data=[{"text": t} for t in texts]
)
```

**来源**: `reference/source_architecture.md:391-405`

### 3. Zilliz Cloud集成优化

```python
# 使用Zilliz Cloud连接
from pymilvus import MilvusClient

# 连接到Zilliz Cloud
client = MilvusClient(
    uri="https://your-cluster.zillizcloud.com:19530",
    token="your-zilliz-cloud-token"
)

# 创建collection with Zilliz Pipeline
client.create_collection(
    collection_name="zilliz_docs",
    dimension=1024,
    function=Function(
        name="zilliz_ef",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=["text"],
        output_field_names=["vector"],
        params={
            "provider": "zilliz",
            "model_deployment_id": "pipeline-xxx"
        }
    )
)
```

---

## 常见陷阱与解决方案

### 陷阱1:Pipeline ID错误

**错误示例**:

```python
# 使用错误的Pipeline ID
params = {
    "model_deployment_id": "pipeline-wrong-id"  # ❌ 错误:ID不存在
}

client.insert(collection_name="docs", data=[{"text": "test"}])
# 错误:Pipeline not found
```

**正确做法**:

```bash
# 从Zilliz Cloud控制台获取正确的Pipeline ID
# 1. 登录Zilliz Cloud
# 2. 进入Pipelines页面
# 3. 复制正确的Pipeline ID
```

```python
# 使用正确的Pipeline ID
params = {
    "model_deployment_id": "pipeline-abc123def456"  # ✅ 正确
}
```

**来源**: `reference/source_architecture.md:467-479`

### 陷阱2:批量大小超限

**错误示例**:

```python
# Zilliz的MaxBatch=64
texts = ["文本1", "文本2", ..., "文本128"]  # ❌ 错误:超过MaxBatch=64

client.insert(collection_name="docs", data=[{"text": t} for t in texts])
# 错误:Batch size 128 exceeds maximum 64
```

**正确做法**:

```python
# 分批处理
batch_size = 64
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    client.insert(collection_name="docs", data=[{"text": t} for t in batch])
```

### 陷阱3:模型参数配置错误

**错误示例**:

```python
# 传递Pipeline不支持的参数
params = {
    "model_deployment_id": "pipeline-xxx",
    "model_params": {
        "unsupported_param": "value"  # ❌ 错误:Pipeline不支持此参数
    }
}
```

**正确做法**:

```python
# 查看Pipeline文档,使用支持的参数
params = {
    "model_deployment_id": "pipeline-xxx",
    "model_params": {
        "temperature": 0.7,  # ✅ 正确:Pipeline支持的参数
        "max_tokens": 512
    }
}
```

### 陷阱4:连接配置错误

**错误示例**:

```python
# 使用本地Milvus连接Zilliz Cloud Pipeline
client = MilvusClient(uri="http://localhost:19530")  # ❌ 错误:本地Milvus

# 尝试使用Zilliz Pipeline
params = {"provider": "zilliz", "model_deployment_id": "pipeline-xxx"}
# 错误:Zilliz Pipeline only works with Zilliz Cloud
```

**正确做法**:

```python
# 使用Zilliz Cloud连接
client = MilvusClient(
    uri="https://your-cluster.zillizcloud.com:19530",  # ✅ 正确
    token="your-zilliz-cloud-token"
)
```

---

## 性能对比

### Zilliz vs OpenAI vs VoyageAI

| 指标 | Zilliz | OpenAI | VoyageAI |
|------|--------|--------|----------|
| **MaxBatch** | 64 | 128 | 128 |
| **延迟(单次)** | ~100ms | ~300ms | ~150ms |
| **成本** | 按用量 | $0.02/M | $0.12/M |
| **云原生集成** | ✅ 原生 | ❌ 第三方 | ❌ 第三方 |
| **认证管理** | ✅ 自动 | ⚠️ 手动 | ⚠️ 手动 |
| **高可用** | ✅ 云原生 | ✅ 支持 | ⚠️ 有限 |

**选择建议**:
- **Zilliz Cloud用户**:选择Zilliz(原生集成)
- **云原生部署**:选择Zilliz(无缝集成)
- **大批量处理**:选择OpenAI或VoyageAI(MaxBatch=128)
- **第三方服务**:选择OpenAI或VoyageAI

**来源**: `reference/source_architecture.md:389-405`, `reference/search_github.md:1-140`

---

## 生产环境建议

### 1. Pipeline管理

```python
# 使用环境变量管理Pipeline ID
import os

pipeline_id = os.getenv("ZILLIZ_PIPELINE_ID")

zilliz_ef = Function(
    name="zilliz_prod",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "zilliz",
        "model_deployment_id": pipeline_id
    }
)
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
            error_msg = str(e).lower()
            if "pipeline" in error_msg:
                # Pipeline错误,检查Pipeline状态
                print("Pipeline error. Check Pipeline status in Zilliz Cloud")
                raise
            elif "rate limit" in error_msg or "quota" in error_msg:
                # 速率限制错误,等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

### 3. 监控与告警

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_zilliz_pipeline(client, collection_name, data):
    """监控Zilliz Pipeline性能"""
    start_time = time.time()
    batch_size = len(data)

    try:
        result = insert_with_retry(client, collection_name, data)
        duration = time.time() - start_time

        # 计算性能指标
        throughput = batch_size / duration
        avg_latency = duration / batch_size * 1000  # ms

        logger.info(f"Zilliz Pipeline performance:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} texts/s")
        logger.info(f"  Avg latency: {avg_latency:.2f}ms")

        # 告警:性能异常
        if avg_latency > 200:
            logger.warning(f"High latency detected: {avg_latency:.2f}ms")

        return result
    except Exception as e:
        logger.error(f"Zilliz Pipeline failed: {e}")
        raise
```

---

## 社区最佳实践

### 1. Zilliz Cloud集成

```python
from pymilvus import MilvusClient, Function, FunctionType

# 连接到Zilliz Cloud
client = MilvusClient(
    uri="https://your-cluster.zillizcloud.com:19530",
    token="your-zilliz-cloud-token"
)

# 创建collection with Zilliz Pipeline
client.create_collection(
    collection_name="zilliz_rag",
    dimension=1024,
    function=Function(
        name="zilliz_ef",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=["text"],
        output_field_names=["vector"],
        params={
            "provider": "zilliz",
            "model_deployment_id": "pipeline-xxx"
        }
    )
)

# 插入数据
docs = [
    {"text": "Milvus is a vector database..."},
    {"text": "RAG combines retrieval and generation..."}
]
client.insert(collection_name="zilliz_rag", data=docs)

# 检索
results = client.search(
    collection_name="zilliz_rag",
    data=["What is Milvus?"],
    limit=5
)
```

**来源**: `reference/search_github.md:1-140`

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:214-239`
   - Zilliz Provider实现细节
   - 认证管理机制
   - 批量大小配置

2. **社区实践**: `reference/search_github.md:1-140`
   - Zilliz Cloud集成案例
   - 生产环境部署经验
   - 性能优化建议

---

## 总结

**Zilliz Provider核心优势**:
1. **云原生集成**:与Zilliz Cloud无缝集成
2. **自动认证**:无需手动管理API密钥
3. **低延迟**:~100ms,云原生优化
4. **高可用**:云原生架构保障

**适用场景**:
- Zilliz Cloud用户
- 云原生部署
- 需要高可用保障的应用
- 无需管理API密钥的场景

**不适用场景**:
- 本地Milvus部署(仅支持Zilliz Cloud)
- 大批量处理(MaxBatch仅64,选择OpenAI或VoyageAI)
- 需要自托管(选择TEI)

**关键注意事项**:
- 仅支持Zilliz Cloud,不支持本地Milvus
- Pipeline ID需要从Zilliz Cloud控制台获取
- 认证由Milvus自动管理,无需手动配置

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
