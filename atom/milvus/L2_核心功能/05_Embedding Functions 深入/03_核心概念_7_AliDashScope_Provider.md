# 核心概念:AliDashScope Provider

> Alibaba DashScope Provider是Milvus Embedding Functions中针对中文优化的提供商

---

## 提供商概述

**AliDashScope Provider**是Milvus支持的10个embedding提供商中专门针对中文场景优化的选择,特别适合中文RAG应用和阿里云生态集成。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 25(默认)/6(v3) | 批量大小较小 |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | text-embedding-v3支持自定义维度 |
| **输入类型** | ✅ 支持 | document vs query优化 |
| **延迟** | ~200ms | 中国区域延迟低 |
| **成本** | ¥0.0007/千tokens | 人民币计费 |
| **可靠性** | ⭐⭐⭐⭐ | 阿里云企业级SLA |

**来源**: `reference/source_architecture.md:240-265`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

dashscope_ef = Function(
    name="dashscope_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "dashscope",                   # 必需:提供商名称
        "model_name": "text-embedding-v3",         # 必需:模型名称
        "api_key": "sk-xxx"                        # 必需:API密钥
    }
)
```

### 可选参数

```python
params = {
    "provider": "dashscope",
    "model_name": "text-embedding-v3",
    "api_key": "sk-xxx",

    # 可选参数
    "dim": 1024,                                   # 输出维度(仅v3支持)
    "url": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
}
```

**来源**: `reference/source_architecture.md:244-265`

---

## 参数说明

### 1. model_name(模型名称)

**支持的模型**:

| 模型 | 默认维度 | MaxBatch | 成本(¥/千tokens) | 适用场景 |
|------|----------|----------|------------------|---------|
| **text-embedding-v3** | 1024 | 6 | 0.0007 | 通用场景(推荐) |
| **text-embedding-v2** | 1536 | 25 | 0.0007 | 旧版模型 |
| **text-embedding-v1** | 1536 | 25 | 0.0005 | 低成本场景 |

**选择建议**:
- 95%场景:使用`text-embedding-v3`(最新模型,中文优化)
- 大批量处理:使用`text-embedding-v2`(MaxBatch=25)
- 成本敏感:使用`text-embedding-v1`

**重要提示**:
- text-embedding-v3的MaxBatch仅为6,远低于其他模型的25
- 大批量处理时需要更多API调用

**来源**: `reference/source_architecture.md:250-253`

### 2. dim(维度控制)

**仅text-embedding-v3支持**

```python
# 原始维度
params = {"model_name": "text-embedding-v3", "dim": 1024}  # 默认

# 降低维度
params = {"model_name": "text-embedding-v3", "dim": 512}   # 节省50%存储

# 支持的维度值
# text-embedding-v3: 任意 ≤ 1024
```

**维度对比**:

| 维度 | 存储大小 | 计算速度 | 质量损失 | 推荐场景 |
|------|----------|----------|----------|---------|
| 1024 | 4KB | 1x | 0% | 标准配置(推荐) |
| 512 | 2KB | 2x | <3% | 性能优先 |
| 256 | 1KB | 4x | <5% | 极致性能 |

### 3. 输入类型优化(自动)

**DashScope根据模式自动优化**

```python
# Milvus自动设置text_type
# InsertMode → text_type = "document"
# SearchMode → text_type = "query"
```

**内部实现**:

```go
// 源码:ali_embedding_provider.go:138-141
if mode == models.InsertMode {
    textType = "document"  // 文档入库优化
} else {
    textType = "query"     // 查询检索优化
}
```

**性能提升**:
- 文档embedding:优化存储表示
- 查询embedding:优化检索匹配
- 中文准确率提升:15-20%

**来源**: `reference/source_architecture.md:259-262`

---

## 完整配置示例

### 示例1:中文RAG系统(推荐)

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# 定义DashScope Embedding Function
dashscope_ef = Function(
    name="dashscope_chinese_rag",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "dashscope",
        "model_name": "text-embedding-v3",       # 最新中文优化模型
        "api_key": "sk-xxx",
        "dim": 1024                              # 标准维度
    }
)

schema = CollectionSchema(fields=fields, functions=[dashscope_ef])
```

**适用场景**:
- 中文文档问答
- 中文知识库检索
- 中文客服系统

**来源**: `reference/source_architecture.md:240-265`

### 示例2:大批量处理配置

```python
# 使用text-embedding-v2获得更大批量
dashscope_batch_ef = Function(
    name="dashscope_batch",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "dashscope",
        "model_name": "text-embedding-v2",       # MaxBatch=25
        "api_key": "sk-xxx"
    }
)
```

**性能对比**:
- text-embedding-v3:MaxBatch=6,需要更多API调用
- text-embedding-v2:MaxBatch=25,批量处理效率高4倍

### 示例3:成本优化配置

```python
# 使用text-embedding-v1降低成本
dashscope_lite_ef = Function(
    name="dashscope_lite",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "dashscope",
        "model_name": "text-embedding-v1",       # 低成本模型
        "api_key": "sk-xxx"
    }
)
```

**成本对比**:
- text-embedding-v1:¥0.0005/千tokens(最便宜)
- text-embedding-v2/v3:¥0.0007/千tokens

---

## 最佳实践

### 1. 模型选择策略

```python
# 场景1:中文RAG系统(推荐)
model_name = "text-embedding-v3"
dim = 1024
# 理由:最新中文优化,质量最高

# 场景2:大批量处理
model_name = "text-embedding-v2"
# 理由:MaxBatch=25,效率高4倍

# 场景3:成本敏感
model_name = "text-embedding-v1"
# 理由:成本降低30%
```

### 2. 批量处理优化

```python
# DashScope批量大小限制
if model_name == "text-embedding-v3":
    batch_size = 6   # v3模型限制
else:
    batch_size = 25  # v1/v2模型

from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 批量插入
texts = ["文本1", "文本2", ..., f"文本{batch_size}"]
client.insert(
    collection_name="chinese_docs",
    data=[{"text": t} for t in texts]
)
```

**来源**: `reference/source_architecture.md:391-405`

### 3. 阿里云生态集成

```python
import os

# 从环境变量读取API密钥
api_key = os.getenv("DASHSCOPE_API_KEY")

# 阿里云生态集成配置
dashscope_ef = Function(
    name="dashscope_aliyun",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "dashscope",
        "model_name": "text-embedding-v3",
        "api_key": api_key,
        "url": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    }
)
```

---

## 常见陷阱与解决方案

### 陷阱1:批量大小超限

**错误示例**:

```python
# text-embedding-v3的MaxBatch=6
texts = ["文本1", "文本2", ..., "文本25"]  # ❌ 错误:超过MaxBatch=6

client.insert(collection_name="docs", data=[{"text": t} for t in texts])
# 错误:Batch size 25 exceeds maximum 6
```

**正确做法**:

```python
# 根据模型选择批量大小
if model_name == "text-embedding-v3":
    batch_size = 6
else:
    batch_size = 25

# 分批处理
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    client.insert(collection_name="docs", data=[{"text": t} for t in batch])
```

**来源**: `reference/source_architecture.md:467-479`

### 陷阱2:维度设置不匹配

**错误示例**:

```python
# text-embedding-v2不支持dim参数
params = {
    "model_name": "text-embedding-v2",
    "dim": 512,  # ❌ 错误:v2不支持dim参数
}
```

**正确做法**:

```python
# 仅v3支持dim参数
params = {
    "model_name": "text-embedding-v3",
    "dim": 512,  # ✅ 正确:v3支持dim
}
```

### 陷阱3:API密钥配置错误

**错误示例**:

```python
# 使用错误的环境变量名
import os
os.environ["ALIYUN_API_KEY"] = "sk-xxx"  # ❌ 错误:变量名不对
```

**正确做法**:

```python
# 使用正确的环境变量名
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"  # ✅ 正确

# 或直接在参数中传递
params = {
    "provider": "dashscope",
    "model_name": "text-embedding-v3",
    "api_key": "sk-xxx"  # ✅ 推荐:直接传递
}
```

**认证优先级**:
1. Function参数中的`api_key`(最高优先级)
2. Milvus YAML配置文件
3. 环境变量`DASHSCOPE_API_KEY`(最低优先级)

**来源**: `reference/source_architecture.md:346-358`

---

## 性能对比

### DashScope vs OpenAI vs VoyageAI

| 指标 | DashScope | OpenAI | VoyageAI |
|------|-----------|--------|----------|
| **MaxBatch** | 6/25 | 128 | 128 |
| **延迟(单次)** | ~200ms | ~300ms | ~150ms |
| **成本** | ¥0.0007/千tokens | $0.02/M tokens | $0.12/M tokens |
| **中文优化** | ✅ 专门优化 | ⚠️ 一般 | ⚠️ 一般 |
| **阿里云集成** | ✅ 原生支持 | ❌ 不支持 | ❌ 不支持 |

**选择建议**:
- **中文场景**:选择DashScope(专门优化)
- **大批量处理**:选择OpenAI或VoyageAI(MaxBatch=128)
- **阿里云生态**:选择DashScope(原生集成)
- **国际化场景**:选择OpenAI或VoyageAI

**来源**: `reference/source_architecture.md:389-405`

---

## 生产环境建议

### 1. 错误处理与重试

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
            if "rate limit" in error_msg or "quota" in error_msg:
                # DashScope速率限制错误,等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "batch size" in error_msg:
                # 批量大小错误,检查MaxBatch设置
                print("Batch size exceeded. Check MaxBatch limit")
                raise
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

### 2. 批量处理优化

```python
def get_max_batch(model_name):
    """根据模型获取最大批量大小"""
    if model_name == "text-embedding-v3":
        return 6
    else:
        return 25

def batch_insert(client, collection_name, texts, model_name):
    """批量插入优化"""
    batch_size = get_max_batch(model_name)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        insert_with_retry(
            client,
            collection_name,
            [{"text": t} for t in batch]
        )
        print(f"Inserted batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
```

### 3. 成本监控

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_embedding_cost(texts, model_name):
    """监控embedding成本"""
    # 估算token数量(中文:1字符≈1.5tokens)
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars * 1.5

    # 计算成本
    if model_name == "text-embedding-v1":
        cost_per_1k = 0.0005
    else:
        cost_per_1k = 0.0007

    estimated_cost = (estimated_tokens / 1000) * cost_per_1k

    logger.info(f"Embedding cost estimation:")
    logger.info(f"  Total characters: {total_chars}")
    logger.info(f"  Estimated tokens: {estimated_tokens:.0f}")
    logger.info(f"  Estimated cost: ¥{estimated_cost:.4f}")

    return estimated_cost
```

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:240-265`
   - DashScope Provider实现细节
   - 批量大小限制
   - 输入类型优化逻辑

2. **社区实践**: `reference/search_github.md:1-140`
   - 中文RAG应用案例
   - 阿里云生态集成经验
   - 性能优化建议

---

## 总结

**DashScope Provider核心优势**:
1. **中文优化**:专门针对中文场景优化,准确率提升15-20%
2. **阿里云集成**:原生支持阿里云生态,部署便捷
3. **成本优势**:人民币计费,中国区域成本更低
4. **低延迟**:中国区域延迟~200ms

**适用场景**:
- 中文RAG系统
- 阿里云生态项目
- 中国区域部署
- 成本敏感场景

**不适用场景**:
- 大批量处理(MaxBatch仅6,选择OpenAI或VoyageAI)
- 国际化场景(选择OpenAI或VoyageAI)
- 需要Int8量化(选择VoyageAI或Cohere)

**关键注意事项**:
- text-embedding-v3的MaxBatch仅为6,远低于其他提供商
- 仅v3支持维度控制
- 专门针对中文优化,英文场景建议使用OpenAI或VoyageAI

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
