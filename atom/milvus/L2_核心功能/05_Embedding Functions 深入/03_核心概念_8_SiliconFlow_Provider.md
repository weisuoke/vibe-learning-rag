# 核心概念:SiliconFlow Provider

> SiliconFlow Provider是Milvus Embedding Functions中针对中国市场的高性价比提供商

---

## 提供商概述

**SiliconFlow Provider**是Milvus支持的10个embedding提供商中专注于中国市场的选择,提供多种开源模型支持和灵活的定价策略。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 32 | 中等批量大小 |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ✅ 支持 | Qwen3系列支持自定义维度 |
| **输入类型** | ❌ 不支持 | 无document/query区分 |
| **延迟** | ~180ms | 中国区域延迟低 |
| **成本** | 灵活定价 | 按模型不同 |
| **可靠性** | ⭐⭐⭐ | 新兴提供商 |

**来源**: `reference/source_architecture.md:266-290`, `reference/fetch_siliconflow_docs.md:1-120`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

siliconflow_ef = Function(
    name="siliconflow_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "siliconflow",                 # 必需:提供商名称
        "model_name": "BAAI/bge-large-zh-v1.5",    # 必需:模型名称
        "api_key": "sk-xxx"                        # 必需:API密钥
    }
)
```

### 可选参数

```python
params = {
    "provider": "siliconflow",
    "model_name": "BAAI/bge-large-zh-v1.5",
    "api_key": "sk-xxx",

    # 可选参数
    "dim": 1024,                                   # 输出维度(仅Qwen3系列支持)
    "url": "https://api.siliconflow.cn/v1/embeddings"  # 自定义端点
}
```

**来源**: `reference/source_architecture.md:270-289`, `reference/fetch_siliconflow_docs.md:30-88`

---

## 参数说明

### 1. model_name(模型名称)

**支持的模型**:

| 模型 | 默认维度 | MaxTokens | 适用场景 |
|------|----------|-----------|---------|
| **BAAI/bge-large-zh-v1.5** | 1024 | 512 | 中文通用(推荐) |
| **BAAI/bge-large-en-v1.5** | 1024 | 512 | 英文通用 |
| **netease-youdao/bce-embedding-base_v1** | 768 | 512 | 中文基础 |
| **BAAI/bge-m3** | 1024 | 8192 | 多语言长文本 |
| **Pro/BAAI/bge-m3** | 1024 | 8192 | 多语言长文本(Pro版) |
| **Qwen/Qwen3-Embedding-*** | 可变 | 32768 | 超长文本 |

**选择建议**:
- 中文场景:使用`BAAI/bge-large-zh-v1.5`(推荐)
- 英文场景:使用`BAAI/bge-large-en-v1.5`
- 长文本:使用`BAAI/bge-m3`(8192 tokens)
- 超长文本:使用`Qwen/Qwen3-Embedding-*`(32768 tokens)

**来源**: `reference/fetch_siliconflow_docs.md:75-88`

### 2. dim(维度控制)

**仅Qwen3系列支持**

```python
# Qwen3系列支持自定义维度
params = {
    "model_name": "Qwen/Qwen3-Embedding-1024",
    "dim": 512  # 支持离散维度值
}

# 其他模型不支持dim参数
params = {
    "model_name": "BAAI/bge-large-zh-v1.5"
    # 不支持dim参数,使用默认1024维度
}
```

**Qwen3维度说明**:
- 支持离散维度值(具体值取决于模型)
- 降低维度可节省存储和计算成本
- 维度降低会有轻微质量损失

**来源**: `reference/fetch_siliconflow_docs.md:84-88`

### 3. Token限制

**不同模型的Token限制差异很大**:

```python
# 短文本模型(512 tokens)
models_512 = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-large-en-v1.5",
    "netease-youdao/bce-embedding-base_v1"
]

# 长文本模型(8192 tokens)
models_8192 = [
    "BAAI/bge-m3",
    "Pro/BAAI/bge-m3"
]

# 超长文本模型(32768 tokens)
models_32768 = [
    "Qwen/Qwen3-Embedding-*"
]
```

**选择策略**:
- 短文档(<500字):使用512 token模型
- 中等文档(500-8000字):使用8192 token模型
- 长文档(>8000字):使用32768 token模型

**来源**: `reference/fetch_siliconflow_docs.md:75-88`

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

# 定义SiliconFlow Embedding Function
siliconflow_ef = Function(
    name="siliconflow_chinese",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "siliconflow",
        "model_name": "BAAI/bge-large-zh-v1.5",   # 中文优化模型
        "api_key": "sk-xxx"
    }
)

schema = CollectionSchema(fields=fields, functions=[siliconflow_ef])
```

**适用场景**:
- 中文文档问答
- 中文知识库检索
- 中文客服系统

**来源**: `reference/source_architecture.md:266-290`

### 示例2:长文本处理

```python
# 使用bge-m3处理长文本
siliconflow_long_ef = Function(
    name="siliconflow_long",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "siliconflow",
        "model_name": "BAAI/bge-m3",               # 支持8192 tokens
        "api_key": "sk-xxx"
    }
)
```

**性能对比**:
- bge-large-zh-v1.5:MaxTokens=512,适合短文本
- bge-m3:MaxTokens=8192,适合长文本(16倍提升)

### 示例3:超长文本处理

```python
# 使用Qwen3处理超长文本
siliconflow_ultra_long_ef = Function(
    name="siliconflow_ultra_long",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "siliconflow",
        "model_name": "Qwen/Qwen3-Embedding-1024", # 支持32768 tokens
        "api_key": "sk-xxx",
        "dim": 1024                                # 自定义维度
    }
)
```

**适用场景**:
- 学术论文检索
- 技术文档问答
- 长篇小说分析

---

## 最佳实践

### 1. 模型选择策略

```python
# 场景1:中文短文本RAG(推荐)
model_name = "BAAI/bge-large-zh-v1.5"
max_tokens = 512
# 理由:中文优化,性价比高

# 场景2:多语言长文本
model_name = "BAAI/bge-m3"
max_tokens = 8192
# 理由:支持多语言,长文本处理

# 场景3:超长文本处理
model_name = "Qwen/Qwen3-Embedding-1024"
max_tokens = 32768
# 理由:超长文本支持
```

### 2. 批量处理优化

```python
# SiliconFlow批量大小:32
batch_size = 32

from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 批量插入
texts = ["文本1", "文本2", ..., "文本32"]  # 32条
client.insert(
    collection_name="chinese_docs",
    data=[{"text": t} for t in texts]
)
```

**来源**: `reference/source_architecture.md:391-405`

### 3. Token限制处理

```python
def select_model_by_text_length(text):
    """根据文本长度选择合适的模型"""
    # 估算token数量(中文:1字符≈1.5tokens)
    estimated_tokens = len(text) * 1.5

    if estimated_tokens <= 512:
        return "BAAI/bge-large-zh-v1.5"
    elif estimated_tokens <= 8192:
        return "BAAI/bge-m3"
    else:
        return "Qwen/Qwen3-Embedding-1024"

# 使用示例
text = "..." * 1000  # 长文本
model_name = select_model_by_text_length(text)
```

---

## 常见陷阱与解决方案

### 陷阱1:Token限制超限

**错误示例**:

```python
# 使用bge-large-zh-v1.5处理长文本
params = {
    "model_name": "BAAI/bge-large-zh-v1.5"  # MaxTokens=512
}

# 插入超长文本
long_text = "..." * 1000  # 超过512 tokens
client.insert(collection_name="docs", data=[{"text": long_text}])
# 错误:Text exceeds maximum token limit 512
```

**正确做法**:

```python
# 根据文本长度选择合适的模型
if len(text) * 1.5 > 512:
    model_name = "BAAI/bge-m3"  # 使用长文本模型
else:
    model_name = "BAAI/bge-large-zh-v1.5"
```

**来源**: `reference/source_architecture.md:467-479`

### 陷阱2:维度参数使用错误

**错误示例**:

```python
# bge-large-zh-v1.5不支持dim参数
params = {
    "model_name": "BAAI/bge-large-zh-v1.5",
    "dim": 512,  # ❌ 错误:该模型不支持dim参数
}
```

**正确做法**:

```python
# 仅Qwen3系列支持dim参数
params = {
    "model_name": "Qwen/Qwen3-Embedding-1024",
    "dim": 512,  # ✅ 正确:Qwen3支持dim
}
```

### 陷阱3:批量大小超限

**错误示例**:

```python
# SiliconFlow的MaxBatch=32
texts = ["文本1", "文本2", ..., "文本64"]  # ❌ 错误:超过MaxBatch=32

client.insert(collection_name="docs", data=[{"text": t} for t in texts])
# 错误:Batch size 64 exceeds maximum 32
```

**正确做法**:

```python
# 分批处理
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    client.insert(collection_name="docs", data=[{"text": t} for t in batch])
```

### 陷阱4:API密钥配置错误

**错误示例**:

```python
# 使用错误的环境变量名
import os
os.environ["SILICON_API_KEY"] = "sk-xxx"  # ❌ 错误:变量名不对
```

**正确做法**:

```python
# 使用正确的环境变量名
import os
os.environ["SILICONFLOW_API_KEY"] = "sk-xxx"  # ✅ 正确

# 或直接在参数中传递
params = {
    "provider": "siliconflow",
    "model_name": "BAAI/bge-large-zh-v1.5",
    "api_key": "sk-xxx"  # ✅ 推荐:直接传递
}
```

**认证优先级**:
1. Function参数中的`api_key`(最高优先级)
2. Milvus YAML配置文件
3. 环境变量`SILICONFLOW_API_KEY`(最低优先级)

**来源**: `reference/source_architecture.md:346-358`

---

## 性能对比

### SiliconFlow vs DashScope vs OpenAI

| 指标 | SiliconFlow | DashScope | OpenAI |
|------|-------------|-----------|--------|
| **MaxBatch** | 32 | 6/25 | 128 |
| **延迟(单次)** | ~180ms | ~200ms | ~300ms |
| **MaxTokens** | 512-32768 | 512-1536 | 8192 |
| **中文优化** | ✅ 专门优化 | ✅ 专门优化 | ⚠️ 一般 |
| **开源模型** | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |

**选择建议**:
- **中文短文本**:选择SiliconFlow或DashScope
- **长文本处理**:选择SiliconFlow(支持32768 tokens)
- **大批量处理**:选择OpenAI(MaxBatch=128)
- **开源模型**:选择SiliconFlow(唯一支持)

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
                # SiliconFlow速率限制错误,等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "token limit" in error_msg:
                # Token限制错误,切换到长文本模型
                print("Token limit exceeded. Switch to long-text model")
                raise
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

### 2. 动态模型选择

```python
def get_optimal_model(text):
    """根据文本长度动态选择最优模型"""
    estimated_tokens = len(text) * 1.5

    if estimated_tokens <= 512:
        return {
            "model_name": "BAAI/bge-large-zh-v1.5",
            "max_tokens": 512,
            "cost_level": "low"
        }
    elif estimated_tokens <= 8192:
        return {
            "model_name": "BAAI/bge-m3",
            "max_tokens": 8192,
            "cost_level": "medium"
        }
    else:
        return {
            "model_name": "Qwen/Qwen3-Embedding-1024",
            "max_tokens": 32768,
            "cost_level": "high"
        }

# 使用示例
text = "..." * 1000
model_config = get_optimal_model(text)
print(f"Selected model: {model_config['model_name']}")
```

### 3. 批量处理优化

```python
def batch_insert_optimized(client, collection_name, texts):
    """优化的批量插入"""
    batch_size = 32  # SiliconFlow MaxBatch

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # 动态选择模型
        max_text_length = max(len(t) for t in batch)
        model_config = get_optimal_model("x" * max_text_length)

        # 插入数据
        insert_with_retry(
            client,
            collection_name,
            [{"text": t} for t in batch]
        )

        print(f"Inserted batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        print(f"Model used: {model_config['model_name']}")
```

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:266-290`
   - SiliconFlow Provider实现细节
   - 批量大小限制
   - 配置参数说明

2. **官方文档**: `reference/fetch_siliconflow_docs.md:1-120`
   - SiliconFlow Embeddings API参数
   - 模型列表与Token限制
   - 请求响应格式

---

## 总结

**SiliconFlow Provider核心优势**:
1. **开源模型支持**:唯一支持开源模型的提供商
2. **灵活Token限制**:512-32768 tokens,覆盖全场景
3. **中文优化**:专门针对中文场景优化
4. **成本优势**:灵活定价,适合中国市场

**适用场景**:
- 中文RAG系统
- 长文本处理(学术论文、技术文档)
- 开源模型部署
- 成本敏感场景

**不适用场景**:
- 大批量处理(MaxBatch仅32,选择OpenAI或VoyageAI)
- 需要Int8量化(选择VoyageAI或Cohere)
- 需要任务类型优化(选择VertexAI)

**关键注意事项**:
- 不同模型的Token限制差异很大(512-32768)
- 仅Qwen3系列支持维度控制
- MaxBatch=32,低于OpenAI/VoyageAI的128

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
