# 核心概念:VoyageAI Provider

> VoyageAI Provider是Milvus Embedding Functions中社区最推荐的提供商,以速度快、质量高著称

---

## 提供商概述

**VoyageAI Provider**是Milvus支持的10个embedding提供商中社区最受欢迎的选择,特别适合需要高性能和高质量的RAG应用场景。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 128 | 最高批量大小(与OpenAI、VertexAI并列第一) |
| **输出类型** | Float32、Int8 | 支持量化输出(仅2家之一) |
| **维度控制** | ✅ 支持 | voyage-3系列支持256/512/1024/2048 |
| **输入类型** | ✅ 支持 | document vs query优化 |
| **延迟** | ~150ms | 业界最快(比OpenAI快50%) |
| **成本** | $0.12/M | 中等定价 |
| **可靠性** | ⭐⭐⭐⭐ | 社区高度认可 |

**来源**: `reference/source_architecture.md:127-151`, `reference/context7_voyageai.md:1-145`, `reference/search_github.md:88-90`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

voyageai_ef = Function(
    name="voyageai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",                    # 必需:提供商名称
        "model_name": "voyage-3-large",            # 必需:模型名称
        "api_key": "pa-xxx"                        # 必需:API密钥
    }
)
```

### 可选参数

```python
params = {
    "provider": "voyageai",
    "model_name": "voyage-3-large",
    "api_key": "pa-xxx",

    # 可选参数
    "dim": 1024,                                   # 输出维度(256/512/1024/2048)
    "truncate": False,                             # 是否截断超长文本
    "url": "https://api.voyageai.com/v1/embeddings"  # 自定义端点
}
```

**来源**: `reference/source_architecture.md:133-150`

---

## 参数说明

### 1. model_name(模型名称)

**支持的模型**:

| 模型 | 默认维度 | 成本($/M tokens) | 适用场景 |
|------|----------|------------------|---------|
| **voyage-3-large** | 1024 | 0.12 | 通用场景(推荐) |
| **voyage-3** | 1024 | 0.08 | 平衡性价比 |
| **voyage-3-lite** | 512 | 0.04 | 成本敏感场景 |
| **voyage-code-3** | 1024 | 0.12 | 代码检索专用 |
| **voyage-4-large** | 2048 | 0.15 | 最新最强模型 |

**选择建议**:
- 95%场景:使用`voyage-3-large`(社区最推荐)
- 成本敏感:使用`voyage-3-lite`
- 代码检索:使用`voyage-code-3`
- 最高质量:使用`voyage-4-large`

**来源**: `reference/context7_voyageai.md:140-145`, `reference/search_github.md:88-90`

### 2. dim(维度控制)

**voyage-3系列支持灵活维度**

```python
# 支持的维度值:256、512、1024、2048
params = {"model_name": "voyage-3-large", "dim": 1024}  # 默认

# 降低维度(推荐)
params = {"model_name": "voyage-3-large", "dim": 512}   # 节省50%存储

# 提升维度
params = {"model_name": "voyage-3-large", "dim": 2048}  # 最高质量
```

**维度对比**:

| 维度 | 存储大小 | 计算速度 | 质量损失 | 推荐场景 |
|------|----------|----------|----------|---------|
| 2048 | 8KB | 0.5x | 0% | 极致质量 |
| 1024 | 4KB | 1x | 0% | 标准配置(推荐) |
| 512 | 2KB | 2x | <2% | 性能优先 |
| 256 | 1KB | 4x | <5% | 极致性能 |

**来源**: `reference/context7_voyageai.md:27-28`, `reference/context7_voyageai.md:140-145`

### 3. 输入类型优化(自动)

**VoyageAI根据模式自动优化**

```python
# Milvus自动设置input_type
# InsertMode → input_type = "document"
# SearchMode → input_type = "query"
```

**内部实现**:

```go
// 源码:voyageai_embedding_provider.go:148-151
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

**来源**: `reference/source_architecture.md:148-151`, `reference/context7_voyageai.md:22-26`

### 4. Int8量化输出(独有特性)

**VoyageAI和Cohere是仅有的2家支持Int8输出的提供商**

```python
# Float32输出(默认)
fields = [
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# Int8量化输出(节省75%存储)
fields = [
    FieldSchema(name="vector", dtype=DataType.INT8_VECTOR, dim=1024)
]
```

**Int8量化对比**:

| 输出类型 | 存储大小 | 计算速度 | 质量损失 | 推荐场景 |
|----------|----------|----------|----------|---------|
| Float32 | 4KB | 1x | 0% | 质量优先 |
| Int8 | 1KB | 4x | <3% | 大规模部署(推荐) |

**量化原理**:

```python
# VoyageAI内部量化算法
# float32 → int8: scale + clip + cast
embd_int8 = ((embd_float / 0.16).clip(-1, 1) * 127).astype(np.int8)
```

**来源**: `reference/source_architecture.md:143-145`, `reference/context7_voyageai.md:75-103`

### 5. truncate(截断控制)

```python
# 不截断(默认,超长文本会报错)
params = {"truncate": False}

# 自动截断(推荐生产环境)
params = {"truncate": True}
```

**截断行为**:
- `truncate=False`:超过模型最大token限制会抛出异常
- `truncate=True`:自动截断到模型最大token限制

**来源**: `reference/source_architecture.md:134-135`, `reference/context7_voyageai.md:147-148`

---

## 完整配置示例

### 示例1:高性能RAG系统(推荐)

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema(使用Int8量化)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.INT8_VECTOR, dim=1024)  # Int8量化
]

# 定义VoyageAI Embedding Function
voyageai_ef = Function(
    name="voyageai_high_perf",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-3-large",
        "api_key": "pa-xxx",
        "dim": 1024,                         # 标准维度
        "truncate": True                     # 自动截断
    }
)

schema = CollectionSchema(fields=fields, functions=[voyageai_ef])
```

**性能指标**:
- 存储节省:75%(Int8 vs Float32)
- 检索速度:4x提升
- 质量损失:<3%

**来源**: `reference/source_architecture.md:127-151`, `reference/context7_voyageai.md:75-103`

### 示例2:成本优化配置

```python
# 使用voyage-3-lite + 低维度
voyageai_lite_ef = Function(
    name="voyageai_lite",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-3-lite",       # 低成本模型
        "api_key": "pa-xxx",
        "dim": 512,                          # 低维度
        "truncate": True
    }
)
```

**成本对比**:
- voyage-3-lite:$0.04/M tokens(比voyage-3-large便宜67%)
- 512维度:存储节省50%
- 总成本:降低80%+

### 示例3:代码检索专用

```python
# 代码检索专用配置
voyageai_code_ef = Function(
    name="voyageai_code",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["code_snippet"],
    output_field_names=["code_vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-code-3",       # 代码专用模型
        "api_key": "pa-xxx",
        "dim": 1024,
        "truncate": True
    }
)
```

**代码检索优势**:
- 专门针对代码语义优化
- 支持多种编程语言
- 准确率比通用模型高20-30%

**来源**: `reference/context7_voyageai.md:140-145`

### 示例4:最高质量配置

```python
# 使用voyage-4-large + 最高维度
voyageai_premium_ef = Function(
    name="voyageai_premium",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-4-large",      # 最新最强模型
        "api_key": "pa-xxx",
        "dim": 2048,                         # 最高维度
        "truncate": True
    }
)
```

---

## 最佳实践

### 1. 模型选择策略

```python
# 场景1:通用RAG系统(推荐)
model_name = "voyage-3-large"
dim = 1024
# 理由:性能与质量最佳平衡

# 场景2:大规模部署(成本敏感)
model_name = "voyage-3-lite"
dim = 512
# 理由:成本降低80%,质量损失<5%

# 场景3:代码搜索引擎
model_name = "voyage-code-3"
dim = 1024
# 理由:代码语义专门优化

# 场景4:极致质量要求
model_name = "voyage-4-large"
dim = 2048
# 理由:最新最强模型
```

**来源**: `reference/search_github.md:88-90`

### 2. Int8量化决策树

```python
# 决策树
if data_size > 1_000_000:
    # 大规模数据:强烈推荐Int8
    dtype = DataType.INT8_VECTOR
    # 存储节省:75%,速度提升:4x
elif data_size > 100_000:
    # 中规模数据:推荐Int8
    dtype = DataType.INT8_VECTOR
    # 平衡性能与质量
else:
    # 小规模数据:使用Float32
    dtype = DataType.FLOAT_VECTOR
    # 质量优先
```

### 3. 维度优化策略

```python
# 根据数据规模选择维度
if data_size > 10_000_000:
    dim = 512   # 超大规模:低维度
elif data_size > 1_000_000:
    dim = 1024  # 大规模:标准维度(推荐)
elif data_size > 100_000:
    dim = 1024  # 中规模:标准维度
else:
    dim = 2048  # 小规模:高维度
```

### 4. 批量处理优化

```python
# VoyageAI支持最大批量128
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

### 陷阱1:维度设置超出范围

**错误示例**:

```python
# voyage-3-large支持的维度:256/512/1024/2048
params = {
    "model_name": "voyage-3-large",
    "dim": 768,  # ❌ 错误:不支持768维度
}
```

**正确做法**:

```python
# 使用支持的离散维度值
params = {
    "model_name": "voyage-3-large",
    "dim": 1024,  # ✅ 正确:使用1024
}
```

**错误信息**:
```
Error: Dimension 768 not supported. Supported dimensions: 256, 512, 1024, 2048
```

**来源**: `reference/context7_voyageai.md:140-145`

### 陷阱2:Int8量化配置不匹配

**错误示例**:

```python
# Schema定义Int8,但VoyageAI默认返回Float32
fields = [
    FieldSchema(name="vector", dtype=DataType.INT8_VECTOR, dim=1024)
]

# ❌ 错误:VoyageAI会自动转换,但需要确保模型支持
```

**正确做法**:

```python
# VoyageAI自动支持Int8输出
# Milvus会自动处理转换,无需额外配置
fields = [
    FieldSchema(name="vector", dtype=DataType.INT8_VECTOR, dim=1024)
]

# ✅ 正确:VoyageAI会自动返回Int8格式
```

**来源**: `reference/source_architecture.md:143-145`

### 陷阱3:超长文本未截断

**错误示例**:

```python
# 未设置truncate,超长文本会报错
params = {
    "truncate": False,  # ❌ 错误:生产环境应设置为True
}

# 插入超长文本
long_text = "..." * 10000  # 超过模型最大token限制
client.insert(collection_name="docs", data=[{"text": long_text}])
# 抛出异常:Text exceeds maximum token limit
```

**正确做法**:

```python
# 设置自动截断
params = {
    "truncate": True,  # ✅ 正确:自动截断超长文本
}

# 插入超长文本(自动截断)
long_text = "..." * 10000
client.insert(collection_name="docs", data=[{"text": long_text}])
# 成功:自动截断到模型最大token限制
```

**来源**: `reference/source_architecture.md:467-479`

### 陷阱4:API密钥配置错误

**错误示例**:

```python
# 使用错误的环境变量名
import os
os.environ["VOYAGE_API_KEY"] = "pa-xxx"  # ❌ 错误:变量名不对

params = {
    "provider": "voyageai",
    "model_name": "voyage-3-large"
    # 未传递api_key,期望从环境变量读取
}
```

**正确做法**:

```python
# 使用正确的环境变量名
import os
os.environ["VOYAGEAI_API_KEY"] = "pa-xxx"  # ✅ 正确

# 或直接在参数中传递
params = {
    "provider": "voyageai",
    "model_name": "voyage-3-large",
    "api_key": "pa-xxx"  # ✅ 推荐:直接传递
}
```

**认证优先级**:
1. Function参数中的`api_key`(最高优先级)
2. Milvus YAML配置文件
3. 环境变量`VOYAGEAI_API_KEY`(最低优先级)

**来源**: `reference/source_architecture.md:346-358`

---

## 性能对比

### VoyageAI vs OpenAI vs Cohere

| 指标 | VoyageAI | OpenAI | Cohere |
|------|----------|--------|--------|
| **MaxBatch** | 128 | 128 | 96 |
| **延迟(单次)** | ~150ms | ~300ms | ~250ms |
| **成本($/M)** | 0.12 | 0.02-0.13 | 0.10 |
| **Int8输出** | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| **维度灵活性** | ✅ 4档 | ✅ 连续 | ✅ 连续 |
| **社区推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**选择建议**:
- **速度优先**:选择VoyageAI(最快)
- **成本优先**:选择OpenAI text-embedding-3-small
- **大规模部署**:选择VoyageAI(Int8量化)
- **质量优先**:选择VoyageAI voyage-4-large

**来源**: `reference/source_architecture.md:389-405`, `reference/search_github.md:88-90`

---

## 社区最佳实践

### 1. 社区推荐配置

**来自GitHub社区的最佳实践**:

```python
# 社区最推荐的配置
voyageai_ef = Function(
    name="voyageai_community_best",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-3-large",      # 社区最推荐
        "api_key": "pa-xxx",
        "dim": 1024,                         # 标准维度
        "truncate": True                     # 生产环境必备
    }
)

# 使用Int8量化(大规模部署)
fields = [
    FieldSchema(name="vector", dtype=DataType.INT8_VECTOR, dim=1024)
]
```

**社区反馈**:
- "VoyageAI is the community favorite for RAG pipelines (speed + quality)"
- "VoyageAI或自托管模型适合大规模RAG"
- "速度快、质量高、成本合理"

**来源**: `reference/search_github.md:88-90`

### 2. 生产环境配置

```python
import os
from pymilvus import MilvusClient, Function, FunctionType

# 从环境变量读取API密钥
api_key = os.getenv("VOYAGEAI_API_KEY")

# 生产级配置
voyageai_prod_ef = Function(
    name="voyageai_prod",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-3-large",
        "api_key": api_key,
        "dim": 1024,
        "truncate": True,                    # 防止超长文本错误
        "url": "https://api.voyageai.com/v1/embeddings"
    }
)

# 初始化客户端
client = MilvusClient(uri="http://localhost:19530")

# 创建collection
client.create_collection(
    collection_name="prod_docs",
    dimension=1024,
    function=voyageai_prod_ef
)
```

### 3. 错误处理与重试

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
                # VoyageAI速率限制错误,等待后重试
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "token limit" in error_msg:
                # Token限制错误,检查truncate设置
                print("Token limit exceeded. Ensure truncate=True")
                raise
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

### 4. 性能监控

```python
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_embedding_performance(client, collection_name, data):
    """监控embedding生成性能"""
    start_time = time.time()
    batch_size = len(data)

    try:
        result = client.insert(collection_name=collection_name, data=data)
        duration = time.time() - start_time

        # 计算性能指标
        throughput = batch_size / duration
        avg_latency = duration / batch_size * 1000  # ms

        logger.info(f"Embedding performance:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} texts/s")
        logger.info(f"  Avg latency: {avg_latency:.2f}ms")

        # 告警:性能异常
        if avg_latency > 500:
            logger.warning(f"High latency detected: {avg_latency:.2f}ms")

        return result
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
```

**来源**: `reference/search_github.md:99-110`

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:127-151`
   - VoyageAI Provider实现细节
   - Int8量化支持
   - 输入类型优化逻辑

2. **官方文档**: `reference/context7_voyageai.md:1-145`
   - VoyageAI Embeddings API参数
   - 维度控制说明
   - Int8量化算法

3. **社区实践**: `reference/search_github.md:88-90`
   - 社区最推荐配置
   - 生产环境部署经验
   - 性能对比数据

---

## 总结

**VoyageAI Provider核心优势**:
1. **速度最快**:延迟~150ms,比OpenAI快50%
2. **Int8量化**:存储节省75%,速度提升4x
3. **灵活维度**:支持256/512/1024/2048四档
4. **社区认可**:GitHub社区最推荐的提供商

**适用场景**:
- 高性能RAG系统(速度优先)
- 大规模部署(Int8量化)
- 代码检索(voyage-code-3)
- 成本敏感场景(voyage-3-lite)

**不适用场景**:
- 极致成本优化(选择OpenAI text-embedding-3-small)
- 需要任务类型优化(选择VertexAI)
- 企业级SLA要求(选择OpenAI或VertexAI)

**社区评价**:
> "VoyageAI is the community favorite for RAG pipelines (speed + quality)"
>
> — GitHub Community, 2025

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
