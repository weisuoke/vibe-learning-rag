# 核心概念:TEI Provider

> Hugging Face TEI Provider是Milvus Embedding Functions中唯一支持自托管部署的提供商

---

## 提供商概述

**TEI (Text Embeddings Inference) Provider**是Milvus支持的10个embedding提供商中唯一支持自托管部署的选择,特别适合需要数据隐私保护和成本优化的场景。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **MaxBatch** | 32(可配置) | 可通过max_client_batch_size调整 |
| **输出类型** | Float32 | 仅支持浮点向量 |
| **维度控制** | ❌ 不支持 | 由模型决定 |
| **Prompt工程** | ✅ 支持 | ingestion_prompt + search_prompt |
| **延迟** | ~50ms | 自托管延迟最低 |
| **成本** | 硬件成本 | 无API调用费用 |
| **可靠性** | ⭐⭐⭐⭐ | 取决于部署环境 |

**来源**: `reference/source_architecture.md:291-320`, `reference/context7_tei.md:1-118`

---

## 配置参数详解

### 必需参数

```python
from pymilvus import Function, FunctionType

tei_ef = Function(
    name="tei_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "tei",                         # 必需:提供商名称
        "endpoint": "http://localhost:8080"        # 必需:TEI服务端点
    }
)
```

### 可选参数

```python
params = {
    "provider": "tei",
    "endpoint": "http://localhost:8080",

    # 可选参数
    "ingestion_prompt": "Represent the document for retrieval: ",  # 文档入库提示
    "search_prompt": "Represent the query for retrieving documents: ",  # 查询检索提示
    "truncate": True,                              # 是否截断超长文本
    "truncation_direction": "Right",               # 截断方向:Left/Right
    "max_client_batch_size": 32,                   # 客户端批量大小
    "api_key": "optional-key"                      # 可选API密钥
}
```

**来源**: `reference/source_architecture.md:297-319`, `reference/context7_tei.md:1-118`

---

## 参数说明

### 1. endpoint(服务端点)

**TEI服务部署地址**

```python
# 本地部署
params = {"endpoint": "http://localhost:8080"}

# Docker部署
params = {"endpoint": "http://tei-container:8080"}

# Kubernetes部署
params = {"endpoint": "http://tei-service.default.svc.cluster.local:8080"}

# 远程部署
params = {"endpoint": "https://tei.example.com"}
```

**部署方式**:

```bash
# Docker部署TEI服务
docker run -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-zh-v1.5 \
  --revision main
```

**来源**: `reference/context7_tei.md:1-118`

### 2. Prompt工程(独有特性)

**TEI是唯一支持Prompt工程的提供商**

```python
# 文档入库提示
ingestion_prompt = "Represent the document for retrieval: "

# 查询检索提示
search_prompt = "Represent the query for retrieving documents: "
```

**内部实现**:

```go
// 源码:tei_embedding_provider.go:138-145
if mode == models.InsertMode {
    // 使用ingestion_prompt
    prompt = provider.ingestionPrompt
} else {
    // 使用search_prompt
    prompt = provider.searchPrompt
}
```

**Prompt优化示例**:

```python
# 中文文档检索
params = {
    "ingestion_prompt": "为检索任务表示文档: ",
    "search_prompt": "为检索文档表示查询: "
}

# 代码检索
params = {
    "ingestion_prompt": "Represent the code snippet for retrieval: ",
    "search_prompt": "Represent the query for retrieving code: "
}

# 学术论文检索
params = {
    "ingestion_prompt": "Represent the academic paper for retrieval: ",
    "search_prompt": "Represent the research question for retrieving papers: "
}
```

**性能提升**:
- 合适的Prompt可提升检索准确率10-15%
- 不同场景需要不同的Prompt策略

**来源**: `reference/source_architecture.md:311-318`

### 3. truncate与truncation_direction

**截断控制**

```python
# 不截断(默认,超长文本会报错)
params = {
    "truncate": False
}

# 自动截断(推荐生产环境)
params = {
    "truncate": True,
    "truncation_direction": "Right"  # 从右侧截断
}

# 从左侧截断
params = {
    "truncate": True,
    "truncation_direction": "Left"   # 从左侧截断
}
```

**截断方向选择**:
- `Right`:保留文本开头,适合标题+正文结构
- `Left`:保留文本结尾,适合对话历史

**来源**: `reference/source_architecture.md:304-307`

### 4. max_client_batch_size

**客户端批量大小配置**

```python
# 默认批量大小
params = {"max_client_batch_size": 32}

# 增大批量(提升吞吐量)
params = {"max_client_batch_size": 64}

# 减小批量(降低内存占用)
params = {"max_client_batch_size": 16}
```

**批量大小选择**:
- 小批量(16):低内存环境
- 中批量(32):标准配置(推荐)
- 大批量(64):高性能环境

**来源**: `reference/source_architecture.md:307-309`

---

## 完整配置示例

### 示例1:本地自托管部署(推荐)

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# 定义TEI Embedding Function
tei_ef = Function(
    name="tei_self_hosted",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "tei",
        "endpoint": "http://localhost:8080",
        "ingestion_prompt": "Represent the document for retrieval: ",
        "search_prompt": "Represent the query for retrieving documents: ",
        "truncate": True,
        "truncation_direction": "Right",
        "max_client_batch_size": 32
    }
)

schema = CollectionSchema(fields=fields, functions=[tei_ef])
```

**部署TEI服务**:

```bash
# 使用Docker部署
docker run -d \
  --name tei-server \
  -p 8080:80 \
  -v $PWD/data:/data \
  --gpus all \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-zh-v1.5 \
  --revision main \
  --max-batch-tokens 16384
```

**适用场景**:
- 数据隐私要求高
- 成本敏感场景
- 高并发需求

**来源**: `reference/source_architecture.md:291-320`

### 示例2:中文RAG系统

```python
# 中文优化配置
tei_chinese_ef = Function(
    name="tei_chinese",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "tei",
        "endpoint": "http://localhost:8080",
        "ingestion_prompt": "为检索任务表示文档: ",
        "search_prompt": "为检索文档表示查询: ",
        "truncate": True,
        "max_client_batch_size": 32
    }
)
```

### 示例3:代码检索系统

```python
# 代码检索专用配置
tei_code_ef = Function(
    name="tei_code",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["code_snippet"],
    output_field_names=["code_vector"],
    params={
        "provider": "tei",
        "endpoint": "http://localhost:8080",
        "ingestion_prompt": "Represent the code snippet for retrieval: ",
        "search_prompt": "Represent the query for retrieving code: ",
        "truncate": True,
        "max_client_batch_size": 32
    }
)
```

---

## 最佳实践

### 1. TEI服务部署策略

```bash
# 生产环境部署(GPU加速)
docker run -d \
  --name tei-prod \
  -p 8080:80 \
  --gpus all \
  --restart unless-stopped \
  -v /data/tei:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-zh-v1.5 \
  --max-batch-tokens 16384 \
  --max-concurrent-requests 512

# CPU部署(成本优化)
docker run -d \
  --name tei-cpu \
  -p 8080:80 \
  --restart unless-stopped \
  -v /data/tei:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id BAAI/bge-base-zh-v1.5 \
  --max-batch-tokens 8192
```

### 2. Prompt优化策略

```python
# 场景1:通用文档检索
ingestion_prompt = "Represent the document for retrieval: "
search_prompt = "Represent the query for retrieving documents: "

# 场景2:问答系统
ingestion_prompt = "Represent the answer for question matching: "
search_prompt = "Represent the question for finding answers: "

# 场景3:代码搜索
ingestion_prompt = "Represent the code snippet for retrieval: "
search_prompt = "Represent the query for retrieving code: "
```

### 3. 批量处理优化

```python
# TEI批量大小:32(可配置)
batch_size = 32

from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 批量插入
texts = ["文本1", "文本2", ..., "文本32"]  # 32条
client.insert(
    collection_name="docs",
    data=[{"text": t} for t in texts]
)
```

**来源**: `reference/source_architecture.md:391-405`

---

## 常见陷阱与解决方案

### 陷阱1:TEI服务未启动

**错误示例**:

```python
# TEI服务未启动
params = {
    "endpoint": "http://localhost:8080"
}

client.insert(collection_name="docs", data=[{"text": "test"}])
# 错误:Connection refused
```

**正确做法**:

```bash
# 先启动TEI服务
docker run -d -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-zh-v1.5

# 等待服务就绪
curl http://localhost:8080/health
```

### 陷阱2:Prompt配置不当

**错误示例**:

```python
# 使用相同的Prompt
params = {
    "ingestion_prompt": "Represent the text: ",
    "search_prompt": "Represent the text: "  # ❌ 错误:与ingestion_prompt相同
}
```

**正确做法**:

```python
# 使用不同的Prompt
params = {
    "ingestion_prompt": "Represent the document for retrieval: ",
    "search_prompt": "Represent the query for retrieving documents: "  # ✅ 正确
}
```

**来源**: `reference/source_architecture.md:467-479`

### 陷阱3:批量大小配置错误

**错误示例**:

```python
# 批量大小过大
params = {
    "max_client_batch_size": 256  # ❌ 错误:可能导致OOM
}
```

**正确做法**:

```python
# 根据硬件资源配置
params = {
    "max_client_batch_size": 32  # ✅ 正确:标准配置
}
```

### 陷阱4:模型路径错误

**错误示例**:

```bash
# 模型ID错误
docker run -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id bge-large-zh  # ❌ 错误:缺少组织名
```

**正确做法**:

```bash
# 完整的模型ID
docker run -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-zh-v1.5  # ✅ 正确
```

---

## 性能对比

### TEI vs OpenAI vs VoyageAI

| 指标 | TEI(自托管) | OpenAI | VoyageAI |
|------|-------------|--------|----------|
| **MaxBatch** | 32(可配置) | 128 | 128 |
| **延迟(单次)** | ~50ms | ~300ms | ~150ms |
| **成本** | 硬件成本 | $0.02/M | $0.12/M |
| **数据隐私** | ✅ 完全控制 | ⚠️ 第三方 | ⚠️ 第三方 |
| **Prompt工程** | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| **部署复杂度** | ⚠️ 需要运维 | ✅ 即开即用 | ✅ 即开即用 |

**选择建议**:
- **数据隐私优先**:选择TEI(自托管)
- **成本敏感**:选择TEI(无API费用)
- **低延迟要求**:选择TEI(~50ms)
- **运维能力有限**:选择OpenAI或VoyageAI

**来源**: `reference/source_architecture.md:389-405`

---

## 生产环境建议

### 1. 高可用部署

```bash
# Kubernetes部署(推荐)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tei-deployment
spec:
  replicas: 3  # 3副本高可用
  selector:
    matchLabels:
      app: tei
  template:
    metadata:
      labels:
        app: tei
    spec:
      containers:
      - name: tei
        image: ghcr.io/huggingface/text-embeddings-inference:latest
        args:
          - --model-id
          - BAAI/bge-large-zh-v1.5
          - --max-batch-tokens
          - "16384"
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
```

### 2. 负载均衡

```python
# 多端点负载均衡
import random

endpoints = [
    "http://tei-1:8080",
    "http://tei-2:8080",
    "http://tei-3:8080"
]

def get_endpoint():
    """随机选择端点"""
    return random.choice(endpoints)

# 使用负载均衡
tei_ef = Function(
    name="tei_lb",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["vector"],
    params={
        "provider": "tei",
        "endpoint": get_endpoint(),
        "truncate": True
    }
)
```

### 3. 监控与告警

```python
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_tei_health(endpoint):
    """监控TEI服务健康状态"""
    import requests

    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"TEI service healthy: {endpoint}")
            return True
        else:
            logger.warning(f"TEI service unhealthy: {endpoint}")
            return False
    except Exception as e:
        logger.error(f"TEI service unreachable: {endpoint}, error: {e}")
        return False

# 定期健康检查
import threading

def health_check_loop(endpoint, interval=60):
    """定期健康检查"""
    while True:
        monitor_tei_health(endpoint)
        time.sleep(interval)

# 启动健康检查线程
threading.Thread(
    target=health_check_loop,
    args=("http://localhost:8080", 60),
    daemon=True
).start()
```

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:291-320`
   - TEI Provider实现细节
   - Prompt工程支持
   - 批量大小配置

2. **官方文档**: `reference/context7_tei.md:1-118`
   - Hugging Face TEI部署指南
   - 模型选择建议
   - 性能优化技巧

---

## 总结

**TEI Provider核心优势**:
1. **自托管部署**:完全控制数据隐私
2. **Prompt工程**:唯一支持Prompt定制的提供商
3. **低延迟**:~50ms,最快的选择
4. **成本优化**:无API调用费用,仅硬件成本

**适用场景**:
- 数据隐私要求高的企业
- 成本敏感的大规模部署
- 需要Prompt定制的场景
- 低延迟要求的应用

**不适用场景**:
- 运维能力有限(选择OpenAI或VoyageAI)
- 需要快速上线(选择云服务)
- 需要Int8量化(选择VoyageAI或Cohere)

**关键注意事项**:
- 需要自行部署和维护TEI服务
- Prompt工程需要针对具体场景优化
- 批量大小需要根据硬件资源调整

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
