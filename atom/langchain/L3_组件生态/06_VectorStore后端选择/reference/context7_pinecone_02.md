---
type: context7_documentation
library: Pinecone Python Client
library_id: /pinecone-io/pinecone-python-client
fetched_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
context7_query: Pinecone Python client for LangChain integration
---

# Context7 文档：Pinecone Python Client 官方文档

## 文档来源
- 库名称：Pinecone Python Client
- Library ID：/pinecone-io/pinecone-python-client
- 官方文档链接：https://github.com/pinecone-io/pinecone-python-client
- Source Reputation：High
- Benchmark Score：85.3

## 关键信息提取

### 1. Pinecone 快速开始

#### 完整工作流程

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

# 1. 初始化 Pinecone 客户端
pc = Pinecone(api_key='YOUR_API_KEY')

# 2. 创建索引
index_config = pc.create_index(
    name="index-name",
    dimension=1536,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    ),
    vector_type=VectorType.DENSE
)

# 3. 实例化索引客户端
idx = pc.Index(host=index_config.host)

# 4. 插入向量
idx.upsert(
    vectors=[
        ("id1", [0.1, 0.2, 0.3, 0.4, ...], {"metadata_key": "value1"}),
        ("id2", [0.2, 0.3, 0.4, 0.5, ...], {"metadata_key": "value2"}),
    ],
    namespace="example-namespace"
)

# 5. 查询索引
query_embedding = [...]  # 向量维度需要匹配索引维度
idx.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True,
    filter={"metadata_key": {"$eq": "value1"}}
)
```

**关键点**：
- 完整的 CRUD 操作流程
- 支持命名空间（namespace）隔离
- 支持元数据过滤
- 支持 top-k 检索

### 2. Serverless 索引创建

#### Dense 和 Sparse 向量索引

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone(api_key='<<PINECONE_API_KEY>>')

# Dense 向量索引
pc.create_index(
    name='index-for-dense-vectors',
    dimension=1536,
    metric=Metric.COSINE,
    vector_type=VectorType.DENSE,  # 默认值，可省略
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
)

# Sparse 向量索引
pc.create_index(
    name='index-for-sparse-vectors',
    metric=Metric.DOTPRODUCT,
    vector_type=VectorType.SPARSE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
)
```

**关键点**：
- 支持 Dense 和 Sparse 两种向量类型
- Dense 向量需要指定维度
- Sparse 向量不需要指定维度
- 支持多种距离度量（COSINE, EUCLIDEAN, DOTPRODUCT）

### 3. 多云支持

#### AWS 部署

```python
pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    )
)
```

#### GCP 部署

```python
from pinecone import GcpRegion

pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.GCP,
        region=GcpRegion.US_CENTRAL1
    )
)
```

**关键点**：
- 支持 AWS、GCP、Azure 多云部署
- 每个云提供商有不同的区域选项
- Serverless 模式自动扩展

### 4. Serverless 索引高级配置

```python
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric, VectorType

pc = Pinecone(api_key="YOUR_API_KEY")

# 创建 serverless 索引
index_config = pc.create_index(
    name="my-serverless-index",
    dimension=1536,
    metric=Metric.COSINE,  # 或 "cosine", "euclidean", "dotproduct"
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    ),
    vector_type=VectorType.DENSE,
    deletion_protection="disabled",  # 或 "enabled"
    tags={"environment": "production", "team": "ml"}
)

print(f"Index host: {index_config.host}")
print(f"Index status: {index_config.status}")
print(f"Dimension: {index_config.dimension}")
```

**关键点**：
- 支持删除保护（deletion_protection）
- 支持标签（tags）管理
- 返回索引配置信息（host, status, dimension）

### 5. Pod-Based 索引（专用基础设施）

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# 创建 pod-based 索引
index_config = pc.create_index(
    name="my-pod-index",
    dimension=768,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",  # Pod 类型：p1.x1, p1.x2, p2.x1, p2.x2, s1.x1, s1.x2
        replicas=2,  # 副本数（高可用）
        shards=1,  # 分片数（水平扩展）
        metadata_config={
            "indexed": ["genre", "year"]  # 元数据字段索引
        }
    ),
    deletion_protection="disabled"
)

print(f"Index created: {index_config.name}")
```

**关键点**：
- Pod-based 提供专用基础设施
- 支持自定义副本数和分片数
- 支持元数据字段索引
- 适合可预测的高性能工作负载

## 索引类型对比

| 特性 | Serverless | Pod-Based |
|------|-----------|-----------|
| 扩展方式 | 自动扩展 | 手动配置 |
| 成本模式 | 按使用量付费 | 按资源预留付费 |
| 适用场景 | 不可预测负载 | 可预测负载 |
| 配置复杂度 | 低 | 高 |
| 性能控制 | 自动优化 | 精细控制 |

## 距离度量选择

| Metric | 适用场景 | 范围 |
|--------|---------|------|
| COSINE | 归一化向量（文本 embedding） | [-1, 1] |
| EUCLIDEAN | 未归一化向量 | [0, ∞) |
| DOTPRODUCT | 归一化向量（性能最快） | [-1, 1] |

## 与 LangChain 的集成要点

1. **包依赖**：`langchain-pinecone`, `pinecone-client`
2. **初始化流程**：
   - 创建 Pinecone 客户端
   - 创建或连接索引
   - 创建 LangChain 向量存储
3. **命名空间**：支持在同一索引中隔离数据
4. **元数据过滤**：支持复杂的元数据查询
5. **云部署**：完全托管，无需运维

## 适用场景

- ✅ 生产环境（完全托管）
- ✅ 大规模应用（无限扩展）
- ✅ 需要快速上线
- ✅ 多租户场景（命名空间）
- ✅ 需要元数据过滤
- ❌ 预算有限（付费服务）
- ❌ 数据敏感（云端存储）
