---
type: context7_documentation
library: milvus
version: v2.6.x
fetched_at: 2026-02-25
knowledge_point: 02_多向量检索
context7_query: AnnSearchRequest RRFRanker WeightedRanker multi-vector hybrid search
---

# Context7 文档：Milvus 多向量检索与混合检索

## 文档来源
- 库名称：Milvus
- 版本：v2.6.x
- 官方文档链接：https://github.com/milvus-io/milvus-docs
- 总代码片段：5075 个
- 信任评分：9.8/10
- 基准评分：84.9/100

## 核心 API 组件

### 1. AnnSearchRequest

**定义**：用于定义单个向量字段的检索请求。

**Python 示例**：
```python
from pymilvus import AnnSearchRequest

# 定义稠密向量检索请求
dense_req = AnnSearchRequest(
    data=[query_dense_embedding],      # 查询向量
    anns_field="dense_vector",         # 向量字段名
    param={"metric_type": "IP"},       # 检索参数
    limit=10                           # 返回结果数量
)

# 定义稀疏向量检索请求
sparse_req = AnnSearchRequest(
    data=[query_sparse_embedding],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10
)
```

**关键参数**：
- `data`：查询向量数据
- `anns_field`：要检索的向量字段名
- `param`：检索参数（metric_type、nprobe 等）
- `limit`：返回结果数量

### 2. RRFRanker（Reciprocal Rank Fusion）

**定义**：使用 Reciprocal Rank Fusion 算法融合多个检索结果。

**Python 示例**：
```python
from pymilvus import RRFRanker

# 创建 RRF 排序器
ranker = RRFRanker()

# 执行混合检索
results = client.hybrid_search(
    collection_name="my_collection",
    reqs=[dense_req, sparse_req],
    ranker=ranker,
    limit=10,
    output_fields=["text", "metadata"]
)
```

**RRF 算法原理**：
- 对每个检索结果计算倒数排名分数
- 公式：`score = 1 / (k + rank)`
- `k` 参数默认值：60（可通过 `hybrid_ranker_params` 配置）
- 融合多个检索结果的分数

**配置示例**：
```python
# 配置 RRF 参数
hybrid_ranker_params = {"k": 60}
```

### 3. WeightedRanker（加权排序器）

**定义**：使用加权求和方式融合多个检索结果。

**Python 示例**：
```python
from pymilvus import WeightedRanker

# 创建加权排序器
ranker = WeightedRanker(
    sparse_weight=0.7,   # 稀疏向量权重
    dense_weight=1.0     # 稠密向量权重
)

# 执行混合检索
results = col.hybrid_search(
    [sparse_req, dense_req],
    rerank=ranker,
    limit=10,
    output_fields=["text"]
)
```

**权重配置**：
- 第一个参数：第一个检索请求的权重
- 第二个参数：第二个检索请求的权重
- 权重可以是任意正数，不需要归一化

**配置示例**（LlamaIndex 集成）：
```python
hybrid_ranker_params = {
    "weights": [0.5, 1.0]  # [稠密向量权重, 稀疏向量权重]
}
```

## 混合检索完整流程

### 基础混合检索示例

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    # 1. 定义稠密向量检索请求
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding],
        "dense_vector",
        dense_search_params,
        limit=limit
    )

    # 2. 定义稀疏向量检索请求
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding],
        "sparse_vector",
        sparse_search_params,
        limit=limit
    )

    # 3. 创建加权排序器
    rerank = WeightedRanker(sparse_weight, dense_weight)

    # 4. 执行混合检索
    res = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text"]
    )[0]

    return [hit.get("text") for hit in res]
```

### 使用 RRF 的混合检索示例

```python
from pymilvus import AnnSearchRequest, RRFRanker

# 定义检索请求
req_dense = AnnSearchRequest(
    data=[rng.random(5) for _ in range(3)],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=10
)

req_sparse = AnnSearchRequest(
    data=[csr_matrix(rng.random(5)) for _ in range(3)],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 创建 RRF 排序器
ranker = RRFRanker()

# 执行混合检索
res = await async_client.hybrid_search(
    collection_name="my_collection",
    reqs=[req_dense, req_sparse],
    ranker=ranker,
    output_fields=["text", "dense_vector", "sparse_vector"]
)
```

### BM25 + 稠密向量混合检索

```python
from pymilvus import AnnSearchRequest, RRFRanker

# 查询文本
query = "what is hybrid search"

# 获取查询向量
query_embedding = get_embeddings([query])[0]

# 1. BM25 稀疏向量检索请求
sparse_search_params = {"metric_type": "BM25"}
sparse_request = AnnSearchRequest(
    [query],                    # 直接传入文本
    "sparse_vector",
    sparse_search_params,
    limit=5
)

# 2. 稠密向量检索请求
dense_search_params = {"metric_type": "IP"}
dense_request = AnnSearchRequest(
    [query_embedding],          # 传入向量
    "dense_vector",
    dense_search_params,
    limit=5
)

# 3. 使用 RRF 融合结果
results = client.hybrid_search(
    collection_name,
    [sparse_request, dense_request],
    ranker=RRFRanker(),
    limit=5,
    output_fields=["content", "metadata"]
)
```

## 多向量字段定义

### Schema 定义示例

```python
from pymilvus import DataType

# 添加多个向量字段
schema.add_field(
    field_name="text_dense_vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=768,
    description="text dense vector"
)

schema.add_field(
    field_name="text_sparse_vector",
    datatype=DataType.SPARSE_FLOAT_VECTOR,
    description="text sparse vector"
)
```

### LangChain 集成示例

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.vectorstores.utils import BM25BuiltInFunction

# 定义多个 Embedding 模型
embedding1 = OpenAIEmbeddings(model="text-embedding-ada-002")
embedding2 = OpenAIEmbeddings(model="text-embedding-3-large")

# 创建 Milvus 向量存储（支持多向量字段）
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=[embedding1, embedding2],
    builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
    vector_field=["dense1", "dense2", "sparse"],
    connection_args={"uri": URI},
    consistency_level="Bounded",
    drop_old=False,
)

print(vectorstore.vector_fields)
# 输出：['dense1', 'dense2', 'sparse']
```

## 多向量检索核心概念

### 1. Multi-Vector Hybrid Search

**定义**：Milvus 支持在多个向量字段上同时进行 ANN 检索，并融合结果。

**应用场景**：
- 文本 + 图片多模态检索
- 多个文本字段描述同一对象
- 稠密向量 + 稀疏向量混合检索

**来源**：
> Milvus supports multi-vector hybrid search by allowing search on multiple vector fields, conducting several Approximate Nearest Neighbor (ANN) searches simultaneously. This is particularly useful if you want to search both text and images, multiple text fields that describe the same object, or dense and sparse vectors to improve search quality.

### 2. Sparse-Dense Vector Search

**定义**：结合稠密向量（语义理解）和稀疏向量（精确关键词匹配）的混合检索。

**优势**：
- 稠密向量：捕获语义关系
- 稀疏向量：精确关键词匹配
- 混合检索：兼顾语义理解和精确匹配

**来源**：
> Sparse-Dense Vector Search combines dense vectors, which are excellent for capturing semantic relationships, with sparse vectors, which are highly effective for precise keyword matching. This hybrid approach provides both a broad conceptual understanding and exact term relevance, thus improving search results by overcoming the limitations of individual methods.

### 3. 多向量字段能力

**Milvus 特性**：
- 一个 Collection 可以存储多个向量字段
- 向量字段可以是稠密或稀疏类型
- 不同向量字段可以有不同的维度

**来源**：
> Milvus enables the storage of multiple vector fields within a single collection, which can be either sparse or dense and may vary in dimensionality. This capability supports multi-vector and hybrid search scenarios, allowing users to perform complex searches across different types of vector representations.

## 其他 Ranker 类型

### Decay Ranker（衰减排序器）

**应用场景**：基于时间或其他字段的衰减排序。

**示例**：
```python
from pymilvus import AnnSearchRequest

# 定义检索请求
dense = AnnSearchRequest(
    data=[your_query_vector_1],
    anns_field="dense_vector",
    param={},
    limit=10
)

sparse = AnnSearchRequest(
    data=[your_query_vector_2],
    anns_field="sparse_vector",
    param={},
    limit=10
)

# 应用衰减排序器
hybrid_results = milvus_client.hybrid_search(
    collection_name,
    [dense, sparse],
    ranker=ranker,  # 衰减排序器
    limit=10,
    output_fields=["title", "venue", "event_date"]
)
```

**衰减类型**：
- Linear Decay（线性衰减）
- Exponential Decay（指数衰减）

## 关键技术要点总结

### 1. API 设计
- `AnnSearchRequest`：定义单个向量字段的检索请求
- `hybrid_search`：执行多向量混合检索
- `ranker`：融合多个检索结果

### 2. Ranker 选择
- **RRFRanker**：适合不确定权重的场景，自动平衡多个检索结果
- **WeightedRanker**：适合已知权重的场景，可精确控制各向量字段的重要性
- **Decay Ranker**：适合需要基于时间或其他字段衰减的场景

### 3. 应用场景
- **多模态检索**：图片 + 文本
- **全文搜索增强**：BM25 + 稠密向量
- **多语言检索**：中文向量 + 英文向量
- **多粒度检索**：标题向量 + 内容向量

### 4. 性能优化
- 每个向量字段都需要创建索引
- `limit` 参数控制每个检索请求的返回数量
- 最终结果数量由 `hybrid_search` 的 `limit` 参数控制

## 下一步调研方向

1. **网络搜索**：
   - 多模态检索实践案例（CLIP 模型集成）
   - RRF vs Weighted Ranker 性能对比
   - 生产环境中的多向量检索优化策略
   - 2025-2026 年最新的多向量检索应用案例

2. **需要深入理解的技术点**：
   - RRF 算法的数学原理和参数调优
   - 加权融合策略的权重调整方法
   - 多向量检索的性能优化技巧
   - 不同 Ranker 的适用场景对比
