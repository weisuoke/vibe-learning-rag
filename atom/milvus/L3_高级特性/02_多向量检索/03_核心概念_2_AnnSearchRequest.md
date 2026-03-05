# 核心概念 2：AnnSearchRequest - 检索请求定义

> **来源**：Milvus 源码分析 + Context7 官方文档
> **知识点**：02_多向量检索
> **文档版本**：v1.0
> **最后更新**：2026-02-25

---

## 1. 核心定义

### 1.1 什么是 AnnSearchRequest？

**AnnSearchRequest** 是 Milvus 多向量检索中用于定义**单个向量字段检索请求**的核心类。

**核心作用**：
- 指定要检索的向量字段
- 提供查询向量数据
- 配置检索参数（距离度量、搜索参数等）
- 设置返回结果数量

**在多向量检索中的位置**：
```
多向量混合检索流程：
1. 创建多个 AnnSearchRequest（每个向量字段一个）
2. 选择 Ranker（RRFRanker 或 WeightedRanker）
3. 调用 hybrid_search API
4. 获取融合后的结果
```

### 1.2 为什么需要 AnnSearchRequest？

在多向量检索场景中，一个 Collection 可能包含多个向量字段（如稠密向量、稀疏向量、图片向量等），每个向量字段需要：
- **独立的查询向量**：不同类型的向量字段需要不同的查询向量
- **独立的检索参数**：不同向量字段可能使用不同的距离度量和搜索参数
- **独立的结果数量**：每个向量字段可以返回不同数量的候选结果

**AnnSearchRequest 提供了统一的接口来定义这些独立的检索请求。**

---

## 2. 参数详解

### 2.1 核心参数概览

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `data` | List | ✓ | 查询向量数据 |
| `anns_field` | String | ✓ | 要检索的向量字段名 |
| `param` | Dict | ✓ | 检索参数（metric_type、nprobe 等） |
| `limit` | Int | ✓ | 返回结果数量 |

### 2.2 参数 1：data（查询向量数据）

**定义**：要检索的查询向量数据。

**数据类型**：
- **稠密向量**：`List[float]` 或 `numpy.ndarray`
- **稀疏向量**：`scipy.sparse.csr_matrix` 或 `Dict[int, float]`
- **BM25 检索**：直接传入文本字符串

**示例**：
```python
# 稠密向量
dense_data = [[0.1, 0.2, 0.3, ..., 0.768]]  # 768 维向量

# 稀疏向量
from scipy.sparse import csr_matrix
sparse_data = [csr_matrix([[0.1, 0.0, 0.3, 0.0, 0.5]])]

# BM25 检索（直接传入文本）
bm25_data = ["what is hybrid search"]
```

**注意事项**：
- 向量维度必须与 Schema 中定义的维度一致
- 支持批量查询（传入多个查询向量）
- BM25 检索时，Milvus 会自动将文本转换为稀疏向量

### 2.3 参数 2：anns_field（向量字段名）

**定义**：指定要检索的向量字段名称。

**要求**：
- 字段名必须在 Collection Schema 中存在
- 字段类型必须是向量类型（FloatVector、SparseFloatVector、BinaryVector、BFloat16Vector）
- 该字段必须已创建索引

**示例**：
```python
# 检索稠密向量字段
anns_field = "dense_vector"

# 检索稀疏向量字段
anns_field = "sparse_vector"

# 检索图片向量字段
anns_field = "image_vector"
```

### 2.4 参数 3：param（检索参数）

**定义**：配置检索行为的参数字典。

**核心参数**：

#### 3.1 metric_type（距离度量）

**支持的度量类型**：
- `L2`：欧氏距离（值越小越相似）
- `IP`：内积（值越大越相似）
- `COSINE`：余弦相似度（值越大越相似）
- `JACCARD`：Jaccard 距离（二值向量）
- `HAMMING`：汉明距离（二值向量）
- `BM25`：BM25 算法（稀疏向量）

**示例**：
```python
# 稠密向量使用内积
param = {"metric_type": "IP"}

# 稀疏向量使用内积
param = {"metric_type": "IP"}

# BM25 检索
param = {"metric_type": "BM25"}
```

#### 3.2 params（索引特定参数）

**HNSW 索引参数**：
```python
param = {
    "metric_type": "IP",
    "params": {
        "ef": 64  # 搜索时的候选集大小
    }
}
```

**IVF 索引参数**：
```python
param = {
    "metric_type": "L2",
    "params": {
        "nprobe": 16  # 搜索的聚类中心数量
    }
}
```

### 2.5 参数 4：limit（返回结果数量）

**定义**：每个检索请求返回的最大结果数量。

**注意事项**：
- 这是**单个检索请求**的返回数量
- 最终混合检索结果数量由 `hybrid_search` 的 `limit` 参数控制
- 建议设置为最终结果数量的 2-5 倍，以提高融合质量

**示例**：
```python
# 单个检索请求返回 10 条结果
limit = 10

# 如果最终需要 5 条结果，建议设置为 10-25
limit = 20
```

---

## 3. 完整代码示例

### 3.1 Python SDK 示例

#### 示例 1：基础稠密向量检索

```python
from pymilvus import AnnSearchRequest, MilvusClient
import numpy as np

# 连接 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 生成查询向量
query_vector = np.random.rand(768).tolist()

# 创建 AnnSearchRequest
dense_req = AnnSearchRequest(
    data=[query_vector],              # 查询向量
    anns_field="dense_vector",        # 向量字段名
    param={
        "metric_type": "IP",          # 内积距离
        "params": {"ef": 64}          # HNSW 参数
    },
    limit=10                          # 返回 10 条结果
)

print(f"检索请求创建成功：{dense_req}")
```

#### 示例 2：稀疏向量检索

```python
from pymilvus import AnnSearchRequest
from scipy.sparse import csr_matrix
import numpy as np

# 生成稀疏向量（只有少数非零元素）
sparse_vector = csr_matrix(np.random.rand(1, 1000))

# 创建稀疏向量检索请求
sparse_req = AnnSearchRequest(
    data=[sparse_vector],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10
)

print(f"稀疏向量检索请求：{sparse_req}")
```

#### 示例 3：BM25 文本检索

```python
from pymilvus import AnnSearchRequest

# 直接传入文本查询
query_text = "what is hybrid search"

# 创建 BM25 检索请求
bm25_req = AnnSearchRequest(
    data=[query_text],                # 直接传入文本
    anns_field="sparse_vector",       # BM25 函数生成的稀疏向量字段
    param={"metric_type": "BM25"},    # 使用 BM25 度量
    limit=5
)

print(f"BM25 检索请求：{bm25_req}")
```

#### 示例 4：多向量混合检索

```python
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient
import numpy as np

# 连接 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 1. 创建稠密向量检索请求
dense_vector = np.random.rand(768).tolist()
dense_req = AnnSearchRequest(
    data=[dense_vector],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 2. 创建稀疏向量检索请求
query_text = "machine learning tutorial"
sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=10
)

# 3. 执行混合检索
results = client.hybrid_search(
    collection_name="my_collection",
    reqs=[dense_req, sparse_req],     # 多个检索请求
    ranker=RRFRanker(),               # 使用 RRF 融合
    limit=5,                          # 最终返回 5 条结果
    output_fields=["text", "metadata"]
)

# 4. 处理结果
for hit in results[0]:
    print(f"ID: {hit['id']}, Score: {hit['distance']}, Text: {hit['text']}")
```

### 3.2 Go SDK 示例

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/milvus-io/milvus-sdk-go/v2/client"
    "github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func main() {
    ctx := context.Background()

    // 连接 Milvus
    c, err := client.NewClient(ctx, client.Config{
        Address: "localhost:19530",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()

    // 生成查询向量
    queryVector := make([]float32, 768)
    for i := range queryVector {
        queryVector[i] = 0.1
    }

    // 创建检索请求
    sp, _ := entity.NewIndexHNSWSearchParam(64)

    // 执行检索
    results, err := c.Search(
        ctx,
        "my_collection",           // Collection 名称
        []string{},                // Partition 名称（空表示全部）
        "",                        // 过滤表达式
        []string{"text"},          // 输出字段
        []entity.Vector{           // 查询向量
            entity.FloatVector(queryVector),
        },
        "dense_vector",            // 向量字段名
        entity.IP,                 // 距离度量
        10,                        // 返回结果数量
        sp,                        // 搜索参数
    )

    if err != nil {
        log.Fatal(err)
    }

    // 处理结果
    for _, result := range results {
        fmt.Printf("ID: %d, Score: %f\n", result.ID, result.Score)
    }
}
```

### 3.3 Java SDK 示例

```java
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.R;

import java.util.Arrays;
import java.util.List;

public class AnnSearchExample {
    public static void main(String[] args) {
        // 连接 Milvus
        MilvusServiceClient client = new MilvusServiceClient(
            ConnectParam.newBuilder()
                .withHost("localhost")
                .withPort(19530)
                .build()
        );

        // 生成查询向量
        List<Float> queryVector = Arrays.asList(0.1f, 0.2f, 0.3f /* ... 768 维 */);
        List<List<Float>> queryVectors = Arrays.asList(queryVector);

        // 创建检索请求
        SearchParam searchParam = SearchParam.newBuilder()
            .withCollectionName("my_collection")
            .withVectorFieldName("dense_vector")
            .withVectors(queryVectors)
            .withMetricType(MetricType.IP)
            .withTopK(10)
            .withParams("{\"ef\": 64}")
            .withOutFields(Arrays.asList("text", "metadata"))
            .build();

        // 执行检索
        R<SearchResults> response = client.search(searchParam);

        if (response.getStatus() != R.Status.Success.getCode()) {
            System.err.println("Search failed: " + response.getMessage());
            return;
        }

        // 处理结果
        SearchResults results = response.getData();
        System.out.println("Found " + results.getResults().getTopK() + " results");
    }
}
```

---

## 4. 使用场景

### 4.1 场景 1：稠密向量 + 稀疏向量混合检索

**应用**：文档问答系统，结合语义理解和关键词匹配。

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 稠密向量检索（语义理解）
dense_req = AnnSearchRequest(
    data=[dense_embedding],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=20
)

# 稀疏向量检索（关键词匹配）
sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=20
)

# 加权融合（稀疏向量权重更高）
results = client.hybrid_search(
    collection_name="documents",
    reqs=[sparse_req, dense_req],
    ranker=WeightedRanker(0.7, 0.3),  # 稀疏 70%，稠密 30%
    limit=10
)
```

### 4.2 场景 2：多模态检索（图片 + 文本）

**应用**：电商搜索，同时检索图片和文本描述。

```python
# 图片向量检索
image_req = AnnSearchRequest(
    data=[image_embedding],
    anns_field="image_vector",
    param={"metric_type": "L2"},
    limit=15
)

# 文本向量检索
text_req = AnnSearchRequest(
    data=[text_embedding],
    anns_field="text_vector",
    param={"metric_type": "IP"},
    limit=15
)

# RRF 融合（自动平衡）
results = client.hybrid_search(
    collection_name="products",
    reqs=[image_req, text_req],
    ranker=RRFRanker(),
    limit=10
)
```

### 4.3 场景 3：多粒度检索（标题 + 内容）

**应用**：新闻推荐，同时检索标题和正文。

```python
# 标题向量检索（精准匹配）
title_req = AnnSearchRequest(
    data=[query_embedding],
    anns_field="title_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 内容向量检索（语义匹配）
content_req = AnnSearchRequest(
    data=[query_embedding],
    anns_field="content_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 加权融合（标题权重更高）
results = client.hybrid_search(
    collection_name="news",
    reqs=[title_req, content_req],
    ranker=WeightedRanker(1.5, 1.0),  # 标题权重 1.5 倍
    limit=10
)
```

---

## 5. 最佳实践

### 5.1 limit 参数设置

**原则**：单个检索请求的 `limit` 应大于最终结果数量。

```python
# ❌ 错误：limit 太小
dense_req = AnnSearchRequest(
    data=[query_vector],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=5  # 最终需要 10 条结果，但只返回 5 条候选
)

# ✅ 正确：limit 是最终结果的 2-5 倍
dense_req = AnnSearchRequest(
    data=[query_vector],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=20  # 最终需要 10 条结果，返回 20 条候选
)
```

**推荐比例**：
- 最终结果 5 条 → 单个请求 limit=10-25
- 最终结果 10 条 → 单个请求 limit=20-50
- 最终结果 50 条 → 单个请求 limit=100-250

### 5.2 metric_type 选择

**稠密向量**：
- 归一化向量 → 使用 `IP`（内积）
- 未归一化向量 → 使用 `L2`（欧氏距离）或 `COSINE`（余弦相似度）

**稀疏向量**：
- BM25 生成的向量 → 使用 `IP`
- TF-IDF 向量 → 使用 `IP` 或 `COSINE`

**二值向量**：
- 使用 `HAMMING` 或 `JACCARD`

### 5.3 批量查询优化

```python
# ❌ 低效：多次单查询
for query in queries:
    req = AnnSearchRequest(
        data=[query],
        anns_field="dense_vector",
        param={"metric_type": "IP"},
        limit=10
    )
    results = client.hybrid_search(...)

# ✅ 高效：批量查询
req = AnnSearchRequest(
    data=queries,  # 传入多个查询向量
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=10
)
results = client.hybrid_search(...)
```

### 5.4 索引参数调优

**HNSW 索引**：
```python
# 平衡性能和召回率
param = {
    "metric_type": "IP",
    "params": {
        "ef": 64  # 默认值，可根据需求调整（16-512）
    }
}
```

**IVF 索引**：
```python
# 平衡性能和召回率
param = {
    "metric_type": "L2",
    "params": {
        "nprobe": 16  # 默认值，可根据需求调整（1-nlist）
    }
}
```

---

## 6. 常见问题

### 6.1 问题 1：向量维度不匹配

**错误信息**：
```
dimension mismatch: expected 768, got 512
```

**原因**：查询向量维度与 Schema 中定义的维度不一致。

**解决方案**：
```python
# 检查 Schema 中的维度定义
collection_info = client.describe_collection("my_collection")
print(collection_info)

# 确保查询向量维度一致
query_vector = np.random.rand(768).tolist()  # 使用正确的维度
```

### 6.2 问题 2：字段不存在

**错误信息**：
```
field 'dense_vector' not found in collection
```

**原因**：`anns_field` 指定的字段名在 Collection 中不存在。

**解决方案**：
```python
# 检查 Collection 的字段列表
collection_info = client.describe_collection("my_collection")
for field in collection_info['fields']:
    print(f"Field: {field['name']}, Type: {field['type']}")

# 使用正确的字段名
req = AnnSearchRequest(
    data=[query_vector],
    anns_field="correct_field_name",  # 使用实际存在的字段名
    param={"metric_type": "IP"},
    limit=10
)
```

### 6.3 问题 3：索引未创建

**错误信息**：
```
index not found for field 'dense_vector'
```

**原因**：向量字段未创建索引。

**解决方案**：
```python
# 为向量字段创建索引
client.create_index(
    collection_name="my_collection",
    field_name="dense_vector",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 256}
    }
)

# 加载 Collection
client.load_collection("my_collection")
```

### 6.4 问题 4：BM25 检索失败

**错误信息**：
```
BM25 function not found for field 'sparse_vector'
```

**原因**：未为稀疏向量字段配置 BM25 函数。

**解决方案**：
```python
from pymilvus import Function, FunctionType

# 在 Schema 中添加 BM25 函数
bm25_function = Function(
    name="sparse_vector",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names="sparse_vector",
    params={}
)
schema.add_function(bm25_function)
```

---

## 7. 性能优化建议

### 7.1 减少网络传输

```python
# ❌ 传输大量无用字段
results = client.hybrid_search(
    collection_name="my_collection",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(),
    limit=10,
    output_fields=["*"]  # 返回所有字段
)

# ✅ 只返回需要的字段
results = client.hybrid_search(
    collection_name="my_collection",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(),
    limit=10,
    output_fields=["id", "text"]  # 只返回必要字段
)
```

### 7.2 合理设置 limit

```python
# 根据实际需求设置 limit
# 如果只需要前 5 条结果，不要设置 limit=100

# ✅ 合理设置
dense_req = AnnSearchRequest(
    data=[query_vector],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=10  # 最终需要 5 条，设置为 10 即可
)
```

### 7.3 使用连接池

```python
from pymilvus import MilvusClient, connections

# 使用连接池
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    pool_size=10  # 连接池大小
)

client = MilvusClient(uri="http://localhost:19530")
```

---

## 8. 总结

### 8.1 核心要点

1. **AnnSearchRequest 是多向量检索的基础单元**，用于定义单个向量字段的检索请求
2. **四个核心参数**：data（查询向量）、anns_field（字段名）、param（检索参数）、limit（结果数量）
3. **支持多种向量类型**：稠密向量、稀疏向量、二值向量、BM25 文本检索
4. **与 Ranker 配合使用**：通过 RRFRanker 或 WeightedRanker 融合多个检索结果
5. **limit 参数设置**：单个请求的 limit 应大于最终结果数量的 2-5 倍

### 8.2 应用场景

- **多模态检索**：图片 + 文本
- **全文搜索增强**：BM25 + 稠密向量
- **多粒度检索**：标题 + 内容
- **多语言检索**：中文向量 + 英文向量

### 8.3 最佳实践

- 合理设置 limit 参数（2-5 倍最终结果数量）
- 根据向量类型选择合适的 metric_type
- 使用批量查询提高性能
- 只返回需要的字段减少网络传输

---

## 9. 参考资料

### 9.1 官方文档
- [Milvus Multi-Vector Search](https://milvus.io/docs/multi-vector-search.md)
- [AnnSearchRequest API Reference](https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/AnnSearchRequest.md)

### 9.2 源码分析
- `tests/python_client/milvus_client_v2/test_milvus_client_hybrid_search_v2.py`
- `tests/integration/hellomilvus/hybridsearch_test.go`

### 9.3 相关知识点
- [03_核心概念_1_多向量字段定义.md](./03_核心概念_1_多向量字段定义.md)
- [03_核心概念_3_RRFRanker.md](./03_核心概念_3_RRFRanker.md)
- [03_核心概念_4_WeightedRanker.md](./03_核心概念_4_WeightedRanker.md)

---

**文档状态**：✅ 已完成
**字数统计**：约 450 行
**代码示例**：Python + Go + Java
**质量检查**：已通过
