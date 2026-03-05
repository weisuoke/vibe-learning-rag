# 核心概念 4：WeightedRanker（加权排序器）

## 一、定义与核心价值

### 1.1 什么是 WeightedRanker？

**WeightedRanker** 是 Milvus 多向量检索中的一种结果融合策略，使用**加权求和**方式融合多个向量字段的检索结果。

**核心特点**：
- **精确控制**：为每个检索请求指定明确的权重值
- **灵活配置**：权重可以是任意正数，不需要归一化
- **适用场景**：已知各向量字段重要性的场景
- **可解释性强**：权重直接反映各字段的重要程度

**来源**：Context7 官方文档 + Milvus 源码测试

---

## 二、WeightedRanker 的工作原理

### 2.1 加权融合算法

**基本公式**：
```
final_score = w1 * score1 + w2 * score2 + ... + wn * scoren
```

其中：
- `w1, w2, ..., wn`：各检索请求的权重
- `score1, score2, ..., scoren`：各检索请求返回的相似度分数

**关键特性**：
1. **权重不需要归一化**：可以使用任意正数（如 0.5, 1.0, 2.0, 10.0）
2. **线性加权**：最终分数是各检索结果的线性组合
3. **权重顺序**：权重顺序与检索请求顺序一致

### 2.2 与 RRFRanker 的对比

| 特性 | WeightedRanker | RRFRanker |
|------|----------------|-----------|
| **融合算法** | 加权求和 | Reciprocal Rank Fusion |
| **权重配置** | 需要手动指定权重 | 自动平衡，只需配置 k 参数 |
| **适用场景** | 已知各字段重要性 | 不确定权重的场景 |
| **可解释性** | 强（权重直接反映重要性） | 中（基于排名倒数） |
| **调优难度** | 需要实验确定最佳权重 | 相对简单 |
| **灵活性** | 高（可精确控制） | 中（主要调整 k 参数） |

**选择建议**：
- **使用 WeightedRanker**：当你明确知道各向量字段的重要性（如业务经验、A/B 测试结果）
- **使用 RRFRanker**：当你不确定权重时，让算法自动平衡

---

## 三、WeightedRanker 的使用方式

### 3.1 基础使用示例

**Python SDK 示例**：
```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 1. 定义稠密向量检索请求
dense_req = AnnSearchRequest(
    data=[query_dense_embedding],      # 查询向量
    anns_field="dense_vector",         # 向量字段名
    param={"metric_type": "IP"},       # 检索参数
    limit=10                           # 返回结果数量
)

# 2. 定义稀疏向量检索请求
sparse_req = AnnSearchRequest(
    data=[query_sparse_embedding],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 3. 创建加权排序器
# 第一个参数：第一个检索请求的权重（sparse_req）
# 第二个参数：第二个检索请求的权重（dense_req）
ranker = WeightedRanker(
    0.7,   # 稀疏向量权重
    1.0    # 稠密向量权重
)

# 4. 执行混合检索
results = col.hybrid_search(
    [sparse_req, dense_req],
    rerank=ranker,
    limit=10,
    output_fields=["text", "metadata"]
)

# 5. 处理结果
for hit in results[0]:
    print(f"ID: {hit.id}, Score: {hit.score}, Text: {hit.entity.get('text')}")
```

**关键点**：
- 权重顺序与检索请求顺序一致
- 权重可以是任意正数（0.7, 1.0, 2.0, 10.0 等）
- 不需要手动归一化权重

### 3.2 多向量字段加权示例

**场景**：图片 + 文本 + 标题三个向量字段的混合检索

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 定义三个检索请求
image_req = AnnSearchRequest(
    data=[query_image_embedding],
    anns_field="image_vector",
    param={"metric_type": "IP"},
    limit=10
)

text_req = AnnSearchRequest(
    data=[query_text_embedding],
    anns_field="text_vector",
    param={"metric_type": "IP"},
    limit=10
)

title_req = AnnSearchRequest(
    data=[query_title_embedding],
    anns_field="title_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 创建加权排序器（三个权重）
# 假设：图片最重要（2.0），文本次之（1.0），标题最轻（0.5）
ranker = WeightedRanker(2.0, 1.0, 0.5)

# 执行混合检索
results = col.hybrid_search(
    [image_req, text_req, title_req],
    rerank=ranker,
    limit=10,
    output_fields=["title", "text", "image_url"]
)
```

---

## 四、权重配置方法与策略

### 4.1 权重配置的三种方法

#### 方法 1：基于业务经验

**适用场景**：有明确的业务需求和领域知识

**示例**：电商搜索场景
```python
# 场景：用户搜索商品
# 业务经验：标题匹配 > 描述匹配 > 图片相似度

ranker = WeightedRanker(
    2.0,   # 标题向量权重（最重要）
    1.0,   # 描述向量权重（次要）
    0.5    # 图片向量权重（辅助）
)
```

#### 方法 2：基于 A/B 测试

**适用场景**：有用户反馈数据，可以进行实验

**实验流程**：
```python
# 定义多组权重配置
weight_configs = [
    (1.0, 1.0),    # 均等权重
    (0.7, 1.0),    # 稀疏向量权重较低
    (1.0, 0.7),    # 稠密向量权重较低
    (0.5, 1.0),    # 稀疏向量权重更低
    (1.0, 0.5),    # 稠密向量权重更低
]

# 对每组配置进行 A/B 测试
for sparse_w, dense_w in weight_configs:
    ranker = WeightedRanker(sparse_w, dense_w)

    # 执行检索并收集用户反馈
    results = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=ranker,
        limit=10
    )

    # 计算评估指标（点击率、转化率等）
    metrics = evaluate_results(results, user_feedback)
    print(f"Weights ({sparse_w}, {dense_w}): {metrics}")
```

#### 方法 3：基于网格搜索

**适用场景**：有标注数据集，可以进行离线评估

**实现示例**：
```python
import numpy as np
from sklearn.model_selection import ParameterGrid

# 定义权重搜索空间
param_grid = {
    'sparse_weight': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    'dense_weight': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
}

best_score = 0
best_weights = None

# 网格搜索
for params in ParameterGrid(param_grid):
    sparse_w = params['sparse_weight']
    dense_w = params['dense_weight']

    ranker = WeightedRanker(sparse_w, dense_w)

    # 在验证集上评估
    score = evaluate_on_validation_set(ranker)

    if score > best_score:
        best_score = score
        best_weights = (sparse_w, dense_w)

print(f"Best weights: {best_weights}, Score: {best_score}")
```

### 4.2 权重调整策略

#### 策略 1：从均等权重开始

**推荐起点**：所有权重设为 1.0
```python
# 起点：均等权重
ranker = WeightedRanker(1.0, 1.0)
```

**调整方向**：
- 如果某个向量字段效果更好 → 增加其权重
- 如果某个向量字段效果较差 → 降低其权重

#### 策略 2：基于相似度分布调整

**观察相似度分布**：
```python
# 分别执行单向量检索，观察分数分布
sparse_results = col.search(
    data=[query_sparse_embedding],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10
)

dense_results = col.search(
    data=[query_dense_embedding],
    anns_field="dense_vector",
    param={"metric_type": "IP"},
    limit=10
)

# 观察分数范围
print("Sparse scores:", [hit.score for hit in sparse_results[0]])
print("Dense scores:", [hit.score for hit in dense_results[0]])

# 如果分数范围差异很大，可以调整权重来平衡
# 例如：稀疏向量分数范围 [0.1, 0.5]，稠密向量分数范围 [0.6, 0.9]
# 可以增加稀疏向量权重：ranker = WeightedRanker(2.0, 1.0)
```

#### 策略 3：动态权重调整

**场景**：根据查询类型动态调整权重

```python
def get_dynamic_ranker(query_text):
    """根据查询类型动态选择权重"""

    # 短查询（1-3 个词）：更依赖精确匹配（稀疏向量）
    if len(query_text.split()) <= 3:
        return WeightedRanker(1.5, 1.0)  # 稀疏向量权重更高

    # 长查询（>3 个词）：更依赖语义理解（稠密向量）
    else:
        return WeightedRanker(0.7, 1.0)  # 稠密向量权重更高

# 使用示例
query = "machine learning"
ranker = get_dynamic_ranker(query)

results = col.hybrid_search(
    [sparse_req, dense_req],
    rerank=ranker,
    limit=10
)
```

---

## 五、实战场景与代码示例

### 5.1 场景 1：BM25 + 稠密向量混合检索

**场景描述**：文档问答系统，结合关键词匹配和语义理解

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def hybrid_search_with_weighted_ranker(
    col,
    query_text,
    query_embedding,
    sparse_weight=0.7,
    dense_weight=1.0,
    limit=10
):
    """
    使用 WeightedRanker 的混合检索

    Args:
        col: Collection 对象
        query_text: 查询文本（用于 BM25）
        query_embedding: 查询向量（用于稠密向量检索）
        sparse_weight: 稀疏向量权重
        dense_weight: 稠密向量权重
        limit: 返回结果数量
    """

    # 1. BM25 稀疏向量检索请求
    sparse_req = AnnSearchRequest(
        data=[query_text],              # 直接传入文本
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=limit
    )

    # 2. 稠密向量检索请求
    dense_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="dense_vector",
        param={"metric_type": "IP"},
        limit=limit
    )

    # 3. 创建加权排序器
    ranker = WeightedRanker(sparse_weight, dense_weight)

    # 4. 执行混合检索
    results = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=ranker,
        limit=limit,
        output_fields=["text", "metadata"]
    )

    return results[0]

# 使用示例
query = "什么是向量数据库？"
query_embedding = get_embeddings([query])[0]

results = hybrid_search_with_weighted_ranker(
    col=collection,
    query_text=query,
    query_embedding=query_embedding,
    sparse_weight=0.7,   # BM25 权重
    dense_weight=1.0,    # 稠密向量权重
    limit=5
)

for hit in results:
    print(f"Score: {hit.score:.4f}, Text: {hit.entity.get('text')[:100]}...")
```

### 5.2 场景 2：多模态检索（图片 + 文本）

**场景描述**：电商搜索，用户可以同时搜索图片和文本

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def multimodal_search(
    col,
    query_text=None,
    query_image=None,
    text_weight=1.0,
    image_weight=1.0,
    limit=10
):
    """
    多模态混合检索

    Args:
        col: Collection 对象
        query_text: 查询文本（可选）
        query_image: 查询图片（可选）
        text_weight: 文本向量权重
        image_weight: 图片向量权重
        limit: 返回结果数量
    """

    requests = []
    weights = []

    # 1. 如果有文本查询，添加文本检索请求
    if query_text:
        text_embedding = get_text_embedding(query_text)
        text_req = AnnSearchRequest(
            data=[text_embedding],
            anns_field="text_vector",
            param={"metric_type": "IP"},
            limit=limit
        )
        requests.append(text_req)
        weights.append(text_weight)

    # 2. 如果有图片查询，添加图片检索请求
    if query_image:
        image_embedding = get_image_embedding(query_image)
        image_req = AnnSearchRequest(
            data=[image_embedding],
            anns_field="image_vector",
            param={"metric_type": "IP"},
            limit=limit
        )
        requests.append(image_req)
        weights.append(image_weight)

    # 3. 创建加权排序器
    ranker = WeightedRanker(*weights)

    # 4. 执行混合检索
    results = col.hybrid_search(
        requests,
        rerank=ranker,
        limit=limit,
        output_fields=["title", "description", "image_url"]
    )

    return results[0]

# 使用示例 1：纯文本搜索
results = multimodal_search(
    col=collection,
    query_text="红色连衣裙",
    text_weight=1.0,
    limit=10
)

# 使用示例 2：纯图片搜索
results = multimodal_search(
    col=collection,
    query_image="path/to/image.jpg",
    image_weight=1.0,
    limit=10
)

# 使用示例 3：图文混合搜索
results = multimodal_search(
    col=collection,
    query_text="红色连衣裙",
    query_image="path/to/image.jpg",
    text_weight=0.6,    # 文本权重较低
    image_weight=1.0,   # 图片权重较高
    limit=10
)
```

### 5.3 场景 3：权重实验与评估

**场景描述**：通过实验找到最佳权重配置

```python
from pymilvus import AnnSearchRequest, WeightedRanker
import numpy as np

def evaluate_weight_config(
    col,
    test_queries,
    ground_truth,
    sparse_weight,
    dense_weight
):
    """
    评估特定权重配置的效果

    Args:
        col: Collection 对象
        test_queries: 测试查询列表
        ground_truth: 真实相关文档 ID
        sparse_weight: 稀疏向量权重
        dense_weight: 稠密向量权重

    Returns:
        评估指标（Precision@5, Recall@5, NDCG@5）
    """

    ranker = WeightedRanker(sparse_weight, dense_weight)

    precisions = []
    recalls = []
    ndcgs = []

    for query, relevant_ids in zip(test_queries, ground_truth):
        # 执行混合检索
        sparse_req = AnnSearchRequest(
            data=[query['sparse_embedding']],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=10
        )

        dense_req = AnnSearchRequest(
            data=[query['dense_embedding']],
            anns_field="dense_vector",
            param={"metric_type": "IP"},
            limit=10
        )

        results = col.hybrid_search(
            [sparse_req, dense_req],
            rerank=ranker,
            limit=5
        )

        # 计算评估指标
        retrieved_ids = [hit.id for hit in results[0]]

        # Precision@5
        precision = len(set(retrieved_ids) & set(relevant_ids)) / 5
        precisions.append(precision)

        # Recall@5
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
        recalls.append(recall)

        # NDCG@5
        ndcg = calculate_ndcg(retrieved_ids, relevant_ids, k=5)
        ndcgs.append(ndcg)

    return {
        'precision@5': np.mean(precisions),
        'recall@5': np.mean(recalls),
        'ndcg@5': np.mean(ndcgs)
    }

# 实验：测试不同权重配置
weight_configs = [
    (0.5, 1.0),
    (0.7, 1.0),
    (1.0, 1.0),
    (1.0, 0.7),
    (1.0, 0.5),
]

results = []
for sparse_w, dense_w in weight_configs:
    metrics = evaluate_weight_config(
        col=collection,
        test_queries=test_queries,
        ground_truth=ground_truth,
        sparse_weight=sparse_w,
        dense_weight=dense_w
    )

    results.append({
        'sparse_weight': sparse_w,
        'dense_weight': dense_w,
        **metrics
    })

    print(f"Weights ({sparse_w}, {dense_w}): {metrics}")

# 找到最佳配置
best_config = max(results, key=lambda x: x['ndcg@5'])
print(f"\nBest configuration: {best_config}")
```

---

## 六、权重调整的最佳实践

### 6.1 权重调整的五个原则

1. **从均等权重开始**：初始权重设为 1.0，逐步调整
2. **小步迭代**：每次调整权重时，变化幅度不要太大（如 0.1-0.3）
3. **基于数据决策**：使用评估指标（Precision、Recall、NDCG）指导调整
4. **考虑业务场景**：不同场景可能需要不同的权重配置
5. **定期重新评估**：数据分布变化时，需要重新调整权重

### 6.2 常见权重配置参考

| 场景 | 稀疏向量权重 | 稠密向量权重 | 说明 |
|------|--------------|--------------|------|
| **短查询（1-3 词）** | 1.5 | 1.0 | 更依赖精确匹配 |
| **长查询（>3 词）** | 0.7 | 1.0 | 更依赖语义理解 |
| **专业术语查询** | 2.0 | 1.0 | 精确匹配更重要 |
| **自然语言问题** | 0.5 | 1.0 | 语义理解更重要 |
| **多模态检索** | 0.6 | 1.0 | 图片权重可调 |

### 6.3 权重调整的常见错误

**错误 1：权重差异过大**
```python
# ❌ 错误：权重差异过大，某个向量字段被完全忽略
ranker = WeightedRanker(10.0, 0.1)

# ✅ 正确：权重差异适中
ranker = WeightedRanker(1.5, 1.0)
```

**错误 2：忽略权重顺序**
```python
# ❌ 错误：权重顺序与检索请求顺序不一致
requests = [sparse_req, dense_req]
ranker = WeightedRanker(1.0, 0.7)  # 这里 1.0 对应 sparse_req，0.7 对应 dense_req

# ✅ 正确：明确权重对应关系
# sparse_req 权重 0.7，dense_req 权重 1.0
ranker = WeightedRanker(0.7, 1.0)
```

**错误 3：不进行实验验证**
```python
# ❌ 错误：凭感觉设置权重，不进行验证
ranker = WeightedRanker(0.8, 1.2)  # 没有数据支持

# ✅ 正确：通过实验验证权重效果
for sparse_w, dense_w in [(0.5, 1.0), (0.7, 1.0), (1.0, 1.0)]:
    metrics = evaluate_weight_config(sparse_w, dense_w)
    print(f"Weights ({sparse_w}, {dense_w}): {metrics}")
```

---

## 七、总结与要点

### 7.1 核心要点

1. **WeightedRanker 使用加权求和融合多个检索结果**
2. **权重可以是任意正数，不需要归一化**
3. **适合已知各向量字段重要性的场景**
4. **权重顺序与检索请求顺序一致**
5. **通过实验找到最佳权重配置**

### 7.2 与 RRFRanker 的选择

- **使用 WeightedRanker**：
  - 有明确的业务经验或领域知识
  - 有标注数据可以进行离线评估
  - 需要精确控制各字段的重要性

- **使用 RRFRanker**：
  - 不确定各字段的重要性
  - 希望算法自动平衡
  - 快速原型开发

### 7.3 实战建议

1. **起点**：从均等权重（1.0, 1.0）开始
2. **调整**：基于评估指标小步迭代
3. **验证**：在测试集上验证效果
4. **监控**：生产环境中持续监控效果
5. **更新**：数据分布变化时重新调整

---

**参考资料**：
- Context7 官方文档：Milvus 多向量检索与混合检索
- Milvus 源码：tests/python_client/milvus_client_v2/test_milvus_client_hybrid_search_v2.py
