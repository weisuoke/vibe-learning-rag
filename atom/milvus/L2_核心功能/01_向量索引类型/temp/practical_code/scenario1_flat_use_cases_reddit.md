# Scenario 1: FLAT Index Use Cases from Reddit

## Search Results

### 1. FLAT | Milvus Documentation
**URL**: https://milvus.io/docs/flat.md
**Description**: FLAT索引是Milvus中最简单的浮点向量索引类型，进行全扫描实现精确最近邻搜索（exact search），保证100% recall，但查询速度较慢，适合小规模数据集或需要绝对准确性的场景。

### 2. Accelerating Similarity Search on Really Big Data with Vector Indexing
**URL**: https://milvus.io/blog/2019-12-05-Accelerating-Similarity-Search-on-Really-Big-Data-with-Vector-Indexing.md
**Description**: FLAT索引适用于百万规模数据集且要求100% recall的精确搜索场景，是唯一保证exact search结果的索引类型，可作为其他近似索引的准确性对比基准。

### 3. In terms of index build time and update flexibility
**URL**: https://milvus.io/ai-quick-reference/in-terms-of-index-build-time-and-update-flexibility-how-do-different-indexing-structures-eg-flat-ivf-hnsw-annoy-compare-with-each-other
**Description**: FLAT索引构建时间最短，支持实时更新和插入，适合数据频繁变更的动态数据集场景，尽管查询时需全扫描导致速度较慢。

### 4. Index Explained | Milvus Documentation
**URL**: https://milvus.io/docs/index-explained.md
**Description**: 当过滤比率超过98%或需要极高召回率（>99%）时，推荐使用Brute-Force (FLAT)索引以获得最准确的搜索结果，尤其适合高过滤比率的精确搜索。

### 5. Which vector database is best for top-1 accuracy?
**URL**: https://www.reddit.com/r/vectordatabase/comments/1ncee4l/which_vector_database_is_best_for_top1_accuracy
**Description**: 若需要100%准确性（top-1 exact match），必须使用brute force search（如Milvus FLAT索引），否则近似索引如HNSW可能丢失关键结果，适用于避免重复的关键场景。

## Key Use Cases

1. **Small-scale Document QA**: < 100K documents requiring exact matches
2. **Deduplication Systems**: Avoiding duplicate entries in critical systems
3. **Accuracy Baseline**: Benchmarking other approximate indexes
4. **High Filter Ratio**: When filtering eliminates > 98% of data
5. **Dynamic Data**: Frequent insertions/updates without rebuild overhead
6. **Critical Applications**: Where missing a result is unacceptable
