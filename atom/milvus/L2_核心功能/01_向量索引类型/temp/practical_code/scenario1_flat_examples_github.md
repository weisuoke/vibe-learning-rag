# Scenario 1: FLAT Index Examples from GitHub

## Search Results

### 1. FLAT | Milvus Documentation
**URL**: https://milvus.io/docs/flat.md
**Description**: Milvus FLAT索引官方文档，提供构建FLAT索引的详细方法和精确搜索示例，适用于需要100%召回率的场景，使用add_index()指定index_type为FLAT，无需额外参数。

### 2. In-memory Index | Milvus Documentation
**URL**: https://milvus.io/docs/index.md
**Description**: Milvus内存索引说明，详细介绍FLAT索引作为唯一保证精确搜索（exact search）的索引类型，适用于百万规模以下数据集的精确最近邻实现。

### 3. Index Explained | Milvus Documentation
**URL**: https://milvus.io/docs/index-explained.md
**Description**: Milvus索引原理解释，说明FLAT（Brute-Force）在高过滤率场景下用于最精确搜索结果的实现方式和适用条件。

### 4. milvus-io/milvus-lite GitHub Repository
**URL**: https://github.com/milvus-io/milvus-lite
**Description**: Milvus Lite轻量版仓库，描述FLAT索引的默认使用和支持情况，包括在小数据集上自动采用FLAT进行精确搜索的实现细节。

### 5. FAISS and Milvus Speed Benchmarking (Flat and HNSW)
**URL**: https://github.com/milvus-io/milvus/discussions/4939
**Description**: Milvus GitHub讨论线程，对比FAISS与Milvus FLAT索引在精确最近邻搜索的性能基准，包含Flat索引的exact NN实现表现。

### 6. How to Build Milvus Integration
**URL**: https://oneuptime.com/blog/post/2026-01-30-milvus-integration/view
**Description**: 2026年Milvus集成指南，包含FLAT索引作为精确搜索（Exact Search）的配置示例和与其他索引类型的对比。

### 7. BIN_FLAT | Milvus Documentation
**URL**: https://milvus.io/docs/bin-flat.md
**Description**: Milvus二进制向量FLAT索引文档，提供BIN_FLAT的构建和精确穷举搜索实现示例，与浮点FLAT类似但针对二进制向量。

## Key Insights

1. **100% Recall Guarantee**: FLAT is the only index type that guarantees exact search results
2. **Use Cases**: Best for datasets under 1M vectors or when absolute accuracy is required
3. **No Parameters**: FLAT index requires no tuning parameters
4. **Baseline for Comparison**: Often used as accuracy baseline for other approximate indexes
5. **Real-time Updates**: Supports real-time insertion and updates without rebuild
6. **High Filtering Scenarios**: Recommended when filter ratio > 98%
