# Scenario 6: Index Selection Guide from GitHub

## Search Results

### 1. Milvus In-memory Index Documentation
**URL**: https://milvus.io/docs/index.md
**Description**: Milvus官方文档详细列出支持的向量索引类型，包括FLAT、IVF_FLAT、IVF_PQ、HNSW、SCANN、DiskANN等，并说明适用场景和参数配置。

### 2. Index Explained | Milvus Documentation
**URL**: https://milvus.io/docs/index-explained.md
**Description**: Milvus索引原理解释，包括PQ与SQ比较、决策矩阵，推荐不同场景下的索引类型如HNSW用于高召回、DiskANN用于磁盘数据。

### 3. How to Choose Between IVF and HNSW for ANN Vector Search
**URL**: https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md
**Description**: Milvus博客比较IVF与HNSW，IVF构建更快、内存更低，适合平衡速度与准确；HNSW查询更快但内存消耗高。

### 4. Milvus Index Selection Guide in OneUptime Integration
**URL**: https://oneuptime.com/blog/post/2026-01-30-milvus-integration/view
**Description**: Milvus索引选择指南表格，比较FLAT、IVF_FLAT、IVF_SQ8、HNSW在内存、速度、准确率上的差异，推荐生产环境用HNSW。

### 5. Milvus GitHub Repository Main Page
**URL**: https://github.com/milvus-io/milvus
**Description**: Milvus官方GitHub仓库，支持多种向量索引类型如HNSW、IVF、DiskANN、SCANN，并强调硬件加速和基准测试。

### 6. Introducing Milvus Sizing Tool with Index Selection
**URL**: https://milvus.io/blog/introducing-the-milvus-sizing-tool-calculating-and-optimizing-your-milvus-deployment-resources.md
**Description**: Milvus资源计算工具介绍，包括HNSW、FLAT、IVF_FLAT、ScaNN、DiskANN等索引的存储、成本、速度、准确权衡。

### 7. Vector Databases in 2025: Top 10 Index Choices Benchmarked
**URL**: https://medium.com/@ThinkingLoop/d3-4-vector-databases-in-2025-top-10-index-choices-benchmarked-1bbce68e1871
**Description**: 2025年向量索引基准比较，包括Milvus支持的HNSW、IVF、DiskANN等，针对速度、准确率、规模的框架选择指南。

## Index Selection Framework

### Decision Factors
1. **Dataset Size**: < 1M, 1M-10M, 10M-100M, > 100M
2. **Recall Requirements**: Exact (100%), High (>99%), Medium (95-99%), Low (<95%)
3. **Query Latency**: < 10ms, 10-50ms, 50-100ms, > 100ms
4. **Memory Budget**: Unlimited, High, Medium, Low
5. **Build Time**: Real-time, Minutes, Hours, Days
6. **Hardware**: CPU-only, GPU available, Disk-based

### Index Selection Matrix

| Dataset Size | Recall | Latency | Memory | Recommended Index |
|--------------|--------|---------|--------|-------------------|
| < 1M | 100% | Any | Any | FLAT |
| 1M-10M | >99% | <50ms | High | HNSW |
| 1M-10M | 95-99% | <50ms | Medium | IVF_FLAT |
| 10M-100M | >99% | <50ms | High | HNSW |
| 10M-100M | 95-99% | <50ms | Low | IVF_SQ8 |
| > 100M | >99% | <50ms | GPU | GPU_CAGRA |
| > 100M | 95-99% | <100ms | Low | IVF_PQ |
| > 1B | 90%+ | <100ms | Very Low | RaBitQ |
| > 100M | >99% | Any | Disk | DiskANN |

### Index Comparison

| Index | Memory | Speed | Recall | Build Time | Use Case |
|-------|--------|-------|--------|------------|----------|
| FLAT | 100% | Slow | 100% | Instant | Small datasets, exact search |
| IVF_FLAT | 100% | Medium | 95-99% | Fast | Medium datasets, balanced |
| IVF_SQ8 | 25% | Medium | 93-97% | Fast | Large datasets, memory-constrained |
| IVF_PQ | 25% | Medium | 90-95% | Fast | Very large datasets, low memory |
| HNSW | 110% | Fast | 99%+ | Medium | Production, high QPS |
| SCANN | 30% | Fast | 95-99% | Medium | Large datasets, balanced |
| GPU_CAGRA | 110% | Very Fast | 99%+ | Fast | GPU available, high throughput |
| RaBitQ | 3-4% | Medium | 90%+ | Fast | Billion-scale, cost-optimized |
| DiskANN | Disk | Medium | 95-99% | Slow | Massive datasets, disk-based |

### Migration Guide
1. **Start with FLAT**: Baseline for accuracy
2. **Benchmark IVF_FLAT**: Test with your data
3. **Try HNSW**: If memory allows, best for production
4. **Optimize with quantization**: IVF_SQ8/IVF_PQ for memory savings
5. **Consider GPU**: GPU_CAGRA for high throughput
6. **Extreme scale**: RaBitQ for billion-scale cost optimization
