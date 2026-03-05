# Scenario 3: HNSW Production Examples from GitHub

## Search Results

### 1. Enhance MilvusVectorDBStorage with Parameterized Configuration Pull Request
**URL**: https://github.com/HKUDS/LightRAG/pull/2672
**Description**: 更新HNSW默认参数以匹配Milvus 2.4+，包括hnsw_m:16、hnsw_ef_construction:360和hnsw_ef:200，优化内存使用和搜索质量，提供生产环境示例。

### 2. Feature Request: Use IVF-PQ + HNSW in Milvus
**URL**: https://github.com/milvus-io/milvus/issues/40705
**Description**: 讨论Milvus中IVF-PQ与HNSW的混合使用，以减少内存消耗并提升速度，包含亿级规模相似性搜索的优化示例。

### 3. Question on HNSW Indexing Size in Milvus
**URL**: https://github.com/milvus-io/milvus/issues/2005
**Description**: 分析HNSW索引在10M向量数据集上的大小，使用M=16、efConstruction=500参数，评估生产环境中的内存和文件大小。

### 4. Configurable Milvus Vector Index Support Issue
**URL**: https://github.com/HKUDS/LightRAG/issues/2667
**Description**: 提出通过环境变量配置Milvus HNSW参数，如M=30、ef_construction=200，支持多种索引类型以优化生产性能。

### 5. EmergentDB: Self-Optimizing Vector Database
**URL**: https://github.com/justrach/emergentDB
**Description**: 使用MAP-Elites自动优化HNSW参数，实现99%召回率下的最大搜索速度，包含Milvus推荐参数的比较。

### 6. VectorLite: In-Process Vector Search
**URL**: https://github.com/1yefuwang1/vectorlite
**Description**: 提供HNSW参数调优示例和基准代码，用于改善生产环境中的召回率，支持完整控制参数。

### 7. VectorDBBench: Benchmark for Vector Databases
**URL**: https://github.com/zilliztech/VectorDBBench
**Description**: Milvus HNSW基准测试工具，支持参数如m、ef-construction、ef-search的优化和多种量化类型。

### 8. HyperspaceDB: High-Performance Hyperbolic Vector Database
**URL**: https://github.com/YARlabs/hyperspace-db
**Description**: 自定义HNSW实现，支持运行时调优ef_search和ef_construction参数，提升生产搜索质量和速度。

### 9. VelesDB: Vector + Graph + ColumnStore Fusion
**URL**: https://github.com/cyberlife-coder/VelesDB
**Description**: 本机HNSW实现，分析ef_search增加对延迟的影响，提供生产性能指标如57-102µs搜索时间。

## Key Production Parameters

### M (Max Connections per Layer)
- **Milvus 2.4+ Default**: 16
- **Production Range**: 16-64
- **Recommendations**:
  - M=16: Balanced memory/performance
  - M=30: Higher recall, more memory
  - M=64: Maximum recall, highest memory
- **Impact**: Higher M = better recall but more memory

### efConstruction (Build-time Search Depth)
- **Milvus 2.4+ Default**: 360
- **Production Range**: 200-500
- **Recommendations**:
  - efConstruction=200: Fast build, good quality
  - efConstruction=360: Balanced (recommended)
  - efConstruction=500: Best quality, slower build
- **Impact**: Higher efConstruction = better index quality but longer build time

### ef (Search-time Depth)
- **Milvus 2.4+ Default**: 200
- **Production Range**: 100-500
- **Recommendations**:
  - ef=100: Fast search, lower recall
  - ef=200: Balanced (recommended)
  - ef=500: High recall, slower search
- **Impact**: Higher ef = better recall but slower search

## Production Best Practices

1. **Memory Planning**: HNSW uses ~1.1x original vector size
2. **Parameter Tuning**: Start with defaults (M=16, efConstruction=360, ef=200)
3. **Scaling**: Add query nodes to boost QPS for distributed deployments
4. **Monitoring**: Track P99 latency, QPS, and memory usage
5. **Optimization**: Use MAP-Elites or grid search for parameter optimization
6. **Connection Pooling**: Essential for high-QPS production systems
7. **Runtime Tuning**: Adjust ef at search time without rebuild
