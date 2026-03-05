# HNSW Index Documentation

## Official Milvus Documentation

### HNSW Index (Inferred from search results)
- **Description**: Hierarchical Navigable Small World graph-based index, the most popular choice for production RAG systems
- **Key advantage**: High accuracy (95-99% recall) with fast search speed

## Key Features

### Graph-Based Structure
- **Hierarchical layers**: Multi-layer graph structure for efficient search
- **Navigable small world**: Short paths between any two nodes
- **Greedy search**: Fast approximate nearest neighbor search
- **High accuracy**: Maintains 95-99% recall with proper tuning

### Performance Characteristics
- **Build time**: Slower than IVF (but one-time cost)
- **Query speed**: Very fast, especially for high-dimensional data
- **Memory usage**: Higher than IVF (stores graph structure)
- **Scalability**: Excellent for production systems

## Algorithm Overview

### Multi-Layer Graph
1. **Layer 0 (base)**: Contains all vectors
2. **Layer 1+**: Progressively sparser layers
3. **Entry point**: Start search from top layer
4. **Greedy descent**: Navigate down layers to find nearest neighbors

### Search Process
1. Start at entry point in top layer
2. Greedily navigate to nearest neighbor in current layer
3. Descend to next layer when local minimum reached
4. Repeat until reaching base layer
5. Return top-k nearest neighbors

## Parameters

### Build Parameters

#### M (max connections per node)
- **Range**: 4 to 64
- **Default**: 16
- **Recommended**: 16-32 for most cases
- **Trade-off**:
  - Higher M = better recall, more memory, slower build
  - Lower M = less memory, faster build, lower recall
- **Formula**: Memory ≈ M * n * 8 bytes per connection

#### efConstruction (build quality)
- **Range**: 8 to 512
- **Default**: 200
- **Recommended**: 200-400 for production
- **Trade-off**:
  - Higher efConstruction = better graph quality, slower build
  - Lower efConstruction = faster build, lower quality
- **Impact**: Affects final index quality, not query speed

### Search Parameters

#### ef (search quality)
- **Range**: top_k to 32768
- **Default**: 100
- **Recommended**: 100-200 for balanced performance
- **Trade-off**:
  - Higher ef = better recall, slower search
  - Lower ef = faster search, lower recall
- **Constraint**: ef >= top_k (number of results requested)

## Performance Benchmarks

### Query Latency
- **1M vectors**: ~1-5ms (ef=100)
- **10M vectors**: ~2-10ms (ef=100)
- **100M vectors**: ~5-20ms (ef=100)
- **Scalability**: Sub-linear growth with dataset size

### Memory Usage
```
Memory = n * (d * 4 + M * 2 * 8) bytes

Example (1M vectors, 128-dim, M=16):
= 1M * (128 * 4 + 16 * 2 * 8)
= 1M * (512 + 256)
= 768 MB
```

### Recall vs Speed Trade-off
- **ef=50**: ~92% recall, fastest
- **ef=100**: ~96% recall, balanced
- **ef=200**: ~98% recall, slower
- **ef=400**: ~99% recall, slowest

## Use Cases

### Ideal Scenarios
1. **Production RAG systems**: High QPS, high accuracy requirements
2. **Real-time search**: Low latency requirements (< 10ms)
3. **High-dimensional data**: 128-1536 dimensions
4. **Stable datasets**: Infrequent updates (index rebuild is expensive)
5. **Sufficient memory**: Can afford graph structure overhead

### Not Recommended For
1. **Memory-constrained**: Use IVF_SQ8 or IVF_PQ instead
2. **Frequent updates**: Index rebuild is expensive
3. **Very large scale**: > 100M vectors (consider GPU_CAGRA or RaBitQ)
4. **Cost-sensitive**: Higher memory cost than IVF

## Comparison with Other Indexes

### vs IVF_FLAT
- **Accuracy**: HNSW slightly better (98% vs 96%)
- **Speed**: HNSW much faster for large datasets
- **Memory**: HNSW uses more memory (graph overhead)
- **Updates**: IVF_FLAT better for frequent updates
- **Recommendation**: HNSW for production, IVF_FLAT for development

### vs FLAT
- **Accuracy**: FLAT 100%, HNSW 95-99%
- **Speed**: HNSW 100-1000x faster
- **Memory**: HNSW uses more memory
- **Use case**: HNSW for production, FLAT for baseline

### vs GPU_CAGRA
- **Speed**: GPU_CAGRA 10x faster (with GPU)
- **Hardware**: HNSW CPU-only, GPU_CAGRA needs GPU
- **Cost**: HNSW cheaper (no GPU required)
- **Scalability**: GPU_CAGRA better for > 10M vectors

## Parameter Tuning Guide

### M Selection
```python
# Conservative (lower memory, lower recall)
M = 8

# Balanced (recommended for most cases)
M = 16

# High quality (higher memory, higher recall)
M = 32

# Maximum quality (production systems with ample memory)
M = 64
```

### efConstruction Selection
```python
# Fast build (development/testing)
efConstruction = 100

# Balanced (recommended)
efConstruction = 200

# High quality (production)
efConstruction = 400

# Maximum quality (critical applications)
efConstruction = 500
```

### ef Selection (Query Time)
```python
# Fast search (90-95% recall)
ef = 50

# Balanced (95-97% recall)
ef = 100

# High accuracy (97-99% recall)
ef = 200

# Maximum accuracy (99%+ recall)
ef = 400
```

## Best Practices

### Build Phase
1. **One-time cost**: Build index once, use many times
2. **Sufficient efConstruction**: Use 200-400 for production
3. **Monitor memory**: Ensure sufficient RAM for graph
4. **Parallel build**: Milvus supports parallel index building

### Query Phase
1. **Tune ef dynamically**: Adjust based on accuracy requirements
2. **Connection pooling**: Reuse connections for high QPS
3. **Batch queries**: Process multiple queries together
4. **Monitor latency**: Track p50, p95, p99 latencies

### Production Deployment
1. **Memory planning**: Allocate 1.5-2x dataset size for graph
2. **Index persistence**: Save index to disk for fast restart
3. **Monitoring**: Track recall, latency, memory usage
4. **Gradual rollout**: Test with small traffic before full deployment

## Common Issues and Solutions

### Issue: High memory usage
**Solution**: Reduce M parameter or switch to IVF_SQ8/IVF_PQ

### Issue: Low recall
**Solution**: Increase ef parameter or rebuild with higher efConstruction

### Issue: Slow queries
**Solution**: Decrease ef parameter or add more CPU cores

### Issue: Slow index building
**Solution**: Decrease efConstruction or use more CPU cores

## Code Example

```python
from pymilvus import Collection, connections

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Create collection
collection = Collection("hnsw_demo")

# Build HNSW index
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,              # Max connections per node
        "efConstruction": 200  # Build quality
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)

# Load collection
collection.load()

# Search with tuned ef
search_params = {
    "metric_type": "L2",
    "params": {
        "ef": 100  # Search quality
    }
}

results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param=search_params,
    limit=10
)
```

## GitHub Discussions

### HNSW Parameter Tuning Issues
- **URL**: https://github.com/milvus-io/milvus/issues/46061
- **Issue**: HNSW + IP + range_search recall cliffs when increasing M
- **Insight**: Parameter tuning can be counter-intuitive in some scenarios

### HNSW Configuration in LightRAG
- **URL**: https://github.com/HKUDS/LightRAG/pull/2670
- **Configuration**: M=16, efConstruction=360, ef=200 (Milvus 2.4+ defaults)
- **Validation**: HNSW_SQ requires Milvus 2.6.8+

## References

1. Milvus In-memory Index: https://milvus.io/docs/index.md
2. HNSW Parameter Tuning: https://github.com/HKUDS/LightRAG/issues/2667
3. HNSW Bug Report: https://github.com/milvus-io/milvus/issues/46061
4. Milvus GitHub: https://github.com/milvus-io/milvus
5. HNSW vs IVF Comparison: https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md
