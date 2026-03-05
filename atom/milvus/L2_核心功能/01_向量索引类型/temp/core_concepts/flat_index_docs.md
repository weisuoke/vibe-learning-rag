# FLAT Index Documentation

## Official Milvus Documentation

### FLAT Index
- **URL**: https://milvus.io/docs/flat.md
- **Description**: Milvus FLAT索引文档，介绍用于小规模数据集的精确暴力搜索索引，无需构建参数，支持100%召回率，适用于需要完美准确性的场景。

### In-memory Index Overview
- **URL**: https://milvus.io/docs/index.md
- **Description**: Milvus内存索引概述，包括FLAT索引详情：无需额外参数、无需构建、保证精确搜索结果，适合百万级数据集的精确向量相似性搜索。

## Key Characteristics

### Exact Search
- **100% recall rate**: Guarantees finding the true nearest neighbors
- **Brute-force algorithm**: Compares query vector with every vector in dataset
- **No approximation**: No accuracy loss, perfect precision
- **No index building**: No preprocessing required

### Performance Characteristics
- **Time complexity**: O(n) - linear search through all vectors
- **Space complexity**: O(n*d) - stores all original vectors
- **Query latency**: Increases linearly with dataset size
- **Best for**: Small datasets (< 1M vectors)

## Technical Details

### Algorithm
1. **No preprocessing**: Vectors stored as-is without transformation
2. **Exhaustive search**: Every query compares with all vectors
3. **Distance calculation**: Computes distance to every vector
4. **Sorting**: Returns top-k results by distance

### Supported Metrics
- **L2 (Euclidean Distance)**: Default metric
- **IP (Inner Product)**: For normalized vectors
- **COSINE**: Cosine similarity

## Configuration

### Index Parameters
```python
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",  # or "IP", "COSINE"
    "params": {}  # No additional parameters needed
}
```

### No Build Parameters
- FLAT index requires **no build parameters**
- No training or preprocessing phase
- Immediate availability after data insertion

## Use Cases

### Ideal Scenarios
1. **Small datasets**: < 1M vectors
2. **High accuracy requirements**: Need 100% recall
3. **Baseline comparison**: Benchmark for other indexes
4. **Development/testing**: Quick prototyping without index tuning

### Not Recommended For
1. **Large datasets**: > 1M vectors (too slow)
2. **High QPS requirements**: Linear search doesn't scale
3. **Production systems**: Better alternatives available (HNSW, IVF)
4. **Real-time search**: Latency increases with data size

## Performance Benchmarks

### Query Latency
- **10K vectors**: ~1-5ms
- **100K vectors**: ~10-50ms
- **1M vectors**: ~100-500ms
- **10M vectors**: ~1-5s (not recommended)

### Memory Usage
- **Full precision**: Stores all vectors in original format
- **No compression**: No memory savings
- **Memory = dataset size**: n_vectors * dimension * 4 bytes (FP32)

## Comparison with Other Indexes

### vs IVF_FLAT
- **Accuracy**: FLAT 100%, IVF_FLAT ~95-99%
- **Speed**: IVF_FLAT 10-100x faster for large datasets
- **Memory**: Similar memory usage
- **Use case**: FLAT for small/accurate, IVF_FLAT for large/fast

### vs HNSW
- **Accuracy**: FLAT 100%, HNSW ~95-99%
- **Speed**: HNSW 100-1000x faster for large datasets
- **Memory**: HNSW uses more memory (graph structure)
- **Use case**: FLAT for baseline, HNSW for production

### vs GPU_CAGRA
- **Accuracy**: FLAT 100%, GPU_CAGRA ~95-99%
- **Speed**: GPU_CAGRA 1000x+ faster with GPU
- **Hardware**: FLAT CPU-only, GPU_CAGRA needs GPU
- **Use case**: FLAT for simplicity, GPU_CAGRA for scale

## Best Practices

### When to Use FLAT
1. **Accuracy validation**: Verify other indexes' recall rates
2. **Small datasets**: < 100K vectors where speed is acceptable
3. **Development phase**: Quick prototyping without tuning
4. **Baseline metrics**: Establish performance benchmarks

### Optimization Tips
1. **Batch queries**: Process multiple queries together
2. **Reduce dimensions**: Use PCA/dimensionality reduction
3. **Filter early**: Apply scalar filters before vector search
4. **Consider alternatives**: Switch to IVF/HNSW for larger datasets

## Code Example

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="FLAT index demo")

# Create collection
collection = Collection("flat_demo", schema)

# Create FLAT index
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",
    "params": {}
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)

# Insert data
import numpy as np
vectors = np.random.rand(10000, 128).tolist()
collection.insert([vectors])

# Load collection
collection.load()

# Search
query_vectors = np.random.rand(1, 128).tolist()
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2"},
    limit=10
)

print(f"Search results: {results}")
```

## Common Questions

### Q: Why use FLAT if it's so slow?
**A**: FLAT provides 100% accuracy baseline for validating other indexes. It's also simple and works well for small datasets.

### Q: Can FLAT be faster than approximate indexes?
**A**: Yes, for very small datasets (< 10K vectors), FLAT can be faster due to no index overhead.

### Q: Does FLAT support updates?
**A**: Yes, FLAT supports real-time updates without index rebuilding.

### Q: What's the maximum dataset size for FLAT?
**A**: Technically unlimited, but practically < 1M vectors for reasonable latency.

## References

1. Milvus FLAT Documentation: https://milvus.io/docs/flat.md
2. In-memory Index Overview: https://milvus.io/docs/index.md
3. Milvus GitHub: https://github.com/milvus-io/milvus
4. Release Notes: https://milvus.io/docs/release_notes.md
