# IVF Series Index Documentation

## IVF_FLAT Index

### Official Documentation
- **URL**: https://milvus.io/docs/ivf-flat.md
- **Description**: Milvus IVF_FLAT索引文档，适用于浮点向量的大规模数据集，提供高召回率和高性能查询，支持nlist参数调优

### Key Features
- **Inverted file structure**: Clusters vectors using k-means
- **Flat compression**: Stores original vectors without quantization
- **High recall**: 95-99% recall with proper tuning
- **Balanced performance**: Good trade-off between speed and accuracy

### Algorithm
1. **Training phase**: Cluster vectors into nlist groups using k-means
2. **Indexing phase**: Assign each vector to nearest cluster
3. **Search phase**: Search nprobe nearest clusters, then find top-k within

### Parameters
- **nlist**: Number of clusters (build parameter)
  - Range: 1 to 65536
  - Recommended: sqrt(n) to 4*sqrt(n)
  - Example: 1024 for 1M vectors

- **nprobe**: Number of clusters to search (query parameter)
  - Range: 1 to nlist
  - Recommended: 8 to 128
  - Trade-off: Higher nprobe = better recall, slower search

### Performance
- **Build time**: O(n*d*nlist*iterations)
- **Query time**: O(nprobe * n/nlist * d)
- **Memory**: Same as FLAT (stores original vectors)
- **Recall**: 95-99% with nprobe=16-64

---

## IVF_SQ8 Index

### Official Documentation
- **URL**: https://milvus.io/docs/ivf-sq8.md
- **Description**: Milvus IVF_SQ8索引文档，基于标量量化（SQ8）的量化索引，适用于内存受限场景，提供极高查询速度并接受少量召回率折衷

### Key Features
- **Scalar quantization**: Quantizes FP32 to 8-bit integers
- **Memory efficient**: 4x memory reduction (32-bit → 8-bit)
- **Fast search**: Faster than IVF_FLAT due to smaller data
- **Slight accuracy loss**: 1-2% recall loss compared to IVF_FLAT

### Algorithm
1. **Training phase**: Same as IVF_FLAT (k-means clustering)
2. **Quantization**: Convert FP32 vectors to 8-bit integers
3. **Indexing**: Store quantized vectors in clusters
4. **Search**: Use quantized vectors for fast distance computation

### Parameters
- **nlist**: Same as IVF_FLAT (number of clusters)
- **nprobe**: Same as IVF_FLAT (clusters to search)
- **nbits**: Fixed at 8 bits (SQ8)

### Performance
- **Memory**: 4x reduction compared to IVF_FLAT
- **Speed**: 1.5-2x faster than IVF_FLAT
- **Recall**: 93-98% (1-2% loss vs IVF_FLAT)
- **Best for**: Memory-constrained environments

### Quantization Details
```
Original: FP32 vector [0.123, -0.456, 0.789, ...]
Quantized: INT8 vector [31, -116, 201, ...]

Quantization formula:
quantized_value = round((value - min) / (max - min) * 255)
```

---

## IVF_PQ Index

### Official Documentation
- **URL**: https://milvus.io/docs/ivf-pq.md
- **Description**: Milvus IVF_PQ索引文档，基于乘积量化（Product Quantization）的索引算法，适用于高维空间的近似最近邻搜索，平衡速度与内存使用

### Key Features
- **Product quantization**: Splits vectors into subvectors, quantizes separately
- **Highest compression**: 8-32x memory reduction
- **Fast search**: Asymmetric distance computation
- **More accuracy loss**: 5-10% recall loss compared to IVF_FLAT

### Algorithm
1. **Training phase**: K-means clustering + PQ codebook training
2. **Vector splitting**: Split d-dimensional vector into m subvectors
3. **Subvector quantization**: Quantize each subvector to nbits
4. **Indexing**: Store quantized codes in clusters
5. **Search**: Asymmetric distance computation using lookup tables

### Parameters
- **nlist**: Number of clusters (same as IVF_FLAT)
- **m**: Number of subquantizers
  - Must divide dimension evenly
  - Recommended: 8, 16, 32
  - Example: m=16 for 128-dim vectors (8-dim subvectors)

- **nbits**: Bits per subquantizer
  - Range: 1 to 16
  - Common: 8 (256 centroids per subvector)
  - Trade-off: Higher nbits = better accuracy, more memory

### Performance
- **Memory**: 8-32x reduction (depends on m and nbits)
- **Speed**: 2-3x faster than IVF_FLAT
- **Recall**: 85-95% (5-10% loss vs IVF_FLAT)
- **Best for**: Very large datasets where memory is critical

### Compression Example
```
Original: 128-dim FP32 vector = 512 bytes
IVF_PQ (m=16, nbits=8):
  - 16 subvectors × 1 byte = 16 bytes
  - Compression ratio: 32x
```

---

## Comparison Table

| Index Type | Memory | Speed | Recall | Best For |
|-----------|--------|-------|--------|----------|
| IVF_FLAT | 1x (baseline) | 1x (baseline) | 95-99% | Balanced performance |
| IVF_SQ8 | 0.25x (4x reduction) | 1.5-2x faster | 93-98% | Memory-constrained |
| IVF_PQ | 0.03-0.125x (8-32x reduction) | 2-3x faster | 85-95% | Very large datasets |

---

## Parameter Tuning Guide

### nlist Selection
```
Small dataset (< 1M): nlist = 1024
Medium dataset (1M-10M): nlist = 4096
Large dataset (> 10M): nlist = 16384

Formula: nlist ≈ sqrt(n) to 4*sqrt(n)
```

### nprobe Selection
```
High accuracy: nprobe = 64-128
Balanced: nprobe = 16-32
High speed: nprobe = 4-8

Trade-off: nprobe/nlist ratio determines recall
```

### IVF_PQ m Selection
```
High dimension (> 512): m = 32-64
Medium dimension (128-512): m = 16-32
Low dimension (< 128): m = 8-16

Constraint: dimension % m == 0
```

---

## Use Case Recommendations

### IVF_FLAT
- **Scenario**: General-purpose vector search
- **Dataset size**: 1M-100M vectors
- **Memory**: Sufficient memory available
- **Accuracy**: Need high recall (95-99%)
- **Example**: RAG systems with moderate scale

### IVF_SQ8
- **Scenario**: Memory-constrained environments
- **Dataset size**: 1M-100M vectors
- **Memory**: Limited memory budget
- **Accuracy**: Can accept 1-2% recall loss
- **Example**: Edge devices, cost-sensitive deployments

### IVF_PQ
- **Scenario**: Billion-scale vector search
- **Dataset size**: 100M+ vectors
- **Memory**: Very limited memory budget
- **Accuracy**: Can accept 5-10% recall loss
- **Example**: Large-scale recommendation systems

---

## Code Examples

### IVF_FLAT
```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}

search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
```

### IVF_SQ8
```python
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}

search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
```

### IVF_PQ
```python
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {
        "nlist": 1024,
        "m": 16,  # 128-dim / 16 = 8-dim subvectors
        "nbits": 8  # 256 centroids per subvector
    }
}

search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
```

---

## References

1. IVF_FLAT Documentation: https://milvus.io/docs/ivf-flat.md
2. IVF_SQ8 Documentation: https://milvus.io/docs/ivf-sq8.md
3. IVF_PQ Documentation: https://milvus.io/docs/ivf-pq.md
4. In-memory Index Overview: https://milvus.io/docs/index.md
5. Index Explained: https://milvus.io/docs/index-explained.md
