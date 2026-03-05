# IVF_PQ Documentation - Milvus Official

Source: https://milvus.io/docs/ivf-pq.md
Fetched: 2026-02-21

## Overview

**IVF_PQ** stands for **Inverted File with Product Quantization**, a hybrid approach that combines indexing and compression for efficient vector search and retrieval.

### IVF Component
- Organizes data into clusters
- Enables search algorithm to focus only on most relevant subsets
- Uses k-means clustering to divide dataset into specified number of clusters
- Each cluster has a centroid (representative vector)

### PQ (Product Quantization) Component

**Key Stages:**

1. **Dimension decomposition**: Decompose each high-dimensional vector into m equal-sized sub-vectors
   - Transforms original D-dimensional space into m disjoint subspaces
   - Each subspace contains D/m dimensions
   - Parameter m controls granularity and compression ratio

2. **Subspace codebook generation**:
   - Apply k-means clustering within each subspace
   - Learn set of representative vectors (centroids)
   - Number of centroids = 2^nbits
   - Example: nbits=8 → 256 centroids per codebook

3. **Vector quantization**:
   - For each sub-vector, find nearest centroid in corresponding subspace
   - Store only the index of matched centroid (not full coordinates)

4. **Compressed representation**:
   - Final representation: m indices (PQ codes)
   - Storage reduction: D × 32 bits → m × nbits bits

### Compression Example

For D=128 dimensions, m=64, nbits=8:
- **Original**: 128 × 32 bits = 4,096 bits
- **PQ-compressed**: 64 × 8 bits = 512 bits
- **Compression ratio**: 8:1

### Distance Computation with PQ

1. **Query preprocessing**:
   - Decompose query vector into m sub-vectors
   - For each sub-vector, compute distances to all centroids in codebook
   - Generate m lookup tables (each with 2^nbits distances)

2. **Distance approximation**:
   - For any database vector (represented by PQ codes)
   - Retrieve pre-computed distance from lookup table using stored centroid index
   - Sum m distances to get approximate distance

## Build Index Parameters

| Parameter     | Description                                                                 | Value Range                              | Tuning Suggestion                                                                                   |
|---------------|-----------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|
| IVF nlist     | Number of clusters to create using k-means during index building            | Type: Integer, Range: [1, 65536], Default: 128 | Larger nlist improves recall but increases build time. Recommended range: [32, 4096]               |
| PQ m          | Number of sub-vectors to divide each vector into                            | Type: Integer, Range: [1, 65536], Default: None | Higher m improves accuracy but increases complexity. m must divide D. Recommended: m = D/2, range: [D/8, D] |
| nbits         | Number of bits for each sub-vector's centroid index                         | Type: Integer, Range: [1, 24], Default: 8 | Higher nbits = larger codebooks = better accuracy but less compression. Recommended range: [1, 16] |

## Search Parameters

| Parameter     | Description                                                                 | Value Range                              | Tuning Suggestion                                                                                   |
|---------------|-----------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|
| IVF nprobe    | Number of clusters to search for candidates                                 | Type: Integer, Range: [1, nlist], Default: 8 | Higher nprobe improves recall but increases latency. Set proportionally to nlist. Recommended range: [1, nlist] |

## Python Code Example

```python
from pymilvus import MilvusClient

# Prepare index building params
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="your_vector_field_name",
    index_type="IVF_PQ",
    index_name="vector_index",
    metric_type="L2",
    params={
        "nlist": 128,  # Number of clusters
        "m": 4,        # Number of sub-vectors
        "nbits": 8     # Bits per sub-vector index
    }
)

# Search parameters
search_params = {
    "params": {
        "nprobe": 10  # Number of clusters to search
    }
}
```
