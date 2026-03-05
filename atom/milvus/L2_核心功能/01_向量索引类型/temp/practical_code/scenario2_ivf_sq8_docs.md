# IVF_SQ8 Documentation - Milvus Official

Source: https://milvus.io/docs/ivf-sq8.md
Fetched: 2026-02-21

## Overview

**IVF_SQ8** is a **quantization-based** indexing algorithm designed for large-scale similarity search. It achieves faster searches with much smaller memory footprint compared to exhaustive search methods.

### Two Key Components

**Inverted File (IVF)**: Organizes data into clusters, enabling search to focus only on most relevant subsets

**Scalar Quantization (SQ8)**: Compresses vectors to compact form, drastically reducing memory while maintaining precision for fast similarity calculations

## SQ8 (Scalar Quantization) Process

Scalar Quantization reduces vector size by replacing values with smaller, more compact representations. **SQ8** uses 8-bit integers instead of 32-bit floating point numbers.

### How SQ8 Works

1. **Range Identification**: Identify minimum and maximum values within the vector

2. **Normalization**: Normalize vector values to range [0, 1]:
   ```
   normalized_value = (value - min) / (max - min)
   ```

3. **8-Bit Compression**: Multiply normalized value by 255 and round to nearest integer
   - Compresses each value into 8-bit representation

### Compression Example

For a dimension value of 1.2, with min=-1.7 and max=2.3:

```
normalized = (1.2 - (-1.7)) / (2.3 - (-1.7)) = 2.9 / 4.0 = 0.725
quantized = round(0.725 × 255) = round(184.875) = 185
```

**Storage savings:**
- Original float32: 32 bits per dimension
- After SQ8: 8 bits per dimension
- **Compression ratio: 4:1**

## IVF + SQ8 Combined

1. **IVF narrows search scope**: Dataset divided into clusters, query compares to cluster centroids, selects most relevant clusters

2. **SQ8 speeds up distance calculations**: Within selected clusters, SQ8 compresses vectors into 8-bit integers, reducing memory and accelerating distance computations

## Build Index Parameters

| Parameter     | Description                                                                 | Value Range                              | Tuning Suggestion                                                                                   |
|---------------|-----------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|
| IVF nlist     | Number of clusters to create using k-means during index building            | Type: Integer, Range: [1, 65536], Default: 128 | Larger nlist improves recall but increases build time. Recommended range: [32, 4096]               |

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
    index_name="vector_index",
    index_type="IVF_SQ8",
    metric_type="L2",
    params={
        "nlist": 64  # Number of clusters
    }
)

# Search parameters
search_params = {
    "params": {
        "nprobe": 8  # Number of clusters to search
    }
}

res = MilvusClient.search(
    collection_name="your_collection_name",
    anns_field="vector_field",
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],
    limit=10,
    search_params=search_params
)
```

## Key Advantages

- **Memory efficiency**: 4:1 compression ratio
- **Fast queries**: Compressed vectors enable faster distance calculations
- **Good balance**: Maintains high recall (90%+) while reducing memory usage
- **Suitable for**: Large-scale search with memory constraints
