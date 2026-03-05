# HNSW Official Documentation - Milvus v2.5.x

Source: https://milvus.io/docs/v2.5.x/hnsw.md
Fetched: 2026-02-21

## Overview

HNSW (Hierarchical Navigable Small World) is a **graph-based** indexing algorithm that offers:
- **Excellent** search accuracy
- **Low** latency
- **High** memory overhead (maintains hierarchical graph structure)

## How HNSW Works

### Multi-Layer Graph Structure

- **Bottom layer**: Contains all data points
- **Upper layers**: Subset of data points sampled from lower layer
- Each layer has nodes (data points) connected by edges (proximity)

### Search Process

1. **Entry point**: Start at fixed entry point at top layer
2. **Greedy search**: Move to closest neighbor until local minimum
3. **Layer descend**: Jump down to lower layer via pre-established connection
4. **Final refinement**: Continue until bottom layer, identify nearest neighbors

## Key Parameters

### M (Maximum connections per node)

- **Description**: Maximum number of edges/connections each node can have at each level
- **Type**: Integer
- **Range**: [2, 2048]
- **Default**: 30 (up to 30 outgoing and 30 incoming edges per node)
- **Effect**:
  - Larger M = higher accuracy + more memory + slower build/search
  - Smaller M = lower memory + faster build but lower accuracy
- **Recommended range**: [5, 100]

### efConstruction (Build-time search depth)

- **Description**: Number of candidate neighbors considered during index construction
- **Type**: Integer
- **Range**: [1, int_max]
- **Default**: 360
- **Effect**:
  - Higher efConstruction = more accurate index + slower build
  - Lower efConstruction = faster build but lower quality graph
- **Recommended range**: [50, 500]

### ef (Query-time search depth)

- **Description**: Number of neighbors evaluated during search (controls breadth of search)
- **Type**: Integer
- **Range**: [1, int_max]
- **Default**: limit (TopK nearest neighbors to return)
- **Effect**:
  - Larger ef = higher search accuracy + slower queries
  - Smaller ef = faster queries but lower recall
- **Recommended range**: [K, 10K] where K is the limit parameter

## Build Index Example

```python
from pymilvus import MilvusClient

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="your_vector_field_name",
    index_type="HNSW",
    index_name="vector_index",
    metric_type="L2",
    params={
        "M": 64,              # Maximum connections per node
        "efConstruction": 100  # Build-time search depth
    }
)
```

## Search Example

```python
search_params = {
    "params": {
        "ef": 10  # Query-time search depth
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

## Parameter Tuning Guidelines

### For High Accuracy (95%+ recall)
- M: 32-64
- efConstruction: 200-400
- ef: 64-128

### For Balanced Performance (90%+ recall)
- M: 16-32
- efConstruction: 100-200
- ef: 32-64

### For Fast Queries (85%+ recall)
- M: 8-16
- efConstruction: 50-100
- ef: 16-32

## Memory Estimation

Memory usage ≈ N × M × 2 × sizeof(int) + N × D × sizeof(float)

Where:
- N = number of vectors
- M = connections per node
- D = vector dimension

Example for 1M vectors, 384 dims, M=32:
- Graph structure: 1M × 32 × 2 × 4 bytes = 256 MB
- Vector data: 1M × 384 × 4 bytes = 1.5 GB
- **Total**: ~1.75 GB
