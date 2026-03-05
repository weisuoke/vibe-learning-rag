# HNSW Milvus Integration Guide - OneUpTime 2026

Source: https://oneuptime.com/blog/post/2026-01-30-milvus-integration/view
Date: Jan 30, 2026
Fetched: 2026-02-21

## Key HNSW Configuration

### Index Parameters

```python
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {
        "M": 16,              # Maximum connections per node
        "efConstruction": 256  # Build-time search depth
    }
}
```

### Search Parameters

```python
search_params = {
    "params": {
        "ef": 64  # Query-time search depth
    }
}
```

## HNSW Architecture

HNSW builds a multi-layered graph structure:
- **Bottom layer**: Contains all data points
- **Upper layers**: Subset of data points sampled from lower layer
- Each layer has nodes (data points) connected by edges (proximity)
- Higher layers provide long-distance jumps for quick navigation
- Lower layers enable fine-grained search for accuracy

## Search Process

1. **Entry point**: Start at fixed entry point at top layer
2. **Greedy search**: Move to closest neighbor at current layer until local minimum
3. **Layer descend**: Jump down to lower layer using pre-established connection
4. **Final refinement**: Continue until bottom layer, identify nearest neighbors

## Key Parameters

### M (connections per node)
- Higher M = denser graph = better recall = more memory
- Default: 16
- Recommended range: [5, 100]

### efConstruction (build-time)
- Higher efConstruction = better quality graph = slower indexing
- Default: 256
- Recommended range: [50, 500]

### ef (query-time)
- Higher ef = more accurate results = slower queries
- Default: 64
- Adjustable at query time (unlike M and efConstruction)
- Recommended range: [K, 10K] where K is top-k results

## Production Best Practices

1. **Connection Pooling**: Reuse connections for better performance
2. **Error Handling**: Implement retry logic and graceful degradation
3. **Monitoring**: Track collection health, query latency, memory usage
4. **Index Selection**: Choose HNSW for high-recall, low-latency scenarios with sufficient memory
