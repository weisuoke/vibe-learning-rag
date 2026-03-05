# GPU_CAGRA Official Documentation - Milvus

Source: https://milvus.io/docs/gpu-cagra.md
Fetched: 2026-02-21

## Overview

**GPU_CAGRA** is a graph-based index optimized for GPUs. Using inference-grade GPUs to run Milvus GPU version can be more cost-effective compared to using expensive training-grade GPUs.

## Key Features

- Graph-based index optimized for GPU
- Cost-effective with inference-grade GPUs
- Hybrid mode: GPU for index building, CPU for search (optional)

## Build Index Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| **intermediate_graph_degree** | Affects recall and build time by determining graph's degree before pruning | 128 | 32 or 64 |
| **graph_degree** | Affects search performance and recall by setting graph's degree after pruning. Must be smaller than intermediate_graph_degree | 64 | - |
| **build_algo** | Graph generation algorithm before pruning. Options: IVF_PQ (higher quality, slower) or NN_DESCENT (quicker, potentially lower recall) | IVF_PQ | - |
| **cache_dataset_on_device** | Whether to cache original dataset in GPU memory. "true" enhances recall by refining search results, "false" saves GPU memory | "false" | - |
| **adapt_for_cpu** | Use GPU for index-building and CPU for search. Requires ef parameter in search requests | "false" | - |

## Search Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **itopk_size** | Size of intermediate results kept during search. Larger value may improve recall at expense of performance. Should be at least equal to final top-k, typically power of 2 (16, 32, 64, 128) | Empty |
| **search_width** | Number of entry points into CAGRA graph during search. Increasing enhances recall but may impact performance (1, 2, 4, 8, 16, 32) | Empty |
| **min_iterations / max_iterations** | Controls search iteration process. Default 0, CAGRA automatically determines based on itopk_size and search_width | 0 |
| **team_size** | Number of CUDA threads for calculating metric distance on GPU. Common values: power of 2 up to 32 (2, 4, 8, 16, 32) | 0 (auto) |
| **ef** | Query time/accuracy trade-off. Higher ef = more accurate but slower. Mandatory if adapt_for_cpu=true | [top_k, int_max] |

## Python Code Example

```python
from pymilvus import MilvusClient

# Build index
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="your_vector_field_name",
    index_type="GPU_CAGRA",
    index_name="vector_index",
    metric_type="L2",
    params={
        "intermediate_graph_degree": 64,
        "graph_degree": 32,
        "build_algo": "IVF_PQ",
        "cache_dataset_on_device": "true",
        "adapt_for_cpu": "false"
    }
)

# Search
search_params = {
    "params": {
        "itopk_size": 16,
        "search_width": 8
    }
}

res = MilvusClient.search(
    collection_name="your_collection_name",
    anns_field="vector_field",
    data=[[0.1, 0.2, 0.3, 0.4, 0.5]],
    limit=3,
    search_params=search_params
)
```

## Enable CPU Search at Load Time (Milvus 2.6.4+)

```yaml
# milvus.yaml
knowhere:
  GPU_CAGRA:
    load:
      adapt_for_cpu: true
```

**Behavior:**
- When `load.adapt_for_cpu=true`, Milvus converts GPU_CAGRA index into CPU-executable format (HNSW-like) during load
- Subsequent searches execute on CPU, even if index was originally built for GPU
- Use in hybrid or cost-sensitive environments where GPU reserved for index building but searches run on CPU

## Memory Usage

GPU_CAGRA memory usage is approximately **1.8x** the size of original vector data.
