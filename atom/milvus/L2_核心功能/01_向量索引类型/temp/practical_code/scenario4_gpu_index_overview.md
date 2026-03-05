# GPU Index Overview - Milvus Documentation

Source: https://milvus.io/docs/gpu_index.md
Fetched: 2026-02-21

## Overview

Milvus supports various GPU index types to accelerate search performance and efficiency, especially in high-throughput and high-recall scenarios.

**Important Note:** Using GPU index may not necessarily reduce latency compared to CPU index. To fully maximize throughput, you need extremely high request pressure or large number of query vectors.

**GPU Support:** Contributed by NVIDIA RAPIDS team

## GPU Index Types

### GPU_CAGRA

Graph-based index optimized for GPUs. Using inference-grade GPUs can be more cost-effective than expensive training-grade GPUs.

**Memory usage:** ~1.8x original vector data

### GPU_IVF_FLAT

Similar to IVF_FLAT, divides vector data into nlist cluster units. Most basic IVF index, encoded data consistent with original data.

**Search limit:** top-K <= 256

**Memory usage:** Equal to original data size

### GPU_IVF_PQ

Performs IVF index clustering before quantizing product of vectors. Index file smaller than IVF_SQ8, but causes accuracy loss during search.

**Search limit:** top-K <= 1024

**Memory usage:** Depends on compression parameter settings (smaller footprint)

### GPU_BRUTE_FORCE

Tailored for cases where extremely high recall is crucial, guaranteeing recall of 1 by comparing each query with all vectors.

**Memory usage:** Equal to original data size

## GPU_CAGRA Detailed Parameters

### Build Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| intermediate_graph_degree | Graph degree before pruning | 128 | Recommended: 32 or 64 |
| graph_degree | Graph degree after pruning | 64 | Must be < intermediate_graph_degree |
| build_algo | Graph generation algorithm | IVF_PQ | IVF_PQ (higher quality) or NN_DESCENT (quicker) |
| cache_dataset_on_device | Cache original dataset in GPU memory | "false" | "true" or "false" |
| adapt_for_cpu | GPU for build, CPU for search | "false" | "true" or "false" |

### Search Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| itopk_size | Intermediate results size | Empty | >= limit, typically power of 2 |
| search_width | Entry points into graph | Empty | e.g., 1, 2, 4, 8, 16, 32 |
| min_iterations / max_iterations | Search iteration control | 0 | Auto-determined if 0 |
| team_size | CUDA threads for distance calc | 0 (auto) | Power of 2 up to 32 |
| ef | Query time/accuracy trade-off | [top_k, int_max] | Mandatory if adapt_for_cpu=true |

### Search Limits

| Parameter | Range |
|-----------|-------|
| limit (top-K) | <= 1024 |
| limit (top-K) | <= max((itopk_size + 31)// 32, search_width) * 32 |

## GPU_IVF_FLAT Parameters

### Build Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| nlist | Number of cluster units | 128 | [1, 65536] |
| cache_dataset_on_device | Cache in GPU memory | "false" | "true" or "false" |

### Search Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| nprobe | Number of units to query | 8 | [1, nlist] |

### Search Limits

| Parameter | Range |
|-----------|-------|
| limit (top-K) | <= 2048 |

## GPU_IVF_PQ Parameters

### Build Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| nlist | Number of cluster units | 128 | [1, 65536] |
| m | Factors of product quantization | 0 | dim mod m == 0 |
| nbits | Bits for each low-dim vector | 8 | [1, 16] |
| cache_dataset_on_device | Cache in GPU memory | "false" | "true" or "false" |

### Search Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| nprobe | Number of units to query | 8 | [1, nlist] |

### Search Limits

| Parameter | Range |
|-----------|-------|
| limit (top-K) | <= 1024 |

## Memory Comparison

| Index Type | Memory Usage |
|------------|--------------|
| GPU_CAGRA | ~1.8x original data |
| GPU_IVF_FLAT | Equal to original data |
| GPU_BRUTE_FORCE | Equal to original data |
| GPU_IVF_PQ | Smaller (depends on compression) |

## Conclusion

All GPU indexes are loaded into GPU memory for efficient search operations. Amount of data that can be loaded depends on GPU memory size.
