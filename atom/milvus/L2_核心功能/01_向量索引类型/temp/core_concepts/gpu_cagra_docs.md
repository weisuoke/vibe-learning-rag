# GPU CAGRA Index Documentation

## Official Milvus Documentation

### GPU_CAGRA Index
- **URL**: https://milvus.io/docs/gpu-cagra.md
- **Description**: Milvus 2.6 中 GPU_CAGRA 索引的官方文档，介绍基于图的 GPU 优化索引、构建参数、搜索参数及 CPU 适配配置，支持 2.6.4+ 版本在加载时启用 CPU 查询。

### GPU Index Overview
- **URL**: https://milvus.io/docs/gpu_index.md
- **Description**: Milvus GPU 索引总览文档，包括 GPU_CAGRA 的详细参数说明、内存使用及性能权衡，适用于 2.6 版本的高吞吐高召回场景。

### Index with GPU Guide
- **URL**: https://milvus.io/docs/index-with-gpu.md
- **Description**: Milvus 使用 GPU 构建索引指南，包含 GPU_CAGRA 示例代码、参数配置及内存池设置，适用于 2.6 版本 GPU 加速向量搜索。

## Key Features

### Performance Characteristics
- **12-15x faster** index building compared to CPU HNSW
- **10x faster** search performance than CPU HNSW
- **High accuracy**: Maintains recall rates comparable to HNSW
- **GPU acceleration**: Leverages NVIDIA CUDA for parallel processing

### Hybrid GPU-CPU Mode (Milvus 2.6.1+)
- **GPU for building**: Fast, high-quality graph construction
- **CPU for querying**: Cost-effective, scalable search
- **adapt_for_cpu parameter**: Enables hybrid mode
- **Best of both worlds**: Performance + cost efficiency

## Blog Articles

### Hybrid GPU-CPU Approach
- **URL**: https://milvus.io/blog/faster-index-builds-and-scalable-queries-with-gpu-cagra-in-milvus.md
- **Title**: Optimizing NVIDIA CAGRA in Milvus: A Hybrid GPU–CPU Approach
- **Key Points**:
  - Milvus 2.6.1 introduces hybrid GPU_CAGRA mode
  - GPU builds high-quality graph 12-15x faster
  - CPU executes queries for lower cost and better scalability
  - Ideal for large-scale workloads

### Milvus 2.6 Introduction
- **URL**: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
- **Key Points**:
  - GPU_CAGRA for billion-scale vector search
  - Hybrid mode: GPU build + CPU query
  - Cost-effective alternative to pure GPU deployment
  - Designed to reduce infrastructure costs

## Technical Architecture

### Graph-Based Index
- **CAGRA**: CUDA-Accelerated Graph-based ANN
- **Hierarchical structure**: Multi-layer graph for efficient search
- **NVIDIA technology**: Optimized for CUDA-capable GPUs
- **Parallel processing**: Leverages thousands of GPU cores

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA version**: Compatible with Milvus 2.6+
- **Memory**: Sufficient GPU memory for index storage
- **Driver**: Latest NVIDIA drivers recommended

## Configuration Parameters

### Build Parameters
- **intermediate_graph_degree**: Controls graph connectivity during build
- **graph_degree**: Final graph degree after optimization
- **build_algo**: Algorithm for graph construction

### Search Parameters
- **itopk_size**: Internal top-k size for search
- **search_width**: Beam width for graph traversal
- **min_iterations**: Minimum search iterations
- **max_iterations**: Maximum search iterations

### Hybrid Mode Parameters
- **adapt_for_cpu**: Enable CPU querying after GPU build
- **cache_dataset_on_device**: Keep dataset in GPU memory

## Performance Benchmarks

### Index Building
- **CPU HNSW**: Baseline performance
- **GPU CAGRA**: 12-15x faster than CPU HNSW
- **Large-scale**: Especially effective for 10M+ vectors

### Query Performance
- **Pure GPU**: 10x faster than CPU HNSW
- **Hybrid mode**: Slightly slower than pure GPU, but more scalable
- **Throughput**: Higher QPS for high-concurrency scenarios

## Use Cases

### Ideal Scenarios
- **Large-scale datasets**: 10M+ vectors
- **High-throughput requirements**: Real-time search at scale
- **RAG systems**: Fast retrieval for LLM applications
- **Production deployments**: Where performance is critical

### Hybrid Mode Benefits
- **Cost optimization**: Reduce GPU infrastructure costs
- **Scalability**: Handle more concurrent queries with CPU
- **Flexibility**: GPU for build, CPU for query
- **Resource efficiency**: Better utilization of available hardware

## GitHub Resources

### Milvus Repository
- **URL**: https://github.com/milvus-io/milvus
- **Description**: Milvus 开源仓库，包含 2.6 系列代码、GPU_CAGRA 实现细节、issue 讨论及最新 release（如 2.6.11），支持 NVIDIA CAGRA GPU 索引。

### Community Discussions
- **Performance tips**: Community-shared optimization strategies
- **Issue tracking**: Known issues and workarounds
- **Feature requests**: Upcoming enhancements

## Integration Example

```python
from pymilvus import Collection, connections

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Create collection with GPU_CAGRA index
collection = Collection("gpu_cagra_demo")

# Build GPU_CAGRA index
index_params = {
    "index_type": "GPU_CAGRA",
    "metric_type": "L2",
    "params": {
        "intermediate_graph_degree": 128,
        "graph_degree": 64,
        "build_algo": "IVF_PQ"
    }
}

collection.create_index(
    field_name="embeddings",
    index_params=index_params
)

# Load collection (with hybrid mode)
collection.load(
    replica_number=1,
    _resource_groups=["gpu"],
    adapt_for_cpu=True  # Enable hybrid mode
)

# Search
results = collection.search(
    data=query_vectors,
    anns_field="embeddings",
    param={"metric_type": "L2", "params": {"itopk_size": 128}},
    limit=10
)
```

## Comparison with Other Indexes

### vs CPU HNSW
- **Build speed**: 12-15x faster
- **Query speed**: 10x faster (pure GPU mode)
- **Memory**: Requires GPU memory
- **Cost**: Higher hardware cost, but better performance

### vs IVF_FLAT
- **Accuracy**: Higher recall
- **Speed**: Faster for large-scale datasets
- **Memory**: More memory intensive
- **Scalability**: Better for billion-scale vectors

### vs RaBitQ
- **Performance**: GPU_CAGRA faster, RaBitQ more memory-efficient
- **Cost**: RaBitQ better for cost optimization
- **Use case**: GPU_CAGRA for speed, RaBitQ for scale

## Best Practices

### When to Use GPU_CAGRA
1. **Large datasets**: 10M+ vectors
2. **High QPS requirements**: Real-time search at scale
3. **GPU availability**: NVIDIA GPUs with CUDA support
4. **Performance priority**: Speed over cost

### When to Use Hybrid Mode
1. **Cost optimization**: Reduce GPU infrastructure costs
2. **Scalability**: Handle more concurrent queries
3. **Flexible deployment**: GPU for build, CPU for query
4. **Production environments**: Balance performance and cost

### Optimization Tips
1. **Tune graph_degree**: Balance accuracy and memory
2. **Adjust search_width**: Trade-off between speed and recall
3. **Use adapt_for_cpu**: Enable hybrid mode for cost savings
4. **Monitor GPU memory**: Ensure sufficient memory for index

## References

1. Milvus GPU_CAGRA Documentation: https://milvus.io/docs/gpu-cagra.md
2. GPU Index Overview: https://milvus.io/docs/gpu_index.md
3. Hybrid GPU-CPU Blog: https://milvus.io/blog/faster-index-builds-and-scalable-queries-with-gpu-cagra-in-milvus.md
4. Milvus GitHub: https://github.com/milvus-io/milvus
