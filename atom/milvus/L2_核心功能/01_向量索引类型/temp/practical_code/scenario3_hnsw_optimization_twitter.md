# Scenario 3: HNSW Performance Tips and Tuning Guide

## Search Results

### 1. How to Spot Search Performance Bottleneck in Vector Databases
**URL**: https://zilliz.com/learn/how-to-spot-search-performance-bottleneck-in-vector-databases
**Description**: This guide explains monitoring search performance in Milvus, identifying bottlenecks, and optimizing HNSW indexes for high QPS, including hardware and parameter tuning tips.

### 2. Hierarchical Navigable Small Worlds (HNSW) in Vector Search — Part 2
**URL**: https://medium.com/@adnanmasood/the-shortcut-through-space-hierarchical-navigable-small-worlds-hnsw-in-vector-search-part-2-ba2e8a64134e
**Description**: Explores HNSW implementation in Milvus, highlighting its superior QPS in benchmarks on large datasets, with discussions on tuning for optimal performance.

### 3. VectorDBBench: Benchmark for vector databases
**URL**: https://github.com/zilliztech/VectorDBBench
**Description**: Benchmarking tool for Milvus and others, focusing on HNSW parameters like ef-construction and ef-search to tune for high performance and QPS.

### 4. Top 10 Vector Indexes for Faster RAG
**URL**: https://medium.com/@connect.hashblock/top-10-vector-indexes-for-faster-rag-35ad7c9b6aa2
**Description**: Compares HNSW with other indexes, providing latency tips, recall trade-offs, and tuning guides for high-QPS vector search in RAG applications.

### 5. Top 10 Vector Databases for RAG Applications
**URL**: https://www.blog.qualitypointtech.com/2025/09/top-10-vector-databases-for-rag.html
**Description**: Reviews Milvus for high-QPS RAG, advising on HNSW tuning versus IVF_PQ, with performance optimization strategies for large-scale deployments.

### 6. How to Build Scalable Enterprise AI with Vector Databases
**URL**: https://bix-tech.com/how-to-build-scalable-enterprise-ai-with-vector-databases-in-2024-and-beyond
**Description**: Details HNSW tuning in Milvus, including ef_search and M parameters for balancing recall and latency in high-QPS enterprise AI systems.

### 7. Milvus-specific tips for performance and accuracy
**URL**: https://huggingface.co/datasets/John6666/forum2/blob/main/vectorizing_form_data_5.md
**Description**: Offers practical tips for optimizing Milvus HNSW indexes, including collection management, partitioning, and parameter adjustments for better QPS.

## Performance Optimization Tips

### High QPS Optimization
1. **Increase Query Nodes**: Distribute load across multiple query nodes
2. **Memory Replicas**: Use memory replicas for small datasets to boost QPS
3. **Connection Pooling**: Essential for handling concurrent requests
4. **Batch Queries**: Group queries to reduce overhead
5. **Hardware**: Use high-memory machines (HNSW is memory-intensive)

### Parameter Tuning for QPS
- **Lower ef**: Reduces search time, increases QPS (trade-off: lower recall)
- **Optimal M**: M=16-32 for balanced QPS/recall
- **efConstruction**: Set once during build, doesn't affect search QPS
- **Partitioning**: Use partitions to reduce search space

### Latency Optimization
- **P99 Latency**: Monitor and optimize for tail latency
- **ef_search Tuning**:
  - ef=64: ~10ms latency, 95% recall
  - ef=128: ~20ms latency, 98% recall
  - ef=256: ~40ms latency, 99% recall
- **Hardware**: SSD for data, RAM for index
- **Network**: Low-latency network for distributed deployments

### Throughput Optimization
- **Horizontal Scaling**: Add more query nodes
- **Vertical Scaling**: Increase memory per node
- **Load Balancing**: Distribute queries evenly
- **Caching**: Cache frequent queries

### Benchmarks (from sources)
- **Milvus HNSW**: Highest single-node QPS in 3 of 4 datasets (1M-10M vectors)
- **vs HNSW**: Outperforms industry standard HNSW in efficiency and speed
- **Small Batches**: Exceptional performance even for small batch sizes
- **Large Datasets**: Maintains performance on 10M+ vector datasets

## Production Monitoring
1. **QPS**: Queries per second
2. **P99 Latency**: 99th percentile latency
3. **Recall**: Accuracy of search results
4. **Memory Usage**: RAM consumption
5. **CPU Usage**: CPU utilization
6. **Network I/O**: Network bandwidth usage
