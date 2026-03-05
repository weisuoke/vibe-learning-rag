# Index Benchmarks from GitHub

## VectorDBBench
- **URL**: https://github.com/zilliztech/VectorDBBench
- **Description**: Benchmark tool for vector databases including Milvus, Qdrant, Weaviate
- **Features**: Search performance, index building, multiple datasets

## ANN Benchmarks
- **URL**: https://github.com/erikbern/ann-benchmarks
- **Description**: Classic ANN benchmark project with Milvus (Knowhere) results

## Milvus Performance Discussion
- **URL**: https://github.com/milvus-io/milvus/discussions/19189
- **Topic**: 500M HNSW index performance optimization
- **Insights**: QPS and latency tuning for production environments

## Key Insights
- HNSW provides best balance for production RAG systems
- IVF series good for memory-constrained scenarios
- GPU CAGRA 10x faster for large-scale deployments
- RaBitQ enables billion-scale with 72% memory savings
