# Performance Tips from Milvus Community

## GPU CAGRA Optimization
- **Source**: https://milvus.io/blog/faster-index-builds-and-scalable-queries-with-gpu-cagra-in-milvus.md
- **Hybrid mode**: GPU for build (12-15x faster), CPU for query (cost-effective)
- **adapt_for_cpu parameter**: Enable hybrid mode in Milvus 2.6.1+
- **Best for**: Large-scale workloads balancing performance and cost

## RaBitQ Memory Optimization
- **Source**: https://milvus.io/blog/bring-vector-compression-to-the-extreme-how-milvus-serves-3%C3%97-more-queries-with-rabitq.md
- **72% memory reduction**: Compress to 1/32 of original size
- **4x query speedup**: Faster than traditional methods
- **<2% accuracy loss**: Maintains high recall with minimal degradation
- **3× more queries**: Serve more traffic with same infrastructure

## Milvus 2.6 Features
- **Source**: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
- **RaBitQ**: Billion-scale vector search with cost optimization
- **GPU CAGRA hybrid**: Balance performance and infrastructure costs
- **Production-ready**: Designed for real-world deployments

## Key Takeaways
1. **HNSW**: Best for production RAG systems (high accuracy + speed)
2. **IVF series**: Good for memory-constrained or filtered searches
3. **GPU CAGRA**: 10x speedup for large-scale with GPU
4. **RaBitQ**: 72% memory savings for billion-scale deployments
5. **Hybrid approaches**: Combine strengths of different indexes
