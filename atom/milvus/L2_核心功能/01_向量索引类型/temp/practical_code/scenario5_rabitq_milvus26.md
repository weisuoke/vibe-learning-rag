# RaBitQ in Milvus 2.6 - Cost Optimization

Source: https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md
Date: June 11, 2025
Fetched: 2026-02-21

## RaBitQ 1-bit Quantization: 72% Memory Reduction with 4× Performance

Traditional quantization methods force you to trade search quality for memory savings. Milvus 2.6 changes this with **RaBitQ 1-bit quantization** combined with an intelligent refinement mechanism.

### How It Works

The new **IVF_RABITQ** index compresses the main index to 1/32 of its original size through 1-bit quantization. When used together with an optional SQ8 refinement, this approach maintains high search quality (95% recall) using only 1/4 of the original memory footprint.

### Performance Benchmarks

VectorDBBench evaluation with 1M vectors of 768 dimensions, tested on AWS m6id.2xlarge:

| Performance Metric       | Traditional IVF_FLAT | RaBitQ (1-bit) Only | RaBitQ (1-bit) + SQ8 Refine |
|--------------------------|----------------------|---------------------|--------------------------------|
| Memory Footprint         | 100% (baseline)      | 3% (97% reduction)  | 28% (72% reduction)            |
| Recall                   | 95.2%                | 76.3%               | 94.9%                          |
| Search Throughput (QPS)  | 236                  | 648 (2.7× faster)   | 946 (4× faster)                |

### Key Insights

The real breakthrough isn't just the 72% memory reduction, but achieving this while simultaneously delivering a **4× throughput improvement**. This means you can:
- Serve the same workload with 75% fewer servers
- Handle 4× more traffic on existing infrastructure
- All without sacrificing recall

### Enterprise Benefits

For enterprise users utilizing fully managed Milvus on Zilliz Cloud, an automated strategy dynamically adjusts RaBitQ parameters based on specific workload characteristics and precision requirements.

## Hot-Cold Tiered Storage: 50% Cost Reduction

Real-world vector search workloads contain data with vastly different access patterns. Milvus 2.6 introduces a tiered storage architecture that automatically classifies data based on access patterns:

- **Intelligent data classification**: Automatically identifies hot (frequently accessed) and cold (rarely accessed) data segments
- **Optimized storage placement**: Hot data remains in high-performance memory/SSD, cold data moves to economical object storage
- **Dynamic data movement**: Data automatically migrates between tiers as usage patterns change
- **Transparent retrieval**: Cold data automatically loaded on demand when queries touch it

**Result**: Up to 50% reduction in storage costs while maintaining query performance for active data.

## Additional Cost Optimizations

- **Int8 vector support** for HNSW indexes
- **Storage v2 format** for optimized structure that reduces IOPS and memory requirements
- **Easier installation** directly through APT/YUM package managers

## Production Recommendations

1. **Start with RaBitQ + SQ8 refinement** for balanced performance
2. **Monitor recall rates** and adjust refinement parameters
3. **Use tiered storage** for datasets with clear hot/cold access patterns
4. **Benchmark with your data** to find optimal configuration
