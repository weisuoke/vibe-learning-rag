# AISAQ in Milvus - 3200x Memory Reduction

Source: https://milvus.io/blog/introducing-aisaq-in-milvus-billion-scale-vector-search-got-3200-cheaper-on-memory.md
Date: December 10, 2025
Fetched: 2026-02-21

## The Memory Cost Problem

Vector databases keep key indexing structures in DRAM (Dynamic Random Access Memory) for fast search. This design is effective for performance, but scales poorly. DRAM usage scales with data size rather than query traffic.

As datasets grow, memory costs quickly become a limiting factor.

## AISAQ: Zero-DRAM-Footprint Architecture

Milvus 2.6 introduces **AISAQ**, a disk-based vector index inspired by DISKANN. Developed by KIOXIA, AISAQ's architecture was designed with a "Zero-DRAM-Footprint Architecture", which stores all search-critical data on disk and optimizes data placement to minimize I/O operations.

### Memory Reduction

In a billion-vector workload, AISAQ reduces memory usage from **32 GB to about 10 MB** — a **3,200× reduction** — while maintaining practical performance.

## How AISAQ Works

AISAQ builds on DISKANN but introduces a critical shift: it eliminates the need to keep PQ (Product Quantization) data in DRAM. Instead of treating compressed vectors as search-critical, always-in-memory structures, AISAQ moves them to SSD and redesigns how graph data is laid out on disk.

### Two Storage Modes

AISAQ provides two disk-based storage modes to address different application requirements:

#### AISAQ-Performance: Optimized for Speed

- Each node's full vector, edge list, and neighbors' PQ codes stored together on disk
- Visiting a node requires only a **single SSD read**
- Comparable performance to DISKANN
- Trade-off: Storage overhead due to redundancy
- **Latency**: ~10ms range (suitable for online semantic search)

#### AISAQ-Scale: Optimized for Storage Efficiency

- PQ data stored separately without redundancy
- Dramatically reduces index size
- May require multiple SSD reads per node
- Optimizations:
  - PQ data rearrangement for better locality
  - PQ cache in DRAM (`pq_read_page_cache_size`)
- **Use case**: RAG requirements at previously unattainable scales

## Comparison with DISKANN

| Feature | DISKANN | AISAQ-Performance | AISAQ-Scale |
|---------|---------|-------------------|-------------|
| **PQ in DRAM** | Yes (always) | No | No |
| **Memory footprint** | Medium | Very low | Extremely low |
| **Storage overhead** | Low | High (redundancy) | Low |
| **Latency** | Low | Low (~10ms) | Medium |
| **Best for** | Balanced | Online search | Ultra-large scale |

## Key Benefits

1. **Massive memory savings**: 3200x reduction for billion-scale datasets
2. **Cost-effective scaling**: Handle billions of vectors without massive DRAM
3. **Flexible modes**: Choose between performance and storage efficiency
4. **Practical performance**: Maintains usable latency even with minimal DRAM

## When to Use AISAQ

- **Billion-scale datasets** where memory costs are prohibitive
- **RAG applications** with massive document collections
- **Cost-sensitive deployments** where DRAM budget is limited
- **Archive search** where slightly higher latency is acceptable

## Production Considerations

1. **SSD performance matters**: Use high-quality SSDs for best results
2. **Cache tuning**: Adjust `pq_read_page_cache_size` based on workload
3. **Mode selection**:
   - Use AISAQ-Performance for online search (<10ms latency)
   - Use AISAQ-Scale for ultra-large datasets (cost priority)
4. **Monitor I/O**: Track SSD read patterns and optimize accordingly
