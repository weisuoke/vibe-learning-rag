# HNSW at Scale - Medium Article Feb 2026

Source: https://medium.com/write-a-catalyst/hnsw-at-scale-why-adding-more-documents-to-your-database-breaks-rag-7642e21f5ab6
Author: Gowtham Boyina
Date: Feb 2026
Fetched: 2026-02-21

## The Problem: HNSW Recall Drift

As datasets grow from 100K to 1M to 10M vectors, HNSW experiences **recall drift** - search quality degrades predictably.

### Symptoms

1. **High similarity but low relevance**: Results show 0.85+ cosine similarity but aren't actually relevant
2. **Rare queries degrade first**: Specific queries return increasingly bad results as corpus grows
3. **Non-linear latency increase**: 10K docs = 50ms, 100K = 200ms, 1M = 2 seconds
4. **Adding data makes things worse**: More documents actually decrease accuracy

## Why HNSW Degrades at Scale

### Problem 1: Local Minima Traps
With small datasets (10K vectors), greedy navigation almost always finds true nearest neighbors. With large datasets, greedy decisions early on can lead down paths that miss actual best results.

### Problem 2: Hubness in High Dimensions
Some nodes become "hubs" that appear in many search paths, even when they're not truly relevant. This is a known phenomenon in high-dimensional spaces.

### Problem 3: RAM Pressure and Cache Misses
As graph grows, more nodes don't fit in CPU cache, causing cache misses and slower traversal.

## Experimental Results

Dataset: 200,000 Jeopardy questions

**At 10,000 vectors:**
- Recall: 100%
- Latency: 91ms

**At 50,000 vectors:**
- Recall: 98%
- Latency: 145ms

**At 100,000 vectors:**
- Recall: 94%
- Latency: 198ms

**At 200,000 vectors:**
- Recall: 87%
- Latency: 312ms

**Key lesson**: If you keep same HNSW parameters as you scale from 10K to 200K vectors, you're either accepting 12x higher latency or significant recall degradation.

## Solutions

### Solution 1: Parameter Tuning
- Increase M from 16 to 32-64 for larger datasets
- Increase efConstruction from 100 to 200-400
- Increase ef_search from 32 to 64-128

**Trade-off**: Higher memory usage and slower indexing

### Solution 2: On-Disk Storage
Use disk-based indexes for very large datasets to reduce memory pressure.

### Solution 3: Quantization + Oversampling
- Use quantization (SQ8, PQ) to reduce memory
- Oversample candidates (retrieve 2x-3x more than needed)
- Re-rank with full precision vectors

### Solution 4: Two-Stage / Hybrid Retrieval
- Stage 1: Fast approximate search (HNSW with lower parameters)
- Stage 2: Re-rank top candidates with more expensive method

## Monitoring Recall Drift

Track these metrics as dataset grows:
- **Recall@K**: Percentage of true top-K results found
- **Latency P95**: 95th percentile query latency
- **Memory usage**: Graph size in RAM
- **Query distribution**: Track performance on rare vs common queries

## Parameter Recommendations by Scale

| Dataset Size | M | efConstruction | ef_search | Expected Recall |
|--------------|---|----------------|-----------|-----------------|
| < 100K | 16 | 100 | 32 | 95%+ |
| 100K - 1M | 32 | 200 | 64 | 90%+ |
| 1M - 10M | 48 | 300 | 128 | 85%+ |
| > 10M | 64 | 400 | 256 | 80%+ |

**Note**: These are starting points. Always benchmark with your specific data and queries.
