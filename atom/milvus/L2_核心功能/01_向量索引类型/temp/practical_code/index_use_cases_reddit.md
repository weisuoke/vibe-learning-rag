# Index Selection Use Cases

## HNSW vs IVF-Flat Comparison
- **Source**: https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md

### IVF_FLAT Best For
- Memory limited but need high query performance
- High recall requirements
- Filtered searches (better than HNSW with filters)

### HNSW Best For
- Sufficient memory available
- Need extremely high recall and low latency
- Low filter ratio scenarios
- Frequent updates not required

## Medium Article Insights
- **URL**: https://medium.com/@nitinprodduturi/hnsw-vs-ivf-flat-choosing-the-right-vector-index-for-similarity-search-921ce576ddb2

### HNSW Scenarios
- High recall requirements
- Low latency critical
- Frequent updates needed

### IVF-Flat Scenarios
- Large-scale deployments
- Stable data (infrequent updates)
- Query throughput > update frequency

## Emergent Mind Summary
- **URL**: https://www.emergentmind.com/topics/milvus-hnsw-ivf
- HNSW better for fine-grained chunking and high-dimensional embeddings
- IVF-Flat provides latency/memory balance for large-scale deployments
