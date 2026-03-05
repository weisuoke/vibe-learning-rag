# Understanding IVF Vector Index - 2025 Blog

Source: https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md
Author: Jack Li
Date: October 26, 2025

## IVF Core Concept

IVF (Inverted File) is one of the most commonly used algorithms in ANN (Approximate Nearest Neighbor) search. It borrows the concept of "inverted index" from text retrieval systems, but instead of processing words and documents, it handles vectors in high-dimensional space.

### How IVF Works

**Two-step search process:**

1. **Find the nearest clusters**: Calculate the distance between the query vector and all centroids, select the closest few buckets
2. **Search within clusters**: Only search within the selected clusters instead of the entire dataset

This approach reduces computation by several orders of magnitude while maintaining high accuracy.

## Building IVF Index

### Step 1: K-means Clustering
- Perform k-means clustering on dataset X
- Divide high-dimensional vector space into `nlist` clusters
- Each cluster is represented by a centroid stored in centroid table C

### Step 2: Vector Assignment
- Assign each vector to the cluster with the nearest centroid
- Form inverted lists (List_i) for each cluster

### Step 3: Compression Encoding (Optional)

**SQ8 (Scalar Quantization)**:
- Quantize each dimension to 8 bits
- Original float32: 4 bytes per dimension
- After SQ8: 1 byte per dimension
- Compression ratio: 4:1

**PQ (Product Quantization)**:
- Split high-dimensional vector into multiple sub-spaces
- Example: 128-dim vector → 8 sub-vectors of 16 dims each
- Train codebook with 256 entries per sub-space (8 bits)
- Original: 512 bytes, After PQ: 8 bytes
- Compression ratio: 64:1

## IVF Variants Comparison

| IVF Variant   | Key Features                                                                 | Use Cases                                                                                   |
|---------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **IVF_FLAT**  | Stores original vectors without compression. Highest accuracy, largest memory. | Medium-scale datasets (up to hundreds of millions), requires very high recall (95%+)       |
| **IVF_PQ**    | Applies Product Quantization. Adjustable compression ratio, significantly reduces memory. | Ultra-large scale search (billions+), acceptable accuracy loss. 64:1 compression achieves ~70% recall, lower compression can reach 90%+ |
| **IVF_SQ8**   | Uses Scalar Quantization. Memory usage between FLAT and PQ.                  | Large-scale search, needs good balance between high recall (90%+) and efficiency           |

## Parameter Tuning

### nlist (Build-time parameter)
- Determines number of clusters created during index building
- **Larger nlist**: More refined clusters, fewer vectors per cluster, faster queries; but longer build time, more memory for centroids
- **Smaller nlist**: Faster build, less memory, but more vectors per cluster, potentially slower queries
- **Rule of thumb**: For million-scale datasets, start with nlist ≈ √n (e.g., 1M vectors → nlist = 1000)

### nprobe (Query-time parameter)
- Controls number of clusters searched during query
- **Larger nprobe**: Higher recall, linearly increased latency
- **Smaller nprobe**: Lower latency, faster queries, but may miss some true neighbors
- **Advantage**: Runtime parameter, can be adjusted without rebuilding index
- **Recommendation**: Experiment with nprobe (e.g., 1-16) to find the lowest latency that meets accuracy requirements

## IVF vs HNSW

| Dimension              | IVF                                          | HNSW                                           |
|------------------------|----------------------------------------------|------------------------------------------------|
| **Algorithm Concept**  | Clustering + Inverted buckets                | Multi-layer navigable small world graph        |
| **Memory Usage**       | Relatively low                               | Relatively high                                |
| **Build Speed**        | Fast (only clustering needed)                | Slow (need to build multi-layer graph)         |
| **Query Speed (no filter)** | Fast, depends on nprobe                 | Very fast, logarithmic complexity              |
| **Query Speed (with filter)** | Stable, small candidate set after coarse filtering at centroid level | Unstable, high filter rate (90%+) causes graph fragmentation, may degrade to near brute-force |
| **Recall Rate**        | Depends on compression; without quantization can reach **95%+** | Usually higher, about **98%+**                 |
| **Key Parameters**     | nlist, nprobe                                | m, ef_construction, ef_search                  |
| **Best Use Cases**     | Memory-constrained but needs high performance and recall; especially suitable for filtered searches | Sufficient memory, pursuing ultimate recall and ultra-low latency |
