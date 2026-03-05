---
source: https://milvus.io/docs/multi-vector-search.md (hybrid-search.md was 404)
title: Multi-Vector Hybrid Search | Milvus Documentation
fetched_at: 2026-02-21
status: active (hybrid-search.md redirected to this page)
---

# Multi-Vector Hybrid Search

In many applications, an object can be searched by a rich set of information such as title and description, or with multiple modalities such as text, images, and audio.

Hybrid search enhances search experience by combining searches across these diverse fields. Milvus supports this by allowing search on **multiple vector fields**, conducting several Approximate Nearest Neighbor (**ANN**) searches simultaneously.

Multi-vector hybrid search is particularly useful if you want to search both text and images, multiple text fields that describe the same object, or **dense** and **sparse** vectors to improve search quality.

## Hybrid Search Workflow

The multi-vector hybrid search integrates different search methods or spans embeddings from various modalities:

**Sparse-Dense Vector Search**: [Dense Vector](/docs/dense-vector.md) are excellent for capturing semantic relationships, while [Sparse Vector](/docs/sparse_vector.md) are highly effective for precise keyword matching.

Hybrid search combines these approaches to provide both a broad conceptual understanding and exact term relevance, thus improving search results. By leveraging the strengths of each method, hybrid search overcomes the limitations of individual approaches, offering better performance for complex queries.

Here is more detailed [guide](/docs/full_text_search_with_milvus.md) on hybrid retrieval that combines semantic search with full-text search.

## Examples of Multi-Vector Hybrid Search

### Search with Multiple Dense Vectors

When an object is described by multiple dense vectors (e.g. title embedding + description embedding + summary embedding), you can search across all of them simultaneously.

### Search with Dense + Sparse Vectors

Combining dense vectors (semantic) with sparse vectors (keyword/BM25 style) is currently one of the most popular hybrid search patterns.

Milvus supports storing both dense and sparse vectors in the same collection and conducting hybrid search using both types.

### Search with Multi-Modal Data

When the data contains multiple modalities (text + image + audio, etc.), each modality can be converted to vectors separately, and hybrid search can be performed across them.

## How to Conduct Multi-Vector Hybrid Search

Milvus provides the `hybrid_search()` API to perform multiple ANN searches in one call.

Each `AnnSearchRequest` represents a single vector search request on one vector field.

After all searches complete, Milvus will merge and rerank the results according to the specified reranking strategy.

### Python Example (pymilvus)

```python
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

client = MilvusClient()

# Assume collection already exists with multiple vector fields
# e.g. "dense_embedding", "sparse_embedding"

search_param_dense = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

search_param_sparse = {
    "metric_type": "IP",   # Inner Product usually for sparse
    "params": {}
}

req1 = AnnSearchRequest(
    data=[dense_query_vector],
    anns_field="dense_embedding",
    param=search_param_dense,
    limit=100
)

req2 = AnnSearchRequest(
    data=[sparse_query_vector],
    anns_field="sparse_embedding",
    param=search_param_sparse,
    limit=100
)

res = client.hybrid_search(
    collection_name="my_hybrid_collection",
    search_requests=[req1, req2],
    ranker=RRFRanker(),   # Reciprocal Rank Fusion
    limit=10,
    output_fields=["title", "text"]
)

for result in res:
    print(result)
```

### Key Concepts

- **AnnSearchRequest** — one vector field search request
- **ranker** — how to merge and reorder results from multiple searches
  - RRFRanker (default, good general choice)
  - WeightedRanker (assign different weights to different search requests)
- **limit** in each request — controls how many candidates per sub-search
- **limit** in hybrid_search — final number of results after reranking

## Supported Rerankers

- **RRFRanker** (Reciprocal Rank Fusion) — parameter-free, robust
- **WeightedRanker** — assign weight to each sub-search request
- **MMRRanker** (in development / experimental)

For more details about reranking strategies → [Reranking](/docs/reranking.md)

## Related Documents

- [Dense Vector](/docs/dense-vector.md)
- [Sparse Vector](/docs/sparse_vector.md)
- [Full Text Search with Milvus](/docs/full_text_search_with_milvus.md)
- [Reranking](/docs/reranking.md)
- [hybrid_search() API Reference](/api-reference/pymilvus/v2.4.x/ORM/Collection/hybrid_search.md)
