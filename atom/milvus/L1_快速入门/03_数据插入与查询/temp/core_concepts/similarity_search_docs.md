---
source: https://milvus.io/docs/search.md
title: Basic Vector Search | Milvus Documentation
fetched_at: 2026-02-21
---

# Basic Vector Search

Based on an index file recording the sorted order of vector embeddings, the Approximate Nearest Neighbor (**ANN**) search locates a subset of vector embeddings based on the query vector carried in a received search request, compares the query vector with those in the subgroup, and returns the most similar results. With ANN search, Milvus provides an efficient search experience. This page helps you to learn how to conduct basic ANN searches.

> **note**
> If you dynamically add new fields after the collection has been created, searches that include these fields will return the defined default values or **NULL** for entities that have not explicitly set values. For details, refer to [Add Fields to an Existing Collection](/docs/add-fields-to-an-existing-collection.md).

## Overview

The ANN and the k-Nearest Neighbors (**kNN**) search are the usual methods in vector similarity searches. In a kNN search, you must compare all vectors in a vector space with the query vector carried in the search request before figuring out the most similar ones, which is time-consuming and resource-intensive.

Unlike kNN searches, an ANN search algorithm asks for an **index** file that records the sorted order of vector embeddings. When a search request comes in, you can use the index file as a reference to quickly locate a subgroup probably containing vector embeddings most similar to the query vector. Then, you can use the specified **metric type** to measure the similarity between the query vector and those in the subgroup, sort the group members based on similarity to the query vector, and figure out the **top-K** group members.

ANN searches depend on pre-built indexes, and the search throughput, memory usage, and search correctness may vary with the index types you choose. You need to balance search performance and correctness.

To reduce the learning curve, Milvus provides **AUTOINDEX**. With **AUTOINDEX**, Milvus can analyze the data distribution within your collection while building the index and sets the most optimized index parameters based on the analysis to strike a balance between search performance and correctness.

In this section, you will find detailed information about the following topics:

- [Single-vector search](#single-vector-search)
- [Bulk-vector search](#bulk-vector-search)
- [ANN search in partition](#ann-search-in-partition)
- [Use output fields](#use-output-fields)
- [Use limit and offset](#use-limit-and-offset)
- [Use level](#use-level)
- [Get Recall Rate](#get-recall-rate)
- [Enhancing ANN search](#enhancing-ann-search)

## Single-Vector Search

In ANN searches, a single-vector search refers to a search that involves only one query vector. Based on the pre-built index and the metric type carried in the search request, Milvus will find the top-K vectors most similar to the query vector.

In this section, you will learn how to conduct a single-vector search. The search request carries a single query vector and asks Milvus to use Inner Product (**IP**) to calculate the similarity between query vectors and vectors in the collection and returns the three most similar ones.

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 4. Single vector search
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
res = client.search(
    collection_name="quick_setup",
    anns_field="vector",
    data=[query_vector],
    limit=3,
    search_params={"metric_type": "IP"}
)

for hits in res:
    for hit in hits:
        print(hit)

# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ]
# ]
```

```java
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;

import java.util.*;

MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
        .uri("http://localhost:19530")
        .token("root:Milvus")
        .build());

FloatVec queryVector = new FloatVec(new float[]{0.3580376395471989f, -0.6023495712049978f, 0.18414012509913835f, -0.26286205330961354f, 0.9029438446296592f});
SearchReq searchReq = SearchReq.builder()
        .collectionName("quick_setup")
        .data(Collections.singletonList(queryVector))
        .annsField("vector")
        .topK(3)
        .build();

SearchResp searchResp = client.search(searchReq);

List<List<SearchResp.SearchResult>> searchResults = searchResp.getSearchResults();
for (List<SearchResp.SearchResult> results : searchResults) {
    System.out.println("TopK results:");
    for (SearchResp.SearchResult result : results) {
        System.out.println(result);
    }
}

// Output
// TopK results:
// SearchResp.SearchResult(entity={}, score=0.95944905, id=5)
// SearchResp.SearchResult(entity={}, score=0.8689616, id=1)
// SearchResp.SearchResult(entity={}, score=0.866088, id=7)
```

Milvus ranks the search results by their similarity scores to the query vector in **descending order**. The similarity score is also termed the distance to the query vector, and its value ranges vary with the metric types in use.

The following table lists the applicable metric types and the corresponding distance ranges.

| Metric Type | Characteristics                        | Distance Range     |
|-------------|----------------------------------------|--------------------|
| L2          | A smaller value indicates a higher similarity. | [0, ∞)            |
| IP          | A greater value indicates a higher similarity. | [-1, 1]           |
| COSINE      | A greater value indicates a higher similarity. | [-1, 1]           |
| JACCARD     | A smaller value indicates a higher similarity. | [0, 1]            |
| HAMMING     | A smaller value indicates a higher similarity. | [0, dim(vector)]  |

## Bulk-Vector Search

Similarly, you can include multiple query vectors in a search request. Milvus will conduct ANN searches for the query vectors in parallel and return two sets of results.

```python
# 7. Search with multiple vectors
# 7.1. Prepare query vectors
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648],
    [0.0039737443, 0.003020432, -0.0006188639, 0.03913546, -0.00089768134]
]

# 7.2. Start search
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    limit=3,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
```
