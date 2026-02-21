# Similarity Search Documentation (Milvus 2.6)

## Source 1: Full Text Search with Milvus

---
source: https://milvus.io/docs/full_text_search_with_milvus.md
title: Full Text Search with Milvus | Milvus Documentation
fetched_at: 2026-02-21
---

# Full Text Search with Milvus

[Full-text search](https://milvus.io/docs/full-text-search.md#Full-Text-Search) is a traditional method for retrieving documents by matching specific keywords or phrases in the text. It ranks results based on relevance scores calculated from factors like term frequency. While semantic search is better at understanding meaning and context, full-text search excels at precise keyword matching, making it a useful complement to semantic search. A common approach to constructing a Retrieval-Augmented Generation (RAG) pipeline involves retrieving documents through both semantic search and full-text search, followed by a reranking process to refine the results.

This approach converts text into sparse vectors for BM25 scoring. To ingest documents, users can simply input raw text without computing the sparse vector manually. Milvus will automatically generate and store the sparse vectors. To search documents, users just need to specify the text search query. Milvus will compute BM25 scores internally and return ranked results.

Milvus also supports hybrid retrieval by combining full-text search with dense vector based semantic search. It usually improves search quality and delivers better results to users by balancing keyword matching and semantic understanding.

## Setup and Configuration

Import the necessary libraries:

```python
from typing import List
from openai import OpenAI

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
```

Connect to Milvus:

```python
# Connect to Milvus
uri = "http://localhost:19530"
collection_name = "full_text_demo"
client = MilvusClient(uri=uri)
```

## Collection Setup for Full-Text Search

Setting up a collection for full-text search requires several configuration steps.

### Text Analysis Configuration

For full-text search, we define how text should be processed. Analyzers are essential in full-text search by breaking sentences into tokens and performing lexical analysis like stemming and stop word removal.

```python
# Define tokenizer parameters for text analysis
analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
```

### Collection Schema and BM25 Function

Now we define the schema with fields for primary key, text content, sparse vectors (for full-text search), dense vectors (for semantic search), and metadata. We also configure the BM25 function for full-text search.

The BM25 function automatically converts text content into sparse vectors, allowing Milvus to handle the complexity of full-text search without requiring manual sparse embedding generation.

```python
# Create schema
schema = MilvusClient.create_schema()
schema.add_field(
    field_name="id",
    datatype=DataType.VARCHAR,
    is_primary=True,
    auto_id=True,
    max_length=100,
)
schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,
    max_length=65535,
    analyzer_params=analyzer_params,
    enable_match=True,  # Enable text matching
    enable_analyzer=True,  # Enable text analysis
)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(
    field_name="dense_vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=1536,  # Dimension for text-embedding-3-small
)
schema.add_field(field_name="metadata", datatype=DataType.JSON)

# Define BM25 function to generate sparse vectors from text
bm25_function = Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content"],
    output_field_names="sparse_vector",
)

# Add the function to schema
schema.add_function(bm25_function)
```

### Indexing and Collection Creation

To optimize search performance, we create indexes for both sparse and dense vector fields, then create the collection in Milvus.

```python
# Define indexes
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)
index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="IP")

# Create the collection
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)
```

## Insert Data

After setting up the collection, we insert data by preparing entities with both text content and their vector representations.

```python
# Set up OpenAI for embeddings
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model_name = "text-embedding-3-small"

# Define embedding generation function for reuse
def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    response = openai_client.embeddings.create(input=texts, model=model_name)
    return [embedding.embedding for embedding in response.data]

# Example documents to insert
documents = [
    {
        "content": "Milvus is a vector database built for embedding similarity search and AI applications.",
        "metadata": {"source": "documentation", "topic": "introduction"},
    },
    {
        "content": "Full-text search in Milvus allows you to search using keywords and phrases.",
        "metadata": {"source": "tutorial", "topic": "full-text search"},
    },
    {
        "content": "Hybrid search combines the power of sparse BM25 retrieval with dense vector search.",
        "metadata": {"source": "blog", "topic": "hybrid search"},
    },
]

# Prepare entities for insertion
entities = []
texts = [doc["content"] for doc in documents]
embeddings = get_embeddings(texts)

for i, doc in enumerate(documents):
    entities.append(
        {
            "content": doc["content"],
            "dense_vector": embeddings[i],
            "metadata": doc.get("metadata", {}),
        }
    )

# Insert data
client.insert(collection_name, entities)
```

## Perform Retrieval

You can flexibly use the `search()` or `hybrid_search()` methods to implement full-text search (sparse), semantic search (dense), and hybrid search.

### Full-Text Search

Sparse search leverages the BM25 algorithm to find documents containing specific keywords or phrases.

```python
# Example query for keyword search
query = "full-text search keywords"

# BM25 sparse vectors
results = client.search(
    collection_name=collection_name,
    data=[query],
    anns_field="sparse_vector",
    limit=5,
    output_fields=["content", "metadata"],
)
sparse_results = results[0]

# Print results
print("\nSparse Search (Full-text search):")
for i, result in enumerate(sparse_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )
```

### Semantic Search

Dense search uses vector embeddings to find documents with similar meaning, even if they don't share the exact same keywords.

```python
# Example query for semantic search
query = "How does Milvus help with similarity search?"

# Generate embedding for query
query_embedding = get_embeddings([query])[0]

# Semantic search using dense vectors
results = client.search(
    collection_name=collection_name,
    data=[query_embedding],
    anns_field="dense_vector",
    limit=5,
    output_fields=["content", "metadata"],
)
dense_results = results[0]

# Print results
print("\nDense Search (Semantic):")
for i, result in enumerate(dense_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )
```

### Hybrid Search

Hybrid search combines both full-text search and semantic dense retrieval. This balanced approach improves search accuracy and robustness by leveraging the strengths of both methods.

```python
# Example query for hybrid search
query = "what is hybrid search"

# Get query embedding
query_embedding = get_embeddings([query])[0]

# Set up BM25 search request
sparse_search_params = {"metric_type": "BM25"}
sparse_request = AnnSearchRequest(
    [query], "sparse_vector", sparse_search_params, limit=5
)

# Set up dense vector search request
dense_search_params = {"metric_type": "IP"}
dense_request = AnnSearchRequest(
    [query_embedding], "dense_vector", dense_search_params, limit=5
)

# Perform hybrid search with reciprocal rank fusion
results = client.hybrid_search(
    collection_name,
    [sparse_request, dense_request],
    ranker=RRFRanker(),  # Reciprocal Rank Fusion for combining results
    limit=5,
    output_fields=["content", "metadata"],
)
hybrid_results = results[0]

# Print results
print("\nHybrid Search (Combined):")
for i, result in enumerate(hybrid_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )
```

---

## Source 2: Multi-Vector Hybrid Search

---
source: https://milvus.io/docs/multi-vector-search.md
title: Multi-Vector Hybrid Search | Milvus Documentation
fetched_at: 2026-02-21
---

# Multi-Vector Hybrid Search

In many applications, an object can be searched by a rich set of information such as title and description, or with multiple modalities such as text, images, and audio. Hybrid search enhances search experience by combining searches across these diverse fields. Milvus supports this by allowing search on multiple vector fields, conducting several Approximate Nearest Neighbor (ANN) searches simultaneously.

## Hybrid Search Workflow

The multi-vector hybrid search integrates different search methods or spans embeddings from various modalities:

**Sparse-Dense Vector Search**: Dense Vector are excellent for capturing semantic relationships, while Sparse Vector are highly effective for precise keyword matching. Hybrid search combines these approaches to provide both a broad conceptual understanding and exact term relevance, thus improving search results.

**Multimodal Vector Search**: Multimodal vector search is a powerful technique that allows you to search across various data types, including text, images, audio, and others. The main advantage of this approach is its ability to unify different modalities into a seamless and cohesive search experience.

## Example

Let's consider a real world use case where each product includes a text description and an image. Based on the available data, we can conduct three types of searches:

**Semantic Text Search:** This involves querying the text description of the product using dense vectors. Text embeddings can be generated using models such as BERT and Transformers or services like OpenAI.

**Full-Text Search:** Here, we query the text description of the product using a keyword match with sparse vectors. Algorithms like BM25 or sparse embedding models such as BGE-M3 or SPLADE can be utilized for this purpose.

**Multimodal Image Search:** This method queries over the image using a text query with dense vectors. Image embeddings can be generated with models like CLIP.

This guide will walk you through an example of a multimodal hybrid search combining the above search methods, given the raw text description and image embeddings of products.

## Create a collection with multiple vector fields

The process of creating a collection involves three key steps: defining the collection schema, configuring the index parameters, and creating the collection.

### Define schema

For multi-vector hybrid search, we should define multiple vector fields within a collection schema.

This example incorporates the following fields into the schema:

- **id**: Serves as the primary key for storing text IDs. This field is of data type INT64.
- **text**: Used for storing textual content. This field is of the data type VARCHAR with a maximum length of 1000 bytes. The enable_analyzer option is set to True to facilitate full-text search.
- **text_dense**: Used to store dense vectors of the texts. This field is of the data type FLOAT_VECTOR with a vector dimension of 768.
- **text_sparse**: Used to store sparse vectors of the texts. This field is of the data type SPARSE_FLOAT_VECTOR.
- **image_dense**: Used to store dense vectors of the product images. This field is of the data type FLOAT_VECTOR with a vector dimension of 512.

Since we will use the built-in BM25 algorithm to perform a full-text search on the text field, it is necessary to add the Milvus `Function` to the schema.

```python
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# Init schema with auto_id disabled
schema = client.create_schema(auto_id=False)

# Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, description="product id")
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True, description="raw text of product description")
schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=768, description="text dense embedding")
schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR, description="text sparse embedding auto-generated by the built-in BM25 function")
schema.add_field(field_name="image_dense", datatype=DataType.FLOAT_VECTOR, dim=512, description="image dense embedding")

# Add function to schema
bm25_function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],
    output_field_names=["text_sparse"],
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function)
```

## Key Points for RAG Development

1. **Hybrid Search is Production Standard (2026)**: Modern RAG systems should use hybrid search combining:
   - Dense vectors for semantic understanding
   - Sparse vectors (BM25) for keyword matching
   - Reranking for result refinement

2. **Automatic Sparse Vector Generation**: Milvus 2.6's BM25 function automatically generates sparse vectors from text, eliminating the need for external BM25 implementations.

3. **Multiple Vector Fields**: A single collection can have multiple vector fields (e.g., text_dense, text_sparse, image_dense), enabling multimodal search.

4. **Reranking Strategies**:
   - **RRFRanker (Reciprocal Rank Fusion)**: Combines results from multiple searches by reciprocal rank
   - **WeightedRanker**: Assigns different weights to different search methods

5. **Performance Considerations**:
   - Hybrid search is more expensive than single-vector search
   - Use appropriate index types for each vector field (SPARSE_INVERTED_INDEX for sparse, HNSW/IVF_FLAT for dense)
   - Consider using filters to reduce search space

6. **Use Cases in RAG**:
   - **Document Retrieval**: Combine semantic understanding with keyword matching for better recall
   - **Multimodal Search**: Search across text and images simultaneously
   - **Question Answering**: Use hybrid search to find relevant context for LLM generation
