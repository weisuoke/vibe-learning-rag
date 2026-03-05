---
source: https://milvus.io/docs/embedding-function-overview.md
title: Embedding Function Overview | Milvus Documentation
fetched_at: 2026-02-21
---

# Embedding Function Overview

Compatible with Milvus 2.6.x

The Function module in Milvus allows you to transform raw text data into vector embeddings by automatically calling external embedding service providers (like OpenAI, AWS Bedrock, Google Vertex AI, etc.). With the Function module, you no longer need to manually interface with embedding APIs—Milvus handles the entire process of sending requests to providers, receiving embeddings, and storing them in your collections. For semantic search, you need to provide only raw query data, not a query vector. Milvus generates the query vector with the same model you used for ingestion, compares it to the stored vectors, and returns the most relevant results.

## Limits

Any input field that the Function module embeds must always contain a value; if a null is supplied, the module will throw an error.
The Function module processes only fields that are explicitly defined in the collection schema; it does not generate embeddings for dynamic fields.
Input fields to be embedded must be of the VARCHAR type.
The Function module can embed an input field to:

- FLOAT_VECTOR
- INT8_VECTOR

Conversions to BINARY_VECTOR, FLOAT16_VECTOR, or BFLOAT16_VECTOR are not supported.

## Supported embedding service providers

| Provider       | Typical Models                  | Embedding Type          | Authentication Method          |
|----------------|---------------------------------|-------------------------|--------------------------------|
| OpenAI         | text-embedding-3-*              | FLOAT_VECTOR            | API key                        |
| Azure OpenAI   | Deployment-based                | FLOAT_VECTOR            | API key                        |
| DashScope      | text-embedding-v3               | FLOAT_VECTOR            | API key                        |
| Bedrock        | amazon.titan-embed-text-v2      | FLOAT_VECTOR            | AK/SK pair                     |
| Vertex AI      | text-embedding-005              | FLOAT_VECTOR            | GCP service account JSON credential |
| Voyage AI      | voyage-3, voyage-lite-02        | FLOAT_VECTOR / INT8_VECTOR | API key                     |
| Cohere         | embed-english-v3.0              | FLOAT_VECTOR / INT8_VECTOR | API key                     |
| SiliconFlow    | BAAI/bge-large-zh-v1.5          | FLOAT_VECTOR            | API key                        |
| Hugging Face   | Any TEI-served model            | FLOAT_VECTOR            | Optional API key               |

## How it works

The following diagram shows how the Function works in Milvus.

1. **Input text**: Users insert raw data (e.g. documents) into Milvus.
2. **Generate embeddings**: The Function module within Milvus automatically calls the configured model provider to convert raw data into vector embeddings.
3. **Store embeddings**: The resulting embeddings are stored in explicitly defined vector fields within Milvus collections.
4. **Query text**: Users submit text queries to Milvus.
5. **Semantic search**: Milvus internally converts queries to vector embeddings, conducts similarity searches against stored embeddings, and retrieves relevant results.
6. **Return results**: Milvus returns top-matching results to the application.

## Configure credentials

Before using an embedding function with Milvus, configure embedding service credentials for Milvus access.

Milvus lets you supply embedding service credentials in two ways:

- **Configuration file** (`milvus.yaml`): The example in this topic demonstrates the **recommended setup** using milvus.yaml.
- **Environment variables**: For details on configuring credentials via environment variables, see the embedding service provider's documentation (for example, OpenAI or Azure OpenAI).

### Step 1: Add credentials to Milvus configuration file

In your `milvus.yaml` file, edit the `credential` block with entries for each provider you need to access:

```yaml
# milvus.yaml credential store section
credential:
  aksk1:
    access_key_id: <YOUR_AK>
    secret_access_key: <YOUR_SK>

  apikey1:
    apikey: <YOUR_API_KEY>

  gcp1:
    credential_json: <BASE64_OF_JSON>
```

### Step 2: Configure provider settings

```yaml
function:
  textEmbedding:
    providers:
      openai:
        credential: apikey1
      bedrock:
        credential: aksk1
        region: us-east-2
      vertexai:
        credential: gcp1
      tei:
        enable: true
```

## Use embedding function

### Step 1: Define schema fields

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530")

schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("document", DataType.VARCHAR, max_length=9000)
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)
```

### Step 2: Add embedding function to schema

```python
text_embedding_function = Function(
    name="openai_embedding",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["document"],
    output_field_names=["dense"],
    params={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    }
)

schema.add_function(text_embedding_function)
```

### Step 3: Configure index

```python
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="dense",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
```

### Step 4: Create collection

```python
client.create_collection(
    collection_name='demo',
    schema=schema,
    index_params=index_params
)
```

### Step 5: Insert data

```python
client.insert('demo', [
    {'id': 1, 'document': 'Milvus simplifies semantic search through embeddings.'},
    {'id': 2, 'document': 'Vector embeddings convert text into searchable numeric data.'},
    {'id': 3, 'document': 'Semantic search helps users find relevant information quickly.'},
])
```

### Step 6: Perform vector search

```python
results = client.search(
    collection_name='demo',
    data=['How does Milvus handle semantic search?'],
    anns_field='dense',
    limit=1,
    output_fields=['document'],
)

print(results)
```
