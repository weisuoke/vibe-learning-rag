### Reducing Embedding Dimensions

Source: https://platform.openai.com/docs/guides/embeddings

Shows how to optimize embedding vectors by reducing their dimensions using the dimensions API parameter or manual normalization. This technique reduces storage and computational costs while maintaining semantic meaning.

```APIDOC
## Embedding Dimension Reduction

### Description
Reduces embedding vector dimensions to optimize storage and performance. The dimensions parameter allows shortening embeddings without significant loss of semantic information.

### Method
POST /v1/embeddings with dimensions parameter

### Parameters
#### Request Body
- **model** (string) - Required - Embedding model ("text-embedding-3-small" or "text-embedding-3-large")
- **input** (string) - Required - Text to embed
- **dimensions** (integer) - Optional - Target dimension size (e.g., 256, 512, 1024)
- **encoding_format** (string) - Optional - Output format ("float")

### Request Example
```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Testing 123",
    encoding_format="float"
)

cut_dim = response.data[0].embedding[:256]
norm_dim = normalize_l2(cut_dim)
print(norm_dim)
```

### Use Cases
- Reduce embedding from 3072 to 1024 dimensions for vector stores with size limitations
- Trade accuracy for reduced storage and computational costs
- Maintain semantic meaning while optimizing performance

### Response
- **embedding** (array) - Shortened and normalized embedding vector
```

--------------------------------

### POST /v1/embeddings

Source: https://platform.openai.com/docs/guides/your-data

The embeddings API supports `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002` models across all regions.

```APIDOC
## POST /v1/embeddings

### Description
The embeddings API supports `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002` models across all regions.

### Method
POST

### Endpoint
/v1/embeddings

### Parameters
#### Path Parameters
Not specified in the provided documentation.

#### Query Parameters
Not specified in the provided documentation.

#### Request Body
Not specified in the provided documentation.

### Request Example
{}

### Response
#### Success Response (200)
Not specified in the provided documentation.

#### Response Example
{}
```

--------------------------------

### POST /v1/batches

Source: https://platform.openai.com/docs/api-reference/batch/cancel

Creates and executes a batch from an uploaded file of requests. The batch will be processed asynchronously within the specified completion window (currently only 24h is supported). Supports endpoints for chat completions, embeddings, completions, moderations, and responses.

```APIDOC
## POST /v1/batches

### Description
Creates and executes a batch from an uploaded file of requests for asynchronous processing.

### Method
POST

### Endpoint
https://api.openai.com/v1/batches

### Parameters

#### Request Body
- **completion_window** (string) - Required - The time frame within which the batch should be processed. Currently only `24h` is supported.
- **endpoint** (string) - Required - The endpoint to be used for all requests in the batch. Currently `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/completions`, and `/v1/moderations` are supported. Note that `/v1/embeddings` batches are restricted to a maximum of 50,000 embedding inputs across all requests.
- **input_file_id** (string) - Required - The ID of an uploaded file that contains requests for the new batch. Must be formatted as JSONL, uploaded with purpose `batch`, contain up to 50,000 requests, and be up to 200 MB in size.
- **metadata** (map) - Optional - Set of 16 key-value pairs for storing additional information. Keys are strings with maximum length of 64 characters. Values are strings with maximum length of 512 characters.
- **output_expires_after** (object) - Optional - The expiration policy for the output and/or error file generated for a batch.

### Request Example
```bash
curl https://api.openai.com/v1/batches \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

### Response

#### Success Response (200)
- **id** (string) - The unique identifier for the batch
- **object** (string) - The object type, always "batch"
- **endpoint** (string) - The endpoint used for all requests in the batch
- **errors** (object) - Any errors encountered during batch creation
- **input_file_id** (string) - The ID of the input file
- **completion_window** (string) - The completion window for the batch
- **status** (string) - The current status of the batch (validating, queued, in_progress, completed, failed, expired, cancelled, cancelling)
- **output_file_id** (string) - The ID of the output file containing results
- **error_file_id** (string) - The ID of the error file containing failed requests
- **created_at** (integer) - Unix timestamp of batch creation
- **in_progress_at** (integer) - Unix timestamp when batch processing started
- **expires_at** (integer) - Unix timestamp when batch expires
- **finalizing_at** (integer) - Unix timestamp when batch started finalizing
- **completed_at** (integer) - Unix timestamp when batch completed
- **failed_at** (integer) - Unix timestamp when batch failed
- **expired_at** (integer) - Unix timestamp when batch expired
- **cancelling_at** (integer) - Unix timestamp when batch cancellation started
- **cancelled_at** (integer) - Unix timestamp when batch was cancelled
- **request_counts** (object) - Count of requests in the batch (total, completed, failed)
- **metadata** (map) - Custom metadata attached to the batch

#### Response Example
```json
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "errors": null,
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "validating",
  "output_file_id": null,
  "error_file_id": null,
  "created_at": 1711471533,
  "in_progress_at": null,
  "expires_at": null,
  "finalizing_at": null,
  "completed_at": null,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {
    "total": 0,
    "completed": 0,
    "failed": 0
  },
  "metadata": {
    "customer_id": "user_123456789",
    "batch_description": "Nightly eval job"
  }
}
```
```

### Vector embeddings > How to get embeddings

Source: https://platform.openai.com/docs/guides/embeddings

To get an embedding, send your text string to the [embeddings API endpoint](/docs/api-reference/embeddings) along with the embedding model name (e.g., `text-embedding-3-small`). The response contains the embedding vector (list of floating point numbers) along with some additional metadata. You can extract the embedding vector, save it in a vector database, and use for many different use cases. By default, the length of the embedding vector is `1536` for `text-embedding-3-small` or `3072` for `text-embedding-3-large`. To reduce the embedding's dimensions without losing its concept-representing properties, pass in the [dimensions parameter](/docs/api-reference/embeddings/create#embeddings-create-dimensions). Find more detail on embedding dimensions in the [embedding use case section](/docs/guides/embeddings#use-cases).

--------------------------------

### Reducing embedding dimensions > Implementation and Flexibility

Source: https://platform.openai.com/docs/guides/embeddings

Using the `dimensions` parameter when creating the embedding is the suggested approach for reducing embedding size. In certain cases, you may need to change the embedding dimension after generation. When changing the dimension manually, you must normalize the dimensions of the embedding to ensure proper functionality. Dynamically changing the dimensions enables very flexible usage, allowing developers to adapt embeddings to specific constraints. For example, when using a vector data store that only supports embeddings up to 1024 dimensions, developers can use the best embedding model `text-embedding-3-large` and specify a value of 1024 for the `dimensions` API parameter, which shortens the embedding from 3072 dimensions while trading off some accuracy for the smaller vector size.