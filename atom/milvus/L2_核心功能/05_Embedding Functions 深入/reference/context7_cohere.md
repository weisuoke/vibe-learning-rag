### Cohere Embed Jobs API

Source: https://docs.cohere.com/v2/reference/create-dataset_explorer=true

Manage embed jobs for batch processing of text data to generate embeddings. Supports creation, listing, fetching, and cancellation.

```APIDOC
POST /v1/embed-jobs
Description: Creates a new embed job for batch embedding generation.
URL: https://docs.cohere.com/v2/reference/create-embed-job?explorer=true
```

```APIDOC
GET /v1/embed-jobs
Description: Lists all available embed jobs.
URL: https://docs.cohere.com/v2/reference/list-embed-jobs?explorer=true
```

```APIDOC
GET /v1/embed-jobs/{jobId}
Description: Fetches a specific embed job by its ID.
URL: https://docs.cohere.com/v2/reference/get-embed-job?explorer=true
```

```APIDOC
POST /v1/embed-jobs/{jobId}/cancel
Description: Cancels an ongoing embed job.
URL: https://docs.cohere.com/v2/reference/cancel-embed-job?explorer=true
```

--------------------------------

### Cohere Embed Jobs API

Source: https://docs.cohere.com/v2/reference/listfinetunedmodels_explorer=true

Manage embed jobs for batch embedding generation. Supports creating, listing, fetching, and canceling jobs.

```APIDOC
POST /v1/embed-jobs
Description: Creates a new embed job for batch embedding generation.
Link: https://docs.cohere.com/v2/reference/create-embed-job?explorer=true

GET /v1/embed-jobs
Description: Lists all available embed jobs.
Link: https://docs.cohere.com/v2/reference/list-embed-jobs?explorer=true

GET /v1/embed-jobs/{job_id}
Description: Fetches a specific embed job by its ID.
Link: https://docs.cohere.com/v2/reference/get-embed-job?explorer=true

POST /v1/embed-jobs/{job_id}/cancel
Description: Cancels an ongoing embed job.
Link: https://docs.cohere.com/v2/reference/cancel-embed-job?explorer=true
```

--------------------------------

### Cohere API Reference - Batch Embedding Jobs

Source: https://docs.cohere.com/v2/docs/amazon-bedrock

How to use the Embed Jobs API for processing large batches of text for embeddings.

```APIDOC
Batch Embedding Jobs:
  https://docs.cohere.com/v2/docs/embed-jobs-api
```

Source: https://docs.cohere.com/v2/docs/embed-jobs-api

Batch Embedding Jobs with the Embed API > How to use the Embed Jobs API: The Embed Jobs API was designed for users who want to leverage the power of retrieval over large corpuses of information. Encoding hundreds of thousands of documents (or chunks) via an API can be painful and slow, often resulting in millions of http-requests sent between your system and our servers. Because it validates, stages, and optimizes batching for the user, the Embed Jobs API is much better suited for encoding a large number (100K+) of documents. The Embed Jobs API also stores the results in a hosted Dataset so there is no need to store the result of your embeddings locally.
The Embed Jobs API works in conjunction with the Embed API; in production use-cases, Embed Jobs is used to stage large periodic updates to your corpus and Embed handles real-time queries and smaller real-time updates.

--------------------------------

Source: https://docs.cohere.com/v2/reference/embed

Request: Show 6 enum values
truncateenumOptionalDefaults to `END`
One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.
Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
Allowed values:NONESTARTEND