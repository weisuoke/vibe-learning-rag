# Embedding Functions Documentation (Milvus 2.6)

## Source 1: Embedding Function Overview

---
source: https://milvus.io/docs/embedding-function-overview.md
title: Embedding Function Overview | Milvus Documentation
fetched_at: 2026-02-21
---

# Embedding Function Overview

**Compatible with Milvus 2.6.x**

The Function module in Milvus allows you to transform raw text data into vector embeddings by automatically calling external embedding service providers (like OpenAI, AWS Bedrock, Google Vertex AI, etc.). With the Function module, you no longer need to manually interface with embedding APIs—Milvus handles the entire process of sending requests to providers, receiving embeddings, and storing them in your collections.

For semantic search, you need to provide only raw query data, not a query vector. Milvus generates the query vector with the same model you used for ingestion, compares it to the stored vectors, and returns the most relevant results.

## Limits

- Any input field that the Function module embeds must always contain a value; if a null is supplied, the module will throw an error.
- The Function module processes only fields that are explicitly defined in the collection schema; it does not generate embeddings for dynamic fields.
- Input fields to be embedded must be of the VARCHAR type.
- The Function module can embed an input field to:
  - FLOAT_VECTOR
  - INT8_VECTOR
- Conversions to BINARY_VECTOR, FLOAT16_VECTOR, or BFLOAT16_VECTOR are not supported.

## Supported embedding service providers

| Provider                               | Typical Models                        | Embedding Type          | Authentication Method              |
|----------------------------------------|---------------------------------------|-------------------------|-------------------------------------|
| [OpenAI](/docs/openai.md)              | text-embedding-3-*                    | FLOAT_VECTOR            | API key                            |
| [Azure OpenAI](/docs/azure-openai.md)  | Deployment-based                      | FLOAT_VECTOR            | API key                            |
| [DashScope](/docs/dashscope.md)        | text-embedding-v3                     | FLOAT_VECTOR            | API key                            |
| [Bedrock](/docs/bedrock.md)            | amazon.titan-embed-text-v2            | FLOAT_VECTOR            | AK/SK pair                         |
| [Vertex AI](/docs/vertex-ai.md)        | text-embedding-005                    | FLOAT_VECTOR            | GCP service account JSON credential|
| [Voyage AI](/docs/voyage-ai.md)        | voyage-3, voyage-lite-02              | FLOAT_VECTOR / INT8_VECTOR | API key                         |
| [Cohere](/docs/cohere.md)              | embed-english-v3.0                    | FLOAT_VECTOR / INT8_VECTOR | API key                         |
| [SiliconFlow](/docs/siliconflow.md)    | BAAI/bge-large-zh-v1.5                | FLOAT_VECTOR            | API key                            |
| [Hugging Face](/docs/hugging-face-tei.md) | Any TEI-served model               | FLOAT_VECTOR            | Optional API key                   |

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
- **Environment variables**: For details on configuring credentials via environment variables, see the embedding service provider's documentation (for example, [OpenAI](/docs/openai.md) or [Azure OpenAI](/docs/azure-openai.md)).

### Step 1: Add credentials to Milvus configuration file

```yaml
# milvus.yaml credential store section
# This section defines all your authentication credentials for external embedding providers
# Each credential gets a unique name (e.g., aksk1, apikey1) that you'll reference elsewhere
credential:
  # For AWS Bedrock or services using access/secret key pairs
  # 'aksk1' is just an example name - you can choose any meaningful identifier
  aksk1:
    access_key_id: <YOUR_AK>
    secret_access_key: <YOUR_SK>

  # For OpenAI, Voyage AI, or other API key-based services
  # 'apikey1' is a custom name you choose to identify this credential
  apikey1:
    apikey: <YOUR_API_KEY>

  # For Google Vertex AI using service account credentials
  # 'gcp1' is an example name for your Google Cloud credentials
  gcp1:
    credential_json: <BASE64_OF_JSON>
```

### Step 2: Configure provider settings

```yaml
function:
  textEmbedding:
    providers:
      openai:                         # calls OpenAI
        credential: apikey1           # Reference to the credential label
        # url:                        # (optional) custom url

      bedrock:                        # calls AWS Bedrock
        credential: aksk1             # Reference to the credential label
        region: us-east-2

      vertexai:                       # calls Google Vertex AI
        credential: gcp1              # Reference to the credential label
        # url:                        # (optional) custom url

      tei:                            # Built-in Tiny Embedding model
        enable: true                  # Whether to enable TEI model service
```

For more information on how to apply Milvus configuration, refer to [Configure Milvus on the Fly](/docs/dynamic_config.md).

## Use embedding function

Once credentials are configured in your Milvus configuration file, follow these steps to define and use embedding functions.

### Step 1: Define schema fields

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

# Initialize Milvus client
client = MilvusClient(
    uri="http://localhost:19530",
)

# Create a new schema for the collection
schema = client.create_schema()

# Add primary field "id"
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)

# Add scalar field "document" for storing textual data
schema.add_field("document", DataType.VARCHAR, max_length=9000)

# Add vector field "dense" for storing embeddings.
# IMPORTANT: Set dim to match the exact output dimension of the embedding model.
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)
```

### Step 2: Add embedding function to schema

```python
# Define embedding function (example: OpenAI provider)
text_embedding_function = Function(
    name="openai_embedding",                  # Unique identifier
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["document"],
    output_field_names=["dense"],
    params={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        # "credential": "apikey1",            # Optional
        # "dim": "1536",                      # Optional
        # "user": "user123"                   # Optional
    }
)

# Add the embedding function to your schema
schema.add_function(text_embedding_function)
```

#### Function 参数说明

| Parameter          | Description                                                                 | Example Value          |
|--------------------|-----------------------------------------------------------------------------|------------------------|
| name               | Unique identifier for the embedding function within Milvus.                | "openai_embedding"     |
| function_type      | Type of function used.                                                      | FunctionType.TEXTEMBEDDING |
| input_field_names  | Scalar field containing raw data to be embedded.                            | ["document"]           |
| output_field_names | Vector field for storing generated embeddings.                              | ["dense"]              |
| params             | Dictionary containing embedding configurations.                             | {...}                  |
| provider           | The embedding model provider.                                               | "openai"               |
| model_name         | Specifies which embedding model to use.                                     | "text-embedding-3-small" |
| credential         | The label of a credential defined in milvus.yaml.                           | "apikey1"              |
| dim                | The number of dimensions for the output embeddings (optional shorten).     | "1536"                 |
| user               | A user-level identifier for tracking API usage.                             | "user123"              |

### Step 3: Configure index

```python
# Prepare index parameters
index_params = client.prepare_index_params()

# Add AUTOINDEX
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
# Insert sample documents
client.insert('demo', [
    {'id': 1, 'document': 'Milvus simplifies semantic search through embeddings.'},
    {'id': 2, 'document': 'Vector embeddings convert text into searchable numeric data.'},
    {'id': 3, 'document': 'Semantic search helps users find relevant information quickly.'},
])
```

---

## Source 2: Data-in, Data-out Blog Post

---
source: https://milvus.io/blog/data-in-and-data-out-in-milvus-2-6.md
title: Introducing the Embedding Function: How Milvus 2.6 Streamlines Vectorization and Semantic Search
fetched_at: 2026-02-21
---

# Introducing the Embedding Function: How Milvus 2.6 Streamlines Vectorization and Semantic Search

**December 02, 2025**
**Xuqi Yang**

If you've ever built a vector search application, you already know the workflow a little too well. Before any data can be stored, it must first be transformed into vectors using an embedding model, cleaned and formatted, and then finally ingested into your vector database. Every query goes through the same process as well: embed the input, run a similarity search, then map the resulting IDs back to your original documents or records. It works — but it creates a distributed tangle of preprocessing scripts, embedding pipelines, and glue code that you have to maintain.

[Milvus](https://milvus.io/), a high-performance open-source vector database, now takes a major step toward simplifying all of that. [Milvus 2.6](https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md) introduces the **Data-in, Data-out feature (also known as the** [Embedding Function](https://milvus.io/docs/embedding-function-overview.md#Embedding-Function-Overview) **)**, a built-in embedding capability that connects directly to major model providers such as OpenAI, AWS Bedrock, Google Vertex AI, and Hugging Face. Instead of managing your own embedding infrastructure, Milvus can now call these models for you. You can also insert and query using raw text — and soon other data types — while Milvus automatically handles vectorization at write and query time.

In the rest of this post, we'll take a closer look at how Data-in, Data-out works under the hood, how to configure providers and embedding functions, and how you can use it to streamline your vector search workflows end-to-end.

## What is Data-in, Data-out?

Data-in, Data-out in Milvus 2.6 is built on the new Function module — a framework that enables Milvus to handle data transformation and embedding generation internally, without any external preprocessing services. (You can follow the design proposal in [GitHub issue #35856](https://github.com/milvus-io/milvus/issues/35856).) With this module, Milvus can take raw input data, call an embedding provider directly, and automatically write the resulting vectors into your collection.

At a high level, the **Function** module turns embedding generation into a native database capability. Instead of running separate embedding pipelines, background workers, or reranker services, Milvus now sends requests to your configured provider, retrieves embeddings, and stores them alongside your data — all inside the ingestion path. This removes the operational overhead of managing your own embedding infrastructure.

Data-in, Data-out introduces three major improvements to the Milvus workflow:

- **Insert raw data directly** – You can now insert unprocessed text, images, or other data types directly into Milvus. No need to convert them into vectors in advance.
- **Configure one embedding function** – Once you configure an embedding model in Milvus, it automatically manages the entire embedding process. Milvus integrates seamlessly with a range of model providers, including OpenAI, AWS Bedrock, Google Vertex AI, Cohere, and Hugging Face.
- **Query with raw inputs** – You can now perform semantic search using raw text or other content-based queries. Milvus uses the same configured model to generate embeddings on the fly, perform similarity search, and return relevant results.

In short, Milvus now automatically embeds — and optionally reranks — your data. Vectorization becomes a built-in database function, eliminating the need for external embedding services or custom preprocessing logic.

## How Data-in, Data-out Works

The Data-in, Data-out workflow can be broken down into six main steps:

1. **Input Data** – The user inserts raw data — such as text, images, or other content types — directly into Milvus without performing any external preprocessing.
2. **Generate Embeddings** – The Function module automatically invokes the configured embedding model through its third-party API, converting the raw input into vector embeddings in real time.
3. **Store Embeddings** – Milvus writes the generated embeddings into the designated vector field within your collection, where they become available for similarity search operations.
4. **Submit a Query** – The user issues a raw-text or content-based query to Milvus, just as with the input stage.
5. **Semantic Search** – Milvus embeds the query using the same configured model, runs a similarity search over the stored vectors, and determines the closest semantic matches.
6. **Return Results** – Milvus returns the top-k most similar results — mapped back to their original data — directly to the application.

## How to Configure Data-in, Data-out

### Prerequisites

- Install the latest version of **Milvus 2.6**.
- Prepare your embedding API key from a supported provider (e.g., OpenAI, AWS Bedrock, or Cohere). In this example, we'll use **Cohere** as the embedding provider.

### Modify the `milvus.yaml` Configuration

If you are running Milvus with **Docker Compose**, you'll need to modify the `milvus.yaml` file to enable the Function module. You can refer to the official documentation for guidance: [Configure Milvus with Docker Compose](https://milvus.io/docs/configure-docker.md?tab=component#Download-a-configuration-file) (Instructions for other deployment methods can also be found here).

In the configuration file, locate the sections `credential` and `function`.

Then, update the fields `apikey1.apikey` and `providers.cohere`.

```yaml
...
credential:
  aksk1:
    access_key_id:  # Your access_key_id
    secret_access_key:  # Your secret_access_key
  apikey1:
    apikey: "***********************" # Edit this section
  gcp1:
    credential_json:  # base64 based gcp credential data

function:
  textEmbedding:
    providers:
      ...
      cohere: # Edit the section below
        credential:  apikey1 # The name in the crendential configuration item
        enable: true # Whether to enable cohere model service
        url:  "https://api.cohere.com/v2/embed" # Your cohere embedding url, Default is the official embedding url
      ...
...
```

Once you've made these changes, restart Milvus to apply the updated configuration.

---

## Source 3: OpenAI Provider Configuration

---
source: https://milvus.io/docs/openai.md
title: OpenAI | Milvus Documentation
fetched_at: 2026-02-21
---

# OpenAI Compatible with Milvus 2.6.x

Use an OpenAI embedding model with Milvus by choosing a model and configuring Milvus with your OpenAI API key.

## Choose an embedding model

Milvus supports all embedding models provided by OpenAI. Below are the currently available OpenAI embedding models for quick reference:

| Model Name            | Dimensions                                      | Max Tokens | Description                                                                                   |
|-----------------------|-------------------------------------------------|------------|-----------------------------------------------------------------------------------------------|
| text-embedding-3-small | Default: 1,536 (can be shortened to a dimension size below 1,536) | 8,191      | Ideal for cost-sensitive and scalable semantic search—offers strong performance at a lower price point. |
| text-embedding-3-large | Default: 3,072 (can be shortened to a dimension size below 3,072) | 8,191      | Best for applications demanding enhanced retrieval accuracy and richer semantic representations. |
| text-embedding-ada-002 | Fixed: 1,536 (cannot be shortened)              | 8,191      | A previous-generation model suited for legacy pipelines or scenarios requiring backward compatibility. |

The third generation embedding models (**text-embedding-3**) support reducing the size of the embedding via a `dim` parameter. Typically larger embeddings are more expensive from a compute, memory, and storage perspective. Being able to adjust the number of dimensions allows more control over overall cost and performance. For more details about each model, refer to [Embedding models](https://platform.openai.com/docs/guides/embeddings#embedding-models) and [OpenAI announcement blog post](https://openai.com/blog/new-embedding-models-and-api-updates).

## Configure credentials

Milvus must know your OpenAI API key before it can request embeddings. Milvus provides two methods to configure credentials:

**Configuration file (recommended):** Store the API key in milvus.yaml so every restart and node picks it up automatically.
**Environment variables:** Inject the key at deploy time—ideal for Docker Compose.

Choose one of the two methods below—the configuration file is easier to maintain on bare-metal and VMs, while the env-var route fits container workflows.

If an API key for the same provider is present in both the configuration file and an environment variable, Milvus always uses the value in `milvus.yaml` and ignores the environment variable.

### Option 1: Configuration file (recommended & higher priority)

Keep your API keys in `milvus.yaml`; Milvus reads them at startup and overrides any environment variable for the same provider.

1. **Declare your keys under credential:**
   You may list one or many API keys—give each a label you invent and will reference later.

   ```yaml
   # milvus.yaml
   credential:
     apikey_dev:           # dev environment
       apikey: <YOUR_DEV_KEY>
     apikey_prod:          # production environment
       apikey: <YOUR_PROD_KEY>
   ```

   Putting the API keys here makes them persistent across restarts and lets you switch keys just by changing a label.

2. **Tell Milvus which key to use for OpenAI calls**
   In the same file, point the OpenAI provider at the label you want it to use.

   ```yaml
   function:
     textEmbedding:
       providers:
         openai:
           credential: apikey_dev     # ← choose any label you defined above
           # url: https://api.openai.com/v1/embeddings   # (optional) custom url
   ```

   This binds a specific key to every request Milvus sends to the OpenAI embeddings endpoint.

### Option 2: Environment variable

Use this method when you run Milvus with Docker Compose and prefer to keep secrets out of files and images.

Milvus falls back to the environment variable only if no key for the provider is found in `milvus.yaml`.

| Variable                  | Required | Description                                                                 |
|---------------------------|----------|-----------------------------------------------------------------------------|
| MILVUSAI_OPENAI_API_KEY   | Yes      | Makes the OpenAI key available inside each Milvus container *(ignored when a key for OpenAI exists in milvus.yaml)* |

In your **docker-compose.yaml** file, set the `MILVUSAI_OPENAI_API_KEY` environment variable.

```yaml
# docker-compose.yaml (standalone service section)
standalone:
  # ... other configurations ...
  environment:
    # ... other environment variables ...
    # Set the environment variable pointing to the OpenAI API key inside the container
    MILVUSAI_OPENAI_API_KEY: <MILVUSAI_OPENAI_API_KEY>
```

The `environment:` block injects the key only into the Milvus container, leaving your host OS untouched. For details, refer to [Configure Milvus with Docker Compose](/docs/configure-docker.md#Configure-Milvus-with-Docker-Compose).

## Use embedding function

Once credentials are configured, follow these steps to define and use embedding functions.

### Step 1: Define schema fields

To use an embedding function, create a collection with a specific schema. This schema must include at least three necessary fields:

- The primary field that uniquely identifies each entity in a collection.
- A scalar field that stores raw data to be embedded.
- A vector field reserved to store vector embeddings that the function will generate for the scalar field.

The following example defines a schema with one scalar field `"document"` for storing textual data and one vector field `"dense"` for storing embeddings to be generated by the Function module. Remember to set the vector dimension (`dim`) to match the output of your chosen embedding model.

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

# Initialize Milvus client
client = MilvusClient(
    uri="http://localhost:19530",
)

# Create a new schema for the collection
schema = client.create_schema()

# Add primary field "id"
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)

# Add scalar field "document" for storing textual data
schema.add_field("document", DataType.VARCHAR, max_length=9000)

# Add vector field "dense" for storing embeddings.
# IMPORTANT: Set dim to match the exact output dimension of the embedding model.
# For instance, OpenAI's text-embedding-3-small model outputs 1536-dimensional vectors.
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)
```

### Step 2: Add embedding function to schema

The Function module in Milvus automatically converts raw data stored in a scalar field into embeddings and stores them into the explicitly defined vector field.

The example below adds a Function module (`openai_embedding`) that converts the scalar field `"document"` into embeddings, storing the resulting vectors in the `"dense"` vector field defined earlier.

Once you have defined your embedding function, add it to your collection schema. This instructs Milvus to use the specified embedding function to process and store embeddings from your text data.

```python
# Define embedding function (example: OpenAI provider)
text_embedding_function = Function(
    name="openai_embedding",                        # Unique identifier for this embedding function
    function_type=FunctionType.TEXTEMBEDDING,       # Type of embedding function
    input_field_names=["document"],                 # Scalar field to embed
    output_field_names=["dense"],                   # Vector field to store embeddings
    params={                                        # Provider-specific configuration (highest priority)
        "provider": "openai",                       # Embedding model provider
        "model_name": "text-embedding-3-small",     # Embedding model
        # Optional parameters:
        # "credential": "apikey_dev",               # Optional: Credential label specified in milvus.yaml
        # "dim": "1536",                            # Optional: Shorten the output vector dimension
        # "user": "user123"                         # Optional: identifier for API tracking
    }
)

# Add the embedding function to your schema
schema.add_function(text_embedding_function)
```

After configuring the embedding function, refer to the [Function Overview](/docs/embedding-function-overview.md) for additional guidance on index configuration, data insertion examples, and semantic search operations.
