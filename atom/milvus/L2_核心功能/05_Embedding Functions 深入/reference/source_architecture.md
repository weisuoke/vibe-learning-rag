# Milvus Embedding Functions - Source Code Architecture Analysis

**Source**: Milvus 2.6 Source Code
**Analysis Date**: 2026-02-24
**Files Analyzed**: 12 core files from `sourcecode/milvus/internal/util/function/embedding/`

---

## 1. Core Architecture Overview

### 1.1 Provider Pattern Design

Milvus implements a **provider pattern** for embedding functions, allowing pluggable embedding service integrations.

**Key Interface**: `textEmbeddingProvider`

```go
type textEmbeddingProvider interface {
    MaxBatch() int
    CallEmbedding(ctx context.Context, texts []string, mode models.TextEmbeddingMode) (any, error)
    FieldDim() int64
}
```

**Source**: `text_embedding_function.go:82-86`

### 1.2 Function Executor Architecture

**File**: `function_executor.go`

The `FunctionExecutor` manages multiple embedding functions and handles:
- **Parallel execution** of multiple functions using goroutines
- **Batch size validation** before processing
- **Error aggregation** from concurrent operations
- **Insert, Search, and BulkInsert** operations

**Key Methods**:
- `ProcessInsert()` - Handles insert operations with parallel function execution
- `ProcessSearch()` - Handles search operations (both regular and advanced)
- `ProcessBulkInsert()` - Handles bulk insert operations serially

**Source**: `function_executor.go:58-318`

### 1.3 Text Embedding Function

**File**: `text_embedding_function.go`

The `TextEmbeddingFunction` is the main wrapper that:
- Validates input/output field types
- Manages embedding provider lifecycle
- Handles two embedding modes: `InsertMode` and `SearchMode`
- Supports both `FloatVector` and `Int8Vector` output types

**Supported Providers** (10 total):
1. `openai` - OpenAI Embeddings API
2. `azure_openai` - Azure OpenAI Service
3. `dashscope` - Alibaba DashScope
4. `bedrock` - AWS Bedrock
5. `vertexai` - Google VertexAI
6. `voyageai` - VoyageAI
7. `cohere` - Cohere Embed API
8. `siliconflow` - SiliconFlow
9. `tei` - Hugging Face Text Embeddings Inference
10. `zilliz` - Zilliz Cloud Pipelines

**Source**: `text_embedding_function.go:120-144`

---

## 2. Provider-Specific Implementation Details

### 2.1 OpenAI Provider

**File**: `openai_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier (e.g., "text-embedding-3-small")
- `dim` - Optional dimension parameter for models that support it
- `user` - Optional user identifier for tracking
- `api_key` - Authentication (from credentials or env: `OPENAI_API_KEY`)
- `url` - Optional custom endpoint (default: `https://api.openai.com/v1/embeddings`)

**Batch Processing**:
- `maxBatch = 128` (base)
- Actual max batch = `batchFactor * 128` (configurable via Milvus config)
- Processes in chunks of 128 texts per API call

**Azure OpenAI Variant**:
- Uses `AZURE_OPENAI_API_KEY` environment variable
- Requires `resource_name` parameter
- URL format: `https://{resource_name}.openai.azure.com`

**Source**: `openai_embedding_provider.go:34-179`

---

### 2.2 AWS Bedrock Provider

**File**: `bedrock_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Bedrock model ID (e.g., "amazon.titan-embed-text-v1")
- `dim` - Optional dimension parameter
- `normalize` - Boolean flag for normalization (default: true)
- `region` - AWS region (required)
- `aws_access_key_id` - AWS credentials (from credentials or env: `AWS_ACCESS_KEY_ID`)
- `aws_secret_access_key` - AWS secret key (env: `AWS_SECRET_ACCESS_KEY`)

**Batch Processing**:
- `maxBatch = 1` (NO batch support)
- Processes one text at a time
- Each text requires a separate API call

**API Request Format**:
```go
type BedRockRequest struct {
    InputText  string `json:"inputText"`
    Dimensions int64  `json:"dimensions,omitempty"`
    Normalize  bool   `json:"normalize,omitempty"`
}
```

**Source**: `bedrock_embedding_provider.go:45-235`

---

### 2.3 VoyageAI Provider

**File**: `voyageai_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier (e.g., "voyage-3-large")
- `dim` - Optional dimension (supported by voyage-3-large and voyage-code-3: 256, 512, 1024, 2048)
- `truncate` - Boolean flag for truncation (default: false)
- `api_key` - Authentication (env: `VOYAGEAI_API_KEY`)
- `url` - Optional custom endpoint (default: `https://api.voyageai.com/v1/embeddings`)

**Batch Processing**:
- `maxBatch = 128`
- Actual max batch = `batchFactor * 128`

**Output Types**:
- Supports both `float32` and `int8` embeddings
- `output_type` parameter: "float" or "int8"

**Input Type Differentiation**:
- `InsertMode` → `input_type = "document"`
- `SearchMode` → `input_type = "query"`

**Source**: `voyageai_embedding_provider.go:34-178`

---

### 2.4 Google VertexAI Provider

**File**: `vertexai_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier (e.g., "text-embedding-004")
- `dim` - Optional dimension parameter
- `location` - GCP region (default: "us-central1")
- `project_id` - GCP project ID (required)
- `task_type` - Task type: "DOC_RETRIEVAL", "CODE_RETRIEVAL", or "STS" (default: "DOC_RETRIEVAL")
- Credentials: GCP service account JSON (from credentials or env: `GOOGLE_APPLICATION_CREDENTIALS`)

**Batch Processing**:
- `maxBatch = 128`
- Actual max batch = `batchFactor * 128`

**Task Type Mapping**:
- `InsertMode` + `DOC_RETRIEVAL` → `"RETRIEVAL_DOCUMENT"`
- `SearchMode` + `DOC_RETRIEVAL` → `"RETRIEVAL_QUERY"`
- `InsertMode` + `CODE_RETRIEVAL` → `"RETRIEVAL_DOCUMENT"`
- `SearchMode` + `CODE_RETRIEVAL` → `"CODE_RETRIEVAL_QUERY"`
- `STS` → `"SEMANTIC_SIMILARITY"` (both modes)

**URL Format**:
```
https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:predict
```

**Source**: `vertexai_embedding_provider.go:75-251`

---

### 2.5 Cohere Provider

**File**: `cohere_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier (e.g., "embed-english-v3.0")
- `dim` - Optional dimension parameter
- `truncate` - Truncation strategy: "NONE", "START", or "END" (default: "END")
- `api_key` - Authentication (env: `COHERE_API_KEY`)
- `url` - Optional custom endpoint (default: `https://api.cohere.com/v2/embed`)

**Batch Processing**:
- `maxBatch = 96`
- Actual max batch = `batchFactor * 96`

**Output Types**:
- Supports both `float32` and `int8` embeddings
- `output_type` parameter: "float" or "int8"

**Input Type Differentiation**:
- `InsertMode` → `input_type = "search_document"`
- `SearchMode` → `input_type = "search_query"`
- **Note**: v2.0 models don't support input_type parameter

**Source**: `cohere_embedding_provider.go:33-180`

---

### 2.6 Zilliz Cloud Pipelines Provider

**File**: `zilliz_embedding_provider.go`

**Configuration Parameters**:
- `model_deployment_id` - Zilliz Cloud pipeline deployment ID (required)
- Additional model-specific parameters passed through `modelParams` map
- Credentials: Managed internally via `extraInfo.ClusterID` and `extraInfo.DBName`

**Batch Processing**:
- `maxBatch = 64`
- Actual max batch = `batchFactor * 64`

**Input Type Differentiation**:
- `InsertMode` → `input_type = "document"`
- `SearchMode` → `input_type = "query"`

**Special Features**:
- Integrated with Zilliz Cloud infrastructure
- No external API key required (uses internal authentication)
- Model parameters are flexible and passed through to the pipeline

**Source**: `zilliz_embedding_provider.go:31-105`

---

### 2.7 Alibaba DashScope Provider

**File**: `ali_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier (e.g., "text-embedding-v3")
- `dim` - Optional dimension parameter
- `api_key` - Authentication (env: `DASHSCOPE_API_KEY`)
- `url` - Optional custom endpoint (default: `https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`)

**Batch Processing**:
- `maxBatch = 25` (default)
- `maxBatch = 6` (for "text-embedding-v3" model)
- Actual max batch = `batchFactor * maxBatch`

**Output Type**:
- Only supports `float32` embeddings
- `output_type = "dense"` (fixed)

**Input Type Differentiation**:
- `InsertMode` → `text_type = "document"`
- `SearchMode` → `text_type = "query"`

**Source**: `ali_embedding_provider.go:33-146`

---

### 2.8 SiliconFlow Provider

**File**: `siliconflow_embedding_provider.go`

**Configuration Parameters**:
- `model_name` - Model identifier
- `dim` - Optional dimension parameter
- `api_key` - Authentication (env: `SILICONFLOW_API_KEY`)
- `url` - Optional custom endpoint (default: `https://api.siliconflow.cn/v1/embeddings`)

**Batch Processing**:
- `maxBatch = 32`
- Actual max batch = `batchFactor * 32`

**Output Type**:
- Only supports `float32` embeddings
- `encoding_format = "float"` (fixed)

**Special Features**:
- No input type differentiation (same behavior for insert and search)
- Simpler configuration compared to other providers

**Source**: `siliconflow_embedding_provider.go:33-126`

---

### 2.9 Hugging Face TEI Provider

**File**: `tei_embedding_provider.go`

**Configuration Parameters**:
- `endpoint` - TEI server endpoint URL (required)
- `ingestion_prompt` - Optional prompt prefix for document ingestion
- `search_prompt` - Optional prompt prefix for search queries
- `truncate` - Boolean flag for truncation (default: false)
- `truncation_direction` - "Left" or "Right" (default: "Right")
- `max_client_batch_size` - Client-side batch size (default: 32)
- `api_key` - Optional authentication

**Batch Processing**:
- `maxBatch = 32` (default, configurable via `max_client_batch_size`)
- Actual max batch = `batchFactor * maxBatch`

**Prompt Differentiation**:
- `InsertMode` → Uses `ingestion_prompt`
- `SearchMode` → Uses `search_prompt`

**Special Features**:
- Self-hosted embedding server support
- Flexible prompt engineering for different modes
- Configurable truncation behavior

**Source**: `tei_embedding_provider.go:34-149`

---

## 3. Common Patterns and Design Principles

### 3.1 Batch Processing Strategy

All providers follow a consistent batch processing pattern:

```go
for i := 0; i < numRows; i += provider.maxBatch {
    end := i + provider.maxBatch
    if end > numRows {
        end = numRows
    }
    // Process batch [i:end]
}
```

**Batch Factor Multiplier**:
- Configured globally via `paramtable.Get().FunctionCfg.GetBatchFactor()`
- Actual max batch = `batchFactor * provider.maxBatch`
- Allows dynamic scaling without code changes

### 3.2 Credential Management

**Priority Order** (highest to lowest):
1. Function schema parameters (`credential` parameter)
2. Milvus YAML configuration (`milvus.yaml`)
3. Environment variables

**Implementation**:
```go
func ParseAKAndURL(credentials *credentials.Credentials,
                   params []*commonpb.KeyValuePair,
                   confParams map[string]string,
                   envKey string,
                   extraInfo *models.ModelExtraInfo) (string, string, error)
```

### 3.3 Error Handling

**Validation Checks**:
- Empty string detection in input texts
- Batch size validation (numRows <= MaxBatch())
- Dimension mismatch detection
- Response count validation

**Example**:
```go
if hasEmptyString(texts) {
    return nil, errors.New("There is an empty string in the input data, TextEmbedding function does not support empty text")
}
```

### 3.4 Mode-Based Behavior

**Two Embedding Modes**:
1. `InsertMode` - For document ingestion
2. `SearchMode` - For query processing

**Provider-Specific Adaptations**:
- **VoyageAI**: `input_type = "document"` vs `"query"`
- **Cohere**: `input_type = "search_document"` vs `"search_query"`
- **VertexAI**: `task_type = "RETRIEVAL_DOCUMENT"` vs `"RETRIEVAL_QUERY"`
- **TEI**: Uses `ingestion_prompt` vs `search_prompt`

---

## 4. Performance Characteristics

### 4.1 Batch Size Comparison

| Provider | Base Max Batch | Multiplier | Effective Max Batch |
|----------|----------------|------------|---------------------|
| OpenAI | 128 | batchFactor | 128 × batchFactor |
| Azure OpenAI | 128 | batchFactor | 128 × batchFactor |
| VoyageAI | 128 | batchFactor | 128 × batchFactor |
| VertexAI | 128 | batchFactor | 128 × batchFactor |
| Cohere | 96 | batchFactor | 96 × batchFactor |
| Zilliz | 64 | batchFactor | 64 × batchFactor |
| SiliconFlow | 32 | batchFactor | 32 × batchFactor |
| TEI | 32 (configurable) | batchFactor | 32 × batchFactor |
| DashScope | 25 (or 6) | batchFactor | 25 × batchFactor |
| Bedrock | 1 | batchFactor | 1 × batchFactor |

**Key Insight**: Bedrock has the lowest throughput due to no batch support.

### 4.2 Parallel Execution

**FunctionExecutor** uses goroutines for parallel processing:

```go
for _, runner := range executor.runners {
    wg.Add(1)
    go func(runner Runner) {
        defer wg.Done()
        data, err := executor.processSingleFunction(ctx, runner, msg)
        // ...
    }(runner)
}
```

**Benefits**:
- Multiple embedding functions can run concurrently
- Reduces total latency when using multiple providers
- Error isolation per function

---

## 5. Integration Points

### 5.1 Schema Validation

**Function**: `ValidateFunctions()`

Validates embedding functions during collection creation:
- Checks provider availability
- Validates credentials
- Tests embedding generation with "check" text
- Verifies dimension consistency

**Source**: `function_executor.go:104-123`

### 5.2 Pipeline Integration

**Insert Pipeline**:
1. `FunctionExecutor.ProcessInsert()` receives `InsertMsg`
2. Extracts input fields based on function schema
3. Calls `runner.ProcessInsert()` for each function
4. Appends output embeddings to message

**Search Pipeline**:
1. `FunctionExecutor.ProcessSearch()` receives `SearchRequest`
2. Extracts placeholder group (query texts)
3. Calls `runner.ProcessSearch()` to generate query embeddings
4. Replaces placeholder with vector embeddings

**Source**: `function_executor.go:168-286`

---

## 6. Key Takeaways for Documentation

### 6.1 Provider Selection Criteria

**Choose based on**:
1. **Batch size requirements** - High throughput → OpenAI/VoyageAI/VertexAI
2. **Output type needs** - Int8 quantization → VoyageAI/Cohere
3. **Cloud ecosystem** - AWS → Bedrock, GCP → VertexAI, Alibaba → DashScope
4. **Self-hosting** - TEI for on-premise deployments
5. **Cost optimization** - Zilliz Cloud Pipelines for integrated solution

### 6.2 Common Pitfalls

1. **Empty strings** - All providers reject empty input texts
2. **Batch size limits** - Exceeding MaxBatch() causes errors
3. **Dimension mismatches** - Field dimension must match model output
4. **Credential priority** - Function params override YAML override env vars
5. **Mode awareness** - Some providers behave differently for insert vs search

### 6.3 Advanced Features

1. **Dimension control** - OpenAI, VoyageAI, Cohere, VertexAI, DashScope support custom dimensions
2. **Quantization** - VoyageAI and Cohere support int8 output
3. **Prompt engineering** - TEI supports custom prompts per mode
4. **Task specialization** - VertexAI supports DOC_RETRIEVAL, CODE_RETRIEVAL, STS
5. **Truncation strategies** - Cohere and TEI offer truncation control

---

## 7. Source File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `function_base.go` | 84 | Base function structure |
| `function_executor.go` | 318 | Function execution orchestration |
| `text_embedding_function.go` | 351 | Main embedding function wrapper |
| `openai_embedding_provider.go` | 179 | OpenAI/Azure OpenAI implementation |
| `bedrock_embedding_provider.go` | 235 | AWS Bedrock implementation |
| `voyageai_embedding_provider.go` | 178 | VoyageAI implementation |
| `vertexai_embedding_provider.go` | 251 | Google VertexAI implementation |
| `cohere_embedding_provider.go` | 180 | Cohere implementation |
| `zilliz_embedding_provider.go` | 105 | Zilliz Cloud Pipelines implementation |
| `ali_embedding_provider.go` | 146 | Alibaba DashScope implementation |
| `siliconflow_embedding_provider.go` | 126 | SiliconFlow implementation |
| `tei_embedding_provider.go` | 149 | Hugging Face TEI implementation |

**Total Lines Analyzed**: ~2,302 lines of Go code

---

**Analysis Complete**: 2026-02-24
**Next Steps**: Query official documentation via Context7, gather community examples via Grok-mcp
