# Milvus 2.6 Embedding Functions - Source Code Analysis

**Analysis Date**: 2026-02-24
**Source**: `sourcecode/milvus/internal/util/function/embedding/`
**Total Files Analyzed**: 10 provider implementations + 4 core files

---

## Architecture Overview

### Core Components

1. **FunctionExecutor** (`function_executor.go`)
   - Central orchestrator for all embedding functions
   - Manages multiple runners concurrently
   - Handles Insert, Search, and BulkInsert operations
   - Implements batch processing with goroutines

2. **TextEmbeddingFunction** (`text_embedding_function.go`)
   - Main interface for text embedding operations
   - Supports 10 providers: OpenAI, Azure OpenAI, Cohere, Bedrock, VertexAI, VoyageAI, AliDashScope, SiliconFlow, TEI, Zilliz
   - Validates embedding dimensions and data types
   - Handles both FloatVector and Int8Vector outputs

3. **FunctionBase** (`function_base.go`)
   - Base structure for all function types
   - Manages schema, output fields, and provider information
   - Provides common functionality across function types

---

## Provider Implementations

### 1. OpenAI Provider (`openai_embedding_provider.go`)

**Key Features**:
- Supports both OpenAI and Azure OpenAI
- Default endpoint: `https://api.openai.com/v1/embeddings`
- Max batch size: 128 (configurable with BatchFactor)
- Timeout: 30 seconds

**Configuration Parameters**:
```go
- model_name: Model identifier (e.g., "text-embedding-3-small")
- dim: Optional dimension parameter for models that support it
- user: Optional user identifier for tracking
- api_key: From credentials or OPENAI_API_KEY env
- url: Custom endpoint URL (optional)
```

**Azure OpenAI Specifics**:
- Requires `resource_name` parameter
- Default URL format: `https://{resource_name}.openai.azure.com`
- Uses AZURE_OPENAI_API_KEY env variable

**Code Pattern**:
```go
// Batch processing with automatic chunking
for i := 0; i < numRows; i += provider.maxBatch {
    end := i + provider.maxBatch
    if end > numRows {
        end = numRows
    }
    resp, err := provider.client.Embedding(...)
    // Validate dimensions and append results
}
```

---

### 2. Cohere Provider (`cohere_embedding_provider.go`)

**Key Features**:
- Default endpoint: `https://api.cohere.com/v2/embed`
- Max batch size: 96
- Supports both float32 and int8 output types
- Input type differentiation for search vs. document

**Configuration Parameters**:
```go
- model_name: Model identifier
- dim: Optional dimension parameter
- truncate: "NONE", "START", or "END" (default: "END")
- output_type: "float" or "int8" (auto-detected from field type)
```

**Input Type Logic**:
```go
func (provider *CohereEmbeddingProvider) getInputType(mode models.TextEmbeddingMode) string {
    if strings.HasSuffix(provider.modelName, "v2.0") {
        return "" // v2 models don't support instructor
    }
    if mode == models.InsertMode {
        return "search_document"
    }
    return "search_query"
}
```

**Special Handling**:
- V2 models don't support input_type parameter
- V3+ models require input_type for optimal performance
- Supports EmbdResult wrapper for type-safe handling

---

### 3. AWS Bedrock Provider (`bedrock_embedding_provider.go`)

**Key Features**:
- Uses AWS SDK v2 for authentication
- Max batch size: 1 (no native batch support)
- Supports dimension parameter and normalization
- Region-based endpoint configuration

**Configuration Parameters**:
```go
- model_name: Bedrock model ID
- region: AWS region (required)
- dim: Optional dimension parameter
- normalize: true/false (default: true)
- aws_access_key_id: From credentials or env
- aws_secret_access_key: From credentials or env
```

**Authentication Priority**:
1. Function parameters (credential name)
2. milvus.yaml configuration
3. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

**Request/Response Format**:
```go
type BedRockRequest struct {
    InputText  string `json:"inputText"`
    Dimensions int64  `json:"dimensions,omitempty"`
    Normalize  bool   `json:"normalize,omitempty"`
}

type BedRockResponse struct {
    Embedding           []float32 `json:"embedding"`
    InputTextTokenCount int       `json:"inputTextTokenCount"`
}
```

**Performance Note**:
- Processes one text at a time (maxBatch=1)
- Milvus implements small batch support on client side
- Suitable for low-latency, single-query scenarios

---

### 4. Google VertexAI Provider (`vertexai_embedding_provider.go`)

**Key Features**:
- Uses Google Cloud service account authentication
- Max batch size: 128
- Supports task-specific embeddings
- Default location: us-central1

**Configuration Parameters**:
```go
- model_name: VertexAI model name
- project_id: GCP project ID (required)
- location: GCP region (default: "us-central1")
- task: "DOC_RETRIEVAL", "CODE_RETRIEVAL", or "STS" (default: "DOC_RETRIEVAL")
- dim: Optional dimension parameter
- credential: GCP service account JSON (from credentials or env)
```

**Task Type Mapping**:
```go
func (provider *VertexAIEmbeddingProvider) getTaskType(mode models.TextEmbeddingMode) string {
    if mode == models.SearchMode {
        switch provider.task {
        case "DOC_RETRIEVAL": return "RETRIEVAL_QUERY"
        case "CODE_RETRIEVAL": return "CODE_RETRIEVAL_QUERY"
        case "STS": return "SEMANTIC_SIMILARITY"
        }
    } else {
        switch provider.task {
        case "DOC_RETRIEVAL", "CODE_RETRIEVAL": return "RETRIEVAL_DOCUMENT"
        case "STS": return "SEMANTIC_SIMILARITY"
        }
    }
}
```

**Authentication**:
- Reads JSON key from file path in GOOGLE_APPLICATION_CREDENTIALS env
- Caches credentials in memory for performance
- Thread-safe credential loading with mutex

**Endpoint Format**:
```
https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:predict
```

---

### 5. VoyageAI Provider (`voyageai_embedding_provider.go`)

**Key Features**:
- Default endpoint: `https://api.voyageai.com/v1/embeddings`
- Max batch size: 128
- Supports both float32 and int8 output types
- Optional truncation control

**Configuration Parameters**:
```go
- model_name: VoyageAI model name
- dim: Optional dimension (supported by voyage-3-large and voyage-code-3: 256, 512, 1024, 2048)
- truncation: true/false (default: false)
- output_type: "float" or "int8" (auto-detected)
```

**Input Type Logic**:
```go
if mode == models.InsertMode {
    textType = "document"
} else {
    textType = "query"
}
```

**Type-Safe Response Handling**:
```go
if provider.embdType == models.Float32Embd {
    resp := r.(*voyageai.EmbeddingResponse[float32])
    // Process float32 embeddings
} else {
    resp := r.(*voyageai.EmbeddingResponse[int8])
    // Process int8 embeddings
}
```

---

### 6. Alibaba DashScope Provider (`ali_embedding_provider.go`)

**Key Features**:
- Default endpoint: `https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`
- Model-specific batch sizes: 25 (default), 6 (text-embedding-v3)
- Supports dense embeddings only
- Text type differentiation

**Configuration Parameters**:
```go
- model_name: DashScope model name
- dim: Optional dimension parameter
- api_key: From credentials or DASHSCOPE_API_KEY env
- url: Custom endpoint (optional)
```

**Model-Specific Batch Limits**:
```go
maxBatch := 25
if modelName == "text-embedding-v3" {
    maxBatch = 6  // v3 has stricter limits
}
```

**Text Type**:
```go
if mode == models.SearchMode {
    textType = "query"
} else {
    textType = "document"
}
```

---

### 7. SiliconFlow Provider (`siliconflow_embedding_provider.go`)

**Key Features**:
- Default endpoint: `https://api.siliconflow.cn/v1/embeddings`
- Max batch size: 32
- Float32 output only
- Simple configuration

**Configuration Parameters**:
```go
- model_name: SiliconFlow model identifier
- dim: Optional dimension parameter
- api_key: From credentials or SILICONFLOW_API_KEY env
- url: Custom endpoint (optional)
```

**API Call**:
```go
resp, err := provider.client.Embedding(
    provider.url,
    provider.modelName,
    texts[i:end],
    "float",  // encoding_format
    int(provider.embedDimParam),
    provider.timeoutSec
)
```

---

### 8. Hugging Face TEI Provider (`tei_embedding_provider.go`)

**Key Features**:
- Text Embeddings Inference (TEI) server support
- Default max batch size: 32 (configurable)
- Supports prompt injection for search/ingestion
- Truncation direction control

**Configuration Parameters**:
```go
- endpoint: TEI server URL (required)
- ingestion_prompt: Prompt prefix for document embeddings
- search_prompt: Prompt prefix for query embeddings
- max_client_batch_size: Client-side batch size (default: 32)
- truncate: true/false (default: false)
- truncation_direction: "Left" or "Right" (default: "Right")
- api_key: Optional authentication token
```

**Prompt Injection**:
```go
var prompt string
if mode == models.InsertMode {
    prompt = provider.ingestionPrompt
} else {
    prompt = provider.searchPrompt
}
resp, err := provider.client.Embedding(texts[i:end], provider.truncate, provider.truncationDirection, prompt, provider.timeoutSec)
```

**Use Case**:
- Self-hosted embedding models
- Custom fine-tuned models
- On-premise deployments
- Cost-sensitive applications

---

### 9. Zilliz Cloud Provider (`zilliz_embedding_provider.go`)

**Key Features**:
- Integrated with Zilliz Cloud Pipelines
- Max batch size: 64
- Automatic cluster and database context
- Model deployment ID based

**Configuration Parameters**:
```go
- model_deployment_id: Zilliz Cloud model deployment identifier
- Additional model-specific parameters passed through
```

**Context Injection**:
```go
c, err := zilliz.NewZilliClient(
    modelDeploymentID,
    extraInfo.ClusterID,  // Automatically injected
    extraInfo.DBName,     // Automatically injected
    params
)
```

**Input Type**:
```go
if mode == models.SearchMode {
    provider.modelParams["input_type"] = "query"
} else {
    provider.modelParams["input_type"] = "document"
}
```

---

## Common Patterns

### 1. Batch Processing

All providers implement similar batch processing logic:

```go
numRows := len(texts)
data := make([][]float32, 0, numRows)
for i := 0; i < numRows; i += provider.maxBatch {
    end := i + provider.maxBatch
    if end > numRows {
        end = numRows
    }
    // Call provider API
    resp, err := provider.client.Embedding(...)
    if err != nil {
        return nil, err
    }
    // Validate and append results
    data = append(data, resp...)
}
return data, nil
```

### 2. Dimension Validation

```go
if len(item.Embedding) != int(provider.fieldDim) {
    return nil, fmt.Errorf(
        "The required embedding dim is [%d], but the embedding obtained from the model is [%d]",
        provider.fieldDim, len(item.Embedding)
    )
}
```

### 3. Credential Priority

Most providers follow this priority order:
1. Function schema parameters (credential name)
2. milvus.yaml configuration
3. Environment variables

```go
// Example from parseAKAndURL
func ParseAKAndURL(credentials *credentials.Credentials, params []*commonpb.KeyValuePair, confParams map[string]string, envKey string, extraInfo *models.ModelExtraInfo) (string, string, error) {
    // 1. Check function params
    // 2. Check yaml config
    // 3. Check environment
}
```

### 4. Mode-Based Behavior

```go
type TextEmbeddingMode int

const (
    InsertMode TextEmbeddingMode = iota
    SearchMode
)
```

Providers adjust behavior based on mode:
- **InsertMode**: Document/ingestion embeddings
- **SearchMode**: Query embeddings

### 5. Error Handling

```go
// Batch size validation
if numRows > runner.MaxBatch() {
    return fmt.Errorf("Embedding supports up to [%d] pieces of data at a time, got [%d]", runner.MaxBatch(), numRows)
}

// Empty string check
if hasEmptyString(texts) {
    return errors.New("There is an empty string in the input data, TextEmbedding function does not support empty text")
}

// Response count validation
if end-i != len(resp.Data) {
    return nil, fmt.Errorf("Get embedding failed. The number of texts and embeddings does not match text:[%d], embedding:[%d]", end-i, len(resp.Data))
}
```

---

## Function Executor Architecture

### Concurrent Processing

```go
func (executor *FunctionExecutor) ProcessInsert(ctx context.Context, msg *msgstream.InsertMsg) error {
    outputs := make(chan []*schemapb.FieldData, len(executor.runners))
    errChan := make(chan error, len(executor.runners))
    var wg sync.WaitGroup

    for _, runner := range executor.runners {
        wg.Add(1)
        go func(runner Runner) {
            defer wg.Done()
            data, err := executor.processSingleFunction(ctx, runner, msg)
            if err != nil {
                errChan <- err
                return
            }
            outputs <- data
        }(runner)
    }

    wg.Wait()
    close(errChan)
    close(outputs)

    // Collect results and errors
}
```

### Metrics Collection

```go
metrics.ProxyFunctionlatency.WithLabelValues(
    strconv.FormatInt(paramtable.GetNodeID(), 10),
    runner.GetCollectionName(),
    runner.GetFunctionTypeName(),
    runner.GetFunctionProvider(),
    runner.GetFunctionName()
).Observe(float64(tr.RecordSpan().Milliseconds()))
```

---

## Provider Comparison Table

| Provider | Max Batch | Output Types | Mode Support | Special Features |
|----------|-----------|--------------|--------------|------------------|
| OpenAI | 128 | float32 | Yes | Azure support, user tracking |
| Cohere | 96 | float32, int8 | Yes | Truncation control, v2/v3 differences |
| Bedrock | 1 | float32 | No | Normalization, AWS regions |
| VertexAI | 128 | float32 | Yes | Task types, GCP auth |
| VoyageAI | 128 | float32, int8 | Yes | Dimension control, truncation |
| AliDashScope | 25/6 | float32 | Yes | Model-specific limits |
| SiliconFlow | 32 | float32 | No | Simple API |
| TEI | 32 | float32 | Yes | Self-hosted, prompt injection |
| Zilliz | 64 | float32 | Yes | Cloud integration |

---

## Test Coverage

From `test_text_embedding_function_e2e.py`:

### Test Cases

1. **Basic Collection Creation**
   - Create collection with TEI provider
   - Verify function schema

2. **Duplicate Schema Handling**
   - Create collection twice with same schema
   - Ensure idempotency

3. **Negative Tests**
   - Unsupported endpoint validation
   - Dimension mismatch detection

### Example Test Configuration

```python
text_embedding_function = Function(
    name="tei",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["document"],
    output_field_names="dense",
    params={
        "provider": "TEI",
        "endpoint": tei_endpoint,
    }
)
```

---

## Key Insights

1. **Unified Interface**: All providers implement the same `textEmbeddingProvider` interface
2. **Batch Optimization**: Each provider has optimized batch sizes based on API limits
3. **Type Safety**: Strong typing with FloatVector and Int8Vector support
4. **Mode Awareness**: Providers differentiate between document and query embeddings
5. **Error Resilience**: Comprehensive validation at multiple levels
6. **Concurrent Execution**: Function executor processes multiple functions in parallel
7. **Metrics Integration**: Built-in latency tracking for all operations
8. **Flexible Authentication**: Multiple credential sources with clear priority
9. **Dimension Flexibility**: Some providers support dynamic dimension configuration
10. **Production Ready**: Timeout handling, batch limits, and error recovery

---

## Source Files Reference

- `function_executor.go`: Lines 1-318
- `text_embedding_function.go`: Lines 1-351
- `function_base.go`: Lines 1-84
- `openai_embedding_provider.go`: Lines 1-179
- `cohere_embedding_provider.go`: Lines 1-180
- `bedrock_embedding_provider.go`: Lines 1-235
- `vertexai_embedding_provider.go`: Lines 1-251
- `voyageai_embedding_provider.go`: Lines 1-178
- `ali_embedding_provider.go`: Lines 1-146
- `siliconflow_embedding_provider.go`: Lines 1-126
- `tei_embedding_provider.go`: Lines 1-149
- `zilliz_embedding_provider.go`: Lines 1-105
- `test_text_embedding_function_e2e.py`: Lines 1-200

---

**Analysis Complete**: 2026-02-24
