---
type: source_code_analysis
source: sourcecode/langchain/libs/partners/openai/langchain_openai/embeddings/base.py
analyzed_files:
  - langchain_openai/embeddings/base.py (前500行)
analyzed_at: 2026-02-25
knowledge_point: Embedding模型集成
---

# 源码分析：OpenAIEmbeddings 实现

## 分析的文件
- `langchain_openai/embeddings/base.py` (前500行) - OpenAI Embeddings 具体实现

## 关键发现

### 1. 类设计：Pydantic BaseModel + Embeddings

**设计模式**：Multiple Inheritance

```python
class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI embedding model integration."""
```

**设计理由**：
1. **BaseModel**：提供配置管理、验证、序列化
2. **Embeddings**：实现 Embeddings 接口

**优势**：
- 自动配置验证
- 类型安全
- 序列化支持
- 环境变量自动加载

### 2. 配置参数设计

**核心参数**（前200行）：

```python
class OpenAIEmbeddings(BaseModel, Embeddings):
    model: str = "text-embedding-ada-002"
    dimensions: int | None = None  # text-embedding-3+ 支持

    # API 配置
    openai_api_key: SecretStr | None | Callable = Field(
        default_factory=secret_from_env("OPENAI_API_KEY")
    )
    openai_api_base: str | None = Field(
        default_factory=from_env("OPENAI_API_BASE")
    )

    # 性能配置
    chunk_size: int = 1000  # 批量大小
    max_retries: int = 2
    request_timeout: float | tuple[float, float] | None = None

    # Token 管理
    embedding_ctx_length: int = 8191
    tiktoken_enabled: bool = True
    check_embedding_ctx_length: bool = True
```

**关键设计点**：

#### A. 环境变量自动加载

```python
openai_api_key: SecretStr | None | Callable = Field(
    default_factory=secret_from_env("OPENAI_API_KEY", default=None)
)
```

**设计优势**：
- 自动从环境变量加载
- 支持 SecretStr（安全）
- 支持 Callable（动态获取）

#### B. 多种 API 密钥类型支持

```python
openai_api_key: (
    SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
)
```

**支持场景**：
1. **SecretStr**：静态密钥
2. **Callable[[], str]**：同步动态获取
3. **Callable[[], Awaitable[str]]**：异步动态获取

**实际应用**：
- 从密钥管理服务动态获取
- 支持密钥轮换
- 支持多租户场景

#### C. 维度参数（text-embedding-3+）

```python
dimensions: int | None = None
"""The number of dimensions the resulting output embeddings should have.

Only supported in 'text-embedding-3' and later models.
"""
```

**设计背景**：
- OpenAI 的 text-embedding-3 模型支持自定义维度
- 可以降低维度以节省存储和计算成本
- 权衡：维度 ↓ → 性能 ↓，但成本 ↓

### 3. 客户端初始化策略

**双客户端设计**：

```python
client: Any = Field(default=None, exclude=True)
async_client: Any = Field(default=None, exclude=True)
```

**初始化逻辑**（validate_environment 方法）：

```python
@model_validator(mode="after")
def validate_environment(self) -> Self:
    # 1. 解析 API 密钥（支持同步/异步 Callable）
    sync_api_key_value, async_api_key_value = _resolve_sync_and_async_api_keys(
        self.openai_api_key
    )

    # 2. 创建同步客户端
    if sync_api_key_value is not None:
        self.client = openai.OpenAI(**client_params, api_key=sync_api_key_value).embeddings

    # 3. 创建异步客户端
    self.async_client = openai.AsyncOpenAI(
        **client_params, api_key=async_api_key_value
    ).embeddings

    return self
```

**关键设计点**：
1. **延迟初始化**：在 Pydantic 验证阶段初始化客户端
2. **分离同步/异步**：支持不同的 API 密钥获取方式
3. **错误处理**：如果没有同步密钥，`client` 为 `None`，调用时抛出错误

### 4. Token 管理与分块策略

**核心方法**：`_tokenize`

```python
def _tokenize(
    self, texts: list[str], chunk_size: int
) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
    """Tokenize and batch input texts.

    Splits texts based on `embedding_ctx_length` and groups them into batches
    of size `chunk_size`.
    """
```

**分块逻辑**：

1. **Tokenization**：
   - 如果 `tiktoken_enabled=True`：使用 tiktoken 分词
   - 如果 `tiktoken_enabled=False`：使用 HuggingFace transformers

2. **长文本处理**：
   - 如果文本超过 `embedding_ctx_length`，自动分块
   - 每个分块独立嵌入
   - 最后加权平均合并

3. **批量处理**：
   - 将多个文本分组为批次（`chunk_size`）
   - 减少 API 调用次数

**加权平均算法**（`_process_batched_chunked_embeddings`）：

```python
# 对于每个文本，如果有多个分块
if len(_result) > 1:
    # 加权平均
    total_weight = sum(num_tokens_in_batch[i])
    average = [
        sum(val * weight for val, weight in zip(embedding, num_tokens_in_batch[i]))
        / total_weight
        for embedding in zip(*_result)
    ]

    # 归一化
    magnitude = sum(val**2 for val in average) ** 0.5
    embeddings.append([val / magnitude for val in average])
```

**设计优势**：
- 自动处理长文本
- 保持向量归一化
- 权重基于 token 数量

### 5. 非 OpenAI 提供商支持

**兼容性设计**：

```python
check_embedding_ctx_length: bool = True
"""Whether to check the token length of inputs and automatically split inputs
longer than `embedding_ctx_length`.

Set to `False` to send raw text strings directly to the API instead of
tokenizing. Useful for many non-OpenAI providers (e.g. OpenRouter, Ollama,
vLLM).
"""
```

**使用场景**（来自源码文档）：

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="...",
    base_url="...",
    check_embedding_ctx_length=False,  # 关键：跳过 tokenization
)
```

**设计理由**：
- 很多 OpenAI 兼容 API 不支持 tiktoken
- 直接发送原始文本字符串
- 避免 tokenization 错误

### 6. 批量处理与性能优化

**批量参数**：

```python
chunk_size: int = 1000
"""Maximum number of texts to embed in each batch"""

show_progress_bar: bool = False
"""Whether to show a progress bar when embedding."""
```

**设计考虑**：
1. **API 限制**：OpenAI API 有批量大小限制
2. **内存管理**：避免一次性处理过多文本
3. **用户体验**：可选的进度条

### 7. 错误处理与重试

**重试配置**：

```python
max_retries: int = 2
"""Maximum number of retries to make when generating."""

retry_min_seconds: int = 4
"""Min number of seconds to wait between retries"""

retry_max_seconds: int = 20
"""Max number of seconds to wait between retries"""
```

**设计模式**：Exponential Backoff

**实际应用**：
- 处理 API 速率限制
- 处理临时网络错误
- 提高系统稳定性

### 8. Azure OpenAI 支持

**Azure 特定参数**：

```python
openai_api_version: str | None = Field(
    default_factory=from_env("OPENAI_API_VERSION", default=None),
    alias="api_version",
)
"""Version of the OpenAI API to use."""

openai_api_type: str | None = Field(
    default_factory=from_env("OPENAI_API_TYPE", default=None)
)

deployment: str | None = model
"""to support Azure OpenAI Service custom deployment names"""
```

**验证逻辑**：

```python
@model_validator(mode="after")
def validate_environment(self) -> Self:
    if self.openai_api_type in ("azure", "azure_ad", "azuread"):
        msg = "If you are using Azure, please use the `AzureOpenAIEmbeddings` class."
        raise ValueError(msg)
```

**设计理由**：
- 引导用户使用专门的 Azure 类
- 避免配置混淆

## 架构决策分析

### 决策1：为什么使用 Pydantic BaseModel？

**优势**：
1. **自动验证**：参数类型和值自动验证
2. **环境变量加载**：`Field(default_factory=from_env(...))`
3. **序列化支持**：可以保存和加载配置
4. **文档生成**：自动生成 API 文档

**权衡**：
- 增加了依赖（Pydantic）
- 但提供了强大的配置管理能力

### 决策2：为什么支持多种 API 密钥类型？

**场景**：
1. **静态密钥**：开发环境
2. **同步动态获取**：从文件或数据库读取
3. **异步动态获取**：从密钥管理服务（如 AWS Secrets Manager）

**设计哲学**：
- 灵活性优先
- 支持企业级场景

### 决策3：为什么自动处理长文本？

**问题**：
- OpenAI API 有 token 限制（8191 for ada-002）
- 用户可能不知道这个限制

**解决方案**：
- 自动分块
- 加权平均合并
- 用户无感知

**权衡**：
- 增加了复杂性
- 但提供了更好的用户体验

### 决策4：为什么支持非 OpenAI 提供商？

**背景**：
- 很多服务提供 OpenAI 兼容 API
- 但不完全兼容（如不支持 tiktoken）

**解决方案**：
- `check_embedding_ctx_length=False`
- 跳过 tokenization，直接发送原始文本

**设计哲学**：
- 兼容性优先
- 支持生态系统

## 与 RAG 开发的联系

### 在 RAG 中的使用

**场景1：基础文档嵌入**

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # 降低维度以节省成本
)

# 嵌入文档
doc_vectors = embeddings.embed_documents(documents)
```

**场景2：使用缓存**

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

store = LocalFileStore("./embeddings_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(),
    store,
    namespace="openai-ada-002"
)
```

**场景3：非 OpenAI 提供商**

```python
embeddings = OpenAIEmbeddings(
    model="custom-model",
    base_url="https://your-api.com/v1",
    check_embedding_ctx_length=False  # 关键
)
```

### 性能优化建议

1. **选择合适的模型**：
   - `text-embedding-3-small`：性能好，成本低
   - `text-embedding-3-large`：性能最好，成本高
   - `text-embedding-ada-002`：旧模型，兼容性好

2. **调整维度**（text-embedding-3+）：
   - 默认：1536 维
   - 降低到 512 或 256 维可以节省 50-75% 成本
   - 权衡：性能略有下降

3. **批量处理**：
   - 设置合理的 `chunk_size`
   - 避免单个文本过长

4. **使用缓存**：
   - 结合 `CacheBackedEmbeddings`
   - 避免重复计算

## 总结

OpenAIEmbeddings 的设计体现了以下原则：
1. **配置灵活性**：支持多种 API 密钥类型、环境变量自动加载
2. **自动化处理**：自动分块、加权平均、错误重试
3. **兼容性**：支持 Azure、非 OpenAI 提供商
4. **性能优化**：批量处理、可选进度条
5. **企业级特性**：动态密钥获取、重试机制

这种设计使得 OpenAIEmbeddings 既易于使用（开箱即用），又足够灵活以支持复杂的企业场景。
