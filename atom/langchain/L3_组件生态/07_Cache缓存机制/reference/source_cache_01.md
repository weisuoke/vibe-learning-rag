---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/caches.py
  - libs/langchain/langchain_classic/cache.py
  - libs/langchain/langchain_classic/embeddings/cache.py
  - libs/langchain/langchain_classic/schema/cache.py
analyzed_at: 2026-02-25
knowledge_point: Cache缓存机制
---

# 源码分析：LangChain Cache 缓存机制

## 分析的文件

### 1. `libs/core/langchain_core/caches.py` - 核心缓存抽象
**关键发现**：
- 定义了 `BaseCache` 抽象基类，是所有缓存实现的基础接口
- 提供了 `InMemoryCache` 内存缓存的参考实现
- 支持同步和异步两套 API（lookup/alookup, update/aupdate, clear/aclear）
- 缓存键由 `(prompt, llm_string)` 二元组生成

### 2. `libs/langchain/langchain_classic/cache.py` - 缓存实现导入
**关键发现**：
- 这是一个导入模块，实际实现在 `langchain_community.cache`
- 支持多种缓存后端：
  - InMemoryCache - 内存缓存
  - SQLiteCache - SQLite 数据库缓存
  - SQLAlchemyCache - 通用数据库缓存
  - RedisCache - Redis 缓存
  - RedisSemanticCache - Redis 语义缓存
  - GPTCache - 第三方缓存库
  - MomentoCache - Momento 云缓存
  - CassandraCache - Cassandra 数据库缓存
  - AstraDBCache - AstraDB 缓存
  - UpstashRedisCache - Upstash Redis 缓存
  - AzureCosmosDBSemanticCache - Azure Cosmos DB 语义缓存

### 3. `libs/langchain/langchain_classic/embeddings/cache.py` - Embedding 缓存
**关键发现**：
- `CacheBackedEmbeddings` 类实现了 Embedding 缓存包装器
- 支持文档 Embedding 缓存（默认启用）和查询 Embedding 缓存（可选）
- 使用哈希算法生成缓存键（支持 SHA-1, BLAKE2B, SHA-256, SHA-512）
- **安全警告**：默认使用 SHA-1，存在碰撞风险，建议生产环境使用更强的哈希算法
- 支持批量处理和异步操作
- 缓存存储基于 `BaseStore` 接口，支持多种后端（LocalFileStore 等）

### 4. `libs/langchain/langchain_classic/schema/cache.py` - 缓存 Schema
**关键发现**：
- 简单的重导出模块，从 `langchain_core.caches` 导入核心类型
- 定义了 `RETURN_VAL_TYPE` 和 `BaseCache`

## 关键代码片段

### BaseCache 接口定义

```python
class BaseCache(ABC):
    """Interface for a caching layer for LLMs and Chat models."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on `prompt` and `llm_string`."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments."""
```

**关键点**：
- 缓存键由 `(prompt, llm_string)` 组成
- `prompt` 是提示词的字符串表示（对于 Chat 模型是序列化后的消息）
- `llm_string` 是 LLM 配置的字符串表示（包含模型名、温度、停止词等参数）
- 缓存未命中返回 `None`，命中返回 `Generation` 列表

### InMemoryCache 实现

```python
class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self, *, maxsize: int | None = None) -> None:
        """Initialize with empty cache.

        Args:
            maxsize: The maximum number of items to store in the cache.
                If `None`, the cache has no maximum size.
                If the cache exceeds the maximum size, the oldest items are removed.
        """
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}
        self._maxsize = maxsize

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""
        if self._maxsize is not None and len(self._cache) == self._maxsize:
            del self._cache[next(iter(self._cache))]  # FIFO 淘汰策略
        self._cache[prompt, llm_string] = return_val
```

**关键点**：
- 使用 Python 字典存储缓存
- 支持最大容量限制（`maxsize`）
- 使用 FIFO（先进先出）淘汰策略
- 异步方法直接调用同步方法（无额外开销）

### CacheBackedEmbeddings 核心逻辑

```python
class CacheBackedEmbeddings(Embeddings):
    """Interface for caching results from embedding models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with caching."""
        # 1. 批量查询缓存
        vectors: list[list[float] | None] = self.document_embedding_store.mget(texts)

        # 2. 找出缓存未命中的文本
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        # 3. 批量计算缺失的 Embedding
        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)

            # 4. 更新缓存
            self.document_embedding_store.mset(
                list(zip(missing_texts, missing_vectors, strict=False))
            )

            # 5. 填充结果
            for index, updated_vector in zip(missing_indices, missing_vectors, strict=False):
                vectors[index] = updated_vector

        return vectors
```

**关键点**：
- 批量查询缓存（`mget`）提高效率
- 只计算缓存未命中的 Embedding
- 支持批量大小控制（`batch_size`）
- 查询 Embedding 默认不缓存（需显式启用）

### Embedding 缓存键生成

```python
def _make_default_key_encoder(namespace: str, algorithm: str) -> Callable[[str], str]:
    """Create a default key encoder function.

    Args:
        namespace: Prefix that segregates keys from different embedding models.
        algorithm:
           * 'sha1' - fast but not collision-resistant
           * 'blake2b' - cryptographically strong, faster than SHA-1
           * 'sha256' - cryptographically strong, slower than SHA-1
           * 'sha512' - cryptographically strong, slower than SHA-1
    """
    def _key_encoder(key: str) -> str:
        """Encode a key using the specified algorithm."""
        if algorithm == "sha1":
            return f"{namespace}{_sha1_hash_to_uuid(key)}"
        if algorithm == "blake2b":
            return f"{namespace}{hashlib.blake2b(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha256":
            return f"{namespace}{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha512":
            return f"{namespace}{hashlib.sha512(key.encode('utf-8')).hexdigest()}"
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return _key_encoder
```

**关键点**：
- 使用 `namespace` 隔离不同 Embedding 模型的缓存
- 支持多种哈希算法，平衡性能和安全性
- SHA-1 最快但不抗碰撞，生产环境建议使用 BLAKE2B 或 SHA-256

## 架构设计洞察

### 1. 分层设计
- **Core 层**（`langchain_core.caches`）：定义抽象接口和基础实现
- **Classic 层**（`langchain_classic.cache`）：导入和重导出
- **Community 层**（`langchain_community.cache`）：具体后端实现

### 2. 缓存键设计
- LLM 缓存：`(prompt, llm_string)` 二元组
  - `prompt`：提示词内容
  - `llm_string`：LLM 配置参数（确保相同输入+相同配置才命中缓存）
- Embedding 缓存：文本内容的哈希值
  - 使用 `namespace` 隔离不同模型
  - 支持自定义哈希算法

### 3. 性能优化策略
- 批量操作（`mget`, `mset`）减少 I/O 次数
- 批量大小控制（`batch_size`）平衡内存和性能
- 异步支持（`alookup`, `aupdate`）提高并发性能
- FIFO 淘汰策略（InMemoryCache）简单高效

### 4. 安全考虑
- SHA-1 碰撞风险警告
- 推荐使用 BLAKE2B 或 SHA-256
- 支持自定义 `key_encoder` 函数

## 缓存后端对比

| 后端 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| InMemoryCache | 开发测试、单机应用 | 最快、零依赖 | 不持久化、不共享 |
| SQLiteCache | 单机应用、需要持久化 | 持久化、零配置 | 不支持分布式 |
| RedisCache | 分布式应用、高并发 | 高性能、支持分布式 | 需要 Redis 服务 |
| RedisSemanticCache | 语义相似查询 | 支持相似度匹配 | 需要 Embedding 计算 |
| SQLAlchemyCache | 企业应用、已有数据库 | 支持多种数据库 | 配置复杂 |

## 使用模式

### LLM 缓存
```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

# 全局启用缓存
set_llm_cache(InMemoryCache())

# 后续所有 LLM 调用自动使用缓存
```

### Embedding 缓存
```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 创建缓存存储
store = LocalFileStore("./my_cache")

# 包装 Embedding 模型
underlying_embedder = OpenAIEmbeddings()
embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embedder,
    store,
    namespace=underlying_embedder.model,
    key_encoder="blake2b"  # 使用更安全的哈希算法
)

# 首次调用计算并缓存
embeddings = embedder.embed_documents(["hello", "world"])

# 再次调用直接从缓存读取
embeddings = embedder.embed_documents(["hello", "world"])
```

## 2025-2026 最新变化

### 安全增强
- 明确警告 SHA-1 的碰撞风险
- 推荐使用 BLAKE2B（速度快且安全）或 SHA-256
- 支持自定义 `key_encoder` 函数

### 性能优化
- 批量操作优化（`batch_iterate`）
- 异步支持完善
- 内存缓存支持 `maxsize` 限制

### 架构演进
- 缓存实现从 `langchain` 迁移到 `langchain_community`
- 核心抽象保持在 `langchain_core`
- 更清晰的分层设计

## 关键要点总结

1. **两种缓存类型**：
   - LLM 缓存：缓存 LLM 调用结果（prompt + config → Generation）
   - Embedding 缓存：缓存 Embedding 计算结果（text → vector）

2. **缓存键设计**：
   - LLM：`(prompt, llm_string)` 确保配置一致性
   - Embedding：文本哈希 + namespace 隔离模型

3. **性能优化**：
   - 批量操作减少 I/O
   - 异步支持提高并发
   - 批量大小控制平衡内存

4. **安全考虑**：
   - SHA-1 存在碰撞风险
   - 生产环境使用 BLAKE2B 或 SHA-256
   - 支持自定义哈希函数

5. **多后端支持**：
   - 内存、SQLite、Redis、数据库等
   - 根据场景选择合适后端
   - 统一接口易于切换
