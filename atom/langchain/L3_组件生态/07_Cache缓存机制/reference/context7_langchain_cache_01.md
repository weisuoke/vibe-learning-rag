---
type: context7_documentation
library: LangChain
library_id: /websites/langchain
version: latest (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: Cache缓存机制
context7_query: cache caching LLM cache embedding cache performance optimization
---

# Context7 文档：LangChain Cache 缓存机制

## 文档来源
- 库名称：LangChain
- Library ID：/websites/langchain
- 最后更新：2026-02-17
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. Embedding 缓存（CacheBackedEmbeddings）

#### 核心概念
Embedding 可以被存储或临时缓存以避免重复计算。使用 `CacheBackedEmbeddings` 包装器实现缓存，将 embedding 存储在键值存储中，文本被哈希后作为缓存键。

#### 初始化方法

**Python 示例（LocalFileStore）**：
```python
import time
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore

# 创建底层 embedding 模型
underlying_embeddings = ... # e.g., OpenAIEmbeddings(), HuggingFaceEmbeddings(), etc.

# 本地文件存储（仅用于开发，不适合生产）
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model  # 避免缓存冲突
)

# 首次调用：计算并缓存
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"First call took: {time.time() - tic:.2f} seconds")

# 后续调用：从缓存读取
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"Second call took: {time.time() - tic:.2f} seconds")
```

**TypeScript 示例（InMemoryStore）**：
```typescript
import { CacheBackedEmbeddings } from "@langchain/classic/embeddings/cache_backed";
import { InMemoryStore } from "@langchain/core/stores";

const underlyingEmbeddings = new OpenAIEmbeddings();
const inMemoryStore = new InMemoryStore();

const cacheBackedEmbeddings = CacheBackedEmbeddings.fromBytesStore(
  underlyingEmbeddings,
  inMemoryStore,
  {
    namespace: underlyingEmbeddings.model,
  }
);

// 缓存查询 embedding
const tic = Date.now();
const queryEmbedding = cacheBackedEmbeddings.embedQuery("Hello, world!");
console.log(`First call took: ${Date.now() - tic}ms`);

// 缓存文档 embedding
const tic2 = Date.now();
const documentEmbedding = cacheBackedEmbeddings.embedDocuments(["Hello, world!"]);
console.log(`Cached creation time: ${Date.now() - tic2}ms`);
```

#### 支持的存储后端

1. **LocalFileStore** - 本地文件存储
   - 适用场景：本地开发、测试
   - 不适合生产环境

2. **InMemoryStore** - 内存存储
   - 适用场景：临时缓存、快速测试
   - 不持久化

3. **BigtableByteStore** - Google Bigtable 存储
   ```python
   from langchain.embeddings import CacheBackedEmbeddings
   from langchain_google_vertexai.embeddings import VertexAIEmbeddings

   underlying_embeddings = VertexAIEmbeddings(
       project=PROJECT_ID, model_name="textembedding-gecko@003"
   )

   # 使用 namespace 避免键冲突
   cached_embedder = CacheBackedEmbeddings.from_bytes_store(
       underlying_embeddings, store, namespace="text-embeddings"
   )
   ```

#### 初始化参数

`from_bytes_store()` 方法参数：
- **underlyingEmbeddings**：用于 embedding 的底层模型
- **documentEmbeddingStore**：任何 `BaseStore` 用于缓存文档 embedding
- **namespace**（可选，默认 `""`）：文档缓存的命名空间，帮助避免冲突（例如，设置为 embedding 模型名称）
- **batch_size**（可选）：控制存储更新之间嵌入的文档数量
- **query_embedding_cache**（可选）：启用查询 embedding 缓存，可以重用文档缓存存储或指定单独的存储

#### 最佳实践

1. **始终设置 namespace**
   - 避免使用不同 embedding 模型时的冲突
   - 推荐使用模型名称作为 namespace

2. **查询 embedding 缓存**
   - 默认不缓存查询 embedding
   - 需要显式指定 `query_embedding_cache` 参数启用

3. **生产环境存储选择**
   - 使用持久化存储（数据库、云存储）
   - 本地文件存储仅用于本地开发和测试

4. **性能优化**
   - 使用 `batch_size` 控制批量处理
   - 避免重复计算相同文本的 embedding

### 2. LLM 缓存

#### RedisCache - 标准 Redis 缓存

**用途**：为 LLM prompt 和 response 提供低延迟内存缓存

**Python 示例**：
```python
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache
import redis

redis_client = redis.Redis.from_url(...)
set_llm_cache(RedisCache(redis_client))
```

**特点**：
- 低延迟内存缓存
- 全局 LLM 缓存设置
- 适合高并发场景

#### UpstashRedisCache - Upstash Redis 缓存

**用途**：使用 Upstash Redis REST API 作为 LangChain LLM 缓存后端

**Python 示例**：
```python
import langchain
from upstash_redis import Redis

URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"

langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN))
```

**配置要求**：
- UPSTASH_REDIS_REST_URL：从 Upstash Console 获取
- UPSTASH_REDIS_REST_TOKEN：从 Upstash Console 获取

**特点**：
- 使用 REST API（无需 Redis 客户端）
- 云托管 Redis 服务
- 所有 LLM 调用自动缓存

#### RedisSemanticCache - Redis 语义缓存

**用途**：基于语义相似度检索缓存的 prompt

**Python 示例**：
```python
from langchain_redis import RedisSemanticCache
from langchain.globals import set_llm_cache
import redis
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

redis_url = "redis://localhost:6379"

set_llm_cache(RedisSemanticCache(
    embedding=FakeEmbeddings(),
    redis_url=redis_url
))
```

**特点**：
- 结合 Redis 作为缓存和 vectorstore
- 需要 embedding 提供者进行相似度计算
- 支持语义相似的 prompt 匹配（不仅是精确匹配）

### 3. 缓存的价值

#### 两大核心价值

1. **节省成本**
   - 减少对 LLM 提供者的 API 调用次数
   - 避免重复请求相同的 completion
   - 特别适合高频重复查询场景

2. **提升速度**
   - 减少 API 调用延迟
   - 从缓存读取比 API 调用快 20 倍以上
   - 改善用户体验

#### 缓存匹配策略

1. **精确字符串匹配**
   - 完全相同的 prompt 命中缓存
   - 适用于标准 LLM 缓存

2. **语义相似匹配**
   - 语义相似的 prompt 也能命中缓存
   - 使用 RedisSemanticCache 实现
   - 需要 embedding 模型支持

### 4. 高级用法

#### Bigtable 缓存示例

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

underlying_embeddings = VertexAIEmbeddings(
    project=PROJECT_ID, model_name="textembedding-gecko@003"
)

# 使用 namespace 避免键冲突
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace="text-embeddings"
)
```

**关键点**：
- 使用 namespace 避免不同数据类型的键冲突
- 适合企业级应用
- 支持大规模数据存储

#### 性能对比

**首次调用（无缓存）**：
- 需要调用 embedding API
- 耗时较长（取决于 API 响应时间）

**后续调用（有缓存）**：
- 直接从缓存读取
- 耗时极短（毫秒级）
- 性能提升显著

### 5. 缓存架构设计

#### 键值存储设计

**Embedding 缓存**：
- **键**：文本内容的哈希值
- **值**：embedding 向量（序列化后的浮点数列表）
- **命名空间**：隔离不同模型的缓存

**LLM 缓存**：
- **键**：prompt + llm_string（配置参数）
- **值**：Generation 对象列表
- **匹配策略**：精确匹配或语义匹配

#### 存储后端选择

| 后端 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| InMemoryStore | 开发测试 | 最快、零配置 | 不持久化、不共享 |
| LocalFileStore | 本地开发 | 持久化、简单 | 不支持分布式 |
| RedisCache | 生产环境 | 高性能、分布式 | 需要 Redis 服务 |
| BigtableByteStore | 企业应用 | 大规模、可靠 | 配置复杂、成本高 |

### 6. 2025-2026 最新特性

#### 官方文档更新
- 最后更新：2026-02-17
- 文档覆盖：26795 个代码片段
- Benchmark Score：83

#### 新增支持
- TypeScript 完整支持
- 多种云存储后端
- 改进的 namespace 机制
- 更好的性能优化

## 实践建议

### 开发阶段
1. 使用 LocalFileStore 或 InMemoryStore
2. 设置合适的 namespace
3. 测试缓存命中率

### 生产阶段
1. 使用 Redis 或云存储后端
2. 监控缓存性能
3. 定期清理过期缓存
4. 考虑使用语义缓存提升命中率

### 性能优化
1. 合理设置 batch_size
2. 启用查询 embedding 缓存（如需要）
3. 使用 namespace 隔离不同应用
4. 监控缓存大小和命中率

## 参考链接

- LangChain 官方文档：https://docs.langchain.com/oss/python/integrations/text_embedding
- Redis 集成文档：https://docs.langchain.com/oss/python/integrations/providers/redis
- Upstash 集成文档：https://docs.langchain.com/oss/python/integrations/providers/upstash
- Bigtable 集成文档：https://docs.langchain.com/oss/python/integrations/stores/bigtable
