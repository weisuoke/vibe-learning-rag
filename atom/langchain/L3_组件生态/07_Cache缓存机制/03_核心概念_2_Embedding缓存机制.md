# 核心概念2：Embedding缓存机制

> Embedding计算结果缓存的原理、实现与最佳实践

---

## 概述

Embedding缓存机制通过存储文本的向量表示，避免重复计算相同文本的Embedding，从而降低成本和延迟。LangChain提供了`CacheBackedEmbeddings`包装器实现Embedding缓存。

---

## 1. CacheBackedEmbeddings包装器

### 核心设计

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 创建底层embedding模型
underlying_embeddings = OpenAIEmbeddings()

# 创建存储后端
store = LocalFileStore("./embedding_cache/")

# 创建缓存包装器
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)
```

**关键特性**：
- 包装器模式：包装任何Embeddings实例
- 透明缓存：无需修改业务代码
- 灵活存储：支持多种存储后端

### from_bytes_store方法

```python
@classmethod
def from_bytes_store(
    cls,
    underlying_embeddings: Embeddings,
    document_embedding_store: BaseStore[str, bytes],
    *,
    namespace: str = "",
    batch_size: Optional[int] = None,
    query_embedding_cache: Optional[BaseStore[str, bytes]] = None,
    key_encoder: Union[str, Callable[[str], str]] = "sha1"
) -> CacheBackedEmbeddings:
    """从字节存储创建缓存embedding"""
```

**参数说明**：
- `underlying_embeddings`: 底层embedding模型
- `document_embedding_store`: 文档embedding缓存存储
- `namespace`: 命名空间（隔离不同模型）
- `batch_size`: 批量处理大小
- `query_embedding_cache`: 查询embedding缓存（可选）
- `key_encoder`: 缓存键编码算法

---

## 2. 缓存键生成机制

### 哈希算法选择

```python
def _make_default_key_encoder(namespace: str, algorithm: str) -> Callable[[str], str]:
    """创建默认的键编码器

    Args:
        namespace: 命名空间前缀
        algorithm: 哈希算法
            - 'sha1': 快速但不抗碰撞
            - 'blake2b': 加密强度高，速度快
            - 'sha256': 加密强度高，速度中等
            - 'sha512': 加密强度高，速度慢
    """
    def _key_encoder(key: str) -> str:
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

**算法对比**：

| 算法 | 速度 | 安全性 | 碰撞风险 | 推荐场景 |
|------|------|--------|----------|----------|
| SHA-1 | 最快 | 低 | 高 | 开发测试 |
| BLAKE2B | 快 | 高 | 极低 | 生产环境（推荐） |
| SHA-256 | 中等 | 高 | 极低 | 生产环境 |
| SHA-512 | 慢 | 最高 | 极低 | 高安全要求 |

### 缓存键格式

```python
# 缓存键格式
cache_key = f"{namespace}{hash(text)}"

# 示例
namespace = "text-embedding-3-small"
text = "Hello, world!"
hash_value = blake2b(text.encode()).hexdigest()
cache_key = f"{namespace}{hash_value}"

# 结果：text-embedding-3-small7fcbc3d0b5f419a...
```

**namespace的作用**：
- 隔离不同embedding模型的缓存
- 避免模型切换时的缓存冲突
- 便于管理和清理

---

## 3. 文档Embedding缓存

### 缓存流程

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """嵌入文档列表（带缓存）"""

    # 1. 批量查询缓存
    vectors: List[Optional[List[float]]] = self.document_embedding_store.mget(texts)

    # 2. 找出缓存未命中的文本
    missing_indices = [i for i, v in enumerate(vectors) if v is None]

    # 3. 批量计算缺失的embedding
    for batch_indices in batch_iterate(self.batch_size, missing_indices):
        missing_texts = [texts[i] for i in batch_indices]
        missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)

        # 4. 更新缓存
        self.document_embedding_store.mset(
            list(zip(missing_texts, missing_vectors))
        )

        # 5. 填充结果
        for index, vector in zip(batch_indices, missing_vectors):
            vectors[index] = vector

    return vectors
```

**性能优化**：
- 批量查询（mget）减少I/O次数
- 只计算缓存未命中的embedding
- 批量更新（mset）提高效率

### 使用示例

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
import time

# 1. 创建缓存embedder
store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,
    key_encoder="blake2b"  # 使用安全哈希
)

# 2. 首次调用（无缓存）
texts = ["Hello, world!", "LangChain is awesome!", "Caching saves time."]

print("=== 首次调用 ===")
start = time.time()
embeddings1 = cached_embedder.embed_documents(texts)
print(f"耗时: {time.time() - start:.2f}秒")
print(f"向量维度: {len(embeddings1[0])}")

# 3. 再次调用（缓存命中）
print("\n=== 缓存命中 ===")
start = time.time()
embeddings2 = cached_embedder.embed_documents(texts)
print(f"耗时: {time.time() - start:.2f}秒")

# 4. 部分缓存命中
texts_mixed = ["Hello, world!", "New text here!"]
print("\n=== 部分缓存命中 ===")
start = time.time()
embeddings3 = cached_embedder.embed_documents(texts_mixed)
print(f"耗时: {time.time() - start:.2f}秒")
print("第1个文本：缓存命中")
print("第2个文本：缓存未命中，需要计算")
```

**输出示例**：
```
=== 首次调用 ===
耗时: 1.23秒
向量维度: 1536

=== 缓存命中 ===
耗时: 0.02秒  ← 速度提升60倍+

=== 部分缓存命中 ===
耗时: 0.45秒  ← 只计算1个新文本
第1个文本：缓存命中
第2个文本：缓存未命中，需要计算
```

---

## 4. 查询Embedding缓存

### 默认行为

```python
# 默认：查询embedding不缓存
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name"
)

# 查询embedding每次都计算
query_embedding = cached_embedder.embed_query("What is AI?")  # 不缓存
```

**原因**：
- 查询通常不重复
- 缓存收益有限
- 避免缓存膨胀

### 启用查询缓存

```python
# 方式1：使用相同的存储
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    query_embedding_cache=store  # 启用查询缓存
)

# 方式2：使用单独的存储
query_store = LocalFileStore("./query_cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    query_embedding_cache=query_store  # 单独的查询缓存
)

# 查询embedding会被缓存
query_embedding1 = cached_embedder.embed_query("What is AI?")  # 计算并缓存
query_embedding2 = cached_embedder.embed_query("What is AI?")  # 缓存命中
```

**适用场景**：
- 固定的查询模板
- 高频重复查询
- 评估和测试场景

---

## 5. 存储后端选择

### LocalFileStore

```python
from langchain_classic.storage import LocalFileStore

# 本地文件存储
store = LocalFileStore("./embedding_cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name"
)
```

**特点**：
- 持久化存储
- 零配置
- 适合单机应用

**注意事项**：
- 不支持分布式
- 文件权限问题
- 定期清理

### InMemoryStore

```python
from langchain_core.stores import InMemoryStore

# 内存存储
store = InMemoryStore()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name"
)
```

**特点**：
- 最快速度
- 零配置
- 适合临时缓存

**注意事项**：
- 不持久化
- 重启丢失
- 内存限制

### Redis存储（推荐生产环境）

```python
from langchain_community.storage import RedisStore
import redis

# Redis存储
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # 存储字节
)

store = RedisStore(redis_client=redis_client)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name"
)
```

**特点**：
- 高性能
- 持久化
- 分布式共享
- 支持TTL

**配置建议**：
```python
# Redis持久化配置
# redis.conf
save 900 1      # 900秒内至少1个key变化则保存
save 300 10     # 300秒内至少10个key变化则保存
save 60 10000   # 60秒内至少10000个key变化则保存

appendonly yes  # 启用AOF持久化
```

---

## 6. 批量处理优化

### batch_size参数

```python
# 设置批量大小
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    batch_size=100  # 每批处理100个文本
)

# 处理大量文档
documents = ["doc1", "doc2", ...] * 1000  # 1000个文档

# 自动分批处理
embeddings = cached_embedder.embed_documents(documents)
```

**batch_size选择**：
- 太小：频繁I/O，性能低
- 太大：内存占用高，可能超时
- 推荐：50-200

### 批量处理示例

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
import time

# 创建缓存embedder
store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,
    batch_size=100
)

# 大规模文档处理
documents = [f"Document {i}" for i in range(1000)]

print("=== 首次处理（无缓存） ===")
start = time.time()
embeddings1 = cached_embedder.embed_documents(documents)
print(f"处理1000个文档耗时: {time.time() - start:.2f}秒")

print("\n=== 再次处理（缓存命中） ===")
start = time.time()
embeddings2 = cached_embedder.embed_documents(documents)
print(f"处理1000个文档耗时: {time.time() - start:.2f}秒")

# 计算加速比
speedup = (time.time() - start) / (time.time() - start)
print(f"\n加速比: {speedup:.1f}x")
```

---

## 7. 在RAG系统中的应用

### 场景1：文档索引构建

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 创建缓存embedder
store = LocalFileStore("./embedding_cache/")
underlying_embeddings = OpenAIEmbeddings()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,
    batch_size=100
)

# 2. 加载文档
loader = DirectoryLoader("./docs/", glob="**/*.md")
documents = loader.load()

# 3. 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"总共{len(chunks)}个文本块")

# 4. 构建向量库（使用缓存）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=cached_embedder,  # 使用缓存embedder
    persist_directory="./chroma_db"
)

print("向量库构建完成")

# 5. 增量更新（只计算新文档）
new_documents = loader.load()  # 加载新文档
new_chunks = text_splitter.split_documents(new_documents)

# 已存在的文本块会命中缓存
vectorstore.add_documents(new_chunks)
print("增量更新完成")
```

### 场景2：重复文档处理

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 创建缓存embedder
store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# 模拟重复文档
documents = [
    "Document A content...",
    "Document B content...",
    "Document A content...",  # 重复
    "Document C content...",
    "Document B content...",  # 重复
]

# 处理文档（重复文档命中缓存）
embeddings = cached_embedder.embed_documents(documents)

print(f"处理{len(documents)}个文档")
print(f"实际计算: 3个（A, B, C）")
print(f"缓存命中: 2个（重复的A和B）")
print(f"缓存命中率: {2/5:.1%}")
```

### 场景3：多次实验迭代

```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 创建缓存embedder
store = LocalFileStore("./cache/")
underlying_embeddings = OpenAIEmbeddings()

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# 固定的测试数据集
test_documents = [
    "Test document 1",
    "Test document 2",
    "Test document 3",
]

# 实验1：首次运行
print("=== 实验1 ===")
embeddings1 = cached_embedder.embed_documents(test_documents)
# 计算并缓存

# 实验2：调整参数后重新运行
print("=== 实验2 ===")
embeddings2 = cached_embedder.embed_documents(test_documents)
# 缓存命中，无需重新计算

# 实验3：再次调整参数
print("=== 实验3 ===")
embeddings3 = cached_embedder.embed_documents(test_documents)
# 缓存命中，无需重新计算

print("多次实验中，embedding计算只执行了1次")
```

---

## 8. 最佳实践

### 1. 始终设置namespace

```python
# ✅ 推荐：设置namespace
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model  # 使用模型名称
)

# ❌ 避免：不设置namespace
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store
)  # 可能导致模型切换时的缓存冲突
```

### 2. 使用安全的哈希算法

```python
# ✅ 推荐：生产环境使用BLAKE2B或SHA-256
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    key_encoder="blake2b"  # 安全且快速
)

# ⚠️ 注意：SHA-1仅用于开发测试
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    key_encoder="sha1"  # 存在碰撞风险
)
```

### 3. 合理设置batch_size

```python
# 根据文档数量和内存调整
if len(documents) < 100:
    batch_size = 50
elif len(documents) < 1000:
    batch_size = 100
else:
    batch_size = 200

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name",
    batch_size=batch_size
)
```

### 4. 定期清理缓存

```python
import os
import time

def clean_old_cache(cache_dir: str, max_age_days: int = 30):
    """清理超过指定天数的缓存文件"""
    now = time.time()
    max_age_seconds = max_age_days * 24 * 3600

    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_age = now - os.path.getmtime(file_path)

            if file_age > max_age_seconds:
                os.remove(file_path)
                print(f"删除过期缓存: {file_path}")

# 定期清理
clean_old_cache("./embedding_cache/", max_age_days=30)
```

### 5. 监控缓存效果

```python
class MonitoredCacheBackedEmbeddings(CacheBackedEmbeddings):
    """带监控的缓存embedder"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_time_saved = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 查询缓存
        vectors = self.document_embedding_store.mget(texts)

        # 统计命中
        hits = sum(1 for v in vectors if v is not None)
        misses = len(texts) - hits

        self.cache_hits += hits
        self.cache_misses += misses

        # 估算节省时间（假设每个embedding计算100ms）
        self.total_time_saved += hits * 0.1

        # 调用父类方法
        return super().embed_documents(texts)

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "缓存命中数": self.cache_hits,
            "缓存未命中数": self.cache_misses,
            "命中率": f"{hit_rate:.2%}",
            "节省时间": f"{self.total_time_saved:.2f}秒"
        }

# 使用监控embedder
monitored_embedder = MonitoredCacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="model-name"
)

# ... 使用embedder ...

# 查看统计
print(monitored_embedder.get_stats())
```

---

## 9. 常见问题

### Q1: 为什么查询embedding默认不缓存？

**原因**：
- 查询通常不重复
- 缓存收益有限
- 避免缓存膨胀

**何时启用**：
- 固定查询模板
- 高频重复查询
- 评估和测试

### Q2: 如何处理embedding模型更新？

**方案1：更换namespace**
```python
# 旧模型
old_embedder = CacheBackedEmbeddings.from_bytes_store(
    old_model,
    store,
    namespace="text-embedding-ada-002"
)

# 新模型
new_embedder = CacheBackedEmbeddings.from_bytes_store(
    new_model,
    store,
    namespace="text-embedding-3-small"  # 不同namespace
)
```

**方案2：清空缓存**
```python
import shutil

# 删除旧缓存
shutil.rmtree("./embedding_cache/")

# 重新创建
os.makedirs("./embedding_cache/")
```

### Q3: 缓存占用空间过大怎么办？

**解决方案**：
1. 定期清理过期缓存
2. 使用Redis的TTL功能
3. 限制缓存大小
4. 压缩存储

```python
# Redis TTL示例
from langchain_community.storage import RedisStore
import redis

redis_client = redis.Redis(host='localhost', port=6379)
store = RedisStore(redis_client=redis_client)

# 设置过期时间（7天）
redis_client.expire("cache_key", 7 * 24 * 3600)
```

---

## 参考资料

**源码**：
- `langchain_classic/embeddings/cache.py` - CacheBackedEmbeddings实现
- `langchain_classic/storage/` - 存储后端

**官方文档**：
- LangChain Embedding Caching文档
- Context7: /websites/langchain

**社区讨论**：
- Twitter: CacheBackedEmbeddings教程
- Reddit: Embedding缓存最佳实践

---

## 下一步

- [核心概念3：缓存键设计](./03_核心概念_3_缓存键设计.md)
- [核心概念4：缓存后端选择](./03_核心概念_4_缓存后端选择.md)
- [实战代码：CacheBackedEmbeddings实战](./07_实战代码_场景3_CacheBackedEmbeddings实战.md)
