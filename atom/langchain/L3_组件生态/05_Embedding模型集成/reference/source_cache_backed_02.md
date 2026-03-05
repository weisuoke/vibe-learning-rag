---
type: source_code_analysis
source: sourcecode/langchain/libs/langchain/langchain_classic/embeddings/cache.py
analyzed_files:
  - langchain_classic/embeddings/cache.py
analyzed_at: 2026-02-25
knowledge_point: Embedding模型集成
---

# 源码分析：CacheBackedEmbeddings 缓存机制

## 分析的文件
- `langchain_classic/embeddings/cache.py` (371行) - 缓存支持的 Embeddings 实现

## 关键发现

### 1. 设计模式：Decorator Pattern

**核心思想**：
CacheBackedEmbeddings 是一个装饰器，包装任何 Embeddings 实现并添加缓存功能。

**代码结构**：
```python
class CacheBackedEmbeddings(Embeddings):
    def __init__(
        self,
        underlying_embeddings: Embeddings,
        document_embedding_store: BaseStore[str, list[float]],
        *,
        batch_size: int | None = None,
        query_embedding_store: BaseStore[str, list[float]] | None = None,
    ) -> None:
        self.underlying_embeddings = underlying_embeddings
        self.document_embedding_store = document_embedding_store
        self.query_embedding_store = query_embedding_store
        self.batch_size = batch_size
```

**设计优势**：
- 不修改原始 Embeddings 实现
- 可以包装任何 Embeddings 子类
- 缓存逻辑与嵌入逻辑分离

### 2. 缓存键生成策略

**哈希算法选择**：

LangChain 提供了多种哈希算法：
```python
def _make_default_key_encoder(namespace: str, algorithm: str) -> Callable[[str], str]:
    """Create a default key encoder function.

    Args:
        algorithm:
           * 'sha1' - fast but not collision-resistant
           * 'blake2b' - cryptographically strong, faster than SHA-1
           * 'sha256' - cryptographically strong, slower than SHA-1
           * 'sha512' - cryptographically strong, slower than SHA-1
    """
```

**默认算法：SHA-1**

**为什么使用 SHA-1？**（来自源码注释）：
> "Deterministic and fast, **but not collision-resistant**. A malicious attacker could try to create two different texts that hash to the same UUID."

**安全警告**：
```python
def _warn_about_sha1_encoder() -> None:
    warnings.warn(
        "Using default key encoder: SHA-1 is *not* collision-resistant. "
        "While acceptable for most cache scenarios, a motivated attacker "
        "can craft two different payloads that map to the same cache key. "
        "If that risk matters in your environment, supply a stronger "
        "encoder (e.g. SHA-256 or BLAKE2) via the `key_encoder` argument.",
        category=UserWarning,
        stacklevel=2,
    )
```

**实际应用建议**：
- 一般场景：SHA-1 足够（性能优先）
- 安全敏感场景：使用 BLAKE2b 或 SHA-256

### 3. 缓存策略：文档 vs 查询

**默认行为**：
- **文档嵌入**：默认缓存
- **查询嵌入**：默认不缓存

**设计理由**（来自源码注释）：
> "Note that by default only document embeddings are cached. To cache query embeddings too, pass in a query_embedding_store to constructor."

**为什么查询默认不缓存？**
1. **查询多样性**：用户查询通常不重复
2. **缓存效率低**：查询缓存命中率低
3. **存储成本**：避免存储大量低价值缓存

**何时缓存查询？**
- 固定查询集（如预定义问题）
- 查询模板（如 Few-shot 示例）
- 评估和测试场景

### 4. 批量处理与缓存更新

**批量处理逻辑**：
```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    # 1. 从缓存中批量获取
    vectors: list[list[float] | None] = self.document_embedding_store.mget(texts)

    # 2. 找出缺失的文本
    all_missing_indices: list[int] = [
        i for i, vector in enumerate(vectors) if vector is None
    ]

    # 3. 分批处理缺失的文本
    for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
        missing_texts = [texts[i] for i in missing_indices]
        missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)

        # 4. 更新缓存
        self.document_embedding_store.mset(
            list(zip(missing_texts, missing_vectors, strict=False))
        )

        # 5. 更新结果
        for index, updated_vector in zip(missing_indices, missing_vectors, strict=False):
            vectors[index] = updated_vector

    return cast("list[list[float]]", vectors)
```

**关键设计点**：
1. **批量获取**：使用 `mget` 一次性获取所有缓存
2. **增量计算**：只计算缺失的嵌入
3. **批量更新**：使用 `mset` 批量更新缓存
4. **分批处理**：支持 `batch_size` 参数控制批次大小

### 5. 异步支持

**异步实现**：
```python
async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    # 使用异步版本的 store 方法
    vectors: list[list[float] | None] = await self.document_embedding_store.amget(texts)

    # ... 类似的逻辑

    missing_vectors = await self.underlying_embeddings.aembed_documents(missing_texts)
    await self.document_embedding_store.amset(...)
```

**关键点**：
- 使用 `amget` / `amset` 异步方法
- 调用底层 Embeddings 的异步方法
- 完全异步的执行路径

### 6. 工厂方法：from_bytes_store

**便捷构造器**：
```python
@classmethod
def from_bytes_store(
    cls,
    underlying_embeddings: Embeddings,
    document_embedding_cache: ByteStore,
    *,
    namespace: str = "",
    batch_size: int | None = None,
    query_embedding_cache: bool | ByteStore = False,
    key_encoder: Callable[[str], str] | Literal["sha1", "blake2b", "sha256", "sha512"] = "sha1",
) -> CacheBackedEmbeddings:
```

**设计目的**：
1. **简化使用**：自动处理序列化和编码
2. **类型转换**：将 ByteStore 转换为 BaseStore[str, list[float]]
3. **默认配置**：提供合理的默认值

**使用示例**（来自源码文档）：
```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

store = LocalFileStore("./my_cache")
underlying_embedder = OpenAIEmbeddings()

embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embedder,
    store,
    namespace=underlying_embedder.model
)

# 第一次调用：计算并缓存
embeddings = embedder.embed_documents(["hello", "goodbye"])

# 第二次调用：从缓存读取
embeddings = embedder.embed_documents(["hello", "goodbye"])
```

### 7. 序列化与反序列化

**值序列化**：
```python
def _value_serializer(value: Sequence[float]) -> bytes:
    """Serialize a value."""
    return json.dumps(value).encode()

def _value_deserializer(serialized_value: bytes) -> list[float]:
    """Deserialize a value."""
    return cast("list[float]", json.loads(serialized_value.decode()))
```

**设计选择**：
- 使用 JSON 序列化（而非 pickle）
- 优点：跨语言兼容、人类可读、安全
- 缺点：比 pickle 慢、占用空间大

**EncoderBackedStore 包装**：
```python
document_embedding_store = EncoderBackedStore[str, list[float]](
    document_embedding_cache,
    key_encoder,
    _value_serializer,
    _value_deserializer,
)
```

## 架构决策分析

### 决策1：为什么默认不缓存查询？

**权衡**：
- **优点**：节省存储空间、避免低效缓存
- **缺点**：某些场景下查询缓存有价值

**设计哲学**：
- 提供灵活性（可选缓存查询）
- 默认行为符合大多数场景

### 决策2：为什么使用 SHA-1 作为默认哈希？

**权衡**：
- **性能**：SHA-1 比 SHA-256 快约 2 倍
- **安全性**：SHA-1 存在碰撞风险

**设计哲学**：
- 缓存场景下碰撞风险可接受
- 提供警告和替代选项
- 性能优先

### 决策3：为什么支持 batch_size？

**场景**：
- 大量文档需要嵌入
- 避免一次性计算过多嵌入（内存/API 限制）

**设计**：
```python
for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
    # 分批处理
```

## 与 RAG 开发的联系

### 在 RAG 中的使用

**场景1：文档索引构建**
```python
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 创建缓存
store = LocalFileStore("./embeddings_cache")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(),
    store,
    namespace="openai-ada-002"
)

# 第一次运行：计算并缓存
vectors = embeddings.embed_documents(documents)

# 后续运行：从缓存读取（节省 API 调用）
vectors = embeddings.embed_documents(documents)
```

**场景2：增量索引更新**
```python
# 新文档会被计算，旧文档从缓存读取
new_documents = ["new doc 1", "new doc 2"]
all_documents = existing_documents + new_documents
vectors = embeddings.embed_documents(all_documents)
```

### 性能优化建议

1. **选择合适的哈希算法**：
   - 一般场景：SHA-1（默认）
   - 安全敏感：BLAKE2b

2. **设置合理的 batch_size**：
   - 小批次：更频繁的缓存更新
   - 大批次：更少的 I/O 操作

3. **使用 namespace**：
   - 区分不同模型的缓存
   - 避免缓存冲突

## 总结

CacheBackedEmbeddings 的设计体现了以下原则：
1. **装饰器模式**：不侵入原始实现
2. **灵活的缓存策略**：文档默认缓存，查询可选
3. **批量优化**：支持批量获取和更新
4. **安全性考虑**：提供多种哈希算法选择
5. **异步支持**：完整的异步实现

这种设计使得缓存功能既易于使用，又足够灵活以适应不同场景。
