---
type: source_code_analysis
source: sourcecode/langchain/libs/partners/qdrant/langchain_qdrant/vectorstores.py
analyzed_files:
  - sourcecode/langchain/libs/partners/qdrant/langchain_qdrant/vectorstores.py
analyzed_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
---

# 源码分析：Qdrant 集成实现

## 分析的文件

- `sourcecode/langchain/libs/partners/qdrant/langchain_qdrant/vectorstores.py` - Qdrant 向量存储集成

## 关键发现

### 1. Qdrant 特点

Qdrant 是一个生产级的向量数据库：
- **高性能**：Rust 实现，性能优异
- **生产就绪**：支持分布式部署、高可用
- **丰富功能**：支持过滤、分片、快照等企业级功能
- **适合场景**：生产环境、大规模应用、高并发场景

### 2. 核心依赖

```python
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
from langchain_core.vectorstores import VectorStore
```

Qdrant 集成依赖：
- `qdrant-client` - Qdrant 官方客户端
- `langchain_core` - LangChain 核心抽象

### 3. 异步支持装饰器

```python
def sync_call_fallback(method: Callable) -> Callable:
    """异步方法回退到同步方法的装饰器

    如果异步方法未实现，自动调用对应的同步方法。
    这个装饰器只应用于类中定义为 async 的方法。
    """
    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            # 如果异步方法未实现，调用同步方法
            # 通过移除方法名的第一个字母（'a'）来获取同步方法名
            # 例如：aadd_texts -> add_texts
            return await run_in_executor(
                None, getattr(self, method.__name__[1:]), *args, **kwargs
            )

    return wrapper
```

### 4. Qdrant 类定义

```python
@deprecated(since="0.1.2", alternative="QdrantVectorStore", removal="0.5.0")
class Qdrant(VectorStore):
    """`Qdrant` vector store.

    Example:
        ```python
        from qdrant_client import QdrantClient
        from langchain_qdrant import Qdrant

        client = QdrantClient()
        collection_name = "MyCollection"
        qdrant = Qdrant(client, collection_name, embedding_function)
        ```
    """

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str | None = None

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embeddings: Embeddings | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: str | None = VECTOR_NAME,
        async_client: Any | None = None,
        embedding_function: Callable | None = None,  # deprecated
    ) -> None:
        """Initialize with necessary components."""

        # 验证客户端类型
        if not isinstance(client, QdrantClient):
            raise TypeError(
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )

        if async_client is not None and not isinstance(async_client, AsyncQdrantClient):
            raise ValueError(
                f"async_client should be an instance of qdrant_client.AsyncQdrantClient"
                f"got {type(async_client)}"
            )

        # 验证 embeddings 参数
        if embeddings is None and embedding_function is None:
            raise ValueError("`embeddings` value can't be None. Pass `embeddings` instance.")

        if embeddings is not None and embedding_function is not None:
            raise ValueError(
                "Both `embeddings` and `embedding_function` are passed. "
                "Use `embeddings` only."
            )

        # 设置实例变量
        self._embeddings = embeddings
        self._embeddings_function = embedding_function
        self.client: QdrantClient = client
        self.async_client: AsyncQdrantClient | None = async_client
        self.collection_name = collection_name
        self.content_payload_key = content_payload_key or self.CONTENT_KEY
        self.metadata_payload_key = metadata_payload_key or self.METADATA_KEY
        self.vector_name = vector_name or self.VECTOR_NAME

        # 处理 deprecated 参数
        if embedding_function is not None:
            warnings.warn(
                "Using `embedding_function` is deprecated. "
                "Pass `Embeddings` instance to `embeddings` instead.",
                stacklevel=2,
            )

        if not isinstance(embeddings, Embeddings):
            warnings.warn(
                "`embeddings` should be an instance of `Embeddings`."
                "Using `embeddings` as `embedding_function` which is deprecated",
                stacklevel=2,
            )
            self._embeddings_function = embeddings
            self._embeddings = None

        # 距离策略
        self.distance_strategy = distance_strategy.upper()

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embeddings
```

## 代码片段

### 初始化示例

```python
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

# 方式1：内存模式（开发测试）
client = QdrantClient(":memory:")

# 方式2：本地持久化
client = QdrantClient(path="/path/to/db")

# 方式3：远程服务器
client = QdrantClient(
    url="http://localhost:6333",
    api_key="your-api-key"  # 可选
)

# 创建 Qdrant 向量存储
qdrant = Qdrant(
    client=client,
    collection_name="my_documents",
    embeddings=OpenAIEmbeddings(),
    distance_strategy="COSINE",  # 或 "EUCLID", "DOT"
)
```

### 距离策略

Qdrant 支持三种距离策略：

1. **COSINE**（余弦相似度）
   - 适用于归一化向量
   - 范围：[-1, 1]
   - 常用于文本 embedding

2. **EUCLID**（欧几里得距离）
   - 适用于未归一化向量
   - 范围：[0, ∞)
   - 常用于图像 embedding

3. **DOT**（点积）
   - 适用于归一化向量
   - 范围：[-1, 1]
   - 性能最快

### 异步客户端支持

```python
from qdrant_client import AsyncQdrantClient

# 创建异步客户端
async_client = AsyncQdrantClient(url="http://localhost:6333")

# 创建 Qdrant 向量存储（支持异步操作）
qdrant = Qdrant(
    client=client,  # 同步客户端
    async_client=async_client,  # 异步客户端
    collection_name="my_documents",
    embeddings=OpenAIEmbeddings(),
)

# 异步添加文档
await qdrant.aadd_texts(texts=["doc1", "doc2"])

# 异步检索
results = await qdrant.asimilarity_search("query")
```

## 架构设计要点

1. **双客户端支持**：同时支持同步和异步客户端
2. **异步回退机制**：异步方法未实现时自动回退到同步方法
3. **灵活的距离策略**：支持多种距离计算方式
4. **类型安全**：严格的类型检查和验证
5. **向后兼容**：保留 deprecated 参数，提供迁移路径

## 优缺点分析

### 优点
1. **生产就绪**：支持分布式、高可用、备份恢复
2. **高性能**：Rust 实现，性能优异
3. **丰富功能**：支持过滤、分片、快照、集群等
4. **完整异步支持**：原生异步 API，适合高并发
5. **灵活部署**：支持内存、本地、远程多种模式

### 缺点
1. **部署复杂度**：相比 Chroma 需要额外的服务部署
2. **学习曲线**：功能丰富，需要学习更多概念
3. **资源消耗**：相比轻量级方案消耗更多资源

## 适用场景

- ✅ 生产环境
- ✅ 大规模应用（> 100K 文档）
- ✅ 高并发场景
- ✅ 需要分布式部署
- ✅ 需要高可用和备份
- ❌ 简单原型（过于复杂）
- ❌ 资源受限环境

## 与其他后端的对比

| 特性 | Qdrant | Chroma | FAISS | Pinecone |
|------|--------|--------|-------|----------|
| 持久化 | ✅ | ✅ | ✅ | ✅ |
| 分布式 | ✅ | ❌ | ❌ | ✅ |
| 高可用 | ✅ | ❌ | ❌ | ✅ |
| 性能 | 高 | 中 | 高 | 高 |
| 部署复杂度 | 中 | 低 | 低 | 低（托管） |
| 适合规模 | 无限 | < 100K | < 1M | 无限 |
| 异步支持 | ✅ 原生 | ❌ | ❌ | ✅ |
| 成本 | 免费（自托管） | 免费 | 免费 | 付费 |

## 部署模式

### 1. 内存模式（开发测试）
```python
client = QdrantClient(":memory:")
```

### 2. 本地持久化
```python
client = QdrantClient(path="/path/to/db")
```

### 3. Docker 部署
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Kubernetes 部署
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant
  replicas: 3
  ...
```

### 5. 云托管
- Qdrant Cloud（官方托管服务）
- AWS/GCP/Azure 自托管
