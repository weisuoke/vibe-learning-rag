# 实战代码 - 场景4：Qdrant 生产部署

> **来源**：基于 `reference/source_qdrant_04.md` 源码分析 + `reference/context7_qdrant_01.md` 官方文档 + `reference/search_vectordb_production_01.md` 生产部署对比

## 场景说明

Qdrant 是一个生产级的向量数据库，适合：
- 生产环境部署
- 大规模应用（> 100K 文档）
- 高并发场景
- 需要分布式部署
- 需要高可用和备份

**核心特点**：
- 高性能：Rust 实现，性能优异
- 生产就绪：支持分布式、高可用、备份恢复
- 丰富功能：支持过滤、分片、快照、集群等
- 完整异步支持：原生异步 API，适合高并发
- 灵活部署：支持内存、本地、远程多种模式

## 环境准备

```bash
# 安装依赖
pip install langchain-qdrant qdrant-client langchain-openai

# 启动 Qdrant 服务器（Docker）
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 完整代码示例

### 示例1：基础生产部署

```python
"""
Qdrant 生产部署 - 基础示例
演示如何使用 Qdrant 进行生产环境部署
"""

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()


def basic_production_deployment():
    """基础生产部署示例"""
    print("=" * 60)
    print("场景1：Qdrant 生产部署 - 基础示例")
    print("=" * 60)

    # 1. 连接到 Qdrant 服务器
    print("\n[步骤1] 连接到 Qdrant 服务器...")
    client = QdrantClient(url="http://localhost:6333")
    print("✓ 连接成功")

    # 2. 创建 collection（如果不存在）
    collection_name = "production_docs"
    print(f"\n[步骤2] 创建 collection: {collection_name}...")
    
    # 检查 collection 是否存在
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if not collection_exists:
        # 创建 collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI text-embedding-3-small 维度
                distance=Distance.COSINE
            )
        )
        print(f"✓ Collection '{collection_name}' 创建成功")
    else:
        print(f"✓ Collection '{collection_name}' 已存在")

    # 3. 初始化 QdrantVectorStore
    print("\n[步骤3] 初始化 QdrantVectorStore...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    print("✓ 初始化完成")

    # 4. 添加文档
    print("\n[步骤4] 添加文档...")
    documents = [
        Document(
            page_content="Qdrant 是一个高性能的向量数据库。",
            metadata={"source": "intro.txt", "category": "database"}
        ),
        Document(
            page_content="Qdrant 使用 Rust 实现，性能优异。",
            metadata={"source": "tech.txt", "category": "performance"}
        ),
        Document(
            page_content="Qdrant 支持分布式部署和高可用。",
            metadata={"source": "deploy.txt", "category": "deployment"}
        ),
    ]
    
    start_time = time.time()
    ids = vector_store.add_documents(documents)
    elapsed = time.time() - start_time
    
    print(f"✓ 添加了 {len(ids)} 个文档，耗时: {elapsed:.2f}秒")
    print(f"  文档 IDs: {ids}")

    # 5. 相似度检索
    print("\n[步骤5] 相似度检索...")
    query = "Qdrant 的性能如何？"
    print(f"  查询: {query}")

    start_time = time.time()
    results = vector_store.similarity_search(query, k=3)
    elapsed = time.time() - start_time

    print(f"✓ 检索完成，耗时: {elapsed*1000:.2f}ms")
    print(f"  返回 {len(results)} 个结果:")
    for i, doc in enumerate(results, 1):
        print(f"\n  结果 {i}:")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category')}")

    # 6. 带分数的检索
    print("\n[步骤6] 带分数的相似度检索...")
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print("✓ 检索完成，结果（按相似度排序）:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n  结果 {i} (相似度: {score:.4f}):")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category')}")

    return vector_store, client


def metadata_filtering():
    """元数据过滤示例"""
    print("\n" + "=" * 60)
    print("场景2：元数据过滤")
    print("=" * 60)

    # 连接到 Qdrant
    client = QdrantClient(url="http://localhost:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    collection_name = "filtered_docs"
    
    # 创建 collection
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # 添加带元数据的文档
    documents = [
        Document(
            page_content="Python 3.13 发布了新特性。",
            metadata={"language": "python", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="JavaScript ES2024 标准发布。",
            metadata={"language": "javascript", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="Python 教程：入门指南。",
            metadata={"language": "python", "year": 2023, "type": "tutorial"}
        ),
        Document(
            page_content="Go 语言并发编程。",
            metadata={"language": "go", "year": 2024, "type": "tutorial"}
        ),
    ]
    
    vector_store.add_documents(documents)

    # 1. 单条件过滤
    print("\n[示例1] 单条件过滤 (language=python):")
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.language",
                match=MatchValue(value="python")
            )
        ]
    )
    
    results = vector_store.similarity_search(
        "编程语言",
        k=5,
        filter=filter_condition
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")

    # 2. 多条件过滤（AND）
    print("\n[示例2] 多条件过滤 (language=python AND year=2024):")
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.language",
                match=MatchValue(value="python")
            ),
            FieldCondition(
                key="metadata.year",
                match=MatchValue(value=2024)
            )
        ]
    )
    
    results = vector_store.similarity_search(
        "编程",
        k=5,
        filter=filter_condition
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")


def hybrid_search():
    """混合检索示例（Dense + Sparse）"""
    print("\n" + "=" * 60)
    print("场景3：混合检索（Dense + Sparse）")
    print("=" * 60)

    from langchain_qdrant import FastEmbedSparse, RetrievalMode
    
    # 初始化
    client = QdrantClient(url="http://localhost:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    collection_name = "hybrid_docs"
    
    # 创建 collection
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    
    # 创建混合检索向量存储
    vector_store = QdrantVectorStore.from_documents(
        documents=[
            Document(page_content="Qdrant 支持混合检索。"),
            Document(page_content="混合检索结合了 dense 和 sparse 向量。"),
            Document(page_content="BM25 是一种稀疏向量算法。"),
        ],
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    # 混合检索
    print("\n[混合检索] 查询: 'Qdrant 混合检索'")
    results = vector_store.similarity_search("Qdrant 混合检索", k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")


def async_operations():
    """异步操作示例"""
    print("\n" + "=" * 60)
    print("场景4：异步操作")
    print("=" * 60)

    import asyncio
    from qdrant_client import AsyncQdrantClient
    
    async def async_example():
        # 创建异步客户端
        async_client = AsyncQdrantClient(url="http://localhost:6333")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        collection_name = "async_docs"
        
        # 创建 collection
        collections = await async_client.get_collections()
        if not any(c.name == collection_name for c in collections.collections):
            await async_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        # 创建向量存储（使用同步客户端 + 异步客户端）
        sync_client = QdrantClient(url="http://localhost:6333")
        vector_store = QdrantVectorStore(
            client=sync_client,
            async_client=async_client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # 异步添加文档
        print("\n[异步操作] 添加文档...")
        documents = [
            Document(page_content=f"异步文档 {i}")
            for i in range(10)
        ]
        
        start_time = time.time()
        await vector_store.aadd_documents(documents)
        elapsed = time.time() - start_time
        print(f"✓ 异步添加完成，耗时: {elapsed:.2f}秒")
        
        # 异步检索
        print("\n[异步操作] 检索文档...")
        start_time = time.time()
        results = await vector_store.asimilarity_search("异步", k=3)
        elapsed = time.time() - start_time
        print(f"✓ 异步检索完成，耗时: {elapsed*1000:.2f}ms")
        print(f"  返回 {len(results)} 个结果")
    
    # 运行异步示例
    asyncio.run(async_example())


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("场景5：性能基准测试")
    print("=" * 60)

    client = QdrantClient(url="http://localhost:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 测试不同数据规模
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n[测试] 数据规模: {size} 个文档")
        
        collection_name = f"benchmark_{size}"
        
        # 创建 collection
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            client.delete_collection(collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # 生成数据
        documents = [
            Document(page_content=f"文档 {i}：测试内容。" * 10)
            for i in range(size)
        ]
        
        # 添加文档
        start_time = time.time()
        vector_store.add_documents(documents)
        build_time = time.time() - start_time
        
        # 检索测试
        query = "测试查询"
        search_times = []
        
        for _ in range(10):  # 多次测试取平均
            start_time = time.time()
            vector_store.similarity_search(query, k=10)
            search_times.append(time.time() - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        print(f"  添加时间: {build_time:.2f}秒")
        print(f"  平均检索时间: {avg_search_time*1000:.2f}ms")
        print(f"  吞吐量: {1/avg_search_time:.1f} 查询/秒")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Qdrant 生产部署 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    basic_production_deployment()
    metadata_filtering()
    hybrid_search()
    async_operations()
    performance_benchmark()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 运行说明

```bash
# 1. 启动 Qdrant 服务器
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 3. 运行示例
python 07_实战代码_场景4_Qdrant生产部署.py
```

## 最佳实践

### 1. 部署模式选择

**内存模式**（开发测试）：
```python
client = QdrantClient(":memory:")
```

**本地持久化**（单机部署）：
```python
client = QdrantClient(path="/path/to/db")
```

**远程服务器**（生产部署）：
```python
client = QdrantClient(
    url="http://localhost:6333",
    api_key="your-api-key"  # 可选
)
```

### 2. Collection 配置

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE  # 或 EUCLID, DOT
    )
)
```

### 3. 距离策略选择

- **COSINE**：归一化向量（文本 embedding）
- **EUCLID**：未归一化向量
- **DOT**：点积（性能最快）

### 4. 异步操作

```python
# 高并发场景使用异步
async_client = AsyncQdrantClient(url="...")
await vector_store.aadd_documents(documents)
results = await vector_store.asimilarity_search(query)
```

## 常见问题

### Q1: Qdrant 和 Chroma 如何选择？

**Qdrant**：
- 生产就绪（分布式、高可用）
- 高性能（Rust 实现）
- 适合大规模应用

**Chroma**：
- 易于使用
- 适合本地开发
- 适合中小规模

### Q2: 如何配置 Qdrant 集群？

```yaml
# docker-compose.yml
version: '3'
services:
  qdrant1:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
  qdrant2:
    image: qdrant/qdrant
    ports:
      - "6334:6333"
```

### Q3: Qdrant 的内存占用如何？

粗略估算：
- 每个向量（1536 维）：约 6KB
- 1000 个文档：约 6MB
- 100K 个文档：约 600MB

### Q4: 如何备份 Qdrant 数据？

```python
# 创建快照
client.create_snapshot(collection_name="my_collection")

# 恢复快照
client.recover_snapshot(
    collection_name="my_collection",
    snapshot_path="/path/to/snapshot"
)
```

### Q5: Qdrant 支持多大规模的数据？

- < 100K 文档：单机部署
- 100K - 1M 文档：单机或小集群
- > 1M 文档：分布式集群

## 总结

Qdrant 是生产环境的首选向量数据库，适合：
- **生产部署**：分布式、高可用、备份恢复
- **高性能**：Rust 实现，性能优异
- **大规模应用**：> 100K 文档

**核心优势**：
- 生产就绪（分布式、高可用）
- 高性能（Rust 实现）
- 丰富功能（过滤、分片、快照）
- 完整异步支持

**主要限制**：
- 部署复杂度（相比 Chroma）
- 学习曲线（功能丰富）
- 资源消耗（相比轻量级方案）

**下一步**：
- 需要云端托管 → 使用 Pinecone
- 需要企业级 → 使用 Milvus
- 需要简单部署 → 使用 Chroma
