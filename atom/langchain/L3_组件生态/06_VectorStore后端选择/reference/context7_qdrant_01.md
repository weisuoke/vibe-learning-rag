---
type: context7_documentation
library: Qdrant
library_id: /websites/qdrant_tech
fetched_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
context7_query: Qdrant vector database integration with LangChain
---

# Context7 文档：Qdrant 官方文档

## 文档来源
- 库名称：Qdrant
- Library ID：/websites/qdrant_tech
- 官方文档链接：https://qdrant.tech/documentation/frameworks/langchain
- Source Reputation：High
- Benchmark Score：89.3

## 关键信息提取

### 1. LangChain 集成方式

#### 从文本创建 Qdrant 向量存储

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

doc_store = QdrantVectorStore.from_texts(
    texts,
    embeddings,
    url="<qdrant-url>",
    api_key="<qdrant-api-key>",
    collection_name="texts"
)
```

**关键点**：
- 使用 `langchain_qdrant` 包进行集成
- 需要提供 Qdrant URL 和 API key
- 支持从文本列表直接创建向量存储

### 2. 混合检索支持

Qdrant 支持混合检索（dense + sparse embeddings）：

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant import QdrantVectorStore

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant = QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    location=":memory:",
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.HYBRID,
)

query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search(query)
```

**关键点**：
- 支持 `RetrievalMode.HYBRID` 混合检索模式
- 可以同时使用 dense 和 sparse embeddings
- `FastEmbedSparse` 提供 BM25 稀疏向量支持

### 3. 本地内存模式

适合测试和快速实验：

```python
from langchain_qdrant import QdrantVectorStore

qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:",  # 本地内存模式
    collection_name="my_documents",
)
```

**关键点**：
- `location=":memory:"` 启用内存模式
- 无需 Qdrant 服务器
- 数据在客户端销毁时丢失
- 适合测试和原型开发

### 4. Qdrant 客户端初始化

```python
from qdrant_client import QdrantClient, models

# 连接到本地 Qdrant 实例
client = QdrantClient(url="http://localhost:6333/")
```

**Docker 部署**：
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. 检索操作

```python
client.search(
    collection_name=collection_name,
    query_vector=openai_client.embeddings.create(
        input=["What is the best to use for vector search scaling?"],
        model=embedding_model,
    )
    .data[0]
    .embedding,
)
```

**关键点**：
- 使用 `client.search()` 进行向量检索
- 需要提供查询向量（通过 embedding 模型生成）
- 返回最相关的文档

## 集成优势

1. **混合检索**：支持 dense + sparse 向量的混合检索
2. **灵活部署**：支持内存、本地、远程多种模式
3. **高性能**：Rust 实现，性能优异
4. **生产就绪**：支持分布式、高可用、备份恢复

## 适用场景

- ✅ 生产环境部署
- ✅ 需要混合检索的场景
- ✅ 大规模向量数据（> 100K）
- ✅ 需要高性能和高可用
- ✅ 本地开发和测试（内存模式）

## 与 LangChain 的集成要点

1. **包依赖**：`langchain-qdrant`
2. **初始化方式**：
   - `from_texts()` - 从文本列表创建
   - `from_documents()` - 从文档列表创建
3. **检索模式**：
   - `RetrievalMode.DENSE` - 仅 dense 向量
   - `RetrievalMode.SPARSE` - 仅 sparse 向量
   - `RetrievalMode.HYBRID` - 混合检索
4. **部署选项**：
   - `:memory:` - 内存模式
   - `path="/path/to/db"` - 本地持久化
   - `url="http://..."` - 远程服务器
