---
type: context7_documentation
library: langchain
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
context7_query: EnsembleRetriever hybrid search, ContextualCompressionRetriever reranking, MultiQueryRetriever, ParentDocumentRetriever, SelfQueryRetriever
---

# Context7 文档：LangChain Retriever 高级策略

## 文档来源
- 库名称：LangChain
- 版本：latest
- 官方文档链接：https://docs.langchain.com

## 1. 混合检索（Hybrid Search）

### ElasticsearchStore 混合检索
- 使用 `DenseVectorStrategy(hybrid=True)` 配置
- 结合近似语义搜索和关键词搜索
- 使用 RRF（Reciprocal Rank Fusion）平衡两种检索方法的分数
- 需要 Elasticsearch 8.9.0+

### Pinecone 混合检索
```python
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)
```

### Oracle 混合检索
- OracleVectorizerPreference 创建数据库端向量化偏好
- create_hybrid_index 创建混合向量索引
- OracleHybridSearchRetriever 执行关键词、语义或混合检索

## 2. 重排序（Reranking）

### ContextualCompressionRetriever + FlashrankRerank
```python
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
compressed_docs = compression_retriever.invoke("query")
```

### ContextualCompressionRetriever + CohereRerank
```python
from langchain_cohere import CohereRerank
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
```

### 其他重排序选项
- JinaRerank（langchain_community）
- VoyageAIRerank（langchain_voyageai）
- LLMLinguaCompressor（文档压缩）

## 3. MultiQueryRetriever
```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
docs = mqr.invoke("query")
```
- 使用 LLM 生成多个查询变体
- 为每个变体检索文档并返回唯一并集
- 可集成到 LCEL 管道中

## 4. MultiVectorRetriever
```python
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

vectorstore = Chroma(collection_name="big2small", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=store,
    search_type=SearchType.mmr, search_kwargs={"k": 2},
)
```
- 子块索引在向量存储中用于语义搜索
- 父文档存储在 docstore 中
- 搜索子块 → 收集父 ID → 获取父文档

## 5. SelfQueryRetriever
```python
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="name", description="The name", type="string"),
]
retriever = SelfQueryRetriever.from_llm(
    llm, db, document_content_description, metadata_field_info,
)
docs = retriever.invoke("Which person is not active?")
```
- LLM 将自然语言转为结构化查询 + 元数据过滤
- 支持 20+ 向量存储后端
