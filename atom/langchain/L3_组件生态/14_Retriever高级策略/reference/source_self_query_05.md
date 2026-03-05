---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/retrievers/self_query/base.py
analyzed_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
---

# 源码分析：SelfQueryRetriever

## 分析的文件
- `libs/langchain/langchain_classic/retrievers/self_query/base.py` - 自查询检索器

## 关键发现

### 类：SelfQueryRetriever(BaseRetriever)
使用 LLM 将自然语言查询解析为结构化查询（语义查询 + 元数据过滤器），然后在向量存储上执行。

### 核心属性
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vectorstore` | `VectorStore` | (必需) | 要查询的向量存储 |
| `query_constructor` | `Runnable[dict, StructuredQuery]` | (必需) | 将自然语言转换为 StructuredQuery 的 LLM 链 |
| `search_type` | `str` | `"similarity"` | 搜索方法名 |
| `structured_query_translator` | `Visitor` | (自动检测) | 将 StructuredQuery 翻译为向量存储原生过滤语法 |
| `use_original_query` | `bool` | `False` | 是否忽略 LLM 重写的查询文本 |

### 支持的向量存储（20+）
Chroma, Pinecone, Milvus, Qdrant, PGVector, Elasticsearch, Weaviate, Neo4j, AstraDB, MongoDB Atlas, Redis, Supabase 等。

### 核心流程
1. 调用 `query_constructor` 将自然语言转为 `StructuredQuery`
2. `_prepare_query` 通过 Visitor 翻译为向量存储原生过滤
3. 调用 `vectorstore.search(new_query, search_type, **kwargs)`

### 设计模式
- **访问者模式**：StructuredQuery 由 Visitor 遍历，生成向量存储特定的过滤语法
- **工厂方法**：`from_llm` 封装 LLM + 翻译器 + 向量存储的复杂连接
- **策略模式**：search_type 在运行时选择搜索方法
