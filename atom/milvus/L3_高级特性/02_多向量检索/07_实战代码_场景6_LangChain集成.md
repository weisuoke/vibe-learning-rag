# 实战代码 - 场景6：LangChain 集成

> 本文展示如何在 LangChain 中使用 Milvus 的多向量检索功能

---

## 场景概述

**目标**：在 LangChain 框架中集成 Milvus 多向量检索，实现稠密向量 + 稀疏向量（BM25）的混合检索。

**应用场景**：
- RAG 系统中的混合检索
- 多模型 Embedding 融合
- 语义理解 + 关键词匹配

**技术栈**：
- LangChain
- Milvus
- OpenAI Embeddings
- BM25 内置函数

---

## 核心概念

### 1. LangChain Milvus 集成

LangChain 提供了 `Milvus` 向量存储类，支持：
- 多个 Embedding 模型
- 多向量字段定义
- BM25 内置函数
- 混合检索

### 2. 多 Embedding 模型

```python
from langchain_openai import OpenAIEmbeddings

# 定义多个 Embedding 模型
embedding1 = OpenAIEmbeddings(model="text-embedding-ada-002")
embedding2 = OpenAIEmbeddings(model="text-embedding-3-large")

# 传入列表
embedding=[embedding1, embedding2]
```

### 3. BM25 内置函数

```python
from langchain_community.vectorstores.utils import BM25BuiltInFunction

# 定义 BM25 函数
builtin_function=BM25BuiltInFunction(output_field_names="sparse")
```

---

## 完整实现

### 场景1：基础多向量集成

```python
"""
LangChain Milvus 多向量检索 - 基础集成
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.vectorstores.utils import BM25BuiltInFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Milvus 连接配置
URI = os.getenv("MILVUS_URI", "http://localhost:19530")

def create_multi_vector_store():
    """创建支持多向量的 Milvus 向量存储"""
    
    # 1. 准备文档数据
    documents = [
        Document(
            page_content="Milvus is a vector database for AI applications",
            metadata={"source": "doc1", "category": "database"}
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications",
            metadata={"source": "doc2", "category": "framework"}
        ),
        Document(
            page_content="Hybrid search combines dense and sparse vectors",
            metadata={"source": "doc3", "category": "search"}
        ),
        Document(
            page_content="RAG retrieval augmented generation improves LLM accuracy",
            metadata={"source": "doc4", "category": "technique"}
        ),
    ]
    
    # 2. 定义多个 Embedding 模型
    embedding1 = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding2 = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 3. 创建 Milvus 向量存储（支持多向量字段）
    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=[embedding1, embedding2],  # 多个 Embedding 模型
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),  # BM25 函数
        vector_field=["dense1", "dense2", "sparse"],  # 向量字段名
        connection_args={"uri": URI},
        collection_name="langchain_multi_vector",
        drop_old=True,  # 删除旧 Collection
    )
    
    print(f"✓ 创建向量存储成功")
    print(f"  向量字段: {vectorstore.vector_fields}")
    
    return vectorstore


def basic_search(vectorstore):
    """基础检索测试"""
    
    query = "What is Milvus?"
    
    # 执行检索
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"\n查询: {query}")
    print(f"结果数量: {len(results)}")
    
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {doc.page_content}")
        print(f"  元数据: {doc.metadata}")


def main():
    """主函数"""
    
    print("=" * 60)
    print("LangChain Milvus 多向量检索 - 基础集成")
    print("=" * 60)
    
    # 创建向量存储
    vectorstore = create_multi_vector_store()
    
    # 基础检索测试
    basic_search(vectorstore)


if __name__ == "__main__":
    main()
```

**输出示例**：
```
============================================================
LangChain Milvus 多向量检索 - 基础集成
============================================================
✓ 创建向量存储成功
  向量字段: ['dense1', 'dense2', 'sparse']

查询: What is Milvus?
结果数量: 2

结果 1:
  内容: Milvus is a vector database for AI applications
  元数据: {'source': 'doc1', 'category': 'database'}

结果 2:
  内容: Hybrid search combines dense and sparse vectors
  元数据: {'source': 'doc3', 'category': 'search'}
```

---

### 场景2：混合检索配置

```python
"""
LangChain Milvus 多向量检索 - 混合检索配置
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.vectorstores.utils import BM25BuiltInFunction
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MILVUS_URI", "http://localhost:19530")


def create_hybrid_vectorstore():
    """创建支持混合检索的向量存储"""
    
    # 准备文档
    documents = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning uses algorithms"),
        Document(page_content="Deep learning is a subset of ML"),
        Document(page_content="Neural networks mimic brain structure"),
        Document(page_content="Transformers revolutionized NLP"),
    ]
    
    # 定义 Embedding
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 创建向量存储
    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
        vector_field=["dense", "sparse"],
        connection_args={"uri": URI},
        collection_name="hybrid_search_demo",
        drop_old=True,
    )
    
    return vectorstore


def hybrid_search_with_weights(vectorstore):
    """使用权重的混合检索"""
    
    query = "What is deep learning?"
    
    # 方法1：使用 similarity_search（默认混合检索）
    print("\n方法1：默认混合检索")
    results = vectorstore.similarity_search(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    # 方法2：使用 similarity_search_with_score
    print("\n方法2：带分数的混合检索")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{i}. [分数: {score:.4f}] {doc.page_content}")


def filter_search(vectorstore):
    """带过滤条件的混合检索"""
    
    # 添加带元数据的文档
    docs_with_metadata = [
        Document(
            page_content="Python 3.11 released with performance improvements",
            metadata={"year": 2023, "category": "release"}
        ),
        Document(
            page_content="Python 3.10 introduced pattern matching",
            metadata={"year": 2021, "category": "feature"}
        ),
    ]
    
    vectorstore.add_documents(docs_with_metadata)
    
    # 带过滤的检索
    query = "Python improvements"
    filter_expr = "year >= 2023"
    
    results = vectorstore.similarity_search(
        query,
        k=2,
        expr=filter_expr  # 过滤表达式
    )
    
    print(f"\n查询: {query}")
    print(f"过滤: {filter_expr}")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   元数据: {doc.metadata}")


def main():
    """主函数"""
    
    print("=" * 60)
    print("LangChain Milvus 多向量检索 - 混合检索配置")
    print("=" * 60)
    
    # 创建向量存储
    vectorstore = create_hybrid_vectorstore()
    
    # 混合检索测试
    hybrid_search_with_weights(vectorstore)
    
    # 过滤检索测试
    filter_search(vectorstore)


if __name__ == "__main__":
    main()
```

---

### 场景3：RAG 完整流程

```python
"""
LangChain Milvus 多向量检索 - RAG 完整流程
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.vectorstores.utils import BM25BuiltInFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MILVUS_URI", "http://localhost:19530")


def create_knowledge_base():
    """创建知识库"""
    
    # 准备长文档
    long_text = """
    Milvus is an open-source vector database built to power embedding similarity search 
    and AI applications. It makes unstructured data search more accessible, and provides 
    a consistent user experience regardless of the deployment environment.
    
    Milvus supports multiple index types including FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, 
    and ANNOY. Each index type has its own characteristics in terms of search performance, 
    memory usage, and accuracy.
    
    Hybrid search in Milvus combines dense vectors (semantic understanding) with sparse 
    vectors (keyword matching) to improve search quality. This is particularly useful 
    for RAG applications where both semantic relevance and exact keyword matches are important.
    """
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = text_splitter.split_text(long_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    print(f"✓ 文档分块完成: {len(documents)} 个块")
    
    # 创建向量存储
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
        vector_field=["dense", "sparse"],
        connection_args={"uri": URI},
        collection_name="milvus_knowledge_base",
        drop_old=True,
    )
    
    print(f"✓ 向量存储创建完成")
    
    return vectorstore


def create_rag_chain(vectorstore):
    """创建 RAG 链"""
    
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 创建 LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # 创建 QA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


def ask_questions(qa_chain):
    """提问测试"""
    
    questions = [
        "What is Milvus?",
        "What index types does Milvus support?",
        "How does hybrid search work in Milvus?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"问题 {i}: {question}")
        print(f"{'=' * 60}")
        
        result = qa_chain({"query": question})
        
        print(f"\n回答:")
        print(result["result"])
        
        print(f"\n来源文档 ({len(result['source_documents'])} 个):")
        for j, doc in enumerate(result["source_documents"], 1):
            print(f"{j}. {doc.page_content[:100]}...")


def main():
    """主函数"""
    
    print("=" * 60)
    print("LangChain Milvus 多向量检索 - RAG 完整流程")
    print("=" * 60)
    
    # 创建知识库
    vectorstore = create_knowledge_base()
    
    # 创建 RAG 链
    qa_chain = create_rag_chain(vectorstore)
    
    # 提问测试
    ask_questions(qa_chain)


if __name__ == "__main__":
    main()
```

---

## 高级配置

### 1. 自定义 Ranker 参数

```python
from langchain_community.vectorstores import Milvus

# 创建向量存储时配置 Ranker
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embedding,
    builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
    vector_field=["dense", "sparse"],
    connection_args={"uri": URI},
    collection_name="custom_ranker",
    # 自定义混合检索参数
    search_params={
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }
)
```

### 2. 多模型 Embedding 融合

```python
from langchain_openai import OpenAIEmbeddings

# 定义多个不同的 Embedding 模型
embedding_models = [
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    OpenAIEmbeddings(model="text-embedding-3-small"),
    OpenAIEmbeddings(model="text-embedding-3-large"),
]

# 创建向量存储
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embedding_models,  # 多个模型
    vector_field=["dense1", "dense2", "dense3"],  # 对应的字段名
    connection_args={"uri": URI},
    collection_name="multi_model_embedding",
)
```

### 3. 批量数据导入

```python
def batch_import_documents(file_paths, vectorstore):
    """批量导入文档"""
    
    from langchain.document_loaders import TextLoader
    
    all_documents = []
    
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
    
    # 批量添加
    vectorstore.add_documents(all_documents)
    
    print(f"✓ 批量导入完成: {len(all_documents)} 个文档")
```

---

## 性能优化

### 1. 连接池配置

```python
connection_args = {
    "uri": URI,
    "pool_size": 10,  # 连接池大小
    "timeout": 30,    # 超时时间
}
```

### 2. 批量检索

```python
def batch_search(vectorstore, queries):
    """批量检索"""
    
    results = []
    
    for query in queries:
        result = vectorstore.similarity_search(query, k=3)
        results.append(result)
    
    return results
```

### 3. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query, k=3):
    """带缓存的检索"""
    return vectorstore.similarity_search(query, k=k)
```

---

## 常见问题

### Q1: 如何指定向量字段名？

```python
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=[embedding1, embedding2],
    vector_field=["custom_dense", "custom_sparse"],  # 自定义字段名
    connection_args={"uri": URI},
)
```

### Q2: 如何删除 Collection？

```python
from pymilvus import connections, utility

connections.connect(uri=URI)
utility.drop_collection("collection_name")
```

### Q3: 如何查看 Collection 信息？

```python
from pymilvus import connections, Collection

connections.connect(uri=URI)
collection = Collection("collection_name")

print(f"实体数量: {collection.num_entities}")
print(f"Schema: {collection.schema}")
```

---

## 总结

LangChain 与 Milvus 的集成提供了：

1. **多向量支持**：支持多个 Embedding 模型
2. **BM25 集成**：内置 BM25 函数
3. **混合检索**：自动融合多个向量字段
4. **RAG 友好**：与 LangChain 生态无缝集成

**最佳实践**：
- 使用多个 Embedding 模型提升检索质量
- 结合 BM25 实现混合检索
- 合理配置连接池和超时参数
- 使用缓存优化重复查询

---

**参考资料**：
- [LangChain Milvus 文档](https://python.langchain.com/docs/integrations/vectorstores/milvus)
- [Milvus 官方文档](https://milvus.io/docs)
- Context7 Milvus 多向量检索文档
