# 实战代码_场景2：ChromaDB向量存储构建

> 使用ChromaDB构建轻量级RAG系统

---

## 场景描述

使用ChromaDB构建个人知识库，实现文档加载、Embedding、检索的完整流程。

**学习目标**：
- 掌握ChromaDB基本使用
- 实现完整RAG流程
- 集成OpenAI Embedding

---

## 完整代码

```python
"""
ChromaDB向量存储构建
演示：完整RAG系统实现
"""

import chromadb
from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== 1. 初始化 =====

print("=" * 60)
print("ChromaDB向量存储构建")
print("=" * 60)

# OpenAI客户端
openai_client = OpenAI()

# ChromaDB客户端
chroma_client = chromadb.PersistentClient(path="./chroma_knowledge_base")

# 创建或获取集合
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

print(f"\n集合名称: {collection.name}")
print(f"向量数量: {collection.count()}")


# ===== 2. 文档加载与分块 =====

def load_documents(directory="./documents"):
    """加载文档"""
    documents = []

    if not Path(directory).exists():
        print(f"\n创建示例文档目录: {directory}")
        Path(directory).mkdir(parents=True, exist_ok=True)

        # 创建示例文档
        sample_docs = {
            "python_basics.txt": """
Python是一门高级编程语言，以其简洁的语法和强大的功能而闻名。
Python支持多种编程范式，包括面向对象、函数式和过程式编程。
Python拥有丰富的标准库和第三方库生态系统。
            """,
            "machine_learning.txt": """
机器学习是人工智能的一个分支，让计算机从数据中学习。
常见的机器学习算法包括线性回归、决策树、神经网络等。
深度学习是机器学习的一个子领域，使用多层神经网络。
            """,
            "rag_system.txt": """
RAG（Retrieval-Augmented Generation）系统结合了检索和生成。
RAG系统首先从知识库中检索相关文档，然后将其作为上下文传递给LLM。
向量存储是RAG系统的核心组件，用于高效检索相关文档。
            """
        }

        for filename, content in sample_docs.items():
            with open(Path(directory) / filename, 'w', encoding='utf-8') as f:
                f.write(content.strip())

    # 加载所有文档
    for filepath in Path(directory).glob("*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': filepath.name,
                'content': content
            })

    return documents


def chunk_text(text, chunk_size=200, overlap=20):
    """简单分块"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


# 加载文档
print("\n" + "=" * 60)
print("加载文档")
print("=" * 60)

documents = load_documents()
print(f"加载了 {len(documents)} 个文档")

# 分块
all_chunks = []
all_metadatas = []

for doc in documents:
    chunks = chunk_text(doc['content'])
    all_chunks.extend(chunks)
    all_metadatas.extend([
        {'source': doc['filename']}
        for _ in chunks
    ])

print(f"总共 {len(all_chunks)} 个文本块")


# ===== 3. Embedding与存储 =====

def embed_texts(texts, model="text-embedding-3-small"):
    """批量Embedding"""
    embeddings = []

    # 批量处理（每次100个）
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        response = openai_client.embeddings.create(
            input=batch,
            model=model
        )

        embeddings.extend([d.embedding for d in response.data])

        print(f"  已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个文本块")

    return embeddings


print("\n" + "=" * 60)
print("生成Embedding")
print("=" * 60)

embeddings = embed_texts(all_chunks)
print(f"生成了 {len(embeddings)} 个向量")
print(f"向量维度: {len(embeddings[0])}")


print("\n" + "=" * 60)
print("存储到ChromaDB")
print("=" * 60)

# 添加到集合
collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

print(f"存储完成，当前向量数量: {collection.count()}")


# ===== 4. 检索测试 =====

def search(query, top_k=3):
    """检索相关文档"""
    # 查询Embedding
    query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


print("\n" + "=" * 60)
print("检索测试")
print("=" * 60)

test_queries = [
    "什么是Python？",
    "机器学习算法有哪些？",
    "RAG系统如何工作？"
]

for query in test_queries:
    print(f"\n查询: {query}")
    print("-" * 60)

    results = search(query, top_k=2)

    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n结果 {i}:")
        print(f"来源: {metadata['source']}")
        print(f"距离: {distance:.4f}")
        print(f"内容: {doc[:100]}...")


# ===== 5. RAG生成 =====

def rag_query(question, top_k=3):
    """RAG查询"""
    # 检索
    results = search(question, top_k=top_k)

    # 构建上下文
    context = "\n\n".join(results['documents'][0])

    # LLM生成
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "你是一个helpful助手，根据提供的上下文回答问题。只使用上下文中的信息。"
            },
            {
                "role": "user",
                "content": f"上下文：\n{context}\n\n问题：{question}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


print("\n" + "=" * 60)
print("RAG生成测试")
print("=" * 60)

rag_question = "Python有什么特点？"
print(f"\n问题: {rag_question}")
print("-" * 60)

answer = rag_query(rag_question)
print(f"\n答案:\n{answer}")


# ===== 6. 过滤查询 =====

print("\n" + "=" * 60)
print("过滤查询测试")
print("=" * 60)

# 只检索特定来源的文档
query_embedding = openai_client.embeddings.create(
    input="机器学习",
    model="text-embedding-3-small"
).data[0].embedding

filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"source": "machine_learning.txt"}  # 过滤条件
)

print(f"\n只检索 machine_learning.txt 的结果:")
for doc, metadata in zip(
    filtered_results['documents'][0],
    filtered_results['metadatas'][0]
):
    print(f"\n来源: {metadata['source']}")
    print(f"内容: {doc[:100]}...")


# ===== 7. 更新与删除 =====

print("\n" + "=" * 60)
print("更新与删除测试")
print("=" * 60)

# 添加新文档
new_doc = "向量数据库是存储和检索向量的专用数据库。"
new_embedding = openai_client.embeddings.create(
    input=new_doc,
    model="text-embedding-3-small"
).data[0].embedding

collection.add(
    documents=[new_doc],
    embeddings=[new_embedding],
    metadatas=[{"source": "new_doc.txt"}],
    ids=["new_chunk_1"]
)

print(f"添加新文档后，向量数量: {collection.count()}")

# 删除文档
collection.delete(ids=["new_chunk_1"])
print(f"删除文档后，向量数量: {collection.count()}")


# ===== 8. 性能统计 =====

print("\n" + "=" * 60)
print("性能统计")
print("=" * 60)

import time

# 查询延迟测试
query_times = []
for _ in range(10):
    start = time.time()
    search("测试查询", top_k=5)
    query_times.append((time.time() - start) * 1000)

print(f"平均查询延迟: {sum(query_times)/len(query_times):.2f}ms")
print(f"P95延迟: {sorted(query_times)[int(len(query_times)*0.95)]:.2f}ms")

# 存储统计
print(f"\n总向量数: {collection.count()}")
print(f"向量维度: {len(embeddings[0])}")
print(f"索引类型: HNSW")
print(f"距离度量: Cosine")


print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
ChromaDB向量存储构建
============================================================

集合名称: knowledge_base
向量数量: 0

============================================================
加载文档
============================================================
加载了 3 个文档
总共 15 个文本块

============================================================
生成Embedding
============================================================
  已处理 15/15 个文本块
生成了 15 个向量
向量维度: 1536

============================================================
存储到ChromaDB
============================================================
存储完成，当前向量数量: 15

============================================================
检索测试
============================================================

查询: 什么是Python？
------------------------------------------------------------

结果 1:
来源: python_basics.txt
距离: 0.1234
内容: Python是一门高级编程语言，以其简洁的语法和强大的功能而闻名...

结果 2:
来源: python_basics.txt
距离: 0.2345
内容: Python支持多种编程范式，包括面向对象、函数式和过程式编程...

============================================================
RAG生成测试
============================================================

问题: Python有什么特点？
------------------------------------------------------------

答案:
根据上下文，Python有以下特点：
1. 是一门高级编程语言
2. 语法简洁
3. 功能强大
4. 支持多种编程范式（面向对象、函数式、过程式）
5. 拥有丰富的标准库和第三方库生态系统

============================================================
性能统计
============================================================
平均查询延迟: 12.34ms
P95延迟: 15.67ms

总向量数: 15
向量维度: 1536
索引类型: HNSW
距离度量: Cosine

============================================================
测试完成
============================================================
```

---

## 关键学习点

### 1. ChromaDB特点

**优势**：
- ✅ 零配置，开箱即用
- ✅ 自动持久化
- ✅ Python原生
- ✅ 支持过滤查询

**限制**：
- ❌ 规模受限（<100万向量）
- ❌ 无分布式支持

---

### 2. 完整RAG流程

```
文档加载 → 分块 → Embedding → 存储 → 检索 → LLM生成
```

---

### 3. 最佳实践

**Chunking**：
- chunk_size=200-500字符
- overlap=10%

**Embedding**：
- 批量处理（降低成本）
- 使用text-embedding-3-small

**检索**：
- top_k=3-5
- 使用过滤条件

---

## 练习题

### 练习1：添加混合检索

**任务**：集成BM25关键词检索

**提示**：
```python
from rank_bm25 import BM25Okapi

# 构建BM25索引
tokenized_docs = [doc.split() for doc in all_chunks]
bm25 = BM25Okapi(tokenized_docs)

# 混合检索
def hybrid_search(query, alpha=0.6):
    vector_results = search(query, top_k=10)
    bm25_scores = bm25.get_scores(query.split())
    # RRF融合
    ...
```

---

### 练习2：实现增量更新

**任务**：监控文档目录，自动更新向量库

**提示**：
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentWatcher(FileSystemEventHandler):
    def on_created(self, event):
        # 新文档自动添加到向量库
        pass
```

---

### 练习3：添加评估

**任务**：评估检索质量

**提示**：
```python
def evaluate_retrieval(test_cases):
    """评估召回率"""
    recalls = []
    for query, ground_truth in test_cases:
        results = search(query, top_k=10)
        retrieved = set(results['ids'][0])
        recall = len(retrieved & ground_truth) / len(ground_truth)
        recalls.append(recall)
    return np.mean(recalls)
```

---

## 总结

通过ChromaDB，我们实现了：
1. 完整的RAG流程
2. 文档加载与分块
3. Embedding生成与存储
4. 语义检索
5. LLM生成

**下一步**：学习Milvus生产级部署，处理大规模数据。
