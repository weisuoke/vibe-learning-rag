# 实战代码02：ChromaDB完整示例

## 代码说明

本示例展示ChromaDB在RAG系统中的完整应用，包括持久化存储、元数据过滤和端到端RAG流程。

**环境要求**：
```bash
pip install chromadb sentence-transformers openai python-dotenv
```

---

## 完整代码

```python
"""
ChromaDB完整RAG示例
演示持久化存储、元数据过滤和完整RAG流程
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# ============================================
# 1. 初始化ChromaDB（持久化存储）
# ============================================

def init_chromadb(persist_directory="./chroma_rag_db"):
    """初始化ChromaDB客户端（持久化模式）"""
    print("=" * 50)
    print("1. 初始化ChromaDB（持久化存储）")
    print("=" * 50)

    # 创建持久化客户端
    client = chromadb.PersistentClient(path=persist_directory)

    print(f"✓ ChromaDB已初始化")
    print(f"  持久化路径: {persist_directory}")
    print(f"  现有collections: {[c.name for c in client.list_collections()]}")

    return client


# ============================================
# 2. 创建Collection（带HNSW配置）
# ============================================

def create_collection(client, collection_name="rag_documents"):
    """创建或获取collection"""
    print(f"\n创建/获取collection: {collection_name}")

    # 获取或创建collection，配置HNSW索引
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",  # 使用余弦相似度
            "hnsw:construction_ef": 200,  # 构建参数
            "hnsw:M": 16  # 连接数
        }
    )

    print(f"✓ Collection已就绪")
    print(f"  文档数量: {collection.count()}")

    return collection


# ============================================
# 3. 添加文档（带元数据）
# ============================================

def add_documents_with_metadata(collection, model):
    """添加文档到collection，包含丰富的元数据"""
    print("\n" + "=" * 50)
    print("2. 添加文档（带元数据）")
    print("=" * 50)

    # 准备文档和元数据
    documents = [
        {
            "text": "RAG是检索增强生成技术，结合检索和生成两个步骤，提升LLM的知识准确性",
            "metadata": {"category": "RAG", "difficulty": "beginner", "source": "manual"}
        },
        {
            "text": "ChromaDB是轻量级向量数据库，适合LLM应用开发，支持持久化存储",
            "metadata": {"category": "VectorDB", "difficulty": "beginner", "source": "manual"}
        },
        {
            "text": "HNSW是高效的索引算法，在大规模向量检索中表现优异，召回率高",
            "metadata": {"category": "Algorithm", "difficulty": "intermediate", "source": "manual"}
        },
        {
            "text": "Prompt Engineering是LLM应用的关键技能，包括角色设定、指令清晰等技巧",
            "metadata": {"category": "PromptEng", "difficulty": "beginner", "source": "manual"}
        },
        {
            "text": "向量数据库通过语义检索快速找到相关文档，是RAG系统的核心组件",
            "metadata": {"category": "VectorDB", "difficulty": "intermediate", "source": "manual"}
        }
    ]

    # 生成embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts).tolist()

    # 添加到collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"✓ 已添加{len(documents)}个文档")
    print(f"  当前总数: {collection.count()}")


# ============================================
# 4. 基础检索
# ============================================

def basic_retrieval(collection, model, query, top_k=3):
    """基础向量检索"""
    print("\n" + "=" * 50)
    print("3. 基础检索")
    print("=" * 50)

    print(f"\n查询: {query}")

    # 生成查询embedding
    query_embedding = model.encode(query).tolist()

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(f"\nTop {top_k}结果:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance
        print(f"\n{i+1}. [{similarity:.3f}] {doc}")
        print(f"   元数据: {metadata}")

    return results


# ============================================
# 5. 元数据过滤检索
# ============================================

def filtered_retrieval(collection, model, query, filter_dict, top_k=3):
    """带元数据过滤的检索"""
    print("\n" + "=" * 50)
    print("4. 元数据过滤检索")
    print("=" * 50)

    print(f"\n查询: {query}")
    print(f"过滤条件: {filter_dict}")

    # 生成查询embedding
    query_embedding = model.encode(query).tolist()

    # 带过滤的检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=filter_dict  # 元数据过滤
    )

    print(f"\nTop {top_k}结果（已过滤）:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance
        print(f"\n{i+1}. [{similarity:.3f}] {doc}")
        print(f"   元数据: {metadata}")

    return results


# ============================================
# 6. 完整RAG流程
# ============================================

def rag_query(collection, model, llm_client, question, top_k=3):
    """完整RAG查询流程：检索 + 生成"""
    print("\n" + "=" * 50)
    print("5. 完整RAG流程")
    print("=" * 50)

    print(f"\n问题: {question}")

    # 步骤1：检索相关文档
    print("\n[步骤1] 检索相关文档...")
    query_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_docs = results['documents'][0]
    print(f"✓ 检索到{len(retrieved_docs)}个相关文档")

    # 步骤2：构建上下文
    context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    print(f"\n[步骤2] 构建上下文（{len(context)}字符）")

    # 步骤3：生成答案
    print("\n[步骤3] 调用LLM生成答案...")

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的AI助手。请基于提供的文档回答用户问题，如果文档中没有相关信息，请明确说明。"
                },
                {
                    "role": "user",
                    "content": f"参考文档：\n{context}\n\n问题：{question}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        print(f"\n✓ 答案生成完成")

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "answer": answer
        }

    except Exception as e:
        print(f"\n✗ LLM调用失败: {e}")
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "answer": f"[错误] 无法生成答案: {e}"
        }


# ============================================
# 7. 批量RAG查询
# ============================================

def batch_rag_queries(collection, model, llm_client, questions):
    """批量处理多个RAG查询"""
    print("\n" + "=" * 50)
    print("6. 批量RAG查询")
    print("=" * 50)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n--- 查询 {i}/{len(questions)} ---")
        result = rag_query(collection, model, llm_client, question, top_k=2)
        results.append(result)

    return results


# ============================================
# 8. 更新和删除文档
# ============================================

def update_and_delete_demo(collection, model):
    """演示文档更新和删除"""
    print("\n" + "=" * 50)
    print("7. 更新和删除文档")
    print("=" * 50)

    # 添加新文档
    new_doc = "LangChain是LLM应用开发框架，简化了RAG系统的构建流程"
    new_embedding = model.encode(new_doc).tolist()

    print(f"\n添加新文档: {new_doc}")
    collection.add(
        documents=[new_doc],
        embeddings=[new_embedding],
        metadatas=[{"category": "Framework", "difficulty": "intermediate", "source": "manual"}],
        ids=["doc_new"]
    )
    print(f"✓ 文档已添加，当前总数: {collection.count()}")

    # 更新文档
    updated_doc = "LangChain是强大的LLM应用开发框架，支持LCEL和Agent"
    updated_embedding = model.encode(updated_doc).tolist()

    print(f"\n更新文档: {updated_doc}")
    collection.update(
        ids=["doc_new"],
        documents=[updated_doc],
        embeddings=[updated_embedding],
        metadatas=[{"category": "Framework", "difficulty": "advanced", "source": "manual"}]
    )
    print(f"✓ 文档已更新")

    # 删除文档
    print(f"\n删除文档: doc_new")
    collection.delete(ids=["doc_new"])
    print(f"✓ 文档已删除，当前总数: {collection.count()}")


# ============================================
# 主函数
# ============================================

def main():
    """主函数"""
    print("ChromaDB完整RAG示例")
    print("=" * 50)

    # 初始化
    client = init_chromadb()
    collection = create_collection(client)

    # 初始化embedding模型
    print("\n加载embedding模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ 模型加载完成")

    # 初始化LLM客户端
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 添加文档
    if collection.count() == 0:
        add_documents_with_metadata(collection, model)
    else:
        print(f"\n跳过添加文档（已有{collection.count()}个文档）")

    # 基础检索
    basic_retrieval(collection, model, "什么是RAG技术？", top_k=3)

    # 元数据过滤检索
    filtered_retrieval(
        collection,
        model,
        "向量数据库",
        filter_dict={"category": "VectorDB"},
        top_k=2
    )

    # 完整RAG流程
    rag_result = rag_query(
        collection,
        model,
        llm_client,
        "ChromaDB适合什么场景？",
        top_k=3
    )

    print("\n" + "=" * 50)
    print("RAG答案:")
    print("=" * 50)
    print(rag_result["answer"])

    # 批量查询
    questions = [
        "如何优化向量检索性能？",
        "Prompt Engineering有哪些技巧？"
    ]
    batch_results = batch_rag_queries(collection, model, llm_client, questions)

    # 更新和删除演示
    update_and_delete_demo(collection, model)

    print("\n" + "=" * 50)
    print("所有示例执行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

---

## 预期输出

```
ChromaDB完整RAG示例
==================================================

==================================================
1. 初始化ChromaDB（持久化存储）
==================================================
✓ ChromaDB已初始化
  持久化路径: ./chroma_rag_db
  现有collections: []

创建/获取collection: rag_documents
✓ Collection已就绪
  文档数量: 0

加载embedding模型...
✓ 模型加载完成

==================================================
2. 添加文档（带元数据）
==================================================
✓ 已添加5个文档
  当前总数: 5

==================================================
3. 基础检索
==================================================

查询: 什么是RAG技术？

Top 3结果:

1. [0.823] RAG是检索增强生成技术，结合检索和生成两个步骤，提升LLM的知识准确性
   元数据: {'category': 'RAG', 'difficulty': 'beginner', 'source': 'manual'}

2. [0.654] 向量数据库通过语义检索快速找到相关文档，是RAG系统的核心组件
   元数据: {'category': 'VectorDB', 'difficulty': 'intermediate', 'source': 'manual'}

3. [0.512] ChromaDB是轻量级向量数据库，适合LLM应用开发，支持持久化存储
   元数据: {'category': 'VectorDB', 'difficulty': 'beginner', 'source': 'manual'}

==================================================
4. 元数据过滤检索
==================================================

查询: 向量数据库
过滤条件: {'category': 'VectorDB'}

Top 2结果（已过滤）:

1. [0.892] 向量数据库通过语义检索快速找到相关文档，是RAG系统的核心组件
   元数据: {'category': 'VectorDB', 'difficulty': 'intermediate', 'source': 'manual'}

2. [0.845] ChromaDB是轻量级向量数据库，适合LLM应用开发，支持持久化存储
   元数据: {'category': 'VectorDB', 'difficulty': 'beginner', 'source': 'manual'}

==================================================
5. 完整RAG流程
==================================================

问题: ChromaDB适合什么场景？

[步骤1] 检索相关文档...
✓ 检索到3个相关文档

[步骤2] 构建上下文（245字符）

[步骤3] 调用LLM生成答案...

✓ 答案生成完成

==================================================
RAG答案:
==================================================
根据文档，ChromaDB适合以下场景：

1. LLM应用开发：ChromaDB是轻量级向量数据库，特别适合LLM应用的快速开发和原型验证。

2. 需要持久化存储的场景：支持数据持久化，确保向量数据不会丢失。

3. RAG系统构建：作为RAG系统的核心组件，通过语义检索快速找到相关文档。

4. 中小规模应用：适合初创公司和中小规模的生产环境，不需要复杂的分布式架构。
```

---

## 关键要点

### 1. 持久化存储

**PersistentClient vs Client**：
```python
# 持久化（推荐生产环境）
client = chromadb.PersistentClient(path="./chroma_db")

# 内存模式（仅用于测试）
client = chromadb.Client()
```

### 2. HNSW配置

```python
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",  # 距离度量
        "hnsw:construction_ef": 200,  # 构建参数
        "hnsw:M": 16  # 连接数
    }
)
```

### 3. 元数据过滤

**支持的过滤操作**：
```python
# 等于
where={"category": "RAG"}

# 不等于
where={"category": {"$ne": "RAG"}}

# 包含
where={"category": {"$in": ["RAG", "VectorDB"]}}

# 组合条件
where={
    "$and": [
        {"category": "RAG"},
        {"difficulty": "beginner"}
    ]
}
```

### 4. RAG最佳实践

**检索参数调优**：
- **top_k**: 通常3-5个文档
- **相似度阈值**: 过滤低相关文档
- **元数据过滤**: 缩小检索范围

**上下文构建**：
- 保持上下文简洁（<2000 tokens）
- 按相关性排序
- 添加文档来源信息

---

## 引用来源

1. **ChromaDB官方文档**：
   - https://docs.trychroma.com/getting-started
   - https://docs.trychroma.com/usage-guide

2. **持久化存储**：
   - https://realpython.com/chromadb-vector-database

3. **RAG集成**：
   - https://oneuptime.com/blog/post/2026-01-30-chromadb-integration/view
   - https://dev.to/aquibpy/building-ragenius-a-production-ready-rag-system-with-fastapi-azure-openai-chromadb-3281

4. **生产实践**：
   - https://promptlyai.in/rag-made-simple
   - https://ai.plainenglish.io/chromadb-end-to-end-tutorial-c18202fa66a2

---

**最后更新**：2026-02-15
**基于资料**：2025-2026最新ChromaDB RAG实践
