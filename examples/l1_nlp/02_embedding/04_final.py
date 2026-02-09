"""
Embedding 原理与选型 实战示例
演示：从文本向量化到语义检索的完整流程
"""

import os
from typing import List
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import chromadb

load_dotenv()

# ===== 1. 基础：获取文本的 Embedding =====
print("=== 1. 获取文本 Embedding ===")
client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """获取文本的 Embedding 向量"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """批量获取 Embedding, 更高效"""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

# 测试单个文本
text = "RAG 是检索增强生成技术，结合了检索和生成的优势"
embedding = get_embedding(text)
print(f"文本: {text}")
print(f"向量维度: {len(embedding)}")
print(f"向量前5个值: {embedding[:5]}")

# ===== 2. 计算语义相似度 =====
print("\n=== 2. 计算语义相似度 ===")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 准备测试文本
texts = [
    "RAG 是检索增强生成技术",
    "检索增强生成是一种 AI 技术",
    "今天天气真好",
    "大语言模型可以生成文本",
]

# 获取所有文本的 Embedding
embeddings = get_embeddings_batch(texts)

# 计算第一个文本与其他文本的相似度
base_text = texts[0]
print(f"基准文本: {base_text}\n")

for i, text in enumerate(texts[1:], 1):
    similarity = cosine_similarity(embeddings[0], embeddings[i])
    print(f"与 '{text}' 的相似度: {similarity:.4f}")

# ===== 3. RAG 开发应用：简单的语义检索 =====
print("\n=== 3. RAG 语义检索示例 ===")

# 模拟知识库文档
knowledge_base = [
    "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术",
    "向量数据库用于存储和检索高维向量",
    "Embedding 将文本转换为稠密向量表示",
    "大语言模型（LLM）可以理解和生成自然语言",
    "Prompt Engineering 是优化 LLM 输入的技术",
    "Chunking 是将长文档分割成小块的过程",
]

# 为知识库文档生成 Embedding
kb_embeddings = get_embeddings_batch(knowledge_base)

def semantic_search(query: str, top_k: int = 3) -> List[tuple]:
    """语义检索：找到与查询最相关的文档"""
    query_embedding = get_embedding(query)

    # 计算与所有文档的相似度
    similarities = []
    for i, doc_embedding in enumerate(kb_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, sim, knowledge_base[i]))

    # 按相似度排序，返回 top_k 个结果
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# 测试语义检索
query = "什么是向量化技术？"
print(f"\n查询: {query}")
print("检索结果:")

results = semantic_search(query, top_k=3)
print(f"检索结果: {results}")
for rank, (idx, score, doc) in enumerate(results, 1):
     print(f"  {rank}. [相似度: {score:.4f}] {doc}")

# ===== 4. 使用 ChromaDB 向量数据库 =====
print("\n=== 4. 使用 ChromaDB 向量数据库 ===")

# 创建客户端和集合
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="rag_knowledge_base",
    metadata={"description": "RAG 学习知识库"}
)

# 添加文档到向量数据库
collection.add(
    documents=knowledge_base,
    ids=[f"doc_{i}" for i in range(len(knowledge_base))]
)

# 使用向量数据库检索
query = "如何优化大模型的输入?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"查询: {query}")
print(f"结果: {results}")
print("ChromaDB 检索结果:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
    print(f"  {i}. [距离: {distance:.4f}] {doc}")

print("\n=== 实战演示完成 ===")