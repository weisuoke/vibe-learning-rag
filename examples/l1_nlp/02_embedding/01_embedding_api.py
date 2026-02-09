from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list:
    """获取文本的 Embedding 向量"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

embedding = get_embedding("你好，世界")
print(f"向量维度: {len(embedding)} 和 向量 embedding: {embedding[:5]}...")  # 打印前5个维度的值示例    

#比较两段文本的相似度
emb1 = get_embedding("苹果是一种水果")
emb2 = get_embedding("香蕉也是水果")
emb3 = get_embedding("Python 是编程语言")

print(f"水果句子相似度: {cosine_similarity(emb1, emb2):.4f}")  # 高
print(f"不同主题相似度: {cosine_similarity(emb1, emb3):.4f}")  # 低