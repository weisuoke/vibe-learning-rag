from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embeddings_batch(texts: list) -> list:
    """批量获取 Embedding, 更高效"""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# 批量处理多个文档
docs = ["文档1", "文档2", "文档3"]
embeddings = get_embeddings_batch(docs)

print(f"批量获取了 {len(embeddings)} 个 Embedding 向量")
for i, emb in enumerate(embeddings):
    print(f"文档 {i+1} 的向量维度: {len(emb)} 和 前5个维度的值示例: {emb[:5]}...")