from sentence_transformers import CrossEncoder
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class ContextOptimizer:
    """上下文优化器"""

    def __init__(self, max_tokens: int = 6000):
        self.max_tokens = max_tokens
        # 使用 CrossEncoder 进行 ReRank
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量（粗略估计：1 token ≈ 4 字符）"""
        return len(text) // 4
    
    def optimize(self, query: str, documents: list) -> str:
        """优化上下文注入"""

        # 步骤1： ReRank 重排序
        print("=== 步骤 1： ReRank 重排序 ===")
        pairs = [(query, doc['content']) for doc in documents]
        scores = self.reranker.predict(pairs)

        # 按相关性排序
        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"重排序后的相关度分数:")
        for i, (doc, score) in enumerate(ranked_docs):
            print(f"  文档 {i+1}: {score:.3f}")

        # 步骤 2：动态选择文档（基于 Token 预算）
        print(f"\n===步骤 2： 动态选择（预算：{self.max_tokens}tokens）===")
        selected_docs=[]
        current_tokens = 0

        for doc, score in ranked_docs:
            doc_tokens = self.estimate_tokens(doc['content'])

            if current_tokens + doc_tokens <= self.max_tokens:
                selected_docs.append((doc, score))
                current_tokens += doc_tokens
                print(f"  ✓ 选择文档（{doc_tokens} tokens，累计：{current_tokens}）")
            else:
                print(f"  ✗ 跳过文档（{doc_tokens} tokens，超出预算）")
                break

        # 步骤 3：构建结构化上下文
        print(f"\n=== 步骤 3：构建结构化上下文 ===")
        context_parts = []

        # 添加元信息
        context_parts.append(f"# 检索结果")
        context_parts.append(f"检索到 {len(documents)} 篇文档，选择了最相关的 {len(selected_docs)} 篇")
        context_parts.append(f"总 tokens：{current_tokens}/{self.max_tokens}\n")

        # 添加文档内容
        for i, (doc, score) in enumerate(selected_docs, 1):
            context_parts.append(f"## 文档 {i} (相关度: {score:.2f})")
            context_parts.append(f"来源：{doc.get('source', '未知')}")
            context_parts.append(f"内容：{doc['content']}\n")

        return "\n".join(context_parts)

# 测试
optimizer = ContextOptimizer(max_tokens=6000)

# 模拟检索到的文档
documents = [
    {
        "content": "RAG（检索增强生成）是一种结合检索和生成的技术，通过在生成前检索相关文档来提升答案准确性。",
        "source": "doc1.pdf"
    },
    {
        "content": "向量数据库用于存储和检索 embeddings，支持语义相似度搜索。",
        "source": "doc2.pdf"
    },
    {
        "content": "LangChain 是一个流行的 RAG 框架，提供了完整的工具链。",
        "source": "doc3.pdf"
    },
    {
        "content": "Python 是一种高级编程语言，广泛用于数据科学和机器学习。",
        "source": "doc4.pdf"
    },
    {
        "content": "Transformer 是现代 LLM 的基础架构，使用自注意力机制。",
        "source": "doc5.pdf"
    }
]

# 优化上下文
optimized_context = optimizer.optimize("什么是 RAG?", documents)

print("\n=== 优化后的上下文 ===")
print(optimized_context)

# 使用优化后的上下文查询
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是专业的 RAG 助手"},
        {"role": "user", "content": f"{optimized_context}\n\n问题：什么是 RAG?"}
    ],
    temperature=0.1
)

print("\n=== RAG 回答 ===")
print(response.choices[0].message.content)