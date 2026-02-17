"""
RAG 生成模块示例
演示：如何将检索结果注入 Prompt，生成最终答案
这是 RAG 系统的核心代码！
"""

from openai import OpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_rag_answer(
    question: str,
    retrieved_docs: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.1
) -> str:
    """
    RAG 生成模块：基于检索结果生成答案

    Args:
        question: 用户问题
        retrieved_docs: 检索到的文档列表
        model: 使用的模型
        temperature: 温度参数

    Returns:
        生成的答案
    """

    # 构建上下文
    context = "\n\n---\n\n".join(retrieved_docs)

    # 构建 Prompt
    messages = [
        {
            "role": "system",
            "content": """
                你是一个专业的问答助手。请基于提供的参考资料回答用户问题。

                规则：
                1. 只使用参考资料中的信息回答
                2. 如果资料中没有相关信息，请明确说明
                3. 回答要简洁准确，不要编造信息
                4. 可以适当组织和总结信息
            """
        },
        {
            "role": "user",
            "content": f"""
                参考资料：{context}

                ---

                问题：{question}

                请基于以上参考资料回答问题
            """
        }
    ]

    # 调用 API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )

    return response.choices[0].message.content

# ===== 测试 RAG 生成 =====
print("=== RAG 生成模块测试 ===\n")

# 模拟检索结果
mock_retrieved_docs = [
    "向量数据库是专门用于存储和检索向量的数据库系统。常见的向量数据库包括 Milvus、Pinecone、Chroma 等。",
    "Embedding 是将文本转换为稠密向量的技术。OpenAI 提供 text-embedding-3-small 和 text-embedding-3-large 两种模型。",
    "RAG（检索增强生成）的核心流程：1. 用户提问 2. 向量检索 3. 构建 Prompt 4. LLM 生成答案"
]

# 用户问题
user_question = "RAG 系统中常用哪些向量数据库？"

# 生成答案
answer = generate_rag_answer(user_question, mock_retrieved_docs)

print(f"问题：{user_question}\n")
print(f"答案：{answer}")