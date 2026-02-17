RELEVANCE_JUDGE_PROMPT = """
任务：判断文档是否与用户问题相关

步骤：
1. 理解用户问题的核心需求
2. 阅读文档内容
3. 判断文档是否包含相关信息
4. 给出相关性评分（0.0-1.0）

判断标准
- 1.0：文档直接回答问题
- 0.7-0.9：文档包含相关信息
- 0.4-0.6：文档部分相关
- 0.0-0.3：文档不相关

当前的输入：
- 问题：{question}
- 文档：{document}

输出格式：
{{
    "is_relevant": true/false,
    "score": 0.0 - 1.0,
    "reason": "判断理由（20字以内）"
}}

示例：
问题："什么是 RAG?"
文档："RAG 是检索增强生成技术，结合检索和生成..."
输出:
{{
    "is_relevant": true,
    "score": 1.0,
    "reason": "文档直接定义了 RAG"
}}
"""

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def judge_relevance(question: str, document: str) -> dict:
    """判断文档相关性"""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是文档相关性判断专家"},
            {"role": "user", "content": RELEVANCE_JUDGE_PROMPT.format(
                question=question,
                document=document
            )}
        ],
        temperature=0.0  # 零温度确保判断一致
    )

    return json.loads(response.choices[0].message.content)

# 测试
doc1 = "RAG 是检索增强生成技术，结合检索和生成..."
doc2 = "Python 是一种编程语言..."

result1 = judge_relevance("什么是 RAG？", doc1)
result2 = judge_relevance("什么是 RAG？", doc2)

print(f"文档1 相关性: {result1['score']} - {result1['reason']}")
print(f"文档2 相关性: {result2['score']} - {result2['reason']}")
