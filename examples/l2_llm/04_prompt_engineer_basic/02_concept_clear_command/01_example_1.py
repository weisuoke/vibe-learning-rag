QUERY_REWRITE_PROMPT = """
任务：将用户的口语化问题转换为适合向量检索的查询

步骤：
1. 识别用户问题的核心意图
2. 提取关键技术术语
3. 扩展同义词和相关概念
4. 生成 2-3 个检索查询变体

成功标准：
- 保留原问题的核心意图
- 包含关键技术术语
- 生成 2-3 个查询变体
- 每个查询 5-15 个词

输入：{user_query}

输出格式:

{{
    "original": "原始问题",
    "intent": "核心意图",
    "queries": ["查询1", "查询2", "查询3"]
}}

示例：
输入："怎么用 Python 做 RAG？"
输出：
{{
  "original": "怎么用 Python 做 RAG？",
  "intent": "学习 Python 实现 RAG 系统",
  "queries": [
    "Python RAG implementation tutorial",
    "retrieval augmented generation Python code",
    "Python vector database RAG example"
  ]
}}
"""

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def rewrite_query(user_query: str) -> dict:
    """查询改写"""
    response = client.chat.completions.create(
        model="gpt-4",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是查询优化专家"},
            {"role": "user", "content": QUERY_REWRITE_PROMPT.format(
                user_query=user_query
            )}
        ],
        temperature=0.1
    )

    return json.loads(response.choices[0].message.content)

# 测试
result = rewrite_query("怎么用 Python 做 RAG？")
print(f"原始问题: {result['original']}")
print(f"核心意图: {result['intent']}")
print(f"检索查询: {result['queries']}")