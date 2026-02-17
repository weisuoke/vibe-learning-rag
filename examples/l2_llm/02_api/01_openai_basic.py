"""
OpenAI API 基础调用示例
演示：基本请求、响应解析、token 统计
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== 1. 创建客户端 =====
client = OpenAI()

# ===== 2. 基础调用 =====
print("=== 基础调用 ===")

response = client.chat.completions.create(
    model = "gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个简洁的助手，回答控制在50字以内"},
        {"role": "user", "content": "什么是RAG？"}
    ],
    temperature=0.1,
    max_tokens=200
)

# 获取回复
answer = response.choices[0].message.content
print(f"回答：{answer}")

# Token 统计
print(f"\n输入 tokens: {response.usage.prompt_tokens}")
print(f"输出 tokens: {response.usage.completion_tokens}")
print(f"总计 tokens: {response.usage.total_tokens}")

# 结束原因
print(f"结束原因: {response.choices[0].finish_reason}")