"""
多轮对话示例
演示：如何维护对话历史，实现上下文连贯
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ===== 多轮对话 =====
print("=== 多轮对话 ===")

# 对话历史
conversation = [
    {"role": "system", "content": "你是Python编程助手"}
]

def chat(user_message: str) -> str:
    """发送消息并获取回复"""
    # 添加用户消息
    conversation.append({"role": "user", "content": user_message})

    # 调用API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        temperature=0.1,
        max_tokens=300
    )

    # 获取回复
    assistant_message = response.choices[0].message.content

    # 添加到历史（保持上下文）
    conversation.append({"role": "assistant", "content": assistant_message})

    return assistant_message

# 模拟多轮对话
print("用户：什么是列表推导式？")
print(f"助手：{chat('什么是列表推导式？')}\n")

print("用户：能举个例子吗？")
print(f"助手：{chat('能举个例子吗？')}\n")

print("用户：如何添加条件过滤？")
print(f"助手：{chat('如何添加条件过滤？')}")
