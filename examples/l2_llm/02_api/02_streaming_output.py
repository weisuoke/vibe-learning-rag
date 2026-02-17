"""
流式输出示例
演示：实时显示生成内容，适合聊天界面
"""
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ===== 流式调用 =====
print("=== 流式输出 ===")

stream = client.chat.completions.create(
     model="gpt-4o",
     messages=[
        {"role": "user", "content": "用三句话介绍 Python 的优点"}
     ],
     stream=True #开启流式
)

# 逐块输出
print("回答 ", end="")
full_response = ""
for chunk in stream:
    # 某些 chunk（例如用量统计）可能不包含 choices，需要跳过
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    content = delta.content if delta else None
    if content:
        print(content, end="", flush=True)
        full_response += content
print("\n")
print(f"完整回答长度：{len(full_response)} 字符")
