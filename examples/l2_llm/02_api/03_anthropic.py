"""
Anthropic Claude API 调用示例
演示：Claude 的调用方式与 OpenAI 的区别
"""
import os

from dotenv import load_dotenv
from anthropic import Anthropic

def _normalize_anthropic_base_url(raw: str) -> str:
    """
    Anthropic SDK 会请求固定路径 `/v1/messages`，因此 base_url 不应包含 `/v1` 或 `/v1/messages`。

    常见错误：
    - ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
    - ANTHROPIC_BASE_URL=http://127.0.0.1:5580/v1/messages
    """
    base_url = raw.strip().rstrip("/")

    for suffix in ("/v1/messages", "/v1"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)].rstrip("/")

    return base_url


# 用 `.env` 作为示例脚本的“单一真相”，避免被已导出的环境变量覆盖
load_dotenv(override=True)

model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

base_url = os.getenv("ANTHROPIC_BASE_URL")

if base_url:
    base_url = _normalize_anthropic_base_url(base_url)

print(f"{model} {base_url}")
client = Anthropic(base_url=base_url) if base_url else Anthropic()

# ===== Claude 调用 =====
print("=== Claude 调用 ===")
print(f"base_url: {client.base_url}")
print(f"model: {model}")

response = client.messages.create(
    model=model,
    max_tokens=200,
    system="你是一个简洁的助手，回答控制在50字以内",  # system 是独立参数
    messages=[
        {"role": "user", "content": "什么是RAG？"}
    ]
)

# 获取回复（注意结构不同）
answer = response.content[0].text
print(f"回答：{answer}")

# Token 统计
print(f"\n输入 tokens: {response.usage.input_tokens}")
print(f"输出 tokens: {response.usage.output_tokens}")
