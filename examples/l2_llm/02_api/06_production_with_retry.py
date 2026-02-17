"""
生产级 API 调用示例
演示：错误处理、重试机制、日志记录
"""

import time
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def robust_llm_call(
    messages: list,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 500
) -> dict: 
    """
    带重试机制的 LLM 调用

    Returns:
        包含 content 和 usage 的字典
    """

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30
        )

        latency = time.time() - start_time

        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "latency": latency
        }
    except RateLimitError as e:
        print(f"[警告] 触发限流，等待重试... {e}")
        raise
    except APIConnectionError as e:
        print(f"[警告] 网络错误，等待重试... {e}")
        raise
    except APIError as e:
        print(f"[错误] API 错误: {e}")
        raise

# ===== 测试 =====
print("=== 生产级调用测试 ===\n")

result = robust_llm_call(
    messages=[
        {"role": "user", "content": "用一句话介绍Python"}
    ]
)

print(f"回答：{result['content']}")
print(f"延迟：{result['latency']:.2f}s")
print(f"Token 使用：{result['usage']}")