"""
BPE 分词练习示例

使用 tiktoken 计算 Token 数量
"""

import tiktoken

def count_tokens(text: str, model: str = "gpt-5-2") -> int:
    """
    计算文本中的 token 数量
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 示例
text = "understanding"
token_count = count_tokens(text)
print(f"Token 数量: {token_count}")  # 约 25-30 个