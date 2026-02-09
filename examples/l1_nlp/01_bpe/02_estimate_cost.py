def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-5-2") -> float:
    """估算 API 调用成本 (美元)"""
    # GPT-4 价格 （2024年参考价）
    prices = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # 每1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    price = prices.get(model)
    cost = (input_tokens / 1000) * price["input"] + (output_tokens / 1000) * price["output"]
    return cost

# 示例： RAG 查询成本估算
context_tokens = 2000  # 检索到的上下文 
query_tokens = 50     # 用户查询
output_tokens = 500  # 模型生成的回答

cost = estimate_cost(context_tokens + query_tokens, output_tokens, model="gpt-4-turbo")
print(f"估算的 API 调用成本: ${cost:.4f}")