import tiktoken
from typing import List, Tuple

# ===== 1. 基础 Token 操作 =====
print("=== 1. 基础 Token 操作 ===\n")

def tokenize_and_analyze(text: str, model: str = "gpt-5-2") -> dict:
    """分词并分析结果"""

    encoder = tiktoken.encoding_for_model(model)
    token_ids = encoder.encode(text)
    tokens = [encoder.decode([tid]) for tid in token_ids]

    return {
        "text": text,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "tokens": tokens,
        "chars_per_token": len(text) / len(token_ids) if token_ids else 0
    }

# 测试不同类型的文本
test_texts = [
    "Hello, world!",  # 英文
    "人工智能正在改变世界",
    "RAG = Retrieval-Augmented Generation",
    "def hello(): print('Hello')"
]

for text in test_texts:
    result = tokenize_and_analyze(text)
    print(f"文本: {result['text']}")
    print(f"Token 数量: {result['token_count']}")
    print(f"Token IDs: {result['token_ids']}")
    print(f"Tokens: {result['tokens']}")
    print(f"平均每 Token 字符数: {result['chars_per_token']:.2f}\n")

# ===== 2. RAG 成本估算器 =====
print("=== 2. RAG 成本估算器 ===\n")

class RAGCostEstimator:
    """RAG 系统成本估算器"""

    # 价格表 (每1K tokens， 美元)
    PRICES = {
        "gpt-4": { "input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        self.price = self.PRICES.get(model, self.PRICES["gpt-4"])

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.encoder.encode(text))
    
    def estimate_query_cost(
        self,
        query: str,
        context_chunks: List[str],
        expected_output_tokens: int = 500
    ) -> dict:
        """估算单次 RAG 查询成本"""

        # 计算各部分 Token 数
        query_tokens = self.count_tokens(query)
        context_tokens = sum(self.count_tokens(chunk) for chunk in context_chunks)

        # 系统提示词（估算）
        system_prompt_tokens = 100

        # 总输入 Token
        total_input = query_tokens + context_tokens + system_prompt_tokens

        # 计算成本
        input_cost = total_input / 1000 * self.price["input"]
        output_cost = expected_output_tokens / 1000 * self.price["output"]
        total_cost = input_cost + output_cost

        return {
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "total_input_tokens": total_input,
            "output_tokens": expected_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
estimator = RAGCostEstimator(model="gpt-4")

query = "什么是 RAG？它有什么优势？"
context_chunks = [
    "RAG(Retrieval-Augmented Generation) 是一种结合检索和生成的技术...",
    "RAG的主要优势包括: 1. 减少幻觉 2. 知识可更新 3. 可追溯来源...",
    "实现 RAG 需要以下组件: 向量数据库、Embedding 模型、LLM..."
]

cost_result = estimator.estimate_query_cost(query, context_chunks)
print(f"查询: {query}")
print(f"上下文块数: {len(context_chunks)}")
print(f"输入 Token: {cost_result['total_input_tokens']}")
print(f"输出 Token: {cost_result['output_tokens']}")
print(f"预估成本: ${cost_result['total_cost']:.4f}")
print()

# ===== 3. 智能文本切分器 =====
print("=== 3. 智能文本切分器 ===\n")

class TokenAwareChunker:
    """基于 Token 的智能文本切分器"""

    def __init__(self, model: str = "gpt-4", chunk_size: int = 500, overlap: int = 50):
        self.encoder = tiktoken.encoding_for_model(model)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        切分文本，返回（chunk_text, token_count）列表
        """
        tokens = self.encoder.encode(text)
        chunks = []

        start = 0

        while start < len(tokens):
            # 计算结束位置
            end = min(start + self.chunk_size, len(tokens))

            # 提取chunk
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append((chunk_text, len(chunk_tokens)))

            # 移动到下一个位置（考虑重叠）
            start = end - self.overlap if end < len(tokens) else end

        return chunks
    
# 使用示例
sample_text = """
RAG (Retrieval-Augmented Generation, 检索增强生成) 是一种将信息检索与文本生成相结合的技术。
它的核心思想是：在生成回答之前， 先从知识库中检索相关信息，然后将这些信息作为上下文提供给语言模型。

RAG 的工作流程包括以下步骤：
1. 文档预处理：将文档切分成小块，并转换为向量存储
2. 查询处理：将用户问题转换为向量
3. 检索：在向量数据库中找到最相关的文档块
4. 生成：将检索到内容与问题一起发送给 LLM 生成回答

RAG 的优势在于：
- 减少幻觉：基于真实文档生成回答
- 知识可更新：只需更新知识库，无需重新训练模型
- 可追溯：可以提供答案来源
""" * 3 # 重复 3 次以获得更长的文本

chunker = TokenAwareChunker(chunk_size=200, overlap=20)
chunks = chunker.chunk_text(sample_text)

print(f"原文 Token 数： {len(tiktoken.encoding_for_model('gpt-4').encode(sample_text))}")
print(f"切分成 {len(chunks)} 个块:")
for i, (chunk, token_count) in enumerate(chunks):
    preview = chunk[:50].replace('\n', ' ') + "..."
    print(f" 块{i+1}: {token_count} tokens - {preview}")

print()

# ===== 4. Token 效率分析 =====
print("=== 4. Token 效率分析 ===\n")

def analyze_token_efficiency(texts: dict, model: str = "gpt-4") -> None:
    """分析不同语言/内容的 Token 效率"""
    encoder = tiktoken.encoding_for_model(model)

    print(f"模型：{model}")
    print("-" * 60)

    for name, text in texts.items():
        tokens = encoder.encode(text)
        chars = len(text)
        efficiency =  chars / len(tokens) if tokens else 0

        print(f"{name}:")
        print(f"  字符数: {chars}, Token数: {len(tokens)}, 效率: {efficiency:.2f} 字符/Token")
    print()

# 测试不同类型内容
test_contents = {
    "英文文本": "The quick brown fox jumps over the lazy dog.",
    "中文文本": "敏捷的棕色狐狸跳过了懒惰的狗",
    "代码片段": "def hello(): return 'Hello, World!'",
    "混合内容": "RAG系统使用embedding进行semantic search",
    "JSON数据": '{"name": "test", "value": 123, "active": true}',
}

analyze_token_efficiency(test_contents)