# 实战代码2：LLMLingua压缩

> **场景**：集成LLMLingua实现智能上下文压缩

---

## 完整代码

```python
"""
LLMLingua上下文压缩
实现智能压缩、LangChain集成、性能对比
"""

from llmlingua import PromptCompressor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-4")


class LLMLinguaCompressor:
    """LLMLingua压缩器"""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
        self.compressor = PromptCompressor(
            model_name=model_name,
            device="cpu"  # 使用CPU，如有GPU可改为"cuda"
        )

    def compress(
        self,
        text: str,
        rate: float = 0.5,
        question: str = None
    ) -> Dict:
        """
        压缩文本
        rate: 压缩率（0.5表示压缩到50%）
        question: 问题感知压缩
        """
        result = self.compressor.compress_prompt(
            text,
            rate=rate,
            question=question,
            force_tokens=['\n', '?', '!', '.', ',']  # 保留标点
        )

        return {
            "compressed_text": result['compressed_prompt'],
            "original_tokens": len(encoding.encode(text)),
            "compressed_tokens": len(encoding.encode(result['compressed_prompt'])),
            "actual_ratio": result.get('ratio', 0),
            "compression_time": result.get('time', 0)
        }

    def compress_documents(
        self,
        documents: List[str],
        rate: float = 0.5,
        question: str = None
    ) -> str:
        """压缩文档列表"""
        combined = "\n\n---\n\n".join(documents)
        result = self.compress(combined, rate, question)
        return result["compressed_text"]


class RAGWithCompression:
    """带压缩的RAG系统"""

    def __init__(self, compression_rate: float = 0.25):
        self.compressor = LLMLinguaCompressor()
        self.compression_rate = compression_rate

    def query(
        self,
        question: str,
        documents: List[str],
        use_compression: bool = True
    ) -> Dict:
        """查询流程"""
        # 1. 构建上下文
        if use_compression:
            # 使用压缩
            context = self.compressor.compress_documents(
                documents,
                rate=self.compression_rate,
                question=question
            )
        else:
            # 不压缩
            context = "\n\n---\n\n".join(documents)

        # 2. 构建Prompt
        prompt = f"""基于以下上下文回答问题。

上下文：
{context}

问题：{question}

答案："""

        # 3. 调用LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一个helpful的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        # 4. 统计
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = response.usage.completion_tokens

        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": self._calculate_cost(prompt_tokens, completion_tokens),
            "compression_used": use_compression
        }

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """计算成本"""
        return (prompt_tokens / 1_000_000) * 10 + (completion_tokens / 1_000_000) * 30


def compare_compression_ratios():
    """对比不同压缩比的效果"""
    compressor = LLMLinguaCompressor()

    # 测试文本
    text = """
    RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
    它先从知识库中检索相关文档，然后将这些文档作为上下文传递给LLM生成答案。
    上下文管理是RAG系统的核心能力。由于LLM的Context Window有限，需要智能地选择和压缩上下文。
    Token是LLM处理文本的基本单位。Token数量直接影响成本和延迟。
    LLMLingua是微软研究院提出的上下文压缩技术，可以实现20x压缩比。
    """

    print("=== 不同压缩比对比 ===\n")

    for rate in [0.7, 0.5, 0.25, 0.1]:
        result = compressor.compress(text, rate=rate)
        print(f"压缩率: {rate} ({int((1-rate)*100)}%压缩)")
        print(f"  原始: {result['original_tokens']} tokens")
        print(f"  压缩后: {result['compressed_tokens']} tokens")
        print(f"  实际压缩比: {result['actual_ratio']:.2f}")
        print(f"  压缩文本预览: {result['compressed_text'][:100]}...")
        print()


def main():
    """主函数"""
    # 测试文档
    documents = [
        "RAG是一种结合检索和生成的技术，先检索相关文档，再传递给LLM生成答案。",
        "上下文管理是RAG核心能力，需要智能选择和压缩上下文。",
        "Token是LLM处理文本的基本单位，影响成本和延迟。",
        "LLMLingua可实现20x压缩比，同时保持甚至提升性能。",
        "Lost in the Middle是指LLM对长上下文中间部分召回率低的现象。"
    ]

    query = "什么是RAG？为什么需要上下文管理？"

    print("=== LLMLingua压缩示例 ===\n")

    # 1. 对比不同压缩比
    compare_compression_ratios()

    # 2. RAG对比（压缩 vs 不压缩）
    print("\n=== RAG性能对比 ===\n")

    rag = RAGWithCompression(compression_rate=0.25)

    # 不压缩
    print("1. 不压缩")
    print("-" * 50)
    result_no_compress = rag.query(query, documents, use_compression=False)
    print(f"Prompt tokens: {result_no_compress['prompt_tokens']}")
    print(f"成本: ${result_no_compress['cost']:.4f}")
    print(f"答案: {result_no_compress['answer'][:100]}...\n")

    # 压缩（4x）
    print("2. 4x压缩")
    print("-" * 50)
    result_compress = rag.query(query, documents, use_compression=True)
    print(f"Prompt tokens: {result_compress['prompt_tokens']}")
    print(f"成本: ${result_compress['cost']:.4f}")
    print(f"答案: {result_compress['answer'][:100]}...\n")

    # 对比
    print("3. 对比结果")
    print("-" * 50)
    token_reduction = (result_no_compress['prompt_tokens'] - result_compress['prompt_tokens']) / result_no_compress['prompt_tokens']
    cost_reduction = (result_no_compress['cost'] - result_compress['cost']) / result_no_compress['cost']

    print(f"Token减少: {token_reduction:.1%}")
    print(f"成本降低: {cost_reduction:.1%}")


if __name__ == "__main__":
    main()
```

---

## 核心要点

### 1. LLMLingua初始化

```python
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device="cpu"
)
```

### 2. 问题感知压缩

```python
result = compressor.compress_prompt(
    text,
    rate=0.25,  # 4x压缩
    question=question  # 保留与问题相关的内容
)
```

### 3. 性能对比

| 压缩比 | Token减少 | 成本降低 | 质量影响 |
|--------|----------|---------|---------|
| 2x (0.5) | 50% | 50% | +5% |
| 4x (0.25) | 75% | 75% | +17% |
| 10x (0.1) | 90% | 90% | +12% |

---

## 总结

**核心功能**：
1. 智能压缩：LLMLingua 20x压缩
2. 问题感知：保留相关内容
3. 性能提升：4x压缩提升17.1%

**最佳实践**：
- 4x压缩是最佳平衡点
- 使用问题感知压缩
- 监控压缩后质量

---

**记住**：LLMLingua不是简单截断，而是智能提炼！
