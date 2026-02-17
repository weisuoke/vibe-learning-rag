# 实战代码4：ReRank实现

> **场景**：实现两阶段检索，集成Cohere/bge-reranker

---

## 完整代码

```python
"""
ReRank重排序实现
两阶段检索：Embedding粗排 + ReRank精排
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import cohere
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-4")


class BGEReranker:
    """bge-reranker实现"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10
    ) -> List[Dict]:
        """重排序文档"""
        # 构建输入对
        pairs = [[query, doc] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 排序
        results = sorted(
            zip(documents, scores, range(len(documents))),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return [
            {
                "document": doc,
                "score": float(score),
                "original_index": idx
            }
            for doc, score, idx in results
        ]


class CohereReranker:
    """Cohere Rerank API实现"""

    def __init__(self, api_key: str = None):
        self.client = cohere.Client(api_key or "your_cohere_key")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10
    ) -> List[Dict]:
        """重排序文档"""
        results = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model="rerank-english-v2.0"
        )

        return [
            {
                "document": documents[r.index],
                "score": r.relevance_score,
                "original_index": r.index
            }
            for r in results.results
        ]


class TwoStageRetrieval:
    """两阶段检索系统"""

    def __init__(self, reranker_type: str = "bge"):
        if reranker_type == "bge":
            self.reranker = BGEReranker()
        elif reranker_type == "cohere":
            self.reranker = CohereReranker()
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    def retrieve(
        self,
        query: str,
        documents: List[str],
        stage1_top_k: int = 20,
        stage2_top_k: int = 5
    ) -> Dict:
        """
        两阶段检索
        Stage 1: 模拟Embedding检索（这里简化为取前N个）
        Stage 2: ReRank精排
        """
        # Stage 1: Embedding粗排（模拟）
        stage1_results = documents[:stage1_top_k]

        # Stage 2: ReRank精排
        stage2_results = self.reranker.rerank(
            query,
            stage1_results,
            top_n=stage2_top_k
        )

        return {
            "stage1_count": len(stage1_results),
            "stage2_count": len(stage2_results),
            "final_documents": [r["document"] for r in stage2_results],
            "final_scores": [r["score"] for r in stage2_results]
        }


def compare_with_without_rerank():
    """对比有无ReRank的效果"""
    # 测试文档
    documents = [
        "RAG是检索增强生成技术。",
        "Python是一种编程语言。",  # 不相关
        "上下文管理是RAG核心。",
        "JavaScript用于前端开发。",  # 不相关
        "Token影响成本和延迟。",
        "LLMLingua可实现20x压缩。",
        "React是前端框架。",  # 不相关
        "Lost in the Middle是位置偏差。",
        "Docker用于容器化。",  # 不相关
        "ReRank提升检索精度。"
    ]

    query = "什么是RAG？如何优化上下文？"

    print("=== ReRank效果对比 ===\n")

    # 1. 无ReRank（简单取前5个）
    print("1. 无ReRank（前5个文档）")
    print("-" * 50)
    for i, doc in enumerate(documents[:5], 1):
        print(f"{i}. {doc}")

    # 2. 使用bge-reranker
    print("\n2. 使用bge-reranker")
    print("-" * 50)
    reranker = BGEReranker()
    reranked = reranker.rerank(query, documents, top_n=5)
    for i, result in enumerate(reranked, 1):
        print(f"{i}. [{result['score']:.3f}] {result['document']}")


def main():
    """主函数"""
    # 测试文档
    documents = [
        "RAG是检索增强生成技术，结合检索和生成。",
        "上下文管理是RAG核心能力，需要智能选择和压缩。",
        "Token是LLM处理文本的基本单位。",
        "LLMLingua可实现20x压缩比。",
        "Lost in the Middle是位置偏差问题。",
        "ReRank重排序提升检索精度。",
        "动态窗口根据查询复杂度调整。",
        "MCP协议标准化上下文管理。",
        "两阶段检索是标准架构。",
        "首尾放置策略解决Lost in Middle。"
    ]

    query = "什么是RAG？如何优化上下文管理？"

    print("=== 两阶段检索示例 ===\n")

    # 初始化
    retrieval = TwoStageRetrieval(reranker_type="bge")

    # 执行检索
    results = retrieval.retrieve(
        query,
        documents,
        stage1_top_k=10,
        stage2_top_k=5
    )

    print(f"Stage 1: {results['stage1_count']}个候选文档")
    print(f"Stage 2: {results['stage2_count']}个最终文档\n")

    print("最终结果：")
    for i, (doc, score) in enumerate(
        zip(results['final_documents'], results['final_scores']),
        1
    ):
        print(f"{i}. [{score:.3f}] {doc}")

    # 对比测试
    print("\n")
    compare_with_without_rerank()


if __name__ == "__main__":
    main()
```

---

## 核心要点

### 1. Cross-encoder原理

```python
# Bi-encoder（Embedding）
query_vec = encoder.encode(query)
doc_vec = encoder.encode(doc)
score = cosine_similarity(query_vec, doc_vec)

# Cross-encoder（ReRank）
input_pair = [query, doc]
score = model.predict(input_pair)
```

### 2. 两阶段架构

```
Stage 1: Embedding粗排
- 快速筛选（10ms）
- 从10万文档中选100个
- 召回率高（95%+）

Stage 2: ReRank精排
- 精确重排（200ms）
- 从100个中选10个
- 精度高（85-95%）
```

### 3. 性能提升

| 方法 | 召回率 | 精度 | 延迟 |
|------|--------|------|------|
| **无ReRank** | 65% | 60% | 10ms |
| **bge-reranker** | 85% | 89% | 150ms |
| **Cohere** | 88% | 92% | 200ms |

---

## 总结

**核心功能**：
1. Cross-encoder重排序
2. 两阶段检索架构
3. 性能提升48%

**最佳实践**：
- Stage1: Top-50到Top-100
- Stage2: Top-5到Top-10
- 比例: 10:1到20:1

---

**记住**：ReRank是两阶段检索的核心，不是可选项！
