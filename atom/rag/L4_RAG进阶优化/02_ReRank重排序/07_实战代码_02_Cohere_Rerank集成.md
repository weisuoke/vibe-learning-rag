# 07_实战代码_02_Cohere_Rerank集成

## 场景说明

Cohere Rerank 是业界领先的商业 reranking 服务，提供多语言支持和高质量的相关性评分。本文展示如何在 RAG 系统中集成 Cohere Rerank v4.0，包括 LangChain 集成、错误处理、批处理优化和成本控制。

**核心价值：**
- 开箱即用的高质量 reranking
- 支持 100+ 语言
- 32k 上下文窗口（v4.0）
- 与 LangChain 无缝集成

---

## 完整实现代码

### 1. 基础集成：Cohere Rerank + LangChain

```python
"""
Cohere Rerank 基础集成示例
展示如何使用 ContextualCompressionRetriever 实现 reranking
"""

import os
from dotenv import load_dotenv
from langchain_cohere import CohereRerank, CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 加载环境变量
load_dotenv()

# 初始化 Cohere Rerank
cohere_rerank = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="rerank-v4.0-pro",  # 或 "rerank-v4.0-fast"
    top_n=5  # 返回前5个最相关的文档
)

# 初始化 Cohere Embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0"
)

# 加载并分块文档
loader = TextLoader("data/documents.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="cohere_demo"
)

# 创建基础检索器
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}  # 初检返回20个候选
)

# 创建压缩检索器（集成 Rerank）
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=base_retriever
)

# 执行检索
query = "What are the benefits of using RAG?"
compressed_docs = compression_retriever.get_relevant_documents(query)

# 打印结果
print(f"检索到 {len(compressed_docs)} 个相关文档：\n")
for i, doc in enumerate(compressed_docs, 1):
    print(f"文档 {i} (相关性分数: {doc.metadata.get('relevance_score', 'N/A')}):")
    print(f"{doc.page_content[:200]}...\n")
```

---

### 2. 高级配置：参数优化与错误处理

```python
"""
Cohere Rerank 高级配置
包含参数优化、错误处理、重试机制
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain.schema import Document
import cohere
from cohere.errors import TooManyRequestsError, InternalServerError

load_dotenv()


class CohereRerankerWithRetry:
    """带重试机制的 Cohere Reranker"""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v4.0-pro",
        top_n: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def rerank(
        self,
        query: str,
        documents: List[str],
        max_tokens_per_doc: int = 4096,
        priority: int = 0
    ) -> List[dict]:
        """
        执行 reranking，带指数退避重试

        Args:
            query: 查询文本
            documents: 文档列表
            max_tokens_per_doc: 每个文档的最大 token 数
            priority: 优先级 (0-999，0最高)

        Returns:
            排序后的文档列表，包含相关性分数
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=documents,
                    top_n=self.top_n,
                    max_tokens_per_doc=max_tokens_per_doc,
                    priority=priority
                )

                # 返回结果
                return [
                    {
                        "index": result.index,
                        "relevance_score": result.relevance_score,
                        "document": result.document.text if hasattr(result.document, 'text') else documents[result.index]
                    }
                    for result in response.results
                ]

            except TooManyRequestsError as e:
                # 429 错误：速率限制
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    print(f"速率限制，等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"达到最大重试次数，仍然遇到速率限制: {e}")

            except InternalServerError as e:
                # 5xx 错误：服务器错误
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"服务器错误，等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"达到最大重试次数，服务器仍然错误: {e}")

            except Exception as e:
                # 其他错误直接抛出
                raise Exception(f"Rerank 失败: {e}")

        return []


# 使用示例
def main():
    reranker = CohereRerankerWithRetry(
        api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-v4.0-fast",  # 使用 fast 版本降低延迟
        top_n=5,
        max_retries=3,
        retry_delay=1.0
    )

    query = "How does RAG improve LLM accuracy?"
    documents = [
        "RAG combines retrieval with generation to provide factual answers.",
        "Large language models can hallucinate without external knowledge.",
        "Vector databases store embeddings for semantic search.",
        "Reranking improves the quality of retrieved documents.",
        "Python is a popular programming language for AI development."
    ]

    results = reranker.rerank(
        query=query,
        documents=documents,
        max_tokens_per_doc=4096,
        priority=0  # 最高优先级
    )

    print("Reranking 结果：\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result['relevance_score']:.4f}]")
        print(f"   {result['document']}\n")


if __name__ == "__main__":
    main()
```

---

### 3. 批处理优化：处理大量文档

```python
"""
Cohere Rerank 批处理优化
处理超过 1000 个文档的场景
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
import cohere
from tqdm import tqdm

load_dotenv()


class BatchReranker:
    """批处理 Reranker"""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v4.0-fast",
        batch_size: int = 1000,  # Cohere 限制每次最多 10000
        top_n_per_batch: int = 100
    ):
        self.client = cohere.Client(api_key)
        self.model = model
        self.batch_size = batch_size
        self.top_n_per_batch = top_n_per_batch

    def rerank_large_corpus(
        self,
        query: str,
        documents: List[str],
        final_top_n: int = 10
    ) -> List[Dict]:
        """
        对大量文档进行分批 reranking

        策略：
        1. 将文档分批（每批 1000 个）
        2. 对每批进行 rerank，保留 top 100
        3. 合并所有批次的 top 100
        4. 对合并结果再次 rerank，得到最终 top N

        Args:
            query: 查询文本
            documents: 文档列表（可能很大）
            final_top_n: 最终返回的文档数量

        Returns:
            排序后的文档列表
        """
        print(f"处理 {len(documents)} 个文档...")

        # 第一阶段：分批 rerank
        all_candidates = []

        for i in tqdm(range(0, len(documents), self.batch_size), desc="批处理 Rerank"):
            batch = documents[i:i + self.batch_size]

            try:
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=batch,
                    top_n=min(self.top_n_per_batch, len(batch))
                )

                # 保存候选结果（带原始索引）
                for result in response.results:
                    all_candidates.append({
                        "original_index": i + result.index,
                        "relevance_score": result.relevance_score,
                        "document": batch[result.index]
                    })

            except Exception as e:
                print(f"批次 {i//self.batch_size + 1} 失败: {e}")
                continue

        print(f"第一阶段完成，获得 {len(all_candidates)} 个候选文档")

        # 第二阶段：对候选结果再次 rerank
        if len(all_candidates) <= final_top_n:
            return all_candidates

        candidate_docs = [c["document"] for c in all_candidates]

        try:
            final_response = self.client.rerank(
                model=self.model,
                query=query,
                documents=candidate_docs,
                top_n=final_top_n
            )

            final_results = []
            for result in final_response.results:
                original_candidate = all_candidates[result.index]
                final_results.append({
                    "original_index": original_candidate["original_index"],
                    "relevance_score": result.relevance_score,
                    "document": original_candidate["document"]
                })

            return final_results

        except Exception as e:
            print(f"第二阶段 rerank 失败: {e}")
            # 降级：返回第一阶段的 top N
            return sorted(
                all_candidates,
                key=lambda x: x["relevance_score"],
                reverse=True
            )[:final_top_n]


# 使用示例
def main():
    # 模拟大量文档
    documents = [
        f"Document {i}: This is a sample document about various topics including AI, ML, and RAG."
        for i in range(5000)
    ]

    # 插入一些相关文档
    documents[100] = "RAG (Retrieval-Augmented Generation) significantly improves LLM accuracy by providing relevant context."
    documents[500] = "Reranking is a crucial step in RAG pipelines to ensure the most relevant documents are used."
    documents[1000] = "Cohere Rerank v4.0 supports 100+ languages and has a 32k context window."

    reranker = BatchReranker(
        api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-v4.0-fast",
        batch_size=1000,
        top_n_per_batch=100
    )

    query = "How does reranking improve RAG systems?"
    results = reranker.rerank_large_corpus(
        query=query,
        documents=documents,
        final_top_n=5
    )

    print("\n最终 Top 5 结果：\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [原始索引: {result['original_index']}, 分数: {result['relevance_score']:.4f}]")
        print(f"   {result['document'][:100]}...\n")


if __name__ == "__main__":
    main()
```

---

### 4. 成本控制：智能缓存与降级策略

```python
"""
Cohere Rerank 成本控制
实现语义缓存和降级策略
"""

import os
import hashlib
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
import cohere
from langchain_cohere import CohereEmbeddings
import numpy as np

load_dotenv()


class CostOptimizedReranker:
    """成本优化的 Reranker"""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v4.0-fast",
        cache_file: str = "rerank_cache.json",
        similarity_threshold: float = 0.95,
        use_fallback: bool = True
    ):
        self.client = cohere.Client(api_key)
        self.model = model
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.use_fallback = use_fallback

        # 加载缓存
        self.cache = self._load_cache()

        # 初始化 embeddings（用于语义缓存）
        self.embeddings = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-english-light-v3.0"  # 使用轻量版降低成本
        )

    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, query: str, documents: List[str]) -> str:
        """生成缓存键"""
        content = query + "|||" + "|||".join(sorted(documents))
        return hashlib.md5(content.encode()).hexdigest()

    def _check_semantic_cache(self, query: str) -> Optional[str]:
        """检查语义缓存"""
        if not self.cache:
            return None

        # 计算查询的 embedding
        query_embedding = self.embeddings.embed_query(query)

        # 与缓存中的查询比较
        for cached_query, cached_data in self.cache.items():
            if "query_embedding" not in cached_data:
                continue

            cached_embedding = cached_data["query_embedding"]
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity >= self.similarity_threshold:
                print(f"命中语义缓存 (相似度: {similarity:.4f})")
                return cached_query

        return None

    def _fallback_rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict]:
        """降级策略：使用 embedding 相似度"""
        print("使用降级策略：embedding 相似度排序")

        # 计算查询和文档的 embeddings
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents(documents)

        # 计算相似度
        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            scores.append({
                "index": i,
                "relevance_score": float(similarity),
                "document": documents[i]
            })

        # 排序并返回 top N
        scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scores[:top_n]

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5
    ) -> List[Dict]:
        """
        执行 reranking，带缓存和降级

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的文档数量

        Returns:
            排序后的文档列表
        """
        # 1. 检查精确缓存
        cache_key = self._get_cache_key(query, documents)
        if cache_key in self.cache:
            print("命中精确缓存")
            return self.cache[cache_key]["results"]

        # 2. 检查语义缓存
        similar_query = self._check_semantic_cache(query)
        if similar_query:
            similar_cache_key = self._get_cache_key(similar_query, documents)
            if similar_cache_key in self.cache:
                return self.cache[similar_cache_key]["results"]

        # 3. 调用 Cohere Rerank API
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n
            )

            results = [
                {
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    "document": documents[result.index]
                }
                for result in response.results
            ]

            # 保存到缓存
            query_embedding = self.embeddings.embed_query(query)
            self.cache[cache_key] = {
                "query": query,
                "query_embedding": query_embedding,
                "results": results
            }
            self._save_cache()

            return results

        except Exception as e:
            print(f"Rerank API 失败: {e}")

            # 4. 降级策略
            if self.use_fallback:
                return self._fallback_rerank(query, documents, top_n)
            else:
                raise


# 使用示例
def main():
    reranker = CostOptimizedReranker(
        api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-v4.0-fast",
        cache_file="rerank_cache.json",
        similarity_threshold=0.95,
        use_fallback=True
    )

    documents = [
        "RAG combines retrieval with generation.",
        "Vector databases enable semantic search.",
        "Reranking improves document relevance.",
        "LLMs can hallucinate without context.",
        "Python is great for AI development."
    ]

    # 第一次查询（调用 API）
    print("=== 第一次查询 ===")
    query1 = "How does RAG work?"
    results1 = reranker.rerank(query1, documents, top_n=3)

    for i, result in enumerate(results1, 1):
        print(f"{i}. [{result['relevance_score']:.4f}] {result['document']}")

    # 第二次查询（命中缓存）
    print("\n=== 第二次查询（相同） ===")
    results2 = reranker.rerank(query1, documents, top_n=3)

    # 第三次查询（语义相似，命中语义缓存）
    print("\n=== 第三次查询（语义相似） ===")
    query3 = "Explain how RAG functions"
    results3 = reranker.rerank(query3, documents, top_n=3)


if __name__ == "__main__":
    main()
```

---

## 代码说明

### 核心组件

1. **基础集成**：使用 `ContextualCompressionRetriever` 包装 `CohereRerank`
2. **错误处理**：指数退避重试机制，处理 429 和 5xx 错误
3. **批处理**：两阶段 reranking，处理超过 1000 个文档
4. **成本控制**：精确缓存 + 语义缓存 + embedding 降级

### 参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `model` | `rerank-v4.0-fast` | 低延迟场景 |
| `model` | `rerank-v4.0-pro` | 高质量场景 |
| `top_n` | 5-10 | 返回的文档数量 |
| `max_tokens_per_doc` | 4096 | 每个文档的最大 token 数 |
| `priority` | 0 | 最高优先级（0-999） |

---

## 运行示例

### 环境准备

```bash
# 安装依赖
pip install cohere langchain langchain-cohere langchain-community chromadb

# 配置 API Key
export COHERE_API_KEY="your_api_key_here"
```

### 执行代码

```bash
# 基础集成
python cohere_basic.py

# 高级配置
python cohere_advanced.py

# 批处理
python cohere_batch.py

# 成本控制
python cohere_cost_optimized.py
```

### 预期输出

```
检索到 5 个相关文档：

文档 1 (相关性分数: 0.9876):
RAG combines retrieval with generation to provide factual answers...

文档 2 (相关性分数: 0.8543):
Reranking improves the quality of retrieved documents...

...
```

---

## 性能优化

### 1. 延迟优化

```python
# 使用 fast 模型
model = "rerank-v4.0-fast"  # ~100ms vs ~300ms (pro)

# 减少候选文档数量
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 而非 20

# 使用优先级参数
priority = 0  # 最高优先级
```

### 2. 成本优化

```python
# 启用缓存
use_cache = True

# 使用轻量级 embedding
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# 批处理优化
batch_size = 1000  # 减少 API 调用次数
```

### 3. 质量优化

```python
# 使用 pro 模型
model = "rerank-v4.0-pro"

# 增加候选文档数量
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# 结构化数据使用 YAML 格式
import yaml
yaml_docs = [yaml.dump(doc, sort_keys=False) for doc in structured_docs]
```

---

## 常见问题

### Q1: 如何处理 429 错误？

**A:** 实现指数退避重试：

```python
for attempt in range(max_retries):
    try:
        response = client.rerank(...)
        break
    except TooManyRequestsError:
        wait_time = retry_delay * (2 ** attempt)
        time.sleep(wait_time)
```

### Q2: 如何降低成本？

**A:** 三种策略：
1. 使用 `rerank-v4.0-fast` 而非 `pro`
2. 启用语义缓存
3. 减少候选文档数量

### Q3: 如何处理超过 10000 个文档？

**A:** 使用两阶段 reranking：
1. 分批处理，每批保留 top 100
2. 合并所有批次的结果，再次 rerank

### Q4: 如何提高 reranking 质量？

**A:**
1. 使用 `rerank-v4.0-pro` 模型
2. 增加初检的候选文档数量（k=50）
3. 结构化数据转为 YAML 格式

---

## 参考资料

### 官方文档
- [Cohere Rerank API Reference](https://docs.cohere.com/reference/rerank) - API 参数详解
- [Cohere Rerank Best Practices](https://docs.cohere.com/docs/reranking-best-practices) - 性能优化建议
- [Cohere Rerank on LangChain](https://docs.cohere.com/docs/rerank-on-langchain) - LangChain 集成指南

### 技术文章
- [Cohere Rerank v4.0 Release](https://docs.cohere.com/changelog/rerank-v4.0) - v4.0 新特性
- [Enhancing RAG with Reranking](https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311) - RAG 中的 reranking
- [Mastering RAG: Reranking Models](https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model) - Reranker 选型指南

### 代码示例
- [Cohere Developer Experience](https://github.com/cohere-ai/cohere-developer-experience) - 官方示例代码
- [LangChain Cohere Integration](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/retrievers/document_compressors) - LangChain 集成源码

---

**版本：** v1.0 (2026年标准)
**最后更新：** 2026-02-16
**代码测试：** Python 3.13 + cohere 5.x + langchain 0.3.x
**API 版本：** Cohere Rerank v4.0
