# 07_实战代码_03_混合检索+Rerank管道

## 场景说明

混合检索结合了 BM25 关键词匹配和向量语义搜索的优势,通过 RRF(Reciprocal Rank Fusion)融合结果,再用 Cross-Encoder 进行精排,是生产级 RAG 系统的标准配置。

**核心价值:**
- BM25 捕获精确关键词匹配
- 向量检索捕获语义相似性
- RRF 融合两种检索结果
- Cross-Encoder 精排提升最终质量

**适用场景:**
- 企业知识库问答
- 文档检索系统
- 智能客服
- 代码搜索

---

## 完整实现代码

### 1. 基础混合检索管道

```python
"""
混合检索 + Rerank 完整管道
BM25 + 向量检索 + RRF 融合 + Cross-Encoder 重排序
"""

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np

load_dotenv()


class HybridRetriever:
    """混合检索器:BM25 + 向量检索 + RRF 融合"""

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "text-embedding-3-small",
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]

        # 初始化 BM25
        tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        # 初始化向量存储
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="hybrid_search"
        )

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 检索"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # 返回 (索引, 分数) 列表
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return ranked

    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """向量检索"""
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k
        )

        # 转换为 (索引, 分数) 格式
        ranked = []
        for doc, score in results:
            # 找到文档在原始列表中的索引
            try:
                idx = self.texts.index(doc.page_content)
                # Chroma 返回的是距离,转换为相似度
                similarity = 1 / (1 + score)
                ranked.append((idx, similarity))
            except ValueError:
                continue

        return ranked

    def reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        RRF 融合算法

        公式: RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            rankings: 多个排序列表 [(doc_idx, score), ...]
            k: RRF 参数,默认 60

        Returns:
            融合后的排序列表
        """
        rrf_scores = {}

        for ranking in rankings:
            for rank, (doc_idx, _) in enumerate(ranking, start=1):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0
                rrf_scores[doc_idx] += 1 / (k + rank)

        # 按 RRF 分数排序
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_docs

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = 20,
        vector_top_k: int = 20
    ) -> List[Document]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 最终返回的文档数量
            bm25_top_k: BM25 初检数量
            vector_top_k: 向量检索初检数量

        Returns:
            融合后的文档列表
        """
        # 1. BM25 检索
        bm25_results = self.bm25_search(query, bm25_top_k)

        # 2. 向量检索
        vector_results = self.vector_search(query, vector_top_k)

        # 3. RRF 融合
        fused_results = self.reciprocal_rank_fusion(
            [bm25_results, vector_results]
        )

        # 4. 返回 top_k 文档
        top_docs = []
        for doc_idx, score in fused_results[:top_k]:
            doc = self.documents[doc_idx]
            # 添加融合分数到元数据
            doc.metadata['rrf_score'] = score
            top_docs.append(doc)

        return top_docs


class CrossEncoderReranker:
    """Cross-Encoder 重排序器"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        使用 Cross-Encoder 重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回的文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 准备 (query, doc) 对
        pairs = [(query, doc.page_content) for doc in documents]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 排序
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # 返回重排序后的文档
        reranked_docs = []
        for idx in ranked_indices:
            doc = documents[idx]
            doc.metadata['rerank_score'] = float(scores[idx])
            reranked_docs.append(doc)

        return reranked_docs


class HybridRAGPipeline:
    """完整的混合检索 + Rerank RAG 管道"""

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "text-embedding-3-small",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        self.retriever = HybridRetriever(
            documents=documents,
            embedding_model=embedding_model
        )
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

    def search(
        self,
        query: str,
        retrieval_top_k: int = 20,
        final_top_k: int = 5
    ) -> List[Document]:
        """
        执行混合检索 + 重排序

        Args:
            query: 查询文本
            retrieval_top_k: 混合检索返回的候选数量
            final_top_k: 最终返回的文档数量

        Returns:
            重排序后的文档列表
        """
        # 1. 混合检索
        candidates = self.retriever.retrieve(query, top_k=retrieval_top_k)

        # 2. Cross-Encoder 重排序
        reranked = self.reranker.rerank(query, candidates, top_k=final_top_k)

        return reranked


# 使用示例
def main():
    # 准备文档
    documents = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines retrieval with generation to provide factual answers.",
            metadata={"source": "doc1"}
        ),
        Document(
            page_content="BM25 is a ranking function used for information retrieval based on term frequency.",
            metadata={"source": "doc2"}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search and similarity matching.",
            metadata={"source": "doc3"}
        ),
        Document(
            page_content="Cross-Encoder models provide superior reranking performance compared to bi-encoders.",
            metadata={"source": "doc4"}
        ),
        Document(
            page_content="Hybrid search combines keyword matching with semantic search for better results.",
            metadata={"source": "doc5"}
        ),
        Document(
            page_content="Python is a popular programming language for AI and machine learning development.",
            metadata={"source": "doc6"}
        ),
        Document(
            page_content="LangChain provides tools for building RAG applications with various retrievers.",
            metadata={"source": "doc7"}
        ),
        Document(
            page_content="Reciprocal Rank Fusion (RRF) is an effective method for combining multiple rankings.",
            metadata={"source": "doc8"}
        ),
    ]

    # 创建管道
    pipeline = HybridRAGPipeline(documents)

    # 执行搜索
    query = "How does hybrid search improve RAG systems?"
    results = pipeline.search(query, retrieval_top_k=20, final_top_k=3)

    # 打印结果
    print(f"查询: {query}\n")
    print("=" * 80)

    for i, doc in enumerate(results, 1):
        print(f"\n文档 {i}:")
        print(f"来源: {doc.metadata.get('source', 'N/A')}")
        print(f"RRF 分数: {doc.metadata.get('rrf_score', 'N/A'):.4f}")
        print(f"Rerank 分数: {doc.metadata.get('rerank_score', 'N/A'):.4f}")
        print(f"内容: {doc.page_content}")
        print("-" * 80)


if __name__ == "__main__":
    main()
```

---

### 2. 优化版:批处理 + 缓存

```python
"""
优化的混合检索管道
支持批处理和缓存
"""

import hashlib
import json
from typing import List, Dict, Optional
from functools import lru_cache


class OptimizedHybridRAGPipeline:
    """优化的混合检索管道"""

    def __init__(
        self,
        documents: List[Document],
        cache_file: str = "search_cache.json",
        enable_cache: bool = True
    ):
        self.retriever = HybridRetriever(documents)
        self.reranker = CrossEncoderReranker()
        self.cache_file = cache_file
        self.enable_cache = enable_cache
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """加载缓存"""
        if not self.enable_cache:
            return {}

        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """保存缓存"""
        if self.enable_cache:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, query: str, retrieval_top_k: int, final_top_k: int) -> str:
        """生成缓存键"""
        content = f"{query}_{retrieval_top_k}_{final_top_k}"
        return hashlib.md5(content.encode()).hexdigest()

    def search(
        self,
        query: str,
        retrieval_top_k: int = 20,
        final_top_k: int = 5,
        use_cache: bool = True
    ) -> List[Document]:
        """
        执行混合检索 + 重排序(带缓存)

        Args:
            query: 查询文本
            retrieval_top_k: 混合检索返回的候选数量
            final_top_k: 最终返回的文档数量
            use_cache: 是否使用缓存

        Returns:
            重排序后的文档列表
        """
        # 检查缓存
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(query, retrieval_top_k, final_top_k)
            if cache_key in self.cache:
                print("命中缓存")
                cached_data = self.cache[cache_key]
                # 重建文档对象
                return [
                    Document(
                        page_content=item['content'],
                        metadata=item['metadata']
                    )
                    for item in cached_data
                ]

        # 执行检索
        candidates = self.retriever.retrieve(query, top_k=retrieval_top_k)
        reranked = self.reranker.rerank(query, candidates, top_k=final_top_k)

        # 保存到缓存
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(query, retrieval_top_k, final_top_k)
            self.cache[cache_key] = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in reranked
            ]
            self._save_cache()

        return reranked

    def batch_search(
        self,
        queries: List[str],
        retrieval_top_k: int = 20,
        final_top_k: int = 5
    ) -> List[List[Document]]:
        """
        批量搜索

        Args:
            queries: 查询列表
            retrieval_top_k: 混合检索返回的候选数量
            final_top_k: 最终返回的文档数量

        Returns:
            每个查询的结果列表
        """
        results = []
        for query in queries:
            result = self.search(query, retrieval_top_k, final_top_k)
            results.append(result)

        return results
```

---

### 3. LangChain 集成版本

```python
"""
使用 LangChain 实现混合检索 + Rerank
"""

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def create_langchain_hybrid_retriever(
    documents: List[Document],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int = 20
):
    """
    创建 LangChain 混合检索器

    Args:
        documents: 文档列表
        bm25_weight: BM25 权重
        vector_weight: 向量检索权重
        top_k: 返回的文档数量

    Returns:
        混合检索器
    """
    # 1. 创建 BM25 检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k

    # 2. 创建向量检索器
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. 创建集成检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )

    return ensemble_retriever


def create_reranked_retriever(
    base_retriever,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    top_n: int = 5
):
    """
    创建带 Rerank 的检索器

    Args:
        base_retriever: 基础检索器
        model_name: Cross-Encoder 模型名称
        top_n: 返回的文档数量

    Returns:
        带 Rerank 的检索器
    """
    # 创建 Cross-Encoder
    model = HuggingFaceCrossEncoder(model_name=model_name)

    # 创建 Reranker
    compressor = CrossEncoderReranker(model=model, top_n=top_n)

    # 创建压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever


# 使用示例
def langchain_example():
    documents = [
        Document(page_content="RAG combines retrieval with generation.", metadata={"id": 1}),
        Document(page_content="BM25 is a ranking function for information retrieval.", metadata={"id": 2}),
        Document(page_content="Vector databases enable semantic search.", metadata={"id": 3}),
        Document(page_content="Cross-Encoder models provide superior reranking.", metadata={"id": 4}),
        Document(page_content="Hybrid search combines keyword and semantic search.", metadata={"id": 5}),
    ]

    # 创建混合检索器
    hybrid_retriever = create_langchain_hybrid_retriever(
        documents,
        bm25_weight=0.5,
        vector_weight=0.5,
        top_k=10
    )

    # 添加 Rerank
    reranked_retriever = create_reranked_retriever(
        hybrid_retriever,
        top_n=3
    )

    # 执行检索
    query = "How does hybrid search work?"
    results = reranked_retriever.get_relevant_documents(query)

    print(f"查询: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   元数据: {doc.metadata}\n")


if __name__ == "__main__":
    langchain_example()
```

---

## 代码说明

### 核心组件

1. **HybridRetriever**: 混合检索器
   - `bm25_search()`: BM25 关键词检索
   - `vector_search()`: 向量语义检索
   - `reciprocal_rank_fusion()`: RRF 融合算法

2. **CrossEncoderReranker**: Cross-Encoder 重排序器
   - 使用 `sentence-transformers` 的 Cross-Encoder 模型
   - 计算 query-document 对的相关性分数

3. **HybridRAGPipeline**: 完整管道
   - 组合混合检索和重排序
   - 提供简单的搜索接口

### RRF 融合算法

```python
RRF_score(d) = Σ 1/(k + rank_i(d))
```

- `k`: 常数,通常设为 60
- `rank_i(d)`: 文档 d 在第 i 个排序列表中的排名
- 对所有排序列表求和

---

## 运行示例

### 环境准备

```bash
# 安装依赖
pip install rank-bm25 sentence-transformers langchain langchain-openai langchain-community chromadb

# 配置 API Key
export OPENAI_API_KEY="your_api_key_here"
```

### 执行代码

```bash
python hybrid_rag_pipeline.py
```

### 预期输出

```
查询: How does hybrid search improve RAG systems?

================================================================================

文档 1:
来源: doc5
RRF 分数: 0.0323
Rerank 分数: 8.9234
内容: Hybrid search combines keyword matching with semantic search for better results.
--------------------------------------------------------------------------------

文档 2:
来源: doc8
RRF 分数: 0.0312
Rerank 分数: 7.8765
内容: Reciprocal Rank Fusion (RRF) is an effective method for combining multiple rankings.
--------------------------------------------------------------------------------

文档 3:
来源: doc1
RRF 分数: 0.0298
Rerank 分数: 7.2341
内容: RAG (Retrieval-Augmented Generation) combines retrieval with generation to provide factual answers.
--------------------------------------------------------------------------------
```

---

## 性能优化

### 1. 延迟优化

```python
# 减少候选文档数量
retrieval_top_k = 10  # 而非 20

# 使用更快的 Cross-Encoder
reranker_model = "cross-encoder/ms-marco-TinyBERT-L2-v2"  # 9000 docs/sec

# 启用缓存
enable_cache = True
```

### 2. 质量优化

```python
# 增加候选文档数量
retrieval_top_k = 50

# 使用更强的 Cross-Encoder
reranker_model = "cross-encoder/ms-marco-MiniLM-L12-v2"  # 更高质量

# 调整 RRF 参数
k = 60  # 默认值,可根据数据集调整
```

### 3. 权重调整

```python
# BM25 权重更高(适合精确匹配场景)
bm25_weight = 0.7
vector_weight = 0.3

# 向量权重更高(适合语义理解场景)
bm25_weight = 0.3
vector_weight = 0.7
```

---

## 常见问题

### Q1: RRF 的 k 参数如何选择?

**A:** 默认 60 适用于大多数场景。k 越小,排名靠前的文档权重越大。

```python
# 更重视排名靠前的文档
k = 30

# 更平滑的融合
k = 100
```

### Q2: BM25 和向量检索的权重如何平衡?

**A:** 根据场景调整:
- 精确匹配场景(如代码搜索): BM25 权重更高
- 语义理解场景(如问答): 向量权重更高
- 通用场景: 各 0.5

### Q3: Cross-Encoder 模型如何选择?

**A:** 根据延迟和质量要求:

| 模型 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| TinyBERT-L2 | 9000 docs/s | 中 | 低延迟 |
| MiniLM-L6 | 1800 docs/s | 高 | 平衡 |
| MiniLM-L12 | 960 docs/s | 很高 | 高质量 |

### Q4: 如何处理大量文档?

**A:** 使用两阶段检索:
1. 混合检索返回 top 100
2. Cross-Encoder 重排序 top 10

---

## 参考资料

### 官方文档
- [Sentence Transformers Cross-Encoder](https://sbert.net/docs/cross_encoder/usage/usage.html) - Cross-Encoder 使用指南
- [Rank BM25 GitHub](https://github.com/dorianbrown/rank_bm25) - BM25 实现
- [LangChain Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) - LangChain 混合检索

### 技术文章
- [Hybrid Search RAG That Actually Works](https://pub.towardsai.net/hybrid-search-rag-that-actually-works-bm25-vectors-reranking-in-python-0c02ade0799d) - 2026年混合检索实践
- [Reciprocal Rank Fusion Explained](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a) - RRF 算法详解
- [RAG at Scale](https://redis.io/blog/rag-at-scale) - 2026年生产级 RAG 架构

### 代码示例
- [MS MARCO Cross-Encoders](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) - Hugging Face 模型
- [OpenSearch RRF](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search) - OpenSearch RRF 实现

---

**版本:** v1.0 (2026年标准)
**最后更新:** 2026-02-16
**代码测试:** Python 3.13 + rank-bm25 0.2.x + sentence-transformers 3.x
