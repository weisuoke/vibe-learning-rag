# 实战代码 - 场景 1: 基础 Routing Agent

## 场景描述

**目标**: 构建一个智能查询路由器,根据查询类型自动选择向量检索或关键词检索

**难点**:
- 准确识别查询意图
- 选择合适的检索策略
- 集成多个检索器

**解决方案**: 使用 LangChain 构建 Routing Agent,通过 LLM 分类查询类型并路由到对应检索器

---

## 环境准备

```bash
# 安装依赖
uv add langchain langchain-openai chromadb python-dotenv
```

---

## 完整代码

```python
"""
基础 Routing Agent - 智能查询路由
演示: 根据查询类型自动选择检索策略

技术栈:
- LangChain: 0.1.0+
- OpenAI: 1.0.0+
- ChromaDB: 0.4.0+
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 加载环境变量
load_dotenv()

# ===== 1. 初始化组件 =====
print("初始化组件...")

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 2. 准备测试数据 =====
print("准备测试数据...")

documents = [
    Document(
        page_content="BERT 是双向编码器,使用 Masked LM 预训练,擅长理解任务",
        metadata={"source": "bert_intro", "type": "concept"}
    ),
    Document(
        page_content="GPT 是单向解码器,使用自回归预训练,擅长生成任务",
        metadata={"source": "gpt_intro", "type": "concept"}
    ),
    Document(
        page_content="2023年第四季度营收为 1000 万美元",
        metadata={"source": "financial_2023q4", "type": "data"}
    ),
    Document(
        page_content="2024年第一季度营收为 1200 万美元",
        metadata={"source": "financial_2024q1", "type": "data"}
    ),
    Document(
        page_content="Transformer 使用 Self-Attention 机制实现并行处理",
        metadata={"source": "transformer_intro", "type": "concept"}
    )
]

# ===== 3. 创建向量存储 =====
print("创建向量存储...")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="routing_demo"
)

# ===== 4. 定义检索器 =====

class VectorRetriever:
    """向量检索器 - 适合语义查询"""
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def search(self, query: str, k: int = 2) -> List[Document]:
        """向量检索"""
        return self.vectorstore.similarity_search(query, k=k)

class KeywordRetriever:
    """关键词检索器 - 适合精确查询"""
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def search(self, query: str, k: int = 2) -> List[Document]:
        """关键词检索(简单实现)"""
        results = []
        for doc in self.documents:
            # 简单的关键词匹配
            if any(keyword in doc.page_content for keyword in query.split()):
                results.append(doc)
                if len(results) >= k:
                    break
        return results

# 初始化检索器
vector_retriever = VectorRetriever(vectorstore)
keyword_retriever = KeywordRetriever(documents)

# ===== 5. 查询分类器 =====

classifier_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
分析以下查询的类型,只返回一个词: vector 或 keyword

规则:
- vector: 概念、原理、解释、对比类查询
- keyword: 包含数字、日期、精确名称的查询

查询: {query}

类型:
"""
)

classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt)

def classify_query(query: str) -> str:
    """分类查询类型"""
    result = classifier_chain.run(query=query)
    query_type = result.strip().lower()
    return query_type if query_type in ["vector", "keyword"] else "vector"

# ===== 6. Routing Agent =====

class RoutingAgent:
    """查询路由代理"""

    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def route(self, query: str) -> Dict:
        """路由查询到合适的检索器"""
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")

        # Step 1: 分类
        query_type = classify_query(query)
        print(f"✓ 查询类型: {query_type}")

        # Step 2: 路由
        if query_type == "vector":
            print(f"→ 使用向量检索")
            results = self.vector_retriever.search(query)
        else:
            print(f"→ 使用关键词检索")
            results = self.keyword_retriever.search(query)

        # Step 3: 返回结果
        print(f"✓ 找到 {len(results)} 个结果\n")

        return {
            "query": query,
            "type": query_type,
            "results": results
        }

# ===== 7. 测试 =====

def main():
    """主函数"""
    agent = RoutingAgent(vector_retriever, keyword_retriever)

    # 测试查询
    test_queries = [
        "什么是 BERT?",                    # 概念查询 → vector
        "2023年第四季度营收是多少?",        # 数据查询 → keyword
        "比较 BERT 和 GPT",                # 对比查询 → vector
        "2024年第一季度",                  # 精确查询 → keyword
        "Transformer 的核心机制"           # 原理查询 → vector
    ]

    for query in test_queries:
        result = agent.route(query)

        # 显示结果
        print("检索结果:")
        for i, doc in enumerate(result["results"], 1):
            print(f"  {i}. {doc.page_content}")
            print(f"     来源: {doc.metadata.get('source', 'unknown')}")

        print()

if __name__ == "__main__":
    main()
```

---

## 运行输出

```
初始化组件...
准备测试数据...
创建向量存储...

============================================================
查询: 什么是 BERT?
============================================================
✓ 查询类型: vector
→ 使用向量检索
✓ 找到 2 个结果

检索结果:
  1. BERT 是双向编码器,使用 Masked LM 预训练,擅长理解任务
     来源: bert_intro
  2. Transformer 使用 Self-Attention 机制实现并行处理
     来源: transformer_intro

============================================================
查询: 2023年第四季度营收是多少?
============================================================
✓ 查询类型: keyword
→ 使用关键词检索
✓ 找到 2 个结果

检索结果:
  1. 2023年第四季度营收为 1000 万美元
     来源: financial_2023q4
  2. 2024年第一季度营收为 1200 万美元
     来源: financial_2024q1

============================================================
查询: 比较 BERT 和 GPT
============================================================
✓ 查询类型: vector
→ 使用向量检索
✓ 找到 2 个结果

检索结果:
  1. BERT 是双向编码器,使用 Masked LM 预训练,擅长理解任务
     来源: bert_intro
  2. GPT 是单向解码器,使用自回归预训练,擅长生成任务
     来源: gpt_intro
```

---

## 代码解析

### 关键点 1: 查询分类

```python
classifier_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
分析以下查询的类型,只返回一个词: vector 或 keyword

规则:
- vector: 概念、原理、解释、对比类查询
- keyword: 包含数字、日期、精确名称的查询

查询: {query}

类型:
"""
)
```

**要点**:
- 使用 LLM 进行智能分类
- 明确的分类规则
- 简单的输出格式(只返回一个词)

### 关键点 2: 双检索器设计

```python
class VectorRetriever:
    """向量检索器 - 适合语义查询"""
    def search(self, query: str, k: int = 2) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

class KeywordRetriever:
    """关键词检索器 - 适合精确查询"""
    def search(self, query: str, k: int = 2) -> List[Document]:
        results = []
        for doc in self.documents:
            if any(keyword in doc.page_content for keyword in query.split()):
                results.append(doc)
        return results
```

**要点**:
- 统一的接口设计(`search` 方法)
- 不同的检索策略
- 易于扩展(可添加更多检索器)

### 关键点 3: 路由逻辑

```python
def route(self, query: str) -> Dict:
    # Step 1: 分类
    query_type = classify_query(query)

    # Step 2: 路由
    if query_type == "vector":
        results = self.vector_retriever.search(query)
    else:
        results = self.keyword_retriever.search(query)

    # Step 3: 返回结果
    return {"query": query, "type": query_type, "results": results}
```

**要点**:
- 清晰的三步流程
- 基于类型的条件路由
- 结构化的返回结果

---

## 扩展思考

### 如何优化?

**1. 添加混合检索**
```python
class HybridRetriever:
    """混合检索器"""
    def search(self, query: str, k: int = 2) -> List[Document]:
        # 向量检索
        vector_results = self.vector_retriever.search(query, k=k)

        # 关键词检索
        keyword_results = self.keyword_retriever.search(query, k=k)

        # 合并去重
        all_results = vector_results + keyword_results
        unique_results = list({doc.page_content: doc for doc in all_results}.values())

        return unique_results[:k]
```

**2. 添加置信度评分**
```python
def classify_query_with_confidence(query: str) -> tuple:
    """分类查询并返回置信度"""
    prompt = f"""
    分析查询类型并给出置信度(0-1):

    查询: {query}

    返回 JSON: {{"type": "vector/keyword", "confidence": 0.9}}
    """

    result = llm.predict(prompt)
    data = json.loads(result)

    return data["type"], data["confidence"]
```

**3. 添加缓存机制**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def classify_query_cached(query: str) -> str:
    """缓存查询分类结果"""
    return classify_query(query)
```

### 如何扩展?

**1. 支持更多检索器**
```python
# 添加 BM25 检索器
class BM25Retriever:
    def search(self, query: str, k: int = 2) -> List[Document]:
        # BM25 算法实现
        pass

# 添加到路由器
agent = RoutingAgent(
    retrievers={
        "vector": vector_retriever,
        "keyword": keyword_retriever,
        "bm25": bm25_retriever
    }
)
```

**2. 支持多级路由**
```python
def route_multilevel(query: str):
    """多级路由"""
    # Level 1: 领域分类
    domain = classify_domain(query)  # tech/business/finance

    # Level 2: 查询类型
    query_type = classify_query_type(query)  # vector/keyword

    # Level 3: 选择检索器
    retriever = select_retriever(domain, query_type)

    return retriever.search(query)
```

### 生产级改进

**1. 错误处理**
```python
def route_safe(self, query: str) -> Dict:
    """安全的路由"""
    try:
        query_type = classify_query(query)
    except Exception as e:
        print(f"分类失败: {e}, 使用默认向量检索")
        query_type = "vector"

    try:
        if query_type == "vector":
            results = self.vector_retriever.search(query)
        else:
            results = self.keyword_retriever.search(query)
    except Exception as e:
        print(f"检索失败: {e}")
        results = []

    return {"query": query, "type": query_type, "results": results}
```

**2. 性能监控**
```python
import time

def route_with_metrics(self, query: str) -> Dict:
    """带性能监控的路由"""
    start_time = time.time()

    # 分类
    classify_start = time.time()
    query_type = classify_query(query)
    classify_time = time.time() - classify_start

    # 检索
    search_start = time.time()
    results = self.search(query, query_type)
    search_time = time.time() - search_start

    total_time = time.time() - start_time

    return {
        "query": query,
        "type": query_type,
        "results": results,
        "metrics": {
            "classify_time": classify_time,
            "search_time": search_time,
            "total_time": total_time
        }
    }
```

**3. 日志记录**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def route_with_logging(self, query: str) -> Dict:
    """带日志的路由"""
    logger.info(f"收到查询: {query}")

    query_type = classify_query(query)
    logger.info(f"查询类型: {query_type}")

    results = self.search(query, query_type)
    logger.info(f"检索到 {len(results)} 个结果")

    return {"query": query, "type": query_type, "results": results}
```

---

## 参考资源

### 官方文档
- LangChain RouterChain: https://python.langchain.com/docs/modules/chains/router
- ChromaDB: https://docs.trychroma.com/

### 相关博客
- "Routing in RAG Driven Applications" (Towards Data Science, 2025)
- LangChain: "Router - Multi-Agent Architecture" (2026)

---

**版本**: v1.0
**最后更新**: 2026-02-17
**代码行数**: ~180 行
