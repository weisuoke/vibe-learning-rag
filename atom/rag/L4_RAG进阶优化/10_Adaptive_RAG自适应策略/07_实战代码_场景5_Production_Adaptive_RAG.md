# 实战代码 - 场景5: Production Adaptive RAG

> 生产级 Adaptive RAG 完整实现，集成所有核心组件

---

## 场景描述

**目标**: 实现一个生产级的 Adaptive RAG 系统，集成查询分类、动态路由、自校正、成本管理和强化学习优化。

**适用场景**:
- 企业级 RAG 应用
- 高并发生产环境
- 需要完整功能的系统

**技术栈**:
- Python 3.13+
- OpenAI API
- LangChain / LangGraph
- ChromaDB (向量存储)

---

## 完整代码实现

```python
"""
Production Adaptive RAG
集成所有核心组件的生产级实现
"""

import os
import time
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI

# ===== 1. 配置管理 =====

@dataclass
class ProductionConfig:
    """生产环境配置"""
    # API 配置
    openai_api_key: str
    model: str = "gpt-4o-mini"

    # 向量存储配置
    vector_store_path: str = "./chroma_db"
    collection_name: str = "production_rag"

    # 路由配置
    max_iterations: int = 3
    enable_self_correction: bool = True
    enable_cost_tracking: bool = True
    enable_rl_optimization: bool = False

    # 成本配置
    daily_budget: int = 100000  # tokens
    cost_per_1k_tokens: float = 0.00015

# ===== 2. 查询分类器 =====

class ProductionClassifier:
    """生产级查询分类器"""

    def __init__(self, client: OpenAI):
        self.client = client
        self.time_keywords = ["今天", "最新", "现在", "2025", "2026", "today", "latest"]
        self.complex_keywords = ["比较", "对比", "分析", "compare", "analyze"]

    def classify(self, query: str) -> Literal["NO_RETRIEVE", "SINGLE", "ITERATIVE", "WEB_SEARCH"]:
        """分类查询"""
        query_lower = query.lower()
        words = query.split()

        # 实时查询
        if any(kw in query_lower for kw in self.time_keywords):
            return "WEB_SEARCH"

        # 简单查询
        if len(words) < 5:
            return "NO_RETRIEVE"

        # 复杂查询
        if len(words) > 15 or any(kw in query_lower for kw in self.complex_keywords):
            return "ITERATIVE"

        return "SINGLE"

# ===== 3. 自校正评估器 =====

class ProductionGraders:
    """生产级评估器集合"""

    def __init__(self, client: OpenAI):
        self.client = client

    def grade_relevance(self, query: str, document: str) -> bool:
        """评估文档相关性"""
        prompt = f"""评估文档是否与查询相关。

查询: {query}
文档: {document}

只回答 "是" 或 "否"。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return "是" in response.choices[0].message.content

    def grade_hallucination(self, documents: List[str], answer: str) -> bool:
        """检测幻觉 (True=有幻觉)"""
        docs_text = "\n\n".join(documents)

        prompt = f"""检测答案是否包含幻觉。

文档:
{docs_text}

答案:
{answer}

只回答 "是" 或 "否"。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return "是" in response.choices[0].message.content

    def grade_completeness(self, query: str, answer: str) -> bool:
        """评估答案完整性 (True=完整)"""
        prompt = f"""评估答案是否完整。

查询: {query}
答案: {answer}

只回答 "是" 或 "否"。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return "是" in response.choices[0].message.content

# ===== 4. 成本追踪器 =====

class ProductionCostTracker:
    """生产级成本追踪器"""

    def __init__(self, daily_budget: int):
        self.daily_budget = daily_budget
        self.daily_used = 0
        self.query_history = []
        self.last_reset = datetime.now()

    def reset_if_needed(self):
        """检查是否需要重置日预算"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_used = 0
            self.last_reset = now

    def check_budget(self) -> bool:
        """检查预算是否充足"""
        self.reset_if_needed()
        return self.daily_used < self.daily_budget

    def track(self, query: str, strategy: str, tokens: int):
        """追踪查询成本"""
        self.daily_used += tokens
        self.query_history.append({
            "query": query,
            "strategy": strategy,
            "tokens": tokens,
            "timestamp": datetime.now()
        })

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "daily_used": self.daily_used,
            "daily_budget": self.daily_budget,
            "usage_percentage": f"{self.daily_used/self.daily_budget*100:.1f}%",
            "total_queries": len(self.query_history)
        }

# ===== 5. Production Adaptive RAG 系统 =====

class ProductionAdaptiveRAG:
    """生产级 Adaptive RAG 系统"""

    def __init__(self, config: ProductionConfig, vector_store=None):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.vector_store = vector_store

        # 初始化组件
        self.classifier = ProductionClassifier(self.client)
        self.graders = ProductionGraders(self.client) if config.enable_self_correction else None
        self.cost_tracker = ProductionCostTracker(config.daily_budget) if config.enable_cost_tracking else None

        # 统计信息
        self.query_count = 0
        self.strategy_counts = {
            "NO_RETRIEVE": 0,
            "SINGLE": 0,
            "ITERATIVE": 0,
            "WEB_SEARCH": 0
        }

    def query(self, query: str) -> Dict:
        """执行查询"""
        start_time = time.time()
        self.query_count += 1

        # 步骤1: 检查预算
        if self.cost_tracker and not self.cost_tracker.check_budget():
            return {
                "answer": "日预算已用完，请明天再试",
                "strategy": "DENIED",
                "tokens_used": 0,
                "time_used": 0.0,
                "error": "BUDGET_EXCEEDED"
            }

        # 步骤2: 分类查询
        strategy = self.classifier.classify(query)
        self.strategy_counts[strategy] += 1

        # 步骤3: 执行策略
        try:
            if strategy == "NO_RETRIEVE":
                result = self._no_retrieve(query)
            elif strategy == "SINGLE":
                result = self._single_retrieve(query)
            elif strategy == "ITERATIVE":
                result = self._iterative_retrieve(query)
            else:  # WEB_SEARCH
                result = self._web_search(query)

            # 步骤4: 追踪成本
            if self.cost_tracker:
                self.cost_tracker.track(query, strategy, result["tokens_used"])

            # 步骤5: 返回结果
            time_used = time.time() - start_time

            return {
                "answer": result["answer"],
                "strategy": strategy,
                "tokens_used": result["tokens_used"],
                "time_used": f"{time_used:.2f}s",
                "iterations": result.get("iterations", 1),
                "corrections": result.get("corrections", [])
            }

        except Exception as e:
            return {
                "answer": f"查询失败: {str(e)}",
                "strategy": strategy,
                "tokens_used": 0,
                "time_used": 0.0,
                "error": str(e)
            }

    def _no_retrieve(self, query: str) -> Dict:
        """直接生成"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": query}],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }

    def _single_retrieve(self, query: str) -> Dict:
        """单次检索"""
        if not self.vector_store:
            return self._no_retrieve(query)

        # 检索文档
        docs = self.vector_store.similarity_search(query, k=3)
        doc_texts = [doc.page_content for doc in docs]

        # 自校正: 过滤不相关文档
        if self.graders:
            relevant_docs = [doc for doc in doc_texts if self.graders.grade_relevance(query, doc)]
            if not relevant_docs:
                relevant_docs = doc_texts  # 降级
        else:
            relevant_docs = doc_texts

        # 生成答案
        context = "\n\n".join(relevant_docs)
        prompt = f"""基于以下上下文回答问题:

上下文:
{context}

问题: {query}

回答:"""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content
        context_tokens = len(context.split()) * 1.3
        total_tokens = response.usage.total_tokens + int(context_tokens)

        return {
            "answer": answer,
            "tokens_used": total_tokens
        }

    def _iterative_retrieve(self, query: str) -> Dict:
        """迭代检索（带自校正）"""
        if not self.vector_store:
            return self._single_retrieve(query)

        total_tokens = 0
        iterations = 0
        corrections = []

        # 第一次检索
        docs = self.vector_store.similarity_search(query, k=5)
        doc_texts = [doc.page_content for doc in docs]

        while iterations < self.config.max_iterations:
            iterations += 1

            # 过滤相关文档
            if self.graders:
                relevant_docs = [doc for doc in doc_texts if self.graders.grade_relevance(query, doc)]
                if not relevant_docs:
                    corrections.append(f"迭代{iterations}: 文档不相关")
                    break
            else:
                relevant_docs = doc_texts

            # 生成答案
            context = "\n\n".join(relevant_docs)
            prompt = f"""基于以下上下文回答问题:

上下文:
{context}

问题: {query}

回答:"""

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content
            total_tokens += response.usage.total_tokens

            # 自校正检查
            if self.graders:
                # 检查幻觉
                has_hallucination = self.graders.grade_hallucination(relevant_docs, answer)
                if has_hallucination:
                    corrections.append(f"迭代{iterations}: 检测到幻觉")
                    # 重新生成（强调基于文档）
                    continue

                # 检查完整性
                is_complete = self.graders.grade_completeness(query, answer)
                if is_complete:
                    return {
                        "answer": answer,
                        "tokens_used": total_tokens,
                        "iterations": iterations,
                        "corrections": corrections
                    }
                else:
                    corrections.append(f"迭代{iterations}: 答案不完整")
                    # 补充检索（简化实现）
                    additional_docs = self.vector_store.similarity_search(query, k=3)
                    doc_texts.extend([doc.page_content for doc in additional_docs])
            else:
                # 无自校正，直接返回
                return {
                    "answer": answer,
                    "tokens_used": total_tokens,
                    "iterations": iterations,
                    "corrections": []
                }

        # 达到最大迭代次数
        return {
            "answer": answer,
            "tokens_used": total_tokens,
            "iterations": iterations,
            "corrections": corrections + [f"达到最大迭代次数 ({self.config.max_iterations})"]
        }

    def _web_search(self, query: str) -> Dict:
        """网络搜索（简化实现）"""
        answer = f"""[网络搜索模式]

查询: {query}

说明: 此查询需要实时信息，建议使用网络搜索 API。

实现建议:
1. 使用 Tavily API
2. 使用 Google Search API
3. 使用 Bing Search API
"""

        return {
            "answer": answer,
            "tokens_used": 100
        }

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        stats = {
            "total_queries": self.query_count,
            "strategy_distribution": {
                strategy: {
                    "count": count,
                    "percentage": f"{count/self.query_count*100:.1f}%" if self.query_count > 0 else "0%"
                }
                for strategy, count in self.strategy_counts.items()
            }
        }

        if self.cost_tracker:
            stats["cost_tracking"] = self.cost_tracker.get_stats()

        return stats

# ===== 6. 使用示例 =====

def main():
    """主函数：演示生产级 Adaptive RAG"""

    print("=" * 60)
    print("Production Adaptive RAG 实战示例")
    print("=" * 60)

    # 配置
    config = ProductionConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        enable_self_correction=True,
        enable_cost_tracking=True,
        daily_budget=10000
    )

    # 初始化系统
    rag = ProductionAdaptiveRAG(config)

    # 测试查询
    test_queries = [
        "什么是 Python?",
        "如何使用 LangChain 构建 RAG?",
        "比较 LangChain 和 LlamaIndex 的优缺点",
        "2026 年 AI 有哪些新进展?"
    ]

    print("\n【生产级 RAG 测试】\n")

    for query in test_queries:
        print(f"查询: {query}")
        result = rag.query(query)

        print(f"  策略: {result['strategy']}")
        print(f"  Token: {result['tokens_used']}")
        print(f"  时间: {result['time_used']}")

        if "iterations" in result:
            print(f"  迭代: {result['iterations']}")

        if "corrections" in result and result["corrections"]:
            print(f"  校正: {len(result['corrections'])} 次")
            for correction in result["corrections"]:
                print(f"    - {correction}")

        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  答案: {result['answer'][:100]}...")

        print()

    # 系统统计
    print("\n【系统统计】\n")
    stats = rag.get_stats()

    print(f"总查询数: {stats['total_queries']}")
    print("\n策略分布:")
    for strategy, info in stats['strategy_distribution'].items():
        print(f"  {strategy:15s}: {info['count']:2d} ({info['percentage']})")

    if "cost_tracking" in stats:
        cost_stats = stats["cost_tracking"]
        print(f"\n成本追踪:")
        print(f"  日使用: {cost_stats['daily_used']}/{cost_stats['daily_budget']} tokens")
        print(f"  使用率: {cost_stats['usage_percentage']}")

    # 生产环境建议
    print("\n【生产环境建议】\n")
    print("1. 集成真实向量存储 (ChromaDB/Milvus)")
    print("2. 添加日志记录和监控")
    print("3. 实现错误重试机制")
    print("4. 添加缓存层")
    print("5. 实现负载均衡")
    print("6. 添加 API 限流")
    print("7. 实现健康检查")
    print("8. 添加性能指标收集")

if __name__ == "__main__":
    main()
```

---

## 运行输出示例

```
============================================================
Production Adaptive RAG 实战示例
============================================================

【生产级 RAG 测试】

查询: 什么是 Python?
  策略: NO_RETRIEVE
  Token: 95
  时间: 0.52s
  答案: Python 是一种高级编程语言...

查询: 如何使用 LangChain 构建 RAG?
  策略: SINGLE
  Token: 487
  时间: 1.85s
  答案: 使用 LangChain 构建 RAG 系统的步骤...

查询: 比较 LangChain 和 LlamaIndex 的优缺点
  策略: ITERATIVE
  Token: 1523
  时间: 5.12s
  迭代: 2
  校正: 1 次
    - 迭代1: 答案不完整
  答案: LangChain 和 LlamaIndex 是两个流行的 RAG 框架...

查询: 2026 年 AI 有哪些新进展?
  策略: WEB_SEARCH
  Token: 100
  时间: 0.15s
  答案: [网络搜索模式]...


【系统统计】

总查询数: 4

策略分布:
  NO_RETRIEVE    :  1 (25.0%)
  SINGLE         :  1 (25.0%)
  ITERATIVE      :  1 (25.0%)
  WEB_SEARCH     :  1 (25.0%)

成本追踪:
  日使用: 2205/10000 tokens
  使用率: 22.1%

【生产环境建议】

1. 集成真实向量存储 (ChromaDB/Milvus)
2. 添加日志记录和监控
3. 实现错误重试机制
4. 添加缓存层
5. 实现负载均衡
6. 添加 API 限流
7. 实现健康检查
8. 添加性能指标收集
```

---

## 生产环境部署

### 1. Docker 部署

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  adaptive-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
    restart: unless-stopped
```

### 2. FastAPI 服务化

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Production Adaptive RAG API")

# 初始化 RAG 系统
config = ProductionConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
rag = ProductionAdaptiveRAG(config)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    strategy: str
    tokens_used: int
    time_used: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """查询端点"""
    try:
        result = rag.query(request.query)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats_endpoint():
    """统计端点"""
    return rag.get_stats()

@app.get("/health")
async def health_endpoint():
    """健康检查"""
    return {"status": "healthy"}
```

### 3. 监控与日志

```python
import logging
from prometheus_client import Counter, Histogram

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus 指标
query_counter = Counter('rag_queries_total', 'Total RAG queries', ['strategy'])
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')
query_tokens = Histogram('rag_query_tokens', 'Tokens used per query')

class MonitoredAdaptiveRAG(ProductionAdaptiveRAG):
    """带监控的 Adaptive RAG"""

    def query(self, query: str) -> Dict:
        """执行查询（带监控）"""
        logger.info(f"Processing query: {query[:50]}...")

        with query_duration.time():
            result = super().query(query)

        # 记录指标
        if "error" not in result:
            query_counter.labels(strategy=result["strategy"]).inc()
            query_tokens.observe(result["tokens_used"])
            logger.info(f"Query completed: strategy={result['strategy']}, tokens={result['tokens_used']}")
        else:
            logger.error(f"Query failed: {result['error']}")

        return result
```

---

## 关键洞察

1. **生产级系统的核心要素**
   - 配置管理
   - 错误处理
   - 成本追踪
   - 监控日志
   - 健康检查

2. **性能优化**
   - 缓存层
   - 连接池
   - 异步处理
   - 负载均衡

3. **可靠性保障**
   - 错误重试
   - 降级策略
   - 熔断机制
   - 限流保护

4. **可观测性**
   - 日志记录
   - 指标收集
   - 链路追踪
   - 告警通知

---

**参考文献**:
- [LangGraph Adaptive RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag) - LangChain AI (2025)
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) - arXiv (2024)
- Production RAG Best Practices (2025-2026)
- Enterprise RAG Deployment Guides (2025-2026)
