# 核心概念 1: Routing Agent 查询路由代理

## 一句话定义

**Routing Agent 是智能查询分类器，根据查询意图自动选择最优检索策略（向量/关键词/混合/工具调用），在 Agentic RAG 中作为第一道智能决策层。**

---

## 详细解释

### 什么是 Routing Agent?

Routing Agent 是 Agentic RAG 系统的"交通指挥官"，负责：
- **意图识别**：分析用户查询的类型和需求
- **策略选择**：决定使用哪种检索方法
- **路由分发**：将查询发送到合适的处理器

**核心价值**：避免"一刀切"的检索策略，根据查询特点动态选择最优方案。

### 为什么需要 Routing Agent?

传统 RAG 的问题：
```python
# 传统 RAG：所有查询都用向量检索
query = "什么是 Transformer?"  # 概念查询
query = "2023年Q4营收是多少?"  # 精确查询
query = "比较 BERT 和 GPT"     # 对比查询

# 都用同一种方法 → 效果参差不齐
results = vector_search(query)
```

**不同查询需要不同策略**：
- **概念查询** → 向量检索（语义理解）
- **精确查询** → 关键词检索（精确匹配）
- **对比查询** → 混合检索（多角度）
- **计算查询** → 工具调用（外部 API）

### Routing Agent 如何工作?

**工作流程**：
```
用户查询
    ↓
[意图分类器]
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│向量检索 │关键词   │混合检索 │工具调用 │
└─────────┴─────────┴─────────┴─────────┘
    ↓
返回结果
```

**关键技术**：
1. **LLM 分类**：用 LLM 判断查询类型
2. **规则路由**：基于关键词/模式匹配
3. **语义路由**：基于 Embedding 相似度
4. **混合路由**：结合多种方法

---

## 核心原理

### 原理图解

```
┌─────────────────────────────────────────┐
│         Routing Agent 架构              │
├─────────────────────────────────────────┤
│                                         │
│  用户查询: "2023年营收是多少?"          │
│       ↓                                 │
│  [意图分析层]                           │
│   - LLM 分类                            │
│   - 关键词匹配                          │
│   - Embedding 相似度                    │
│       ↓                                 │
│  [决策层]                               │
│   判断: 精确查询 → 关键词检索           │
│       ↓                                 │
│  [执行层]                               │
│   调用: KeywordRetriever                │
│       ↓                                 │
│  返回: 精确匹配的财报数据               │
│                                         │
└─────────────────────────────────────────┘
```

### 工作流程

**Step 1: 查询分析**
```python
def analyze_query(query: str) -> QueryType:
    """分析查询类型"""
    prompt = f"""
    分析以下查询的类型：
    查询: {query}

    类型选项:
    - semantic: 概念/语义查询
    - keyword: 精确/关键词查询
    - hybrid: 需要多种检索
    - tool: 需要外部工具

    返回类型:
    """
    return llm.predict(prompt)
```

**Step 2: 路由决策**
```python
def route_query(query: str, query_type: QueryType):
    """根据类型路由查询"""
    if query_type == "semantic":
        return vector_retriever.search(query)
    elif query_type == "keyword":
        return keyword_retriever.search(query)
    elif query_type == "hybrid":
        return hybrid_retriever.search(query)
    elif query_type == "tool":
        return tool_executor.run(query)
```

**Step 3: 结果返回**
```python
def routing_agent(query: str):
    """完整路由流程"""
    query_type = analyze_query(query)
    results = route_query(query, query_type)
    return results
```

### 关键技术

**1. LLM 路由（最灵活）**
```python
from langchain.chains.router import LLMRouterChain

router_prompt = """
给定用户查询，选择最合适的检索方法：

查询: {input}

选项:
- vector: 语义/概念查询
- keyword: 精确/关键词查询
- hybrid: 复杂查询

选择:
"""

router = LLMRouterChain.from_llm(llm, router_prompt)
```

**2. 语义路由（最快）**
```python
from langchain.chains.router import EmbeddingRouterChain

# 预定义路由规则
routes = [
    {
        "name": "vector",
        "description": "概念解释、原理说明、技术对比"
    },
    {
        "name": "keyword",
        "description": "数字查询、日期查询、精确匹配"
    }
]

router = EmbeddingRouterChain.from_names_and_descriptions(
    routes, embeddings
)
```

**3. 规则路由（最可控）**
```python
import re

def rule_based_router(query: str):
    """基于规则的路由"""
    # 数字查询 → 关键词
    if re.search(r'\d{4}年|\d+月', query):
        return "keyword"

    # 对比查询 → 混合
    if any(word in query for word in ["比较", "对比", "区别"]):
        return "hybrid"

    # 默认 → 向量
    return "vector"
```

---

## 手写实现

```python
"""
Routing Agent 从零实现
演示：智能查询路由系统
"""

from typing import Literal, List, Dict
from openai import OpenAI
import os

# ===== 1. 初始化 =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QueryType = Literal["vector", "keyword", "hybrid", "tool"]

# ===== 2. 意图分类器 =====
def classify_query(query: str) -> QueryType:
    """使用 LLM 分类查询意图"""
    prompt = f"""
    分析查询类型，只返回一个词：vector/keyword/hybrid/tool

    规则：
    - vector: 概念、原理、解释类查询
    - keyword: 包含数字、日期、精确名称
    - hybrid: 需要多角度分析
    - tool: 需要计算、实时数据

    查询: {query}
    类型:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()
    return result if result in ["vector", "keyword", "hybrid", "tool"] else "vector"

# ===== 3. 模拟检索器 =====
class VectorRetriever:
    """向量检索器（模拟）"""
    def search(self, query: str) -> List[str]:
        return [f"[向量检索] {query} 的语义相关文档"]

class KeywordRetriever:
    """关键词检索器（模拟）"""
    def search(self, query: str) -> List[str]:
        return [f"[关键词检索] 精确匹配 '{query}' 的文档"]

class HybridRetriever:
    """混合检索器（模拟）"""
    def search(self, query: str) -> List[str]:
        return [f"[混合检索] 向量+关键词检索 '{query}'"]

class ToolExecutor:
    """工具执行器（模拟）"""
    def run(self, query: str) -> List[str]:
        return [f"[工具调用] 执行外部工具处理 '{query}'"]

# ===== 4. Routing Agent =====
class RoutingAgent:
    """查询路由代理"""

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.hybrid_retriever = HybridRetriever()
        self.tool_executor = ToolExecutor()

    def route(self, query: str) -> Dict:
        """路由查询到合适的检索器"""
        # Step 1: 分类
        query_type = classify_query(query)
        print(f"查询类型: {query_type}")

        # Step 2: 路由
        if query_type == "vector":
            results = self.vector_retriever.search(query)
        elif query_type == "keyword":
            results = self.keyword_retriever.search(query)
        elif query_type == "hybrid":
            results = self.hybrid_retriever.search(query)
        else:  # tool
            results = self.tool_executor.run(query)

        return {
            "query": query,
            "type": query_type,
            "results": results
        }

# ===== 5. 测试 =====
if __name__ == "__main__":
    agent = RoutingAgent()

    test_queries = [
        "什么是 Transformer 架构?",           # vector
        "2023年Q4营收是多少?",                # keyword
        "比较 BERT 和 GPT 的优缺点",          # hybrid
        "计算 1000 个文档的平均长度"          # tool
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        result = agent.route(query)
        print(f"结果: {result['results']}")
```

---

## 在 RAG 中的应用

### 应用场景 1: 企业知识库

**问题**：不同类型的查询需要不同策略
- "公司愿景是什么?" → 向量检索（概念）
- "2023年员工数量" → 关键词检索（精确）
- "对比2022和2023年业绩" → 混合检索

**解决方案**：
```python
# 企业知识库路由
def enterprise_router(query: str):
    if "是什么" in query or "如何" in query:
        return vector_search(query)
    elif re.search(r'\d{4}年', query):
        return keyword_search(query)
    elif "对比" in query or "比较" in query:
        return hybrid_search(query)
```

### 应用场景 2: 技术文档问答

**问题**：技术查询类型多样
- "解释 API 原理" → 向量检索
- "API 错误码 404" → 关键词检索
- "API 性能优化方案" → 混合检索

**解决方案**：
```python
# 技术文档路由
def tech_doc_router(query: str):
    if "原理" in query or "解释" in query:
        return vector_search(query)
    elif "错误码" in query or re.search(r'\d{3}', query):
        return keyword_search(query)
    else:
        return hybrid_search(query)
```

### 应用场景 3: 客服系统

**问题**：客服查询需要快速准确
- "如何退款?" → 向量检索（流程）
- "订单号 12345 状态" → 工具调用（查询系统）
- "退款政策和流程" → 混合检索

**解决方案**：
```python
# 客服路由
def customer_service_router(query: str):
    if "订单号" in query:
        return call_order_api(query)
    elif "如何" in query or "流程" in query:
        return vector_search(query)
    else:
        return hybrid_search(query)
```

---

## 主流框架实现

### LangChain 实现

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# 定义不同的检索链
vector_chain = ConversationChain(
    llm=ChatOpenAI(),
    verbose=True
)

keyword_chain = ConversationChain(
    llm=ChatOpenAI(),
    verbose=True
)

# 定义路由规则
prompt_infos = [
    {
        "name": "vector",
        "description": "适合概念、原理、解释类查询",
        "prompt_template": "使用语义检索: {input}"
    },
    {
        "name": "keyword",
        "description": "适合精确、数字、日期类查询",
        "prompt_template": "使用关键词检索: {input}"
    }
]

# 创建路由链
router_chain = MultiPromptChain.from_prompts(
    llm=ChatOpenAI(),
    prompt_infos=prompt_infos,
    default_chain=vector_chain
)

# 使用
result = router_chain.run("什么是 Transformer?")
```

### LangGraph 实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    query_type: str
    results: list

def classify_node(state: AgentState):
    """分类节点"""
    query = state["query"]
    query_type = classify_query(query)
    return {"query_type": query_type}

def vector_node(state: AgentState):
    """向量检索节点"""
    results = vector_retriever.search(state["query"])
    return {"results": results}

def keyword_node(state: AgentState):
    """关键词检索节点"""
    results = keyword_retriever.search(state["query"])
    return {"results": results}

def route_decision(state: AgentState):
    """路由决策"""
    if state["query_type"] == "vector":
        return "vector"
    else:
        return "keyword"

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("vector", vector_node)
workflow.add_node("keyword", keyword_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    route_decision,
    {
        "vector": "vector",
        "keyword": "keyword"
    }
)
workflow.add_edge("vector", END)
workflow.add_edge("keyword", END)

app = workflow.compile()

# 使用
result = app.invoke({"query": "什么是 RAG?"})
```

### LlamaIndex 实现

```python
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# 创建不同的索引
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = SimpleKeywordTableIndex.from_documents(documents)

# 创建查询引擎
vector_engine = vector_index.as_query_engine()
keyword_engine = keyword_index.as_query_engine()

# 创建路由器
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        {
            "query_engine": vector_engine,
            "description": "适合概念和语义查询"
        },
        {
            "query_engine": keyword_engine,
            "description": "适合精确和关键词查询"
        }
    ]
)

# 使用
response = router_engine.query("什么是 Transformer?")
```

---

## 最佳实践（2025-2026）

### 性能优化

**1. 缓存分类结果**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def classify_query_cached(query: str) -> QueryType:
    """缓存查询分类结果"""
    return classify_query(query)
```

**2. 并行路由**
```python
import asyncio

async def parallel_route(query: str):
    """并行执行多个检索器，取最佳结果"""
    tasks = [
        vector_retriever.search_async(query),
        keyword_retriever.search_async(query)
    ]
    results = await asyncio.gather(*tasks)
    return merge_results(results)
```

**3. 快速路由规则**
```python
# 先用规则快速过滤，再用 LLM 精确分类
def fast_route(query: str):
    # 规则路由（快）
    if re.search(r'\d{4}年', query):
        return "keyword"

    # LLM 路由（慢但准）
    return classify_query(query)
```

### 成本控制

**1. 使用小模型分类**
```python
# 用 gpt-4o-mini 而非 gpt-4 进行分类
classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**2. 批量分类**
```python
def batch_classify(queries: List[str]) -> List[QueryType]:
    """批量分类降低 API 调用次数"""
    prompt = f"分类以下查询：\n" + "\n".join(queries)
    # 一次调用分类多个查询
    return llm.predict(prompt).split("\n")
```

### 错误处理

**1. 默认路由**
```python
def safe_route(query: str):
    """安全路由，失败时使用默认策略"""
    try:
        query_type = classify_query(query)
    except Exception as e:
        print(f"分类失败: {e}")
        query_type = "vector"  # 默认向量检索

    return route_query(query, query_type)
```

**2. 路由回退**
```python
def route_with_fallback(query: str):
    """路由失败时回退到混合检索"""
    try:
        return route_query(query, classify_query(query))
    except Exception:
        return hybrid_retriever.search(query)
```

---

## 常见问题

### 问题 1: 路由准确率低怎么办?

**原因**：
- LLM 分类不准确
- 路由规则不完善
- 查询类型模糊

**解决方案**：
```python
# 1. 使用更好的分类 Prompt
better_prompt = """
你是查询分类专家。分析查询类型：

查询: {query}

分类标准：
- vector: 包含"是什么"、"如何"、"原理"等概念词
- keyword: 包含数字、日期、专有名词
- hybrid: 包含"对比"、"比较"、多个实体
- tool: 需要计算、实时数据

思考过程：
1. 查询包含哪些关键词？
2. 用户想要什么类型的答案？
3. 哪种检索方法最合适？

类型:
"""

# 2. 添加置信度判断
def classify_with_confidence(query: str):
    result = llm.predict(better_prompt.format(query=query))
    if "不确定" in result:
        return "hybrid"  # 不确定时用混合检索
    return parse_type(result)
```

### 问题 2: 路由延迟太高怎么办?

**原因**：
- LLM 调用慢
- 每次查询都分类

**解决方案**：
```python
# 1. 使用规则优先
def fast_route(query: str):
    # 规则路由（<1ms）
    rule_type = rule_based_router(query)
    if rule_type != "unknown":
        return rule_type

    # LLM 路由（100-500ms）
    return classify_query(query)

# 2. 异步分类
async def async_route(query: str):
    query_type = await classify_query_async(query)
    return await route_query_async(query, query_type)
```

### 问题 3: 如何评估路由效果?

**评估指标**：
```python
def evaluate_router(test_cases: List[Dict]):
    """评估路由准确率"""
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        predicted = classify_query(case["query"])
        if predicted == case["expected_type"]:
            correct += 1

    accuracy = correct / total
    print(f"路由准确率: {accuracy:.2%}")
    return accuracy

# 测试用例
test_cases = [
    {"query": "什么是 RAG?", "expected_type": "vector"},
    {"query": "2023年营收", "expected_type": "keyword"},
    {"query": "对比 BERT 和 GPT", "expected_type": "hybrid"}
]

evaluate_router(test_cases)
```

---

## 参考资源

### 论文
- "Routing in RAG Driven Applications" (Towards Data Science, 2025)
- "Agentic RAG: A Survey" (arXiv 2501.09136, 2025)

### 博客
- LangChain: "Router - Multi-Agent Architecture" (2026)
  https://docs.langchain.com/oss/python/langchain/multi-agent/router
- "Part 2: Building an Agentic RAG Workflow with Query Router" (2025)
  https://sajalsharma.com/posts/agentic-rag-query-router-langgraph

### 框架文档
- LangChain RouterChain: https://python.langchain.com/docs/modules/chains/router
- LangGraph Conditional Edges: https://langchain-ai.github.io/langgraph/
- LlamaIndex RouterQueryEngine: https://docs.llamaindex.ai/en/stable/

---

**版本**: v1.0
**最后更新**: 2026-02-17
**字数**: ~450 行
