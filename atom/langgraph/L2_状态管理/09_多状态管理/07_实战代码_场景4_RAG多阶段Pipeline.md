# 实战代码 - 场景4：RAG 多阶段 Pipeline

## 场景说明

本场景演示如何用 LangGraph 的多状态管理构建一个**实际的 RAG 多阶段 Pipeline**。重点展示三种多状态技术的协作：

1. **Input/Output Schema 分离**——外部只传 `question`，只拿 `answer` + `sources`
2. **节点级状态分组**——每个节点声明自己只需要哪些字段（通过函数参数类型推断）
3. **内部私有状态**——`metadata`、`rewritten_query` 等中间字段对外不可见

Pipeline 流程：用户问题 → 查询改写 → 文档检索 → 重排序 → 生成回答

核心知识点：
1. 三层 Schema 协作——InputSchema / OutputSchema / OverallState 各司其职
2. 节点级 TypedDict——每个节点函数的参数类型决定它能看到哪些字段
3. Reducer 与覆盖模式混用——`documents` 用 `operator.add` 累积，其他字段用覆盖
4. 私有状态隐藏——`metadata`、`rewritten_query` 不出现在输出中
5. 模拟 RAG 全流程——不依赖 API Key，本地即可运行验证

[来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]
[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 图结构

```
         ┌──────────────┐
         │    START     │  ← 只接受 RAGInput (question)
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │   rewrite    │  ← 读 question → 写 rewritten_query
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │    search    │  ← 读 rewritten_query → 写 documents
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │     rank     │  ← 读 documents → 写 ranked_documents
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │   generate   │  ← 读 question + ranked_documents → 写 answer, sources
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │     END      │  ← 只输出 RAGOutput (answer, sources)
         └──────────────┘
```

关键点：
- START 只接受 `RAGInput` 的字段（`question`）
- END 只输出 `RAGOutput` 的字段（`answer`, `sources`）
- 中间的 `rewritten_query`、`documents`、`ranked_documents`、`metadata` 全部是内部私有状态

---

## 完整代码

```python
"""
LangGraph 多状态管理 - RAG 多阶段 Pipeline
演示：使用 Input/Output Schema 分离 + 节点级状态分组构建 RAG 系统

Pipeline: 用户问题 → 查询改写 → 文档检索 → 重排序 → 生成回答

核心技术：
- Input/Output Schema 分离（隐藏内部实现）
- 节点级状态分组（每个节点只看到需要的字段）
- Reducer 与覆盖模式混用
- 私有状态隐藏（metadata 不暴露给外部）

运行环境：Python 3.13+, langgraph
安装依赖：uv add langgraph
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator


# ============================================================
# 1. 定义多层 Schema
# ============================================================

# --- 外部接口 Schema ---

class RAGInput(TypedDict):
    """RAG 系统的输入接口

    外部调用者只需要传这一个字段。
    这就是"菜单"——顾客只需要说想吃什么。
    """
    question: str


class RAGOutput(TypedDict):
    """RAG 系统的输出接口

    外部调用者只能拿到这两个字段。
    这就是"出餐口"——顾客只拿到菜和小票。
    """
    answer: str
    sources: list[str]


# --- 内部完整状态 ---

class RAGState(TypedDict):
    """RAG 内部完整状态

    包含所有字段：输入字段 + 输出字段 + 中间状态字段。
    这就是"厨房"——所有食材、工具、半成品都在这里。

    字段分类：
    - question: 来自 RAGInput（输入）
    - answer, sources: 来自 RAGOutput（输出）
    - rewritten_query: 查询改写结果（内部私有）
    - documents: 检索到的文档（内部私有，使用 Reducer 累积）
    - ranked_documents: 重排序后的文档（内部私有）
    - metadata: 处理元数据（内部私有）
    """
    # 输入字段
    question: str

    # 输出字段
    answer: str
    sources: list[str]

    # 内部私有字段——对外不可见
    rewritten_query: str
    documents: Annotated[list[dict], operator.add]  # Reducer：累积追加
    ranked_documents: list[dict]
    metadata: dict


# --- 节点级 Schema（每个节点只看到需要的字段）---

class QueryState(TypedDict):
    """查询改写节点的输入视图：只需要原始问题"""
    question: str


class SearchState(TypedDict):
    """文档检索节点的输入视图：只需要改写后的查询"""
    rewritten_query: str


class RankState(TypedDict):
    """重排序节点的输入视图：只需要文档列表"""
    documents: list[dict]


class GenerateState(TypedDict):
    """生成节点的输入视图：需要问题和排序后的文档"""
    question: str
    ranked_documents: list[dict]


# ============================================================
# 2. 定义节点函数
# ============================================================

def rewrite_query(state: QueryState) -> dict:
    """查询改写节点

    通过函数参数类型 QueryState，LangGraph 自动推断：
    这个节点只需要 question 字段。

    职责：优化用户查询，提高检索召回率
    - 补充同义词和领域关键词
    - 模拟 LLM 改写（实际中调用 OpenAI API）

    读取：question（通过 QueryState 限定）
    写入：rewritten_query, metadata
    """
    query = state["question"]

    # 模拟 LLM 查询改写
    rewritten = f"{query} 原理 实现 最佳实践"

    print(f"[rewrite] '{query}' → '{rewritten}'")
    return {
        "rewritten_query": rewritten,
        "metadata": {"step": "rewrite", "original_length": len(query)},
    }


def search_documents(state: SearchState) -> dict:
    """文档检索节点

    通过函数参数类型 SearchState，LangGraph 自动推断：
    这个节点只需要 rewritten_query 字段。

    职责：根据改写后的查询检索相关文档
    - 模拟向量数据库检索（实际中调用 ChromaDB / Milvus）
    - 返回带分数的文档列表

    读取：rewritten_query（通过 SearchState 限定）
    写入：documents（使用 operator.add Reducer 累积）
    """
    query = state["rewritten_query"]

    # 模拟向量检索结果
    docs = [
        {
            "content": "向量数据库是专门存储和检索高维向量的数据库系统",
            "source": "vector_db_intro.md",
            "score": 0.95,
        },
        {
            "content": "常见的向量数据库包括 Milvus、Pinecone、Weaviate 等",
            "source": "vector_db_comparison.md",
            "score": 0.88,
        },
        {
            "content": "向量检索的核心是近似最近邻算法，如 HNSW 和 IVF",
            "source": "ann_algorithms.md",
            "score": 0.82,
        },
        {
            "content": "Python 是一种通用编程语言",
            "source": "python_intro.md",
            "score": 0.25,
        },
        {
            "content": "向量数据库在 RAG 系统中负责存储文档的 Embedding 表示",
            "source": "rag_architecture.md",
            "score": 0.91,
        },
    ]

    print(f"[search] 检索到 {len(docs)} 篇文档")
    return {"documents": docs}


def rank_documents(state: RankState) -> dict:
    """重排序节点

    通过函数参数类型 RankState，LangGraph 自动推断：
    这个节点只需要 documents 字段。

    职责：对检索结果进行重排序和过滤
    - 过滤低分文档（score < 0.5）
    - 按分数降序排列
    - 只保留 Top-3

    读取：documents（通过 RankState 限定）
    写入：ranked_documents
    """
    docs = state["documents"]

    # 过滤低分文档
    filtered = [d for d in docs if d["score"] >= 0.5]

    # 按分数降序排列，取 Top-3
    ranked = sorted(filtered, key=lambda d: d["score"], reverse=True)[:3]

    print(f"[rank] {len(docs)} 篇 → 过滤 → 排序 → Top-{len(ranked)}")
    for i, doc in enumerate(ranked):
        print(f"  #{i+1} [{doc['score']}] {doc['source']}")

    return {"ranked_documents": ranked}


def generate_answer(state: GenerateState) -> dict:
    """生成回答节点

    通过函数参数类型 GenerateState，LangGraph 自动推断：
    这个节点需要 question 和 ranked_documents 两个字段。

    职责：基于查询和排序后的文档生成最终答案
    - 组装上下文
    - 模拟 LLM 生成（实际中调用 OpenAI API）
    - 提取来源信息

    读取：question, ranked_documents（通过 GenerateState 限定）
    写入：answer, sources
    """
    question = state["question"]
    docs = state["ranked_documents"]

    # 组装上下文
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[{i}] {doc['content']}")
    context = "\n".join(context_parts)

    # 模拟 LLM 生成答案
    answer = (
        f"关于「{question}」：\n\n"
        f"向量数据库是专门用于存储和检索高维向量数据的数据库系统。"
        f"在 RAG 架构中，它负责存储文档的 Embedding 表示，"
        f"支持基于语义相似度的快速检索。"
        f"常见实现包括 Milvus、Pinecone、Weaviate 等，"
        f"底层使用 HNSW、IVF 等近似最近邻算法。\n\n"
        f"（基于 {len(docs)} 篇参考文档生成）"
    )

    # 提取来源
    sources = [doc["source"] for doc in docs]

    print(f"[generate] 基于 {len(docs)} 篇文档生成答案 ({len(answer)} chars)")
    return {"answer": answer, "sources": sources}


# ============================================================
# 3. 构建 RAG Pipeline
# ============================================================

builder = StateGraph(
    RAGState,                    # 内部完整状态
    input_schema=RAGInput,       # 外部输入接口
    output_schema=RAGOutput,     # 外部输出接口
)

# 添加节点——LangGraph 自动从函数参数推断每个节点的 input_schema
builder.add_node("rewrite", rewrite_query)       # 推断为 QueryState
builder.add_node("search", search_documents)     # 推断为 SearchState
builder.add_node("rank", rank_documents)         # 推断为 RankState
builder.add_node("generate", generate_answer)    # 推断为 GenerateState

# 线性串联
builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "search")
builder.add_edge("search", "rank")
builder.add_edge("rank", "generate")
builder.add_edge("generate", END)

# 编译
rag_pipeline = builder.compile()


# ============================================================
# 4. 测试1：invoke 模式——验证 Input/Output 隔离
# ============================================================

print("=" * 60)
print("=== 测试1：invoke 模式 - Input/Output 隔离验证 ===")
print("=" * 60)

# 外部只传 question（RAGInput 的字段）
result = rag_pipeline.invoke({"question": "什么是向量数据库？"})

print(f"\n--- 输出结果 ---")
print(f"result 的键: {list(result.keys())}")
print(f"\n回答:\n{result['answer']}")
print(f"\n来源: {result['sources']}")

# 关键验证：输出中没有内部私有字段！
print(f"\n--- 隔离验证 ---")
print(f"'answer' in result: {'answer' in result}")
print(f"'sources' in result: {'sources' in result}")
print(f"'metadata' in result: {'metadata' in result}")
print(f"'rewritten_query' in result: {'rewritten_query' in result}")
print(f"'documents' in result: {'documents' in result}")
print(f"'ranked_documents' in result: {'ranked_documents' in result}")


# ============================================================
# 5. 测试2：stream 模式——观察每个节点的状态变化
# ============================================================

print("\n" + "=" * 60)
print("=== 测试2：stream 模式 - 逐步观察状态变化 ===")
print("=" * 60)

for step in rag_pipeline.stream({"question": "RAG 的核心组件有哪些？"}):
    for node_name, node_output in step.items():
        print(f"\n--- 节点: {node_name} ---")
        print(f"  返回的键: {list(node_output.keys())}")
        for key, value in node_output.items():
            display = str(value)
            if len(display) > 80:
                display = display[:80] + "..."
            print(f"  {key}: {display}")


# ============================================================
# 6. 测试3：对比有无 Schema 分离的差异
# ============================================================

print("\n" + "=" * 60)
print("=== 测试3：对比有无 Schema 分离 ===")
print("=" * 60)

# 构建一个没有 Schema 分离的版本
builder_no_schema = StateGraph(RAGState)  # 不指定 input/output schema
builder_no_schema.add_node("rewrite", rewrite_query)
builder_no_schema.add_node("search", search_documents)
builder_no_schema.add_node("rank", rank_documents)
builder_no_schema.add_node("generate", generate_answer)
builder_no_schema.add_edge(START, "rewrite")
builder_no_schema.add_edge("rewrite", "search")
builder_no_schema.add_edge("search", "rank")
builder_no_schema.add_edge("rank", "generate")
builder_no_schema.add_edge("generate", END)
pipeline_no_schema = builder_no_schema.compile()

# 没有 Schema 分离时，需要传入更多初始值
result_no_schema = pipeline_no_schema.invoke({
    "question": "什么是向量数据库？",
    "answer": "",
    "sources": [],
    "rewritten_query": "",
    "documents": [],
    "ranked_documents": [],
    "metadata": {},
})

print(f"\n无 Schema 分离 - 输出的键: {list(result_no_schema.keys())}")
print(f"  包含 metadata: {'metadata' in result_no_schema}")
print(f"  包含 rewritten_query: {'rewritten_query' in result_no_schema}")
print(f"  包含 documents: {'documents' in result_no_schema}")
print(f"  → 所有内部状态都暴露了！")

print(f"\n有 Schema 分离 - 输出的键: {list(result.keys())}")
print(f"  包含 metadata: {'metadata' in result}")
print(f"  包含 rewritten_query: {'rewritten_query' in result}")
print(f"  包含 documents: {'documents' in result}")
print(f"  → 只有 answer 和 sources！干净！")


# ============================================================
# 7. 测试4：节点级状态分组验证
# ============================================================

print("\n" + "=" * 60)
print("=== 测试4：节点级状态分组验证 ===")
print("=" * 60)

print("""
每个节点函数的参数类型决定了它能看到哪些字段：

  rewrite_query(state: QueryState)
    → 只能看到: question
    → 看不到: documents, ranked_documents, answer...

  search_documents(state: SearchState)
    → 只能看到: rewritten_query
    → 看不到: question, documents, answer...

  rank_documents(state: RankState)
    → 只能看到: documents
    → 看不到: question, rewritten_query, answer...

  generate_answer(state: GenerateState)
    → 只能看到: question, ranked_documents
    → 看不到: rewritten_query, documents, metadata...

这就是「最小权限原则」在状态管理中的体现：
每个节点只拿到它需要的数据，不多不少。
""")

# 验证：打印每个节点级 Schema 的字段
from typing import get_type_hints

schemas = {
    "QueryState": QueryState,
    "SearchState": SearchState,
    "RankState": RankState,
    "GenerateState": GenerateState,
}

for name, schema in schemas.items():
    fields = list(get_type_hints(schema).keys())
    print(f"  {name}: {fields}")
```

---

## 预期输出

```
============================================================
=== 测试1：invoke 模式 - Input/Output 隔离验证 ===
============================================================
[rewrite] '什么是向量数据库？' → '什么是向量数据库？ 原理 实现 最佳实践'
[search] 检索到 5 篇文档
[rank] 5 篇 → 过滤 → 排序 → Top-3
  #1 [0.95] vector_db_intro.md
  #2 [0.91] rag_architecture.md
  #3 [0.88] vector_db_comparison.md
[generate] 基于 3 篇文档生成答案 (168 chars)

--- 输出结果 ---
result 的键: ['answer', 'sources']

回答:
关于「什么是向量数据库？」：

向量数据库是专门用于存储和检索高维向量数据的数据库系统。在 RAG 架构中，它负责存储文档的 Embedding 表示，支持基于语义相似度的快速检索。常见实现包括 Milvus、Pinecone、Weaviate 等，底层使用 HNSW、IVF 等近似最近邻算法。

（基于 3 篇参考文档生成）

来源: ['vector_db_intro.md', 'rag_architecture.md', 'vector_db_comparison.md']

--- 隔离验证 ---
'answer' in result: True
'sources' in result: True
'metadata' in result: False
'rewritten_query' in result: False
'documents' in result: False
'ranked_documents' in result: False

============================================================
=== 测试2：stream 模式 - 逐步观察状态变化 ===
============================================================
[rewrite] 'RAG 的核心组件有哪些？' → 'RAG 的核心组件有哪些？ 原理 实现 最佳实践'

--- 节点: rewrite ---
  返回的键: ['rewritten_query', 'metadata']
  rewritten_query: RAG 的核心组件有哪些？ 原理 实现 最佳实践
  metadata: {'step': 'rewrite', 'original_length': 12}
[search] 检索到 5 篇文档

--- 节点: search ---
  返回的键: ['documents']
  documents: [{'content': '向量数据库是专门存储和检索高维向量的数据库系统', 'source': 'vector_db_i...
[rank] 5 篇 → 过滤 → 排序 → Top-3
  #1 [0.95] vector_db_intro.md
  #2 [0.91] rag_architecture.md
  #3 [0.88] vector_db_comparison.md

--- 节点: rank ---
  返回的键: ['ranked_documents']
  ranked_documents: [{'content': '向量数据库是专门存储和检索高维向量的数据库系统', 'source': 'vect...
[generate] 基于 3 篇文档生成答案 (172 chars)

--- 节点: generate ---
  返回的键: ['answer', 'sources']
  answer: 关于「RAG 的核心组件有哪些？」：

向量数据库是专门用于存储和检索高维向量数据的数据库系统。在 RAG ...
  sources: ['vector_db_intro.md', 'rag_architecture.md', 'vector_db_comparison.md']

============================================================
=== 测试3：对比有无 Schema 分离 ===
============================================================
[rewrite] '什么是向量数据库？' → '什么是向量数据库？ 原理 实现 最佳实践'
[search] 检索到 5 篇文档
[rank] 5 篇 → 过滤 → 排序 → Top-3
  #1 [0.95] vector_db_intro.md
  #2 [0.91] rag_architecture.md
  #3 [0.88] vector_db_comparison.md
[generate] 基于 3 篇文档生成答案 (168 chars)

无 Schema 分离 - 输出的键: ['question', 'answer', 'sources', 'rewritten_query', 'documents', 'ranked_documents', 'metadata']
  包含 metadata: True
  包含 rewritten_query: True
  包含 documents: True
  → 所有内部状态都暴露了！

有 Schema 分离 - 输出的键: ['answer', 'sources']
  包含 metadata: False
  包含 rewritten_query: False
  包含 documents: False
  → 只有 answer 和 sources！干净！

============================================================
=== 测试4：节点级状态分组验证 ===
============================================================

每个节点函数的参数类型决定了它能看到哪些字段：

  rewrite_query(state: QueryState)
    → 只能看到: question
    → 看不到: documents, ranked_documents, answer...

  search_documents(state: SearchState)
    → 只能看到: rewritten_query
    → 看不到: question, documents, answer...

  rank_documents(state: RankState)
    → 只能看到: documents
    → 看不到: question, rewritten_query, answer...

  generate_answer(state: GenerateState)
    → 只能看到: question, ranked_documents
    → 看不到: rewritten_query, documents, metadata...

这就是「最小权限原则」在状态管理中的体现：
每个节点只拿到它需要的数据，不多不少。

  QueryState: ['question']
  SearchState: ['rewritten_query']
  RankState: ['documents']
  GenerateState: ['question', 'ranked_documents']
```

---

## 关键知识点解析

### 1. 三层 Schema 的职责划分

本场景使用了三层 Schema，各司其职：

| Schema 层级 | 类名 | 字段 | 职责 |
|-------------|------|------|------|
| 输入接口 | `RAGInput` | `question` | 外部调用者传入的数据 |
| 输出接口 | `RAGOutput` | `answer`, `sources` | 外部调用者拿到的数据 |
| 内部完整状态 | `RAGState` | 全部 8 个字段 | 节点间共享的完整数据 |
| 节点级视图 | `QueryState` 等 | 各自 1-2 个字段 | 每个节点只看到需要的字段 |

这就像一个 API 的设计：
- `RAGInput` = 请求体（Request Body）
- `RAGOutput` = 响应体（Response Body）
- `RAGState` = 服务器内部状态（对外不可见）
- 节点级 Schema = 每个中间件只处理自己关心的 Header

### 2. Input/Output Schema 的源码机制

```python
# StateGraph 构造函数
builder = StateGraph(
    RAGState,                    # state_schema：内部完整状态
    input_schema=RAGInput,       # 入口白名单
    output_schema=RAGOutput,     # 出口白名单
)
```

源码层面的行为：
- **START 节点**只写入 `input_schema` 的字段 → 只有 `question` 进入图
- **普通节点**可以读写所有 `state_schema` 的字段 → 内部自由操作
- **图的返回值**只包含 `output_schema` 的字段 → 只有 `answer` 和 `sources` 出去

```python
# langgraph/graph/state.py
def __init__(self, state_schema, context_schema=None, *, input_schema=None, output_schema=None):
    self.input_schema = input_schema or state_schema   # 默认 = 完整状态
    self.output_schema = output_schema or state_schema  # 默认 = 完整状态
```

如果不指定 `input_schema` 和 `output_schema`，它们默认等于 `state_schema`——这就是测试3中"无 Schema 分离"版本暴露所有字段的原因。

### 3. 节点级状态分组的自动推断

LangGraph 会自动从函数参数的类型注解推断节点的 input_schema：

```python
# 你写的代码
def rewrite_query(state: QueryState) -> dict:
    ...

# LangGraph 内部做的事
# 1. 检查 rewrite_query 的第一个参数类型 → QueryState
# 2. 提取 QueryState 的字段 → {"question": str}
# 3. 设置节点的 input_schema = QueryState
# 4. 执行时，只把 question 字段传给这个节点
```

源码中的推断逻辑：

```python
# langgraph/graph/state.py - add_node 方法
def add_node(self, node, action=None, *, input_schema=None, ...):
    if input_schema is not None:
        # 显式指定
        self.nodes[node] = StateNodeSpec(..., input_schema=input_schema)
    elif inferred_input_schema is not None:
        # 从函数参数类型自动推断
        self.nodes[node] = StateNodeSpec(..., input_schema=inferred_input_schema)
```

### 4. Reducer 与覆盖模式的混用策略

```python
class RAGState(TypedDict):
    question: str                                      # 覆盖模式
    rewritten_query: str                               # 覆盖模式
    documents: Annotated[list[dict], operator.add]     # Reducer：累积追加
    ranked_documents: list[dict]                       # 覆盖模式
    answer: str                                        # 覆盖模式
    sources: list[str]                                 # 覆盖模式
    metadata: dict                                     # 覆盖模式
```

为什么 `documents` 用 `operator.add` Reducer？

在更复杂的 RAG 系统中，可能有多个检索源（向量检索 + 关键词检索 + 知识图谱检索）并行执行，每个检索节点返回一批文档。使用 `operator.add` 可以自动合并多个来源的结果：

```python
# 并行检索场景（本例未实现，但 Schema 已预留）
# vector_search 返回: {"documents": [doc1, doc2]}
# keyword_search 返回: {"documents": [doc3, doc4]}
# 合并后: documents = [doc1, doc2, doc3, doc4]
```

其他字段用覆盖模式，因为它们在流程中只被写入一次，不需要累积。

### 5. 有无 Schema 分离的对比

```python
# ❌ 无 Schema 分离：调用者需要知道所有字段
result = pipeline.invoke({
    "question": "...",
    "answer": "",           # 为什么要传空字符串？
    "sources": [],          # 为什么要传空列表？
    "rewritten_query": "",  # 这是什么？
    "documents": [],        # 调用者不需要关心这个
    "ranked_documents": [],
    "metadata": {},
})
# result 包含所有 8 个字段——内部细节全暴露

# ✅ 有 Schema 分离：调用者只需要知道输入和输出
result = pipeline.invoke({"question": "..."})
# result 只包含 answer 和 sources——干净清晰
```

这不仅是美观问题，更是**接口稳定性**问题。如果你在内部增加一个 `embedding_cache` 字段，无 Schema 分离的版本会影响所有调用者（他们需要传入这个新字段的初始值），而有 Schema 分离的版本完全不受影响。

---

## 在 RAG 系统中的应用

本场景本身就是一个 RAG 系统的骨架。在生产环境中的扩展方向：

### 扩展1：增加并行检索

```python
# 多源并行检索——documents 的 Reducer 派上用场
builder.add_node("vector_search", vector_search_node)
builder.add_node("keyword_search", keyword_search_node)

# 并行执行
builder.add_edge("rewrite", "vector_search")
builder.add_edge("rewrite", "keyword_search")

# 汇合后排序
builder.add_edge("vector_search", "rank")
builder.add_edge("keyword_search", "rank")
# documents 字段自动合并两个检索源的结果
```

### 扩展2：增加 ReRank 节点

```python
class ReRankState(TypedDict):
    question: str
    ranked_documents: list[dict]

def rerank_node(state: ReRankState) -> dict:
    # 用交叉编码器对候选文档精排
    ...

builder.add_node("rerank", rerank_node)
builder.add_edge("rank", "rerank")
builder.add_edge("rerank", "generate")
```

### 扩展3：替换为真实 API 调用

```python
# 查询改写：替换为 OpenAI API
def rewrite_query(state: QueryState) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"改写查询: {state['question']}"}]
    )
    return {"rewritten_query": response.choices[0].message.content}

# 文档检索：替换为 ChromaDB
def search_documents(state: SearchState) -> dict:
    results = collection.query(query_texts=[state["rewritten_query"]], n_results=10)
    docs = [{"content": c, "score": s} for c, s in zip(results["documents"][0], results["distances"][0])]
    return {"documents": docs}
```

Pipeline 结构不变，只需替换节点内部的模拟逻辑。这就是多状态管理 + 单一职责节点的好处。

---

## 学习检查清单

- [ ] 理解 RAGInput / RAGOutput / RAGState 三层 Schema 的职责划分
- [ ] 能解释为什么 `metadata` 不出现在 `invoke()` 的返回值中
- [ ] 理解节点级 TypedDict 的自动推断机制（从函数参数类型推断）
- [ ] 知道 `documents` 为什么用 `operator.add` Reducer（预留并行检索）
- [ ] 能对比有无 Schema 分离的差异（接口稳定性、信息隐藏）
- [ ] 能将模拟节点替换为真实 API 调用，不改变 Pipeline 结构

---

## 下一步

掌握了 RAG 多阶段 Pipeline 后，可以继续探索：
- 增加条件路由——根据查询类型走不同检索策略
- 增加 `context_schema`——传入 API Key、模型配置等运行时不可变参数
- 结合子图嵌套——将检索模块抽成独立子图，支持复用和独立测试
- 增加 human-in-the-loop——在生成答案前加入人工审核节点
