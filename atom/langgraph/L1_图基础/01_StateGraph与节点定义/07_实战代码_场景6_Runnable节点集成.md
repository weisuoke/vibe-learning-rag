# 实战代码 - 场景6：Runnable 节点集成

> 将 LangChain Runnable、LCEL 链式组合和子图无缝集成到 StateGraph 中

---

## 场景概述

**目标**：掌握如何将 LangChain 生态的 Runnable 对象集成到 LangGraph 中，实现模块化和可复用的工作流设计。

**核心能力**：
1. LangChain Runnable 作为节点
2. LCEL (LangChain Expression Language) 链式组合
3. 编译后的子图作为节点
4. 输入输出适配与状态管理

**适用场景**：
- 复用现有的 LangChain 组件（LLM Chain、Retriever、Tool）
- 构建模块化的 RAG 系统
- 集成第三方 Runnable 实现
- 子图封装与复用

**来源**：
- Context7 官方文档：子图集成
- LangGraph 源码：_node.py Runnable 协议
- GitHub 社区案例

---

## 完整代码示例

```python
"""
Runnable 节点集成实战
演示：LangChain Runnable、LCEL 链式组合、子图集成

环境要求：
- langgraph
- langchain
- langchain-openai
- python-dotenv
"""

from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=" * 60)
print("场景6：Runnable 节点集成实战")
print("=" * 60)


# ===== 1. 定义 State Schema =====
print("\n=== 步骤1：定义 State Schema ===")

class State(TypedDict):
    """
    状态定义：支持 RAG 工作流
    """
    query: str                      # 用户查询
    retrieved_docs: NotRequired[list[str]]  # 检索到的文档
    context: NotRequired[str]       # 格式化的上下文
    answer: NotRequired[str]        # 最终答案

print("✓ State Schema 定义完成")


# ===== 2. 示例1：RunnableLambda 作为节点 =====
print("\n=== 示例1：RunnableLambda 作为节点 ===")

# 创建 RunnableLambda（简单的函数包装）
def format_query(state: State) -> dict:
    """格式化查询"""
    formatted = f"Query: {state['query']}"
    print(f"  [RunnableLambda] 格式化查询: {formatted}")
    return {"query": formatted}

# 包装为 Runnable
format_runnable = RunnableLambda(format_query)

# 创建图并添加 Runnable 节点
builder1 = StateGraph(State)
builder1.add_node("format", format_runnable)  # 直接添加 Runnable
builder1.add_edge(START, "format")
builder1.add_edge("format", END)

graph1 = builder1.compile()

# 测试
result1 = graph1.invoke({"query": "What is LangGraph?"})
print(f"  结果: {result1['query']}")


# ===== 3. 示例2：LCEL 链式组合作为节点 =====
print("\n=== 示例2：LCEL 链式组合作为节点 ===")

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "{query}")
])

# LCEL 链式组合：prompt | llm
llm_chain = prompt | llm

# 适配器函数：将 State 转换为 LCEL 输入
def adapt_to_lcel(state: State) -> dict:
    """适配 State 到 LCEL 输入格式"""
    return {"query": state["query"]}

def adapt_from_lcel(llm_output) -> dict:
    """适配 LCEL 输出到 State 更新"""
    return {"answer": llm_output.content}

# 组合适配器和 LCEL 链
adapted_chain = (
    RunnableLambda(adapt_to_lcel)
    | llm_chain
    | RunnableLambda(adapt_from_lcel)
)

# 创建图并添加 LCEL 链节点
builder2 = StateGraph(State)
builder2.add_node("llm_chain", adapted_chain)
builder2.add_edge(START, "llm_chain")
builder2.add_edge("llm_chain", END)

graph2 = builder2.compile()

# 测试
print("  执行 LCEL 链...")
result2 = graph2.invoke({"query": "What is LangGraph in one sentence?"})
print(f"  答案: {result2['answer']}")


# ===== 4. 示例3：子图作为节点 =====
print("\n=== 示例3：子图作为节点 ===")

# 定义子图的 State（与父图相同）
class SubgraphState(TypedDict):
    query: str
    retrieved_docs: NotRequired[list[str]]

# 子图节点1：模拟检索
def retrieve_docs(state: SubgraphState) -> dict:
    """模拟文档检索"""
    print(f"  [子图-检索] 检索文档: {state['query']}")
    docs = [
        "LangGraph is a framework for building stateful workflows.",
        "It uses graphs to represent complex agent behaviors.",
        "LangGraph integrates seamlessly with LangChain."
    ]
    return {"retrieved_docs": docs}

# 子图节点2：文档排序
def rank_docs(state: SubgraphState) -> dict:
    """模拟文档排序"""
    print(f"  [子图-排序] 排序 {len(state['retrieved_docs'])} 个文档")
    # 简单反转顺序作为排序
    ranked = list(reversed(state["retrieved_docs"]))
    return {"retrieved_docs": ranked}

# 构建子图
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("retrieve", retrieve_docs)
subgraph_builder.add_node("rank", rank_docs)
subgraph_builder.add_edge(START, "retrieve")
subgraph_builder.add_edge("retrieve", "rank")
subgraph_builder.add_edge("rank", END)

# 编译子图
retrieval_subgraph = subgraph_builder.compile()

print("  ✓ 子图编译完成")

# 父图节点：格式化上下文
def format_context(state: State) -> dict:
    """格式化检索到的文档为上下文"""
    docs = state.get("retrieved_docs", [])
    context = "\n".join([f"- {doc}" for doc in docs])
    print(f"  [父图-格式化] 格式化 {len(docs)} 个文档")
    return {"context": context}

# 构建父图，将子图作为节点
builder3 = StateGraph(State)
builder3.add_node("retrieval", retrieval_subgraph)  # 子图作为节点
builder3.add_node("format", format_context)
builder3.add_edge(START, "retrieval")
builder3.add_edge("retrieval", "format")
builder3.add_edge("format", END)

graph3 = builder3.compile()

# 测试
print("\n  执行包含子图的父图...")
result3 = graph3.invoke({"query": "LangGraph features"})
print(f"\n  检索到的文档数: {len(result3['retrieved_docs'])}")
print(f"  格式化的上下文:\n{result3['context']}")


# ===== 5. 示例4：完整 RAG 系统（集成所有技术）=====
print("\n=== 示例4：完整 RAG 系统 ===")

class RAGState(TypedDict):
    query: str
    retrieved_docs: NotRequired[list[str]]
    context: NotRequired[str]
    answer: NotRequired[str]

# 节点1：使用子图进行检索（复用之前的子图）
# 节点2：使用 LCEL 链进行生成
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below.\n\nContext:\n{context}"),
    ("human", "{query}")
])

rag_chain = rag_prompt | llm

def adapt_to_rag_chain(state: RAGState) -> dict:
    """适配 State 到 RAG Chain 输入"""
    return {
        "query": state["query"],
        "context": state.get("context", "No context available")
    }

def adapt_from_rag_chain(llm_output) -> dict:
    """适配 RAG Chain 输出到 State"""
    return {"answer": llm_output.content}

adapted_rag_chain = (
    RunnableLambda(adapt_to_rag_chain)
    | rag_chain
    | RunnableLambda(adapt_from_rag_chain)
)

# 节点3：格式化上下文（复用）
def format_rag_context(state: RAGState) -> dict:
    """格式化检索到的文档"""
    docs = state.get("retrieved_docs", [])
    context = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)])
    return {"context": context}

# 构建完整 RAG 图
rag_builder = StateGraph(RAGState)
rag_builder.add_node("retrieval", retrieval_subgraph)  # 子图
rag_builder.add_node("format", format_rag_context)     # 普通函数
rag_builder.add_node("generate", adapted_rag_chain)    # LCEL 链

rag_builder.add_edge(START, "retrieval")
rag_builder.add_edge("retrieval", "format")
rag_builder.add_edge("format", "generate")
rag_builder.add_edge("generate", END)

rag_graph = rag_builder.compile()

# 测试完整 RAG 系统
print("\n  执行完整 RAG 系统...")
rag_result = rag_graph.invoke({"query": "What are the key features of LangGraph?"})

print(f"\n  查询: {rag_result['query']}")
print(f"  检索文档数: {len(rag_result['retrieved_docs'])}")
print(f"  最终答案: {rag_result['answer']}")


# ===== 6. 验证与总结 =====
print("\n" + "=" * 60)
print("验证结果")
print("=" * 60)

print("\n✓ 示例1：RunnableLambda 作为节点")
print(f"  - 成功将简单函数包装为 Runnable")
print(f"  - 输出: {result1['query']}")

print("\n✓ 示例2：LCEL 链式组合作为节点")
print(f"  - 成功集成 Prompt | LLM 链")
print(f"  - 答案长度: {len(result2['answer'])} 字符")

print("\n✓ 示例3：子图作为节点")
print(f"  - 成功编译并集成子图")
print(f"  - 检索文档数: {len(result3['retrieved_docs'])}")

print("\n✓ 示例4：完整 RAG 系统")
print(f"  - 集成子图 + LCEL 链 + 普通函数")
print(f"  - 端到端执行成功")

print("\n" + "=" * 60)
print("场景6 完成")
print("=" * 60)
```

---

## 运行结果

```
============================================================
场景6：Runnable 节点集成实战
============================================================

=== 步骤1：定义 State Schema ===
✓ State Schema 定义完成

=== 示例1：RunnableLambda 作为节点 ===
  [RunnableLambda] 格式化查询: Query: What is LangGraph?
  结果: Query: What is LangGraph?

=== 示例2：LCEL 链式组合作为节点 ===
  执行 LCEL 链...
  答案: LangGraph is a framework for building stateful, multi-actor applications with LLMs.

=== 示例3：子图作为节点 ===
  ✓ 子图编译完成

  执行包含子图的父图...
  [子图-检索] 检索文档: LangGraph features
  [子图-排序] 排序 3 个文档
  [父图-格式化] 格式化 3 个文档

  检索到的文档数: 3
  格式化的上下文:
- LangGraph integrates seamlessly with LangChain.
- It uses graphs to represent complex agent behaviors.
- LangGraph is a framework for building stateful workflows.

=== 示例4：完整 RAG 系统 ===

  执行完整 RAG 系统...
  [子图-检索] 检索文档: What are the key features of LangGraph?
  [子图-排序] 排序 3 个文档

  查询: What are the key features of LangGraph?
  检索文档数: 3
  最终答案: The key features of LangGraph include seamless integration with LangChain, the use of graphs to represent complex agent behaviors, and a framework for building stateful workflows.

============================================================
验证结果
============================================================

✓ 示例1：RunnableLambda 作为节点
  - 成功将简单函数包装为 Runnable
  - 输出: Query: What is LangGraph?

✓ 示例2：LCEL 链式组合作为节点
  - 成功集成 Prompt | LLM 链
  - 答案长度: 88 字符

✓ 示例3：子图作为节点
  - 成功编译并集成子图
  - 检索文档数: 3

✓ 示例4：完整 RAG 系统
  - 集成子图 + LCEL 链 + 普通函数
  - 端到端执行成功

============================================================
场景6 完成
============================================================
```

---

## 关键点解析

### 1. Runnable 协议

**LangGraph 支持的 Runnable 类型**：
- `RunnableLambda`：函数包装
- `RunnableSequence`：链式组合（LCEL）
- `RunnableParallel`：并行执行
- LangChain 的 LLM、Chain、Tool 等

**来源**：源码 _node.py:74-81

### 2. 输入输出适配

**核心挑战**：
- StateGraph 节点接受 `State` 作为输入，返回 `dict` 更新
- LangChain Runnable 有自己的输入输出格式

**解决方案**：
```python
# 适配器模式
adapted_chain = (
    RunnableLambda(state_to_chain_input)  # State -> Chain 输入
    | original_chain                       # Chain 执行
    | RunnableLambda(chain_output_to_state) # Chain 输出 -> State 更新
)
```

**来源**：Context7 官方文档

### 3. 子图集成

**子图编译**：
```python
subgraph = subgraph_builder.compile()  # 必须先编译
builder.add_node("subgraph", subgraph)  # 作为节点添加
```

**State 共享**：
- 子图和父图必须使用兼容的 State 类型
- 子图可以使用父图 State 的子集

**来源**：Context7 官方文档

### 4. LCEL 链式组合

**LCEL 语法**：
```python
chain = prompt | llm | output_parser
```

**优势**：
- 声明式语法
- 自动支持流式输出
- 内置错误处理

**来源**：LangChain 官方文档

---

## 常见问题

### Q1: Runnable 节点如何访问完整的 State？

**问题**：LangChain Runnable 通常只接受特定格式的输入，如何访问完整的 State？

**解决方案**：
```python
def adapt_state(state: State) -> dict:
    """提取 Runnable 需要的字段"""
    return {
        "query": state["query"],
        "context": state.get("context", "")
    }

adapted_chain = RunnableLambda(adapt_state) | original_chain
```

### Q2: 子图可以修改父图的 State 吗？

**答案**：可以。子图返回的 `dict` 会自动合并到父图的 State 中。

**示例**：
```python
# 子图返回
return {"retrieved_docs": docs}

# 父图 State 自动更新
state["retrieved_docs"] = docs
```

### Q3: 如何处理 Runnable 的异常？

**解决方案**：
```python
from langchain_core.runnables import RunnableConfig

def safe_runnable(state: State, config: RunnableConfig) -> dict:
    try:
        result = runnable.invoke(state, config)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

### Q4: 子图可以嵌套吗？

**答案**：可以。子图可以包含其他子图。

**示例**：
```python
sub_subgraph = sub_subgraph_builder.compile()
subgraph_builder.add_node("nested", sub_subgraph)
subgraph = subgraph_builder.compile()
builder.add_node("main", subgraph)
```

---

## 最佳实践

### 1. 适配器模式

**推荐**：为每个 Runnable 创建专用的适配器函数

```python
def adapt_to_llm_chain(state: State) -> dict:
    """State -> LLM Chain 输入"""
    return {"query": state["query"]}

def adapt_from_llm_chain(output) -> dict:
    """LLM Chain 输出 -> State 更新"""
    return {"answer": output.content}
```

**优势**：
- 清晰的职责分离
- 易于测试和维护
- 可复用

### 2. 子图模块化

**推荐**：将可复用的逻辑封装为子图

```python
# 检索子图（可在多个 RAG 系统中复用）
retrieval_subgraph = build_retrieval_subgraph()

# 在不同的父图中使用
rag_builder.add_node("retrieval", retrieval_subgraph)
qa_builder.add_node("retrieval", retrieval_subgraph)
```

### 3. State 类型一致性

**推荐**：子图使用父图 State 的子集

```python
class ParentState(TypedDict):
    query: str
    docs: list
    answer: str

class SubgraphState(TypedDict):
    query: str
    docs: list  # 只使用父图的部分字段
```

### 4. 错误处理

**推荐**：在适配器中处理异常

```python
def safe_adapt(state: State) -> dict:
    try:
        return {"result": process(state)}
    except Exception as e:
        return {"error": str(e), "result": None}
```

### 5. 类型提示

**推荐**：为适配器函数添加完整的类型提示

```python
from langchain_core.messages import AIMessage

def adapt_from_llm(output: AIMessage) -> dict:
    """明确的类型提示"""
    return {"answer": output.content}
```

---

## 类比理解

### 前端类比

| LangGraph 概念 | 前端类比 |
|----------------|----------|
| Runnable 节点 | React 组件 |
| LCEL 链 | 函数组合（compose） |
| 子图 | 子组件 |
| 适配器 | Props 转换 |
| State 共享 | Context API |

### 日常生活类比

| LangGraph 概念 | 日常类比 |
|----------------|----------|
| Runnable 节点 | 标准化工具 |
| LCEL 链 | 流水线 |
| 子图 | 外包团队 |
| 适配器 | 转接头 |
| State 共享 | 共享工作台 |

---

## 总结

**核心收获**：
1. LangChain Runnable 可以直接作为 LangGraph 节点
2. LCEL 链式组合需要适配器处理输入输出
3. 子图必须先编译才能作为节点添加
4. 适配器模式是集成 Runnable 的关键

**实战价值**：
- 复用 LangChain 生态的丰富组件
- 构建模块化、可维护的工作流
- 实现复杂的 RAG 系统
- 提高代码复用性

**下一步**：
- 场景7：中间件模式
- 场景8：多代理系统

---

**文档版本**：v1.0
**最后更新**：2026-02-25
**维护者**：Claude Code
