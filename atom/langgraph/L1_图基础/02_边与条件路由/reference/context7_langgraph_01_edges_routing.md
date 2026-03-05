---
type: context7_documentation
library: langgraph
library_id: /langchain-ai/langgraph
version: main (latest)
fetched_at: 2026-02-25
knowledge_point: 02_边与条件路由
context7_query: add_edge add_conditional_edges routing
total_snippets: 234
trust_score: 9.2
benchmark_score: 77.5
---

# Context7 文档：LangGraph 边与条件路由

## 文档来源
- 库名称：LangGraph
- 库 ID：/langchain-ai/langgraph
- 版本：main (最新)
- 官方文档链接：https://context7.com/langchain-ai/langgraph
- 总代码片段数：234
- 信任评分：9.2/10
- 基准评分：77.5/100

## 关键信息提取

### 1. 动态路由与条件边的核心概念

**来源**：https://context7.com/langchain-ai/langgraph/llms.txt

**核心示例**：使用 `add_conditional_edges` 实现动态路由

```python
from langgraph.graph import START, END, StateGraph
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    value: int
    path_taken: str

def process_input(state: State) -> dict:
    return {"value": state["value"]}

def route_by_value(state: State) -> Literal["high_path", "low_path"]:
    """Route based on state value."""
    if state["value"] > 50:
        return "high_path"
    return "low_path"

def high_handler(state: State) -> dict:
    return {"path_taken": "high", "value": state["value"] * 2}

def low_handler(state: State) -> dict:
    return {"path_taken": "low", "value": state["value"] + 10}

builder = StateGraph(State)
builder.add_node("process", process_input)
builder.add_node("high_path", high_handler)
builder.add_node("low_path", low_handler)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    route_by_value,
    {"high_path": "high_path", "low_path": "low_path"}
)
builder.add_edge("high_path", END)
builder.add_edge("low_path", END)

graph = builder.compile()

# Test with high value
result = graph.invoke({"value": 75, "path_taken": ""})
print(result)  # {'value': 150, 'path_taken': 'high'}

# Test with low value
result = graph.invoke({"value": 25, "path_taken": ""})
print(result)  # {'value': 35, 'path_taken': 'low'}
```

**关键点**：
1. **路由函数签名**：接受 `state` 参数，返回目标节点名称
2. **类型提示**：使用 `Literal` 类型提示明确可能的路由目标
3. **path_map 参数**：将路由函数返回值映射到实际节点名称
4. **动态决策**：基于状态值动态选择执行路径

### 2. RAG 工作流中的条件边应用

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

**场景**：文档相关性评分后的路由决策

```python
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```

**关键点**：
1. **决策函数**：`decide_to_generate` 根据文档相关性决定下一步
2. **多路由选择**：
   - `transform_query`：文档不相关，需要改写查询
   - `generate`：文档相关，直接生成答案
3. **路由后续流程**：不同路由有不同的后续节点

### 3. 复杂 RAG 工作流的条件边组合

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb

**场景**：多层条件路由的自适应 RAG 系统

```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # rag
workflow.add_node("llm_fallback", llm_fallback)  # llm

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# Compile
app = workflow.compile()
```

**关键点**：
1. **入口条件路由**：`START` 节点使用条件边决定初始路径
   - `web_search`：需要网络搜索
   - `vectorstore`：使用向量检索
   - `llm_fallback`：直接使用 LLM
2. **中间条件路由**：`grade_documents` 评估文档质量
3. **输出条件路由**：`generate` 评估生成质量
   - `not supported`：幻觉检测，重新生成
   - `not useful`：答案无用，回退到网络搜索
   - `useful`：结束流程
4. **循环路由**：支持重新生成和回退机制

### 4. 自适应 RAG 的查询转换路由

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb

**场景**：查询转换与检索的循环流程

```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()
```

**关键点**：
1. **查询转换循环**：`transform_query` → `retrieve` → `grade_documents`
2. **多次尝试机制**：如果文档不相关，转换查询后重新检索
3. **生成质量评估**：
   - `not supported`：重新生成
   - `not useful`：转换查询
   - `useful`：结束
4. **避免无限循环**：需要在状态中添加计数器限制尝试次数

### 5. Self-RAG 的条件边模式

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_pinecone_movies.ipynb

**场景**：自我反思的 RAG 系统

```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()
```

**关键点**：
1. **固定入口**：从 `retrieve` 开始
2. **文档评分**：决定是否需要转换查询
3. **生成评分**：决定是否需要重新生成或转换查询
4. **自我修正**：通过循环实现自我改进

## 条件边的设计模式总结

### 模式 1：简单分支（if-else）
```python
def route_func(state) -> Literal["path_a", "path_b"]:
    if condition:
        return "path_a"
    return "path_b"

graph.add_conditional_edges(
    "source",
    route_func,
    {"path_a": "node_a", "path_b": "node_b"}
)
```

### 模式 2：多路由（switch-case）
```python
def route_func(state) -> Literal["path_a", "path_b", "path_c", END]:
    if condition_a:
        return "path_a"
    elif condition_b:
        return "path_b"
    elif condition_c:
        return "path_c"
    return END

graph.add_conditional_edges("source", route_func)
```

### 模式 3：循环路由（retry/fallback）
```python
def route_func(state) -> Literal["retry", "fallback", END]:
    if should_retry(state):
        return "retry"
    elif should_fallback(state):
        return "fallback"
    return END

graph.add_conditional_edges(
    "process",
    route_func,
    {
        "retry": "process",  # 循环回自己
        "fallback": "fallback_node",
        END: END
    }
)
```

### 模式 4：入口条件路由
```python
graph.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
```

### 模式 5：多层条件路由
```python
# 第一层路由
graph.add_conditional_edges(START, route_entry, {...})

# 第二层路由
graph.add_conditional_edges("process", route_middle, {...})

# 第三层路由
graph.add_conditional_edges("generate", route_exit, {...})
```

## 路由函数的最佳实践

### 1. 使用类型提示
```python
def route_func(state: State) -> Literal["path_a", "path_b", END]:
    # 明确返回类型，便于图可视化
    ...
```

### 2. 路由逻辑清晰
```python
def route_func(state: State) -> str:
    # 使用清晰的条件判断
    if state["score"] > 0.8:
        return "high_quality"
    elif state["score"] > 0.5:
        return "medium_quality"
    else:
        return "low_quality"
```

### 3. 避免复杂逻辑
```python
# ❌ 不推荐：复杂的嵌套逻辑
def route_func(state):
    if state["a"]:
        if state["b"]:
            if state["c"]:
                return "path1"
            else:
                return "path2"
        else:
            return "path3"
    else:
        return "path4"

# ✅ 推荐：扁平化逻辑
def route_func(state):
    if state["a"] and state["b"] and state["c"]:
        return "path1"
    if state["a"] and state["b"]:
        return "path2"
    if state["a"]:
        return "path3"
    return "path4"
```

### 4. 添加日志和调试信息
```python
def route_func(state: State) -> str:
    score = state["score"]
    print(f"Routing decision: score={score}")

    if score > 0.8:
        print("→ high_quality path")
        return "high_quality"
    else:
        print("→ low_quality path")
        return "low_quality"
```

### 5. 处理边界情况
```python
def route_func(state: State) -> str:
    # 处理缺失值
    score = state.get("score", 0.0)

    # 处理异常值
    if score < 0 or score > 1:
        return "error_handler"

    # 正常路由
    if score > 0.5:
        return "pass"
    return "fail"
```

## 常见错误与解决方案

### 错误 1：路由函数返回未定义的节点
```python
# ❌ 错误
def route_func(state):
    return "undefined_node"  # 节点不存在

# ✅ 正确
def route_func(state) -> Literal["node_a", "node_b"]:
    return "node_a"  # 确保节点已定义
```

### 错误 2：path_map 与返回值不匹配
```python
# ❌ 错误
def route_func(state):
    return "yes"  # 返回 "yes"

graph.add_conditional_edges(
    "source",
    route_func,
    {"true": "node_a", "false": "node_b"}  # path_map 中没有 "yes"
)

# ✅ 正确
graph.add_conditional_edges(
    "source",
    route_func,
    {"yes": "node_a", "no": "node_b"}  # 匹配返回值
)
```

### 错误 3：忘记添加 END 路由
```python
# ❌ 错误：没有结束路径
def route_func(state):
    if state["done"]:
        return "process"  # 永远不会结束
    return "process"

# ✅ 正确：添加结束路径
def route_func(state) -> Literal["process", END]:
    if state["done"]:
        return END
    return "process"
```

### 错误 4：循环路由没有退出条件
```python
# ❌ 错误：无限循环
def route_func(state):
    if state["retry"]:
        return "process"  # 一直循环
    return "process"

# ✅ 正确：添加计数器
def route_func(state):
    if state.get("retry_count", 0) < 3 and state["retry"]:
        return "process"
    return END
```

## 性能优化建议

### 1. 避免在路由函数中执行耗时操作
```python
# ❌ 不推荐
def route_func(state):
    result = expensive_api_call()  # 耗时操作
    if result > 0.5:
        return "path_a"
    return "path_b"

# ✅ 推荐：在节点中执行耗时操作
def process_node(state):
    result = expensive_api_call()
    return {"result": result}

def route_func(state):
    # 使用已计算的结果
    if state["result"] > 0.5:
        return "path_a"
    return "path_b"
```

### 2. 缓存路由决策
```python
def route_func(state):
    # 检查缓存
    if "route_decision" in state:
        return state["route_decision"]

    # 计算路由
    decision = compute_route(state)
    return decision
```

### 3. 使用简单的条件判断
```python
# ✅ 快速
def route_func(state):
    return "path_a" if state["flag"] else "path_b"

# ❌ 慢
def route_func(state):
    import time
    time.sleep(0.1)  # 模拟复杂计算
    return "path_a"
```

## 总结

LangGraph 的条件边机制提供了强大的动态路由能力：

1. **核心方法**：`add_conditional_edges(source, path, path_map)`
2. **路由函数**：接受状态，返回目标节点名称
3. **类型提示**：使用 `Literal` 明确可能的路由目标
4. **path_map**：将路由函数返回值映射到节点名称
5. **设计模式**：简单分支、多路由、循环路由、入口路由、多层路由
6. **最佳实践**：清晰逻辑、类型提示、错误处理、性能优化
7. **常见应用**：RAG 工作流、自适应系统、错误重试、质量评估

通过合理使用条件边，可以构建复杂的自适应工作流系统。
