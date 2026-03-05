---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-25
knowledge_point: 07_边的类型与选择
context7_query: add_edge add_conditional_edges edge types
---

# Context7 文档：LangGraph 边的类型

## 文档来源
- 库名称：LangGraph
- 版本：latest (1.0.8)
- 官方文档链接：https://context7.com/langchain-ai/langgraph

## 关键信息提取

### 1. 动态路由的条件边

**核心示例**：

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
```

**关键特征**：
- 路由函数使用 `Literal` 类型注解
- `path_map` 字典映射路由函数返回值到节点名称
- 条件边根据状态值动态决定下一个节点

### 2. RAG 工作流的边定义

**示例1：CRAG 工作流**

```python
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

custom_graph = workflow.compile()
```

**示例2：Self-RAG 工作流**

```python
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
```

**关键特征**：
- 多个条件边组合实现复杂的决策流程
- 条件边可以形成循环（如 `transform_query` -> `retrieve`）
- 条件边可以路由到 `END` 节点

### 3. 边的类型总结

**普通边（Normal Edge）**：
- API：`add_edge(start, end)`
- 特征：固定路由，直接连接两个节点
- 使用场景：确定性流程

**条件边（Conditional Edge）**：
- API：`add_conditional_edges(source, path, path_map)`
- 特征：动态路由，根据路由函数返回值决定下一个节点
- 使用场景：需要根据状态进行决策的场景

**序列边（Sequence Edge）**：
- API：`add_sequence(nodes)`
- 特征：语法糖，自动添加节点和边
- 使用场景：线性流程

### 4. 路由函数设计

**类型注解**：
- 使用 `Literal` 类型注解提升可维护性
- 类型注解可以自动推断 `path_map`

**路由逻辑**：
- 路由函数应该是纯函数，只读取状态并返回路由决策
- 路由函数可以返回单个节点名称或多个节点名称（并行执行）

**path_map 映射**：
- 字典形式：`{"return_value": "node_name"}`
- 列表形式：`["node1", "node2"]`（路由函数直接返回节点名称）
- 自动推断：如果路由函数有 `Literal` 返回类型注解，自动推断 `path_map`

### 5. 实际应用模式

**模式1：二分支决策**
```python
def route_by_condition(state: State) -> Literal["path_a", "path_b"]:
    if condition:
        return "path_a"
    return "path_b"

builder.add_conditional_edges(
    "source",
    route_by_condition,
    {"path_a": "node_a", "path_b": "node_b"}
)
```

**模式2：多分支决策**
```python
def route_by_category(state: State) -> Literal["cat1", "cat2", "cat3"]:
    # 根据状态分类
    return category

builder.add_conditional_edges(
    "source",
    route_by_category,
    {
        "cat1": "handler1",
        "cat2": "handler2",
        "cat3": "handler3"
    }
)
```

**模式3：循环与退出**
```python
def route_with_loop(state: State) -> Literal["continue", "exit"]:
    if should_continue:
        return "continue"
    return "exit"

builder.add_conditional_edges(
    "process",
    route_with_loop,
    {
        "continue": "process",  # 循环
        "exit": END
    }
)
```

## 参考价值

这些官方文档提供了：
- **标准用法**：LangGraph 边的标准 API 使用方式
- **代码示例**：完整可运行的代码示例
- **设计模式**：常见的边使用模式
- **最佳实践**：类型注解、路由函数设计等最佳实践
