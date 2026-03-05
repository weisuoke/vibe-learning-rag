---
type: context7_documentation
library: LangGraph
library_id: /websites/langchain_oss_python_langgraph
version: main
fetched_at: 2026-02-25
knowledge_point: 01_StateGraph与节点定义
context7_query: StateGraph class creation node definition add_node START END
---

# Context7 文档：LangGraph - StateGraph 与节点定义

## 文档来源
- 库名称：LangGraph
- Library ID：`/websites/langchain_oss_python_langgraph`
- 版本：main
- 最后更新：2026-02-17
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/
- 总 Snippets：900
- Trust Score：10
- Benchmark Score：86.9

## 关键信息提取

### 1. StateGraph 基础创建

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心模式**：
```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# Add nodes
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# Add edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")

graph = builder.compile()
```

**关键点**：
- StateGraph 是 builder 模式
- 需要先定义 State（TypedDict）
- 使用 `add_node` 添加节点
- 使用 `add_edge` 连接节点
- 使用 `compile()` 生成可执行图

### 2. State 定义与 Reducer

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

**State 定义示例**：
```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}
```

**关键点**：
- State 使用 TypedDict 定义
- 使用 `Annotated[type, reducer]` 添加 reducer 函数
- `operator.add` 使列表变为 append-only
- 节点函数返回部分状态更新

### 3. 节点函数定义

**来源**：https://docs.langchain.com/oss/python/langgraph/use-time-travel

**节点函数示例**：
```python
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0,
)

def generate_topic(state: State):
    """LLM call to generate a topic for the joke"""
    msg = model.invoke("Give me a funny topic for a joke")
    return {"topic": msg.content}

def write_joke(state: State):
    """LLM call to write a joke based on the topic"""
    msg = model.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}
```

**关键点**：
- 节点函数接受 State 作为参数
- 节点函数返回部分状态更新（dict）
- 可以使用 `NotRequired` 标记可选字段
- 节点函数可以调用 LLM

### 4. 条件路由

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

**条件边示例**：
```python
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
```

**关键点**：
- 使用 `add_conditional_edges` 添加条件边
- 路由函数返回目标节点名称或 END
- 使用 `Literal` 类型提示可能的路由目标

### 5. START 和 END 节点

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

**START 和 END 的使用**：
```python
from langgraph.graph import START, END

# START: 图的入口点
builder.add_edge(START, "first_node")

# END: 图的出口点
builder.add_edge("last_node", END)

# 条件路由到 END
def route(state: State) -> Literal["continue", END]:
    if condition:
        return "continue"
    else:
        return END
```

**关键点**：
- `START` 是特殊的入口节点
- `END` 是特殊的出口节点
- 可以在条件路由中返回 END

### 6. 图的编译

**来源**：https://docs.langchain.com/oss/python/langgraph/graph-api

**compile 方法**：
```python
graph = graph_builder.compile(...)
```

**关键点**：
- `compile()` 是必需的步骤
- 编译时会进行结构检查
- 可以指定 checkpointer、breakpoints 等运行时参数
- 返回可执行的 Pregel 实例

### 7. Pregel 实例

**来源**：https://docs.langchain.com/oss/python/langgraph/pregel

**Pregel 应用示例**：
```python
from typing import TypedDict
from langgraph.constants import START
from langgraph.graph import StateGraph

class Essay(TypedDict):
    topic: str
    content: str | None
    score: float | None

def write_essay(essay: Essay):
    return {
        "content": f"Essay about {essay['topic']}",
    }

def score_essay(essay: Essay):
    return {
        "score": 10
    }

builder = StateGraph(Essay)
builder.add_node(write_essay)
builder.add_node(score_essay)
builder.add_edge(START, "write_essay")
builder.add_edge("write_essay", "score_essay")

# Compile the graph.
# This will return a Pregel instance.
graph = builder.compile()

print(graph.nodes)
print(graph.channels)
```

**关键点**：
- StateGraph 自动编译为 Pregel 实例
- Pregel 实例有 `nodes` 和 `channels` 属性
- Pregel 是 LangGraph 的执行引擎

### 8. 子图集成

**来源**：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

**子图示例**：
```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class State(TypedDict):
    foo: str

# Subgraph
def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(State)
builder.add_node("node_1", subgraph)  # 添加编译后的子图
builder.add_edge(START, "node_1")
graph = builder.compile()
```

**关键点**：
- 子图需要先编译
- 编译后的子图可以作为节点添加到父图
- 子图和父图共享相同的 State 类型

### 9. 边的类型

**来源**：https://docs.langchain.com/oss/python/langgraph/graph-api

**普通边**：
```python
graph.add_edge("node_a", "node_b")
```

**关键点**：
- 普通边是无条件的直接转换
- 从 node_a 完成后总是转到 node_b

### 10. 完整工作流示例

**来源**：https://docs.langchain.com/oss/python/langgraph/use-time-travel

**完整示例**：
```python
import uuid
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0,
)

def generate_topic(state: State):
    """LLM call to generate a topic for the joke"""
    msg = model.invoke("Give me a funny topic for a joke")
    return {"topic": msg.content}

def write_joke(state: State):
    """LLM call to write a joke based on the topic"""
    msg = model.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", END)
```

**关键点**：
- 完整的工作流包含：State 定义、节点函数、图构建、边连接
- 可以使用 InMemorySaver 作为 checkpointer
- 节点可以有显式名称（字符串）

## 核心概念总结

### StateGraph 的本质
- **Builder 模式**：逐步构建图结构
- **类型安全**：通过 TypedDict 定义状态
- **部分更新**：节点返回部分状态，自动合并
- **Reducer 函数**：控制状态更新策略

### 节点定义规范
- **输入**：接受完整的 State
- **输出**：返回部分状态更新（dict）
- **命名**：可以自动推断或显式指定
- **类型**：函数、Runnable、编译后的子图

### 边的类型
- **普通边**：`add_edge(start, end)` - 无条件转换
- **条件边**：`add_conditional_edges(source, route_fn)` - 动态路由
- **START 边**：`add_edge(START, node)` - 入口点
- **END 边**：`add_edge(node, END)` - 出口点

### 编译与执行
- **编译**：`compile()` 生成 Pregel 实例
- **验证**：编译时检查图结构
- **执行**：Pregel 实例支持 invoke、stream 等方法

## 实战模式

### 模式 1：顺序执行
```python
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)
```

### 模式 2：条件分支
```python
builder.add_edge(START, "decision_node")
builder.add_conditional_edges("decision_node", route_fn)
```

### 模式 3：循环执行
```python
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)  # 可能返回 "b" 或 END
builder.add_edge("b", "a")  # 循环回 a
```

### 模式 4：子图模块化
```python
subgraph = subgraph_builder.compile()
builder.add_node("subgraph_node", subgraph)
```

## 常见陷阱

1. **忘记编译**：StateGraph 必须调用 `compile()` 才能执行
2. **状态类型不匹配**：节点返回的字段必须在 State 中定义
3. **循环无终止**：条件路由必须有退出条件
4. **节点名称冲突**：节点名称必须唯一
5. **START/END 误用**：START 不能作为终点，END 不能作为起点
