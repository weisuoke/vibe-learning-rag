---
type: context7_documentation
library: LangGraph
version: latest (2026-02-17)
fetched_at: 2026-02-27
knowledge_point: 01_并行执行与分支合并
context7_query: Send parallel execution map reduce branch merge conditional edges
---

# Context7 文档：LangGraph 并行执行与分支合并

## 文档来源
- 库名称：LangGraph
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/
- Context7 库 ID：`/websites/langchain_oss_python_langgraph`

## 关键信息提取

### 1. 并行图执行与状态 Reducers

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心概念**:
- 使用 `operator.add` 作为 reducer 来累积值
- 确保新值被追加而不是覆盖现有值
- 图结构从节点 'a' 扇出到 'b' 和 'c'，然后扇入到 'd'
- 允许 'b' 和 'c' 在同一个超步中并发执行

**代码示例**:
```python
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```

**关键点**:
- 节点 'a' 扇出到 'b' 和 'c'，这两个节点并行执行
- 节点 'd' 等待 'b' 和 'c' 都完成后再执行（扇入）
- 使用 `Annotated[list, operator.add]` 确保结果被追加而不是覆盖

### 2. 并行 LLM 调用

**来源**: https://docs.langchain.com/oss/python/langgraph/workflows-agents

**核心概念**:
- 并行执行多个 LLM 调用（笑话、故事、诗歌生成）
- 使用聚合器节点合并结果
- 通过边管理工作流

**代码示例**:
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

# Nodes
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    """Second LLM call to generate story"""
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    """Third LLM call to generate poem"""
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """Combine the joke, story and poem into a single output"""
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}

# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Invoke
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])
```

**关键点**:
- 从 START 节点扇出到三个 LLM 调用节点
- 三个 LLM 调用并行执行
- 聚合器节点等待所有 LLM 调用完成后再执行
- 最终输出合并所有结果

### 3. Send API 实现 Map-Reduce

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心概念**:
- 使用 Send API 实现 Map-Reduce 模式
- 动态创建多个并行任务
- 使用 `operator.add` reducer 合并结果

**代码示例**:
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    return {"subjects": ["lions", "elephants", "penguins"]}

def generate_joke(state: OverallState):
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }
    return {"jokes": [joke_map[state["subject"]]]}

def continue_to_jokes(state: OverallState):
    # 返回多个 Send 对象实现并行执行
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def best_joke(state: OverallState):
    return {"best_selected_joke": "penguins"}

builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)
graph = builder.compile()
```

**关键点**:
- `continue_to_jokes` 函数返回多个 `Send` 对象
- 每个 `Send` 对象指定目标节点和输入状态
- `generate_joke` 节点被调用多次，每次使用不同的 subject
- 使用 `Annotated[list[str], operator.add]` 自动合并所有笑话

### 4. 并行处理多个数据源

**来源**: https://docs.langchain.com/oss/python/langgraph/choosing-apis

**核心概念**:
- 并行获取多个数据源
- 使用合并节点等待所有并行操作完成

**代码示例**:
```python
# Parallel processing of multiple data sources
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("fetch_weather", fetch_weather)
workflow.add_node("fetch_stocks", fetch_stocks)
workflow.add_node("combine_data", combine_all_data)

# All fetch operations run in parallel
workflow.add_edge(START, "fetch_news")
workflow.add_edge(START, "fetch_weather")
workflow.add_edge(START, "fetch_stocks")

# Combine waits for all parallel operations to complete
workflow.add_edge("fetch_news", "combine_data")
workflow.add_edge("fetch_weather", "combine_data")
workflow.add_edge("fetch_stocks", "combine_data")
```

**关键点**:
- 从 START 节点扇出到三个数据获取节点
- 三个数据获取操作并行执行
- `combine_data` 节点等待所有数据获取完成后再执行

### 5. 并行节点执行机制

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心概念**:
- LangGraph 允许并行执行图节点
- 当一个节点扇出到多个后续节点时，这些节点可以在同一个超步中并发运行
- 使用 reducers 管理并行分支的状态更新

**关键点**:
- **扇出模式**: 节点 A 扇出到节点 B 和 C，B 和 C 并行执行
- **Reducer 函数**: 使用 `operator.add` reducer 可以累积值而不是覆盖
- **列表状态**: 对于列表状态，新列表元素会与现有元素连接
- **同步点**: 扇入节点等待所有并行节点完成后再执行

### 6. 条件分支

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心概念**:
- 使用 `add_conditional_edges` 设置条件分支
- 根据状态动态确定下一个节点
- 允许运行时变化执行流程

**代码示例**:
```python
import operator
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str  # 用于确定分支的键

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"], "which": "c"}  # 设置分支方向

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)

def conditional_edge(state: State) -> Literal["b", "c"]:
    # 使用状态确定下一个节点
    return state["which"]

builder.add_conditional_edges("a", conditional_edge)
graph = builder.compile()
```

**关键点**:
- 条件边函数根据状态返回下一个节点名称
- 可以返回单个节点名称或节点名称列表
- 支持复杂的决策逻辑

### 7. 条件循环

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**核心概念**:
- 使用条件边实现循环
- 根据状态决定是否继续循环或结束

**代码示例**:
```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

**关键点**:
- 条件边函数可以返回 END 来终止执行
- 循环边从节点 'b' 回到节点 'a'
- 根据状态条件决定是否继续循环

## 总结

LangGraph 的并行执行与分支合并机制包括以下核心特性：

1. **扇出/扇入模式**: 通过添加多条边实现节点扇出，并行节点自动在同一超步中执行
2. **Send API**: 动态创建并行任务，支持 Map-Reduce 模式
3. **Reducer 函数**: 使用 `operator.add` 等 reducer 自动合并并行节点的结果
4. **条件分支**: 使用 `add_conditional_edges` 根据状态动态路由
5. **同步机制**: 基于 Bulk Synchronous Parallel 模型，确保并行节点在同一超步中执行
6. **状态管理**: 使用 `Annotated` 类型和 reducer 函数管理并行更新

这些特性使得 LangGraph 能够优雅地处理复杂的并行工作流场景。
