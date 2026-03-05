---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-25
knowledge_point: 04_图的编译与执行
context7_query: invoke method graph execution stream durability
---

# Context7 文档：LangGraph invoke 方法

## 文档来源
- 库名称：LangGraph
- 版本：latest
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/

## 关键信息提取

### 1. Execute LangGraph Workflow: Invoke, Stream, Ainvoke, Astream

**来源**：https://docs.langchain.com/oss/python/langgraph/functional-api

演示同步和异步执行 LangGraph 工作流的方法。这些方法允许立即获取结果或流式输出块。它们需要配置，包括用于状态管理的 thread ID。

**同步 invoke**：
```python
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}
my_workflow.invoke(some_input, config)  # 同步等待结果
```

**异步 ainvoke**：
```python
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}
await my_workflow.ainvoke(some_input, config)  # 异步等待结果
```

**同步 stream**：
```python
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

for chunk in my_workflow.stream(some_input, config):
    print(chunk)
```

**异步 astream**：
```python
config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

async for chunk in my_workflow.astream(some_input, config):
    print(chunk)
```

**关键点**：
- 所有执行方法都需要 `thread_id` 配置
- `invoke/ainvoke` 等待完整结果
- `stream/astream` 流式输出块

### 2. Configure Durability Mode in LangGraph Stream

**来源**：https://docs.langchain.com/oss/python/langgraph/durable-execution

演示如何在流式执行 LangGraph 工作流时设置持久化模式。展示了 'sync' 模式，确保在下一步开始前同步持久化更改，提供高持久性但有性能成本。

```python
graph.stream(
    {"input": "test"},
    durability="sync"
)
```

**持久化模式**：
- `"sync"`: 同步持久化（高持久性，性能成本高）
- `"async"`: 异步持久化（默认）
- `"exit"`: 仅在图退出时持久化

### 3. Stream Full Graph State Values in LangGraph Python

**来源**：https://docs.langchain.com/oss/python/langgraph/streaming

启用在每个步骤后流式输出完整的图执行状态。'values' 模式在每个执行点提供完整的状态对象，提供图条件的全面视图。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",  # 流式输出完整状态
):
    print(chunk)
```

**关键点**：
- `stream_mode="values"` 输出完整状态
- 每个步骤后都会输出当前状态

### 4. Stream Graph State Updates in LangGraph Python

**来源**：https://docs.langchain.com/oss/python/langgraph/streaming

启用流式输出图执行状态。'updates' 模式仅流式输出每个节点执行后的状态更改，包括节点名称和状态更新。这提供了状态修改的细粒度视图。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",  # 仅流式输出更新
):
    print(chunk)
```

**关键点**：
- `stream_mode="updates"` 仅输出状态更改
- 包括节点名称和更新内容
- 提供细粒度的状态修改视图

### 5. Execute LangGraph Stream for Joke Generation in Python

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

演示如何流式执行编译后的 LangGraph。它接受一个带有 'topic' 的初始状态，并打印图执行的每个步骤，显示生成的笑话和最终的最佳笑话选择。

```python
# 调用图：这里我们调用它来生成笑话列表
for step in graph.stream({"topic": "animals"}):
    print(step)
```

**关键点**：
- 流式执行显示每个步骤
- 可以实时观察图的执行过程

## 总结

### invoke 方法核心特性

1. **配置要求**：
   - 必须提供 `thread_id` 用于状态管理
   - 通过 `config` 参数传递

2. **执行模式**：
   - `invoke()` - 同步执行，等待完整结果
   - `ainvoke()` - 异步执行，等待完整结果
   - `stream()` - 同步流式执行
   - `astream()` - 异步流式执行

3. **流模式**：
   - `stream_mode="values"` - 输出完整状态
   - `stream_mode="updates"` - 仅输出状态更新

4. **持久化控制**：
   - `durability="sync"` - 同步持久化
   - `durability="async"` - 异步持久化（默认）
   - `durability="exit"` - 仅在退出时持久化

### 实际应用场景

1. **简单执行**：使用 `invoke()` 获取最终结果
2. **实时监控**：使用 `stream()` 观察执行过程
3. **高并发**：使用 `ainvoke()` 和 `astream()` 异步执行
4. **高可靠性**：使用 `durability="sync"` 确保数据持久化
