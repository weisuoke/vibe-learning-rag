---
type: context7_documentation
library: langgraph
version: latest (2026-02)
fetched_at: 2026-02-28
knowledge_point: 05_动态图构建
context7_query: Send Command dynamic routing conditional edges map reduce
---

# Context7 文档：LangGraph 动态图构建

## 文档来源
- 库名称：LangGraph
- 版本：latest
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/graph-api

## 一、Send API - 动态边

### 核心概念
Send API 允许节点返回多个边，每个边可以携带不同的状态。适用于 map-reduce 模式，下游任务数量在编译时未知。

### 基本用法
```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

### 关键特性
- 接受两个参数：目标节点名 + 传递的状态
- 支持返回 Send 对象列表实现并行调用
- 每个 Send 可以携带不同的状态
- 与条件边配合使用

## 二、Command API - 组合状态更新与路由

### 核心概念
Command API 允许节点在同一个函数调用中同时执行状态更新和确定下一个节点。

### 基本用法
```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="my_other_node"
    )
```

### 条件路由
```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

### 重要注意事项
- Command 的 goto 不会阻止静态边的执行
- 如果节点有静态边 + Command goto，两者都会执行
- 使用 `Command[Literal["node_name"]]` 类型注解帮助可视化

### 父图导航
```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",
        graph=Command.PARENT
    )
```
- `graph=Command.PARENT` 导航到最近的父图
- 共享状态键必须在父图中定义 reducer
- 适用于多代理切换场景

## 三、条件边（Conditional Edges）

### 基本用法
```python
graph.add_conditional_edges("node_a", routing_function)
```

### 带路径映射
```python
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
```

### 多路返回
```python
def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]
```
- 返回序列可以同时路由到多个节点

### LLM 驱动的路由
```python
class Route(BaseModel):
    step: Literal["poem", "story", "joke"]

router = llm.with_structured_output(Route)

def llm_call_router(state: State):
    decision = router.invoke([
        SystemMessage(content="Route the input..."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

router_builder.add_conditional_edges("llm_call_router", route_decision, {...})
```

## 四、并行执行

### Fan-out / Fan-in 模式
```python
builder.add_edge(START, "fetch_news")
builder.add_edge(START, "fetch_weather")
builder.add_edge(START, "fetch_stocks")
builder.add_edge("fetch_news", "combine_data")
builder.add_edge("fetch_weather", "combine_data")
builder.add_edge("fetch_stocks", "combine_data")
```

### 状态 Reducer 处理并行结果
```python
class State(TypedDict):
    aggregate: Annotated[list, operator.add]  # append-only
```
- `operator.add` reducer 确保并行分支的结果被合并而非覆盖

## 关键信息提取

### 动态图构建的三种模式
1. **Send 模式**：运行时动态创建并行任务（数量未知）
2. **Command 模式**：节点内部动态路由 + 状态更新
3. **条件边模式**：基于状态的运行时路由决策

### 适用场景
- Map-Reduce 工作流（Send）
- 多代理切换（Command + 父图导航）
- LLM 驱动的智能路由（条件边 + 结构化输出）
- 并行数据处理（Fan-out/Fan-in）
