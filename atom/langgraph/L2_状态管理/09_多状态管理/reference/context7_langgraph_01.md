---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 09_多状态管理
context7_query: multi-state management nested state subgraph state input output schema separation
---

# Context7 文档：LangGraph 多状态管理

## 文档来源
- 库名称：LangGraph
- Context7 ID：/websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph

## 关键信息提取

### 1. Input/Output Schema 分离

LangGraph 支持为图定义独立的输入和输出 Schema：

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    pass

def answer_node(state: InputState):
    return {"answer": "bye", "question": state["question"]}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()

print(graph.invoke({"question": "hi"}))
# 输出只包含 OutputState 的字段: {'answer': 'bye'}
```

### 2. 多 Schema 协作（含私有状态）

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
# ... add nodes and edges ...
result = graph.invoke({"user_input": "My"})
# {'graph_output': 'My name is Lance'}
```

### 3. 子图状态隔离（不同 Schema）

```python
class SubgraphState(TypedDict):
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
# ... build subgraph ...
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_2(state: ParentState):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

builder = StateGraph(ParentState)
# ... build parent graph ...
```

### 4. 子图共享键通信

```python
class SubgraphState(TypedDict):
    foo: str  # 与父图共享
    bar: str  # 子图私有

class ParentState(TypedDict):
    foo: str

# 当子图作为编译图直接添加为节点时，
# 共享键（foo）自动在父子图之间传递
builder = StateGraph(ParentState)
builder.add_node("node_2", subgraph)  # 直接传入编译后的子图
```

### 5. 多层嵌套子图

LangGraph 支持多层嵌套：parent -> child -> grandchild
每层有独立的 TypedDict 状态，数据通过显式转换传递。

### 6. context_schema 与 Runtime

```python
from langgraph.runtime import Runtime

class ContextSchema(TypedDict):
    my_runtime_value: str

class State(TypedDict):
    my_state_value: str

def node(state: State, runtime: Runtime[ContextSchema]):
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}

builder = StateGraph(State, context_schema=ContextSchema)
```
