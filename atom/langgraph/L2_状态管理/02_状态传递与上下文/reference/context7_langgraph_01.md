---
type: context7_documentation
library: langgraph
version: main (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: 02_状态传递与上下文
context7_query: state passing context management channel read write
---

# Context7 文档：LangGraph 状态传递与上下文管理

## 文档来源
- 库名称：LangGraph
- 版本：main (2026-02-17)
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/
- Context7 库 ID：/websites/langchain_oss_python_langgraph
- 总代码片段：900 个
- 信任评分：10/10
- 基准评分：86.9/100

## 关键信息提取

### 1. 多状态 Schema 定义

**核心概念**：LangGraph 支持定义多个状态 Schema（InputState, OutputState, OverallState, PrivateState）

**代码示例**：
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

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
    # 从 InputState 读取，写入 OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # 从 OverallState 读取，写入 PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # 从 PrivateState 读取，写入 OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
result = graph.invoke({"user_input":"My"})
# 输出: {'graph_output': 'My name is Lance'}
```

**关键发现**：
- 节点可以从不同的状态 Schema 读取和写入
- 状态在节点间自动传递和合并
- 支持私有状态（PrivateState）不暴露给外部

### 2. Runtime Context（运行时上下文）

**核心概念**：通过 `context_schema` 定义运行时上下文，用于传递非状态数据（如配置、依赖）

**代码示例**：
```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    # 从 runtime context 获取 user_id
    user_id = runtime.context.user_id

    # 命名空间化内存
    namespace = (user_id, "memories")

    # 创建新的内存 ID
    memory_id = str(uuid.uuid4())

    # 存储内存
    await runtime.store.aput(namespace, memory_id, {"memory": memory})
```

**使用方式**：
```python
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),  # 传递 context
):
    print(update)
```

**关键发现**：
- `context_schema` 用于定义运行时上下文的类型
- 通过 `Runtime[Context]` 在节点中访问上下文
- 上下文在运行时不可修改（immutable）
- 适合传递配置、数据库连接、用户 ID 等

### 3. 节点参数类型

**核心概念**：节点可以接受三种类型的参数

**代码示例**：
```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

# 1. 只接受 state
def plain_node(state: State):
    return state

# 2. 接受 state + runtime（访问 context）
def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("In node: ", runtime.context.user_id)
    return {"results": f"Hello, {state['input']}!"}

# 3. 接受 state + config（访问配置）
def node_with_config(state: State, config: RunnableConfig):
    print("In node with thread_id: ", config["configurable"]["thread_id"])
    return {"results": f"Hello, {state['input']}!"}

builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_config", node_with_config)
```

**关键发现**：
- **state**：图的状态数据
- **runtime**：运行时上下文（通过 `context_schema` 定义）
- **config**：RunnableConfig 对象（包含 thread_id、tags 等）

### 4. Context Schema 定义与使用

**核心概念**：在创建图时指定 `context_schema`，在节点中通过 `Runtime[ContextSchema]` 访问

**代码示例**：
```python
from langgraph.graph import END, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

# 1. 定义 context schema
class ContextSchema(TypedDict):
    my_runtime_value: str

# 2. 定义状态
class State(TypedDict):
    my_state_value: str

# 3. 在节点中访问 context
def node(state: State, runtime: Runtime[ContextSchema]):
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        raise ValueError("Unknown values.")

# 4. 创建图时指定 context_schema
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()
```

**关键发现**：
- `context_schema` 在图创建时指定
- 节点通过 `Runtime[ContextSchema]` 类型注解访问
- 支持运行时动态传递不同的 context 值

### 5. 动态 LLM 选择（Runtime Context 应用）

**核心概念**：使用 Runtime Context 在运行时动态选择 LLM

**代码示例**：
```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.runtime import Runtime

@dataclass
class ContextSchema:
    model_provider: str = "anthropic"

MODELS = {
    "anthropic": init_chat_model("claude-haiku-4-5-20251001"),
    "openai": init_chat_model("gpt-4.1-mini"),
}

def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
    model = MODELS[runtime.context.model_provider]
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState, context_schema=ContextSchema)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# 使用默认 context（Anthropic）
response_1 = graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    context=ContextSchema()
)

# 使用 OpenAI
response_2 = graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    context={"model_provider": "openai"}
)
```

**关键发现**：
- Runtime Context 适合传递运行时配置
- 无需重新编译图即可改变行为
- 支持字典和 dataclass 两种方式传递 context

### 6. Functional API 中的 Injectable Parameters

**核心概念**：在 Functional API 中，可以请求注入的参数

**代码示例**：
```python
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StreamWriter

in_memory_checkpointer = InMemorySaver(...)
in_memory_store = InMemoryStore(...)

@entrypoint(
    checkpointer=in_memory_checkpointer,
    store=in_memory_store
)
def my_workflow(
    some_input: dict,  # 输入（通过 invoke 传递）
    *,
    previous: Any = None,  # 短期记忆
    store: BaseStore,  # 长期记忆
    writer: StreamWriter,  # 流式输出
    config: RunnableConfig  # 配置
) -> ...:
    ...
```

**关键发现**：
- **previous**：短期记忆（checkpoint）
- **store**：长期记忆（持久化存储）
- **writer**：流式输出
- **config**：运行时配置

### 7. 共享状态管理

**核心概念**：多个节点可以访问和修改共享状态

**代码示例**：
```python
class WorkflowState(TypedDict):
    user_input: str
    search_results: list
    generated_response: str
    validation_status: str

def search_node(state):
    # 访问共享状态
    results = search(state["user_input"])
    return {"search_results": results}

def validation_node(state):
    # 访问前一个节点的结果
    is_valid = validate(state["generated_response"])
    return {"validation_status": "valid" if is_valid else "invalid"}
```

**关键发现**：
- 所有节点共享同一个状态对象
- 节点返回部分状态更新
- 状态自动合并（通过 Reducer 函数）

## 核心设计模式

### 1. 状态传递模式
- **输入状态** → **节点处理** → **输出状态**
- 节点只需返回部分状态更新
- 框架自动处理状态合并

### 2. 上下文传递模式
- **context_schema** 定义上下文类型
- **Runtime[Context]** 在节点中访问
- **context 参数** 在 invoke/stream 时传递

### 3. 配置传递模式
- **RunnableConfig** 包含运行时配置
- **thread_id**：线程标识
- **tags**：追踪标签
- **configurable**：自定义配置

## 与 LangChain 的集成

- LangGraph 基于 LangChain 的 Runnable 接口
- RunnableConfig 来自 langchain_core.runnables.config
- 节点可以是任何 Runnable（包括 LangChain 的 Chain）

## 最佳实践

1. **状态设计**：
   - 使用 TypedDict 定义状态类型
   - 区分 InputState、OverallState、OutputState
   - 使用 PrivateState 隐藏内部状态

2. **上下文使用**：
   - 用于传递不可变的运行时配置
   - 适合传递数据库连接、用户 ID、模型配置
   - 不要在 context 中传递会变化的数据

3. **节点设计**：
   - 根据需要选择参数类型（state、runtime、config）
   - 只返回需要更新的状态字段
   - 保持节点函数的纯粹性

4. **类型安全**：
   - 使用 TypedDict 定义所有 Schema
   - 使用 Runtime[Context] 类型注解
   - 利用 Python 类型检查工具
