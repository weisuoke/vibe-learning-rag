---
type: context7_documentation
library: langgraph
version: main (latest)
fetched_at: 2026-02-26
knowledge_point: 03_部分状态更新
context7_query: partial state update reducer annotated field
library_id: /langchain-ai/langgraph
---

# Context7 文档：LangGraph 部分状态更新

## 文档来源
- 库名称：LangGraph
- 版本：main (latest)
- 官方文档链接：https://github.com/langchain-ai/langgraph
- Context7 库 ID：/langchain-ai/langgraph
- 总代码片段：234 个
- 信任评分：9.2/10
- 基准评分：77.5/100

## 关键信息提取

### 1. Annotated 字段与 Reducer 函数

**核心概念：使用 Annotated 定义字段的更新策略**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # Reducer adds new items to list
    counter: int  # Simple overwrite
```

**来源：** https://context7.com/langchain-ai/langgraph/llms.txt

**关键特性：**
- `Annotated[list[str], operator.add]` - 使用 `operator.add` 作为 Reducer，追加新元素到列表
- `counter: int` - 没有 Reducer，使用简单覆盖策略
- Reducer 定义了如何处理状态更新

### 2. add_messages 函数

**核心概念：消息累积而非替换**

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**来源：** https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

**关键特性：**
- `add_messages` 函数定义了如何处理更新
- 默认行为是替换（replace）
- `add_messages` 指定追加（append）行为
- 用于维护对话历史

### 3. 节点函数返回部分状态

**核心概念：节点函数可以只返回部分字段**

```python
def first_node(state: State) -> dict:
    return {"messages": ["Hello from first node"], "counter": state["counter"] + 1}

def second_node(state: State) -> dict:
    return {"messages": ["Hello from second node"], "counter": state["counter"] + 1}
```

**来源：** https://context7.com/langchain-ai/langgraph/llms.txt

**关键特性：**
- 节点函数返回字典（dict）
- 只需要返回需要更新的字段
- 未返回的字段保持原值不变

### 4. StateGraph 工作原理

**核心概念：节点通过共享状态通信**

```python
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# Build the graph
builder = StateGraph(State)
builder.add_node("first", first_node)
builder.add_node("second", second_node)
builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

# Compile with checkpointer for persistence
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# Invoke the graph
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"messages": [], "counter": 0}, config)
print(result)
# {'messages': ['Hello from first node', 'Hello from second node'], 'counter': 2}
```

**来源：** https://context7.com/langchain-ai/langgraph/llms.txt

**关键特性：**
- `StateGraph` 是主要的构建器类
- 节点是接收当前状态并返回部分更新的 Python 函数
- 状态模式可以为字段定义 Reducer（如列表聚合）
- 其他字段使用简单覆盖语义（如计数器）
- 必须使用 `compile()` 方法编译图
- 使用 checkpointer 实现持久化

### 5. 状态更新语义

**核心概念：不同字段的更新策略**

**默认策略（覆盖）：**
```python
class State(TypedDict):
    counter: int  # 简单覆盖
```

**Reducer 策略（聚合）：**
```python
class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # 追加到列表
```

**来源：** https://context7.com/langchain-ai/langgraph/llms.txt

**关键特性：**
- 没有 Reducer 的字段：使用覆盖语义
- 有 Reducer 的字段：使用聚合语义
- Reducer 函数接收两个参数：当前值和新值
- 返回合并后的值

### 6. 实际应用示例

**示例 1：代码助手状态**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

**来源：** https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

**关键特性：**
- `messages` 字段使用 `add_messages` Reducer
- `error`, `generation`, `iterations` 字段使用覆盖策略
- 用于迭代代码修正工作流

**示例 2：RAG 代理状态**

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**来源：** https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

**关键特性：**
- 简单的状态定义
- 只有一个字段 `messages`
- 使用 `add_messages` 维护对话历史

## 核心概念总结

### 1. 部分状态返回
- 节点函数返回字典（dict）
- 只需要返回需要更新的字段
- 未返回的字段保持原值不变

### 2. Reducer 函数
- 使用 `Annotated[type, reducer]` 定义
- 常用 Reducer：`operator.add`, `add_messages`
- Reducer 接收当前值和新值，返回合并后的值

### 3. 更新策略
- **默认策略**：覆盖（没有 Reducer）
- **Reducer 策略**：聚合（有 Reducer）

### 4. add_messages 函数
- 专门用于消息列表的 Reducer
- 追加新消息而不是替换
- 用于维护对话历史

### 5. StateGraph 工作原理
- 节点通过共享状态通信
- 节点函数接收当前状态，返回部分更新
- 状态更新根据 Reducer 定义自动合并

## 技术点识别

1. **Annotated 类型注解** - 使用 `Annotated[type, reducer]` 定义字段的更新策略
2. **operator.add Reducer** - 使用 `operator.add` 实现列表追加
3. **add_messages Reducer** - 专门用于消息列表的 Reducer
4. **部分状态返回** - 节点函数只返回需要更新的字段
5. **状态合并机制** - 根据 Reducer 定义自动合并状态

## 依赖库识别

1. **typing** - 标准库，提供 `Annotated`, `TypedDict` 等类型注解
2. **typing_extensions** - 扩展类型注解支持
3. **operator** - 标准库，提供 `operator.add` 等操作符函数
4. **langgraph.graph.message** - 提供 `add_messages` 函数
5. **langchain_core.messages** - 提供消息类型（`BaseMessage`, `AnyMessage`）
