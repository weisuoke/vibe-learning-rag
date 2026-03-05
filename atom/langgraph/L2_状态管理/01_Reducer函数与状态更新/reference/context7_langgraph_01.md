---
type: context7_documentation
library: langgraph
version: main (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: Reducer函数与状态更新
context7_query: reducer function state management annotated
library_id: /langchain-ai/langgraph
---

# Context7 文档：LangGraph Reducer 函数与状态管理

## 文档来源
- 库名称: LangGraph
- Library ID: `/langchain-ai/langgraph`
- 版本: main branch
- 最后更新: 2026-02-17
- Trust Score: 9.2
- Benchmark Score: 77.5
- 总代码片段: 234 个

## 关键信息提取

### 1. Reducer 函数的基本使用

#### 使用 add_messages Reducer (官方推荐)

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

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

**关键点**:
- `messages` 字段使用 `Annotated` 绑定 `add_messages` Reducer
- `add_messages` 用于管理对话历史
- 适用于聊天机器人、代码助手等场景

#### 使用 operator.add Reducer

**来源**: https://context7.com/langchain-ai/langgraph/llms.txt

```python
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
import operator

# Define state schema with optional reducer for list aggregation
class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # Reducer adds new items to list
    counter: int  # Simple overwrite

# Define node functions
def first_node(state: State) -> dict:
    return {"messages": ["Hello from first node"], "counter": state["counter"] + 1}

def second_node(state: State) -> dict:
    return {"messages": ["Hello from second node"], "counter": state["counter"] + 1}

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

**关键点**:
- `operator.add` 用于列表拼接
- `counter` 字段没有 Reducer,使用简单覆盖策略
- 展示了 Reducer 和非 Reducer 字段的对比

### 2. add_messages 的详细说明

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**官方解释**:
- **默认行为**: 替换 (replace)
- **add_messages 行为**: 追加 (append)
- **用途**: 启用对话历史跟踪

### 3. InjectedState 与状态访问

**来源**: https://context7.com/langchain-ai/langgraph/llms.txt

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    preferences: dict

@tool
def personalized_search(
    query: str,
    state: Annotated[dict, InjectedState]  # Full state injected
) -> str:
    """Search with user context."""
    user_id = state.get("user_id", "unknown")
    prefs = state.get("preferences", {})
    return f"Results for '{query}' (user: {user_id}, prefs: {prefs})"

@tool
def get_user_preference(
    key: str,
    preferences: Annotated[dict, InjectedState("preferences")]  # Specific field
) -> str:
    """Get a specific user preference."""
    return preferences.get(key, "Not set")

tool_node = ToolNode([personalized_search, get_user_preference])

# Test direct invocation
from langchain_core.messages import AIMessage

state = {
    "messages": [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "personalized_search", "args": {"query": "restaurants"}, "id": "1"},
                {"name": "get_user_preference", "args": {"key": "cuisine"}, "id": "2"}
            ]
        )
    ],
    "user_id": "user-123",
    "preferences": {"cuisine": "italian", "budget": "moderate"}
}

result = tool_node.invoke(state)
for msg in result["messages"]:
    print(f"{msg.name}: {msg.content}")
```

**关键点**:
- `InjectedState` 允许工具访问图的状态
- 可以注入完整状态或特定字段
- 不会将状态暴露给 LLM

### 4. 实际应用场景

#### 场景 1: 代码生成与自我修正

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb

```python
class GraphState(TypedDict):
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

**应用**:
- 跟踪错误信息
- 累积对话历史
- 记录生成的代码
- 计数迭代次数

#### 场景 2: RAG 代理

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**应用**:
- 维护用户问题和 AI 回答的历史
- 支持多轮对话
- 上下文保持

## 技术要点总结

### 1. Reducer 的作用
- **默认行为**: 状态字段的值会被新值替换
- **Reducer 行为**: 使用自定义函数合并旧值和新值
- **常见 Reducer**:
  - `add_messages`: 消息列表合并 (按 ID)
  - `operator.add`: 列表拼接、字符串拼接、数值相加
  - `operator.or_`: 字典合并

### 2. Annotated 的使用
```python
# 语法
field_name: Annotated[type, reducer_function]

# 示例
messages: Annotated[list[str], operator.add]
messages: Annotated[list[AnyMessage], add_messages]
```

### 3. 状态更新策略
- **有 Reducer**: 合并旧值和新值
- **无 Reducer**: 直接替换为新值

### 4. 最佳实践
- 对于消息列表,使用 `add_messages`
- 对于简单列表拼接,使用 `operator.add`
- 对于字典合并,使用 `operator.or_`
- 对于计数器等简单值,不使用 Reducer

## 官方文档链接

1. **Code Assistant Example**: https://github.com/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant_mistral.ipynb
2. **RAG Agent Example**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
3. **LangGraph LLMs.txt**: https://context7.com/langchain-ai/langgraph/llms.txt

## 版本信息

- **最后更新**: 2026-02-17
- **分支**: main
- **总代码片段**: 234 个
- **Trust Score**: 9.2/10
- **Benchmark Score**: 77.5/100
