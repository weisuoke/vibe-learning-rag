---
type: context7_documentation
library: langgraph
version: main
fetched_at: 2026-02-26
knowledge_point: 05_Annotated字段
context7_query: Annotated reducer state annotation TypedDict
---

# Context7 文档：LangGraph Annotated 字段

## 文档来源
- 库名称：LangGraph
- 版本：main (最后更新: 2026-02-17)
- Library ID：/langchain-ai/langgraph
- 官方文档链接：https://github.com/langchain-ai/langgraph

## 关键信息提取

### 1. 基础用法示例

#### 示例 1：使用 operator.add 作为 reducer

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # Reducer 添加新项到列表
    counter: int  # 简单覆盖
```

**关键点**：
- `Annotated[list[str], operator.add]` - 使用 `operator.add` 作为 reducer
- 没有 Annotated 的字段（如 `counter`）使用默认覆盖策略

#### 示例 2：使用 add_messages 函数

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

**关键点**：
- `add_messages` 是预定义的 reducer 函数
- 用于管理对话历史，支持按 ID 更新消息

### 2. 完整工作流示例

```python
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
import operator

# 定义状态 schema，带可选的 reducer
class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # Reducer 添加新项到列表
    counter: int  # 简单覆盖

# 定义节点函数
def first_node(state: State) -> dict:
    return {"messages": ["Hello from first node"], "counter": state["counter"] + 1}

def second_node(state: State) -> dict:
    return {"messages": ["Hello from second node"], "counter": state["counter"] + 1}

# 构建图
builder = StateGraph(State)
builder.add_node("first", first_node)
builder.add_node("second", second_node)
builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

# 使用 checkpointer 编译以实现持久化
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# 调用图
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"messages": [], "counter": 0}, config)
print(result)
# {'messages': ['Hello from first node', 'Hello from second node'], 'counter': 2}
```

**执行流程**：
1. 初始状态：`{"messages": [], "counter": 0}`
2. `first_node` 返回：`{"messages": ["Hello from first node"], "counter": 1}`
   - `messages` 使用 `operator.add`：`[] + ["Hello from first node"]` = `["Hello from first node"]`
   - `counter` 覆盖：`1`
3. `second_node` 返回：`{"messages": ["Hello from second node"], "counter": 2}`
   - `messages` 使用 `operator.add`：`["Hello from first node"] + ["Hello from second node"]`
   - `counter` 覆盖：`2`
4. 最终状态：`{"messages": ["Hello from first node", "Hello from second node"], "counter": 2}`

### 3. add_messages 函数详解

#### 基本用法

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

#### 特性

1. **按 ID 更新消息**：
   ```python
   from langchain_core.messages import AIMessage, HumanMessage

   msgs1 = [HumanMessage(content="Hello", id="1")]
   msgs2 = [AIMessage(content="Hi there!", id="2")]
   add_messages(msgs1, msgs2)
   # [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
   ```

2. **覆盖现有消息**：
   ```python
   msgs1 = [HumanMessage(content="Hello", id="1")]
   msgs2 = [HumanMessage(content="Hello again", id="1")]
   add_messages(msgs1, msgs2)
   # [HumanMessage(content='Hello again', id='1')]
   ```

3. **支持格式化**：
   ```python
   class State(TypedDict):
       messages: Annotated[list, add_messages(format="langchain-openai")]
   ```

### 4. InjectedState 注解

用于在工具中访问图的状态：

```python
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

@tool
def personalized_search(
    query: str,
    state: Annotated[dict, InjectedState]  # 注入完整状态
) -> str:
    """使用用户上下文搜索。"""
    user_id = state.get("user_id", "unknown")
    prefs = state.get("preferences", {})
    return f"Results for '{query}' (user: {user_id}, prefs: {prefs})"

@tool
def get_user_preference(
    key: str,
    preferences: Annotated[dict, InjectedState("preferences")]  # 注入特定字段
) -> str:
    """获取特定用户偏好。"""
    return preferences.get(key, "Not set")
```

### 5. 代码助手示例

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class GraphState(TypedDict):
    """
    表示图的状态。

    属性:
        error : 控制流的二进制标志，指示是否触发测试错误
        messages : 包含用户问题、错误消息、推理
        generation : 代码解决方案
        iterations : 尝试次数
    """
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

## 核心设计模式

### 1. Reducer 函数签名

```python
def reducer(old_value: T, new_value: T) -> T:
    """
    Args:
        old_value: 当前状态中的值
        new_value: 节点返回的值

    Returns:
        合并后的值
    """
    return merge(old_value, new_value)
```

### 2. 常用 Reducer 函数

| Reducer | 用途 | 示例 |
|---------|------|------|
| `operator.add` | 列表/字符串拼接 | `[1, 2] + [3]` = `[1, 2, 3]` |
| `operator.or_` | 字典合并 | `{a: 1} \| {b: 2}` = `{a: 1, b: 2}` |
| `add_messages` | 消息列表管理 | 按 ID 更新或追加 |
| `lambda x, y: y or x` | 优先使用新值 | 新值非空则使用新值 |
| `lambda x, y: y if y is not None else x` | 条件覆盖 | 新值非 None 则覆盖 |

### 3. 状态更新策略

#### 默认策略（无 Annotated）
```python
class State(TypedDict):
    counter: int  # 直接覆盖
```

#### Reducer 策略（有 Annotated）
```python
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 累积
```

## 实际应用场景

### 1. 对话系统
```python
class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
```

### 2. 代码生成系统
```python
class CodeGenState(TypedDict):
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

### 3. 多步推理系统
```python
class ReasoningState(TypedDict):
    steps: Annotated[list[str], operator.add]
    current_result: str
    confidence: float
```

## 最佳实践

1. **选择合适的 Reducer**：
   - 列表累积：`operator.add`
   - 字典合并：`operator.or_`
   - 消息管理：`add_messages`
   - 条件覆盖：自定义 lambda

2. **状态设计原则**：
   - 需要累积的数据使用 Annotated
   - 简单覆盖的数据不使用 Annotated
   - 复杂逻辑使用自定义 reducer

3. **类型安全**：
   - 使用 TypedDict 定义状态
   - 为 Annotated 字段指定正确的类型
   - 确保 reducer 函数签名正确

## 常见误区

1. ❌ **忘记使用 Annotated**：
   ```python
   class State(TypedDict):
       messages: list  # 会被覆盖，不会累积
   ```

2. ❌ **Reducer 签名错误**：
   ```python
   def bad_reducer(value):  # 缺少第二个参数
       return value
   ```

3. ❌ **类型不匹配**：
   ```python
   class State(TypedDict):
       messages: Annotated[list, operator.or_]  # or_ 用于字典，不适合列表
   ```

## 参考资源

- GitHub 示例：https://github.com/langchain-ai/langgraph/blob/main/examples/
- 官方文档：https://langchain-ai.github.io/langgraph/
- Context7 库：/langchain-ai/langgraph
