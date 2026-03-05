# 核心概念2：State Schema 定义

> 本文档详细讲解 LangGraph 中 State Schema 的定义方式，包括 TypedDict、Annotated 类型与 reducer 函数、以及 NotRequired 可选字段。

**来源标注：**
- 源码分析：`sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py`
- Context7 官方文档：https://docs.langchain.com/oss/python/langgraph/
- 社区实践：GitHub、Twitter、Reddit

---

## 核心概念1：TypedDict 状态定义

**TypedDict 是 LangGraph 状态的基础类型，提供类型安全的字典结构定义。**

### 基础语法

```python
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[str]
    user_id: str
    count: int
```

**来源：** Context7 官方文档 - https://docs.langchain.com/oss/python/langgraph/use-graph-api

### 为什么使用 TypedDict？

**1. 类型安全**
- 编译时类型检查
- IDE 自动补全支持
- 减少运行时错误

**2. 文档化**
- 状态结构一目了然
- 字段类型明确
- 便于团队协作

**3. 与 LangGraph 集成**
- StateGraph 自动推断状态结构
- 节点函数参数类型提示
- 编译时验证状态字段

### 完整示例

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态结构
class AgentState(TypedDict):
    input: str           # 用户输入
    output: str          # 最终输出
    intermediate: list   # 中间结果
    step_count: int      # 步骤计数

# 节点函数使用类型化状态
def process_node(state: AgentState) -> dict:
    """节点函数接受类型化状态，返回部分更新"""
    return {
        "intermediate": state["intermediate"] + ["processed"],
        "step_count": state["step_count"] + 1
    }

# 创建图
builder = StateGraph(AgentState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# 执行
result = graph.invoke({
    "input": "hello",
    "output": "",
    "intermediate": [],
    "step_count": 0
})
print(result)
# {'input': 'hello', 'output': '', 'intermediate': ['processed'], 'step_count': 1}
```

**来源：** Context7 官方文档示例改编

### 在 LangGraph 中的实际应用

**场景1：多步推理系统**
```python
class ReasoningState(TypedDict):
    question: str
    thoughts: list[str]
    answer: str
    confidence: float
```

**场景2：文档处理流程**
```python
class DocumentState(TypedDict):
    raw_text: str
    chunks: list[str]
    embeddings: list[list[float]]
    metadata: dict
```

**场景3：多代理协作**
```python
class MultiAgentState(TypedDict):
    task: str
    agent_outputs: dict[str, str]
    final_result: str
    coordinator_decision: str
```

**来源：** 社区实践案例（Reddit、GitHub）

---

## 核心概念2：Annotated 与 Reducer 函数

**Annotated 类型允许为状态字段添加 reducer 函数，控制状态更新策略。**

### 基础语法

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    # 使用 operator.add 作为 reducer，实现列表追加
    messages: Annotated[list, operator.add]
```

**来源：** Context7 官方文档 - https://docs.langchain.com/oss/python/langgraph/use-graph-api

### Reducer 函数的作用

**默认行为（无 Reducer）：**
```python
class State(TypedDict):
    value: int

# 节点返回 {"value": 5}
# 状态更新：直接覆盖 state["value"] = 5
```

**使用 Reducer：**
```python
def add_reducer(current: int, new: int) -> int:
    return current + new

class State(TypedDict):
    value: Annotated[int, add_reducer]

# 节点返回 {"value": 5}
# 状态更新：state["value"] = add_reducer(state["value"], 5)
```

**来源：** 源码分析 - state.py:141-180

### 常用 Reducer 模式

**1. 列表追加（operator.add）**
```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    logs: Annotated[list, operator.add]

# 节点1返回 {"logs": ["step1"]}
# 节点2返回 {"logs": ["step2"]}
# 最终状态：{"logs": ["step1", "step2"]}
```

**2. 自定义 Reducer**
```python
def merge_dicts(current: dict, new: dict) -> dict:
    """合并字典，保留所有键"""
    return {**current, **new}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

**3. 条件更新**
```python
def max_reducer(current: float, new: float) -> float:
    """只保留最大值"""
    return max(current, new)

class State(TypedDict):
    confidence: Annotated[float, max_reducer]
```

**来源：** Context7 官方文档 + 社区实践

### 完整实战示例

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 自定义 reducer：累加计数
def add_count(current: int, new: int | None) -> int:
    if new is None:
        return current
    return current + new

# 定义状态
class State(TypedDict):
    # 列表追加
    aggregate: Annotated[list, operator.add]
    # 自定义累加
    total: Annotated[int, add_count]

def node_a(state: State):
    print(f'Node A sees aggregate: {state["aggregate"]}, total: {state["total"]}')
    return {"aggregate": ["A"], "total": 1}

def node_b(state: State):
    print(f'Node B sees aggregate: {state["aggregate"]}, total: {state["total"]}')
    return {"aggregate": ["B"], "total": 2}

# 构建图
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()

# 执行
result = graph.invoke({"aggregate": [], "total": 0})
print(f"\nFinal result: {result}")
# Node A sees aggregate: [], total: 0
# Node B sees aggregate: ['A'], total: 1
# Final result: {'aggregate': ['A', 'B'], 'total': 3}
```

**来源：** Context7 官方文档示例改编

### 在 LangGraph 中的实际应用

**场景1：对话历史管理**
```python
from langchain_core.messages import BaseMessage

def add_messages(current: list[BaseMessage], new: list[BaseMessage]) -> list[BaseMessage]:
    """合并消息列表，去重"""
    seen = {msg.id for msg in current}
    return current + [msg for msg in new if msg.id not in seen]

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**场景2：多路径结果聚合**
```python
def merge_results(current: dict, new: dict) -> dict:
    """合并多个代理的结果"""
    for key, value in new.items():
        if key in current:
            current[key].append(value)
        else:
            current[key] = [value]
    return current

class MultiPathState(TypedDict):
    results: Annotated[dict, merge_results]
```

**来源：** 社区实践案例（Twitter、Reddit）

---

## 核心概念3：NotRequired 可选字段

**NotRequired 标记可选字段，允许状态字段在初始化时不提供值。**

### 基础语法

```python
from typing_extensions import TypedDict, NotRequired

class State(TypedDict):
    # 必需字段
    input: str
    # 可选字段
    output: NotRequired[str]
    intermediate: NotRequired[list]
```

**来源：** Context7 官方文档 - https://docs.langchain.com/oss/python/langgraph/use-time-travel

### 为什么需要 NotRequired？

**1. 渐进式状态构建**
- 初始状态只包含必需字段
- 节点逐步添加可选字段
- 避免初始化时提供所有字段

**2. 条件性字段**
- 某些字段只在特定路径下生成
- 避免为所有路径提供默认值

**3. 类型安全**
- 明确区分必需和可选字段
- IDE 提示更准确

### 完整示例

```python
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: NotRequired[str]      # 可选：生成的主题
    joke: NotRequired[str]        # 可选：生成的笑话

def generate_topic(state: State):
    """生成主题"""
    return {"topic": "programming"}

def write_joke(state: State):
    """基于主题写笑话"""
    topic = state.get("topic", "general")
    return {"joke": f"A joke about {topic}"}

# 构建图
builder = StateGraph(State)
builder.add_node("generate_topic", generate_topic)
builder.add_node("write_joke", write_joke)
builder.add_edge(START, "generate_topic")
builder.add_edge("generate_topic", "write_joke")
builder.add_edge("write_joke", END)

graph = builder.compile()

# 执行 - 初始状态为空字典
result = graph.invoke({})
print(result)
# {'topic': 'programming', 'joke': 'A joke about programming'}
```

**来源：** Context7 官方文档示例改编

### NotRequired vs 默认值

**使用 NotRequired：**
```python
class State(TypedDict):
    count: NotRequired[int]

# 初始化
graph.invoke({})  # ✅ 合法
```

**使用默认值（不推荐）：**
```python
class State(TypedDict):
    count: int  # 必需字段

# 初始化
graph.invoke({"count": 0})  # 必须提供
```

**推荐做法：**
- 使用 NotRequired 标记可选字段
- 在节点函数中使用 `state.get(key, default)` 安全访问

**来源：** 社区最佳实践（Twitter）

### 在 LangGraph 中的实际应用

**场景1：多步生成流程**
```python
class GenerationState(TypedDict):
    prompt: str                          # 必需：用户输入
    outline: NotRequired[str]            # 可选：大纲
    draft: NotRequired[str]              # 可选：草稿
    final: NotRequired[str]              # 可选：最终版本
    revisions: NotRequired[list[str]]    # 可选：修订历史
```

**场景2：条件路由状态**
```python
class RoutingState(TypedDict):
    query: str                           # 必需：查询
    route: NotRequired[str]              # 可选：路由决策
    result: NotRequired[str]             # 可选：结果
    error: NotRequired[str]              # 可选：错误信息
```

**场景3：人机协作**
```python
class HumanLoopState(TypedDict):
    task: str                            # 必需：任务描述
    ai_suggestion: NotRequired[str]      # 可选：AI 建议
    human_feedback: NotRequired[str]     # 可选：人类反馈
    final_decision: NotRequired[str]     # 可选：最终决策
```

**来源：** 社区实践案例（Reddit、GitHub）

---

## 综合实战示例

### 场景：多代理内容生成系统

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END

# 定义状态
class ContentState(TypedDict):
    # 必需字段
    topic: str
    # 可选字段
    outline: NotRequired[str]
    draft: NotRequired[str]
    final: NotRequired[str]
    # 使用 reducer 的字段
    feedback: Annotated[list[str], operator.add]
    revision_count: Annotated[int, lambda a, b: a + b]

def create_outline(state: ContentState):
    """创建大纲"""
    return {
        "outline": f"Outline for {state['topic']}",
        "feedback": ["Outline created"]
    }

def write_draft(state: ContentState):
    """写草稿"""
    return {
        "draft": f"Draft based on: {state.get('outline', 'no outline')}",
        "feedback": ["Draft written"],
        "revision_count": 1
    }

def finalize(state: ContentState):
    """最终化"""
    return {
        "final": f"Final version of: {state.get('draft', 'no draft')}",
        "feedback": ["Content finalized"],
        "revision_count": 1
    }

# 构建图
builder = StateGraph(ContentState)
builder.add_node("outline", create_outline)
builder.add_node("draft", write_draft)
builder.add_node("finalize", finalize)

builder.add_edge(START, "outline")
builder.add_edge("outline", "draft")
builder.add_edge("draft", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()

# 执行
result = graph.invoke({
    "topic": "LangGraph Tutorial",
    "feedback": [],
    "revision_count": 0
})

print("Final State:")
print(f"Topic: {result['topic']}")
print(f"Outline: {result.get('outline', 'N/A')}")
print(f"Draft: {result.get('draft', 'N/A')}")
print(f"Final: {result.get('final', 'N/A')}")
print(f"Feedback: {result['feedback']}")
print(f"Revision Count: {result['revision_count']}")
```

**预期输出：**
```
Final State:
Topic: LangGraph Tutorial
Outline: Outline for LangGraph Tutorial
Draft: Draft based on: Outline for LangGraph Tutorial
Final: Final version of: Draft based on: Outline for LangGraph Tutorial
Feedback: ['Outline created', 'Draft written', 'Content finalized']
Revision Count: 2
```

**来源：** 综合示例（基于 Context7 文档和社区实践）

---

## 最佳实践总结

### 1. TypedDict 使用建议

**✅ 推荐：**
- 为所有状态使用 TypedDict
- 字段命名清晰有意义
- 添加类型注解

**❌ 避免：**
- 使用普通 dict
- 字段名过于简短（如 `x`, `y`）
- 缺少类型注解

### 2. Reducer 函数使用建议

**✅ 推荐：**
- 列表追加使用 `operator.add`
- 复杂逻辑使用自定义 reducer
- Reducer 函数保持纯函数

**❌ 避免：**
- Reducer 函数有副作用
- 过度复杂的 reducer 逻辑
- 忘记处理 None 值

### 3. NotRequired 使用建议

**✅ 推荐：**
- 标记所有可选字段
- 使用 `state.get(key, default)` 安全访问
- 在节点函数中检查字段存在性

**❌ 避免：**
- 所有字段都标记为 NotRequired
- 直接访问可选字段（`state["key"]`）
- 混淆必需和可选字段

**来源：** 社区最佳实践（Twitter、Reddit）

---

## 常见陷阱

### 陷阱1：忘记 Reducer 导致状态覆盖

**错误示例：**
```python
class State(TypedDict):
    messages: list  # 没有 reducer

# 节点1返回 {"messages": ["A"]}
# 节点2返回 {"messages": ["B"]}
# 最终状态：{"messages": ["B"]}  # ❌ 覆盖了！
```

**正确示例：**
```python
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 使用 reducer

# 最终状态：{"messages": ["A", "B"]}  # ✅ 追加
```

### 陷阱2：直接访问 NotRequired 字段

**错误示例：**
```python
def node(state: State):
    topic = state["topic"]  # ❌ KeyError if not present
```

**正确示例：**
```python
def node(state: State):
    topic = state.get("topic", "default")  # ✅ 安全访问
```

### 陷阱3：Reducer 函数有副作用

**错误示例：**
```python
external_list = []

def bad_reducer(current: list, new: list) -> list:
    external_list.extend(new)  # ❌ 副作用
    return current + new
```

**正确示例：**
```python
def good_reducer(current: list, new: list) -> list:
    return current + new  # ✅ 纯函数
```

**来源：** 社区常见问题（Reddit）

---

## 参考资源

### 官方文档
- LangGraph State Schema: https://docs.langchain.com/oss/python/langgraph/use-graph-api
- TypedDict 文档: https://docs.python.org/3/library/typing.html#typing.TypedDict

### 源码参考
- `langgraph/graph/state.py`: StateGraph 核心实现
- `langgraph/graph/_node.py`: 节点协议定义

### 社区资源
- GitHub 示例: https://github.com/langchain-ai/langgraph/tree/main/examples
- LangChain Academy: https://github.com/langchain-ai/langchain-academy

---

**文档版本：** v1.0
**最后更新：** 2026-02-25
**维护者：** Claude Code
