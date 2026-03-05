# 03_核心概念_1 - Send 类与动态并行

> LangGraph 并行执行的核心机制：通过 Send 类动态创建并行任务

---

## 概念定义

**Send 类是 LangGraph 中实现动态并行执行的核心类，用于在条件边中动态调用节点，并可以发送自定义状态到目标节点。**

Send 类使得我们可以在运行时根据状态动态决定：
1. 要并行执行哪些节点
2. 每个节点接收什么输入
3. 并行任务的数量

这是 LangGraph 区别于传统工作流引擎的关键特性之一。

---

## Send 类的定义与属性

### 源码定义

```python
# 来源：langgraph/types.py:289-362
class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.

    One such example is a "map-reduce" workflow where your graph invokes
    the same node multiple times in parallel with different states,
    before aggregating the results back into the main graph's state.
    """

    __slots__ = ("node", "arg")

    node: str  # 目标节点名称
    arg: Any   # 发送的状态或消息
```

### 核心属性

#### 1. `node` - 目标节点名称

**类型**: `str`

**作用**: 指定要调用的目标节点

**示例**:
```python
Send("generate_joke", {...})  # 调用 "generate_joke" 节点
Send("process_doc", {...})    # 调用 "process_doc" 节点
```

#### 2. `arg` - 输入状态

**类型**: `Any`（通常是字典）

**作用**: 传递给目标节点的输入状态

**关键特性**:
- 可以与图的核心状态不同
- 支持部分状态传递
- 允许自定义数据结构

**示例**:
```python
# 传递完整状态
Send("node", state)

# 传递部分状态
Send("node", {"subject": "cats", "style": "funny"})

# 传递单个值
Send("node", {"doc": documents[0]})
```

---

## 动态创建并行任务的机制

### 基本原理

**核心思想**: 在条件边函数中返回多个 `Send` 对象，LangGraph 会自动并行执行这些 Send 对象对应的节点。

### 实现步骤

#### 步骤 1: 定义状态 Schema

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class OverallState(TypedDict):
    # 输入数据
    subjects: list[str]

    # 并行任务的结果（使用 reducer 自动合并）
    jokes: Annotated[list[str], operator.add]

    # 最终结果
    best_joke: str
```

**关键点**:
- `Annotated[list[str], operator.add]` 确保并行任务的结果被追加而不是覆盖
- `operator.add` 是 reducer 函数，用于合并并行任务的结果

#### 步骤 2: 定义条件边函数

```python
def continue_to_jokes(state: OverallState):
    """
    根据 subjects 列表动态创建并行任务
    返回多个 Send 对象，每个对应一个并行任务
    """
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]
```

**关键点**:
- 返回 `Send` 对象列表
- 每个 `Send` 对象指定目标节点和输入状态
- 列表长度决定并行任务数量（运行时确定）

#### 步骤 3: 定义并行执行的节点

```python
def generate_joke(state: dict):
    """
    单个并行任务的实现
    接收 Send 传递的状态，返回结果
    """
    subject = state["subject"]

    # 调用 LLM 生成笑话
    joke = llm.invoke(f"Tell me a joke about {subject}")

    # 返回结果（会被 reducer 自动合并）
    return {"jokes": [joke.content]}
```

**关键点**:
- 接收的 `state` 是 `Send` 传递的 `arg`
- 返回的字典会被合并到全局状态
- 使用 `{"jokes": [joke]}` 而不是 `{"jokes": joke}` 以配合 `operator.add`

#### 步骤 4: 构建图

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(OverallState)

# 添加节点
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", select_best_joke)

# 添加边
builder.add_edge(START, "generate_topics")

# 关键：添加条件边，返回 Send 对象列表
builder.add_conditional_edges(
    "generate_topics",
    continue_to_jokes,
    ["generate_joke"]  # 可能的目标节点列表
)

builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

graph = builder.compile()
```

**关键点**:
- `add_conditional_edges` 的第二个参数是条件函数
- 条件函数返回 `Send` 对象列表时，触发并行执行
- 第三个参数是可能的目标节点列表（用于验证）

---

## 与条件边的配合使用

### 条件边的三种返回方式

#### 方式 1: 返回节点名称字符串（串行）

```python
def route(state):
    if state["count"] < 10:
        return "continue"
    else:
        return "finish"

builder.add_conditional_edges("check", route, ["continue", "finish"])
```

**特点**: 串行执行，只调用一个节点

#### 方式 2: 返回 Send 对象列表（并行）

```python
def route(state):
    return [
        Send("process", {"item": item})
        for item in state["items"]
    ]

builder.add_conditional_edges("split", route, ["process"])
```

**特点**: 并行执行，调用多个节点实例

#### 方式 3: 混合返回（并行 + 串行）

```python
def route(state):
    sends = [Send("process", {"item": item}) for item in state["items"]]
    sends.append("log")  # 同时执行 log 节点
    return sends

builder.add_conditional_edges("split", route, ["process", "log"])
```

**特点**: 部分并行，部分串行

### 动态决策示例

```python
def dynamic_route(state):
    """根据状态动态决定并行策略"""
    query = state["query"]

    # 分析查询，决定需要哪些工具
    if "weather" in query and "news" in query:
        # 并行调用两个工具
        return [
            Send("weather_tool", {"location": extract_location(query)}),
            Send("news_tool", {"topic": extract_topic(query)})
        ]
    elif "weather" in query:
        # 只调用一个工具
        return [Send("weather_tool", {"location": extract_location(query)})]
    else:
        # 不调用工具，直接回答
        return "answer"
```

---

## Map-Reduce 模式的实现

### Map-Reduce 概念

**Map 阶段**: 将数据分发到多个并行任务
**Reduce 阶段**: 聚合所有任务的结果

### 完整实现示例

```python
"""
Map-Reduce 示例：并行生成多个主题的笑话，然后选择最佳笑话
"""

from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ===== 1. 定义状态 Schema =====
class OverallState(TypedDict):
    topic: str                                      # 输入：主题
    subjects: list[str]                             # Map 输入：子主题列表
    jokes: Annotated[list[str], operator.add]       # Map 输出：笑话列表（自动合并）
    best_selected_joke: str                         # Reduce 输出：最佳笑话

# ===== 2. Map 阶段：生成子主题 =====
def generate_topics(state: OverallState):
    """根据主题生成多个子主题"""
    topic = state["topic"]

    # 这里简化处理，实际可以调用 LLM
    if topic == "animals":
        subjects = ["lions", "elephants", "penguins"]
    else:
        subjects = [topic]

    return {"subjects": subjects}

# ===== 3. Map 阶段：并行生成笑话 =====
def generate_joke(state: dict):
    """为单个主题生成笑话（并行执行）"""
    subject = state["subject"]

    # 模拟 LLM 调用
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers? They find it hard to break the ice."
    }

    joke = joke_map.get(subject, f"A joke about {subject}")

    # 返回结果（会被 operator.add 自动追加到 jokes 列表）
    return {"jokes": [joke]}

# ===== 4. Map 阶段：动态创建并行任务 =====
def continue_to_jokes(state: OverallState):
    """
    关键函数：返回多个 Send 对象实现并行执行
    """
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

# ===== 5. Reduce 阶段：选择最佳笑话 =====
def best_joke(state: OverallState):
    """聚合所有笑话，选择最佳的一个"""
    jokes = state["jokes"]

    # 这里简化处理，实际可以调用 LLM 评分
    # 假设选择最长的笑话
    best = max(jokes, key=len)

    return {"best_selected_joke": best}

# ===== 6. 构建图 =====
builder = StateGraph(OverallState)

# 添加节点
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

# 添加边
builder.add_edge(START, "generate_topics")

# 关键：条件边返回 Send 对象列表，触发并行执行
builder.add_conditional_edges(
    "generate_topics",
    continue_to_jokes,
    ["generate_joke"]
)

builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

# 编译图
graph = builder.compile()

# ===== 7. 执行 =====
result = graph.invoke({"topic": "animals"})

print("生成的笑话:")
for joke in result["jokes"]:
    print(f"  - {joke}")

print(f"\n最佳笑话: {result['best_selected_joke']}")
```

### 执行流程

```
1. START → generate_topics
   输入: {"topic": "animals"}
   输出: {"subjects": ["lions", "elephants", "penguins"]}

2. generate_topics → continue_to_jokes (条件边)
   返回: [
       Send("generate_joke", {"subject": "lions"}),
       Send("generate_joke", {"subject": "elephants"}),
       Send("generate_joke", {"subject": "penguins"})
   ]

3. 并行执行 generate_joke (3次)
   - generate_joke({"subject": "lions"}) → {"jokes": ["Why don't lions..."]}
   - generate_joke({"subject": "elephants"}) → {"jokes": ["Why don't elephants..."]}
   - generate_joke({"subject": "penguins"}) → {"jokes": ["Why don't penguins..."]}

   结果通过 operator.add 自动合并:
   {"jokes": ["Why don't lions...", "Why don't elephants...", "Why don't penguins..."]}

4. generate_joke → best_joke
   输入: {"jokes": [...], "subjects": [...], "topic": "animals"}
   输出: {"best_selected_joke": "Why don't penguins..."}

5. best_joke → END
```

---

## 在 LangGraph 工作流中的应用

### 应用场景 1: 多文档并行处理

```python
class DocumentState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]

def split_documents(state: DocumentState):
    """为每个文档创建并行任务"""
    return [
        Send("summarize", {"doc": doc})
        for doc in state["documents"]
    ]

def summarize(state: dict):
    """单个文档摘要"""
    doc = state["doc"]
    summary = llm.invoke(f"Summarize: {doc}")
    return {"summaries": [summary.content]}
```

### 应用场景 2: 多智能体协作

```python
class AgentState(TypedDict):
    task: str
    agent_results: Annotated[list[dict], operator.add]

def route_to_agents(state: AgentState):
    """根据任务分配给不同的 Agent"""
    task = state["task"]

    agents = []
    if "research" in task:
        agents.append(Send("research_agent", {"task": task}))
    if "code" in task:
        agents.append(Send("code_agent", {"task": task}))
    if "write" in task:
        agents.append(Send("writer_agent", {"task": task}))

    return agents
```

### 应用场景 3: 多工具并行调用

```python
class ToolState(TypedDict):
    query: str
    tool_results: Annotated[list[dict], operator.add]

def route_to_tools(state: ToolState):
    """并行调用多个工具"""
    query = state["query"]

    # 分析查询，决定需要哪些工具
    tools_needed = analyze_query(query)

    return [
        Send(f"{tool}_tool", {"query": query})
        for tool in tools_needed
    ]
```

---

## 关键特性总结

### 1. 动态性

**编译时 vs 运行时**:
- 传统并行：编译时确定任务数量
- Send API：运行时动态确定任务数量

**优势**:
- 根据输入数据动态调整
- 支持复杂的决策逻辑
- 适应不同的业务场景

### 2. 灵活性

**状态传递**:
- 可以传递完整状态
- 可以传递部分状态
- 可以传递自定义数据

**节点调用**:
- 同一节点多次调用（不同输入）
- 不同节点并行调用
- 混合串行和并行

### 3. 自动化

**结果合并**:
- 使用 Reducer 函数自动合并
- 无需手动管理状态
- 避免状态覆盖问题

**同步机制**:
- 基于 BSP 模型自动同步
- 确保所有并行任务完成后再继续
- 无需手动管理同步点

---

## 最佳实践

### 1. 合理使用 Reducer

```python
# ✅ 正确：使用 operator.add 追加列表
results: Annotated[list, operator.add]

# ❌ 错误：不使用 reducer，结果会被覆盖
results: list
```

### 2. 控制并行度

```python
def route_with_limit(state):
    """限制并行任务数量"""
    items = state["items"]

    # 限制最多 10 个并行任务
    max_parallel = 10
    limited_items = items[:max_parallel]

    return [Send("process", {"item": item}) for item in limited_items]
```

### 3. 错误处理

```python
def safe_process(state: dict):
    """在并行任务中处理错误"""
    try:
        result = process_item(state["item"])
        return {"results": [{"success": True, "data": result}]}
    except Exception as e:
        return {"results": [{"success": False, "error": str(e)}]}
```

### 4. 性能优化

```python
# 静态并行（性能更好）
builder.add_edge(START, "task1")
builder.add_edge(START, "task2")
builder.add_edge(START, "task3")

# 动态并行（更灵活）
builder.add_conditional_edges(START, lambda s: [
    Send("task1", s),
    Send("task2", s),
    Send("task3", s)
])
```

---

## 参考资料

### 源码
- `langgraph/types.py:289-362` - Send 类定义
- `langgraph/graph/_branch.py` - 分支路由实现

### 官方文档
- [Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#send-api)
- [Map-Reduce with Send](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

### 社区资源
- [Implementing Map-Reduce with LangGraph](https://medium.com/@astropomeai/implementing-map-reduce-with-langgraph-creating-flexible-branches-for-parallel-execution-b6dc44327c0e)
- [Leveraging LangGraph's Send API](https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd)

---

**版本**: v1.0
**最后更新**: 2026-02-27
**作者**: Claude Code
**知识点**: 并行执行与分支合并 - Send 类与动态并行
