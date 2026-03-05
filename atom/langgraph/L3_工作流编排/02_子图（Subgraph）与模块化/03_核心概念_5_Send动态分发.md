# 核心概念 5：Send 动态分发

> Send 是在条件边中动态创建并行任务的机制，可将不同输入分发到相同或不同的子图节点，实现 map-reduce 模式

---

## 概述

在很多实际场景中，我们需要对一组输入进行并行处理——比如对多个文档分别做摘要、对多个主题分别生成内容、对多个 API 分别调用。LangGraph 的 `Send` 机制正是为此设计的：它允许在条件边函数中返回一组 `Send` 对象，每个 `Send` 将不同的输入分发到指定的节点（或子图），所有分发的任务并行执行，结果通过 reducer 收集汇总。这本质上就是 **map-reduce** 模式。

**[来源: reference/source_subgraph_01.md:76-85]**

---

## 1. 核心定义

### 什么是 Send？

**Send 是一个轻量级指令对象，用于在条件边中动态创建并行任务，每个 Send 指定一个目标节点和该任务的输入数据。**

```python
from langgraph.types import Send

# 基本结构
send = Send(
    node="target_node",  # 要调用的节点名称
    arg={"key": "value"}  # 传给该节点的输入数据
)
```

**[来源: reference/source_subgraph_01.md:78-85]**

### Send 的两个参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `node` | `str` | 目标节点的名称（必须是图中已注册的节点） |
| `arg` | `Any` | 传递给目标节点的输入数据（通常是字典） |

---

## 2. 工作原理

### 2.1 在条件边中返回 Send 列表

Send 不是在节点函数中使用的，而是在**条件边函数**中返回的。条件边函数返回一个 `Send` 列表，LangGraph 会为每个 `Send` 创建一个并行任务。

```python
from langgraph.types import Send

def dispatch_tasks(state: OverallState):
    """条件边函数：为每个 subject 创建一个并行任务"""
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

# 在图定义中使用
builder.add_conditional_edges("start_node", dispatch_tasks)
```

### 2.2 执行流程

```
                    条件边函数返回 Send 列表
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        Send("node",  Send("node",  Send("node",
         {"s":"A"})    {"s":"B"})    {"s":"C"})
              │            │            │
              ↓            ↓            ↓
         ┌────────┐  ┌────────┐  ┌────────┐
         │ node   │  │ node   │  │ node   │
         │ 处理 A │  │ 处理 B │  │ 处理 C │
         └───┬────┘  └───┬────┘  └───┬────┘
              │            │            │
              └────────────┼────────────┘
                           ↓
                    Reducer 合并结果
                   (operator.add)
```

### 2.3 与 Reducer 配合收集结果

每个并行任务的输出需要通过 **reducer** 合并到父图状态中。最常用的模式是用 `Annotated[list, operator.add]` 将所有结果追加到一个列表中。

```python
from typing import Annotated, TypedDict
import operator

class OverallState(TypedDict):
    subjects: list[str]                              # 输入列表
    jokes: Annotated[list[str], operator.add]         # 结果收集（用 reducer 追加）
```

**为什么必须用 reducer？** 因为多个并行任务同时完成后，会同时尝试更新 `jokes` 字段。没有 reducer，后一个的结果会覆盖前一个；有了 `operator.add`，所有结果会依次追加。

---

## 3. Map-Reduce 模式详解

### 3.1 模式概念

Send 本质上实现了经典的 **map-reduce** 模式：

```
┌─────────────────────────────────────────────────────┐
│                  Map-Reduce 模式                     │
│                                                     │
│  输入: ["主题A", "主题B", "主题C"]                    │
│                                                     │
│  ┌─── MAP 阶段（Send 分发）──────────────────────┐  │
│  │                                               │  │
│  │  Send("node", "主题A") → 结果A                │  │
│  │  Send("node", "主题B") → 结果B    [并行执行]   │  │
│  │  Send("node", "主题C") → 结果C                │  │
│  │                                               │  │
│  └───────────────────────────────────────────────┘  │
│                       │                             │
│                       ↓                             │
│  ┌─── REDUCE 阶段（Reducer 合并）────────────────┐  │
│  │                                               │  │
│  │  results = [结果A] + [结果B] + [结果C]         │  │
│  │          (operator.add)                       │  │
│  │                                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  输出: ["结果A", "结果B", "结果C"]                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 与传统 map-reduce 的对比

| 特性 | 传统 Map-Reduce | LangGraph Send |
|------|----------------|----------------|
| Map 阶段 | `map(func, items)` | `[Send(node, item) for item in items]` |
| 并行执行 | 依赖多进程/多线程 | LangGraph 自动并行 |
| Reduce 阶段 | `reduce(func, results)` | `Annotated[list, operator.add]` |
| 适用场景 | 数据处理 | AI 工作流（LLM 调用、子图处理） |

---

## 4. 完整代码示例

### 场景：多主题笑话生成（参考 fanout_to_subgraph.py）

这是 LangGraph 官方基准测试中的经典示例，展示了完整的 Send + 子图 map-reduce 模式。

```python
"""
Send 动态分发实战示例
演示：输入多个主题，并行生成笑话，收集所有结果
参考：langgraph/bench/fanout_to_subgraph.py
"""
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ============================================================
# 第一部分：定义状态类型
# ============================================================

# 父图状态
class OverallState(TypedDict):
    subjects: list[str]                              # 输入：主题列表
    jokes: Annotated[list[str], operator.add]         # 输出：笑话列表（reducer 追加）


# 子图输入（只接收单个 subject）
class JokeInput(TypedDict):
    subject: str


# 子图内部状态
class JokeState(TypedDict):
    subject: str
    joke: str


# 子图输出（只返回 joke）
class JokeOutput(TypedDict):
    joke: str


# ============================================================
# 第二部分：定义子图（笑话生成器）
# ============================================================

def generate(state: JokeState) -> JokeState:
    """生成笑话（简化版，实际可调用 LLM）"""
    subject = state["subject"]
    joke = f"关于{subject}的笑话：为什么{subject}过马路？因为它想到对面去！"
    return {"joke": joke}


def edit(state: JokeState) -> JokeState:
    """编辑润色笑话"""
    joke = state["joke"]
    edited = f"{joke}（经过精心润色）"
    return {"joke": edited}


# 构建子图
joke_builder = StateGraph(
    JokeState,
    input=JokeInput,     # 子图只接收 subject
    output=JokeOutput,   # 子图只返回 joke
)
joke_builder.add_node("generate", generate)
joke_builder.add_node("edit", edit)
joke_builder.add_edge(START, "generate")
joke_builder.add_edge("generate", "edit")
joke_builder.add_edge("edit", END)

joke_subgraph = joke_builder.compile()


# ============================================================
# 第三部分：定义父图
# ============================================================

def collect_joke(state: JokeOutput) -> dict:
    """将子图输出的单个 joke 转换为父图的 jokes 列表格式"""
    return {"jokes": [state["joke"]]}


def continue_to_jokes(state: OverallState):
    """条件边函数：为每个 subject 创建一个 Send"""
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]


def summarize(state: OverallState) -> dict:
    """汇总所有笑话"""
    total = len(state["jokes"])
    return {"jokes": [f"\n--- 共生成 {total} 个笑话 ---"]}


# 构建父图
parent_builder = StateGraph(OverallState)
parent_builder.add_node("generate_joke", joke_subgraph)
parent_builder.add_node("collect", collect_joke)
parent_builder.add_node("summarize", summarize)

# START → 条件边（Send 分发）→ generate_joke（并行）
parent_builder.add_conditional_edges(START, continue_to_jokes)
parent_builder.add_edge("generate_joke", "collect")
parent_builder.add_edge("collect", "summarize")
parent_builder.add_edge("summarize", END)

app = parent_builder.compile()


# ============================================================
# 第四部分：运行测试
# ============================================================

result = app.invoke({
    "subjects": ["程序员", "AI", "Python"],
    "jokes": [],
})

print("=== 生成的笑话 ===")
for joke in result["jokes"]:
    print(f"  {joke}")
```

**运行输出示例：**
```
=== 生成的笑话 ===
  关于程序员的笑话：为什么程序员过马路？因为它想到对面去！（经过精心润色）
  关于AI的笑话：为什么AI过马路？因为它想到对面去！（经过精心润色）
  关于Python的笑话：为什么Python过马路？因为它想到对面去！（经过精心润色）

--- 共生成 3 个笑话 ---
```

---

## 5. 状态类型设计要点

### 5.1 四种状态类型的关系

在 Send + 子图的模式中，通常需要定义多种状态类型：

```
┌──────────────────────────────────────────────────┐
│  OverallState（父图状态）                         │
│  ├── subjects: list[str]      ← 输入数据          │
│  └── jokes: list[str]         ← 收集结果（reducer）│
└─────────────────────┬────────────────────────────┘
                      │ Send("node", {"subject": s})
                      ↓
┌──────────────────────────────────────────────────┐
│  JokeInput（子图输入 Schema）                     │
│  └── subject: str             ← Send 传入的数据    │
├──────────────────────────────────────────────────┤
│  JokeState（子图内部状态）                         │
│  ├── subject: str             ← 从 input 映射      │
│  └── joke: str                ← 内部处理过程        │
├──────────────────────────────────────────────────┤
│  JokeOutput（子图输出 Schema）                     │
│  └── joke: str                ← 返回给父图的数据    │
└──────────────────────────────────────────────────┘
```

### 5.2 关键设计原则

1. **Send 的 `arg` 必须匹配子图的 input schema**
   ```python
   # Send 传入的数据
   Send("generate_joke", {"subject": "AI"})

   # 子图的 input schema 必须匹配
   class JokeInput(TypedDict):
       subject: str  # ✅ 匹配
   ```

2. **子图的 output 通过 reducer 合并到父图**
   ```python
   # 父图收集字段必须有 reducer
   class OverallState(TypedDict):
       jokes: Annotated[list[str], operator.add]  # ✅
   ```

3. **子图可以有比 input/output 更丰富的内部状态**
   ```python
   # 子图内部可以有额外字段
   class JokeState(TypedDict):
       subject: str   # input
       joke: str      # 内部处理
       draft: str     # 额外的内部字段（不对外暴露）
   ```

---

## 6. Send 的进阶用法

### 6.1 分发到不同的节点

Send 不要求所有任务都发到同一个节点，可以根据条件分发到不同节点：

```python
def dispatch_by_type(state: OverallState):
    """根据任务类型分发到不同的处理节点"""
    sends = []
    for task in state["tasks"]:
        if task["type"] == "translate":
            sends.append(Send("translate_node", task))
        elif task["type"] == "summarize":
            sends.append(Send("summarize_node", task))
        else:
            sends.append(Send("general_node", task))
    return sends
```

### 6.2 动态数量的并行任务

Send 的强大之处在于任务数量是**运行时动态决定**的：

```python
def dynamic_dispatch(state):
    """任务数量在运行时才确定"""
    # 可能是 1 个，也可能是 100 个
    documents = state["documents"]
    return [
        Send("process_doc", {"doc": doc, "index": i})
        for i, doc in enumerate(documents)
    ]
```

### 6.3 Send 与 Command 的组合

在子图内部，可以用 Command 控制流程；在父图层面，用 Send 分发任务：

```python
# 父图：用 Send 分发
def dispatch(state):
    return [Send("agent", {"query": q}) for q in state["queries"]]

# 子图内部：用 Command 导航
def agent_node(state) -> Command:
    if needs_escalation(state):
        return Command(goto="escalate", graph=Command.PARENT)
    return Command(goto="respond", update={"result": "done"})
```

---

## 7. 应用场景

### 7.1 批量文档处理

```python
def dispatch_docs(state):
    """将每个文档分发到处理子图"""
    return [
        Send("doc_processor", {"content": doc["text"], "doc_id": doc["id"]})
        for doc in state["documents"]
    ]

# 父图状态
class PipelineState(TypedDict):
    documents: list[dict]
    summaries: Annotated[list[str], operator.add]  # 收集所有摘要
```

### 7.2 并行 API 调用

```python
def dispatch_api_calls(state):
    """并行调用多个外部 API"""
    return [
        Send("call_api", {"url": url, "params": params})
        for url, params in state["api_requests"]
    ]
```

### 7.3 多主题内容生成（RAG 场景）

```python
def dispatch_rag_queries(state):
    """对多个子问题并行进行 RAG 检索"""
    return [
        Send("rag_retrieve", {"query": sub_query})
        for sub_query in state["sub_queries"]
    ]
```

---

## 8. 常见错误与排查

### 错误 1：缺少 Reducer 导致结果丢失

```python
# ❌ 错误：没有 reducer，只保留最后一个并行任务的结果
class State(TypedDict):
    results: list[str]  # 无 reducer

# ✅ 正确：用 operator.add 收集所有结果
class State(TypedDict):
    results: Annotated[list[str], operator.add]
```

### 错误 2：Send 的 arg 与子图 input schema 不匹配

```python
# ❌ 错误：Send 传入 "topic"，但子图期望 "subject"
Send("joke_node", {"topic": "AI"})

class JokeInput(TypedDict):
    subject: str  # 期望 "subject"，收到 "topic"

# ✅ 正确：key 名称必须一致
Send("joke_node", {"subject": "AI"})
```

### 错误 3：条件边函数返回类型错误

```python
# ❌ 错误：返回单个 Send 而不是列表
def dispatch(state):
    return Send("node", {"x": 1})  # 应该返回列表

# ✅ 正确：返回 Send 列表
def dispatch(state):
    return [Send("node", {"x": 1})]
```

---

## 9. 速查表

```
┌─────────────────────────────────────────────────────────┐
│                  Send 动态分发速查                        │
├──────────────────┬──────────────────────────────────────┤
│ 基本用法         │ Send("node_name", {"key": "value"})  │
│ 条件边中使用     │ return [Send(...) for item in items]  │
│ 结果收集         │ Annotated[list, operator.add]         │
│ 子图配合         │ StateGraph(State, input=In, output=O) │
├──────────────────┼──────────────────────────────────────┤
│ 并行执行         │ 自动，无需手动管理线程                  │
│ 动态数量         │ 运行时决定 Send 列表的长度              │
│ 分发到不同节点   │ 每个 Send 可以指定不同的 node           │
│ 与 Command 组合  │ Send 分发 + Command 控制子图内流程     │
└──────────────────┴──────────────────────────────────────┘
```

---

## 学习检查清单

- [ ] 理解 Send 的两个参数：node 和 arg
- [ ] 能在条件边函数中返回 Send 列表
- [ ] 理解为什么收集结果需要 reducer（operator.add）
- [ ] 能设计完整的 map-reduce 工作流（父图状态 + 子图状态 + Send + Reducer）
- [ ] 理解 Send 的 arg 必须匹配子图的 input schema
- [ ] 能区分 Send（并行分发）和 Command（导航控制）的使用场景

---

## 下一步学习

- **03_核心概念_4_Command跨图通信.md** - 学习 Command.PARENT 实现子图到父图的导航控制
- **03_核心概念_6_子图Checkpointer与持久化.md** - 子图的状态持久化与独立记忆
