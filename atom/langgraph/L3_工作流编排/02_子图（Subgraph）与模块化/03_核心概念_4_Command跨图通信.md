# 核心概念 4：Command 跨图通信

> Command 是 LangGraph 中同时支持状态更新和导航控制的指令类型，通过 `graph=Command.PARENT` 实现子图到父图的跨图通信

---

## 概述

在多图嵌套的 LangGraph 系统中，子图内部的节点经常需要与父图进行交互——比如将执行结果传递给父图、或者直接跳转到父图中的某个节点。`Command` 对象就是为此设计的统一指令类型，它能在一次返回中同时完成**状态更新**和**导航控制**，而 `graph=Command.PARENT` 参数则打通了子图到父图的通信通道。

**[来源: reference/source_subgraph_01.md:60-74, reference/context7_langgraph_01.md:61-97]**

---

## 1. 核心定义

### 什么是 Command 跨图通信？

**Command 是一种特殊的返回值类型，通过 `graph` 参数指定命令的目标图，实现子图节点直接向父图发送状态更新和导航指令。**

```python
from langgraph.types import Command

# 子图节点返回 Command，向父图发送命令
def subgraph_node(state: SubgraphState) -> Command:
    return Command(
        update={"result": "处理完成"},    # 更新父图状态
        goto="parent_next_node",          # 跳转到父图的某个节点
        graph=Command.PARENT,             # 目标：最近的父图
    )
```

**[来源: reference/source_subgraph_01.md:60-74]**

---

## 2. Command 的三个关键参数

### 2.1 参数总览

```
┌──────────────────────────────────────────────┐
│              Command 对象结构                  │
├──────────┬───────────────────────────────────┤
│  update  │  要应用到目标图状态的更新内容         │
│  goto    │  要导航到的目标节点名称              │
│  graph   │  命令发送的目标图                    │
│          │  ├─ None → 当前图（默认）            │
│          │  └─ Command.PARENT → 最近的父图     │
└──────────┴───────────────────────────────────┘
```

### 2.2 `update` 参数 —— 状态更新

`update` 是一个字典，定义要更新到目标图状态中的键值对。

```python
# 更新当前图的状态
Command(update={"count": 10})

# 更新父图的状态（必须配合 graph=Command.PARENT）
Command(update={"result": "done"}, graph=Command.PARENT)
```

**关键规则**：当 `graph=Command.PARENT` 时，`update` 中的 key 必须存在于父图的 State Schema 中，且父图必须为该 key 定义 reducer。

### 2.3 `goto` 参数 —— 导航控制

`goto` 指定目标图中要跳转到的节点名称。

```python
# 跳转到当前图的 "process" 节点
Command(goto="process")

# 跳转到父图的 "summarize" 节点
Command(goto="summarize", graph=Command.PARENT)

# 也可以同时指定多个目标（并行执行）
Command(goto=["node_a", "node_b"])
```

### 2.4 `graph` 参数 —— 目标图选择

```python
# graph=None（默认）：命令发送到当前图
Command(goto="next_node", graph=None)

# graph=Command.PARENT：命令发送到最近的父图
Command(goto="parent_node", graph=Command.PARENT)
```

**注意**：`Command.PARENT` 只会到达**最近的一层**父图。如果是多层嵌套（grandparent → parent → child），child 中的 `Command.PARENT` 只到 parent，不会直接到 grandparent。

---

## 3. Reducer 要求（最重要的注意事项）

### 3.1 为什么需要 Reducer？

当子图通过 `Command.PARENT` 更新父图的某个 key 时，LangGraph 需要知道**如何合并**这个更新。如果父图的 key 没有定义 reducer，更新将被直接覆盖——在并行场景下这可能导致数据丢失。

```python
from typing import Annotated, TypedDict
import operator

# ❌ 错误：没有 reducer，子图通过 Command.PARENT 更新时会报错
class ParentState(TypedDict):
    messages: list[str]

# ✅ 正确：定义了 reducer，子图可以安全地追加消息
class ParentState(TypedDict):
    messages: Annotated[list[str], operator.add]
```

### 3.2 Reducer 规则总结

| 场景 | 是否需要 Reducer | 原因 |
|------|-----------------|------|
| 子图通过共享 key 自动传递 | 不强制（但推荐） | 自动映射，无并发冲突 |
| 子图通过 Command.PARENT 更新 | **必须** | 跨图更新需要明确的合并策略 |
| 并行子图更新同一个 key | **必须** | 多个更新需要合并 |

---

## 4. 完整代码示例

### 场景 1：子图内部使用 Command 导航

最基础的用法——在当前图内部用 Command 实现动态路由。

```python
"""
场景1：子图内部使用 Command 导航
演示：根据条件在子图内部动态跳转节点
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义子图状态 =====
class ReviewState(TypedDict):
    document: str
    quality_score: int
    feedback: str


# ===== 2. 定义子图节点 =====
def evaluate(state: ReviewState) -> Command:
    """评估文档质量，根据得分决定下一步"""
    doc = state["document"]
    score = len(doc) * 10  # 简化的评分逻辑

    if score >= 50:
        # 质量达标，直接到通过节点
        return Command(
            update={"quality_score": score},
            goto="approve",
        )
    else:
        # 质量不达标，到修改节点
        return Command(
            update={"quality_score": score},
            goto="revise",
        )


def approve(state: ReviewState):
    """通过审核"""
    return {"feedback": f"文档审核通过，得分: {state['quality_score']}"}


def revise(state: ReviewState):
    """需要修改"""
    return {"feedback": f"文档需要修改，当前得分: {state['quality_score']}"}


# ===== 3. 构建子图 =====
review_graph = StateGraph(ReviewState)
review_graph.add_node("evaluate", evaluate)
review_graph.add_node("approve", approve)
review_graph.add_node("revise", revise)

review_graph.add_edge(START, "evaluate")
# 注意：evaluate 通过 Command 导航，不需要静态边
review_graph.add_edge("approve", END)
review_graph.add_edge("revise", END)

app = review_graph.compile()

# ===== 4. 运行测试 =====
result = app.invoke({"document": "Hello World", "quality_score": 0, "feedback": ""})
print(f"结果: {result['feedback']}")
# 输出: 结果: 文档审核通过，得分: 110
```

**要点**：`Command` 在当前图内导航时 `graph` 参数为 `None`（默认值），等价于条件边的动态路由，但更灵活——可以同时更新状态。

---

### 场景 2：子图通过 Command.PARENT 跳转到父图节点

这是 Command 跨图通信的核心场景——子图节点完成任务后，直接通知父图跳转到指定节点。

```python
"""
场景2：子图通过 Command.PARENT 跳转到父图的另一个节点
演示：分析子图完成后，通知父图进入汇总阶段
"""
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义父图状态（注意 reducer） =====
class ParentState(TypedDict):
    topic: str
    # 使用 operator.add reducer，允许子图安全追加结果
    results: Annotated[list[str], operator.add]
    summary: str


# ===== 2. 定义子图状态 =====
class AnalysisState(TypedDict):
    topic: str          # 与父图共享的 key
    analysis: str       # 子图私有的 key


# ===== 3. 子图节点 =====
def analyze(state: AnalysisState) -> Command:
    """执行分析，然后通过 Command.PARENT 将结果发送给父图"""
    topic = state["topic"]
    analysis_result = f"对 [{topic}] 的深度分析结果"

    # 向父图发送命令：更新 results 并跳转到 summarize 节点
    return Command(
        update={"results": [analysis_result]},  # 追加到父图的 results 列表
        goto="summarize",                        # 跳转到父图的 summarize 节点
        graph=Command.PARENT,                    # 目标：父图
    )


# ===== 4. 构建子图 =====
analysis_builder = StateGraph(AnalysisState)
analysis_builder.add_node("analyze", analyze)
analysis_builder.add_edge(START, "analyze")
analysis_subgraph = analysis_builder.compile()


# ===== 5. 父图节点 =====
def prepare(state: ParentState):
    """准备阶段"""
    return {"results": [f"准备分析主题: {state['topic']}"]}


def summarize(state: ParentState):
    """汇总阶段 —— 由子图的 Command.PARENT 跳转到这里"""
    all_results = "\n".join(state["results"])
    return {"summary": f"汇总报告:\n{all_results}"}


# ===== 6. 构建父图 =====
parent_builder = StateGraph(ParentState)
parent_builder.add_node("prepare", prepare)
parent_builder.add_node("analysis", analysis_subgraph)  # 子图作为节点
parent_builder.add_node("summarize", summarize)

parent_builder.add_edge(START, "prepare")
parent_builder.add_edge("prepare", "analysis")
# 注意：analysis 子图通过 Command.PARENT 跳转到 summarize
# 不需要 add_edge("analysis", "summarize")
parent_builder.add_edge("summarize", END)

parent_app = parent_builder.compile()

# ===== 7. 运行 =====
result = parent_app.invoke({
    "topic": "AI Agent 架构",
    "results": [],
    "summary": "",
})
print(f"最终摘要: {result['summary']}")
```

**运行输出示例：**
```
最终摘要: 汇总报告:
准备分析主题: AI Agent 架构
对 [AI Agent 架构] 的深度分析结果
```

**关键理解**：
- 子图的 `analyze` 节点返回 `Command(graph=Command.PARENT)`，直接跳过了子图的后续流程
- 父图的 `results` key 定义了 `operator.add` reducer，所以子图的更新是**追加**而非覆盖
- 父图不需要从 `analysis` 到 `summarize` 的静态边，因为子图已经通过 Command 完成了跳转

---

### 场景 3：多代理 Handoff（代理 A 切换到代理 B）

这是 `Command.PARENT` 最典型的应用场景——多代理系统中的代理切换。

```python
"""
场景3：多代理 Handoff 系统
演示：客服系统中 通用代理 → 技术代理 的切换
"""
from typing import Annotated, TypedDict, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义父图状态（协调层） =====
class OrchestratorState(TypedDict):
    user_query: str
    messages: Annotated[list[str], operator.add]  # reducer: 追加消息
    active_agent: str
    final_answer: str


# ===== 2. 定义通用代理子图 =====
class GeneralAgentState(TypedDict):
    user_query: str       # 共享 key
    messages: Annotated[list[str], operator.add]  # 共享 key


def general_agent_process(state: GeneralAgentState) -> Command:
    """通用代理：判断是否需要转交给技术代理"""
    query = state["user_query"]

    # 简化的判断逻辑
    is_technical = any(
        keyword in query for keyword in ["代码", "bug", "API", "部署", "错误"]
    )

    if is_technical:
        # 需要技术支持 → 切换到技术代理
        return Command(
            update={
                "messages": [f"[通用代理] 检测到技术问题，转交技术代理处理"],
                "active_agent": "technical",
            },
            goto="technical_agent",      # 父图中的技术代理节点
            graph=Command.PARENT,         # 跳到父图
        )
    else:
        # 通用问题，直接回答
        return Command(
            update={
                "messages": [f"[通用代理] 回答: 关于'{query}'的解答..."],
                "active_agent": "general",
            },
            goto="collect_answer",        # 父图中的收集答案节点
            graph=Command.PARENT,
        )


general_builder = StateGraph(GeneralAgentState)
general_builder.add_node("process", general_agent_process)
general_builder.add_edge(START, "process")
general_agent = general_builder.compile()


# ===== 3. 定义技术代理子图 =====
class TechAgentState(TypedDict):
    user_query: str
    messages: Annotated[list[str], operator.add]


def tech_agent_process(state: TechAgentState) -> Command:
    """技术代理：处理技术问题"""
    query = state["user_query"]
    return Command(
        update={
            "messages": [f"[技术代理] 技术解答: 关于'{query}'的技术方案..."],
            "active_agent": "technical",
        },
        goto="collect_answer",
        graph=Command.PARENT,
    )


tech_builder = StateGraph(TechAgentState)
tech_builder.add_node("process", tech_agent_process)
tech_builder.add_edge(START, "process")
technical_agent = tech_builder.compile()


# ===== 4. 构建父图（协调器） =====
def route_query(state: OrchestratorState):
    """初始路由：所有查询先到通用代理"""
    return {"messages": [f"[协调器] 收到查询: {state['user_query']}"]}


def collect_answer(state: OrchestratorState):
    """收集最终答案"""
    last_message = state["messages"][-1] if state["messages"] else ""
    return {"final_answer": last_message}


orchestrator = StateGraph(OrchestratorState)
orchestrator.add_node("route", route_query)
orchestrator.add_node("general_agent", general_agent)       # 通用代理子图
orchestrator.add_node("technical_agent", technical_agent)    # 技术代理子图
orchestrator.add_node("collect_answer", collect_answer)

orchestrator.add_edge(START, "route")
orchestrator.add_edge("route", "general_agent")
# general_agent 和 technical_agent 通过 Command.PARENT 跳转
orchestrator.add_edge("collect_answer", END)

app = orchestrator.compile()

# ===== 5. 测试 =====
# 测试1：通用问题
result1 = app.invoke({
    "user_query": "你们的营业时间是什么？",
    "messages": [],
    "active_agent": "",
    "final_answer": "",
})
print(f"通用问题: {result1['final_answer']}")

# 测试2：技术问题（会触发 handoff）
result2 = app.invoke({
    "user_query": "API 调用返回错误码 500",
    "messages": [],
    "active_agent": "",
    "final_answer": "",
})
print(f"技术问题: {result2['final_answer']}")
print(f"消息记录: {result2['messages']}")
```

**运行输出示例：**
```
通用问题: [通用代理] 回答: 关于'你们的营业时间是什么？'的解答...
技术问题: [技术代理] 技术解答: 关于'API 调用返回错误码 500'的技术方案...
消息记录: ['[协调器] 收到查询: API 调用返回错误码 500',
           '[通用代理] 检测到技术问题，转交技术代理处理',
           '[技术代理] 技术解答: ...']
```

**Handoff 流程图：**

```
┌─────────────────────────────────────────────────────────┐
│                     父图（协调器）                        │
│                                                         │
│  START → [route] → [general_agent]                      │
│                         │                               │
│                    ┌────┴──────────┐                     │
│                    │ 子图判断       │                     │
│                    │ 是技术问题？   │                     │
│                    └────┬──────────┘                     │
│              Command.PARENT                              │
│               ┌────┴────┐                               │
│               ↓         ↓                               │
│    [technical_agent]  [collect_answer] → END             │
│         │                    ↑                           │
│         └────────────────────┘                           │
│           Command.PARENT                                │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Command 与静态边的关系

### 5.1 Command 不会取消静态边

一个常见误解：如果节点返回了 `Command(goto="B")`，是否会取消已定义的 `add_edge("node", "C")`？

**答案：不会。** Command 的 `goto` 是**额外的**导航指令，它不会阻止已有的静态边。

```python
# 假设有以下定义
builder.add_edge("node_a", "node_c")  # 静态边

def node_a(state):
    return Command(goto="node_b", update={"x": 1})
    # 结果：node_b 和 node_c 都会执行！
```

### 5.2 最佳实践

| 做法 | 推荐度 | 说明 |
|------|--------|------|
| Command 替代条件边 | 推荐 | 用 Command 实现动态路由，替代 add_conditional_edges |
| Command + 静态边混用 | 谨慎 | 可能导致多个节点同时执行 |
| Command.PARENT + 静态边 | 避免 | 子图的 Command.PARENT 后不应有静态边到其他节点 |

---

## 6. 在 LangGraph 中的典型应用

### 6.1 多代理系统

```
协调器（父图）
├── 代理A（子图）──→ Command.PARENT → 代理B
├── 代理B（子图）──→ Command.PARENT → 代理C
└── 代理C（子图）──→ Command.PARENT → 汇总节点
```

每个代理独立处理，通过 `Command.PARENT` 实现代理间的无缝切换。

### 6.2 审批流程

```
审批系统（父图）
├── 初审子图 ──→ Command.PARENT → 复审子图 / 驳回节点
├── 复审子图 ──→ Command.PARENT → 终审子图 / 退回初审
└── 终审子图 ──→ Command.PARENT → 完成节点
```

审批的每一级都是独立子图，通过 Command 控制审批流转方向。

### 6.3 RAG 多步骤处理

```
RAG 系统（父图）
├── 查询理解子图 ──→ Command.PARENT → 检索子图
├── 检索子图     ──→ Command.PARENT → 生成子图 / 查询改写子图
└── 生成子图     ──→ Command.PARENT → 结果输出
```

---

## 7. 常见错误与排查

### 错误 1：父图未定义 Reducer

```python
# ❌ 报错：InvalidUpdateError
class ParentState(TypedDict):
    results: list[str]  # 没有 reducer

# 子图中
return Command(update={"results": ["new"]}, graph=Command.PARENT)
# 报错：Cannot update key 'results' without a reducer
```

**修复**：为共享 key 添加 reducer。

```python
# ✅ 正确
class ParentState(TypedDict):
    results: Annotated[list[str], operator.add]
```

### 错误 2：更新了父图中不存在的 Key

```python
# ❌ 报错：子图尝试更新父图中不存在的 key
return Command(
    update={"nonexistent_key": "value"},
    graph=Command.PARENT,
)
```

**修复**：确保 `update` 中的 key 在目标图的 State Schema 中存在。

### 错误 3：goto 指向不存在的节点

```python
# ❌ 报错：父图中没有 "missing_node" 节点
return Command(goto="missing_node", graph=Command.PARENT)
```

**修复**：确认父图中已注册目标节点。

---

## 8. 速查表

```
┌─────────────────────────────────────────────────────────┐
│                Command 跨图通信速查                       │
├──────────────────┬──────────────────────────────────────┤
│ 当前图内导航      │ Command(goto="node")                 │
│ 当前图内更新+导航 │ Command(update={...}, goto="node")   │
│ 跳转到父图       │ Command(goto="node",                 │
│                  │   graph=Command.PARENT)               │
│ 更新父图+跳转    │ Command(update={...}, goto="node",   │
│                  │   graph=Command.PARENT)               │
├──────────────────┼──────────────────────────────────────┤
│ 父图 reducer     │ 使用 Command.PARENT 时必须定义        │
│ 静态边           │ Command 不会取消已有的静态边           │
│ 多层嵌套         │ PARENT 只到最近一层父图               │
└──────────────────┴──────────────────────────────────────┘
```

---

## 学习检查清单

- [ ] 理解 Command 的三个参数：update、goto、graph
- [ ] 理解 `graph=None` 和 `graph=Command.PARENT` 的区别
- [ ] 掌握父图 reducer 的要求
- [ ] 能编写子图通过 Command.PARENT 跳转到父图的代码
- [ ] 理解 Command 与静态边的关系
- [ ] 能设计多代理 handoff 系统

---

## 下一步学习

- **03_核心概念_5_Send动态分发.md** - 学习另一种跨图通信机制：map-reduce 模式的并行分发
- **03_核心概念_6_子图Checkpointer与持久化.md** - 子图的状态持久化策略
