# 核心概念 6：子图 Checkpointer

> 子图 Checkpointer 决定子图如何持久化执行状态，支持继承父图、独立配置或完全禁用三种模式

---

## 概述

**子图 Checkpointer 是什么？**

在 LangGraph 中，Checkpointer 负责在图执行过程中保存每一步的状态快照，使得图可以暂停、恢复、回溯和断点续传。当子图作为父图的节点运行时，子图的 Checkpointer 策略决定了它如何与父图的持久化机制协作。

LangGraph 提供了三种 Checkpointer 配置模式：

| 模式 | 编译参数 | 行为 |
|------|----------|------|
| 继承父图 | `checkpointer=None`（默认） | 子图使用父图的 Checkpointer |
| 独立配置 | `checkpointer=True` | 子图拥有独立的 Checkpointer |
| 完全禁用 | `checkpointer=False` | 子图不使用任何 Checkpointer |

**为什么需要理解子图 Checkpointer？**

- 多代理系统中，不同代理可能需要独立的对话历史
- 复杂工作流中，子图可能需要断点续传但不影响父图
- 性能优化时，某些子图不需要持久化可以减少开销
- 调试时，需要理解 checkpoint 数据在父子图之间的关系

**[来源: reference/source_subgraph_01.md:87-92, reference/context7_langgraph_01.md:86-90]**

---

## 1. Checkpointer 基础回顾

### 什么是 Checkpointer？

**Checkpointer 是 LangGraph 的状态持久化机制，在图执行的每个超级步（super step）后自动保存状态快照，支持暂停恢复、时间回溯和断点续传。**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 创建图并设置 Checkpointer
builder = StateGraph(MyState)
# ... 添加节点和边 ...
graph = builder.compile(checkpointer=InMemorySaver())

# 执行时通过 thread_id 标识不同的执行线程
config = {"configurable": {"thread_id": "thread-001"}}
result = graph.invoke({"query": "你好"}, config=config)

# 同一个 thread_id 可以恢复之前的状态
result2 = graph.invoke({"query": "继续刚才的话题"}, config=config)
```

### Checkpointer 保存了什么？

```
┌──────────────────────────────────────────┐
│            Checkpoint 快照                │
├──────────────────────────────────────────┤
│  thread_id   : "thread-001"              │
│  step        : 3                         │
│  state       : {完整的状态字典}            │
│  metadata    : {执行元信息}               │
│  parent_id   : "上一个 checkpoint 的 ID"  │
│  timestamp   : "2026-02-27T10:30:00Z"    │
└──────────────────────────────────────────┘
```

---

## 2. 模式 1：继承父图（默认）

### 一句话定义

**`checkpointer=None` 是默认模式，子图不自带 Checkpointer，当作为父图节点运行时自动继承父图的 Checkpointer。**

### 工作原理

当父图编译时设置了 Checkpointer，它会自动传播到所有子图。子图的状态变化作为父图执行的一部分被记录，不会产生独立的 checkpoint 记录。

```
┌──────────────────────────────────────────────────────┐
│  父图 (checkpointer=InMemorySaver())                  │
│                                                      │
│  ┌──────────┐    ┌──────────────────┐    ┌────────┐  │
│  │  节点 A   │───→│  子图节点          │───→│ 节点 C  │  │
│  └──────────┘    │  (checkpointer    │    └────────┘  │
│                  │   =None, 继承)    │                │
│                  └──────────────────┘                │
│                                                      │
│  Checkpointer 自动覆盖整个执行流程                      │
│  子图的状态变化记录在父图的 checkpoint 中                 │
└──────────────────────────────────────────────────────┘
```

### 代码示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ===== 子图定义 =====
class SubgraphState(TypedDict):
    query: str
    sub_result: str


def sub_process(state: SubgraphState) -> dict:
    """子图节点：处理查询"""
    return {"sub_result": f"子图处理了: {state['query']}"}


sub_builder = StateGraph(SubgraphState)
sub_builder.add_node("process", sub_process)
sub_builder.add_edge(START, "process")
sub_builder.add_edge("process", END)

# 关键：checkpointer=None（默认），继承父图
subgraph = sub_builder.compile()  # 等同于 compile(checkpointer=None)


# ===== 父图定义 =====
class ParentState(TypedDict):
    query: str
    sub_result: str
    final_result: str


def pre_process(state: ParentState) -> dict:
    return {"query": state["query"].strip().lower()}


def post_process(state: ParentState) -> dict:
    return {"final_result": f"最终结果: {state['sub_result']}"}


parent_builder = StateGraph(ParentState)
parent_builder.add_node("pre", pre_process)
parent_builder.add_node("sub", subgraph)  # 子图作为节点
parent_builder.add_node("post", post_process)
parent_builder.add_edge(START, "pre")
parent_builder.add_edge("pre", "sub")
parent_builder.add_edge("sub", "post")
parent_builder.add_edge("post", END)

# 父图设置 Checkpointer，自动传播到子图
graph = parent_builder.compile(checkpointer=InMemorySaver())

# 执行
config = {"configurable": {"thread_id": "inherit-demo"}}
result = graph.invoke({"query": "  Hello World  "}, config=config)
print(result)
# {'query': 'hello world', 'sub_result': '子图处理了: hello world', 'final_result': '最终结果: 子图处理了: hello world'}
```

### 适用场景

| 场景 | 说明 |
|------|------|
| **简单子图** | 子图只是父图流程的一部分，不需要独立状态 |
| **数据处理管道** | 文档解析、文本清洗等无状态子流程 |
| **工具调用子图** | 工具执行是一次性的，不需要记住历史 |
| **大部分常见场景** | 默认行为即可满足需求 |

### 源码依据

```python
# langgraph/graph/state.py line 1057
# If `None`, it may inherit the parent graph's checkpointer when used as a subgraph.
```

**[来源: reference/source_subgraph_01.md:89-90, LangGraph 源码 state.py:1057]**

---

## 3. 模式 2：独立 Checkpointer

### 一句话定义

**`checkpointer=True` 使子图拥有独立的持久化能力，子图的每一步执行都会产生独立的 checkpoint 记录，与父图的 checkpoint 互不干扰。**

### 工作原理

当子图以 `checkpointer=True` 编译时，LangGraph 会在运行时为子图创建一个独立的 checkpoint 命名空间。子图的状态快照独立于父图存储，拥有自己的 checkpoint 历史链。

```
┌──────────────────────────────────────────────────────┐
│  父图 (checkpointer=InMemorySaver())                  │
│  checkpoint 命名空间: "parent"                         │
│                                                      │
│  ┌──────────┐    ┌──────────────────┐    ┌────────┐  │
│  │  节点 A   │───→│  子图节点          │───→│ 节点 C  │  │
│  └──────────┘    │  (checkpointer   │    └────────┘  │
│                  │   =True, 独立)   │                │
│                  │                  │                │
│                  │  独立 checkpoint  │                │
│                  │  命名空间: "sub"   │                │
│                  └──────────────────┘                │
│                                                      │
│  父图和子图各自维护独立的 checkpoint 历史               │
└──────────────────────────────────────────────────────┘
```

### 代码示例

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator


# ===== 代理子图：拥有独立记忆 =====
class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    task: str


def agent_think(state: AgentState) -> dict:
    """代理思考节点：基于历史消息进行推理"""
    history_count = len(state.get("messages", []))
    response = f"[代理回复] 收到任务: {state['task']}（历史消息数: {history_count}）"
    return {"messages": [response]}


agent_builder = StateGraph(AgentState)
agent_builder.add_node("think", agent_think)
agent_builder.add_edge(START, "think")
agent_builder.add_edge("think", END)

# 关键：checkpointer=True，子图拥有独立记忆
agent_subgraph = agent_builder.compile(checkpointer=True)


# ===== 编排父图 =====
class OrchestratorState(TypedDict):
    task: str
    messages: Annotated[list[str], operator.add]
    final_answer: str


def dispatch(state: OrchestratorState) -> dict:
    return {"task": state["task"]}


def summarize(state: OrchestratorState) -> dict:
    return {"final_answer": f"汇总: {state['messages'][-1]}"}


orchestrator_builder = StateGraph(OrchestratorState)
orchestrator_builder.add_node("dispatch", dispatch)
orchestrator_builder.add_node("agent", agent_subgraph)
orchestrator_builder.add_node("summarize", summarize)
orchestrator_builder.add_edge(START, "dispatch")
orchestrator_builder.add_edge("dispatch", "agent")
orchestrator_builder.add_edge("agent", "summarize")
orchestrator_builder.add_edge("summarize", END)

graph = orchestrator_builder.compile(checkpointer=InMemorySaver())

# 第一次调用
config = {"configurable": {"thread_id": "multi-agent-001"}}
result1 = graph.invoke({"task": "分析市场数据"}, config=config)
print(result1["final_answer"])

# 第二次调用 - 子图的独立 checkpointer 保留了之前的 messages
result2 = graph.invoke({"task": "生成报告"}, config=config)
print(result2["final_answer"])
```

### 适用场景

| 场景 | 说明 |
|------|------|
| **多代理系统** | 每个代理需要独立的对话历史和记忆 |
| **断点续传子流程** | 子图需要独立暂停/恢复，不影响父图 |
| **有状态工具** | 工具需要记住之前的调用结果 |
| **长时间运行子任务** | 子任务可能失败重试，需要独立的进度记录 |

**[来源: reference/context7_langgraph_01.md:88-90, reference/source_subgraph_01.md:91]**

---

## 4. 模式 3：禁用 Checkpointer

### 一句话定义

**`checkpointer=False` 明确禁止子图使用任何 Checkpointer，即使父图有 Checkpointer 也不会继承，子图的执行不产生任何持久化记录。**

### 工作原理

这是一种显式的"退出"机制。与默认的 `None`（可能继承）不同，`False` 是一个明确的声明：这个子图不需要也不应该有 checkpoint。

```
┌──────────────────────────────────────────────────────┐
│  父图 (checkpointer=InMemorySaver())                  │
│                                                      │
│  ┌──────────┐    ┌──────────────────┐    ┌────────┐  │
│  │  节点 A   │───→│  子图节点          │───→│ 节点 C  │  │
│  └──────────┘    │  (checkpointer   │    └────────┘  │
│                  │   =False, 禁用)  │                │
│                  │                  │                │
│                  │  无 checkpoint    │                │
│                  │  不继承父图       │                │
│                  └──────────────────┘                │
└──────────────────────────────────────────────────────┘
```

### 代码示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ===== 纯计算子图：不需要持久化 =====
class ComputeState(TypedDict):
    data: list[float]
    result: float


def compute_average(state: ComputeState) -> dict:
    """纯计算节点：计算平均值，无需记录历史"""
    avg = sum(state["data"]) / len(state["data"]) if state["data"] else 0.0
    return {"result": avg}


compute_builder = StateGraph(ComputeState)
compute_builder.add_node("compute", compute_average)
compute_builder.add_edge(START, "compute")
compute_builder.add_edge("compute", END)

# 关键：checkpointer=False，明确禁用
compute_subgraph = compute_builder.compile(checkpointer=False)


# ===== 父图 =====
class PipelineState(TypedDict):
    data: list[float]
    result: float
    report: str


def generate_report(state: PipelineState) -> dict:
    return {"report": f"计算结果: {state['result']:.2f}"}


pipeline_builder = StateGraph(PipelineState)
pipeline_builder.add_node("compute", compute_subgraph)
pipeline_builder.add_node("report", generate_report)
pipeline_builder.add_edge(START, "compute")
pipeline_builder.add_edge("compute", "report")
pipeline_builder.add_edge("report", END)

# 父图有 checkpointer，但计算子图不继承
graph = pipeline_builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "compute-001"}}
result = graph.invoke({"data": [1.0, 2.0, 3.0, 4.0, 5.0]}, config=config)
print(result["report"])  # 计算结果: 3.00
```

### 适用场景

| 场景 | 说明 |
|------|------|
| **纯计算子图** | 确定性计算，不需要记录中间状态 |
| **敏感数据处理** | 子图处理不应被持久化的敏感信息 |
| **高性能要求** | 避免 checkpoint 写入的开销 |
| **临时性子任务** | 一次性执行，结果通过状态传回父图即可 |

### 源码依据

```python
# langgraph/graph/state.py line 1057
# If `False`, it will not use or inherit any checkpointer.
```

**[来源: reference/source_subgraph_01.md:92, LangGraph 源码 state.py:1057]**

---

## 5. 三种模式对比

### 决策表

```
                    继承 (None)     独立 (True)      禁用 (False)
                    ───────────    ───────────     ────────────
是否有 checkpoint？    取决于父图       始终有            始终无
独立于父图？          否              是               N/A
能暂停恢复？          随父图           独立可恢复        不能
有独立历史？          无              有               无
性能开销             随父图           额外开销          最小
适合场景             常规子流程       多代理/有状态      纯计算/敏感
```

### 决策流程图

```
开始选择子图 Checkpointer 模式
│
├── 子图需要独立的执行历史吗？
│   ├── 是 → checkpointer=True（独立模式）
│   └── 否 ↓
│
├── 子图处理敏感数据或需要最大性能？
│   ├── 是 → checkpointer=False（禁用模式）
│   └── 否 ↓
│
└── 使用默认值 → checkpointer=None（继承模式）
```

---

## 6. 深入：继承模式的传播机制

### 父图如何传播 Checkpointer 到子图

当父图通过 `compile(checkpointer=InMemorySaver())` 编译后，运行时如果遇到一个子图节点且该子图的 checkpointer 为 `None`，LangGraph 会将父图的 checkpointer 实例传递给子图使用。这个过程是**运行时**发生的，不是编译时。

```python
# 伪代码：LangGraph 内部传播逻辑
class CompiledGraph:
    def _execute_node(self, node_name, state, config):
        node = self.nodes[node_name]

        if isinstance(node, CompiledGraph):  # 节点是子图
            subgraph = node
            if subgraph.checkpointer is None:
                # 运行时继承父图的 checkpointer
                subgraph_config = {
                    **config,
                    "checkpointer": self.checkpointer
                }
                return subgraph.invoke(state, subgraph_config)
            elif subgraph.checkpointer is False:
                # 明确禁用，不传递
                return subgraph.invoke(state, config)
            else:
                # 独立 checkpointer，使用自己的
                return subgraph.invoke(state, config)
```

### 嵌套子图的 Checkpoint 命名空间

当子图嵌套多层时，每层的 checkpoint 通过命名空间隔离：

```
父图 checkpoint 命名空间:
  thread_id: "main-001"

  └── 子图 A checkpoint:
      namespace: "main-001:subgraph_a"

      └── 子子图 B checkpoint:
          namespace: "main-001:subgraph_a:subgraph_b"
```

这种命名空间机制确保了不同层级的 checkpoint 不会互相冲突。

---

## 7. 实战：多代理独立记忆系统

### 场景描述

构建一个多代理系统：研究代理和写作代理各自维护独立的对话历史，编排器协调两者。

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator


# ===== 研究代理子图 =====
class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]


def research(state: ResearchState) -> dict:
    """模拟研究过程"""
    history_count = len(state.get("findings", []))
    finding = f"[研究发现 #{history_count + 1}] 关于 '{state['topic']}' 的调研结果"
    return {"findings": [finding]}


research_builder = StateGraph(ResearchState)
research_builder.add_node("research", research)
research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

# 研究代理：独立 checkpointer，保留研究历史
research_agent = research_builder.compile(checkpointer=True)


# ===== 写作代理子图 =====
class WritingState(TypedDict):
    topic: str
    findings: list[str]
    drafts: Annotated[list[str], operator.add]


def write(state: WritingState) -> dict:
    """模拟写作过程"""
    draft_count = len(state.get("drafts", []))
    latest_finding = state["findings"][-1] if state.get("findings") else "无"
    draft = f"[草稿 v{draft_count + 1}] 基于: {latest_finding}"
    return {"drafts": [draft]}


writing_builder = StateGraph(WritingState)
writing_builder.add_node("write", write)
writing_builder.add_edge(START, "write")
writing_builder.add_edge("write", END)

# 写作代理：独立 checkpointer，保留草稿历史
writing_agent = writing_builder.compile(checkpointer=True)


# ===== 编排器父图 =====
class OrchestratorState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    drafts: Annotated[list[str], operator.add]
    final_output: str


def prepare(state: OrchestratorState) -> dict:
    return {"topic": state["topic"]}


def finalize(state: OrchestratorState) -> dict:
    return {
        "final_output": (
            f"主题: {state['topic']}\n"
            f"研究发现数: {len(state.get('findings', []))}\n"
            f"草稿数: {len(state.get('drafts', []))}"
        )
    }


orchestrator_builder = StateGraph(OrchestratorState)
orchestrator_builder.add_node("prepare", prepare)
orchestrator_builder.add_node("research", research_agent)
orchestrator_builder.add_node("writing", writing_agent)
orchestrator_builder.add_node("finalize", finalize)

orchestrator_builder.add_edge(START, "prepare")
orchestrator_builder.add_edge("prepare", "research")
orchestrator_builder.add_edge("research", "writing")
orchestrator_builder.add_edge("writing", "finalize")
orchestrator_builder.add_edge("finalize", END)

graph = orchestrator_builder.compile(checkpointer=InMemorySaver())

# 第一轮执行
config = {"configurable": {"thread_id": "project-alpha"}}
result1 = graph.invoke({"topic": "AI 趋势"}, config=config)
print(result1["final_output"])
# 主题: AI 趋势
# 研究发现数: 1
# 草稿数: 1

# 第二轮执行 - 研究代理和写作代理各自保留了之前的记忆
result2 = graph.invoke({"topic": "AI 趋势（深入）"}, config=config)
print(result2["final_output"])
# 主题: AI 趋势（深入）
# 研究发现数: 2  ← 研究代理累积了历史
# 草稿数: 2      ← 写作代理也累积了历史
```

---

## 8. 注意事项

### 8.1 thread_id 的使用

```python
# thread_id 是 checkpoint 的核心标识
config = {"configurable": {"thread_id": "unique-id"}}

# 同一 thread_id → 恢复之前的状态
# 不同 thread_id → 全新的执行上下文

# 重要：父图和子图共享同一个 thread_id
# 子图的 checkpoint 通过命名空间与父图区分
```

### 8.2 独立模式下的 Checkpointer 类型

```python
# checkpointer=True 表示"使用独立 checkpointer"
# LangGraph 运行时会自动配置具体的 checkpointer 实例
subgraph = builder.compile(checkpointer=True)

# 注意：True 不是传入一个具体的 checkpointer 对象
# 而是一个标志，告诉 LangGraph "这个子图需要独立的 checkpoint 能力"
# 实际使用的 checkpointer 类型由父图的 checkpointer 决定
```

### 8.3 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 子图单独运行无 checkpoint | `checkpointer=None` 的子图如果不在父图中运行，就没有 checkpointer | 单独测试时手动设置 checkpointer |
| 混淆 None 和 False | `None` 可能继承，`False` 绝对不继承 | 明确意图时用 `False` |
| 独立模式下状态膨胀 | 独立 checkpointer 会累积大量历史 | 定期清理或设置 TTL |
| 嵌套子图命名空间冲突 | 多层嵌套时需要注意命名空间 | LangGraph 自动处理，但调试时需了解 |

---

## 9. 与 RAG 开发的联系

在 RAG 系统中，子图 Checkpointer 策略在以下场景中至关重要：

| RAG 场景 | Checkpointer 策略 | 原因 |
|----------|-------------------|------|
| 文档解析子图 | `checkpointer=False` | 纯计算，无需持久化 |
| 对话式 RAG | `checkpointer=True` | 每个用户会话需要独立历史 |
| 多知识库检索 | `checkpointer=None` | 作为父图流程的一部分即可 |
| Agent + RAG 混合 | `checkpointer=True` | Agent 需要独立的推理历史 |

---

## 10. 总结

**子图 Checkpointer 三种模式的核心区别：**

- **`None`（继承）**：子图是父图流程的一部分，状态由父图统一管理 -- 大部分场景的默认选择
- **`True`（独立）**：子图是独立的有状态实体，需要自己的记忆 -- 多代理系统的核心能力
- **`False`（禁用）**：子图是无状态的纯函数，不需要任何持久化 -- 性能优先或安全要求

**一句话记忆：None 跟随父图走，True 独立有记忆，False 轻装无负担。**

**[来源: reference/source_subgraph_01.md:87-92, reference/context7_langgraph_01.md:86-90, LangGraph 源码 state.py:1057]**
