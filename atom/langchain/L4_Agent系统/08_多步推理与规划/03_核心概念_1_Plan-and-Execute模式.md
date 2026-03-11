# 核心概念 1：Plan-and-Execute 模式

> LangChain Agent 系统 · 第 8 个知识点 · 核心概念 1/5

---

## 一句话定义

**Plan-and-Execute 是将"规划"和"执行"分离为独立组件的 Agent 架构，先用大模型生成完整计划，再用小模型或工具逐步执行。**

---

## 架构全景

### 标准 Plan-and-Execute 架构

```
┌─────────────────────────────────────────────────────────┐
│                 Plan-and-Execute Agent                   │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐       │
│  │ Planner  │───→│ Executor │───→│  Replanner   │       │
│  │ (规划器)  │    │ (执行器)  │    │ (重规划器)    │       │
│  │          │    │          │    │              │       │
│  │ GPT-4    │    │ GPT-4-mini│   │ GPT-4        │       │
│  │ Claude   │    │ 工具调用   │    │ Claude       │       │
│  └──────────┘    └──────────┘    └──┬───────────┘       │
│       ↑                             │                   │
│       │          ┌──────────┐       │                   │
│       └──────────│ 需要重规划 │←──────┘                   │
│                  └──────────┘                           │
│                       │                                 │
│                  ┌──────────┐                           │
│                  │ 完成任务  │──→ 最终输出                │
│                  └──────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### 三个核心组件

#### 1. Planner（规划器）

**职责**：接收用户输入，生成完整的多步任务计划。

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# 定义计划的结构化输出
class Plan(BaseModel):
    """Agent 的执行计划"""
    steps: List[str] = Field(
        description="按顺序需要执行的步骤列表，每个步骤应该是一个明确的任务"
    )

# 使用 Structured Output 让 LLM 生成结构化计划
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
planner = planner_llm.with_structured_output(Plan)

# 生成计划
plan = planner.invoke(
    "帮我对比 Python 和 Rust 在 Web 开发中的优劣势，写一份分析报告"
)
print(plan.steps)
# ['1. 搜索 Python Web 框架的优劣势',
#  '2. 搜索 Rust Web 框架的优劣势',
#  '3. 对比两者的性能差异',
#  '4. 对比两者的开发效率',
#  '5. 编写对比分析报告']
```

#### 2. Executor（执行器）

**职责**：接收单个步骤，使用工具完成该步骤的任务。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索相关信息"""
    # 实际实现会调用搜索 API
    return f"搜索结果：关于 {query} 的信息..."

@tool
def write_report(content: str) -> str:
    """编写报告"""
    return f"报告已生成：{content[:100]}..."

# 执行器使用更轻量的模型
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools([search, write_report])

# 执行单个步骤
def execute_step(state):
    """执行计划中的当前步骤"""
    current_step = state["plan"][state["step_index"]]
    # 让小模型决定调用什么工具并执行
    result = executor_llm_with_tools.invoke(
        f"请完成以下任务：{current_step}\n"
        f"之前的执行结果：{state.get('past_results', [])}"
    )
    return {"result": result, "step_index": state["step_index"] + 1}
```

#### 3. Replanner（重规划器）

**职责**：检查执行结果，决定是完成还是重新规划。

```python
class ReplanDecision(BaseModel):
    """重规划决策"""
    is_complete: bool = Field(description="任务是否已完成")
    final_answer: str = Field(default="", description="如果完成，最终答案")
    new_plan: List[str] = Field(default_factory=list, description="如果未完成，新的计划步骤")

replanner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
replanner = replanner_llm.with_structured_output(ReplanDecision)

def replan_step(state):
    """检查进度，决定下一步"""
    decision = replanner.invoke(
        f"原始目标：{state['objective']}\n"
        f"已执行步骤和结果：{state['past_results']}\n"
        f"剩余计划：{state['plan'][state['step_index']:]}\n"
        f"请判断任务是否完成。如果未完成，请调整计划。"
    )
    return decision
```

---

## 与 ReAct 模式的对比

| 维度 | ReAct 模式 | Plan-and-Execute 模式 |
|------|-----------|---------------------|
| **推理方式** | 每一步都推理（Thought-Action-Observation） | 先全局规划，再逐步执行 |
| **LLM 调用次数** | 每个工具调用都需要 LLM | 只在规划和重规划时调用大模型 |
| **成本** | 高（大模型调用 N 次） | 低（大模型调用 2-3 次） |
| **速度** | 慢（串行 LLM+工具） | 快（规划后并行执行） |
| **适用场景** | 简单任务、需要灵活应对的场景 | 复杂多步任务、可预期的工作流 |
| **全局视角** | ❌ 一次只看一步 | ✅ 先看全局再行动 |
| **错误恢复** | 有限（靠 LLM 自纠错） | 强（可重规划） |

### 何时用哪种？

```python
# 用 ReAct 的场景：
# - 简单问答（1-2步就能完成）
# - 需要实时灵活应对的对话
# - 任务不可预期，需要探索

# 用 Plan-and-Execute 的场景：
# - 复杂多步任务（3步以上）
# - 任务可以预先规划
# - 需要成本优化
# - 需要进度追踪和可观测性
```

---

## 三种 Plan-and-Execute 变体

### 变体1：标准 Plan-and-Execute

```
Planner → [步骤1, 步骤2, 步骤3]
              ↓
Executor → 逐步执行（串行）
              ↓
Replanner → 检查/调整
```

**特点**：最简单的实现，步骤串行执行。
**论文**：Wang et al. "Plan-and-Solve Prompting" (2023)

### 变体2：ReWOO（Reasoning WithOut Observations）

```
Planner → [Plan+E1, Plan+E2(#E1), Plan+E3(#E2)]
              ↓
Worker → 顺序执行，替换变量引用（#E1 → 结果1）
              ↓
Solver → 整合所有结果生成答案
```

**特点**：支持变量引用（`#E1`），每个任务只有必要上下文。
**论文**：Xu et al. "ReWOO" (2023)

```python
# ReWOO 计划示例
plan = """
Plan: 我需要找到今年超级碗的参赛队伍
E1: Search[今年超级碗的参赛队伍是哪些？]
Plan: 我需要知道第一支队的四分卫
E2: LLM[#E1 中第一支队的四分卫是谁？]
Plan: 我需要查找该四分卫的数据
E3: Search[#E2 的本赛季数据]
"""
# 注意：E2 引用了 E1 的结果，E3 引用了 E2 的结果
```

### 变体3：LLMCompiler

```
Planner → DAG任务图（流式输出）
              ↓
Task Fetching Unit → 依赖满足后立即并行执行
              ↓
Joiner → 检查/重规划
```

**特点**：支持 DAG 并行执行，速度最快（3.6x 加速）。
**论文**：Kim et al. "LLMCompiler" (2023)

```python
# LLMCompiler 计划示例（DAG）
# 任务1和任务2可以并行执行
plan_dag = {
    "task_1": {"tool": "search", "args": "Python web frameworks", "deps": []},
    "task_2": {"tool": "search", "args": "Rust web frameworks", "deps": []},
    "task_3": {"tool": "compare", "args": "${1}, ${2}", "deps": [1, 2]},
    # task_3 依赖 task_1 和 task_2 的结果
}
```

---

## 在 LangChain/LangGraph 中的实现

### 完整实现骨架

```python
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import operator

# ===== 1. 定义状态 =====
class PlanExecuteState(TypedDict):
    """Agent 的状态"""
    input: str                                    # 用户输入
    plan: List[str]                               # 当前计划
    step_index: int                               # 当前步骤索引
    past_steps: Annotated[List[tuple], operator.add]  # 已执行的步骤和结果
    response: str                                 # 最终响应

# ===== 2. 定义节点函数 =====
planner_llm = ChatOpenAI(model="gpt-4o")
executor_llm = ChatOpenAI(model="gpt-4o-mini")

class Plan(BaseModel):
    steps: List[str] = Field(description="执行步骤列表")

async def plan_step(state: PlanExecuteState):
    """规划节点：生成执行计划"""
    planner = planner_llm.with_structured_output(Plan)
    plan = await planner.ainvoke(
        f"为以下任务制定执行计划：\n{state['input']}"
    )
    return {"plan": plan.steps, "step_index": 0}

async def execute_step(state: PlanExecuteState):
    """执行节点：执行当前步骤"""
    current_step = state["plan"][state["step_index"]]
    result = await executor_llm.ainvoke(
        f"执行任务：{current_step}\n已有结果：{state['past_steps']}"
    )
    return {
        "past_steps": [(current_step, result.content)],
        "step_index": state["step_index"] + 1
    }

async def replan_step(state: PlanExecuteState):
    """重规划节点：检查是否完成"""
    # 简化逻辑：如果所有步骤都执行完了，生成最终答案
    if state["step_index"] >= len(state["plan"]):
        response = await planner_llm.ainvoke(
            f"目标：{state['input']}\n执行结果：{state['past_steps']}\n请生成最终答案。"
        )
        return {"response": response.content}
    return {}

# ===== 3. 定义路由逻辑 =====
def should_continue(state: PlanExecuteState):
    """决定是继续执行还是结束"""
    if state.get("response"):
        return "end"
    if state["step_index"] < len(state["plan"]):
        return "execute"
    return "replan"

# ===== 4. 构建图 =====
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", plan_step)
graph.add_node("executor", execute_step)
graph.add_node("replanner", replan_step)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges(
    "executor",
    should_continue,
    {"execute": "executor", "replan": "replanner", "end": END}
)
graph.add_conditional_edges(
    "replanner",
    should_continue,
    {"execute": "executor", "replan": "replanner", "end": END}
)

app = graph.compile()
```

---

## 实际应用场景

| 场景 | 规划器做什么 | 执行器做什么 |
|------|-----------|-----------|
| **研究报告** | 列出研究步骤 | 搜索、阅读、总结 |
| **数据分析** | 确定分析流程 | 查数据库、计算、可视化 |
| **代码生成** | 设计代码架构 | 编写各模块代码 |
| **客服工单** | 分解问题解决步骤 | 查知识库、调用 API |
| **项目管理** | 分解为子任务 | 分配和跟踪任务 |

---

## 关键要点

1. **核心思想**：规划和执行分离，大模型管规划，小模型管执行
2. **三大优势**：更快、更省、更好
3. **三种变体**：标准 P&E（串行）、ReWOO（变量引用）、LLMCompiler（DAG 并行）
4. **适用场景**：复杂多步任务，特别是可预先规划的工作流
5. **LangGraph 实现**：用 StateGraph 构建 Planner→Executor→Replanner 图

---

[来源: reference/fetch_plan_execute_01.md | https://blog.langchain.com/planning-agents/]
[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]
[来源: sourcecode/langchain/libs/langchain/langchain_classic/agents/agent.py]

**下一步**：阅读 [03_核心概念_2_任务分解策略.md](03_核心概念_2_任务分解策略.md) 学习如何拆分任务
