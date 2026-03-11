# 核心概念 5：LangGraph 规划工作流

> LangChain Agent 系统 · 第 8 个知识点 · 核心概念 5/5

---

## 一句话定义

**LangGraph 规划工作流是用 StateGraph 或 Functional API 实现 Plan-and-Execute 架构的具体技术方案，包括规划节点、执行节点、重规划节点的图结构编排，以及 TodoListMiddleware 等原生规划能力。**

---

## LangGraph 实现规划的两种方式

### 方式1：StateGraph（类定义方式）

```
┌──────────────┐
│ 用户输入      │
└──────┬───────┘
       ↓
┌──────────────┐
│ plan_node    │  ← 规划节点（大模型）
└──────┬───────┘
       ↓
┌──────────────┐
│ execute_node │  ← 执行节点（小模型+工具）
└──────┬───────┘
       ↓
┌──────────────┐    ┌──────────────┐
│ replan_node  │───→│   完成？      │
└──────────────┘    └──────┬───────┘
       ↑                   │
       │              是 → END
       └── 否 ────────────┘
```

### 方式2：Functional API（函数式方式）

使用 `@task` 和 `@entrypoint` 装饰器，更简洁。

---

## StateGraph 完整实现

```python
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import operator

# ===== 1. 定义状态 =====
class PlanExecuteState(TypedDict):
    input: str                                       # 用户输入
    plan: List[str]                                  # 当前计划
    step_index: int                                  # 当前步骤索引
    past_steps: Annotated[List[tuple], operator.add] # 已执行步骤
    response: str                                    # 最终响应

# ===== 2. 定义 Structured Output =====
class Plan(BaseModel):
    steps: List[str] = Field(description="按顺序执行的步骤列表")

class ReplanDecision(BaseModel):
    is_complete: bool = Field(description="任务是否已完成")
    final_answer: str = Field(default="", description="最终答案")
    updated_plan: List[str] = Field(
        default_factory=list, description="更新后的剩余步骤"
    )

# ===== 3. 定义节点 =====
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def plan_node(state: PlanExecuteState):
    """规划节点：生成执行计划"""
    planner = planner_llm.with_structured_output(Plan)
    plan = await planner.ainvoke(
        f"为以下任务制定 3-7 个执行步骤：\n{state['input']}"
    )
    return {"plan": plan.steps, "step_index": 0}

async def execute_node(state: PlanExecuteState):
    """执行节点：执行当前步骤"""
    current_step = state["plan"][state["step_index"]]
    # 执行器可以绑定工具
    result = await executor_llm.ainvoke(
        f"执行任务：{current_step}\n已有结果：{state['past_steps']}"
    )
    return {
        "past_steps": [(current_step, result.content)],
        "step_index": state["step_index"] + 1,
    }

async def replan_node(state: PlanExecuteState):
    """重规划节点：检查进度，决定下一步"""
    replanner = planner_llm.with_structured_output(ReplanDecision)
    decision = await replanner.ainvoke(
        f"目标：{state['input']}\n"
        f"已完成：{state['past_steps']}\n"
        f"剩余计划：{state['plan'][state['step_index']:]}\n"
        f"请判断任务是否完成。如果未完成，更新剩余步骤。"
    )
    if decision.is_complete:
        return {"response": decision.final_answer}
    return {
        "plan": [s for _, s in state["past_steps"]]
              + decision.updated_plan,
        "step_index": len(state["past_steps"]),
    }

# ===== 4. 路由逻辑 =====
def should_continue(state: PlanExecuteState):
    if state.get("response"):
        return "end"
    if state["step_index"] < len(state["plan"]):
        return "execute"
    return "replan"

# ===== 5. 构建图 =====
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", plan_node)
graph.add_node("executor", execute_node)
graph.add_node("replanner", replan_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {
    "execute": "executor",
    "replan": "replanner",
    "end": END,
})
graph.add_conditional_edges("replanner", should_continue, {
    "execute": "executor",
    "replan": "replanner",
    "end": END,
})

app = graph.compile()
```

---

## Functional API 实现

LangGraph 的 `@task` 和 `@entrypoint` 提供更简洁的函数式实现。

```python
from langgraph.func import task, entrypoint
from langchain_openai import ChatOpenAI

planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@task
def generate_plan(objective: str) -> list[str]:
    """生成执行计划"""
    planner = planner_llm.with_structured_output(Plan)
    plan = planner.invoke(
        f"为以下任务制定执行步骤：{objective}"
    )
    return plan.steps

@task
def execute_step(step: str, context: list) -> str:
    """执行单个步骤"""
    result = executor_llm.invoke(
        f"执行：{step}\n上下文：{context}"
    )
    return result.content

@task
def check_completion(objective: str, results: list) -> dict:
    """检查是否完成"""
    replanner = planner_llm.with_structured_output(ReplanDecision)
    return replanner.invoke(
        f"目标：{objective}\n执行结果：{results}\n是否完成？"
    )

@entrypoint()
def plan_execute_agent(objective: str):
    """Plan-and-Execute Agent 入口"""
    # 1. 规划
    steps = generate_plan(objective).result()

    results = []
    for step in steps:
        # 2. 逐步执行
        result = execute_step(step, results).result()
        results.append((step, result))

    # 3. 检查完成
    decision = check_completion(objective, results).result()
    if decision.is_complete:
        return decision.final_answer

    # 4. 如果未完成，可以继续迭代（简化处理）
    return f"已完成 {len(results)} 步，结果：{results}"
```

---

## Orchestrator-Worker 模式

适合报告生成、多源分析等场景，先规划章节结构，再分发给 Worker 并行执行。

```python
from langgraph.func import task, entrypoint
from pydantic import BaseModel, Field
from typing import List

class Section(BaseModel):
    name: str = Field(description="章节名称")
    description: str = Field(description="章节内容概要")

class Sections(BaseModel):
    sections: List[Section] = Field(description="报告章节列表")

@task
def plan_sections(topic: str) -> list[Section]:
    """规划报告章节"""
    planner = planner_llm.with_structured_output(Sections)
    result = planner.invoke(
        f"为以下主题规划报告章节：{topic}"
    )
    return result.sections

@task
def write_section(section: Section, topic: str) -> str:
    """编写单个章节（Worker）"""
    result = executor_llm.invoke(
        f"主题：{topic}\n请编写章节：{section.name}\n"
        f"内容要求：{section.description}"
    )
    return result.content

@task
def synthesize(topic: str, sections: list) -> str:
    """合成最终报告"""
    result = planner_llm.invoke(
        f"主题：{topic}\n章节内容：{sections}\n请合成完整报告。"
    )
    return result.content

@entrypoint()
def report_agent(topic: str):
    """报告生成 Agent"""
    # 1. Orchestrator: 规划章节
    sections = plan_sections(topic).result()

    # 2. Workers: 并行编写章节
    section_futures = [
        write_section(section, topic) for section in sections
    ]
    section_results = [f.result() for f in section_futures]

    # 3. Synthesizer: 合成报告
    report = synthesize(topic, section_results).result()
    return report
```

---

## Human-in-the-Loop 规划审批

LangGraph 支持在规划后暂停，等待人工审批。

```python
from langgraph.types import interrupt, Command

@entrypoint()
def agent_with_approval(objective: str):
    """带人工审批的规划 Agent"""
    # 1. 生成计划
    steps = generate_plan(objective).result()

    # 2. 暂停，等待人工审批
    approved = interrupt({
        "plan": steps,
        "question": "是否批准此执行计划？"
    })

    if not approved:
        # 3a. 人工拒绝，重新规划
        steps = generate_plan(
            f"{objective}\n用户反馈：请调整计划"
        ).result()
    # 3b. 人工批准，继续执行
    results = []
    for step in steps:
        result = execute_step(step, results).result()
        results.append(result)

    return results

# 运行
config = {"configurable": {"thread_id": "user-1"}}

# 首次调用 → 会在 interrupt 处暂停
result = agent_with_approval.invoke("写一份技术调研报告", config=config)

# 恢复执行 → 批准计划
result = agent_with_approval.invoke(
    Command(resume=True), config=config
)
```

---

## TodoListMiddleware（2026 新特性）

LangChain 2026 原生的任务规划中间件，无需手动构建规划节点。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

# 创建带规划能力的 Agent
agent = create_agent(
    model="gpt-4.1",
    tools=[search, read_file, write_file],
    middleware=[TodoListMiddleware()]
)

# TodoListMiddleware 自动提供：
# - write_todos 工具：Agent 可调用此工具分解任务
# - 任务进度跟踪：自动追踪每个步骤状态
# - 规划引导：系统提示词引导 Agent 先规划再执行
```

### Deep Agents SDK

更高级的原生规划方案，组合多个中间件：

```python
from langchain.agents import create_deep_agent

agent = create_deep_agent(
    model="gpt-4.1",
    tools=[search, code_executor, file_manager],
    # 自动包含三大中间件：
    # - TodoListMiddleware: 任务规划和跟踪
    # - FilesystemMiddleware: 文件系统和长期记忆
    # - SubAgentMiddleware: 子代理管理
)

# Agent 自动具备：
# 1. 任务分解能力 (write_todos)
# 2. 文件读写能力 (长期记忆)
# 3. 子代理派生能力 (复杂任务分发)
```

---

## 两种方式对比

| 维度 | StateGraph | Functional API |
|------|-----------|---------------|
| **代码风格** | 类 + 字典状态 | 函数 + 装饰器 |
| **灵活度** | 极高（完全自定义） | 高（简洁但稍受限） |
| **学习成本** | 中等 | 低 |
| **适用场景** | 复杂工作流、需要精细控制 | 标准规划流程 |
| **并行执行** | 需要手动实现 | `@task` 自动支持 |
| **中断恢复** | `interrupt_before` | `interrupt()` |

---

## 关键要点

1. **StateGraph 最灵活**：Planner/Executor/Replanner 三节点 + 条件边
2. **Functional API 更简洁**：`@task` + `@entrypoint` 快速构建
3. **Orchestrator-Worker**：适合报告生成、多源分析等并行场景
4. **Human-in-the-Loop**：`interrupt()` 支持规划审批
5. **TodoListMiddleware**：2026 原生方案，无需手动构建规划节点
6. **Deep Agents SDK**：规划 + 文件系统 + 子代理三位一体

---

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]
[来源: reference/context7_langchain_01.md | TodoListMiddleware/Deep Agents 文档]
[来源: reference/fetch_plan_execute_01.md | LangChain Blog]

**下一步**：阅读 [04_最小可用.md](04_最小可用.md) 掌握 20% 核心知识解决 80% 问题
