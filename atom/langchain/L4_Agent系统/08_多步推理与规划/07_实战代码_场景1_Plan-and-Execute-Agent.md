# 实战代码 场景1：Plan-and-Execute Agent

> LangChain Agent 系统 · 第 8 个知识点 · 实战代码 1/3
> 难度：⭐⭐ 进阶

---

## 场景描述

**构建一个完整的 Plan-and-Execute Agent，能够接收复杂任务，自动分解为子步骤，逐步执行并在必要时动态调整计划。**

### 应用场景

- 技术调研报告生成
- 多数据源对比分析
- 复杂问题分步解答

---

## 完整实现

### 第一步：环境准备

```python
# 安装依赖
# pip install langchain langchain-openai langgraph pydantic

from typing import List, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import operator
import json
from dotenv import load_dotenv

load_dotenv()
```

### 第二步：定义工具

```python
@tool
def search(query: str) -> str:
    """搜索网络获取相关信息"""
    # 实际项目中接入搜索 API（如 Tavily、SerpAPI）
    return f"[搜索结果] 关于 '{query}' 的最新信息：" \
           f"这是一个模拟搜索结果，包含了 {query} 的关键信息。"

@tool
def analyze(data: str) -> str:
    """分析和对比数据"""
    return f"[分析结果] 对以下数据的分析：{data[:100]}..."

@tool
def write_report(content: str) -> str:
    """编写报告"""
    return f"[报告] 已生成报告，包含 {len(content)} 字符的内容。"

tools = [search, analyze, write_report]
```

### 第三步：定义状态和模型

```python
# === 状态定义 ===
class PlanExecuteState(TypedDict):
    input: str                                       # 用户输入
    plan: List[str]                                  # 当前计划
    step_index: int                                  # 当前步骤
    past_steps: Annotated[List[tuple], operator.add] # 已完成步骤
    response: str                                    # 最终响应

# === Structured Output 模型 ===
class Plan(BaseModel):
    """执行计划"""
    steps: List[str] = Field(
        description="按顺序执行的步骤列表，每步对应具体工具调用，3-7步"
    )

class StepResult(BaseModel):
    """步骤执行结果"""
    tool_name: str = Field(description="使用的工具名")
    tool_input: str = Field(description="工具输入")
    summary: str = Field(description="结果摘要")

class ReplanDecision(BaseModel):
    """重规划决策"""
    action: Literal["continue", "replan", "complete"] = Field(
        description="continue=继续执行, replan=调整计划, complete=任务完成"
    )
    final_answer: str = Field(
        default="", description="如果完成，最终答案"
    )
    updated_steps: List[str] = Field(
        default_factory=list,
        description="如果重规划，更新后的剩余步骤"
    )
    reason: str = Field(description="决策理由")

# === 模型 ===
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
executor_with_tools = executor_llm.bind_tools(tools)
```

### 第四步：定义节点函数

```python
async def plan_node(state: PlanExecuteState):
    """规划节点：生成执行计划"""
    planner = planner_llm.with_structured_output(Plan)
    plan = await planner.ainvoke(
        f"""请为以下任务制定执行计划。

要求：
- 步骤数量控制在 3-7 步
- 每步应对应一个具体的工具调用
- 步骤按依赖关系排序
- 最后一步应该是验证或总结

可用工具：search（搜索信息）、analyze（分析数据）、write_report（写报告）

任务：{state['input']}"""
    )
    print(f"📋 生成计划：{len(plan.steps)} 步")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")
    return {"plan": plan.steps, "step_index": 0}

async def execute_node(state: PlanExecuteState):
    """执行节点：执行当前步骤"""
    current_step = state["plan"][state["step_index"]]
    step_num = state["step_index"] + 1
    total = len(state["plan"])

    print(f"\n⚡ 执行步骤 {step_num}/{total}: {current_step}")

    # 让执行器决定调用什么工具
    context = ""
    if state["past_steps"]:
        context = "\n已有结果：\n" + "\n".join(
            f"- {step}: {result[:200]}" for step, result in state["past_steps"]
        )

    response = await executor_with_tools.ainvoke(
        f"请完成以下任务：{current_step}{context}"
    )

    # 处理工具调用
    result = response.content
    if response.tool_calls:
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            if tc["name"] in tool_map:
                tool_result = tool_map[tc["name"]].invoke(tc["args"])
                result = f"[{tc['name']}] {tool_result}"

    print(f"  ✅ 结果: {result[:150]}...")

    return {
        "past_steps": [(current_step, result)],
        "step_index": state["step_index"] + 1,
    }

async def replan_node(state: PlanExecuteState):
    """重规划节点：评估进度，决定下一步"""
    replanner = planner_llm.with_structured_output(ReplanDecision)

    remaining = state["plan"][state["step_index"]:]
    past_summary = "\n".join(
        f"- {step}: {result[:200]}" for step, result in state["past_steps"]
    )

    decision = await replanner.ainvoke(
        f"""评估当前执行进度，决定下一步行动。

原始目标：{state['input']}

已完成步骤和结果：
{past_summary}

剩余计划：{remaining if remaining else "无（所有步骤已执行）"}

决策规则：
- 如果还有剩余步骤且方向正确 → continue
- 如果需要调整剩余步骤 → replan（提供更新后的步骤）
- 如果目标已达成 → complete（提供最终答案）"""
    )

    print(f"\n🔄 Replanner 决策: {decision.action} - {decision.reason}")

    if decision.action == "complete":
        return {"response": decision.final_answer}
    elif decision.action == "replan":
        print(f"  📝 更新计划: {decision.updated_steps}")
        return {
            "plan": [s for s, _ in state["past_steps"]]
                  + decision.updated_steps,
            "step_index": len(state["past_steps"]),
        }
    return {}  # continue
```

### 第五步：构建图

```python
def should_continue(state: PlanExecuteState):
    """路由逻辑"""
    if state.get("response"):
        return "end"
    if state["step_index"] < len(state["plan"]):
        return "execute"
    return "replan"

# 构建 StateGraph
graph = StateGraph(PlanExecuteState)

# 添加节点
graph.add_node("planner", plan_node)
graph.add_node("executor", execute_node)
graph.add_node("replanner", replan_node)

# 添加边
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

# 编译
app = graph.compile()
```

### 第六步：运行

```python
import asyncio

async def main():
    result = await app.ainvoke({
        "input": "帮我对比 ChromaDB 和 Milvus 在 RAG 项目中的适用性，"
                 "包括性能、易用性、扩展性，给出选型建议。"
    })
    print("\n" + "="*60)
    print("📊 最终结果：")
    print(result["response"])

asyncio.run(main())
```

### 预期输出

```
📋 生成计划：5 步
  1. 搜索 ChromaDB 的核心特性、性能指标和适用场景
  2. 搜索 Milvus 的核心特性、性能指标和适用场景
  3. 对比分析两者在性能、易用性、扩展性方面的差异
  4. 结合 RAG 项目需求分析各自的优劣势
  5. 综合生成选型建议报告

⚡ 执行步骤 1/5: 搜索 ChromaDB 的核心特性...
  ✅ 结果: [search] ChromaDB 是轻量级嵌入式向量数据库...

⚡ 执行步骤 2/5: 搜索 Milvus 的核心特性...
  ✅ 结果: [search] Milvus 是高性能分布式向量数据库...

⚡ 执行步骤 3/5: 对比分析...
  ✅ 结果: [analyze] 性能对比：Milvus 在大规模数据上领先...

⚡ 执行步骤 4/5: 结合 RAG 项目需求分析...
  ✅ 结果: [analyze] RAG 场景下：原型用 ChromaDB，生产用 Milvus...

⚡ 执行步骤 5/5: 综合生成选型建议报告...
  ✅ 结果: [write_report] 已生成报告...

🔄 Replanner 决策: complete - 所有步骤已执行，目标达成

============================================================
📊 最终结果：
建议：小规模 RAG 原型选 ChromaDB（易上手），
生产环境大规模 RAG 选 Milvus（高性能、高扩展）。
```

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 规划器模型 | gpt-4o | 规划需要强推理能力 |
| 执行器模型 | gpt-4o-mini | 执行简单任务，降低成本 |
| 步骤数限制 | 3-7 步 | 平衡粒度和效率 |
| 重规划触发 | 所有步骤执行完后 | 简单可靠 |
| 状态管理 | Annotated + operator.add | 自动累积历史结果 |

---

## 扩展方向

1. **加入真实搜索工具**（Tavily API）替换模拟搜索
2. **加入 Human-in-the-Loop**（规划后暂停等审批）
3. **加入 Reflection**（执行后自我评估）
4. **支持并行执行**（无依赖步骤同时执行）

---

[来源: reference/fetch_plan_execute_01.md | LangChain Blog]
[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

**下一步**：阅读 [07_实战代码_场景2_自适应规划系统.md](07_实战代码_场景2_自适应规划系统.md) 实现动态任务分解
