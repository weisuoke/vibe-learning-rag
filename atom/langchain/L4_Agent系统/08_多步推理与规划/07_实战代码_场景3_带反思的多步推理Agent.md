# 实战代码 场景3：带反思的多步推理 Agent

> LangChain Agent 系统 · 第 8 个知识点 · 实战代码 3/3
> 难度：⭐⭐⭐ 高级

---

## 场景描述

**构建一个研究助手 Agent，在 Plan-and-Execute 基础上加入 Reflection（自我反思）和 Self-Critique（自我批评）机制，确保输出质量。**

### 核心特性

- **Plan-and-Execute 架构**：先规划后执行
- **Reflection 反思**：执行后自我评估输出质量
- **Self-Critique**：主动挑战自己的结论
- **质量门控**：不达标则迭代改进

---

## 完整实现

### 第一步：定义反思模型

```python
from typing import List, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

# === 反思模型 ===
class QualityAssessment(BaseModel):
    """质量评估结果"""
    completeness: int = Field(description="完整性评分 1-10")
    accuracy: int = Field(description="准确性评分 1-10")
    relevance: int = Field(description="相关性评分 1-10")
    depth: int = Field(description="深度评分 1-10")
    overall: int = Field(description="综合评分 1-10")
    issues: List[str] = Field(
        default_factory=list,
        description="发现的问题"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="改进建议"
    )
    pass_threshold: bool = Field(
        description="是否通过质量阈值（综合>=7分）"
    )

class SelfCritiqueResult(BaseModel):
    """自我批评结果"""
    original_point: str = Field(description="原始论点")
    counter_argument: str = Field(description="反面论据")
    weakness: str = Field(description="最大弱点")
    balanced_view: str = Field(description="更平衡的观点")

class Plan(BaseModel):
    steps: List[str] = Field(description="执行步骤")

# === 状态 ===
class ReflectiveState(TypedDict):
    input: str
    plan: List[str]
    step_index: int
    past_steps: Annotated[List[tuple], operator.add]
    draft_response: str                # 草稿回答
    reflections: List[dict]            # 反思历史
    response: str                      # 最终回答
    reflection_count: int              # 反思次数

# === 模型和工具 ===
planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def search(query: str) -> str:
    """搜索研究资料"""
    return f"搜索 '{query}' 的结果：找到了相关研究论文和技术文章。"

@tool
def deep_analyze(data: str) -> str:
    """深度分析研究材料"""
    return f"深度分析结果：{data[:100]} 的核心发现和洞察。"

tools = [search, deep_analyze]
```

### 第二步：规划和执行节点

```python
async def plan_node(state: ReflectiveState):
    """规划节点"""
    planner = planner_llm.with_structured_output(Plan)
    plan = await planner.ainvoke(
        f"为以下研究任务制定 3-5 步执行计划：\n{state['input']}"
    )
    print(f"📋 研究计划: {plan.steps}")
    return {"plan": plan.steps, "step_index": 0, "reflection_count": 0}

async def execute_node(state: ReflectiveState):
    """执行节点"""
    step = state["plan"][state["step_index"]]
    print(f"⚡ 执行步骤 {state['step_index']+1}/{len(state['plan'])}: {step}")

    context = ""
    if state["past_steps"]:
        context = "\n已有结果：" + "\n".join(
            f"- {s}: {r[:150]}" for s, r in state["past_steps"]
        )

    result = await executor_llm.ainvoke(f"执行：{step}{context}")
    print(f"  ✅ 完成")

    return {
        "past_steps": [(step, result.content)],
        "step_index": state["step_index"] + 1,
    }
```

### 第三步：生成草稿节点

```python
async def draft_node(state: ReflectiveState):
    """生成草稿回答"""
    print(f"\n📝 生成研究报告草稿...")

    # 检查是否有之前的反思建议需要采纳
    improvement_hints = ""
    if state.get("reflections"):
        latest = state["reflections"][-1]
        improvement_hints = (
            f"\n\n请特别注意改进以下问题：\n"
            f"- 问题：{latest.get('issues', [])}\n"
            f"- 建议：{latest.get('suggestions', [])}"
        )

    results_summary = "\n".join(
        f"- {step}: {result}" for step, result in state["past_steps"]
    )

    draft = await planner_llm.ainvoke(
        f"基于以下研究结果，撰写研究报告。\n\n"
        f"研究目标：{state['input']}\n\n"
        f"研究结果：\n{results_summary}"
        f"{improvement_hints}\n\n"
        f"要求：结构清晰、论据充分、结论明确。"
    )
    print(f"  ✅ 草稿完成（{len(draft.content)} 字符）")
    return {"draft_response": draft.content}
```

### 第四步：反思节点（核心）

```python
async def reflect_node(state: ReflectiveState):
    """反思节点：评估草稿质量"""
    print(f"\n🔍 反思评估（第 {state.get('reflection_count', 0)+1} 轮）...")

    # 1. 质量评估
    assessor = judge_llm.with_structured_output(QualityAssessment)
    assessment = await assessor.ainvoke(
        f"请严格评估以下研究报告的质量。\n\n"
        f"研究目标：{state['input']}\n\n"
        f"报告内容：\n{state['draft_response']}\n\n"
        f"评估维度：完整性、准确性、相关性、深度。\n"
        f"综合评分 >= 7 视为通过。请严格打分。"
    )

    print(f"  完整性: {assessment.completeness}/10")
    print(f"  准确性: {assessment.accuracy}/10")
    print(f"  相关性: {assessment.relevance}/10")
    print(f"  深度: {assessment.depth}/10")
    print(f"  综合: {assessment.overall}/10")
    print(f"  通过: {'✅' if assessment.pass_threshold else '❌'}")

    if assessment.issues:
        print(f"  问题: {assessment.issues}")

    # 2. 自我批评（仅在有明确结论时）
    critique_info = ""
    if assessment.overall >= 5:
        critiquer = judge_llm.with_structured_output(SelfCritiqueResult)
        critique = await critiquer.ainvoke(
            f"请对以下研究报告中的核心结论提出批评。\n\n"
            f"报告：{state['draft_response'][:500]}\n\n"
            f"找出最大的弱点和偏见，提出更平衡的观点。"
        )
        critique_info = (
            f"自我批评 - 弱点: {critique.weakness}, "
            f"平衡观点: {critique.balanced_view}"
        )
        print(f"  🎯 {critique_info}")

    # 记录反思结果
    reflection_record = {
        "round": state.get("reflection_count", 0) + 1,
        "overall_score": assessment.overall,
        "passed": assessment.pass_threshold,
        "issues": assessment.issues,
        "suggestions": assessment.suggestions,
        "critique": critique_info,
    }

    reflections = state.get("reflections", [])
    reflections.append(reflection_record)

    return {
        "reflections": reflections,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }
```

### 第五步：路由逻辑

```python
def after_execute(state: ReflectiveState):
    """执行后路由"""
    if state.get("response"):
        return "end"
    if state["step_index"] < len(state["plan"]):
        return "execute"
    return "draft"  # 所有步骤完成，生成草稿

def after_reflect(state: ReflectiveState):
    """反思后路由"""
    reflections = state.get("reflections", [])
    if not reflections:
        return "draft"

    latest = reflections[-1]
    reflection_count = state.get("reflection_count", 0)

    # 通过质量阈值 → 完成
    if latest.get("passed", False):
        return "finalize"

    # 超过最大反思次数 → 强制完成
    if reflection_count >= 3:
        print("  ⚠️ 达到最大反思次数，使用当前最佳版本")
        return "finalize"

    # 未通过 → 重新生成草稿
    print(f"  🔄 质量未达标，重新生成草稿")
    return "draft"

async def finalize_node(state: ReflectiveState):
    """最终输出节点"""
    reflections = state.get("reflections", [])
    rounds = len(reflections)
    final_score = reflections[-1]["overall_score"] if reflections else "N/A"

    print(f"\n✨ 最终输出（经过 {rounds} 轮反思，评分 {final_score}/10）")
    return {"response": state["draft_response"]}
```

### 第六步：构建图

```python
graph = StateGraph(ReflectiveState)

# 添加节点
graph.add_node("planner", plan_node)
graph.add_node("executor", execute_node)
graph.add_node("drafter", draft_node)
graph.add_node("reflector", reflect_node)
graph.add_node("finalizer", finalize_node)

# 添加边
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", after_execute, {
    "execute": "executor",
    "draft": "drafter",
    "end": END,
})
graph.add_edge("drafter", "reflector")
graph.add_conditional_edges("reflector", after_reflect, {
    "draft": "drafter",     # 质量不达标，重新生成
    "finalize": "finalizer", # 质量达标或达到上限
})
graph.add_edge("finalizer", END)

app = graph.compile()
```

### 第七步：运行

```python
import asyncio

async def main():
    result = await app.ainvoke({
        "input": "分析 RAG 技术在 2026 年的发展趋势，"
                 "包括关键技术突破、主要挑战和未来方向，"
                 "给出对 AI 应用开发者的建议。"
    })
    print("\n" + "="*60)
    print("📊 研究报告：")
    print(result["response"])
    print(f"\n📈 反思历程: {len(result.get('reflections', []))} 轮")
    for r in result.get("reflections", []):
        print(f"  第{r['round']}轮: 评分{r['overall_score']}/10 "
              f"{'✅' if r['passed'] else '❌'}")

asyncio.run(main())
```

---

## 执行流程可视化

```
用户输入
  ↓
Planner → [步骤1, 步骤2, 步骤3]
  ↓
Executor → 逐步执行
  ↓
Drafter → 生成研究报告草稿（第1版）
  ↓
Reflector → 质量评估 + 自我批评
  │
  ├── 通过（≥7分）→ Finalizer → 输出
  │
  └── 未通过 → Drafter → 改进草稿（第2版）
                  ↓
              Reflector → 再次评估
                  │
                  ├── 通过 → 输出
                  └── 未通过且 < 3轮 → 继续改进
                  └── 达到3轮 → 强制输出
```

---

## 关键设计亮点

| 特性 | 实现方式 | 价值 |
|------|---------|------|
| 质量门控 | `QualityAssessment.pass_threshold` | 确保输出达标 |
| 自我批评 | `SelfCritiqueResult` | 避免偏见和片面 |
| 迭代改进 | 反思 → 重新生成 → 再反思 | 质量逐轮提升 |
| 安全退出 | `reflection_count >= 3` | 防止无限循环 |
| 多维评估 | 完整性/准确性/相关性/深度 | 全面质量评估 |
| 改进指导 | issues + suggestions 传入下轮 | 定向改进 |

---

## 三个场景的递进关系

```
场景1（基础 P&E）      → 会规划、会执行
场景2（自适应规划）     → 会调整、会重试
场景3（带反思的推理）   → 会反思、会改进
              ↓
      完整的多步推理 Agent
```

---

[来源: reference/search_多步推理_01.md | LangGraph Reflexion 教程]
[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

**下一步**：阅读 [08_面试必问.md](08_面试必问.md) 准备面试常见问题
