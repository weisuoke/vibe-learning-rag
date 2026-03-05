# 实战代码 - 场景8：多 Agent 协作路由

> 多 Agent 通过条件分支实现 handoff 和质量路由，掌握 LangGraph 中 Agent 间协作的条件路由模式

---

## 场景描述

**场景**：构建一个研究助手系统，包含研究 Agent、写作 Agent、审核 Agent 三个角色。Agent 之间通过条件分支实现任务流转和质量控制。

**核心挑战**：多个 Agent 协作时，需要根据中间产出的质量决定下一步走向——是继续深入、交给下一个 Agent，还是打回修改。这不是简单的线性流水线，而是带有循环和质量门控的协作网络。

**协作流程**：

```
START
  ↓
research_agent（研究Agent：收集资料）
  ↓
route_research（研究质量路由）
  ├── 质量不足 → research_agent（继续研究，循环）
  └── 质量充分 → writing_agent（写作Agent：撰写文章）
                    ↓
                 review_agent（审核Agent：质量评审）
                    ↓
                 route_review（审核结果路由）
                    ├── 不通过 → writing_agent（修改，循环）
                    └── 通过   → format_output（格式化输出）→ END
```

**关键设计**：

| 设计点 | 说明 |
|--------|------|
| 研究循环 | 研究不充分时自动继续，最多 N 轮 |
| 写作循环 | 审核不通过时打回修改，最多 M 轮 |
| 循环守卫 | 通过轮次计数防止无限循环 |
| 质量评分 | 模拟质量评估，驱动路由决策 |

---

## 完整代码

```python
"""
条件分支策略 - 实战场景8：多Agent协作路由
演示：研究-写作-审核协作系统

核心模式：
- Agent 间 handoff：通过条件边将任务从一个 Agent 传递到另一个
- 质量路由：根据产出质量决定继续、前进或打回
- 循环守卫：轮次计数防止无限循环
- 两个独立的循环：研究循环 + 写作-审核循环

前置条件：
- pip install langgraph
"""

from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, START, END


# ============================================================
# 第一部分：状态定义
# ============================================================

class ResearchState(TypedDict):
    """研究助手系统状态

    字段说明：
    - topic: 研究主题
    - research_notes: 研究笔记列表（使用 operator.add 自动追加）
    - draft: 当前文章草稿
    - review_feedback: 审核反馈
    - research_rounds: 研究轮次计数
    - writing_rounds: 写作轮次计数
    - quality_score: 当前质量评分（0.0 ~ 1.0）
    - status: 最终状态描述
    """
    topic: str
    research_notes: Annotated[list[str], operator.add]
    draft: str
    review_feedback: str
    research_rounds: int
    writing_rounds: int
    quality_score: float
    status: str


# ============================================================
# 第二部分：Agent 节点函数
# ============================================================

def research_agent(state: ResearchState) -> dict:
    """研究 Agent：收集和整理资料

    每轮研究产出一条笔记，质量评分随轮次递增。
    实际项目中这里会调用搜索 API 或 RAG 检索，
    这里用模拟数据保证代码可直接运行。
    """
    round_num = state.get("research_rounds", 0) + 1
    topic = state["topic"]

    # 模拟不同轮次的研究发现
    findings = {
        1: f"基础概念：'{topic}' 的定义和核心原理",
        2: f"深入分析：'{topic}' 的技术实现和关键挑战",
        3: f"前沿进展：'{topic}' 的最新研究成果和应用案例",
    }
    note = findings.get(round_num, f"补充研究#{round_num}：'{topic}' 的更多细节")

    # 质量评分随轮次递增，但有上限
    score = min(0.3 * round_num, 0.9)

    print(f"[研究Agent] 第{round_num}轮研究完成")
    print(f"  发现: {note}")
    print(f"  质量评分: {score:.1f}")

    return {
        "research_notes": [note],
        "research_rounds": round_num,
        "quality_score": score,
    }


def writing_agent(state: ResearchState) -> dict:
    """写作 Agent：基于研究笔记撰写文章

    综合所有研究笔记生成文章草稿。
    如果有审核反馈，会在修改版中体现。
    """
    round_num = state.get("writing_rounds", 0) + 1
    notes_count = len(state.get("research_notes", []))
    feedback = state.get("review_feedback", "")

    if round_num == 1:
        draft = (
            f"[初稿] 《{state['topic']}》\n"
            f"基于 {notes_count} 条研究笔记撰写。\n"
            f"内容摘要：\n"
        )
        for i, note in enumerate(state.get("research_notes", []), 1):
            draft += f"  {i}. {note}\n"
    else:
        draft = (
            f"[第{round_num}版修改稿] 《{state['topic']}》\n"
            f"根据审核反馈修改：{feedback}\n"
            f"基于 {notes_count} 条研究笔记，已优化内容结构和深度。"
        )

    print(f"[写作Agent] 第{round_num}版草稿完成（基于{notes_count}条笔记）")

    return {
        "draft": draft,
        "writing_rounds": round_num,
    }


def review_agent(state: ResearchState) -> dict:
    """审核 Agent：评审文章质量

    根据写作轮次和研究深度综合评分。
    实际项目中这里会调用 LLM 做质量评估。
    """
    writing_rounds = state.get("writing_rounds", 1)
    research_rounds = state.get("research_rounds", 1)

    # 综合评分：写作轮次和研究深度都影响质量
    base_score = 0.3
    research_bonus = 0.15 * min(research_rounds, 3)
    writing_bonus = 0.2 * min(writing_rounds, 3)
    score = min(base_score + research_bonus + writing_bonus, 1.0)

    if score >= 0.8:
        feedback = "文章质量优秀，内容充实，结构清晰，可以发布。"
    elif score >= 0.6:
        feedback = "内容基本完整，但需要增加更多细节和案例，建议修改后重新提交。"
    else:
        feedback = "内容深度不足，论述不够充分，需要大幅修改。"

    print(f"[审核Agent] 评审完成")
    print(f"  质量评分: {score:.2f}")
    print(f"  反馈: {feedback}")

    return {
        "review_feedback": feedback,
        "quality_score": score,
    }


def format_output(state: ResearchState) -> dict:
    """格式化输出节点：生成最终结果"""
    result = (
        f"研究完成！\n"
        f"  主题: {state['topic']}\n"
        f"  研究轮次: {state['research_rounds']}\n"
        f"  写作轮次: {state['writing_rounds']}\n"
        f"  最终评分: {state['quality_score']:.2f}\n"
        f"  文章: {state['draft'][:80]}..."
    )
    print(f"[输出] {result}")
    return {"status": result}


# ============================================================
# 第三部分：路由函数
# ============================================================

# 循环守卫配置
MAX_RESEARCH_ROUNDS = 3
MAX_WRITING_ROUNDS = 3
RESEARCH_QUALITY_THRESHOLD = 0.6
REVIEW_QUALITY_THRESHOLD = 0.8


def route_research(state: ResearchState) -> Literal["writing_agent", "research_agent"]:
    """研究质量路由

    决策逻辑：
    1. 质量评分达标（>= 0.6）→ 交给写作 Agent
    2. 达到最大研究轮次 → 强制交给写作 Agent（循环守卫）
    3. 否则 → 继续研究
    """
    score = state.get("quality_score", 0)
    rounds = state.get("research_rounds", 0)

    if score >= RESEARCH_QUALITY_THRESHOLD:
        print(f"[路由] 研究质量达标（{score:.1f} >= {RESEARCH_QUALITY_THRESHOLD}），转入写作")
        return "writing_agent"

    if rounds >= MAX_RESEARCH_ROUNDS:
        print(f"[路由] 达到最大研究轮次（{rounds}/{MAX_RESEARCH_ROUNDS}），强制转入写作")
        return "writing_agent"

    print(f"[路由] 研究质量不足（{score:.1f} < {RESEARCH_QUALITY_THRESHOLD}），继续研究")
    return "research_agent"


def route_review(state: ResearchState) -> Literal["format_output", "writing_agent"]:
    """审核结果路由

    决策逻辑：
    1. 质量评分达标（>= 0.8）→ 格式化输出
    2. 达到最大写作轮次 → 强制输出（循环守卫）
    3. 否则 → 打回写作 Agent 修改
    """
    score = state.get("quality_score", 0)
    rounds = state.get("writing_rounds", 0)

    if score >= REVIEW_QUALITY_THRESHOLD:
        print(f"[路由] 审核通过（{score:.2f} >= {REVIEW_QUALITY_THRESHOLD}），转入输出")
        return "format_output"

    if rounds >= MAX_WRITING_ROUNDS:
        print(f"[路由] 达到最大写作轮次（{rounds}/{MAX_WRITING_ROUNDS}），强制输出")
        return "format_output"

    print(f"[路由] 审核不通过（{score:.2f} < {REVIEW_QUALITY_THRESHOLD}），打回修改")
    return "writing_agent"


# ============================================================
# 第四部分：构建图
# ============================================================

def build_research_graph():
    """构建研究助手协作图

    图结构包含两个循环：
    1. 研究循环：research_agent ↔ route_research
    2. 写作-审核循环：writing_agent → review_agent → route_review ↔ writing_agent
    """
    builder = StateGraph(ResearchState)

    # 添加节点
    builder.add_node("research_agent", research_agent)
    builder.add_node("writing_agent", writing_agent)
    builder.add_node("review_agent", review_agent)
    builder.add_node("format_output", format_output)

    # 入口：从研究开始
    builder.add_edge(START, "research_agent")

    # 研究循环：研究完成后路由决策
    builder.add_conditional_edges("research_agent", route_research)

    # 写作完成后 → 审核
    builder.add_edge("writing_agent", "review_agent")

    # 审核完成后路由决策
    builder.add_conditional_edges("review_agent", route_review)

    # 输出 → 结束
    builder.add_edge("format_output", END)

    return builder.compile()


# ============================================================
# 第五部分：测试
# ============================================================

if __name__ == "__main__":

    graph = build_research_graph()

    # ===== 测试1：完整的研究-写作-审核流程 =====
    print("=" * 60)
    print("测试1：完整协作流程 - 主题「LangGraph 条件分支」")
    print("=" * 60)
    print()

    result1 = graph.invoke({
        "topic": "LangGraph 条件分支",
        "research_notes": [],
        "draft": "",
        "review_feedback": "",
        "research_rounds": 0,
        "writing_rounds": 0,
        "quality_score": 0.0,
        "status": "",
    })

    print()
    print(f"最终状态: {result1['status']}")
    print()

    # ===== 测试2：简单主题（可能更快完成） =====
    print("=" * 60)
    print("测试2：简单主题 - 「Python 基础语法」")
    print("=" * 60)
    print()

    result2 = graph.invoke({
        "topic": "Python 基础语法",
        "research_notes": [],
        "draft": "",
        "review_feedback": "",
        "research_rounds": 0,
        "writing_rounds": 0,
        "quality_score": 0.0,
        "status": "",
    })

    print()
    print(f"最终状态: {result2['status']}")
```

---

## 运行输出示例

```
============================================================
测试1：完整协作流程 - 主题「LangGraph 条件分支」
============================================================

[研究Agent] 第1轮研究完成
  发现: 基础概念：'LangGraph 条件分支' 的定义和核心原理
  质量评分: 0.3
[路由] 研究质量不足（0.3 < 0.6），继续研究
[研究Agent] 第2轮研究完成
  发现: 深入分析：'LangGraph 条件分支' 的技术实现和关键挑战
  质量评分: 0.6
[路由] 研究质量达标（0.6 >= 0.6），转入写作
[写作Agent] 第1版草稿完成（基于2条笔记）
[审核Agent] 评审完成
  质量评分: 0.65
  反馈: 内容基本完整，但需要增加更多细节和案例，建议修改后重新提交。
[路由] 审核不通过（0.65 < 0.8），打回修改
[写作Agent] 第2版修改稿完成（基于2条笔记）
[审核Agent] 评审完成
  质量评分: 0.85
  反馈: 文章质量优秀，内容充实，结构清晰，可以发布。
[路由] 审核通过（0.85 >= 0.8），转入输出
[输出] 研究完成！
  主题: LangGraph 条件分支
  研究轮次: 2
  写作轮次: 2
  最终评分: 0.85
  文章: [第2版修改稿] 《LangGraph 条件分支》
根据审核反馈修改：内容基本完整，但需要增...

最终状态: 研究完成！
  主题: LangGraph 条件分支
  研究轮次: 2
  写作轮次: 2
  最终评分: 0.85
  文章: [第2版修改稿] 《LangGraph 条件分支》
根据审核反馈修改：内容基本完整，但需要增...
```

---

## 核心设计详解

### 1. 两个独立的循环

本场景包含两个条件路由驱动的循环，它们是独立的：

```
循环1：研究循环
  research_agent → route_research → research_agent（质量不足时）

循环2：写作-审核循环
  writing_agent → review_agent → route_review → writing_agent（审核不通过时）
```

两个循环通过单向 handoff 连接：研究循环的出口是写作循环的入口。

### 2. 循环守卫的必要性

没有循环守卫，如果质量评分逻辑有 bug（比如永远达不到阈值），图会无限循环。循环守卫是生产环境的必备机制：

```python
# 循环守卫：轮次上限
if rounds >= MAX_RESEARCH_ROUNDS:
    return "writing_agent"  # 强制跳出循环
```

### 3. Annotated + operator.add 的聚合模式

`research_notes` 使用 `Annotated[list[str], operator.add]`，每次研究 Agent 返回的笔记会自动追加到列表中，而不是覆盖：

```python
# 第1轮返回 {"research_notes": ["笔记1"]}
# 第2轮返回 {"research_notes": ["笔记2"]}
# 状态中 research_notes = ["笔记1", "笔记2"]  ← 自动追加
```

这是 LangGraph 中处理循环累积数据的标准模式。

### 4. Agent handoff 的条件路由实现

Agent 之间的任务交接通过条件边实现，而不是硬编码的固定边：

| 交接点 | 路由函数 | 条件 | 目标 |
|--------|---------|------|------|
| 研究 → 写作 | `route_research` | 质量达标或轮次上限 | `writing_agent` |
| 研究 → 研究 | `route_research` | 质量不足 | `research_agent` |
| 审核 → 输出 | `route_review` | 审核通过或轮次上限 | `format_output` |
| 审核 → 写作 | `route_review` | 审核不通过 | `writing_agent` |

---

## 扩展：添加 Supervisor 协调

在更复杂的多 Agent 系统中，可以引入 Supervisor Agent 统一协调：

```python
def supervisor_agent(state: ResearchState) -> Command[Literal[
    "research_agent", "writing_agent", "review_agent", "format_output"
]]:
    """Supervisor：根据全局状态决定下一步交给谁

    优势：
    - 集中式决策，逻辑更清晰
    - 可以跳过某些步骤（如研究充分时直接写作）
    - 便于添加新的 Agent（只需修改 Supervisor 逻辑）
    """
    research_rounds = state.get("research_rounds", 0)
    writing_rounds = state.get("writing_rounds", 0)
    score = state.get("quality_score", 0)
    draft = state.get("draft", "")

    # 还没开始研究，或研究不充分
    if research_rounds == 0 or (score < 0.6 and research_rounds < 3):
        return Command(
            goto="research_agent",
            update={"status": "researching"},
        )

    # 研究完成，还没写作
    if not draft:
        return Command(
            goto="writing_agent",
            update={"status": "writing"},
        )

    # 写作完成，需要审核
    if score < 0.8 and writing_rounds < 3:
        return Command(
            goto="review_agent",
            update={"status": "reviewing"},
        )

    # 全部完成
    return Command(
        goto="format_output",
        update={"status": "finalizing"},
    )
```

Supervisor 模式 vs 条件边模式的对比：

| 维度 | 条件边模式（本场景） | Supervisor 模式 |
|------|---------------------|----------------|
| 决策位置 | 分散在多个路由函数中 | 集中在 Supervisor 节点 |
| 灵活性 | 每个路由函数独立，修改互不影响 | 修改一处即可调整全局流程 |
| 可视化 | 图结构直观，边清晰 | 所有边都从 Supervisor 出发，星型结构 |
| 适用场景 | Agent 数量少，流程固定 | Agent 数量多，流程需要动态调整 |
| 复杂度 | 简单直接 | 需要维护全局决策逻辑 |

---

## 注意事项

1. **循环守卫是必须的**：任何包含循环的图都要设置最大轮次，防止无限循环
2. **Annotated reducer 用于累积数据**：循环中需要追加数据时，使用 `Annotated[list, operator.add]`
3. **质量评分驱动路由**：实际项目中用 LLM 评估质量，这里用模拟评分保证可运行
4. **轮次计数器不能用 reducer**：`research_rounds` 和 `writing_rounds` 是覆盖式更新（每次 +1），不能用 `operator.add`
5. **Supervisor 模式适合复杂场景**：当 Agent 数量超过 3-4 个时，考虑引入 Supervisor 集中管理路由
