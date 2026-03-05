# 实战代码 - 场景4：多Agent状态协调

## 场景说明

本场景演示多Agent协作系统中的状态共享与隔离模式。你将看到：

1. 主图（协调器）持有共享状态，子图（各Agent）持有私有状态
2. 子图通过"同名键"自动与主图交换数据，私有键对主图完全不可见
3. 三个Agent各自独立工作，互不干扰，最终结果汇聚到主图

核心思路：把主图当成"项目经理"——它只关心任务分配和最终交付物，不关心每个Agent内部怎么干活。子图的私有字段就是Agent的"草稿纸"，用完即弃。

[来源: 03_核心概念_5_状态组织与模块化.md | 子图状态隔离]

---

## 设计思路

```
主图 CoordinatorState
├── task           ← 输入：任务描述
├── research_data  ← 研究Agent写入，分析Agent读取
├── analysis_data  ← 分析Agent写入，写作Agent读取
├── final_report   ← 写作Agent写入，最终输出
│
├── 研究子图 ResearchState
│   ├── task, research_data        ← 共享键（与主图通信）
│   └── sources, raw_notes         ← 私有键（主图看不到）
│
├── 分析子图 AnalysisState
│   ├── research_data, analysis_data  ← 共享键
│   └── metrics, scratch            ← 私有键
│
└── 写作子图 WritingState
    ├── task, research_data, analysis_data, final_report  ← 共享键
    └── outline, draft              ← 私有键
```

共享键靠"同名"自动映射——子图字段名和主图一致的，LangGraph 自动传入传出。

---

## 完整代码

```python
"""
多Agent状态协调实战
演示：子图隔离、同名键自动映射、共享 vs 私有状态

场景：研究Agent → 分析Agent → 写作Agent 协作完成一份报告
每个Agent有自己的私有工作状态，通过共享键传递成果

运行环境：Python 3.13+, langgraph
安装依赖：uv add langgraph
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 主图状态 —— 只包含Agent间需要传递的共享数据
# ============================================================

class CoordinatorState(TypedDict):
    """协调器状态：Agent间的"公告板"

    设计原则：只放需要跨Agent传递的数据，
    各Agent的中间草稿、临时变量不要放这里。
    """
    task: str               # 输入：任务描述
    research_data: str      # 研究 → 分析：研究成果
    analysis_data: str      # 分析 → 写作：分析结论
    final_report: str       # 写作 → 输出：最终报告


# ============================================================
# 2. 研究Agent子图 —— 有自己的私有状态
# ============================================================

class ResearchState(TypedDict):
    """研究Agent的完整状态"""
    task: str               # 共享键：从主图接收任务
    research_data: str      # 共享键：研究成果返回主图
    sources: list[str]      # 私有键：信息来源（主图看不到）
    raw_notes: str          # 私有键：原始笔记（主图看不到）


def search_sources(state: ResearchState) -> dict:
    """步骤1：检索信息来源"""
    task = state["task"]
    sources = [
        f"[论文] {task}的最新研究进展 (2024)",
        f"[报告] {task}市场分析报告",
        f"[访谈] 行业专家谈{task}趋势",
    ]
    print(f"  [研究Agent] 检索到 {len(sources)} 个来源")
    return {"sources": sources}


def compile_findings(state: ResearchState) -> dict:
    """步骤2：整理研究发现，写入共享键"""
    sources = state["sources"]
    raw_notes = " | ".join(sources)
    # research_data 是共享键，会自动传回主图
    research_data = f"基于 {len(sources)} 个来源的研究发现：{raw_notes}"
    print(f"  [研究Agent] 整理完成，写入 research_data")
    return {"raw_notes": raw_notes, "research_data": research_data}


def build_research_subgraph() -> StateGraph:
    builder = StateGraph(ResearchState)
    builder.add_node("search", search_sources)
    builder.add_node("compile", compile_findings)
    builder.add_edge(START, "search")
    builder.add_edge("search", "compile")
    builder.add_edge("compile", END)
    return builder.compile()


# ============================================================
# 3. 分析Agent子图 —— 有自己的私有状态
# ============================================================

class AnalysisState(TypedDict):
    """分析Agent的完整状态"""
    research_data: str      # 共享键：从主图接收研究成果
    analysis_data: str      # 共享键：分析结论返回主图
    metrics: dict           # 私有键：分析指标（主图看不到）
    scratch: str            # 私有键：草稿区（主图看不到）


def extract_metrics(state: AnalysisState) -> dict:
    """步骤1：从研究数据中提取关键指标"""
    data = state["research_data"]
    metrics = {
        "source_count": data.count("来源"),
        "confidence": 0.85,
        "key_themes": ["技术突破", "市场增长", "人才需求"],
    }
    print(f"  [分析Agent] 提取指标: {len(metrics['key_themes'])} 个主题")
    return {"metrics": metrics, "scratch": f"原始数据长度: {len(data)}"}


def generate_insights(state: AnalysisState) -> dict:
    """步骤2：生成分析洞察，写入共享键"""
    themes = state["metrics"]["key_themes"]
    confidence = state["metrics"]["confidence"]
    # analysis_data 是共享键，会自动传回主图
    analysis_data = (
        f"分析结论（置信度 {confidence}）：核心趋势为 "
        + "、".join(themes)
    )
    print(f"  [分析Agent] 生成洞察，写入 analysis_data")
    return {"analysis_data": analysis_data}


def build_analysis_subgraph() -> StateGraph:
    builder = StateGraph(AnalysisState)
    builder.add_node("extract", extract_metrics)
    builder.add_node("insights", generate_insights)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "insights")
    builder.add_edge("insights", END)
    return builder.compile()


# ============================================================
# 4. 写作Agent —— 普通节点（不需要子图也行）
# ============================================================

def writing_agent(state: CoordinatorState) -> dict:
    """写作Agent：直接在主图中运行

    为什么不用子图？因为写作Agent逻辑简单，
    不需要私有状态，直接读共享数据、写最终报告。
    这也是一种设计选择——不是所有Agent都需要子图。
    """
    task = state["task"]
    research = state["research_data"]
    analysis = state["analysis_data"]

    final_report = (
        f"=== {task} 研究报告 ===\n"
        f"研究摘要: {research[:50]}...\n"
        f"分析结论: {analysis}\n"
        f"建议: 持续关注该领域发展"
    )
    print(f"  [写作Agent] 生成最终报告")
    return {"final_report": final_report}


# ============================================================
# 5. 主图 —— 协调三个Agent的执行顺序
# ============================================================

research_subgraph = build_research_subgraph()
analysis_subgraph = build_analysis_subgraph()

coordinator = StateGraph(CoordinatorState)

# 子图作为节点嵌入主图
# LangGraph 自动通过同名键传递数据：
#   主图.task → 研究子图.task（自动传入）
#   研究子图.research_data → 主图.research_data（自动传出）
coordinator.add_node("research_agent", research_subgraph)
coordinator.add_node("analysis_agent", analysis_subgraph)
coordinator.add_node("writing_agent", writing_agent)

# 执行顺序：研究 → 分析 → 写作
coordinator.add_edge(START, "research_agent")
coordinator.add_edge("research_agent", "analysis_agent")
coordinator.add_edge("analysis_agent", "writing_agent")
coordinator.add_edge("writing_agent", END)

app = coordinator.compile()


# ============================================================
# 6. 运行演示
# ============================================================

print("=" * 55)
print("  多Agent协作系统")
print("=" * 55)

task = "AI Agent 发展趋势"
print(f"\n任务: {task}\n")

print("--- 阶段1: 研究Agent ---")
# 注意：只需要传共享字段的初始值
result = app.invoke({
    "task": task,
    "research_data": "",
    "analysis_data": "",
    "final_report": "",
})

print("\n--- 阶段2: 分析Agent ---")
# （已在上面的 invoke 中自动执行）

print("\n--- 阶段3: 写作Agent ---")
# （已在上面的 invoke 中自动执行）

print(f"\n{'=' * 55}")
print("  最终输出")
print("=" * 55)
print(result["final_report"])


# ============================================================
# 7. 验证状态隔离
# ============================================================

print(f"\n{'=' * 55}")
print("  状态隔离验证")
print("=" * 55)

print(f"\n主图返回的键: {list(result.keys())}")
print(f"  包含 task: {'task' in result}")
print(f"  包含 research_data: {'research_data' in result}")
print(f"  包含 analysis_data: {'analysis_data' in result}")
print(f"  包含 final_report: {'final_report' in result}")

# 子图的私有字段不会泄漏到主图
print(f"\n  泄漏 sources: {'sources' in result}")
print(f"  泄漏 raw_notes: {'raw_notes' in result}")
print(f"  泄漏 metrics: {'metrics' in result}")
print(f"  泄漏 scratch: {'scratch' in result}")
print(f"\n  → 子图私有字段完全隔离，主图看不到")
```

---

## 预期输出

```text
=======================================================
  多Agent协作系统
=======================================================

任务: AI Agent 发展趋势

--- 阶段1: 研究Agent ---
  [研究Agent] 检索到 3 个来源
  [研究Agent] 整理完成，写入 research_data

--- 阶段2: 分析Agent ---
  [分析Agent] 提取指标: 3 个主题
  [分析Agent] 生成洞察，写入 analysis_data

--- 阶段3: 写作Agent ---
  [写作Agent] 生成最终报告

=======================================================
  最终输出
=======================================================
=== AI Agent 发展趋势 研究报告 ===
研究摘要: 基于 3 个来源的研究发现：[论文] AI Agent 发展趋势的最新研...
分析结论: 分析结论（置信度 0.85）：核心趋势为 技术突破、市场增长、人才需求
建议: 持续关注该领域发展

=======================================================
  状态隔离验证
=======================================================

主图返回的键: ['task', 'research_data', 'analysis_data', 'final_report']
  包含 task: True
  包含 research_data: True
  包含 analysis_data: True
  包含 final_report: True

  泄漏 sources: False
  泄漏 raw_notes: False
  泄漏 metrics: False
  泄漏 scratch: False

  → 子图私有字段完全隔离，主图看不到
```

---

## 状态协调要点

### 共享 vs 隔离的边界划分

```
问自己一个问题：这个字段需要被其他Agent读取吗？

需要 → 放在主图状态中（共享键）
不需要 → 放在子图状态中（私有键）

示例：
  research_data  → 分析Agent要读 → 共享
  sources        → 只有研究Agent用 → 私有
  metrics        → 只有分析Agent用 → 私有
```

### 同名键映射机制

```python
# 主图状态
class CoordinatorState(TypedDict):
    task: str               # ← 同名键
    research_data: str      # ← 同名键

# 子图状态
class ResearchState(TypedDict):
    task: str               # ← 同名：自动从主图接收
    research_data: str      # ← 同名：自动返回主图
    sources: list[str]      # ← 不同名：私有，不传递
```

LangGraph 的规则很简单：子图中与主图同名的字段自动双向映射，其余字段完全隔离。

### 三种Agent集成方式对比

| 方式 | 适用场景 | 状态隔离 | 复杂度 |
|------|----------|----------|--------|
| 子图（本例研究/分析Agent） | 需要私有工作状态 | 完全隔离 | 中 |
| 普通节点（本例写作Agent） | 逻辑简单，无需私有状态 | 无隔离 | 低 |
| 函数包装 + 手动映射 | 需要字段重命名 | 手动控制 | 高 |

### 避免状态泄漏的检查清单

- [ ] 子图的私有字段名是否与主图字段名不同？（同名会自动映射）
- [ ] 共享键是否最小化？（只传必要数据）
- [ ] 是否需要 Input/Output Schema 进一步限制对外接口？
- [ ] 运行时资源（db连接等）是否用了 Managed Values 而非普通状态？

---

## 学习检查清单

- [ ] 理解子图通过"同名键"自动与主图交换数据
- [ ] 能区分哪些字段该共享、哪些该私有
- [ ] 理解不是所有Agent都需要子图（简单逻辑用普通节点）
- [ ] 能验证子图私有字段不会泄漏到主图
- [ ] 掌握多Agent系统的状态设计流程：先定共享接口，再定私有状态
