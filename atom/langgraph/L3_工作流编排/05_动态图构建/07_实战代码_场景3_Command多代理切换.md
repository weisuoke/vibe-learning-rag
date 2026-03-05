# 实战场景3：Command 多代理切换 - 跨图导航

> 构建一个「多代理协作系统」：主图中的 router 根据任务类型分配给不同的子图代理（research / writing / review），子图完成后通过 Command(graph=Command.PARENT) 返回主图，代理之间可以互相切换。

---

## 场景描述

想象你在构建一个 AI 写作助手。用户提交一个任务（比如"写一篇关于 RAG 的技术博客"），系统需要多个"专家代理"协作完成：

1. **Router（路由器）**：分析任务类型，决定交给哪个代理
2. **Research Agent（调研代理）**：负责收集资料、整理要点
3. **Writing Agent（写作代理）**：负责根据资料撰写内容
4. **Review Agent（审核代理）**：负责审核质量、给出修改建议

关键挑战：
- 每个代理是一个**独立的子图**，有自己的内部逻辑
- 代理完成后需要**返回主图**继续流程
- 代理之间可以**互相切换**（调研完 → 写作 → 审核）

### 双重类比

**前端类比：** 就像微前端架构——主应用（主图）负责路由和状态管理，每个子应用（子图）独立运行。子应用完成后通过 `postMessage`（Command.PARENT）通知主应用切换到下一个子应用。共享状态就像 `window.__SHARED_STATE__`。

**日常生活类比：** 就像一个出版社的工作流——总编辑（Router）收到稿件需求，先交给记者（Research）采访调研，记者写完采访笔记后交回总编辑，总编辑再转给作家（Writing）撰稿，最后交给校对（Review）审核。每个人在自己的工位独立工作，但通过总编辑协调。

### 工作流图示

```
主图:
                    ┌──→ [Research子图] ──┐
                    │                      │
[START] → [Router] ┼──→ [Writing子图]  ──┼──→ [Review] → [END]
                    │                      │
                    └──────────────────────┘
                    （代理之间可通过 Command.PARENT 互相切换）

Research子图:                          Writing子图:
[collect] → [analyze] → 返回主图       [draft] → [polish] → 返回主图
```

---

## 完整可运行代码

```python
"""
实战场景3：Command 多代理切换 - 多代理协作系统
演示：Command API + Command.PARENT 实现子图与父图之间的导航

运行要求：pip install langgraph
无需外部 API，完全本地运行
"""
import operator
import time
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义状态 =====

class MainState(TypedDict):
    """主图状态 - 所有代理共享的状态

    Attributes:
        task: 用户的原始任务描述
        task_type: 路由器判断的任务类型
        research_notes: 调研代理的输出（共享键，需要 Reducer）
        draft: 写作代理的输出
        review_feedback: 审核代理的反馈
        workflow_log: 工作流日志（Reducer: operator.add 追加）
        final_output: 最终输出
    """
    task: str
    task_type: str
    research_notes: Annotated[list[str], operator.add]
    draft: str
    review_feedback: str
    workflow_log: Annotated[list[str], operator.add]
    final_output: str


# ===== 2. Router 节点 =====

def router(state: MainState) -> Command[Literal["research_agent", "writing_agent", "review"]]:
    """路由器 - 根据任务类型和当前状态决定下一步

    使用 Command 同时完成两件事：
    1. 更新状态（update）
    2. 决定下一个节点（goto）

    Command 的类型注解 Command[Literal["research_agent", "writing_agent", "review"]]
    告诉 LangGraph 这个节点可能路由到哪些节点（用于图可视化）。
    """
    task = state["task"]
    research_notes = state.get("research_notes", [])
    draft = state.get("draft", "")

    print(f"\n[Router] 分析任务: {task}")

    # 决策逻辑：根据当前进度决定下一步
    if not research_notes:
        # 还没有调研资料 → 先调研
        print(f"[Router] 决策: 需要先调研 → 分配给 Research Agent")
        return Command(
            update={
                "task_type": "research",
                "workflow_log": ["[Router] 任务开始，分配给 Research Agent"],
            },
            goto="research_agent",
        )
    elif not draft:
        # 有调研资料但没有草稿 → 去写作
        print(f"[Router] 决策: 调研完成，需要写作 → 分配给 Writing Agent")
        return Command(
            update={
                "task_type": "writing",
                "workflow_log": ["[Router] 调研完成，分配给 Writing Agent"],
            },
            goto="writing_agent",
        )
    else:
        # 有草稿了 → 去审核
        print(f"[Router] 决策: 草稿完成，需要审核 → 分配给 Review Agent")
        return Command(
            update={
                "task_type": "review",
                "workflow_log": ["[Router] 草稿完成，分配给 Review Agent"],
            },
            goto="review",
        )


# ===== 3. Research Agent 子图 =====

class ResearchState(TypedDict):
    """调研代理的内部状态"""
    task: str
    research_notes: Annotated[list[str], operator.add]
    workflow_log: Annotated[list[str], operator.add]


def collect_info(state: ResearchState) -> dict:
    """收集资料节点"""
    task = state["task"]
    print(f"  [Research/收集] 正在为 '{task}' 收集资料...")
    time.sleep(0.1)

    notes = [
        f"资料1: {task} 的核心概念和定义",
        f"资料2: {task} 的主要应用场景",
        f"资料3: {task} 的最新发展趋势",
    ]
    print(f"  [Research/收集] 找到 {len(notes)} 条资料")

    return {
        "research_notes": notes,
        "workflow_log": [f"[Research] 收集了 {len(notes)} 条资料"],
    }


def analyze_info(state: ResearchState) -> Command[Literal["router"]]:
    """分析资料节点 - 完成后返回主图

    关键：使用 Command(graph=Command.PARENT) 返回父图！
    - graph=Command.PARENT：导航到最近的父图（主图）
    - goto="router"：在父图中跳转到 router 节点
    - update：更新的状态会合并到父图的 MainState 中
    """
    notes = state["research_notes"]
    print(f"  [Research/分析] 分析 {len(notes)} 条资料，提炼要点...")
    time.sleep(0.1)

    summary_note = f"调研总结: 基于 {len(notes)} 条资料，提炼了核心要点和关键数据"
    print(f"  [Research/分析] 分析完成，返回主图")

    # 关键代码：通过 Command.PARENT 返回父图
    return Command(
        update={
            "research_notes": [summary_note],
            "workflow_log": ["[Research] 分析完成，返回主图"],
        },
        goto="router",              # 在父图中跳转到 router
        graph=Command.PARENT,       # 导航到父图！
    )


def build_research_agent():
    """构建调研代理子图"""
    graph = StateGraph(ResearchState)

    graph.add_node("collect", collect_info)
    graph.add_node("analyze", analyze_info)

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "analyze")
    # 注意：analyze 节点通过 Command.PARENT 返回父图
    # 所以不需要 add_edge("analyze", END)

    return graph.compile()


# ===== 4. Writing Agent 子图 =====

class WritingState(TypedDict):
    """写作代理的内部状态"""
    task: str
    research_notes: Annotated[list[str], operator.add]
    draft: str
    workflow_log: Annotated[list[str], operator.add]


def write_draft(state: WritingState) -> dict:
    """撰写草稿节点"""
    task = state["task"]
    notes = state.get("research_notes", [])
    print(f"  [Writing/草稿] 基于 {len(notes)} 条资料撰写 '{task}' 草稿...")
    time.sleep(0.1)

    draft = (
        f"# {task}\n\n"
        f"## 引言\n"
        f"本文将深入探讨{task}的核心概念和实践应用。\n\n"
        f"## 核心内容\n"
        f"基于调研资料，{task}具有以下关键特点...\n\n"
        f"## 总结\n"
        f"{task}是一个值得深入学习的领域。"
    )
    print(f"  [Writing/草稿] 草稿完成，{len(draft)} 字")

    return {
        "draft": draft,
        "workflow_log": [f"[Writing] 草稿完成，{len(draft)} 字"],
    }


def polish_draft(state: WritingState) -> Command[Literal["router"]]:
    """润色草稿节点 - 完成后返回主图"""
    draft = state["draft"]
    print(f"  [Writing/润色] 润色草稿中...")
    time.sleep(0.1)

    polished = draft + "\n\n---\n*本文经过 AI 润色优化*"
    print(f"  [Writing/润色] 润色完成，返回主图")

    return Command(
        update={
            "draft": polished,
            "workflow_log": ["[Writing] 润色完成，返回主图"],
        },
        goto="router",
        graph=Command.PARENT,
    )


def build_writing_agent():
    """构建写作代理子图"""
    graph = StateGraph(WritingState)

    graph.add_node("write_draft", write_draft)
    graph.add_node("polish", polish_draft)

    graph.add_edge(START, "write_draft")
    graph.add_edge("write_draft", "polish")

    return graph.compile()


# ===== 5. Review Agent（主图节点，非子图）=====

def review(state: MainState) -> dict:
    """审核节点 - 审核草稿质量

    这个节点直接在主图中，不是子图。
    简单的节点不需要做成子图。
    """
    draft = state["draft"]
    print(f"\n[Review] 审核草稿（{len(draft)} 字）...")
    time.sleep(0.1)

    # 模拟审核逻辑
    feedback = "审核通过。内容结构清晰，论述充分，建议补充更多实际案例。"
    final_output = (
        f"{draft}\n\n"
        f"---\n"
        f"审核意见: {feedback}"
    )

    print(f"[Review] 审核完成: {feedback}")

    return {
        "review_feedback": feedback,
        "final_output": final_output,
        "workflow_log": [f"[Review] 审核完成: {feedback}"],
    }


# ===== 6. 构建主图 =====

def build_multi_agent_system():
    """构建多代理协作系统

    关键步骤：
    1. 构建子图（research_agent, writing_agent）
    2. 创建主图
    3. 将子图作为节点添加到主图
    4. 用 Command 实现动态路由
    """
    # 构建子图
    research_agent = build_research_agent()
    writing_agent = build_writing_agent()

    # 构建主图
    main_graph = StateGraph(MainState)

    # --- 添加节点 ---
    # router 是普通函数节点
    main_graph.add_node("router", router)
    # research_agent 和 writing_agent 是编译后的子图
    main_graph.add_node("research_agent", research_agent)
    main_graph.add_node("writing_agent", writing_agent)
    # review 是普通函数节点
    main_graph.add_node("review", review)

    # --- 边连接 ---
    main_graph.add_edge(START, "router")
    # router 通过 Command(goto=...) 动态路由，不需要显式添加边
    # 子图通过 Command(graph=Command.PARENT, goto="router") 返回
    main_graph.add_edge("review", END)

    return main_graph.compile()


# ===== 7. 运行测试 =====

if __name__ == "__main__":
    app = build_multi_agent_system()

    # 测试：完整的多代理协作流程
    print("=" * 60)
    print("多代理协作系统 - 完整流程测试")
    print("=" * 60)

    result = app.invoke({
        "task": "RAG 检索增强生成技术",
        "task_type": "",
        "research_notes": [],
        "draft": "",
        "review_feedback": "",
        "workflow_log": [],
        "final_output": "",
    })

    print("\n" + "=" * 60)
    print("最终输出:")
    print("=" * 60)
    print(result["final_output"])

    print("\n" + "=" * 60)
    print("完整工作流日志:")
    print("=" * 60)
    for i, log in enumerate(result["workflow_log"], 1):
        print(f"  {i}. {log}")
```

---

## 运行输出示例

```
============================================================
多代理协作系统 - 完整流程测试
============================================================

[Router] 分析任务: RAG 检索增强生成技术
[Router] 决策: 需要先调研 → 分配给 Research Agent
  [Research/收集] 正在为 'RAG 检索增强生成技术' 收集资料...
  [Research/收集] 找到 3 条资料
  [Research/分析] 分析 3 条资料，提炼要点...
  [Research/分析] 分析完成，返回主图

[Router] 分析任务: RAG 检索增强生成技术
[Router] 决策: 调研完成，需要写作 → 分配给 Writing Agent
  [Writing/草稿] 基于 4 条资料撰写 'RAG 检索增强生成技术' 草稿...
  [Writing/草稿] 草稿完成，153 字
  [Writing/润色] 润色草稿中...
  [Writing/润色] 润色完成，返回主图

[Router] 分析任务: RAG 检索增强生成技术
[Router] 决策: 草稿完成，需要审核 → 分配给 Review Agent

[Review] 审核草稿（175 字）...
[Review] 审核完成: 审核通过。内容结构清晰，论述充分，建议补充更多实际案例。

============================================================
最终输出:
============================================================
# RAG 检索增强生成技术

## 引言
本文将深入探讨RAG 检索增强生成技术的核心概念和实践应用。

## 核心内容
基于调研资料，RAG 检索增强生成技术具有以下关键特点...

## 总结
RAG 检索增强生成技术是一个值得深入学习的领域。

---
*本文经过 AI 润色优化*

---
审核意见: 审核通过。内容结构清晰，论述充分，建议补充更多实际案例。

============================================================
完整工作流日志:
============================================================
  1. [Router] 任务开始，分配给 Research Agent
  2. [Research] 收集了 3 条资料
  3. [Research] 分析完成，返回主图
  4. [Router] 调研完成，分配给 Writing Agent
  5. [Writing] 草稿完成，153 字
  6. [Writing] 润色完成，返回主图
  7. [Router] 草稿完成，分配给 Review Agent
  8. [Review] 审核完成: 审核通过。内容结构清晰，论述充分，建议补充更多实际案例。
```

---

## 代码逐行解析

### 1. Command 的基本用法

```python
def router(state: MainState) -> Command[Literal["research_agent", "writing_agent", "review"]]:
    return Command(
        update={"task_type": "research", "workflow_log": ["..."]},
        goto="research_agent",
    )
```

Command 同时做两件事：
- `update`：更新状态（等价于普通节点的 `return {"key": "value"}`）
- `goto`：指定下一个节点（等价于条件边的路由函数返回值）

类型注解 `Command[Literal["research_agent", "writing_agent", "review"]]` 不影响运行时行为，但告诉 LangGraph 这个节点可能路由到哪些节点，用于生成图的可视化。

**前端类比：** Command 就像 Redux 的 `dispatch` 同时触发了 `setState` 和 `navigate`：

```javascript
// 传统方式：分开做
dispatch({ type: 'SET_TASK_TYPE', payload: 'research' });  // 更新状态
navigate('/research');                                        // 路由跳转

// Command 方式：一步到位
return Command({ update: {...}, goto: "research_agent" });
```

### 2. Command.PARENT 的工作机制

```python
def analyze_info(state: ResearchState) -> Command[Literal["router"]]:
    return Command(
        update={"research_notes": [...], "workflow_log": [...]},
        goto="router",
        graph=Command.PARENT,    # 关键！
    )
```

`graph=Command.PARENT` 告诉 LangGraph："不要在当前子图中找 router 节点，而是去父图中找"。

执行流程：
1. 子图的 analyze 节点执行完毕
2. LangGraph 看到 `graph=Command.PARENT`
3. 将 `update` 中的状态合并到**父图的 MainState** 中
4. 在**父图**中跳转到 `goto` 指定的 router 节点
5. 子图执行结束

> [来源: LangGraph 源码 types.py - Command 类定义，graph 参数说明（行 367-418）]

### 3. 子图与父图的状态传递

这是最容易出错的地方。子图和父图的状态是**独立的**，但通过**共享键**传递数据：

```
MainState（父图）:
  task: str                                    ← 子图可以读取
  research_notes: Annotated[list, operator.add] ← 共享键，需要 Reducer
  workflow_log: Annotated[list, operator.add]   ← 共享键，需要 Reducer
  draft: str
  ...

ResearchState（子图）:
  task: str                                    ← 从父图继承
  research_notes: Annotated[list, operator.add] ← 共享键，Reducer 必须一致
  workflow_log: Annotated[list, operator.add]   ← 共享键，Reducer 必须一致
```

当子图通过 `Command(graph=Command.PARENT, update={...})` 返回时：
- `update` 中的键如果在父图 State 中存在，就会按父图的 Reducer 规则合并
- `update` 中的键如果在父图 State 中不存在，会被忽略

### 4. 为什么共享键需要 Reducer

```python
# 父图
research_notes: Annotated[list[str], operator.add]

# 子图
research_notes: Annotated[list[str], operator.add]
```

如果共享键没有 Reducer（比如只是 `research_notes: list[str]`），子图返回的值会**覆盖**父图中已有的值。使用 `operator.add` Reducer 后，子图返回的列表会**追加**到父图已有的列表中。

**重要规则：** 父图和子图中同名共享键的 Reducer 必须一致。如果父图用 `operator.add`，子图也必须用 `operator.add`，否则行为不可预测。

> [来源: Context7 LangGraph 文档 - 共享状态键必须在父图中定义 reducer]

### 5. 子图作为节点添加

```python
research_agent = build_research_agent()  # 编译后的子图
main_graph.add_node("research_agent", research_agent)
```

编译后的子图可以直接作为节点添加到父图中。LangGraph 会自动处理：
- 父图状态 → 子图状态的映射（同名键自动传递）
- 子图状态 → 父图状态的回传（通过 Command.PARENT）

---

## 注意事项与常见陷阱

### 陷阱1：Command.goto 不会阻止静态边

```python
# 错误示范：同时有静态边和 Command.goto
graph.add_edge("router", "some_node")  # 静态边

def router(state):
    return Command(goto="research_agent")  # Command.goto
```

如果 router 节点既有静态边又返回 Command.goto，**两者都会执行**！这通常不是你想要的。解决方案：router 节点不要添加任何静态出边，完全依赖 Command.goto 路由。

### 陷阱2：子图的 END 与 Command.PARENT 冲突

```python
# 错误示范：子图节点既有到 END 的边，又返回 Command.PARENT
graph.add_edge("analyze", END)  # 静态边到 END

def analyze(state):
    return Command(goto="router", graph=Command.PARENT)  # 返回父图
```

如果子图节点通过 Command.PARENT 返回父图，就不需要（也不应该）添加到 END 的边。否则子图会同时尝试结束和返回父图，导致不可预测的行为。

### 陷阱3：共享键的 Reducer 不一致

```python
# 父图
class MainState(TypedDict):
    notes: Annotated[list[str], operator.add]  # 用 operator.add

# 子图
class SubState(TypedDict):
    notes: list[str]  # 没有 Reducer！
```

这会导致子图返回的 notes 覆盖父图中已有的值，而不是追加。确保共享键在父图和子图中使用相同的 Reducer。

### 陷阱4：忘记在子图 State 中声明共享键

```python
# 子图 State 中没有 workflow_log
class ResearchState(TypedDict):
    task: str
    research_notes: Annotated[list[str], operator.add]
    # 缺少 workflow_log！

def analyze(state):
    return Command(
        update={"workflow_log": ["..."]},  # 这个更新会被忽略
        graph=Command.PARENT,
    )
```

如果子图 State 中没有声明某个键，但 Command.update 中包含了它，行为取决于 LangGraph 版本。最安全的做法是：**子图 State 中声明所有需要回传给父图的键**。

---

## 关键收获

| 概念 | 本场景中的体现 |
|------|---------------|
| `Command(goto=...)` | router 节点动态路由到不同代理 |
| `Command(graph=Command.PARENT)` | 子图节点返回父图继续执行 |
| `Command(update=...)` | 在路由的同时更新状态 |
| 子图作为节点 | research_agent、writing_agent 是编译后的子图 |
| 共享状态键 | task、research_notes、workflow_log 在父子图间共享 |
| Reducer 一致性 | 父子图的共享键必须使用相同的 Reducer |
| 类型注解 | `Command[Literal["node1", "node2"]]` 辅助图可视化 |

---

## 参考来源

- [LangGraph 源码: types.py - Command 类定义（行 367-418）](sourcecode/langgraph/types.py)
- [LangGraph 源码: graph/state.py - add_node 支持子图作为节点](sourcecode/langgraph/graph/state.py)
- [Context7 LangGraph 文档 - Command.PARENT 父图导航](https://docs.langchain.com/oss/python/langgraph/graph-api)
