# 03_核心概念_4 - Command 自循环模式

> LangGraph 中通过 Command API 实现节点内部自路由与循环控制

---

## 概念定义

**Command 是 LangGraph 中唯一能在同一次调用中同时完成"状态更新"和"路由决策"的 API。** 节点返回 `Command(update={...}, goto="next_node")` 时，既修改了图状态，又指定了下一个要执行的节点，无需预先定义条件边。

这使得 Command 特别适合以下循环场景：
1. 节点自循环：`Command(goto="self")` 让节点反复执行直到满足条件
2. 动态路由：根据 LLM 输出决定下一步走向
3. 多智能体交接：在切换 Agent 的同时传递上下文信息

---

## Command 类的定义与结构

### 源码定义

```python
# [来源: sourcecode/langgraph/types.py L367-L417]
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """一次性完成状态更新 + 路由控制的指令对象"""

    graph: str | None = None       # 目标图（None=当前图，Command.PARENT=父图）
    update: Any | None = None      # 状态更新内容
    resume: dict[str, Any] | Any | None = None  # 恢复中断时的数据
    goto: Send | Sequence[Send | N] | N = ()     # 路由目标
```

### 四个字段详解

#### 1. `goto` - 路由目标（最核心）

**类型**: `Send | Sequence[Send | N] | N`，其中 `N` 通常是字符串

```python
from langgraph.types import Command

# 路由到单个节点
Command(goto="node_b")

# 路由到多个节点（并行触发）
Command(goto=["node_b", "node_c"])

# 路由到 END 结束图
Command(goto="__end__")

# 结合 Send 实现动态并行
Command(goto=[Send("process", {"item": x}) for x in items])
```

#### 2. `update` - 状态更新

```python
# 更新状态字段
Command(update={"score": 0.95, "status": "approved"}, goto="next")

# 不更新状态，只路由
Command(goto="next")
```

#### 3. `graph` - 跨图控制

```python
# 子图向父图发送命令
Command(graph=Command.PARENT, goto="parent_node", update={"result": data})
```

#### 4. `resume` - 中断恢复

```python
# 恢复被中断的执行
Command(resume={"user_input": "approved"})
```

---

## Command 自循环的实现原理

### 底层处理流程

当节点返回 Command 时，LangGraph 通过 `_control_branch()` 函数处理路由逻辑：

```python
# [来源: sourcecode/langgraph/graph/state.py L1492-L1518]
def _control_branch(value: Any) -> Sequence[tuple[str, Any]]:
    commands: list[Command] = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)

    rtn: list[tuple[str, Any]] = []
    for command in commands:
        if command.graph == Command.PARENT:
            raise ParentCommand(command)
        goto_targets = (
            [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
        )
        for go in goto_targets:
            if isinstance(go, Send):
                rtn.append((TASKS, go))           # Send 写入 TASKS channel
            elif isinstance(go, str) and go != END:
                # 写入目标节点的触发 channel
                rtn.append((_CHANNEL_BRANCH_TO.format(go), None))
    return rtn
```

**关键机制**: Command 的 `goto` 最终写入 `branch:to:{node_name}` channel，这与条件边的底层机制完全相同。区别在于 Command 是从节点内部发起的，而条件边是在节点之间定义的。

### 自循环的数据流

```
节点 A 执行
  │
  ├── 返回 Command(update={"count": n+1}, goto="a")
  │
  ▼
_control_branch() 处理
  │
  ├── update 写入状态 channels → count 被更新
  ├── goto 写入 branch:to:a channel → 节点 A 被重新触发
  │
  ▼
PregelLoop.tick()
  │
  ├── prepare_next_tasks() 发现 branch:to:a 有值
  ├── 创建节点 A 的新任务
  │
  ▼
节点 A 再次执行（使用更新后的状态）
```

---

## Command vs 条件边对比

### 核心区别

| 特性 | Command | 条件边 (add_conditional_edges) |
|------|---------|-------------------------------|
| 定义位置 | 节点函数内部 | 图构建时（节点之间） |
| 状态更新 | 可以同时更新 | 不能更新状态 |
| 路由决策 | 节点自己决定 | 外部函数决定 |
| 静态边交互 | 不覆盖静态边 | 替代静态边 |
| 类型注解 | 需要 `Command[Literal[...]]` | 路由函数返回类型 |
| 适用场景 | 更新+路由同时需要 | 纯路由逻辑 |

### 重要警告：Command 不覆盖静态边

这是一个容易踩的坑。来自 Context7 文档的明确说明：

```python
# [来源: Context7 LangGraph 文档]
def node_a(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")

# 如果同时定义了静态边：
builder.add_edge("node_a", "node_b")

# 结果：node_a 执行后，node_b 和 my_other_node 都会被触发！
# Command 不会阻止静态边的执行
```

**最佳实践**: 使用 Command 路由时，不要为同一个源节点定义静态边。

### 类型注解要求

Command 必须通过返回类型注解声明可能的路由目标：

```python
from typing import Literal
from langgraph.types import Command

# 必须声明可能路由到的节点
def my_node(state: State) -> Command[Literal["node_a", "node_b"]]:
    if state["score"] > 0.8:
        return Command(goto="node_a")
    else:
        return Command(goto="node_b")
```

这个注解有两个作用：
1. 让 LangGraph 知道该节点可能路由到哪些节点（用于图的渲染和验证）
2. 提供类型检查支持

---

## Command 与 Send 的组合

Command 的 `goto` 字段可以接收 Send 对象，实现从节点内部发起动态并行：

```python
from langgraph.types import Command, Send

def dispatcher(state: State) -> Command[Literal["worker"]]:
    """从节点内部动态创建并行任务"""
    items = state["items"]

    return Command(
        update={"status": "dispatched"},
        goto=[Send("worker", {"item": item}) for item in items]
    )
```

### 处理机制

在 `_control_branch()` 中，Send 对象被写入 TASKS channel：

```python
# [来源: sourcecode/langgraph/graph/state.py L1510-L1512]
for go in goto_targets:
    if isinstance(go, Send):
        rtn.append((TASKS, go))  # Send → TASKS channel
    elif isinstance(go, str) and go != END:
        rtn.append((_CHANNEL_BRANCH_TO.format(go), None))  # 字符串 → branch channel
```

---

## 实战代码

### 场景 1：自纠正循环（Self-Correcting Loop）

一个节点反复检查和修正自己的输出，直到质量达标：

```python
"""
自纠正循环：节点通过 Command 自循环，直到输出质量达标
"""
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义状态 =====
class State(TypedDict):
    query: str                                          # 用户问题
    draft: str                                          # 当前草稿
    feedback: str                                       # 反馈意见
    revision_count: int                                 # 修改次数
    history: Annotated[list[str], operator.add]         # 修改历史


# ===== 2. 生成初始草稿 =====
def generate_draft(state: State) -> Command[Literal["review"]]:
    """生成初始回答草稿"""
    query = state["query"]

    # 模拟 LLM 生成（实际项目中调用 openai 等）
    draft = f"关于'{query}'的初始回答：这是一个需要详细解释的问题..."

    return Command(
        update={
            "draft": draft,
            "revision_count": 0,
            "history": [f"v0: {draft[:50]}..."]
        },
        goto="review"
    )


# ===== 3. 审查节点（自循环核心） =====
def review(state: State) -> Command[Literal["review", "__end__"]]:
    """
    审查草稿质量，决定是继续修改还是通过
    这是自循环的核心：通过 Command(goto="review") 实现自循环
    """
    draft = state["draft"]
    count = state["revision_count"]

    # 模拟质量评分（实际项目中可用 LLM 评估）
    quality_score = min(0.5 + count * 0.2, 1.0)

    # 终止条件：质量达标或修改次数过多
    if quality_score >= 0.9 or count >= 3:
        return Command(
            update={
                "feedback": f"通过审查！质量分数: {quality_score:.1f}",
                "history": [f"审查通过 (score={quality_score:.1f})"]
            },
            goto="__end__"  # 结束循环
        )

    # 未达标：修改草稿并自循环
    improved_draft = f"{draft} [第{count+1}次修改: 补充了更多细节和示例]"

    return Command(
        update={
            "draft": improved_draft,
            "feedback": f"需要改进，当前分数: {quality_score:.1f}",
            "revision_count": count + 1,
            "history": [f"v{count+1}: 修改后 score={quality_score:.1f}"]
        },
        goto="review"  # 自循环：回到自己
    )


# ===== 4. 构建图 =====
builder = StateGraph(State)

builder.add_node("generate_draft", generate_draft)
builder.add_node("review", review)

builder.add_edge(START, "generate_draft")
# 注意：不需要 add_conditional_edges，Command 自己处理路由

graph = builder.compile()

# ===== 5. 执行 =====
result = graph.invoke({"query": "什么是 RAG？"})

print(f"最终草稿: {result['draft'][:80]}...")
print(f"修改次数: {result['revision_count']}")
print(f"反馈: {result['feedback']}")
print(f"\n修改历史:")
for h in result["history"]:
    print(f"  - {h}")
```

**执行流程**:

```
START → generate_draft
  │
  ├── Command(goto="review")
  │
  ▼
review (第1次, score=0.5)
  │
  ├── Command(goto="review")  ← 自循环
  │
  ▼
review (第2次, score=0.7)
  │
  ├── Command(goto="review")  ← 自循环
  │
  ▼
review (第3次, score=0.9)
  │
  ├── Command(goto="__end__")  ← 达标，结束
  │
  ▼
END
```

### 场景 2：多智能体动态路由

多个 Agent 之间根据任务类型动态切换，同时传递上下文：

```python
"""
多智能体交接：通过 Command 实现 Agent 之间的动态路由和状态传递
"""
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    task: str
    current_agent: str
    result: str
    handoff_count: int


# ===== 2. 路由器节点 =====
def router(state: AgentState) -> Command[Literal["researcher", "coder", "writer"]]:
    """根据任务内容路由到合适的 Agent"""
    task = state["task"].lower()

    if "研究" in task or "调研" in task:
        target = "researcher"
    elif "代码" in task or "编程" in task:
        target = "coder"
    else:
        target = "writer"

    return Command(
        update={"current_agent": target, "handoff_count": 0},
        goto=target
    )


# ===== 3. 研究 Agent =====
def researcher(state: AgentState) -> Command[Literal["coder", "writer", "__end__"]]:
    """研究 Agent：完成研究后可能交接给其他 Agent"""
    task = state["task"]

    # 模拟研究结果
    research_result = f"研究发现：关于'{task}'的关键信息已收集完毕"

    # 如果任务还需要编码，交接给 coder
    if "代码" in task:
        return Command(
            update={
                "result": research_result,
                "handoff_count": state["handoff_count"] + 1
            },
            goto="coder"  # 交接给编码 Agent
        )

    return Command(
        update={"result": research_result},
        goto="__end__"
    )


# ===== 4. 编码 Agent =====
def coder(state: AgentState) -> Command[Literal["writer", "__end__"]]:
    """编码 Agent"""
    prev_result = state.get("result", "")
    code_result = f"{prev_result} | 代码实现已完成"

    return Command(
        update={"result": code_result},
        goto="__end__"
    )


# ===== 5. 写作 Agent =====
def writer(state: AgentState) -> Command[Literal["__end__"]]:
    """写作 Agent"""
    prev_result = state.get("result", "")
    write_result = f"{prev_result} | 文档已撰写完成"

    return Command(
        update={"result": write_result},
        goto="__end__"
    )


# ===== 6. 构建图 =====
builder = StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)
builder.add_node("writer", writer)

builder.add_edge(START, "router")

graph = builder.compile()

# ===== 7. 测试不同任务 =====
# 测试 1：纯研究任务
result1 = graph.invoke({"task": "调研 RAG 最新进展"})
print(f"任务1: {result1['result']}")

# 测试 2：研究 + 编码任务
result2 = graph.invoke({"task": "研究并编写代码实现向量检索"})
print(f"任务2: {result2['result']}")

# 测试 3：写作任务
result3 = graph.invoke({"task": "撰写技术文档"})
print(f"任务3: {result3['result']}")
```

### 场景 3：Command 与 Send 组合 - 动态并行分发

```python
"""
Command + Send 组合：从节点内部动态创建并行任务
"""
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send


# ===== 1. 定义状态 =====
class BatchState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]
    status: str


# ===== 2. 分析节点：决定如何分发 =====
def analyze(state: BatchState) -> Command[Literal["summarize"]]:
    """分析文档列表，通过 Command + Send 动态创建并行任务"""
    docs = state["documents"]

    return Command(
        update={"status": f"正在处理 {len(docs)} 个文档"},
        goto=[Send("summarize", {"doc": doc, "index": i})
              for i, doc in enumerate(docs)]
    )


# ===== 3. 摘要节点：并行处理每个文档 =====
def summarize(state: dict) -> dict:
    """为单个文档生成摘要（并行执行）"""
    doc = state["doc"]
    index = state["index"]

    summary = f"[文档{index}摘要] {doc[:30]}..."
    return {"summaries": [summary]}


# ===== 4. 汇总节点 =====
def aggregate(state: BatchState) -> dict:
    """汇总所有摘要"""
    count = len(state["summaries"])
    return {"status": f"完成！共处理 {count} 个文档"}


# ===== 5. 构建图 =====
builder = StateGraph(BatchState)

builder.add_node("analyze", analyze)
builder.add_node("summarize", summarize)
builder.add_node("aggregate", aggregate)

builder.add_edge(START, "analyze")
builder.add_edge("summarize", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()

# ===== 6. 执行 =====
result = graph.invoke({
    "documents": [
        "RAG 是检索增强生成的缩写，它结合了检索和生成两个步骤",
        "LangGraph 是一个用于构建有状态多智能体应用的框架",
        "向量数据库用于存储和检索高维向量数据"
    ]
})

print(f"状态: {result['status']}")
for s in result["summaries"]:
    print(f"  {s}")
```

---

## 在 RAG 开发中的应用

### 1. 自纠正 RAG（Self-Correcting RAG）

```python
def rag_with_correction(state) -> Command[Literal["retrieve", "generate", "__end__"]]:
    """RAG 节点：检查回答质量，不满意则重新检索"""
    answer = state["answer"]
    query = state["query"]

    # 评估回答质量
    is_relevant = evaluate_relevance(answer, query)

    if not is_relevant and state["retry_count"] < 3:
        # 改写查询，重新检索
        return Command(
            update={
                "query": rewrite_query(query),
                "retry_count": state["retry_count"] + 1
            },
            goto="retrieve"  # 回到检索步骤
        )

    return Command(goto="__end__")
```

### 2. 多步推理 RAG

```python
def reasoning_step(state) -> Command[Literal["reasoning_step", "final_answer"]]:
    """多步推理：每一步可能需要额外检索"""
    steps = state["reasoning_steps"]

    if needs_more_info(steps):
        # 继续推理
        new_step = generate_next_step(steps)
        return Command(
            update={"reasoning_steps": steps + [new_step]},
            goto="reasoning_step"  # 自循环继续推理
        )

    return Command(goto="final_answer")
```

---

## 常见陷阱与最佳实践

### 陷阱 1：忘记类型注解

```python
# 错误：缺少返回类型注解，图无法正确渲染
def my_node(state: State):
    return Command(goto="other_node")

# 正确：必须声明可能的路由目标
def my_node(state: State) -> Command[Literal["other_node"]]:
    return Command(goto="other_node")
```

### 陷阱 2：Command 与静态边冲突

```python
# 危险：同时有静态边和 Command 路由
builder.add_edge("node_a", "node_b")  # 静态边

def node_a(state) -> Command[Literal["node_c"]]:
    return Command(goto="node_c")
# 结果：node_b 和 node_c 都会执行！

# 安全做法：使用 Command 时不定义静态边
```

### 陷阱 3：自循环没有终止条件

```python
# 危险：可能无限循环
def loop_node(state) -> Command[Literal["loop_node"]]:
    return Command(goto="loop_node")  # 永远自循环

# 安全做法：始终有退出路径
def loop_node(state) -> Command[Literal["loop_node", "__end__"]]:
    if should_stop(state):
        return Command(goto="__end__")
    return Command(goto="loop_node")
```

### 最佳实践总结

1. **始终声明类型注解**: `-> Command[Literal["node_a", "node_b"]]`
2. **避免与静态边混用**: Command 不覆盖静态边
3. **确保终止条件**: 自循环必须有退出路径
4. **配合 recursion_limit**: 作为兜底保护
5. **优先用条件边**: 如果只需要路由不需要更新状态，用条件边更清晰

---

## 参考资料

### 源码
- `langgraph/types.py:367-417` - Command 类定义
- `langgraph/graph/state.py:1492-1518` - `_control_branch()` 处理 Command 路由
- `langgraph/pregel/_io.py:56-78` - `map_command()` 处理外部 Command

### 文档
- [Command API](https://docs.langchain.com/oss/python/langgraph/graph-api) - 官方 API 文档
- [Multi-agent Handoffs](https://langchain-ai.github.io/langgraph/how-tos/command/) - Command 使用指南

---

**版本**: v1.0
**最后更新**: 2026-02-28
**作者**: Claude Code
**知识点**: 循环与迭代 - Command 自循环模式
