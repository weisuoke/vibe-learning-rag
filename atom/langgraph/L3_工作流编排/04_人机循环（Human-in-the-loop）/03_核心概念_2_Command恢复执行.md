# 核心概念2：Command 恢复执行

> Command 是 LangGraph 的恢复执行原语，通过 resume 参数向中断的图传递人类决策

---

## 什么是 Command？

**Command** 是 LangGraph 中用于恢复中断图执行的核心类。当图通过 `interrupt()` 暂停后，你需要通过 `Command(resume=value)` 告诉图"人类已经做出决策，继续执行"。

把它想象成一个遥控器：图暂停了，你按下遥控器上的按钮（Command），同时告诉它你的选择（resume 值），图就从暂停的地方继续运行。

---

## Command 类的完整结构

从源码 `langgraph/types.py` 中，Command 类的定义如下：

```python
# 源码位置：libs/langgraph/langgraph/types.py:368
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None      # 目标图
    update: Any | None = None     # 状态更新
    resume: dict[str, Any] | Any | None = None  # 恢复值
    goto: Send | Sequence[Send | N] | N = ()     # 导航目标
```

### 四个参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `graph` | `str \| None` | `None` | 目标图。`None` = 当前图，`Command.PARENT` = 父图 |
| `update` | `Any \| None` | `None` | 在恢复前更新图的状态 |
| `resume` | `dict[str, Any] \| Any \| None` | `None` | 传递给 `interrupt()` 的恢复值 |
| `goto` | `Send \| Sequence[Send \| N] \| N` | `()` | 恢复后导航到指定节点 |

### 参数1：graph — 目标图

```python
# 默认：恢复当前图
Command(resume="yes")

# 恢复父图（在子图中使用）
Command(resume="yes", graph=Command.PARENT)
```

当你的图有子图嵌套时，`graph` 参数决定了 Command 作用于哪一层。大多数场景下保持默认 `None` 即可。

### 参数2：update — 状态更新

在恢复执行之前，先修改图的状态：

```python
# 恢复的同时更新状态
Command(
    resume="approved",
    update={"reviewer": "Alice", "review_time": "2026-02-28"}
)
```

这在审批场景中非常有用——你不仅告诉图"批准了"，还记录了"谁批准的"和"什么时候批准的"。

### 参数3：resume — 恢复值（核心参数）

这是 Command 最重要的参数，支持两种形式：

**形式一：单值恢复（最常用）**

```python
# 恢复下一个待处理的中断
Command(resume="approved")
Command(resume=True)
Command(resume={"action": "proceed", "comment": "looks good"})
```

单值恢复会按顺序匹配下一个未处理的 `interrupt()` 调用。

**形式二：按 ID 恢复（精确控制）**

```python
# 通过中断 ID 恢复特定的中断
Command(resume={"interrupt-id-abc123": "approved"})
```

当一个节点有多个 `interrupt()` 调用时，按 ID 恢复可以精确指定恢复哪一个。中断 ID 可以从 `StateSnapshot.interrupts` 中获取：

```python
# 获取中断 ID
snapshot = graph.get_state(config)
for intr in snapshot.interrupts:
    print(f"中断 ID: {intr.id}, 值: {intr.value}")
```

### 参数4：goto — 导航目标

恢复后跳转到指定节点，而不是按照图的默认边走：

```python
# 恢复后跳转到 "proceed" 节点
Command(resume="approved", goto="proceed")

# 恢复后跳转到多个节点（并行执行）
Command(resume="approved", goto=["notify", "log"])

# 使用 Send 对象传递不同状态
from langgraph.types import Send
Command(resume="approved", goto=[Send("process", {"priority": "high"})])
```

---

## 四种恢复方式

LangGraph 支持四种方式恢复中断的图，覆盖同步/异步和普通/流式场景：

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}

# 方式1：同步恢复（最常用）
result = graph.invoke(Command(resume="approved"), config=config)

# 方式2：异步恢复
result = await graph.ainvoke(Command(resume="approved"), config=config)

# 方式3：流式恢复（可以实时看到中间结果）
for chunk in graph.stream(Command(resume="approved"), config=config):
    print(chunk)

# 方式4：异步流式恢复
async for chunk in graph.astream(Command(resume="approved"), config=config):
    print(chunk)
```

### 如何选择恢复方式？

| 方式 | 场景 | 特点 |
|------|------|------|
| `invoke` | 脚本、后端 API | 简单直接，等待完成 |
| `ainvoke` | 异步后端（FastAPI） | 不阻塞事件循环 |
| `stream` | 需要实时反馈 | 逐步返回中间结果 |
| `astream` | 异步 + 实时反馈 | 最灵活，适合 WebSocket |

---

## resume 的两种形式详解

### 形式一：单值恢复

这是最常用的方式。你传入一个值，它会自动匹配下一个待处理的 `interrupt()`：

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional

class State(TypedDict):
    task: str
    decision: Optional[str]

def review_node(state: State):
    """审核节点 - 包含一个中断点"""
    # interrupt() 第一次执行时暂停，恢复时返回 resume 值
    decision = interrupt({
        "question": "请审核此任务",
        "task": state["task"]
    })
    return {"decision": decision}

def execute_node(state: State):
    return {"task": f"已执行: {state['task']} (决策: {state['decision']})"}

# 构建图
builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_node("execute", execute_node)
builder.add_edge(START, "review")
builder.add_edge("review", "execute")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "demo-1"}}

# 第一次运行 - 遇到 interrupt 暂停
result = graph.invoke({"task": "部署新版本"}, config=config)
# 图在 review_node 的 interrupt() 处暂停

# 恢复执行 - 传入单值
result = graph.invoke(Command(resume="批准部署"), config=config)
print(result)
# {'task': '已执行: 部署新版本 (决策: 批准部署)', 'decision': '批准部署'}
```

### 形式二：按 ID 恢复

当一个节点有多个 `interrupt()` 调用时，你可能需要精确恢复某一个特定的中断：

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional

class MultiReviewState(TypedDict):
    content: str
    tech_review: Optional[str]
    legal_review: Optional[str]

def multi_review_node(state: MultiReviewState):
    """包含多个中断点的节点"""
    # 第一个中断：技术审核
    tech_decision = interrupt({
        "type": "tech_review",
        "question": "技术方案是否可行？",
        "content": state["content"]
    })

    # 第二个中断：法务审核
    legal_decision = interrupt({
        "type": "legal_review",
        "question": "是否符合合规要求？",
        "content": state["content"]
    })

    return {
        "tech_review": tech_decision,
        "legal_review": legal_decision
    }

builder = StateGraph(MultiReviewState)
builder.add_node("review", multi_review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "multi-review-1"}}

# 第一次运行 - 在第一个 interrupt 处暂停
result = graph.invoke({"content": "新功能上线方案"}, config=config)

# 查看中断信息，获取中断 ID
snapshot = graph.get_state(config)
print(f"待处理中断数: {len(snapshot.interrupts)}")
for intr in snapshot.interrupts:
    print(f"  ID: {intr.id}, 值: {intr.value}")

# 方式A：单值恢复 - 按顺序恢复第一个中断
result = graph.invoke(Command(resume="技术通过"), config=config)
# 此时图在第二个 interrupt 处再次暂停

# 继续恢复第二个中断
result = graph.invoke(Command(resume="合规通过"), config=config)
print(result)
# {'content': '新功能上线方案', 'tech_review': '技术通过', 'legal_review': '合规通过'}
```

**按 ID 恢复的使用场景**：

```python
# 当你需要精确控制恢复哪个中断时
snapshot = graph.get_state(config)
interrupt_id = snapshot.interrupts[0].id

# 通过字典形式按 ID 恢复
result = graph.invoke(
    Command(resume={interrupt_id: "技术通过"}),
    config=config
)
```

> **什么时候用按 ID 恢复？**
> - 节点内有多个 `interrupt()` 且你需要跳过某些中断
> - 并行节点各自有中断，需要分别恢复
> - 需要精确控制恢复顺序的复杂场景
>
> 大多数情况下，单值恢复就够用了。

---

## Command 组合使用

Command 的强大之处在于可以同时使用多个参数，实现"恢复 + 更新状态 + 跳转"的组合操作：

### 组合1：resume + goto（条件路由）

根据人类决策跳转到不同节点：

```python
from typing import Literal, Optional, TypedDict
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    request: str
    status: Optional[str]
    reason: Optional[str]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "reject"]]:
    """审批节点 - 根据人类决策路由"""
    decision = interrupt({
        "question": "是否批准此请求？",
        "request": state["request"]
    })

    if decision.get("approved"):
        return Command(goto="proceed")
    else:
        return Command(goto="reject")

def proceed_node(state: ApprovalState):
    return {"status": "approved", "reason": "审批通过"}

def reject_node(state: ApprovalState):
    return {"status": "rejected", "reason": "审批拒绝"}

builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("reject", reject_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("reject", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "approval-1"}}

# 第一次运行
graph.invoke({"request": "申请服务器扩容"}, config=config)

# 恢复：批准
result = graph.invoke(
    Command(resume={"approved": True}),
    config=config
)
print(result["status"])  # "approved"
```

### 组合2：resume + update（恢复并更新状态）

在恢复的同时注入额外信息到图状态中：

```python
# 恢复并记录审批人信息
result = graph.invoke(
    Command(
        resume="approved",
        update={
            "reviewer": "张三",
            "review_time": "2026-02-28 14:30",
            "review_comment": "方案可行，同意执行"
        }
    ),
    config=config
)
```

### 组合3：resume + goto + update（完整组合）

三个参数同时使用，实现最灵活的恢复控制：

```python
# 完整组合：恢复 + 更新状态 + 跳转到指定节点
result = graph.invoke(
    Command(
        resume="approved_with_changes",
        update={"modifications": ["调整预算", "延长工期"]},
        goto="modified_proceed"  # 跳转到"有修改的执行"节点
    ),
    config=config
)
```

---

## thread_id 一致性要求

**这是最容易踩的坑**：恢复时必须使用与初始运行相同的 `thread_id`。

```python
config = {"configurable": {"thread_id": "my-thread-123"}}

# 初始运行
graph.invoke({"task": "重要任务"}, config=config)

# 恢复 - 必须用同一个 config（同一个 thread_id）
graph.invoke(Command(resume="go"), config=config)  # 正确

# 错误！不同的 thread_id 会找不到之前的中断状态
wrong_config = {"configurable": {"thread_id": "different-thread"}}
graph.invoke(Command(resume="go"), config=wrong_config)  # 错误：找不到中断
```

**为什么？** 因为 `thread_id` 是 Checkpointer 定位保存状态的关键标识。中断时的状态保存在这个 thread_id 下，恢复时需要从同一个位置读取。

---

## 完整实战示例：文档审批工作流

```python
"""
完整示例：多步骤文档审批工作流
演示 Command 的各种用法：resume、goto、update 的组合
"""

from typing import Literal, Optional, TypedDict, Annotated
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator


# ===== 1. 定义状态 =====
class DocState(TypedDict):
    title: str
    content: str
    status: str
    review_log: Annotated[list[str], operator.add]

# ===== 2. 定义节点 =====
def draft_node(state: DocState):
    """草稿节点"""
    print(f"[Draft] 创建文档: {state['title']}")
    return {
        "status": "draft_complete",
        "review_log": [f"文档 '{state['title']}' 草稿完成"]
    }

def review_node(state: DocState) -> Command[Literal["publish", "revise", "archive"]]:
    """审核节点 - 人工审核"""
    decision = interrupt({
        "type": "document_review",
        "title": state["title"],
        "content": state["content"],
        "question": "请审核此文档。选择: approve / revise / reject"
    })

    action = decision.get("action", "reject")
    comment = decision.get("comment", "")

    if action == "approve":
        return Command(
            goto="publish",
            update={
                "status": "approved",
                "review_log": [f"审核通过: {comment}"]
            }
        )
    elif action == "revise":
        return Command(
            goto="revise",
            update={
                "status": "needs_revision",
                "review_log": [f"需要修改: {comment}"]
            }
        )
    else:
        return Command(
            goto="archive",
            update={
                "status": "rejected",
                "review_log": [f"审核拒绝: {comment}"]
            }
        )

def publish_node(state: DocState):
    """发布节点"""
    print(f"[Publish] 文档已发布: {state['title']}")
    return {
        "status": "published",
        "review_log": ["文档已发布"]
    }

def revise_node(state: DocState):
    """修改节点 - 回到审核"""
    print(f"[Revise] 文档需要修改: {state['title']}")
    return {"review_log": ["文档已修改，重新提交审核"]}

def archive_node(state: DocState):
    """归档节点"""
    print(f"[Archive] 文档已归档: {state['title']}")
    return {
        "status": "archived",
        "review_log": ["文档已归档"]
    }


# ===== 3. 构建图 =====
builder = StateGraph(DocState)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_node("publish", publish_node)
builder.add_node("revise", revise_node)
builder.add_node("archive", archive_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
# review 节点通过 Command(goto=...) 动态路由，不需要静态边
builder.add_edge("publish", END)
builder.add_edge("revise", "review")  # 修改后回到审核
builder.add_edge("archive", END)

graph = builder.compile(checkpointer=MemorySaver())

# ===== 4. 执行流程 =====
config = {"configurable": {"thread_id": "doc-review-001"}}

# 步骤1：提交文档
print("=" * 50)
print("步骤1：提交文档")
print("=" * 50)
result = graph.invoke(
    {
        "title": "Q1 技术方案",
        "content": "本方案提出使用微服务架构重构现有系统...",
        "status": "new",
        "review_log": []
    },
    config=config
)
print(f"当前状态: {result.get('status', 'interrupted')}")

# 步骤2：第一次审核 - 要求修改
print("\n" + "=" * 50)
print("步骤2：审核 - 要求修改")
print("=" * 50)
result = graph.invoke(
    Command(resume={"action": "revise", "comment": "请补充性能测试数据"}),
    config=config
)
print(f"当前状态: {result.get('status', 'interrupted')}")

# 步骤3：第二次审核 - 批准
print("\n" + "=" * 50)
print("步骤3：审核 - 批准")
print("=" * 50)
result = graph.invoke(
    Command(resume={"action": "approve", "comment": "方案完善，同意执行"}),
    config=config
)
print(f"当前状态: {result['status']}")
print(f"审核日志: {result['review_log']}")
```

**运行输出**：
```
==================================================
步骤1：提交文档
==================================================
[Draft] 创建文档: Q1 技术方案
当前状态: interrupted

==================================================
步骤2：审核 - 要求修改
==================================================
[Revise] 文档需要修改: Q1 技术方案
当前状态: interrupted

==================================================
步骤3：审核 - 批准
==================================================
[Publish] 文档已发布: Q1 技术方案
当前状态: published
审核日志: ['文档 'Q1 技术方案' 草稿完成', '需要修改: 请补充性能测试数据',
           '文档已修改，重新提交审核', '审核通过: 方案完善，同意执行', '文档已发布']
```


---

## 执行流程图

```
用户调用 graph.invoke(input, config)
    ↓
节点执行 → 遇到 interrupt(value)
    ↓
抛出 GraphInterrupt 异常
    ↓
Checkpointer 保存当前状态（绑定 thread_id）
    ↓
返回中断信息给客户端
    ↓
客户端展示中断信息，等待人类决策
    ↓
人类做出决策
    ↓
graph.invoke(Command(resume=决策值), config)  ← 同一个 config!
    ↓
Checkpointer 加载保存的状态
    ↓
节点从头重新执行，interrupt() 返回 resume 值
    ↓
继续执行后续节点
```

---

## 与前端开发的类比

| Command 概念 | 前端类比 | 说明 |
|-------------|----------|------|
| `Command(resume=value)` | `Promise.resolve(value)` | 向暂停的异步操作传递结果 |
| `thread_id` | `requestId` | 标识一次完整的请求会话 |
| `resume` 单值 | `callback(result)` | 回调函数传递结果 |
| `resume` 字典 | `eventEmitter.emit(id, data)` | 按事件 ID 触发特定回调 |
| `goto` | `router.push("/path")` | 导航到指定路由 |
| `update` | `setState({...})` | 在继续前更新状态 |

**前端类比示例**：
```javascript
// Command(resume=value) 就像 Promise 的 resolve
const userDecision = await new Promise((resolve) => {
  // 显示审批弹窗
  showApprovalDialog({
    onApprove: () => resolve("approved"),
    onReject: () => resolve("rejected")
  });
});
// userDecision 就是 resume 的值
```

---

## 日常生活类比

**Command 就像餐厅点餐的确认环节**：

1. 服务员（图）拿着菜单来到你桌前，问"请问要点什么？"（`interrupt()`）
2. 服务员站在旁边等你（图暂停）
3. 你看完菜单做出决定（人类决策）
4. 你告诉服务员"我要宫保鸡丁"（`Command(resume="宫保鸡丁")`）
5. 服务员拿着你的选择去厨房下单（图恢复执行）

**组合使用的类比**：
- `resume` = 你的点餐内容（"宫保鸡丁"）
- `update` = 额外备注（"少放辣"、"加饭"）
- `goto` = 指定处理方式（"打包" vs "堂食"）

---

## 常见误区

### 误区1：恢复时不需要 config ❌

**错误**：
```python
graph.invoke(Command(resume="go"))  # 缺少 config！
```

**正确**：
```python
graph.invoke(Command(resume="go"), config=config)  # 必须传入相同的 config
```

### 误区2：Command 会从中断点继续执行 ❌

**错误理解**：恢复后从 `interrupt()` 的下一行开始执行。

**正确理解**：恢复后节点从头重新执行。`interrupt()` 在重新执行时不再抛出异常，而是直接返回 `resume` 值。

```python
def my_node(state):
    print("这行每次恢复都会执行！")  # 重新执行
    result = interrupt("请确认")      # 恢复时直接返回 resume 值
    print(f"收到: {result}")          # 只在恢复时执行到这里
    return {"result": result}
```

### 误区3：一次 Command 可以恢复所有中断 ❌

**错误理解**：一个 `Command(resume=value)` 可以一次性恢复节点内的所有 `interrupt()`。

**正确理解**：单值 resume 只恢复下一个未处理的中断。如果节点有 3 个 `interrupt()`，你需要恢复 3 次。

---

## 总结

**Command 是连接"人类决策"和"图恢复执行"的桥梁**：

1. **核心参数**：`resume` 传递决策值，`goto` 控制路由，`update` 更新状态
2. **两种恢复形式**：单值（按顺序）和字典（按 ID）
3. **四种调用方式**：invoke / ainvoke / stream / astream
4. **关键约束**：thread_id 必须一致，节点会从头重新执行
5. **组合使用**：resume + goto + update 实现灵活的恢复控制

---

## 参考资料

**源码**：
- `langgraph/types.py:368` - Command 类定义
- `langgraph/types.py:420` - interrupt() 函数
- `langgraph/types.py:161` - Interrupt 数据类

**官方文档**：
- [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Command API Reference](https://docs.langchain.com/oss/python/langgraph/reference/types#command)

---

**下一步学习**：
- 核心概念3：静态断点配置（interrupt_before/interrupt_after）
- 实战代码：审批工作流
- 实战代码：工具调用审批
