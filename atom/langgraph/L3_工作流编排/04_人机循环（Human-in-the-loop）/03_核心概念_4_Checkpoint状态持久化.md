# 03_核心概念_4 - Checkpoint 状态持久化

> Checkpoint 是 LangGraph 的状态快照机制，为中断提供状态保存和恢复的基础设施

---

## 概念定义

**Checkpoint 是 LangGraph 在每个超步结束时自动保存的状态快照，是 Human-in-the-loop 机制的必要基础设施。** 没有 Checkpoint，`interrupt()` 调用会直接报错——因为中断后的状态无处保存，恢复也就无从谈起。

Checkpoint 解决了三个核心问题：
1. **中断时保存完整图状态**：节点执行到 `interrupt()` 时，当前所有通道值、待执行节点、中断信息都被持久化
2. **恢复时加载状态继续执行**：通过 `thread_id` 找到对应的 Checkpoint，从中断点恢复执行
3. **支持跨进程、跨时间的恢复**：用户可以在几秒后恢复，也可以在几天后恢复——只要 Checkpoint 还在

---

## 为什么 HITL 必须有 Checkpoint

### 没有 Checkpoint 会怎样？

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

def my_node(state):
    # 没有 checkpointer，这行会直接报错
    answer = interrupt("请确认是否继续？")
    return {"result": answer}

builder = StateGraph(dict)
builder.add_node("ask", my_node)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)

# 没有传入 checkpointer
graph = builder.compile()

# 运行时报错：
# ValueError: Checkpointer is required for interrupt
graph.invoke({"input": "test"})
```

**报错原因**：`interrupt()` 需要将当前状态保存到某个地方，然后在恢复时读取回来。没有 Checkpointer，这个"某个地方"不存在。

### Checkpoint 在中断-恢复流程中的角色

```
第一次调用 graph.invoke(input, config)
    ↓
节点开始执行
    ↓
遇到 interrupt(value)
    ↓
┌─────────────────────────────────┐
│  Checkpointer 保存状态快照：     │
│  - 所有通道的当前值              │
│  - 下一步要执行的节点            │
│  - 中断信息（value + id）        │
│  - 元数据（时间戳、步骤号等）     │
└─────────────────────────────────┘
    ↓
抛出 GraphInterrupt，返回给客户端
    ↓
... 时间流逝（可能是几秒，也可能是几天）...
    ↓
第二次调用 graph.invoke(Command(resume=value), config)
    ↓
┌─────────────────────────────────┐
│  Checkpointer 加载状态快照：     │
│  - 通过 thread_id 找到快照       │
│  - 恢复所有通道值                │
│  - 恢复中断信息                  │
└─────────────────────────────────┘
    ↓
从节点开头重新执行
    ↓
interrupt() 返回 resume 值（不再抛异常）
    ↓
继续后续节点
```

---

## Checkpointer 类型

LangGraph 提供了三种 Checkpointer 实现，适用于不同场景：

### 1. MemorySaver — 内存存储

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 数据存储在 Python 进程内存中
- 进程结束后数据丢失
- 速度最快，零配置
- 适合开发调试和单元测试

**适用场景**：本地开发、Jupyter Notebook 实验、单元测试

### 2. PostgresSaver — PostgreSQL 存储

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 同步版本
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)

# 异步版本
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)

graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 数据持久化到 PostgreSQL 数据库
- 支持多进程、多服务器共享状态
- 支持事务和并发控制
- 需要额外安装：`pip install langgraph-checkpoint-postgres`

**适用场景**：生产环境、分布式部署、需要高可用的系统

### 3. SqliteSaver — SQLite 存储

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 文件存储
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 内存存储（类似 MemorySaver 但有 SQL 接口）
checkpointer = SqliteSaver.from_conn_string(":memory:")

graph = builder.compile(checkpointer=checkpointer)
```

**特点**：
- 轻量级文件数据库，无需额外服务
- 数据持久化到本地文件
- 单进程访问（不支持并发写入）
- 需要额外安装：`pip install langgraph-checkpoint-sqlite`

**适用场景**：轻量级持久化、单机部署、原型验证

### 三种 Checkpointer 对比

| 特性 | MemorySaver | SqliteSaver | PostgresSaver |
|------|-------------|-------------|---------------|
| 持久化 | 否（进程内） | 是（文件） | 是（数据库） |
| 并发支持 | 单进程 | 单进程 | 多进程 |
| 配置复杂度 | 零配置 | 低 | 中 |
| 性能 | 最快 | 快 | 中等 |
| 适用环境 | 开发测试 | 轻量生产 | 生产环境 |
| 额外依赖 | 无 | langgraph-checkpoint-sqlite | langgraph-checkpoint-postgres |

---

## StateSnapshot 结构

当图被中断时，可以通过 `get_state()` 获取当前的状态快照。这个快照是一个 `StateSnapshot` 对象，包含了恢复执行所需的全部信息。

### 源码定义

```python
# 来源：langgraph/types.py:268
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any        # 当前所有通道的值
    next: tuple[str, ...]               # 下一步要执行的节点名称
    config: RunnableConfig              # 当前配置（含 thread_id）
    metadata: CheckpointMetadata | None # 元数据（步骤号、来源等）
    created_at: str | None              # 创建时间
    parent_config: RunnableConfig | None # 父检查点配置
    tasks: tuple[PregelTask, ...]       # 当前步骤的任务列表
    interrupts: tuple[Interrupt, ...]   # 待解决的中断列表
```

### 各字段详解

**`values`** — 当前状态的完整快照：
```python
# 包含所有通道的当前值
snapshot.values
# {'messages': [...], 'status': 'pending', 'count': 3}
```

**`next`** — 下一步要执行的节点：
```python
# 中断时，next 包含被中断的节点
snapshot.next
# ('approval_node',)  ← 这个节点被中断了，恢复时会重新执行
```

**`tasks`** — 当前步骤的任务信息：
```python
# 每个任务包含节点名、状态、中断信息
for task in snapshot.tasks:
    print(f"节点: {task.name}")
    print(f"中断: {task.interrupts}")
    # 节点: approval_node
    # 中断: (Interrupt(value='请确认是否继续？', id='abc123'),)
```

**`interrupts`** — 待解决的中断列表：
```python
# 直接获取所有待解决的中断
for intr in snapshot.interrupts:
    print(f"中断值: {intr.value}")
    print(f"中断ID: {intr.id}")
```

---

## get_state() 与 get_state_history()

### get_state() — 获取当前状态

```python
config = {"configurable": {"thread_id": "thread-1"}}

# 获取当前最新的状态快照
snapshot = graph.get_state(config)

print(f"当前状态值: {snapshot.values}")
print(f"下一步节点: {snapshot.next}")
print(f"待解决中断: {snapshot.interrupts}")
print(f"创建时间: {snapshot.created_at}")
```

**典型用途**：
- 检查图是否处于中断状态
- 获取中断传递的信息（用于展示给用户）
- 调试时查看当前状态

### get_state_history() — 获取状态历史

```python
config = {"configurable": {"thread_id": "thread-1"}}

# 获取该线程的所有历史状态
for snapshot in graph.get_state_history(config):
    print(f"时间: {snapshot.created_at}")
    print(f"状态: {snapshot.values}")
    print(f"下一步: {snapshot.next}")
    print(f"中断: {snapshot.interrupts}")
    print("---")
```

**典型用途**：
- 查看图的完整执行历史
- 实现"时间旅行"调试（回到某个历史状态）
- 审计和日志记录

---

## thread_id 的作用

`thread_id` 是 Checkpoint 系统中最重要的标识符，它将一次完整的对话/工作流串联起来。

### 核心规则

```python
# 规则1：同一个 thread_id = 同一个工作流实例
config = {"configurable": {"thread_id": "workflow-abc"}}

# 第一次调用：开始工作流
graph.invoke({"input": "start"}, config)

# 恢复时必须用同一个 thread_id
graph.invoke(Command(resume="approved"), config)  # ✅ 正确

# 用不同的 thread_id 会开始一个全新的工作流
other_config = {"configurable": {"thread_id": "workflow-xyz"}}
graph.invoke(Command(resume="approved"), other_config)  # ❌ 找不到中断状态
```

### thread_id 的设计建议

```python
import uuid

# 方式1：UUID（最常用）
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 方式2：业务含义（推荐生产环境）
config = {"configurable": {"thread_id": f"order-approval-{order_id}"}}

# 方式3：用户会话（对话式场景）
config = {"configurable": {"thread_id": f"user-{user_id}-session-{session_id}"}}
```

**最佳实践**：
- 开发测试时用简单字符串（如 `"test-1"`）
- 生产环境用有业务含义的 ID
- 确保 thread_id 在恢复时保持一致
- 不同的工作流实例使用不同的 thread_id

---

## 完整代码示例：中断前后的状态检查

下面的示例展示了如何在中断前后检查 Checkpoint 状态，理解整个持久化流程。

```python
"""
Checkpoint 状态持久化完整示例
演示中断前后的状态检查和恢复流程
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# 1. 定义状态
# ============================================================

class OrderState(TypedDict):
    order_id: str
    amount: float
    status: Optional[str]
    approver: Optional[str]


# ============================================================
# 2. 定义节点
# ============================================================

def validate_order(state: OrderState):
    """验证订单"""
    print(f"[验证] 订单 {state['order_id']}，金额 {state['amount']}")
    return {"status": "validated"}


def request_approval(state: OrderState):
    """请求人工审批 — 这里会中断"""
    print(f"[审批] 等待审批，订单 {state['order_id']}")

    # interrupt() 暂停执行，将审批信息发送给客户端
    decision = interrupt({
        "question": "是否批准此订单？",
        "order_id": state["order_id"],
        "amount": state["amount"],
    })

    # 恢复后，decision 就是用户传入的 resume 值
    print(f"[审批] 收到决策: {decision}")

    if decision.get("approved"):
        return {
            "status": "approved",
            "approver": decision.get("approver", "unknown"),
        }
    else:
        return {"status": "rejected"}


def execute_order(state: OrderState):
    """执行订单"""
    print(f"[执行] 订单 {state['order_id']} 状态: {state['status']}")
    if state["status"] == "approved":
        return {"status": "completed"}
    else:
        return {"status": "cancelled"}


# ============================================================
# 3. 构建图
# ============================================================

builder = StateGraph(OrderState)
builder.add_node("validate", validate_order)
builder.add_node("approve", request_approval)
builder.add_node("execute", execute_order)

builder.add_edge(START, "validate")
builder.add_edge("validate", "approve")
builder.add_edge("approve", "execute")
builder.add_edge("execute", END)

# 关键：必须传入 checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ============================================================
# 4. 执行并观察 Checkpoint 状态
# ============================================================

def inspect_state(graph, config, label=""):
    """检查当前状态快照"""
    snapshot = graph.get_state(config)
    print(f"\n{'='*50}")
    print(f"状态检查 [{label}]")
    print(f"{'='*50}")
    print(f"  values:     {snapshot.values}")
    print(f"  next:       {snapshot.next}")
    print(f"  interrupts: {snapshot.interrupts}")

    if snapshot.tasks:
        for task in snapshot.tasks:
            print(f"  task:       name={task.name}, interrupts={task.interrupts}")
    print()


# 配置 thread_id
config = {"configurable": {"thread_id": "order-001"}}

# --- 第一阶段：初始调用，遇到中断 ---
print("=" * 60)
print("第一阶段：初始调用")
print("=" * 60)

result = graph.invoke(
    {
        "order_id": "ORD-2024-001",
        "amount": 5000.0,
        "status": None,
        "approver": None,
    },
    config,
)

# 检查中断后的状态
inspect_state(graph, config, "中断后")

# 查看返回的中断信息
if "__interrupt__" in result:
    for intr in result["__interrupt__"]:
        print(f"中断信息: {intr.value}")
        print(f"中断 ID:  {intr.id}")


# --- 第二阶段：查看状态历史 ---
print("\n" + "=" * 60)
print("第二阶段：查看状态历史")
print("=" * 60)

for i, snapshot in enumerate(graph.get_state_history(config)):
    print(f"\n  历史 #{i}:")
    print(f"    values: {snapshot.values}")
    print(f"    next:   {snapshot.next}")
    print(f"    time:   {snapshot.created_at}")


# --- 第三阶段：恢复执行 ---
print("\n" + "=" * 60)
print("第三阶段：恢复执行")
print("=" * 60)

# 用 Command(resume=...) 恢复，传入审批决策
final_result = graph.invoke(
    Command(resume={"approved": True, "approver": "张经理"}),
    config,
)

# 检查恢复后的最终状态
inspect_state(graph, config, "恢复完成后")

print(f"最终结果: {final_result}")
```

### 预期输出

```
============================================================
第一阶段：初始调用
============================================================
[验证] 订单 ORD-2024-001，金额 5000.0
[审批] 等待审批，订单 ORD-2024-001

==================================================
状态检查 [中断后]
==================================================
  values:     {'order_id': 'ORD-2024-001', 'amount': 5000.0, 'status': 'validated', 'approver': None}
  next:       ('approve',)
  interrupts: (Interrupt(value={'question': '是否批准此订单？', ...}, id='...'),)
  task:       name=approve, interrupts=(Interrupt(...),)

中断信息: {'question': '是否批准此订单？', 'order_id': 'ORD-2024-001', 'amount': 5000.0}

============================================================
第二阶段：查看状态历史
============================================================
  历史 #0:
    values: {'order_id': 'ORD-2024-001', 'amount': 5000.0, 'status': 'validated', ...}
    next:   ('approve',)
    time:   2026-02-28T10:00:01+00:00

  历史 #1:
    values: {'order_id': 'ORD-2024-001', 'amount': 5000.0, 'status': 'validated', ...}
    next:   ('validate',)
    time:   2026-02-28T10:00:00+00:00

============================================================
第三阶段：恢复执行
============================================================
[审批] 等待审批，订单 ORD-2024-001
[审批] 收到决策: {'approved': True, 'approver': '张经理'}
[执行] 订单 ORD-2024-001 状态: approved

==================================================
状态检查 [恢复完成后]
==================================================
  values:     {'order_id': 'ORD-2024-001', 'amount': 5000.0, 'status': 'completed', 'approver': '张经理'}
  next:       ()
  interrupts: ()

最终结果: {'order_id': 'ORD-2024-001', 'amount': 5000.0, 'status': 'completed', 'approver': '张经理'}
```

### 关键观察点

1. **中断后 `next` 指向被中断的节点**：`('approve',)` 表示恢复时会重新执行 `approve` 节点
2. **`interrupts` 包含中断详情**：可以从中获取传递给用户的信息和中断 ID
3. **恢复时节点从头重新执行**：注意输出中 `[审批] 等待审批` 打印了两次——第一次中断，第二次恢复
4. **状态历史记录了每个超步**：可以通过 `get_state_history()` 回溯整个执行过程

---

## Checkpoint 的内部工作原理

### 保存时机

Checkpoint 在以下时机自动保存：

```
每个超步结束时
    ↓
┌─────────────────────────────────────┐
│ 1. 节点正常完成 → 保存 Checkpoint   │
│ 2. 遇到 interrupt() → 保存后中断    │
│ 3. interrupt_before 触发 → 保存后中断│
│ 4. interrupt_after 触发 → 保存后中断 │
└─────────────────────────────────────┘
```

### 版本追踪机制

```python
# 来源：langgraph/pregel/_algo.py
# Checkpoint 通过 channel_versions 追踪状态变化

def should_interrupt(checkpoint, interrupt_nodes, tasks):
    """检查是否应该中断"""
    # 获取上次中断时的版本号
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})

    # 检查自上次中断以来是否有通道更新
    any_updates = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )

    # 有更新且节点在中断列表中 → 触发中断
    return [task for task in tasks if task.name in interrupt_nodes] if any_updates else []
```

这个机制确保了：
- 同一个中断不会重复触发
- 只有状态发生变化后才会触发新的中断
- 版本号单调递增，保证顺序性

---

## 常见问题与注意事项

### 1. 忘记配置 Checkpointer

```python
# ❌ 错误：没有 checkpointer
graph = builder.compile()
graph.invoke(input)  # interrupt() 会报错

# ✅ 正确：配置 checkpointer
graph = builder.compile(checkpointer=MemorySaver())
```

### 2. thread_id 不一致

```python
config1 = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}

graph.invoke(input, config1)  # 在 thread-1 中断

# ❌ 错误：用不同的 thread_id 恢复
graph.invoke(Command(resume="yes"), config2)  # 找不到中断

# ✅ 正确：用相同的 thread_id 恢复
graph.invoke(Command(resume="yes"), config1)
```

### 3. MemorySaver 在生产环境的风险

```python
# ❌ 生产环境不要用 MemorySaver
# 服务重启后所有中断状态丢失
checkpointer = MemorySaver()

# ✅ 生产环境用 PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(conn_string)
```

### 4. 异步场景的 Checkpointer 选择

```python
# 异步图需要使用异步版本的 Checkpointer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
graph = builder.compile(checkpointer=checkpointer)

# 异步调用
result = await graph.ainvoke(input, config)
result = await graph.ainvoke(Command(resume=value), config)
```

---

## 总结

**Checkpoint 是 HITL 的地基**——没有它，中断和恢复就无法实现。

核心要点：
1. **必须配置 Checkpointer** 才能使用 `interrupt()`
2. **MemorySaver** 用于开发，**PostgresSaver** 用于生产
3. **StateSnapshot** 包含恢复所需的全部信息（values、next、interrupts）
4. **thread_id** 是串联中断和恢复的关键标识
5. **get_state()** 检查当前状态，**get_state_history()** 查看完整历史
6. Checkpoint 在每个超步结束时自动保存，通过版本号追踪状态变化

---

## 引用来源

1. **源码分析**：`langgraph/types.py:268` — StateSnapshot 定义
2. **源码分析**：`langgraph/pregel/_algo.py` — should_interrupt() 版本追踪机制
3. **源码分析**：`langgraph/pregel/_loop.py` — PregelLoop 中断状态管理
4. **官方文档**：LangGraph Human-in-the-loop — https://docs.langchain.com/oss/python/langgraph/interrupts

---

**相关概念**：
- [03_核心概念_1_interrupt函数.md](./03_核心概念_1_interrupt函数.md)
- [03_核心概念_2_Command恢复执行.md](./03_核心概念_2_Command恢复执行.md)
- [03_核心概念_3_静态断点配置.md](./03_核心概念_3_静态断点配置.md)
- [03_核心概念_5_结构化人机交互.md](./03_核心概念_5_结构化人机交互.md)
