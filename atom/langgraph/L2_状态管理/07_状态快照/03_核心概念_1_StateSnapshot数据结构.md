# StateSnapshot 数据结构

> LangGraph 状态快照的用户层数据结构，包含 8 个字段，完整描述图在某一时刻的全部状态

---

## 一句话定义

**StateSnapshot 是一个 NamedTuple，包含 8 个字段，是 LangGraph 暴露给用户的"状态照片"——你通过 `get_state()` 拿到的就是它。**

---

## 为什么需要 StateSnapshot？

想象你在玩一个 RPG 游戏，每次存档时，游戏会记录：
- 你当前的血量、装备（**values**）
- 你下一步要去哪个地图（**next**）
- 这个存档的编号和位置（**config**）
- 存档的附加信息——第几关、怎么来的（**metadata**）
- 存档时间（**created_at**）
- 上一个存档是哪个（**parent_config**）
- 当前正在执行的任务（**tasks**）
- 暂停等待你操作的事件（**interrupts**）

StateSnapshot 就是 LangGraph 的"游戏存档"数据结构。

---

## 源码定义

```python
# 来源：langgraph/types.py:268-286
class StateSnapshot(NamedTuple):
    """Snapshot of the state of the graph at the beginning of a step."""
    values: dict[str, Any] | Any
    next: tuple[str, ...]
    config: RunnableConfig
    metadata: CheckpointMetadata | None
    created_at: str | None
    parent_config: RunnableConfig | None
    tasks: tuple[PregelTask, ...]
    interrupts: tuple[Interrupt, ...]
```

它是一个 `NamedTuple`，意味着：
- 不可变（创建后不能修改）
- 可以用 `.values`、`.next` 等属性访问
- 也可以用索引访问（`snapshot[0]` 等于 `snapshot.values`）

---

## 8 个字段逐一详解

### 字段 1：`values` — 当前状态值

**类型：** `dict[str, Any] | Any`

**作用：** 保存图在这一时刻所有 channel 的值，也就是你定义的 State 中每个字段的当前值。

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph

# 定义状态
class MyState(TypedDict):
    messages: Annotated[list[str], add]
    count: int

# 构建并运行图后获取快照
snapshot = graph.get_state(config)

# values 就是当前的状态字典
print(snapshot.values)
# 输出: {'messages': ['你好', '你好！有什么可以帮你的？'], 'count': 1}

# 访问具体字段
print(snapshot.values["messages"])
# 输出: ['你好', '你好！有什么可以帮你的？']
```

**关键点：**
- 大多数情况下是一个字典，key 是你定义的 State 字段名
- 如果图使用了自定义 output schema，`values` 可能只包含 output 字段
- 这是 checkpoint 中 `channel_values` 的用户友好版本

**在实际应用中：**
- 查看对话历史：`snapshot.values["messages"]`
- 检查中间结果：`snapshot.values["search_results"]`
- 调试状态流转：对比不同快照的 values 变化

---

### 字段 2：`next` — 下一步要执行的节点

**类型：** `tuple[str, ...]`

**作用：** 告诉你从这个快照恢复执行时，接下来会运行哪些节点。

```python
snapshot = graph.get_state(config)

# 查看下一步
print(snapshot.next)
# 可能输出: ('chatbot',)        — 下一步执行 chatbot 节点
# 可能输出: ('tool', 'search')  — 下一步并行执行 tool 和 search
# 可能输出: ()                  — 图已执行完毕，没有下一步
```

**关键点：**
- 空元组 `()` 表示图已经运行结束（到达 END 节点）
- 元组中可能有多个节点名，表示它们会并行执行
- 这个值由 `prepare_next_tasks()` 函数计算得出

**前端类比：** 就像 React Router 中的"下一个要渲染的路由"——告诉你接下来会展示哪个页面。

**在实际应用中：**
- 判断图是否执行完毕：`if not snapshot.next: print("已完成")`
- 判断是否在等待人工审核：`if "human_review" in snapshot.next`
- 理解执行流程：追踪每一步的 next 变化

---

### 字段 3：`config` — 快照配置

**类型：** `RunnableConfig`

**作用：** 包含获取这个快照所需的配置信息，最重要的是 `thread_id` 和 `checkpoint_id`。

```python
snapshot = graph.get_state(config)

# config 的结构
print(snapshot.config)
# 输出:
# {
#     'configurable': {
#         'thread_id': '1',
#         'checkpoint_ns': '',
#         'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'
#     }
# }

# 提取关键信息
thread_id = snapshot.config["configurable"]["thread_id"]
checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
```

**关键点：**
- `thread_id`：标识一个对话线程（类似会话 ID）
- `checkpoint_id`：标识这个具体的快照（UUID6 格式，单调递增）
- `checkpoint_ns`：命名空间，用于子图隔离（通常为空字符串）
- 这个 config 可以直接传给 `get_state()` 来重新获取同一个快照

**日常生活类比：** 就像快递单号——通过 thread_id 找到是哪个订单，通过 checkpoint_id 找到是哪次物流更新。

**在实际应用中：**
- 保存快照引用：存储 config 以便后续回溯
- 时间旅行：用特定 checkpoint_id 回到过去的状态
- 分叉执行：基于某个 config 创建新的执行分支

---

### 字段 4：`metadata` — 快照元数据

**类型：** `CheckpointMetadata | None`

**作用：** 记录这个快照的来源、步骤编号、父级关系和运行 ID。

```python
snapshot = graph.get_state(config)

print(snapshot.metadata)
# 输出:
# {
#     'source': 'loop',
#     'step': 2,
#     'parents': {},
#     'run_id': 'a1b2c3d4-...'
# }
```

**`source` 字段的 4 种取值：**

| source 值 | 含义 | 触发场景 |
|-----------|------|----------|
| `"input"` | 来自用户输入 | `graph.invoke()` 或 `graph.stream()` 的初始输入 |
| `"loop"` | 来自图的循环执行 | 节点执行完毕后自动创建 |
| `"update"` | 来自手动更新 | 调用 `graph.update_state()` |
| `"fork"` | 从另一个快照复制 | 从历史快照分叉创建 |

**`step` 字段：**
- `-1`：首次输入（还没开始执行）
- `0`：第一个循环步骤
- `1, 2, 3...`：后续步骤

```python
# 通过 metadata 判断快照来源
if snapshot.metadata["source"] == "input":
    print("这是用户输入创建的快照")
elif snapshot.metadata["source"] == "loop":
    print(f"这是第 {snapshot.metadata['step']} 步执行后的快照")
elif snapshot.metadata["source"] == "update":
    print("这是手动修改状态创建的快照")
```

**在实际应用中：**
- 调试执行流程：通过 step 追踪执行到了第几步
- 审计日志：通过 source 和 run_id 追踪状态变更来源
- 筛选历史：在 `get_state_history()` 中按 source 过滤

---

### 字段 5：`created_at` — 创建时间戳

**类型：** `str | None`

**作用：** 记录这个快照的创建时间，ISO 8601 格式。

```python
snapshot = graph.get_state(config)

print(snapshot.created_at)
# 输出: '2026-02-27T10:30:45.123456+00:00'

# 解析时间戳
from datetime import datetime
created_time = datetime.fromisoformat(snapshot.created_at)
print(f"创建于: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
```

**关键点：**
- 来自底层 Checkpoint 的 `ts` 字段
- 可能为 None（如果 checkpoint 数据不完整）
- 时间戳是 UTC 时区

**在实际应用中：**
- 展示对话时间线
- 计算步骤间的执行耗时
- 按时间筛选历史快照

---

### 字段 6：`parent_config` — 父快照配置

**类型：** `RunnableConfig | None`

**作用：** 指向上一个快照的配置，形成快照链。通过它可以逐步回溯整个执行历史。

```python
snapshot = graph.get_state(config)

# 查看父快照
if snapshot.parent_config:
    parent_id = snapshot.parent_config["configurable"]["checkpoint_id"]
    print(f"父快照 ID: {parent_id}")

    # 获取父快照
    parent_snapshot = graph.get_state(snapshot.parent_config)
    print(f"父快照的状态: {parent_snapshot.values}")
else:
    print("这是第一个快照，没有父级")
```

**快照链示意：**

```
快照1 (input, step=-1)
  ↓ parent_config
快照2 (loop, step=0)
  ↓ parent_config
快照3 (loop, step=1)    ← 当前快照
  ↓ parent_config
快照4 (loop, step=2)    ← 最新快照
```

**前端类比：** 就像浏览器的历史记录——每个页面都知道"上一页"是什么，你可以一路点"返回"回到起点。

**在实际应用中：**
- 实现"撤销"功能：回到 parent_config 指向的状态
- 构建执行树：可视化整个执行路径
- 时间旅行：沿着 parent 链回溯到任意历史节点

---

### 字段 7：`tasks` — 当前步骤的任务

**类型：** `tuple[PregelTask, ...]`

**作用：** 描述在这个快照时刻，当前步骤要执行（或正在执行）的任务列表。

```python
snapshot = graph.get_state(config)

for task in snapshot.tasks:
    print(f"任务 ID: {task.id}")
    print(f"节点名: {task.name}")
    print(f"错误: {task.error}")
    print(f"中断: {task.interrupts}")
    print(f"子图状态: {task.state}")
    print("---")
```

**PregelTask 的关键属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 任务唯一 ID |
| `name` | `str` | 对应的节点名称 |
| `error` | `Exception \| None` | 执行错误（如果有） |
| `interrupts` | `tuple[Interrupt, ...]` | 任务产生的中断 |
| `state` | `RunnableConfig \| StateSnapshot \| None` | 子图状态 |

**关键点：**
- `tasks` 和 `next` 的关系：`next` 是节点名列表，`tasks` 是更详细的任务对象
- 如果任务有 `error`，说明执行出错了
- `state` 字段在 `subgraphs=True` 时包含子图的完整 StateSnapshot

**在实际应用中：**
- 错误诊断：检查 `task.error` 找到失败原因
- 中断处理：检查 `task.interrupts` 处理人工审核请求
- 子图调试：通过 `task.state` 深入子图内部状态

---

### 字段 8：`interrupts` — 待处理的中断

**类型：** `tuple[Interrupt, ...]`

**作用：** 汇总所有任务中待处理的中断事件。这是 Human-in-the-Loop 模式的核心。

```python
snapshot = graph.get_state(config)

# 检查是否有中断
if snapshot.interrupts:
    for interrupt in snapshot.interrupts:
        print(f"中断值: {interrupt.value}")
        print(f"是否已恢复: {interrupt.resumable}")

    # 处理中断后恢复执行
    graph.invoke(
        Command(resume="approved"),
        config=snapshot.config
    )
else:
    print("没有待处理的中断")
```

**关键点：**
- `interrupts` 是所有 tasks 中 interrupts 的汇总（便捷访问）
- 中断通常由 `interrupt()` 函数在节点内触发
- 处理中断后，通过 `Command(resume=...)` 恢复执行

**日常生活类比：** 就像工厂流水线上的"暂停按钮"——某个环节需要人工检查，整条线暂停，检查完毕后按"继续"恢复。

**在实际应用中：**
- Human-in-the-Loop：AI 生成内容后暂停，等待人工审核
- 工具确认：调用外部 API 前暂停，让用户确认参数
- 错误恢复：出错后暂停，让用户决定如何处理

---

## 完整使用示例

```python
"""
StateSnapshot 完整字段探索示例
"""
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== 1. 定义状态和图 =====
class ChatState(TypedDict):
    messages: Annotated[list[str], add]
    step_count: int

def chatbot(state: ChatState) -> dict:
    reply = f"收到第 {state['step_count'] + 1} 条消息"
    return {
        "messages": [reply],
        "step_count": state["step_count"] + 1
    }

graph = (
    StateGraph(ChatState)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile(checkpointer=InMemorySaver())
)

# ===== 2. 运行图 =====
config = {"configurable": {"thread_id": "demo-1"}}
graph.invoke({"messages": ["你好"], "step_count": 0}, config)

# ===== 3. 获取并探索 StateSnapshot =====
snapshot = graph.get_state(config)

print("=== StateSnapshot 8 字段 ===")
print(f"1. values:        {snapshot.values}")
print(f"2. next:          {snapshot.next}")
print(f"3. config:        {snapshot.config['configurable']}")
print(f"4. metadata:      {snapshot.metadata}")
print(f"5. created_at:    {snapshot.created_at}")
print(f"6. parent_config: {snapshot.parent_config is not None}")
print(f"7. tasks:         {len(snapshot.tasks)} 个任务")
print(f"8. interrupts:    {len(snapshot.interrupts)} 个中断")

# ===== 4. 判断执行状态 =====
if not snapshot.next:
    print("\n图已执行完毕")
elif snapshot.interrupts:
    print("\n图在等待人工处理")
else:
    print(f"\n图将继续执行: {snapshot.next}")
```

**预期输出：**
```
=== StateSnapshot 8 字段 ===
1. values:        {'messages': ['你好', '收到第 1 条消息'], 'step_count': 1}
2. next:          ()
3. config:        {'thread_id': 'demo-1', 'checkpoint_ns': '', 'checkpoint_id': '...'}
4. metadata:      {'source': 'loop', 'step': 1, 'parents': {}, 'run_id': '...'}
5. created_at:    2026-02-27T...
6. parent_config: True
7. tasks:         0 个任务
8. interrupts:    0 个中断

图已执行完毕
```

---

## StateSnapshot 字段速查表

| 字段 | 类型 | 一句话说明 | 常用场景 |
|------|------|-----------|----------|
| `values` | `dict[str, Any] \| Any` | 当前所有状态值 | 查看对话历史、检查中间结果 |
| `next` | `tuple[str, ...]` | 下一步要执行的节点 | 判断是否完成、查看执行流向 |
| `config` | `RunnableConfig` | 快照的唯一标识配置 | 时间旅行、保存引用 |
| `metadata` | `CheckpointMetadata \| None` | 来源、步骤、运行 ID | 调试、审计、筛选历史 |
| `created_at` | `str \| None` | ISO 8601 创建时间 | 时间线展示、耗时计算 |
| `parent_config` | `RunnableConfig \| None` | 父快照配置 | 回溯历史、撤销操作 |
| `tasks` | `tuple[PregelTask, ...]` | 当前步骤的任务详情 | 错误诊断、子图调试 |
| `interrupts` | `tuple[Interrupt, ...]` | 待处理的中断 | Human-in-the-Loop |

---

## 字段间的关系图

```
StateSnapshot
├── values ←────────── 从 Checkpoint.channel_values 恢复
├── next ←──────────── 由 prepare_next_tasks() 计算
├── config ←────────── 包含 thread_id + checkpoint_id
│   └── configurable
│       ├── thread_id ──────── 标识对话线程
│       ├── checkpoint_id ──── 标识具体快照
│       └── checkpoint_ns ──── 子图命名空间
├── metadata ←──────── 来自 CheckpointMetadata
│   ├── source ─────── input/loop/update/fork
│   ├── step ──────── -1, 0, 1, 2...
│   ├── parents ────── 父级 checkpoint 映射
│   └── run_id ─────── 运行标识
├── created_at ←────── 来自 Checkpoint.ts
├── parent_config ←─── 来自 CheckpointTuple.parent_config
│   └── 指向上一个快照的 config
├── tasks ←─────────── 由 prepare_next_tasks() 生成
│   └── PregelTask
│       ├── id
│       ├── name
│       ├── error
│       ├── interrupts
│       └── state (子图)
└── interrupts ←────── 汇总所有 tasks 的 interrupts
```

---

## 学习检查清单

- [ ] 能说出 StateSnapshot 的 8 个字段名称和类型
- [ ] 理解 `values` 和底层 `channel_values` 的关系
- [ ] 知道 `next` 为空元组意味着什么
- [ ] 能从 `config` 中提取 `thread_id` 和 `checkpoint_id`
- [ ] 理解 `metadata.source` 的 4 种取值
- [ ] 知道 `parent_config` 如何形成快照链
- [ ] 理解 `tasks` 和 `interrupts` 在 Human-in-the-Loop 中的作用

---

## 下一步学习

- **03_核心概念_2_Checkpoint存储格式.md** — 了解 StateSnapshot 背后的底层存储结构
- **03_核心概念_3_get_state获取快照.md** — 了解如何获取 StateSnapshot
