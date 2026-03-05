# get_state() 获取快照

> 从 Checkpointer 中读取 Checkpoint 并转换为用户友好的 StateSnapshot 的完整流程

---

## 一句话定义

**`get_state()` 是 LangGraph 图的核心方法，通过 config 中的 thread_id/checkpoint_id 从 Checkpointer 读取底层 Checkpoint，经过 `_prepare_state_snapshot()` 转换为用户友好的 StateSnapshot。**

---

## 为什么需要 get_state()？

**日常生活类比：** 你去银行查余额。你只需要提供账号（config），银行从数据库（Checkpointer）里查出原始记录（Checkpoint），然后把它转换成你能看懂的余额页面（StateSnapshot）。你不需要知道数据库的表结构，`get_state()` 帮你搞定了这一切。

**前端类比：** 就像 Redux 的 `useSelector()`——你传入一个选择器（config），它从 Store（Checkpointer）中提取数据，返回组件能直接使用的格式（StateSnapshot）。

---

## 方法签名

```python
def get_state(
    self,
    config: RunnableConfig,
    *,
    subgraphs: bool = False,
) -> StateSnapshot:
    """Get the current state of the graph."""
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config` | `RunnableConfig` | 是 | 包含 `thread_id`，可选 `checkpoint_id` |
| `subgraphs` | `bool` | 否 | 是否递归获取子图状态，默认 `False` |

**返回值：** `StateSnapshot` — 包含 8 个字段的命名元组

---

## 前置条件：必须配置 Checkpointer

`get_state()` 依赖 Checkpointer 来读取持久化的状态。如果没有配置，会抛出错误。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ❌ 没有 checkpointer — get_state() 会报错
graph_no_ckpt = (
    StateGraph(MyState)
    .add_node("node", my_func)
    .add_edge(START, "node")
    .add_edge("node", END)
    .compile()  # 没有传 checkpointer
)

try:
    graph_no_ckpt.get_state({"configurable": {"thread_id": "1"}})
except ValueError as e:
    print(f"错误: {e}")
    # 错误: No checkpointer set

# ✅ 配置了 checkpointer — 正常工作
graph = (
    StateGraph(MyState)
    .add_node("node", my_func)
    .add_edge(START, "node")
    .add_edge("node", END)
    .compile(checkpointer=InMemorySaver())  # 传入 checkpointer
)

snapshot = graph.get_state({"configurable": {"thread_id": "1"}})
# 正常返回 StateSnapshot
```

---

## config 参数详解

### 最小配置：只传 thread_id

```python
# 获取线程 "1" 的最新状态
config = {"configurable": {"thread_id": "1"}}
snapshot = graph.get_state(config)
```

只传 `thread_id` 时，返回该线程的**最新**快照。

### 精确定位：传 thread_id + checkpoint_id

```python
# 获取特定 checkpoint 的状态
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"
    }
}
snapshot = graph.get_state(config)
```

同时传 `checkpoint_id` 时，返回该**特定时刻**的快照。这是时间旅行的基础。

### 子图命名空间：checkpoint_ns

```python
# 获取子图的状态（高级用法）
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_ns": "subgraph_name",
    }
}
```

`checkpoint_ns` 用于隔离子图的 checkpoint，通常由框架自动管理。

---

## 核心流程：从 config 到 StateSnapshot

```
get_state(config)
│
├── 1. 获取 checkpointer
│   └── 检查 self.checkpointer 是否存在
│       └── 不存在 → raise ValueError("No checkpointer set")
│
├── 2. 处理子图命名空间（如果 config 中有 checkpoint_ns）
│   └── 找到对应的子图实例
│
├── 3. 调用 checkpointer.get_tuple(config)
│   └── 返回 CheckpointTuple 或 None
│       ├── CheckpointTuple 包含:
│       │   ├── config (含 checkpoint_id)
│       │   ├── checkpoint (Checkpoint 数据)
│       │   ├── metadata (CheckpointMetadata)
│       │   ├── parent_config
│       │   └── pending_writes
│       └── None → 返回空的 StateSnapshot
│
├── 4. 调用 _prepare_state_snapshot(config, saved)
│   └── 将 CheckpointTuple 转换为 StateSnapshot
│       （详见下方 _prepare_state_snapshot 详解）
│
└── 5. 返回 StateSnapshot
```

---

## _prepare_state_snapshot() 内部工作原理

这是 `get_state()` 的核心转换逻辑，把底层的 CheckpointTuple 变成用户友好的 StateSnapshot。

### 源码流程（简化版）

```python
# 来源：langgraph/pregel/main.py:996-1113（简化）
def _prepare_state_snapshot(self, config, saved, recurse=False):
    # 步骤 1：空数据 → 返回空 StateSnapshot
    if not saved:
        return StateSnapshot(values={}, next=(), config=config, ...)

    # 步骤 2：格式迁移（旧版 checkpoint 自动升级到 v4）
    checkpoint = migrate_checkpoint(saved.checkpoint)

    # 步骤 3：从 checkpoint 恢复 channels
    channels = channels_from_checkpoint(checkpoint, self.channel_specs)

    # 步骤 4：应用 pending writes
    if saved.pending_writes:
        apply_writes(channels, saved.pending_writes)

    # 步骤 5：准备下一步任务
    tasks = prepare_next_tasks(checkpoint, channels, self.nodes, self.edges)

    # 步骤 6：处理子图状态（recurse=True 时递归获取）
    # 步骤 7：提取 values
    values = read_channels(channels, self.output_channels)

    # 步骤 8：汇总 interrupts
    interrupts = tuple(i for t in tasks for i in t.interrupts)

    # 步骤 9：组装返回
    return StateSnapshot(
        values=values,
        next=tuple(t.name for t in tasks),
        config=saved.config,
        metadata=saved.metadata,
        created_at=checkpoint["ts"],
        parent_config=saved.parent_config,
        tasks=tuple(tasks),
        interrupts=interrupts,
    )
```

### 9 个步骤的可视化

```
CheckpointTuple
├── checkpoint ─────────────────────────────────────┐
│   ├── channel_values ──→ channels_from_checkpoint() ──→ channels
│   ├── channel_versions ──→ prepare_next_tasks() ──→ tasks ──→ next
│   ├── versions_seen ─────→ prepare_next_tasks()        │
│   └── ts ──────────────────────────────────────────→ created_at
├── metadata ────────────────────────────────────────→ metadata
├── config ──────────────────────────────────────────→ config
├── parent_config ───────────────────────────────────→ parent_config
└── pending_writes ──→ apply_writes(channels) ──→ 更新 values
                                                      │
                                                      ↓
                                                 StateSnapshot
```

---

## subgraphs 参数的作用

当图包含子图时，`subgraphs=True` 会递归获取子图的完整状态。

```python
# ===== 不递归（默认）=====
snapshot = graph.get_state(config, subgraphs=False)

for task in snapshot.tasks:
    print(task.name, type(task.state))
    # 输出: subgraph_node <class 'dict'>
    # task.state 只是一个 config 字典

# ===== 递归获取子图状态 =====
snapshot = graph.get_state(config, subgraphs=True)

for task in snapshot.tasks:
    print(task.name, type(task.state))
    # 输出: subgraph_node <class 'StateSnapshot'>
    # task.state 是完整的 StateSnapshot！

    if isinstance(task.state, StateSnapshot):
        print(f"  子图状态: {task.state.values}")
        print(f"  子图下一步: {task.state.next}")
```

**什么时候用 `subgraphs=True`？**
- 调试子图内部状态
- 需要查看子图的执行进度
- 构建完整的状态树可视化

**什么时候不需要？**
- 只关心主图状态
- 性能敏感场景（递归获取有额外开销）

---

## 异步版本：aget_state()

```python
# 在 FastAPI 等异步框架中使用
snapshot = await graph.aget_state(config)
snapshot = await graph.aget_state(config, subgraphs=True)
```

适用于异步框架（FastAPI）、并发获取多线程状态、异步 Checkpointer 后端。

---

## 错误处理

### 错误 1：没有配置 Checkpointer

```python
try:
    snapshot = graph.get_state(config)
except ValueError as e:
    # "No checkpointer set"
    print("需要在 compile() 时传入 checkpointer")
```

### 错误 2：线程不存在

```python
config = {"configurable": {"thread_id": "不存在的线程"}}
snapshot = graph.get_state(config)

# 不会报错！返回空的 StateSnapshot
print(snapshot.values)   # {}
print(snapshot.next)     # ()
print(snapshot.metadata) # None
```

这是一个重要的设计决策：查询不存在的线程不会抛异常，而是返回空快照。你需要通过检查 `metadata` 是否为 `None` 来判断。

### 错误 3：checkpoint_id 不存在

```python
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "不存在的ID"
    }
}
snapshot = graph.get_state(config)
# 同样返回空的 StateSnapshot
```

---

## 完整使用示例

```python
"""
get_state() 完整使用示例
演示：获取快照、检查状态、时间旅行
"""
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== 1. 定义状态和图 =====
class WorkflowState(TypedDict):
    messages: Annotated[list[str], add]
    step: int

def analyst(state: WorkflowState) -> dict:
    return {
        "messages": [f"分析完成（步骤 {state['step']})"],
        "step": state["step"] + 1,
    }

def reviewer(state: WorkflowState) -> dict:
    return {
        "messages": [f"审核通过（步骤 {state['step']})"],
        "step": state["step"] + 1,
    }

checkpointer = InMemorySaver()
graph = (
    StateGraph(WorkflowState)
    .add_node("analyst", analyst)
    .add_node("reviewer", reviewer)
    .add_edge(START, "analyst")
    .add_edge("analyst", "reviewer")
    .add_edge("reviewer", END)
    .compile(checkpointer=checkpointer)
)

# ===== 2. 运行图 =====
config = {"configurable": {"thread_id": "workflow-1"}}
result = graph.invoke(
    {"messages": ["开始工作流"], "step": 0},
    config
)
print("=== 运行结果 ===")
print(f"最终消息: {result['messages']}")
print(f"最终步骤: {result['step']}")

# ===== 3. 获取最新状态 =====
print("\n=== 获取最新状态 ===")
latest = graph.get_state(config)
print(f"Values: {latest.values}")
print(f"Next: {latest.next}")
print(f"Source: {latest.metadata['source']}")
print(f"Step: {latest.metadata['step']}")

# ===== 4. 判断执行状态 =====
print("\n=== 执行状态判断 ===")
if not latest.next:
    print("工作流已完成")
elif latest.interrupts:
    print("工作流等待人工处理")
else:
    print(f"工作流将继续: {latest.next}")

# ===== 5. 时间旅行：获取历史快照 =====
print("\n=== 历史快照 ===")
history = list(graph.get_state_history(config))
for i, snap in enumerate(history):
    print(f"[{i}] Step {snap.metadata['step']:>2} | "
          f"Source: {snap.metadata['source']:<6} | "
          f"Next: {snap.next or '(END)'}")

# ===== 6. 回到过去的状态 =====
print("\n=== 时间旅行 ===")
# 获取 analyst 执行前的快照（step=0）
past_snapshots = [s for s in history if s.metadata["step"] == 0]
if past_snapshots:
    past = past_snapshots[0]
    print(f"回到 Step 0:")
    print(f"  Values: {past.values}")
    print(f"  Next: {past.next}")

    # 用过去的 config 重新获取（验证）
    past_again = graph.get_state(past.config)
    print(f"  重新获取验证: {past_again.values == past.values}")

# ===== 7. 检查不存在的线程 =====
print("\n=== 不存在的线程 ===")
empty = graph.get_state({"configurable": {"thread_id": "ghost"}})
print(f"Values: {empty.values}")
print(f"Metadata: {empty.metadata}")
print(f"是空快照: {empty.metadata is None}")
```

**预期输出：**
```
=== 运行结果 ===
最终消息: ['开始工作流', '分析完成（步骤 0)', '审核通过（步骤 1)']
最终步骤: 2

=== 获取最新状态 ===
Values: {'messages': ['开始工作流', '分析完成（步骤 0)', '审核通过（步骤 1)'], 'step': 2}
Next: ()
Source: loop
Step: 2

=== 执行状态判断 ===
工作流已完成

=== 历史快照 ===
[0] Step  2 | Source: loop   | Next: (END)
[1] Step  1 | Source: loop   | Next: ('reviewer',)
[2] Step  0 | Source: loop   | Next: ('analyst',)
[3] Step -1 | Source: input  | Next: ('__start__',)

=== 时间旅行 ===
回到 Step 0:
  Values: {'messages': ['开始工作流'], 'step': 0}
  Next: ('analyst',)
  重新获取验证: True

=== 不存在的线程 ===
Values: {}
Metadata: None
是空快照: True
```

---

## 数据流全景图

```
graph.get_state(config)
  → checkpointer.get_tuple(config) → CheckpointTuple
  → _prepare_state_snapshot():
      checkpoint.channel_values → channels_from_checkpoint() → channels
        + pending_writes → apply_writes() → read_channels() → values
      checkpoint.channel_versions + versions_seen
        → prepare_next_tasks() → tasks → next, interrupts
      checkpoint.ts → created_at
      saved.metadata → metadata
      saved.config → config
      saved.parent_config → parent_config
  → StateSnapshot (返回给用户)
```

---

## 常见使用模式速览

```python
# 模式 1：判断是否完成
snapshot = graph.get_state(config)
if not snapshot.next:
    print("执行完成")

# 模式 2：Human-in-the-Loop
if snapshot.interrupts:
    from langgraph.types import Command
    graph.invoke(Command(resume="approved"), config=snapshot.config)

# 模式 3：状态对比调试
history = list(graph.get_state_history(config))
before, after = history[1], history[0]
for key in after.values:
    if after.values[key] != before.values.get(key):
        print(f"{key}: {before.values.get(key)} → {after.values[key]}")
```

---

## 学习检查清单

- [ ] 能写出 `get_state()` 的基本调用方式
- [ ] 知道 config 中 `thread_id` 和 `checkpoint_id` 的区别
- [ ] 理解没有 checkpointer 时会发生什么
- [ ] 理解查询不存在的线程返回什么
- [ ] 能描述 `_prepare_state_snapshot()` 的核心步骤
- [ ] 知道 `subgraphs=True` 的作用和使用场景
- [ ] 能使用 `get_state()` 实现基本的状态检查

---

## 下一步学习

- **03_核心概念_4_get_state_history历史遍历.md** — 了解如何遍历完整的快照历史
- **03_核心概念_5_update_state状态修改.md** — 了解如何手动修改状态
