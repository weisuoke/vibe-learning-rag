# Checkpoint 存储格式

> StateSnapshot 背后的底层存储结构——Checkpoint 是 LangGraph 持久化的真正数据格式

---

## 一句话定义

**Checkpoint 是 LangGraph 底层的状态存储格式（TypedDict），记录了 channel 值、版本号和节点执行信息，是 StateSnapshot 的"数据库记录"。**

---

## StateSnapshot vs Checkpoint：用户层 vs 存储层

在深入 Checkpoint 之前，先理解它和 StateSnapshot 的关系：

```
用户层（你直接接触的）          存储层（底层持久化的）
┌─────────────────┐          ┌──────────────────┐
│  StateSnapshot   │  ←转换←  │   Checkpoint      │
│  - values        │          │   - channel_values │
│  - next          │  计算得出 │   - channel_versions│
│  - config        │          │   - versions_seen  │
│  - metadata      │  ←来自←  │   CheckpointMetadata│
│  - created_at    │  ←来自←  │   - ts             │
│  - parent_config │  ←来自←  │   CheckpointTuple  │
│  - tasks         │  计算得出 │   - pending_writes │
│  - interrupts    │  计算得出 │                    │
└─────────────────┘          └──────────────────┘
```

**前端类比：** StateSnapshot 就像 React 组件的 `props`（用户友好的接口），Checkpoint 就像 Redux Store 的内部结构（底层存储格式）。你平时用 props，但理解 Store 能帮你调试复杂问题。

**日常生活类比：** StateSnapshot 是你看到的银行账户余额页面，Checkpoint 是银行数据库里的那条交易记录——包含更多技术细节（版本号、时间戳、关联 ID）。

---

## Checkpoint TypedDict 详解

### 源码定义

```python
# 来源：langgraph/checkpoint/base/__init__.py:61-92
class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""
    v: int                                    # 版本号（当前为 4）
    id: str                                   # 唯一且单调递增的 ID（uuid6）
    ts: str                                   # ISO 8601 时间戳
    channel_values: dict[str, Any]            # channel 名到序列化值的映射
    channel_versions: ChannelVersions         # channel 名到版本号的映射
    versions_seen: dict[str, ChannelVersions] # 节点 ID 到已见 channel 版本的映射
    updated_channels: list[str] | None        # 本次更新的 channel 列表
```

### 字段 1：`v` — 版本号

```python
checkpoint = {
    "v": 4,  # 当前 Checkpoint 格式版本
    # ...
}
```

- 当前版本为 **4**
- LangGraph 内部用它做格式迁移（旧版本的 checkpoint 会自动升级）
- 你通常不需要关心这个字段

### 字段 2：`id` — 唯一标识

```python
checkpoint = {
    "id": "1ef663ba-28fe-6528-8002-5a559208592c",
    # ...
}
```

- 使用 **UUID6** 格式，保证唯一且单调递增
- 单调递增意味着：后创建的 checkpoint，ID 一定比先创建的"大"
- 这个 ID 就是 StateSnapshot.config 中的 `checkpoint_id`

**为什么用 UUID6 而不是 UUID4？**
UUID4 是完全随机的，无法排序。UUID6 基于时间戳生成，天然有序——这对于"按时间顺序遍历历史"至关重要。

### 字段 3：`ts` — 时间戳

```python
checkpoint = {
    "ts": "2026-02-27T10:30:45.123456+00:00",
    # ...
}
```

- ISO 8601 格式
- 对应 StateSnapshot 的 `created_at` 字段
- UTC 时区

### 字段 4：`channel_values` — channel 值映射

```python
checkpoint = {
    "channel_values": {
        "messages": ["你好", "你好！有什么可以帮你的？"],
        "count": 1,
        "__start__": None,  # 内部 channel
    },
    # ...
}
```

**这是最核心的字段。** 它保存了图中每个 channel 的序列化值。

- key 是 channel 名称（对应你定义的 State 字段名）
- value 是该 channel 的序列化值
- 可能包含内部 channel（如 `__start__`），这些在 StateSnapshot.values 中会被过滤掉

**channel_values → StateSnapshot.values 的转换：**
```python
# 底层 checkpoint 的 channel_values
channel_values = {
    "messages": [序列化的消息列表],
    "count": 1,
    "__start__": None,       # 内部 channel
    "__end__": None,         # 内部 channel
}

# 转换为 StateSnapshot.values 时：
# 1. 过滤掉内部 channel（以 __ 开头的）
# 2. 反序列化值
# 结果：
values = {
    "messages": ["你好", "你好！有什么可以帮你的？"],
    "count": 1,
}
```

### 字段 5：`channel_versions` — channel 版本映射

```python
checkpoint = {
    "channel_versions": {
        "messages": 3,    # messages 已更新到版本 3
        "count": 2,       # count 已更新到版本 2
        "__start__": 1,   # __start__ 版本 1
    },
    # ...
}
```

**作用：** 记录每个 channel 当前的版本号。每次 channel 被更新，版本号递增。

**为什么需要版本号？** 这是 LangGraph 判断"哪些节点需要执行"的关键机制。

### 字段 6：`versions_seen` — 节点已见版本映射

```python
checkpoint = {
    "versions_seen": {
        "chatbot": {
            "messages": 2,   # chatbot 上次执行时，messages 版本是 2
            "count": 1,      # chatbot 上次执行时，count 版本是 1
        },
        "tool_node": {
            "messages": 1,
        },
    },
    # ...
}
```

**作用：** 记录每个节点上次执行时"看到"的 channel 版本。

**这是 LangGraph 调度的核心逻辑：**

```
判断节点是否需要执行：
  对于节点 N 订阅的每个 channel C：
    如果 channel_versions[C] > versions_seen[N][C]：
      说明 C 在 N 上次执行后被更新了
      → N 需要重新执行
```

**具体例子：**

```python
# 假设当前状态：
channel_versions = {"messages": 3, "count": 2}
versions_seen = {
    "chatbot": {"messages": 2, "count": 2},  # chatbot 看到的 messages 版本是 2
}

# 判断 chatbot 是否需要执行：
# messages: channel_versions["messages"](3) > versions_seen["chatbot"]["messages"](2)
# → messages 有新更新！chatbot 需要执行

# 如果 versions_seen 是：
versions_seen = {
    "chatbot": {"messages": 3, "count": 2},  # chatbot 已经看到最新版本
}
# → 没有新更新，chatbot 不需要执行
```

**前端类比：** 就像 React 的 `shouldComponentUpdate`——通过比较 props 的版本号决定是否重新渲染。`channel_versions` 是当前 props 版本，`versions_seen` 是组件上次渲染时的 props 版本。

### 字段 7：`updated_channels` — 本次更新的 channel 列表

```python
checkpoint = {
    "updated_channels": ["messages", "count"],
    # ...
}
```

- 记录这次 checkpoint 创建时哪些 channel 被更新了
- 可能为 `None`（旧版本 checkpoint）
- 主要用于优化：只序列化变更的 channel

---

## CheckpointMetadata 详解

```python
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]
    step: int
    parents: dict[str, str]
    run_id: str
```

### `source` — 快照来源

| source 值 | 含义 | 触发场景 |
|-----------|------|----------|
| `"input"` | 来自用户输入 | `graph.invoke()` 的初始输入 |
| `"loop"` | 来自图的循环执行 | 节点执行完毕后自动创建 |
| `"update"` | 来自手动更新 | 调用 `graph.update_state()` |
| `"fork"` | 从另一个快照复制 | 从历史快照分叉创建 |

### `step` — 执行步骤

`-1` 为初始输入，`0` 为第一个节点执行完毕，依次递增。

### `parents` 和 `run_id`

- `parents`：checkpoint 命名空间到父 ID 的映射（无子图时为 `{}`）
- `run_id`：同一次 `invoke()` 调用中所有 checkpoint 共享的运行 ID

---

## CheckpointTuple — 存储层的完整记录

```python
class CheckpointTuple(NamedTuple):
    config: RunnableConfig                        # 配置（含 thread_id, checkpoint_id）
    checkpoint: Checkpoint                        # Checkpoint 数据
    metadata: CheckpointMetadata                  # 元数据
    parent_config: RunnableConfig | None = None   # 父级配置
    pending_writes: list[PendingWrite] | None = None  # 待写入数据
```

**CheckpointTuple 是 Checkpointer 存取的基本单位。** 它把 Checkpoint、元数据、配置打包在一起。

```
Checkpointer.get_tuple(config) → CheckpointTuple
                                    ├── config          → StateSnapshot.config
                                    ├── checkpoint      → 恢复 channels → StateSnapshot.values
                                    │   ├── channel_values
                                    │   ├── channel_versions  → 计算 next/tasks
                                    │   └── versions_seen     → 计算 next/tasks
                                    ├── metadata        → StateSnapshot.metadata
                                    ├── parent_config   → StateSnapshot.parent_config
                                    └── pending_writes  → 应用后影响 values/tasks
```

### `pending_writes` — 待写入数据

类型为 `tuple[str, str, Any]`（task_id, channel_name, value）。当节点执行完但还没创建新 checkpoint 时，写入暂存在这里。`_prepare_state_snapshot()` 会把它应用到 channel 值上，保证 StateSnapshot.values 反映最新状态。

---

## Checkpoint 工具函数

LangGraph 提供了 4 个核心工具函数来操作 Checkpoint：

### `empty_checkpoint()` — 创建空 checkpoint

```python
from langgraph.pregel._checkpoint import empty_checkpoint

checkpoint = empty_checkpoint()
# 返回:
# {
#     "v": 4,
#     "id": "00000000-0000-0000-0000-000000000000",
#     "ts": "2026-02-27T...",
#     "channel_values": {},
#     "channel_versions": {},
#     "versions_seen": {},
#     "updated_channels": None,
# }
```

用于初始化——图第一次运行时，从空 checkpoint 开始。

### `create_checkpoint()` — 从现有 checkpoint 创建新的

为每个 channel 调用 `channel.checkpoint()` 序列化当前值，生成新 UUID6 ID，更新时间戳和版本号。

### `channels_from_checkpoint()` — 从 checkpoint 恢复 channels

`create_checkpoint()` 的逆操作：为每个 channel 调用 `channel.from_checkpoint()` 反序列化，恢复完整状态。

### `copy_checkpoint()` — 深拷贝 checkpoint

创建 checkpoint 的深拷贝，在分叉（fork）操作中使用，避免修改原始数据。

---

## channel_versions 与 versions_seen 的协作机制

这是理解 LangGraph 调度的关键。核心逻辑：

```
判断节点 N 是否需要执行：
  如果 channel_versions[C] > versions_seen[N][C] → N 需要执行
```

**完整例子（图结构：START → node_a → node_b → END）：**

```python
# Checkpoint 0 (input, step=-1): invoke({"x": 1, "y": 0})
# channel_versions: {"x": 1, "y": 1}
# versions_seen: {}
# → node_a 需要执行（x:1 > 0）

# Checkpoint 1 (loop, step=0): node_a 返回 {"x": 10}
# channel_versions: {"x": 2, "y": 1}
# versions_seen: {"node_a": {"x": 1, "y": 1}}
# → node_b 需要执行（x:2 > 0）

# Checkpoint 2 (loop, step=1): node_b 返回 {"y": 100}
# channel_versions: {"x": 2, "y": 2}
# versions_seen: {"node_a": {"x": 1, "y": 1}, "node_b": {"x": 2, "y": 1}}
# → 没有更多节点 → END
```

**版本变化时间线：**

```
channel_versions:  x: 1→2→2  |  y: 1→1→2
versions_seen:     node_a: {}→{x:1,y:1}  |  node_b: {}→{}→{x:2,y:1}
事件:              input → node_a完成 → node_b完成
```

---

## InMemorySaver 的存储结构

InMemorySaver 使用**三层索引**组织 Checkpoint：`thread_id → checkpoint_ns → checkpoint_id`

```python
# 内部存储示意
storage = {
    "thread-1": {
        "": {  # 默认命名空间
            "ckpt-001": (checkpoint_data, metadata, None),
            "ckpt-002": (checkpoint_data, metadata, parent_config),
        }
    },
}

# 这就是 config 中需要这三个字段的原因
config = {
    "configurable": {
        "thread_id": "thread-1",       # 第一层
        "checkpoint_ns": "",           # 第二层
        "checkpoint_id": "ckpt-002",   # 第三层
    }
}
```

---

## 完整使用示例

```python
"""
Checkpoint 存储格式探索示例
通过 get_state_history 观察 Checkpoint 的变化
"""
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== 1. 定义状态和图 =====
class CounterState(TypedDict):
    values: Annotated[list[str], add]
    counter: int

def step_a(state: CounterState) -> dict:
    return {"values": ["a"], "counter": state["counter"] + 1}

def step_b(state: CounterState) -> dict:
    return {"values": ["b"], "counter": state["counter"] + 10}

graph = (
    StateGraph(CounterState)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .add_edge("step_b", END)
    .compile(checkpointer=InMemorySaver())
)

# ===== 2. 运行图 =====
config = {"configurable": {"thread_id": "demo"}}
graph.invoke({"values": ["start"], "counter": 0}, config)

# ===== 3. 遍历历史，观察 metadata 变化 =====
print("=== Checkpoint 历史（从新到旧）===\n")
for snapshot in graph.get_state_history(config):
    meta = snapshot.metadata
    print(f"Step {meta['step']:>2} | Source: {meta['source']:<6} | "
          f"Next: {snapshot.next or '(END)'}")
    print(f"         Values: {snapshot.values}")
    print(f"         Checkpoint ID: {snapshot.config['configurable']['checkpoint_id'][:12]}...")
    print()
```

**预期输出：**
```
=== Checkpoint 历史（从新到旧）===

Step  2 | Source: loop   | Next: (END)
         Values: {'values': ['start', 'a', 'b'], 'counter': 11}
         Checkpoint ID: 1ef663ba-28...

Step  1 | Source: loop   | Next: ('step_b',)
         Values: {'values': ['start', 'a'], 'counter': 1}
         Checkpoint ID: 1ef663ba-27...

Step  0 | Source: loop   | Next: ('step_a',)
         Values: {'values': ['start'], 'counter': 0}
         Checkpoint ID: 1ef663ba-26...

Step -1 | Source: input  | Next: ('__start__',)
         Values: {'values': ['start'], 'counter': 0}
         Checkpoint ID: 1ef663ba-25...
```

---

## StateSnapshot 与 Checkpoint 的映射关系总结

| StateSnapshot 字段 | 数据来源 | 转换方式 |
|-------------------|----------|----------|
| `values` | `Checkpoint.channel_values` | `channels_from_checkpoint()` 反序列化 + 过滤内部 channel |
| `next` | `Checkpoint.channel_versions` + `versions_seen` | `prepare_next_tasks()` 计算 |
| `config` | `CheckpointTuple.config` | 直接传递 |
| `metadata` | `CheckpointTuple.metadata` | 直接传递 |
| `created_at` | `Checkpoint.ts` | 直接传递 |
| `parent_config` | `CheckpointTuple.parent_config` | 直接传递 |
| `tasks` | `prepare_next_tasks()` 结果 | 结合 `pending_writes` 计算 |
| `interrupts` | `tasks` 中的 interrupts | 汇总提取 |

---

## 学习检查清单

- [ ] 能说出 Checkpoint 的 7 个字段
- [ ] 理解 `channel_versions` 和 `versions_seen` 如何决定节点执行
- [ ] 知道 CheckpointTuple 包含哪些内容
- [ ] 理解 `metadata.source` 的 4 种取值及触发场景
- [ ] 知道 4 个 Checkpoint 工具函数的作用
- [ ] 能解释 StateSnapshot 和 Checkpoint 的映射关系

---

## 下一步学习

- **03_核心概念_3_get_state获取快照.md** — 了解 Checkpoint 如何被转换为 StateSnapshot
- **03_核心概念_4_get_state_history历史遍历.md** — 了解如何遍历 Checkpoint 历史
