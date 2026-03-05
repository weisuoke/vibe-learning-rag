# stream_mode 调试模式

> LangGraph 提供 7 种流式模式，每种模式输出不同粒度的调试信息，是追踪图执行过程的核心手段

---

## 一句话定义

**`stream_mode` 决定了 `graph.stream()` 输出什么——从完整状态快照到逐 token 的 LLM 消息，6 种调试相关模式覆盖了从粗到细的所有观测粒度。**

---

## 为什么需要不同的 stream_mode？

调试多步骤工作流时，你面临的问题粒度不同：

- "最终状态对不对？" → 看每步完整状态（**values**）
- "哪个节点改了什么？" → 只看增量更新（**updates**）
- "任务什么时候开始/结束？" → 任务生命周期事件（**tasks**）
- "检查点保存了什么？" → 检查点快照（**checkpoints**）
- "给我所有调试信息！" → 全量调试事件（**debug**）
- "我想输出自定义中间数据" → 自定义流（**custom**）

不同的 stream_mode 就像不同焦距的镜头——广角看全貌，长焦看细节。

---

## 源码中的类型定义

```python
# [来源: langgraph/types.py:95-109]
StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
```

本文聚焦 6 种调试相关模式（`messages` 模式专注 LLM token 流，不在本文讨论范围）。

---

## 基础示例图（后续所有模式共用）

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[str], add]
    count: int

def node_a(state: State) -> dict:
    return {"messages": ["A处理完毕"], "count": state["count"] + 1}

def node_b(state: State) -> dict:
    return {"messages": ["B处理完毕"], "count": state["count"] + 10}

graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_node("node_b", node_b)
    .add_edge(START, "node_a")
    .add_edge("node_a", "node_b")
    .add_edge("node_b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "debug-demo"}}
initial_input = {"messages": ["开始"], "count": 0}
```

---

## 模式 1：stream_mode="values"

### 作用

每个节点执行完毕后，输出当前的**完整状态**。你看到的是状态的"全家福"，而不是"谁变了"。

### 适用场景

- 查看每步后的状态全貌
- 验证状态累积是否正确
- 初学者调试首选（最直观）

### 代码示例

```python
print("=== stream_mode='values' ===")
for chunk in graph.stream(initial_input, config, stream_mode="values"):
    print(chunk)
```

### 预期输出

```
{'messages': ['开始'], 'count': 0}
{'messages': ['开始', 'A处理完毕'], 'count': 1}
{'messages': ['开始', 'A处理完毕', 'B处理完毕'], 'count': 11}
```

### 输出解读

| 输出顺序 | 含义 |
|----------|------|
| 第 1 个 | 初始输入后的状态（step=-1 → step=0） |
| 第 2 个 | node_a 执行后的完整状态 |
| 第 3 个 | node_b 执行后的完整状态（最终状态） |

### 关键点

- 每个 chunk 是一个完整的状态字典，不是增量
- 如果状态很大（比如包含长对话历史），每个 chunk 都会很大
- 中断事件也会作为 values 输出

---

## 模式 2：stream_mode="updates"

### 作用

仅输出**节点名称**和该节点**返回的更新**。你看到的是"谁改了什么"，而不是"改完后整体是什么样"。

### 适用场景

- 追踪增量变化
- 定位"哪个节点修改了哪个字段"
- 状态很大时，避免重复输出完整状态

### 代码示例

```python
config2 = {"configurable": {"thread_id": "debug-demo-2"}}
for chunk in graph.stream(initial_input, config2, stream_mode="updates"):
    print(chunk)
    print("---")
```

### 预期输出

```
{'node_a': {'messages': ['A处理完毕'], 'count': 1}}
{'node_b': {'messages': ['B处理完毕'], 'count': 11}}
```

### 输出解读

每个 chunk 是 `{节点名: 节点返回值}` 的字典。

| 输出 | 含义 |
|------|------|
| `{'node_a': {...}}` | node_a 返回了 messages 和 count 的更新 |
| `{'node_b': {...}}` | node_b 返回了 messages 和 count 的更新 |

### values vs updates 对比

```
values 模式:  [完整状态] → [完整状态] → [完整状态]
updates 模式: [node_a的更新] → [node_b的更新]
```

前端类比：`values` 像 React 的 `state`（每次拿到完整状态），`updates` 像 Redux 的 `action`（只看发生了什么变化）。

---

## 模式 3：stream_mode="tasks"

### 作用

输出任务的**生命周期事件**——任务开始时输出 `TaskPayload`，任务完成时输出 `TaskResultPayload`。

### 适用场景

- 追踪任务执行时序
- 监控任务是否出错
- 检查任务的输入和输出

### 源码中的数据结构

```python
# [来源: langgraph/pregel/debug.py:31-43]
class TaskPayload(TypedDict):
    id: str          # 任务唯一 ID
    name: str        # 节点名称
    input: Any       # 任务输入
    triggers: list[str]  # 触发器列表

class TaskResultPayload(TypedDict):
    id: str          # 任务唯一 ID
    name: str        # 节点名称
    error: str | None    # 错误信息
    interrupts: list[dict]  # 中断列表
    result: dict[str, Any]  # 执行结果
```

### 代码示例

```python
config3 = {"configurable": {"thread_id": "debug-demo-3"}}
for chunk in graph.stream(initial_input, config3, stream_mode="tasks"):
    if "input" in chunk:
        print(f"[任务开始] {chunk['name']} | 输入: {chunk['input']}")
    elif "result" in chunk:
        print(f"[任务完成] {chunk['name']} | 结果: {chunk['result']}")
        if chunk.get("error"):
            print(f"  错误: {chunk['error']}")
```

### 预期输出

```
[任务开始] node_a | 输入: {'messages': ['开始'], 'count': 0}
[任务完成] node_a | 结果: {'messages': ['A处理完毕'], 'count': 1}
[任务开始] node_b | 输入: {'messages': ['开始', 'A处理完毕'], 'count': 1}
[任务完成] node_b | 结果: {'messages': ['B处理完毕'], 'count': 11}
```

### 关键点

- 任务开始事件包含 `input` 和 `triggers` 字段
- 任务完成事件包含 `result`、`error`、`interrupts` 字段
- 通过 `id` 字段可以匹配同一任务的开始和完成事件
- 源码中由 `map_debug_tasks()` 和 `map_debug_task_results()` 生成

---

## 模式 4：stream_mode="checkpoints"

### 作用

每次创建检查点时输出 `CheckpointPayload`，格式与 `get_state()` 返回的 `StateSnapshot` 类似。

### 适用场景

- 监控检查点创建过程
- 查看每个检查点的完整信息（包括 config、metadata、next）
- 调试持久化相关问题

### 源码中的数据结构

```python
# [来源: langgraph/pregel/debug.py:54-60]
class CheckpointPayload(TypedDict):
    config: RunnableConfig | None      # 检查点配置
    metadata: CheckpointMetadata       # 元数据（source, step, run_id）
    values: dict[str, Any]             # 当前状态值
    next: list[str]                    # 下一步要执行的节点
    parent_config: RunnableConfig | None  # 父检查点配置
    tasks: list[CheckpointTask]        # 任务列表
```

### 代码示例

```python
config4 = {"configurable": {"thread_id": "debug-demo-4"}}
print("=== stream_mode='checkpoints' ===")
for chunk in graph.stream(initial_input, config4, stream_mode="checkpoints"):
    step = chunk['metadata']['step']
    source = chunk['metadata']['source']
    print(f"[检查点] step={step}, source={source}, next={chunk['next']}")
    print(f"  values: {chunk['values']}")
```

### 预期输出

```
[检查点] step=-1, source=input, next=['node_a']
  values: {'messages': ['开始'], 'count': 0}
[检查点] step=0, source=loop, next=['node_b']
  values: {'messages': ['开始', 'A处理完毕'], 'count': 1}
[检查点] step=1, source=loop, next=[]
  values: {'messages': ['开始', 'A处理完毕', 'B处理完毕'], 'count': 11}
```

### 关键点

- 每个检查点都有 `metadata.step` 标识执行步骤
- `next` 为空列表表示图执行完毕
- `config` 中的 `checkpoint_id` 可用于时间旅行
- 源码中由 `map_debug_checkpoint()` 函数生成

---

## 模式 5：stream_mode="debug"

### 作用

最详细的调试模式。实际上是 `checkpoints` + `tasks` 的组合，每个事件额外包装了 `step`、`timestamp`、`type` 信息。

### 源码实现

```python
# [来源: langgraph/pregel/_loop.py:885-910] 简化后的核心逻辑
def _emit(self, mode, values, *args, **kwargs):
    debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
    if debug_remap:
        self.stream((self.checkpoint_ns, "debug", {
            "step": self.step - 1 if mode == "checkpoints" else self.step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "checkpoint" if mode == "checkpoints"
                    else "task_result" if "result" in v else "task",
            "payload": v,
        }))
```

### 三种事件类型

| type 值 | 对应 payload | 触发时机 |
|---------|-------------|----------|
| `"task"` | `TaskPayload` | 任务开始执行 |
| `"task_result"` | `TaskResultPayload` | 任务执行完毕 |
| `"checkpoint"` | `CheckpointPayload` | 检查点创建 |

### 代码示例

```python
config5 = {"configurable": {"thread_id": "debug-demo-5"}}
print("=== stream_mode='debug' ===")
for event in graph.stream(initial_input, config5, stream_mode="debug"):
    event_type = event["type"]
    step = event["step"]
    timestamp = event["timestamp"]
    payload = event["payload"]

    if event_type == "task":
        print(f"[step={step}] TASK_START: {payload['name']}")
        print(f"  时间: {timestamp}")
        print(f"  输入: {payload['input']}")

    elif event_type == "task_result":
        print(f"[step={step}] TASK_DONE: {payload['name']}")
        print(f"  结果: {payload['result']}")
        if payload.get("error"):
            print(f"  错误: {payload['error']}")

    elif event_type == "checkpoint":
        print(f"[step={step}] CHECKPOINT")
        print(f"  values: {payload['values']}")
        print(f"  next:   {payload['next']}")

    print("---")
```

### 预期输出（节选）

```
[step=-1] CHECKPOINT
  values: {'messages': ['开始'], 'count': 0}
  next:   ['node_a']
---
[step=0] TASK_START: node_a
  时间: 2026-02-27T10:30:45.123456+00:00
  输入: {'messages': ['开始'], 'count': 0}
---
[step=0] TASK_DONE: node_a
  结果: {'messages': ['A处理完毕'], 'count': 1}
---
[step=0] CHECKPOINT
  values: {'messages': ['开始', 'A处理完毕'], 'count': 1}
  next:   ['node_b']
---
... (node_b 的 TASK_START → TASK_DONE → CHECKPOINT 同理)
```

### 关键点

- debug 模式不是独立实现，而是复用 checkpoints 和 tasks 的事件
- 每个事件都有统一的 `step` + `timestamp` + `type` + `payload` 结构
- checkpoint 的 step 是 `self.step - 1`，task 的 step 是 `self.step`（源码 line 898-900）
- 这是信息量最大的模式，适合深度排查问题

---

## 模式 6：stream_mode="custom"

### 作用

允许你在节点内部通过 `get_stream_writer()` 发送**自定义数据**到流中。其他模式输出的是框架自动生成的事件，custom 模式输出的是你手动写入的任意数据。

### 适用场景

- 输出中间计算结果（如检索到的文档片段）
- 发送进度信息（"正在处理第 3/10 个文档"）
- 传递自定义调试数据

### 代码示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    query: str
    answer: str

def retrieve(state: State) -> dict:
    writer = get_stream_writer()
    writer({"stage": "retrieval", "status": "开始检索..."})
    docs = ["文档1: Python基础", "文档2: LangGraph入门"]
    writer({"stage": "retrieval", "found": len(docs), "docs": docs})
    return {"answer": f"基于 {len(docs)} 篇文档生成回答"}

def generate(state: State) -> dict:
    writer = get_stream_writer()
    writer({"stage": "generation", "status": "正在生成回答..."})
    return {"answer": state["answer"] + " -> 最终回答"}

graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .add_edge("generate", END)
    .compile(checkpointer=InMemorySaver())
)

config6 = {"configurable": {"thread_id": "custom-demo"}}
for chunk in graph.stream({"query": "什么是RAG?", "answer": ""}, config6, stream_mode="custom"):
    print(chunk)
# 输出:
# {'stage': 'retrieval', 'status': '开始检索...'}
# {'stage': 'retrieval', 'found': 2, 'docs': ['文档1: Python基础', '文档2: LangGraph入门']}
# {'stage': 'generation', 'status': '正在生成回答...'}
```

### 关键点

- 必须通过 `get_stream_writer()` 获取写入器
- 写入器接受任意 Python 对象作为参数
- 如果不使用 `stream_mode="custom"`，写入器是一个空操作（no-op）
- Python < 3.11 的异步场景下 `get_stream_writer()` 不可用（contextvar 限制）

---

## 多模式组合

`stream_mode` 可以传入列表，同时使用多种模式。此时每个 chunk 变成 `(mode, data)` 元组。

```python
config7 = {"configurable": {"thread_id": "multi-mode"}}
for mode, chunk in graph.stream(
    {"query": "什么是RAG?", "answer": ""},
    config7,
    stream_mode=["values", "updates", "custom"]
):
    print(f"[{mode}] {chunk}")
```

### 关键点

- 单模式时 chunk 直接是数据，多模式时 chunk 是 `(mode_name, data)` 元组
- 可以组合任意数量的模式
- 常用组合：`["values", "custom"]`（状态 + 自定义进度）、`["updates", "tasks"]`（增量 + 任务生命周期）

---

## 模式选择决策表

| 你想知道什么？ | 推荐模式 | 输出粒度 |
|---------------|----------|----------|
| 每步后的完整状态 | `values` | 粗 |
| 哪个节点改了什么 | `updates` | 中 |
| 任务的开始/完成/错误 | `tasks` | 中 |
| 检查点的完整快照 | `checkpoints` | 细 |
| 所有调试事件（任务+检查点） | `debug` | 最细 |
| 自定义的中间数据 | `custom` | 自定义 |
| 状态全貌 + 增量变化 | `["values", "updates"]` | 组合 |
| 快速排查问题 | `debug` | 最细 |
| 生产环境监控 | `["updates", "custom"]` | 组合 |

---

## 源码关键函数速查

以下函数位于 `debug.py`，是 `tasks`/`checkpoints`/`debug` 模式的事件生成核心：

| 函数 | 作用 | 产出类型 | 源码位置 |
|------|------|----------|----------|
| `map_debug_tasks()` | 生成任务开始事件 | `TaskPayload` | debug.py:66-77 |
| `map_debug_task_results()` | 生成任务完成事件 | `TaskResultPayload` | debug.py:112-134 |
| `map_debug_checkpoint()` | 生成检查点事件 | `CheckpointPayload` | debug.py:150-212 |

[来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/debug.py]

---

## 学习检查清单

- [ ] 能说出 6 种调试相关 stream_mode 的名称和用途
- [ ] 理解 `values` 和 `updates` 的区别（完整状态 vs 增量更新）
- [ ] 知道 `debug` 模式实际上是 `checkpoints` + `tasks` 的组合
- [ ] 能区分 `TaskPayload`（任务开始）和 `TaskResultPayload`（任务完成）
- [ ] 知道如何使用 `get_stream_writer()` 发送自定义数据
- [ ] 理解多模式组合时输出格式变为 `(mode, data)` 元组
- [ ] 能根据调试需求选择合适的 stream_mode

---

## 下一步学习

- **03_核心概念_2_get_state状态检查.md** — 了解如何用 `get_state()` 在任意时刻检查状态快照
- **03_核心概念_3_get_state_history历史追溯.md** — 了解如何遍历完整的状态历史
