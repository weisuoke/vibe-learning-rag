# get_state 状态检查

> get_state() 是 LangGraph 中最直接的运行时状态检查方法——一个调用，拿到图执行的完整快照

---

## 一句话定义

**get_state(config) 根据 thread_id 返回 StateSnapshot 对象，包含当前状态值、下一步节点、任务详情、错误和中断信息，是调试 LangGraph 工作流的第一工具。**

---

## 为什么需要 get_state？

调试多步骤 AI 工作流时，你需要知道：当前状态里存了什么？图停在哪个节点？有没有报错或中断？`get_state()` 一个调用全部告诉你。

**前端类比：** 就像 React DevTools 的 "Components" 面板——点一下组件，立刻看到它的 state 和 props。

**日常生活类比：** 就像快递查询——输入单号（thread_id），立刻看到包裹在哪、下一站去哪、有没有异常。

---

## 基本用法

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    messages: list[str]
    count: int

# 构建图（省略节点定义）
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 运行图
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": ["你好"], "count": 0}, config)

# 检查状态——就这一行
snapshot = graph.get_state(config)
```

**关键前提：** 图必须配置了 checkpointer，否则 `get_state()` 无法工作。

---

## StateSnapshot 数据结构详解

`get_state()` 返回的是一个 `StateSnapshot` 对象。来自源码 `types.py`：

```python
# 来源: langgraph/types.py
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any        # 当前通道值
    next: tuple[str, ...]               # 下一步要执行的节点
    config: RunnableConfig              # 获取快照的配置
    metadata: CheckpointMetadata | None # 检查点元数据
    created_at: str | None              # 创建时间戳
    parent_config: RunnableConfig | None # 父快照配置
    tasks: tuple[PregelTask, ...]       # 任务元组
    interrupts: tuple[Interrupt, ...]   # 中断元组
```

8 个字段，每个都有明确的调试用途。下面逐一拆解。

---

## 字段 1：values — 当前状态值

```python
snapshot = graph.get_state(config)
print(snapshot.values)
# {'messages': ['你好', '你好！有什么可以帮你的？'], 'count': 1}
```

**最常用的字段。** 返回图当前的完整状态——所有你定义的 State 字段的当前值。

```python
# 调试场景：检查消息是否正确累积
messages = snapshot.values.get("messages", [])
print(f"消息数量: {len(messages)}")
for i, msg in enumerate(messages):
    print(f"  [{i}] {msg}")
```

**注意：** `values` 已过滤掉内部 channel（如 `__start__`），你看到的都是你定义的字段。

---

## 字段 2：next — 下一步节点

```python
print(snapshot.next)
# ('chatbot',)  ← 下一步要执行 chatbot 节点
# ()            ← 空元组 = 图已执行完毕
```

| next 的值 | 含义 |
|-----------|------|
| `()` 空元组 | 图已执行完毕（到达 END） |
| `('node_a',)` | 下一步执行 node_a |
| `('node_a', 'node_b')` | 下一步并行执行 node_a 和 node_b |

```python
if not snapshot.next:
    print("图已执行完毕")
else:
    print(f"图还在运行，下一步: {snapshot.next}")
```

---

## 字段 3：config — 快照配置

```python
print(snapshot.config)
# {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-...'}}
```

主要用途：获取 `checkpoint_id` 用于历史回溯，确认 `thread_id` 是否正确。

```python
checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
print(f"当前 checkpoint: {checkpoint_id}")
```

---

## 字段 4：metadata — 检查点元数据

```python
print(snapshot.metadata)
# {'source': 'loop', 'step': 2, 'writes': {...}, 'parents': {}, 'run_id': 'abc123-...'}
```

**最有用的两个字段：**

| source 值 | 含义 | 触发场景 |
|-----------|------|---------|
| `"input"` | 用户输入 | `graph.invoke()` 刚被调用 |
| `"loop"` | 节点执行完毕 | 节点跑完后自动创建 |
| `"update"` | 手动更新 | 调用了 `graph.update_state()` |
| `"fork"` | 从历史分叉 | 从历史快照创建新分支 |

`step` 编号：`-1` 为初始输入，`0` 为第一个节点完成，依次递增。

```python
meta = snapshot.metadata
print(f"来源: {meta['source']}, 步骤: {meta['step']}")
```

---

## 字段 5：created_at — 创建时间戳

```python
snapshot = graph.get_state(config)
print(snapshot.created_at)
# '2026-02-27T10:30:45.123456+00:00'
```

ISO 8601 格式。调试时可以计算两个快照之间的时间差来定位性能瓶颈：

```python
from datetime import datetime

snapshots = list(graph.get_state_history(config))
if len(snapshots) >= 2:
    t1 = datetime.fromisoformat(snapshots[1].created_at)
    t0 = datetime.fromisoformat(snapshots[0].created_at)
    print(f"最后一步耗时: {(t0 - t1).total_seconds():.3f}s")
```

---

## 字段 6：parent_config — 父快照配置

```python
print(snapshot.parent_config)
# {'configurable': {'thread_id': '1', 'checkpoint_id': '1ef663ba-27...'}}
# 或者 None（第一个快照）
```

parent_config 指向上一个快照，可以沿链条回溯整个执行历史：

```python
snapshot = graph.get_state(config)
while snapshot.parent_config:
    print(f"Step {snapshot.metadata['step']}: {snapshot.values}")
    snapshot = graph.get_state(snapshot.parent_config)
print(f"Step {snapshot.metadata['step']}: {snapshot.values} (初始)")
```

---

## 字段 7：tasks — 任务详情

这是调试错误和中断的核心字段。每个 task 是一个 `PregelTask` 对象：

```python
# 来源: langgraph/types.py
class PregelTask(NamedTuple):
    id: str                                    # 任务唯一 ID
    name: str                                  # 节点名称
    path: tuple[str | int | tuple, ...]        # 任务路径
    error: Exception | None = None             # 执行错误（如有）
    interrupts: tuple[Interrupt, ...] = ()     # 中断信息
    state: None | RunnableConfig | StateSnapshot = None  # 子图状态
    result: Any | None = None                  # 任务结果
```

### 检查任务错误

```python
snapshot = graph.get_state(config)
for task in snapshot.tasks:
    if task.error:
        print(f"节点 '{task.name}' 出错了!")
        print(f"  错误类型: {type(task.error).__name__}")
        print(f"  错误信息: {task.error}")
    else:
        print(f"节点 '{task.name}' 正常，等待执行")
```

### 检查中断状态

```python
snapshot = graph.get_state(config)
for task in snapshot.tasks:
    if task.interrupts:
        print(f"节点 '{task.name}' 被中断了!")
        for interrupt in task.interrupts:
            print(f"  中断值: {interrupt.value}")
```

**tasks 字段的含义取决于 next：**
- `next` 不为空时：tasks 列出即将执行的任务
- `next` 为空时：tasks 为空（图已完成）
- 有错误时：tasks 中的 error 字段记录异常

---

## 字段 8：interrupts — 中断汇总

```python
print(snapshot.interrupts)
# (Interrupt(value='请确认是否继续', resumable=True, ns=('node_a:xxx',)),)
```

所有 tasks 中 interrupts 的汇总。快速判断有没有中断：

```python
if snapshot.interrupts:
    print(f"有 {len(snapshot.interrupts)} 个中断需要处理")
```

---

## 指定 checkpoint_id 检查历史状态

默认 `get_state()` 返回最新快照。通过 `checkpoint_id` 可查看任意历史时刻：

```python
config_history = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-26fe-6528-8002-5a559208592c"
    }
}
historical = graph.get_state(config_history)
print(f"历史状态 step={historical.metadata['step']}: {historical.values}")
```

checkpoint_id 来源：从 `snapshot.config["configurable"]["checkpoint_id"]` 获取，或从 `get_state_history()` 遍历获取。

---

## 子图状态检查

当图包含子图时，通过 `subgraphs=True` 获取子图状态：

```python
snapshot = graph.get_state(config, subgraphs=True)

for task in snapshot.tasks:
    if task.state:
        print(f"子图 '{task.name}' 的状态: {task.state}")
```

适用场景：子图内部出错、需要检查子图执行进度、调试子图中断。

---

## 异步版本

```python
# 在异步代码中使用 aget_state()，API 完全一致
snapshot = await graph.aget_state(config)
```

---

## 实际调试场景汇总

### 场景 1：检查执行是否完成

```python
def is_graph_done(graph, config) -> bool:
    """检查图是否执行完毕"""
    snapshot = graph.get_state(config)
    return len(snapshot.next) == 0

# 使用
if is_graph_done(graph, config):
    print("执行完毕，可以读取最终结果")
    result = graph.get_state(config).values
else:
    print("还在执行中...")
```

### 场景 2：检查是否有错误

```python
def check_errors(graph, config) -> list[dict]:
    """检查图执行中的错误"""
    snapshot = graph.get_state(config)
    errors = []
    for task in snapshot.tasks:
        if task.error:
            errors.append({
                "node": task.name,
                "error_type": type(task.error).__name__,
                "message": str(task.error),
            })
    return errors

# 使用
errors = check_errors(graph, config)
if errors:
    for e in errors:
        print(f"[ERROR] 节点 {e['node']}: {e['error_type']} - {e['message']}")
```

### 场景 3：检查中断并恢复

```python
def handle_interrupts(graph, config):
    """检查中断并决定如何恢复"""
    snapshot = graph.get_state(config)

    if not snapshot.interrupts:
        print("没有中断")
        return

    for interrupt in snapshot.interrupts:
        print(f"中断: {interrupt.value}")

    # 恢复执行（传入 None 表示继续）
    graph.invoke(None, config)

```

### 场景 4：一站式状态诊断函数

```python
def diagnose_state(graph, config):
    """一站式状态诊断——调试时直接调用"""
    snapshot = graph.get_state(config)
    meta = snapshot.metadata or {}

    print("=" * 50)
    print("LangGraph 状态诊断报告")
    print("=" * 50)
    print(f"Thread: {snapshot.config['configurable']['thread_id']}")
    print(f"来源: {meta.get('source', 'N/A')}, 步骤: {meta.get('step', 'N/A')}")
    print(f"状态: {'已完成' if not snapshot.next else f'进行中 → {snapshot.next}'}")

    for key, value in snapshot.values.items():
        val_str = str(value)[:80]
        print(f"  {key}: {val_str}")

    errors = [t for t in snapshot.tasks if t.error]
    if errors:
        for task in errors:
            print(f"  [ERROR] 节点 '{task.name}': {task.error}")

    if snapshot.interrupts:
        for intr in snapshot.interrupts:
            print(f"  [INTERRUPT] {intr.value}")

    print("=" * 50)

# 使用：一行搞定调试
diagnose_state(graph, config)
```

---

## 完整可运行示例

```python
"""get_state() 状态检查完整示例"""
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(TypedDict):
    messages: Annotated[list[str], add]
    step_count: int

def greet(state: ChatState) -> dict:
    return {"messages": ["你好！我是助手。"], "step_count": state["step_count"] + 1}

def respond(state: ChatState) -> dict:
    return {"messages": ["有什么可以帮你的？"], "step_count": state["step_count"] + 1}

builder = StateGraph(ChatState)
builder.add_node("greet", greet)
builder.add_node("respond", respond)
builder.add_edge(START, "greet")
builder.add_edge("greet", "respond")
builder.add_edge("respond", END)
graph = builder.compile(checkpointer=InMemorySaver())

# 运行
config = {"configurable": {"thread_id": "demo-1"}}
graph.invoke({"messages": ["用户: 你好"], "step_count": 0}, config)

# 检查状态
snapshot = graph.get_state(config)
print(f"values: {snapshot.values}")
print(f"next: {snapshot.next}")           # () = 执行完毕
print(f"step: {snapshot.metadata['step']}")
print(f"done: {not snapshot.next}")
```

---

## 常见陷阱

### 陷阱 1：忘记配置 checkpointer

```python
graph = builder.compile()  # 没传 checkpointer → get_state 报错
# 解决：graph = builder.compile(checkpointer=InMemorySaver())
```

### 陷阱 2：thread_id 不匹配

```python
graph.invoke(input, {"configurable": {"thread_id": "thread-A"}})
# 检查时用了 thread-B → 找不到记录
snapshot = graph.get_state({"configurable": {"thread_id": "thread-B"}})
```

### 陷阱 3：混淆 next 为空的含义

```python
# 错误：next 不会是 None，它是空元组 ()
if snapshot.next == None: ...

# 正确
if not snapshot.next:
    print("完成")
```

---

## 与 RAG 开发的关联

在 RAG 工作流中，`get_state()` 的典型用途：检查检索结果是否正确、确认传给 LLM 的上下文是否完整、检查消息历史是否正确累积、当 RAG 返回错误答案时逐步检查每个节点的输出。

```python
snapshot = graph.get_state(config)
values = snapshot.values
print(f"检索到 {len(values.get('retrieved_docs', []))} 个文档")
print(f"答案: {values.get('answer', '')[:100]}...")
if values.get("hallucination_detected"):
    print("警告: 检测到幻觉!")
```

---

## 学习检查清单

- [ ] 能写出 `get_state()` 的基本调用方式
- [ ] 知道 StateSnapshot 的 8 个字段及其用途
- [ ] 能通过 `next` 判断图是否执行完毕
- [ ] 能通过 `tasks` 检查错误和中断
- [ ] 知道如何用 `checkpoint_id` 查看历史状态
- [ ] 理解 `subgraphs=True` 的作用
- [ ] 能写出一个完整的状态诊断函数

---

## 参考来源

- [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py] — StateSnapshot、PregelTask 定义
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/protocol.py] — get_state 协议接口
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/main.py] — get_state 实现
