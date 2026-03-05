# 核心概念 4：get_state_history() 历史遍历

## 概述

`get_state_history()` 是 LangGraph 提供的**状态历史遍历方法**，它以逆时间顺序返回图执行过程中保存的所有状态快照。你可以把它想象成 Git 的 `git log`——不是看代码变更历史，而是看状态变更历史。每一个快照都是图在某个时间点的完整状态，包含当时的数据、下一步要执行的节点、元数据等信息。

[来源: reference/source_状态快照_01.md | LangGraph 源码分析]

---

## 为什么需要历史遍历？

### 核心问题：图执行过程是黑盒

当你调用 `graph.invoke()` 时，图会经过多个节点，每个节点都会修改状态。如果最终结果不对，你怎么知道是哪个节点出了问题？

```
用户输入 → 节点A → 节点B → 节点C → 最终输出（不对！）
                                        ↑
                                    哪里出错了？
```

没有历史遍历，你只能看到最终状态。有了 `get_state_history()`，你可以看到每一步的状态：

```
checkpoint_0: 用户输入后的状态     ← 正常
checkpoint_1: 节点A执行后的状态    ← 正常
checkpoint_2: 节点B执行后的状态    ← 这里数据变了！问题在节点B
checkpoint_3: 节点C执行后的状态    ← 基于错误数据继续
```

### 三大使用场景

1. **调试**：定位哪个节点产生了错误的状态变更
2. **审计**：记录完整的执行轨迹，满足合规要求
3. **回溯**：找到某个历史状态，从那里重新开始（时间旅行的基础）

[来源: reference/search_状态快照_01.md | 社区实践]

---

## 方法签名详解

### 完整签名

```python
def get_state_history(
    self,
    config: RunnableConfig,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> Iterator[StateSnapshot]:
    """获取图的状态历史。"""
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config` | `RunnableConfig` | 是 | 包含 `thread_id` 的配置，指定要查询哪个线程的历史 |
| `filter` | `dict` | 否 | 按 metadata 字段过滤，如 `{"source": "loop"}` |
| `before` | `RunnableConfig` | 否 | 获取指定 checkpoint 之前的历史 |
| `limit` | `int` | 否 | 限制返回的快照数量 |

### 返回值

返回 `Iterator[StateSnapshot]`，按**逆时间顺序**排列（最新的在前面）。

每个 `StateSnapshot` 包含：

```python
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any          # 当前状态值
    next: tuple[str, ...]                  # 下一步要执行的节点
    config: RunnableConfig                 # 此快照的配置（含 checkpoint_id）
    metadata: CheckpointMetadata | None    # 元数据（source, step 等）
    created_at: str | None                 # 创建时间戳
    parent_config: RunnableConfig | None   # 父快照配置
    tasks: tuple[PregelTask, ...]          # 当前步骤的任务
    interrupts: tuple[Interrupt, ...]      # 待处理的中断
```

[来源: reference/source_状态快照_01.md | types.py:268-286]

---

## 基础用法

### 最小完整示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 定义状态
class State(TypedDict):
    topic: str
    joke: str

# 2. 定义节点
def think_node(state: State):
    return {"joke": f"关于{state['topic']}的笑话：为什么..."}

def polish_node(state: State):
    return {"joke": state["joke"] + " 因为它太有趣了！"}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("think", think_node)
builder.add_node("polish", polish_node)
builder.add_edge(START, "think")
builder.add_edge("think", "polish")
builder.add_edge("polish", END)

# 4. 编译（必须配置 checkpointer）
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 5. 执行
config = {"configurable": {"thread_id": "demo-1"}}
result = graph.invoke({"topic": "程序员"}, config=config)
print(f"最终结果: {result['joke']}")

# 6. 遍历历史
print("\n=== 状态历史（逆时间顺序）===")
for i, state in enumerate(graph.get_state_history(config)):
    print(f"\n--- 快照 {i} ---")
    print(f"  状态值: {state.values}")
    print(f"  下一步: {state.next}")
    print(f"  checkpoint_id: {state.config['configurable']['checkpoint_id']}")
    print(f"  元数据: source={state.metadata.get('source')}, step={state.metadata.get('step')}")
    print(f"  创建时间: {state.created_at}")
```

**预期输出**：

```
最终结果: 关于程序员的笑话：为什么... 因为它太有趣了！

=== 状态历史（逆时间顺序）===

--- 快照 0 ---
  状态值: {'topic': '程序员', 'joke': '关于程序员的笑话：为什么... 因为它太有趣了！'}
  下一步: ()
  checkpoint_id: 1ef...003
  元数据: source=loop, step=2
  创建时间: 2026-02-27T10:00:02+00:00

--- 快照 1 ---
  状态值: {'topic': '程序员', 'joke': '关于程序员的笑话：为什么...'}
  下一步: ('polish',)
  checkpoint_id: 1ef...002
  元数据: source=loop, step=1
  创建时间: 2026-02-27T10:00:01+00:00

--- 快照 2 ---
  状态值: {'topic': '程序员'}
  下一步: ('think',)
  checkpoint_id: 1ef...001
  元数据: source=input, step=-1
  创建时间: 2026-02-27T10:00:00+00:00
```

**关键观察**：
- 快照 0 是最终状态（`next` 为空，说明执行完毕）
- 快照 1 是 think 节点执行后的状态（`next` 指向 polish）
- 快照 2 是初始输入状态（`source=input`，`step=-1`）
- 逆时间顺序：最新的排在最前面

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 参数详解

### filter 参数：按元数据过滤

`filter` 参数让你按 `CheckpointMetadata` 的字段筛选快照。最常用的过滤字段是 `source`。

```python
# source 的四种取值：
# "input"  - 来自 invoke/stream 的初始输入
# "loop"   - 来自图的循环执行（节点执行后）
# "update" - 来自手动 update_state()
# "fork"   - 从另一个 checkpoint 分叉

# 只看节点执行产生的快照（跳过初始输入）
loop_states = list(graph.get_state_history(
    config,
    filter={"source": "loop"}
))
print(f"节点执行产生了 {len(loop_states)} 个快照")

for state in loop_states:
    step = state.metadata.get("step")
    print(f"  step {step}: next={state.next}")
```

**预期输出**：

```
节点执行产生了 2 个快照
  step 2: next=()
  step 1: next=('polish',)
```

**前端类比**：就像 Redux DevTools 中的 action 过滤器——你可以只看某种类型的 action，忽略其他的。

**日常生活类比**：就像银行流水筛选——你可以只看"转账"类型的记录，忽略"查询余额"的记录。

### before 参数：获取指定时间点之前的历史

`before` 参数接受一个包含 `checkpoint_id` 的配置，返回该 checkpoint 之前的所有快照。

```python
# 先获取所有历史
all_states = list(graph.get_state_history(config))

# 假设我们想看第一个快照之前的历史
if len(all_states) >= 2:
    middle_state = all_states[1]  # 取中间的快照

    # 获取这个快照之前的历史
    earlier_states = list(graph.get_state_history(
        config,
        before=middle_state.config
    ))

    print(f"在 checkpoint {middle_state.config['configurable']['checkpoint_id']} 之前：")
    for state in earlier_states:
        print(f"  checkpoint: {state.config['configurable']['checkpoint_id']}")
        print(f"  step: {state.metadata.get('step')}")
```

### limit 参数：限制返回数量

当历史记录很长时，用 `limit` 控制返回数量，避免内存问题。

```python
# 只获取最近 3 个快照
recent_states = list(graph.get_state_history(config, limit=3))
print(f"获取了 {len(recent_states)} 个最近的快照")

# 只获取最新的 1 个（等价于 get_state，但返回格式不同）
latest = list(graph.get_state_history(config, limit=1))
```

### 参数组合使用

```python
# 组合使用：获取最近 5 个由节点执行产生的快照
filtered_recent = list(graph.get_state_history(
    config,
    filter={"source": "loop"},
    limit=5
))
```

[来源: reference/source_状态快照_01.md | main.py:1319-1368]

---

## 内部实现原理

### 源码流程

`get_state_history()` 的内部实现分为三步：

```
get_state_history(config, filter, before, limit)
    ↓
1. 获取 checkpointer 实例
    ↓
2. checkpointer.list(config, before=before, limit=limit, filter=filter)
   → 返回 Iterator[CheckpointTuple]（逆时间顺序）
    ↓
3. 对每个 CheckpointTuple 调用 _prepare_state_snapshot()
   → 转换为 StateSnapshot
    ↓
返回 Iterator[StateSnapshot]
```

### 关键步骤解析

**步骤 1：获取 checkpointer**

```python
# 简化的源码逻辑
checkpointer = config["configurable"].get("checkpoint") or self.checkpointer
if checkpointer is None:
    raise ValueError("需要配置 checkpointer 才能使用 get_state_history")
```

**步骤 2：checkpointer.list()**

这是存储层的方法，负责从底层存储中按条件检索 checkpoint。以 `InMemorySaver` 为例：

```python
# InMemorySaver 的存储结构
# thread_id → checkpoint_ns → checkpoint_id → checkpoint 数据
storage: defaultdict[str, dict[str, dict[str, tuple[...]]]]
```

`list()` 方法会遍历指定 thread 的所有 checkpoint，按时间倒序排列，并应用 `filter`、`before`、`limit` 条件。

**步骤 3：_prepare_state_snapshot()**

这是核心转换方法，将底层的 `CheckpointTuple` 转换为用户友好的 `StateSnapshot`：

```
CheckpointTuple
  ├─ checkpoint.channel_values  →  StateSnapshot.values
  ├─ prepare_next_tasks()       →  StateSnapshot.next + StateSnapshot.tasks
  ├─ config + checkpoint_id     →  StateSnapshot.config
  ├─ metadata                   →  StateSnapshot.metadata
  ├─ checkpoint.ts              →  StateSnapshot.created_at
  └─ parent_config              →  StateSnapshot.parent_config
```

[来源: reference/source_状态快照_01.md | main.py:996-1113]

---

## 异步版本：aget_state_history()

LangGraph 为所有状态操作提供了异步版本，适用于 `async` 环境。

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class State(TypedDict):
    message: str

async def process_node(state: State):
    return {"message": state["message"] + " → 已处理"}

# 构建图
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

async def main():
    config = {"configurable": {"thread_id": "async-demo"}}

    # 异步执行
    result = await graph.ainvoke({"message": "你好"}, config=config)
    print(f"结果: {result}")

    # 异步遍历历史
    print("\n=== 异步历史遍历 ===")
    async for state in graph.aget_state_history(config):
        print(f"step={state.metadata.get('step')}, next={state.next}")

asyncio.run(main())
```

**签名对比**：

| 同步版本 | 异步版本 | 返回类型 |
|---------|---------|---------|
| `get_state_history()` | `aget_state_history()` | `Iterator` / `AsyncIterator` |
| `get_state()` | `aget_state()` | `StateSnapshot` |
| `update_state()` | `aupdate_state()` | `RunnableConfig` |

---

## 实际应用场景

### 场景 1：RAG 系统调试——定位检索质量问题

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class RAGState(TypedDict):
    query: str
    documents: Annotated[list[str], add]
    answer: str

def retrieve_node(state: RAGState):
    # 模拟检索
    query = state["query"]
    docs = [f"文档1: {query}相关内容", f"文档2: {query}补充信息"]
    return {"documents": docs}

def generate_node(state: RAGState):
    context = "\n".join(state["documents"])
    return {"answer": f"基于以下内容回答：{context}"}

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 执行
config = {"configurable": {"thread_id": "rag-debug-1"}}
result = graph.invoke({"query": "什么是向量数据库？"}, config=config)

# 调试：检查每一步的状态
print("=== RAG 执行轨迹 ===")
for state in graph.get_state_history(config):
    step = state.metadata.get("step", "?")
    source = state.metadata.get("source", "?")
    print(f"\n[step={step}, source={source}]")

    if "documents" in state.values:
        print(f"  检索到 {len(state.values['documents'])} 篇文档")
        for doc in state.values["documents"]:
            print(f"    - {doc[:50]}...")

    if "answer" in state.values and state.values["answer"]:
        print(f"  生成答案: {state.values['answer'][:80]}...")
```

### 场景 2：执行审计——记录完整操作日志

```python
import json

def export_execution_audit(graph, config, output_file="audit.json"):
    """导出完整的执行审计日志"""
    audit_log = []

    for state in graph.get_state_history(config):
        entry = {
            "checkpoint_id": state.config["configurable"]["checkpoint_id"],
            "step": state.metadata.get("step"),
            "source": state.metadata.get("source"),
            "created_at": state.created_at,
            "next_nodes": list(state.next),
            "state_keys": list(state.values.keys()) if isinstance(state.values, dict) else [],
        }
        audit_log.append(entry)

    # 反转为正序（时间从早到晚）
    audit_log.reverse()

    print(json.dumps(audit_log, indent=2, ensure_ascii=False))
    return audit_log

# 使用
audit = export_execution_audit(graph, config)
```

[来源: reference/search_状态快照_01.md | 社区实践案例]

---

## 与 get_state() 的区别

| 特性 | `get_state()` | `get_state_history()` |
|------|--------------|----------------------|
| 返回数量 | 1 个快照 | 所有快照（可限制） |
| 返回类型 | `StateSnapshot` | `Iterator[StateSnapshot]` |
| 默认行为 | 返回最新状态 | 返回完整历史 |
| 过滤能力 | 无 | 支持 filter/before/limit |
| 典型用途 | 查看当前状态 | 调试、审计、回溯 |
| 性能 | O(1) | O(n)，n 为历史长度 |

**选择建议**：
- 只需要当前状态 → 用 `get_state()`
- 需要查看执行过程 → 用 `get_state_history()`
- 需要找到特定历史状态进行时间旅行 → 用 `get_state_history()` + `filter`

---

## 常见误区

### 误区 1："get_state_history() 返回的是正序列表" ❌

**事实**：返回的是**逆时间顺序**的迭代器，最新的快照在最前面。如果你需要正序，要手动反转：

```python
# 逆序（默认）
states = list(graph.get_state_history(config))
# states[0] 是最新的，states[-1] 是最早的

# 正序
states_chronological = list(reversed(states))
# states_chronological[0] 是最早的
```

### 误区 2："没有 checkpointer 也能用 get_state_history()" ❌

**事实**：`get_state_history()` 依赖 checkpointer 存储的数据。如果编译图时没有配置 checkpointer，调用会报错。

```python
# ❌ 没有 checkpointer
graph = builder.compile()
# graph.get_state_history(config)  # 报错！

# ✅ 配置了 checkpointer
graph = builder.compile(checkpointer=MemorySaver())
# graph.get_state_history(config)  # 正常工作
```

### 误区 3："历史记录会无限增长" ❌

**事实**：不同的 checkpointer 有不同的存储策略。`InMemorySaver` 确实会在内存中保留所有历史，但生产环境的 checkpointer（如 PostgresSaver）可以配置保留策略。使用 `limit` 参数也能控制查询时的内存消耗。

---

## 总结

### 核心要点

1. `get_state_history()` 以**逆时间顺序**返回所有状态快照
2. 三个过滤参数：`filter`（按元数据）、`before`（时间点之前）、`limit`（数量限制）
3. 内部流程：`checkpointer.list()` → `_prepare_state_snapshot()` → `StateSnapshot`
4. 必须配置 checkpointer 才能使用
5. 异步版本为 `aget_state_history()`

### 下一步

理解了历史遍历后，下一个核心概念将讲解 **update_state() 状态修改**——如何基于历史快照创建新的 checkpoint，这是实现时间旅行的关键操作。

---

**参考资料**：
- [LangGraph 源码 - main.py](reference/source_状态快照_01.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [社区实践案例](reference/search_状态快照_01.md)
