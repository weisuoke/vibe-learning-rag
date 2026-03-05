# get_state_history() 历史追溯

> 通过遍历图执行的完整状态历史，定位问题步骤并从任意历史节点分叉重新执行

---

## 一句话定义

**`get_state_history()` 返回一个按时间倒序排列的 StateSnapshot 迭代器，让你遍历图执行的完整历史，实现问题定位、状态对比和历史分叉。**

---

## 为什么需要 get_state_history()？

**日常生活类比：** Word 的"版本历史"可以让你看到每一次保存的完整内容，某次修改引入错误时回到之前的版本重新开始。`get_state_history()` 就是 LangGraph 的"版本历史"——记录了图执行过程中每一步的完整状态快照。

**前端类比：** 就像 Redux DevTools 的时间旅行——你可以看到每一次 action dispatch 后的完整 state 变化，点击任意历史记录就能回到那个时刻。

---

## 方法签名

```python
def get_state_history(
    self,
    config: RunnableConfig,
    *,
    filter: Optional[dict[str, Any]] = None,
    before: Optional[RunnableConfig] = None,
    limit: Optional[int] = None,
) -> Iterator[StateSnapshot]:
    """Get the history of the graph execution."""
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config` | `RunnableConfig` | 是 | 包含 `thread_id`，指定要查询的线程 |
| `filter` | `dict[str, Any]` | 否 | 按 metadata 字段筛选历史记录 |
| `before` | `RunnableConfig` | 否 | 只返回该 config 之前的历史记录 |
| `limit` | `int` | 否 | 限制返回的最大记录数 |

**返回值：** `Iterator[StateSnapshot]` — 按时间倒序排列的快照迭代器

**前置条件：** 必须配置 Checkpointer（`compile(checkpointer=InMemorySaver())`），否则抛出 `ValueError("No checkpointer set")`。

---

## 基本用法

```python
config = {"configurable": {"thread_id": "1"}}

# 遍历所有历史快照（最新的在前）
for snapshot in graph.get_state_history(config):
    print(f"Step: {snapshot.metadata.get('step')}")
    print(f"Source: {snapshot.metadata.get('source')}")
    print(f"Values: {snapshot.values}")
    print(f"Next: {snapshot.next}")
    print("---")

# 转为列表以支持随机访问
history = list(graph.get_state_history(config))
print(f"共 {len(history)} 个快照")
print(f"最新: Step {history[0].metadata['step']}")
print(f"最早: Step {history[-1].metadata['step']}")
```

---

## 历史快照的结构

每个快照是 `StateSnapshot` 对象，按时间倒序排列，通过 `parent_config` 形成链式关系。

```
时间线（从左到右执行）：

  input → __start__ → analyst → reviewer → END
  Step -1   Step 0     Step 1    Step 2

get_state_history() 返回顺序（最新在前）：

  [0] Step 2  (reviewer 执行后)  ← 最新
  [1] Step 1  (analyst 执行后)
  [2] Step 0  (__start__ 执行后)
  [3] Step -1 (用户输入)         ← 最早
```

### metadata 中的关键字段

```python
snapshot = history[0]

# step: 执行步骤编号（-1 表示初始输入）
print(snapshot.metadata["step"])        # 2

# source: 快照来源
#   "input"  — 用户输入产生
#   "loop"   — 图执行循环产生
#   "update" — update_state() 手动修改产生
print(snapshot.metadata["source"])      # "loop"

# writes: 该步骤的写入内容
print(snapshot.metadata.get("writes"))  # {"reviewer": {"messages": [...], "step": 2}}
```

### 快照之间的链式关系

```python
# 每个快照的 parent_config 指向上一个快照的 config
for i, snap in enumerate(history[:-1]):
    parent_id = snap.parent_config["configurable"]["checkpoint_id"]
    prev_id = history[i + 1].config["configurable"]["checkpoint_id"]
    assert parent_id == prev_id  # True — parent 指向前一个快照
```

---

## 筛选历史记录

### 按 metadata 筛选（filter）

```python
# 只获取图执行循环产生的快照
for snap in graph.get_state_history(config, filter={"source": "loop"}):
    print(f"Step {snap.metadata['step']}: {snap.values}")

# 只获取手动修改产生的快照
for snap in graph.get_state_history(config, filter={"source": "update"}):
    print(f"手动修改: {snap.values}")
```

### 按时间点截断（before）

```python
history = list(graph.get_state_history(config))
# 获取 Step 1 之前的所有历史
older = list(graph.get_state_history(config, before=history[1].config))
```

### 限制返回数量（limit）

```python
# 只获取最近 3 条
recent = list(graph.get_state_history(config, limit=3))

# 组合使用：最近 5 条循环快照
recent_loops = list(graph.get_state_history(
    config, filter={"source": "loop"}, limit=5,
))
```

---

## 定位问题步骤

调试核心思路：**找到状态从"正常"变为"异常"的那一步，检查该步骤的写入内容，确定问题节点。**

### 追踪字段变化轨迹

```python
history = list(graph.get_state_history(config))

# 正序查看 score 变化（从早到晚）
for snap in reversed(history):
    score = snap.values.get("score", "N/A")
    step = snap.metadata.get("step")
    print(f"Step {step}: score = {score}")
```

### 对比相邻步骤的状态差异

```python
def diff_snapshots(before, after):
    """对比两个快照的状态差异"""
    all_keys = set(list(before.values.keys()) + list(after.values.keys()))
    for key in all_keys:
        old_val = before.values.get(key)
        new_val = after.values.get(key)
        if old_val != new_val:
            print(f"  {key}: {old_val} → {new_val}")

# 对比每一步的变化
for i in range(len(history) - 1):
    after, before = history[i], history[i + 1]
    step = after.metadata.get("step")
    print(f"\n--- Step {step} ---")
    diff_snapshots(before, after)
```

### 通过 writes 确定问题节点

```python
for snap in graph.get_state_history(config):
    writes = snap.metadata.get("writes", {})
    for node_name, output in writes.items():
        if isinstance(output, dict) and output.get("score", 0) < 0:
            print(f"问题节点: {node_name} (Step {snap.metadata['step']})")
```

---

## 从历史状态分叉

每个快照的 `config` 包含精确的 `checkpoint_id`，可以作为新执行的起点。

### 基本分叉

```python
# 找到目标历史状态，从该点重新执行
for snapshot in graph.get_state_history(config):
    if snapshot.metadata.get("step") == 1:
        result = graph.invoke(None, snapshot.config)
        print(f"从 Step 1 重新执行: {result}")
        break
```

### 修改后分叉

```python
# 发现 Step 1 的 score 不对，修正后重新执行
step1 = next(s for s in history if s.metadata.get("step") == 1)

graph.update_state(step1.config, {"score": 95})  # 修正
result = graph.invoke(None, step1.config)         # 从修正后继续
```

### 条件分叉

```python
# 找到最后一个 score > 80 的快照，从那里重新执行
for snapshot in graph.get_state_history(config):
    if snapshot.values.get("score", 0) > 80:
        result = graph.invoke(None, snapshot.config)
        break
```

---

## 异步版本

```python
# 异步遍历
async for snapshot in graph.aget_state_history(config):
    print(f"Step: {snapshot.metadata.get('step')}")

# 异步 + 筛选
async for snapshot in graph.aget_state_history(
    config, filter={"source": "loop"}, limit=5,
):
    print(f"Step {snapshot.metadata['step']}: {snapshot.values}")
```

---

## 与 get_state() 的区别

| 对比维度 | `get_state()` | `get_state_history()` |
|----------|---------------|----------------------|
| 返回类型 | 单个 `StateSnapshot` | `Iterator[StateSnapshot]` |
| 默认行为 | 返回最新状态 | 返回完整历史（倒序） |
| 筛选能力 | `checkpoint_id` 精确定位 | `filter`/`before`/`limit` |
| 典型用途 | 检查当前状态 | 调试、问题定位、时间旅行 |
| 性能开销 | 低（单次查询） | 较高（遍历 checkpoint） |

**选择建议：** 只看当前状态用 `get_state()`，需要遍历/搜索/对比历史用 `get_state_history()`。

---

## 性能注意事项

每次 invoke 产生多个 checkpoint（输入 1 个 + 每个节点 1 个），多次调用后历史会持续增长。

```python
# ⚠️ 大量历史时避免一次性加载
# history = list(graph.get_state_history(config))  # 可能占用大量内存

# ✅ 使用 limit 限制
history = list(graph.get_state_history(config, limit=20))

# ✅ 流式处理，找到目标就停止
for snap in graph.get_state_history(config):
    if found_target(snap):
        process(snap)
        break

# ✅ 使用 before 做分页
page1 = list(graph.get_state_history(config, limit=10))
if page1:
    page2 = list(graph.get_state_history(config, before=page1[-1].config, limit=10))
```

| Checkpointer | 查询性能 | 适用场景 |
|---------------|----------|----------|
| `InMemorySaver` | 最快 | 开发调试、单元测试 |
| `SqliteSaver` | 中等 | 本地开发 |
| `PostgresSaver` | 较慢 | 生产环境 |

---

## 完整使用示例

```python
"""
get_state_history() 完整调试示例
"""
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== 定义状态和图 =====
class DebugState(TypedDict):
    messages: Annotated[list[str], add]
    score: int

def analyzer(state: DebugState) -> dict:
    return {"messages": [f"分析完成 (score={state['score']})"], "score": state["score"] + 10}

def evaluator(state: DebugState) -> dict:
    new_score = state["score"] // 2  # 模拟问题：score 被意外减半
    return {"messages": [f"评估完成 ({state['score']}→{new_score})"], "score": new_score}

def reporter(state: DebugState) -> dict:
    return {"messages": [f"报告 (score={state['score']})"]}

graph = (
    StateGraph(DebugState)
    .add_node("analyzer", analyzer)
    .add_node("evaluator", evaluator)
    .add_node("reporter", reporter)
    .add_edge(START, "analyzer")
    .add_edge("analyzer", "evaluator")
    .add_edge("evaluator", "reporter")
    .add_edge("reporter", END)
    .compile(checkpointer=InMemorySaver())
)

# ===== 运行 =====
config = {"configurable": {"thread_id": "debug-demo"}}
result = graph.invoke({"messages": ["开始"], "score": 80}, config)
print(f"最终: score={result['score']}")
# 最终: score=45（80 → 90 → 45 → 45）

# ===== 遍历历史 =====
print("\n=== 执行历史 ===")
history = list(graph.get_state_history(config))
for snap in reversed(history):
    step = snap.metadata.get("step")
    source = snap.metadata.get("source")
    score = snap.values.get("score", "N/A")
    print(f"Step {step:>3} | {source:<6} | score={score}")

# ===== 定位问题 =====
print("\n=== Score 下降追踪 ===")
for i in range(len(history) - 1):
    after, before = history[i], history[i + 1]
    s_before = before.values.get("score", 0)
    s_after = after.values.get("score", 0)
    if s_after < s_before:
        writes = after.metadata.get("writes", {})
        culprit = list(writes.keys())[0] if writes else "unknown"
        print(f"Score 下降: {s_before} → {s_after}, 问题节点: {culprit}")

# ===== 筛选 =====
print("\n=== loop 快照 ===")
for snap in graph.get_state_history(config, filter={"source": "loop"}):
    print(f"Step {snap.metadata['step']}: score={snap.values.get('score')}")
```

**预期输出：**
```
最终: score=45

=== 执行历史 ===
Step  -1 | input  | score=80
Step   0 | loop   | score=80
Step   1 | loop   | score=90
Step   2 | loop   | score=45
Step   3 | loop   | score=45

=== Score 下降追踪 ===
Score 下降: 90 → 45, 问题节点: evaluator

=== loop 快照 ===
Step 3: score=45
Step 2: score=45
Step 1: score=90
Step 0: score=80
```

---

## 源码实现要点

[来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/main.py]

```
get_state_history(config, filter, before, limit)
│
├── 1. 检查 self.checkpointer 是否存在
│
├── 2. 调用 checkpointer.list(config, filter, before, limit)
│   └── 返回 Iterator[CheckpointTuple]（按时间倒序）
│
├── 3. 对每个 CheckpointTuple 调用 _prepare_state_snapshot()
│   └── 与 get_state() 使用相同的转换逻辑
│
└── 4. yield StateSnapshot（惰性求值，逐个返回）
```

关键点：这是一个**生成器函数**，不会一次性加载所有历史到内存。只要不 `list()` 全部转换，就可以安全遍历大量历史。

---

## 学习检查清单

- [ ] 能写出 `get_state_history()` 的基本调用方式
- [ ] 理解返回的快照是按时间倒序排列的
- [ ] 知道 `filter`、`before`、`limit` 三个筛选参数的用法
- [ ] 能通过历史快照定位状态异常的步骤
- [ ] 能对比两个快照的状态差异
- [ ] 理解如何从历史状态分叉重新执行
- [ ] 知道 `get_state()` 和 `get_state_history()` 的区别
- [ ] 了解性能注意事项和生产环境建议

---

## 下一步学习

- **03_核心概念_4_自定义调试工具.md** — 日志、StreamWriter 等自定义调试方案
- **07_实战代码_场景3_时间旅行调试.md** — 完整的时间旅行调试实战案例
