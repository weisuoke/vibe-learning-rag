# 核心概念 5：update_state() 状态修改

## 概述

`update_state()` 是 LangGraph 提供的**手动状态修改方法**，它允许你在图执行之外修改状态，创建一个新的 checkpoint。你可以把它想象成 Git 的 `git commit --amend` 或者创建一个新的 commit——它不会修改历史，而是基于某个历史状态创建一个新的分支点。这是实现时间旅行（Fork 分叉）和 Human-in-the-loop（人工干预）的核心 API。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 为什么需要手动修改状态？

### 三个核心场景

**场景 1：Human-in-the-loop（人工审核）**

图执行到某个节点后暂停，人工审核状态，修改后继续执行。

```
用户输入 → 检索节点 → [暂停] → 人工审核检索结果 → 修改/确认 → 生成节点 → 输出
                         ↑
                    update_state() 在这里介入
```

**场景 2：错误修复**

发现某个历史状态有问题，回到那个状态修改后重新执行。

```
执行历史：A → B → C → D（结果错误）
                 ↑
            回到 B，修改状态，重新执行 → C' → D'（结果正确）
```

**场景 3：A/B 测试**

从同一个历史状态出发，尝试不同的状态值，比较结果。

```
历史状态 B → 修改为 X → 执行 → 结果1
历史状态 B → 修改为 Y → 执行 → 结果2
```

[来源: reference/search_状态快照_01.md | 社区实践]

---

## 方法签名详解

### 完整签名

```python
def update_state(
    self,
    config: RunnableConfig,
    values: dict[str, Any] | Any,
    as_node: str | None = None,
) -> RunnableConfig:
    """更新图的状态，创建新的 checkpoint。"""
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config` | `RunnableConfig` | 是 | 指定要基于哪个 checkpoint 进行修改 |
| `values` | `dict` | 是 | 要更新的状态值 |
| `as_node` | `str` | 否 | 模拟哪个节点的输出（影响 reducer 行为和下一步路由） |

### 返回值

返回 `RunnableConfig`，包含新创建的 checkpoint 的 `checkpoint_id`。你可以用这个配置继续执行图。

[来源: reference/source_状态快照_01.md | main.py]

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
    return {"joke": f"关于{state['topic']}的笑话初稿"}

def polish_node(state: State):
    return {"joke": state["joke"] + "（已润色）"}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("think", think_node)
builder.add_node("polish", polish_node)
builder.add_edge(START, "think")
builder.add_edge("think", "polish")
builder.add_edge("polish", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 4. 执行
config = {"configurable": {"thread_id": "update-demo"}}
result = graph.invoke({"topic": "程序员"}, config=config)
print(f"原始结果: {result['joke']}")

# 5. 查看当前状态
current = graph.get_state(config)
print(f"当前状态: {current.values}")
print(f"当前 next: {current.next}")  # () 表示已执行完毕

# 6. 手动修改状态
new_config = graph.update_state(
    config,
    values={"topic": "猫咪", "joke": "关于猫咪的全新笑话"}
)
print(f"\n新 checkpoint_id: {new_config['configurable']['checkpoint_id']}")

# 7. 验证修改生效
updated = graph.get_state(config)
print(f"修改后状态: {updated.values}")
```

**预期输出**：

```
原始结果: 关于程序员的笑话初稿（已润色）
当前状态: {'topic': '程序员', 'joke': '关于程序员的笑话初稿（已润色）'}
当前 next: ()

新 checkpoint_id: 1ef...004
修改后状态: {'topic': '猫咪', 'joke': '关于猫咪的全新笑话'}
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## Reducer 规则：覆盖 vs 合并

这是 `update_state()` 最重要的行为特征。更新状态时，LangGraph 会遵循字段定义的 reducer 规则。

### 规则总结

| 字段类型 | 行为 | 示例 |
|---------|------|------|
| 无 reducer（普通字段） | **覆盖**：新值直接替换旧值 | `foo: int` → 新值覆盖旧值 |
| 有 reducer（Annotated 字段） | **合并**：按 reducer 函数处理 | `bar: Annotated[list, add]` → 新值追加到列表 |

### 代码示例

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    count: int                              # 无 reducer → 覆盖
    messages: Annotated[list[str], add]     # 有 reducer → 追加

def node_a(state: State):
    return {"count": 1, "messages": ["节点A执行"]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_edge("a", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 执行
config = {"configurable": {"thread_id": "reducer-demo"}}
result = graph.invoke({"count": 0, "messages": ["初始化"]}, config=config)
print(f"执行后: count={result['count']}, messages={result['messages']}")
# 执行后: count=1, messages=['初始化', '节点A执行']

# 手动更新状态
graph.update_state(config, {"count": 100, "messages": ["手动添加"]})

updated = graph.get_state(config)
print(f"更新后: count={updated.values['count']}, messages={updated.values['messages']}")
# 更新后: count=100, messages=['初始化', '节点A执行', '手动添加']
#         ↑ 覆盖为 100                    ↑ 追加了 '手动添加'
```

**关键理解**：
- `count` 没有 reducer，所以 `update_state` 直接覆盖为 100
- `messages` 有 `add` reducer，所以 `["手动添加"]` 被追加到现有列表末尾

**前端类比**：就像 React 的 `setState`——普通字段是 `setState({count: 100})`（覆盖），而带 reducer 的字段是 `dispatch({type: 'ADD', payload: '手动添加'})`（按规则合并）。

**日常生活类比**：普通字段像改名字（直接替换），带 reducer 的字段像往购物车加东西（追加，不清空原有的）。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## as_node 参数：模拟节点输出

`as_node` 参数让你指定这次更新"假装"是哪个节点的输出。这会影响两件事：

1. **Reducer 行为**：不同节点可能对同一字段有不同的写入方式
2. **下一步路由**：图会根据"哪个节点刚执行完"来决定下一步走哪条边

### 基础示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    query: str
    result: str

def search_node(state: State):
    return {"result": f"搜索结果: {state['query']}"}

def answer_node(state: State):
    return {"result": f"最终答案: {state['result']}"}

builder = StateGraph(State)
builder.add_node("search", search_node)
builder.add_node("answer", answer_node)
builder.add_edge(START, "search")
builder.add_edge("search", "answer")
builder.add_edge("answer", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 执行到 search 节点后
config = {"configurable": {"thread_id": "as-node-demo"}}
graph.invoke({"query": "什么是 RAG？"}, config=config)

# 获取 search 执行后的状态
states = list(graph.get_state_history(config))
after_search = states[1]  # search 执行后、answer 执行前的状态
print(f"search 后: next={after_search.next}")  # ('answer',)

# 模拟 search 节点重新输出（替换检索结果）
new_config = graph.update_state(
    after_search.config,
    values={"result": "更好的搜索结果"},
    as_node="search"  # 假装是 search 节点的输出
)

# 检查更新后的状态
updated = graph.get_state(new_config)
print(f"更新后: next={updated.next}")  # ('answer',) — 下一步仍然是 answer
print(f"更新后: result={updated.values['result']}")
```

### as_node 的路由影响

```
图结构：START → search → answer → END

不指定 as_node：
  update_state(config, values)
  → next 可能不确定

指定 as_node="search"：
  update_state(config, values, as_node="search")
  → LangGraph 认为 search 刚执行完
  → 根据边定义，next = ("answer",)
```

**什么时候需要 as_node？**
- 当你想从某个中间状态继续执行时，需要告诉 LangGraph "当前执行到了哪个节点"
- 在 Human-in-the-loop 场景中，人工修改后需要指定从哪个节点的输出继续

[来源: reference/source_状态快照_01.md | LangGraph 源码分析]

---

## source 标记：update vs fork

`update_state()` 创建的新 checkpoint 会在元数据中标记 `source` 字段，区分两种情况：

### "update"：基于最新状态修改

当你基于当前最新的 checkpoint 调用 `update_state()` 时，source 标记为 `"update"`。

```python
# 基于最新状态修改
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"topic": "AI"}, config=config)

# 这是 update（基于最新状态）
graph.update_state(config, {"topic": "ML"})

state = graph.get_state(config)
print(state.metadata["source"])  # "update"
```

### "fork"：基于历史状态分叉

当你基于一个非最新的历史 checkpoint 调用 `update_state()` 时，source 标记为 `"fork"`。

```python
# 获取历史状态
states = list(graph.get_state_history(config))
old_state = states[-1]  # 最早的状态

# 这是 fork（基于历史状态）
new_config = graph.update_state(
    old_state.config,  # 使用历史状态的 config
    {"topic": "新方向"}
)

forked = graph.get_state(new_config)
print(forked.metadata["source"])  # "fork"
```

### 可视化理解

```
原始执行线：
  input → step0 → step1 → step2（最新）

update（基于最新）：
  input → step0 → step1 → step2 → update_step（source="update"）

fork（基于历史）：
  input → step0 → step1 → step2
               ↘
                fork_step（source="fork"）→ 可以从这里继续执行
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 从历史状态分叉：完整工作流

这是 `update_state()` 最强大的用法——从历史中的某个点创建分叉，探索不同的执行路径。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    topic: str
    style: str
    content: str

def draft_node(state: State):
    return {"content": f"[{state['style']}风格] 关于{state['topic']}的文章草稿"}

def review_node(state: State):
    return {"content": state["content"] + " → 审核通过"}

builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ===== 原始执行 =====
config = {"configurable": {"thread_id": "fork-demo"}}
result = graph.invoke({"topic": "AI", "style": "正式"}, config=config)
print(f"原始结果: {result['content']}")

# ===== 找到 draft 执行前的状态 =====
states = list(graph.get_state_history(config))
print(f"\n共有 {len(states)} 个历史快照")

# 找到 input 状态（draft 执行前）
input_state = None
for s in states:
    if s.metadata.get("source") == "input":
        input_state = s
        break

print(f"找到输入状态: step={input_state.metadata.get('step')}")
print(f"输入状态值: {input_state.values}")

# ===== 从输入状态分叉，修改风格 =====
fork_config = graph.update_state(
    input_state.config,
    values={"style": "幽默"},  # 改为幽默风格
    as_node="__start__"  # 模拟从起点开始
)

print(f"\n分叉 checkpoint_id: {fork_config['configurable']['checkpoint_id']}")

# ===== 从分叉点继续执行 =====
fork_result = graph.invoke(None, config=fork_config)
print(f"分叉结果: {fork_result['content']}")

# ===== 对比两条路径 =====
print(f"\n=== 对比 ===")
print(f"原始路径: {result['content']}")
print(f"分叉路径: {fork_result['content']}")
```

**预期输出**：

```
原始结果: [正式风格] 关于AI的文章草稿 → 审核通过

共有 3 个历史快照
找到输入状态: step=-1
输入状态值: {'topic': 'AI', 'style': '正式'}

分叉 checkpoint_id: 1ef...005

分叉结果: [幽默风格] 关于AI的文章草稿 → 审核通过

=== 对比 ===
原始路径: [正式风格] 关于AI的文章草稿 → 审核通过
分叉路径: [幽默风格] 关于AI的文章草稿 → 审核通过
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 与 get_state() 的配合使用

`update_state()` 通常不会单独使用，而是和 `get_state()` / `get_state_history()` 配合，形成"查看 → 修改 → 继续"的工作流。

### 典型工作流

```python
# 步骤 1：执行图
config = {"configurable": {"thread_id": "workflow-demo"}}
graph.invoke({"topic": "Python"}, config=config)

# 步骤 2：查看当前状态
current = graph.get_state(config)
print(f"当前: {current.values}")
print(f"下一步: {current.next}")

# 步骤 3：如果需要修改
if current.values.get("topic") != "期望的值":
    new_config = graph.update_state(config, {"topic": "修正后的值"})
    print(f"已修改，新 checkpoint: {new_config}")

# 步骤 4：如果需要从修改后继续执行
if current.next:  # 如果还有下一步
    result = graph.invoke(None, config=new_config)
    print(f"继续执行结果: {result}")
```

### 流程图

```
                    ┌─────────────┐
                    │ graph.invoke│
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  get_state  │ ← 查看当前状态
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
              ┌─ NO ┤ 需要修改？  │
              │     └──────┬──────┘
              │            │ YES
              │            ↓
              │     ┌──────────────┐
              │     │ update_state │ ← 修改状态
              │     └──────┬───────┘
              │            ↓
              │     ┌──────────────┐
              │     │ 还有下一步？  │
              │     └──────┬───────┘
              │            │ YES
              │            ↓
              │     ┌──────────────────┐
              │     │ invoke(None, cfg)│ ← 继续执行
              │     └──────┬───────────┘
              │            ↓
              └────→ 完成
```

---

## 异步版本：aupdate_state()

```python
import asyncio

async def main():
    config = {"configurable": {"thread_id": "async-update"}}

    # 异步执行
    await graph.ainvoke({"topic": "AI"}, config=config)

    # 异步更新状态
    new_config = await graph.aupdate_state(
        config,
        values={"topic": "ML"}
    )

    # 异步获取更新后的状态
    updated = await graph.aget_state(new_config)
    print(f"异步更新后: {updated.values}")

asyncio.run(main())
```

---

## 常见误区

### 误区 1："update_state() 会修改原来的 checkpoint" ❌

**事实**：`update_state()` 永远不会修改已有的 checkpoint。它创建一个**全新的** checkpoint，原来的历史完整保留。这和 Git 的理念一样——历史是不可变的。

```python
# 修改前
states_before = list(graph.get_state_history(config))
count_before = len(states_before)

# 修改
graph.update_state(config, {"topic": "新值"})

# 修改后
states_after = list(graph.get_state_history(config))
count_after = len(states_after)

print(f"修改前: {count_before} 个快照")
print(f"修改后: {count_after} 个快照")  # 多了 1 个
```

### 误区 2："update_state() 后图会自动继续执行" ❌

**事实**：`update_state()` 只是创建新的 checkpoint，不会触发图的执行。如果你想从新状态继续执行，需要显式调用 `graph.invoke(None, config=new_config)`。

```python
# update_state 只修改状态，不执行
new_config = graph.update_state(config, {"topic": "新值"})
# 此时图没有继续执行

# 需要显式调用 invoke 继续执行
result = graph.invoke(None, config=new_config)
```

### 误区 3："不指定 as_node 也能正确路由" ❌

**事实**：在需要从中间状态继续执行的场景中，不指定 `as_node` 可能导致路由错误。LangGraph 需要知道"当前执行到了哪个节点"才能确定下一步走哪条边。

```python
# ❌ 可能路由错误
graph.update_state(some_historical_config, {"result": "新结果"})

# ✅ 明确指定节点
graph.update_state(some_historical_config, {"result": "新结果"}, as_node="search")
```

---

## 总结

### 核心要点

1. `update_state()` 创建新 checkpoint，不修改历史
2. 遵循 reducer 规则：无 reducer 覆盖，有 reducer 合并
3. `as_node` 参数影响路由和 reducer 行为
4. source 标记区分 `"update"`（基于最新）和 `"fork"`（基于历史）
5. 通常与 `get_state()` / `get_state_history()` 配合使用

### 下一步

理解了状态修改后，下一个核心概念将讲解 **时间旅行机制**——如何将 `get_state_history()` 和 `update_state()` 组合起来，实现 Replay（重放）和 Fork（分叉）两大时间旅行操作。

---

**参考资料**：
- [LangGraph 源码 - main.py](reference/source_状态快照_01.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [社区实践案例](reference/search_状态快照_01.md)
