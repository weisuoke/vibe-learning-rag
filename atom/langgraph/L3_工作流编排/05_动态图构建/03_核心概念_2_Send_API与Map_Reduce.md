# 核心概念2：Send API 与 Map-Reduce

## 一句话定义

**Send 是 LangGraph 实现运行时动态并行的核心机制——在条件边中返回 Send 对象列表，每个 Send 携带不同状态并行调用目标节点，再通过 Reducer 合并结果，构成完整的 Map-Reduce 模式。**

---

## 双重类比

### 前端类比：Promise.all 动态并发

Send 就像前端的 `Promise.all`，但任务数量在运行时才确定：

```javascript
// 前端：动态创建并行请求
const urls = getUrlsFromState(); // 运行时才知道有几个
const results = await Promise.all(
  urls.map(url => fetch(url)) // 每个请求携带不同参数
);
const merged = results.flat(); // 合并结果
```

对应到 LangGraph：
- `urls.map(url => fetch(url))` → `[Send("node", {"url": url}) for url in urls]`
- `Promise.all` → LangGraph 的并行执行引擎
- `results.flat()` → `Annotated[list, operator.add]` Reducer 合并

### 日常生活类比：老板分配任务

想象你是一个项目经理，收到一批需求：

1. **Map 阶段**：你把需求拆成独立任务，分给不同的开发者（每个 Send = 一个任务分配单）
2. **并行执行**：开发者们同时干活，互不干扰
3. **Reduce 阶段**：所有人完成后，你把结果汇总成一份报告（Reducer 合并）

关键点：你事先不知道有多少需求，任务数量是运行时才确定的。

---

## Send 类的工作原理

### Send(node, arg) 的两个参数

```python
from langgraph.types import Send

# node: 目标节点名称（字符串）
# arg:  传递给目标节点的状态（任意类型，通常是 dict）
send_obj = Send("generate_joke", {"subject": "猫"})
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `node` | `str` | 目标节点的名称，必须是图中已注册的节点 |
| `arg` | `Any` | 传递给目标节点的状态数据，每个 Send 可以不同 |

### 在条件边中返回 Send 列表

Send 不能单独使用，必须在条件边的路由函数中返回：

```python
def continue_to_jokes(state: OverallState):
    """路由函数：为每个主题创建一个 Send"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# 注册为条件边
graph.add_conditional_edges("collect_subjects", continue_to_jokes)
```

### 每个 Send 可以携带不同的状态

这是 Send 最强大的特性——不同于静态并行中所有节点共享同一个状态，Send 让每个并行任务拥有独立的输入：

```python
# 三个 Send，三份不同的状态
sends = [
    Send("process", {"doc_id": 1, "language": "zh"}),
    Send("process", {"doc_id": 2, "language": "en"}),
    Send("process", {"doc_id": 3, "language": "ja"}),
]
```

---

## Map-Reduce 模式详解

### 整体流程

```
                         Map 阶段（Send 创建并行任务）
                        ┌─→ generate_joke({"subject": "猫"}) ─┐
collect_subjects ───────┼─→ generate_joke({"subject": "狗"}) ─┼─→ best_joke (Reduce)
                        └─→ generate_joke({"subject": "鸟"}) ─┘
                                                          ↑
                                              Reducer: operator.add 合并列表
```

### Map 阶段：Send 创建多个并行任务

路由函数根据当前状态，动态生成 N 个 Send 对象。N 在编译时未知，完全由运行时数据决定：

```python
def fan_out(state: OverallState) -> list[Send]:
    """Map 阶段：将任务分发到多个并行节点"""
    subjects = state["subjects"]  # 运行时才知道有几个
    return [Send("generate_joke", {"subject": s}) for s in subjects]
```

### Reduce 阶段：通过 Reducer 合并结果

并行节点的输出需要通过 Reducer 函数合并回主状态。没有 Reducer，后执行的结果会覆盖先执行的：

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class OverallState(TypedDict):
    subjects: list[str]
    # 关键：operator.add 作为 Reducer，将多个列表拼接
    jokes: Annotated[list[str], operator.add]
```

### operator.add 作为列表 Reducer

`operator.add` 对列表执行拼接操作：

```python
import operator

# operator.add 对列表的效果
operator.add(["笑话1"], ["笑话2"])  # → ["笑话1", "笑话2"]
operator.add(["笑话1", "笑话2"], ["笑话3"])  # → ["笑话1", "笑话2", "笑话3"]
```

当三个并行节点分别返回 `{"jokes": ["笑话A"]}`、`{"jokes": ["笑话B"]}`、`{"jokes": ["笑话C"]}` 时，Reducer 会将它们合并为 `["笑话A", "笑话B", "笑话C"]`。

---

## 源码层面解释

### 1. types.py 中 Send 类的定义

```python
# langgraph/types.py（简化版，行 289-362）
class Send:
    """用于在条件边中动态调用节点的对象。

    允许向每个节点调用发送不同的状态，
    支持 map-reduce 工作流的并行节点调用。
    """
    node: str  # 目标节点名
    arg: Any   # 传递的状态/消息

    def __init__(self, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, _default_serializer(self.arg)))

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )
```

Send 是一个简单的数据类，核心就是 `node` + `arg` 两个字段。它实现了 `__hash__` 和 `__eq__`，这样 LangGraph 可以对 Send 对象进行去重和比较。

### 2. pregel/_algo.py 中 prepare_next_tasks 如何处理 PUSH 任务

当条件边返回 Send 对象时，LangGraph 将它们放入 TASKS 通道。`prepare_next_tasks` 函数负责从通道中取出 Send 对象并创建执行任务：

```python
# langgraph/pregel/_algo.py（简化版）
def prepare_next_tasks(checkpoint, processes, channels, ...):
    tasks = []

    # 1. 处理 PUSH 任务（来自 Send 对象）
    for idx, packet in enumerate(channels[TASKS]):
        if isinstance(packet, Send):
            # 为每个 Send 创建一个独立的执行任务
            node = packet.node
            arg = packet.arg
            task = PregelTask(node, arg, id=..., idx=idx)
            tasks.append(task)

    # 2. 处理 PULL 任务（由普通边触发的节点）
    for node_name, triggers in trigger_to_nodes.items():
        if any(channels_updated(t) for t in triggers):
            task = PregelTask(node_name, state, ...)
            tasks.append(task)

    return tasks
```

关键区别：
- **PUSH 任务**（Send）：每个 Send 对象创建一个独立任务，携带自定义状态
- **PULL 任务**（普通边）：由通道更新触发，使用共享状态

### 3. pregel/_write.py 中 ChannelWrite 如何处理 Send 对象

条件边的路由函数返回 Send 列表后，`ChannelWrite` 负责将它们写入 TASKS 通道：

```python
# langgraph/pregel/_write.py（简化版）
class ChannelWrite:
    def _write(self, values, config):
        for value in values:
            if isinstance(value, Send):
                # Send 对象写入 TASKS 通道，等待下一步执行
                channels[TASKS].append(value)
            else:
                # 普通值写入对应的状态通道
                channels[value.channel].update(value.value)
```

整个流程：
1. 条件边路由函数返回 `[Send(...), Send(...), ...]`
2. `ChannelWrite` 将 Send 对象写入 TASKS 通道
3. 下一个 BSP 步骤中，`prepare_next_tasks` 从 TASKS 通道取出 Send 对象
4. 为每个 Send 创建独立的执行任务，并行运行
5. 各任务的输出通过 Reducer 合并回主状态

---

## 完整代码示例：动态生成笑话（经典 Map-Reduce）

```python
"""
Send API 与 Map-Reduce 完整示例
场景：根据用户提供的主题列表，动态并行生成笑话，最后选出最佳笑话
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send


# ===== 1. 定义状态 =====

class OverallState(TypedDict):
    """主状态：管理整个 Map-Reduce 流程"""
    subjects: list[str]                              # 输入：主题列表
    jokes: Annotated[list[str], operator.add]         # Reduce：笑话列表（自动合并）
    best_joke: str                                    # 输出：最佳笑话


class JokeState(TypedDict):
    """子任务状态：单个笑话生成任务"""
    subject: str


# ===== 2. 定义节点函数 =====

def generate_joke(state: JokeState) -> dict:
    """
    Map 节点：为单个主题生成笑话
    这个函数会被并行调用 N 次（N = 主题数量）
    """
    subject = state["subject"]
    # 实际项目中这里会调用 LLM
    joke = f"为什么 {subject} 要过马路？因为它想到对面去！"
    print(f"  [Map] 生成笑话 - 主题: {subject}")
    return {"jokes": [joke]}


def select_best_joke(state: OverallState) -> dict:
    """
    Reduce 节点：从所有笑话中选出最佳
    """
    jokes = state["jokes"]
    print(f"\n  [Reduce] 从 {len(jokes)} 个笑话中选择最佳...")
    # 实际项目中这里会用 LLM 评分
    best = jokes[0] if jokes else "没有笑话"
    return {"best_joke": best}


# ===== 3. 定义路由函数（Map 阶段的分发器） =====

def fan_out_jokes(state: OverallState) -> list[Send]:
    """
    路由函数：为每个主题创建一个 Send 对象
    返回的 Send 列表长度 = 并行任务数量
    """
    subjects = state["subjects"]
    print(f"[Fan-out] 收到 {len(subjects)} 个主题，创建并行任务...")
    return [Send("generate_joke", {"subject": s}) for s in subjects]


# ===== 4. 构建图 =====

builder = StateGraph(OverallState)

# 添加节点
builder.add_node("generate_joke", generate_joke)
builder.add_node("select_best_joke", select_best_joke)

# Map 阶段：START → 条件边（Send）→ generate_joke（并行）
builder.add_conditional_edges(START, fan_out_jokes)

# Reduce 阶段：generate_joke → select_best_joke → END
builder.add_edge("generate_joke", "select_best_joke")
builder.add_edge("select_best_joke", END)

# 编译
graph = builder.compile()


# ===== 5. 运行 =====

if __name__ == "__main__":
    print("=== Send API Map-Reduce 示例 ===\n")

    result = graph.invoke({
        "subjects": ["程序员", "产品经理", "设计师", "测试工程师"],
        "jokes": [],
        "best_joke": "",
    })

    print(f"\n[结果]")
    print(f"  生成笑话数: {len(result['jokes'])}")
    print(f"  最佳笑话: {result['best_joke']}")
    print(f"\n  所有笑话:")
    for i, joke in enumerate(result["jokes"], 1):
        print(f"    {i}. {joke}")
```

**运行输出：**
```
=== Send API Map-Reduce 示例 ===

[Fan-out] 收到 4 个主题，创建并行任务...
  [Map] 生成笑话 - 主题: 程序员
  [Map] 生成笑话 - 主题: 产品经理
  [Map] 生成笑话 - 主题: 设计师
  [Map] 生成笑话 - 主题: 测试工程师

  [Reduce] 从 4 个笑话中选择最佳...

[结果]
  生成笑话数: 4
  最佳笑话: 为什么 程序员 要过马路？因为它想到对面去！

  所有笑话:
    1. 为什么 程序员 要过马路？因为它想到对面去！
    2. 为什么 产品经理 要过马路？因为它想到对面去！
    3. 为什么 设计师 要过马路？因为它想到对面去！
    4. 为什么 测试工程师 要过马路？因为它想到对面去！
```

---

## 与静态并行（多条边）的对比

### 静态并行：编译时确定

```python
# 静态并行：编译时就知道有 3 个并行分支
builder.add_edge(START, "fetch_news")
builder.add_edge(START, "fetch_weather")
builder.add_edge(START, "fetch_stocks")
```

### 动态并行：运行时确定（Send）

```python
# 动态并行：运行时才知道有几个任务
def fan_out(state):
    return [Send("fetch", {"source": s}) for s in state["sources"]]

builder.add_conditional_edges(START, fan_out)
```

### 对比表

| 特性 | 静态并行（多条边） | 动态并行（Send） |
|------|-------------------|-----------------|
| 并行数量 | 编译时固定 | 运行时动态决定 |
| 目标节点 | 不同节点 | 可以是同一个节点的多次调用 |
| 状态传递 | 共享同一个状态 | 每个 Send 携带独立状态 |
| 定义方式 | 多个 `add_edge` | 条件边返回 `Send` 列表 |
| 适用场景 | 固定的并行流水线 | 批量处理、Map-Reduce |
| 灵活性 | 低 | 高 |

---

## 实际应用场景

### 场景1：批量文档处理

```python
def fan_out_documents(state: State) -> list[Send]:
    """将文档列表分发到并行处理节点"""
    return [
        Send("process_document", {
            "doc_id": doc["id"],
            "content": doc["content"],
            "doc_type": doc["type"],
        })
        for doc in state["documents"]
    ]
```

### 场景2：多查询并行检索（RAG）

```python
def fan_out_queries(state: RAGState) -> list[Send]:
    """将多个查询变体并行发送到检索节点"""
    queries = state["query_variants"]  # 由 Query 改写生成
    return [
        Send("retrieve", {"query": q, "top_k": 5})
        for q in queries
    ]
```

### 场景3：多语言翻译

```python
def fan_out_translations(state: State) -> list[Send]:
    """将文本并行翻译成多种语言"""
    target_langs = state["target_languages"]
    return [
        Send("translate", {
            "text": state["source_text"],
            "target_lang": lang,
        })
        for lang in target_langs
    ]
```

---

## 使用注意事项

### 1. 必须配合 Reducer

```python
# ✅ 正确：使用 operator.add 合并并行结果
class State(TypedDict):
    results: Annotated[list[str], operator.add]

# ❌ 错误：没有 Reducer，后执行的结果会覆盖前面的
class State(TypedDict):
    results: list[str]
```

### 2. 不能 Send 到 END 节点

```python
# ❌ 会抛出 InvalidUpdateError
return [Send(END, {"data": "..."})]

# ✅ Send 到普通节点，再用边连接到 END
return [Send("process", {"data": "..."})]
builder.add_edge("process", END)
```

### 3. 空列表的处理

```python
def safe_fan_out(state: State) -> list[Send]:
    items = state.get("items", [])
    if not items:
        # 返回空列表时，不会有任何节点被执行
        # 需要确保下游逻辑能处理这种情况
        return []
    return [Send("process", {"item": item}) for item in items]
```

### 4. 子状态类型要匹配

```python
class TaskState(TypedDict):
    task_id: int
    data: str

# ✅ 传递的 dict 结构与 TaskState 匹配
Send("process", {"task_id": 1, "data": "hello"})

# ❌ 缺少必需字段，运行时可能出错
Send("process", {"task_id": 1})
```

---

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 源码分析** - `langgraph/types.py` Send 类定义（行 289-362）、`langgraph/pregel/_algo.py` PUSH 任务处理、`langgraph/pregel/_write.py` ChannelWrite 机制
   - 来源：`reference/source_dynamic_graph_01.md`

2. **Context7 官方文档** - Send API 基本用法、Map-Reduce 模式、并行执行与 Reducer
   - 来源：`reference/context7_langgraph_01.md`

3. **Dev.to 技术文章** - 条件边高级功能与并行执行实践
   - 来源：`reference/fetch_conditional_edges_01.md`

---

## 总结

Send API 是 LangGraph 动态并行的核心：在条件边中返回 `[Send(node, arg), ...]` 列表实现 Map，通过 `Annotated[list, operator.add]` Reducer 实现 Reduce。与静态并行不同，Send 的任务数量在运行时才确定，每个任务携带独立状态，适用于批量文档处理、多查询检索等场景。
