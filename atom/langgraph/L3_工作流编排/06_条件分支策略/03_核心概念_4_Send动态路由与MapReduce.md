# 核心概念 4：Send 动态路由与 Map-Reduce

## 概念定义

**Send 是 LangGraph 中向特定节点发送自定义状态的消息包，通过在条件边中返回 Send 列表，可以在运行时动态创建任意数量的并行任务，实现 Map-Reduce 模式。**

## API 签名

```python
from langgraph.types import Send

# 创建 Send 对象
send_obj = Send(node: str, arg: Any)
```

**参数说明：**
- `node` (str): 目标节点的名称，指定消息发送到哪个节点
- `arg` (Any): 发送给目标节点的自定义状态，可以与主图状态完全不同

**源码定义：**

```python
class Send:
    """A message or packet to send to a specific node in the graph.
    Used within conditional edges to dynamically invoke a node with custom state.
    Sent state can differ from core graph's state."""
    node: str  # 目标节点名
    arg: Any   # 发送给目标节点的自定义状态

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

## Send vs 普通路由

理解 Send 的关键在于它和普通条件边路由的本质区别：

```python
# ===== 普通路由：返回节点名字符串 =====
def normal_route(state):
    if state["score"] > 0.8:
        return "good_path"    # 只能选一条路
    return "bad_path"         # 传递的是完整的图状态

# ===== Send 路由：返回 Send 对象列表 =====
def send_route(state):
    return [
        Send("process", {"topic": "AI"}),      # 自定义状态
        Send("process", {"topic": "ML"}),       # 每个 Send 独立
        Send("process", {"topic": "DL"}),       # 并行执行
    ]
```

**核心区别：**

| 特性 | 普通路由 | Send 路由 |
|------|---------|----------|
| 返回值 | 节点名字符串 | Send 对象列表 |
| 并行度 | 单目标 | 多目标并行 |
| 状态传递 | 完整图状态 | 自定义子状态 |
| 任务数量 | 编译时确定 | 运行时动态决定 |
| 典型场景 | if-else 分支 | 批量处理、Map-Reduce |

## Map-Reduce 模式详解

Map-Reduce 是 Send 最核心的应用模式。理解这个模式，就理解了 Send 存在的意义。

### 什么是 Map-Reduce？

```
                         Map 阶段                    Reduce 阶段
                    ┌─→ process("cats") ──┐
输入 ──→ 条件边 ────┼─→ process("dogs") ──┼──→ 结果聚合 ──→ 输出
                    └─→ process("fish") ──┘
```

- **Map 阶段**：条件边返回 Send 列表，每个 Send 创建一个并行任务
- **Reduce 阶段**：通过 `Annotated[list, operator.add]` reducer 自动聚合所有并行任务的结果

### 状态隔离机制

Map-Reduce 模式的一个关键设计是**状态隔离**：子任务使用独立的 TypedDict，与主图状态分离。

```python
# 主图状态：包含全局信息
class OverallState(TypedDict):
    subjects: list[str]                              # 输入数据
    jokes: Annotated[list[str], operator.add]         # 聚合结果

# 子任务状态：只包含单个任务需要的信息
class JokeState(TypedDict):
    subject: str                                      # 单个主题
```

**为什么要状态隔离？**
- 子任务不需要知道全局状态的全貌
- 避免并行任务之间的状态干扰
- 每个 Send 的 `arg` 就是子任务的完整输入

### 完整 Map-Reduce 示例

```python
"""
Send 动态路由 - Map-Reduce 基础示例
演示：为多个主题并行生成笑话
"""

import operator
from typing import Annotated, TypedDict
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

# ===== 1. 定义状态 =====

class OverallState(TypedDict):
    """主图状态"""
    subjects: list[str]                              # 输入：主题列表
    jokes: Annotated[list[str], operator.add]         # 输出：笑话列表（自动聚合）

class JokeState(TypedDict):
    """子任务状态（与主图状态完全独立）"""
    subject: str                                      # 单个主题

# ===== 2. 定义 Map 函数（条件边路由） =====

def continue_to_jokes(state: OverallState):
    """Map 阶段：为每个主题创建一个 Send"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

# ===== 3. 定义处理节点 =====

def generate_joke(state: JokeState):
    """处理单个主题（并行执行多次）"""
    # 注意：接收的是 JokeState，不是 OverallState
    return {"jokes": [f"Joke about {state['subject']}"]}

# ===== 4. 构建图 =====

builder = StateGraph(OverallState)
builder.add_node("generate_joke", generate_joke)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)
graph = builder.compile()

# ===== 5. 运行 =====

result = graph.invoke({"subjects": ["cats", "dogs"]})
print(result)
# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
```

**执行流程解析：**

```
1. invoke({"subjects": ["cats", "dogs"]})
   │
2. START → 条件边调用 continue_to_jokes()
   │
3. 返回 [Send("generate_joke", {"subject": "cats"}),
   │      Send("generate_joke", {"subject": "dogs"})]
   │
4. 并行执行：
   │  ├─ generate_joke({"subject": "cats"}) → {"jokes": ["Joke about cats"]}
   │  └─ generate_joke({"subject": "dogs"}) → {"jokes": ["Joke about dogs"]}
   │
5. Reduce：operator.add 聚合 jokes 列表
   │  jokes = ["Joke about cats"] + ["Joke about dogs"]
   │
6. → END
```

## 高级用法

### 1. 多级 Map-Reduce（链式 Send）

当一个 Map-Reduce 的输出需要再次 fan-out 时，可以链式使用 Send：

```python
"""
多级 Map-Reduce 示例
演示：生成方案 → 并行评估 → 并行深化 → 汇总排名
"""

import operator
from typing import Annotated, TypedDict
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

# ===== 状态定义 =====

class OverallState(TypedDict):
    problem: str
    solutions: Annotated[list[str], operator.add]
    evaluations: Annotated[list[str], operator.add]
    final_ranking: str

class SolutionState(TypedDict):
    solution: str

class EvalState(TypedDict):
    evaluation: str

# ===== 节点函数 =====

def generate_solutions(state: OverallState):
    """生成多个候选方案"""
    problem = state["problem"]
    return {
        "solutions": [
            f"方案A: 针对'{problem}'的保守策略",
            f"方案B: 针对'{problem}'的激进策略",
            f"方案C: 针对'{problem}'的平衡策略",
        ]
    }

def fan_out_to_eval(state: OverallState):
    """第一级 Map：将每个方案分发到评估节点"""
    return [Send("evaluate", {"solution": s}) for s in state["solutions"]]

def evaluate(state: SolutionState):
    """评估单个方案"""
    return {"evaluations": [f"评估结果: {state['solution']} → 可行性 85%"]}

def rank_solutions(state: OverallState):
    """Reduce：汇总所有评估结果并排名"""
    all_evals = "\n".join(state["evaluations"])
    return {"final_ranking": f"最终排名:\n{all_evals}"}

# ===== 构建图 =====

builder = StateGraph(OverallState)
builder.add_node("generate_solutions", generate_solutions)
builder.add_node("evaluate", evaluate)
builder.add_node("rank_solutions", rank_solutions)

builder.add_edge(START, "generate_solutions")
builder.add_conditional_edges("generate_solutions", fan_out_to_eval)
builder.add_edge("evaluate", "rank_solutions")
builder.add_edge("rank_solutions", END)

graph = builder.compile()

# ===== 运行 =====

result = graph.invoke({"problem": "如何提升 RAG 检索质量"})
print(result["final_ranking"])
```

**执行流程：**

```
START → generate_solutions → [Send × 3] → evaluate(并行×3) → rank_solutions → END
         生成3个方案          Map阶段       并行评估            Reduce阶段
```

### 2. Send 与条件逻辑结合

Send 不仅可以做简单的 fan-out，还可以根据数据特征路由到不同节点：

```python
def smart_fan_out(state: OverallState):
    """根据内容特征路由到不同的处理节点"""
    sends = []
    for doc in state["documents"]:
        if doc["type"] == "pdf":
            sends.append(Send("process_pdf", {"doc": doc}))
        elif doc["type"] == "html":
            sends.append(Send("process_html", {"doc": doc}))
        else:
            sends.append(Send("process_text", {"doc": doc}))
    return sends
```

### 3. 动态子任务数量

Send 的强大之处在于子任务数量完全由运行时数据决定：

```python
def dynamic_chunking(state: OverallState):
    """根据文档长度动态决定分块数量"""
    text = state["document"]
    chunk_size = 500

    # 运行时计算分块数量
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    print(f"文档长度 {len(text)}，分成 {len(chunks)} 个块并行处理")

    return [
        Send("process_chunk", {"chunk": c, "index": i})
        for i, c in enumerate(chunks)
    ]
```

## Reducer 函数的选择

Reduce 阶段的行为完全取决于你选择的 reducer 函数：

```python
import operator
from typing import Annotated

# ===== 列表追加（最常用） =====
class State1(TypedDict):
    results: Annotated[list[str], operator.add]
    # Send A 返回 {"results": ["a"]}
    # Send B 返回 {"results": ["b"]}
    # 最终：results = ["a", "b"]

# ===== 数值累加 =====
class State2(TypedDict):
    total_score: Annotated[float, lambda a, b: a + b]
    # Send A 返回 {"total_score": 0.8}
    # Send B 返回 {"total_score": 0.9}
    # 最终：total_score = 1.7

# ===== 自定义聚合（取最大值） =====
class State3(TypedDict):
    best_score: Annotated[float, max]
    # Send A 返回 {"best_score": 0.8}
    # Send B 返回 {"best_score": 0.9}
    # 最终：best_score = 0.9

# ===== 字典合并 =====
def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

class State4(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

## 注意事项与常见陷阱

### 1. Send 不能发送到 END 节点

```python
# ❌ 错误：会抛出 InvalidUpdateError
def bad_router(state):
    return [Send(END, {"data": "done"})]

# ✅ 正确：Send 到处理节点，再用普通边连接 END
def good_router(state):
    return [Send("final_process", {"data": "done"})]

builder.add_edge("final_process", END)
```

### 2. Send 的 arg 必须是可序列化的

```python
# ❌ 错误：lambda 不可序列化
Send("process", {"callback": lambda x: x + 1})

# ❌ 错误：数据库连接不可序列化
Send("process", {"db": db_connection})

# ✅ 正确：使用基本数据类型
Send("process", {"id": 1, "name": "test", "scores": [0.8, 0.9]})
```

### 3. 没有 reducer 会导致结果丢失

```python
# ❌ 错误：没有 reducer，并行结果会互相覆盖
class BadState(TypedDict):
    results: list[str]  # 最后一个 Send 的结果会覆盖前面的

# ✅ 正确：使用 Annotated + reducer
class GoodState(TypedDict):
    results: Annotated[list[str], operator.add]  # 所有结果正确聚合
```

### 4. 空 Send 列表的处理

```python
def safe_router(state: OverallState):
    """处理可能为空的输入"""
    items = state.get("items", [])
    if not items:
        # 返回空列表时，图会直接跳过该分支
        # 需要确保后续节点能处理空结果
        return []
    return [Send("process", {"item": item}) for item in items]
```

### 5. 并行度控制

```python
def controlled_fan_out(state: OverallState):
    """限制最大并行数，避免资源耗尽"""
    MAX_PARALLEL = 20
    items = state["items"]

    if len(items) > MAX_PARALLEL:
        print(f"警告：{len(items)} 个任务超过上限，截取前 {MAX_PARALLEL} 个")
        items = items[:MAX_PARALLEL]

    return [Send("process", {"item": item}) for item in items]
```

## 在 RAG 中的典型应用

### 1. 多文档并行处理

```python
def fan_out_documents(state):
    """将多个文档分发到并行处理节点"""
    return [
        Send("process_document", {
            "doc_id": doc["id"],
            "content": doc["content"],
            "source": doc["source"]
        })
        for doc in state["documents"]
    ]
```

### 2. 多查询并行检索

```python
def fan_out_queries(state):
    """Query 改写后并行检索多个变体"""
    return [
        Send("retrieve", {"query": q, "top_k": 5})
        for q in state["rewritten_queries"]
    ]
```

### 3. Tree of Thoughts 并行推理

```python
def fan_out_thoughts(state):
    """为每个候选思路创建独立的推理分支"""
    return [
        Send("reason", {"thought": t, "depth": state["current_depth"] + 1})
        for t in state["candidate_thoughts"]
    ]
```

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 源码分析**
   - 文件：`langgraph/types.py` - Send 类定义
   - 文件：`langgraph/graph/_branch.py` - Send 路由执行机制
   - 来源：`reference/source_conditional_branching_01.md`

2. **Map-Reduce 教程**
   - 标题：Implementing Map-Reduce with LangGraph - Send API
   - 来源：`reference/fetch_map_reduce_01.md`

3. **LangGraph 最佳实践**
   - 来源：`reference/fetch_best_practices_01.md`

## 总结

**Send 是 LangGraph 实现动态并行的核心原语：**

- **本质**：向指定节点发送自定义状态的消息包
- **Map 阶段**：条件边返回 Send 列表，每个 Send 创建一个并行任务
- **Reduce 阶段**：通过 `Annotated[list, operator.add]` 等 reducer 自动聚合结果
- **状态隔离**：子任务使用独立的 TypedDict，与主图状态分离
- **动态性**：子任务数量在运行时由数据决定，而非编译时固定

掌握 Send + Map-Reduce 模式，就能优雅地处理 LangGraph 中所有需要动态并行的场景。