# 核心概念2：Fan-out/Fan-in 机制

> 从一个节点分支到多个并行路径，再汇聚到一个节点的图结构设计模式

---

## 什么是 Fan-out/Fan-in？

**Fan-out/Fan-in** 是 LangGraph 中实现并行执行的核心图结构模式：

- **Fan-out（扇出）**：从一个节点分支到多个并行节点
- **Fan-in（扇入）**：多个并行节点汇聚到一个节点

这种模式允许工作流在某个阶段并行处理多个任务，然后在下一个阶段合并结果。

---

## 为什么需要 Fan-out/Fan-in？

### 核心问题

在复杂的工作流中，我们经常遇到这样的场景：

1. **并行任务执行**：多个独立任务可以同时执行，不需要等待
2. **结果汇总**：并行任务完成后，需要将结果合并处理
3. **性能优化**：通过并行执行减少总体执行时间

**传统串行方式的问题**：
```python
# 串行执行 - 慢
result1 = task1()  # 耗时 2 秒
result2 = task2()  # 耗时 2 秒
result3 = task3()  # 耗时 2 秒
# 总耗时：6 秒

# 并行执行 - 快
results = parallel_execute([task1, task2, task3])
# 总耗时：2 秒（假设任务独立）
```

---

## Fan-out/Fan-in 的两种实现方式

### 1. 静态并行（Static Parallel）

**定义**：在图编译时确定并行度，使用多条 `add_edge` 实现。

**特点**：
- ✅ 性能更好（编译时优化）
- ✅ 结构清晰，易于理解
- ❌ 并行度固定，不够灵活
- ❌ 无法根据运行时状态调整

**实现方式**：
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import operator
from typing import Annotated

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def node_a(state: State):
    print(f'Node A: {state["aggregate"]}')
    return {"aggregate": ["A"]}

def node_b(state: State):
    print(f'Node B: {state["aggregate"]}')
    return {"aggregate": ["B"]}

def node_c(state: State):
    print(f'Node C: {state["aggregate"]}')
    return {"aggregate": ["C"]}

def node_d(state: State):
    print(f'Node D: {state["aggregate"]}')
    return {"aggregate": ["D"]}

# 构建图
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("d", node_d)

# Fan-out: a -> b, c（静态并行）
builder.add_edge(START, "a")
builder.add_edge("a", "b")  # 第一条并行边
builder.add_edge("a", "c")  # 第二条并行边

# Fan-in: b, c -> d
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

graph = builder.compile()

# 执行
result = graph.invoke({"aggregate": []})
print(f"Final result: {result['aggregate']}")
# 输出：['A', 'B', 'C', 'D']
```

**执行流程**：
```
START
  ↓
  a (添加 'A')
  ├─→ b (添加 'B') ─┐
  └─→ c (添加 'C') ─┤
                    ↓
                    d (添加 'D')
                    ↓
                   END
```

**关键点**：
- 节点 `b` 和 `c` 在同一个超步（Superstep）中并行执行
- 节点 `d` 等待 `b` 和 `c` 都完成后再执行
- 使用 `operator.add` reducer 自动合并结果

---

### 2. 动态并行（Dynamic Parallel）

**定义**：在运行时根据状态动态创建并行任务，使用 `Send` API 实现。

**特点**：
- ✅ 灵活，可根据运行时状态调整并行度
- ✅ 支持 Map-Reduce 模式
- ✅ 可以发送不同的状态到不同节点
- ❌ 额外开销（动态创建任务）
- ❌ 内存压力（大量并行任务）

**实现方式**：
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

class JokeState(TypedDict):
    subject: str

def generate_topics(state: OverallState):
    """生成主题列表"""
    return {"subjects": ["cats", "dogs", "birds"]}

def generate_joke(state: JokeState):
    """为单个主题生成笑话"""
    subject = state["subject"]
    joke_map = {
        "cats": "Why don't cats play poker? Too many cheetahs!",
        "dogs": "Why do dogs run in circles? It's hard to run in squares!",
        "birds": "Why do birds fly south? It's too far to walk!"
    }
    return {"jokes": [joke_map.get(subject, f"Joke about {subject}")]}

def continue_to_jokes(state: OverallState):
    """动态创建并行任务"""
    # 返回多个 Send 对象，每个对象对应一个并行任务
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

def best_joke(state: OverallState):
    """选择最佳笑话"""
    print(f"All jokes: {state['jokes']}")
    return {"jokes": state["jokes"]}

# 构建图
builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

# 静态边
builder.add_edge(START, "generate_topics")

# 动态并行边：根据 subjects 数量动态创建并行任务
builder.add_conditional_edges(
    "generate_topics",
    continue_to_jokes,
    ["generate_joke"]
)

# Fan-in 边
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

graph = builder.compile()

# 执行
result = graph.invoke({"subjects": [], "jokes": []})
print(f"Final jokes: {result['jokes']}")
```

**执行流程**：
```
START
  ↓
generate_topics (生成 ["cats", "dogs", "birds"])
  ↓
continue_to_jokes (返回 3 个 Send 对象)
  ├─→ generate_joke(subject="cats") ─┐
  ├─→ generate_joke(subject="dogs") ─┤
  └─→ generate_joke(subject="birds")─┤
                                      ↓
                                  best_joke
                                      ↓
                                     END
```

**关键点**：
- `continue_to_jokes` 返回多个 `Send` 对象
- 每个 `Send` 对象指定目标节点和输入状态
- `generate_joke` 节点被调用 3 次，每次使用不同的 `subject`
- 使用 `Annotated[list[str], operator.add]` 自动合并所有笑话

---

## 静态并行 vs 动态并行对比

| 特性 | 静态并行 | 动态并行 |
|------|----------|----------|
| **并行度** | 编译时确定 | 运行时确定 |
| **灵活性** | 固定结构 | 动态调整 |
| **性能** | 更好（编译优化） | 稍慢（动态创建） |
| **内存** | 固定开销 | 取决于并行度 |
| **实现方式** | 多条 `add_edge` | `Send` API |
| **适用场景** | 固定数量的并行任务 | 数量不确定的并行任务 |
| **状态传递** | 共享相同状态 | 可传递不同状态 |

**选择建议**：
- **静态并行**：适合固定数量的并行任务（如：并行调用 3 个 LLM）
- **动态并行**：适合数量不确定的并行任务（如：Map-Reduce、多智能体协作）

---

## 图结构设计模式

### 模式1：简单扇出扇入

**场景**：固定数量的并行任务，结果需要汇总。

**结构**：
```
    A
   / \
  B   C
   \ /
    D
```

**代码**：
```python
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
```

**应用**：并行 LLM 调用、多数据源获取。

---

### 模式2：Map-Reduce

**场景**：对列表中的每个元素执行相同操作，然后汇总结果。

**结构**：
```
  Generate Topics
       ↓
  [Send, Send, Send]  (Map 阶段)
   ↓    ↓    ↓
  Process × N
       ↓
   Aggregate  (Reduce 阶段)
```

**代码**：
```python
def map_function(state):
    return [Send("process", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("generate", map_function, ["process"])
builder.add_edge("process", "aggregate")
```

**应用**：批量文档处理、多任务并行执行。

---

### 模式3：多层扇出扇入

**场景**：多个阶段的并行执行。

**结构**：
```
    A
   / \
  B   C
 / \ / \
D  E F  G
 \ | | /
    H
```

**代码**：
```python
# 第一层扇出
builder.add_edge("a", "b")
builder.add_edge("a", "c")

# 第二层扇出
builder.add_edge("b", "d")
builder.add_edge("b", "e")
builder.add_edge("c", "f")
builder.add_edge("c", "g")

# 扇入
builder.add_edge("d", "h")
builder.add_edge("e", "h")
builder.add_edge("f", "h")
builder.add_edge("g", "h")
```

**应用**：复杂的多阶段并行工作流。

---

### 模式4：条件扇出

**场景**：根据条件决定扇出到哪些节点。

**结构**：
```
    A
    ↓
  Condition
   / | \
  B  C  D  (根据条件选择)
   \ | /
    E
```

**代码**：
```python
def conditional_fanout(state):
    sends = []
    if state["need_b"]:
        sends.append(Send("b", state))
    if state["need_c"]:
        sends.append(Send("c", state))
    if state["need_d"]:
        sends.append(Send("d", state))
    return sends

builder.add_conditional_edges("a", conditional_fanout, ["b", "c", "d"])
builder.add_edge("b", "e")
builder.add_edge("c", "e")
builder.add_edge("d", "e")
```

**应用**：动态决策、智能路由。

---

## 完整实战示例：并行 LLM 调用

```python
"""
并行 LLM 调用示例
演示：同时生成笑话、故事、诗歌，然后汇总结果
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import os

# 模拟 LLM 调用
def mock_llm_call(prompt: str) -> str:
    """模拟 LLM 调用"""
    responses = {
        "joke": "Why don't scientists trust atoms? Because they make up everything!",
        "story": "Once upon a time, in a land far away, there lived a brave knight...",
        "poem": "Roses are red, violets are blue, LangGraph is awesome, and so are you!"
    }
    for key, value in responses.items():
        if key in prompt.lower():
            return value
    return "Generic response"

# ===== 1. 定义状态 =====
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

# ===== 2. 定义节点 =====
def generate_joke(state: State):
    """生成笑话"""
    print(f"[Joke] Generating joke about {state['topic']}...")
    joke = mock_llm_call(f"Write a joke about {state['topic']}")
    return {"joke": joke}

def generate_story(state: State):
    """生成故事"""
    print(f"[Story] Generating story about {state['topic']}...")
    story = mock_llm_call(f"Write a story about {state['topic']}")
    return {"story": story}

def generate_poem(state: State):
    """生成诗歌"""
    print(f"[Poem] Generating poem about {state['topic']}...")
    poem = mock_llm_call(f"Write a poem about {state['topic']}")
    return {"poem": poem}

def aggregate_results(state: State):
    """汇总结果"""
    print("[Aggregator] Combining all results...")
    combined = f"""
=== Content about {state['topic']} ===

JOKE:
{state['joke']}

STORY:
{state['story']}

POEM:
{state['poem']}
"""
    return {"combined_output": combined}

# ===== 3. 构建图 =====
builder = StateGraph(State)

# 添加节点
builder.add_node("generate_joke", generate_joke)
builder.add_node("generate_story", generate_story)
builder.add_node("generate_poem", generate_poem)
builder.add_node("aggregate", aggregate_results)

# Fan-out: START -> 三个生成节点（并行执行）
builder.add_edge(START, "generate_joke")
builder.add_edge(START, "generate_story")
builder.add_edge(START, "generate_poem")

# Fan-in: 三个生成节点 -> aggregate
builder.add_edge("generate_joke", "aggregate")
builder.add_edge("generate_story", "aggregate")
builder.add_edge("generate_poem", "aggregate")

# aggregate -> END
builder.add_edge("aggregate", END)

# 编译图
graph = builder.compile()

# ===== 4. 执行 =====
print("=== 并行 LLM 调用示例 ===\n")

result = graph.invoke({
    "topic": "artificial intelligence",
    "joke": "",
    "story": "",
    "poem": "",
    "combined_output": ""
})

print("\n=== 最终结果 ===")
print(result["combined_output"])
```

**运行输出**：
```
=== 并行 LLM 调用示例 ===

[Joke] Generating joke about artificial intelligence...
[Story] Generating story about artificial intelligence...
[Poem] Generating poem about artificial intelligence...
[Aggregator] Combining all results...

=== 最终结果 ===

=== Content about artificial intelligence ===

JOKE:
Why don't scientists trust atoms? Because they make up everything!

STORY:
Once upon a time, in a land far away, there lived a brave knight...

POEM:
Roses are red, violets are blue, LangGraph is awesome, and so are you!
```

---

## 性能考虑

### 1. 并行度控制

**问题**：过多的并行任务可能导致资源耗尽。

**解决方案**：
```python
def controlled_fanout(state):
    """控制并行度"""
    items = state["items"]
    max_parallel = 10  # 最大并行度

    # 只处理前 N 个
    return [
        Send("process", {"item": item})
        for item in items[:max_parallel]
    ]
```

### 2. 同步开销

**问题**：扇入节点需要等待所有并行节点完成。

**优化**：
- 减少不必要的并行任务
- 使用批处理减少任务数量
- 考虑使用流式处理

### 3. 内存使用

**问题**：每个并行任务都有独立的状态副本。

**优化**：
- 只传递必要的状态字段
- 使用引用而非复制大对象
- 及时清理不需要的状态

---

## 常见误区

### 误区1：扇入节点会被调用多次 ❌

**错误理解**：扇入节点会被每个并行节点调用一次。

**正确理解**：扇入节点只会被调用一次，等待所有并行节点完成后再执行。

```python
# 扇入节点只执行一次
def fan_in_node(state):
    print("This is called ONCE after all parallel nodes complete")
    return state
```

### 误区2：静态并行可以根据状态调整 ❌

**错误理解**：静态并行可以根据运行时状态调整并行度。

**正确理解**：静态并行的并行度在编译时确定，无法动态调整。需要动态调整请使用 `Send` API。

### 误区3：并行节点之间可以共享状态 ❌

**错误理解**：并行节点可以直接修改彼此的状态。

**正确理解**：并行节点之间的状态是隔离的，只能通过 reducer 函数在扇入时合并。

---

## 与前端开发的类比

| LangGraph 概念 | 前端类比 | 说明 |
|----------------|----------|------|
| Fan-out | `Promise.all()` | 同时发起多个异步请求 |
| Fan-in | `.then()` | 等待所有请求完成后处理 |
| 静态并行 | 固定的并行请求 | `Promise.all([api1(), api2()])` |
| 动态并行 | 动态生成请求 | `Promise.all(items.map(api))` |
| Reducer | 结果合并 | `results.reduce((acc, r) => [...acc, r])` |

**前端示例**：
```javascript
// Fan-out: 并行请求
const promises = [
  fetch('/api/joke'),
  fetch('/api/story'),
  fetch('/api/poem')
];

// Fan-in: 等待所有请求完成
Promise.all(promises)
  .then(responses => Promise.all(responses.map(r => r.json())))
  .then(results => {
    // 汇总结果
    console.log('All results:', results);
  });
```

---

## 日常生活类比

**Fan-out/Fan-in 就像餐厅的厨房**：

1. **Fan-out（扇出）**：
   - 主厨接到订单后，将任务分配给不同的厨师
   - 一个做汤，一个做主菜，一个做甜点
   - 三个厨师同时工作（并行执行）

2. **Fan-in（扇入）**：
   - 服务员等待所有菜品都准备好
   - 将所有菜品放在一个托盘上
   - 一起送到客人桌上

**静态并行 vs 动态并行**：
- **静态并行**：固定菜单，每次都是汤+主菜+甜点
- **动态并行**：根据客人点单，动态分配任务

---

## 总结

**Fan-out/Fan-in 机制是 LangGraph 实现并行执行的核心模式**：

1. **两种实现方式**：
   - 静态并行：编译时确定，性能更好
   - 动态并行：运行时确定，更灵活

2. **关键组件**：
   - Fan-out：使用多条边或 `Send` API
   - Fan-in：自动等待所有并行节点完成
   - Reducer：合并并行节点的结果

3. **应用场景**：
   - 并行 LLM 调用
   - Map-Reduce 工作流
   - 多数据源获取
   - 多智能体协作

4. **性能优化**：
   - 控制并行度
   - 减少同步开销
   - 优化内存使用

---

## 参考资料

**官方文档**：
- [LangGraph Branching](https://docs.langchain.com/oss/python/langgraph/how-tos/branching/)
- [Parallel Execution](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

**源码**：
- `langgraph/types.py` - Send 类定义
- `langgraph/graph/_branch.py` - 分支路由实现
- `langgraph/pregel/main.py` - Pregel 算法

**社区资源**：
- [LangGraph for Node.js Developers](https://medium.com/@ashithvl/langgraph-for-node-js-developers-the-ultimate-guide-a64d9494dddb)
- [Best practices for parallel nodes](https://forum.langchain.com/t/best-practices-for-parallel-nodes-fanouts/1900)

---

**下一步学习**：
- 核心概念3：状态合并与 Reducer
- 实战代码：Map-Reduce 工作流
- 实战代码：并行 LLM 调用
