# 核心概念3：Pregel 算法与执行流程

> **来源**：基于 LangGraph 源码分析和官方文档整理

---

## 一句话定义

**Pregel 是 LangGraph 的运行时引擎,基于 Google Pregel 算法的 BSP（批量同步并行）模型,通过 Actors 和 Channels 架构实现迭代式图计算,每个超步包含 Plan、Execution、Update 三个阶段。**

---

## 详细解释

### 1. Pregel 算法的起源

**Google Pregel**：
- Google 开发的大规模图处理算法
- 发表于 2010 年的论文
- 用于处理 Web 图、社交网络等大规模图数据
- 基于 BSP（Bulk Synchronous Parallel）模型

**LangGraph 的 Pregel**：
- 借鉴 Google Pregel 的核心思想
- 针对 AI 工作流进行优化
- 支持持久化、中断、流式输出等特性
- 集成 LangChain 生态系统

### 2. BSP 模型

**BSP（Bulk Synchronous Parallel）模型**：
- 批量同步并行计算模型
- 将计算组织成多个超步（supersteps）
- 每个超步包含三个阶段：计算、通信、同步

**在 LangGraph 中的应用**：
- 超步 = 图的一次迭代
- 计算 = 执行节点
- 通信 = 通过 Channels 传递数据
- 同步 = 等待所有节点完成

---

## Pregel 核心架构

### 1. Actors（执行者）

**定义**：图中的节点，执行计算任务。

**特性**：
- 每个 Actor 对应一个节点函数
- 从 Channels 读取数据
- 向 Channels 写入数据
- 可以并行执行

**示例**：
```python
def actor_a(state: State):
    # 从 state 读取数据
    count = state["count"]

    # 执行计算
    new_count = count + 1

    # 返回更新（写入 Channel）
    return {"count": new_count}
```

### 2. Channels（通道）

**定义**：用于在 Actors 之间传递数据的通道。

**类型**：
- **EphemeralValue**：临时值，每次读取后清空
- **LastValue**：保留最后一个值
- **BinaryOperatorAggregate**：聚合多个值（如求和）

**特性**：
- Actors 订阅 Channels
- Actors 写入 Channels
- Channels 管理数据流转

**示例**：
```python
from langgraph.channels import EphemeralValue, LastValue

channels = {
    "input": EphemeralValue(str),  # 临时输入
    "count": LastValue(int),       # 保留最后的计数
    "output": LastValue(str),      # 保留最后的输出
}
```

### 3. 执行流程图

```
开始
  ↓
初始化 Channels
  ↓
┌─────────────────────────────────┐
│  超步循环（Superstep Loop）      │
│                                 │
│  ┌─────────────────────────┐   │
│  │ 阶段1：Plan（计划）      │   │
│  │ - 确定要执行的 Actors    │   │
│  │ - 检查中断点            │   │
│  │ - 检查递归限制          │   │
│  └─────────────────────────┘   │
│           ↓                     │
│  ┌─────────────────────────┐   │
│  │ 阶段2：Execution（执行） │   │
│  │ - 并行执行 Actors        │   │
│  │ - 收集写入操作          │   │
│  └─────────────────────────┘   │
│           ↓                     │
│  ┌─────────────────────────┐   │
│  │ 阶段3：Update（更新）    │   │
│  │ - 应用写入到 Channels    │   │
│  │ - 保存 Checkpoint        │   │
│  │ - 流式输出状态          │   │
│  └─────────────────────────┘   │
│           ↓                     │
│  检查终止条件：                 │
│  - 没有 Actors 被选中？         │
│  - 达到递归限制？               │
│  - 遇到中断点？                 │
│           ↓                     │
└─────────────────────────────────┘
  ↓
返回最终结果
```

---

## 三个执行阶段详解

### 阶段1：Plan（计划）

**作用**：确定要执行哪些 Actors。

**步骤**：
1. 检查哪些 Channels 有新数据
2. 找出订阅这些 Channels 的 Actors
3. 检查中断点（interrupt_before）
4. 检查递归限制
5. 返回要执行的 Actors 列表

**伪代码**：
```python
def plan_phase(channels, actors, interrupt_before, recursion_count):
    # 1. 找出有新数据的 Channels
    updated_channels = [ch for ch in channels if ch.has_update()]

    # 2. 找出订阅这些 Channels 的 Actors
    actors_to_run = []
    for actor in actors:
        if actor.subscribes_to_any(updated_channels):
            actors_to_run.append(actor)

    # 3. 检查中断点
    for actor in actors_to_run:
        if actor.name in interrupt_before:
            raise GraphInterrupt(f"Interrupted before {actor.name}")

    # 4. 检查递归限制
    if recursion_count >= max_recursion:
        raise RecursionError("Max recursion limit reached")

    return actors_to_run
```

### 阶段2：Execution（执行）

**作用**：并行执行选定的 Actors。

**步骤**：
1. 从 Channels 读取数据
2. 并行执行 Actors
3. 收集写入操作
4. 处理异常

**伪代码**：
```python
def execution_phase(actors_to_run, channels):
    writes = []

    # 并行执行 Actors
    with ThreadPoolExecutor() as executor:
        futures = []
        for actor in actors_to_run:
            # 读取 Actor 订阅的 Channels
            input_data = {
                ch_name: channels[ch_name].get()
                for ch_name in actor.subscribes_to
            }

            # 提交执行任务
            future = executor.submit(actor.run, input_data)
            futures.append((actor, future))

        # 收集结果
        for actor, future in futures:
            try:
                result = future.result()
                writes.append((actor, result))
            except Exception as e:
                handle_error(actor, e)

    return writes
```

### 阶段3：Update（更新）

**作用**：使用 Actors 写入的值更新 Channels。

**步骤**：
1. 应用写入操作到 Channels
2. 保存 Checkpoint（如果配置）
3. 流式输出状态
4. 检查中断点（interrupt_after）

**伪代码**：
```python
def update_phase(writes, channels, checkpointer, interrupt_after):
    # 1. 应用写入操作
    for actor, result in writes:
        for ch_name, value in result.items():
            channels[ch_name].update(value)

    # 2. 保存 Checkpoint
    if checkpointer:
        checkpoint = create_checkpoint(channels)
        checkpointer.save(checkpoint)

    # 3. 流式输出状态
    current_state = {
        ch_name: ch.get()
        for ch_name, ch in channels.items()
    }
    yield current_state

    # 4. 检查中断点
    for actor, _ in writes:
        if actor.name in interrupt_after:
            raise GraphInterrupt(f"Interrupted after {actor.name}")
```

---

## 代码示例

### 示例1：基础 Pregel 执行

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int
    message: str

def increment(state: State):
    """Actor 1: 增加计数"""
    return {"count": state["count"] + 1}

def format_message(state: State):
    """Actor 2: 格式化消息"""
    return {"message": f"Count is {state['count']}"}

# 构建图
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_node("format", format_message)
builder.add_edge(START, "increment")
builder.add_edge("increment", "format")
builder.add_edge("format", END)

# 编译和执行
graph = builder.compile()
result = graph.invoke({"count": 0, "message": ""})

print(result)
# 输出: {"count": 1, "message": "Count is 1"}
```

**执行流程**：
```
超步1:
  Plan: 选择 increment（START 有数据）
  Execution: 执行 increment，返回 {"count": 1}
  Update: 更新 count Channel

超步2:
  Plan: 选择 format（count 有新数据）
  Execution: 执行 format，返回 {"message": "Count is 1"}
  Update: 更新 message Channel

超步3:
  Plan: 没有 Actors 被选中（到达 END）
  终止执行
```

### 示例2：循环执行

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int
    max_count: int

def increment(state: State):
    """增加计数"""
    return {"count": state["count"] + 1}

def should_continue(state: State) -> Literal["increment", "end"]:
    """条件路由：决定是否继续"""
    if state["count"] < state["max_count"]:
        return "increment"
    return "end"

# 构建图
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "increment": "increment",  # 循环
        "end": END
    }
)

# 编译和执行
graph = builder.compile()
result = graph.invoke({"count": 0, "max_count": 3})

print(result)
# 输出: {"count": 3, "max_count": 3}
```

**执行流程**：
```
超步1: count=0 → increment → count=1 → should_continue → "increment"
超步2: count=1 → increment → count=2 → should_continue → "increment"
超步3: count=2 → increment → count=3 → should_continue → "end"
终止执行
```

### 示例3：并行执行

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    result_a: str
    result_b: str
    final: str

def process_a(state: State):
    """并行任务 A"""
    return {"result_a": f"A processed: {state['input']}"}

def process_b(state: State):
    """并行任务 B"""
    return {"result_b": f"B processed: {state['input']}"}

def combine(state: State):
    """合并结果"""
    return {"final": f"{state['result_a']} | {state['result_b']}"}

# 构建图
builder = StateGraph(State)
builder.add_node("process_a", process_a)
builder.add_node("process_b", process_b)
builder.add_node("combine", combine)

# 并行执行 A 和 B
builder.add_edge(START, "process_a")
builder.add_edge(START, "process_b")

# 等待 A 和 B 完成后执行 combine
builder.add_edge("process_a", "combine")
builder.add_edge("process_b", "combine")
builder.add_edge("combine", END)

# 编译和执行
graph = builder.compile()
result = graph.invoke({"input": "test", "result_a": "", "result_b": "", "final": ""})

print(result)
# 输出: {"input": "test", "result_a": "A processed: test",
#        "result_b": "B processed: test", "final": "A processed: test | B processed: test"}
```

**执行流程**：
```
超步1:
  Plan: 选择 process_a 和 process_b（START 有数据）
  Execution: 并行执行 process_a 和 process_b
  Update: 更新 result_a 和 result_b Channels

超步2:
  Plan: 选择 combine（result_a 和 result_b 都有新数据）
  Execution: 执行 combine
  Update: 更新 final Channel

超步3:
  Plan: 没有 Actors 被选中（到达 END）
  终止执行
```

### 示例4：递归限制

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def increment(state: State):
    return {"count": state["count"] + 1}

# 构建无限循环图
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", "increment")  # 循环到自己

# 编译
graph = builder.compile()

# 执行（设置递归限制）
try:
    result = graph.invoke(
        {"count": 0},
        config={"recursion_limit": 10}  # 最多执行 10 次
    )
except RecursionError as e:
    print(f"达到递归限制: {e}")
```

---

## 在实际应用中的使用

### 1. 迭代式工作流

```python
# 适合：需要多次迭代的任务
class State(TypedDict):
    query: str
    answer: str
    quality_score: float

def generate_answer(state: State):
    # 生成答案
    answer = llm.invoke(state["query"])
    return {"answer": answer}

def evaluate_quality(state: State):
    # 评估质量
    score = evaluate(state["answer"])
    return {"quality_score": score}

def should_retry(state: State) -> Literal["generate", "end"]:
    # 质量不够则重试
    if state["quality_score"] < 0.8:
        return "generate"
    return "end"

# 构建迭代图
builder = StateGraph(State)
builder.add_node("generate", generate_answer)
builder.add_node("evaluate", evaluate_quality)
builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate",
    should_retry,
    {"generate": "generate", "end": END}
)

graph = builder.compile()
```

### 2. 并行处理

```python
# 适合：多个独立任务并行执行
class State(TypedDict):
    document: str
    summary: str
    keywords: list[str]
    sentiment: str

def summarize(state: State):
    return {"summary": llm.summarize(state["document"])}

def extract_keywords(state: State):
    return {"keywords": llm.extract_keywords(state["document"])}

def analyze_sentiment(state: State):
    return {"sentiment": llm.analyze_sentiment(state["document"])}

# 构建并行图
builder = StateGraph(State)
builder.add_node("summarize", summarize)
builder.add_node("keywords", extract_keywords)
builder.add_node("sentiment", analyze_sentiment)

# 并行执行
builder.add_edge(START, "summarize")
builder.add_edge(START, "keywords")
builder.add_edge(START, "sentiment")

builder.add_edge("summarize", END)
builder.add_edge("keywords", END)
builder.add_edge("sentiment", END)

graph = builder.compile()
```

### 3. 大规模图计算

```python
# 适合：处理复杂的图结构
class State(TypedDict):
    nodes: dict[str, dict]
    edges: list[tuple[str, str]]
    page_rank: dict[str, float]

def initialize_scores(state: State):
    # 初始化 PageRank 分数
    scores = {node: 1.0 for node in state["nodes"]}
    return {"page_rank": scores}

def update_scores(state: State):
    # 更新 PageRank 分数（一次迭代）
    new_scores = calculate_page_rank(
        state["nodes"],
        state["edges"],
        state["page_rank"]
    )
    return {"page_rank": new_scores}

def has_converged(state: State) -> Literal["update", "end"]:
    # 检查是否收敛
    if check_convergence(state["page_rank"]):
        return "end"
    return "update"

# 构建 PageRank 图
builder = StateGraph(State)
builder.add_node("initialize", initialize_scores)
builder.add_node("update", update_scores)
builder.add_edge(START, "initialize")
builder.add_edge("initialize", "update")
builder.add_conditional_edges(
    "update",
    has_converged,
    {"update": "update", "end": END}
)

graph = builder.compile()
```

---

## 循环控制机制

### 1. 条件边控制

```python
def should_continue(state: State) -> Literal["continue", "end"]:
    if state["count"] < 10:
        return "continue"
    return "end"

builder.add_conditional_edges(
    "node",
    should_continue,
    {"continue": "node", "end": END}
)
```

### 2. 递归限制控制

```python
# 设置最大步骤数
config = {"recursion_limit": 100}
result = graph.invoke(input_data, config=config)
```

### 3. 节点返回 None 停止循环

```python
def process(state: State):
    if state["count"] >= 10:
        return None  # 停止循环
    return {"count": state["count"] + 1}
```

---

## 性能优化

### 1. 并行执行优化

**优点**：
- 多个独立 Actors 并行执行
- 提高执行效率
- 充分利用多核 CPU

**示例**：
```python
# 并行执行多个独立任务
builder.add_edge(START, "task1")
builder.add_edge(START, "task2")
builder.add_edge(START, "task3")
```

### 2. Channel 类型选择

**EphemeralValue**：
- 临时值，读取后清空
- 适合一次性数据传递
- 节省内存

**LastValue**：
- 保留最后一个值
- 适合状态管理
- 支持多次读取

**BinaryOperatorAggregate**：
- 聚合多个值
- 适合累加、合并等操作

### 3. Checkpoint 优化

**异步持久化**：
```python
result = graph.invoke(
    input_data,
    config=config,
    durability="async"  # 异步持久化
)
```

**退出时持久化**：
```python
result = graph.invoke(
    input_data,
    config=config,
    durability="exit"  # 仅退出时持久化
)
```

---

## 关键技术点总结

### 1. Pregel 算法的核心

- **BSP 模型**：批量同步并行计算
- **超步迭代**：将执行组织成多个步骤
- **消息传递**：通过 Channels 传递数据
- **并行执行**：在每个超步中并行执行节点

### 2. 三个执行阶段

- **Plan**：确定要执行的 Actors
- **Execution**：并行执行 Actors
- **Update**：更新 Channels 和保存 Checkpoint

### 3. 循环控制

- 条件边引导到 END
- 递归限制防止无限循环
- 节点返回 None 停止循环

### 4. 并行执行

- 多个 Actors 可以并行执行
- 提高执行效率
- 适合大规模图计算

---

## 常见问题

### Q1：Pregel 和普通的图执行有什么区别？

**答**：Pregel 基于 BSP 模型，将执行组织成超步，支持并行执行和迭代计算。

**对比**：
| 特性 | 普通图执行 | Pregel 执行 |
|------|-----------|------------|
| 执行模式 | 顺序执行 | 批量并行 |
| 迭代支持 | 需要手动实现 | 内置支持 |
| 并行性 | 有限 | 高度并行 |
| 适用场景 | 简单流程 | 复杂迭代 |

### Q2：如何防止无限循环？

**答**：使用三种方法：
1. 条件边引导到 END
2. 设置递归限制
3. 节点返回 None

```python
# 方法1：条件边
builder.add_conditional_edges(
    "node",
    lambda s: "end" if s["count"] >= 10 else "continue",
    {"continue": "node", "end": END}
)

# 方法2：递归限制
config = {"recursion_limit": 100}

# 方法3：返回 None
def node(state):
    if state["count"] >= 10:
        return None
    return {"count": state["count"] + 1}
```

### Q3：如何实现并行执行？

**答**：让多个节点订阅相同的 Channel，它们会在同一个超步中并行执行。

```python
# 并行执行 A 和 B
builder.add_edge(START, "node_a")
builder.add_edge(START, "node_b")
```

### Q4：Pregel 的性能如何？

**答**：Pregel 适合迭代式图计算，性能取决于：
- 节点数量
- 并行度
- Checkpoint 频率
- 网络延迟（分布式场景）

**优化建议**：
- 减少 Checkpoint 频率
- 使用异步持久化
- 增加并行度
- 减少节点间依赖

### Q5：Pregel 和 LangChain 的 Chain 有什么区别？

**答**：
| 特性 | LangChain Chain | LangGraph Pregel |
|------|----------------|------------------|
| 执行模式 | 顺序执行 | 并行 + 迭代 |
| 循环支持 | 有限 | 内置支持 |
| 状态管理 | 简单 | 复杂（Checkpoint） |
| 中断恢复 | 不支持 | 支持 |
| 适用场景 | 简单流程 | 复杂工作流 |

---

## 参考来源

1. **源码分析**：
   - `libs/langgraph/langgraph/pregel/main.py` - Pregel 实现
   - `libs/langgraph/langgraph/pregel/_loop.py` - 执行循环
   - `libs/langgraph/langgraph/pregel/_algo.py` - 算法实现

2. **官方文档**：
   - https://docs.langchain.com/oss/python/langgraph/pregel - Pregel 运行时
   - https://docs.langchain.com/oss/python/langgraph/use-graph-api - 循环控制

3. **Context7 文档**：
   - LangGraph Pregel Runtime Overview
   - LangGraph Pregel: Cycle Example
   - Create and control loops

4. **学术论文**：
   - Malewicz et al. (2010). "Pregel: A System for Large-Scale Graph Processing"

---

## 下一步学习

- **实战代码**：基础编译与执行
- **实战代码**：流式执行与监控
- **实战代码**：持久化模式与性能优化
- **高级模式**：子图与模块化设计
