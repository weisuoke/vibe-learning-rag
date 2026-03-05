# 核心概念 5：动态路由 Send

## 概念定义

**动态路由 Send 是 LangGraph 中用于实现动态并行执行（fan-out）的机制，允许条件边根据状态动态地向多个目标节点发送自定义状态，实现 map-reduce 模式。**

## API 签名

```python
from langgraph.types import Send

# 创建 Send 对象
send_obj = Send(node: str, arg: Any)
```

**参数说明：**
- `node` (str): 目标节点的名称
- `arg` (Any): 发送给目标节点的状态数据

**返回值：**
- `Send` 对象，用于条件边的路由函数返回值

## 核心特征

### 1. 动态并行执行

Send 允许在运行时动态决定并行执行的节点数量和传递的状态：

```python
def fan_out_router(state: OverallState) -> list[Send]:
    """根据状态动态创建多个并行任务"""
    return [
        Send("process_task", {"task_id": i, "data": item})
        for i, item in enumerate(state["items"])
    ]
```

### 2. 自定义状态传递

每个 Send 对象可以携带不同的状态数据：

```python
# 不同的任务接收不同的状态
sends = [
    Send("summarize", {"topic": "AI"}),
    Send("summarize", {"topic": "ML"}),
    Send("summarize", {"topic": "DL"})
]
```

### 3. Map-Reduce 模式

Send 是实现 map-reduce 模式的核心机制：

```
                    ┌─→ Node A (state1) ─┐
Router (fan-out) ───┼─→ Node B (state2) ─┼─→ Aggregator
                    └─→ Node C (state3) ─┘
```

## 详细解释

### Send 的工作原理

**1. 条件边返回 Send 列表**

条件边的路由函数可以返回 `Send` 对象的列表，而不是简单的节点名称：

```python
from langgraph.graph import StateGraph
from langgraph.types import Send

def route_function(state: State) -> list[Send]:
    # 动态创建多个 Send 对象
    return [Send("target_node", custom_state) for custom_state in ...]
```

**2. 并行执行目标节点**

LangGraph 会为每个 Send 对象并行执行目标节点：

```python
# 如果返回 3 个 Send 对象，目标节点会被并行执行 3 次
sends = [
    Send("process", {"id": 1}),
    Send("process", {"id": 2}),
    Send("process", {"id": 3})
]
# → "process" 节点会被并行调用 3 次，每次接收不同的状态
```

**3. 状态聚合**

目标节点的输出会通过 reducer 函数聚合到主状态中：

```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    results: Annotated[list[str], operator.add]  # 使用 add reducer 聚合结果
```

### Send 与普通条件边的区别

| 特性 | 普通条件边 | Send 动态路由 |
|------|-----------|--------------|
| 返回值 | 节点名称字符串 | Send 对象列表 |
| 执行次数 | 目标节点执行 1 次 | 目标节点执行 N 次（N = Send 数量） |
| 状态传递 | 传递完整状态 | 每个 Send 携带自定义状态 |
| 并行执行 | 不支持 | 支持动态并行 |
| 使用场景 | 简单分支决策 | Map-reduce、批量处理 |

### 实现机制

**源码层面（基于 `_branch.py`）：**

```python
# langgraph/graph/_branch.py (简化版)
class BranchSpec:
    def _route(self, input: Any, config: RunnableConfig, ...) -> Runnable:
        # 执行路径函数
        result = self.path.invoke(value, config)

        # 处理 Send 对象
        if self.ends:
            destinations = [
                r if isinstance(r, Send) else self.ends[r]
                for r in result
            ]
        else:
            destinations = result

        # 检查不能发送到 END 节点
        if any(p.node == END for p in destinations if isinstance(p, Send)):
            raise InvalidUpdateError("Cannot send a packet to the END node")

        return destinations
```

**关键点：**
1. 路由函数返回的 `Send` 对象会被识别并处理
2. 每个 `Send` 对象指定目标节点和自定义状态
3. LangGraph 会为每个 `Send` 创建独立的执行任务
4. 不能向 `END` 节点发送 Send 对象

## 完整代码示例

### 示例 1：基础 Send 使用（主题并行摘要）

```python
"""
动态路由 Send - 基础示例
演示：使用 Send 实现多主题并行摘要
"""

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

# ===== 1. 定义状态结构 =====

class OverallState(TypedDict):
    """主状态：包含主题列表和摘要结果"""
    topics: list[str]
    summaries: Annotated[list[str], operator.add]  # 使用 add reducer 聚合

class TopicState(TypedDict):
    """子任务状态：单个主题"""
    topic: str

# ===== 2. 定义路由函数（fan-out） =====

def fan_out_to_topics(state: OverallState) -> list[Send]:
    """
    根据主题列表动态创建并行任务
    返回：Send 对象列表，每个 Send 对应一个主题
    """
    print(f"\n[Router] 收到 {len(state['topics'])} 个主题，开始 fan-out...")

    # 为每个主题创建一个 Send 对象
    return [
        Send("summarize_topic", {"topic": topic})
        for topic in state["topics"]
    ]

# ===== 3. 定义处理节点 =====

def summarize_topic(state: TopicState) -> dict:
    """
    处理单个主题（会被并行调用多次）
    """
    topic = state["topic"]
    print(f"  [Summarize] 正在处理主题: {topic}")

    # 模拟摘要生成
    summary = f"关于 {topic} 的摘要内容"

    return {"summaries": [summary]}

# ===== 4. 构建图 =====

builder = StateGraph(OverallState)

# 添加节点
builder.add_node("summarize_topic", summarize_topic)

# 添加条件边（从 START 到 summarize_topic，使用 Send）
builder.add_conditional_edges(
    START,
    fan_out_to_topics  # 路由函数返回 Send 列表
)

# summarize_topic 完成后回到 END
builder.add_edge("summarize_topic", END)

# 编译图
graph = builder.compile()

# ===== 5. 运行示例 =====

print("=== 示例 1：基础 Send 使用 ===\n")

result = graph.invoke({
    "topics": ["人工智能", "机器学习", "深度学习"],
    "summaries": []
})

print("\n[结果]")
print(f"主题数量: {len(result['topics'])}")
print(f"摘要数量: {len(result['summaries'])}")
print("\n生成的摘要:")
for i, summary in enumerate(result["summaries"], 1):
    print(f"  {i}. {summary}")
```

**运行输出：**
```
=== 示例 1：基础 Send 使用 ===

[Router] 收到 3 个主题，开始 fan-out...
  [Summarize] 正在处理主题: 人工智能
  [Summarize] 正在处理主题: 机器学习
  [Summarize] 正在处理主题: 深度学习

[结果]
主题数量: 3
摘要数量: 3

生成的摘要:
  1. 关于 人工智能 的摘要内容
  2. 关于 机器学习 的摘要内容
  3. 关于 深度学习 的摘要内容
```

### 示例 2：条件 Send（根据数据特征动态路由）

```python
"""
动态路由 Send - 条件路由示例
演示：根据数据特征将任务路由到不同的处理节点
"""

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

# ===== 1. 定义状态 =====

class BatchState(TypedDict):
    """批量任务状态"""
    items: list[dict]
    results: Annotated[list[str], operator.add]

class ItemState(TypedDict):
    """单个任务状态"""
    item_id: int
    value: int
    category: str

# ===== 2. 智能路由函数 =====

def smart_router(state: BatchState) -> list[Send]:
    """
    根据数据特征将任务路由到不同的处理节点
    - 高优先级任务 → fast_processor
    - 普通任务 → normal_processor
    """
    sends = []

    for i, item in enumerate(state["items"]):
        value = item["value"]

        # 根据值的大小决定路由目标
        if value > 50:
            target_node = "fast_processor"
            category = "high_priority"
        else:
            target_node = "normal_processor"
            category = "normal"

        sends.append(
            Send(target_node, {
                "item_id": i,
                "value": value,
                "category": category
            })
        )

    print(f"\n[Router] 路由了 {len(sends)} 个任务")
    return sends

# ===== 3. 处理节点 =====

def fast_processor(state: ItemState) -> dict:
    """快速处理器（处理高优先级任务）"""
    print(f"  [Fast] 处理任务 #{state['item_id']} (value={state['value']})")
    result = f"Fast处理: 任务{state['item_id']} → 结果{state['value'] * 2}"
    return {"results": [result]}

def normal_processor(state: ItemState) -> dict:
    """普通处理器"""
    print(f"  [Normal] 处理任务 #{state['item_id']} (value={state['value']})")
    result = f"Normal处理: 任务{state['item_id']} → 结果{state['value'] + 10}"
    return {"results": [result]}

# ===== 4. 构建图 =====

builder = StateGraph(BatchState)

# 添加两个处理节点
builder.add_node("fast_processor", fast_processor)
builder.add_node("normal_processor", normal_processor)

# 条件边：从 START 动态路由
builder.add_conditional_edges(START, smart_router)

# 两个处理器都连接到 END
builder.add_edge("fast_processor", END)
builder.add_edge("normal_processor", END)

graph = builder.compile()

# ===== 5. 运行示例 =====

print("=== 示例 2：条件 Send 路由 ===\n")

result = graph.invoke({
    "items": [
        {"value": 75},  # → fast_processor
        {"value": 25},  # → normal_processor
        {"value": 60},  # → fast_processor
        {"value": 30},  # → normal_processor
    ],
    "results": []
})

print("\n[结果]")
for result_str in result["results"]:
    print(f"  {result_str}")
```

**运行输出：**
```
=== 示例 2：条件 Send 路由 ===

[Router] 路由了 4 个任务
  [Fast] 处理任务 #0 (value=75)
  [Normal] 处理任务 #1 (value=25)
  [Fast] 处理任务 #2 (value=60)
  [Normal] 处理任务 #3 (value=30)

[结果]
  Fast处理: 任务0 → 结果150
  Normal处理: 任务1 → 结果35
  Fast处理: 任务2 → 结果120
  Normal处理: 任务3 → 结果40
```

### 示例 3：RAG 场景 - 多文档并行检索

```python
"""
动态路由 Send - RAG 实战示例
演示：根据查询动态检索多个文档源
"""

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

# ===== 1. 定义状态 =====

class RAGState(TypedDict):
    """RAG 主状态"""
    query: str
    document_sources: list[str]
    retrieved_docs: Annotated[list[dict], operator.add]

class RetrievalState(TypedDict):
    """单个检索任务状态"""
    query: str
    source: str

# ===== 2. 路由函数 =====

def route_to_sources(state: RAGState) -> list[Send]:
    """
    根据文档源列表创建并行检索任务
    """
    query = state["query"]
    sources = state["document_sources"]

    print(f"\n[Router] 查询: '{query}'")
    print(f"[Router] 将从 {len(sources)} 个源并行检索...")

    return [
        Send("retrieve_from_source", {
            "query": query,
            "source": source
        })
        for source in sources
    ]

# ===== 3. 检索节点 =====

def retrieve_from_source(state: RetrievalState) -> dict:
    """
    从单个文档源检索（会被并行调用）
    """
    query = state["query"]
    source = state["source"]

    print(f"  [Retrieve] 从 {source} 检索: '{query}'")

    # 模拟检索结果
    docs = [
        {
            "source": source,
            "content": f"来自 {source} 的相关文档内容",
            "score": 0.85
        }
    ]

    return {"retrieved_docs": docs}

# ===== 4. 构建图 =====

builder = StateGraph(RAGState)

builder.add_node("retrieve_from_source", retrieve_from_source)

builder.add_conditional_edges(START, route_to_sources)
builder.add_edge("retrieve_from_source", END)

graph = builder.compile()

# ===== 5. 运行示例 =====

print("=== 示例 3：RAG 多源并行检索 ===\n")

result = graph.invoke({
    "query": "什么是 LangGraph？",
    "document_sources": ["官方文档", "技术博客", "GitHub Issues"],
    "retrieved_docs": []
})

print("\n[检索结果]")
print(f"查询: {result['query']}")
print(f"检索到 {len(result['retrieved_docs'])} 个文档:\n")
for doc in result["retrieved_docs"]:
    print(f"  来源: {doc['source']}")
    print(f"  内容: {doc['content']}")
    print(f"  分数: {doc['score']}\n")
```

**运行输出：**
```
=== 示例 3：RAG 多源并行检索 ===

[Router] 查询: '什么是 LangGraph？'
[Router] 将从 3 个源并行检索...
  [Retrieve] 从 官方文档 检索: '什么是 LangGraph？'
  [Retrieve] 从 技术博客 检索: '什么是 LangGraph？'
  [Retrieve] 从 GitHub Issues 检索: '什么是 LangGraph？'

[检索结果]
查询: 什么是 LangGraph？
检索到 3 个文档:

  来源: 官方文档
  内容: 来自 官方文档 的相关文档内容
  分数: 0.85

  来源: 技术博客
  内容: 来自 技术博客 的相关文档内容
  分数: 0.85

  来源: GitHub Issues
  内容: 来自 GitHub Issues 的相关文档内容
  分数: 0.85
```

## 在 LangGraph 工作流中的应用

### 1. Map-Reduce 模式

**场景：** 批量数据处理

```python
def map_phase(state: State) -> list[Send]:
    """Map: 将数据分发到多个处理节点"""
    return [Send("process", {"data": item}) for item in state["items"]]

# Reduce 通过 Annotated reducer 自动完成
class State(TypedDict):
    results: Annotated[list, operator.add]  # 自动聚合
```

### 2. 动态并行任务

**场景：** 根据运行时条件决定并行度

```python
def dynamic_parallel(state: State) -> list[Send]:
    """根据数据量动态调整并行任务数"""
    batch_size = 10
    batches = [
        state["data"][i:i+batch_size]
        for i in range(0, len(state["data"]), batch_size)
    ]
    return [Send("process_batch", {"batch": b}) for b in batches]
```

### 3. 多策略并行执行

**场景：** 同时尝试多种策略，选择最佳结果

```python
def try_multiple_strategies(state: State) -> list[Send]:
    """并行尝试多种处理策略"""
    strategies = ["fast", "accurate", "balanced"]
    return [
        Send("execute_strategy", {"strategy": s, "data": state["data"]})
        for s in strategies
    ]
```

### 4. 分层处理

**场景：** 根据数据特征分层处理

```python
def hierarchical_routing(state: State) -> list[Send]:
    """根据优先级分层路由"""
    sends = []
    for item in state["items"]:
        if item["priority"] == "high":
            sends.append(Send("high_priority_handler", item))
        elif item["priority"] == "medium":
            sends.append(Send("medium_priority_handler", item))
        else:
            sends.append(Send("low_priority_handler", item))
    return sends
```

## 使用注意事项

### 1. 状态聚合

使用 `Annotated` 和 reducer 函数确保并行结果正确聚合：

```python
from typing_extensions import Annotated
import operator

class State(TypedDict):
    # ✅ 正确：使用 operator.add 聚合列表
    results: Annotated[list[str], operator.add]

    # ❌ 错误：没有 reducer，结果会被覆盖
    # results: list[str]
```

### 2. 不能发送到 END

```python
# ❌ 错误：不能向 END 发送 Send 对象
def bad_router(state: State) -> list[Send]:
    return [Send(END, state)]  # 会抛出 InvalidUpdateError

# ✅ 正确：Send 到普通节点，然后用边连接到 END
def good_router(state: State) -> list[Send]:
    return [Send("process", state)]

builder.add_edge("process", END)
```

### 3. 状态类型匹配

确保 Send 传递的状态与目标节点期望的状态类型匹配：

```python
class NodeState(TypedDict):
    task_id: int
    data: str

def router(state: State) -> list[Send]:
    # ✅ 正确：状态结构匹配
    return [Send("process", {"task_id": 1, "data": "test"})]

    # ❌ 错误：缺少必需字段
    # return [Send("process", {"task_id": 1})]
```

### 4. 性能考虑

并行执行数量会影响性能：

```python
def optimized_router(state: State) -> list[Send]:
    """限制并行任务数量"""
    MAX_PARALLEL = 10
    items = state["items"][:MAX_PARALLEL]  # 限制并行度
    return [Send("process", {"item": item}) for item in items]
```

## 与其他边类型的对比

| 特性 | 普通边 | 条件边（返回字符串） | 条件边（返回 Send） |
|------|--------|---------------------|-------------------|
| 路由方式 | 固定 | 动态（单目标） | 动态（多目标） |
| 并行执行 | 否 | 否 | 是 |
| 自定义状态 | 否 | 否 | 是 |
| 使用场景 | 线性流程 | 简单分支 | Map-reduce、批量处理 |
| 复杂度 | 低 | 中 | 高 |

## 最佳实践

### 1. 明确命名

```python
# ✅ 好的命名
def fan_out_to_documents(state: State) -> list[Send]:
    """清晰表达 fan-out 意图"""
    pass

# ❌ 不好的命名
def route(state: State) -> list[Send]:
    """命名过于通用"""
    pass
```

### 2. 添加日志

```python
def router_with_logging(state: State) -> list[Send]:
    """添加日志便于调试"""
    sends = [Send("process", {"id": i}) for i in range(10)]
    print(f"[Router] 创建了 {len(sends)} 个并行任务")
    return sends
```

### 3. 错误处理

```python
def safe_router(state: State) -> list[Send]:
    """处理空列表情况"""
    items = state.get("items", [])
    if not items:
        print("[Router] 警告：没有任务需要处理")
        return []
    return [Send("process", {"item": item}) for item in items]
```

### 4. 使用类型注解

```python
from typing import List

def typed_router(state: State) -> List[Send]:
    """使用类型注解提高代码可读性"""
    return [Send("process", {"id": i}) for i in range(10)]
```

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 源码分析**
   - 文件：`langgraph/graph/_branch.py`
   - 内容：Send 对象的实现机制和路由逻辑
   - 来源：`atom/langgraph/L1_图基础/07_边的类型与选择/reference/source_边的类型_01.md`

2. **Context7 官方文档**
   - 标题：Execute LangGraph nodes in parallel using Send
   - 来源：https://context7.com/langchain-ai/langgraph/llms.txt
   - 内容：Send 的标准使用方式和 map-reduce 模式示例
   - 引用位置：`atom/langgraph/L1_图基础/07_边的类型与选择/reference/context7_langgraph_02.txt`

3. **Context7 官方文档**
   - 标题：Implement Dynamic Routing with Conditional Edges in LangGraph
   - 来源：https://context7.com/langchain-ai/langgraph/llms.txt
   - 内容：条件边与 Send 的结合使用
   - 引用位置：`atom/langgraph/L1_图基础/07_边的类型与选择/reference/context7_langgraph_02.txt`

## 总结

**动态路由 Send 是 LangGraph 实现动态并行执行的核心机制：**

- **核心价值**：支持 map-reduce 模式和动态并行任务
- **关键特性**：自定义状态传递、运行时决定并行度
- **典型场景**：批量数据处理、多源检索、多策略并行
- **使用要点**：配合 Annotated reducer 聚合结果，注意状态类型匹配

Send 让 LangGraph 能够优雅地处理需要动态并行执行的复杂工作流场景。
