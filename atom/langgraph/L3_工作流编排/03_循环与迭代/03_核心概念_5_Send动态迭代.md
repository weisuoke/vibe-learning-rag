# 03_核心概念_5 - Send 动态迭代

> LangGraph 中通过 Send 类实现运行时动态迭代与 Map-Reduce 模式

---

## 概念定义

**Send 是 LangGraph 中实现动态迭代的核心类，它允许在运行时根据状态动态创建任意数量的并行任务，每个任务可以携带不同的输入数据。** 这是实现 Map-Reduce 模式的基础设施。

与静态边在编译时确定执行路径不同，Send 的核心价值在于：
1. 迭代次数在运行时才确定（由数据驱动）
2. 每次迭代可以接收不同的输入
3. 所有迭代自动并行执行，结果通过 Reducer 自动合并

---

## Send 类的定义

### 源码定义

```python
# [来源: sourcecode/langgraph/types.py L289-L361]
class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.

    One such example is a "map-reduce" workflow where your graph invokes
    the same node multiple times in parallel with different states,
    before aggregating the results back into the main graph's state.
    """

    __slots__ = ("node", "arg")

    node: str  # 目标节点名称
    arg: Any   # 发送给目标节点的输入数据

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

### 设计特点

**`__slots__` 轻量设计**: Send 使用 `__slots__` 而非普通属性，这意味着：
- 内存占用更小（没有 `__dict__`）
- 属性访问更快
- 适合大量创建（Map 阶段可能创建数百个 Send 对象）

**只有两个字段**:
- `node`: 字符串，指定目标节点
- `arg`: 任意类型，传递给目标节点的输入（通常是字典）

```python
from langgraph.types import Send

# 基本用法
Send("generate_joke", {"subject": "cats"})

# arg 可以是任意数据
Send("process", {"doc": document, "config": {"max_length": 500}})
Send("worker", [1, 2, 3])  # 列表也可以
```

---

## Send 在条件边中的使用

### 基本模式

在条件边函数中返回 Send 对象列表，LangGraph 会为每个 Send 创建一个独立的任务：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

def fan_out(state):
    """条件边函数：返回 Send 列表实现动态迭代"""
    return [
        Send("process", {"item": item})
        for item in state["items"]
    ]

builder = StateGraph(State)
builder.add_node("process", process_fn)
builder.add_conditional_edges("splitter", fan_out, ["process"])
```

### 底层处理机制

条件边返回 Send 对象时，经过 `_finish()` 方法处理：

```python
# [来源: sourcecode/langgraph/graph/_branch.py L192-L225]
def _finish(self, writer, input, result, config):
    if not isinstance(result, (list, tuple)):
        result = [result]
    if self.ends:
        destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
    else:
        destinations = cast(Sequence[Send | str], result)
    # 验证：不能 Send 到 END
    if any(p.node == END for p in destinations if isinstance(p, Send)):
        raise InvalidUpdateError("Cannot send a packet to the END node")
    entries = writer(destinations, False)
```

在 `attach_branch()` 中，Send 对象被写入 TASKS channel：

```python
# [来源: sourcecode/langgraph/graph/state.py L1323-L1370]
def get_writes(packets, static=False):
    writes = [
        (ChannelWriteEntry(
            p if p == END else _CHANNEL_BRANCH_TO.format(p), None
        ) if not isinstance(p, Send) else p)  # Send 对象直接传递
        for p in packets
    ]
    return writes
```

**关键**: Send 对象不写入 `branch:to:{node}` channel，而是写入 TASKS channel。TASKS channel 是一个特殊的 channel，用于存储动态创建的任务。

---

## Map-Reduce 模式

### 模式概述

```
         ┌─→ Send("worker", item_1) → worker(item_1) ─┐
         │                                              │
splitter ├─→ Send("worker", item_2) → worker(item_2) ─├─→ aggregator → END
         │                                              │
         └─→ Send("worker", item_3) → worker(item_3) ─┘

  Fan-out          Map (并行)              Fan-in (Reducer 自动合并)
```

**三个阶段**:
1. **Fan-out（扇出）**: 条件边函数根据状态创建多个 Send 对象
2. **Map（映射）**: 每个 Send 触发目标节点的一个独立实例，并行执行
3. **Fan-in（扇入）**: 所有实例的结果通过 Reducer（如 `operator.add`）自动合并回主状态

### 结果合并的关键：Reducer

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    items: list[str]
    # Reducer 是 Map-Reduce 的关键
    # operator.add 确保并行任务的结果被追加而非覆盖
    results: Annotated[list[str], operator.add]
```

**没有 Reducer 会怎样？** 多个并行任务同时写入同一个字段，后写入的会覆盖先写入的，导致数据丢失。`Annotated[list, operator.add]` 确保所有结果被追加到列表中。

---

## 完整实战代码

### 场景 1：文档批量处理（经典 Map-Reduce）

```python
"""
Map-Reduce 模式：并行处理多个文档，然后汇总结果
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====
class DocState(TypedDict):
    documents: list[str]                                # 输入：文档列表
    summaries: Annotated[list[dict], operator.add]      # Map 输出：摘要列表
    final_report: str                                   # Reduce 输出：最终报告


# ===== 2. Fan-out：动态创建并行任务 =====
def split_documents(state: DocState):
    """
    条件边函数：为每个文档创建一个 Send 对象
    返回 Send 列表 → LangGraph 自动并行执行
    """
    return [
        Send("process_doc", {"doc": doc, "index": i})
        for i, doc in enumerate(state["documents"])
    ]


# ===== 3. Map：并行处理每个文档 =====
def process_doc(state: dict) -> dict:
    """
    处理单个文档（并行执行）
    注意：接收的 state 是 Send 的 arg，不是全局状态
    """
    doc = state["doc"]
    index = state["index"]

    # 模拟文档处理（实际项目中调用 LLM）
    word_count = len(doc)
    keywords = doc.split()[:3]

    summary = {
        "index": index,
        "word_count": word_count,
        "keywords": keywords,
        "summary": f"文档{index}的核心内容: {doc[:50]}..."
    }

    # 返回结果，通过 operator.add 自动追加到 summaries 列表
    return {"summaries": [summary]}


# ===== 4. Fan-in / Reduce：汇总所有结果 =====
def aggregate_results(state: DocState) -> dict:
    """汇总所有并行任务的结果"""
    summaries = state["summaries"]

    total_words = sum(s["word_count"] for s in summaries)
    all_keywords = []
    for s in summaries:
        all_keywords.extend(s["keywords"])

    report = (
        f"处理完成！共 {len(summaries)} 个文档，"
        f"总字数 {total_words}，"
        f"关键词: {', '.join(set(all_keywords)[:5])}"
    )

    return {"final_report": report}


# ===== 5. 构建图 =====
builder = StateGraph(DocState)

# 添加节点
builder.add_node("split", lambda state: state)  # 占位节点
builder.add_node("process_doc", process_doc)
builder.add_node("aggregate", aggregate_results)

# 添加边
builder.add_edge(START, "split")

# 关键：条件边返回 Send 列表，触发并行执行
builder.add_conditional_edges(
    "split",
    split_documents,
    ["process_doc"]  # 声明可能的目标节点
)

# 所有 process_doc 完成后 → aggregate
builder.add_edge("process_doc", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()

# ===== 6. 执行 =====
result = graph.invoke({
    "documents": [
        "RAG 检索增强生成是一种结合检索和生成的技术架构",
        "LangGraph 提供了有状态的工作流编排能力",
        "向量数据库如 Milvus 和 ChromaDB 用于语义检索",
        "Embedding 模型将文本转换为高维向量表示",
    ]
})

print(f"报告: {result['final_report']}")
print(f"\n各文档摘要:")
for s in result["summaries"]:
    print(f"  文档{s['index']}: {s['summary']}")
```

**执行流程**:

```
1. START → split
   输入: {"documents": [doc1, doc2, doc3, doc4]}

2. split → split_documents (条件边)
   返回: [
       Send("process_doc", {"doc": doc1, "index": 0}),
       Send("process_doc", {"doc": doc2, "index": 1}),
       Send("process_doc", {"doc": doc3, "index": 2}),
       Send("process_doc", {"doc": doc4, "index": 3}),
   ]

3. 并行执行 process_doc (4次)
   每个实例接收不同的 {doc, index}
   每个实例返回 {"summaries": [summary_dict]}

   结果通过 operator.add 自动合并:
   {"summaries": [summary_0, summary_1, summary_2, summary_3]}

4. process_doc → aggregate
   汇总所有摘要，生成最终报告

5. aggregate → END
```

### 场景 2：动态查询扩展（RAG 应用）

```python
"""
RAG 场景：将一个复杂查询拆分为多个子查询，并行检索后合并结果
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====
class RAGState(TypedDict):
    original_query: str                                     # 原始查询
    sub_queries: list[str]                                  # 拆分后的子查询
    retrieved_docs: Annotated[list[dict], operator.add]     # 检索结果（自动合并）
    answer: str                                             # 最终回答


# ===== 2. 查询拆分 =====
def decompose_query(state: RAGState) -> dict:
    """将复杂查询拆分为多个子查询"""
    query = state["original_query"]

    # 模拟查询拆分（实际项目中用 LLM）
    if "和" in query or "与" in query:
        parts = query.replace("与", "和").split("和")
        sub_queries = [p.strip() + "是什么？" for p in parts]
    else:
        sub_queries = [query, f"{query}的应用场景", f"{query}的优缺点"]

    return {"sub_queries": sub_queries}


# ===== 3. Fan-out：为每个子查询创建检索任务 =====
def fan_out_retrieval(state: RAGState):
    """条件边：为每个子查询创建一个 Send"""
    return [
        Send("retrieve", {"query": q, "query_index": i})
        for i, q in enumerate(state["sub_queries"])
    ]


# ===== 4. Map：并行检索 =====
def retrieve(state: dict) -> dict:
    """为单个子查询执行检索（并行执行）"""
    query = state["query"]
    index = state["query_index"]

    # 模拟向量检索（实际项目中调用向量数据库）
    mock_docs = {
        0: [{"content": f"关于'{query}'的文档A", "score": 0.95}],
        1: [{"content": f"关于'{query}'的文档B", "score": 0.88}],
        2: [{"content": f"关于'{query}'的文档C", "score": 0.82}],
    }

    docs = mock_docs.get(index, [{"content": f"关于'{query}'的通用文档", "score": 0.7}])

    # 返回结果，通过 operator.add 合并
    return {"retrieved_docs": docs}


# ===== 5. Fan-in：合并检索结果并生成回答 =====
def generate_answer(state: RAGState) -> dict:
    """基于所有检索结果生成最终回答"""
    docs = state["retrieved_docs"]
    query = state["original_query"]

    # 按相关度排序
    sorted_docs = sorted(docs, key=lambda d: d["score"], reverse=True)

    # 模拟生成回答
    context = " | ".join(d["content"] for d in sorted_docs[:3])
    answer = f"基于 {len(docs)} 个检索结果回答'{query}': {context}"

    return {"answer": answer}


# ===== 6. 构建图 =====
builder = StateGraph(RAGState)

builder.add_node("decompose", decompose_query)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate_answer)

builder.add_edge(START, "decompose")
builder.add_conditional_edges("decompose", fan_out_retrieval, ["retrieve"])
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ===== 7. 执行 =====
result = graph.invoke({
    "original_query": "RAG和LangGraph的关系"
})

print(f"原始查询: {result['original_query']}")
print(f"子查询: {result['sub_queries']}")
print(f"检索到 {len(result['retrieved_docs'])} 个文档")
print(f"回答: {result['answer']}")
```

### 场景 3：递归分解（Recursive Decomposition）

```python
"""
递归分解：将大任务拆分为子任务，子任务可以继续拆分
结合 Send 的动态迭代和条件边的循环控制
"""
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===== 1. 定义状态 =====
class TaskState(TypedDict):
    tasks: list[str]                                    # 待处理任务
    results: Annotated[list[str], operator.add]         # 处理结果
    depth: int                                          # 当前递归深度


# ===== 2. 任务分发 =====
def dispatch(state: TaskState):
    """将任务列表分发为并行的 Send"""
    return [
        Send("worker", {"task": task, "depth": state.get("depth", 0)})
        for task in state["tasks"]
    ]


# ===== 3. 工作节点 =====
def worker(state: dict) -> dict:
    """处理单个任务"""
    task = state["task"]
    depth = state["depth"]

    # 模拟任务处理
    result = f"[深度{depth}] 完成: {task}"
    return {"results": [result]}


# ===== 4. 构建图 =====
builder = StateGraph(TaskState)

builder.add_node("dispatch_node", lambda s: s)
builder.add_node("worker", worker)
builder.add_node("done", lambda s: {"depth": s.get("depth", 0) + 1})

builder.add_edge(START, "dispatch_node")
builder.add_conditional_edges("dispatch_node", dispatch, ["worker"])
builder.add_edge("worker", "done")
builder.add_edge("done", END)

graph = builder.compile()

# ===== 5. 执行 =====
result = graph.invoke({
    "tasks": ["分析需求", "设计架构", "编写代码", "测试验证"],
    "depth": 0
})

print("处理结果:")
for r in result["results"]:
    print(f"  {r}")
```

---

## Send vs Command(goto=Send(...)) 对比

两种方式都能创建动态并行任务，但使用场景不同：

### Send 在条件边中

```python
# 在条件边函数中返回 Send 列表
def fan_out(state):
    return [Send("worker", {"item": i}) for i in state["items"]]

builder.add_conditional_edges("source", fan_out, ["worker"])
```

**特点**:
- 路由逻辑定义在图的边上（节点之间）
- 源节点不需要知道路由细节
- 更符合"关注点分离"原则

### Send 在 Command 中

```python
# 在节点函数中通过 Command 返回 Send
def dispatcher(state) -> Command[Literal["worker"]]:
    return Command(
        update={"status": "dispatching"},
        goto=[Send("worker", {"item": i}) for i in state["items"]]
    )
```

**特点**:
- 路由逻辑定义在节点内部
- 可以同时更新状态
- 适合节点需要"做一些处理后再分发"的场景

### 底层统一

两种方式最终都将 Send 对象写入 TASKS channel：

```python
# [来源: sourcecode/langgraph/graph/state.py L1510-L1512]
# Command 中的 Send
if isinstance(go, Send):
    rtn.append((TASKS, go))

# [来源: sourcecode/langgraph/graph/state.py L1323-L1340]
# 条件边中的 Send
writes = [
    p if isinstance(p, Send) else ...  # Send 直接传递到 TASKS
    for p in packets
]
```

---

## 迭代模式总结

### 模式 1：动态迭代（数据驱动）

迭代次数由输入数据决定：

```python
# 处理 N 个文档，N 在运行时才知道
def fan_out(state):
    return [Send("process", {"doc": d}) for d in state["docs"]]
```

### 模式 2：批量处理（固定模式，动态数量）

对一批数据执行相同操作：

```python
# 批量 Embedding
def batch_embed(state):
    chunks = state["chunks"]
    batch_size = 10
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    return [Send("embed_batch", {"batch": b, "batch_id": i})
            for i, b in enumerate(batches)]
```

### 模式 3：条件迭代（混合 Send + 字符串）

部分并行，部分串行：

```python
def smart_route(state):
    sends = [Send("process", {"item": i}) for i in state["items"]]
    sends.append("log")  # 同时触发 log 节点
    return sends
```

---

## 最佳实践

### 1. 始终使用 Reducer

```python
# 正确：使用 operator.add 合并并行结果
results: Annotated[list, operator.add]

# 错误：没有 Reducer，并行结果会互相覆盖
results: list  # 只保留最后一个任务的结果
```

### 2. 控制并行度

```python
def fan_out_with_limit(state):
    """限制最大并行数，避免资源耗尽"""
    items = state["items"]
    MAX_PARALLEL = 20

    if len(items) > MAX_PARALLEL:
        # 分批处理
        batch = items[:MAX_PARALLEL]
    else:
        batch = items

    return [Send("process", {"item": i}) for i in batch]
```

### 3. 错误处理

```python
def safe_worker(state: dict) -> dict:
    """在并行任务中捕获错误，避免整个图失败"""
    try:
        result = do_work(state["item"])
        return {"results": [{"status": "success", "data": result}]}
    except Exception as e:
        return {"results": [{"status": "error", "error": str(e)}]}
```

### 4. Send 的 arg 保持简洁

```python
# 好：只传递必要数据
Send("process", {"doc_id": doc.id, "content": doc.text})

# 不好：传递大量不需要的数据
Send("process", entire_state)  # 浪费内存
```

---

## 参考资料

### 源码
- `langgraph/types.py:289-361` - Send 类定义
- `langgraph/graph/_branch.py:192-225` - `_finish()` 处理 Send 路由
- `langgraph/graph/state.py:1323-1370` - `attach_branch()` 中 Send 写入 TASKS channel

### 文档
- [Map-Reduce with Send](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) - 官方 Map-Reduce 教程
- [Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#send-api) - Send API 参考

---

**版本**: v1.0
**最后更新**: 2026-02-28
**作者**: Claude Code
**知识点**: 循环与迭代 - Send 动态迭代
