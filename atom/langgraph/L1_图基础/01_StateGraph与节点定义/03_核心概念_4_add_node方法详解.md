# 核心概念 4：add_node 方法详解

> 本文档是【01_StateGraph与节点定义】知识点的核心概念系列文档之一，专注于深入讲解 `add_node` 方法的设计与使用。

---

## 文档说明

**资料来源：**
- 源码分析：`sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py` (lines 289-783)
- Context7 官方文档：LangGraph 官方文档
- GitHub 教程：社区最佳实践
- Twitter/Reddit：实战经验总结

**本文档涵盖：**
- add_node 方法的 3 个重载版本
- 参数详细说明（defer, metadata, input_schema, retry_policy, cache_policy, destinations）
- 链式调用机制
- 节点命名规则
- 源码设计分析
- 实战代码示例

---

## 1. add_node 方法概述

### 1.1 方法定位

`add_node` 是 StateGraph 的核心方法之一，用于向图中添加节点。它是构建工作流的基础操作。

**核心特点：**
- **Builder 模式**：返回 `Self`，支持链式调用
- **类型安全**：通过泛型和重载提供类型推断
- **灵活配置**：支持多种节点类型和配置选项
- **自动命名**：可以自动推断节点名称

**来源：** `state.py:289-783`

### 1.2 方法签名总览

```python
# 重载 1：自动推断节点名称
def add_node(
    self,
    node: StateNode[NodeInputT, ContextT],
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
) -> Self: ...

# 重载 2：显式指定节点名称
def add_node(
    self,
    name: str,
    node: StateNode[NodeInputT, ContextT],
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
) -> Self: ...

# 重载 3：自定义 input_schema
def add_node(
    self,
    name: str,
    node: StateNode[NodeInputT, ContextT],
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: type[NodeInputT],
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
) -> Self: ...
```

**来源：** `state.py:289-354`

---

## 2. 三个重载版本详解

### 2.1 重载 1：自动推断节点名称

**使用场景：** 节点函数有明确的 `__name__` 属性时

```python
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

def my_node(state: State) -> dict:
    return {"x": state["x"] + 1}

builder = StateGraph(State)
builder.add_node(my_node)  # 节点名称自动推断为 "my_node"
builder.add_edge(START, "my_node")
```

**关键点：**
- 节点名称从函数的 `__name__` 属性获取
- 适用于普通函数和命名的 lambda
- Runnable 对象也会尝试推断名称

**来源：** `context7_langgraph_01.md:328-349`

### 2.2 重载 2：显式指定节点名称

**使用场景：** 需要自定义节点名称，或节点是匿名函数/Runnable

```python
builder = StateGraph(State)

# 显式命名
builder.add_node("step_1", my_node)

# 匿名函数必须显式命名
builder.add_node("increment", lambda state: {"x": state["x"] + 1})

# Runnable 对象显式命名
from langchain_core.runnables import RunnableLambda
runnable = RunnableLambda(lambda x: x + 1)
builder.add_node("runnable_node", runnable)
```

**关键点：**
- 名称必须是字符串
- 名称在图中必须唯一
- 推荐使用描述性名称（如 "generate_topic" 而非 "node1"）

**来源：** `context7_langgraph_01.md:306-323`

### 2.3 重载 3：自定义 input_schema

**使用场景：** 节点的输入类型与图的 state_schema 不同

```python
from typing_extensions import TypedDict

class GraphState(TypedDict):
    x: int
    y: int

class NodeInput(TypedDict):
    x: int  # 只需要部分字段

def specialized_node(state: NodeInput) -> dict:
    return {"y": state["x"] * 2}

builder = StateGraph(GraphState)
builder.add_node(
    "specialized",
    specialized_node,
    input_schema=NodeInput  # 显式指定输入类型
)
```

**关键点：**
- `input_schema` 必须是 `state_schema` 的子集
- 用于类型安全和性能优化
- 节点只接收声明的字段

**来源：** `source_StateGraph_01.md:110-142`

---

## 3. 参数详细说明

### 3.1 defer (bool)

**作用：** 控制节点是否延迟执行

```python
builder.add_node("expensive_task", expensive_node, defer=True)
```

**使用场景：**
- 长时间运行的任务
- 需要异步执行的操作
- 优化执行顺序

**默认值：** `False`

**来源：** `source_StateGraph_01.md:216-227`

### 3.2 metadata (dict[str, Any] | None)

**作用：** 为节点附加元数据，用于调试、监控和可视化

```python
builder.add_node(
    "llm_call",
    call_llm,
    metadata={
        "description": "调用 GPT-4 生成回复",
        "cost_per_call": 0.03,
        "timeout": 30
    }
)
```

**使用场景：**
- 节点描述和文档
- 成本追踪
- 性能监控
- 图可视化标注

**默认值：** `None`

**来源：** `source_StateGraph_01.md:216-227`

### 3.3 input_schema (type[NodeInputT] | None)

**作用：** 显式指定节点的输入类型

```python
class FullState(TypedDict):
    query: str
    context: list[str]
    answer: str

class RetrievalInput(TypedDict):
    query: str  # 只需要 query

builder.add_node(
    "retrieve",
    retrieve_docs,
    input_schema=RetrievalInput
)
```

**使用场景：**
- 节点只需要部分状态字段
- 提高类型安全性
- 优化数据传递

**默认值：** `None`（使用图的 state_schema）

**来源：** `source_StateGraph_01.md:110-142`

### 3.4 retry_policy (RetryPolicy | Sequence[RetryPolicy] | None)

**作用：** 配置节点失败时的重试策略

```python
from langgraph.retry import RetryPolicy

builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=3,
        backoff_factor=2.0,
        retry_on=(ConnectionError, TimeoutError)
    )
)
```

**使用场景：**
- 调用外部 API
- 网络请求
- 不稳定的操作

**默认值：** `None`（不重试）

**来源：** `source_StateGraph_01.md:216-227`

### 3.5 cache_policy (CachePolicy | None)

**作用：** 配置节点结果的缓存策略

```python
from langgraph.cache import CachePolicy

builder.add_node(
    "embedding",
    generate_embedding,
    cache_policy=CachePolicy(
        ttl=3600,  # 缓存 1 小时
        key_fn=lambda state: state["text"]
    )
)
```

**使用场景：**
- 昂贵的计算（如 Embedding 生成）
- 重复查询优化
- 减少 API 调用成本

**默认值：** `None`（不缓存）

**来源：** `source_StateGraph_01.md:216-227`

### 3.6 destinations (dict[str, str] | tuple[str, ...] | None)

**作用：** 用于图可视化，标注节点的可能目标节点

```python
builder.add_node(
    "router",
    route_query,
    destinations=("search", "qa", "chitchat")
)
```

**使用场景：**
- 图可视化
- 文档生成
- 静态分析

**注意：** 这是提示性信息，不影响实际执行逻辑

**默认值：** `None`

**来源：** `source_StateGraph_01.md:216-227`

---

## 4. 链式调用机制

### 4.1 返回值设计

`add_node` 返回 `Self`，支持链式调用：

```python
builder = (
    StateGraph(State)
    .add_node("step_1", step_1)
    .add_node("step_2", step_2)
    .add_node("step_3", step_3)
    .add_edge(START, "step_1")
    .add_edge("step_1", "step_2")
    .add_edge("step_2", "step_3")
)
```

**来源：** `source_StateGraph_01.md:112-127`

### 4.2 链式调用最佳实践

**推荐：** 按逻辑分组

```python
# 先添加所有节点
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate", generate)

# 再添加所有边
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", END)
```

**来源：** `search_StateGraph_03.md:125-139`

---

## 5. 节点命名规则

### 5.1 命名约定

**推荐命名风格：**
- 使用动词短语：`generate_topic`, `write_joke`, `retrieve_docs`
- 使用下划线分隔：`call_llm` 而非 `callLLM`
- 描述性命名：`rerank_results` 而非 `step_2`

**来源：** `search_StateGraph_02.md:78-89`

### 5.2 命名冲突处理

```python
# ❌ 错误：重复添加同名节点
builder.add_node("process", process_v1)
builder.add_node("process", process_v2)  # 会覆盖前一个

# ✅ 正确：使用不同名称
builder.add_node("process_v1", process_v1)
builder.add_node("process_v2", process_v2)
```

**来源：** `search_StateGraph_03.md:84-90`

### 5.3 特殊节点名称

**保留名称：**
- `START`：图的入口点（不能作为节点名）
- `END`：图的出口点（不能作为节点名）

```python
# ❌ 错误
builder.add_node("START", my_node)  # 会报错

# ✅ 正确
builder.add_edge(START, "my_node")
```

**来源：** `context7_langgraph_01.md:143-169`

---

## 6. 源码设计分析

### 6.1 StateNodeSpec 数据结构

节点内部使用 `StateNodeSpec` 存储：

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]  # 节点函数或 Runnable
    metadata: dict[str, Any] | None            # 元数据
    input_schema: type[NodeInputT]             # 输入类型
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None  # destinations
    defer: bool                                # 是否延迟执行
```

**来源：** `source_StateGraph_01.md:216-227`

### 6.2 节点存储机制

```python
# 内部实现（简化版）
class StateGraph:
    def __init__(self, state_schema):
        self.nodes: dict[str, StateNodeSpec] = {}

    def add_node(self, name_or_node, node=None, **kwargs):
        # 推断名称
        if node is None:
            node = name_or_node
            name = self._infer_name(node)
        else:
            name = name_or_node

        # 创建 StateNodeSpec
        spec = StateNodeSpec(
            runnable=node,
            metadata=kwargs.get("metadata"),
            input_schema=kwargs.get("input_schema") or self.state_schema,
            retry_policy=kwargs.get("retry_policy"),
            cache_policy=kwargs.get("cache_policy"),
            ends=kwargs.get("destinations"),
            defer=kwargs.get("defer", False)
        )

        # 存储节点
        self.nodes[name] = spec
        return self
```

**来源：** `source_StateGraph_01.md:183-195`

---

## 7. 实战代码示例

### 7.1 基础用法

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[str]

def add_greeting(state: State) -> dict:
    return {"messages": state["messages"] + ["Hello!"]}

def add_farewell(state: State) -> dict:
    return {"messages": state["messages"] + ["Goodbye!"]}

# 构建图
builder = StateGraph(State)
builder.add_node(add_greeting)      # 自动命名为 "add_greeting"
builder.add_node(add_farewell)      # 自动命名为 "add_farewell"
builder.add_edge(START, "add_greeting")
builder.add_edge("add_greeting", "add_farewell")
builder.add_edge("add_farewell", END)

graph = builder.compile()
result = graph.invoke({"messages": []})
print(result)  # {"messages": ["Hello!", "Goodbye!"]}
```

**来源：** `context7_langgraph_01.md:328-349`

### 7.2 异步节点

```python
import asyncio

async def async_node(state: State) -> dict:
    await asyncio.sleep(1)  # 模拟异步操作
    return {"messages": state["messages"] + ["Async done"]}

builder.add_node(async_node)  # LangGraph 自动处理异步
```

**来源：** `search_StateGraph_03.md:43-54`

### 7.3 Runnable 节点

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# LangChain Runnable
llm = ChatOpenAI(model="gpt-4")
builder.add_node("llm", llm)

# 自定义 Runnable
custom_runnable = RunnableLambda(lambda x: {"result": x["input"].upper()})
builder.add_node("uppercase", custom_runnable)
```

**来源：** `context7_langgraph_01.md:232-261`

### 7.4 中间件模式

```python
def logging_middleware(state: State) -> dict:
    """日志记录中间件"""
    print(f"[LOG] Current state: {state}")
    return {}  # 不修改状态

builder.add_node("logger", logging_middleware)
builder.add_edge("step_1", "logger")
builder.add_edge("logger", "step_2")
```

**来源：** `search_StateGraph_03.md:57-66`

---

## 8. 使用场景总结

### 8.1 常见模式

| 场景 | 推荐用法 | 示例 |
|------|----------|------|
| 简单函数节点 | 自动命名 | `builder.add_node(my_func)` |
| 匿名函数 | 显式命名 | `builder.add_node("inc", lambda s: {...})` |
| LLM 调用 | Runnable + 元数据 | `builder.add_node("llm", llm, metadata={...})` |
| 外部 API | 重试策略 | `builder.add_node("api", call_api, retry_policy=...)` |
| 昂贵计算 | 缓存策略 | `builder.add_node("embed", embed, cache_policy=...)` |
| 部分状态 | input_schema | `builder.add_node("node", fn, input_schema=Partial)` |

### 8.2 最佳实践

1. **优先使用自动命名**：函数名即文档
2. **添加元数据**：便于调试和监控
3. **配置重试**：提高系统鲁棒性
4. **使用缓存**：优化性能和成本
5. **类型安全**：使用 input_schema 明确输入

**来源：** `search_StateGraph_02.md:42-97`

---

## 9. 常见问题

### Q1: 节点名称冲突怎么办？

**A:** 节点名称必须唯一，重复添加会覆盖前一个节点。建议使用描述性名称避免冲突。

**来源：** `search_StateGraph_03.md:84-90`

### Q2: 异步节点需要特殊处理吗？

**A:** 不需要，LangGraph 自动检测并处理异步节点。

**来源：** `search_StateGraph_03.md:43-54`

### Q3: 可以动态添加节点吗？

**A:** 不可以，节点必须在 `compile()` 之前添加。编译后图结构不可变。

**来源：** `context7_langgraph_01.md:172-183`

### Q4: 节点函数必须返回 dict 吗？

**A:** 是的，节点函数必须返回 dict，表示状态的部分更新。

**来源：** `context7_langgraph_01.md:86-120`

---

## 10. 参考资源

### 官方文档
- LangGraph 官方文档：https://docs.langchain.com/oss/python/langgraph/
- StateGraph API 参考：https://docs.langchain.com/oss/python/langgraph/graph-api

### 源码位置
- `langgraph/graph/state.py` (lines 289-783)
- `langgraph/graph/_node.py` (节点协议定义)

### 社区资源
- LangChain Academy：https://github.com/langchain-ai/langchain-academy
- LangGraph 示例：https://github.com/langchain-ai/langgraph/tree/main/examples

---

**文档版本：** v1.0
**最后更新：** 2026-02-25
**维护者：** Claude Code
