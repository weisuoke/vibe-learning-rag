# 03_核心概念_1 - add_conditional_edges 基础

> LangGraph 条件分支的核心 API：通过路由函数在运行时动态选择执行路径

---

## 概念定义

**`add_conditional_edges()` 是 StateGraph 的核心方法，它将一个路由函数绑定到源节点，使得源节点执行完毕后，根据路由函数的返回值动态决定下一个要执行的节点。**

这是 LangGraph 中使用频率最高的条件分支 API，覆盖了绝大多数的动态路由场景：ReAct Agent 的工具调用循环、RAG 的检索质量判断、多 Agent 的任务分发等。

---

## API 签名详解

### 完整签名

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
        | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
```

### 参数 1：`source` — 源节点名称

**类型**：`str`

条件边从哪个节点出发。当这个节点执行完毕后，路由函数会被调用。

```python
# source 就是"在哪个路口设置红绿灯"
builder.add_conditional_edges(
    "agent",      # ← 当 agent 节点执行完后，触发路由判断
    should_continue
)
```

**注意事项**：
- source 必须是已通过 `add_node()` 添加的节点名
- 一个节点可以同时有普通边和条件边（但要注意冲突）
- source 不能是 `END`

### 参数 2：`path` — 路由函数

**类型**：`Callable[..., Hashable | Sequence[Hashable]]` 或 `Runnable`

路由函数是条件分支的灵魂。它接收当前状态，返回下一个要执行的节点名（或 Send 对象列表）。

```python
# 路由函数的基本形式
def should_continue(state: AgentState) -> str:
    """根据状态决定下一步"""
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

**路由函数可以是**：
1. 普通同步函数：`def route(state) -> str`
2. 异步函数：`async def route(state) -> str`
3. LangChain Runnable 对象：实现了 `invoke()` 方法的对象

**路由函数的输入**：
- 接收当前节点执行后的最新 state
- 编译时，LangGraph 会用 `reader` 读取最新状态传给路由函数

**路由函数的返回值**：
- 单个字符串：目标节点名（如 `"tools"`、`"__end__"`）
- 单个 Send 对象：`Send("node", custom_state)`
- 列表：多个字符串或 Send 对象的混合列表（触发并行执行）

### 参数 3：`path_map` — 路径映射

**类型**：`dict[Hashable, str] | list[str] | None`

path_map 将路由函数的返回值映射到实际的节点名。它有三种形式：

#### 形式 1：`None`（默认，自动推断）

```python
# 不传 path_map，LangGraph 自动从 Literal 类型注解推断
def route(state) -> Literal["tools", "__end__"]:
    if state["should_call_tools"]:
        return "tools"
    return END

builder.add_conditional_edges("agent", route)
# LangGraph 自动推断出可能的目标：["tools", "__end__"]
```

**自动推断的源码逻辑**：

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py]
# 从函数返回类型的 Literal 注解中提取可能的目标
if rtn_type := get_type_hints(func).get("return"):
    if get_origin(rtn_type) is Literal:
        path_map_ = {name: name for name in get_args(rtn_type)}
```

**优点**：代码简洁，类型安全，IDE 有提示
**缺点**：必须使用 Literal 类型注解

#### 形式 2：`dict`（键值映射）

```python
# 路由函数返回自定义键，path_map 映射到实际节点名
def route(state) -> str:
    if state["intent"] == "search":
        return "need_search"      # 自定义键
    return "direct_answer"        # 自定义键

builder.add_conditional_edges("classifier", route, {
    "need_search": "retriever",   # 自定义键 → 实际节点名
    "direct_answer": "generator", # 自定义键 → 实际节点名
})
```

**优点**：解耦路由逻辑和节点命名，路由函数不需要知道节点名
**缺点**：多一层映射，需要维护一致性

#### 形式 3：`list`（名称列表）

```python
# 等价于 {name: name for name in list}
builder.add_conditional_edges("agent", route, ["tools", "end"])
# 等价于
builder.add_conditional_edges("agent", route, {"tools": "tools", "end": "end"})
```

**优点**：比 dict 简洁
**缺点**：不能做键值映射，只是声明合法目标

---

## 路由函数的三种写法

### 写法 1：返回字符串节点名

最简单、最常用的写法。路由函数直接返回目标节点的名称。

```python
"""
写法 1：返回字符串节点名
适用场景：简单的二选一或多选一路由

图拓扑：
    START → agent → (条件判断)
                      ↓ 有工具调用    ↓ 无工具调用
                    tools            END
                      ↓
                    agent（循环）
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    tool_calls_count: int


def agent(state: State):
    count = state.get("tool_calls_count", 0)
    if count < 2:
        return {
            "messages": [f"Agent: 需要调用工具 (第{count+1}次)"],
            "tool_calls_count": count + 1,
        }
    return {
        "messages": ["Agent: 任务完成，直接回答"],
        "tool_calls_count": count,
    }


def tools(state: State):
    return {"messages": ["Tool: 执行完毕，返回结果"]}


# 路由函数：返回字符串节点名
def should_continue(state: State) -> str:
    last_msg = state["messages"][-1]
    if "需要调用工具" in last_msg:
        return "tools"    # 返回节点名字符串
    return "__end__"      # "__end__" 是 END 的字符串形式


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()
result = graph.invoke({"messages": [], "tool_calls_count": 0})
print(result["messages"])
# ['Agent: 需要调用工具 (第1次)', 'Tool: 执行完毕，返回结果',
#  'Agent: 需要调用工具 (第2次)', 'Tool: 执行完毕，返回结果',
#  'Agent: 任务完成，直接回答']
```

**注意**：这种写法没有 Literal 类型注解，LangGraph 无法在编译时推断所有可能的目标。图的可视化中可能不会显示所有分支。

### 写法 2：返回 Literal 类型注解（推荐）

通过 Literal 类型注解声明所有可能的返回值，LangGraph 可以自动推断分支。

```python
"""
写法 2：Literal 类型注解（推荐写法）
适用场景：所有可能的目标在编写时已知

图拓扑：
    START → classifier → (条件判断)
                           ↓ search      ↓ chat       ↓ code
                         retriever    chat_bot    code_gen
                           ↓            ↓            ↓
                          END          END          END
"""
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    intent: str
    response: str


def classifier(state: State):
    """模拟意图分类"""
    query = state["query"].lower()
    if "搜索" in query or "查找" in query:
        intent = "search"
    elif "代码" in query or "编程" in query:
        intent = "code"
    else:
        intent = "chat"
    return {"intent": intent}


def retriever(state: State):
    return {"response": f"检索结果: 关于 '{state['query']}' 的文档"}


def chat_bot(state: State):
    return {"response": f"闲聊回复: 你好！关于 '{state['query']}'..."}


def code_gen(state: State):
    return {"response": f"代码生成: # {state['query']}\nprint('hello')"}


# 路由函数：使用 Literal 类型注解
def route_by_intent(state: State) -> Literal["retriever", "chat_bot", "code_gen"]:
    """Literal 注解让 LangGraph 自动推断所有可能的分支"""
    intent = state["intent"]
    if intent == "search":
        return "retriever"
    elif intent == "code":
        return "code_gen"
    return "chat_bot"


builder = StateGraph(State)
builder.add_node("classifier", classifier)
builder.add_node("retriever", retriever)
builder.add_node("chat_bot", chat_bot)
builder.add_node("code_gen", code_gen)

builder.add_edge(START, "classifier")
# path_map=None，自动从 Literal 推断
builder.add_conditional_edges("classifier", route_by_intent)
builder.add_edge("retriever", END)
builder.add_edge("chat_bot", END)
builder.add_edge("code_gen", END)

graph = builder.compile()

# 测试不同意图
for query in ["帮我搜索 Python 教程", "今天天气怎么样", "写一段代码排序"]:
    result = graph.invoke({"query": query, "intent": "", "response": ""})
    print(f"查询: {query} → {result['response']}")
```

**为什么推荐这种写法？**
1. **类型安全**：IDE 和类型检查器可以验证返回值
2. **自动推断**：LangGraph 编译时自动知道所有可能的分支
3. **可视化友好**：图的可视化中会显示所有分支
4. **自文档化**：函数签名就说明了所有可能的路径

### 写法 3：返回 Send 对象列表

当需要动态创建并行任务时，路由函数返回 Send 对象列表。

```python
"""
写法 3：返回 Send 对象列表
适用场景：Map-Reduce 模式，动态并行

图拓扑：
    START → splitter → (条件边返回 Send 列表)
                         ↓ Send("worker", {topic: "A"})
                         ↓ Send("worker", {topic: "B"})
                         ↓ Send("worker", {topic: "C"})
                       worker (并行执行 3 次)
                         ↓
                       aggregator → END
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class OverallState(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    topic: str


def splitter(state: OverallState):
    """准备要处理的主题列表"""
    return {"topics": state["topics"]}


def worker(state: WorkerState):
    """处理单个主题（每个 Send 会独立调用一次）"""
    topic = state["topic"]
    return {"summaries": [f"摘要: {topic} 的分析结果"]}


def aggregator(state: OverallState):
    """聚合所有并行结果"""
    combined = " | ".join(state["summaries"])
    return {"summaries": [f"最终报告: [{combined}]"]}


# 路由函数：返回 Send 对象列表
def fan_out_to_workers(state: OverallState):
    """为每个主题创建一个并行任务"""
    return [Send("worker", {"topic": topic}) for topic in state["topics"]]


builder = StateGraph(OverallState)
builder.add_node("splitter", splitter)
builder.add_node("worker", worker)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "splitter")
# 条件边返回 Send 列表 → 动态并行
builder.add_conditional_edges("splitter", fan_out_to_workers)
builder.add_edge("worker", "aggregator")
builder.add_edge("aggregator", END)

graph = builder.compile()
result = graph.invoke({
    "topics": ["机器学习", "深度学习", "强化学习"],
    "summaries": [],
})
print(result["summaries"])
# ['摘要: 机器学习 的分析结果', '摘要: 深度学习 的分析结果',
#  '摘要: 强化学习 的分析结果', '最终报告: [...]']
```

**Send 的关键特性**：
- 每个 Send 对象指定目标节点和自定义输入状态
- 多个 Send 在同一个 superstep 中并行执行
- worker 节点接收的是 Send 中的自定义状态，不是全局状态
- 并行结果通过 Reducer（如 `operator.add`）自动合并到全局状态

---

## 与普通 add_edge 的对比

| 特性 | `add_edge()` | `add_conditional_edges()` |
|------|-------------|--------------------------|
| 路由方式 | 无条件，固定路径 | 有条件，运行时动态决策 |
| 参数 | `(source, target)` | `(source, path_fn, path_map)` |
| 执行时机 | 源节点完成后一定执行 | 源节点完成后根据路由函数决定 |
| 目标数量 | 固定 1 个 | 动态 1 个或多个 |
| 并行支持 | 多条边可并行 | Send 列表可并行 |
| 可视化 | 显示固定箭头 | 显示多条可能的箭头 |
| 典型场景 | 固定流程步骤 | 决策点、循环控制 |

```python
# add_edge：A 完成后一定去 B
builder.add_edge("A", "B")

# add_conditional_edges：A 完成后根据状态决定去 B 还是 C
builder.add_conditional_edges("A", lambda state: "B" if state["ok"] else "C")
```

**什么时候用哪个？**
- 流程中的固定步骤 → `add_edge()`
- 需要根据状态决策 → `add_conditional_edges()`
- 循环控制（是否继续循环）→ `add_conditional_edges()`
- 错误处理分支 → `add_conditional_edges()`

---

## 内部执行流程

理解 `add_conditional_edges()` 的内部机制有助于调试和优化。

### 编译阶段

```python
# 1. add_conditional_edges() 被调用时
#    将路由函数包装为 BranchSpec，存入 self.branches
self.branches[source].append(BranchSpec(path=path_runnable, ends=path_map))

# 2. compile() 编译时
#    attach_branch() 将 BranchSpec 挂载到源节点的 writers
def attach_branch(self, start, name, branch):
    # 创建 writer 函数，将路由结果转化为 channel 写入
    self.nodes[start].writers.append(branch.run(get_writes, reader))
```

### 运行阶段

```
源节点执行完毕
    ↓
源节点的 writers 执行（包括条件边的 BranchSpec）
    ↓
BranchSpec._route() 被调用
    ↓
reader 读取最新 state
    ↓
path 函数被调用，传入 state
    ↓
path 函数返回目标（字符串 / Send 列表）
    ↓
BranchSpec._finish() 处理返回值
    ↓
如果有 path_map，通过映射解析实际节点名
    ↓
验证目标有效性（不能是 None、START）
    ↓
写入 "branch:to:{target}" channel
    ↓
下一个 superstep，目标节点被触发执行
```

### 源码中的验证逻辑

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/graph/_branch.py]
def _finish(self, writer, input, result, config):
    if not isinstance(result, (list, tuple)):
        result = [result]

    if self.ends:
        # 有 path_map：通过映射解析
        destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
    else:
        # 无 path_map：直接使用返回值
        destinations = cast(Sequence[Send | str], result)

    # 验证：不能返回 None 或 START
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")

    # 验证：Send 不能指向 END
    if any(isinstance(dest, Send) and dest.node == END for dest in destinations):
        raise ValueError("Cannot send to END node")

    # 写入 channel 触发目标节点
    entries = writer(destinations, False)
```

---

## 常见错误和注意事项

### 错误 1：路由函数返回未注册的节点名

```python
# ❌ 错误：返回了不存在的节点名
def route(state) -> Literal["tools", "unknown_node"]:
    return "unknown_node"  # 这个节点没有 add_node！

# 编译时会报错：
# ValueError: Found edge starting at 'agent' connecting to 'unknown_node'
# which is not a known node.
```

**解决方案**：确保路由函数返回的所有节点名都已通过 `add_node()` 注册。

### 错误 2：path_map 的键与路由函数返回值不匹配

```python
# ❌ 错误：路由函数返回 "search"，但 path_map 中没有这个键
def route(state):
    return "search"

builder.add_conditional_edges("agent", route, {
    "retrieve": "retriever",  # 键是 "retrieve"，不是 "search"
})
# 运行时会报 KeyError
```

**解决方案**：确保 path_map 的键覆盖路由函数所有可能的返回值。

### 错误 3：忘记处理 END 路径

```python
# ❌ 危险：没有 END 退出路径，可能无限循环
def route(state) -> Literal["tools"]:
    return "tools"  # 永远返回 tools，没有退出！

# ✅ 正确：始终包含 END 退出路径
def route(state) -> Literal["tools", "__end__"]:
    if state["done"]:
        return END
    return "tools"
```

### 错误 4：条件边和普通边冲突

```python
# ⚠️ 注意：同一个源节点同时有普通边和条件边
builder.add_edge("agent", "tools")              # 普通边：一定去 tools
builder.add_conditional_edges("agent", route)    # 条件边：可能去 END

# 结果：普通边和条件边都会执行！
# agent 完成后，tools 一定会被触发（普通边），
# 同时路由函数的目标也会被触发（条件边）
```

**解决方案**：避免同一个源节点同时有普通边和条件边指向不同目标。如果需要，用条件边统一管理所有路由。

### 错误 5：异步路由函数忘记 await

```python
# ❌ 错误：异步函数但忘记声明 async
def route(state):
    result = some_async_call(state)  # 返回的是 coroutine，不是结果！
    return result

# ✅ 正确：使用 async def
async def route(state):
    result = await some_async_call(state)
    return result
```

---

## 在 Agent 开发中的应用

### 应用 1：ReAct Agent 工具调用循环

这是 LangGraph 中最经典的条件分支模式：

```python
"""
ReAct Agent：LLM 决定是否调用工具
这是 LangGraph 官方 create_react_agent 的核心模式
"""
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: list[dict]


def agent(state: AgentState):
    """调用 LLM，可能返回 tool_calls"""
    messages = state["messages"]
    if len(messages) < 3:
        return {"messages": messages + [
            {"role": "assistant", "tool_calls": [{"name": "search", "args": {}}]}
        ]}
    return {"messages": messages + [
        {"role": "assistant", "content": "最终答案：找到了相关信息"}
    ]}


def tools(state: AgentState):
    """执行工具调用"""
    return {"messages": state["messages"] + [
        {"role": "tool", "content": "搜索结果：Python 是一种编程语言"}
    ]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """核心路由：检查最后一条消息是否包含 tool_calls"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict) and last_msg.get("tool_calls"):
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)  # 核心条件边
builder.add_edge("tools", "agent")  # 工具执行后回到 agent

graph = builder.compile()
```

### 应用 2：RAG 检索质量路由

```python
"""
Self-RAG 模式：根据检索质量决定是否重试
"""
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    query: str
    documents: list[str]
    relevance: float
    attempt: int
    answer: str


def retrieve(state: RAGState):
    attempt = state.get("attempt", 0) + 1
    # 模拟检索，质量随重试次数提升
    score = min(0.3 * attempt, 1.0)
    return {
        "documents": [f"文档_{attempt}"],
        "relevance": score,
        "attempt": attempt,
    }


def generate(state: RAGState):
    return {"answer": f"基于 {state['documents']} 生成的回答"}


def rewrite(state: RAGState):
    return {"query": f"优化后的查询: {state['query']}"}


def check_relevance(state: RAGState) -> Literal["generate", "rewrite"]:
    """检索质量路由：质量好就生成，质量差就改写重试"""
    if state["relevance"] >= 0.7:
        return "generate"
    return "rewrite"


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("rewrite", rewrite)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", check_relevance)
builder.add_edge("generate", END)
builder.add_edge("rewrite", "retrieve")  # 重试循环

graph = builder.compile()
```

### 应用 3：多 Agent Supervisor 路由

```python
"""
Supervisor 模式：中央调度器将任务分发给不同的专家 Agent
"""
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class SupervisorState(TypedDict):
    task: str
    assigned_to: str
    result: str


def supervisor(state: SupervisorState):
    """分析任务，决定分配给谁"""
    task = state["task"].lower()
    if "翻译" in task:
        assigned = "translator"
    elif "总结" in task:
        assigned = "summarizer"
    else:
        assigned = "general"
    return {"assigned_to": assigned}


def translator(state: SupervisorState):
    return {"result": f"翻译结果: {state['task']}"}


def summarizer(state: SupervisorState):
    return {"result": f"摘要结果: {state['task']}"}


def general(state: SupervisorState):
    return {"result": f"通用处理: {state['task']}"}


def route_to_expert(
    state: SupervisorState,
) -> Literal["translator", "summarizer", "general"]:
    """根据 supervisor 的分配结果路由到专家"""
    return state["assigned_to"]


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("translator", translator)
builder.add_node("summarizer", summarizer)
builder.add_node("general", general)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_expert)
builder.add_edge("translator", END)
builder.add_edge("summarizer", END)
builder.add_edge("general", END)

graph = builder.compile()
```

---

## 三种写法对比总结

| 特性 | 返回字符串 | Literal 注解 | Send 列表 |
|------|-----------|-------------|-----------|
| 类型安全 | 无 | 有 | 无 |
| 自动推断 | 不支持 | 支持 | 不适用 |
| 可视化 | 可能不完整 | 完整显示所有分支 | 显示动态分支 |
| 并行支持 | 不支持 | 不支持 | 支持 |
| 适用场景 | 简单路由 | 大多数场景（推荐） | Map-Reduce |
| 代码复杂度 | 低 | 低 | 中 |

**选择建议**：
- 默认使用 **Literal 注解**（写法 2），兼顾类型安全和可视化
- 需要动态并行时使用 **Send 列表**（写法 3）
- 快速原型时可以用 **字符串返回**（写法 1），后续再加类型注解

---

## 最佳实践

### 1. 路由函数保持简单

```python
# ✅ 好：路由函数只做判断，不做业务逻辑
def route(state) -> Literal["tools", "__end__"]:
    if state["messages"][-1].get("tool_calls"):
        return "tools"
    return END

# ❌ 差：路由函数里塞了业务逻辑
def route(state) -> Literal["tools", "__end__"]:
    # 不要在路由函数里做复杂计算！
    processed = heavy_computation(state["data"])
    state["processed"] = processed  # 不要修改 state！
    if processed.score > 0.5:
        return "tools"
    return END
```

### 2. 始终包含 END 退出路径

```python
# ✅ 好：有明确的退出路径
def route(state) -> Literal["retry", "__end__"]:
    if state["attempt"] < 3 and not state["success"]:
        return "retry"
    return END  # 明确的退出

# ❌ 差：可能永远不会结束
def route(state) -> Literal["retry"]:
    return "retry"  # 没有 END！
```

### 3. 使用 Literal 类型注解

```python
# ✅ 推荐：Literal 注解
def route(state) -> Literal["a", "b", "__end__"]:
    ...

# ⚠️ 不推荐：裸字符串返回
def route(state) -> str:
    ...
```

### 4. 配合 recursion_limit 防止无限循环

```python
# 即使路由函数有 bug 导致无限循环，recursion_limit 也能兜底
graph.invoke(
    {"messages": []},
    {"recursion_limit": 25}  # 最多执行 25 个 superstep
)
```

---

## 参考资料

### 源码
- `langgraph/graph/state.py` L839-L887 - `add_conditional_edges()` 方法定义
- `langgraph/graph/_branch.py` L83-L225 - `BranchSpec` 类和路由执行逻辑
- `langgraph/graph/state.py` L1323-L1370 - `attach_branch()` 编译时挂载
- `langgraph/prebuilt/chat_agent_executor.py` - ReAct Agent 中的实际用法

### 官方文档
- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [LangGraph How-to: Branching](https://langchain-ai.github.io/langgraph/how-tos/branching/)

---

**版本**: v1.0
**最后更新**: 2026-03-01
**作者**: Claude Code
**知识点**: 条件分支策略 - add_conditional_edges 基础



