# 核心概念5：Command API 状态转换

> Command 是 LangGraph 提供的"状态更新 + 路由控制"二合一原语——节点不再只能返回数据，还能同时告诉图"下一步去哪"。

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/types.py` — Command 类定义与属性

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 1. Command API 是什么？

### 核心问题：节点想同时做两件事

在传统 LangGraph 写法中，节点和路由是分开的：

```
节点函数 → 返回状态更新（dict）
条件边函数 → 决定下一步去哪（str）
```

这意味着你需要写两个函数：一个处理数据，一个决定路由。当路由逻辑和数据处理紧密耦合时，这种分离反而增加了复杂度。

**Command 把两者合二为一：**

```
节点函数 → 返回 Command（状态更新 + 路由目标）
```

### 一句话定义

**Command 是一个数据类，让节点在返回状态更新的同时，指定下一个要执行的节点。**

### 双重类比

**前端类比：React Router 的 `navigate()` 带 state**

```javascript
// 前端：跳转页面的同时传递数据
navigate("/dashboard", { state: { user: "张三", role: "admin" } });
```

在 React Router 中，`navigate()` 同时完成了两件事：跳转到目标页面 + 传递状态数据。Command 做的事情完全一样——更新状态 + 指定下一个节点。

**日常生活类比：快递员送完包裹报告下一站**

快递员送完包裹后，不是回到调度中心等指令，而是直接告诉系统："包裹已送达，下一站去 B 小区。" 调度中心不需要额外判断，快递员自己就决定了下一步。

---

## 2. 基本用法

### 最简单的 Command

```python
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    route: str
    result: str


def router_node(state: State) -> Command[Literal["search", "generate"]]:
    """根据查询内容决定路由，同时记录路由选择"""
    if "搜索" in state["query"]:
        return Command(
            update={"route": "search"},   # 更新状态
            goto="search"                  # 指定下一个节点
        )
    else:
        return Command(
            update={"route": "generate"},
            goto="generate"
        )


def search_node(state: State) -> dict:
    return {"result": f"搜索结果: {state['query']}"}


def generate_node(state: State) -> dict:
    return {"result": f"生成结果: {state['query']}"}


# 构建图
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("search", search_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "router")
# 注意：不需要 add_conditional_edges！Command 自带路由
graph.add_edge("search", END)
graph.add_edge("generate", END)

app = graph.compile()

# 测试
result = app.invoke({"query": "搜索最新新闻", "route": "", "result": ""})
print(result)
# {'query': '搜索最新新闻', 'route': 'search', 'result': '搜索结果: 搜索最新新闻'}
```

### 关键点

1. **返回类型注解是必须的** — `Command[Literal["search", "generate"]]` 告诉 LangGraph 这个节点可能跳转到哪些节点
2. **`update` 字段** — 和普通节点返回 dict 一样，遵循相同的 reducer 规则
3. **`goto` 字段** — 指定下一个要执行的节点名称
4. **不需要条件边** — Command 自己就包含了路由信息

---

## 3. Command 的四个属性

从源码中可以看到 Command 类的完整定义：

```python
# 源码：libs/langgraph/langgraph/types.py
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None        # 目标图（None=当前图，PARENT=父图）
    update: Any | None = None       # 状态更新
    resume: dict[str, Any] | Any | None = None  # 恢复中断的值
    goto: Send | Sequence[Send | N] | N = ()     # 下一个节点
```

[来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py]

### 属性详解

| 属性 | 类型 | 作用 | 默认值 |
|------|------|------|--------|
| `update` | `dict` 或其他 | 状态更新，与普通返回 dict 相同 | `None` |
| `goto` | `str` / `list[str]` / `Send` | 下一个节点（单个、多个或 Send 对象） | `()` 空元组 |
| `graph` | `str` / `None` | 目标图，`None` 表示当前图 | `None` |
| `resume` | `Any` | 恢复 `interrupt()` 中断时传递的值 | `None` |

### 每个属性的使用场景

```python
from langgraph.types import Command, Send

# 1. 只更新状态，不路由
Command(update={"count": 1})

# 2. 只路由，不更新状态
Command(goto="next_node")

# 3. 更新 + 路由（最常见）
Command(update={"route": "search"}, goto="search")

# 4. 路由到多个节点
Command(goto=["node_a", "node_b"])

# 5. 发送到父图
Command(update={"result": "done"}, graph=Command.PARENT)

# 6. 恢复中断
Command(resume="用户确认继续")
```

---

## 4. 多目标路由：同时去多个节点

Command 的 `goto` 可以是一个列表，实现扇出（fan-out）模式：

```python
from langgraph.types import Command
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from operator import add


class State(TypedDict):
    query: str
    tasks: Annotated[list[str], add]  # 用 reducer 累积任务结果
    summary: str
    translation: str


def dispatcher(state: State) -> Command[Literal["summarizer", "translator"]]:
    """同时派发到摘要和翻译节点"""
    return Command(
        update={"tasks": ["dispatched"]},
        goto=["summarizer", "translator"]  # 同时去两个节点
    )


def summarizer(state: State) -> dict:
    return {
        "summary": f"摘要: {state['query'][:20]}...",
        "tasks": ["summarized"]
    }


def translator(state: State) -> dict:
    return {
        "translation": f"Translation: {state['query']}",
        "tasks": ["translated"]
    }
```

### 使用 Send 对象实现差异化输入

当你需要给不同节点发送不同的输入时，使用 `Send` 对象：

```python
from langgraph.types import Command, Send

def dispatcher(state: State) -> Command[Literal["worker"]]:
    """给同一个节点发送多个不同的任务"""
    return Command(
        goto=[
            Send("worker", {"task": "summarize", "text": state["query"]}),
            Send("worker", {"task": "translate", "text": state["query"]}),
        ]
    )
```

---

## 5. Command vs 条件边：什么时候用哪个？

### 对比表

| 特性 | Command | 条件边 (conditional_edges) |
|------|---------|---------------------------|
| 状态更新 | 同时更新 | 需要在节点中单独处理 |
| 路由决策 | 在节点内决定 | 在独立的边函数中决定 |
| 类型安全 | `Literal` 类型注解 | 返回字符串映射 |
| 代码位置 | 路由逻辑在节点内 | 路由逻辑在边函数中 |
| 适用场景 | 路由与数据处理紧耦合 | 路由逻辑独立于数据处理 |

### 传统条件边写法

```python
# 需要两个函数：一个处理数据，一个决定路由
def classify_node(state: State) -> dict:
    """节点：只负责分类"""
    category = classify(state["query"])
    return {"category": category}

def route_by_category(state: State) -> str:
    """边函数：只负责路由"""
    if state["category"] == "question":
        return "qa_node"
    return "chat_node"

# 构建图时需要额外的条件边
graph.add_conditional_edges("classify", route_by_category)
```

### Command 写法

```python
# 一个函数搞定：分类 + 路由
def classify_and_route(state: State) -> Command[Literal["qa_node", "chat_node"]]:
    """节点：分类并路由"""
    category = classify(state["query"])
    target = "qa_node" if category == "question" else "chat_node"
    return Command(
        update={"category": category},
        goto=target
    )

# 构建图时不需要条件边
graph.add_edge(START, "classify_and_route")
```

### 选择建议

- **用 Command**：当路由决策依赖于节点内部的计算结果时（比如 LLM 判断、分类结果）
- **用条件边**：当路由逻辑简单且独立，只需要读取状态中已有的字段时

---

## 6. 重要警告：Command 不覆盖静态边

这是最容易踩的坑。**Command 的 `goto` 只是添加动态边，不会替换已有的静态边。**

```python
# 危险示例：静态边 + Command 会导致两个节点都执行
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_node("C", node_c)

graph.add_edge("A", "B")  # 静态边：A → B

# 如果 node_a 返回 Command(goto="C")
# 结果：B 和 C 都会执行！不是只执行 C！
```

### 正确做法

如果你想用 Command 控制路由，**不要给该节点添加静态边**：

```python
# 正确：不给 A 添加静态出边
graph.add_node("A", node_a)  # node_a 返回 Command
graph.add_node("B", node_b)
graph.add_node("C", node_c)

graph.add_edge(START, "A")
# A 的出边完全由 Command 控制
graph.add_edge("B", END)
graph.add_edge("C", END)
```

### 为什么这样设计？

从图论角度看，静态边是图结构的一部分，Command 是运行时的动态行为。LangGraph 的设计哲学是"动态边是附加的，不是替代的"。这保证了图的静态结构始终可预测。

---

## 7. 子图通信：Command.PARENT

在嵌套图（子图）中，节点默认只能更新当前图的状态。通过 `graph=Command.PARENT`，子图节点可以直接更新父图的状态。

```python
from langgraph.types import Command
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ===== 子图 =====
class SubState(TypedDict):
    sub_input: str
    sub_result: str


def sub_process(state: SubState) -> dict:
    """子图内部处理"""
    return {"sub_result": f"处理完成: {state['sub_input']}"}


def sub_report(state: SubState) -> Command:
    """子图节点：将结果发送到父图"""
    return Command(
        update={"parent_result": state["sub_result"]},  # 更新父图的字段
        graph=Command.PARENT  # 指定目标是父图
    )


sub_graph = StateGraph(SubState)
sub_graph.add_node("process", sub_process)
sub_graph.add_node("report", sub_report)
sub_graph.add_edge(START, "process")
sub_graph.add_edge("process", "report")
sub_graph.add_edge("report", END)


# ===== 父图 =====
class ParentState(TypedDict):
    input: str
    parent_result: str


parent_graph = StateGraph(ParentState)
parent_graph.add_node("sub", sub_graph.compile())
parent_graph.add_edge(START, "sub")
parent_graph.add_edge("sub", END)

app = parent_graph.compile()
```

### 关键点

- `Command.PARENT` 的值是字符串 `"__parent__"`
- `update` 中的字段名必须匹配**父图**的 State 定义
- 这是子图向父图"报告结果"的标准方式

---

## 8. 与 interrupt 配合：人机协作

Command 的 `resume` 属性用于恢复被 `interrupt()` 暂停的图执行：

```python
from langgraph.types import Command, interrupt

def review_node(state: State) -> dict:
    """需要人工审核的节点"""
    human_input = interrupt("请审核以下查询是否安全: " + state["query"])
    return {"approved": human_input == "approve"}

# 客户端恢复执行时：
# app.invoke(Command(resume="approve"), config=thread_config)
```

流程：图执行到 `interrupt()` 暂停 → 客户端收到中断信息 → 用户决定后调用 `Command(resume="approve")` → 图从中断点恢复。

---

## 9. 与工具调用结合

在 Agent 场景中，工具函数也可以返回 Command 来更新图状态：

```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command


@tool
def lookup_user(tool_call_id: str, config: dict):
    """查找用户信息的工具——同时更新图状态"""
    # 模拟查找用户
    user_info = {"name": "张三", "level": "VIP"}

    return Command(
        update={
            "user_info": user_info,
            "messages": [
                ToolMessage(
                    content="已找到用户信息",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
```

### 为什么工具需要返回 Command？

普通工具只能返回一个字符串结果（作为 ToolMessage 的 content）。但有时工具需要做更多事情——比如把查到的用户信息写入图状态的 `user_info` 字段，而不仅仅是返回一段文字。Command 让工具拥有了"直接操作图状态"的能力。

---

## 10. 类型注解的重要性

```python
# ❌ 缺少类型注解 — LangGraph 无法知道可能的路由目标
def my_node(state: State):
    return Command(goto="search")

# ✅ 有类型注解 — LangGraph 在编译时就知道可能的边
def my_node(state: State) -> Command[Literal["search", "generate"]]:
    return Command(goto="search")
```

LangGraph 在编译图时需要知道所有可能的边，才能正确构建图结构。`Literal` 类型注解就是在告诉框架："这个节点可能跳转到 search 或 generate"。`goto` 的实际值必须在 `Literal` 列表中。

---

## 11. 最佳实践总结

### 什么时候用 Command

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| LLM 判断后路由 | Command | 判断和路由在同一个节点完成 |
| 简单状态字段路由 | 条件边 | 路由逻辑独立，更清晰 |
| 子图向父图报告 | Command.PARENT | 唯一的跨图通信方式 |
| 人机协作恢复 | Command.resume | interrupt 的标准恢复方式 |
| 工具更新图状态 | Command in tool | 工具需要写入非消息字段 |

### 常见错误清单

```python
# ❌ 错误1：忘记类型注解
def node(state) -> dict:  # 应该是 Command[Literal[...]]
    return Command(goto="next")

# ❌ 错误2：Command + 静态边冲突
graph.add_edge("A", "B")  # A 已经有静态边
# 如果 A 返回 Command(goto="C")，B 和 C 都会执行

# ❌ 错误3：goto 目标不在 Literal 中
def node(state) -> Command[Literal["a", "b"]]:
    return Command(goto="c")  # "c" 不在类型注解中

# ✅ 正确：类型注解与实际 goto 一致
def node(state) -> Command[Literal["a", "b"]]:
    if condition:
        return Command(goto="a")
    return Command(goto="b")
```

### 在 RAG 中的应用

Command 在 RAG 系统中特别有用：

```python
def query_analyzer(state: State) -> Command[Literal["vector_search", "keyword_search", "hybrid_search"]]:
    """分析查询类型，选择最佳检索策略"""
    query = state["query"]

    # 根据查询特征选择检索方式
    if is_semantic_query(query):
        strategy = "vector_search"
    elif is_exact_match_query(query):
        strategy = "keyword_search"
    else:
        strategy = "hybrid_search"

    return Command(
        update={"search_strategy": strategy, "analyzed": True},
        goto=strategy
    )
```

这比条件边更自然——查询分析和路由决策本来就是同一个思考过程，没必要拆成两个函数。
