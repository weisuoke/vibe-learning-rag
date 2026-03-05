# 核心概念 5：Command 高级路由控制

## 概念定义

**Command 是 LangGraph 中节点返回的指令对象，能够在一次操作中同时完成状态更新和路由控制，还支持跨图导航和中断恢复，是最强大的路由机制。**

## API 签名

```python
from langgraph.types import Command

# 创建 Command 对象
cmd = Command(
    goto="next_node",           # 导航目标
    update={"key": "value"},    # 状态更新
    graph=None,                 # 目标图（None=当前图）
    resume=None                 # 中断恢复值
)
```

**源码定义：**

```python
@dataclass
class Command(Generic[N]):
    graph: str | None = None      # 目标图（None=当前图，PARENT=父图）
    update: Any | None = None     # 状态更新
    resume: dict[str, Any] | Any | None = None  # 中断恢复值
    goto: Send | Sequence[Send | N] | N = ()     # 导航目标
    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

**四个属性详解：**

| 属性 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `goto` | `str \| list \| Send` | 导航目标，支持节点名、列表、Send 对象 | `()` 空元组 |
| `update` | `Any \| None` | 要更新的状态字段 | `None` |
| `graph` | `str \| None` | 目标图，`None` 表示当前图 | `None` |
| `resume` | `Any \| None` | 与 `interrupt()` 配合的恢复值 | `None` |

## Command vs add_conditional_edges

理解 Command 的关键在于它把路由决策从"边"搬到了"节点内部"：

```python
# ===== 方式1：条件边（路由逻辑在边定义中） =====
def router_function(state):
    """路由函数：只负责返回目标节点名"""
    if state["intent"] == "search":
        return "search_node"
    return "chat_node"

# 路由逻辑和状态更新是分离的
builder.add_conditional_edges("classifier", router_function)

# ===== 方式2：Command（路由逻辑在节点内部） =====
def classifier_node(state):
    """节点函数：同时完成分类、状态更新、路由"""
    intent = classify(state["input"])
    return Command(
        goto="search_node" if intent == "search" else "chat_node",
        update={"intent": intent, "classified_at": "2024-01-01"}
    )

# 不需要 add_conditional_edges，节点自己决定去哪
builder.add_node("classifier", classifier_node)
```

**什么时候用哪个？**

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 简单 if-else 分支 | `add_conditional_edges` | 逻辑简单，边定义更清晰 |
| 路由时需要同时更新状态 | `Command` | 一次操作完成两件事 |
| 路由逻辑依赖复杂计算 | `Command` | 计算结果可以同时用于路由和状态更新 |
| 需要跨图导航 | `Command` | 只有 Command 支持 `graph` 参数 |
| 人机循环中断恢复 | `Command` | 只有 Command 支持 `resume` 参数 |

## 基本用法

### 1. goto + update：最常用的组合

```python
from langgraph.types import Command
from typing import Literal

def router_node(state) -> Command[Literal["search_node", "chat_node"]]:
    """根据意图路由，同时记录路由信息"""
    if state["intent"] == "search":
        return Command(
            goto="search_node",
            update={"routed": True, "route_target": "search"}
        )
    return Command(
        goto="chat_node",
        update={"routed": True, "route_target": "chat"}
    )
```

**注意类型注解**：`Command[Literal["search_node", "chat_node"]]` 告诉 LangGraph 这个节点可能路由到哪些目标，用于图的可视化和验证。

### 2. 完整示例：意图分类路由

```python
"""
Command 高级路由 - 基础示例
演示：基于用户意图的智能路由
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ===== 1. 定义状态 =====

class State(TypedDict):
    user_input: str
    intent: str
    response: str
    routed: bool

# ===== 2. 定义节点 =====

def classify_and_route(state: State) -> Command[Literal["search", "chat", "help"]]:
    """分类用户意图并路由到对应处理节点"""
    text = state["user_input"].lower()

    if "搜索" in text or "查找" in text:
        intent = "search"
    elif "帮助" in text or "怎么" in text:
        intent = "help"
    else:
        intent = "chat"

    print(f"[分类] 输入: '{state['user_input']}' → 意图: {intent}")

    return Command(
        goto=intent,
        update={"intent": intent, "routed": True}
    )

def search(state: State):
    return {"response": f"搜索结果: 关于 '{state['user_input']}' 的信息..."}

def chat(state: State):
    return {"response": f"闲聊回复: 你好！你说的是 '{state['user_input']}'"}

def help_node(state: State):
    return {"response": f"帮助信息: 这是关于 '{state['user_input']}' 的使用指南"}

# ===== 3. 构建图 =====

builder = StateGraph(State)
builder.add_node("classify_and_route", classify_and_route)
builder.add_node("search", search)
builder.add_node("chat", chat)
builder.add_node("help", help_node)

builder.add_edge(START, "classify_and_route")
# 注意：不需要 add_conditional_edges，Command 自己控制路由
builder.add_edge("search", END)
builder.add_edge("chat", END)
builder.add_edge("help", END)

graph = builder.compile()

# ===== 4. 运行 =====

print("=== Command 路由示例 ===\n")

for user_input in ["搜索 LangGraph 教程", "你好啊", "怎么使用这个工具"]:
    result = graph.invoke({"user_input": user_input, "routed": False})
    print(f"  回复: {result['response']}\n")
```

**运行输出：**
```
=== Command 路由示例 ===

[分类] 输入: '搜索 LangGraph 教程' → 意图: search
  回复: 搜索结果: 关于 '搜索 LangGraph 教程' 的信息...

[分类] 输入: '你好啊' → 意图: chat
  回复: 闲聊回复: 你好！你说的是 '你好啊'

[分类] 输入: '怎么使用这个工具' → 意图: help
  回复: 帮助信息: 这是关于 '怎么使用这个工具' 的使用指南
```

## 高级特性

### 1. goto 支持多种类型

`goto` 参数非常灵活，支持三种形式：

```python
# ===== 形式1：字符串 → 单目标路由 =====
Command(goto="next_node")

# ===== 形式2：列表 → 多目标并行 =====
Command(goto=["node_a", "node_b", "node_c"])
# 等价于同时触发三个节点

# ===== 形式3：Send 对象 → 带自定义状态的并行 =====
from langgraph.types import Send

Command(goto=[
    Send("process", {"topic": "AI"}),
    Send("process", {"topic": "ML"}),
])
# 结合了 Command 的状态更新能力和 Send 的并行能力
```

**goto 列表 + update 组合示例：**

```python
def broadcast_node(state):
    """广播消息到多个处理节点，同时更新状态"""
    return Command(
        goto=["logger", "notifier", "processor"],
        update={"broadcast_count": state.get("broadcast_count", 0) + 1}
    )
```

### 2. graph 参数：跨图导航

在子图（subgraph）场景中，`graph` 参数允许子图中的节点控制父图的路由：

```python
# ===== 子图中的节点 =====
def subgraph_node(state):
    """子图节点完成任务后，通知父图继续"""
    result = process(state)

    # 导航到父图的 "next_step" 节点
    return Command(
        goto="next_step",
        update={"sub_result": result},
        graph=Command.PARENT  # 关键：指定目标是父图
    )
```

**`graph` 参数的取值：**

| 值 | 含义 |
|----|------|
| `None`（默认） | 当前图 |
| `Command.PARENT` | 父图（即 `"__parent__"`） |

**典型场景：** 多 Agent 系统中，子 Agent 完成任务后将控制权交还给 Supervisor。

```python
def agent_node(state):
    """子 Agent 完成任务后交还控制权"""
    result = do_work(state)

    if result["quality"] == "good":
        # 告诉父图（Supervisor）：任务完成，去下一步
        return Command(
            goto="aggregator",
            update={"agent_result": result},
            graph=Command.PARENT
        )
    else:
        # 告诉父图：需要重试
        return Command(
            goto="retry_dispatcher",
            update={"error": "quality too low"},
            graph=Command.PARENT
        )
```

### 3. resume 参数：人机循环

`resume` 与 `interrupt()` 配合，实现人工审批等中断-恢复场景：

```python
from langgraph.types import Command, interrupt

def approval_node(state):
    """需要人工审批的节点"""
    # 中断执行，等待人工输入
    human_decision = interrupt(
        {"question": "是否批准这个操作？", "data": state["draft"]}
    )

    # 人工恢复后继续执行
    if human_decision == "approve":
        return Command(goto="execute", update={"approved": True})
    elif human_decision == "revise":
        return Command(goto="revise", update={"feedback": "需要修改"})
    else:
        return Command(goto="reject", update={"rejected": True})
```

**恢复执行时：**

```python
# 人工审批后恢复图的执行
graph.invoke(
    Command(resume="approve"),  # 传入人工决策
    config={"configurable": {"thread_id": "thread-1"}}
)
```

### 4. update + goto 组合的威力

Command 最大的优势是**一次操作同时完成路由和状态更新**，避免了条件边模式下需要额外节点来更新状态的问题：

```python
# ===== 条件边模式：需要两步 =====
# 步骤1：节点更新状态
def evaluator(state):
    score = evaluate(state["draft"])
    return {"score": score, "quality": "high" if score > 0.8 else "low"}

# 步骤2：条件边读取状态做路由
def route_by_quality(state):
    return "approve" if state["quality"] == "high" else "revise"

builder.add_conditional_edges("evaluator", route_by_quality)

# ===== Command 模式：一步到位 =====
def evaluator(state) -> Command[Literal["approve", "revise"]]:
    score = evaluate(state["draft"])
    if score > 0.8:
        return Command(goto="approve", update={"score": score, "quality": "high"})
    return Command(goto="revise", update={"score": score, "quality": "low"})
```

## 三种路由机制对比

| 特性 | add_conditional_edges | Send | Command |
|------|----------------------|------|---------|
| 路由位置 | 边定义（图结构层） | 条件边返回值 | 节点返回值 |
| 状态更新 | 不支持（需额外节点） | 自定义子状态 | 直接支持 `update` |
| 并行执行 | 不支持 | 支持（Send 列表） | 通过 `goto` 列表或 Send |
| 跨图导航 | 不支持 | 不支持 | 支持（`graph=PARENT`） |
| 中断恢复 | 不支持 | 不支持 | 支持（`resume`） |
| 类型推断 | Literal 返回类型 | 无 | `Command[Literal[...]]` |
| 可视化 | 自动显示分支 | 显示为条件边 | 需要 Literal 注解 |
| 适用场景 | 简单分支决策 | Map-Reduce 并行 | 复杂路由 + 状态更新 |

**选择建议：**

```
需要并行处理？ ──→ 是 ──→ Send（Map-Reduce 模式）
       │
       否
       │
需要同时更新状态？ ──→ 是 ──→ Command
       │
       否
       │
需要跨图导航？ ──→ 是 ──→ Command
       │
       否
       │
简单分支 ──→ add_conditional_edges
```

## 实战示例：RAG 质量评估路由

```python
"""
Command 高级路由 - RAG 实战示例
演示：检索质量评估 → 动态路由到不同处理策略
"""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator

# ===== 1. 定义状态 =====

class RAGState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    retrieval_score: float
    strategy: str
    response: str
    retry_count: int

# ===== 2. 定义节点 =====

def retrieve(state: RAGState):
    """模拟检索"""
    query = state["query"]
    # 模拟检索结果和质量分数
    docs = [{"content": f"关于 {query} 的文档", "score": 0.75}]
    avg_score = sum(d["score"] for d in docs) / len(docs)
    return {"retrieved_docs": docs, "retrieval_score": avg_score}

def evaluate_and_route(state: RAGState) -> Command[Literal["generate", "rewrite_query", "fallback"]]:
    """评估检索质量并路由到对应策略"""
    score = state["retrieval_score"]
    retry_count = state.get("retry_count", 0)

    print(f"[评估] 检索分数: {score}, 重试次数: {retry_count}")

    if score > 0.8:
        # 质量好：直接生成
        return Command(
            goto="generate",
            update={"strategy": "direct_generate"}
        )
    elif score > 0.5 and retry_count < 2:
        # 质量一般且未超过重试上限：改写查询重试
        return Command(
            goto="rewrite_query",
            update={
                "strategy": "query_rewrite",
                "retry_count": retry_count + 1
            }
        )
    else:
        # 质量差或重试耗尽：降级处理
        return Command(
            goto="fallback",
            update={"strategy": "fallback"}
        )

def rewrite_query(state: RAGState):
    """改写查询"""
    return {"query": f"{state['query']}（改写版）"}

def generate(state: RAGState):
    """基于检索结果生成回答"""
    docs_text = " ".join(d["content"] for d in state["retrieved_docs"])
    return {"response": f"基于检索结果生成: {docs_text}"}

def fallback(state: RAGState):
    """降级处理"""
    return {"response": f"抱歉，无法找到关于 '{state['query']}' 的准确信息"}

# ===== 3. 构建图 =====

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("evaluate_and_route", evaluate_and_route)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("fallback", fallback)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "evaluate_and_route")
# Command 自动处理路由，不需要 add_conditional_edges
builder.add_edge("rewrite_query", "retrieve")  # 改写后重新检索
builder.add_edge("generate", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# ===== 4. 运行 =====

print("=== RAG 质量评估路由 ===\n")
result = graph.invoke({
    "query": "LangGraph 条件分支",
    "retrieved_docs": [],
    "retrieval_score": 0.0,
    "strategy": "",
    "response": "",
    "retry_count": 0
})
print(f"\n策略: {result['strategy']}")
print(f"回复: {result['response']}")
```

## 最佳实践

### 1. 始终添加 Literal 类型注解

```python
# ✅ 好：类型注解让图可视化正确显示分支
def node(state) -> Command[Literal["a", "b", "c"]]:
    return Command(goto="a")

# ❌ 差：没有类型注解，图可视化无法显示可能的分支
def node(state):
    return Command(goto="a")
```

### 2. 保持路由逻辑简单

```python
# ✅ 好：路由逻辑清晰，一目了然
def route(state) -> Command[Literal["fast", "slow"]]:
    if state["priority"] == "high":
        return Command(goto="fast", update={"priority_handled": True})
    return Command(goto="slow", update={"priority_handled": True})

# ❌ 差：路由逻辑过于复杂，难以理解和调试
def route(state) -> Command[Literal["a", "b", "c", "d", "e"]]:
    score = state["s1"] * 0.3 + state["s2"] * 0.7
    if score > 0.9 and state["flag1"]:
        return Command(goto="a", update={...})
    elif score > 0.7 and not state["flag2"]:
        # ... 10 行条件判断
        pass
```

### 3. update 只包含必要字段

```python
# ✅ 好：只更新路由相关的状态
return Command(goto="next", update={"routed": True, "route_target": "next"})

# ❌ 差：在 Command 中做大量状态计算
return Command(goto="next", update={
    "routed": True,
    "processed_data": heavy_computation(state),  # 应该在节点逻辑中完成
    "statistics": calculate_stats(state),
})
```

### 4. 调试技巧

```python
def debug_command_node(state) -> Command[Literal["a", "b"]]:
    """添加日志便于调试路由决策"""
    decision = "a" if state["score"] > 0.5 else "b"

    print(f"[Debug] 状态: score={state['score']}")
    print(f"[Debug] 决策: goto={decision}")

    cmd = Command(goto=decision, update={"last_route": decision})
    print(f"[Debug] Command: {cmd}")
    return cmd
```

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 源码分析**
   - 文件：`langgraph/types.py` - Command 类定义
   - 来源：`reference/source_conditional_branching_01.md`

2. **Command 动态路由教程**
   - 标题：A Beginner's Guide to Dynamic Routing in LangGraph with Command()
   - 来源：`reference/fetch_command_routing_01.md`

3. **LangGraph 最佳实践**
   - 来源：`reference/fetch_best_practices_01.md`

## 总结

**Command 是 LangGraph 最强大的路由机制：**

- **核心能力**：一次操作同时完成路由（`goto`）和状态更新（`update`）
- **独有特性**：跨图导航（`graph=PARENT`）、中断恢复（`resume`）
- **goto 灵活性**：支持字符串、列表、Send 对象三种形式
- **适用场景**：路由决策需要伴随状态更新、子图与父图交互、人机循环审批

简单分支用 `add_conditional_edges`，并行处理用 `Send`，复杂路由控制用 `Command` -- 三者各有所长，按需选择。