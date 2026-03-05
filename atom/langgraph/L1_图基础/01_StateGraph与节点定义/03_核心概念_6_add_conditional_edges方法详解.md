# add_conditional_edges 方法详解

> **来源声明**：本文档基于 LangGraph 源码分析（state.py:839-887）、Context7 官方文档和社区实践案例

---

## 方法概述

**`add_conditional_edges` 是 StateGraph 中实现动态路由的核心方法，允许根据状态内容决定下一步执行哪个节点。**

**核心特征**：
- 动态路由：根据运行时状态决定流向
- 类型安全：使用 Literal 类型提示可能的路由目标
- 灵活映射：支持 path_map 将路由结果映射到节点
- 多分支支持：可以路由到多个不同的目标节点

**来源**：LangGraph 源码 state.py:839-887

---

## 核心概念 1：条件路由函数

**条件路由函数是一个接受状态并返回目标节点名称的函数。**

### 函数签名

```python
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    count: int

# 路由函数返回节点名称或 END
def route_fn(state: State) -> Literal["continue", "end"]:
    if state["count"] < 5:
        return "continue"
    else:
        return "end"
```

**来源**：Context7 官方文档

### 路由函数的要求

1. **输入**：接受完整的 State
2. **输出**：返回目标节点名称（字符串）或 END
3. **类型提示**：使用 Literal 标注所有可能的返回值
4. **纯函数**：不应修改状态，只读取状态做决策

### 实际应用场景

**场景 1：循环控制**
```python
def should_continue(state: State) -> Literal["process", END]:
    """决定是否继续处理"""
    if state["remaining_items"] > 0:
        return "process"
    return END
```

**场景 2：错误处理**
```python
def handle_result(state: State) -> Literal["retry", "success", "fail"]:
    """根据结果决定下一步"""
    if state["error"] and state["retry_count"] < 3:
        return "retry"
    elif state["success"]:
        return "success"
    else:
        return "fail"
```

**来源**：Reddit 社区实践案例

---

## 核心概念 2：path_map 映射

**path_map 是一个字典，将路由函数的返回值映射到实际的节点名称。**

### 基础用法

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# 添加节点
builder.add_node("node_a", node_a_fn)
builder.add_node("node_b", node_b_fn)
builder.add_node("node_c", node_c_fn)

# 路由函数返回逻辑名称
def route(state: State) -> Literal["option1", "option2", "finish"]:
    if state["value"] < 10:
        return "option1"
    elif state["value"] < 20:
        return "option2"
    else:
        return "finish"

# 使用 path_map 映射到实际节点
builder.add_conditional_edges(
    "node_a",
    route,
    path_map={
        "option1": "node_b",
        "option2": "node_c",
        "finish": END
    }
)
```

**来源**：Context7 官方文档

### path_map 的优势

1. **解耦逻辑**：路由逻辑与节点名称分离
2. **可读性**：使用语义化的路由名称
3. **灵活性**：可以轻松重新映射路由目标
4. **维护性**：修改节点名称时只需更新 path_map

### 不使用 path_map 的简化写法

```python
# 路由函数直接返回节点名称
def route(state: State) -> Literal["node_b", "node_c", END]:
    if state["value"] < 10:
        return "node_b"
    elif state["value"] < 20:
        return "node_c"
    else:
        return END

# 不需要 path_map
builder.add_conditional_edges("node_a", route)
```

**来源**：LangGraph 源码分析

---

## 核心概念 3：BranchSpec 规范

**BranchSpec 是 LangGraph 内部用于表示条件分支的数据结构。**

### 源码定义

```python
# 来源：state.py:839-887
@dataclass
class BranchSpec:
    """条件分支规范"""
    path: Callable  # 路由函数
    path_map: dict[str, str] | None  # 路由映射
    then: str | None  # 所有分支后的汇聚节点
```

### BranchSpec 的作用

1. **存储分支信息**：保存路由函数和映射关系
2. **编译时验证**：检查路由目标是否存在
3. **执行时调用**：运行时调用路由函数决定流向

### 内部工作流程

```python
# 伪代码：LangGraph 内部处理流程
def execute_conditional_edge(state, branch_spec):
    # 1. 调用路由函数
    route_result = branch_spec.path(state)
    
    # 2. 应用 path_map 映射
    if branch_spec.path_map:
        target_node = branch_spec.path_map[route_result]
    else:
        target_node = route_result
    
    # 3. 执行目标节点
    return execute_node(target_node, state)
```

**来源**：LangGraph 源码分析

---

## 核心概念 4：Literal 类型提示

**Literal 类型提示用于明确标注路由函数可能返回的所有值，提供类型安全和 IDE 支持。**

### 为什么需要 Literal

```python
from typing import Literal

# ✅ 好的实践：使用 Literal
def route(state: State) -> Literal["a", "b", "c", END]:
    ...

# ❌ 不推荐：使用 str
def route(state: State) -> str:
    ...
```

**优势**：
1. **类型检查**：编译时发现拼写错误
2. **IDE 支持**：自动补全和提示
3. **文档作用**：清晰展示所有可能的路由
4. **重构安全**：修改节点名称时 IDE 会提示

**来源**：Context7 官方文档

### Literal 与 path_map 的配合

```python
# 路由函数使用逻辑名称
def route(state: State) -> Literal["success", "failure", "retry"]:
    ...

# path_map 映射到实际节点
builder.add_conditional_edges(
    "processor",
    route,
    path_map={
        "success": "success_handler",
        "failure": "error_handler",
        "retry": "retry_handler"
    }
)
```

---

## 方法签名与参数

### 完整方法签名

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[[StateT], str] | Callable[[StateT], Awaitable[str]],
    path_map: dict[str, str] | None = None,
    then: str | None = None,
) -> Self:
    """
    添加条件边到图中
    
    参数:
        source: 源节点名称
        path: 路由函数，接受状态返回目标节点名称
        path_map: 可选的路由映射字典
        then: 可选的汇聚节点，所有分支执行后都会到达此节点
    
    返回:
        Self: 支持链式调用
    """
```

**来源**：LangGraph 源码 state.py:839-887

### 参数详解

**source**：
- 条件边的起始节点
- 必须是已添加的节点名称
- 不能是 END

**path**：
- 路由函数，决定下一步流向
- 支持同步和异步函数
- 必须返回字符串或 END

**path_map**：
- 可选的映射字典
- 键：路由函数返回值
- 值：实际节点名称

**then**：
- 可选的汇聚节点
- 所有分支执行后都会到达此节点
- 用于多分支后的统一处理

---

## 实战示例

### 示例 1：简单循环

```python
from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

def should_continue(state: State) -> Literal["increment", END]:
    if state["count"] < 5:
        return "increment"
    return END

builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges("increment", should_continue)

graph = builder.compile()
result = graph.invoke({"count": 0})
print(result)  # {'count': 5}
```

**来源**：Context7 官方文档

### 示例 2：多分支路由

```python
class State(TypedDict):
    value: int
    result: str

def classify(state: State) -> Literal["small", "medium", "large"]:
    if state["value"] < 10:
        return "small"
    elif state["value"] < 100:
        return "medium"
    else:
        return "large"

def handle_small(state: State) -> dict:
    return {"result": "处理小值"}

def handle_medium(state: State) -> dict:
    return {"result": "处理中值"}

def handle_large(state: State) -> dict:
    return {"result": "处理大值"}

builder = StateGraph(State)
builder.add_node("handle_small", handle_small)
builder.add_node("handle_medium", handle_medium)
builder.add_node("handle_large", handle_large)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier",
    classify,
    path_map={
        "small": "handle_small",
        "medium": "handle_medium",
        "large": "handle_large"
    }
)

# 所有分支都汇聚到 END
builder.add_edge("handle_small", END)
builder.add_edge("handle_medium", END)
builder.add_edge("handle_large", END)
```

**来源**：Reddit 社区实践案例

### 示例 3：错误重试模式

```python
class State(TypedDict):
    error: bool
    retry_count: int
    max_retries: int

def check_result(state: State) -> Literal["retry", "success", "failed"]:
    if state["error"]:
        if state["retry_count"] < state["max_retries"]:
            return "retry"
        else:
            return "failed"
    return "success"

def retry_node(state: State) -> dict:
    return {"retry_count": state["retry_count"] + 1}

builder = StateGraph(State)
builder.add_node("process", process_fn)
builder.add_node("retry", retry_node)
builder.add_node("success", success_fn)
builder.add_node("failed", failed_fn)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    check_result,
    path_map={
        "retry": "retry",
        "success": "success",
        "failed": "failed"
    }
)
builder.add_edge("retry", "process")  # 重试后回到 process
builder.add_edge("success", END)
builder.add_edge("failed", END)
```

**来源**：Twitter 最佳实践

---

## 常见模式

### 模式 1：循环执行

```python
def should_continue(state: State) -> Literal["continue", END]:
    return "continue" if state["condition"] else END

builder.add_conditional_edges("node", should_continue)
builder.add_edge("continue", "node")  # 循环回自己
```

### 模式 2：多路分发

```python
def route(state: State) -> Literal["path1", "path2", "path3"]:
    # 根据状态选择路径
    ...

builder.add_conditional_edges("dispatcher", route)
```

### 模式 3：汇聚模式

```python
builder.add_conditional_edges(
    "router",
    route_fn,
    then="aggregator"  # 所有分支后都到 aggregator
)
```

**来源**：Context7 官方文档 + Reddit 社区实践

---

## 最佳实践

### 1. 使用 Literal 类型提示

```python
# ✅ 推荐
def route(state: State) -> Literal["a", "b", END]:
    ...

# ❌ 不推荐
def route(state: State) -> str:
    ...
```

### 2. 路由函数保持纯净

```python
# ✅ 推荐：只读取状态
def route(state: State) -> Literal["next", END]:
    return "next" if state["count"] < 10 else END

# ❌ 不推荐：修改状态
def route(state: State) -> Literal["next", END]:
    state["count"] += 1  # 不要在路由函数中修改状态
    return "next"
```

### 3. 使用 path_map 提高可读性

```python
# ✅ 推荐：语义化的路由名称
def route(state: State) -> Literal["needs_review", "auto_approve", "reject"]:
    ...

builder.add_conditional_edges(
    "validator",
    route,
    path_map={
        "needs_review": "human_review_node",
        "auto_approve": "approval_node",
        "reject": "rejection_node"
    }
)
```

### 4. 避免过深的嵌套

```python
# ❌ 不推荐：过深的条件嵌套
def complex_route(state: State) -> Literal["a", "b", "c", "d", "e"]:
    if state["x"]:
        if state["y"]:
            if state["z"]:
                return "a"
            return "b"
        return "c"
    return "d"

# ✅ 推荐：扁平化逻辑
def simple_route(state: State) -> Literal["a", "b", "c", "d"]:
    if state["x"] and state["y"] and state["z"]:
        return "a"
    if state["x"] and state["y"]:
        return "b"
    if state["x"]:
        return "c"
    return "d"
```

**来源**：Twitter 最佳实践 + Reddit 社区经验

---

## 常见陷阱

### 陷阱 1：路由目标不存在

```python
# ❌ 错误：路由到不存在的节点
def route(state: State) -> Literal["node_x", END]:
    return "node_x"

builder.add_conditional_edges("start", route)
# 编译时会报错：node_x 不存在
```

### 陷阱 2：忘记处理所有分支

```python
# ❌ 错误：某些分支没有出口
builder.add_conditional_edges("router", route_fn)
builder.add_edge("path_a", END)
# 忘记添加 path_b 的边，导致图不完整
```

### 陷阱 3：循环无终止条件

```python
# ❌ 错误：永远返回 "continue"
def bad_route(state: State) -> Literal["continue", END]:
    return "continue"  # 永远不会到 END

builder.add_conditional_edges("node", bad_route)
builder.add_edge("continue", "node")  # 无限循环
```

**来源**：Reddit 社区常见问题

---

## 与 add_edge 的对比

| 特性 | add_edge | add_conditional_edges |
|------|----------|----------------------|
| 路由方式 | 静态，固定目标 | 动态，运行时决定 |
| 参数 | 起点、终点 | 起点、路由函数、映射 |
| 使用场景 | 确定的流程 | 需要分支决策 |
| 复杂度 | 简单 | 较复杂 |
| 灵活性 | 低 | 高 |

**来源**：Context7 官方文档

---

## 总结

**add_conditional_edges 是 LangGraph 实现动态工作流的核心机制，通过路由函数和 path_map 实现灵活的条件分支。**

**关键要点**：
1. 路由函数决定下一步流向
2. path_map 提供逻辑名称到节点的映射
3. Literal 类型提示提供类型安全
4. BranchSpec 是内部实现的数据结构
5. 支持循环、多分支、汇聚等多种模式

**最佳实践**：
- 使用 Literal 类型提示
- 保持路由函数纯净
- 使用 path_map 提高可读性
- 避免过深的条件嵌套
- 确保所有分支都有出口

**来源汇总**：
- LangGraph 源码：state.py:839-887
- Context7 官方文档：/websites/langchain_oss_python_langgraph
- Reddit 社区实践：r/LangChain
- Twitter 最佳实践：@LangChain, @saen_dev

---

**文档版本**：v1.0  
**最后更新**：2026-02-25  
**维护者**：Claude Code
