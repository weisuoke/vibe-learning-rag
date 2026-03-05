# 核心概念 7：START 与 END 节点

> 本文档详细讲解 LangGraph 中的 START 与 END 特殊节点，包括它们的本质、使用方法和实战模式。

---

## 概述

START 和 END 是 LangGraph 中的两个特殊节点常量，用于标记图的入口点和出口点。它们不是普通的节点，而是图结构的边界标记，帮助 LangGraph 理解工作流的起点和终点。

**核心价值：**
- 明确图的执行起点和终点
- 支持条件路由到终点
- 简化图结构的定义
- 提供清晰的语义表达

---

## 核心概念 1：START 与 END 特殊节点常量

### 一句话定义

**START 和 END 是 LangGraph 预定义的特殊节点常量，分别表示图的入口点和出口点，用于标记工作流的边界。**

### 源码定义

**来源：** `langgraph/graph/__init__.py` [source_StateGraph_01.md]

```python
from langgraph.constants import END, START

__all__ = (
    "END",
    "START",
    "StateGraph",
    # ...
)
```

**关键点：**
- START 和 END 是从 `langgraph.constants` 导入的常量
- 它们是字符串类型的特殊标识符
- 在整个 LangGraph 框架中具有特殊含义

### 详细解释

#### 1. START 节点

**本质：** START 是图的虚拟入口节点，表示工作流的起点。

**特性：**
- 不是真实的执行节点，不包含任何逻辑
- 只能作为边的起点，不能作为终点
- 每个图必须有至少一条从 START 出发的边
- 可以连接到多个节点（并行执行）

**使用场景：**
```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)
builder.add_node("first_node", first_node_fn)

# START 作为边的起点
builder.add_edge(START, "first_node")
```

#### 2. END 节点

**本质：** END 是图的虚拟出口节点，表示工作流的终点。

**特性：**
- 不是真实的执行节点，不包含任何逻辑
- 只能作为边的终点，不能作为起点
- 可以有多条边指向 END（多个退出点）
- 到达 END 后，图的执行结束

**使用场景：**
```python
# END 作为边的终点
builder.add_edge("last_node", END)

# 条件路由到 END
def route(state: State) -> Literal["continue", END]:
    if state["done"]:
        return END
    return "continue"

builder.add_conditional_edges("decision_node", route)
```

### 代码示例

**来源：** Context7 官方文档 [context7_langgraph_01.md]

```python
from typing import TypedDict, Literal
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    """增加计数"""
    return {"count": state["count"] + 1}

def check_done(state: State) -> Literal["increment", END]:
    """检查是否完成"""
    if state["count"] >= 5:
        return END
    return "increment"

# 构建图
builder = StateGraph(State)
builder.add_node("increment", increment)

# START 连接到第一个节点
builder.add_edge(START, "increment")

# 条件路由：继续或结束
builder.add_conditional_edges("increment", check_done)

# 编译并执行
graph = builder.compile()
result = graph.invoke({"count": 0})
print(result)  # {'count': 5}
```

### 在 LangGraph 中的应用

**1. 线性工作流**

```python
# START -> node1 -> node2 -> END
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)
```

**2. 并行执行**

```python
# START -> [node1, node2] -> merge -> END
builder.add_edge(START, "node1")
builder.add_edge(START, "node2")
builder.add_edge(["node1", "node2"], "merge")
builder.add_edge("merge", END)
```

**3. 条件终止**

```python
# START -> decision -> [continue, END]
builder.add_edge(START, "decision")
builder.add_conditional_edges("decision", route_fn)
```

---

## 核心概念 2：set_entry_point 与 set_finish_point 方法

### 一句话定义

**set_entry_point 和 set_finish_point 是 StateGraph 的便捷方法，分别等价于 add_edge(START, node) 和 add_edge(node, END)。**

### 源码示例

**来源：** 源码分析 [source_StateGraph_01.md]

```python
# 示例代码（state.py:141-180）
graph = StateGraph(state_schema=State, context_schema=Context)

def node(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)
    x = state["x"][-1]
    next_value = x * r * (1 - x)
    return {"x": next_value}

graph.add_node("A", node)

# 使用 set_entry_point 和 set_finish_point
graph.set_entry_point("A")
graph.set_finish_point("A")

compiled = graph.compile()
```

### 详细解释

#### 1. set_entry_point(node_name)

**功能：** 设置图的入口点，等价于 `add_edge(START, node_name)`

**签名：**
```python
def set_entry_point(self, node_name: str) -> Self:
    """Set the entry point of the graph."""
    return self.add_edge(START, node_name)
```

**使用场景：**
- 当只有一个入口节点时，使用 set_entry_point 更语义化
- 代码更简洁易读

**示例对比：**
```python
# 方式 1：使用 add_edge
builder.add_edge(START, "first_node")

# 方式 2：使用 set_entry_point（推荐）
builder.set_entry_point("first_node")
```

#### 2. set_finish_point(node_name)

**功能：** 设置图的出口点，等价于 `add_edge(node_name, END)`

**签名：**
```python
def set_finish_point(self, node_name: str) -> Self:
    """Set the finish point of the graph."""
    return self.add_edge(node_name, END)
```

**使用场景：**
- 当只有一个出口节点时，使用 set_finish_point 更语义化
- 明确表达"这是最后一个节点"

**示例对比：**
```python
# 方式 1：使用 add_edge
builder.add_edge("last_node", END)

# 方式 2：使用 set_finish_point（推荐）
builder.set_finish_point("last_node")
```

### 代码示例

**完整示例：**

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    message: str
    processed: bool

def process(state: State) -> dict:
    """处理消息"""
    return {
        "message": state["message"].upper(),
        "processed": True
    }

# 构建图
builder = StateGraph(State)
builder.add_node("process", process)

# 使用便捷方法设置入口和出口
builder.set_entry_point("process")
builder.set_finish_point("process")

# 编译并执行
graph = builder.compile()
result = graph.invoke({"message": "hello", "processed": False})
print(result)  # {'message': 'HELLO', 'processed': True}
```

### 在 LangGraph 中的应用

**1. 简单线性流程**

```python
builder.set_entry_point("step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.set_finish_point("step3")
```

**2. 单节点图**

```python
# 最简单的图：START -> node -> END
builder.add_node("single_node", node_fn)
builder.set_entry_point("single_node")
builder.set_finish_point("single_node")
```

**3. 与条件路由结合**

```python
builder.set_entry_point("start_node")
builder.add_conditional_edges("start_node", route_fn)
# 某些分支可能直接到 END，某些分支继续执行
```

---

## 核心概念 3：条件路由到 END

### 一句话定义

**条件路由到 END 是指在条件边的路由函数中返回 END 常量，实现动态终止工作流的能力。**

### 详细解释

#### 1. 条件路由的本质

**来源：** Context7 官方文档 [context7_langgraph_01.md]

条件路由允许根据状态动态决定下一个节点，包括：
- 返回普通节点名称（字符串）
- 返回 END 常量（终止执行）

**关键点：**
- 路由函数的返回类型必须包含 END
- 使用 `Literal` 类型提示所有可能的返回值
- END 是一个有效的路由目标

#### 2. 路由函数签名

```python
from typing import Literal
from langgraph.graph import END

def route(state: State) -> Literal["next_node", "another_node", END]:
    """
    根据状态决定下一步

    返回值：
    - "next_node": 继续到 next_node
    - "another_node": 继续到 another_node
    - END: 终止执行
    """
    if state["done"]:
        return END
    elif state["retry"]:
        return "another_node"
    else:
        return "next_node"
```

#### 3. 使用场景

**场景 1：循环终止条件**

```python
def should_continue(state: State) -> Literal["loop", END]:
    """检查是否继续循环"""
    if state["iterations"] >= state["max_iterations"]:
        return END
    return "loop"

builder.add_edge(START, "loop")
builder.add_conditional_edges("loop", should_continue)
```

**场景 2：多分支决策**

```python
def route_decision(state: State) -> Literal["path_a", "path_b", END]:
    """根据条件选择路径"""
    if state["error"]:
        return END  # 错误时直接终止
    elif state["use_path_a"]:
        return "path_a"
    else:
        return "path_b"
```

**场景 3：提前退出**

```python
def check_validity(state: State) -> Literal["process", END]:
    """验证输入，无效时提前退出"""
    if not state["valid"]:
        return END  # 无效输入，不继续处理
    return "process"
```

### 代码示例

**来源：** Context7 官方文档 [context7_langgraph_01.md]

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # 使用 operator.add 作为 reducer，列表只增不减
    aggregate: Annotated[list, operator.add]

def a(state: State):
    """节点 A：添加 'A' 到列表"""
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    """节点 B：添加 'B' 到列表"""
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

def route(state: State) -> Literal["b", END]:
    """路由函数：列表长度小于 7 时继续，否则结束"""
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

# 构建图
builder = StateGraph(State)
builder.add_node("a", a)
builder.add_node("b", b)

# 设置边
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)  # a -> [b, END]
builder.add_edge("b", "a")  # b -> a（形成循环）

# 编译并执行
graph = builder.compile()
result = graph.invoke({"aggregate": []})

print(f"Final result: {result}")
# 输出：
# Node A sees []
# Node B sees ['A']
# Node A sees ['A', 'B']
# Node B sees ['A', 'B', 'A']
# Node A sees ['A', 'B', 'A', 'B']
# Node B sees ['A', 'B', 'A', 'B', 'A']
# Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
# Final result: {'aggregate': ['A', 'B', 'A', 'B', 'A', 'B', 'A']}
```

### 在 LangGraph 中的应用

**1. 迭代优化流程**

```python
def should_refine(state: State) -> Literal["refine", END]:
    """检查是否需要继续优化"""
    if state["quality_score"] >= state["target_score"]:
        return END
    if state["iterations"] >= state["max_iterations"]:
        return END
    return "refine"
```

**2. 多代理协作**

**来源：** Reddit 实践案例 [search_StateGraph_03.md]

```python
def supervisor_route(state: State) -> Literal["agent1", "agent2", END]:
    """Supervisor 决定下一步"""
    if state["task_complete"]:
        return END
    elif state["need_research"]:
        return "agent1"
    else:
        return "agent2"

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", supervisor_route)
```

**3. 错误处理**

```python
def error_handler(state: State) -> Literal["retry", "fallback", END]:
    """错误处理路由"""
    if state["error_count"] >= 3:
        return END  # 超过重试次数，终止
    elif state["can_retry"]:
        return "retry"
    else:
        return "fallback"
```

---

## 实战模式总结

### 模式 1：线性流程

```python
# START -> node1 -> node2 -> END
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

# 或使用便捷方法
builder.set_entry_point("node1")
builder.add_edge("node1", "node2")
builder.set_finish_point("node2")
```

### 模式 2：条件分支

```python
# START -> decision -> [path_a -> END, path_b -> END]
builder.add_edge(START, "decision")
builder.add_conditional_edges("decision", route_fn)
builder.add_edge("path_a", END)
builder.add_edge("path_b", END)
```

### 模式 3：循环 + 条件终止

```python
# START -> loop -> [loop, END]
builder.add_edge(START, "loop")
builder.add_conditional_edges("loop", should_continue)
```

### 模式 4：并行执行

```python
# START -> [node1, node2] -> merge -> END
builder.add_edge(START, "node1")
builder.add_edge(START, "node2")
builder.add_edge(["node1", "node2"], "merge")
builder.add_edge("merge", END)
```

---

## 常见陷阱与最佳实践

### 陷阱 1：START 作为终点

**错误示例：**
```python
# ❌ 错误：START 不能作为边的终点
builder.add_edge("some_node", START)
```

**正确做法：**
```python
# ✅ 正确：START 只能作为起点
builder.add_edge(START, "some_node")
```

### 陷阱 2：END 作为起点

**错误示例：**
```python
# ❌ 错误：END 不能作为边的起点
builder.add_edge(END, "some_node")
```

**正确做法：**
```python
# ✅ 正确：END 只能作为终点
builder.add_edge("some_node", END)
```

### 陷阱 3：忘记类型提示

**错误示例：**
```python
# ❌ 错误：缺少 END 的类型提示
def route(state: State) -> str:
    if state["done"]:
        return END  # 类型检查会报错
    return "next_node"
```

**正确做法：**
```python
# ✅ 正确：使用 Literal 包含 END
def route(state: State) -> Literal["next_node", END]:
    if state["done"]:
        return END
    return "next_node"
```

### 最佳实践

**来源：** Twitter 最佳实践 [search_StateGraph_02.md]

1. **优先使用 set_entry_point/set_finish_point**
   - 当只有一个入口/出口时，使用便捷方法更清晰

2. **明确条件路由的所有可能返回值**
   - 使用 `Literal` 类型提示所有可能的路由目标
   - 包括 END 在内的所有分支

3. **避免无终止条件的循环**
   - 确保循环路由中有明确的 END 条件
   - 设置最大迭代次数作为安全网

4. **使用语义化的节点名称**
   - 节点名称应该清晰表达其功能
   - 便于理解图的执行流程

---

## 参考资料

1. **源码分析**
   - `langgraph/graph/__init__.py` - START/END 导出
   - `langgraph/graph/state.py` - set_entry_point/set_finish_point 实现
   - 来源：[source_StateGraph_01.md]

2. **Context7 官方文档**
   - StateGraph 基础创建
   - 条件路由示例
   - 来源：[context7_langgraph_01.md]

3. **社区实践**
   - Reddit 实践案例：Supervisor 模式
   - Twitter 最佳实践：早期建模为状态机
   - 来源：[search_StateGraph_02.md, search_StateGraph_03.md]

---

**文档版本：** v1.0
**最后更新：** 2026-02-25
**知识点：** 01_StateGraph与节点定义 - 核心概念 7
