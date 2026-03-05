# 核心概念 5：路由映射 path_map

> 本文档详细讲解 LangGraph 条件边中的路由映射（path_map）机制

---

## 什么是路由映射 path_map？

**路由映射（path_map）是条件边中将路由函数返回值映射到实际节点名称的机制。**

在 `add_conditional_edges` 方法中，`path_map` 参数定义了如何将路由函数的返回值转换为图中的节点名称。

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## path_map 的三种形式

### 形式 1：None（默认）

**定义**：不提供 `path_map` 参数，路由函数直接返回节点名称。

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def route_func(state: State) -> Literal["node_a", "node_b", END]:
    """路由函数直接返回节点名称"""
    if state["value"] > 50:
        return "node_a"
    elif state["value"] > 0:
        return "node_b"
    else:
        return END

graph = StateGraph(State)
graph.add_node("process", lambda s: s)
graph.add_node("node_a", lambda s: s)
graph.add_node("node_b", lambda s: s)

# path_map=None（默认）
graph.add_conditional_edges(
    "process",
    route_func
    # 不提供 path_map，路由函数直接返回节点名称
)
```

**关键点**：
1. **类型提示必需**：必须使用 `Literal` 类型提示明确可能的返回值
2. **返回值即节点名**：路由函数返回的字符串必须是已定义的节点名称
3. **图可视化**：类型提示帮助图可视化工具理解可能的路由路径

[来源: reference/source_edges_02_add_conditional_edges.md]

---

### 形式 2：dict（映射字典）

**定义**：提供一个字典，将路由函数返回的键映射到节点名称。

```python
def route_func(state: State) -> str:
    """路由函数返回键（不是节点名称）"""
    if state["value"] > 50:
        return "high"  # 返回键
    elif state["value"] > 0:
        return "medium"  # 返回键
    else:
        return "low"  # 返回键

graph.add_conditional_edges(
    "process",
    route_func,
    {
        "high": "node_a",      # 键 -> 节点名称
        "medium": "node_b",    # 键 -> 节点名称
        "low": END             # 键 -> END
    }
)
```

**关键点**：
1. **解耦路由逻辑**：路由函数返回的是逻辑键（如 "high"），而不是具体节点名称
2. **灵活性**：可以在不修改路由函数的情况下更改目标节点
3. **可读性**：路由逻辑更清晰，键名可以表达业务含义

**应用场景**：
- 路由逻辑与节点名称解耦
- 多个路由函数共享相同的映射
- 业务逻辑需要清晰的语义键

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 形式 3：list（节点列表）

**定义**：提供一个节点名称列表，用于图可视化和验证。

```python
def route_func(state: State) -> str:
    """路由函数返回节点名称"""
    if state["value"] > 50:
        return "node_a"
    elif state["value"] > 0:
        return "node_b"
    else:
        return "node_c"

graph.add_conditional_edges(
    "process",
    route_func,
    ["node_a", "node_b", "node_c"]  # 可能的目标节点列表
)
```

**关键点**：
1. **验证作用**：列表中的节点名称用于验证路由函数返回值
2. **可视化辅助**：帮助图可视化工具理解可能的路由路径
3. **不做映射**：列表不做键值映射，路由函数仍然返回节点名称

**与 None 的区别**：
- `None`：依赖类型提示（`Literal`）
- `list`：显式列出可能的目标节点

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## path_map 的工作原理

### 源码实现

```python
# StateGraph.add_conditional_edges 方法
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    # ...
    # 保存条件边
    self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
    # ...
```

**BranchSpec.from_path 方法**：
- 接受 `path`（路由函数）和 `path_map`（路径映射）
- 创建 `BranchSpec` 对象，封装路由逻辑
- `infer_schema=True` 表示推断输入模式

[来源: reference/source_edges_02_add_conditional_edges.md]

---

### 执行流程

```
1. 节点执行完成
   ↓
2. 调用路由函数（传入当前状态）
   ↓
3. 路由函数返回值（键或节点名称）
   ↓
4. 应用 path_map 映射（如果提供）
   ↓
5. 获取目标节点名称
   ↓
6. 执行目标节点
```

**示例**：

```python
# 路由函数返回 "high"
route_result = route_func(state)  # "high"

# 应用 path_map 映射
path_map = {"high": "node_a", "medium": "node_b", "low": END}
target_node = path_map[route_result]  # "node_a"

# 执行目标节点
execute_node(target_node)
```

---

## 三种形式的对比

| 特性 | None | dict | list |
|------|------|------|------|
| **路由函数返回值** | 节点名称 | 键（任意） | 节点名称 |
| **类型提示** | 必需（`Literal`） | 可选 | 可选 |
| **映射作用** | 无映射 | 键 → 节点 | 无映射 |
| **灵活性** | 低 | 高 | 低 |
| **可读性** | 中 | 高 | 中 |
| **验证** | 类型提示 | 运行时 | 运行时 |
| **适用场景** | 简单路由 | 复杂业务逻辑 | 需要显式列出目标 |

---

## 实战示例

### 示例 1：使用 None（简单场景）

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict

class State(TypedDict):
    score: float

def simple_route(state: State) -> Literal["pass", "fail"]:
    """简单的及格/不及格路由"""
    return "pass" if state["score"] >= 60 else "fail"

def pass_handler(state: State) -> dict:
    return {"result": "通过"}

def fail_handler(state: State) -> dict:
    return {"result": "不通过"}

# 构建图
graph = StateGraph(State)
graph.add_node("check", lambda s: s)
graph.add_node("pass", pass_handler)
graph.add_node("fail", fail_handler)

graph.add_edge(START, "check")
graph.add_conditional_edges(
    "check",
    simple_route
    # 不提供 path_map，依赖类型提示
)
graph.add_edge("pass", END)
graph.add_edge("fail", END)

app = graph.compile()

# 测试
result = app.invoke({"score": 75})
print(result)  # {'score': 75, 'result': '通过'}
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 示例 2：使用 dict（业务逻辑解耦）

```python
class State(TypedDict):
    priority: str
    urgency: str

def business_route(state: State) -> str:
    """业务路由逻辑"""
    if state["priority"] == "high" and state["urgency"] == "urgent":
        return "critical"  # 业务键
    elif state["priority"] == "high":
        return "important"  # 业务键
    elif state["urgency"] == "urgent":
        return "urgent"  # 业务键
    else:
        return "normal"  # 业务键

# 构建图
graph = StateGraph(State)
graph.add_node("classify", lambda s: s)
graph.add_node("critical_handler", lambda s: {"status": "立即处理"})
graph.add_node("important_handler", lambda s: {"status": "优先处理"})
graph.add_node("urgent_handler", lambda s: {"status": "加急处理"})
graph.add_node("normal_handler", lambda s: {"status": "正常处理"})

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    business_route,
    {
        "critical": "critical_handler",
        "important": "important_handler",
        "urgent": "urgent_handler",
        "normal": "normal_handler"
    }
)
graph.add_edge("critical_handler", END)
graph.add_edge("important_handler", END)
graph.add_edge("urgent_handler", END)
graph.add_edge("normal_handler", END)

app = graph.compile()

# 测试
result = app.invoke({"priority": "high", "urgency": "urgent"})
print(result)  # {'priority': 'high', 'urgency': 'urgent', 'status': '立即处理'}
```

**优势**：
- 路由函数返回业务语义键（"critical"、"important"）
- 节点名称可以随意更改，不影响路由逻辑
- 业务逻辑清晰，易于维护

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 示例 3：使用 list（显式验证）

```python
class State(TypedDict):
    category: str

def category_route(state: State) -> str:
    """分类路由"""
    return state["category"]

# 构建图
graph = StateGraph(State)
graph.add_node("router", lambda s: s)
graph.add_node("category_a", lambda s: {"result": "A类处理"})
graph.add_node("category_b", lambda s: {"result": "B类处理"})
graph.add_node("category_c", lambda s: {"result": "C类处理"})

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    category_route,
    ["category_a", "category_b", "category_c"]  # 显式列出可能的目标
)
graph.add_edge("category_a", END)
graph.add_edge("category_b", END)
graph.add_edge("category_c", END)

app = graph.compile()

# 测试
result = app.invoke({"category": "category_a"})
print(result)  # {'category': 'category_a', 'result': 'A类处理'}
```

**优势**：
- 显式列出所有可能的目标节点
- 帮助图可视化工具理解路由路径
- 运行时验证路由函数返回值

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## 常见错误与解决方案

### 错误 1：path_map 与返回值不匹配

```python
# ❌ 错误
def route_func(state):
    return "yes"  # 返回 "yes"

graph.add_conditional_edges(
    "source",
    route_func,
    {"true": "node_a", "false": "node_b"}  # path_map 中没有 "yes"
)
```

**错误原因**：路由函数返回 "yes"，但 `path_map` 中只有 "true" 和 "false"。

**解决方案**：

```python
# ✅ 正确
def route_func(state):
    return "yes"

graph.add_conditional_edges(
    "source",
    route_func,
    {"yes": "node_a", "no": "node_b"}  # 匹配返回值
)
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 错误 2：路由函数返回未定义的节点

```python
# ❌ 错误
def route_func(state):
    return "undefined_node"  # 节点不存在

graph.add_conditional_edges("source", route_func)
```

**错误原因**：路由函数返回的节点名称在图中不存在。

**解决方案**：

```python
# ✅ 正确
def route_func(state) -> Literal["node_a", "node_b"]:
    return "node_a"  # 确保节点已定义

# 先添加节点
graph.add_node("node_a", lambda s: s)
graph.add_node("node_b", lambda s: s)

# 再添加条件边
graph.add_conditional_edges("source", route_func)
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 错误 3：忘记类型提示（使用 None 时）

```python
# ❌ 错误：没有类型提示
def route_func(state):
    return "node_a"

graph.add_conditional_edges("source", route_func)
# 图可视化会假设可以转到任何节点
```

**警告信息**：
> Without type hints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
> or a path_map, the graph visualization assumes the edge could transition to any node in the graph.

**解决方案**：

```python
# ✅ 正确：添加类型提示
def route_func(state) -> Literal["node_a", "node_b", END]:
    return "node_a"

graph.add_conditional_edges("source", route_func)
```

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## 最佳实践

### 1. 选择合适的 path_map 形式

**使用 None**：
- 简单的路由逻辑
- 节点名称清晰且不会改变
- 路由目标少（2-3个）

**使用 dict**：
- 复杂的业务逻辑
- 需要解耦路由逻辑和节点名称
- 路由键有明确的业务含义

**使用 list**：
- 需要显式列出所有可能的目标
- 帮助图可视化
- 运行时验证路由函数返回值

---

### 2. 使用类型提示

```python
# ✅ 推荐：使用 Literal 类型提示
def route_func(state: State) -> Literal["path_a", "path_b", END]:
    ...

# ❌ 不推荐：没有类型提示
def route_func(state):
    ...
```

**优势**：
- 帮助 IDE 提供代码补全
- 帮助图可视化工具理解路由路径
- 提前发现类型错误

---

### 3. 使用有意义的键名

```python
# ✅ 推荐：业务语义清晰
path_map = {
    "high_quality": "premium_handler",
    "medium_quality": "standard_handler",
    "low_quality": "basic_handler"
}

# ❌ 不推荐：键名不清晰
path_map = {
    "a": "node1",
    "b": "node2",
    "c": "node3"
}
```

---

### 4. 处理边界情况

```python
def route_func(state: State) -> Literal["node_a", "node_b", END]:
    # 处理缺失值
    value = state.get("score", 0)

    # 处理异常值
    if value < 0 or value > 100:
        return END  # 异常情况直接结束

    # 正常路由
    if value > 50:
        return "node_a"
    return "node_b"
```

---

## 在状态化工作流中的应用

### RAG 工作流中的路由映射

```python
class RAGState(TypedDict):
    query: str
    documents: list[str]
    relevance_score: float

def rag_route(state: RAGState) -> str:
    """RAG 路由逻辑"""
    if state["relevance_score"] > 0.8:
        return "high_relevance"
    elif state["relevance_score"] > 0.5:
        return "medium_relevance"
    else:
        return "low_relevance"

graph.add_conditional_edges(
    "grade_documents",
    rag_route,
    {
        "high_relevance": "generate",           # 直接生成
        "medium_relevance": "rerank",           # 重排序
        "low_relevance": "transform_query"      # 转换查询
    }
)
```

**优势**：
- 业务逻辑清晰（"high_relevance"、"medium_relevance"）
- 节点名称可以灵活调整
- 易于扩展新的路由路径

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## 总结

**path_map 的三种形式**：
1. **None**：路由函数直接返回节点名称，依赖类型提示
2. **dict**：将路由函数返回的键映射到节点名称，解耦业务逻辑
3. **list**：显式列出可能的目标节点，用于验证和可视化

**选择建议**：
- 简单场景：使用 None + 类型提示
- 复杂业务：使用 dict 映射
- 需要验证：使用 list 列表

**最佳实践**：
- 始终使用类型提示（`Literal`）
- 使用有意义的键名
- 处理边界情况
- 确保 path_map 与返回值匹配

通过合理使用 path_map，可以构建清晰、灵活、易维护的状态化工作流。

---

**参考资料**：
- [source_edges_02_add_conditional_edges.md](reference/source_edges_02_add_conditional_edges.md) - path_map 参数源码分析
- [context7_langgraph_01_edges_routing.md](reference/context7_langgraph_01_edges_routing.md) - path_map 使用示例
