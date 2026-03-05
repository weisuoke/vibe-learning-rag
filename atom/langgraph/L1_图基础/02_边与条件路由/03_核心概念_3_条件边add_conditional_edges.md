# 03_核心概念_3_条件边add_conditional_edges

## 概念定义

**条件边（Conditional Edge）是 LangGraph 中的动态路由边，通过 `add_conditional_edges` 方法添加，根据状态值动态选择目标节点。**

---

## 数据来源

本文档基于以下资料生成：
- [来源: reference/source_edges_02_add_conditional_edges.md] - add_conditional_edges 方法源码分析
- [来源: reference/context7_langgraph_01_edges_routing.md] - LangGraph 官方文档

---

## 核心要点

### 什么是条件边？

条件边是动态路由的边，通过路由函数根据状态值选择不同的目标节点，实现分支决策和动态工作流。

### 条件边的三个特点

1. **动态路由**：根据状态值选择目标节点
2. **路由函数**：通过函数实现决策逻辑
3. **多目标支持**：可以路由到多个不同的节点

---

## 方法签名

```python
def add_conditional_edges(
    self,
    source: str,                              # 起始节点
    path: Callable[..., Hashable],           # 路由函数
    path_map: dict[Hashable, str] | None     # 路径映射（可选）
) -> Self:
    """从起始节点添加条件边到任意数量的目标节点
    
    Args:
        source: 起始节点，条件边会在退出该节点时运行
        path: 路由函数，决定下一个节点或多个节点
        path_map: 可选的路径映射
        
    Returns:
        Self: StateGraph 实例，支持链式调用
    """
```

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## 路由函数设计

### 基本结构

```python
from typing import Literal

def route_func(state: State) -> Literal["node_a", "node_b", END]:
    """路由函数示例
    
    Args:
        state: 当前状态
        
    Returns:
        目标节点名称
    """
    if state["condition"]:
        return "node_a"
    return "node_b"
```

### 关键要素

1. **输入参数**：接受 `state` 参数
2. **返回值**：返回目标节点名称（字符串）
3. **类型提示**：使用 `Literal` 明确可能的返回值
4. **决策逻辑**：基于状态值的条件判断

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## path_map 参数

### 三种形式

#### 形式 1：None（默认）

```python
def route_func(state: State) -> Literal["node_a", "node_b"]:
    return "node_a" if state["flag"] else "node_b"

graph.add_conditional_edges("source", route_func)
# 路由函数直接返回节点名称
```

#### 形式 2：dict（映射）

```python
def route_func(state: State) -> Literal["yes", "no"]:
    return "yes" if state["flag"] else "no"

graph.add_conditional_edges(
    "source",
    route_func,
    {"yes": "node_a", "no": "node_b"}  # 映射返回值到节点
)
```

#### 形式 3：list（列表）

```python
graph.add_conditional_edges(
    "source",
    route_func,
    ["node_a", "node_b", "node_c"]  # 用于验证和可视化
)
```

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## 实战示例

### 示例 1：简单条件分支

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class State(TypedDict):
    value: int
    path: str

def check(state: State) -> dict:
    return {"value": state["value"]}

def route_by_value(state: State) -> Literal["high", "low"]:
    """根据值路由"""
    if state["value"] > 50:
        return "high"
    return "low"

def process_high(state: State) -> dict:
    return {"path": "high", "value": state["value"] * 2}

def process_low(state: State) -> dict:
    return {"path": "low", "value": state["value"] + 10}

# 构建图
graph = StateGraph(State)
graph.add_node("check", check)
graph.add_node("high", process_high)
graph.add_node("low", process_low)

graph.add_edge(START, "check")
graph.add_conditional_edges(
    "check",
    route_by_value,
    {"high": "high", "low": "low"}
)
graph.add_edge("high", END)
graph.add_edge("low", END)

app = graph.compile()
result = app.invoke({"value": 75, "path": ""})
print(result)  # {'value': 150, 'path': 'high'}
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

### 示例 2：多路由决策

```python
from typing import Literal

def route_by_score(state: State) -> Literal["excellent", "good", "poor", END]:
    """根据分数多路由"""
    score = state["score"]
    if score > 0.9:
        return "excellent"
    elif score > 0.7:
        return "good"
    elif score > 0.5:
        return "poor"
    return END

graph.add_conditional_edges(
    "evaluate",
    route_by_score,
    {
        "excellent": "reward",
        "good": "continue",
        "poor": "retry",
        END: END
    }
)
```

---

### 示例 3：循环路由

```python
def route_with_retry(state: State) -> Literal["retry", "success", END]:
    """带重试的路由"""
    if state.get("error") and state.get("retry_count", 0) < 3:
        return "retry"
    elif state.get("success"):
        return "success"
    return END

graph.add_conditional_edges(
    "process",
    route_with_retry,
    {
        "retry": "process",  # 循环回自己
        "success": "finalize",
        END: END
    }
)
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## BranchSpec 对象

### 定义

`BranchSpec` 是条件边的规范对象，封装了路由函数和路径映射。

### 创建方式

```python
# 内部实现（源码）
BranchSpec.from_path(path, path_map, infer_schema=True)
```

### 存储结构

```python
self.branches: defaultdict[str, dict[str, BranchSpec]]

# 示例
{
    "node1": {
        "condition1": BranchSpec(...),
        "condition2": BranchSpec(...)
    }
}
```

[来源: reference/source_edges_02_add_conditional_edges.md]

---

## 设计模式

### 模式 1：简单分支（if-else）

```python
def route(state) -> Literal["a", "b"]:
    return "a" if condition else "b"
```

### 模式 2：多路由（switch-case）

```python
def route(state) -> Literal["a", "b", "c", END]:
    if cond_a: return "a"
    elif cond_b: return "b"
    elif cond_c: return "c"
    return END
```

### 模式 3：循环路由（retry）

```python
def route(state) -> Literal["retry", END]:
    if should_retry(state):
        return "retry"
    return END
```

### 模式 4：入口路由

```python
graph.add_conditional_edges(START, route_func, {...})
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## 最佳实践

### 1. 使用类型提示

```python
# ✅ 推荐
def route(state: State) -> Literal["a", "b", END]:
    ...

# ❌ 不推荐
def route(state):
    ...
```

### 2. 保持逻辑简单

```python
# ✅ 推荐：扁平化逻辑
def route(state):
    if state["a"] and state["b"]:
        return "path1"
    if state["a"]:
        return "path2"
    return "path3"

# ❌ 不推荐：复杂嵌套
def route(state):
    if state["a"]:
        if state["b"]:
            return "path1"
        else:
            return "path2"
    else:
        return "path3"
```

### 3. 处理边界情况

```python
def route(state):
    # 处理缺失值
    score = state.get("score", 0.0)
    
    # 处理异常值
    if score < 0 or score > 1:
        return "error"
    
    # 正常路由
    return "pass" if score > 0.5 else "fail"
```

### 4. 添加日志

```python
def route(state):
    score = state["score"]
    print(f"Routing: score={score}")
    
    if score > 0.8:
        print("→ high_quality")
        return "high_quality"
    print("→ low_quality")
    return "low_quality"
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## 常见错误

### 错误 1：返回未定义的节点

```python
# ❌ 错误
def route(state):
    return "undefined_node"  # 节点不存在

# ✅ 正确
def route(state) -> Literal["node_a", "node_b"]:
    return "node_a"
```

### 错误 2：path_map 不匹配

```python
# ❌ 错误
def route(state):
    return "yes"

graph.add_conditional_edges(
    "source",
    route,
    {"true": "node_a"}  # 没有 "yes"
)

# ✅ 正确
graph.add_conditional_edges(
    "source",
    route,
    {"yes": "node_a", "no": "node_b"}
)
```

### 错误 3：忘记 END 路由

```python
# ❌ 错误：无限循环
def route(state):
    return "process"  # 永远不结束

# ✅ 正确
def route(state) -> Literal["process", END]:
    if state.get("retry_count", 0) < 3:
        return "process"
    return END
```

[来源: reference/context7_langgraph_01_edges_routing.md]

---

## 参考资料

- [来源: reference/source_edges_02_add_conditional_edges.md] - add_conditional_edges 源码分析
- [来源: reference/context7_langgraph_01_edges_routing.md] - LangGraph 官方文档

---

**记住**：条件边是 LangGraph 的动态路由机制，通过路由函数根据状态值选择执行路径，实现复杂的分支决策和自适应工作流。
