---
type: source_code_analysis
source: sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py
analyzed_files:
  - sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py (lines 839-887)
analyzed_at: 2026-02-25
knowledge_point: 02_边与条件路由
---

# 源码分析：add_conditional_edges 方法

## 分析的文件
- `sourcecode/langgraph/libs/langgraph/langgraph/graph/state.py:839-887` - StateGraph.add_conditional_edges 方法实现

## 方法签名

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
```

## 关键发现

### 1. 方法功能
`add_conditional_edges` 用于从起始节点添加条件边到任意数量的目标节点。

**核心特性**：
- 条件边会在退出起始节点时运行
- 通过路由函数（path）动态决定下一个节点
- 支持多目标路由

### 2. 参数说明

- `source: str` - 起始节点，条件边会在退出该节点时运行
- `path: Callable | Runnable` - 路由函数，决定下一个节点或多个节点
  - 如果不指定 `path_map`，应该返回节点名称
  - 如果返回 `'END'`，图会停止执行
- `path_map: dict | list | None` - 可选的路径映射
  - 如果省略，`path` 返回的路径应该是节点名称

### 3. 核心实现逻辑

#### A. 编译状态检查（lines 869-873）
```python
if self.compiled:
    logger.warning(
        "Adding an edge to a graph that has already been compiled. This will "
        "not be reflected in the compiled graph."
    )
```
**关键点**：与 `add_edge` 相同，如果图已编译，添加条件边不会生效。

#### B. 路由函数处理（lines 875-877）
```python
# find a name for the condition
path = coerce_to_runnable(path, name=None, trace=True)
name = path.name or "condition"
```

**关键点**：
1. **路由函数转换**：将路由函数转换为 Runnable 对象
2. **命名**：为条件边分配名称，默认为 "condition"
3. **追踪**：`trace=True` 表示启用追踪功能

#### C. 条件验证（lines 878-882）
```python
# validate the condition
if name in self.branches[source]:
    raise ValueError(
        f"Branch with name `{path.name}` already exists for node `{source}`"
    )
```

**关键点**：
- **唯一性检查**：同一个起始节点不能有重名的条件边
- **错误信息**：明确指出哪个节点的哪个分支已存在

#### D. 保存条件边（lines 883-887）
```python
# save it
self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
if schema := self.branches[source][name].input_schema:
    self._add_schema(schema)
return self
```

**关键点**：
1. **BranchSpec 创建**：使用 `BranchSpec.from_path` 创建分支规范
   - `path`: 路由函数
   - `path_map`: 路径映射
   - `True`: `infer_schema` 参数，推断输入模式
2. **Schema 注册**：如果分支有输入模式，注册到图中
3. **存储位置**：`self.branches[source][name]`
   - `branches` 是一个 `defaultdict[str, dict[str, BranchSpec]]`
   - 第一层键是起始节点名称
   - 第二层键是条件边名称

### 4. 条件边的存储结构

```python
# StateGraph 类定义中
branches: defaultdict[str, dict[str, BranchSpec]]
```

**结构说明**：
- 外层字典：键是起始节点名称，值是该节点的所有条件边
- 内层字典：键是条件边名称，值是 BranchSpec 对象

**示例**：
```python
{
    "node1": {
        "condition1": BranchSpec(...),
        "condition2": BranchSpec(...)
    },
    "node2": {
        "router": BranchSpec(...)
    }
}
```

### 5. BranchSpec 的作用

`BranchSpec` 是条件边的规范对象，包含：
- `path`: 路由函数（Runnable）
- `ends`: 路径映射（dict[Hashable, str] | None）
- `input_schema`: 输入模式（type[Any] | None）

**创建方式**：
```python
BranchSpec.from_path(path, path_map, infer_schema=True)
```

### 6. 路由函数的类型

路由函数可以是：
1. **同步函数**：`Callable[..., Hashable | Sequence[Hashable]]`
2. **异步函数**：`Callable[..., Awaitable[Hashable | Sequence[Hashable]]]`
3. **Runnable 对象**：`Runnable[Any, Hashable | Sequence[Hashable]]`

**返回值类型**：
- `Hashable`: 单个目标节点（字符串）
- `Sequence[Hashable]`: 多个目标节点（列表）

### 7. path_map 的三种形式

#### A. None（默认）
```python
graph.add_conditional_edges("node1", route_func)
```
- 路由函数直接返回节点名称
- 需要在路由函数的返回类型中使用 `Literal` 类型提示

#### B. dict（映射）
```python
graph.add_conditional_edges(
    "node1",
    route_func,
    {"yes": "node2", "no": "node3"}
)
```
- 路由函数返回键（"yes" 或 "no"）
- `path_map` 将键映射到节点名称

#### C. list（列表）
```python
graph.add_conditional_edges(
    "node1",
    route_func,
    ["node2", "node3", "node4"]
)
```
- 路由函数返回节点名称
- `path_map` 列表用于图可视化和验证

### 8. 类型提示的重要性

```python
!!! warning
    Without type hints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
    or a path_map, the graph visualization assumes the edge could transition to any node in the graph.
```

**关键点**：
- 没有类型提示或 `path_map` 时，图可视化会假设可以转到任何节点
- 使用 `Literal` 类型提示可以明确指定可能的目标节点

## 代码片段

### 完整实现
```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    """Add a conditional edge from the starting node to any number of destination nodes.

    Args:
        source: The starting node. This conditional edge will run when
            exiting this node.
        path: The callable that determines the next node or nodes.

            If not specifying `path_map` it should return one or more nodes.

            If it returns `'END'`, the graph will stop execution.
        path_map: Optional mapping of paths to node names.

            If omitted the paths returned by `path` should be node names.

    Returns:
        Self: The instance of the graph, allowing for method chaining.

    !!! warning
        Without type hints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
        or a path_map, the graph visualization assumes the edge could transition to any node in the graph.

    """
    if self.compiled:
        logger.warning(
            "Adding an edge to a graph that has already been compiled. This will "
            "not be reflected in the compiled graph."
        )

    # find a name for the condition
    path = coerce_to_runnable(path, name=None, trace=True)
    name = path.name or "condition"
    # validate the condition
    if name in self.branches[source]:
        raise ValueError(
            f"Branch with name `{path.name}` already exists for node `{source}`"
        )
    # save it
    self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
    if schema := self.branches[source][name].input_schema:
        self._add_schema(schema)
    return self
```

## 在 LangGraph 中的应用

### 1. 简单条件分支（if-else）
```python
def route_func(state):
    if state["value"] > 0:
        return "positive"
    else:
        return "negative"

graph.add_conditional_edges(
    "check",
    route_func,
    {"positive": "node_pos", "negative": "node_neg"}
)
```

### 2. 多路由决策（switch-case）
```python
def route_func(state) -> Literal["a", "b", "c", END]:
    if state["type"] == "A":
        return "a"
    elif state["type"] == "B":
        return "b"
    elif state["type"] == "C":
        return "c"
    else:
        return END

graph.add_conditional_edges("router", route_func)
```

### 3. 动态多目标路由
```python
def route_func(state):
    # 返回多个目标节点
    targets = []
    if state["need_validation"]:
        targets.append("validate")
    if state["need_logging"]:
        targets.append("log")
    return targets

graph.add_conditional_edges("process", route_func)
```

### 4. 快捷方法
- `set_conditional_entry_point(path, path_map)` = `add_conditional_edges(START, path, path_map)`

## 设计决策分析

### 1. 为什么使用 BranchSpec？
- **封装**：将路由函数、路径映射、输入模式封装在一起
- **延迟执行**：在编译时才真正创建路由逻辑
- **类型推断**：支持自动推断输入模式

### 2. 为什么支持多种 path_map 形式？
- **dict**：最灵活，支持任意键到节点的映射
- **list**：简化场景，路由函数直接返回节点名称
- **None**：最简洁，依赖类型提示

### 3. 为什么需要 infer_schema？
- **类型安全**：推断路由函数的输入类型
- **验证**：在编译时验证状态类型匹配
- **优化**：可以根据输入类型优化执行

### 4. 为什么条件边可以有多个？
- **复杂路由**：同一个节点可能需要多个不同的路由逻辑
- **模块化**：不同的路由逻辑可以独立定义和测试
- **命名空间**：通过名称区分不同的条件边

## 与 add_edge 的对比

| 特性 | add_edge | add_conditional_edges |
|------|----------|----------------------|
| 路由类型 | 固定路由 | 动态路由 |
| 目标节点 | 单个固定节点 | 多个动态节点 |
| 决策逻辑 | 无（直接连接） | 路由函数 |
| 存储位置 | `self.edges` | `self.branches` |
| 多起始节点 | 支持（等待所有完成） | 不支持 |
| 路径映射 | 无 | 支持 dict/list |
| 类型推断 | 无 | 支持输入模式推断 |

## 相关源码引用

- `langgraph.graph._branch.BranchSpec` - 分支规范类
- `langgraph._internal._runnable.coerce_to_runnable` - 转换为 Runnable
- `StateGraph.branches: defaultdict[str, dict[str, BranchSpec]]` - 条件边存储
- `StateGraph._add_schema` - 注册输入模式
