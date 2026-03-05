---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/graph/_branch.py
  - libs/langgraph/langgraph/pregel/_algo.py
  - libs/langgraph/langgraph/pregel/_loop.py
analyzed_at: 2026-03-01
knowledge_point: 07_工作流模式
---

# 源码分析：LangGraph 工作流模式核心实现

## 分析的文件

- `libs/langgraph/langgraph/graph/state.py` - StateGraph 构建器，支持顺序/并行/条件边
- `libs/langgraph/langgraph/graph/_branch.py` - 条件分支实现（BranchSpec）
- `libs/langgraph/langgraph/pregel/_algo.py` - Pregel 算法核心，任务调度与并行执行
- `libs/langgraph/langgraph/pregel/_loop.py` - 执行循环，superstep 迭代

## 关键发现

### 1. 工作流模式的底层实现

LangGraph 基于 Pregel 算法实现所有工作流模式：
- **顺序执行**：通过 `add_edge(A, B)` 建立单链
- **并行执行**：通过 `add_edge(A, B)` + `add_edge(A, C)` 实现 fan-out，同一 superstep 内并行
- **条件分支**：通过 `add_conditional_edges(source, path_fn, path_map)` 实现动态路由
- **Map-Reduce**：通过 `Send(node_name, arg)` 实现动态并行

### 2. StateGraph 关键方法

```python
class StateGraph:
    def add_node(name, action)          # 添加节点
    def add_edge(start, end)            # 顺序边
    def add_conditional_edges(source, path, path_map)  # 条件边
    def add_sequence(*nodes)            # 快捷顺序链
    def compile()                       # 编译为可执行图
```

### 3. 并行执行机制

- 同一 superstep 内的多个节点自动并行执行
- `NamedBarrierValue` 通道实现 fan-in 同步
- `operator.add` reducer 实现结果聚合
- `defer=True` 支持延迟节点（等待所有并行分支完成）

### 4. Send API（Map-Reduce 模式）

```python
# 动态创建并行任务
def continue_to_jokes(state):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```

- Send 创建 PUSH 类型任务
- 通过 Topic 通道路由
- 支持动态数量的并行任务

### 5. 条件分支实现

```python
class BranchSpec(NamedTuple):
    path: Runnable          # 路由函数
    ends: dict | None       # 目标映射
    input_schema: type | None
```

- 路由函数返回目标节点名称
- 支持返回列表（多目标并行）
- 支持 END 作为终止目标
