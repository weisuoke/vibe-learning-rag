# Context7 查询结果：LangGraph 官方文档

> 库：`/websites/langchain_oss_python_langgraph`

## 查询主题：State Schema 与类型化状态

### 1. Annotated 与 Reducer

**官方文档说明**：

LangGraph 使用 `Annotated` 类型注解来定义状态字段的 reducer 函数。

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
```

**Reducer 执行时机**：
- 节点返回部分状态更新时
- 多个并行节点的输出需要聚合时
- 手动调用 `update_state()` 时

**内置 Reducer**：
- `operator.add` - 列表追加、字符串拼接、数值相加
- `operator.or_` - 字典合并
- `operator.and_` - 集合交集
- `operator.sub` - 集合差集

**自定义 Reducer**：

```python
def merge_messages(existing: list, new: list) -> list:
    """自定义消息合并逻辑"""
    # 去重合并
    seen = {msg['id'] for msg in existing}
    return existing + [msg for msg in new if msg['id'] not in seen]

class State(TypedDict):
    messages: Annotated[list, merge_messages]
```

### 2. TypedDict 状态定义

**官方推荐方式**：

```python
from typing import TypedDict, Annotated
from typing_extensions import Required, NotRequired

class State(TypedDict):
    # 必需字段
    user_id: Required[str]

    # 可选字段
    count: NotRequired[int]

    # 带 reducer 的字段
    messages: Annotated[list, operator.add]
```

**优势**：
- 轻量级，无需额外依赖
- 类型检查支持
- 与 Python 类型系统原生集成
- 性能开销小

### 3. 多 Schema 架构

**官方文档示例**：

```python
from typing import TypedDict

# 整体状态
class OverallState(TypedDict):
    messages: list
    user_id: str
    internal_data: dict

# 输入状态（节点接收）
class InputState(TypedDict):
    messages: list
    user_id: str

# 输出状态（节点返回）
class OutputState(TypedDict):
    messages: list

# 创建图
graph = StateGraph(
    state_schema=OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

**使用场景**：
- **InputState**：限制节点可以访问的字段
- **OutputState**：限制节点可以修改的字段
- **OverallState**：图的完整状态
- **PrivateState**：节点内部状态，不暴露给其他节点

### 4. 状态更新机制

**官方文档说明**：

节点返回的部分状态会与当前状态合并：

```python
def my_node(state: State) -> dict:
    """节点只需返回要更新的字段"""
    return {
        "messages": [{"role": "assistant", "content": "Hello"}]
    }

# StateGraph 自动应用 reducer 合并状态
```

**更新流程**：
1. 节点返回部分状态（dict）
2. StateGraph 遍历返回的字段
3. 对每个字段应用对应的 reducer
4. 更新内部状态

### 5. 并行执行与状态聚合

**官方文档示例**：

```python
from langgraph.graph import StateGraph

# 定义并行节点
graph = StateGraph(State)
graph.add_node("node1", node1_func)
graph.add_node("node2", node2_func)

# Fan-out: 并行执行
graph.add_edge(START, "node1")
graph.add_edge(START, "node2")

# Fan-in: 聚合结果
graph.add_edge("node1", "merge")
graph.add_edge("node2", "merge")
```

**聚合机制**：
- 使用 reducer 自动聚合多个节点的输出
- 确保状态一致性
- 支持自定义聚合逻辑

### 6. Optional 字段处理

**官方文档说明**：

```python
from typing import TypedDict, Optional
from typing_extensions import NotRequired

class State(TypedDict):
    # 方式一：使用 Optional
    count: Optional[int]

    # 方式二：使用 NotRequired（推荐）
    metadata: NotRequired[dict]
```

**NotRequired vs Optional**：
- `NotRequired[T]`：字段可以不存在
- `Optional[T]`：字段必须存在，但值可以是 None

**Reducer 行为**：
- Optional 字段的 reducer 需要处理 None 值
- 建议使用默认值避免 None 检查

```python
def safe_add(a: Optional[list], b: list) -> list:
    """安全的列表追加"""
    if a is None:
        return b
    return a + b

class State(TypedDict):
    messages: Annotated[Optional[list], safe_add]
```

### 7. Superstep 概念

**官方文档说明**：

Superstep 是 LangGraph 的执行单元：
- 一个 superstep 包含所有可以并行执行的节点
- Superstep 结束后，状态被聚合
- 下一个 superstep 开始

**示例**：

```
Superstep 1: [node1, node2] 并行执行
  -> 状态聚合
Superstep 2: [node3] 执行
  -> 状态聚合
Superstep 3: [node4, node5] 并行执行
  -> 状态聚合
```

### 8. 状态持久化

**官方文档示例**：

```python
from langgraph.checkpoint import MemorySaver, PostgresSaver

# 内存持久化
memory_saver = MemorySaver()
graph = graph.compile(checkpointer=memory_saver)

# PostgreSQL 持久化
postgres_saver = PostgresSaver.from_conn_string("postgresql://...")
graph = graph.compile(checkpointer=postgres_saver)
```

**Checkpoint 机制**：
- 每个 superstep 后自动创建 checkpoint
- 支持状态恢复和回溯
- 支持多种存储后端

### 9. 类型验证

**官方文档说明**：

LangGraph 在运行时验证状态类型：

```python
class State(TypedDict):
    count: int

def bad_node(state: State) -> dict:
    return {"count": "not an int"}  # 运行时错误

# TypeError: Expected int, got str for field 'count'
```

**验证时机**：
- 节点返回时
- 状态更新时
- Checkpoint 恢复时

### 10. 最佳实践

**官方推荐**：

1. **优先使用 TypedDict**：除非需要验证或复杂默认值
2. **明确定义 reducer**：避免依赖默认行为
3. **分离 schema**：使用不同的 input/output schema 提高灵活性
4. **类型注解完整**：确保所有字段都有类型注解
5. **测试 reducer**：确保 reducer 逻辑正确
6. **避免大状态**：状态过大会影响性能
7. **使用 NotRequired**：而非 Optional，语义更清晰
8. **文档化 schema**：为复杂 schema 添加注释

## 参考链接

- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/
- State Schema 指南：https://langchain-ai.github.io/langgraph/concepts/state/
- Reducer 文档：https://langchain-ai.github.io/langgraph/concepts/reducers/
- 多 Schema 示例：https://langchain-ai.github.io/langgraph/how-tos/state-model/
