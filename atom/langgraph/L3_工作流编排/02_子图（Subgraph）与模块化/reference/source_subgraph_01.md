---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/bench/fanout_to_subgraph.py
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/graph/_node.py
  - libs/langgraph/langgraph/types.py
analyzed_at: 2026-02-27
knowledge_point: 02_子图（Subgraph）与模块化
---

# 源码分析：LangGraph 子图（Subgraph）核心实现

## 分析的文件

- `libs/langgraph/bench/fanout_to_subgraph.py` - 子图 fanout 性能基准测试
- `libs/langgraph/langgraph/graph/state.py` - StateGraph 核心实现，包含 add_node、compile、input_schema/output_schema
- `libs/langgraph/langgraph/graph/_node.py` - StateNodeSpec 定义
- `libs/langgraph/langgraph/types.py` - Command、Send 等类型定义

## 关键发现

### 1. 子图定义与添加

子图本质上是一个 **编译后的 StateGraph（CompiledStateGraph）**，通过 `add_node()` 添加到父图中。

```python
# 子图定义
subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
subgraph.add_node("edit", edit)
subgraph.add_node("generate", generate)
subgraphc = subgraph.compile()

# 父图中添加子图
builder = StateGraph(OverallState)
builder.add_node("generate_joke", subgraphc)  # 直接传入编译后的子图
```

### 2. input_schema 和 output_schema

StateGraph 构造函数支持 `input_schema` 和 `output_schema` 参数：
- `input_schema`：定义子图接受的输入结构（默认为 state_schema）
- `output_schema`：定义子图返回的输出结构（默认为 state_schema）
- 这允许子图有不同于内部状态的输入/输出接口

```python
# state.py 中的构造函数
def __init__(
    self,
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
):
    self.input_schema = cast(type[InputT], input_schema or state_schema)
    self.output_schema = cast(type[OutputT], output_schema or state_schema)
```

### 3. Command 类型用于跨图通信

`Command` 类支持 `graph` 参数，用于子图向父图发送命令：

```python
class Command(Generic[N], ToolOutputMixin):
    """
    Args:
        graph: Graph to send the command to. Supported values:
            - None: the current graph
            - Command.PARENT: closest parent graph
        update: Update to apply to the graph's state.
        goto: Node to navigate to.
    """
```

### 4. Send 类型用于动态分发

`Send` 类用于在条件边中动态将不同输入发送到子图节点：

```python
class Send:
    """用于在条件边中动态调用节点并传入自定义状态"""
    node: str
    arg: Any
```

### 5. Checkpointer 继承

子图可以继承父图的 checkpointer：
- `compile(checkpointer=None)` → 从父图继承
- `compile(checkpointer=True)` → 使用独立 checkpointer
- `compile(checkpointer=False)` → 不使用 checkpointer

### 6. StateNodeSpec

节点定义包含 `input_schema` 字段，用于指定节点接收的输入类型：

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    ends: tuple[str, ...] | dict[str, str] | None
    defer: bool = False
```
