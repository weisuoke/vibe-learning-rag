---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/channels/base.py
  - libs/langgraph/langgraph/_internal/_constants.py
  - libs/langgraph/langgraph/pregel/main.py
analyzed_at: 2026-02-27
knowledge_point: 09_多状态管理
---

# 源码分析：LangGraph 多状态管理架构

## 分析的文件

- `libs/langgraph/langgraph/graph/state.py` - StateGraph 核心实现，四种 Schema 参数
- `libs/langgraph/langgraph/channels/base.py` - Channel 基类，状态字段的底层抽象
- `libs/langgraph/langgraph/_internal/_constants.py` - 命名空间常量定义
- `libs/langgraph/langgraph/pregel/main.py` - 子图执行与命名空间组装

## 关键发现

### 1. StateGraph 四种 Schema 参数

StateGraph 构造函数接受四个 Schema 参数：

```python
def __init__(
    self,
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
) -> None:
```

默认行为：
- `input_schema` 默认等于 `state_schema`
- `output_schema` 默认等于 `state_schema`
- `context_schema` 默认为 None

### 2. Schema 到 Channel 的映射

`_get_channels()` 函数将 Schema 的类型注解转换为 Channel：

```python
def _get_channels(schema: type[dict]) -> tuple[dict[str, BaseChannel], dict[str, ManagedValueSpec], dict[str, Any]]:
    type_hints = get_type_hints(schema, include_extras=True)
    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in type_hints.items()
        if name != "__slots__"
    }
    return (
        {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
        type_hints,
    )
```

### 3. 多 Schema 合并机制

`_add_schema()` 方法将多个 Schema 的 Channel 合并到统一的 Channel 池：

- `self.schemas` 字典：每个 Schema 类型映射到其字段-Channel 映射
- `self.channels` 字典：跨所有 Schema 的统一 Channel 注册表
- 同名字段必须兼容（LastValue 与任何 Channel 兼容）
- 不同 Reducer 的同名字段会抛出错误

### 4. Input/Output Schema 在编译时的作用

- START 节点只写入 input_schema 的字段
- 普通节点可以写入所有 Channel
- 图的返回值只包含 output_schema 的字段
- 节点可以有自己的 input_schema（通过函数参数类型推断）

### 5. 命名空间系统

```python
NS_SEP = "|"   # 分隔层级：graph|subgraph|subsubgraph
NS_END = ":"   # 分隔命名空间和 task_id
```

命名空间格式：`{node_name}:{task_id}|{child_node_name}:{child_task_id}`

示例：
- `""` - 根图
- `"agent:abc123"` - 一级子图
- `"agent:abc123|researcher:def456"` - 二级子图

### 6. 子图状态完全隔离

- 每个子图有自己的 StateGraph 和 Channel 集合
- 父子图通信路径：
  1. 调用子图时传入的输入
  2. 子图返回的输出
  3. `Command(graph=Command.PARENT)` 显式更新父图状态
  4. 通过 config keys 共享资源（checkpointer, store, cache, stream）

### 7. context_schema 与 Runtime

context_schema 不创建 Channel，而是定义不可变的运行时上下文：

```python
class Context(TypedDict):
    r: float

graph = StateGraph(state_schema=State, context_schema=Context)

def node(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)
    return {"x": state["x"] * r}

compiled.invoke({"x": 0.5}, context={"r": 3.0})
```

### 8. 节点级 Input Schema

每个节点可以有自己的 input_schema，控制它接收的状态子集：

```python
def add_node(self, node, action=None, *, input_schema=None, ...):
    if input_schema is not None:
        self.nodes[node] = StateNodeSpec(action, metadata, input_schema=input_schema, ...)
    elif inferred_input_schema is not None:
        # 从函数第一个参数的类型提示推断
        self.nodes[node] = StateNodeSpec(action, metadata, input_schema=inferred_input_schema, ...)
```

自动推断机制：如果节点函数的第一个参数有类型注解且是带 type hints 的类，LangGraph 自动将其作为节点的 input_schema。
