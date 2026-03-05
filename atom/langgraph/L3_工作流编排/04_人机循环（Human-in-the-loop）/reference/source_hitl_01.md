---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/types.py
  - libs/langgraph/langgraph/errors.py
  - libs/prebuilt/langgraph/prebuilt/interrupt.py
analyzed_at: 2026-02-28
knowledge_point: 04_人机循环（Human-in-the-loop）
---

# 源码分析：Human-in-the-loop 核心类型与函数

## 分析的文件

- `libs/langgraph/langgraph/types.py` - interrupt()函数、Interrupt类、Command类
- `libs/langgraph/langgraph/errors.py` - GraphInterrupt异常、GraphBubbleUp基类
- `libs/prebuilt/langgraph/prebuilt/interrupt.py` - HumanInterruptConfig、HumanInterrupt、HumanResponse

## 关键发现

### 1. interrupt() 函数 (types.py:420)

核心HITL机制，在节点内部调用以暂停图执行：

```python
def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.

    - 第一次调用时抛出 GraphInterrupt 异常，暂停执行
    - value 会发送给客户端（必须JSON可序列化）
    - 恢复时使用 Command(resume=value)
    - 图从节点开头重新执行（re-executing all logic）
    - 多个interrupt调用按顺序匹配resume值
    - 必须启用checkpointer
    """
```

关键行为：
- 节点重新执行：恢复时从节点开头重新运行
- 顺序匹配：多个interrupt按顺序与resume值配对
- 任务作用域：resume值列表限定在执行该节点的特定任务内

### 2. Interrupt 数据类 (types.py:161)

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any      # 中断关联的值
    id: str         # 中断ID，用于直接恢复
```

- v0.6.0 移除了 ns、when、resumable、interrupt_id 属性
- id 通过 xxh3_128_hexdigest 生成

### 3. Command 类 (types.py:368)

```python
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None      # 目标图（None=当前图，PARENT=父图）
    update: Any | None = None     # 状态更新
    resume: dict[str, Any] | Any | None = None  # 恢复值
    goto: Send | Sequence[Send | N] | N = ()     # 导航目标
```

resume 支持两种形式：
- 单个值：恢复下一个中断
- 字典 {interrupt_id: value}：按ID恢复特定中断

### 4. GraphInterrupt 异常 (errors.py:84)

```python
class GraphBubbleUp(Exception):
    pass

class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""
    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```

- 继承自 GraphBubbleUp（冒泡异常）
- 子图中断时抛出，被根图抑制
- 不会直接暴露给用户

### 5. NodeInterrupt (已废弃)

```python
@deprecated("NodeInterrupt is deprecated. Please use `interrupt` instead.")
class NodeInterrupt(GraphInterrupt):
    """Raised by a node to interrupt execution."""
```

### 6. HumanInterruptConfig (prebuilt/interrupt.py)

```python
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool      # 可跳过当前步骤
    allow_respond: bool     # 可提供文本反馈
    allow_edit: bool        # 可编辑内容/状态
    allow_accept: bool      # 可批准当前状态
```

注意：已标记为废弃，迁移到 `langchain.agents.interrupt`

### 7. HumanInterrupt / HumanResponse

```python
class HumanInterrupt(TypedDict):
    action_request: ActionRequest   # 请求的动作
    config: HumanInterruptConfig    # 允许的操作配置
    description: str | None         # 详细描述

class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

### 8. StateSnapshot (types.py:268)

```python
class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any
    next: tuple[str, ...]
    config: RunnableConfig
    metadata: CheckpointMetadata | None
    created_at: str | None
    parent_config: RunnableConfig | None
    tasks: tuple[PregelTask, ...]
    interrupts: tuple[Interrupt, ...]  # 待解决的中断
```

## 架构总结

```
用户调用 graph.invoke/stream
    ↓
节点执行 → 调用 interrupt(value)
    ↓
抛出 GraphInterrupt 异常
    ↓
Checkpointer 保存状态
    ↓
返回 __interrupt__ 给客户端
    ↓
客户端获取中断信息
    ↓
用户决策
    ↓
graph.invoke(Command(resume=value), config)
    ↓
从节点开头重新执行，interrupt() 返回 resume 值
    ↓
继续后续节点
```
