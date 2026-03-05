---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/_algo.py
  - libs/langgraph/langgraph/pregel/_loop.py
  - libs/langgraph/langgraph/pregel/main.py
  - libs/langgraph/langgraph/graph/state.py
analyzed_at: 2026-02-28
knowledge_point: 04_人机循环（Human-in-the-loop）
---

# 源码分析：中断检测与执行循环

## 分析的文件

- `libs/langgraph/langgraph/pregel/_algo.py` - should_interrupt() 中断检测算法
- `libs/langgraph/langgraph/pregel/_loop.py` - PregelLoop 中断循环处理
- `libs/langgraph/langgraph/pregel/main.py` - Pregel 主类中断配置
- `libs/langgraph/langgraph/graph/state.py` - StateGraph.compile() 中断参数

## 关键发现

### 1. should_interrupt() 算法 (_algo.py)

```python
def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: All | Sequence[str],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
    """检查图是否应该中断"""
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})

    # 检查自上次中断以来是否有通道更新
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )

    # 如果有更新且触发的节点在中断列表中
    return (
        [
            task for task in tasks
            if (
                (not task.config or TAG_HIDDEN not in task.config.get("tags", EMPTY_SEQ))
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        ]
        if any_updates_since_prev_interrupt
        else []
    )
```

关键逻辑：
- 通过 channel_versions 跟踪状态变化
- 使用 versions_seen[INTERRUPT] 避免重复中断
- 支持 "*" 通配符（所有节点）和具体节点名列表
- 隐藏节点（TAG_HIDDEN）在通配符模式下被排除

### 2. PregelLoop 中断处理 (_loop.py)

```python
class PregelLoop:
    interrupt_after: All | Sequence[str]
    interrupt_before: All | Sequence[str]

    status: Literal[
        "input",
        "pending",
        "done",
        "interrupt_before",
        "interrupt_after",
        "out_of_steps",
    ]
```

中断前检查：
```python
if self.interrupt_before and should_interrupt(
    self.checkpoint, self.interrupt_before, self.tasks.values()
):
    self.status = "interrupt_before"
    raise GraphInterrupt()
```

中断后检查：
```python
if self.interrupt_after and should_interrupt(
    self.checkpoint, self.interrupt_after, self.tasks.values()
):
    self.status = "interrupt_after"
    raise GraphInterrupt()
```

### 3. StateGraph.compile() 中断参数 (state.py)

```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
```

### 4. Pregel 主类 (main.py)

```python
class Pregel:
    interrupt_after_nodes: All | Sequence[str]
    interrupt_before_nodes: All | Sequence[str]
```

### 5. PregelProtocol (protocol.py)

stream/invoke 方法也支持运行时中断配置：
```python
def stream(
    self,
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    ...
) -> Iterator[dict[str, Any] | Any]: ...
```

## 中断机制对比

| 特性 | interrupt() 函数 | interrupt_before/after |
|------|-----------------|----------------------|
| 类型 | 动态中断 | 静态断点 |
| 位置 | 节点内部任意位置 | 节点执行前/后 |
| 灵活性 | 高（支持条件逻辑） | 低（固定节点列表） |
| 数据传递 | 可传递任意JSON值 | 无数据传递 |
| 推荐度 | 官方推荐 | 遗留方式 |
| 配置方式 | 代码内调用 | compile()参数 |

## 执行流程图

```
graph.invoke(input, config)
    ↓
PregelLoop 初始化
    ↓
┌─→ 检查 interrupt_before
│   ├─ 匹配 → status="interrupt_before", 抛出 GraphInterrupt
│   └─ 不匹配 → 继续
│       ↓
│   执行节点
│   ├─ 节点内调用 interrupt() → 抛出 GraphInterrupt
│   └─ 正常完成
│       ↓
│   检查 interrupt_after
│   ├─ 匹配 → status="interrupt_after", 抛出 GraphInterrupt
│   └─ 不匹配 → 继续
│       ↓
│   检查是否有下一步
│   ├─ 有 → 回到循环开头
│   └─ 无 → status="done"
└───────┘
```
