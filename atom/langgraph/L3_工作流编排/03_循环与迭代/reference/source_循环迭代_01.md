---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - langgraph/graph/state.py
  - langgraph/graph/_branch.py
  - langgraph/types.py
  - langgraph/pregel/_loop.py
  - langgraph/pregel/_io.py
  - langgraph/managed/is_last_step.py
  - langgraph/errors.py
  - langgraph/_internal/_scratchpad.py
analyzed_at: 2026-02-28
knowledge_point: 03_循环与迭代
---

# LangGraph 循环与迭代机制 - 源码分析

## 一、循环的构建方式

LangGraph 中循环的本质是：**边（edge）指向图中已经存在的节点，形成有向环**。LangGraph 提供了四种构建循环的机制。

### 1.1 普通边 `add_edge()` 构建循环

源码位置：`state.py` L785-L837

```python
# state.py L785
def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
    # ...
    if isinstance(start_key, str):
        if start_key == END:
            raise ValueError("END cannot be a start node")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        self.edges.add((start_key, end_key))
        return self
```

`add_edge` 本身不阻止循环。它只禁止 `END -> X` 和 `X -> START`，但允许 `node_B -> node_A` 这样的回边。

编译时，`attach_edge()` 将边转化为 channel 写入（`state.py` L1297-L1305）：

```python
# state.py L1297
def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
    if isinstance(starts, str):
        if end != END:
            # 向目标节点的触发 channel 写入，激活该节点
            self.nodes[starts].writers.append(
                ChannelWrite(
                    (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
                )
            )
```

关键机制：每个节点有一个 `branch:to:{node_name}` 的 EphemeralValue channel 作为触发器。当上游节点写入这个 channel 时，下游节点被激活。循环就是让下游节点写入上游节点的触发 channel。

### 1.2 条件边 `add_conditional_edges()` 构建循环

源码位置：`state.py` L839-L887

```python
# state.py L839
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
    | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
    | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
    # 将 path 函数包装为 Runnable
    path = coerce_to_runnable(path, name=None, trace=True)
    name = path.name or "condition"
    # 保存为 BranchSpec
    self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
```

条件边是循环的核心构建方式。`path` 函数在运行时决定下一个节点，可以返回已经执行过的节点名，从而形成循环。

#### BranchSpec 路由解析

源码位置：`_branch.py` L83-L225

```python
# _branch.py L83
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None
```

路由执行流程（`_branch.py` L146-L225）：

```python
# _branch.py L146
def _route(self, input, config, *, reader, writer):
    if reader:
        value = reader(config)  # 读取最新 state
        if isinstance(value, dict) and isinstance(input, dict) and self.input_schema is None:
            value = {**input, **value}
    else:
        value = input
    result = self.path.invoke(value, config)  # 调用路由函数
    return self._finish(writer, input, result, config)

# _branch.py L192
def _finish(self, writer, input, result, config):
    if not isinstance(result, (list, tuple)):
        result = [result]
    if self.ends:
        destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
    else:
        destinations = cast(Sequence[Send | str], result)
    # 验证：不能返回 None 或 START
    if any(dest is None or dest == START for dest in destinations):
        raise ValueError("Branch did not return a valid destination")
    # 验证：不能 Send 到 END
    if any(p.node == END for p in destinations if isinstance(p, Send)):
        raise InvalidUpdateError("Cannot send a packet to the END node")
    entries = writer(destinations, False)
    # ...
```

条件边编译时通过 `attach_branch()` 挂载（`state.py` L1323-L1370）：

```python
# state.py L1323
def attach_branch(self, start, name, branch, *, with_reader=True):
    def get_writes(packets, static=False):
        writes = [
            (ChannelWriteEntry(
                p if p == END else _CHANNEL_BRANCH_TO.format(p), None
            ) if not isinstance(p, Send) else p)
            for p in packets
            if (True if static else p != END)
        ]
        return writes
    # 将 branch runner 添加到源节点的 writers
    self.nodes[start].writers.append(branch.run(get_writes, reader))
```

### 1.3 Command 的 `goto` 构建循环（节点内部自路由）

源码位置：`types.py` L367-L417

```python
# types.py L367
@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

Command 允许节点在返回值中直接指定下一个要执行的节点，无需预先定义条件边。

节点返回 Command 时的处理流程（`state.py` L1492-L1518）：

```python
# state.py L1492
def _control_branch(value: Any) -> Sequence[tuple[str, Any]]:
    # ...
    commands: list[Command] = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)
    rtn: list[tuple[str, Any]] = []
    for command in commands:
        if command.graph == Command.PARENT:
            raise ParentCommand(command)
        goto_targets = (
            [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
        )
        for go in goto_targets:
            if isinstance(go, Send):
                rtn.append((TASKS, go))
            elif isinstance(go, str) and go != END:
                # 写入目标节点的触发 channel
                rtn.append((_CHANNEL_BRANCH_TO.format(go), None))
    return rtn
```

Command 作为外部输入时的处理（`_io.py` L56-L78）：

```python
# _io.py L56
def map_command(cmd: Command) -> Iterator[tuple[str, str, Any]]:
    if cmd.graph == Command.PARENT:
        raise InvalidUpdateError("There is no parent graph")
    if cmd.goto:
        if isinstance(cmd.goto, (tuple, list)):
            sends = cmd.goto
        else:
            sends = [cmd.goto]
        for send in sends:
            if isinstance(send, Send):
                yield (NULL_TASK_ID, TASKS, send)
            elif isinstance(send, str):
                yield (NULL_TASK_ID, f"branch:to:{send}", START)
```

### 1.4 Send 实现动态迭代（Map-Reduce 模式）

源码位置：`types.py` L289-L361

```python
# types.py L289
class Send:
    __slots__ = ("node", "arg")
    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg
```

Send 的核心用途：在条件边或 Command 中，向同一个节点发送多个不同的输入，实现并行 map 操作。

```python
# types.py 文档示例
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder.add_conditional_edges(START, continue_to_jokes)
```

Send 在 `_control_branch` 中被写入 TASKS channel（`state.py` L1493-L1494）：

```python
if isinstance(value, Send):
    return ((TASKS, value),)
```

## 二、执行引擎 PregelLoop 的迭代机制

### 2.1 PregelLoop 核心属性

源码位置：`_loop.py` L140-L203

```python
# _loop.py L140
class PregelLoop:
    step: int       # 当前步数
    stop: int       # 最大步数（step > stop 时停止）
    status: Literal[
        "input",
        "pending",
        "done",
        "interrupt_before",
        "interrupt_after",
        "out_of_steps",
    ]
    tasks: dict[str, PregelExecutableTask]
```

### 2.2 step 和 stop 的初始化

源码位置：`_loop.py` L1119-L1120（SyncPregelLoop.__enter__）

```python
# _loop.py L1119
self.step = self.checkpoint_metadata["step"] + 1
self.stop = self.step + self.config["recursion_limit"] + 1
```

`stop = step + recursion_limit + 1`。默认 `recursion_limit=25`，所以图最多执行 26 个 superstep（包含 input step）。

### 2.3 tick() - 单次迭代

源码位置：`_loop.py` L459-L536

```python
# _loop.py L459
def tick(self) -> bool:
    """Execute a single iteration of the Pregel loop.
    Returns: True if more iterations are needed.
    """
    # 1. 检查是否超出步数限制
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    # 2. 准备下一批任务
    self.tasks = prepare_next_tasks(
        self.checkpoint,
        self.checkpoint_pending_writes,
        self.nodes,
        self.channels,
        self.managed,
        self.config,
        self.step,
        self.stop,
        for_execution=True,
        # ...
    )

    # 3. 没有任务则结束
    if not self.tasks:
        self.status = "done"
        return False

    # 4. 检查 interrupt_before
    if self.interrupt_before and should_interrupt(
        self.checkpoint, self.interrupt_before, self.tasks.values()
    ):
        self.status = "interrupt_before"
        raise GraphInterrupt()

    return True
```

### 2.4 after_tick() - 迭代后处理

源码位置：`_loop.py` L538-L571

```python
# _loop.py L538
def after_tick(self) -> None:
    # 1. 收集所有任务的写入
    writes = [w for t in self.tasks.values() for w in t.writes]

    # 2. 应用写入到 channels（这会触发下一轮的任务）
    self.updated_channels = apply_writes(
        self.checkpoint,
        self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        self.trigger_to_nodes,
    )

    # 3. 清除 pending writes
    self.checkpoint_pending_writes.clear()

    # 4. 保存 checkpoint
    self._put_checkpoint({"source": "loop"})

    # 5. 检查 interrupt_after
    if self.interrupt_after and should_interrupt(
        self.checkpoint, self.interrupt_after, self.tasks.values()
    ):
        self.status = "interrupt_after"
        raise GraphInterrupt()

    # 6. step 在 _put_checkpoint 中递增
```

step 递增发生在 `_put_checkpoint` 中（`_loop.py` L811-L813）：

```python
# _loop.py L811
if not exiting:
    self.step += 1
```

### 2.5 主循环驱动

源码位置：`main.py` L2643-L2674

```python
# main.py L2643
while loop.tick():
    for task in loop.match_cached_writes():
        loop.output_writes(task.id, task.writes, cached=True)
    for _ in runner.tick(
        [t for t in loop.tasks.values() if not t.writes],
        timeout=self.step_timeout,
        get_waiter=get_waiter,
        schedule_task=loop.accept_push,
    ):
        yield from _output(...)
    loop.after_tick()

# 循环结束后检查状态
if loop.status == "out_of_steps":
    msg = create_error_message(
        message=(
            f"Recursion limit of {config['recursion_limit']} reached "
            "without hitting a stop condition. You can increase the "
            "limit by setting the `recursion_limit` config key."
        ),
        error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
    )
    raise GraphRecursionError(msg)
```

完整的迭代流程：
1. `tick()` 检查步数限制，准备任务
2. `runner.tick()` 执行所有任务（并行）
3. `after_tick()` 应用写入、保存 checkpoint、递增 step
4. 回到 1，直到没有任务或超出限制

## 三、递归限制与 IsLastStep 机制

### 3.1 递归限制的计算

递归限制通过 `config["recursion_limit"]` 设置，默认值为 25。

```python
# _loop.py L1120 (SyncPregelLoop.__enter__)
self.stop = self.step + self.config["recursion_limit"] + 1

# _loop.py L1301 (AsyncPregelLoop.__aenter__)
self.stop = self.step + self.config["recursion_limit"] + 1
```

`stop` 的含义：当 `self.step > self.stop` 时停止。由于 `stop = step + limit + 1`，实际允许执行 `limit + 1` 个 superstep。

### 3.2 IsLastStep 和 RemainingSteps

源码位置：`is_last_step.py` L1-L24

```python
# is_last_step.py
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

PregelScratchpad 的定义（`_scratchpad.py` L8-L19）：

```python
# _scratchpad.py L8
@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    call_counter: Callable[[], int]
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    subgraph_counter: Callable[[], int]
```

使用方式：节点可以通过类型注解注入 `IsLastStep` 或 `RemainingSteps`，在最后一步时采取不同策略（如强制结束循环、返回当前最佳结果）。

```python
# 用法示例
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

def my_node(state, is_last_step: IsLastStep):
    if is_last_step:
        return {"result": "forced final answer"}
    # 正常逻辑...
```

### 3.3 递归限制的触发时机

```
step=0: input 处理
step=1: 第一轮 tick
step=2: 第二轮 tick
...
step=N: tick() 检查 step > stop，如果超出则 status="out_of_steps"
```

当 `tick()` 返回 False 且 `status == "out_of_steps"` 时，主循环抛出 `GraphRecursionError`。

## 四、错误处理

### 4.1 GraphRecursionError

源码位置：`errors.py` L45-L65

```python
# errors.py L45
class GraphRecursionError(RecursionError):
    """Raised when the graph has exhausted the maximum number of steps.

    This prevents infinite loops. To increase the maximum number of steps,
    run your graph with a config specifying a higher `recursion_limit`.

    Examples:
        graph = builder.compile()
        graph.invoke(
            {"messages": [("user", "Hello, world!")]},
            {"recursion_limit": 1000},
        )
    """
    pass
```

继承自 Python 内置的 `RecursionError`。

### 4.2 GraphInterrupt

源码位置：`errors.py` L84-L89

```python
# errors.py L84
class GraphInterrupt(GraphBubbleUp):
    """Raised when a subgraph is interrupted, suppressed by the root graph.
    Never raised directly, or surfaced to the user."""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```

`GraphInterrupt` 在 `interrupt_before` 和 `interrupt_after` 检查时抛出，被 `_suppress_interrupt` 捕获（`_loop.py` L815-L875）。对于根图，中断被抑制并转化为输出；对于子图，中断向上冒泡。

### 4.3 ParentCommand

源码位置：`errors.py` L111-L115

```python
# errors.py L111
class ParentCommand(GraphBubbleUp):
    args: tuple[Command]

    def __init__(self, command: Command) -> None:
        super().__init__(command)
```

当节点返回 `Command(graph=Command.PARENT, ...)` 时，在 `_control_branch` 中抛出 `ParentCommand`，将控制权交给父图。

### 4.4 ErrorCode 枚举

```python
# errors.py L29
class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"
```

## 五、循环执行的完整数据流

```
用户输入
  │
  ▼
PregelLoop.__enter__()
  ├── 加载 checkpoint
  ├── 初始化 channels
  ├── step = checkpoint_step + 1
  ├── stop = step + recursion_limit + 1
  └── _first(): 处理输入/恢复
        │
        ▼
  ┌─→ tick()
  │     ├── step > stop? → status="out_of_steps", return False
  │     ├── prepare_next_tasks() → 根据 channel 状态确定要执行的节点
  │     ├── 没有任务? → status="done", return False
  │     ├── interrupt_before 检查
  │     └── return True
  │     │
  │     ▼
  │   runner.tick() → 并行执行所有任务
  │     │
  │     ▼
  │   after_tick()
  │     ├── apply_writes() → 将任务输出写入 channels
  │     │     └── 如果写入了 "branch:to:X" channel → X 节点在下一轮被触发
  │     │     └── 如果写入了 TASKS channel (Send) → 新任务在下一轮被创建
  │     ├── _put_checkpoint() → 保存 checkpoint, step += 1
  │     └── interrupt_after 检查
  │     │
  └─────┘ (回到 tick)

循环结束:
  ├── status=="done" → 正常结束，返回输出
  ├── status=="out_of_steps" → 抛出 GraphRecursionError
  └── GraphInterrupt → 被 _suppress_interrupt 处理
```

## 六、关键设计总结

| 机制 | 源码位置 | 作用 |
|------|----------|------|
| `add_edge(A, B)` | `state.py` L785 | 静态边，A 完成后触发 B |
| `add_conditional_edges(A, fn)` | `state.py` L839 | 动态路由，fn 返回下一个节点名 |
| `Command(goto="B")` | `types.py` L367 | 节点内部自路由，无需预定义边 |
| `Send(node, arg)` | `types.py` L289 | 动态创建并行任务（map-reduce） |
| `branch:to:{node}` channel | `state.py` L90, L1275 | 节点触发机制的底层实现 |
| `PregelLoop.tick()` | `_loop.py` L459 | 单次迭代：检查限制、准备任务 |
| `PregelLoop.after_tick()` | `_loop.py` L538 | 迭代后：应用写入、递增 step |
| `step > stop` 检查 | `_loop.py` L467 | 递归限制的守卫条件 |
| `IsLastStep` | `is_last_step.py` L9 | 节点感知是否为最后一步 |
| `RemainingSteps` | `is_last_step.py` L18 | 节点感知剩余步数 |
| `GraphRecursionError` | `errors.py` L45 | 超出递归限制时的错误 |
| `GraphInterrupt` | `errors.py` L84 | 中断循环（human-in-the-loop） |

### 循环的底层统一模型

无论使用哪种方式构建循环，底层都归结为同一个机制：

1. 节点执行后产生 writes
2. writes 被 `apply_writes()` 应用到 channels
3. `prepare_next_tasks()` 根据 channel 变化确定下一批任务
4. 如果有任务，继续迭代；否则结束

循环的形成条件：某个节点的 writes 触发了图中已经执行过的节点的 trigger channel。递归限制是唯一的安全阀，防止无限循环。
