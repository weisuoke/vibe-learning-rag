---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/channels/base.py
  - libs/langgraph/langgraph/pregel/_read.py
  - libs/langgraph/langgraph/pregel/_write.py
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/pregel/main.py
analyzed_at: 2026-02-26
knowledge_point: 02_状态传递与上下文
---

# 源码分析：LangGraph 状态传递机制

## 分析的文件

### 1. `channels/base.py` - Channel 基础抽象
- **BaseChannel** - 所有 Channel 的基类
- 定义了状态存储的核心接口：`get()`, `update()`, `checkpoint()`, `from_checkpoint()`
- 泛型设计：`BaseChannel[Value, Update, Checkpoint]`

### 2. `pregel/_read.py` - 状态读取机制
- **ChannelRead** - 实现从 `CONFIG_KEY_READ` 读取状态的逻辑
- 支持单个或多个 channel 读取
- `fresh` 参数控制是否读取最新值
- `mapper` 参数支持读取后的数据转换

**关键代码片段：**
```python
class ChannelRead(RunnableCallable):
    """Implements the logic for reading state from CONFIG_KEY_READ."""

    channel: str | list[str]
    fresh: bool = False
    mapper: Callable[[Any], Any] | None = None

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)
```

### 3. `pregel/_write.py` - 状态写入机制
- **ChannelWrite** - 实现向 `CONFIG_KEY_SEND` 写入状态的逻辑
- **ChannelWriteEntry** - 单个 channel 写入条目
- **ChannelWriteTupleEntry** - 批量写入条目
- 支持 `PASSTHROUGH` 模式（直接传递输入）
- 支持 `skip_none` 跳过 None 值
- 支持 `mapper` 转换写入值

**关键代码片段：**
```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH
    skip_none: bool = False
    mapper: Callable | None = None

class ChannelWrite(RunnableCallable):
    """Implements the logic for sending writes to CONFIG_KEY_SEND."""

    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(_assemble_writes(writes))
```

### 4. `pregel/_read.py` - PregelNode 状态绑定
- **PregelNode** - 节点容器，管理节点与 Channel 的连接
- `channels` - 节点输入的 channel（单个或多个）
- `triggers` - 触发节点执行的 channel 列表
- `writers` - 节点输出的 writer 列表
- `bound` - 节点的主要逻辑（Runnable）
- `mapper` - 输入转换函数

**关键代码片段：**
```python
class PregelNode:
    """A node in a Pregel graph."""

    channels: str | list[str]
    """The channels that will be passed as input to `bound`."""

    triggers: list[str]
    """If any of these channels is written to, this node will be triggered."""

    mapper: Callable[[Any], Any] | None
    """A function to transform the input before passing it to `bound`."""

    writers: list[Runnable]
    """Writers that will be executed after `bound`."""

    bound: Runnable[Any, Any]
    """The main logic of the node."""
```

### 5. `graph/state.py` - StateGraph 状态管理
- **StateGraph** - 状态图的顶层抽象
- `channels` - 所有 channel 的字典
- `state_schema` - 状态的类型定义
- `context_schema` - 运行时上下文的类型定义
- `input_schema` / `output_schema` - 输入输出的类型定义

**关键代码片段：**
```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    The signature of each node is `State -> Partial<State>`.
    """

    channels: dict[str, BaseChannel]
    state_schema: type[StateT]
    context_schema: type[ContextT] | None
    input_schema: type[InputT]
    output_schema: type[OutputT]
```

### 6. `pregel/main.py` - Pregel 执行引擎
- **Pregel** - 图执行引擎
- 管理 `CONFIG_KEY_READ` 和 `CONFIG_KEY_SEND` 的注入
- 通过 RunnableConfig 传递上下文
- 支持 checkpoint、store、cache 等高级特性

**关键常量：**
```python
CONFIG_KEY_READ = "read"
CONFIG_KEY_SEND = "send"
CONFIG_KEY_RUNTIME = "runtime"
CONFIG_KEY_CHECKPOINTER = "checkpointer"
CONFIG_KEY_CHECKPOINT_NS = "checkpoint_ns"
```

## 关键发现

### 1. 状态传递的核心机制
- **Channel** 是状态存储的基本单元
- **RunnableConfig** 是上下文传递的载体
- 通过 `CONFIG_KEY_READ` 和 `CONFIG_KEY_SEND` 实现读写分离

### 2. 节点与状态的连接
- 节点通过 `channels` 参数声明输入
- 节点通过 `writers` 声明输出
- 节点通过 `triggers` 声明触发条件

### 3. 状态流转路径
```
StateGraph.channels (存储)
    ↓
PregelNode.channels (声明输入)
    ↓
ChannelRead.do_read() (读取)
    ↓
PregelNode.bound (节点逻辑)
    ↓
ChannelWrite.do_write() (写入)
    ↓
StateGraph.channels (更新)
```

### 4. 上下文传递机制
- RunnableConfig 包含 `CONF` 字典
- `CONF[CONFIG_KEY_READ]` - 读取函数
- `CONF[CONFIG_KEY_SEND]` - 写入函数
- `CONF[CONFIG_KEY_RUNTIME]` - 运行时上下文

### 5. 设计模式
- **依赖注入** - 通过 RunnableConfig 注入读写函数
- **策略模式** - Channel 的不同实现（LastValue, BinaryOperator 等）
- **命令模式** - ChannelWrite 封装写入操作
- **观察者模式** - triggers 机制触发节点执行

## 实现细节

### Channel 的生命周期
1. **创建** - StateGraph 初始化时创建 channels
2. **读取** - ChannelRead 通过 CONFIG_KEY_READ 读取
3. **更新** - ChannelWrite 通过 CONFIG_KEY_SEND 写入
4. **检查点** - checkpoint() 序列化当前状态
5. **恢复** - from_checkpoint() 从检查点恢复

### 状态更新流程
1. 节点执行完毕，返回部分状态
2. ChannelWrite 收集所有写入操作
3. 通过 CONFIG_KEY_SEND 发送写入请求
4. Pregel 引擎调用 Channel.update() 更新状态
5. 触发依赖该 channel 的节点

### 上下文保持策略
- **RunnableConfig 传递** - 所有节点共享同一个 config
- **Runtime 对象** - 通过 CONFIG_KEY_RUNTIME 访问
- **Context Schema** - 类型化的上下文定义
- **不可变性** - context 在运行时不可修改

## 与 LangChain 的集成
- LangGraph 基于 LangChain 的 Runnable 接口
- RunnableConfig 来自 langchain_core.runnables.config
- 节点可以是任何 Runnable（包括 LangChain 的 Chain）

## 性能考虑
- **惰性读取** - 只读取节点需要的 channel
- **批量写入** - 收集所有写入后一次性更新
- **fresh 参数** - 控制是否读取最新值（避免缓存）
- **PASSTHROUGH 模式** - 避免不必要的数据复制
