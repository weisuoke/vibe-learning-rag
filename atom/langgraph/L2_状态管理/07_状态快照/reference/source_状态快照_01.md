---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/types.py
  - libs/langgraph/langgraph/pregel/main.py
  - libs/langgraph/langgraph/pregel/_checkpoint.py
  - libs/checkpoint/langgraph/checkpoint/base/__init__.py
  - libs/checkpoint/langgraph/checkpoint/memory/__init__.py
analyzed_at: 2026-02-27
knowledge_point: 07_状态快照
---

# 源码分析：LangGraph 状态快照核心实现

## 分析的文件

- `libs/langgraph/langgraph/types.py` - StateSnapshot、PregelTask、Interrupt 类型定义
- `libs/langgraph/langgraph/pregel/main.py` - get_state()、get_state_history()、_prepare_state_snapshot() 实现
- `libs/langgraph/langgraph/pregel/_checkpoint.py` - Checkpoint 工具函数
- `libs/checkpoint/langgraph/checkpoint/base/__init__.py` - Checkpoint、CheckpointMetadata、CheckpointTuple、BaseCheckpointSaver 定义
- `libs/checkpoint/langgraph/checkpoint/memory/__init__.py` - InMemorySaver 实现

## 关键发现

### 1. StateSnapshot 数据结构 (types.py:268-286)

```python
class StateSnapshot(NamedTuple):
    """Snapshot of the state of the graph at the beginning of a step."""
    values: dict[str, Any] | Any          # 当前 channel 值
    next: tuple[str, ...]                  # 下一步要执行的节点名
    config: RunnableConfig                 # 获取此快照的配置
    metadata: CheckpointMetadata | None    # 快照元数据
    created_at: str | None                 # 创建时间戳
    parent_config: RunnableConfig | None   # 父快照配置
    tasks: tuple[PregelTask, ...]          # 当前步骤的任务
    interrupts: tuple[Interrupt, ...]      # 待处理的中断
```

### 2. Checkpoint 存储格式 (base/__init__.py:61-92)

```python
class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""
    v: int                                    # 版本号（当前为 4）
    id: str                                   # 唯一且单调递增的 ID
    ts: str                                   # ISO 8601 时间戳
    channel_values: dict[str, Any]            # channel 名到值的映射
    channel_versions: ChannelVersions         # channel 名到版本号的映射
    versions_seen: dict[str, ChannelVersions] # 节点 ID 到已见 channel 版本的映射
    updated_channels: list[str] | None        # 本次更新的 channel 列表
```

### 3. CheckpointMetadata (base/__init__.py:31-55)

```python
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]
    # "input": 来自 invoke/stream 的输入
    # "loop": 来自 pregel 循环内部
    # "update": 来自手动状态更新
    # "fork": 从另一个 checkpoint 复制
    step: int       # -1 为首次输入，0+ 为循环步骤
    parents: dict[str, str]  # checkpoint 命名空间到父 ID 的映射
    run_id: str     # 创建此 checkpoint 的运行 ID
```

### 4. CheckpointTuple (base/__init__.py:108-115)

```python
class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None
```

### 5. get_state() 方法 (main.py:1235-1275)

核心流程：
1. 获取 checkpointer（从 config 或 self.checkpointer）
2. 处理子图命名空间（如果有）
3. 调用 `checkpointer.get_tuple(config)` 获取 CheckpointTuple
4. 调用 `_prepare_state_snapshot()` 构建 StateSnapshot

支持通过 `subgraphs=True` 递归获取子图状态。

### 6. get_state_history() 方法 (main.py:1319-1368)

核心流程：
1. 获取 checkpointer
2. 调用 `checkpointer.list(config, before=before, limit=limit, filter=filter)`
3. 对每个 CheckpointTuple 调用 `_prepare_state_snapshot()` 生成 StateSnapshot
4. 以迭代器形式返回（逆时间顺序）

### 7. _prepare_state_snapshot() 方法 (main.py:996-1113)

这是核心方法，从 CheckpointTuple 构建 StateSnapshot：
1. 如果没有保存的数据，返回空 StateSnapshot
2. 迁移 checkpoint 格式（如需要）
3. 从 checkpoint 恢复 channels：`channels_from_checkpoint()`
4. 准备下一步任务：`prepare_next_tasks()`
5. 处理子图状态（递归或仅返回 config）
6. 应用 pending writes（如需要）
7. 组装并返回 StateSnapshot

### 8. Checkpoint 工具函数 (_checkpoint.py)

- `empty_checkpoint()`: 创建空 checkpoint（v=4, 空 channel_values）
- `create_checkpoint()`: 从现有 checkpoint 创建新的（调用 channel.checkpoint() 序列化）
- `channels_from_checkpoint()`: 从 checkpoint 恢复 channels（调用 channel.from_checkpoint()）
- `copy_checkpoint()`: 深拷贝 checkpoint

### 9. InMemorySaver 存储结构 (memory/__init__.py)

```python
# thread ID -> checkpoint NS -> checkpoint ID -> checkpoint 映射
storage: defaultdict[str, dict[str, dict[str, tuple[...]]]]
# (thread ID, checkpoint NS, checkpoint ID) -> (task ID, write idx)
writes: defaultdict[tuple[str, str, str], dict[...]]
# (thread id, checkpoint ns, channel, version) -> blob
blobs: dict[tuple[str, str, str, str | int | float], tuple[str, bytes]]
```

### 10. 数据流

```
get_state(config)
  ↓
checkpointer.get_tuple(config)  → CheckpointTuple
  ↓
_prepare_state_snapshot(config, CheckpointTuple)
  ├─ channels_from_checkpoint()  → 恢复 channel 值
  ├─ prepare_next_tasks()        → 确定下一步任务
  ├─ 处理子图状态
  ├─ apply_pending_writes()      → 应用待写入
  └─ return StateSnapshot
```
