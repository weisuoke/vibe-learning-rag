---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/debug.py
  - libs/langgraph/langgraph/types.py
  - libs/langgraph/langgraph/pregel/_loop.py
  - libs/langgraph/langgraph/pregel/protocol.py
  - libs/langgraph/langgraph/pregel/main.py
analyzed_at: 2026-02-27
knowledge_point: 11_状态调试技巧
---

# 源码分析：LangGraph 状态调试核心机制

## 分析的文件

### 1. `libs/langgraph/langgraph/pregel/debug.py` - 调试事件生成核心

**关键数据结构：**

- `TaskPayload(TypedDict)`: 任务事件载荷
  - `id`: 任务ID
  - `name`: 节点名称
  - `input`: 输入数据
  - `triggers`: 触发器列表

- `TaskResultPayload(TypedDict)`: 任务结果载荷
  - `id`, `name`: 任务标识
  - `error`: 错误信息（如有）
  - `interrupts`: 中断列表
  - `result`: 结果字典

- `CheckpointTask(TypedDict)`: 检查点任务
  - `id`, `name`, `error`, `interrupts`
  - `state`: StateSnapshot | RunnableConfig | None

- `CheckpointPayload(TypedDict)`: 检查点事件载荷
  - `config`: 运行配置
  - `metadata`: 检查点元数据
  - `values`: 当前通道值
  - `next`: 下一步要执行的节点
  - `parent_config`: 父快照配置
  - `tasks`: 任务列表

**关键函数：**

- `map_debug_tasks()`: 生成 stream_mode="debug" 的 "task" 事件
  - 过滤 TAG_HIDDEN 标记的任务
  - 产出 TaskPayload

- `map_debug_task_results()`: 生成 "task_result" 事件
  - 包含错误、中断、结果信息
  - 使用 `map_task_result_writes()` 聚合同通道多次写入

- `map_debug_checkpoint()`: 生成 "checkpoint" 事件
  - 包含完整状态快照
  - 处理子图的 checkpoint_ns
  - 使用 `tasks_w_writes()` 应用写入到任务

- `tasks_w_writes()`: 将 pending_writes 应用到任务
  - 处理 RETURN、ERROR、INTERRUPT 通道
  - 聚合任务写入结果

- `get_colored_text()` / `get_bolded_text()`: 终端格式化工具

### 2. `libs/langgraph/langgraph/types.py` - 核心类型定义

**StateSnapshot(NamedTuple):**
- `values`: 当前通道值
- `next`: 下一步要执行的节点
- `config`: 获取快照的配置
- `metadata`: 检查点元数据
- `created_at`: 创建时间戳
- `parent_config`: 父快照配置
- `tasks`: 任务元组
- `interrupts`: 中断元组

**PregelTask(NamedTuple):**
- `id`, `name`, `path`: 任务标识
- `error`: 异常（如有）
- `interrupts`: 中断元组
- `state`: 状态快照或配置
- `result`: 任务结果

**StreamMode 类型:**
- `"values"`: 每步后所有状态值
- `"updates"`: 仅节点名和更新
- `"checkpoints"`: 检查点事件
- `"tasks"`: 任务开始/完成事件
- `"debug"`: "checkpoints" + "tasks" 组合
- `"messages"`: LLM 消息逐 token
- `"custom"`: 自定义数据

### 3. `libs/langgraph/langgraph/pregel/protocol.py` - 协议接口

**PregelProtocol 定义的调试相关方法：**
- `get_state(config, subgraphs=False)` → StateSnapshot
- `aget_state()` → 异步版本
- `get_state_history()` → Iterator[StateSnapshot]
- `aget_state_history()` → 异步版本
- `stream()` → 支持 stream_mode 参数
- `astream()` → 异步版本

### 4. `libs/langgraph/langgraph/pregel/main.py` - 主实现

- `get_state()` (line 1235): 获取当前状态快照
- `get_state_history()` (line 1319): 获取历史快照
- `stream()` (line 2407): 主流式接口，支持 stream_mode

### 5. `libs/langgraph/langgraph/pregel/_loop.py` - 执行循环

- 导入 debug 函数：`map_debug_tasks`, `map_debug_checkpoint`, `map_debug_task_results`
- Line 449-450: 推送任务时发出 "tasks" 事件
- Line 491-495: 发出 "checkpoint" 事件
- Line 885-896: "debug" 模式重映射为 "checkpoints"/"tasks"

## 关键发现

1. **调试模式是 checkpoints + tasks 的组合**：stream_mode="debug" 实际上同时流式输出检查点事件和任务事件
2. **状态快照包含完整执行上下文**：包括当前值、下一步节点、任务详情、错误和中断
3. **子图调试通过 checkpoint_ns 实现**：每个子图有独立的命名空间
4. **终端格式化工具内置**：提供彩色和加粗文本输出
5. **多种流式模式满足不同调试需求**：从粗粒度（values）到细粒度（debug）
