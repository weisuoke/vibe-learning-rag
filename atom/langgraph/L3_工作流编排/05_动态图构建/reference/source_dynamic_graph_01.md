---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/types.py
  - libs/langgraph/langgraph/graph/_branch.py
  - libs/langgraph/langgraph/graph/_node.py
  - libs/langgraph/langgraph/pregel/_read.py
  - libs/langgraph/langgraph/pregel/_write.py
  - libs/langgraph/langgraph/pregel/_algo.py
  - libs/langgraph/langgraph/pregel/main.py
  - libs/langgraph/langgraph/runtime.py
analyzed_at: 2026-02-28
knowledge_point: 05_动态图构建
---

# 源码分析：LangGraph 动态图构建核心机制

## 分析的文件

### 1. graph/state.py - StateGraph 构建器
- `StateGraph` 类：主要的图构建器
  - `add_node(node, action, defer, metadata, input_schema, retry_policy, cache_policy, destinations)` - 动态添加节点
  - `add_edge(start_key, end_key)` - 创建有向边
  - `add_conditional_edges(source, path, path_map)` - 添加条件路由
  - `add_sequence(nodes)` - 按顺序添加节点序列
  - `set_entry_point(key)` / `set_conditional_entry_point(path, path_map)` - 设置入口
  - `set_finish_point(key)` - 标记终点
  - `compile()` - 编译为 CompiledStateGraph

- `CompiledStateGraph` 类：编译后的图（继承 Pregel）
  - `attach_node(key, node)` - 附加节点到编译图
  - `attach_edge(starts, end)` - 附加边
  - `attach_branch(start, name, branch, with_reader)` - 附加条件分支

### 2. types.py - Send 和 Command 核心类型

#### Send 类（行 289-362）
```python
class Send:
    node: str  # 目标节点名
    arg: Any   # 传递的状态/消息
```
- 用于条件边中动态调用节点
- 支持 map-reduce 工作流的并行节点调用
- 允许向每个节点调用发送不同状态
- 示例：`[Send("generate_joke", {"subject": s}) for s in state["subjects"]]`

#### Command 类（行 367-418）
```python
@dataclass
class Command(Generic[N], ToolOutputMixin):
    graph: str | None = None           # 目标图（None=当前, Command.PARENT=父图）
    update: Any | None = None          # 状态更新
    resume: dict[str, Any] | Any | None = None  # 中断恢复值
    goto: Send | Sequence[Send | N] | N = ()    # 下一个节点
```
- 支持动态图导航
- 支持父图通信
- 在单个命令中同时更新状态和路由
- 支持中断恢复

### 3. graph/_branch.py - 分支规范
- `BranchSpec` 类：条件分支规范
  - `path` - 决定下一个节点的 Runnable
  - `ends` - 路径结果到节点名的映射
  - `from_path()` - 工厂方法

### 4. graph/_node.py - 节点定义
- `StateNode` 类型别名：支持多种节点签名
  - `_Node` - 基础：`(state) -> Any`
  - `_NodeWithConfig` - 带配置：`(state, config) -> Any`
  - `_NodeWithWriter` - 带流写入：`(state, *, writer) -> Any`
  - `_NodeWithStore` - 带存储：`(state, *, store) -> Any`
  - `_NodeWithRuntime` - 带运行时：`(state, *, runtime) -> Any`

- `StateNodeSpec` 类：节点规范
  - `runnable`, `metadata`, `input_schema`, `retry_policy`, `cache_policy`
  - `ends` - 可能的目标节点（用于可视化）
  - `defer` - 是否延迟执行

### 5. pregel/_algo.py - 执行算法
- `prepare_next_tasks()` - 准备下一步任务
  - 处理 PUSH 任务（来自 TASKS 通道的 Send 对象）
  - 处理 PULL 任务（由边触发的节点）
  - 使用 trigger_to_nodes 映射优化节点选择
- `apply_writes()` - 应用待处理写入到通道
- BSP（批量同步并行）执行模型：计划 → 执行 → 更新

### 6. pregel/_write.py - 通道写入与任务路由
- `ChannelWrite` 类：写入通道
  - 支持 Send 对象进行动态路由
  - PASSTHROUGH 用于输入透传
  - SKIP_WRITE 用于条件跳过

### 7. runtime.py - 运行时上下文
- `Runtime[ContextT]` 类：注入到节点的运行时上下文
  - `context` - 用户定义上下文
  - `store` - 持久化存储
  - `stream_writer` - 自定义流输出

## 关键发现

### 动态图构建的三大核心机制

1. **Send 机制**：运行时动态创建并行节点调用，数量在编译时未知
2. **Command 机制**：节点内部同时更新状态和控制路由，支持跨图导航
3. **条件边（Conditional Edges）**：基于状态的运行时路由决策

### 架构特点

1. **两阶段架构**：构建（StateGraph）→ 编译（CompiledStateGraph → Pregel）
2. **通道通信**：节点通过类型化通道通信，非直接调用
3. **触发激活**：节点由通道更新触发，非显式调用
4. **BSP 执行模型**：三阶段步骤（计划 → 执行 → 更新）
