---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/_draw.py
  - libs/langgraph/langgraph/pregel/main.py
analyzed_at: 2026-02-25
knowledge_point: 05_图可视化
---

# 源码分析:LangGraph 图可视化核心实现

## 分析的文件

- `libs/langgraph/langgraph/pregel/_draw.py` - 核心图绘制逻辑
- `libs/langgraph/langgraph/pregel/main.py` - get_graph 方法实现

## 关键发现

### 1. get_graph() 方法

**位置**: `libs/langgraph/langgraph/pregel/main.py`

**核心功能**:
- 返回一个可绘制的图表示(Graph 对象)
- 支持 xray 参数用于递归展示子图
- 调用 draw_graph() 函数生成图结构

**关键代码**:
```python
def get_graph(
    self, config: RunnableConfig | None = None, *, xray: int | bool = False
) -> Graph:
    """Return a drawable representation of the computation graph."""
    # gather subgraphs
    if xray:
        subgraphs = {
            k: v.get_graph(
                config,
                xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
            )
            for k, v in self.get_subgraphs()
        }
    else:
        subgraphs = {}

    return draw_graph(
        merge_configs(self.config, config),
        nodes=self.nodes,
        specs=self.channels,
        input_channels=self.input_channels,
        interrupt_after_nodes=self.interrupt_after_nodes,
        interrupt_before_nodes=self.interrupt_before_nodes,
        trigger_to_nodes=self.trigger_to_nodes,
        checkpointer=self.checkpointer,
        subgraphs=subgraphs,
    )
```

**特性**:
- xray 参数:控制子图展开深度
- 支持递归获取子图
- 传递配置信息到 draw_graph

### 2. draw_graph() 函数

**位置**: `libs/langgraph/langgraph/pregel/_draw.py`

**核心功能**:
- 模拟图的执行流程来发现所有边
- 构建完整的图结构(节点 + 边)
- 支持条件边、普通边、中断点等特性

**关键数据结构**:
```python
class Edge(NamedTuple):
    source: str
    target: str
    conditional: bool
    data: str | None

class TriggerEdge(NamedTuple):
    source: str
    conditional: bool
    data: str | None
```

**执行流程**:
1. 初始化空图和通道状态
2. 模拟执行图的 Pregel 循环(最多 250 步)
3. 收集所有任务的写操作
4. 根据写操作推断边的连接关系
5. 处理条件边和静态声明的写操作
6. 添加 START 和 END 节点
7. 替换子图(如果有)

**关键代码片段**:
```python
# 运行 pregel 循环发现边
for step in range(step, limit):
    if not tasks:
        break
    conditionals: dict[tuple[str, str, Any], str | None] = {}
    # 运行任务写入器
    for task in tasks.values():
        for w in task.writers:
            # 应用常规写入
            if isinstance(w, ChannelWrite):
                empty_input = (
                    cast(BaseChannel, specs["__root__"]).ValueType()
                    if "__root__" in specs
                    else None
                )
                w.invoke(empty_input, task.config)
            # 应用条件写入(仅用于静态分析,只执行一次)
            if w not in static_seen:
                static_seen.add(w)
                if writes := ChannelWrite.get_static_writes(w):
                    # END 写入不被写入,而是直接成为边
                    for t in writes:
                        if t[0] == END:
                            edges.add(Edge(task.name, t[0], True, t[2]))
```

### 3. Jupyter 集成

**位置**: `libs/langgraph/langgraph/pregel/main.py`

**功能**: 在 Jupyter 中自动显示图表

```python
def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
    """Mime bundle used by Jupyter to display the graph"""
    return {
        "text/plain": repr(self),
        "image/png": self.get_graph().draw_mermaid_png(),
    }
```

**特性**:
- 返回 MIME bundle 用于 Jupyter 显示
- 自动调用 draw_mermaid_png() 生成 PNG 图像
- 提供文本表示作为后备

### 4. Graph 对象

**来源**: `langchain_core.runnables.graph`

**方法**:
- `draw_mermaid()` - 生成 Mermaid 文本
- `draw_mermaid_png()` - 生成 Mermaid PNG 图像
- `add_node()` - 添加节点
- `add_edge()` - 添加边

**使用示例**(从测试文件):
```python
# 基础使用
graph.get_graph().draw_mermaid(with_styles=False)

# 带样式
graph.get_graph().draw_mermaid()
```

### 5. 支持的特性

**节点元数据**:
- `defer` - 延迟节点标记
- `__interrupt` - 中断点标记("before", "after", "before,after")

**边类型**:
- 普通边(regular edges)
- 条件边(conditional edges)
- 隐式边(implicit edges)

**子图支持**:
- 递归展开子图
- 子图节点替换
- 边的重新连接

## 关键技术点

### 1. 静态分析 vs 动态执行

draw_graph() 使用"模拟执行"的方式来发现图的结构:
- 不实际执行节点函数
- 只执行写入器(writers)来发现边
- 使用空的 checkpoint 和 channels

### 2. 边的发现机制

通过以下方式发现边:
1. 任务的 triggers - 哪些通道触发了这个任务
2. 任务的 writes - 这个任务写入了哪些通道
3. trigger_to_nodes - 通道到节点的映射关系

### 3. 条件边的处理

条件边通过 `ChannelWrite.get_static_writes()` 静态分析:
- 提取所有可能的写入目标
- 标记为条件边(conditional=True)
- 保留边的标签(data)

### 4. 子图的集成

子图通过以下方式集成到主图:
1. 递归调用 get_graph() 获取子图
2. 移除子图的 START 和 END 节点
3. 用子图的内部结构替换子图节点
4. 重新连接边到子图的第一个和最后一个节点

## 依赖库识别

从源码分析中识别的关键依赖:

1. **langchain_core.runnables.graph** - Graph 和 Node 类
2. **langchain_core.runnables.config** - RunnableConfig
3. **langgraph.checkpoint.base** - BaseCheckpointSaver
4. **langgraph.channels.base** - BaseChannel
5. **langgraph.constants** - START, END 常量

## 调试辅助功能

### 1. xray 参数

用于递归展示子图结构:
- `xray=True` - 展开所有子图
- `xray=N` - 展开 N 层子图
- `xray=False` - 不展开子图(默认)

### 2. 中断点可视化

通过节点元数据显示中断点:
- `__interrupt="before"` - 在节点前中断
- `__interrupt="after"` - 在节点后中断
- `__interrupt="before,after"` - 在节点前后都中断

### 3. 延迟节点标记

通过 `defer=True` 元数据标记延迟执行的节点

## 实际使用模式

从测试文件中提取的使用模式:

```python
# 1. 基础可视化
app = StateGraph(State)
# ... 添加节点和边 ...
app = app.compile()
mermaid_text = app.get_graph().draw_mermaid()

# 2. 无样式可视化(用于快照测试)
mermaid_text = app.get_graph().draw_mermaid(with_styles=False)

# 3. Jupyter 自动显示
# 在 Jupyter 中直接显示 app 对象会自动调用 _repr_mimebundle_
app  # 自动显示图表

# 4. 子图可视化
mermaid_text = app.get_graph(xray=True).draw_mermaid()
```

## 性能考虑

1. **limit 参数**: draw_graph() 默认最多模拟 250 步
2. **静态分析**: 条件写入只分析一次(通过 static_seen 集合)
3. **子图缓存**: 子图只在需要时(xray=True)才获取

## 总结

LangGraph 的图可视化通过以下方式实现:
1. 模拟执行图的 Pregel 循环来发现边
2. 构建 Graph 对象(来自 langchain_core)
3. 使用 Mermaid 格式生成可视化
4. 支持 Jupyter 自动显示
5. 提供调试辅助功能(xray, 中断点, 延迟节点)
