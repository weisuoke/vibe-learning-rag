# 图可视化 - 核心概念1: get_graph() 方法

## 概念定位

**核心概念**: get_graph() 方法
**重要程度**: ⭐⭐⭐⭐⭐ (最高)
**使用频率**: 极高
**难度等级**: ⭐ (简单)

## 一句话定义

**get_graph() 是 LangGraph 提供的核心方法,用于获取计算图的可绘制表示(Graph 对象),支持通过 xray 参数递归展开子图结构。**

## 详细解释

### 方法签名

```python
def get_graph(
    self, 
    config: RunnableConfig | None = None, 
    *, 
    xray: int | bool = False
) -> Graph:
    """Return a drawable representation of the computation graph."""
```

**参数说明**:
- `config`: 可选的运行配置,用于合并到图的配置中
- `xray`: 控制子图展开深度
  - `False` (默认): 不展开子图
  - `True`: 递归展开所有子图
  - `int`: 展开指定层数的子图

**返回值**:
- `Graph` 对象: 来自 `langchain_core.runnables.graph`,包含节点和边的完整图结构

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

### 核心功能

#### 1. 获取可绘制的图对象

get_graph() 的主要作用是将 LangGraph 的内部表示转换为可绘制的 Graph 对象:

```python
from langgraph.graph import StateGraph, START, END

# 定义状态
class State(TypedDict):
    messages: list[str]

# 创建图
workflow = StateGraph(State)
workflow.add_node("process", lambda x: {"messages": x["messages"] + ["processed"]})
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

# 编译图
app = workflow.compile()

# 获取可绘制的图对象
graph = app.get_graph()
print(type(graph))  # <class 'langchain_core.runnables.graph.Graph'>
```

**输出**:
```
<class 'langchain_core.runnables.graph.Graph'>
```

[来源: reference/context7_langgraph_01.md - 基础可视化模式]

#### 2. 支持配置合并

get_graph() 可以接受配置参数,用于合并到图的配置中:

```python
from langchain_core.runnables import RunnableConfig

# 创建配置
config = RunnableConfig(
    tags=["production"],
    metadata={"version": "1.0"}
)

# 获取带配置的图对象
graph = app.get_graph(config=config)
```

这在需要根据不同环境或配置生成不同图表时很有用。

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

#### 3. 递归展开子图 (xray 参数)

xray 参数是 get_graph() 的核心特性,用于控制子图的展开:

```python
# 不展开子图 (默认)
graph_simple = app.get_graph(xray=False)

# 展开所有子图
graph_detailed = app.get_graph(xray=True)

# 只展开一层子图
graph_one_level = app.get_graph(xray=1)

# 展开两层子图
graph_two_levels = app.get_graph(xray=2)
```

**xray 参数的实现逻辑**:

```python
def get_graph(self, config=None, *, xray=False):
    # 如果 xray=True,递归获取子图
    if xray:
        subgraphs = {
            k: v.get_graph(
                config,
                # 如果 xray 是布尔值,保持 True
                # 如果 xray 是整数,递减层数
                xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
            )
            for k, v in self.get_subgraphs()
        }
    else:
        subgraphs = {}
    
    return draw_graph(
        # ... 其他参数
        subgraphs=subgraphs,
    )
```

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

### 内部实现原理

#### 1. 调用 draw_graph() 函数

get_graph() 内部调用 `draw_graph()` 函数来生成图结构:

```python
def get_graph(self, config=None, *, xray=False):
    # 1. 收集子图
    if xray:
        subgraphs = {k: v.get_graph(config, xray=...) for k, v in self.get_subgraphs()}
    else:
        subgraphs = {}
    
    # 2. 调用 draw_graph() 生成图对象
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

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

#### 2. draw_graph() 的模拟执行

draw_graph() 通过模拟执行图的 Pregel 循环来发现所有边:

```python
# draw_graph() 的核心逻辑
for step in range(step, limit):  # 最多 250 步
    if not tasks:
        break
    
    # 运行任务写入器
    for task in tasks.values():
        for w in task.writers:
            # 应用常规写入
            if isinstance(w, ChannelWrite):
                w.invoke(empty_input, task.config)
            
            # 应用条件写入(静态分析)
            if w not in static_seen:
                static_seen.add(w)
                if writes := ChannelWrite.get_static_writes(w):
                    for t in writes:
                        if t[0] == END:
                            edges.add(Edge(task.name, t[0], True, t[2]))
```

**关键点**:
- 不实际执行节点函数,只执行写入器
- 使用空的 checkpoint 和 channels
- 限制最多 250 步,避免无限循环

[来源: reference/source_图可视化_01.md - draw_graph() 函数核心逻辑]

### 实际应用场景

#### 场景1: 基础可视化

```python
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# 创建简单的图
workflow = StateGraph(State)
workflow.add_node("node1", lambda x: x)
workflow.add_node("node2", lambda x: x)
workflow.add_edge(START, "node1")
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", END)

app = workflow.compile()

# 获取图对象并可视化
graph = app.get_graph()
display(Image(graph.draw_mermaid_png()))
```

[来源: reference/context7_langgraph_01.md - 基础可视化模式]

#### 场景2: 子图可视化

```python
# 创建包含子图的复杂图
main_workflow = StateGraph(State)
sub_workflow = StateGraph(State)

# 添加子图节点
sub_workflow.add_node("sub_node1", lambda x: x)
sub_workflow.add_node("sub_node2", lambda x: x)
sub_workflow.add_edge(START, "sub_node1")
sub_workflow.add_edge("sub_node1", "sub_node2")
sub_workflow.add_edge("sub_node2", END)

sub_app = sub_workflow.compile()

# 将子图添加到主图
main_workflow.add_node("main_node1", lambda x: x)
main_workflow.add_node("sub_graph", sub_app)
main_workflow.add_node("main_node2", lambda x: x)
main_workflow.add_edge(START, "main_node1")
main_workflow.add_edge("main_node1", "sub_graph")
main_workflow.add_edge("sub_graph", "main_node2")
main_workflow.add_edge("main_node2", END)

main_app = main_workflow.compile()

# 不展开子图
graph_simple = main_app.get_graph(xray=False)
print("简单视图:", graph_simple.draw_mermaid())

# 展开所有子图
graph_detailed = main_app.get_graph(xray=True)
print("详细视图:", graph_detailed.draw_mermaid())
```

[来源: reference/context7_langgraph_01.md - xray 参数使用]

#### 场景3: 调试工作流

```python
# 使用 get_graph() 调试工作流结构
app = workflow.compile()

# 获取图对象
graph = app.get_graph(xray=True)

# 检查节点数量
print(f"节点数量: {len(graph.nodes)}")

# 检查边数量
print(f"边数量: {len(graph.edges)}")

# 打印所有节点名称
print("节点列表:")
for node in graph.nodes:
    print(f"  - {node.id}")

# 打印所有边
print("边列表:")
for edge in graph.edges:
    print(f"  - {edge.source} -> {edge.target}")
```

[来源: reference/context7_langgraph_01.md - 调试模式]

### 与其他方法的关系

#### 1. get_graph() → draw_mermaid()

```python
# 获取图对象
graph = app.get_graph()

# 生成 Mermaid 文本
mermaid_text = graph.draw_mermaid()
print(mermaid_text)
```

#### 2. get_graph() → draw_mermaid_png()

```python
# 获取图对象
graph = app.get_graph()

# 生成 PNG 图像
png_data = graph.draw_mermaid_png()

# 保存到文件
with open("graph.png", "wb") as f:
    f.write(png_data)
```

#### 3. 链式调用

```python
# 常见的链式调用模式
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

[来源: reference/context7_langchain_01.md - Python API]

### 性能考虑

#### 1. xray 参数的性能影响

```python
import time

# 测试不同 xray 参数的性能
start = time.time()
graph_simple = app.get_graph(xray=False)
print(f"xray=False: {time.time() - start:.3f}s")

start = time.time()
graph_detailed = app.get_graph(xray=True)
print(f"xray=True: {time.time() - start:.3f}s")
```

**性能建议**:
- 开发阶段: 使用 `xray=False` 快速查看顶层结构
- 调试阶段: 使用 `xray=True` 深入查看子图
- 生产环境: 避免频繁调用 get_graph()

[来源: reference/context7_langgraph_01.md - 最佳实践]

#### 2. 缓存图对象

```python
# 缓存图对象避免重复计算
class CachedApp:
    def __init__(self, app):
        self.app = app
        self._graph_cache = {}
    
    def get_graph(self, xray=False):
        cache_key = f"xray_{xray}"
        if cache_key not in self._graph_cache:
            self._graph_cache[cache_key] = self.app.get_graph(xray=xray)
        return self._graph_cache[cache_key]

cached_app = CachedApp(app)
graph1 = cached_app.get_graph(xray=True)  # 计算
graph2 = cached_app.get_graph(xray=True)  # 从缓存获取
```

### 常见问题

#### Q1: get_graph() 返回的 Graph 对象是什么?

**A**: Graph 对象来自 `langchain_core.runnables.graph`,包含:
- `nodes`: 节点列表
- `edges`: 边列表
- `draw_mermaid()`: 生成 Mermaid 文本的方法
- `draw_mermaid_png()`: 生成 PNG 图像的方法

[来源: reference/source_图可视化_01.md - Graph 对象]

#### Q2: xray 参数什么时候使用?

**A**:
- `xray=False`: 只查看顶层结构,适合快速了解图的整体架构
- `xray=True`: 递归展开所有子图,适合深入调试和理解复杂工作流
- `xray=N`: 只展开 N 层子图,适合控制可视化的复杂度

#### Q3: get_graph() 会实际执行图吗?

**A**: 不会。get_graph() 只是模拟执行来发现边,不会实际执行节点函数,因此:
- 无副作用
- 不会调用 LLM
- 不会修改状态
- 性能开销较小

[来源: reference/source_图可视化_01.md - 静态分析 vs 动态执行]

#### Q4: 如何在 Jupyter 外使用 get_graph()?

**A**:
```python
# 方法1: 生成 Mermaid 文本
mermaid_text = app.get_graph().draw_mermaid()
with open("graph.mmd", "w") as f:
    f.write(mermaid_text)

# 方法2: 生成 PNG 图像
png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
```

### 最佳实践

#### 1. 开发阶段

```python
# 快速查看图结构
print(app.get_graph().draw_mermaid())
```

#### 2. 调试阶段

```python
# 深入查看子图
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

#### 3. 文档阶段

```python
# 生成高质量图像
png_data = app.get_graph(xray=True).draw_mermaid_png()
with open("docs/workflow.png", "wb") as f:
    f.write(png_data)
```

#### 4. 生产环境

```python
# 避免在生产代码中频繁调用
# 只在初始化或配置变更时调用一次
if os.getenv("ENV") == "development":
    graph = app.get_graph(xray=True)
    display(Image(graph.draw_mermaid_png()))
```

[来源: reference/context7_langgraph_01.md - 最佳实践]

### 总结

get_graph() 是 LangGraph 图可视化的核心入口方法:

1. **核心功能**: 获取可绘制的 Graph 对象
2. **关键特性**: 支持 xray 参数递归展开子图
3. **实现原理**: 通过模拟执行发现所有边
4. **使用场景**: 开发、调试、文档生成
5. **性能考虑**: 避免在生产环境频繁调用

**记住**: get_graph() 是可视化的第一步,后续需要调用 draw_mermaid() 或 draw_mermaid_png() 来生成实际的图表。

---

**版本**: v1.0
**最后更新**: 2026-02-25
**维护者**: Claude Code
**数据来源**: [reference/source_图可视化_01.md, reference/context7_langgraph_01.md, reference/context7_langchain_01.md]
