# 图可视化 - 核心概念4: xray 参数详解

## 概念定位

**核心概念**: xray 参数
**重要程度**: ⭐⭐⭐⭐ (高)
**使用频率**: 中等
**难度等级**: ⭐⭐ (中等)

## 一句话定义

**xray 参数是 get_graph() 方法的关键参数,用于控制子图的递归展开深度,是调试和理解复杂嵌套图结构的核心工具。**

## 详细解释

### 参数签名

```python
def get_graph(
    self, 
    config: RunnableConfig | None = None, 
    *, 
    xray: int | bool = False  # 关键参数
) -> Graph:
    """Return a drawable representation of the computation graph."""
```

**参数类型**:
- `bool`: `False` (默认) 或 `True`
- `int`: 正整数,表示展开层数

**参数含义**:
- `xray=False`: 不展开子图,只显示顶层结构 (默认)
- `xray=True`: 递归展开所有子图
- `xray=N`: 只展开 N 层子图

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

### 核心功能

#### 1. 控制子图展开深度

xray 参数的核心作用是控制子图的展开深度:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 定义状态
class State(TypedDict):
    messages: list[str]

# 创建子图
sub_workflow = StateGraph(State)
sub_workflow.add_node("sub_node1", lambda x: x)
sub_workflow.add_node("sub_node2", lambda x: x)
sub_workflow.add_edge(START, "sub_node1")
sub_workflow.add_edge("sub_node1", "sub_node2")
sub_workflow.add_edge("sub_node2", END)
sub_app = sub_workflow.compile()

# 创建主图
main_workflow = StateGraph(State)
main_workflow.add_node("main_node1", lambda x: x)
main_workflow.add_node("sub_graph", sub_app)  # 嵌入子图
main_workflow.add_node("main_node2", lambda x: x)
main_workflow.add_edge(START, "main_node1")
main_workflow.add_edge("main_node1", "sub_graph")
main_workflow.add_edge("sub_graph", "main_node2")
main_workflow.add_edge("main_node2", END)
main_app = main_workflow.compile()

# 不展开子图 (默认)
graph_simple = main_app.get_graph(xray=False)
print("简单视图节点数:", len(graph_simple.nodes))  # 4 个节点

# 展开所有子图
graph_detailed = main_app.get_graph(xray=True)
print("详细视图节点数:", len(graph_detailed.nodes))  # 6 个节点
```

**输出**:
```
简单视图节点数: 4
详细视图节点数: 6
```

[来源: reference/context7_langgraph_01.md - xray 参数使用]

#### 2. 递归展开机制

xray 参数的实现逻辑:

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

**递归逻辑**:
1. 如果 `xray=False`,不获取子图
2. 如果 `xray=True`,递归获取所有子图 (无限深度)
3. 如果 `xray=N` (整数),递归获取 N 层子图,每层递减 1

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

#### 3. 子图替换机制

当 xray=True 时,draw_graph() 会替换子图节点:

```python
# draw_graph() 的子图替换逻辑
if subgraphs:
    for subgraph_name, subgraph in subgraphs.items():
        # 1. 移除子图的 START 和 END 节点
        subgraph_nodes = [n for n in subgraph.nodes if n.id not in (START, END)]
        
        # 2. 用子图的内部节点替换子图节点
        graph.nodes.remove(subgraph_node)
        graph.nodes.extend(subgraph_nodes)
        
        # 3. 重新连接边
        # 入边: 连接到子图的第一个节点
        # 出边: 从子图的最后一个节点连接出去
```

[来源: reference/source_图可视化_01.md - 子图的集成]

### 使用场景

#### 场景1: 快速查看顶层结构

```python
# 只查看顶层结构,不展开子图
graph_simple = app.get_graph(xray=False)
print(graph_simple.draw_mermaid())
```

**适用情况**:
- 快速了解图的整体架构
- 子图内部细节不重要
- 图表需要保持简洁

[来源: reference/context7_langgraph_01.md - 基础可视化模式]

#### 场景2: 深入调试子图

```python
# 展开所有子图,深入查看内部结构
graph_detailed = app.get_graph(xray=True)

from IPython.display import Image, display
display(Image(graph_detailed.draw_mermaid_png()))
```

**适用情况**:
- 调试子图内部逻辑
- 理解复杂的嵌套结构
- 验证子图的连接关系

[来源: reference/context7_langgraph_01.md - 调试模式]

#### 场景3: 控制展开层数

```python
# 只展开一层子图
graph_one_level = app.get_graph(xray=1)

# 展开两层子图
graph_two_levels = app.get_graph(xray=2)

# 对比不同层数的节点数量
print(f"0 层: {len(app.get_graph(xray=0).nodes)} 节点")
print(f"1 层: {len(app.get_graph(xray=1).nodes)} 节点")
print(f"2 层: {len(app.get_graph(xray=2).nodes)} 节点")
print(f"全部: {len(app.get_graph(xray=True).nodes)} 节点")
```

**适用情况**:
- 图有多层嵌套
- 只需要查看部分层级
- 控制图表的复杂度

#### 场景4: RAG 工作流可视化

```python
# RAG 工作流通常包含子图
# 使用 xray=True 查看完整流程
from IPython.display import Image, display

display(Image(rag_app.get_graph(xray=True).draw_mermaid_png()))
```

**RAG 工作流示例**:
```python
# 主工作流
main_workflow = StateGraph(State)
main_workflow.add_node("retrieve", retrieve_func)
main_workflow.add_node("grade", grade_func)
main_workflow.add_node("generate", generate_func)

# 子工作流: 查询改写
query_rewrite_workflow = StateGraph(State)
query_rewrite_workflow.add_node("analyze", analyze_func)
query_rewrite_workflow.add_node("rewrite", rewrite_func)
query_rewrite_app = query_rewrite_workflow.compile()

# 将子工作流嵌入主工作流
main_workflow.add_node("query_rewrite", query_rewrite_app)

# 使用 xray=True 查看完整结构
rag_app = main_workflow.compile()
display(Image(rag_app.get_graph(xray=True).draw_mermaid_png()))
```

[来源: reference/context7_langgraph_01.md - RAG 工作流可视化]

### 实现原理

#### 1. 递归获取子图

```python
# get_graph() 的递归逻辑
if xray:
    subgraphs = {
        k: v.get_graph(
            config,
            xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
        )
        for k, v in self.get_subgraphs()
    }
```

**递归过程**:
1. 调用 `self.get_subgraphs()` 获取所有子图
2. 对每个子图递归调用 `get_graph(xray=...)`
3. 如果 xray 是整数,每层递减 1
4. 如果 xray 是布尔值,保持 True

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

#### 2. 子图节点替换

```python
# draw_graph() 的子图替换逻辑
for subgraph_name, subgraph in subgraphs.items():
    # 1. 找到子图节点
    subgraph_node = next(n for n in graph.nodes if n.id == subgraph_name)
    
    # 2. 获取子图的内部节点 (排除 START 和 END)
    subgraph_nodes = [n for n in subgraph.nodes if n.id not in (START, END)]
    
    # 3. 替换节点
    graph.nodes.remove(subgraph_node)
    graph.nodes.extend(subgraph_nodes)
    
    # 4. 重新连接边
    # 入边: 原本指向子图节点的边,现在指向子图的第一个节点
    # 出边: 原本从子图节点出发的边,现在从子图的最后一个节点出发
```

[来源: reference/source_图可视化_01.md - 子图的集成]

#### 3. 边的重新连接

```python
# 边的重新连接逻辑
# 1. 找到所有指向子图节点的边
incoming_edges = [e for e in graph.edges if e.target == subgraph_name]

# 2. 找到子图的第一个节点 (从 START 出发的节点)
first_nodes = [e.target for e in subgraph.edges if e.source == START]

# 3. 重新连接入边
for edge in incoming_edges:
    for first_node in first_nodes:
        graph.edges.append(Edge(edge.source, first_node, edge.conditional, edge.data))

# 4. 找到所有从子图节点出发的边
outgoing_edges = [e for e in graph.edges if e.source == subgraph_name]

# 5. 找到子图的最后一个节点 (指向 END 的节点)
last_nodes = [e.source for e in subgraph.edges if e.target == END]

# 6. 重新连接出边
for edge in outgoing_edges:
    for last_node in last_nodes:
        graph.edges.append(Edge(last_node, edge.target, edge.conditional, edge.data))
```

[来源: reference/source_图可视化_01.md - 子图的集成]

### 性能考虑

#### 1. 展开深度与性能

```python
import time

# 测试不同 xray 参数的性能
configs = [False, 1, 2, True]

for xray_value in configs:
    start = time.time()
    graph = app.get_graph(xray=xray_value)
    elapsed = time.time() - start
    
    print(f"xray={xray_value}: {len(graph.nodes)} 节点, {elapsed:.3f}s")
```

**性能影响**:
- `xray=False`: 最快,只处理顶层节点
- `xray=1`: 中等,处理一层子图
- `xray=True`: 最慢,递归处理所有子图

#### 2. 图表复杂度

```python
# 测试不同 xray 参数的图表复杂度
configs = [False, 1, 2, True]

for xray_value in configs:
    graph = app.get_graph(xray=xray_value)
    mermaid_text = graph.draw_mermaid()
    
    print(f"xray={xray_value}:")
    print(f"  节点数: {len(graph.nodes)}")
    print(f"  边数: {len(graph.edges)}")
    print(f"  Mermaid 文本长度: {len(mermaid_text)} 字符")
```

**复杂度影响**:
- 节点数: 随展开深度指数增长
- 边数: 随展开深度指数增长
- 文本长度: 随展开深度线性增长

### 常见问题

#### Q1: xray=True 和 xray=1 有什么区别?

**A**:
- `xray=True`: 递归展开所有子图,无限深度
- `xray=1`: 只展开一层子图,子图的子图不展开

```python
# 示例
graph_true = app.get_graph(xray=True)   # 展开所有层级
graph_1 = app.get_graph(xray=1)         # 只展开一层
```

#### Q2: 如何判断图是否包含子图?

**A**:
```python
# 方法1: 检查 get_subgraphs() 返回值
subgraphs = list(app.get_subgraphs())
has_subgraphs = len(subgraphs) > 0

# 方法2: 对比 xray=False 和 xray=True 的节点数
nodes_simple = len(app.get_graph(xray=False).nodes)
nodes_detailed = len(app.get_graph(xray=True).nodes)
has_subgraphs = nodes_detailed > nodes_simple
```

#### Q3: xray 参数会影响图的执行吗?

**A**: 不会。xray 参数只影响可视化,不影响图的实际执行:
- `get_graph()` 只是生成图的表示,不执行节点函数
- xray 参数只控制子图的展开,不改变图的逻辑

[来源: reference/source_图可视化_01.md - 静态分析 vs 动态执行]

#### Q4: 如何处理超大图的可视化?

**A**:
```python
# 策略1: 使用 xray=False 只查看顶层
graph_simple = app.get_graph(xray=False)

# 策略2: 使用 xray=1 只展开一层
graph_one_level = app.get_graph(xray=1)

# 策略3: 分别可视化每个子图
for name, subgraph_app in app.get_subgraphs():
    subgraph = subgraph_app.get_graph(xray=False)
    print(f"子图 {name}:")
    print(subgraph.draw_mermaid())
```

### 最佳实践

#### 1. 开发阶段

```python
# 快速查看顶层结构
print(app.get_graph(xray=False).draw_mermaid())
```

#### 2. 调试阶段

```python
# 深入查看子图结构
from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

#### 3. 文档阶段

```python
# 生成两个版本的图表
# 简单版: 用于概览
png_simple = app.get_graph(xray=False).draw_mermaid_png()
with open("docs/workflow_simple.png", "wb") as f:
    f.write(png_simple)

# 详细版: 用于深入理解
png_detailed = app.get_graph(xray=True).draw_mermaid_png()
with open("docs/workflow_detailed.png", "wb") as f:
    f.write(png_detailed)
```

#### 4. 性能优化

```python
# 缓存不同 xray 参数的结果
class CachedGraph:
    def __init__(self, app):
        self.app = app
        self._cache = {}
    
    def get_graph(self, xray=False):
        cache_key = f"xray_{xray}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.app.get_graph(xray=xray)
        return self._cache[cache_key]

cached = CachedGraph(app)
graph1 = cached.get_graph(xray=True)  # 计算
graph2 = cached.get_graph(xray=True)  # 从缓存获取
```

[来源: reference/context7_langgraph_01.md - 最佳实践]

### 总结

xray 参数是图可视化的核心特性:

1. **核心功能**: 控制子图的递归展开深度
2. **参数类型**: 布尔值 (True/False) 或整数 (层数)
3. **实现原理**: 递归获取子图,替换节点,重新连接边
4. **使用场景**: 开发、调试、文档生成
5. **性能考虑**: 展开深度越大,性能开销越大

**记住**: xray 参数只影响可视化,不影响图的实际执行。合理使用 xray 参数可以帮助你更好地理解和调试复杂的嵌套图结构。

---

**版本**: v1.0
**最后更新**: 2026-02-25
**维护者**: Claude Code
**数据来源**: [reference/source_图可视化_01.md, reference/context7_langgraph_01.md, reference/context7_langchain_01.md]
