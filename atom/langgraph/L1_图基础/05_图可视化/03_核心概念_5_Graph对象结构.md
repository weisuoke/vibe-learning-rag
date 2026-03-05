# 图可视化 - 核心概念5: Graph 对象结构

## 概念定位

**核心概念**: Graph 对象结构
**重要程度**: ⭐⭐⭐ (中)
**使用频率**: 中等
**难度等级**: ⭐⭐ (中等)

## 一句话定义

**Graph 对象是来自 langchain_core.runnables.graph 的数据结构,包含节点(nodes)和边(edges)的完整图表示,是 LangGraph 可视化的核心数据载体。**

## 详细解释

### 对象来源

```python
from langchain_core.runnables.graph import Graph, Node, Edge

# Graph 对象由 get_graph() 返回
graph = app.get_graph()
print(type(graph))  # <class 'langchain_core.runnables.graph.Graph'>
```

**关键点**:
- Graph 对象来自 `langchain_core.runnables.graph`
- 不是 LangGraph 特有的,而是 LangChain Core 的通用组件
- 用于表示任何可运行对象(Runnable)的图结构

[来源: reference/source_图可视化_01.md - Graph 对象]

### 核心属性

#### 1. nodes 属性

```python
# 获取所有节点
graph = app.get_graph()
print(f"节点数量: {len(graph.nodes)}")

# 遍历节点
for node in graph.nodes:
    print(f"节点 ID: {node.id}")
    print(f"节点名称: {node.name}")
    print(f"节点类型: {type(node)}")
```

**Node 对象结构**:
```python
class Node:
    id: str          # 节点唯一标识符
    name: str        # 节点显示名称
    data: Any        # 节点关联的数据
    metadata: dict   # 节点元数据
```

**特殊节点**:
- `__start__`: 图的起始节点
- `__end__`: 图的结束节点
- 普通节点: 用户定义的节点

[来源: reference/source_图可视化_01.md - Graph 对象]

#### 2. edges 属性

```python
# 获取所有边
graph = app.get_graph()
print(f"边数量: {len(graph.edges)}")

# 遍历边
for edge in graph.edges:
    print(f"边: {edge.source} -> {edge.target}")
    print(f"条件边: {edge.conditional}")
    if edge.data:
        print(f"边标签: {edge.data}")
```

**Edge 对象结构**:
```python
class Edge:
    source: str       # 源节点 ID
    target: str       # 目标节点 ID
    conditional: bool # 是否为条件边
    data: str | None  # 边的标签或数据
```

**边的类型**:
- 普通边: `conditional=False`
- 条件边: `conditional=True`

[来源: reference/source_图可视化_01.md - 关键数据结构]

### 核心方法

#### 1. draw_mermaid()

```python
# 生成 Mermaid 文本
graph = app.get_graph()
mermaid_text = graph.draw_mermaid()
print(mermaid_text)
```

**方法签名**:
```python
def draw_mermaid(
    self,
    *,
    with_styles: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_colors: NodeStyles = NodeStyles.default(),
    wrap_label_n_words: int = 9,
) -> str:
    """Draw the graph as a Mermaid syntax string."""
```

[来源: reference/context7_langchain_01.md - Python API]

#### 2. draw_mermaid_png()

```python
# 生成 PNG 图像
graph = app.get_graph()
png_data = graph.draw_mermaid_png()

# 保存到文件
with open("graph.png", "wb") as f:
    f.write(png_data)
```

**方法签名**:
```python
def draw_mermaid_png(
    self,
    *,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    background_color: str = "white",
    padding: int = 10,
) -> bytes:
    """Draw the graph as a PNG image."""
```

[来源: reference/context7_langchain_01.md - Python API]

#### 3. add_node()

```python
# 添加节点 (通常由 draw_graph() 内部调用)
graph = Graph()
graph.add_node(Node(id="node1", name="Node 1"))
```

#### 4. add_edge()

```python
# 添加边 (通常由 draw_graph() 内部调用)
graph = Graph()
graph.add_edge(Edge(source="node1", target="node2", conditional=False, data=None))
```

### 实际应用场景

#### 场景1: 分析图结构

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 定义状态
class State(TypedDict):
    messages: list[str]

# 创建图
workflow = StateGraph(State)
workflow.add_node("node1", lambda x: x)
workflow.add_node("node2", lambda x: x)
workflow.add_node("node3", lambda x: x)
workflow.add_edge(START, "node1")
workflow.add_edge("node1", "node2")
workflow.add_conditional_edges(
    "node2",
    lambda x: "node3" if x.get("condition") else END,
    {
        "node3": "node3",
        END: END
    }
)
workflow.add_edge("node3", END)

app = workflow.compile()

# 获取 Graph 对象
graph = app.get_graph()

# 分析节点
print(f"总节点数: {len(graph.nodes)}")
print("节点列表:")
for node in graph.nodes:
    print(f"  - {node.id}")

# 分析边
print(f"\n总边数: {len(graph.edges)}")
print("边列表:")
for edge in graph.edges:
    edge_type = "条件边" if edge.conditional else "普通边"
    label = f" ({edge.data})" if edge.data else ""
    print(f"  - {edge.source} -> {edge.target} [{edge_type}]{label}")

# 分析入度和出度
from collections import defaultdict

in_degree = defaultdict(int)
out_degree = defaultdict(int)

for edge in graph.edges:
    out_degree[edge.source] += 1
    in_degree[edge.target] += 1

print("\n节点度数:")
for node in graph.nodes:
    print(f"  - {node.id}: 入度={in_degree[node.id]}, 出度={out_degree[node.id]}")
```

**输出示例**:
```
总节点数: 5
节点列表:
  - __start__
  - node1
  - node2
  - node3
  - __end__

总边数: 5
边列表:
  - __start__ -> node1 [普通边]
  - node1 -> node2 [普通边]
  - node2 -> node3 [条件边] (node3)
  - node2 -> __end__ [条件边]
  - node3 -> __end__ [普通边]

节点度数:
  - __start__: 入度=0, 出度=1
  - node1: 入度=1, 出度=1
  - node2: 入度=1, 出度=2
  - node3: 入度=1, 出度=1
  - __end__: 入度=2, 出度=0
```

#### 场景2: 验证图结构

```python
# 验证图的完整性
def validate_graph(graph):
    """验证图结构的完整性"""
    errors = []
    
    # 1. 检查是否有 START 节点
    start_nodes = [n for n in graph.nodes if n.id == "__start__"]
    if not start_nodes:
        errors.append("缺少 START 节点")
    
    # 2. 检查是否有 END 节点
    end_nodes = [n for n in graph.nodes if n.id == "__end__"]
    if not end_nodes:
        errors.append("缺少 END 节点")
    
    # 3. 检查所有节点是否可达
    node_ids = {n.id for n in graph.nodes}
    reachable = set()
    
    # 从 START 开始 BFS
    queue = ["__start__"]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        
        # 找到所有出边
        for edge in graph.edges:
            if edge.source == current and edge.target not in reachable:
                queue.append(edge.target)
    
    unreachable = node_ids - reachable
    if unreachable:
        errors.append(f"不可达节点: {unreachable}")
    
    # 4. 检查边的有效性
    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"边的源节点不存在: {edge.source}")
        if edge.target not in node_ids:
            errors.append(f"边的目标节点不存在: {edge.target}")
    
    return errors

# 验证图
graph = app.get_graph()
errors = validate_graph(graph)

if errors:
    print("图结构验证失败:")
    for error in errors:
        print(f"  - {error}")
else:
    print("图结构验证通过")
```

#### 场景3: 导出图数据

```python
import json

# 导出图数据为 JSON
def export_graph_to_json(graph):
    """导出图数据为 JSON 格式"""
    data = {
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "conditional": edge.conditional,
                "data": edge.data,
            }
            for edge in graph.edges
        ]
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

# 导出图数据
graph = app.get_graph()
json_data = export_graph_to_json(graph)

# 保存到文件
with open("graph.json", "w", encoding="utf-8") as f:
    f.write(json_data)

print("图数据已导出到 graph.json")
```

**输出示例** (graph.json):
```json
{
  "nodes": [
    {"id": "__start__", "name": "__start__"},
    {"id": "node1", "name": "node1"},
    {"id": "node2", "name": "node2"},
    {"id": "__end__", "name": "__end__"}
  ],
  "edges": [
    {"source": "__start__", "target": "node1", "conditional": false, "data": null},
    {"source": "node1", "target": "node2", "conditional": false, "data": null},
    {"source": "node2", "target": "__end__", "conditional": false, "data": null}
  ]
}
```

#### 场景4: 图结构对比

```python
# 对比两个图的结构
def compare_graphs(graph1, graph2):
    """对比两个图的结构差异"""
    # 对比节点
    nodes1 = {n.id for n in graph1.nodes}
    nodes2 = {n.id for n in graph2.nodes}
    
    added_nodes = nodes2 - nodes1
    removed_nodes = nodes1 - nodes2
    
    # 对比边
    edges1 = {(e.source, e.target) for e in graph1.edges}
    edges2 = {(e.source, e.target) for e in graph2.edges}
    
    added_edges = edges2 - edges1
    removed_edges = edges1 - edges2
    
    # 输出差异
    print("图结构对比:")
    if added_nodes:
        print(f"  新增节点: {added_nodes}")
    if removed_nodes:
        print(f"  删除节点: {removed_nodes}")
    if added_edges:
        print(f"  新增边: {added_edges}")
    if removed_edges:
        print(f"  删除边: {removed_edges}")
    
    if not (added_nodes or removed_nodes or added_edges or removed_edges):
        print("  两个图结构相同")

# 对比图
graph1 = app_v1.get_graph()
graph2 = app_v2.get_graph()
compare_graphs(graph1, graph2)
```

### 与 LangGraph 的关系

#### 1. 生成流程

```
LangGraph StateGraph
    ↓ compile()
LangGraph CompiledGraph
    ↓ get_graph()
langchain_core Graph 对象
    ↓ draw_mermaid() / draw_mermaid_png()
可视化输出
```

#### 2. 数据转换

```python
# LangGraph 内部表示 → Graph 对象
def get_graph(self, config=None, *, xray=False):
    # 1. 收集子图
    if xray:
        subgraphs = {...}
    else:
        subgraphs = {}
    
    # 2. 调用 draw_graph() 生成 Graph 对象
    return draw_graph(
        config=merge_configs(self.config, config),
        nodes=self.nodes,              # LangGraph 节点
        specs=self.channels,           # LangGraph 通道
        input_channels=self.input_channels,
        interrupt_after_nodes=self.interrupt_after_nodes,
        interrupt_before_nodes=self.interrupt_before_nodes,
        trigger_to_nodes=self.trigger_to_nodes,
        checkpointer=self.checkpointer,
        subgraphs=subgraphs,
    )
```

[来源: reference/source_图可视化_01.md - get_graph() 方法实现]

### 常见问题

#### Q1: Graph 对象可以修改吗?

**A**: 可以,但通常不建议:
```python
# 可以添加节点和边
graph = app.get_graph()
graph.add_node(Node(id="new_node", name="New Node"))
graph.add_edge(Edge(source="node1", target="new_node", conditional=False, data=None))

# 但这不会影响原始的 LangGraph 图
# 只会影响可视化输出
```

#### Q2: 如何获取特定节点?

**A**:
```python
# 方法1: 遍历查找
graph = app.get_graph()
target_node = next((n for n in graph.nodes if n.id == "node1"), None)

# 方法2: 使用字典
node_dict = {n.id: n for n in graph.nodes}
target_node = node_dict.get("node1")
```

#### Q3: 如何获取节点的所有入边和出边?

**A**:
```python
def get_node_edges(graph, node_id):
    """获取节点的所有入边和出边"""
    incoming = [e for e in graph.edges if e.target == node_id]
    outgoing = [e for e in graph.edges if e.source == node_id]
    return incoming, outgoing

# 使用
graph = app.get_graph()
incoming, outgoing = get_node_edges(graph, "node1")
print(f"入边: {len(incoming)}, 出边: {len(outgoing)}")
```

#### Q4: Graph 对象可以序列化吗?

**A**: 不能直接序列化,但可以转换为字典:
```python
# 转换为字典
def graph_to_dict(graph):
    return {
        "nodes": [{"id": n.id, "name": n.name} for n in graph.nodes],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "conditional": e.conditional,
                "data": e.data
            }
            for e in graph.edges
        ]
    }

# 序列化
import json
graph_dict = graph_to_dict(graph)
json_str = json.dumps(graph_dict, indent=2)
```

### 最佳实践

#### 1. 分析图结构

```python
# 使用 Graph 对象分析图结构
graph = app.get_graph()

# 统计信息
print(f"节点数: {len(graph.nodes)}")
print(f"边数: {len(graph.edges)}")
print(f"条件边数: {sum(1 for e in graph.edges if e.conditional)}")
```

#### 2. 验证图完整性

```python
# 验证图结构
errors = validate_graph(graph)
if errors:
    raise ValueError(f"图结构无效: {errors}")
```

#### 3. 导出图数据

```python
# 导出为 JSON 用于其他工具
json_data = export_graph_to_json(graph)
with open("graph.json", "w") as f:
    f.write(json_data)
```

#### 4. 对比图版本

```python
# 对比不同版本的图结构
graph_v1 = app_v1.get_graph()
graph_v2 = app_v2.get_graph()
compare_graphs(graph_v1, graph_v2)
```

### 总结

Graph 对象是 LangGraph 可视化的核心数据结构:

1. **来源**: langchain_core.runnables.graph
2. **核心属性**: nodes (节点列表), edges (边列表)
3. **核心方法**: draw_mermaid(), draw_mermaid_png()
4. **使用场景**: 分析图结构、验证完整性、导出数据、对比版本
5. **关键特性**: 通用的图表示,不限于 LangGraph

**记住**: Graph 对象是只读的数据结构,用于表示和可视化图,不用于修改图的逻辑。

---

**版本**: v1.0
**最后更新**: 2026-02-25
**维护者**: Claude Code
**数据来源**: [reference/source_图可视化_01.md, reference/context7_langgraph_01.md, reference/context7_langchain_01.md]
