---
type: context7_documentation
library: LangGraph
version: latest (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: 05_图可视化
context7_query: graph visualization mermaid syntax xray debugging
---

# Context7 文档: LangGraph 图可视化详细文档

## 文档来源
- 库名称: LangGraph
- 版本: latest (2026-02-17)
- 官方文档链接: https://docs.langchain.com/oss/python/langgraph
- Context7 库 ID: /websites/langchain_oss_python_langgraph

## 关键信息提取

### 1. 快速入门中的可视化

**来源**: https://docs.langchain.com/oss/python/langgraph/quickstart

**代码示例**:
```python
from IPython.display import Image, display

# 显示代理图
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 调用代理
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

**功能说明**:
- 使用 `xray=True` 参数可视化编译后的代理图
- 使用 `draw_mermaid_png()` 生成 PNG 图像
- 在 Jupyter 中使用 `display()` 显示图像
- 可视化后可以立即调用代理进行测试

### 2. SQL 代理可视化

**来源**: https://docs.langchain.com/oss/python/langgraph/sql-agent

**代码示例**:
```python
from IPython.display import Image, display

# 假设 'agent' 是一个编译后的 LangGraph 代理对象
display(Image(agent.get_graph().draw_mermaid_png()))
```

**功能说明**:
- 可视化 SQL 代理的计算图
- 使用 Mermaid 格式渲染
- 需要 `IPython.display` 来显示图像
- 主要依赖是代理对象本身,必须有 `get_graph()` 方法

### 3. 使用 Graph API

**来源**: https://docs.langchain.com/oss/python/langgraph/use-graph-api

**生成 Mermaid 语法**:
```python
print(app.get_graph().draw_mermaid())
```

**生成 PNG 图像**:
```python
from IPython.display import Image, display

# 假设 'graph' 已经定义并编译
# display(Image(graph.get_graph().draw_mermaid_png()))
```

**使用 Mermaid.ink API**:
```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(Image(app.get_graph().draw_mermaid_png()))
```

**功能说明**:
- `draw_mermaid()` 输出 Mermaid 文本语法字符串
- `draw_mermaid_png()` 使用 Mermaid.ink API 渲染 PNG 图像
- 可以导入 `CurveStyle`, `MermaidDrawMethod`, `NodeStyles` 进行样式定制
- 不需要额外的包,LangGraph 自带这些功能

### 4. RAG 工作流可视化

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb

**完整工作流示例**:
```python
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

custom_graph = workflow.compile()

display(Image(custom_graph.get_graph(xray=True).draw_mermaid_png()))
```

**功能说明**:
- 定义完整的 RAG 工作流边和条件边
- 编译工作流为自定义图
- 使用 `xray=True` 提供详细的节点级可视化
- 展示图的拓扑结构

### 5. 自适应 RAG 工作流

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb

**复杂工作流示例**:
```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("web_search", web_search)  # 网络搜索
workflow.add_node("retrieve", retrieve)  # 检索
workflow.add_node("grade_documents", grade_documents)  # 文档评分
workflow.add_node("generate", generate)  # 生成
workflow.add_node("transform_query", transform_query)  # 查询转换

# 构建图
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# 编译
app = workflow.compile()
```

**功能说明**:
- 展示复杂的自适应 RAG 工作流
- 包含多个条件边和路由函数
- 使用 `StateGraph` 构建状态图
- 支持动态路由和决策

### 6. Cohere 自适应 RAG

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb

**完整工作流示例**:
```python
import pprint
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("web_search", web_search)  # 网络搜索
workflow.add_node("retrieve", retrieve)  # 检索
workflow.add_node("grade_documents", grade_documents)  # 文档评分
workflow.add_node("generate", generate)  # RAG
workflow.add_node("llm_fallback", llm_fallback)  # LLM 后备

# 构建图
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # 幻觉:重新生成
        "not useful": "web_search",  # 无法回答问题:回退到网络搜索
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# 编译
app = workflow.compile()
```

**功能说明**:
- 展示带有 LLM 后备的自适应 RAG
- 包含幻觉检测和重新生成逻辑
- 支持多种路由策略
- 使用条件边实现复杂的决策流程

## 核心模式总结

### 1. 基础可视化模式

```python
# 模式 1: 文本输出
print(app.get_graph().draw_mermaid())

# 模式 2: PNG 图像
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# 模式 3: 带 xray 的详细视图
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

### 2. 工作流构建模式

```python
# 1. 创建 StateGraph
workflow = StateGraph(GraphState)

# 2. 添加节点
workflow.add_node("node_name", node_function)

# 3. 添加边
workflow.add_edge(START, "first_node")
workflow.add_edge("node1", "node2")
workflow.add_edge("last_node", END)

# 4. 添加条件边
workflow.add_conditional_edges(
    "source_node",
    routing_function,
    {
        "option1": "target_node1",
        "option2": "target_node2",
    }
)

# 5. 编译
app = workflow.compile()

# 6. 可视化
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

### 3. 调试模式

```python
# 1. 可视化图结构
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# 2. 调用并查看结果
result = app.invoke({"input": "test"})

# 3. 打印消息
for m in result["messages"]:
    m.pretty_print()
```

## 实际应用场景

### 1. 开发阶段
- 在构建工作流时实时查看图结构
- 验证节点和边的连接关系
- 检查条件路由的逻辑

### 2. 调试阶段
- 使用 `xray=True` 深入查看子图结构
- 识别工作流中的瓶颈和问题
- 验证路由函数的决策逻辑

### 3. 文档阶段
- 为项目文档生成图表
- 导出为 PNG 图像嵌入文档
- 生成 Mermaid 语法用于 Markdown

### 4. 团队协作
- 分享图结构给团队成员
- 讨论工作流设计和优化
- 代码审查时可视化变更

## 依赖和要求

### Python 依赖
- `IPython` - 用于 Jupyter 显示
- `langgraph` - 核心库
- `langchain_core` - 基础组件

### 可选依赖
- Mermaid.ink API (用于 PNG 生成)
- 图像渲染库 (根据环境)

## 最佳实践

1. **开发阶段**: 使用 `draw_mermaid()` 快速查看文本表示
2. **文档阶段**: 使用 `draw_mermaid_png()` 生成高质量图像
3. **调试阶段**: 使用 `xray=True` 深入查看子图结构
4. **生产环境**: 避免在生产代码中生成可视化 (性能考虑)
5. **异常处理**: 使用 try-except 包裹可视化代码,因为 PNG 生成可能失败

## 注意事项

1. PNG 生成需要额外依赖,可能在某些环境中失败
2. 大型图可能生成较大的图像文件
3. `xray` 参数会递归展开所有子图,可能导致图表过于复杂
4. Mermaid 语法有长度限制,超大图可能无法正确渲染
5. 在 Jupyter 外使用时需要手动保存图像文件

## 高级特性

### 样式定制

```python
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# 可以导入这些类进行样式定制
# 具体用法需要查看 langchain_core 文档
```

### 条件边标签

条件边可以包含标签,用于说明路由决策:

```python
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # 标签: 幻觉检测
        "not useful": "web_search",   # 标签: 无法回答
        "useful": END,                # 标签: 成功
    },
)
```

这些标签会在可视化中显示,帮助理解路由逻辑。
