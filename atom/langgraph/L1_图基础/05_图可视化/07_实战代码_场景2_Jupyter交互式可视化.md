# 图可视化 - 实战代码: 场景2 Jupyter交互式可视化

## 场景描述

在 Jupyter Notebook 中进行交互式开发,需要实时查看图结构的变化,快速迭代和调试工作流。

## 完整代码

```python
"""
场景2: Jupyter 交互式可视化
演示: 在 Jupyter Notebook 中交互式开发和调试 LangGraph 工作流
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display, Markdown

# ===== 1. 定义状态 =====
class State(TypedDict):
    """工作流状态定义"""
    query: str
    documents: list[str]
    answer: str
    step: int

# ===== 2. 定义节点函数 =====
def retrieve(state: State) -> State:
    """检索节点"""
    print(f"[Retrieve] 查询: {state['query']}")
    # 模拟检索
    documents = [
        f"文档1: 关于 {state['query']} 的信息",
        f"文档2: {state['query']} 的详细说明",
        f"文档3: {state['query']} 的实例"
    ]
    return {
        **state,
        "documents": documents,
        "step": state.get("step", 0) + 1
    }

def grade(state: State) -> State:
    """评分节点"""
    print(f"[Grade] 评分文档数量: {len(state['documents'])}")
    # 模拟评分
    relevant_docs = [doc for doc in state['documents'] if "详细" in doc or "实例" in doc]
    return {
        **state,
        "documents": relevant_docs,
        "step": state.get("step", 0) + 1
    }

def generate(state: State) -> State:
    """生成节点"""
    print(f"[Generate] 基于 {len(state['documents'])} 个文档生成答案")
    # 模拟生成
    answer = f"基于 {len(state['documents'])} 个相关文档,关于 {state['query']} 的答案是..."
    return {
        **state,
        "answer": answer,
        "step": state.get("step", 0) + 1
    }

# ===== 3. 创建工作流 - 版本1 (简单) =====
print("=== 版本1: 简单工作流 ===")
workflow_v1 = StateGraph(State)
workflow_v1.add_node("retrieve", retrieve)
workflow_v1.add_node("generate", generate)
workflow_v1.add_edge(START, "retrieve")
workflow_v1.add_edge("retrieve", "generate")
workflow_v1.add_edge("generate", END)

app_v1 = workflow_v1.compile()

# 在 Jupyter 中自动显示
display(Markdown("### 版本1: 简单工作流"))
app_v1  # 自动显示图表

# ===== 4. 创建工作流 - 版本2 (添加评分) =====
print("\n=== 版本2: 添加评分节点 ===")
workflow_v2 = StateGraph(State)
workflow_v2.add_node("retrieve", retrieve)
workflow_v2.add_node("grade", grade)
workflow_v2.add_node("generate", generate)
workflow_v2.add_edge(START, "retrieve")
workflow_v2.add_edge("retrieve", "grade")
workflow_v2.add_edge("grade", "generate")
workflow_v2.add_edge("generate", END)

app_v2 = workflow_v2.compile()

# 在 Jupyter 中自动显示
display(Markdown("### 版本2: 添加评分节点"))
app_v2  # 自动显示图表

# ===== 5. 对比两个版本 =====
display(Markdown("## 版本对比"))

# 版本1
display(Markdown("### 版本1 (简单)"))
display(Image(app_v1.get_graph().draw_mermaid_png()))

# 版本2
display(Markdown("### 版本2 (添加评分)"))
display(Image(app_v2.get_graph().draw_mermaid_png()))

# ===== 6. 执行并对比结果 =====
display(Markdown("## 执行结果对比"))

# 测试输入
test_input = {"query": "LangGraph", "documents": [], "answer": "", "step": 0}

# 执行版本1
display(Markdown("### 版本1 执行结果"))
result_v1 = app_v1.invoke(test_input)
print(f"最终答案: {result_v1['answer']}")
print(f"文档数量: {len(result_v1['documents'])}")
print(f"步骤数: {result_v1['step']}")

# 执行版本2
display(Markdown("### 版本2 执行结果"))
result_v2 = app_v2.invoke(test_input)
print(f"最终答案: {result_v2['answer']}")
print(f"文档数量: {len(result_v2['documents'])}")
print(f"步骤数: {result_v2['step']}")

# ===== 7. 交互式调试 =====
display(Markdown("## 交互式调试"))

# 创建带条件边的工作流
def decide_to_generate(state: State) -> str:
    """决策函数"""
    if len(state['documents']) > 0:
        return "generate"
    else:
        return "retrieve"

workflow_v3 = StateGraph(State)
workflow_v3.add_node("retrieve", retrieve)
workflow_v3.add_node("grade", grade)
workflow_v3.add_node("generate", generate)

workflow_v3.add_edge(START, "retrieve")
workflow_v3.add_edge("retrieve", "grade")
workflow_v3.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "retrieve": "retrieve"
    }
)
workflow_v3.add_edge("generate", END)

app_v3 = workflow_v3.compile()

# 显示条件边
display(Markdown("### 版本3: 带条件边"))
display(Image(app_v3.get_graph().draw_mermaid_png()))

# ===== 8. 使用 xray 参数深入查看 =====
display(Markdown("## 使用 xray 参数"))

# 简单视图
display(Markdown("### 简单视图 (xray=False)"))
display(Image(app_v3.get_graph(xray=False).draw_mermaid_png()))

# 详细视图
display(Markdown("### 详细视图 (xray=True)"))
display(Image(app_v3.get_graph(xray=True).draw_mermaid_png()))

# ===== 9. 实时修改和查看 =====
display(Markdown("## 实时修改和查看"))

# 添加新节点
def rewrite(state: State) -> State:
    """查询改写节点"""
    print(f"[Rewrite] 改写查询: {state['query']}")
    new_query = f"{state['query']} (改写)"
    return {
        **state,
        "query": new_query,
        "step": state.get("step", 0) + 1
    }

workflow_v4 = StateGraph(State)
workflow_v4.add_node("rewrite", rewrite)
workflow_v4.add_node("retrieve", retrieve)
workflow_v4.add_node("grade", grade)
workflow_v4.add_node("generate", generate)

workflow_v4.add_edge(START, "rewrite")
workflow_v4.add_edge("rewrite", "retrieve")
workflow_v4.add_edge("retrieve", "grade")
workflow_v4.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "retrieve": "rewrite"  # 回到改写
    }
)
workflow_v4.add_edge("generate", END)

app_v4 = workflow_v4.compile()

# 显示最终版本
display(Markdown("### 版本4: 添加查询改写"))
display(Image(app_v4.get_graph().draw_mermaid_png()))

# ===== 10. 性能对比 =====
display(Markdown("## 性能对比"))

import time

versions = [
    ("版本1 (简单)", app_v1),
    ("版本2 (评分)", app_v2),
    ("版本3 (条件边)", app_v3),
    ("版本4 (改写)", app_v4)
]

for name, app in versions:
    start = time.time()
    result = app.invoke(test_input)
    elapsed = time.time() - start
    
    print(f"{name}:")
    print(f"  执行时间: {elapsed:.3f}s")
    print(f"  步骤数: {result['step']}")
    print(f"  文档数: {len(result['documents'])}")
    print()

print("=== Jupyter 交互式可视化完成 ===")
```

## 在 Jupyter 中的显示效果

### 版本1: 简单工作流
![简单工作流图表]

### 版本2: 添加评分节点
![添加评分节点图表]

### 版本3: 带条件边
![带条件边图表]

### 版本4: 添加查询改写
![添加查询改写图表]

## 关键点说明

### 1. Jupyter 自动显示

```python
# 最简洁的方式
app  # 直接输入对象名,自动显示图表
```

### 2. 手动显示控制

```python
# 使用 display() 和 Image()
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

### 3. Markdown 标题

```python
# 使用 Markdown 添加标题
from IPython.display import Markdown, display
display(Markdown("## 工作流图表"))
app
```

### 4. 版本对比

```python
# 对比不同版本
display(Markdown("### 版本1"))
app_v1

display(Markdown("### 版本2"))
app_v2
```

## 使用场景

1. **快速原型开发**: 实时查看图结构变化
2. **迭代优化**: 对比不同版本的工作流
3. **团队演示**: 在 Notebook 中展示工作流设计
4. **教学培训**: 交互式讲解 LangGraph 概念

## 最佳实践

### 1. 使用 Markdown 组织内容

```python
display(Markdown("# 工作流开发"))
display(Markdown("## 版本1: 基础版本"))
app_v1
display(Markdown("## 版本2: 优化版本"))
app_v2
```

### 2. 结合代码和可视化

```python
# 显示代码
display(Markdown("```python\nworkflow.add_node('process', func)\n```"))

# 显示图表
app
```

### 3. 添加说明文字

```python
display(Markdown("### 说明"))
display(Markdown("这个工作流包含3个节点..."))
app
```

### 4. 使用 xray 参数调试

```python
# 简单视图
display(Markdown("### 简单视图"))
app

# 详细视图
display(Markdown("### 详细视图 (xray=True)"))
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```

## 常见问题

### Q1: 如何在 Jupyter 中禁用自动显示?

**A**: 使用分号或赋值:
```python
app;  # 分号抑制输出
result = app  # 赋值不显示
```

### Q2: 如何同时显示多个图表?

**A**: 使用 display():
```python
display(Image(app1.get_graph().draw_mermaid_png()))
display(Image(app2.get_graph().draw_mermaid_png()))
```

### Q3: 如何保存 Notebook 中的图表?

**A**: 右键图表 → 保存图像,或使用代码:
```python
png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
```

## 扩展练习

1. 创建多个版本的工作流并对比
2. 使用 Markdown 创建完整的开发文档
3. 结合执行结果和可视化进行分析
4. 尝试在 Jupyter 中进行实时调试

---

**版本**: v1.0
**最后更新**: 2026-02-25
**维护者**: Claude Code
**数据来源**: [reference/context7_langgraph_01.md - Jupyter 交互式可视化]
