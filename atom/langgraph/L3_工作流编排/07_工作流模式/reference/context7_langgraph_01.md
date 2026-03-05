---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-03-01
knowledge_point: 07_工作流模式
context7_query: workflow patterns sequential parallel branching fan-out fan-in map-reduce
---

# Context7 文档：LangGraph 工作流模式

## 文档来源
- 库名称：langgraph
- Context7 ID：/websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/workflows-agents

## 官方定义的六大工作流模式

### 1. Prompt Chaining（顺序链式模式）
- 多个 LLM 调用按顺序执行
- 前一步输出作为后一步输入
- 支持中间 Gate 检查（条件继续/终止）
- 示例：生成笑话 → 检查质量 → 改进 → 润色

### 2. Routing（路由模式）
- 根据输入动态选择执行路径
- 使用 LLM structured output 做路由决策
- `add_conditional_edges` 实现多路分支
- 示例：根据用户请求路由到 story/joke/poem

### 3. Parallelization（并行模式）
- Fan-out：一个节点触发多个并行节点
- Fan-in：多个并行节点汇聚到一个节点
- 使用 `operator.add` reducer 聚合结果
- `defer=True` 支持不等长分支的同步

### 4. Orchestrator-Worker（编排者-工作者模式）
- 编排者动态分解任务
- 工作者并行执行子任务
- 合成器汇总结果
- 使用 Functional API 的 `@task` 装饰器

### 5. Evaluator-Optimizer（评估者-优化者模式）
- 生成 → 评估 → 反馈循环
- 条件边控制循环终止
- 支持 structured output 做评估
- 示例：生成笑话 → 评估是否好笑 → 根据反馈改进

### 6. Map-Reduce（映射-归约模式）
- 使用 `Send` API 动态创建并行任务
- 支持动态数量的并行分支
- 结果通过 reducer 聚合
- 示例：生成主题列表 → 为每个主题生成笑话 → 选最佳

## 两种 API 风格

### Graph API（StateGraph）
- 显式定义节点和边
- 适合复杂工作流
- 支持可视化

### Functional API（@entrypoint + @task）
- 更 Pythonic 的写法
- 适合简单工作流
- 自动推断图结构

## 关键代码模式

### 顺序模式
```python
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)
```

### 并行模式
```python
builder.add_edge(START, "a")
builder.add_edge("a", "b")  # fan-out
builder.add_edge("a", "c")  # fan-out
builder.add_edge("b", "d")  # fan-in
builder.add_edge("c", "d")  # fan-in
builder.add_edge("d", END)
```

### 路由模式
```python
builder.add_conditional_edges(
    "router", route_fn, {"path_a": "node_a", "path_b": "node_b"}
)
```

### Map-Reduce 模式
```python
def fan_out(state):
    return [Send("worker", {"item": i}) for i in state["items"]]

builder.add_conditional_edges("splitter", fan_out, ["worker"])
```
