---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-25
knowledge_point: 04_图的编译与执行
context7_query: compile invoke execution flow 2025 2026
---

# Context7 文档：LangGraph 源码仓库中的编译和执行示例

## 文档来源
- 库名称：LangGraph
- 版本：latest
- 源码仓库：https://github.com/langchain-ai/langgraph/

## 关键信息提取

### 1. Execute a LangGraph Workflow and Stream Outputs (Python)

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb

演示如何执行编译后的 LangGraph 工作流（`app`）并流式输出。使用 `app.stream()` 迭代工作流的执行，打印每个节点的输出。这允许观察 RAG 过程的逐步进展并调试流程。

```python
# Run
inputs = {
    "question": "What player are the Bears expected to draft first in the 2024 NFL draft?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
    pprint.pprint("\n---\n")
```

**关键点**：
- 使用 `app.stream(inputs)` 流式执行
- 迭代输出，每个输出是一个字典
- 字典的键是节点名称，值是节点的输出
- 可以打印每个节点的完整状态

### 2. Execute Graph with Alternative Query Input

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_pinecone_movies.ipynb

演示使用不同的输入问题流式执行编译后的工作流。展示相同的图结构如何处理多个查询并通过条件路由逻辑产生相应的生成。

```python
inputs = {"question": "Which movies are about aliens?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
```

**关键点**：
- 相同的编译后的图可以处理不同的输入
- 最终结果在 `value["generation"]` 中
- 条件路由逻辑自动处理不同的查询

### 3. Stream Graph Execution with Input Question

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_pinecone_movies.ipynb

执行编译后的 LangGraph 工作流并流式输出，迭代节点执行结果并打印最终生成。演示如何处理图输出并在执行期间访问状态值。

```python
from pprint import pprint

# Run
inputs = {"question": "Movies that star Daniel Craig"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
```

**关键点**：
- 使用 `pprint` 美化输出
- 可以在执行期间访问状态值
- 最终生成在循环结束后访问

### 4. Stream Workflow Execution and Output Results - Python

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb

执行编译后的工作流并流式输出每个节点的输出。演示迭代节点输出并访问最终生成结果。

```python
from pprint import pprint

# Run
inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
```

**关键点**：
- 可选地打印每个节点的完整状态
- 使用 `value["keys"]` 访问状态的键
- 支持自定义打印格式（indent, width, depth）

### 5. Compile and Stream LangGraph Workflow Results

**来源**：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_local.ipynb

将工作流编译成可执行应用程序并流式执行结果。迭代 `app.stream()` 的输出以打印节点名称及其相应的状态值。演示在工作流完成后访问最终生成输出。

```python
app = workflow.compile()

from pprint import pprint

inputs = {"question": "Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
final_generation = value["generation"]
pprint(final_generation)
```

**关键点**：
- 先编译工作流：`app = workflow.compile()`
- 然后流式执行：`app.stream(inputs)`
- 最终生成在循环结束后访问

## 总结

### 编译和执行的典型模式

1. **编译阶段**：
   ```python
   app = workflow.compile()
   ```

2. **执行阶段**：
   ```python
   inputs = {"question": "..."}
   for output in app.stream(inputs):
       for key, value in output.items():
           print(f"Node '{key}':")
   ```

3. **访问最终结果**：
   ```python
   final_result = value["generation"]
   ```

### 实际应用模式

1. **RAG 应用**：
   - 输入：用户问题
   - 流式输出：每个节点的执行结果
   - 最终输出：生成的答案

2. **调试和监控**：
   - 打印每个节点的名称
   - 可选地打印完整状态
   - 观察执行流程

3. **多查询处理**：
   - 相同的编译后的图
   - 不同的输入问题
   - 自动路由和处理

### 最佳实践

1. **编译一次，多次执行**：
   - 编译是一次性操作
   - 编译后的图可以多次调用

2. **流式输出用于监控**：
   - 使用 `stream()` 观察执行过程
   - 使用 `invoke()` 获取最终结果

3. **状态访问**：
   - 在循环中访问中间状态
   - 在循环结束后访问最终状态

4. **错误处理**：
   - 在循环中捕获异常
   - 打印节点名称帮助定位问题
