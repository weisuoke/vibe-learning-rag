---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-25
knowledge_point: 04_图的编译与执行
context7_query: compile method StateGraph checkpointer interrupt
---

# Context7 文档：LangGraph compile 方法

## 文档来源
- 库名称：LangGraph
- 版本：latest
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/

## 关键信息提取

### 1. Compile Time Interrupts in LangGraph (Python)

**来源**：https://docs.langchain.com/oss/python/langgraph/interrupts

演示在编译时为 LangGraph 设置静态中断点。它需要 checkpointer，并展示如何调用图以触发断点然后恢复它。`interrupt_before` 和 `interrupt_after` 参数指定暂停执行的节点。

```python
graph = builder.compile(
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    checkpointer=checkpointer,
)

# 向图传递 thread ID
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# 运行图直到断点
graph.invoke(inputs, config=config)

# 恢复图
graph.invoke(None, config=config)
```

**关键点**：
- `interrupt_before` - 在指定节点前中断
- `interrupt_after` - 在指定节点后中断
- 需要 `checkpointer` 保存状态
- 使用相同的 `thread_id` 恢复执行
- 恢复时传递 `None` 作为输入

### 2. Build and Invoke Approval Graph with Interrupt (Python)

**来源**：https://docs.langchain.com/oss/python/langgraph/interrupts

提供使用 LangGraph 的 StateGraph 创建有状态图的完整示例。它包括批准、继续和取消的节点，利用 `interrupt` 函数暂停执行以等待用户决策。图使用 MemorySaver checkpointer 编译，并使用初始状态和配置调用。

```python
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]


def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    # 暴露详情以便调用者可以在 UI 中渲染
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })

    # 恢复后路由到适当的节点
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState):
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# 在生产环境中使用更持久的 checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=config,
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

# 使用决策恢复；True 路由到 proceed，False 路由到 cancel
resumed = graph.invoke(Command(resume=True), config=config)
print(resumed["status"])  # -> "approved"
```

**关键点**：
- 使用 `interrupt()` 函数动态中断
- 中断值在 `__interrupt__` 字段中返回
- 使用 `Command(resume=...)` 恢复执行
- 使用 `Command(goto=...)` 控制路由
- `MemorySaver` 用于内存中的 checkpointing

### 3. Full LangGraph Example: Age Collection with Validation (Python)

**来源**：https://docs.langchain.com/oss/python/langgraph/interrupts

这个全面的 Python 示例展示了一个完整的 LangGraph 应用程序，用于收集和验证用户年龄。它利用 `SqliteSaver` 进行 checkpointing，`StateGraph` 定义工作流，以及 `interrupt` 进行交互式输入验证。该示例演示了使用无效和有效输入调用图，以展示重新提示和状态更新机制。

```python
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None


def get_age_node(state: FormState):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload 在 result["__interrupt__"] 中显示

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}' is not a valid age. Please enter a positive number."


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

checkpointer = SqliteSaver(sqlite3.connect("forms.db"))
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age": None}, config=config)
print(first["__interrupt__"])  # -> [Interrupt(value='What is your age?', ...)]

# 提供无效数据；节点重新提示
retry = graph.invoke(Command(resume="thirty"), config=config)
print(retry["__interrupt__"])  # -> [Interrupt(value=''thirty' is not a valid age...', ...)]

# 提供有效数据；循环退出并更新状态
final = graph.invoke(Command(resume=30), config=config)
print(final["age"])  # -> 30
```

**关键点**：
- `SqliteSaver` 用于持久化 checkpointing
- `interrupt()` 可以在循环中使用
- 支持输入验证和重新提示
- 使用 `Command(resume=...)` 提供用户输入

### 4. Interrupts

**来源**：https://docs.langchain.com/oss/python/langgraph/interrupts

中断通过在图节点的任何点调用 `interrupt()` 函数来工作。该函数接受任何 JSON 可序列化的值，该值会暴露给调用者。当你准备继续时，使用 `Command` 重新调用图来恢复执行，然后它成为节点内 `interrupt()` 调用的返回值。与静态断点（在特定节点前后暂停）不同，中断是**动态的**——它们可以放置在代码的任何位置，并且可以基于应用程序逻辑有条件地触发。

**关键特性**：
- **Checkpointing 保持你的位置**：checkpointer 写入确切的图状态，以便你可以稍后恢复，即使在错误状态下也是如此。
- **`thread_id` 是你的指针**：设置 `config={"configurable": {"thread_id": ...}}` 告诉 checkpointer 加载哪个状态。
- **中断 payload 作为 `__interrupt__` 显示**：你传递给 `interrupt()` 的值在 `__interrupt__` 字段中返回给调用者，以便你知道图在等待什么。

你选择的 `thread_id` 实际上是你的持久游标。重用它会恢复相同的 checkpoint；使用新值会以空状态启动全新的线程。

### 5. LangGraph Python > Concepts > Graph Compilation

**来源**：https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph

LangGraph 中的图编译过程涉及定义节点及其连接。对于需要持久化的工作流，如使用 `interrupt()` 的工作流，必须在编译期间提供 checkpointer。`StateGraph` 用于定义节点和边，`compile()` 方法最终确定图，可选地包括用于状态管理的 checkpointer。

**关键点**：
- 编译过程定义节点和连接
- 需要持久化时必须提供 checkpointer
- `compile()` 方法最终确定图

## 总结

### compile 方法核心特性

1. **Checkpointer 配置**：
   - 必须提供 checkpointer 用于持久化
   - `MemorySaver` - 内存中的 checkpointing
   - `SqliteSaver` - SQLite 持久化
   - 其他：PostgreSQL, Redis 等

2. **中断点配置**：
   - **静态中断**：
     - `interrupt_before` - 在节点前中断
     - `interrupt_after` - 在节点后中断
   - **动态中断**：
     - 使用 `interrupt()` 函数
     - 可以放置在代码的任何位置
     - 可以有条件地触发

3. **恢复机制**：
   - 使用相同的 `thread_id` 恢复
   - 静态中断：传递 `None` 作为输入
   - 动态中断：使用 `Command(resume=...)` 提供数据

4. **中断数据**：
   - 中断值在 `__interrupt__` 字段中返回
   - 可以传递任何 JSON 可序列化的值
   - 用于向调用者传递上下文信息

### 实际应用场景

1. **人机协作**：使用中断等待用户批准或输入
2. **调试和测试**：在特定节点前后暂停检查状态
3. **长时间运行任务**：使用 checkpointer 支持断点续传
4. **交互式应用**：使用动态中断实现表单验证等交互
5. **错误恢复**：使用 checkpointer 在错误后恢复执行
