---
type: context7_documentation
library: langgraph
version: latest (2026-02)
fetched_at: 2026-02-28
knowledge_point: 04_人机循环（Human-in-the-loop）
context7_query: human in the loop interrupt Command resume checkpoint
---

# Context7 文档：LangGraph Human-in-the-loop

## 文档来源
- 库名称：LangGraph (Python)
- Library ID: /websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/interrupts

## 关键信息提取

### 1. 基础中断与恢复模式

```python
from langgraph.types import interrupt, Command

def approval_node(state: State):
    approved = interrupt("Do you approve this action?")
    return {"approved": approved}

# 初始运行 - 遇到中断暂停
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)
print(result["__interrupt__"])
# > [Interrupt(value='Do you approve this action?')]

# 使用 Command(resume=...) 恢复
resumed = graph.invoke(Command(resume="Approved"), config=config)
```

### 2. 完整审批工作流示例

```python
from typing import Literal, Optional, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
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

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=config,
)
# resume: True → proceed, False → cancel
resumed = graph.invoke(Command(resume=True), config=config)
print(resumed["status"])  # "approved"
```

### 3. 流式处理中断

```python
async for metadata, mode, chunk in graph.astream(
    initial_input,
    stream_mode=["messages", "updates"],
    subgraphs=True,
    config=config
):
    if mode == "updates":
        if "__interrupt__" in chunk:
            interrupt_info = chunk["__interrupt__"][0].value
            user_response = get_user_input(interrupt_info)
            initial_input = Command(resume=user_response)
            break
```

### 4. 多种恢复方式

```python
# 同步恢复
graph.invoke(Command(resume=value), config)

# 异步恢复
await graph.ainvoke(Command(resume=value), config)

# 流式恢复
for chunk in graph.stream(Command(resume=value), config):
    print(chunk)

# 异步流式恢复
async for chunk in graph.astream(Command(resume=value), config):
    print(chunk)
```

### 5. 关键要点

- interrupt() 的 value 必须是 JSON 可序列化的
- 必须启用 checkpointer（如 MemorySaver、PostgresSaver）
- thread_id 必须保持一致才能恢复
- 恢复时节点从头重新执行
- 多个 interrupt 按顺序匹配 resume 值
- Command 支持 resume + goto + update 组合
