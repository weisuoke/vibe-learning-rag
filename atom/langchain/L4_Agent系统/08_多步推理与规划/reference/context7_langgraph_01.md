---
type: context7_documentation
library: LangGraph
version: latest (2026)
fetched_at: 2026-03-06
knowledge_point: 08_多步推理与规划
context7_query: plan and execute agent planning replanning task decomposition
---

# Context7 文档：LangGraph 规划与执行

## 文档来源
- 库名称：LangGraph
- Library ID: /websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph

## 关键信息提取

### 1. Agent 工作流模式 (Functional API)

LangGraph 提供 Functional API 构建 Agent 工作流：
- `@task` 装饰器定义独立任务（如 call_llm, call_tool）
- `@entrypoint()` 定义 Agent 入口
- 循环执行：LLM → 检查 tool_calls → 执行工具 → 更新消息 → 重复

```python
@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return llm_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant...")] + messages
    )

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()
    while True:
        if not model_response.tool_calls:
            break
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()
    messages = add_messages(messages, model_response)
    return messages
```

### 2. Orchestrator-Worker 模式 (规划 + 分发 + 合成)

实现报告生成等复杂任务的规划分发模式：

```python
class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of topics.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

@task
def orchestrator(topic: str):
    """Orchestrator that generates a plan for the report"""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {topic}"),
    ])
    return report_sections.sections
```

### 3. 执行中断与恢复 (Human-in-the-Loop)

LangGraph 支持在规划执行过程中中断和恢复：

```python
# 中断执行
graph.invoke(inputs, interrupt_before=["node_a"], config=config)

# 恢复执行
graph.invoke(None, config=config)

# 带 Command 恢复
from langgraph.types import Command
graph.invoke(Command(resume=some_resume_value), config=config)
```

### 4. 错误恢复与重试

```python
# 在错误后恢复执行
config = {"configurable": {"thread_id": "some_thread_id"}}
my_workflow.invoke(None, config)
```

### 5. 流式中断处理

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
