---
type: context7_documentation
library: LangGraph
version: latest (2026)
fetched_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
context7_query: create_react_agent prebuilt ReAct agent reasoning loop
---

# Context7 文档：LangGraph ReAct Agent

## 文档来源
- 库名称：LangGraph
- Context7 Library ID: /websites/langchain_oss_python_langgraph

## 关键信息提取

### LangGraph Agent 推理循环

**核心模式：LLM → Tool Calls → Tool Execution → Loop**

```python
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

### 条件路由决策逻辑

```python
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END
```

**决策逻辑：**
- 有 tool_calls → 继续执行工具 → 回到 LLM
- 无 tool_calls → 结束（返回最终答案）

### StateGraph 构建

```python
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()
```

### 与经典 ReAct 的对比

| 特性 | 经典 ReAct (AgentExecutor) | LangGraph ReAct |
|------|---------------------------|-----------------|
| 推理格式 | 文本 Thought/Action | 原生 tool_calls |
| 状态管理 | intermediate_steps | MessagesState |
| 循环控制 | max_iterations | 条件边 + END |
| 并行工具 | 不支持 | 支持 (futures) |
| 可扩展性 | 有限 | 高度可扩展 |
