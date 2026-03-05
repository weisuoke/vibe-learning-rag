---
type: context7_documentation
library: langchain
version: v1.0 (2025-2026)
fetched_at: 2026-02-28
knowledge_point: 02_AgentExecutor执行循环
context7_query: AgentExecutor execution loop how agents work tool calling cycle ReAct pattern
---

# Context7 文档：LangChain Agent 执行循环

## 文档来源
- 库名称：LangChain
- 版本：v1.0 (2025-2026)
- 官方文档链接：https://docs.langchain.com/oss/python/langchain/agents

## 关键信息提取

### 1. ReAct 循环模式

Agent 遵循 ReAct（Reasoning + Acting）模式：
- 交替进行推理步骤和工具调用
- 每个推理阶段决定下一步调用哪个工具
- 每个工具结果指导后续推理
- 直到能给出最终答案

### 2. Agent 循环的两个主要步骤

典型的 agent 循环包含两个主要步骤：
1. **Model call** - 调用 LLM，传入 prompt 和可用工具，返回响应或工具调用请求
2. **Tool execution** - 执行 LLM 请求的工具，返回工具结果

循环持续直到 LLM 决定结束（不再请求工具调用）。

### 3. 现代 Agent 循环（LangGraph 风格）

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

### 4. ReAct 循环示例流程

```
Human: Find headphones and check stock
  ↓
AI: [Tool Call: search_products("wireless headphones")]
  ↓
Tool: Found 5 products...
  ↓
AI: [Tool Call: check_inventory("WH-1000XM5")]
  ↓
Tool: 10 units in stock
  ↓
AI: I found wireless headphones with 10 units in stock...
```

### 5. Context Engineering in Agents

Agent 循环中的上下文工程：
- 每次迭代都将之前的工具结果添加到消息历史
- LLM 基于完整的对话历史做出决策
- 中间步骤（intermediate_steps）记录完整的推理轨迹
