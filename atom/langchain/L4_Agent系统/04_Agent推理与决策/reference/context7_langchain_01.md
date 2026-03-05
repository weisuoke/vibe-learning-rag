---
type: context7_documentation
library: LangChain
version: latest (2026)
fetched_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
context7_query: ReAct agent reasoning chain of thought decision making
---

# Context7 文档：LangChain Agent 推理与决策

## 文档来源
- 库名称：LangChain
- 版本：latest (2026)
- Context7 Library ID: /websites/langchain, /websites/langchain_oss_python

## 关键信息提取

### 1. ReAct Agent LCEL 链构建

**标准 ReAct 链结构：**
```python
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)
```

**Prompt 配置：**
```python
prompt = hub.pull("hwchase17/react")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])
```

### 2. ReAct 推理循环描述（官方文档）

> Agents follow the ReAct ("Reasoning + Acting") pattern, alternating between
> brief reasoning steps with targeted tool calls and feeding the resulting
> observations into subsequent decisions until they can deliver a final answer.
> This pattern enables agents to break down complex tasks into manageable steps,
> where each reasoning phase informs which tool to call next, and each tool
> result guides the subsequent reasoning.

### 3. create_agent 与 ReAct（2026 最新）

**SQL Agent 示例：**
> Use `create_agent` to build a ReAct agent with minimal code. The agent will
> interpret the request and generate a SQL command, which the tools will execute.
> If the command has an error, the error message is returned to the model. The
> model can then examine the original request and the new error message and
> generate a new command. This can continue until the LLM generates the command
> successfully or reaches an end count. This pattern of providing a model with
> feedback - error messages in this case - is very powerful.

### 4. JSON 格式 ReAct Prompt

```
Question: the input question you must answer
Thought: you should always think about what to do
Action:
```json
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### 5. Structured Output 与推理（2026 新特性）

**create_agent 结构化输出：**
- 主循环集成：结构化输出在主循环中生成，无需额外 LLM 调用
- 策略选择：模型可选择调用工具或使用 provider-side 结构化输出
- 成本降低：消除额外 LLM 调用的开销
- 错误处理：通过 `handle_errors` 参数控制

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openrouter:openai/gpt-4.1",
    tools=[weather_tool],
    response_format=ProviderStrategy(Weather),
)
```

## 核心概念总结

1. **ReAct 模式**：Reasoning + Acting 交替循环
2. **Scratchpad**：累积推理历史，维持上下文
3. **Stop Sequence**：防止 LLM 幻觉生成 Observation
4. **Output Parser**：从 LLM 输出中提取结构化决策
5. **Self-Correction**：错误反馈机制，让 LLM 自我修正
6. **Structured Output**：2026 新特性，主循环集成结构化输出
