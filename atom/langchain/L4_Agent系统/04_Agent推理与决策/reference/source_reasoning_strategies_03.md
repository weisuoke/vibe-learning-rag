---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/agents/mrkl/prompt.py
  - libs/langchain/langchain_classic/agents/mrkl/base.py
  - libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
  - libs/langchain/langchain_classic/agents/format_scratchpad/tools.py
  - libs/langchain/langchain_classic/agents/format_scratchpad/xml.py
analyzed_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
---

# 源码分析：推理提示工程与策略对比

## 分析的文件

### 1. mrkl/prompt.py - MRKL 推理提示模板

**标准 MRKL 格式指令：**
```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

**关键设计要素：**
- 明确的格式指令（Format Instructions）
- 工具列表注入 `{tools}`
- 思维链引导 `Thought:` 前缀
- 循环模式说明 `can repeat N times`
- 终止信号 `Final Answer`

### 2. mrkl/base.py - ZeroShotAgent（MRKL 实现）

```python
class ZeroShotAgent(Agent):
    """Agent for the MRKL chain."""

    @property
    def _agent_type(self) -> str:
        return "zero-shot-react-description"

    @property
    def observation_prefix(self) -> str:
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        return "Thought: "

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        tool_strings = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        ...
```

**关键发现：**
- "zero-shot-react-description" = 零样本 + ReAct + 工具描述
- 工具描述格式：`tool_name: tool_description`
- Prompt 由 4 部分拼接：prefix + tools + format_instructions + suffix
- `llm_prefix = "Thought: "` 引导 LLM 开始推理

### 3. tool_calling_agent/base.py - 现代工具调用 Agent

```python
def create_tool_calling_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    *,
    message_formatter: _MessageFormatter = format_to_tool_messages,
) -> Runnable:
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: message_formatter(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent
```

**关键发现：**
- 使用 `llm.bind_tools(tools)` 绑定工具（原生函数调用）
- 不需要文本格式的推理指令
- 使用 `ToolsAgentOutputParser` 直接从 tool_calls 提取
- scratchpad 使用消息格式（不是字符串）

### 4. format_scratchpad/tools.py - 工具调用格式化

```python
def format_to_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = list(agent_action.message_log) + [
                _create_tool_message(agent_action, observation)
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
            messages.append(
                _get_tool_message(observation, agent_action)
            )
    return messages
```

**关键发现：**
- 工具调用模式使用 ToolMessage（不是 HumanMessage）
- 保留原始 AIMessage（包含 tool_calls）
- 每个工具调用对应一个 ToolMessage 响应

### 5. format_scratchpad/xml.py - XML 格式化

```python
def format_xml(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> str:
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"{action.log}<observation>{observation}</observation>"
        )
    return log
```

**关键发现：**
- XML 格式最简洁
- 使用 `<observation>` 标签包裹观察结果
- 适合 Claude 等偏好 XML 的模型

## 三种推理策略对比

### 策略1：ReAct 文本推理（经典）
```
Thought: I need to search for...
Action: search
Action Input: "query"
Observation: result...
Thought: Now I know...
Final Answer: answer
```
- **优点**：显式推理过程，可解释性强
- **缺点**：依赖文本解析，容易出错
- **适用**：需要透明推理的场景

### 策略2：工具调用（现代）
```python
# LLM 直接输出结构化 tool_calls
AIMessage(
    content="",
    tool_calls=[{"name": "search", "args": {"query": "..."}}]
)
```
- **优点**：结构化输出，解析可靠
- **缺点**：推理过程隐式（在 content 中）
- **适用**：生产环境，需要可靠性

### 策略3：XML 推理
```xml
<tool>search</tool>
<tool_input>query</tool_input>
```
- **优点**：结构清晰，适合特定模型
- **缺点**：不如原生 tool_calls 可靠
- **适用**：Claude 等偏好 XML 的模型

## 推理策略选型决策树

```
需要显式推理过程？
├── 是 → 模型支持原生函数调用？
│   ├── 是 → 工具调用 + system prompt 要求 CoT
│   └── 否 → ReAct 文本推理
└── 否 → 模型支持原生函数调用？
    ├── 是 → 工具调用（推荐）
    └── 否 → 模型偏好 XML？
        ├── 是 → XML 推理
        └── 否 → ReAct JSON 推理
```
