---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/agents/react/agent.py
  - libs/langchain/langchain_classic/agents/react/output_parser.py
  - libs/langchain/langchain_classic/agents/react/wiki_prompt.py
  - libs/langchain/langchain_classic/agents/format_scratchpad/log.py
  - libs/langchain/langchain_classic/agents/format_scratchpad/log_to_messages.py
analyzed_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
---

# 源码分析：ReAct 推理模式实现

## 分析的文件

### 1. react/agent.py - create_react_agent 工厂函数

**核心签名：**
```python
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: AgentOutputParser | None = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: bool | list[str] = True,
) -> Runnable
```

**关键实现逻辑：**
- 使用 `prompt.partial()` 注入 tools 描述和 tool_names
- 绑定 stop sequence（默认 `["\nObservation"]`）防止 LLM 幻觉
- 构建 LCEL 链：`RunnablePassthrough → Prompt → LLM(with stop) → OutputParser`
- 默认使用 `ReActSingleInputOutputParser`

**LCEL 链结构：**
```python
agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm_with_stop
    | output_parser
)
```

### 2. react/output_parser.py - ReAct 输出解析器

**ReActOutputParser 核心逻辑：**
```python
class ReActOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            raise OutputParserException(f"Could not parse LLM Output: {text}")
        action_block = text.strip().split("\n")[-1]
        action_str = action_block[len(action_prefix):]

        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise OutputParserException(...)

        action, action_input = re_matches.group(1), re_matches.group(2)
        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)
```

**关键发现：**
- 使用正则 `(.*?)\[(.*?)\]` 提取 Action 和参数
- `Finish[answer]` 触发 AgentFinish（停止循环）
- 其他 Action 触发 AgentAction（继续循环）

### 3. react/wiki_prompt.py - ReAct 推理示例

**6个完整推理示例，展示多步推理模式：**

示例1（多步搜索+查找）：
```
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building...
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains.
Thought 3: High Plains rise in elevation from around 1,800 to 7,000 ft.
Action 3: Finish[1,800 to 7,000 ft]
```

**推理模式总结：**
- Thought → Action → Observation 循环
- 支持 Search（搜索）和 Lookup（查找）两种动作
- 支持多步推理（最多6步）
- Finish 动作结束推理

### 4. format_scratchpad/log.py - 思维链格式化

```python
def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
```

**关键发现：**
- 将 intermediate_steps 格式化为字符串形式的"思维链"
- 每步包含：action.log（包含Thought和Action）+ Observation
- 末尾添加 `Thought: ` 前缀，引导 LLM 继续推理

### 5. format_scratchpad/log_to_messages.py - 消息格式化

```python
def format_log_to_messages(
    intermediate_steps: List[Tuple[AgentAction, str]],
    template_tool_response: str = "{observation}",
) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        messages.append(AIMessage(content=action.log))
        messages.append(HumanMessage(
            content=template_tool_response.format(observation=observation)
        ))
    return messages
```

**关键发现：**
- 将推理步骤转换为消息列表（适用于 ChatModel）
- AI 消息包含推理和动作
- Human 消息包含观察结果
- 这种格式更适合现代 Chat API

## 核心架构总结

```
Input → Prompt(tools + scratchpad) → LLM(stop=Observation) → OutputParser
                                                                    ↓
                                                          AgentAction / AgentFinish
                                                                    ↓
                                                          Tool Execution → Observation
                                                                    ↓
                                                          Append to scratchpad
                                                                    ↓
                                                          Loop back to Prompt
```
