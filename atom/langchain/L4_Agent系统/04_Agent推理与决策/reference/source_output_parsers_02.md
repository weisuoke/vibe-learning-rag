---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/agents/output_parsers/react_single_input.py
  - libs/langchain/langchain_classic/agents/output_parsers/react_json_single_input.py
  - libs/langchain/langchain_classic/agents/output_parsers/tools.py
  - libs/langchain/langchain_classic/agents/output_parsers/xml.py
  - libs/langchain/langchain_classic/agents/output_parsers/mrkl/output_parser.py
analyzed_at: 2026-03-01
knowledge_point: 04_Agent推理与决策
---

# 源码分析：输出解析与决策提取

## 分析的文件

### 1. react_single_input.py - ReAct 单输入解析器

**核心类：ReActSingleInputOutputParser**

```python
class ReActSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*"
            r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)

        if action_match:
            if includes_answer:
                return AgentFinish(
                    {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()},
                    text,
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ").strip('"')
            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()},
                text,
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        ...
```

**关键发现：**
- 使用正则匹配 `Action:` 和 `Action Input:`
- 优先检查 `Final Answer` 关键词
- 解析失败时设置 `send_to_llm=True`，让 LLM 自我修正
- MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE 提供修正提示

### 2. react_json_single_input.py - JSON 格式 ReAct 解析器

```python
class ReActJsonSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style output with JSON action blocks."""

    pattern = re.compile(
        r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$",
        re.DOTALL,
    )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text

        try:
            found = self.pattern.search(text)
            if not found:
                # JSON not in markdown code block
                raise ValueError("action not found")

            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response

            if includes_answer and includes_action:
                ...
            elif includes_answer:
                return AgentFinish(
                    {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()},
                    text,
                )
            elif includes_action:
                return AgentAction(
                    response["action"],
                    response.get("action_input", {}),
                    text,
                )
            ...
```

**关键发现：**
- 从 markdown 代码块中提取 JSON
- JSON 格式：`{"action": "tool_name", "action_input": "input"}`
- 比纯文本解析更结构化，减少解析错误

### 3. tools.py - 工具调用解析器（现代方式）

```python
class ToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish using tool calls."""

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Union[list[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError(...)

        message = result[0].message
        return parse_ai_message_to_tool_action(message)
```

```python
def parse_ai_message_to_tool_action(
    message: BaseMessage,
) -> Union[list[AgentAction], AgentFinish]:
    if isinstance(message, AIMessage) and message.tool_calls:
        actions = []
        for tool_call in message.tool_calls:
            function_name = tool_call["name"]
            _tool_input = tool_call["args"]
            actions.append(
                ToolAgentAction(
                    tool=function_name,
                    tool_input=_tool_input,
                    log=f"\nInvoking: `{function_name}` with `{_tool_input}`\n",
                    message_log=[message],
                    tool_call_id=tool_call["id"],
                )
            )
        return actions

    return AgentFinish(
        return_values={"output": message.content},
        log=str(message.content),
    )
```

**关键发现：**
- 现代方式：直接从 AIMessage.tool_calls 提取
- 支持多工具并行调用（MultiAction）
- 无需正则解析，依赖模型原生函数调用能力
- 没有 tool_calls 时自动返回 AgentFinish

### 4. xml.py - XML 格式解析器

```python
class XMLAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in XML format."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            return AgentAction(
                tool=_tool, tool_input=_tool_input, log=text
            )
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(
                return_values={"output": answer}, log=text
            )
        ...
```

**关键发现：**
- XML 格式适合 Claude 等偏好 XML 的模型
- 使用 `<tool>` 和 `<final_answer>` 标签
- 解析逻辑简单直接

### 5. mrkl/output_parser.py - MRKL 零样本解析器

```python
class MRKLOutputParser(ReActSingleInputOutputParser):
    """MRKL output parser inherits from ReAct parser."""
    pass
```

**关键发现：**
- MRKL 解析器直接继承 ReAct 解析器
- 说明 MRKL 和 ReAct 的输出格式本质相同
- 区别在于 prompt 设计，不在解析逻辑

## 决策提取模式对比

| 解析器 | 格式 | 适用场景 | 优缺点 |
|--------|------|----------|--------|
| ReActSingleInput | 纯文本 | 通用 LLM | 灵活但易出错 |
| ReActJsonSingleInput | JSON 代码块 | 支持 JSON 的 LLM | 更结构化 |
| ToolsAgent | 原生 tool_calls | OpenAI/Anthropic | 最可靠 |
| XMLAgent | XML 标签 | Claude 等 | 适合特定模型 |
| MRKL | 同 ReAct | 零样本场景 | 继承 ReAct |

## 自我修正机制

所有解析器都支持错误反馈：
```python
raise OutputParserException(
    f"Could not parse LLM output: `{text}`",
    observation="Invalid Format: ...",  # 错误提示
    llm_output=text,
    send_to_llm=True,  # 将错误发回 LLM 自我修正
)
```
