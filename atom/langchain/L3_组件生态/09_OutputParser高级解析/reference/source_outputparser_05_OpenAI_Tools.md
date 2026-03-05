---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/output_parsers
analyzed_files:
  - openai_tools.py
analyzed_at: 2026-02-26
knowledge_point: OutputParser高级解析
---

# 源码分析：OpenAI Tools 集成（JsonOutputToolsParser & PydanticToolsParser）

## 分析的文件

- `langchain_core/output_parsers/openai_tools.py` - OpenAI Tools 解析器实现

## 关键发现

### 1. OpenAI Tools 背景

**什么是 OpenAI Tools？**
- OpenAI 的 Function Calling 功能
- 允许 LLM 调用预定义的工具/函数
- 返回结构化的工具调用信息

**工具调用格式**：
```json
{
  "id": "call_abc123",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"San Francisco\"}"
  }
}
```

### 2. parse_tool_call() - 单个工具调用解析

**函数签名**（openai_tools.py:27-78）：

```python
def parse_tool_call(
    raw_tool_call: dict[str, Any],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> dict[str, Any] | None:
    """Parse a single tool call.

    Args:
        raw_tool_call: The raw tool call to parse.
        partial: Whether to parse partial JSON.
        strict: Whether to allow non-JSON-compliant strings.
        return_id: Whether to return the tool call id.

    Returns:
        The parsed tool call.

    Raises:
        OutputParserException: If the tool call is not valid JSON.
    """
```

**关键逻辑**：

1. **检查格式**：
```python
if "function" not in raw_tool_call:
    return None
```

2. **提取参数**：
```python
arguments = raw_tool_call["function"]["arguments"]
```

3. **解析参数（支持部分解析）**：
```python
if partial:
    try:
        function_args = parse_partial_json(arguments, strict=strict)
    except (JSONDecodeError, TypeError):
        return None
# Handle None or empty string arguments for parameter-less tools
elif not arguments:
    function_args = {}
else:
    try:
        function_args = json.loads(arguments, strict=strict)
    except JSONDecodeError as e:
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{arguments}\n\nare not valid JSON. "
            f"Received JSONDecodeError {e}"
        )
        raise OutputParserException(msg) from e
```

4. **构造返回值**：
```python
parsed = {
    "name": raw_tool_call["function"]["name"] or "",
    "args": function_args or {},
}
if return_id:
    parsed["id"] = raw_tool_call.get("id")
    parsed = create_tool_call(**parsed)
return parsed
```

**关键特性**：
- 支持部分解析（`partial=True`）
- 支持无参数工具（`arguments` 为空）
- 支持 `strict` 模式（JSON 严格解析）
- 可选返回工具调用 ID

### 3. make_invalid_tool_call() - 创建无效工具调用

**函数签名**（openai_tools.py:81-99）：

```python
def make_invalid_tool_call(
    raw_tool_call: dict[str, Any],
    error_msg: str | None,
) -> InvalidToolCall:
    """Create an `InvalidToolCall` from a raw tool call.

    Args:
        raw_tool_call: The raw tool call.
        error_msg: The error message.

    Returns:
        An `InvalidToolCall` instance with the error message.
    """
    return invalid_tool_call(
        name=raw_tool_call["function"]["name"],
        args=raw_tool_call["function"]["arguments"],
        id=raw_tool_call.get("id"),
        error=error_msg,
    )
```

**用途**：
- 当工具调用解析失败时，创建 `InvalidToolCall` 对象
- 保留原始信息和错误消息
- 用于错误处理和调试

### 4. parse_tool_calls() - 批量工具调用解析

**函数签名**（openai_tools.py:102-137）：

```python
def parse_tool_calls(
    raw_tool_calls: list[dict],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> list[dict[str, Any]]:
    """Parse a list of tool calls.

    Args:
        raw_tool_calls: The raw tool calls to parse.
        partial: Whether to parse partial JSON.
        strict: Whether to allow non-JSON-compliant strings.
        return_id: Whether to return the tool call id.

    Returns:
        The parsed tool calls.

    Raises:
        OutputParserException: If any of the tool calls are not valid JSON.
    """
    final_tools: list[dict[str, Any]] = []
    exceptions = []
    for tool_call in raw_tool_calls:
        try:
            parsed = parse_tool_call(
                tool_call, partial=partial, strict=strict, return_id=return_id
            )
            if parsed:
                final_tools.append(parsed)
        except OutputParserException as e:
            exceptions.append(str(e))
            continue
    if exceptions:
        raise OutputParserException("\n\n".join(exceptions))
    return final_tools
```

**关键特性**：
- 批量解析多个工具调用
- 收集所有异常并一起抛出
- 跳过解析失败的工具调用（继续处理其他）

### 5. JsonOutputToolsParser - JSON 工具解析器

**类定义**（openai_tools.py:140-150）：

```python
class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse tools from OpenAI response."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    """
```

**关键特性**：
- 继承自 `BaseCumulativeTransformOutputParser`（支持流式解析）
- `strict` 参数控制 JSON 解析严格性
- 用于解析 OpenAI 的工具调用响应

**从源码中推断的实现**（未完全显示）：
- 从 `AIMessage` 中提取 `tool_calls`
- 使用 `parse_tool_calls()` 解析
- 支持流式解析（累积模式）

### 6. 工具调用数据结构

**ToolCall 对象**（从 `create_tool_call` 推断）：

```python
{
    "name": str,      # 工具名称
    "args": dict,     # 工具参数（已解析的 JSON）
    "id": str,        # 工具调用 ID（可选）
}
```

**InvalidToolCall 对象**（从 `invalid_tool_call` 推断）：

```python
{
    "name": str,      # 工具名称
    "args": str,      # 原始参数字符串（未解析）
    "id": str,        # 工具调用 ID（可选）
    "error": str,     # 错误消息
}
```

### 7. 与 AIMessage 的集成

**AIMessage 结构**（从导入推断）：

```python
from langchain_core.messages import AIMessage, InvalidToolCall

class AIMessage:
    tool_calls: list[ToolCall]           # 有效的工具调用
    invalid_tool_calls: list[InvalidToolCall]  # 无效的工具调用
```

**解析流程**：
1. LLM 返回 `AIMessage`
2. `AIMessage` 包含 `tool_calls` 列表
3. `JsonOutputToolsParser` 解析 `tool_calls`
4. 返回解析后的工具调用列表

### 8. 部分解析支持

**流式场景**：

```python
# 输入流（逐块接收）
chunk1: '{"function": {"name": "get_weather", "arguments": "{"'
chunk2: '"location": "San Francisco"'
chunk3: '}"}'

# 部分解析
parse_tool_call(chunk1, partial=True)  # 返回 None（不完整）
parse_tool_call(chunk1 + chunk2, partial=True)  # 返回部分结果
parse_tool_call(chunk1 + chunk2 + chunk3, partial=True)  # 返回完整结果
```

**关键函数**：
```python
from langchain_core.utils.json import parse_partial_json
```

### 9. strict 参数的作用

**strict=False（默认）**：
- 允许 Unicode 字符
- 允许换行符
- 更宽松的 JSON 解析

**strict=True**：
- 严格的 JSON 解析
- 不允许非标准字符
- 符合 JSON 规范

**示例**：
```python
# strict=False 可以解析
arguments = '{"text": "Hello\nWorld"}'  # 包含换行符

# strict=True 会失败
arguments = '{"text": "Hello\nWorld"}'  # 抛出 JSONDecodeError
```

### 10. PydanticToolsParser（从 __init__.py 推断）

**预期功能**：
- 继承自 `JsonOutputToolsParser`
- 使用 Pydantic 模型验证工具参数
- 类似于 `PydanticOutputParser` 的验证逻辑

**预期用法**：
```python
from pydantic import BaseModel

class WeatherArgs(BaseModel):
    location: str
    unit: str = "celsius"

parser = PydanticToolsParser(pydantic_object=WeatherArgs)
```

### 11. JsonOutputKeyToolsParser（从 __init__.py 推断）

**预期功能**：
- 从工具调用中提取特定键的值
- 用于简化工具调用结果的访问

**预期用法**：
```python
parser = JsonOutputKeyToolsParser(key="location")
# 从 {"name": "get_weather", "args": {"location": "SF"}} 中提取 "SF"
```

## 使用示例

### 基础工具调用解析

```python
from langchain_core.output_parsers.openai_tools import parse_tool_call

raw_tool_call = {
    "id": "call_abc123",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco", "unit": "celsius"}'
    }
}

parsed = parse_tool_call(raw_tool_call)
# 输出: {
#     "id": "call_abc123",
#     "name": "get_weather",
#     "args": {"location": "San Francisco", "unit": "celsius"}
# }
```

### 批量工具调用解析

```python
from langchain_core.output_parsers.openai_tools import parse_tool_calls

raw_tool_calls = [
    {
        "id": "call_1",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "SF"}'
        }
    },
    {
        "id": "call_2",
        "function": {
            "name": "get_time",
            "arguments": '{"timezone": "PST"}'
        }
    }
]

parsed = parse_tool_calls(raw_tool_calls)
# 输出: [
#     {"id": "call_1", "name": "get_weather", "args": {"location": "SF"}},
#     {"id": "call_2", "name": "get_time", "args": {"timezone": "PST"}}
# ]
```

### JsonOutputToolsParser 使用

```python
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
parser = JsonOutputToolsParser()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# 绑定工具到模型
model_with_tools = model.bind_tools(tools)

# 调用
chain = model_with_tools | parser
result = chain.invoke("What's the weather in SF?")
# 输出: [{"name": "get_weather", "args": {"location": "San Francisco"}}]
```

### 流式工具调用解析

```python
# 流式解析工具调用
for chunk in chain.stream("What's the weather in SF and LA?"):
    print(chunk)
    # 输出: [{"name": "get_weather", "args": {"location": "San Francisco"}}]
    # 输出: [{"name": "get_weather", "args": {"location": "Los Angeles"}}]
```

## 架构设计亮点

1. **部分解析支持**：支持流式场景下的部分工具调用解析
2. **错误容忍**：批量解析时收集所有错误，不会因单个失败而中断
3. **灵活配置**：`strict` 参数控制 JSON 解析严格性
4. **无参数工具**：正确处理无参数的工具调用
5. **类型安全**：使用 Pydantic 验证工具参数（PydanticToolsParser）

## 与 Function Calling 的集成

**完整流程**：

1. **定义工具**：
```python
tools = [{"type": "function", "function": {...}}]
```

2. **绑定工具到模型**：
```python
model_with_tools = model.bind_tools(tools)
```

3. **调用模型**：
```python
response = model_with_tools.invoke("user query")
# response 是 AIMessage，包含 tool_calls
```

4. **解析工具调用**：
```python
parser = JsonOutputToolsParser()
parsed_tools = parser.invoke(response)
```

5. **执行工具**：
```python
for tool_call in parsed_tools:
    result = execute_tool(tool_call["name"], tool_call["args"])
```

## 性能考虑

1. **JSON 解析开销**：每个工具调用都需要解析 JSON
2. **部分解析开销**：`parse_partial_json()` 比完整解析慢
3. **批量处理**：批量解析比逐个解析更高效

## 总结

OpenAI Tools 解析器通过以下设计实现了高效的工具调用解析：
- **部分解析**：支持流式场景
- **错误容忍**：批量解析时收集所有错误
- **灵活配置**：`strict` 参数控制解析严格性
- **类型安全**：Pydantic 验证工具参数
- **无缝集成**：与 LangChain 的 Function Calling 无缝集成
