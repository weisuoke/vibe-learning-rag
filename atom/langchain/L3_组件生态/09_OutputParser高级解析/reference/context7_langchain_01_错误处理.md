---
type: context7_documentation
library: langchain
version: latest (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: OutputParser高级解析
context7_query: output parsers streaming error handling LCEL
---

# Context7 文档：LangChain Output Parsers（错误处理与结构化输出）

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/oss/python/langchain/structured-output
- Context7 库 ID：/websites/langchain
- 总文档片段：26,795
- 信任分数：10/10
- 基准分数：83/100

## 关键信息提取

### 1. Output Parsers 概述

**定义**：
> Output parsers are responsible for taking the output of a model and transforming it to a more suitable format for downstream tasks. They are useful when you are using LLMs to generate structured data, or to normalize output from chat models and LLMs. Output parsers accept a string or BaseMessage as input and can return an arbitrary type.

**关键点**：
- 负责将模型输出转换为更适合下游任务的格式
- 用于生成结构化数据或规范化输出
- 接受 `string` 或 `BaseMessage` 作为输入
- 可以返回任意类型

### 2. 多个结构化输出错误处理

**场景**：LLM 尝试返回多个结构化输出，但只期望一个

**Python 示例**：

```python
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # Default: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})
```

**关键特性**：
- 使用 `ToolStrategy` 定义期望的输出类型
- 支持 `Union` 类型（多个 Pydantic 模型）
- 默认启用错误处理（`handle_errors=True`）
- 自动提供错误反馈并引导模型纠正

### 3. 错误处理工作流程

**TypeScript 示例中的错误处理流程**：

```typescript
const result = await agent.invoke({
    messages: [
        {
        role: "user",
        content: "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th",
        },
    ],
});

/**
 * {
 *   messages: [
 *     { role: "user", content: "Extract info: ..." },
 *     { role: "assistant", content: "", tool_calls: [
 *         { name: "ContactInfo", args: { name: "John Doe", email: "john@email.com" }, id: "call_1" },
 *         { name: "EventDetails", args: { event_name: "Tech Conference", date: "March 15th" }, id: "call_2" }
 *     ] },
 *     { role: "tool", content: "Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.\n Please fix your mistakes.", tool_call_id: "call_1", name: "ContactInfo" },
 *     { role: "tool", content: "Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.\n Please fix your mistakes.", tool_call_id: "call_2", name: "EventDetails" },
 *     { role: "assistant", content: "", tool_calls: [
 *         { name: "ContactInfo", args: { name: "John Doe", email: "john@email.com" }, id: "call_3" }
 *     ] },
 *     { role: "tool", content: "Returning structured response: {'name': 'John Doe', 'email': 'john@email.com'}", tool_call_id: "call_3", name: "ContactInfo" }
 *   ],
 *   structuredResponse: { name: "John Doe", email: "john@email.com" }
 * }
 */
```

**错误处理步骤**：
1. **模型错误**：返回多个工具调用（ContactInfo 和 EventDetails）
2. **错误反馈**：系统自动生成 `ToolMessage` 提示错误
3. **模型重试**：模型根据错误反馈重新生成正确的单个输出
4. **成功返回**：最终返回正确的结构化响应

### 4. 流式工具调用和结构化输出的错误处理

**来源**：https://docs.langchain.com/oss/python/integrations/chat/bedrock

**关键点**：
> When using fine-grained tool streaming, the system may emit invalid or partial JSON inputs during the streaming process. It is important to implement proper error handling and validation in your code to account for these edge cases, ensuring your application can gracefully handle incomplete JSON chunks that may arrive during streaming.

**流式解析的挑战**：
- 可能发出无效或部分 JSON 输入
- 需要实现适当的错误处理和验证
- 必须优雅地处理不完整的 JSON 块

### 5. 异常类型过滤

**来源**：https://docs.langchain.com/oss/javascript/langchain/structured-output

**关键点**：
> Error handling strategies support filtering specific exception types to apply custom handling only to certain errors. By checking the error instance type within the handler function, you can return different messages for different error categories. For example, you can handle `ToolInputParsingException` differently from other error types, providing targeted guidance for validation errors while allowing other errors to propagate with their default messages.

**异常类型过滤示例**：

```python
def handle_error(error):
    if isinstance(error, ToolInputParsingException):
        return "Error: Invalid tool input format. Please check your JSON syntax."
    else:
        return f"Error: {str(error)}"
```

**支持的错误类型**：
- `ToolInputParsingException`：工具输入解析错误
- 其他错误类型：使用默认错误消息

## 与 OutputParser 的关系

### 1. 结构化输出 vs OutputParser

**现代方法（2025+）**：
- 大多数 LLM 支持原生结构化输出（通过 Function Calling）
- 使用 `ToolStrategy` 和 Pydantic 模型定义输出格式
- 自动错误处理和重试机制

**传统方法（OutputParser）**：
- 用于不支持原生结构化输出的模型
- 手动解析和验证输出
- 需要显式错误处理

### 2. 何时使用 OutputParser

根据文档概述：
> Output parsers remain valuable when working with models that do not support structured output natively, or when you require additional processing or validation of the model's output beyond its inherent capabilities.

**适用场景**：
- 模型不支持原生结构化输出
- 需要额外的处理或验证
- 需要自定义解析逻辑

### 3. 错误处理策略对比

| 特性 | 原生结构化输出 | OutputParser |
|------|----------------|--------------|
| 错误检测 | 自动（LLM 内置） | 手动（解析时检测） |
| 错误反馈 | 自动（ToolMessage） | 手动（抛出异常） |
| 重试机制 | 自动（Agent 循环） | 手动（需要实现） |
| 验证 | Pydantic 自动验证 | 需要显式验证 |
| 流式支持 | 部分支持 | 完全支持 |

## 实战建议

### 1. 优先使用原生结构化输出

```python
# 推荐（2025+）
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="gpt-5",
    response_format=ToolStrategy(MyPydanticModel)
)
```

### 2. 回退到 OutputParser

```python
# 当模型不支持原生结构化输出时
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MyPydanticModel)
chain = model | parser
```

### 3. 流式场景的错误处理

```python
# 流式解析时处理部分 JSON
try:
    for chunk in chain.stream(input):
        # 处理部分结果
        process_chunk(chunk)
except OutputParserException as e:
    # 处理解析错误
    handle_error(e)
```

## 总结

Context7 文档主要关注：
1. **现代方法**：使用原生结构化输出（ToolStrategy + Pydantic）
2. **错误处理**：自动错误检测、反馈和重试机制
3. **流式支持**：处理不完整 JSON 块的挑战
4. **异常过滤**：针对不同错误类型的定制处理

**与 OutputParser 的关系**：
- OutputParser 是传统方法，用于不支持原生结构化输出的模型
- 现代 LLM（2025+）优先使用原生结构化输出
- OutputParser 仍然有价值：自定义解析逻辑、额外验证、流式支持

**下一步**：需要查询更多关于 OutputParser 具体类型（JSON、Pydantic、List、XML 等）的文档。
