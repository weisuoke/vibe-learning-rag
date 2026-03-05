# 实战代码 - 场景6：OpenAI Tools 集成

> **数据来源**：
> - reference/source_outputparser_05_OpenAI_Tools.md（源码分析）

---

## 场景概述

OpenAI Tools（Function Calling）允许 LLM 调用预定义的工具/函数。LangChain 提供了专门的解析器来处理工具调用响应：
- `JsonOutputToolsParser`：解析工具调用为 JSON 格式
- `JsonOutputKeyToolsParser`：提取工具调用中的特定键
- `PydanticToolsParser`：使用 Pydantic 模型验证工具参数

**适用场景**：
- Agent 系统中的工具调用
- 多步推理任务
- 结构化数据提取
- API 调用编排

---

## 示例1：基础工具调用 - JsonOutputToolsParser

**场景**：定义工具并解析 LLM 的工具调用响应

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate

# 初始化模型和解析器
model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 绑定工具到模型
model_with_tools = model.bind_tools(tools)

# 创建链
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{query}")
])

chain = prompt | model_with_tools | parser

# 执行
result = chain.invoke({"query": "What's the weather in San Francisco?"})

print(result)
# 输出: [
#     {
#         "name": "get_weather",
#         "args": {"location": "San Francisco", "unit": "celsius"},
#         "id": "call_abc123"
#     }
# ]

# 访问工具调用信息
if result:
    tool_call = result[0]
    print(f"Tool: {tool_call['name']}")
    print(f"Arguments: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
```

**关键点**：
- `bind_tools()` 将工具绑定到模型
- `JsonOutputToolsParser` 自动解析工具调用
- 返回列表，因为 LLM 可能调用多个工具
- 每个工具调用包含 `name`、`args`、`id`

---

## 示例2：多工具调用 - 并行工具执行

**场景**：LLM 同时调用多个工具

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputToolsParser

model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser()

# 定义多个工具
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search for restaurants",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "cuisine": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)
chain = model_with_tools | parser

# 复杂查询可能触发多个工具调用
result = chain.invoke(
    "What's the weather and time in San Francisco? Also find Italian restaurants there."
)

print(f"Total tool calls: {len(result)}")
for i, tool_call in enumerate(result, 1):
    print(f"\nTool Call {i}:")
    print(f"  Name: {tool_call['name']}")
    print(f"  Args: {tool_call['args']}")

# 输出示例:
# Total tool calls: 3
#
# Tool Call 1:
#   Name: get_weather
#   Args: {'location': 'San Francisco'}
#
# Tool Call 2:
#   Name: get_time
#   Args: {'timezone': 'America/Los_Angeles'}
#
# Tool Call 3:
#   Name: search_restaurants
#   Args: {'location': 'San Francisco', 'cuisine': 'Italian'}
```

**关键点**：
- LLM 可以并行调用多个工具
- 解析器返回所有工具调用的列表
- 每个工具调用独立处理

---

## 示例3：流式工具调用解析

**场景**：实时解析流式输出中的工具调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputToolsParser

model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
parser = JsonOutputToolsParser()

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "operation": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)
chain = model_with_tools | parser

# 流式输出
print("Streaming tool calls:")
for chunk in chain.stream("Calculate 123 * 456 and 789 + 321"):
    print(chunk)
    # 输出: 逐步接收到的工具调用
    # [{'name': 'calculate', 'args': {'expression': '123 * 456'}}]
    # [{'name': 'calculate', 'args': {'expression': '789 + 321'}}]

# 完整输出
result = chain.invoke("Calculate 123 * 456")
print("\nFinal result:", result)
```

**关键点**：
- `JsonOutputToolsParser` 继承自 `BaseCumulativeTransformOutputParser`
- 支持流式解析（累积模式）
- 每个 chunk 包含当前已解析的工具调用

---

## 示例4：PydanticToolsParser - 类型验证

**场景**：使用 Pydantic 模型验证工具参数

```python
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticToolsParser

# 定义 Pydantic 模型
class WeatherQuery(BaseModel):
    """Weather query parameters"""
    location: str = Field(description="City name")
    unit: Optional[str] = Field(
        default="celsius",
        description="Temperature unit (celsius or fahrenheit)"
    )

class TimeQuery(BaseModel):
    """Time query parameters"""
    timezone: str = Field(description="Timezone name")

# 初始化
model = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticToolsParser(tools=[WeatherQuery, TimeQuery])

# 定义工具（使用 Pydantic 模型）
tools = [
    {
        "type": "function",
        "function": {
            "name": "WeatherQuery",
            "description": "Get weather information",
            "parameters": WeatherQuery.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TimeQuery",
            "description": "Get current time",
            "parameters": TimeQuery.model_json_schema()
        }
    }
]

model_with_tools = model.bind_tools(tools)
chain = model_with_tools | parser

# 执行
result = chain.invoke("What's the weather in Tokyo?")

print(result)
# 输出: [WeatherQuery(location='Tokyo', unit='celsius')]

# 类型安全访问
if result:
    weather_query = result[0]
    print(f"Location: {weather_query.location}")
    print(f"Unit: {weather_query.unit}")
    print(f"Type: {type(weather_query)}")  # <class 'WeatherQuery'>
```

**关键点**：
- `PydanticToolsParser` 返回 Pydantic 模型实例
- 自动类型验证和转换
- 类型安全的属性访问
- 支持默认值和可选字段

---

## 示例5：JsonOutputKeyToolsParser - 提取特定键

**场景**：只提取工具调用中的特定参数

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputKeyToolsParser

model = ChatOpenAI(model="gpt-4o-mini")

# 只提取 "location" 参数
parser = JsonOutputKeyToolsParser(key_name="location")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)
chain = model_with_tools | parser

result = chain.invoke("What's the weather in Paris and London?")

print(result)
# 输出: ['Paris', 'London']  # 只返回 location 值

# 对比：使用 JsonOutputToolsParser
from langchain_core.output_parsers import JsonOutputToolsParser
full_parser = JsonOutputToolsParser()
full_chain = model_with_tools | full_parser
full_result = full_chain.invoke("What's the weather in Paris?")

print(full_result)
# 输出: [{'name': 'get_weather', 'args': {'location': 'Paris', 'unit': 'celsius'}, 'id': '...'}]
```

**关键点**：
- `JsonOutputKeyToolsParser` 简化输出
- 只返回指定键的值
- 适用于只关心特定参数的场景

---

## 示例6：工具执行循环 - Agent 模式

**场景**：完整的工具调用 → 执行 → 返回结果循环

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 模拟工具执行函数
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """执行工具并返回结果"""
    if tool_name == "get_weather":
        location = tool_args.get("location")
        return f"Weather in {location}: Sunny, 22°C"
    elif tool_name == "get_time":
        timezone = tool_args.get("timezone")
        return f"Current time in {timezone}: 14:30"
    else:
        return "Tool not found"

# 初始化
model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string"}
                },
                "required": ["timezone"]
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)

# Agent 循环
def run_agent(query: str, max_iterations: int = 5):
    """运行 Agent 循环"""
    messages = [HumanMessage(content=query)]

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        # 1. LLM 生成响应（可能包含工具调用）
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # 2. 解析工具调用
        tool_calls = parser.invoke(response)

        if not tool_calls:
            # 没有工具调用，返回最终答案
            print("Final answer:", response.content)
            return response.content

        # 3. 执行工具
        for tool_call in tool_calls:
            print(f"Calling tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

            # 执行工具
            result = execute_tool(tool_call['name'], tool_call['args'])
            print(f"Result: {result}")

            # 4. 将工具结果添加到消息历史
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call['id']
                )
            )

    return "Max iterations reached"

# 运行
final_answer = run_agent("What's the weather in Tokyo and the time in PST?")

# 输出示例:
# --- Iteration 1 ---
# Calling tool: get_weather
# Arguments: {'location': 'Tokyo'}
# Result: Weather in Tokyo: Sunny, 22°C
# Calling tool: get_time
# Arguments: {'timezone': 'America/Los_Angeles'}
# Result: Current time in America/Los_Angeles: 14:30
#
# --- Iteration 2 ---
# Final answer: The weather in Tokyo is sunny with 22°C. The current time in PST is 14:30.
```

**关键点**：
- 完整的 Agent 循环实现
- 工具调用 → 执行 → 结果反馈
- 使用 `ToolMessage` 传递工具结果
- 支持多轮迭代直到获得最终答案

---

## 示例7：错误处理 - 无效工具调用

**场景**：处理 LLM 生成的无效工具调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.exceptions import OutputParserException

model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(strict=False)  # 宽松模式

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)

def safe_parse_tools(response):
    """安全解析工具调用"""
    try:
        tool_calls = parser.invoke(response)

        # 检查是否有无效的工具调用
        if hasattr(response, 'invalid_tool_calls'):
            invalid_calls = response.invalid_tool_calls
            if invalid_calls:
                print(f"Warning: {len(invalid_calls)} invalid tool calls")
                for invalid in invalid_calls:
                    print(f"  - {invalid.name}: {invalid.error}")

        return tool_calls

    except OutputParserException as e:
        print(f"Parsing error: {e}")
        return []

# 测试
response = model_with_tools.invoke("Calculate something complex")
tool_calls = safe_parse_tools(response)

if tool_calls:
    print("Valid tool calls:", tool_calls)
else:
    print("No valid tool calls")
```

**关键点**：
- `strict=False` 允许宽松的 JSON 解析
- 检查 `invalid_tool_calls` 属性
- 使用 try-except 捕获解析错误
- 提供降级处理策略

---

## 常见错误与解决方案

### 错误1：工具未绑定到模型

```python
# ❌ 错误
model = ChatOpenAI()
chain = model | JsonOutputToolsParser()
result = chain.invoke("Get weather")  # 不会调用工具

# ✅ 正确
model_with_tools = model.bind_tools(tools)
chain = model_with_tools | JsonOutputToolsParser()
```

### 错误2：忘记处理空工具调用

```python
# ❌ 错误
result = chain.invoke("Hello")
tool_call = result[0]  # IndexError: list index out of range

# ✅ 正确
result = chain.invoke("Hello")
if result:
    tool_call = result[0]
else:
    print("No tool calls")
```

### 错误3：Pydantic 模型名称不匹配

```python
# ❌ 错误
class WeatherQuery(BaseModel):
    location: str

tools = [{
    "function": {
        "name": "get_weather",  # 名称不匹配
        "parameters": WeatherQuery.model_json_schema()
    }
}]

# ✅ 正确
tools = [{
    "function": {
        "name": "WeatherQuery",  # 与类名一致
        "parameters": WeatherQuery.model_json_schema()
    }
}]
```

---

## 最佳实践

### 1. 工具定义规范

```python
# 清晰的工具描述
tools = [
    {
        "type": "function",
        "function": {
            "name": "tool_name",  # 简洁的名称
            "description": "Clear description of what the tool does",  # 详细描述
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "What this parameter is for"  # 参数说明
                    }
                },
                "required": ["param1"]  # 明确必需参数
            }
        }
    }
]
```

### 2. 选择合适的解析器

```python
# 简单场景：JsonOutputToolsParser
parser = JsonOutputToolsParser()

# 需要类型验证：PydanticToolsParser
parser = PydanticToolsParser(tools=[MyModel])

# 只需特定字段：JsonOutputKeyToolsParser
parser = JsonOutputKeyToolsParser(key_name="location")
```

### 3. 错误处理

```python
def safe_tool_execution(tool_calls):
    """安全执行工具"""
    results = []
    for tool_call in tool_calls:
        try:
            result = execute_tool(tool_call['name'], tool_call['args'])
            results.append(result)
        except Exception as e:
            results.append(f"Error: {e}")
    return results
```

### 4. 日志记录

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_tool_call(tool_call):
    """记录工具调用"""
    logger.info(f"Tool: {tool_call['name']}")
    logger.info(f"Args: {tool_call['args']}")
    logger.info(f"ID: {tool_call['id']}")
```

---

## 性能优化

### 1. 批量工具调用

```python
# 允许 LLM 并行调用多个工具
model_with_tools = model.bind_tools(tools, parallel_tool_calls=True)
```

### 2. 缓存工具定义

```python
# 避免重复创建工具定义
TOOLS = [...]  # 全局常量

def create_chain():
    return model.bind_tools(TOOLS) | parser
```

### 3. 流式处理

```python
# 使用流式处理提高响应速度
for chunk in chain.stream(query):
    process_chunk(chunk)
```

---

## 何时使用 OpenAI Tools 解析器

**适用场景**：
- Agent 系统（工具调用是核心）
- 多步推理任务
- 需要结构化输出的场景
- API 调用编排

**不适用场景**：
- 简单的文本生成（使用 `StrOutputParser`）
- 不需要工具调用的场景
- 模型不支持 Function Calling

---

## 与其他解析器的对比

| 解析器 | 用途 | 返回类型 | 验证 |
|--------|------|----------|------|
| JsonOutputToolsParser | 工具调用解析 | List[dict] | 无 |
| PydanticToolsParser | 工具调用 + 验证 | List[BaseModel] | Pydantic |
| JsonOutputKeyToolsParser | 提取特定键 | List[Any] | 无 |
| JsonOutputParser | 通用 JSON | dict | 无 |
| PydanticOutputParser | 通用验证 | BaseModel | Pydantic |

---

**数据来源总结**：
- 工具调用机制：源码分析（source_outputparser_05_OpenAI_Tools.md）
- 解析器实现：parse_tool_call(), parse_tool_calls() 函数
- 流式支持：BaseCumulativeTransformOutputParser 继承
- 错误处理：InvalidToolCall 和 OutputParserException
