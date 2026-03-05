---
type: context7_documentation
library: langchain
version: latest (2026-02-26)
fetched_at: 2026-03-02
knowledge_point: 自定义Tool开发
context7_query: custom tools @tool decorator BaseTool StructuredTool
---

# Context7 文档：LangChain 自定义工具开发

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-26)
- 官方文档链接：https://docs.langchain.com/oss/python/langchain/tools
- Context7 Library ID: /websites/langchain

## 关键信息提取

### 1. @tool 装饰器的多种使用方式

#### 基础用法（无参数装饰器）
```python
from langchain.tools import tool

@tool
def get_supercopa_trophies_count(club_name: str) -> int | None:
    """Returns information about supercopa trophies count.

    Args:
        club_name: Club you want to investigate info of supercopa trophies about

    Returns:
        Number of supercopa trophies or None if there is no info about requested club
    """
    if club_name == "Barcelona":
        return 15
    elif club_name == "Real Madrid":
        return 13
    elif club_name == "Atletico Madrid":
        return 2
    else:
        return None
```

#### 自定义描述
```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

**关键点**：
- 自定义描述帮助 LLM 更好理解工具用途
- 描述应该清晰说明何时调用该工具

#### 使用 Pydantic Schema
```python
from pydantic import BaseModel, Field

class MagicFunctionInput(BaseModel):
    magic_function_input: int = Field(description="The input value for magic function")

@tool("get_magic_function", args_schema=MagicFunctionInput)
def magic_function(magic_function_input: int):
    """Get the value of magic function for an input."""
    return magic_function_input + 2
```

### 2. 高级 Schema 定义

#### 使用 Pydantic 定义复杂输入
```python
from typing import Literal
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

**设计要点**：
- 使用 `Field()` 提供详细描述
- 使用 `Literal` 限制可选值
- 提供默认值提升易用性

#### 使用 JSON Schema（字典格式）
```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    # ... implementation
```

### 3. StructuredTool.from_function() 用法

#### 集成外部 API 包装器
```python
from langchain.tools import StructuredTool
from langchain_community.utilities.infobip import InfobipAPIWrapper
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    body: str = Field(description="Email body text")
    to: str = Field(description="Email address to send to. Example: email@example.com")
    sender: str = Field(
        description="Email address to send from, must be 'validemail@example.com'"
    )
    subject: str = Field(description="Email subject")
    channel: str = Field(description="Email channel, must be 'email'")

infobip_api_wrapper: InfobipAPIWrapper = InfobipAPIWrapper()
infobip_tool = StructuredTool.from_function(
    name="infobip_email",
    description="Send Email via Infobip. If you need to send email, use infobip_email",
    func=infobip_api_wrapper.run,
    args_schema=EmailInput,
)
```

**适用场景**：
- 包装现有的 API 客户端
- 需要显式控制工具名称和描述
- 复杂的输入验证需求

### 4. 工具与 Agent 集成

#### 创建 Agent 并绑定工具
```python
from langchain_classic import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

instructions = "You are a coding teacher."
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = ChatOpenAI(temperature=0)

tools = [infobip_tool]

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({
    "input": "Hi, can you please send me an example of Python recursion to my email email@example.com"
})
```

### 5. 工具调用模式

#### 异步工具支持
```python
from typing import List
from langchain.tools import tool

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True
```

**关键点**：
- 工具支持同步和异步执行
- 使用 `_run()` 方法同步执行
- 使用 `_arun()` 方法异步执行

### 6. Retriever 工具模式

#### 将 Retriever 包装为工具
```python
from langchain.tools import tool

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts
```

**应用场景**：
- RAG 系统中的文档检索
- 知识库查询
- 向量数据库搜索

### 7. BaseTool 继承模式

#### 工具类接口定义
```python
from langchain.tools import BaseTool

class FMPDataToolkit:
    """Creates a collection of FMP data tools based on queries."""

    def __init__(
        self,
        query: str | None = None,
        num_results: int = 3,
        similarity_threshold: float = 0.3,
        cache_dir: str | None = None,
    ): ...

    def get_tools(self) -> list[BaseTool]:
        """Returns a list of relevant FMP API tools based on the query."""
        ...
```

**BaseTool 接口要求**：
1. 继承自 `BaseTool` 基类
2. 实现 3 个属性和 2 个方法
3. 定义输入 schema（args_schema）
4. 实现 `_run()` 方法（必需）
5. 可选实现 `_arun()` 方法（异步）

### 8. 工具执行方法

#### 直接调用工具
```python
# 使用 run() 方法
tool.run("ai")

# 使用 _run() 方法（内部）
search_results = search_tool._run(
    query="When was the last time the New York Knicks won the NBA Championship?",
    num_results=5,
    text_contents_options=True,
    highlights=True
)
```

## 最佳实践总结

### 1. 选择合适的创建方式
- **简单函数** → `@tool` 装饰器
- **需要显式 schema** → `@tool(args_schema=...)`
- **包装现有 API** → `StructuredTool.from_function()`
- **复杂状态管理** → 继承 `BaseTool`

### 2. Schema 设计原则
- 使用 Pydantic 提供类型安全
- 使用 `Field()` 添加详细描述
- 使用 `Literal` 限制枚举值
- 提供合理的默认值

### 3. 描述编写技巧
- 清晰说明工具功能
- 说明何时应该使用该工具
- 包含输入输出示例
- 使用 Google-style docstring

### 4. 与 Agent 集成
- 使用 `create_openai_functions_agent` 创建 Agent
- 通过 `AgentExecutor` 执行工具调用
- 设置 `verbose=True` 便于调试

### 5. 异步支持
- 工具自动支持同步和异步
- 优先实现异步版本（性能更好）
- 使用 `_run()` 和 `_arun()` 方法
