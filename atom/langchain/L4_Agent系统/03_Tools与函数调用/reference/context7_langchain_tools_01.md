---
type: context7_documentation
library: langchain
version: latest (2026)
fetched_at: 2026-02-28
knowledge_point: 03_Tools与函数调用
context7_query: tools definition @tool decorator StructuredTool BaseTool bind_tools function calling
---

# Context7 文档：LangChain Tools & Function Calling

## 文档来源
- 库名称：LangChain
- 版本：latest (2026)
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. @tool 装饰器 + bind_tools 基本用法

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

### 2. 多参数工具定义

```python
@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.
    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True

llm_with_tools = llm.bind_tools([validate_user])
result = llm_with_tools.invoke("Could you validate user 123?...")
result.tool_calls
```

### 3. 动态工具选择 - Middleware 模式 (2026 新)

#### 类方式
```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class ToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4.1",
    tools=all_tools,
    middleware=[ToolSelectorMiddleware()],
)
```

#### 装饰器方式
```python
@wrap_model_call
def select_tools(request, handler):
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return handler(request.override(tools=relevant_tools))
```

#### 内置 LLMToolSelectorMiddleware
```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4.1-mini",
            max_tools=3,
            always_include=["search"],
        ),
    ],
)
```

### 4. 工具选择的优势
- **更短的 Prompt**: 减少复杂度，只暴露相关工具
- **更高准确率**: 模型从更少选项中正确选择
- **权限控制**: 基于用户权限动态过滤工具

### 5. tool_calls 返回格式
```python
# AIMessage.tool_calls 格式
[{
    'name': 'multiply',
    'args': {'first_int': 5, 'second_int': 42},
    'id': 'call_f0c2cc49307f480db78a45',
    'type': 'tool_call'
}]
```
