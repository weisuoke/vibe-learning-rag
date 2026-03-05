---
type: context7_documentation
library: langchain
version: 1.0+
fetched_at: 2026-02-28
knowledge_point: 01_create_agent标准抽象（2026新）
context7_query: create_agent middleware AgentMiddleware
---

# Context7 文档：LangChain create_agent & Middleware

## 文档来源
- 库名称：LangChain
- 版本：1.0+
- 官方文档链接：https://docs.langchain.com/oss/python/langchain/agents

## 关键信息提取

### 1. create_agent 基础用法

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    system_prompt="You are a helpful assistant",
)
```

### 2. Middleware 两种定义方式

#### 装饰器方式（简单场景）
```python
@before_model
def log_before_model(state, runtime):
    print(f"About to call model with {len(state['messages'])} messages")
    return None

@wrap_model_call
def retry_model(request, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2: raise
            print(f"Retry {attempt + 1}/3")
```

#### 类方式（复杂场景）
```python
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Messages: {len(state['messages'])}")
        return None
    def after_model(self, state, runtime):
        print(f"Response: {state['messages'][-1].content}")
        return None
```

### 3. 动态 Prompt 中间件
```python
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")
    if user_role == "expert":
        return "Provide detailed technical responses."
    return "Explain concepts simply."
```

### 4. 内置 Middleware 示例
- ModelCallLimitMiddleware: 限制模型调用次数
- ModelFallbackMiddleware: 模型降级
- ModelRetryMiddleware: 自动重试
- HumanInTheLoopMiddleware: 人工审核
- SummarizationMiddleware: 上下文摘要
- ToolRetryMiddleware: 工具重试

### 5. 动态工具选择
```python
class ToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))
```

### 6. context_schema 运行时上下文
```python
class Context(TypedDict):
    user_role: str

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

result = agent.invoke(
    {"messages": [...]},
    context={"user_role": "expert"}
)
```
