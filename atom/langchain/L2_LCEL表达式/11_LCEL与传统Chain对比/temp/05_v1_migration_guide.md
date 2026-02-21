# LangChain v1 Migration Guide

**Source**: https://docs.langchain.com/oss/python/migrate/langchain-v1
**Fetched**: 2026-02-20

## Simplified Package

The `langchain` package namespace has been significantly reduced in v1 to focus on essential building blocks for agents.

### Available Modules

| Module | What's available | Notes |
|--------|------------------|-------|
| langchain.agents | Core agent creation functionality | |
| langchain.messages | Message types, content blocks, trim_messages | Re-exported from langchain-core |
| langchain.tools | @tool, BaseTool, injection helpers | Re-exported from langchain-core |
| langchain.chat_models | Unified model initialization | |
| langchain.embeddings | Embedding models | |

### langchain-classic

Legacy functionality moved to `langchain-classic`:
- Legacy chains (LLMChain, ConversationChain, etc.)
- Retrievers (MultiQueryRetriever, etc.)
- Indexing API
- Hub module
- Embeddings modules
- langchain-community re-exports

```bash
pip install langchain-classic
```

## Migrate to create_agent

### Import Path Change

```python
# Old
from langgraph.prebuilt import create_react_agent

# New
from langchain.agents import create_agent
```

### Key Changes Summary

| Section | What's Changed |
|---------|----------------|
| Import path | Package moved from langgraph.prebuilt to langchain.agents |
| Prompts | Parameter renamed to system_prompt, dynamic prompts use middleware |
| Pre-model hook | Replaced by middleware with before_model method |
| Post-model hook | Replaced by middleware with after_model method |
| Custom state | TypedDict only, defined via state_schema or middleware |
| Model | Dynamic selection via middleware, pre-bound models not supported |
| Tools | Tool error handling moved to middleware with wrap_tool_call |
| Structured output | prompted output removed, use ToolStrategy/ProviderStrategy |
| Streaming node name | Node name changed from "agent" to "model" |
| Runtime context | Dependency injection via context argument instead of config["configurable"] |

### Prompts

#### Static Prompt Rename

```python
# Old
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt="You are a helpful assistant"
)

# New
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=tools,
    system_prompt="You are a helpful assistant"
)
```

#### Dynamic Prompts

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply."
    else:
        return base_prompt

agent = create_agent(
    model="gpt-4.1",
    tools=tools,
    middleware=[dynamic_prompt],
    context_schema=Context
)
```

### Middleware System

#### Pre-model Hook (Summarization Example)

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=tools,
    middleware=[
        SummarizationMiddleware(
            model="claude-sonnet-4-5-20250929",
            trigger={"tokens": 1000}
        )
    ]
)
```

#### Post-model Hook (Human-in-the-Loop Example)

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[read_email, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "description": "Please review this email before sending",
                    "allowed_decisions": ["approve", "reject"]
                }
            }
        )
    ]
)
```

### Custom State

#### State Type Restrictions

`create_agent` only supports `TypedDict` for state schemas. Pydantic models and dataclasses are no longer supported.

```python
from langchain.agents import AgentState, create_agent

# AgentState is a TypedDict
class CustomAgentState(AgentState):
    user_id: str

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=tools,
    state_schema=CustomAgentState
)
```

### Tools

#### Tool Error Handling

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[check_weather, search_web],
    middleware=[handle_tool_errors]
)
```

### Structured Output

#### Tool and Provider Strategies

```python
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str

# Using ToolStrategy
agent = create_agent(
    model="gpt-4.1-mini",
    tools=tools,
    response_format=ToolStrategy(OutputSchema)
)
```

**Prompted output is no longer supported** via response_format.

### Runtime Context

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    session_id: str

agent = create_agent(
    model=model,
    tools=tools,
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    context=Context(user_id="123", session_id="abc")
)
```

## Standard Content Blocks

Messages gain provider-agnostic standard content blocks via `message.content_blocks`.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
response = model.invoke("Explain AI")

for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(block.get("reasoning"))
    elif block["type"] == "text":
        print(block.get("text"))
```

## Breaking Changes

### Dropped Python 3.9 Support
All LangChain packages now require **Python 3.10 or higher**.

### Updated Return Type for Chat Models
Return type signature fixed from `BaseMessage` to `AIMessage`.

### Default max_tokens in langchain-anthropic
Now defaults to higher values based on model chosen (previously 1024).

### Legacy Code Moved to langchain-classic
Existing functionality outside core abstractions moved to `langchain-classic` package.

### Text Property
The `.text()` method is now a **property** (no parentheses):

```python
text = response.text  # Correct
# text = response.text()  # Deprecated
```

### example Parameter Removed from AIMessage
Use `additional_kwargs` for extra metadata instead.
