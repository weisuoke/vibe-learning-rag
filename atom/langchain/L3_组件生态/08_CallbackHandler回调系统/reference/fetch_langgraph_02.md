---
type: fetched_content
source: https://github.com/orgs/langfuse/discussions/10711
title: How to customize the callback handler when multiple llm calls are being made in a langgraph state?
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: LangGraph回调处理
---

# How to customize the callback handler when multiple llm calls are being made in a langgraph state?

**langfuse · Discussion #10711**

**Labels**: [integration-langchain](https://github.com/orgs/langfuse/discussions?discussions_q=label%3Aintegration-langchain)

**Participants**: 2

## 原始帖子 (Original Post)

**Describe your question**

Hello I have a use case where I have a streaming LLM call. On some X amount of chunks I run it through a LLM Moderation call such that it looks like the the following

```python
def call_model():
    streaming_response = call_main_model()  # Uses langchain's `ChatBedrockConverse`
    call_moderation_model(streaming_response)  # Also uses langchain's `ChatBedrockConverse`

def call_moderation_model(streaming_response):
    with span.start_as_current_span("critic.orchestrator.evaluate"):
        ChatBedrockConverse(streaming_response)
```

This is a state in langgraph and I use the callback handler to automatically observe it. The issue is that the traces now look like this. I have 3 moderation models running creating 3 bedrock spans.

Is there a way I can customize it in such a way I can customize the name of the bedrock span? The bedrock calls appear outside of the critic.orchestrator.evaluate even though they were executed within the span. How do i make it so its a sub span of the critic.orchestrator.evaluate call.

**Langfuse Cloud or Self-Hosted?** Self-Hosted
**If Self-Hosted** 3.131
**SDK and integration versions** *No response*

**Pre-Submission Checklist**
I have checked for existing issues/discussions and consulted Langfuse AI.

## 回复 1：Inkeep AI Assistant

Based on the available sources, there are known context propagation issues with the CallbackHandler in LangGraph, particularly when trying to nest observations properly.

**Issue with Nesting**
The LangChain CallbackHandler uses `start_span` instead of `start_as_current_span`, which means it doesn't register observations in the current context. This is why your Bedrock calls appear outside of your `critic.orchestrator.evaluate` span even though they're executed within it.

**Recommended Approach**
The recommended way to ensure correct span context and nesting is to use the `@observe` decorator or the `langfuse.start_as_current_span()` context manager.

```python
from langfuse import observe

@observe()
def call_moderation_model(streaming_response):
    # This will automatically create a span and set it as current
    ChatBedrockConverse(streaming_response)
```

**Alternative: Explicit Context Passing**

```python
from langfuse import get_client
langfuse = get_client()

def call_moderation_model(streaming_response):
    with langfuse.start_as_current_span(name="critic.orchestrator.evaluate") as span:
        # Initialize handler within the span context
        langfuse_handler = CallbackHandler()
        # Your LangChain call will now be nested under this span
        ChatBedrockConverse(streaming_response, callbacks=[langfuse_handler])
```

**Known Limitations**
The current CallbackHandler implementation has limitations with context propagation in concurrent or complex scenarios. For customizing span names when using CallbackHandler with LangChain models, span names are typically derived from the LangChain component names automatically.

**Sources**
- https://github.com/langfuse/langfuse/issues/8584
- https://github.com/langfuse/langfuse/issues/8001
- https://github.com/orgs/langfuse/discussions/8009
- https://langfuse.com/integrations/frameworks/langchain

## 回复 2 (OP)

@inkeep i added @observe to the call_moderation_model call and it did not work as expected. For reference im running langfuse 3.9 of the sdk

## 回复 3 (Langfuse Team)

HI @dahnny012 - could you please provide a minimum example of what you are doing for us to reproduce?

## 回复 4 (OP)

see below comment

## 最小复现代码示例 (by OP)

```python
import asyncio
from typing import List, Dict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrockConverse
import os
# Langfuse
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
import dotenv
import time

dotenv.load_dotenv()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

class WorkflowState(MessagesState):
    prompt: str
    result: str

llm = ChatBedrockConverse(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
)

def router(state: dict):
    langfuse.update_current_trace(tags=["router"])
    return {"bedrock_call": state["prompt"]}

async def call_moderation(user_query: str, response: str):
    langfuse.update_current_trace(tags=["call_moderation"])
    messages = [
        HumanMessage(content=f"Analyze this LLM's response and score it 1-5 on how well it fit the user's query <query>{user_query}</query> <response>{response}</response>"),
        HumanMessage(content=f"Analyze this LLM's response and score it 1-5 on how safe it is to show to a user <query>{user_query}</query> <response>{response}</response>")
    ]
    results = []
    for msg in messages:
        with langfuse.start_as_current_span(name="evaluation") as span:
            span.update(input=user_query)
            result = await llm.ainvoke([msg])
            span.update(output=[result.content])
            results.append(result.content)
    return results

async def call_bedrock(state: dict):
    langfuse.update_current_trace(tags=["call_bedrock"])
    prompt = state["prompt"]
    main_msg = HumanMessage(content=f"Analyze this users response and make up a tool you would use to answer it <query>{prompt}</query>")
    main_response = await llm.ainvoke([main_msg])
    moderation_response = await call_moderation(prompt, main_response.content)
    return {"result": {"main": main_response.content, "moderation": moderation_response}}

def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("router", router)
    graph.add_node("bedrock_call", call_bedrock)
    graph.set_entry_point("router")
    graph.add_edge("router", "bedrock_call")
    graph.add_edge("bedrock_call", END)
    return graph.compile()

async def main():
    graph = build_graph()
    inputs = {"prompt": "Write a haiku about oceans."}
    langfuse = get_client()
    trace_id = f"trace-id-{time.time()}"
    langfuse_trace_id = langfuse.create_trace_id(seed=trace_id)
    lf_handler = CallbackHandler()
    config: RunnableConfig = {"callbacks": [lf_handler], "run_id": langfuse_trace_id}
    with langfuse.start_as_current_span(name="chat-request") as span:
        span.update_trace(name=trace_id, input={"user_query": inputs["prompt"]}, metadata={"iteration": 0})
        span.update(input={"user_query": inputs["prompt"]})
        result = await graph.ainvoke(inputs, config=config)
        span.update(output={"response": result["result"]})

if __name__ == "__main__":
    asyncio.run(main())
```

## 回复 5 (OP 进一步说明)

After some testing. If i pass the callback handler into the llm.ainvoke call of call_moderation it get the nesting i want. Is that recommended to pass the callback handler in multiple places? I thought passing it at the top level graph invoke was enough but for LLM calls specifically that operate in another function it looks like i need to pass the callback handler to get proper nesting?

**更新后的 `call_moderation` 示例代码：**

```python
async def call_moderation(user_query: str, response: str):
    messages = [
        HumanMessage(content=...),
        HumanMessage(content=...)
    ]
    results = []
    callback_handler = CallbackHandler()
    for msg in messages:
        with langfuse.start_as_current_span(name="evaluation") as span:
            langfuse.update_current_trace(tags=["call_moderation"])
            span.update(input=user_query)
            result = await llm.ainvoke(
                [msg],
                config={"callbacks": [callback_handler]}
            )
            span.update(output=[result.content])
            results.append(result.content)
    return results
```

**页脚信息**
You can't perform that action at this time.
**Category**：Support
**Sign in to comment** required.
