---
type: fetched_content
source: https://github.com/langfuse/langfuse/issues/6761
title: bug: prompt not linking to CallbackHandler trace when using LangGraph · Issue #6761 · langfuse/langfuse
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: LangGraph回调处理
---

```markdown
---
source: https://github.com/langfuse/langfuse/issues/6761
title: bug: prompt not linking to CallbackHandler trace when using LangGraph · Issue #6761 · langfuse/langfuse
fetched_at: February 25, 2026 01:27 AM PST
---

# bug: prompt not linking to CallbackHandler trace when using LangGraph

**Issue #6761** · [langfuse/langfuse](https://github.com/langfuse/langfuse)

## Metadata

- **Status**: Open
- **Author**: [hassiebp](https://github.com/hassiebp)
- **Assignees**: None
- **Labels**: `stale`, `🐞❔ unconfirmed bug`
- **Projects**: None
- **Milestone**: None
- **Linked pull requests**: None
- **Participants**: 1 (hassiebp)

## Description

### Describe the bug

When using the CallbackHandler for LangGraph I am getting automatically generated traces on the system. I want to link LangFuse prompts to these traces for the purposes of collecting metrics, however when I use the observe decorator (as demonstrated in the [documentation](https://langfuse.com/docs/prompts/get-started#link-with-langfuse-tracing-optional) ), it creates a new trace instead of calling from the trace that was created by the CallbackHandler. The langfuse_context does not seem to realise that there is a current trace happening. The code below contains the exact setup.

### To reproduce

Run the following code with the .env containing the Langfuse variables:

```
"""Minimal example of prompts not being linked"""
from typing import Literal

from dotenv import load_dotenv
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe, langfuse_context

load_dotenv()

# Create LangFuse classes
langfuse_handler = CallbackHandler()
langfuse = Langfuse()

# Create LangGraph node
def thinker(
    state: MessagesState,
) -> Command[Literal["__end__"]]:
    """LangGraph node"""
    prompt = get_prompt("test-prompt")
    return Command(goto="__end__", update={"messages": [prompt]})

# Get prompt with observation, comes up as separate trace
@observe(as_type="generation")
def get_prompt(name: str) -> str:
    """Get the prompt fron LangFuse by name"""
    langfuse_prompt = langfuse.get_prompt(name)
    langfuse_context.update_current_observation(
        prompt=langfuse_prompt,
    )
    return langfuse_prompt.prompt

# Build Graph
graph = StateGraph(MessagesState)
graph.add_node("thinker", thinker)
graph.add_edge(START, "thinker")
agent = graph.compile()

# Start flow
config = {"callbacks": [langfuse_handler]}
user_input = {"role": "user", "content": input("\n[You] ")}
response = agent.invoke({"messages": [user_input], "next": []}, config)
print(f"\n[Agent]\n{response["messages"][-1].content}\n")
```

### SDK and container versions
*No response*

### Additional information
*No response*

### Are you interested to contribute a fix for this bug?
No

## Comments

**No comments.**

## Sidebar

- **Participants**: [hassiebp](https://github.com/hassiebp)
```
