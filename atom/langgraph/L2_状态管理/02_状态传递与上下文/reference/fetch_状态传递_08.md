---
type: fetched_content
source: https://github.com/FareedKhan-dev/contextual-engineering-guide
title: LangChain AI Agents Using Contextual Engineering - Implementation with LangGraph
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# LangChain AI Agents Using Contextual Engineering

**Repository Metadata**
- **Description**: Implementation of contextual engineering pipeline with LangChain and LangGraph Agents
- **Stars**: 83
- **Forks**: 18
- **Watchers**: 1
- **Main Language**: Jupyter Notebook (主要)，Python
- **License**: 存在 LICENSE 文件
- **Files in repository**: LICENSE, README.md, contextual_engineering.ipynb, requirements.txt, utils.py

Context engineering means creating the right setup for an AI before giving it a task. This setup includes:

* **Instructions** on how the AI should act, like being a helpful budget travel guide
* Access to **useful info** from databases, documents, or live sources.
* Remembering **past conversations** to avoid repeats or forgetting.
* **Tools** the AI can use, such as calculators or search features.
* Important details about you, like your **preferences** or location.

*Context Engineering (From LangChain and 12Factor )*

[AI engineers are now shifting](https://diamantai.substack.com/p/why-ai-experts-are-moving-from-prompt) from prompt engineering to context engineering because…

> context engineering focuses on providing AI with the right background and tools, making its answers smarter and more useful.

In this blog, we will explore how **LangChain** and **LangGraph** two powerful tools for building AI agents, RAG apps, and LLM apps can be used to implement **contextual engineering** effectively to improve our AI Agents.

This guide is created on top of [langgchain ai](https://github.com/FareedKhan-dev/contextual-engineering-guide) guide.

---

### Table of Contents

* [What is Context Engineering?](#what-is-context-engineering)
* [Scratchpad with LangGraph](#scratchpad-with-langgraph)
* [Creating StateGraph](#creating-stategraph)
* [Memory Writing in LangGraph](#memory-writing-in-langgraph)
* [Scratchpad Selection Approach](#scratchpad-selection-approach)
* [Memory Selection Ability](#memory-selection-ability)
* [Advantage of LangGraph BigTool Calling](#advantage-of-langgraph-bigtool-calling)
* [RAG with Contextual Engineering](#rag-with-contextual-engineering)
* [Compression Strategy with knowledgeable Agents](#compression-strategy-with-knowledgeable-agents)
* [Isolating Context using Sub-Agents Architecture](#isolating-context-using-sub-agents-architecture)
* [Isolation using Sandboxed Environments](#isolation-using-sandboxed-environments)
* [State Isolation in LangGraph](#state-isolation-in-langgraph)
* [Summarizing Everything](#summarizing-everything)

### What is Context Engineering?

LLMs work like a new type of operating system. The LLM acts like the CPU, and its context window works like RAM, serving as its short-term memory. But, like RAM, the context window has limited space for different information.

> Just as an operating system decides what goes into RAM, "context engineering" is about choosing what the LLM should keep in its context.

When building LLM applications, we need to manage different types of context. Context engineering covers these main types:

* Instructions: prompts, examples, memories, and tool descriptions
* Knowledge: facts, stored information, and memories
* Tools: feedback and results from tool calls

This year, more people are interested in agents because LLMs are better at thinking and using tools. Agents work on long tasks by using LLMs and tools together, choosing the next step based on the tool's feedback.

But long tasks and collecting too much feedback from tools use a lot of tokens. This can create problems: the context window can overflow, costs and delays can increase, and the agent might work worse.

Drew Breunig explained how too much context can hurt performance, including:

* Context Poisoning: [when a mistake or hallucination gets added to the context](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-poisoning)
* Context Distraction: [when too much context confuses the model](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-distraction)
* Context Confusion: [when extra, unnecessary details affect the answer](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-confusion)
* Context Clash: [when parts of the context give conflicting information](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-clash)

Anthropic [in their research](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) stressed the need for it:

> Agents often have conversations with hundreds of turns, so managing context carefully is crucial.

So, how are people solving this problem today? Common strategies for agent context engineering can be grouped into four main types:

* Write: creating clear and useful context
* Select: picking only the most relevant information
* Compress: shortening context to save space
* Isolate: keeping different types of context separate

*Categories of Context Engineering (From LangChain docs )*

[LangGraph](https://www.langchain.com/langgraph) is built to support all these strategies. We will go through each of these components one by one in [LangGraph](https://www.langchain.com/langgraph) and see how they help make our AI agents work better.

### Scratchpad with LangGraph

Just like humans take notes to remember things for later tasks, agents can do the same using a [scratchpad](https://www.anthropic.com/engineering/claude-think-tool). It stores information outside the context window so the agent can access it whenever needed.

*First Component of CE (From LangChain docs )*

A good example is [Anthropic multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system) :

> The LeadResearcher plans its approach and saves it to memory, because if the context window goes beyond 200,000 tokens, it gets cut off so saving the plan ensures it isn't lost.

Scratchpads can be implemented in different ways:

* As a [tool call](https://www.anthropic.com/engineering/claude-think-tool) that [writes to a file](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem).
* As a field in a runtime [state object](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) that persists during the session.

In short, scratchpads help agents keep important notes during a session to complete tasks effectively.

In terms of LangGraph, it supports both [short-term](https://langchain-ai.github.io/langgraph/concepts/memory/#short-term-memory) (thread-scoped) and [long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory).

* Short-term memory uses [checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/) to save the [agent state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) during a session. It works like a scratchpad, letting you store information while the agent runs and retrieve it later.

The state object is the main structure passed between graph nodes. You can define its format (usually a Python dictionary). It acts as a shared scratchpad, where each node can read and update specific fields.

> We will only import the modules when we need them, so we can learn step by step in a clear way.

For better and cleaner output, we will use Python `pprint` module for pretty printing and the `Console` module from the `rich` library. Let's import and initialize them first:

```python
# Import necessary libraries
from typing import TypedDict  # For defining the state schema with type hints

from rich.console import Console  # For pretty-printing output
from rich.pretty import pprint  # For pretty-printing Python objects

# Initialize a console for rich, formatted output in the notebook.
console = Console()
```

Next, we will create a `TypedDict` for the state object.

```markdown
# Define the schema for the graph's state using TypedDict.
# This class acts as a data structure that will be passed between nodes in the graph.
# It ensures that the state has a consistent shape and provides type hints.
class State(TypedDict):
    """
    Defines the structure of the state for our joke generator workflow.

    Attributes:
        topic: The input topic for which a joke will be generated.
        joke: The output field where the generated joke will be stored.
    """

    topic: str
    joke: str
```

This state object will store the topic and the joke that we ask our agent to generate based on the given topic.

### Creating StateGraph

Once we define a state object, we can write context to it using a [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph).

A StateGraph is LangGraph's main tool for building stateful [agents or workflows](https://langchain-ai.github.io/langgraph/concepts/workflows/). Think of it as a directed graph:

* Nodes are steps in the workflow. Each node takes the current state as input, updates it, and returns the changes.
* Edges connect nodes, defining how execution flows this can be linear, conditional, or even cyclical.

Next, we will:

1. Create a [chat model](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) by choosing from [Anthropic models](https://docs.anthropic.com/en/docs/about-claude/models/overview).
2. Use it in a LangGraph workflow.

```python
# Import necessary libraries for environment management, display, and LangGraph
import getpass
import os

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

# --- Environment and Model Setup ---
# Set the Anthropic API key to authenticate requests
from dotenv import load_dotenv
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY in environment")

# Initialize the chat model to be used in the workflow
# We use a specific Claude model with temperature=0 for deterministic outputs
llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0)
```

We've initialized our Sonnet model. LangChain supports many open-source and closed models through their APIs, so you can use any of them.

Now, we need to create a function that generates a response using this Sonnet model.

```python
# --- Define Workflow Node ---
def generate_joke(state: State) -> dict[str, str]:
    """
    A node function that generates a joke based on the topic in the current state.

    This function reads the 'topic' from the state, uses the LLM to generate a joke,
    and returns a dictionary to update the 'joke' field in the state.

    Args:
        state: The current state of the graph, which must contain a 'topic'.

    Returns:
        A dictionary with the 'joke' key to update the state.
    """
    # Read the topic from the state
    topic = state["topic"]
    print(f"Generating a joke about: {topic}")

    # Invoke the language model to generate a joke
    msg = llm.invoke(f"Write a short joke about {topic}")

    # Return the generated joke to be written back to the state
    return {"joke": msg.content}
```

This function simply returns a dictionary containing the generated response (the joke).

Now, using the StateGraph, we can easily build and compile the graph. Let's do that next.

```python
# --- Build and Compile the Graph ---
# Initialize a new StateGraph with the predefined State schema
workflow = StateGraph(State)

# Add the 'generate_joke' function as a node in the graph
workflow.add_node("generate_joke", generate_joke)

# Define the workflow's execution path:
# The graph starts at the START entrypoint and flows to our 'generate_joke' node.
workflow.add_edge(START, "generate_joke")
# After 'generate_joke' completes, the graph execution ends.
workflow.add_edge("generate_joke", END)

# Compile the workflow into an executable chain
chain = workflow.compile()

# --- Visualize the Graph ---
# Display a visual representation of the compiled workflow graph
display(Image(chain.get_graph().draw_mermaid_png()))
```

---

**Note**: The complete implementation with additional sections (Memory Writing, RAG, Compression, Isolation, etc.) is available in the repository's Jupyter Notebook `contextual_engineering.ipynb`. This README provides the conceptual framework and initial implementation examples for understanding contextual engineering with LangGraph.

**Repository Files**:
- `contextual_engineering.ipynb`: Complete implementation with all examples
- `requirements.txt`: Python dependencies
- `utils.py`: Helper functions
- `LICENSE`: Repository license
- `README.md`: This documentation
