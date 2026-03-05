---
type: fetched_content
source: https://blog.langchain.com/context-engineering-for-agents
title: Context Engineering
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# Context Engineering

11 min read · Jul 2, 2025

### TL;DR

Agents need context to perform tasks. Context engineering is the art and science of filling the context window with just the right information at each step of an agent's trajectory. In this post, we break down some common strategies — **write, select, compress, and isolate —** for context engineering by reviewing various popular agents and papers. We then explain how LangGraph is designed to support them!

**Also, see our video on context engineering** [here](https://youtu.be/4GiqzUHD5AA?ref=blog.langchain.com).

*General categories of context engineering*

## Context Engineering

As Andrej Karpathy puts it, LLMs are like a [new kind of operating system](https://www.youtube.com/watch?si=-aKY-x57ILAmWTdw&t=620&v=LCEmiRjPEtQ&feature=youtu.be&ref=blog.langchain.com). The LLM is like the CPU and its [context window](https://docs.anthropic.com/en/docs/build-with-claude/context-windows?ref=blog.langchain.com) is like the RAM, serving as the model's working memory. Just like RAM, the LLM context window has limited [capacity](https://lilianweng.github.io/posts/2023-06-23-agent/?ref=blog.langchain.com) to handle various sources of context. And just as an operating system curates what fits into a CPU's RAM, we can think about "context engineering" playing a similar role. [Karpathy summarizes this well](https://x.com/karpathy/status/1937902205765607626?ref=blog.langchain.com):

> [Context engineering is the] "…delicate art and science of filling the context window with just the right information for the next step."

*Context types commonly used in LLM applications*

What are the types of context that we need to manage when building LLM applications? Context engineering as an [umbrella](https://x.com/dexhorthy/status/1933283008863482067?ref=blog.langchain.com) that applies across a few different context types:

* **Instructions** – prompts, memories, few-shot examples, tool descriptions, etc
* **Knowledge** – facts, memories, etc
* **Tools** – feedback from tool calls

## Context Engineering for Agents

This year, interest in [agents](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) has grown tremendously as LLMs get better at [reasoning](https://platform.openai.com/docs/guides/reasoning?api-mode=responses&ref=blog.langchain.com) and [tool calling](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com). [Agents](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) interleave [LLM invocations and tool calls](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com), often for [long-running tasks](https://blog.langchain.com/introducing-ambient-agents/). Agents interleave [LLM calls and tool calls](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com), using tool feedback to decide the next step.

*Agents interleave LLM calls and tool calls, using tool feedback to decide the next step*

However, long-running tasks and accumulating feedback from tool calls mean that agents often utilize a large number of tokens. This can cause numerous problems: it can [exceed the size of the context window](https://cognition.ai/blog/kevin-32b?ref=blog.langchain.com), balloon cost / latency, or degrade agent performance. Drew Breunig [nicely outlined](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com) a number of specific ways that longer context can cause perform problems, including:

* [Context Poisoning: When a hallucination makes it into the context](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-poisoning)
* [Context Distraction: When the context overwhelms the training](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-distraction)
* [Context Confusion: When superfluous context influences the response](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-confusion)
* [Context Clash: When parts of the context disagree](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-clash)

*Context from tool calls accumulates over multiple agent turns*

With this in mind, [Cognition](https://cognition.ai/blog/dont-build-multi-agents?ref=blog.langchain.com) called out the importance of context engineering:

> "Context engineering" … is effectively the #1 job of engineers building AI agents.

[Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) also laid it out clearly:

> Agents often engage in conversations spanning hundreds of turns, requiring careful context management strategies.

So, how are people tackling this challenge today? We group common strategies for agent context engineering into four buckets — **write, select, compress, and isolate —** and give examples of each from review of some popular agent products and papers. We then explain how LangGraph is designed to support them!

*General categories of context engineering*

## Write Context

*Writing context means saving it outside the context window to help an agent perform a task.*

### Scratchpads

When humans solve tasks, we take notes and remember things for future, related tasks. Agents are also gaining these capabilities! Note-taking via a " [scratchpad](https://www.anthropic.com/engineering/claude-think-tool?ref=blog.langchain.com) " is one approach to persist information while an agent is performing a task. The idea is to save information outside of the context window so that it's available to the agent. [Anthropic's multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) illustrates a clear example of this:

> The LeadResearcher begins by thinking through the approach and saving its plan to Memory to persist the context, since if the context window exceeds 200,000 tokens it will be truncated and it is important to retain the plan.

Scratchpads can be implemented in a few different ways. They can be a [tool call](https://www.anthropic.com/engineering/claude-think-tool?ref=blog.langchain.com) that simply [writes to a file](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem?ref=blog.langchain.com). They can also be a field in a runtime [state object](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#state) that persists during the session. In either case, scratchpads let agents save useful information to help them accomplish a task.

### Memories

Scratchpads help agents solve a task within a given session (or [thread](https://langchain-ai.github.io/langgraph/concepts/persistence/?ref=blog.langchain.com#threads)), but sometimes agents benefit from remembering things across *many* sessions! [Reflexion](https://arxiv.org/abs/2303.11366?ref=blog.langchain.com) introduced the idea of reflection following each agent turn and re-using these self-generated memories. [Generative Agents](https://ar5iv.labs.arxiv.org/html/2304.03442?ref=blog.langchain.com) created memories synthesized periodically from collections of past agent feedback.

*An LLM can be used to update or create memories*

These concepts made their way into popular products like [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq?ref=blog.langchain.com), [Cursor](https://forum.cursor.com/t/0-51-memories-feature/98509?ref=blog.langchain.com), and [Windsurf](https://docs.windsurf.com/windsurf/cascade/memories?ref=blog.langchain.com), which all have mechanisms to auto-generate long-term memories that can persist across sessions based on user-agent interactions.

*Cursor memories are auto-generated from user-agent interactions*

## Select Context

*Selecting context means choosing what to include in the context window.*

### Retrieval

Retrieval is a well-established approach for selecting context. The idea is to use a query to retrieve relevant information from a knowledge base. This is the foundation of [RAG](https://blog.langchain.com/deconstructing-rag/?ref=blog.langchain.com), which has become a popular way to ground LLM responses in external knowledge. For agents, retrieval can be used to select relevant context from various sources:

* **Tool feedback** – retrieve relevant past tool calls
* **Memories** – retrieve relevant memories from past sessions
* **Knowledge** – retrieve relevant facts from a knowledge base

*Retrieval selects relevant context from various sources*

### Filtering

Filtering is another way to select context. The idea is to use rules or heuristics to decide what context to include. For example, [Anthropic's multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) uses filtering to manage context:

> We implemented a context management system that tracks token usage and automatically truncates older messages when approaching limits, while preserving critical information like the research plan.

*Filtering uses rules to decide what context to include*

## Compress Context

*Compressing context means reducing its size while preserving important information.*

### Summarization

Summarization is a common way to compress context. The idea is to use an LLM to create a shorter version of the context that captures the key information. [Anthropic's multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) uses summarization:

> When the context window approaches its limit, we summarize older messages to preserve the essential information while reducing token count.

*Summarization compresses context by creating shorter versions*

### Prompt Compression

Prompt compression is another approach. The idea is to use specialized models or techniques to reduce the size of prompts while maintaining their effectiveness. [LLMLingua](https://arxiv.org/abs/2310.05736?ref=blog.langchain.com) is an example of this approach.

## Isolate Context

*Isolating context means separating different types of context or tasks.*

### Multi-Agent Systems

Multi-agent systems isolate context by distributing tasks across multiple agents, each with their own context window. [Anthropic's multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) is a good example:

> We built a multi-agent system where different agents handle different parts of the research process, each with their own context window.

*Multi-agent systems isolate context across different agents*

### Hierarchical Planning

Hierarchical planning isolates context by breaking down tasks into subtasks, each with their own context. [Plan-and-Solve](https://arxiv.org/abs/2305.04091?ref=blog.langchain.com) is an example of this approach.

## Context Engineering with LangGraph

LangGraph is designed to support all of these context engineering strategies:

* **Write**: Use [state](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#state) to implement scratchpads and [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/?ref=blog.langchain.com) for memories
* **Select**: Use [retrieval](https://blog.langchain.com/deconstructing-rag/?ref=blog.langchain.com) and custom logic for filtering
* **Compress**: Use LLMs for summarization in your graph nodes
* **Isolate**: Use [subgraphs](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#subgraphs) and [multi-agent patterns](https://langchain-ai.github.io/langgraph/concepts/multi_agent/?ref=blog.langchain.com)

*LangGraph supports all context engineering strategies*

## Conclusion

Context engineering is critical for building effective agents. By understanding and applying these strategies — write, select, compress, and isolate — you can build agents that handle long-running tasks efficiently. LangGraph provides the primitives to implement all of these strategies, giving you full control over how your agents manage context.
