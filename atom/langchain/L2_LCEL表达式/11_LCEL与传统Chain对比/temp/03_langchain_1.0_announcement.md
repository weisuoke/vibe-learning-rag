# LangChain and LangGraph 1.0 Milestones

**Source**: https://blog.langchain.com/langchain-langgraph-1dot0/
**Fetched**: 2026-02-20
**Date**: October 22, 2025

## Framework Purposes

- **LangChain**: Fastest way to build an AI agent with standard tool calling architecture, provider agnostic design, and middleware for customization
- **LangGraph**: Lower level framework and runtime for highly custom and controllable agents, designed for production-grade, long running agents

## LangChain 1.0 Key Changes

### 1. create_agent Abstraction

**Setup**: Select a model, give it tools and a prompt

**Execution Loop**:
1. Send request to model
2. Model responds with either:
   - Tool calls → execute tool and add results to conversation
   - Final answer → return result
3. Repeat from step 1

Built on LangGraph runtime for reliable agents.

### 2. Middleware System

Middleware defines hooks for customizing agent loop behavior:

**Built-in Middlewares**:
- **Human-in-the-loop**: Pause execution for user approval/editing of tool calls
- **Summarization**: Condense message history when approaching context limits
- **PII redaction**: Pattern matching to identify and redact sensitive information

**Custom Middleware**: Hook into various points in agent loop

### 3. Structured Output Generation

Improved by incorporating into main model ↔ tools loop:
- Reduces latency and cost (eliminates extra LLM call)
- Fine-grained control via tool calling or provider-native structured output

### 4. Standard Content Blocks

Provider-agnostic interfaces for:
- Consistent content types across providers
- Support for reasoning traces, citations, tool calls
- Typed interfaces for complex response structures
- Full backward compatibility

### 5. Simplified Package

Legacy functionality moved to `langchain-classic` for backwards compatibility.

## LangGraph 1.0 Features

### Production-Ready Capabilities

- **Durable state**: Execution state persists automatically, survives server restarts
- **Built-in persistence**: Save and resume workflows at any point
- **Human-in-the-loop patterns**: First-class API support for human review/approval

### Breaking Changes

Only notable change: deprecation of `langgraph.prebuilt` module, with enhanced functionality moved to `langchain.agents`.

## When to Use Each Framework

### Choose LangChain 1.0 for:
- Shipping quickly with standard agent patterns
- Agents that fit the default loop (model → tools → response)
- Middleware-based customization
- Higher-level abstractions over low-level control

### Choose LangGraph 1.0 for:
- Workflows with mixture of deterministic and agentic components
- Long running business process automation
- Sensitive workflows requiring oversight/human in the loop
- Highly custom or complex workflows
- Applications where latency and/or cost need careful control

## Key Quote

> "We rely heavily on the durable runtime that LangGraph provides under the hood to support our agent developments, and the new agent prebuilt and middleware in LangChain 1.0 makes it far more flexible than before."
> — Ankur Bhatt, Head of AI at Rippling

## Adoption

With 90M monthly downloads and powering production applications at Uber, JP Morgan, Blackrock, Cisco, and more.
