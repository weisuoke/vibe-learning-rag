---
type: fetched_content
source: https://blog.langchain.com/planning-agents/
title: Plan-and-Execute Agents - LangChain Blog
fetched_at: 2026-03-06
status: success
author: LangChain Team
knowledge_point: 08_多步推理与规划
fetch_tool: grok-mcp
---

# Plan-and-Execute Agents

Plan and execute agents promise faster, cheaper, and more performant task execution over previous agent designs. Learn how to build 3 types of planning agents in LangGraph.

5 min read Feb 13, 2024

### Links

- Plan-and-execute (Python: https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb, JS)
- LLMCompiler (Python: https://github.com/langchain-ai/langgraph/blob/main/examples/llm-compiler/LLMCompiler.ipynb)
- ReWOO (Python: https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb)

## 三大优势

⏰ **更快**：不需要在每次动作后咨询大模型。子任务可以不需要额外 LLM 调用（或调用轻量级 LLM）。

💸 **更省**：子任务可以使用更小的领域专用模型。大模型仅用于（重新）规划和生成最终响应。

🏆 **更好**：强制规划器显式"思考"完成整个任务所需的所有步骤。将问题细分也允许更聚焦的任务执行。

## Background - ReAct 的局限

ReAct 循环: Thought → Act → Observation (repeat)

两个主要缺点：
1. 每次工具调用都需要一次 LLM 调用
2. LLM 一次只规划一个子问题，可能导致次优轨迹

## Plan-And-Execute 架构

基于 Wang 等人的 Plan-and-Solve Prompting 论文和 BabyAGI 项目。

两个基本组件：
1. **Planner (规划器)**：提示 LLM 生成多步计划来完成大任务
2. **Executor (执行器)**：接受用户查询和计划中的一步，调用工具完成任务

执行完成后，Agent 被再次调用进行 re-planning，决定是直接响应还是生成后续计划。

优点：避免在每次工具调用时调用大型规划 LLM
限制：仍受串行工具调用限制，每个任务仍使用 LLM

## ReWOO (Reasoning WithOut Observations)

基于 Xu 等人论文。关键创新：允许变量赋值。

Planner 生成交替的 "Plan" 和 "E#" 行：
```
Plan: I need to know the teams playing in the superbowl
E1: Search[Who is competing in the superbowl?]
Plan: I need to know the quarterbacks for each team
E2: LLM[Quarterback for the first team of #E1]
E3: LLM[Quarter back for the second team of #E1]
E4: Search[Stats for #E2]
E5: Search[Stats for #E3]
```

三个组件：
- **Planner**: 生成带变量引用的计划
- **Worker**: 循环执行每个任务，替换变量
- **Solver**: 整合所有输出生成最终答案

优点：每个任务只有必要的上下文
限制：仍依赖顺序执行

## LLMCompiler

基于 Kim 等人论文。进一步提升执行速度。

三个主要组件：
1. **Planner**: 流式输出 DAG 任务图。每个任务包含工具、参数和依赖列表
2. **Task Fetching Unit**: 依赖满足后立即调度执行。支持并行执行
3. **Joiner**: 动态决定重新规划或完成

关键加速思想：
- Planner 输出是流式的
- Task Fetching Unit 在依赖满足后立即调度
- 任务参数可以是变量（如 `search("${1}")`）

论文声称 3.6x 速度提升。

## 结论

这三种架构是"plan-and-execute"设计模式的典型代表，将 LLM 驱动的"规划器"与工具执行运行时分离。
