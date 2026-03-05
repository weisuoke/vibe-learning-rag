---
type: fetched_content
source: https://blog.langchain.com/agent-middleware
title: Agent Middleware - LangChain Blog
fetched_at: 2026-02-28
status: success
author: LangChain Team
knowledge_point: 01_create_agent标准抽象（2026新）
fetch_tool: grok-mcp
---

# Agent Middleware (2025年9月)

## 核心观点

LangChain 近三年的 Agent 抽象存在共同问题：开发者在非简单场景下缺乏对上下文工程的控制力。LangChain 1.0 引入 Middleware 解决此问题。

## 为什么难以将 Agent 带入生产

答案是 **上下文工程（Context Engineering）**。模型的输入决定输出质量。简单的 Agent 状态和循环适合入门，但随着复杂度增加，需要更多控制：

1. 调整 Agent 状态（不仅仅是 messages）
2. 控制模型输入内容
3. 控制执行步骤序列

## LangChain 的演进历程

过去两年的改进（按时间顺序）：
- 允许用户指定运行时配置
- 允许任意状态 schema
- 允许函数返回 prompt（动态 prompt）
- 允许函数返回消息列表（完全控制）
- 允许 pre model hook（模型调用前）
- 允许 post model hook（模型调用后）
- 允许函数返回模型（动态模型切换）

问题：参数过多，参数间有依赖关系，难以组合。

## Middleware 解决方案

核心 Agent 循环仍由 model node 和 tool node 组成，但 Middleware 可以指定：
- **before_model**: 模型调用前运行，可更新状态或跳转
- **after_model**: 模型调用后运行，可更新状态或跳转
- **modify_model_request**: 模型调用前修改工具、prompt、消息列表、模型、设置、输出格式、工具选择

多个 Middleware 按 Web 服务器中间件模式运行：进入时顺序执行，返回时逆序执行。

Middleware 还可以指定自定义状态 schema 和工具。

## 统一架构

Middleware 帮助统一不同 Agent 架构：supervisor, swarm, bigtool, deepagents, reflection 等都可以用 Middleware 复现。

## 首批内置 Middleware

1. **Human-in-the-loop**: 使用 after_model 提供人工审核
2. **Summarization**: 使用 before_model 在消息累积超过阈值时摘要
3. **Anthropic Prompt Caching**: 使用 modify_model_request 添加缓存标签
