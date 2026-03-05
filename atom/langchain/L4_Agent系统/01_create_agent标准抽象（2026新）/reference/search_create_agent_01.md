---
type: search_result
search_query: LangChain create_agent API 2025 2026 new agent abstraction middleware
search_engine: grok-mcp
searched_at: 2026-02-28
knowledge_point: 01_create_agent标准抽象（2026新）
---

# 搜索结果：LangChain create_agent 2025-2026

## 搜索摘要

LangChain 1.0（2025年10月发布）引入了 create_agent 作为统一的 Agent 创建 API，取代了旧版的 AgentExecutor 和多种 create_xxx_agent 函数。核心创新是 Middleware 系统，提供细粒度的 Agent 行为控制。

## 相关链接

- [Agent Middleware - LangChain Blog](https://blog.langchain.com/agent-middleware) - LangChain 1.0 Middleware 抽象介绍
- [LangChain and LangGraph 1.0](https://blog.langchain.com/langchain-langgraph-1dot0) - 1.0 发布公告
- [LangChain Python Tutorial 2026](https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026) - JetBrains 2026 完整教程
- [LangChain Middlewares](https://medium.com/@ale.garavaglia/langchain-middlewares-lightweight-hooks-for-more-structured-agents-f0abba828934) - 社区 Middleware 实践
- [Build AI agents with LangChain v1](https://www.codecademy.com/article/build-ai-agents-with-langchain-v1) - Codecademy 教程
- [Lessons Learned from Upgrading to LangChain 1.0](https://towardsdatascience.com/lessons-learnt-from-upgrading-to-langchain-1-0-in-production) - 生产环境迁移经验

## 关键信息提取

### 1. create_agent 取代旧版 Agent 创建方式
- LangChain 1.0 用单一 create_agent() 函数替换了 create_react_agent, create_openai_functions_agent, create_tool_calling_agent 等多种函数
- AgentExecutor 正式弃用
- 返回 CompiledStateGraph（基于 LangGraph）

### 2. Middleware 是核心创新
- 提供 Agent 循环内的细粒度控制
- 支持装饰器和类两种定义方式
- 19 个内置 Middleware 覆盖常见场景
- 比直接使用 LangGraph 更轻量

### 3. 社区反馈
- 生产环境迁移需要注意状态管理变化
- Middleware 系统被认为是比 LangGraph 更简单的自定义方式
- 模型字符串标识符（如 "openai:gpt-4"）简化了模型切换

### 4. 迁移路径
- 官方提供了从 AgentExecutor → create_agent 的迁移指南
- 从 create_react_agent → create_agent 的迁移指南
- 主要变化：状态定义、工具注册、执行方式
