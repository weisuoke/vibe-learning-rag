---
type: search_result
search_query: LangChain AgentExecutor execution loop 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-28
knowledge_point: 02_AgentExecutor执行循环
---

# 搜索结果：AgentExecutor 执行循环

## 搜索摘要

搜索覆盖了 AgentExecutor 的工作原理、与 LangGraph 的对比、以及 2025-2026 年的迁移趋势。

## 相关链接

### 技术深度文章
- [LangChain Agent Executor Deep Dive](https://www.aurelio.ai/learn/langchain-agent-executor) - 深入探讨 AgentExecutor 工作原理，v0.3 版本自定义代理执行循环
- [LangChain: How an Agent works](https://nakamasato.medium.com/langchain-how-an-agent-works-7dce1569933d) - AgentExecutor 执行逻辑深度剖析，包含伪代码

### 官方文档与公告
- [AgentExecutor 官方参考](https://reference.langchain.com/v0.3/python/langchain/agents/langchain.agents.agent.AgentExecutor.html) - 官方 API 文档
- [LangChain and LangGraph v1.0](https://blog.langchain.com/langchain-langgraph-1dot0) - 2025 年 v1.0 发布，create_agent 取代 AgentExecutor

### 迁移指南
- [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1) - 从 create_react_agent 迁移到 create_agent
- [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1) - AgentExecutor 迁移到 create_agent

### 社区讨论
- [Create_react_agent internal loop](https://forum.langchain.com/t/create-react-agent-internal-loop/641) - 社区讨论内部循环机制
- [LangGraph Agent vs. LangChain Agent](https://medium.com/@seahorse.technologies.sl/langgraph-agent-vs-langchain-agent-63b105d6e5e5) - 对比分析

### 2026 年最新
- [How to Build Agents with LangChain (2026)](https://oneuptime.com/blog/post/2026-02-02-langchain-agents/view) - 2026 年构建指南
- [Complete Guide to LangChain & LangGraph 2025](https://ai.plainenglish.io/the-complete-guide-to-langchain-langgraph-2025-updates-and-production-ready-ai-frameworks-58bdb49a34b6) - 完整指南

## 关键信息提取

### 1. AgentExecutor 执行循环核心
- while 循环调用 agent.plan() 获取下一步动作
- 执行工具获取 observation
- 将 observation 反馈给 LLM
- 直到 AgentFinish 或达到 max_iterations

### 2. 2025-2026 迁移趋势
- AgentExecutor 被标记为 legacy（langchain_classic）
- 新标准：`create_agent`（基于 LangGraph）
- create_agent 提供中间件系统、状态持久化、流式输出
- 底层执行循环逻辑类似，但架构更灵活

### 3. 关键区别
| 特性 | AgentExecutor | create_agent (v1) |
|------|--------------|-------------------|
| 架构 | Chain-based | LangGraph StateGraph |
| 状态管理 | intermediate_steps 列表 | 图状态 + Checkpointer |
| 错误处理 | handle_parsing_errors | 中间件 + retry |
| 流式输出 | 有限支持 | 原生支持 |
| 循环控制 | max_iterations + max_execution_time | 中间件 + tool_call_limit |
| 扩展性 | 子类化 | 中间件组合 |
