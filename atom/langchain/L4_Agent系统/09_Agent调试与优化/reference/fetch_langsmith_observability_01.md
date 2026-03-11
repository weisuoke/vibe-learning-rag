---
type: fetched_content
source: https://docs.smith.langchain.com/observability
title: LangSmith Observability Guide
fetched_at: 2026-03-06
status: success
author: LangChain
knowledge_point: 09_Agent调试与优化
fetch_tool: grok-mcp
---

# LangSmith Observability（可观测性）指南

## 概述

LangSmith 提供完整的可观测性解决方案，用于追踪、监控和分析 LLM 应用和 Agent 的执行。

## 核心功能模块

### 1. 追踪设置 (Tracing Setup)
- **集成 (Integrations)**: 支持 OpenAI, Anthropic, CrewAI, Vercel AI SDK, Pydantic AI 等
- **手动埋点 (Manual Instrumentation)**: 自定义追踪点
- **线程 (Threads)**: 按对话线程组织追踪

### 2. 配置与故障排除
- **项目与环境设置**: 分环境管理追踪数据
- **成本追踪 (Cost Tracking)**: 监控 API 调用成本
- **高级追踪技巧**: 深度定制追踪行为
- **数据与隐私**: 数据保护策略
- **故障排除指南**: 常见问题解决方案

### 3. 查看与管理追踪
- **过滤追踪**: 按条件筛选追踪记录
- **配置运行预览**: 自定义输入输出显示
- **查询追踪 (SDK)**: 编程式追踪查询
- **对比追踪**: 并排比较不同运行
- **分享追踪**: 公开分享追踪用于协作
- **查看服务端日志**: 追踪关联的后端日志
- **批量导出**: 大规模数据导出

### 4. 自动化 (Automations)
- 规则驱动的自动化工作流
- Webhook 集成
- 在线评估自动触发

### 5. 反馈与评估
- **用户反馈日志**: SDK 收集用户反馈
- **在线评估器**: 自动评估输出质量

### 6. 监控与告警
- 自定义仪表盘
- 性能告警
- 异常检测

### 7. Polly AI 助手
LangSmith 内置的 AI 助手，用于：
- 分析追踪数据
- 提供 AI 驱动的性能洞察
- 辅助调试和优化

## 部署选项
- **云托管**: LangSmith Cloud
- **混合部署**: Hybrid
- **自托管**: Self-hosted
- 所有选项都包含观测性、评估、提示工程和部署功能
