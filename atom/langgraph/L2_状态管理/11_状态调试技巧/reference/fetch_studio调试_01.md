---
type: fetched_content
source: https://mem0.ai/blog/visual-ai-agent-debugging-langgraph-studio
title: Getting Started with LangGraph Studio - The Complete Agent Development Tool
fetched_at: 2026-02-27
status: success
author: Taranjeet Singh
knowledge_point: 11_状态调试技巧
fetch_tool: grok-mcp
---

# LangGraph Studio 调试指南摘要

## 核心功能

### 1. Graph Mode（图模式）
- 完整的执行路径可视化
- 中间状态检查
- LangSmith 集成
- 适用于复杂多步代理调试

### 2. Chat Mode（聊天模式）
- 简化的对话界面
- 适用于聊天代理测试

### 3. 实时调试与状态管理
- **AgentState 中断**：暂停代理执行，查看所有决策
- **时间旅行**：步进执行历史，查看任意时刻的 AgentState
- **状态编辑**：在节点运行前后编辑状态
- **热重载**：代码修改后自动检测，无需重启

### 4. 开发工作流集成
- 自动检测代码文件变更
- 代码编辑器与代理运行时的桥梁
- LangSmith 完整可观测性

## 功能对比表

| 功能 | 描述 | 最佳场景 |
|------|------|----------|
| Graph Mode | 详细执行路径可视化 | 复杂多步代理调试 |
| Chat Mode | 简化对话界面 | 聊天代理测试 |
| Interrupt | 节点执行前后编辑状态 | 微调代理行为 |
| Hot Reload | 自动检测代码变更 | 快速迭代开发 |
| LangSmith | 完整可观测性和监控 | 生产性能分析 |

## 关键洞察
- LangGraph Studio 是首个专为 AI 代理开发设计的 IDE
- 传统 IDE 无法有效处理代理的非确定性执行流
- 状态管理可视化是代理调试的核心需求
- 时间旅行调试可以回放场景并修改参数
