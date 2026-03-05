---
type: fetched_content
source: https://blog.langchain.com/langchain-langgraph-1dot0
title: LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones
fetched_at: 2026-02-28
status: success
author: Sydney Runkle, LangChain OSS team
knowledge_point: 01_create_agent标准抽象（2026新）
fetch_tool: grok-mcp
---

# LangChain 1.0 发布公告 (2025年10月22日)

## 核心变化

### 1. create_agent 抽象
- 最快的 Agent 构建方式
- 基于 LangGraph 运行时
- 取代 langgraph.prebuilt 中的 create_react_agent

### 2. Middleware 系统
- 定义一组钩子自定义 Agent 循环行为
- 内置 Middleware：Human-in-the-loop、Summarization、PII redaction
- 支持自定义 Middleware

### 3. 结构化输出改进
- 集成到主 model <-> tools 循环中
- 减少延迟和成本（消除额外 LLM 调用）
- 支持 ToolStrategy 和 ProviderStrategy

### 4. 标准内容块 (Standard Content Blocks)
- 跨提供商一致的内容类型
- 支持推理轨迹、引用、工具调用
- 完全向后兼容

### 5. 包精简
- 旧功能移至 langchain-classic
- 核心聚焦 Agent 相关抽象
- Python 3.10+ 要求

## 何时使用 LangChain vs LangGraph

### LangChain 1.0 适用于：
- 快速发布标准 Agent 模式
- 符合默认循环的 Agent（model → tools → response）
- 基于 Middleware 的自定义
- 高层抽象优先

### LangGraph 1.0 适用于：
- 确定性 + 代理性混合工作流
- 长时间运行的业务流程自动化
- 需要更多监督/人工审核的敏感工作流
- 高度自定义或复杂工作流
- 需要精确控制延迟和成本的应用

## 关键引用

> "We rely heavily on the durable runtime that LangGraph provides under the hood to support our agent developments, and the new agent prebuilt and middleware in LangChain 1.0 makes it far more flexible than before." – Ankur Bhatt, Head of AI at Rippling

## 安装

```bash
uv pip install --upgrade langchain
uv pip install langchain-classic  # 向后兼容
```
