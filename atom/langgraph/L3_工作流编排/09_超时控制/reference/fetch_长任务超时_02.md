---
type: fetched_content
source: https://github.com/CopilotKit/CopilotKit/issues/2059
title: Bug: Timeout during long running tasks from langgraph agent
fetched_at: 2026-03-07
status: success
author: GitHub issue page (author metadata partially unavailable in fetch output)
knowledge_point: 09_超时控制
fetch_tool: grok-mcp
---

# Bug: Timeout during long running tasks from langgraph agent

## 核心要点提取

### 1. 问题背景
- 在长运行搜索节点中，任务会持续约 2 分钟；
- 期间故意不发送任何状态更新；
- 前端出现 network error，后续状态更新也不再被实时接收。

### 2. 这不是“节点逻辑错了”，而是“交互层 timeout 失配”
- 后端任务最终可能完成；
- 但前端/中间层已经先因为长时间无响应而判定超时；
- 刷新页面后结果又存在，说明状态并未完全丢失。

### 3. 对 LangGraph 超时设计的启示
- 长任务如果要同步挂着等，需要持续发进度；
- 更稳妥的架构是后台执行 + 前端轮询 / 订阅；
- 这类问题不能只靠 `step_timeout` 解决。

### 4. 和官方 durable execution 思路一致
- 官方文档更强调 checkpoint / interrupt / resume；
- 该社区案例从反面证明：如果把长任务强塞进单次同步交互，很容易撞到 UI / 网关 timeout。

## 抓取内容摘要

抓取页面包含：
- 一段完整的长运行复现代码；
- “2 分钟无状态更新”这一关键条件；
- 预期行为与实际行为的对照；
- 相关前端 SDK 版本信息。

## 结论

这份抓取内容最有价值的结论是：**在生产里，timeout 不只是执行器问题，更是“是否持续可观测、是否异步化”的系统设计问题。**

