---
type: fetched_content
source: https://www.getmaxim.ai/articles/retries-fallbacks-and-circuit-breakers-in-llm-apps-a-production-guide/
title: "Retries, Fallbacks, and Circuit Breakers in LLM Apps: A Production Guide"
fetched_at: 2026-03-07
status: success
author: Maxim/Bifrost
knowledge_point: 08_错误处理与重试
fetch_tool: grok-mcp
---

# LLM 应用的重试、回退与断路器生产指南

## 核心要点提取

### 1. 何时重试（可重试的 HTTP 状态码）
- **429** (Rate Limit Exceeded): 速率限制，退避后自动恢复
- **500** (Internal Server Error): 临时服务端问题
- **502** (Bad Gateway): 代理/负载均衡器问题
- **503** (Service Unavailable): 临时容量或维护
- **504** (Gateway Timeout): 请求超时

### 2. 不可重试的错误
- **400** (Bad Request): 请求格式错误
- **401** (Unauthorized): 认证无效
- **403** (Forbidden): 权限不足
- **404** (Not Found): 端点不存在

### 3. Fallback 链设计
```
Primary: OpenAI GPT-4
Fallback 1: Anthropic Claude
Fallback 2: Google Gemini
Fallback 3: Azure OpenAI
```
- 每个提供者按顺序尝试，直到成功或全部耗尽
- 也可以在同一提供者内降级模型

### 4. 断路器三种状态
- **Closed（关闭）**：正常运行，监控失败率
- **Open（打开）**：阻止请求，直接走 fallback
- **Half-Open（半开）**：允许少量测试请求判断恢复

### 5. 断路器配置参数
- **Failure threshold**: 触发断路的失败次数/比例
- **Timeout period**: 断路器保持打开的时长
- **Success threshold**: 从半开恢复到关闭需要的成功次数
- **Rolling window**: 计算失败率的时间窗口

### 6. 三种模式协同工作流程
```
请求到达
  → 主提供者尝试
    → 失败？重试（指数退避）
      → 所有重试失败？Fallback 提供者 1
        → 失败？Fallback 提供者 2
          → 全部失败？返回错误
断路器在整个过程中监控失败率
```

### 7. 监控指标
**重试指标：** 重试次数/成功率/延迟占比
**Fallback 指标：** 激活频率/各提供者处理比例/质量差异
**断路器指标：** 状态转换/各状态停留时间/半开测试成功率

### 8. Python 断路器库推荐
- PyBreaker: https://github.com/danielfm/pybreaker
