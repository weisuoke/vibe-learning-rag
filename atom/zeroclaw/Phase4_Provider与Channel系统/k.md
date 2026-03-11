# Phase 4: Provider 与 Channel 系统 - 知识点列表

> 目标：深入 LLM 集成和多通道消息系统的源码实现
> 学习时长：第 8-9 周
> 前置要求：Phase 1-3 完成

---

## 知识点列表

### 27. Provider Trait 详解
- Provider trait 的方法签名、chat/stream、模型信息、能力查询
- 前端类比：fetch API 的适配器模式
- ZeroClaw 场景：providers/mod.rs 中的 trait 定义和设计考量

### 28. OpenAI-Compatible 实现
- 统一的 OpenAI API 格式、请求构造、响应解析、Header 处理
- 前端类比：axios 封装统一接口
- ZeroClaw 场景：providers/compatible.rs — 最通用的 Provider 实现

### 29. Ollama 本地模型集成
- Ollama API 对接、本地模型管理、性能优化、离线运行
- 前端类比：localhost dev server
- ZeroClaw 场景：providers/ollama.rs — 本地推理的最佳实践

### 30. 流式响应处理
- SSE 解析、chunk 拼接、流式回调、错误恢复
- 前端类比：EventSource / ReadableStream / SSE
- ZeroClaw 场景：Provider 中的 stream 处理逻辑

### 31. Provider 可靠性与降级
- 重试策略、超时控制、fallback Provider、错误分类
- 前端类比：请求重试 + 降级方案（如 SWR stale-while-revalidate）
- ZeroClaw 场景：providers/reliable.rs — 生产环境的可靠性保障

### 32. Channel Trait 详解
- Channel trait 方法签名、消息接收/发送、配对、生命周期
- 前端类比：WebSocket 连接抽象层
- ZeroClaw 场景：channels/mod.rs 中的 trait 定义

### 33. Telegram Channel 实现
- grammy/telegram API、Webhook vs Polling、消息格式、媒体处理
- 前端类比：微信 Bot 开发
- ZeroClaw 场景：channels/telegram.rs 的完整实现解析

### 34. Discord Channel 实现
- Discord Gateway WebSocket、REST API、Slash Commands、消息格式
- 前端类比：Socket.IO 实时通信
- ZeroClaw 场景：channels/discord.rs 的实现与 Telegram 的对比

### 35. 通道路由与消息分发
- 多通道并发监听、消息路由到 Agent、响应回送、错误隔离
- 前端类比：API Gateway 路由 / Express Router
- ZeroClaw 场景：消息从 Channel 入口到 Agent 处理再返回的完整链路

### 36. 自定义 Provider/Channel 开发
- 实现 Trait → 注册到工厂 → 配置启用 → 测试验证
- 前端类比：写一个 Express 中间件
- ZeroClaw 场景：参考 examples/custom_provider.rs 和 custom_channel.rs
