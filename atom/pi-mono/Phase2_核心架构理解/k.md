# Phase 2: 核心架构理解

> 第 3-5 周 | 6个知识点 | 目标：理解 pi-ai 和 pi-agent-core 的设计，掌握运行时机制

---

## 知识点清单

| 编号 | 知识点名称 | 核心概念 | 应用场景 | 学习时长 |
|------|-----------|---------|---------|---------|
| 07 | Pi AI 统一 LLM API 设计 | 多 Provider 抽象、统一消息格式、工具调用标准化 | 多模型集成、API 适配 | 2小时 |
| 08 | Agent Core 运行时机制 | Agent 执行循环、工具注册、状态管理 | Agent 开发、运行时理解 | 2.5小时 |
| 09 | 工具调用与状态管理 | Tool schema、执行流程、状态持久化 | 自定义工具、状态追踪 | 2小时 |
| 10 | 消息队列与流式响应 | Steering/Follow-up message、传递模式、Transport | 实时交互、消息处理 | 1.5小时 |
| 11 | Session 存储与树形结构 | JSONL 格式、树形结构、分支管理 | 历史管理、数据持久化 | 1.5小时 |
| 12 | Compaction 压缩机制 | 自动压缩、压缩策略、手动压缩 | 性能优化、存储管理 | 1小时 |

**总学习时长：** 约 10.5 小时

---

## 学习顺序建议

### 架构理解路径（核心）
按编号顺序学习，理解 pi-mono 的核心设计：
1. **07 - Pi AI 统一 LLM API 设计**（理解多模型抽象）
2. **08 - Agent Core 运行时机制**（理解 Agent 执行）
3. **09 - 工具调用与状态管理**（理解工具系统）
4. **10 - 消息队列与流式响应**（理解消息处理）
5. **11 - Session 存储与树形结构**（理解数据持久化）
6. **12 - Compaction 压缩机制**（理解性能优化）

### 快速理解路径（3小时）
适合想快速理解核心机制的开发者：
1. **08 - Agent Core 运行时机制**（核心）
2. **09 - 工具调用与状态管理**（核心）
3. **11 - Session 存储与树形结构**（核心）

### 源码阅读路径
结合源码深入理解：
1. 07 - 阅读 `packages/pi-ai/src/`
2. 08 - 阅读 `packages/pi-agent-core/src/agent.ts`
3. 09 - 阅读 `packages/pi-agent-core/src/tools.ts`
4. 10 - 阅读 `packages/pi-agent-core/src/messages.ts`
5. 11 - 阅读 `packages/pi-coding-agent/src/session.ts`
6. 12 - 阅读 `packages/pi-coding-agent/src/compaction.ts`

---

## 与 AI Agent 开发的关系

### 架构理解
- **07**：理解如何统一不同 LLM Provider 的 API
- **08**：理解 Agent 的执行循环和生命周期
- **09**：理解工具系统的设计和实现
- **10**：理解消息传递和流式响应机制

### 为后续学习打基础
- **Phase 3**：基于架构理解进行定制化开发
- **Phase 4**：基于核心机制实现高级扩展
- **Phase 5**：基于架构设计构建实战项目

### 构建者视角
- 理解 pi-mono 的设计思想（极简、可扩展）
- 掌握核心组件的工作原理
- 为自定义 Agent 开发打基础

---

## 学习检查清单

完成本 Phase 后，你应该能够：

### Pi AI 统一 API
- [ ] 理解多 Provider 抽象层的设计
- [ ] 理解统一消息格式的转换
- [ ] 理解工具调用的标准化
- [ ] 能够阅读 pi-ai 源码
- [ ] 理解如何添加新 Provider

### Agent Core 运行时
- [ ] 理解 Agent 执行循环
- [ ] 理解工具注册机制
- [ ] 理解状态管理
- [ ] 能够阅读 pi-agent-core 源码
- [ ] 理解 Agent 生命周期

### 工具调用
- [ ] 理解 Tool schema 定义（Zod）
- [ ] 理解工具执行流程
- [ ] 理解状态持久化机制
- [ ] 能够设计自定义工具
- [ ] 理解工具调用的错误处理

### 消息队列
- [ ] 理解 Steering message vs Follow-up message
- [ ] 理解消息传递模式（one-at-a-time vs all）
- [ ] 理解 Transport 选择（SSE、WebSocket）
- [ ] 理解流式响应的实现
- [ ] 能够处理消息队列

### Session 存储
- [ ] 理解 JSONL 文件格式
- [ ] 理解树形结构实现（id、parentId）
- [ ] 理解分支管理
- [ ] 能够解析 Session 文件
- [ ] 理解 Session 的持久化

### Compaction 机制
- [ ] 理解自动压缩触发条件
- [ ] 理解压缩策略
- [ ] 理解手动压缩（/compact）
- [ ] 理解压缩对性能的影响
- [ ] 能够优化 Session 存储

### 实战能力
- [ ] 能够阅读 pi-mono 核心源码
- [ ] 能够解释 Agent 的执行流程
- [ ] 能够设计自定义工具的 schema
- [ ] 能够调试 Agent 运行时问题
- [ ] 能够优化 Session 性能

---

## 双重类比速查表

| Pi-mono 概念 | TypeScript/Node.js 类比 | 日常生活类比 |
|-------------|------------------------|--------------|
| **Pi AI 统一 API** |
| Provider 抽象 | 数据库驱动（Prisma、TypeORM） | 不同品牌的插座 |
| 统一消息格式 | DTO (Data Transfer Object) | 标准化表格 |
| 工具调用标准化 | OpenAPI/Swagger | 统一的接口规范 |
| **Agent Core** |
| Agent 执行循环 | Express 中间件链 | 流水线工人 |
| 工具注册 | 路由注册（Express Router） | 注册服务窗口 |
| 状态管理 | Redux/Zustand | 记事本 |
| **工具调用** |
| Tool schema | Zod schema | 函数签名 |
| 工具执行 | 函数调用 | 使用工具 |
| 状态持久化 | localStorage/sessionStorage | 保存进度 |
| **消息队列** |
| Steering message | HTTP 请求 | 发送指令 |
| Follow-up message | WebSocket 消息 | 追问 |
| Transport | HTTP/WebSocket | 通信方式 |
| 流式响应 | Server-Sent Events | 实时推送 |
| **Session 存储** |
| JSONL 格式 | 追加日志文件（.log） | 日记本逐行记录 |
| 树形结构 | DOM 树 | 家族树 |
| 分支管理 | Git 分支 | 平行宇宙 |
| **Compaction** |
| 自动压缩 | 日志轮转（logrotate） | 整理笔记本 |
| 压缩策略 | 缓存淘汰策略（LRU） | 删除旧记录 |
| 手动压缩 | 手动清理缓存 | 手动整理 |

---

## 核心架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         Pi Coding Agent                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    User Interface                      │  │
│  │  (pi-tui: Terminal UI / pi-web-ui: Web UI)           │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Agent Core                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Agent     │  │   Tools     │  │   State     │  │  │
│  │  │  Executor   │→ │  Registry   │→ │  Manager    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                      Pi AI                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Provider   │  │   Message   │  │    Tool     │  │  │
│  │  │  Adapter    │→ │   Format    │→ │   Schema    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  LLM Providers                         │  │
│  │  [Anthropic] [OpenAI] [GitHub] [Google] [Custom]     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Session Storage                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  JSONL File (Tree Structure)                          │  │
│  │  ┌─────┐  ┌─────┐  ┌─────┐                           │  │
│  │  │ Msg │→ │ Msg │→ │ Msg │  (Linear append)          │  │
│  │  └─────┘  └─────┘  └─────┘                           │  │
│  │     ↓         ↓         ↓                             │  │
│  │  [Branch 1] [Branch 2] [Branch 3]  (Tree structure)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Compaction                           │  │
│  │  (Auto/Manual compression for performance)            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q1: Pi AI 如何统一不同 Provider 的 API？
**A:** 通过 Provider Adapter 将不同 API 格式转换为统一的内部格式，类似于数据库驱动的抽象层。

### Q2: Agent Core 的执行循环是什么？
**A:** Agent 接收消息 → 调用 LLM → 解析工具调用 → 执行工具 → 返回结果 → 继续循环，直到任务完成。

### Q3: 工具调用是同步还是异步？
**A:** 可以是异步的。工具执行可以返回 Promise，Agent Core 会等待 Promise resolve。

### Q4: Session 为什么使用 JSONL 而不是 JSON？
**A:** JSONL 支持追加写入，不需要重写整个文件，性能更好。同时保留完整历史，支持树形结构。

### Q5: Compaction 会丢失历史吗？
**A:** 不会。Compaction 只是优化存储格式，不删除历史消息。可以通过 JSONL 文件完整恢复历史。

### Q6: 如何调试 Agent 运行时问题？
**A:** 使用 VS Code Debugger 附加到 pi 进程，或在源码中添加 console.log，或使用 /debug 命令。

---

## 下一步学习

完成 Phase 2 后，你已经理解了 pi-mono 的核心架构。接下来：

- **Phase 3: 定制化开发** - 学习 Prompt Templates、Skills、Extensions
- **Phase 4: 高级扩展** - 实现自定义 Provider、MCP 集成、Sub-Agents
- **Phase 5: 实战项目** - 基于 pi-mono 构建实际应用

---

## 参考资源

- **源码仓库**: https://github.com/badlogic/pi-mono
- **pi-ai 源码**: https://github.com/badlogic/pi-mono/tree/main/packages/pi-ai
- **pi-agent-core 源码**: https://github.com/badlogic/pi-mono/tree/main/packages/pi-agent-core
- **pi-coding-agent 源码**: https://github.com/badlogic/pi-mono/tree/main/packages/pi-coding-agent
- **Discord 社区**: https://discord.gg/pi-mono

---

**版本：** v1.0
**最后更新：** 2026-02-17
**维护者：** Claude Code
