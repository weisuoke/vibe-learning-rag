# ZeroClaw 学习路径设计文档

> 为前端/TypeScript 开发者设计的 ZeroClaw（Rust AI Agent 运行时）完整学习路径

---

## 项目背景

### 什么是 ZeroClaw

ZeroClaw 是一个 **100% Rust** 编写的超轻量 AI Agent 运行时，是 OpenClaw 的高性能替代方案。

**核心特性（2026 年 3 月，v0.1.7）：**
- 单一二进制文件 ~3.4-8.8MB，运行内存 <5MB
- 冷启动 <10ms（比 OpenClaw 快 400 倍）
- 可在 $10 硬件上运行（ARM/x86/RISC-V/Android）
- Trait 驱动架构：Provider/Tool/Memory/Channel 全部可插拔
- 22+ LLM Provider、14+ 消息通道、44+ 内置工具
- 混合检索内存（SQLite 向量余弦 + FTS5 BM25）
- Secure-by-default：沙箱、配对认证、命令白名单

**源码地址：** `sourcecode/zeroclaw/`
**GitHub：** https://github.com/zeroclaw-labs/zeroclaw

### 学习者画像

- **技术背景**：前端/TypeScript 开发者，熟悉 React、Express、npm 生态
- **Rust 基础**：零基础
- **学习目标**：使用 ZeroClaw + 阅读源码 + 贡献代码
- **学习原则**：20% 核心知识 → 80% 能力
- **学习时长**：3-4 个月（16 周）

---

## 学习路径总览

### 8 个 Phase，70 个知识点

| Phase | 主题 | 知识点数 | 学习时长 | 里程碑 |
|-------|------|---------|---------|--------|
| Phase 1 | Rust 速成基础 | 8 | 第 1-2 周 | 能读懂 ZeroClaw 源码中的 Rust |
| Phase 2 | ZeroClaw 快速上手 | 8 | 第 3-4 周 | 本地运行 Agent + Telegram 对话 |
| Phase 3 | 核心架构理解 | 10 | 第 5-7 周 | 说清消息全链路，理解 Trait 设计 |
| Phase 4 | Provider 与 Channel 系统 | 10 | 第 8-9 周 | 读懂任意 Provider/Channel 实现 |
| Phase 5 | 工具与内存系统 | 10 | 第 10-11 周 | 理解工具调度和混合检索引擎 |
| Phase 6 | 安全与运维 | 8 | 第 12-13 周 | 生产部署 ZeroClaw 实例 |
| Phase 7 | 扩展开发实战 | 8 | 第 14-15 周 | 独立开发 Provider/Channel/Tool |
| Phase 8 | 高级主题与社区贡献 | 8 | 第 16 周 | 提交 PR，成为社区贡献者 |

### 设计原则

1. **Rust 按需学习**：不教"完整的 Rust"，只教"读懂 ZeroClaw 源码需要的 Rust"
2. **先用后读**：Phase 2 先会用，Phase 3 再看源码
3. **前端类比贯穿**：所有 Rust 和 ZeroClaw 概念用前端类比解释
4. **贡献能力递进**：Phase 4（自定义 Provider/Channel）→ Phase 5（自定义 Tool）→ Phase 7（完整实战）→ Phase 8（提交 PR）
5. **连接已有知识**：与 RAG 学习路径、Python 后端知识关联

---

## Phase 详细设计

### Phase 1: Rust 速成基础（第 1-2 周）

**目标：** 掌握 20% 的 Rust 核心知识，能读懂 ZeroClaw 源码

**设计依据：** 基于 ZeroClaw 源码中高频出现的 Rust 模式：
- `#[async_trait]` + `Box<dyn Provider>` — 每个模块都用
- `Result<T, anyhow::Error>` — 错误处理无处不在
- `Arc<dyn Memory>` / `Vec<Box<dyn Tool>>` — 组件注入
- `tokio::spawn` + `.await` — 异步运行时
- `serde::Serialize/Deserialize` — 配置和消息序列化
- `struct` + `impl` + Builder pattern — AgentBuilder 等

| 编号 | 知识点 | 前端类比 | 为什么 ZeroClaw 需要 |
|------|--------|---------|---------------------|
| 01 | 所有权与借用 | JS 值传递 vs 引用传递 | 理解为什么函数签名有 `&self`、`&mut self` |
| 02 | Struct 与 Enum | TS interface/type | Agent、Message、Config 全是 struct |
| 03 | Trait 与泛型 | TS interface + generics | Provider/Tool/Memory 全靠 Trait 定义 |
| 04 | 错误处理 Result/Option | TS try/catch + optional | 每个函数都返回 `Result<T>` |
| 05 | 动态分发与 Trait Object | TS 多态 + 依赖注入 | `Box<dyn Provider>`、`Arc<dyn Memory>` |
| 06 | async/await 与 Tokio | JS Promise + Event Loop | 整个运行时基于 Tokio async |
| 07 | Cargo 与模块系统 | npm + ES modules | 理解 Cargo.toml、mod.rs、use 语句 |
| 08 | 常用库速查（serde/anyhow/clap） | axios/yargs/zod | ZeroClaw 依赖这些库的 API |

### Phase 2: ZeroClaw 快速上手（第 3-4 周）

**目标：** 作为用户掌握 ZeroClaw 的安装、配置和基本使用

| 编号 | 知识点 | 前端类比 | 核心目标 |
|------|--------|---------|---------|
| 09 | ZeroClaw 安装与环境配置 | npm install -g | cargo install、一键脚本、环境变量 |
| 10 | Onboarding Wizard 与首次配置 | create-react-app 脚手架 | Provider 选择、API Key 配置、config.toml |
| 11 | CLI 基础命令 | npm scripts | agent、daemon、config、channel 子命令 |
| 12 | 第一个 Agent 对话 | Postman 发请求 | CLI 模式下与 Agent 交互 |
| 13 | Provider 配置与切换 | 切换 API 后端地址 | Ollama 本地 / OpenRouter / Anthropic |
| 14 | Channel 配对（Telegram 实战） | OAuth 扫码授权 | 配对码机制、Webhook 设置 |
| 15 | Gateway 启动与管理 | Express 服务器启动 | 端口配置、Webhook 路由、健康检查 |
| 16 | 基础故障排查 | Chrome DevTools 调试 | 日志查看、常见错误、doctor 命令 |

### Phase 3: 核心架构理解（第 5-7 周）

**目标：** 理解 ZeroClaw 的顶层设计哲学和 Agent 核心循环

| 编号 | 知识点 | 前端类比 | 源码对应 |
|------|--------|---------|---------|
| 17 | 源码目录结构总览 | monorepo 结构 | src/ 下 30+ 模块的分层逻辑 |
| 18 | Trait 驱动架构设计 | 依赖注入 + 接口解耦 | Provider/Tool/Memory/Channel 四大 Trait |
| 19 | AgentBuilder 与构建器模式 | React Context Provider 配置 | agent.rs 中的 Builder pattern |
| 20 | Agent 核心循环（ReAct） | Redux dispatch → reducer 循环 | 提示→LLM调用→工具执行→更新→迭代 |
| 21 | 消息模型与数据流 | Redux action/state 流 | Message struct、Role enum、序列化 |
| 22 | 配置系统详解（config.toml） | next.config.js | 配置加载、Schema 验证、热重载 |
| 23 | 模块注册与工厂模式 | Plugin 注册表 | Provider/Channel/Tool 的动态注册 |
| 24 | 错误传播与 anyhow | Error Boundary 链 | Result 链式传播、context() 附加信息 |
| 25 | 日志与可观测性 | console.log + 性能监控 | tracing crate、Observer trait |
| 26 | 入口分析：main.rs 解读 | index.ts 启动流程 | CLI 解析 → 配置加载 → Agent 启动 |

### Phase 4: Provider 与 Channel 系统（第 8-9 周）

**目标：** 深入 LLM 集成和多通道消息系统的源码实现

| 编号 | 知识点 | 前端类比 | 源码对应 |
|------|--------|---------|---------|
| 27 | Provider Trait 详解 | fetch API 的适配器模式 | providers/mod.rs 中的 trait 定义 |
| 28 | OpenAI-Compatible 实现 | axios 封装统一接口 | providers/compatible.rs |
| 29 | Ollama 本地模型集成 | localhost dev server | providers/ollama.rs |
| 30 | 流式响应处理 | EventSource / SSE | providers/ 中的 stream 处理逻辑 |
| 31 | Provider 可靠性与降级 | 请求重试 + fallback | providers/reliable.rs |
| 32 | Channel Trait 详解 | WebSocket 连接抽象 | channels/mod.rs 中的 trait 定义 |
| 33 | Telegram Channel 实现 | 微信 Bot 开发 | channels/telegram.rs |
| 34 | Discord Channel 实现 | Socket.IO 实时通信 | channels/discord.rs |
| 35 | 通道路由与消息分发 | API Gateway 路由 | 消息从 Channel → Agent 的分发逻辑 |
| 36 | 自定义 Provider/Channel 开发 | 写一个 Express 中间件 | 实现 Trait → 注册 → 配置启用 |

### Phase 5: 工具与内存系统（第 10-11 周）

**目标：** 理解 Tool 调度和 Memory 混合检索引擎

| 编号 | 知识点 | 前端类比 | 源码对应 |
|------|--------|---------|---------|
| 37 | Tool Trait 详解 | React Hook 接口规范 | tools/mod.rs |
| 38 | ToolDispatcher 与工具调度 | Redux middleware dispatch | 工具调用解析→路由→执行→结果格式化 |
| 39 | Shell 工具实现 | child_process.exec | tools/shell.rs |
| 40 | 文件操作工具 | fs.readFile / writeFile | tools/file_read.rs + file_write.rs |
| 41 | HTTP 与 Browser 工具 | fetch + Puppeteer | tools/http_request.rs + tools/browser.rs |
| 42 | 自定义 Tool 开发 | 写一个自定义 Hook | 实现 Tool trait → 注册 → 测试 |
| 43 | Memory Trait 详解 | localStorage + 搜索引擎 | memory/mod.rs |
| 44 | SQLite 混合检索引擎 | Algolia 全文+向量搜索 | 向量余弦 + FTS5 BM25，0.7/0.3 权重 |
| 45 | Markdown Memory 后端 | 文件系统缓存 | memory/markdown/ |
| 46 | Memory 上下文管理与裁剪 | 虚拟列表按需加载 | 历史消息裁剪、上下文窗口管理 |

### Phase 6: 安全与运维（第 12-13 周）

**目标：** 理解安全机制，掌握生产部署能力

| 编号 | 知识点 | 前端类比 | 源码对应 |
|------|--------|---------|---------|
| 47 | SecurityPolicy Trait 与安全模型 | CORS + CSP 安全策略 | security/ |
| 48 | 配对认证机制 | 扫码登录 / 2FA | identity.rs |
| 49 | 沙箱执行（Native/Docker） | iframe sandbox | runtime/ |
| 50 | Secret Store 与密钥管理 | .env + Vault | 加密存储、环境变量 |
| 51 | Gateway 架构与 Webhook | Express + nginx 反向代理 | gateway/ |
| 52 | Daemon 守护进程 | pm2 进程管理 | daemon/ |
| 53 | Cron 调度系统 | node-cron 定时任务 | cron/ |
| 54 | 隧道与远程访问 | ngrok 内网穿透 | tunnel/ |

### Phase 7: 扩展开发实战（第 14-15 周）

**目标：** 动手实现完整的扩展，达到贡献代码的能力

| 编号 | 知识点 | 前端类比 | 核心目标 |
|------|--------|---------|---------|
| 55 | 开发环境搭建（从源码） | git clone + pnpm install | Cargo build、feature flags、测试运行 |
| 56 | 测试体系与质量保证 | Vitest + Playwright | 单元测试、集成测试、fuzz 测试 |
| 57 | 完整实战：自定义 Provider | 封装一个 API SDK | 从零实现一个 LLM Provider（含流式） |
| 58 | 完整实战：自定义 Channel | 写一个 WebSocket 服务 | 从零实现一个消息通道 |
| 59 | 完整实战：自定义 Tool | 写一个 CLI 插件 | 从零实现一个复杂工具 |
| 60 | 完整实战：自定义 Memory 后端 | 写一个缓存适配器 | 实现 Memory trait 对接 Redis/PostgreSQL |
| 61 | Skill 系统与 SOP | npm scripts 编排 | skills/ + sop/ |
| 62 | Python 伴侣包开发 | npm 包发布 | python/ — LangGraph 工具调用集成 |

### Phase 8: 高级主题与社区贡献（第 16 周）

**目标：** 覆盖进阶功能，成为社区贡献者

| 编号 | 知识点 | 前端类比 | 源码对应 |
|------|--------|---------|---------|
| 63 | AIEOS 身份系统 | 用户 Profile + 主题切换 | identity.rs |
| 64 | RAG 能力集成 | 搜索引擎 + 知识库 | rag/ |
| 65 | Web Dashboard（React） | 你的主场！ | web/ — React + TypeScript 前端 |
| 66 | 硬件与固件集成 | IoT 设备控制 | peripherals/ + firmware/ |
| 67 | 性能优化与 Benchmark | Lighthouse 性能评分 | benches/ |
| 68 | CI/CD 与发布流程 | GitHub Actions + npm publish | scripts/ |
| 69 | PR 工作流与代码审查 | GitHub PR 流程 | CONTRIBUTING.md |
| 70 | 社区参与与生态贡献 | 开源项目维护 | Issue 分类、RFC 流程 |

---

## 知识关联

### 与 RAG 学习路径的关联

| ZeroClaw 知识点 | RAG 学习路径关联 |
|-----------------|-----------------|
| 44. SQLite 混合检索引擎 | L3_RAG核心流程/05_检索器设计 |
| 64. RAG 能力集成 | L3_RAG核心流程 全部 |
| 43. Memory Trait | L3_RAG核心流程/04_向量存储 |
| 30. 流式响应处理 | L2_LLM核心/02_大模型API调用 |

### 与 OpenClaw 学习路径的对比

| 维度 | OpenClaw | ZeroClaw |
|------|----------|----------|
| 语言 | TypeScript | Rust |
| 前置知识 | Phase 1 直接上手 | Phase 1 需要 Rust 速成 |
| 架构重点 | Gateway + Agent + Extension | Trait 驱动 + Agent 循环 |
| 贡献入口 | Extension/Plugin | 实现 Trait（Provider/Channel/Tool） |
| 总知识点 | 94 个 | 70 个（20%核心原则精简） |

---

## 实施计划

1. 创建 `atom/zeroclaw/CLAUDE_ZEROCLAW.md` — ZeroClaw 特定配置
2. 创建 8 个 Phase 目录 + `k.md` 知识点列表
3. 按 Phase 顺序逐步生成知识点文档

---

**版本：** v1.0
**创建日期：** 2026-03-09
**维护者：** Claude Code
