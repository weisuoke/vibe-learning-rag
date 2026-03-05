# 原子化知识点生成规范 - OpenClaw 开发专用

> 本文档定义了为 OpenClaw 多通道 AI 网关学习项目生成原子化知识点文档的标准和要求

---

## 文档概述

**项目目标：** 为 OpenClaw 开发学习构建完整的原子化知识体系

**核心理念：**
- **原子化**：每个知识点独立完整，可独立学习
- **全面覆盖**：知识点包含多个子概念时，全部详细讲解，不遗漏
- **实战导向**：所有知识点都要联系 OpenClaw 开发的实际应用
- **初学者友好**：假设有 TypeScript/Node.js 基础，用简单语言和丰富类比
- **速成高效**：抓住20%核心解决80%问题
- **双重类比**：同时提供后端开发类比 + 日常生活类比
- **TypeScript 优先**：所有代码示例使用 TypeScript/JavaScript
- **多通道架构**：体现 OpenClaw 的多通道消息系统设计理念

---

## 模板引用

本文档基于通用原子化知识点模板：**`prompt/atom_template.md`**

OpenClaw 开发的特殊要求在下方 **OpenClaw 开发特定配置** 章节中定义。

---

## 生成流程

### 第一步：确认输入信息

在开始生成前，确认以下信息：

1. **知识点名称**：从 `atom/openclaw/[Phase]/k.md` 中获取
2. **Phase 目录**：如 `Phase1_快速上手与环境搭建`、`Phase2_核心架构深入理解` 等
3. **目标受众**：有 TypeScript/Node.js 基础的开发者
4. **文件位置**：`atom/openclaw/[Phase]/[编号]_[知识点名称]/`

### 第二步：读取模板

**通用模板：** `prompt/atom_template.md` - 定义10个维度的标准结构

**10个必需维度：**
1. 【30字核心】
2. 【第一性原理】
3. 【3个核心概念】
4. 【最小可用】
5. 【双重类比】
6. 【反直觉点】
7. 【实战代码】
8. 【面试必问】
9. 【化骨绵掌】
10. 【一句话总结】

### 第三步：按规范生成内容

参考 `prompt/atom_template.md` 的详细规范，结合下方的 OpenClaw 开发特定配置生成内容。

### 第四步：质量检查

使用 `prompt/atom_template.md` 中的检查清单验证质量。

---

## OpenClaw 开发特定配置

### 应用场景强调

**每个部分都要联系 OpenClaw 开发实际应用：**
- ✅ 这个知识在 OpenClaw 开发中如何体现？
- ✅ 为什么 OpenClaw 开发需要这个？
- ✅ 实际场景举例（多通道消息、Gateway 路由、Agent 集成、浏览器自动化、语音集成）

**重点强调：**
- Gateway 架构与消息路由
- 多通道抽象与集成
- Agent 系统（pi-agent-core 集成）
- 扩展系统（Extension、Plugin SDK、Hook）
- 跨平台支持（Mac、iOS、Android、Web）
- 实战案例（前端、后端、全栈）

### OpenClaw 类比对照表

在【双重类比】维度中，优先使用以下类比：

| OpenClaw 概念 | 后端开发类比 | 日常生活类比 |
|--------------|-------------|--------------|
| **核心架构** |
| Gateway | Nginx 反向代理 | 邮局分拣中心 |
| Channel 抽象 | 数据库驱动接口 | 不同品牌的插座 |
| 消息路由 | API Gateway 路由 | 快递分拣系统 |
| Agent 集成 | 微服务调用 | 外包团队 |
| Session 管理 | Redis Session | 聊天记录本 |
| JSONL 存储 | 日志文件 | 日记本逐行记录 |
| **通道系统** |
| WhatsApp 集成 | WebSocket 连接 | 微信聊天 |
| Telegram Bot | HTTP Webhook | 机器人客服 |
| Discord 集成 | WebSocket + REST | 游戏语音频道 |
| 通道配对 | OAuth 认证 | 扫码登录 |
| 消息队列 | RabbitMQ | 排队系统 |
| **扩展系统** |
| Extension | Express 中间件 | 浏览器扩展 |
| Plugin SDK | npm 包 | 工具箱 |
| Hook 系统 | 生命周期钩子 | 事件监听器 |
| Skill | Lambda 函数 | 技能卡片 |
| **高级功能** |
| Browser 自动化 | Puppeteer | 自动化测试 |
| Canvas | WebSocket 实时渲染 | 远程桌面 |
| Voice 集成 | 语音识别 API | 语音助手 |
| Cron 调度 | Cron Job | 定时闹钟 |
| ACP | gRPC 协议 | 标准接口 |

### 推荐库列表

在【实战代码】维度中，优先使用以下库：

| 用途 | 推荐库 |
|------|--------|
| **开发环境** | Node.js 22+, pnpm, TypeScript 5.0+ |
| **包管理** | pnpm workspaces |
| **测试** | Vitest, Playwright |
| **类型检查** | TypeScript, tsc |
| **代码规范** | Oxlint, Oxfmt |
| **调试** | VS Code Debugger |
| **Agent 系统** | @mariozechner/pi-agent-core, @mariozechner/pi-ai |
| **通道集成** | grammy (Telegram), @slack/bolt (Slack), discord.js (Discord) |
| **消息处理** | @whiskeysockets/baileys (WhatsApp) |
| **浏览器自动化** | playwright-core |
| **Web 框架** | Express, Lit |
| **CLI 工具** | Commander.js, @clack/prompts |
| **Schema 验证** | @sinclair/typebox, zod |

### OpenClaw 常见误区

在【反直觉点】维度中，可参考以下常见误区：

**架构误区：**
- "Gateway 只是简单的消息转发"（Gateway 是完整的控制平面）
- "每个通道都需要独立实现"（通道抽象层统一接口）
- "Agent 和 Gateway 是分离的"（紧密集成）
- "Extension 需要修改核心代码"（Plugin SDK 提供标准接口）

**使用误区：**
- "只能用 Anthropic 模型"（支持多种 Provider）
- "必须使用所有通道"（按需配置）
- "Session 存储在内存中"（JSONL 文件持久化）
- "配置需要重启 Gateway"（支持热重载）

**开发误区：**
- "Extension 必须用 JavaScript"（TypeScript 优先）
- "自定义通道很复杂"（Channel 接口简化开发）
- "测试需要真实通道"（Mock 和 Docker 测试）
- "部署只能用 npm"（支持 Docker、源码等多种方式）

---

## TypeScript/Node.js 环境配置

### 环境要求

- **Node.js 版本**: 22+ (推荐 22.12.0+)
- **包管理器**: pnpm (推荐 10.23.0+)
- **TypeScript**: 5.9+
- **操作系统**: macOS, Linux, Windows (WSL2 强烈推荐)

### 快速开始

```bash
# 1. 全局安装 OpenClaw
npm install -g openclaw@latest
# 或: pnpm add -g openclaw@latest

# 2. 运行 Onboarding Wizard
openclaw onboard --install-daemon

# 3. 启动 Gateway
openclaw gateway --port 18789 --verbose

# 4. 发送消息测试
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# 5. Agent 交互
openclaw agent --message "Ship checklist" --thinking high
```

### 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 2. 安装依赖
pnpm install

# 3. 构建所有包
pnpm build

# 4. 运行测试
pnpm test

# 5. 从源码运行
pnpm openclaw gateway --port 18789
```

### 可用的库

所有代码示例可以使用以下库：

| 用途 | 库名 |
|------|------|
| **Agent 系统** | `@mariozechner/pi-agent-core`, `@mariozechner/pi-ai`, `@mariozechner/pi-coding-agent` |
| **通道集成** | `grammy`, `@slack/bolt`, `discord.js`, `@whiskeysockets/baileys` |
| **浏览器自动化** | `playwright-core` |
| **Web 框架** | `express`, `lit` |
| **CLI 工具** | `commander`, `@clack/prompts` |
| **Schema 验证** | `@sinclair/typebox`, `zod` |
| **工具** | `dotenv`, `chalk`, `tslog` |

### 环境管理

```bash
# 添加新依赖
pnpm add <package-name>

# 添加开发依赖
pnpm add -D <package-name>

# 更新依赖
pnpm update
```

### 配置 API 密钥

1. 设置环境变量：
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...
   ```

2. 或使用 OpenClaw 配置：
   ```bash
   openclaw config set provider anthropic
   openclaw config set model claude-opus-4
   ```

3. 或在项目中配置 `~/.openclaw/config/settings.json`：
   ```json
   {
     "provider": "anthropic",
     "model": "claude-opus-4"
   }
   ```

---

## 文件组织规范

### 文件命名

**格式：** `[编号]_[知识点名称]/`（目录形式）

**示例：**
```
atom/openclaw/Phase1_快速上手与环境搭建/01_OpenClaw安装与配置/
atom/openclaw/Phase1_快速上手与环境搭建/02_Onboarding_Wizard详解/
atom/openclaw/Phase2_核心架构深入理解/09_Gateway架构设计/
atom/openclaw/Phase3_多通道消息系统/19_通道抽象层设计/
```

**编号规则：**
- 按学习顺序编号（01, 02, 03...）
- 反映知识点的依赖关系
- 与 `k.md` 中的顺序一致
- 跨 Phase 连续编号（Phase1: 01-08, Phase2: 09-18, Phase3: 19-30...）

### 目录结构

```
atom/
└── openclaw/                               # OpenClaw 学习路径（9个阶段）
    ├── CLAUDE_OPENCLAW.md                  # OpenClaw 特定配置文档
    ├── Phase1_快速上手与环境搭建/
    │   ├── k.md                            # 知识点列表
    │   ├── 01_OpenClaw安装与配置/
    │   │   ├── 00_概览.md
    │   │   ├── 01_30字核心.md
    │   │   ├── 02_第一性原理.md
    │   │   ├── 03_核心概念_01_npm全局安装.md
    │   │   ├── 03_核心概念_02_环境要求.md
    │   │   ├── 03_核心概念_03_配置文件.md
    │   │   ├── 04_最小可用.md
    │   │   ├── 05_双重类比.md
    │   │   ├── 06_反直觉点.md
    │   │   ├── 07_实战代码_01_基础安装.md
    │   │   ├── 07_实战代码_02_配置Gateway.md
    │   │   ├── 07_实战代码_03_测试运行.md
    │   │   ├── 08_面试必问.md
    │   │   ├── 09_化骨绵掌.md
    │   │   └── 10_一句话总结.md
    │   ├── 02_Onboarding_Wizard详解/
    │   ├── 03_CLI基础命令/
    │   ├── 04_Gateway启动与管理/
    │   ├── 05_第一个消息发送/
    │   ├── 06_基础故障排查/
    │   ├── 07_开发环境搭建（从源码）/
    │   └── 08_源码编译与调试/
    │
    ├── Phase2_核心架构深入理解/
    │   ├── k.md
    │   ├── 09_Gateway架构设计/
    │   ├── 10_Agent系统集成（Pi_agent_core）/
    │   ├── 11_配置系统详解/
    │   ├── 12_会话管理机制/
    │   ├── 13_守护进程管理/
    │   ├── 14_日志系统/
    │   ├── 15_错误处理机制/
    │   ├── 16_性能监控/
    │   ├── 17_协议设计（Protocol）/
    │   └── 18_依赖注入与模块化/
    │
    ├── Phase3_多通道消息系统/
    │   ├── k.md
    │   ├── 19_通道抽象层设计/
    │   ├── 20_通道路由机制/
    │   ├── 21_WhatsApp集成/
    │   ├── 22_Telegram集成/
    │   ├── 23_Slack集成/
    │   ├── 24_Discord集成/
    │   ├── 25_Signal集成/
    │   ├── 26_iMessage集成/
    │   ├── 27_其他通道/
    │   ├── 28_通道配对与认证/
    │   ├── 29_消息队列与分发/
    │   └── 30_通道状态管理/
    │
    ├── Phase4_Agent与AI集成/
    │   ├── k.md
    │   ├── 31_Pi_agent_core深入/
    │   ├── 32_Pi_ai统一API/
    │   ├── 33_Model配置与切换/
    │   ├── 34_Provider管理/
    │   ├── 35_Prompt工程/
    │   ├── 36_工具调用机制/
    │   ├── 37_流式响应处理/
    │   ├── 38_Context管理/
    │   ├── 39_Memory系统/
    │   └── 40_Agent模式/
    │
    ├── Phase5_扩展与定制开发/
    │   ├── k.md
    │   ├── 41_Extension_API架构/
    │   ├── 42_Plugin_SDK详解/
    │   ├── 43_自定义Extension开发实战/
    │   ├── 44_Hook系统深入/
    │   ├── 45_自定义Hook开发/
    │   ├── 46_Skills系统详解/
    │   ├── 47_自定义Skill开发/
    │   ├── 48_配置扩展机制/
    │   ├── 49_自定义通道开发/
    │   ├── 50_自定义工具开发/
    │   ├── 51_UI组件扩展/
    │   └── 52_Extension打包与发布/
    │
    ├── Phase6_高级功能实现/
    │   ├── k.md
    │   ├── 53_Browser自动化/
    │   ├── 54_Canvas功能架构与实现/
    │   ├── 55_Voice集成（macOS）/
    │   ├── 56_Voice集成（iOS）/
    │   ├── 57_Voice集成（Android）/
    │   ├── 58_Cron调度系统详解/
    │   ├── 59_Auto_reply系统设计/
    │   ├── 60_ACP深入/
    │   ├── 61_多Agent协作机制/
    │   └── 62_安全与权限控制系统/
    │
    ├── Phase7_平台与部署/
    │   ├── k.md
    │   ├── 63_CLI工具架构深入/
    │   ├── 64_Mac应用开发/
    │   ├── 65_iOS应用开发/
    │   ├── 66_Android应用开发/
    │   ├── 67_Web_UI开发/
    │   ├── 68_Docker部署与容器化/
    │   ├── 69_生产环境配置与优化/
    │   ├── 70_监控日志与告警/
    │   ├── 71_性能优化与调优/
    │   └── 72_安全加固与最佳实践/
    │
    ├── Phase8_源码贡献与高级主题/
    │   ├── k.md
    │   ├── 73_代码库结构深入分析/
    │   ├── 74_Monorepo架构/
    │   ├── 75_开发环境完整搭建/
    │   ├── 76_测试框架/
    │   ├── 77_代码规范/
    │   ├── 78_构建系统/
    │   ├── 79_发布流程/
    │   ├── 80_PR工作流与代码审查/
    │   ├── 81_社区贡献指南/
    │   └── 82_高级调试与性能分析/
    │
    └── Phase9_实战项目与案例分析/
        ├── k.md
        ├── 83_React_Vite_Electron桌面客户端开发/
        ├── 84_Next_js仪表盘开发/
        ├── 85_WebChat前端集成/
        ├── 86_Live2D_Avatar前端框架/
        ├── 87_FastAPI_WebSocket实时通信/
        ├── 88_Flask_API端点设计/
        ├── 89_Express_Gateway集成/
        ├── 90_第三方认证与API安全/
        ├── 91_多代理协作系统/
        ├── 92_支付系统集成/
        ├── 93_企业级应用/
        └── 94_生产环境部署与监控/
```

---

## 快速启动模板

### 生成新知识点的步骤

1. **读取通用模板** (`prompt/atom_template.md`)
2. **读取本文档** (`atom/openclaw/CLAUDE_OPENCLAW.md`) - OpenClaw 特定配置
3. **读取知识点列表** (`atom/openclaw/[Phase]/k.md`)
4. **确认目标知识点**（第几个）
5. **按规范生成内容**（10个维度）
6. **质量检查**（使用检查清单）
7. **保存文件**（`atom/openclaw/[Phase]/[编号]_[知识点]/`）

### 提示词模板

```
根据 @prompt/atom_template.md 的通用规范和 @atom/openclaw/CLAUDE_OPENCLAW.md 的 OpenClaw 特定配置，为 @atom/openclaw/[Phase]/k.md 中的第[N]个知识点 "[知识点名称]" 生成一个完整的学习文档。

要求：
- 按照10个维度完整生成
- 有 TypeScript/Node.js 基础的开发者友好
- 代码可运行（TypeScript/JavaScript）
- 双重类比（后端开发 + 日常生活）
- 与 OpenClaw 多通道 AI 网关开发紧密结合
- 体现 Gateway 架构和多通道消息系统设计理念

文件保存到：atom/openclaw/[Phase]/[编号]_[知识点名称]/
```

---

## 核心原则总结

1. **原子化**：每个知识点独立完整
2. **全面覆盖**：知识点所有子概念都要讲到
3. **实战导向**：联系 OpenClaw 开发应用
4. **TypeScript 友好**：简单语言 + 双重类比（后端开发 + 日常生活）
5. **速成高效**：20%核心 + 80%效果
6. **代码可运行**：所有示例都能跑（TypeScript/JavaScript）
7. **体系完整**：10个维度全覆盖
8. **质量保证**：严格检查清单
9. **多通道架构**：体现 OpenClaw 的多通道消息系统设计理念
10. **实战案例**：强调前端、后端、Agent 开发的结合

---

## 学习路径设计

### 用户背景

- **技术背景**：TypeScript 和 Node.js 熟练
- **AI 基础**：LLM API 和 AI agent 开发有基础了解
- **学习目标**：全面掌握 OpenClaw（使用 + 架构 + 开发）

### 学习目标

1. **使用者视角** - 把 OpenClaw 作为个人 AI 助手使用
2. **架构理解** - 深入理解 OpenClaw 的 Gateway 架构和多通道系统
3. **构建者视角** - 基于 OpenClaw 构建自己的 AI 应用和扩展

### 学习深度

**全面掌握（3-4个月）**

### 9个阶段，94个知识点

| Phase | 知识点数 | 学习时长 | 目标 |
|-------|---------|---------|------|
| Phase 1: 快速上手与环境搭建 | 8 | 第 1-2 周 | 能够安装、配置和使用 OpenClaw |
| Phase 2: 核心架构深入理解 | 10 | 第 3-5 周 | 理解 Gateway 架构和核心组件 |
| Phase 3: 多通道消息系统 | 12 | 第 6-8 周 | 掌握多通道集成和消息路由 |
| Phase 4: Agent 与 AI 集成 | 10 | 第 9-10 周 | 理解 Agent 系统和 AI 模型集成 |
| Phase 5: 扩展与定制开发 | 12 | 第 11-12 周 | 掌握扩展开发和定制能力 |
| Phase 6: 高级功能实现 | 10 | 第 13-14 周 | 掌握高级功能的实现原理 |
| Phase 7: 平台与部署 | 10 | 第 15-16 周 | 掌握跨平台开发和生产部署 |
| Phase 8: 源码贡献与高级主题 | 10 | 第 17-18 周 | 能够为 OpenClaw 贡献代码 |
| Phase 9: 实战项目与案例分析 | 12 | 第 19-20 周 | 通过实战案例掌握全栈开发 |

---

## 实战案例资源

### 前端开发案例
- **clawUI**: https://github.com/Kt-L/clawUI - React/Vite/Electron 桌面客户端
- **webclaw**: https://github.com/ibelick/webclaw - 快速 Web 客户端
- **PinchChat**: https://github.com/MarlBurroW/pinchchat - 轻量级 Webchat 前端
- **Mission Control**: https://github.com/abhi1693/openclaw-mission-control - Next.js 仪表盘
- **Live2D**: https://github.com/Singularity-Engine/openclaw-live2d - Live2D Avatar 框架

### 后端开发案例
- **ClawWork**: https://github.com/HKUDS/ClawWork - FastAPI + WebSocket
- **Operating System Guide**: https://gist.github.com/behindthegarage/db5e15213a4daf566caccc9d40fcd02d - Flask 后端

### 全栈项目案例
- **Awesome OpenClaw**: https://github.com/rohitg00/awesome-openclaw - 资源合集
- **Awesome Skills**: https://github.com/VoltAgent/awesome-openclaw-skills - 技能合集

### 社区资源
- **GitHub**: https://github.com/openclaw/openclaw
- **文档**: https://docs.openclaw.ai
- **Discord**: https://discord.gg/clawd
- **DeepWiki**: https://deepwiki.com/openclaw/openclaw

---

**版本：** v1.0 (OpenClaw 开发专用版 - 基于通用模板)
**最后更新：** 2026-02-22
**维护者：** Claude Code

---

**记住：** 生成每个新知识点前，先读取 `prompt/atom_template.md` 和本文档！
