# OpenClaw 全面学习计划

> 为 OpenClaw 多通道 AI 网关开发构建的完整原子化知识体系

---

## 项目概述

**学习目标：** 全面掌握 OpenClaw 的使用、架构和开发能力

**OpenClaw 简介：**
- 个人 AI 助手，运行在自己的设备上
- 多通道消息系统（WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Google Chat, Microsoft Teams, WebChat 等）
- Gateway 架构（控制平面）
- 基于 pi-mono 的 Agent 系统
- 支持语音、Canvas、浏览器自动化等高级功能
- 跨平台支持（Mac, iOS, Android, Linux, Windows）

**技术栈：**
- TypeScript/Node.js (Node 22+)
- pnpm 包管理
- Vitest 测试框架
- Pi-agent-core, Pi-ai, Pi-tui
- 多通道集成库

**学习深度：** 综合掌握（使用 + 架构 + 开发）

**学习时长：** 3-4 个月（20 周）

---

## 学习路径

### 9 个阶段，94 个知识点

| Phase | 知识点数 | 学习时长 | 目标 |
|-------|---------|---------|------|
| [Phase 1: 快速上手与环境搭建](./Phase1_快速上手与环境搭建/) | 8 | 第 1-2 周 | 能够安装、配置和使用 OpenClaw |
| [Phase 2: 核心架构深入理解](./Phase2_核心架构深入理解/) | 10 | 第 3-5 周 | 理解 Gateway 架构和核心组件 |
| [Phase 3: 多通道消息系统](./Phase3_多通道消息系统/) | 12 | 第 6-8 周 | 掌握多通道集成和消息路由 |
| [Phase 4: Agent 与 AI 集成](./Phase4_Agent与AI集成/) | 10 | 第 9-10 周 | 理解 Agent 系统和 AI 模型集成 |
| [Phase 5: 扩展与定制开发](./Phase5_扩展与定制开发/) | 12 | 第 11-12 周 | 掌握扩展开发和定制能力 |
| [Phase 6: 高级功能实现](./Phase6_高级功能实现/) | 10 | 第 13-14 周 | 掌握高级功能的实现原理 |
| [Phase 7: 平台与部署](./Phase7_平台与部署/) | 10 | 第 15-16 周 | 掌握跨平台开发和生产部署 |
| [Phase 8: 源码贡献与高级主题](./Phase8_源码贡献与高级主题/) | 10 | 第 17-18 周 | 能够为 OpenClaw 贡献代码 |
| [Phase 9: 实战项目与案例分析](./Phase9_实战项目与案例分析/) | 12 | 第 19-20 周 | 通过实战案例掌握全栈开发 |

---

## 快速开始

### 环境要求

- **Node.js 版本**: 22+ (推荐 22.12.0+)
- **包管理器**: pnpm (推荐 10.23.0+)
- **TypeScript**: 5.9+
- **操作系统**: macOS, Linux, Windows (WSL2 强烈推荐)

### 安装 OpenClaw

```bash
# 全局安装
npm install -g openclaw@latest
# 或: pnpm add -g openclaw@latest

# 运行 Onboarding Wizard
openclaw onboard --install-daemon

# 启动 Gateway
openclaw gateway --port 18789 --verbose

# 发送消息测试
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# Agent 交互
openclaw agent --message "Ship checklist" --thinking high
```

### 从源码开发

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 安装依赖
pnpm install

# 构建所有包
pnpm build

# 运行测试
pnpm test

# 从源码运行
pnpm openclaw gateway --port 18789
```

---

## 学习方法

### 原子化学习理念

每个知识点都按照 10 个维度进行组织：

1. **【30字核心】** - 一句话理解核心概念
2. **【第一性原理】** - 从根本原理理解
3. **【3个核心概念】** - 掌握关键概念
4. **【最小可用】** - 最小可运行示例
5. **【双重类比】** - 后端开发类比 + 日常生活类比
6. **【反直觉点】** - 常见误区和陷阱
7. **【实战代码】** - 可运行的完整代码
8. **【面试必问】** - 核心面试问题
9. **【化骨绵掌】** - 深入理解和最佳实践
10. **【一句话总结】** - 记忆要点

### 学习建议

- **边学边实践**：每个知识点都要动手操作
- **阅读源码**：理解架构的最佳方式
- **参与社区**：学习他人经验
- **从简单开始**：从简单的 Extension 开始，逐步深入
- **关注官方**：关注 GitHub Issues 和 Discussions
- **实战项目**：检验学习成果的最好方式

---

## 阶段性验证

### Phase 1 验证
- ✅ 成功安装 OpenClaw
- ✅ 启动 Gateway
- ✅ 发送第一条消息
- ✅ 从源码编译运行

### Phase 2 验证
- ✅ 理解 Gateway 架构
- ✅ 阅读核心源码
- ✅ 理解配置系统
- ✅ 理解会话管理

### Phase 3 验证
- ✅ 配置至少 3 个通道
- ✅ 理解通道路由
- ✅ 阅读通道源码
- ✅ 测试消息收发

### Phase 4 验证
- ✅ 配置多个 Model Provider
- ✅ 理解 Agent 运行时
- ✅ 测试工具调用
- ✅ 理解流式响应

### Phase 5 验证
- ✅ 开发一个简单 Extension
- ✅ 开发一个自定义 Skill
- ✅ 开发一个自定义 Hook
- ✅ 发布 Extension

### Phase 6 验证
- ✅ 使用 Browser 自动化
- ✅ 测试 Canvas 功能
- ✅ 测试 Voice 集成
- ✅ 配置 Cron 任务

### Phase 7 验证
- ✅ 构建 Mac 应用
- ✅ 构建 iOS/Android 应用
- ✅ Docker 部署
- ✅ 生产环境配置

### Phase 8 验证
- ✅ 运行完整测试套件
- ✅ 提交一个 PR
- ✅ 通过代码审查
- ✅ 理解发布流程

### Phase 9 验证
- ✅ 完成一个前端项目
- ✅ 完成一个后端项目
- ✅ 完成一个全栈项目
- ✅ 部署到生产环境

---

## 关键文件路径

### 核心源码
- `src/gateway/` - Gateway 核心
- `src/agents/` - Agent 系统
- `src/channels/` - 通道抽象
- `src/cli/` - CLI 工具
- `src/commands/` - 命令实现
- `src/config/` - 配置系统
- `src/infra/` - 基础设施
- `src/hooks/` - Hook 系统

### 通道实现
- `src/telegram/` - Telegram
- `src/discord/` - Discord
- `src/slack/` - Slack
- `src/signal/` - Signal
- `src/imessage/` - iMessage
- `src/line/` - LINE
- `src/web/` - WhatsApp Web
- `extensions/msteams/` - Microsoft Teams

### 平台应用
- `apps/macos/` - Mac 应用
- `apps/ios/` - iOS 应用
- `apps/android/` - Android 应用
- `ui/` - Web UI

### 扩展系统
- `extensions/` - 扩展目录
- `skills/` - Skills 目录
- `src/extensionAPI.ts` - Extension API
- `dist/plugin-sdk/` - Plugin SDK

---

## 学习资源

### 官方资源
- **GitHub**: https://github.com/openclaw/openclaw
- **文档**: https://docs.openclaw.ai
- **Discord**: https://discord.gg/clawd
- **网站**: https://openclaw.ai

### 社区资源
- **Awesome OpenClaw**: https://github.com/rohitg00/awesome-openclaw
- **Awesome Skills**: https://github.com/VoltAgent/awesome-openclaw-skills
- **DeepWiki**: https://deepwiki.com/openclaw/openclaw

### 实战案例

#### 前端开发
- **clawUI**: https://github.com/Kt-L/clawUI - React/Vite/Electron 桌面客户端
- **webclaw**: https://github.com/ibelick/webclaw - 快速 Web 客户端
- **PinchChat**: https://github.com/MarlBurroW/pinchchat - 轻量级 Webchat 前端
- **Mission Control**: https://github.com/abhi1693/openclaw-mission-control - Next.js 仪表盘
- **Live2D**: https://github.com/Singularity-Engine/openclaw-live2d - Live2D Avatar 框架

#### 后端开发
- **ClawWork**: https://github.com/HKUDS/ClawWork - FastAPI + WebSocket
- **Operating System Guide**: https://gist.github.com/behindthegarage/db5e15213a4daf566caccc9d40fcd02d - Flask 后端

#### 全栈项目
- **Awesome OpenClaw**: https://github.com/rohitg00/awesome-openclaw - 资源合集
- **Awesome Skills**: https://github.com/VoltAgent/awesome-openclaw-skills - 技能合集

### 学习指南
- **Ultimate Guide**: https://www.reddit.com/r/ThinkingDeeplyAI/comments/1qsoq4h/
- **Operating System Guide**: https://gist.github.com/behindthegarage/db5e15213a4daf566caccc9d40fcd02d

---

## 文档规范

本学习计划遵循原子化知识点生成规范，详见：
- **通用模板**: `../../prompt/atom_template.md`
- **OpenClaw 特定配置**: `./CLAUDE_OPENCLAW.md`

---

## 贡献

欢迎为本学习计划贡献内容：
1. 完善知识点文档
2. 添加实战案例
3. 修正错误和改进
4. 分享学习心得

---

**版本：** v1.0
**创建日期：** 2026-02-22
**维护者：** Claude Code
**许可证：** MIT

---

**开始学习：** 从 [Phase 1: 快速上手与环境搭建](./Phase1_快速上手与环境搭建/k.md) 开始你的 OpenClaw 学习之旅！
