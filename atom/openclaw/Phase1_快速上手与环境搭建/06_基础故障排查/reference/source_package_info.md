# OpenClaw 包信息

**来源**: `sourcecode/openclaw/package.json`

## 基本信息

- **名称**: openclaw
- **版本**: 2026.2.22
- **描述**: Multi-channel AI gateway with extensible messaging integrations
- **许可证**: MIT
- **仓库**: https://github.com/openclaw/openclaw

## 环境要求

- **Node.js**: >= 22.12.0
- **包管理器**: pnpm@10.23.0

## 核心依赖

### Agent 系统
- `@mariozechner/pi-agent-core`: 0.54.0
- `@mariozechner/pi-ai`: 0.54.0
- `@mariozechner/pi-coding-agent`: 0.54.0
- `@mariozechner/pi-tui`: 0.54.0

### 通道集成
- `@whiskeysockets/baileys`: 7.0.0-rc.9 (WhatsApp)
- `grammy`: ^1.40.0 (Telegram)
- `@slack/bolt`: ^4.6.0 (Slack)
- `discord-api-types`: ^0.38.40 (Discord)
- `@line/bot-sdk`: ^10.6.0 (LINE)

### 浏览器自动化
- `playwright-core`: 1.58.2

### Web 框架
- `express`: ^5.2.1

### CLI 工具
- `commander`: ^14.0.3
- `@clack/prompts`: ^1.0.1

### Schema 验证
- `@sinclair/typebox`: 0.34.48
- `zod`: ^4.3.6

### 其他核心库
- `ws`: ^8.19.0 (WebSocket)
- `dotenv`: ^17.3.1 (环境变量)
- `chalk`: ^5.6.2 (终端颜色)
- `tslog`: ^4.10.2 (日志)

## 开发依赖

- `typescript`: ^5.9.3
- `vitest`: ^4.0.18
- `oxlint`: ^1.49.0
- `oxfmt`: 0.34.0
- `tsx`: ^4.21.0

## 关键脚本

### 开发
- `gateway:dev`: 开发模式运行网关（跳过通道）
- `gateway:watch`: 监视模式运行网关
- `gateway:dev:reset`: 重置开发环境

### 构建
- `build`: 构建所有包
- `ui:build`: 构建 UI 资源

### 测试
- `test`: 运行并行测试
- `test:e2e`: 运行端到端测试
- `test:live`: 运行实时测试

### 代码质量
- `lint`: 运行 oxlint 类型感知检查
- `format`: 运行 oxfmt 格式化
- `check`: 运行格式检查 + 类型检查 + lint

### 平台特定
- `android:run`: 运行 Android 应用
- `ios:run`: 运行 iOS 应用
- `mac:package`: 打包 Mac 应用

## 核心概念

1. **多通道架构**: 支持 WhatsApp, Telegram, Slack, Discord, LINE 等多个通道
2. **Agent 系统**: 基于 pi-agent-core 的 AI 代理系统
3. **浏览器自动化**: 使用 Playwright 进行浏览器控制
4. **跨平台支持**: macOS, iOS, Android, Web
5. **TypeScript 优先**: 完整的 TypeScript 支持
6. **Monorepo 架构**: 使用 pnpm workspaces
7. **开发工具链**: oxlint, oxfmt, vitest
