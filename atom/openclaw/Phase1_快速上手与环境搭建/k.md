# Phase 1: 快速上手与环境搭建

**目标：** 能够安装、配置和使用 OpenClaw 进行基本操作

**学习时长：** 第 1-2 周

**知识点数：** 8 个

---

## 知识点列表

### 01. OpenClaw 安装与配置
- npm/pnpm 全局安装
- Onboarding wizard 使用
- 环境要求（Node 22+）
- 配置文件位置（~/.openclaw/）

### 02. Onboarding Wizard 详解
- Gateway 设置
- Workspace 配置
- Channel 配置
- Skills 安装
- 守护进程安装（launchd/systemd）

### 03. CLI 基础命令
- `openclaw gateway` - 启动网关
- `openclaw message send` - 发送消息
- `openclaw agent` - Agent 交互
- `openclaw channels status` - 通道状态
- `openclaw config` - 配置管理

### 04. Gateway 启动与管理
- Gateway 启动模式（前台/后台）
- 端口配置（默认 18789）
- 日志查看
- 守护进程管理

### 05. 第一个消息发送
- 通道配对
- 消息发送到不同通道
- Agent 响应机制
- 基础故障排查

### 06. 基础故障排查
- `openclaw doctor` 诊断工具
- 日志位置和查看
- 常见错误解决
- 配置验证

### 07. 开发环境搭建（从源码）
- 克隆仓库
- pnpm install
- pnpm build
- 从源码运行

### 08. 源码编译与调试
- TypeScript 编译
- 开发模式运行
- 调试配置
- 热重载

---

**验证标准：**
- ✅ 成功安装 OpenClaw
- ✅ 启动 Gateway
- ✅ 发送第一条消息
- ✅ 从源码编译运行
