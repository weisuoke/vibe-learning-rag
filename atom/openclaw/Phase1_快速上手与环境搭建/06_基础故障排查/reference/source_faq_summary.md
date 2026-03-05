# OpenClaw FAQ 摘要

**来源**: `sourcecode/openclaw/docs/help/faq.md`（前500行）

## 60秒快速诊断（如果出现问题）

### 1. 快速状态（首次检查）
```bash
openclaw status
```
快速本地摘要：OS + 更新、网关/服务可达性、代理/会话、提供商配置 + 运行时问题

### 2. 可粘贴报告（安全分享）
```bash
openclaw status --all
```
只读诊断带日志尾部（token 已编辑）

### 3. 守护进程 + 端口状态
```bash
openclaw gateway status
```
显示监督器运行时 vs RPC 可达性、探测目标 URL 以及服务可能使用的配置

### 4. 深度探测
```bash
openclaw status --deep
```
运行网关健康检查 + 提供商探测（需要可达的网关）

### 5. 跟踪最新日志
```bash
openclaw logs --follow
```

如果 RPC 关闭，回退到：
```bash
tail -f "$(ls -t /tmp/openclaw/openclaw-*.log | head -1)"
```

### 6. 运行 doctor（修复）
```bash
openclaw doctor
```
修复/迁移配置/状态 + 运行健康检查

### 7. 网关快照
```bash
openclaw health --json
openclaw health --verbose   # 在错误时显示目标 URL + 配置路径
```

## 快速启动和首次运行设置

### 我卡住了，最快的解决方法是什么？

使用可以**看到你的机器**的本地 AI 代理。这比在 Discord 中询问更有效，因为大多数"我卡住了"的情况是**本地配置或环境问题**，远程帮助者无法检查。

- **Claude Code**: https://www.anthropic.com/claude-code/
- **OpenAI Codex**: https://openai.com/codex/

这些工具可以读取仓库、运行命令、检查日志并帮助修复你的机器级设置（PATH、服务、权限、认证文件）。

使用可破解（git）安装给它们**完整的源代码检出**：

```bash
curl -fsSL https://openclaw.ai/install.sh | bash -s -- --install-method git
```

首先运行这些命令（在寻求帮助时分享输出）：

```bash
openclaw status
openclaw models status
openclaw doctor
```

### 推荐的安装和设置方式是什么？

仓库推荐从源代码运行并使用 onboarding wizard：

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
openclaw onboard --install-daemon
```

向导还可以自动构建 UI 资源。onboarding 后，你通常在端口 **18789** 上运行 Gateway。

从源代码（贡献者/开发）：

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
pnpm build
pnpm ui:build # 首次运行时自动安装 UI 依赖
openclaw onboard
```

### 需要什么运行时？

需要 Node **>= 22**。推荐 `pnpm`。**不推荐** Bun 用于 Gateway。

### 它能在 Raspberry Pi 上运行吗？

是的。Gateway 是轻量级的 - 文档列出 **512MB-1GB RAM**、**1 核心**和约 **500MB** 磁盘足够个人使用，并注意 **Raspberry Pi 4 可以运行它**。

如果你想要额外的余量（日志、媒体、其他服务），**推荐 2GB**，但这不是硬性最低要求。

提示：小型 Pi/VPS 可以托管 Gateway，你可以在笔记本电脑/手机上配对**节点**以进行本地屏幕/相机/画布或命令执行。

### onboarding 卡在"wake up my friend"

该屏幕依赖于 Gateway 可达和已认证。TUI 还会在首次孵化时自动发送"Wake up, my friend!"。如果你看到该行**无回复**且 token 保持在 0，代理从未运行。

1. 重启 Gateway：
```bash
openclaw gateway restart
```

2. 检查状态 + 认证：
```bash
openclaw status
openclaw models status
openclaw logs --follow
```

3. 如果仍然挂起，运行：
```bash
openclaw doctor
```

## 核心诊断工具总结

1. **openclaw status**: 快速本地摘要
2. **openclaw status --all**: 完整可分享报告
3. **openclaw gateway status**: 守护进程和端口状态
4. **openclaw doctor**: 健康检查和自动修复
5. **openclaw logs --follow**: 实时日志跟踪
6. **openclaw health**: 网关快照

## 环境要求

- **Node.js**: >= 22
- **包管理器**: pnpm（推荐）
- **操作系统**: macOS, Linux, Windows (WSL2 强烈推荐)
- **最小资源**: 512MB-1GB RAM, 1 核心, 500MB 磁盘

## 关键概念

1. **60秒诊断流程**: 快速系统化的故障排查方法
2. **本地 AI 代理辅助**: 使用 Claude Code 或 Codex 进行机器级诊断
3. **从源代码安装**: 可破解安装提供完整代码访问
4. **Onboarding Wizard**: 自动化设置和配置
5. **轻量级设计**: 可在 Raspberry Pi 等低资源设备上运行
6. **多平台支持**: macOS, Linux, Windows (WSL2)
