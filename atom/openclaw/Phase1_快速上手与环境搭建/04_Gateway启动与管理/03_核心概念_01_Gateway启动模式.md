# 核心概念 1：Gateway 启动模式

## 概述

Gateway 启动模式是 OpenClaw 运行时控制的基础，决定了 Gateway 进程如何启动、运行和管理。OpenClaw 提供两种启动模式：**前台启动（Foreground）**和**守护进程（Daemon）**，分别适用于开发调试和生产部署场景。

## 一句话定义

**Gateway 启动模式是控制 Gateway 进程运行方式的机制，前台模式用于开发调试（实时日志、交互式控制），守护进程模式用于生产部署（后台运行、自动重启、系统集成）。**

---

## 第一性原理：为什么需要两种启动模式？

### 问题根源

任何服务器程序都面临两个矛盾的需求：

1. **开发调试需求**：
   - 需要立即看到日志输出
   - 需要快速启动和停止
   - 需要交互式控制（Ctrl+C）
   - 需要快速迭代测试

2. **生产部署需求**：
   - 需要后台持续运行
   - 需要系统启动时自动运行
   - 需要崩溃后自动重启
   - 需要与系统服务管理集成

这两个需求无法用同一种运行方式满足。

### 解决方案

OpenClaw 通过两种启动模式解决这个矛盾：

- **前台模式**：进程运行在当前终端，日志输出到 stdout/stderr，适合开发调试
- **守护进程模式**：进程运行在后台，由系统服务管理（launchd/systemd），适合生产部署

---

## 前台启动模式（Foreground Mode）

### 定义

**前台启动模式**是指 Gateway 进程运行在当前终端会话中，日志实时输出到控制台，进程生命周期与终端会话绑定。

### 特征

| 特征 | 说明 |
|------|------|
| **运行位置** | 当前终端前台 |
| **日志输出** | stdout/stderr（实时可见） |
| **生命周期** | 与终端会话绑定 |
| **停止方式** | Ctrl+C 或关闭终端 |
| **自动重启** | 无 |
| **系统集成** | 无 |

### 启动命令

**基础启动：**
```bash
# 最简单的前台启动
openclaw gateway

# 等价于
openclaw gateway run
```

**开发模式启动：**
```bash
# 详细日志输出（debug 级别）
openclaw gateway --verbose

# 开发模式（创建 dev 配置）
openclaw gateway --dev

# 开发模式 + 重置
openclaw gateway --dev --reset
```

**调试选项：**
```bash
# 完整 WebSocket 日志
openclaw gateway --ws-log full

# 紧凑 WebSocket 日志
openclaw gateway --ws-log compact

# 只显示 Claude CLI 日志
openclaw gateway --claude-cli-logs

# 原始流日志
openclaw gateway --raw-stream
```

### 日志输出示例

```bash
$ openclaw gateway --verbose

[2026-02-22T06:45:58.288Z] [gateway] Gateway starting on port 18789
[2026-02-22T06:45:58.289Z] [gateway] Bind mode: loopback
[2026-02-22T06:45:58.290Z] [gateway] Auth mode: token
[2026-02-22T06:45:58.291Z] [gateway] Loading config from ~/.openclaw/openclaw.json
[2026-02-22T06:45:58.292Z] [gateway] Config validation passed
[2026-02-22T06:45:58.293Z] [channels] Initializing WhatsApp channel
[2026-02-22T06:45:58.294Z] [channels] WhatsApp channel ready
[2026-02-22T06:45:58.295Z] [gateway] Gateway ready
[2026-02-22T06:45:58.296Z] [gateway] Control UI: http://127.0.0.1:18789
```

### 使用场景

#### 场景 1：开发调试

**需求：**
- 快速测试代码更改
- 实时查看日志输出
- 方便启动和停止

**操作：**
```bash
# Terminal 1: 启动 Gateway
openclaw gateway --verbose --dev

# Terminal 2: 测试功能
openclaw agent --message "Test feature"

# Terminal 1: 查看日志输出（实时）
# [2026-02-22T06:46:00.123Z] [agent] Received message: Test feature
# [2026-02-22T06:46:00.124Z] [agent] Processing...
# [2026-02-22T06:46:01.456Z] [agent] Response sent

# Terminal 1: 停止 Gateway
# Ctrl+C
```

#### 场景 2：故障排查

**需求：**
- 查看详细日志定位问题
- 测试不同配置选项
- 快速重启测试

**操作：**
```bash
# 启动 Gateway 并查看详细日志
openclaw gateway --verbose --ws-log full

# 观察日志输出，定位问题
# 修改配置后，Ctrl+C 停止
# 重新启动测试
```

#### 场景 3：快速测试

**需求：**
- 测试新功能
- 验证配置更改
- 临时运行

**操作：**
```bash
# 快速启动测试
openclaw gateway --dev --verbose

# 测试完成后 Ctrl+C 停止
```

### 优点

✅ **立即反馈**：日志实时输出，问题立即可见
✅ **快速控制**：Ctrl+C 即可停止，无需额外命令
✅ **简单直接**：无需安装守护进程，直接运行
✅ **开发友好**：适合快速迭代开发

### 缺点

❌ **不持久**：终端关闭后进程停止
❌ **无自动重启**：崩溃后需要手动重启
❌ **占用终端**：需要保持终端窗口打开
❌ **不适合生产**：无法实现 24/7 运行

### 最佳实践

1. **开发环境使用前台模式**
   ```bash
   openclaw gateway --verbose --dev
   ```

2. **启用详细日志**
   ```bash
   openclaw gateway --verbose --ws-log full
   ```

3. **使用多终端**
   - Terminal 1: Gateway 前台运行
   - Terminal 2: 测试命令
   - Terminal 3: 日志监控（如需要）

4. **快速迭代**
   ```bash
   # 启动 -> 测试 -> Ctrl+C -> 修改 -> 重启
   openclaw gateway --dev --verbose
   ```

---

## 守护进程模式（Daemon Mode）

### 定义

**守护进程模式**是指 Gateway 进程作为系统服务运行在后台，由操作系统的服务管理器（macOS 的 launchd 或 Linux 的 systemd）管理，独立于终端会话。

### 特征

| 特征 | 说明 |
|------|------|
| **运行位置** | 系统后台 |
| **日志输出** | 文件（~/.openclaw/logs/） |
| **生命周期** | 独立于终端会话 |
| **停止方式** | 服务管理命令 |
| **自动重启** | 支持（崩溃后自动重启） |
| **系统集成** | 完全集成（开机自启） |

### 安装守护进程

**通过 Onboarding Wizard：**
```bash
# 交互式安装（推荐）
openclaw onboard --install-daemon

# 向导会：
# 1. 创建配置文件
# 2. 安装守护进程服务
# 3. 启动 Gateway
# 4. 验证运行状态
```

**直接安装：**
```bash
# 安装守护进程
openclaw gateway install

# 带选项安装
openclaw gateway install --port 18789 --token <token>
```

### 服务管理命令

**跨平台命令（推荐）：**
```bash
# 启动 Gateway
openclaw gateway start

# 停止 Gateway
openclaw gateway stop

# 重启 Gateway
openclaw gateway restart

# 查看状态
openclaw gateway status

# 卸载守护进程
openclaw gateway uninstall
```

**macOS (launchd) 原生命令：**
```bash
# 服务文件位置
~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 加载服务
launchctl load ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 启动服务
launchctl start ai.openclaw.gateway

# 停止服务
launchctl stop ai.openclaw.gateway

# 卸载服务
launchctl unload ~/Library/LaunchAgents/ai.openclaw.gateway.plist

# 查看服务列表
launchctl list | grep openclaw
```

**Linux (systemd) 原生命令：**
```bash
# 服务文件位置
~/.config/systemd/user/openclaw-gateway.service

# 启用服务（开机自启）
systemctl --user enable openclaw-gateway

# 启动服务
systemctl --user start openclaw-gateway

# 停止服务
systemctl --user stop openclaw-gateway

# 重启服务
systemctl --user restart openclaw-gateway

# 查看状态
systemctl --user status openclaw-gateway

# 查看日志
journalctl --user -u openclaw-gateway -f

# 禁用服务
systemctl --user disable openclaw-gateway

# 启用 lingering（用户登出后继续运行）
loginctl enable-linger $USER
```

### 日志查看

**使用 OpenClaw CLI：**
```bash
# 查看最近日志
openclaw logs

# 实时跟踪日志
openclaw logs --follow

# 查看最后 100 行
openclaw logs --tail 100

# 查看特定日期日志
openclaw logs --date 2026-02-22
```

**直接查看日志文件：**
```bash
# 日志文件位置
~/.openclaw/logs/gateway.log

# 实时查看
tail -f ~/.openclaw/logs/gateway.log

# 查看最后 100 行
tail -n 100 ~/.openclaw/logs/gateway.log
```

### 使用场景

#### 场景 1：生产部署

**需求：**
- 24/7 持续运行
- 系统启动时自动运行
- 崩溃后自动重启

**操作：**
```bash
# 1. 安装守护进程
openclaw onboard --install-daemon

# 2. 验证运行状态
openclaw gateway status

# 3. 查看日志
openclaw logs --follow

# 4. 系统重启后自动运行（无需手动操作）
```

#### 场景 2：服务器环境

**需求：**
- 无图形界面
- SSH 连接后需要持续运行
- 断开 SSH 后继续运行

**操作：**
```bash
# SSH 连接到服务器
ssh user@server

# 安装守护进程
openclaw onboard --install-daemon

# 断开 SSH（Gateway 继续运行）
exit

# 重新连接查看状态
ssh user@server
openclaw gateway status
```

#### 场景 3：多用户环境

**需求：**
- 多个用户共享服务器
- 每个用户独立的 Gateway 实例
- 用户登出后继续运行

**操作：**
```bash
# 用户 A
openclaw onboard --install-daemon
# Gateway 运行在用户 A 的上下文

# 用户 B
openclaw onboard --install-daemon
# Gateway 运行在用户 B 的上下文

# 两个 Gateway 独立运行，互不干扰
```

### 优点

✅ **持久运行**：终端关闭后继续运行
✅ **自动重启**：崩溃后自动恢复
✅ **系统集成**：开机自动启动
✅ **生产就绪**：适合 24/7 运行

### 缺点

❌ **日志延迟**：需要查看日志文件，不如前台模式直观
❌ **安装复杂**：需要安装守护进程服务
❌ **调试不便**：无法直接看到实时输出
❌ **管理命令**：需要使用服务管理命令

### 最佳实践

1. **生产环境使用守护进程**
   ```bash
   openclaw onboard --install-daemon
   ```

2. **启用 lingering（Linux）**
   ```bash
   loginctl enable-linger $USER
   ```

3. **定期检查状态**
   ```bash
   openclaw gateway status
   ```

4. **监控日志**
   ```bash
   openclaw logs --follow
   ```

5. **配置自动重启**
   - macOS launchd 默认支持
   - Linux systemd 需要配置 `Restart=always`

---

## 两种模式对比

### 功能对比表

| 特性 | 前台模式 | 守护进程模式 |
|------|----------|--------------|
| **运行位置** | 终端前台 | 系统后台 |
| **日志输出** | 控制台（实时） | 文件（延迟） |
| **终端依赖** | 依赖终端会话 | 独立于终端 |
| **停止方式** | Ctrl+C | 服务管理命令 |
| **自动重启** | ❌ 不支持 | ✅ 支持 |
| **开机自启** | ❌ 不支持 | ✅ 支持 |
| **调试便利性** | ✅ 高 | ❌ 低 |
| **生产适用性** | ❌ 不适合 | ✅ 适合 |
| **安装复杂度** | ✅ 简单 | ❌ 复杂 |
| **资源占用** | 相同 | 相同 |

### 使用场景对比

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| **本地开发** | 前台模式 | 快速迭代，实时反馈 |
| **调试问题** | 前台模式 | 详细日志，交互式控制 |
| **快速测试** | 前台模式 | 启动快，停止方便 |
| **生产部署** | 守护进程 | 持久运行，自动重启 |
| **服务器环境** | 守护进程 | 独立于 SSH 会话 |
| **24/7 运行** | 守护进程 | 系统集成，自动恢复 |

### 切换模式

**从前台模式切换到守护进程：**
```bash
# 1. 停止前台 Gateway（Ctrl+C）

# 2. 安装守护进程
openclaw onboard --install-daemon

# 3. 启动守护进程
openclaw gateway start
```

**从守护进程切换到前台模式：**
```bash
# 1. 停止守护进程
openclaw gateway stop

# 2. 前台启动
openclaw gateway --verbose
```

---

## 实际应用示例

### 示例 1：开发工作流

```bash
# 开发阶段：使用前台模式
openclaw gateway --verbose --dev

# 测试功能
openclaw agent --message "Test"

# 修改代码后，Ctrl+C 停止，重新启动
openclaw gateway --verbose --dev

# 开发完成后，切换到守护进程
openclaw gateway stop  # 如果之前有守护进程在运行
openclaw onboard --install-daemon
```

### 示例 2：服务器部署

```bash
# SSH 连接到服务器
ssh user@production-server

# 安装 OpenClaw
npm install -g openclaw@latest

# 配置 Gateway
openclaw onboard --install-daemon

# 验证运行
openclaw gateway status

# 查看日志
openclaw logs --tail 50

# 断开 SSH（Gateway 继续运行）
exit
```

### 示例 3：故障排查

```bash
# 1. 停止守护进程
openclaw gateway stop

# 2. 前台启动查看详细日志
openclaw gateway --verbose --ws-log full

# 3. 观察日志，定位问题

# 4. 修复问题后，重新启动守护进程
openclaw gateway start
```

---

## 常见问题

### Q1: 如何判断当前使用的是哪种模式？

```bash
# 查看 Gateway 状态
openclaw gateway status

# 输出示例（守护进程模式）：
# Gateway Status:
#   Service: running (launchd)
#   PID: 12345
#   Port: 18789
#   Uptime: 2h 34m

# 输出示例（前台模式）：
# Gateway Status:
#   Service: not installed
#   RPC: reachable (ws://127.0.0.1:18789)
```

### Q2: 前台模式下如何后台运行？

**不推荐**，但可以使用 `nohup` 或 `screen`：

```bash
# 使用 nohup（不推荐）
nohup openclaw gateway > gateway.log 2>&1 &

# 使用 screen（不推荐）
screen -S openclaw
openclaw gateway
# Ctrl+A, D 分离会话

# 推荐：直接使用守护进程模式
openclaw onboard --install-daemon
```

### Q3: 守护进程模式下如何查看实时日志？

```bash
# 使用 OpenClaw CLI
openclaw logs --follow

# 或直接 tail 日志文件
tail -f ~/.openclaw/logs/gateway.log

# Linux systemd
journalctl --user -u openclaw-gateway -f
```

### Q4: 如何在开发时使用守护进程？

```bash
# 使用 dev profile
openclaw onboard --install-daemon --profile dev

# 启动 dev profile 守护进程
openclaw gateway start --profile dev

# 查看 dev profile 日志
openclaw logs --follow --profile dev
```

---

## 参考资源

### 官方文档

- [OpenClaw CLI Gateway 文档](https://docs.openclaw.ai/cli/gateway) - Gateway 命令详解
- [Gateway Runbook](https://docs.openclaw.ai/gateway) - Gateway 运行手册
- [Getting Started](https://docs.openclaw.ai/start/getting-started) - 快速入门指南

### 源码参考

- `sourcecode/openclaw/src/cli/gateway-cli/run.ts:101-353` - Gateway 启动实现
- `sourcecode/openclaw/src/cli/gateway-cli/run-loop.ts` - Gateway 运行循环

### 社区资源

- [OpenClaw GitHub](https://github.com/openclaw/openclaw) - 官方仓库
- [OpenClaw Architecture Explained](https://ppaolo.substack.com/p/openclaw-system-architecture-overview) - 架构详解

---

## 总结

Gateway 启动模式是 OpenClaw 运行时控制的基础：

- **前台模式**：适合开发调试，实时反馈，快速迭代
- **守护进程模式**：适合生产部署，持久运行，自动恢复

选择合适的启动模式可以显著提升开发效率和生产可靠性。在开发阶段使用前台模式快速迭代，在生产环境使用守护进程模式确保服务稳定运行。
