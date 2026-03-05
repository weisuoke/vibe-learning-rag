# 核心概念 2：macOS App 引导流程

**本文档讲解 macOS App 的 Onboarding 流程，对比 CLI Wizard 的差异，以及 GUI 特有的设计和问题。**

---

## 概述

OpenClaw 提供两种 Onboarding 方式：
1. **CLI Wizard**：命令行交互式向导（跨平台）
2. **macOS App**：图形界面引导流程（仅 macOS）

macOS App 提供更友好的图形界面，但底层仍然调用 CLI 工具完成配置。

---

## macOS App vs CLI Wizard 对比

### 核心差异

| 维度 | CLI Wizard | macOS App |
|------|-----------|-----------|
| **界面** | 命令行文本界面 | 原生 macOS GUI |
| **交互方式** | 键盘输入 + 选择 | 鼠标点击 + 表单输入 |
| **Gateway 管理** | 手动启动/停止 | 自动管理（LaunchAgent） |
| **配置方式** | 交互式问答 | 表单填写 |
| **错误提示** | 文本错误信息 | 图形化错误对话框 |
| **适用场景** | 开发者、服务器部署 | 普通用户、桌面使用 |
| **可用性** | 所有平台（macOS/Linux/Windows） | 仅 macOS |

### 相同点

- 都配置相同的核心组件（Model/Auth、Workspace、Gateway、Channels）
- 都生成相同的配置文件（`~/.openclaw/openclaw.json`）
- 都安装相同的守护进程（LaunchAgent）
- 都进行相同的健康检查

---

## macOS App Onboarding 流程

### 整体流程图

```
用户启动 OpenClaw.app（首次）
    ↓
┌─────────────────────────────────────┐
│  1. 欢迎屏幕 (Welcome Screen)       │
│  - 显示 OpenClaw Logo               │
│  - 简介和功能说明                   │
│  - "Get Started" 按钮               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 模式选择 (Setup Mode)           │
│  - This Mac (Local)                 │
│  - Remote Gateway                   │
│  - Configure Later                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Model/Auth 配置                 │
│  - Provider 选择（下拉菜单）        │
│  - API Key 输入（密码框）           │
│  - Model 选择（下拉菜单）           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Gateway 配置                    │
│  - Port 输入（文本框）              │
│  - Bind 选择（单选按钮）            │
│  - Auth 配置（Token/Password）      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. Gateway 启动与健康检查          │
│  - 安装 LaunchAgent                 │
│  - 启动 Gateway                     │
│  - 等待就绪（进度条）               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  6. 完成屏幕 (Completion)           │
│  - 显示配置摘要                     │
│  - "Open Dashboard" 按钮            │
│  - "Start Chatting" 按钮            │
└─────────────────────────────────────┘
```

---

## 阶段 1：欢迎屏幕

### UI 设计

```
┌─────────────────────────────────────┐
│                                     │
│         🦞 OpenClaw                 │
│                                     │
│   Your Personal AI Assistant        │
│                                     │
│   • Chat with Claude via any app    │
│   • Automate tasks with skills      │
│   • Keep your data private          │
│                                     │
│         [Get Started]               │
│                                     │
└─────────────────────────────────────┘
```

### 功能

- 展示 OpenClaw 的核心价值
- 提供简洁的功能介绍
- 单一明确的行动按钮（Get Started）

---

## 阶段 2：模式选择

### UI 设计

```
┌─────────────────────────────────────┐
│  How do you want to set up?         │
│                                     │
│  ○ This Mac (Local)                 │
│    Run OpenClaw on this computer    │
│    Recommended for most users       │
│                                     │
│  ○ Remote Gateway                   │
│    Connect to existing gateway      │
│    For advanced setups              │
│                                     │
│  ○ Configure Later                  │
│    Skip setup for now               │
│                                     │
│         [Continue]                  │
│                                     │
└─────────────────────────────────────┘
```

### 选项说明

#### This Mac (Local)

- **行为**：在本机安装和配置 Gateway
- **适用场景**：大多数用户（桌面使用）
- **后续步骤**：完整的配置流程

#### Remote Gateway

- **行为**：配置连接到远程 Gateway
- **适用场景**：高级用户（多机部署）
- **后续步骤**：输入远程 Gateway URL 和 Token

#### Configure Later

- **行为**：跳过配置，稍后手动配置
- **适用场景**：想先探索 App，稍后配置
- **问题**：旧版本中不可逆（已在 PR #8260 修复）

**修复后的行为**：
- 添加菜单栏 "Run Setup..." 选项
- 可以随时重新进入配置流程

---

## 阶段 3：Model/Auth 配置

### UI 设计

```
┌─────────────────────────────────────┐
│  Configure Model Provider           │
│                                     │
│  Provider:                          │
│  [Anthropic (recommended)    ▼]     │
│                                     │
│  API Key:                           │
│  [••••••••••••••••••••••••••]       │
│  Get your key at console.anthropic.com │
│                                     │
│  Default Model:                     │
│  [claude-sonnet-4-5          ▼]     │
│                                     │
│         [Back]  [Continue]          │
│                                     │
└─────────────────────────────────────┘
```

### 表单验证

```swift
// 伪代码示例
func validateAPIKey(_ key: String) -> Bool {
    if key.isEmpty {
        showError("API key is required")
        return false
    }

    if !key.hasPrefix("sk-ant-") {
        showError("Invalid Anthropic API key format")
        return false
    }

    return true
}

func validateForm() -> Bool {
    guard validateAPIKey(apiKeyField.text) else {
        return false
    }

    guard !modelField.text.isEmpty else {
        showError("Please select a model")
        return false
    }

    return true
}
```

### Provider 选项

| Provider | API Key 格式 | 验证方式 |
|----------|-------------|---------|
| **Anthropic** | `sk-ant-...` | 前缀验证 |
| **OpenAI** | `sk-...` | 前缀验证 |
| **Custom** | 任意 | 无验证（用户自定义） |

---

## 阶段 4：Gateway 配置

### UI 设计

```
┌─────────────────────────────────────┐
│  Configure Gateway                  │
│                                     │
│  Port:                              │
│  [18789                      ]      │
│                                     │
│  Bind Address:                      │
│  ○ Loopback (127.0.0.1)             │
│    Recommended for local use        │
│  ○ LAN (0.0.0.0)                    │
│    Allow network access             │
│                                     │
│  Authentication:                    │
│  ○ Token (recommended)              │
│  ○ Password                         │
│                                     │
│  [Auto-generate token]              │
│                                     │
│         [Back]  [Continue]          │
│                                     │
└─────────────────────────────────────┘
```

### 默认值

macOS App 使用与 CLI QuickStart 相同的默认值：

```swift
let defaultGatewayConfig = GatewayConfig(
    port: 18789,
    bind: .loopback,  // 127.0.0.1
    authMode: .token,
    token: generateRandomToken(),  // 自动生成
    tailscaleMode: .off
)
```

### Token 生成

```swift
func generateRandomToken() -> String {
    let bytes = (0..<16).map { _ in UInt8.random(in: 0...255) }
    let hex = bytes.map { String(format: "%02x", $0) }.joined()
    return "oc_\(hex)"
}
```

---

## 阶段 5：Gateway 启动与健康检查

### UI 设计

```
┌─────────────────────────────────────┐
│  Starting Gateway...                │
│                                     │
│  [████████████░░░░░░░░░░] 60%       │
│                                     │
│  • Installing LaunchAgent...  ✓     │
│  • Starting Gateway service... ✓    │
│  • Waiting for Gateway ready... ⏳  │
│                                     │
│  This may take a few moments        │
│                                     │
└─────────────────────────────────────┘
```

### 启动流程

```swift
func startGateway() async throws {
    // 1. 安装 LaunchAgent
    updateProgress("Installing LaunchAgent...", 20)
    try await installLaunchAgent()

    // 2. 启动 Gateway 服务
    updateProgress("Starting Gateway service...", 40)
    try await startGatewayService()

    // 3. 等待 Gateway 就绪
    updateProgress("Waiting for Gateway ready...", 60)
    try await waitForGatewayReady(timeout: 12.0)

    // 4. 完成
    updateProgress("Gateway is ready!", 100)
}
```

### 健康检查实现

```swift
func waitForGatewayReady(timeout: TimeInterval) async throws {
    let startTime = Date()
    let gatewayURL = URL(string: "http://127.0.0.1:\(port)/health")!

    while Date().timeIntervalSince(startTime) < timeout {
        do {
            var request = URLRequest(url: gatewayURL)
            if let token = gatewayToken {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            let (_, response) = try await URLSession.shared.data(for: request)

            if let httpResponse = response as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                return  // Gateway 就绪
            }
        } catch {
            // Gateway 未启动，继续轮询
        }

        try await Task.sleep(nanoseconds: 1_000_000_000)  // 1 秒
    }

    throw GatewayError.notReady
}
```

---

## 阶段 6：完成屏幕

### UI 设计

```
┌─────────────────────────────────────┐
│  Setup Complete! 🎉                 │
│                                     │
│  Your OpenClaw is ready to use      │
│                                     │
│  Configuration Summary:             │
│  • Model: claude-sonnet-4-5         │
│  • Gateway: http://127.0.0.1:18789  │
│  • Auth: Token                      │
│                                     │
│  What's next?                       │
│                                     │
│  [Open Dashboard]  [Start Chatting] │
│                                     │
│  You can also configure channels    │
│  later via Settings → Channels      │
│                                     │
└─────────────────────────────────────┘
```

### 行动按钮

#### Open Dashboard

```swift
func openDashboard() {
    let dashboardURL = URL(string: "http://127.0.0.1:\(port)")!
    NSWorkspace.shared.open(dashboardURL)
}
```

#### Start Chatting

```swift
func startChatting() {
    // 打开 TUI 或 WebChat
    let chatWindow = ChatWindowController()
    chatWindow.showWindow(nil)
}
```

---

## macOS App 特有功能

### 1. 菜单栏集成

```
┌─────────────────────────────────────┐
│  🦞 OpenClaw                        │
│  ────────────────────────────────   │
│  Open Dashboard                     │
│  Start Chat                         │
│  ────────────────────────────────   │
│  Gateway Status: Running ✓          │
│  ────────────────────────────────   │
│  Run Setup...                       │  ← PR #8260 添加
│  Settings...                        │
│  ────────────────────────────────   │
│  Quit OpenClaw                      │
└─────────────────────────────────────┘
```

### 2. 系统通知

```swift
func showNotification(title: String, message: String) {
    let notification = NSUserNotification()
    notification.title = title
    notification.informativeText = message
    notification.soundName = NSUserNotificationDefaultSoundName

    NSUserNotificationCenter.default.deliver(notification)
}

// 使用示例
showNotification(
    title: "Gateway Started",
    message: "OpenClaw is ready to use"
)
```

### 3. Dock 图标状态

```swift
enum GatewayStatus {
    case running
    case stopped
    case error
}

func updateDockIcon(status: GatewayStatus) {
    switch status {
    case .running:
        NSApp.dockTile.badgeLabel = "✓"
    case .stopped:
        NSApp.dockTile.badgeLabel = "●"
    case .error:
        NSApp.dockTile.badgeLabel = "!"
    }
}
```

---

## 常见问题与解决方案

### 问题 1：Gateway 未就绪（一直卡在等待）

**症状**：

```
Starting Gateway...
[████████████░░░░░░░░░░] 60%
• Installing LaunchAgent...  ✓
• Starting Gateway service... ✓
• Waiting for Gateway ready... ⏳
```

一直卡在 "Waiting for Gateway ready..."，最终超时。

**根本原因**：

1. **CLI 不在 PATH**：LaunchAgent 无法找到 `openclaw` 命令
2. **端口被占用**：18789 端口已被其他进程占用
3. **权限问题**：Accessibility 权限未授予

**解决方案**：

```bash
# 1. 确认 CLI 在 PATH
which openclaw
# 应该输出：/usr/local/bin/openclaw

# 2. 检查端口占用
lsof -i :18789

# 3. 检查 Accessibility 权限
# System Settings → Privacy & Security → Accessibility
# 启用 OpenClaw

# 4. 查看日志
tail -f ~/Library/Logs/OpenClaw/gateway.log
```

**详细排查**：查看 `temp/troubleshooting/macos_gateway_readiness_issue_6156.md`

### 问题 2："Configure Later" 不可逆（旧版本）

**症状**：

选择 "Configure Later" 后，无法重新进入配置流程。

**根本原因**：

旧版本没有提供重新运行配置的入口。

**解决方案**：

**方式 1：升级到新版本**（推荐）

```bash
# 下载最新版本
# PR #8260 已添加 "Run Setup..." 菜单项
```

**方式 2：使用 CLI 向导**

```bash
openclaw onboard
```

**方式 3：手动删除配置**

```bash
rm ~/.openclaw/openclaw.json
# 重启 App，会重新显示配置向导
```

### 问题 3：LaunchAgent 安装失败

**症状**：

```
Error: Failed to install LaunchAgent
```

**根本原因**：

- `~/Library/LaunchAgents/` 目录不存在
- 权限不足
- LaunchAgent 文件格式错误

**解决方案**：

```bash
# 1. 创建目录
mkdir -p ~/Library/LaunchAgents

# 2. 检查权限
ls -la ~/Library/LaunchAgents

# 3. 手动安装
openclaw gateway install

# 4. 验证
launchctl list | grep openclaw
```

---

## macOS App 与 CLI 的协作

### 底层实现

macOS App 实际上是 CLI 的图形化封装：

```swift
// macOS App 调用 CLI
func runOnboarding() async throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/local/bin/openclaw")
    process.arguments = [
        "onboard",
        "--non-interactive",
        "--auth-choice", "anthropic-api-key",
        "--anthropic-api-key", apiKey,
        "--gateway-port", String(port),
        "--gateway-bind", "loopback",
    ]

    try process.run()
    process.waitUntilExit()

    if process.terminationStatus != 0 {
        throw OnboardingError.cliFailed
    }
}
```

### 配置文件共享

macOS App 和 CLI 共享相同的配置文件：

```
~/.openclaw/openclaw.json
```

这意味着：
- 在 App 中配置后，CLI 可以直接使用
- 在 CLI 中修改配置，App 会自动读取
- 两者完全兼容

---

## 设计理念

### 1. 简化复杂性

macOS App 隐藏了 CLI 的复杂性：
- 无需理解命令行参数
- 无需手动编辑配置文件
- 无需理解 LaunchAgent 配置

### 2. 渐进式披露

与 CLI QuickStart 类似：
- 默认使用安全配置
- 高级选项隐藏在 Settings 中
- 用户可以稍后调整

### 3. 即时反馈

GUI 提供更好的反馈：
- 进度条显示启动进度
- 实时错误提示
- 系统通知

### 4. 原生体验

充分利用 macOS 特性：
- 菜单栏集成
- Dock 图标状态
- 系统通知
- Accessibility 权限

---

## 最佳实践

### 首次使用

1. **选择 "This Mac (Local)"**（推荐）
2. **使用 Anthropic Provider**（推荐）
3. **使用默认 Gateway 配置**（Loopback + Token）
4. **完成后立即测试**（Open Dashboard）

### 故障排查

1. **查看日志**：`~/Library/Logs/OpenClaw/gateway.log`
2. **检查 CLI**：`which openclaw`
3. **检查权限**：System Settings → Privacy & Security
4. **重新安装**：删除配置文件，重启 App

### 高级配置

1. **使用 CLI 进行高级配置**：`openclaw configure`
2. **手动编辑配置文件**：`~/.openclaw/openclaw.json`
3. **配置 Channels**：Settings → Channels

---

## 总结

macOS App Onboarding 流程的特点：

1. **图形化界面**：更友好的用户体验
2. **自动化管理**：自动安装和管理 LaunchAgent
3. **即时反馈**：进度条、通知、状态显示
4. **CLI 兼容**：底层调用 CLI，配置文件共享
5. **原生集成**：菜单栏、Dock、系统通知

理解 macOS App 与 CLI 的关系，可以帮助你更好地使用和排查问题。
