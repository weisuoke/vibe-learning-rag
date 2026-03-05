# OpenClaw 网关深度故障排查

**来源**: `sourcecode/openclaw/docs/gateway/troubleshooting.md`

## 命令梯队

首先按顺序运行这些命令：

```bash
openclaw status
openclaw gateway status
openclaw logs --follow
openclaw doctor
openclaw channels status --probe
```

## 预期健康信号

- `openclaw gateway status` 显示 `Runtime: running` 和 `RPC probe: ok`
- `openclaw doctor` 报告无阻塞性配置/服务问题
- `openclaw channels status --probe` 显示 connected/ready 通道

## 深度故障排查场景

### 1. 无回复 (No replies)

如果通道已启动但无响应，在重新连接任何内容之前检查路由和策略。

**诊断命令**:
```bash
openclaw status
openclaw channels status --probe
openclaw pairing list <channel>
openclaw config get channels
openclaw logs --follow
```

**查找内容**:
- DM 发送者的配对待定
- 组提及门控（`requireMention`, `mentionPatterns`）
- 通道/组允许列表不匹配

**常见特征**:
- `drop guild message (mention required` → 消息被忽略直到提及
- `pairing request` → 发送者需要批准
- `blocked` / `allowlist` → 发送者/通道被策略过滤

### 2. 控制面板 UI 连接性 (Dashboard control ui connectivity)

当控制面板/控制 UI 无法连接时，验证 URL、认证模式和安全上下文假设。

**诊断命令**:
```bash
openclaw gateway status
openclaw status
openclaw logs --follow
openclaw doctor
openclaw gateway status --json
```

**查找内容**:
- 正确的探测 URL 和控制面板 URL
- 客户端和网关之间的认证模式/token 不匹配
- 需要设备身份的 HTTP 使用

**常见特征**:
- `device identity required` → 非安全上下文或缺少设备认证
- `unauthorized` / 重连循环 → token/password 不匹配
- `gateway connect failed:` → 错误的主机/端口/URL 目标

### 3. 网关服务未运行 (Gateway service not running)

当服务已安装但进程无法保持运行时使用此方法。

**诊断命令**:
```bash
openclaw gateway status
openclaw status
openclaw logs --follow
openclaw doctor
openclaw gateway status --deep
```

**查找内容**:
- `Runtime: stopped` 带退出提示
- 服务配置不匹配（`Config (cli)` vs `Config (service)`）
- 端口/监听器冲突

**常见特征**:
- `Gateway start blocked: set gateway.mode=local` → 本地网关模式未启用
  - 修复：在配置中设置 `gateway.mode="local"`（或运行 `openclaw configure`）
  - 如果通过 Podman 使用专用 `openclaw` 用户运行，配置位于 `~openclaw/.openclaw/openclaw.json`
- `refusing to bind gateway ... without auth` → 非回环绑定没有 token/password
- `another gateway instance is already listening` / `EADDRINUSE` → 端口冲突

### 4. 通道连接但消息不流动 (Channel connected messages not flowing)

如果通道状态已连接但消息流已死，关注策略、权限和通道特定的传递规则。

**诊断命令**:
```bash
openclaw channels status --probe
openclaw pairing list <channel>
openclaw status --deep
openclaw logs --follow
openclaw config get channels
```

**查找内容**:
- DM 策略（`pairing`, `allowlist`, `open`, `disabled`）
- 组允许列表和提及要求
- 缺少通道 API 权限/范围

**常见特征**:
- `mention required` → 消息被组提及策略忽略
- `pairing` / 待批准跟踪 → 发送者未批准
- `missing_scope`, `not_in_channel`, `Forbidden`, `401/403` → 通道认证/权限问题

### 5. Cron 和心跳传递 (Cron and heartbeat delivery)

如果 cron 或心跳未运行或未传递，首先验证调度器状态，然后验证传递目标。

**诊断命令**:
```bash
openclaw cron status
openclaw cron list
openclaw cron runs --id <jobId> --limit 20
openclaw system heartbeat last
openclaw logs --follow
```

**查找内容**:
- Cron 已启用且存在下次唤醒
- 作业运行历史状态（`ok`, `skipped`, `error`）
- 心跳跳过原因（`quiet-hours`, `requests-in-flight`, `alerts-disabled`）

**常见特征**:
- `cron: scheduler disabled; jobs will not run automatically` → cron 已禁用
- `cron: timer tick failed` → 调度器滴答失败；检查文件/日志/运行时错误
- `heartbeat skipped` with `reason=quiet-hours` → 在活动时间窗口之外
- `heartbeat: unknown accountId` → 心跳传递目标的账户 ID 无效

### 6. 节点配对但工具失败 (Node paired tool fails)

如果节点已配对但工具失败，隔离前台、权限和批准状态。

**诊断命令**:
```bash
openclaw nodes status
openclaw nodes describe --node <idOrNameOrIp>
openclaw approvals get --node <idOrNameOrIp>
openclaw logs --follow
openclaw status
```

**查找内容**:
- 节点在线且具有预期功能
- 相机/麦克风/位置/屏幕的 OS 权限授予
- Exec 批准和允许列表状态

**常见特征**:
- `NODE_BACKGROUND_UNAVAILABLE` → 节点应用必须在前台
- `*_PERMISSION_REQUIRED` / `LOCATION_PERMISSION_REQUIRED` → 缺少 OS 权限
- `SYSTEM_RUN_DENIED: approval required` → exec 批准待定
- `SYSTEM_RUN_DENIED: allowlist miss` → 命令被允许列表阻止

### 7. 浏览器工具失败 (Browser tool fails)

当浏览器工具操作失败时使用此方法，即使网关本身是健康的。

**诊断命令**:
```bash
openclaw browser status
openclaw browser start --browser-profile openclaw
openclaw browser profiles
openclaw logs --follow
openclaw doctor
```

**查找内容**:
- 有效的浏览器可执行路径
- CDP 配置文件可达性
- `profile="chrome"` 的扩展中继选项卡附件

**常见特征**:
- `Failed to start Chrome CDP on port` → 浏览器进程启动失败
- `browser.executablePath not found` → 配置的路径无效
- `Chrome extension relay is running, but no tab is connected` → 扩展中继未附加
- `Browser attachOnly is enabled ... not reachable` → 仅附加配置文件没有可达目标

## 升级后突然中断

大多数升级后的中断是配置漂移或现在强制执行的更严格默认值。

### 1) 认证和 URL 覆盖行为已更改

```bash
openclaw gateway status
openclaw config get gateway.mode
openclaw config get gateway.remote.url
openclaw config get gateway.auth.mode
```

**检查内容**:
- 如果 `gateway.mode=remote`，CLI 调用可能针对远程，而本地服务正常
- 显式 `--url` 调用不会回退到存储的凭据

**常见特征**:
- `gateway connect failed:` → 错误的 URL 目标
- `unauthorized` → 端点可达但认证错误

### 2) 绑定和认证防护更严格

```bash
openclaw config get gateway.bind
openclaw config get gateway.auth.token
openclaw gateway status
openclaw logs --follow
```

**检查内容**:
- 非回环绑定（`lan`, `tailnet`, `custom`）需要配置认证
- 旧键如 `gateway.token` 不替换 `gateway.auth.token`

**常见特征**:
- `refusing to bind gateway ... without auth` → 绑定+认证不匹配
- `RPC probe: failed` 而运行时正在运行 → 网关活动但使用当前认证/URL 无法访问

### 3) 配对和设备身份状态已更改

```bash
openclaw devices list
openclaw pairing list <channel>
openclaw logs --follow
openclaw doctor
```

**检查内容**:
- 控制面板/节点的待定设备批准
- 策略或身份更改后的待定 DM 配对批准

**常见特征**:
- `device identity required` → 设备认证未满足
- `pairing required` → 发送者/设备必须批准

如果服务配置和运行时在检查后仍然不一致，从相同的配置文件/状态目录重新安装服务元数据：

```bash
openclaw gateway install --force
openclaw gateway restart
```

## 核心概念

1. **分层诊断**: 从快速检查到深度探测的系统化方法
2. **日志特征识别**: 通过特定日志消息快速定位问题类型
3. **配置验证**: 检查配置文件和运行时状态的一致性
4. **权限管理**: OS 级权限、API 权限、配对批准的多层权限系统
5. **升级兼容性**: 升级后的配置迁移和兼容性检查
6. **服务状态管理**: 守护进程、端口、进程状态的管理
7. **认证模式**: token、password、设备身份等多种认证方式
