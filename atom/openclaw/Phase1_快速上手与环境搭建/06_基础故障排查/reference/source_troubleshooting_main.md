# OpenClaw 故障排查主文档

**来源**: `sourcecode/openclaw/docs/help/troubleshooting.md`

## 60秒快速诊断流程

```bash
openclaw status
openclaw status --all
openclaw gateway probe
openclaw gateway status
openclaw doctor
openclaw channels status --probe
openclaw logs --follow
```

## 健康输出标准

- `openclaw status` → 显示已配置的通道，无明显认证错误
- `openclaw status --all` → 完整报告可分享
- `openclaw gateway probe` → 预期网关目标可达
- `openclaw gateway status` → `Runtime: running` 和 `RPC probe: ok`
- `openclaw doctor` → 无阻塞性配置/服务错误
- `openclaw channels status --probe` → 通道报告 `connected` 或 `ready`
- `openclaw logs --follow` → 稳定活动，无重复致命错误

## 故障决策树

OpenClaw 提供了一个决策树来诊断不同类型的故障：

### 1. 无回复 (No replies)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw channels status --probe
openclaw pairing list <channel>
openclaw logs --follow
```

**常见日志特征**:
- `drop guild message (mention required` → 提及门控阻止了 Discord 中的消息
- `pairing request` → 发送者未批准，等待 DM 配对批准
- `blocked` / `allowlist` → 发送者、房间或组被过滤

### 2. 控制面板无法连接 (Dashboard or Control UI will not connect)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw logs --follow
openclaw doctor
openclaw channels status --probe
```

**常见日志特征**:
- `device identity required` → HTTP/非安全上下文无法完成设备认证
- `unauthorized` / 重连循环 → 错误的 token/password 或认证模式不匹配
- `gateway connect failed:` → UI 目标错误的 URL/端口或无法访问网关

### 3. 网关无法启动 (Gateway will not start)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw logs --follow
openclaw doctor
openclaw channels status --probe
```

**常见日志特征**:
- `Gateway start blocked: set gateway.mode=local` → 网关模式未设置/远程
- `refusing to bind gateway ... without auth` → 非回环绑定没有 token/password
- `another gateway instance is already listening` / `EADDRINUSE` → 端口已被占用

### 4. 通道连接但消息不流动 (Channel connects but messages do not flow)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw logs --follow
openclaw doctor
openclaw channels status --probe
```

**常见日志特征**:
- `mention required` → 组提及门控阻止了处理
- `pairing` / `pending` → DM 发送者尚未批准
- `not_in_channel`, `missing_scope`, `Forbidden`, `401/403` → 通道权限 token 问题

### 5. Cron 或心跳未触发 (Cron or heartbeat did not fire)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw cron status
openclaw cron list
openclaw cron runs --id <jobId> --limit 20
openclaw logs --follow
```

**常见日志特征**:
- `cron: scheduler disabled; jobs will not run automatically` → cron 已禁用
- `heartbeat skipped` with `reason=quiet-hours` → 在配置的活动时间之外
- `requests-in-flight` → 主通道繁忙；心跳唤醒被推迟
- `unknown accountId` → 心跳传递目标账户不存在

### 6. 节点配对但工具失败 (Node is paired but tool fails)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw nodes status
openclaw nodes describe --node <idOrNameOrIp>
openclaw logs --follow
```

**常见日志特征**:
- `NODE_BACKGROUND_UNAVAILABLE` → 将节点应用带到前台
- `*_PERMISSION_REQUIRED` → OS 权限被拒绝/缺失
- `SYSTEM_RUN_DENIED: approval required` → exec 批准待定
- `SYSTEM_RUN_DENIED: allowlist miss` → 命令不在 exec 允许列表中

### 7. 浏览器工具失败 (Browser tool fails)
**诊断命令**:
```bash
openclaw status
openclaw gateway status
openclaw browser status
openclaw logs --follow
openclaw doctor
```

**常见日志特征**:
- `Failed to start Chrome CDP on port` → 本地浏览器启动失败
- `browser.executablePath not found` → 配置的二进制路径错误
- `Chrome extension relay is running, but no tab is connected` → 扩展未附加
- `Browser attachOnly is enabled ... not reachable` → 仅附加配置文件没有活动的 CDP 目标

## 核心诊断工具

### 1. openclaw status
快速本地摘要：OS + 更新、网关/服务可达性、代理/会话、提供商配置 + 运行时问题

### 2. openclaw status --all
只读诊断，带日志尾部（token 已编辑）

### 3. openclaw gateway status
显示监督器运行时 vs RPC 可达性、探测目标 URL 以及服务可能使用的配置

### 4. openclaw doctor
健康检查 + 配置/状态的快速修复

### 5. openclaw logs --follow
实时日志跟踪

## 关键概念

1. **60秒诊断流程**: 快速运行一系列命令来诊断问题
2. **决策树**: 根据症状选择不同的诊断路径
3. **日志特征识别**: 通过特定的日志消息识别问题类型
4. **分层诊断**: 从快速检查到深度探测
5. **命令组合**: 使用多个命令组合来全面诊断问题
