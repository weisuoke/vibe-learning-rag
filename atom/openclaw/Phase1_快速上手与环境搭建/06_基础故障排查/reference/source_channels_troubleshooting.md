# OpenClaw 通道级故障排查

**来源**: `sourcecode/openclaw/docs/channels/troubleshooting.md`

## 命令梯队

按顺序首先运行这些命令：

```bash
openclaw status
openclaw gateway status
openclaw logs --follow
openclaw doctor
openclaw channels status --probe
```

健康基线：
- `Runtime: running`
- `RPC probe: ok`
- 通道探测显示 connected/ready

## WhatsApp 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| 已连接但无 DM 回复 | `openclaw pairing list whatsapp` | 批准发送者或切换 DM 策略/允许列表 |
| 组消息被忽略 | 检查配置中的 `requireMention` + 提及模式 | 提及机器人或放宽该组的提及策略 |
| 随机断开/重新登录循环 | `openclaw channels status --probe` + logs | 重新登录并验证凭据目录健康 |

## Telegram 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| `/start` 但无可用回复流程 | `openclaw pairing list telegram` | 批准配对或更改 DM 策略 |
| 机器人在线但组保持沉默 | 验证提及要求和机器人隐私模式 | 禁用组可见性的隐私模式或提及机器人 |
| 发送失败，网络错误 | 检查日志中的 Telegram API 调用失败 | 修复到 `api.telegram.org` 的 DNS/IPv6/代理路由 |
| 升级后允许列表阻止你 | `openclaw security audit` 和配置允许列表 | 运行 `openclaw doctor --fix` 或用数字发送者 ID 替换 `@username` |

## Discord 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| 机器人在线但无公会回复 | `openclaw channels status --probe` | 允许公会/频道并验证消息内容意图 |
| 组消息被忽略 | 检查日志中的提及门控丢弃 | 提及机器人或设置公会/频道 `requireMention: false` |
| DM 回复缺失 | `openclaw pairing list discord` | 批准 DM 配对或调整 DM 策略 |

## Slack 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| Socket 模式已连接但无响应 | `openclaw channels status --probe` | 验证应用 token + 机器人 token 和所需范围 |
| DM 被阻止 | `openclaw pairing list slack` | 批准配对或放宽 DM 策略 |
| 频道消息被忽略 | 检查 `groupPolicy` 和频道允许列表 | 允许频道或将策略切换到 `open` |

## iMessage 和 BlueBubbles 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| 无入站事件 | 验证 webhook/服务器可达性和应用权限 | 修复 webhook URL 或 BlueBubbles 服务器状态 |
| 可以发送但 macOS 上无接收 | 检查 macOS 隐私权限以进行消息自动化 | 重新授予 TCC 权限并重启通道进程 |
| DM 发送者被阻止 | `openclaw pairing list imessage` 或 `openclaw pairing list bluebubbles` | 批准配对或更新允许列表 |

## Signal 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| 守护进程可达但机器人沉默 | `openclaw channels status --probe` | 验证 `signal-cli` 守护进程 URL/账户和接收模式 |
| DM 被阻止 | `openclaw pairing list signal` | 批准发送者或调整 DM 策略 |
| 组回复不触发 | 检查组允许列表和提及模式 | 添加发送者/组或放宽门控 |

## Matrix 故障特征

| 症状 | 最快检查 | 修复 |
|------|---------|------|
| 已登录但忽略房间消息 | `openclaw channels status --probe` | 检查 `groupPolicy` 和房间允许列表 |
| DM 不处理 | `openclaw pairing list matrix` | 批准发送者或调整 DM 策略 |
| 加密房间失败 | 验证加密模块和加密设置 | 启用加密支持并重新加入/同步房间 |

## 核心概念

1. **通道特定故障**: 每个通道有其独特的故障模式
2. **配对机制**: DM 发送者需要批准才能交互
3. **提及门控**: 组消息可能需要提及机器人才能触发
4. **权限和范围**: 通道 API 需要正确的权限配置
5. **允许列表策略**: 通过允许列表控制谁可以与机器人交互
