# OpenClaw doctor 命令

**来源**: `sourcecode/openclaw/docs/cli/doctor.md`

## 概述

`openclaw doctor` 是网关和通道的健康检查 + 快速修复工具。

## 示例

```bash
openclaw doctor
openclaw doctor --repair
openclaw doctor --deep
```

## 注意事项

- 交互式提示（如钥匙串/OAuth 修复）仅在 stdin 是 TTY 且**未**设置 `--non-interactive` 时运行
- 无头运行（cron、Telegram、无终端）将跳过提示
- `--fix`（`--repair` 的别名）将备份写入 `~/.openclaw/openclaw.json.bak` 并删除未知配置键，列出每个删除

## macOS: `launchctl` 环境覆盖

如果你之前运行过 `launchctl setenv OPENCLAW_GATEWAY_TOKEN ...`（或 `...PASSWORD`），该值会覆盖你的配置文件并可能导致持续的"未授权"错误。

```bash
launchctl getenv OPENCLAW_GATEWAY_TOKEN
launchctl getenv OPENCLAW_GATEWAY_PASSWORD

launchctl unsetenv OPENCLAW_GATEWAY_TOKEN
launchctl unsetenv OPENCLAW_GATEWAY_PASSWORD
```

## 核心功能

1. **健康检查**: 验证配置和服务状态
2. **自动修复**: 使用 `--repair` 标志自动修复常见问题
3. **配置备份**: 修复前自动备份配置
4. **环境变量检查**: 检测可能导致问题的环境变量覆盖
5. **交互式修复**: 在 TTY 环境中提供交互式修复选项

## 使用场景

- 升级后验证配置
- 诊断连接问题
- 修复配置错误
- 清理未知配置键
- 检查环境变量冲突
