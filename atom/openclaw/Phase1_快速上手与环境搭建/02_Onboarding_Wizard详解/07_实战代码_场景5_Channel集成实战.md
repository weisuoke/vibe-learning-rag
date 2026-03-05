# 实战代码 场景5：Channel 集成实战

**本文档演示如何集成各种聊天平台 Channel，包括 WhatsApp、Telegram、Discord 的完整配置流程。**

---

## 场景概述

**目标**：将 OpenClaw 连接到主流聊天平台，实现多渠道 AI 助手。

**支持的 Channel**：
- **WhatsApp**：通过 Baileys 库连接（QR 扫码）
- **Telegram**：通过 Bot API 连接（Bot Token）
- **Discord**：通过 Discord.js 连接（Bot Token）
- **Signal**：通过 Signal CLI 连接（手机号验证）
- **BlueBubbles**：通过 REST API 连接（iMessage）

**核心优势**：
- 多平台统一接口
- 独立配置和管理
- 灵活的访问控制
- 会话隔离

---

## WhatsApp 集成

### 前置条件

- 有效的 WhatsApp 账号
- 手机上安装 WhatsApp App
- 稳定的网络连接

### 配置流程

#### 步骤 1：运行 Onboarding Wizard

```bash
# 启动向导
openclaw onboard

# 或使用非交互式模式
openclaw onboard --non-interactive \
  --auth-choice anthropic-api-key \
  --anthropic-api-key "$ANTHROPIC_API_KEY"
```

#### 步骤 2：选择 WhatsApp Channel

```
◆  Which channels do you want to set up?
│  ● WhatsApp
│  ○ Telegram
│  ○ Discord
│  ○ Skip for now
```

选择 `WhatsApp`

#### 步骤 3：扫描 QR 码

```
◇  WhatsApp
│  Scan this QR code with WhatsApp:
│
│  ████████████████████████████
│  ██ ▄▄▄▄▄ █▀ █▀▀██ ▄▄▄▄▄ ██
│  ██ █   █ █▀▄ ▀▄▀█ █   █ ██
│  ██ █▄▄▄█ █▀ ▀ ▄ █ █▄▄▄█ ██
│  ██▄▄▄▄▄▄▄█▄▀ ▀▄█▄▄▄▄▄▄▄██
│  ████████████████████████████
│
│  1. Open WhatsApp on your phone
│  2. Tap Menu or Settings → Linked Devices
│  3. Tap Link a Device
│  4. Scan this QR code
```

**操作步骤**：

1. 打开手机上的 WhatsApp
2. 点击 **设置** → **已连接的设备**
3. 点击 **连接设备**
4. 扫描终端显示的 QR 码

#### 步骤 4：等待连接

```
◇  WhatsApp
│  Waiting for QR scan...
│  Connected! (Phone: +1234567890)
│
│  Session saved to: ~/.openclaw/sessions/whatsapp
```

#### 步骤 5：配置访问策略

```
◆  WhatsApp DM policy
│  ● Pairing (require pairing code)
│  ○ Allowlist (only allowed numbers)
│  ○ Open (anyone can message)
│  ○ Disabled (no DMs)
```

**推荐**：选择 `Pairing`（最安全）

#### 步骤 6：配置群组策略

```
◆  WhatsApp groups
│  ● Require @mention in groups
│  ○ Respond to all messages in groups
```

**推荐**：选择 `Require @mention`（避免消息轰炸）

### 完整自动化脚本

```bash
#!/bin/bash
# setup-whatsapp.sh

set -e

echo "Setting up WhatsApp channel..."

# 1. 配置 OpenClaw（如果未配置）
if ! openclaw status &>/dev/null; then
  openclaw onboard --non-interactive \
    --auth-choice anthropic-api-key \
    --anthropic-api-key "$ANTHROPIC_API_KEY"
fi

# 2. 配置 WhatsApp Channel
openclaw configure --section channels

# 向导会提示：
# - 选择 WhatsApp
# - 扫描 QR 码
# - 配置 DM policy
# - 配置群组策略

# 3. 验证配置
openclaw config get channels.whatsapp

echo "WhatsApp channel configured successfully!"
```

### 手动配置

```json5
// ~/.openclaw/openclaw.json
{
  channels: {
    whatsapp: {
      enabled: true,
      dmPolicy: "pairing",  // pairing | allowlist | open | disabled
      allowFrom: [],  // 白名单（pairing 模式下为空）
      groups: {
        "*": {
          requireMention: true,  // 群组需要 @提及
        },
      },
      sessionStore: "~/.openclaw/sessions/whatsapp",
    },
  },
}
```

### 测试 WhatsApp 连接

```bash
# 1. 检查 Channel 状态
openclaw channels status whatsapp

# 2. 发送测试消息
# 在手机上给自己发送消息："你好"

# 3. 查看日志
openclaw logs gateway | grep whatsapp
```

---

## Telegram 集成

### 前置条件

- Telegram 账号
- 访问 @BotFather

### 配置流程

#### 步骤 1：创建 Telegram Bot

```
1. 打开 Telegram，搜索 @BotFather
2. 发送 /newbot
3. 输入 Bot 名称（如 "My OpenClaw Bot"）
4. 输入 Bot 用户名（如 "my_openclaw_bot"，必须以 _bot 结尾）
5. 获取 Bot Token（如 "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"）
```

#### 步骤 2：配置 Bot Token

```bash
# 方式 1：使用环境变量
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"

# 方式 2：在向导中输入
openclaw configure --section channels
# 选择 Telegram
# 输入 Bot Token
```

#### 步骤 3：配置访问策略

```
◆  Telegram DM policy
│  ● Pairing (require pairing code)
│  ○ Allowlist (only allowed users)
│  ○ Open (anyone can message)
│  ○ Disabled (no DMs)
```

#### 步骤 4：启动 Bot

```bash
# 重启 Gateway 以加载 Telegram Channel
openclaw gateway restart

# 验证 Bot 状态
openclaw channels status telegram
```

#### 步骤 5：测试 Bot

```
1. 在 Telegram 中搜索你的 Bot（@my_openclaw_bot）
2. 发送 /start
3. 如果使用 Pairing 模式，Bot 会返回配对码
4. 批准配对码：openclaw pairing approve <code>
5. 发送消息测试
```

### 完整自动化脚本

```bash
#!/bin/bash
# setup-telegram.sh

set -e

# 检查 Bot Token
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
  echo "Error: TELEGRAM_BOT_TOKEN is not set"
  exit 1
fi

echo "Setting up Telegram channel..."

# 1. 配置 Telegram Channel
cat > /tmp/telegram-config.json <<EOF
{
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "$TELEGRAM_BOT_TOKEN",
      "dmPolicy": "pairing",
      "allowFrom": [],
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    }
  }
}
EOF

# 2. 合并配置
openclaw config merge /tmp/telegram-config.json

# 3. 重启 Gateway
openclaw gateway restart

# 4. 验证
openclaw channels status telegram

echo "Telegram channel configured successfully!"
echo "Bot username: @$(openclaw config get channels.telegram.botUsername)"
```

### 手动配置

```json5
// ~/.openclaw/openclaw.json
{
  channels: {
    telegram: {
      enabled: true,
      botToken: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
      dmPolicy: "pairing",
      allowFrom: [],
      groups: {
        "*": {
          requireMention: true,
        },
      },
    },
  },
}
```

### Telegram Bot 命令

```bash
# 配置 Bot 命令（可选）
# 在 @BotFather 中发送 /setcommands
# 选择你的 Bot
# 输入命令列表：

start - 开始对话
help - 获取帮助
reset - 重置会话
status - 查看状态
```

---

## Discord 集成

### 前置条件

- Discord 账号
- 访问 Discord Developer Portal

### 配置流程

#### 步骤 1：创建 Discord Application

```
1. 访问 https://discord.com/developers/applications
2. 点击 "New Application"
3. 输入 Application 名称（如 "OpenClaw Bot"）
4. 点击 "Create"
```

#### 步骤 2：创建 Bot 用户

```
1. 在左侧菜单选择 "Bot"
2. 点击 "Add Bot"
3. 确认 "Yes, do it!"
4. 复制 Bot Token（点击 "Reset Token" 并复制）
```

#### 步骤 3：配置 Bot 权限

```
1. 在 "Bot" 页面，启用以下 Privileged Gateway Intents：
   - ✅ Presence Intent
   - ✅ Server Members Intent
   - ✅ Message Content Intent

2. 在 "OAuth2" → "URL Generator" 中：
   - Scopes: 选择 "bot"
   - Bot Permissions: 选择以下权限
     - ✅ Read Messages/View Channels
     - ✅ Send Messages
     - ✅ Read Message History
     - ✅ Add Reactions
```

#### 步骤 4：生成邀请链接

```
1. 在 "OAuth2" → "URL Generator" 中
2. 复制生成的 URL（如 https://discord.com/api/oauth2/authorize?client_id=...）
3. 在浏览器中打开 URL
4. 选择要添加 Bot 的服务器
5. 点击 "授权"
```

#### 步骤 5：配置 Bot Token

```bash
# 方式 1：使用环境变量
export DISCORD_BOT_TOKEN="MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GaBcDe.FgHiJkLmNoPqRsTuVwXyZ"

# 方式 2：在向导中输入
openclaw configure --section channels
# 选择 Discord
# 输入 Bot Token
```

#### 步骤 6：配置访问策略

```
◆  Discord DM policy
│  ● Pairing (require pairing code)
│  ○ Allowlist (only allowed users)
│  ○ Open (anyone can message)
│  ○ Disabled (no DMs)

◆  Discord guilds (servers)
│  ● Require @mention in servers
│  ○ Respond to all messages in servers
```

#### 步骤 7：启动 Bot

```bash
# 重启 Gateway
openclaw gateway restart

# 验证 Bot 状态
openclaw channels status discord
```

### 完整自动化脚本

```bash
#!/bin/bash
# setup-discord.sh

set -e

# 检查 Bot Token
if [ -z "$DISCORD_BOT_TOKEN" ]; then
  echo "Error: DISCORD_BOT_TOKEN is not set"
  exit 1
fi

echo "Setting up Discord channel..."

# 1. 配置 Discord Channel
cat > /tmp/discord-config.json <<EOF
{
  "channels": {
    "discord": {
      "enabled": true,
      "botToken": "$DISCORD_BOT_TOKEN",
      "dmPolicy": "pairing",
      "allowFrom": [],
      "guilds": {
        "*": {
          "requireMention": true
        }
      }
    }
  }
}
EOF

# 2. 合并配置
openclaw config merge /tmp/discord-config.json

# 3. 重启 Gateway
openclaw gateway restart

# 4. 验证
openclaw channels status discord

echo "Discord channel configured successfully!"
```

### 手动配置

```json5
// ~/.openclaw/openclaw.json
{
  channels: {
    discord: {
      enabled: true,
      botToken: "MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GaBcDe.FgHiJkLmNoPqRsTuVwXyZ",
      dmPolicy: "pairing",
      allowFrom: [],
      guilds: {
        "*": {
          requireMention: true,
        },
        "123456789012345678": {
          // 特定服务器配置
          requireMention: false,
          channels: {
            "987654321098765432": {
              // 特定频道配置
              enabled: true,
            },
          },
        },
      },
    },
  },
}
```

---

## 多 Channel 环境配置

### 同时配置多个 Channel

```bash
#!/bin/bash
# setup-all-channels.sh

set -e

echo "Setting up all channels..."

# 1. 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice anthropic-api-key \
  --anthropic-api-key "$ANTHROPIC_API_KEY"

# 2. 配置所有 Channel
cat > /tmp/all-channels-config.json <<EOF
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "dmPolicy": "pairing",
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    },
    "telegram": {
      "enabled": true,
      "botToken": "$TELEGRAM_BOT_TOKEN",
      "dmPolicy": "pairing",
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    },
    "discord": {
      "enabled": true,
      "botToken": "$DISCORD_BOT_TOKEN",
      "dmPolicy": "pairing",
      "guilds": {
        "*": {
          "requireMention": true
        }
      }
    }
  }
}
EOF

# 3. 合并配置
openclaw config merge /tmp/all-channels-config.json

# 4. 重启 Gateway
openclaw gateway restart

# 5. 验证所有 Channel
openclaw channels list

echo "All channels configured successfully!"
```

### Channel 路由配置

```json5
{
  channels: {
    whatsapp: {
      enabled: true,
      dmPolicy: "allowlist",
      allowFrom: ["+1234567890"],
    },
    telegram: {
      enabled: true,
      botToken: "...",
      dmPolicy: "allowlist",
      allowFrom: ["tg:123456789"],
    },
    discord: {
      enabled: true,
      botToken: "...",
      dmPolicy: "allowlist",
      allowFrom: ["dc:123456789012345678"],
    },
  },
  routing: {
    rules: [
      {
        channel: "whatsapp",
        from: "+1234567890",
        agent: "work",
      },
      {
        channel: "telegram",
        from: "tg:123456789",
        agent: "personal",
      },
      {
        channel: "discord",
        from: "dc:123456789012345678",
        agent: "main",
      },
    ],
  },
}
```

---

## Channel 管理

### 列出所有 Channel

```bash
# 列出所有 Channel
openclaw channels list

# 输出示例：
# Channels:
# - whatsapp (enabled)
#   Status: Connected
#   Phone: +1234567890
# - telegram (enabled)
#   Status: Running
#   Bot: @my_openclaw_bot
# - discord (enabled)
#   Status: Running
#   Bot: OpenClaw Bot#1234
```

### 启用/禁用 Channel

```bash
# 启用 Channel
openclaw channels enable whatsapp

# 禁用 Channel
openclaw channels disable whatsapp

# 查看 Channel 状态
openclaw channels status whatsapp
```

### 重启 Channel

```bash
# 重启单个 Channel
openclaw channels restart whatsapp

# 重启所有 Channel
openclaw gateway restart
```

### 查看 Channel 日志

```bash
# 查看 WhatsApp 日志
openclaw logs gateway | grep whatsapp

# 查看 Telegram 日志
openclaw logs gateway | grep telegram

# 查看 Discord 日志
openclaw logs gateway | grep discord
```

---

## 配对管理

### 查看配对请求

```bash
# 列出所有配对请求
openclaw pairing list

# 输出示例：
# Pairing Requests:
# - Code: 123456
#   Channel: telegram
#   User: tg:123456789
#   Expires: 2026-02-22 12:00:00
```

### 批准配对请求

```bash
# 批准配对码
openclaw pairing approve 123456

# 批准并添加到白名单
openclaw pairing approve 123456 --add-to-allowlist
```

### 拒绝配对请求

```bash
# 拒绝配对码
openclaw pairing reject 123456
```

### 清理过期配对请求

```bash
# 清理过期的配对请求
openclaw pairing cleanup
```

---

## 故障排查

### 问题 1：WhatsApp QR 码不显示

**症状**：

```
Error: Failed to generate QR code
```

**解决方案**：

```bash
# 1. 检查 Gateway 日志
openclaw logs gateway | grep whatsapp

# 2. 清理旧会话
rm -rf ~/.openclaw/sessions/whatsapp

# 3. 重新配置
openclaw configure --section channels
# 选择 WhatsApp

# 4. 手动启动 WhatsApp Channel
openclaw channels start whatsapp --debug
```

### 问题 2：Telegram Bot 无响应

**症状**：

Bot 不回复消息

**解决方案**：

```bash
# 1. 验证 Bot Token
curl https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe

# 2. 检查 Bot 权限
# 确保 Bot 有 "Read Messages" 权限

# 3. 检查 Channel 状态
openclaw channels status telegram

# 4. 重启 Channel
openclaw channels restart telegram

# 5. 查看日志
openclaw logs gateway | grep telegram
```

### 问题 3：Discord Bot 离线

**症状**：

Bot 显示离线状态

**解决方案**：

```bash
# 1. 检查 Bot Token
openclaw config get channels.discord.botToken

# 2. 检查 Intents
# 确保启用了 Message Content Intent

# 3. 检查 Channel 状态
openclaw channels status discord

# 4. 重启 Channel
openclaw channels restart discord

# 5. 查看连接日志
openclaw logs gateway | grep discord | grep -i "connect"
```

### 问题 4：配对码过期

**症状**：

```
Error: Pairing code expired
```

**解决方案**：

```bash
# 1. 延长配对码有效期
openclaw config set channels.telegram.pairingCodeExpiry "30m"

# 2. 重新生成配对码
# 用户重新发送 /start

# 3. 立即批准
openclaw pairing approve <code>
```

---

## 最佳实践

### 1. 使用 Pairing 策略

```json5
{
  channels: {
    whatsapp: {
      dmPolicy: "pairing",  // ✅ 推荐
    },
    telegram: {
      dmPolicy: "pairing",  // ✅ 推荐
    },
    discord: {
      dmPolicy: "pairing",  // ✅ 推荐
    },
  },
}
```

### 2. 启用群组 Mention 要求

```json5
{
  channels: {
    whatsapp: {
      groups: {
        "*": {
          requireMention: true,  // ✅ 推荐
        },
      },
    },
  },
}
```

### 3. 定期审计白名单

```bash
# 查看所有白名单
openclaw channels allowlist list --all

# 移除不活跃用户
openclaw channels allowlist prune --inactive-days 90
```

### 4. 备份 Channel 配置

```bash
# 备份配置
cp ~/.openclaw/openclaw.json ~/backups/openclaw-$(date +%Y%m%d).json

# 备份 WhatsApp 会话
tar -czf ~/backups/whatsapp-session-$(date +%Y%m%d).tar.gz \
  ~/.openclaw/sessions/whatsapp
```

### 5. 监控 Channel 状态

```bash
#!/bin/bash
# monitor-channels.sh

while true; do
  echo "[$(date)] Checking channel status..."

  # 检查所有 Channel
  openclaw channels list --json | jq -r '.[] | "\(.name): \(.status)"'

  # 如果有 Channel 离线，发送告警
  if openclaw channels list --json | jq -e '.[] | select(.status != "running")' > /dev/null; then
    echo "Warning: Some channels are offline!"
    # 发送告警（如邮件、Slack 等）
  fi

  sleep 300  # 每 5 分钟检查一次
done
```

---

## 完整部署示例

### 生产环境 Channel 部署

```bash
#!/bin/bash
# production-channels-deploy.sh

set -e

echo "Deploying channels for production..."

# 1. 检查必需的环境变量
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY is not set"
  exit 1
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
  echo "Error: TELEGRAM_BOT_TOKEN is not set"
  exit 1
fi

if [ -z "$DISCORD_BOT_TOKEN" ]; then
  echo "Error: DISCORD_BOT_TOKEN is not set"
  exit 1
fi

# 2. 配置 OpenClaw
openclaw onboard --non-interactive \
  --auth-choice anthropic-api-key \
  --anthropic-api-key "$ANTHROPIC_API_KEY"

# 3. 配置所有 Channel
cat > /tmp/production-channels.json <<EOF
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "dmPolicy": "allowlist",
      "allowFrom": ["+1234567890"],
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    },
    "telegram": {
      "enabled": true,
      "botToken": "$TELEGRAM_BOT_TOKEN",
      "dmPolicy": "allowlist",
      "allowFrom": ["tg:123456789"],
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    },
    "discord": {
      "enabled": true,
      "botToken": "$DISCORD_BOT_TOKEN",
      "dmPolicy": "allowlist",
      "allowFrom": ["dc:123456789012345678"],
      "guilds": {
        "*": {
          "requireMention": true
        }
      }
    }
  }
}
EOF

# 4. 合并配置
openclaw config merge /tmp/production-channels.json

# 5. 重启 Gateway
openclaw gateway restart

# 6. 等待 Gateway 启动
sleep 5

# 7. 验证所有 Channel
openclaw channels list

# 8. 健康检查
openclaw health

echo "Channels deployed successfully!"
```

---

## 总结

Channel 集成实战的核心要点：

1. **WhatsApp**：QR 扫码连接，会话持久化
2. **Telegram**：Bot Token 配置，命令支持
3. **Discord**：Bot 权限配置，服务器集成
4. **访问控制**：Pairing 策略 + Mention 要求
5. **多 Channel 管理**：统一配置，独立路由

完成这些步骤后，你就可以将 OpenClaw 连接到所有主流聊天平台了！🎉
