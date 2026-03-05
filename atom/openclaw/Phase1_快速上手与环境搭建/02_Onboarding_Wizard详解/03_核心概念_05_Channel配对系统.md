# 核心概念 5：Channel 配对系统

**本文档深入讲解 Channel 的配对系统，包括认证流程、配对机制和安全策略。**

---

## 概述

Channel 是 OpenClaw 与外部聊天平台（WhatsApp、Telegram、Discord 等）的连接桥梁。Channel 配对系统负责：

1. **身份认证**：验证 Channel 账号的合法性
2. **连接建立**：建立与聊天平台的持久连接
3. **消息路由**：将消息路由到对应的 Agent
4. **访问控制**：控制谁可以与 Agent 对话

---

## Channel 配对机制对比

### 主流 Channel 配对方式

| Channel | 配对方式 | 认证凭证 | 配对难度 | 安全性 |
|---------|---------|---------|---------|-------|
| **WhatsApp** | QR 扫码 | 会话密钥 | 中 | 高 |
| **Telegram** | Bot Token | API Token | 低 | 中 |
| **Discord** | OAuth | OAuth Token | 中 | 高 |
| **Signal** | 手机号 + 验证码 | 注册密钥 | 高 | 极高 |
| **BlueBubbles** | REST API | API Key | 低 | 中 |
| **iMessage** | macOS 集成 | 系统凭证 | 高 | 高 |

---

## WhatsApp 配对系统

### 配对流程

```
用户运行 openclaw onboard
    ↓
选择 WhatsApp Channel
    ↓
┌─────────────────────────────────────┐
│  1. 生成 QR 码                      │
│  - Gateway 启动 Baileys 客户端      │
│  - 生成配对 QR 码                   │
│  - 显示在终端或 WebUI               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 用户扫码                        │
│  - 打开 WhatsApp 手机 App           │
│  - 设置 → 已连接的设备 → 连接设备  │
│  - 扫描 QR 码                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. 建立连接                        │
│  - WhatsApp 服务器验证 QR 码        │
│  - 返回会话密钥                     │
│  - Baileys 保存会话到本地           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. 持久化会话                      │
│  - 保存到 ~/.openclaw/sessions/     │
│  - 包含加密的会话密钥               │
│  - 下次启动自动恢复                 │
└─────────────────────────────────────┘
```

### 配置示例

```json5
{
  channels: {
    whatsapp: {
      enabled: true,
      dmPolicy: "allowlist",  // pairing | allowlist | open | disabled
      allowFrom: ["+1234567890"],  // 白名单电话号码
      groups: {
        "*": {
          requireMention: true,  // 群组需要 @提及
        },
      },
      sessionStore: "~/.openclaw/sessions/whatsapp",  // 会话存储路径
    },
  },
}
```

### 实现细节

```typescript
// 伪代码示例
async function setupWhatsAppChannel(config: WhatsAppConfig) {
  // 1. 初始化 Baileys 客户端
  const { state, saveCreds } = await useMultiFileAuthState(config.sessionStore);

  const sock = makeWASocket({
    auth: state,
    printQRInTerminal: true,  // 在终端打印 QR 码
  });

  // 2. 监听连接事件
  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      // 显示 QR 码
      console.log("Scan this QR code with WhatsApp:");
      qrcode.generate(qr, { small: true });
    }

    if (connection === "open") {
      console.log("WhatsApp connected!");
    }

    if (connection === "close") {
      // 处理断开连接
      const shouldReconnect = lastDisconnect?.error?.output?.statusCode !== DisconnectReason.loggedOut;
      if (shouldReconnect) {
        setupWhatsAppChannel(config);  // 重新连接
      }
    }
  });

  // 3. 监听消息
  sock.ev.on("messages.upsert", async (m) => {
    const message = m.messages[0];
    if (!message.key.fromMe && m.type === "notify") {
      await handleIncomingMessage(message);
    }
  });

  // 4. 保存凭证
  sock.ev.on("creds.update", saveCreds);
}
```

### 会话持久化

```
~/.openclaw/sessions/whatsapp/
├── creds.json                    # 加密的会话凭证
├── app-state-sync-key-*.json     # 应用状态同步密钥
└── session-*.json                # 会话数据
```

**安全特性**：
- 会话密钥加密存储
- 仅用户可读写（`0o600`）
- 支持多设备同步

---

## Telegram 配对系统

### 配对流程

```
用户运行 openclaw onboard
    ↓
选择 Telegram Channel
    ↓
┌─────────────────────────────────────┐
│  1. 创建 Bot                        │
│  - 与 @BotFather 对话               │
│  - 发送 /newbot                     │
│  - 设置 Bot 名称和用户名            │
│  - 获取 Bot Token                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 配置 Bot Token                  │
│  - 在向导中输入 Bot Token           │
│  - 或设置环境变量                   │
│  - Gateway 验证 Token 有效性        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. 启动 Bot                        │
│  - Gateway 连接 Telegram API        │
│  - 使用 grammY 库                   │
│  - 开始接收消息                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. 配对用户                        │
│  - 用户发送 /start 给 Bot           │
│  - Bot 返回配对码（如果 dmPolicy=pairing）│
│  - 或直接开始对话（如果在 allowlist）│
└─────────────────────────────────────┘
```

### 配置示例

```json5
{
  channels: {
    telegram: {
      enabled: true,
      botToken: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",  // Bot Token
      dmPolicy: "pairing",  // pairing | allowlist | open | disabled
      allowFrom: ["tg:123456789"],  // 白名单用户 ID
      groups: {
        "*": {
          requireMention: true,  // 群组需要 @提及
        },
      },
    },
  },
}
```

### 实现细节

```typescript
// 伪代码示例
async function setupTelegramChannel(config: TelegramConfig) {
  // 1. 初始化 grammY Bot
  const bot = new Bot(config.botToken);

  // 2. 验证 Bot Token
  const me = await bot.api.getMe();
  console.log(`Telegram Bot connected: @${me.username}`);

  // 3. 处理 /start 命令
  bot.command("start", async (ctx) => {
    const userId = ctx.from?.id;
    const username = ctx.from?.username;

    if (config.dmPolicy === "pairing") {
      // 生成配对码
      const pairingCode = generatePairingCode();
      await ctx.reply(
        `Welcome! Your pairing code is: ${pairingCode}\n\n` +
        `Please share this code with the bot administrator to approve your access.`
      );

      // 等待管理员批准
      await waitForPairingApproval(userId, pairingCode);
    } else if (config.dmPolicy === "allowlist") {
      // 检查白名单
      if (config.allowFrom.includes(`tg:${userId}`)) {
        await ctx.reply("Welcome! You can start chatting now.");
      } else {
        await ctx.reply("Sorry, you are not authorized to use this bot.");
      }
    } else if (config.dmPolicy === "open") {
      await ctx.reply("Welcome! You can start chatting now.");
    }
  });

  // 4. 处理消息
  bot.on("message:text", async (ctx) => {
    const userId = ctx.from?.id;

    // 检查访问权限
    if (!await checkAccess(userId, config)) {
      return;
    }

    // 路由到 Agent
    await handleIncomingMessage({
      channel: "telegram",
      userId: `tg:${userId}`,
      text: ctx.message.text,
    });
  });

  // 5. 启动 Bot
  await bot.start();
}
```

### Bot Token 获取

```
1. 打开 Telegram，搜索 @BotFather
2. 发送 /newbot
3. 输入 Bot 名称（如 "My OpenClaw Bot"）
4. 输入 Bot 用户名（如 "my_openclaw_bot"，必须以 _bot 结尾）
5. 获取 Bot Token（如 "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"）
6. 保存 Token 到配置文件或环境变量
```

---

## Discord 配对系统

### 配对流程

```
用户运行 openclaw onboard
    ↓
选择 Discord Channel
    ↓
┌─────────────────────────────────────┐
│  1. 创建 Discord Application        │
│  - 访问 Discord Developer Portal    │
│  - 创建 New Application             │
│  - 添加 Bot 用户                    │
│  - 获取 Bot Token                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 配置权限和 Intents              │
│  - 启用 Message Content Intent      │
│  - 配置 Bot 权限（读取消息、发送消息）│
│  - 生成 OAuth2 URL                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. 邀请 Bot 到服务器               │
│  - 使用 OAuth2 URL 邀请             │
│  - 选择目标服务器                   │
│  - 授权权限                         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. 配置 Bot Token                  │
│  - 在向导中输入 Bot Token           │
│  - Gateway 连接 Discord Gateway     │
│  - 开始接收消息                     │
└─────────────────────────────────────┘
```

### 配置示例

```json5
{
  channels: {
    discord: {
      enabled: true,
      botToken: "MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GaBcDe.FgHiJkLmNoPqRsTuVwXyZ",  // Bot Token
      dmPolicy: "pairing",  // pairing | allowlist | open | disabled
      allowFrom: ["dc:123456789012345678"],  // 白名单用户 ID
      guilds: {
        "*": {
          requireMention: true,  // 服务器需要 @提及
        },
      },
    },
  },
}
```

### 实现细节

```typescript
// 伪代码示例
async function setupDiscordChannel(config: DiscordConfig) {
  // 1. 初始化 Discord Client
  const client = new Client({
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.MessageContent,
      GatewayIntentBits.DirectMessages,
    ],
  });

  // 2. 监听 ready 事件
  client.once("ready", () => {
    console.log(`Discord Bot connected: ${client.user?.tag}`);
  });

  // 3. 处理消息
  client.on("messageCreate", async (message) => {
    // 忽略 Bot 自己的消息
    if (message.author.bot) return;

    // 检查访问权限
    if (!await checkAccess(message.author.id, config)) {
      return;
    }

    // 检查是否需要 @提及
    if (message.guild && config.guilds["*"]?.requireMention) {
      if (!message.mentions.has(client.user!)) {
        return;  // 未 @提及，忽略
      }
    }

    // 路由到 Agent
    await handleIncomingMessage({
      channel: "discord",
      userId: `dc:${message.author.id}`,
      text: message.content,
      guildId: message.guild?.id,
      channelId: message.channel.id,
    });
  });

  // 4. 登录
  await client.login(config.botToken);
}
```

### OAuth2 URL 生成

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=2048&scope=bot

参数说明：
- client_id: Application ID
- permissions: 权限位掩码（2048 = 发送消息）
- scope: bot（Bot 权限）
```

---

## DM Policy（访问控制策略）

### 策略类型

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| **pairing** | 未知用户需要配对码 | 个人使用（推荐） |
| **allowlist** | 仅白名单用户可访问 | 团队使用 |
| **open** | 所有用户可访问 | 公开服务（需谨慎） |
| **disabled** | 禁用 DM | 仅群组使用 |

### Pairing 策略

**流程**：

```
未知用户发送消息
    ↓
Bot 返回配对码（6 位数字）
    ↓
用户将配对码发送给管理员
    ↓
管理员批准（openclaw pairing approve <code>）
    ↓
用户可以开始对话
```

**配置示例**：

```json5
{
  channels: {
    telegram: {
      dmPolicy: "pairing",
      pairingCodeExpiry: "15m",  // 配对码有效期
    },
  },
}
```

**实现逻辑**：

```typescript
async function handlePairingRequest(userId: string, channel: string) {
  // 1. 生成配对码
  const pairingCode = generatePairingCode();  // 6 位数字

  // 2. 保存到数据库
  await savePairingRequest({
    userId,
    channel,
    code: pairingCode,
    expiresAt: Date.now() + 15 * 60 * 1000,  // 15 分钟
  });

  // 3. 返回配对码
  return pairingCode;
}

async function approvePairing(code: string) {
  // 1. 查找配对请求
  const request = await findPairingRequest(code);

  if (!request) {
    throw new Error("Invalid pairing code");
  }

  if (request.expiresAt < Date.now()) {
    throw new Error("Pairing code expired");
  }

  // 2. 添加到白名单
  await addToAllowlist(request.userId, request.channel);

  // 3. 删除配对请求
  await deletePairingRequest(code);

  // 4. 通知用户
  await sendMessage(request.userId, request.channel, "Pairing approved! You can start chatting now.");
}
```

### Allowlist 策略

**配置示例**：

```json5
{
  channels: {
    whatsapp: {
      dmPolicy: "allowlist",
      allowFrom: [
        "+1234567890",  // 电话号码
        "+9876543210",
      ],
    },
    telegram: {
      dmPolicy: "allowlist",
      allowFrom: [
        "tg:123456789",  // 用户 ID
        "tg:987654321",
      ],
    },
    discord: {
      dmPolicy: "allowlist",
      allowFrom: [
        "dc:123456789012345678",  // 用户 ID
        "dc:987654321098765432",
      ],
    },
  },
}
```

**动态管理**：

```bash
# 添加到白名单
openclaw channels allowlist add whatsapp +1234567890

# 从白名单移除
openclaw channels allowlist remove whatsapp +1234567890

# 查看白名单
openclaw channels allowlist list whatsapp
```

### Open 策略

**配置示例**：

```json5
{
  channels: {
    telegram: {
      dmPolicy: "open",
      allowFrom: ["*"],  // 必须显式设置为 ["*"]
    },
  },
}
```

**安全警告**：

```
⚠️  Warning: Open DM policy allows anyone to message your bot.
    This can lead to:
    - Spam and abuse
    - Resource exhaustion
    - Security risks

    Recommended mitigations:
    - Rate limiting
    - Content filtering
    - Monitoring and alerting
```

---

## 群组/服务器配置

### WhatsApp 群组

```json5
{
  channels: {
    whatsapp: {
      groups: {
        "*": {
          requireMention: true,  // 所有群组需要 @提及
        },
        "120363123456789012@g.us": {
          requireMention: false,  // 特定群组不需要 @提及
          allowFrom: ["*"],  // 所有成员可访问
        },
      },
    },
  },
}
```

### Telegram 群组

```json5
{
  channels: {
    telegram: {
      groups: {
        "*": {
          requireMention: true,  // 所有群组需要 @提及
        },
        "-1001234567890": {
          requireMention: false,  // 特定群组不需要 @提及
          allowFrom: ["tg:123456789"],  // 仅特定用户可访问
        },
      },
    },
  },
}
```

### Discord 服务器

```json5
{
  channels: {
    discord: {
      guilds: {
        "*": {
          requireMention: true,  // 所有服务器需要 @提及
        },
        "123456789012345678": {
          requireMention: false,  // 特定服务器不需要 @提及
          channels: {
            "987654321098765432": {
              enabled: true,  // 仅特定频道启用
            },
          },
        },
      },
    },
  },
}
```

---

## 会话管理

### 会话 Scope

```json5
{
  session: {
    dmScope: "per-channel-peer",  // main | per-peer | per-channel-peer | per-account-channel-peer
  },
}
```

| Scope | 行为 | 适用场景 |
|-------|------|---------|
| **main** | 所有对话共享一个会话 | 单用户使用 |
| **per-peer** | 每个用户一个会话 | 多用户，跨渠道共享上下文 |
| **per-channel-peer** | 每个渠道的每个用户一个会话 | 多用户，渠道隔离（推荐） |
| **per-account-channel-peer** | 每个账号的每个渠道的每个用户一个会话 | 多账号，完全隔离 |

### 会话重置

```json5
{
  session: {
    reset: {
      mode: "daily",  // manual | daily | idle
      atHour: 4,  // 每日重置时间（凌晨 4 点）
      idleMinutes: 120,  // 空闲 2 小时后重置
    },
  },
}
```

---

## 安全最佳实践

### 1. 使用 Pairing 策略

```json5
{
  channels: {
    whatsapp: {
      dmPolicy: "pairing",  // 推荐
    },
    telegram: {
      dmPolicy: "pairing",  // 推荐
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
          requireMention: true,  // 防止群组消息轰炸
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

### 4. 监控异常访问

```bash
# 查看访问日志
openclaw logs gateway | grep "Unauthorized access"

# 设置告警
openclaw alerts add --condition "unauthorized_access > 10" \
  --action "notify:admin@example.com"
```

---

## 故障排查

### 问题 1：WhatsApp QR 码不显示

```bash
# 检查 Gateway 日志
openclaw logs gateway | grep "whatsapp"

# 手动启动 WhatsApp Channel
openclaw channels start whatsapp --debug

# 检查会话文件
ls -la ~/.openclaw/sessions/whatsapp/
```

### 问题 2：Telegram Bot 无响应

```bash
# 验证 Bot Token
curl https://api.telegram.org/bot<TOKEN>/getMe

# 检查 Bot 权限
# 确保 Bot 有 "Read Messages" 权限

# 重启 Channel
openclaw channels restart telegram
```

### 问题 3：Discord Bot 离线

```bash
# 检查 Bot Token
openclaw config get channels.discord.botToken

# 检查 Intents
# 确保启用了 Message Content Intent

# 查看连接状态
openclaw channels status discord
```

### 问题 4：配对码过期

```bash
# 查看配对请求
openclaw pairing list

# 延长配对码有效期
openclaw config set channels.telegram.pairingCodeExpiry "30m"

# 重新生成配对码
openclaw pairing regenerate <user-id>
```

---

## 总结

Channel 配对系统的核心要点：

1. **多样化配对方式**：QR 扫码（WhatsApp）、Bot Token（Telegram）、OAuth（Discord）
2. **灵活的访问控制**：Pairing、Allowlist、Open、Disabled 四种策略
3. **会话隔离**：支持多种会话 Scope，适应不同使用场景
4. **安全默认值**：Pairing 策略 + Mention 要求 + 会话隔离
5. **动态管理**：支持运行时添加/移除白名单，无需重启

理解 Channel 配对系统，可以帮助你安全、高效地连接各种聊天平台。
