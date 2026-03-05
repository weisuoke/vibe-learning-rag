# 03_核心概念_04_channels_status命令

## 命令概述

`openclaw channels` 是 OpenClaw 的通道管理命令，用于管理和监控多个消息通道（WhatsApp、Telegram、Slack、Discord、Signal、iMessage 等）。通道是 OpenClaw 的核心抽象，统一了不同平台的消息接口。

---

## 命令语法

```bash
openclaw channels <subcommand> [options]
```

---

## 子命令

### 1. `channels list` - 列出所有通道

**用途**：显示所有已配置的通道

**语法**：
```bash
openclaw channels list [--json]
```

**输出示例**：
```
Configured Channels:

1. WhatsApp
   Status: Connected
   Phone: +1234567890
   Last Activity: 2 minutes ago

2. Telegram
   Status: Connected
   Username: @mybot
   Last Activity: 5 minutes ago

3. Slack
   Status: Disconnected
   Workspace: my-workspace
   Last Activity: 1 hour ago
```

**JSON 输出**：
```bash
openclaw channels list --json
```

```json
{
  "channels": [
    {
      "id": "whatsapp-1",
      "type": "whatsapp",
      "status": "connected",
      "phone": "+1234567890",
      "lastActivity": "2024-02-22T06:00:00Z"
    },
    {
      "id": "telegram-1",
      "type": "telegram",
      "status": "connected",
      "username": "@mybot",
      "lastActivity": "2024-02-22T05:55:00Z"
    }
  ]
}
```

---

### 2. `channels status` - 检查通道状态

**用途**：检查所有通道的连接状态

**语法**：
```bash
openclaw channels status [--json]
```

**输出示例**：
```
Channel Status:

✅ WhatsApp: Connected (2/2 sessions)
✅ Telegram: Connected
❌ Slack: Disconnected (auth expired)
⚠️  Discord: Degraded (rate limited)

Overall: 2/4 channels healthy
```

**状态说明**：
- `Connected`: 通道正常工作
- `Disconnected`: 通道未连接
- `Degraded`: 通道部分功能受限
- `Pairing`: 等待配对确认

---

### 3. `channels logs` - 查看通道日志

**用途**：查看特定通道的日志

**语法**：
```bash
openclaw channels logs <channel-id> [--tail <n>] [--follow]
```

**选项**：
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--tail <n>` | 显示最后 N 行 | 100 |
| `--follow` | 实时跟踪日志 | false |

**示例**：
```bash
# 查看 WhatsApp 日志
openclaw channels logs whatsapp-1

# 实时跟踪日志
openclaw channels logs whatsapp-1 --follow

# 查看最后 50 行
openclaw channels logs telegram-1 --tail 50
```

---

### 4. `channels add` - 添加新通道

**用途**：添加并配置新的消息通道

**语法**：
```bash
openclaw channels add <type> [options]
```

**支持的通道类型**：
- `whatsapp`: WhatsApp
- `telegram`: Telegram Bot
- `slack`: Slack
- `discord`: Discord
- `signal`: Signal
- `imessage`: iMessage (macOS only)
- `google-chat`: Google Chat
- `matrix`: Matrix

**示例**：
```bash
# 添加 WhatsApp
openclaw channels add whatsapp

# 添加 Telegram Bot
openclaw channels add telegram --token <bot-token>

# 添加 Slack
openclaw channels add slack --token <slack-token>
```

---

### 5. `channels remove` - 移除通道

**用途**：移除已配置的通道

**语法**：
```bash
openclaw channels remove <channel-id>
```

**示例**：
```bash
# 移除 WhatsApp 通道
openclaw channels remove whatsapp-1

# 移除 Telegram 通道
openclaw channels remove telegram-1
```

---

### 6. `channels logout` - 登出通道

**用途**：从通道登出（保留配置）

**语法**：
```bash
openclaw channels logout <channel-id>
```

**示例**：
```bash
# 从 WhatsApp 登出
openclaw channels logout whatsapp-1

# 从 Telegram 登出
openclaw channels logout telegram-1
```

**与 remove 的区别**：
- `logout`: 登出但保留配置，可以重新登录
- `remove`: 完全删除通道配置

---

## 通道架构

### 通道抽象层

```
┌─────────────────────────────────────────┐
│           OpenClaw Gateway              │
├─────────────────────────────────────────┤
│  ┌───────────────────────────────────┐  │
│  │      Channel Abstraction          │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │   Unified Message Interface │  │  │
│  │  │   - send()                  │  │  │
│  │  │   - receive()               │  │  │
│  │  │   - getStatus()             │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      Channel Implementations      │  │
│  │  - WhatsApp (Baileys)             │  │
│  │  - Telegram (Grammy)              │  │
│  │  - Slack (Bolt)                   │  │
│  │  - Discord (discord.js)           │  │
│  │  - Signal                         │  │
│  │  - iMessage (macOS)               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 消息流程

```typescript
// 消息接收流程
1. 通道接收消息 → 2. 标准化格式 → 3. 路由到 Agent → 4. 生成响应 → 5. 发送到通道

// 消息发送流程
1. Agent 生成响应 → 2. 选择通道 → 3. 格式化消息 → 4. 通道发送 → 5. 确认送达
```

---

## 使用场景

### 场景 1: 初始化通道

```bash
# 1. 添加 WhatsApp
openclaw channels add whatsapp

# 2. 扫码配对（会显示二维码）
# 扫码后等待连接

# 3. 检查状态
openclaw channels status

# 4. 测试发送
openclaw message send --to +1234567890 --message "Hello from OpenClaw"
```

---

### 场景 2: 多通道管理

```bash
# 添加多个通道
openclaw channels add whatsapp
openclaw channels add telegram --token <bot-token>
openclaw channels add slack --token <slack-token>

# 查看所有通道
openclaw channels list

# 检查状态
openclaw channels status

# 查看特定通道日志
openclaw channels logs whatsapp-1 --follow
```

---

### 场景 3: 通道故障排查

```bash
# 1. 检查通道状态
openclaw channels status

# 2. 查看日志
openclaw channels logs whatsapp-1 --tail 100

# 3. 重新登录
openclaw channels logout whatsapp-1
openclaw channels add whatsapp

# 4. 验证连接
openclaw message send --to +1234567890 --message "Test"
```

---

### 场景 4: 通道切换

```bash
# 移除旧通道
openclaw channels remove whatsapp-old

# 添加新通道
openclaw channels add whatsapp

# 验证新通道
openclaw channels status
```

---

## TypeScript 集成

### 示例 1: 通道管理器

```typescript
// src/channel-manager.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface Channel {
  id: string;
  type: string;
  status: 'connected' | 'disconnected' | 'degraded' | 'pairing';
  lastActivity?: string;
  metadata?: Record<string, any>;
}

export class ChannelManager {
  /**
   * 获取所有通道
   */
  async list(): Promise<Channel[]> {
    const { stdout } = await execAsync('openclaw channels list --json');
    const result = JSON.parse(stdout);
    return result.channels;
  }

  /**
   * 获取通道状态
   */
  async getStatus(channelId?: string): Promise<Record<string, string>> {
    const command = channelId
      ? `openclaw channels status ${channelId} --json`
      : 'openclaw channels status --json';

    const { stdout } = await execAsync(command);
    return JSON.parse(stdout);
  }

  /**
   * 添加通道
   */
  async add(type: string, options?: Record<string, string>): Promise<void> {
    let command = `openclaw channels add ${type}`;

    if (options) {
      for (const [key, value] of Object.entries(options)) {
        command += ` --${key} ${value}`;
      }
    }

    await execAsync(command);
    console.log(`✅ Channel ${type} added successfully`);
  }

  /**
   * 移除通道
   */
  async remove(channelId: string): Promise<void> {
    await execAsync(`openclaw channels remove ${channelId}`);
    console.log(`✅ Channel ${channelId} removed`);
  }

  /**
   * 登出通道
   */
  async logout(channelId: string): Promise<void> {
    await execAsync(`openclaw channels logout ${channelId}`);
    console.log(`✅ Logged out from ${channelId}`);
  }

  /**
   * 获取通道日志
   */
  async getLogs(channelId: string, tail = 100): Promise<string> {
    const { stdout } = await execAsync(
      `openclaw channels logs ${channelId} --tail ${tail}`
    );
    return stdout;
  }

  /**
   * 检查通道是否健康
   */
  async isHealthy(channelId: string): Promise<boolean> {
    const status = await this.getStatus(channelId);
    return status[channelId] === 'connected';
  }

  /**
   * 获取已连接的通道
   */
  async getConnectedChannels(): Promise<Channel[]> {
    const channels = await this.list();
    return channels.filter(c => c.status === 'connected');
  }

  /**
   * 获取断开的通道
   */
  async getDisconnectedChannels(): Promise<Channel[]> {
    const channels = await this.list();
    return channels.filter(c => c.status === 'disconnected');
  }
}
```

### 示例 2: 通道监控

```typescript
// src/channel-monitor.ts
import { EventEmitter } from 'events';
import { ChannelManager } from './channel-manager';

export class ChannelMonitor extends EventEmitter {
  private manager = new ChannelManager();
  private interval: NodeJS.Timeout | null = null;
  private previousStatus: Record<string, string> = {};

  /**
   * 开始监控
   */
  start(checkInterval = 30000): void {
    this.interval = setInterval(async () => {
      await this.checkChannels();
    }, checkInterval);

    console.log('📡 Channel monitoring started');
  }

  /**
   * 停止监控
   */
  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      console.log('📡 Channel monitoring stopped');
    }
  }

  /**
   * 检查所有通道
   */
  private async checkChannels(): Promise<void> {
    try {
      const channels = await this.manager.list();

      for (const channel of channels) {
        const previousStatus = this.previousStatus[channel.id];
        const currentStatus = channel.status;

        // 状态变化检测
        if (previousStatus && previousStatus !== currentStatus) {
          this.emit('status-changed', {
            channelId: channel.id,
            from: previousStatus,
            to: currentStatus,
          });

          if (currentStatus === 'disconnected') {
            this.emit('channel-disconnected', channel);
          } else if (currentStatus === 'connected' && previousStatus === 'disconnected') {
            this.emit('channel-reconnected', channel);
          }
        }

        this.previousStatus[channel.id] = currentStatus;
      }
    } catch (error) {
      this.emit('error', error);
    }
  }

  /**
   * 获取监控统计
   */
  async getStats() {
    const channels = await this.manager.list();
    const connected = channels.filter(c => c.status === 'connected').length;
    const disconnected = channels.filter(c => c.status === 'disconnected').length;
    const degraded = channels.filter(c => c.status === 'degraded').length;

    return {
      total: channels.length,
      connected,
      disconnected,
      degraded,
      healthRate: channels.length > 0 ? (connected / channels.length) * 100 : 0,
    };
  }
}

// 使用示例
const monitor = new ChannelMonitor();

monitor.on('status-changed', ({ channelId, from, to }) => {
  console.log(`📊 Channel ${channelId} status: ${from} → ${to}`);
});

monitor.on('channel-disconnected', (channel) => {
  console.log(`❌ Channel ${channel.id} disconnected`);
  // 发送告警通知
});

monitor.on('channel-reconnected', (channel) => {
  console.log(`✅ Channel ${channel.id} reconnected`);
});

monitor.start();
```

### 示例 3: 通道路由器

```typescript
// src/channel-router.ts
import { ChannelManager } from './channel-manager';

export interface Message {
  to: string;
  content: string;
  channel?: string;
}

export class ChannelRouter {
  private manager = new ChannelManager();

  /**
   * 智能路由消息到最佳通道
   */
  async route(message: Message): Promise<string> {
    // 如果指定了通道，直接使用
    if (message.channel) {
      return message.channel;
    }

    // 根据收件人格式判断通道类型
    const channelType = this.detectChannelType(message.to);

    // 获取该类型的可用通道
    const channels = await this.manager.getConnectedChannels();
    const availableChannels = channels.filter(c => c.type === channelType);

    if (availableChannels.length === 0) {
      throw new Error(`No available ${channelType} channels`);
    }

    // 选择最近活跃的通道
    const bestChannel = availableChannels.sort((a, b) => {
      const aTime = new Date(a.lastActivity || 0).getTime();
      const bTime = new Date(b.lastActivity || 0).getTime();
      return bTime - aTime;
    })[0];

    return bestChannel.id;
  }

  /**
   * 检测通道类型
   */
  private detectChannelType(recipient: string): string {
    // 电话号码 → WhatsApp
    if (/^\+?\d{10,15}$/.test(recipient)) {
      return 'whatsapp';
    }

    // @username → Telegram
    if (recipient.startsWith('@')) {
      return 'telegram';
    }

    // #channel → Slack
    if (recipient.startsWith('#')) {
      return 'slack';
    }

    // 默认 WhatsApp
    return 'whatsapp';
  }

  /**
   * 发送消息（自动路由）
   */
  async send(message: Message): Promise<void> {
    const channelId = await this.route(message);
    console.log(`📤 Routing message to ${channelId}`);

    // 这里调用实际的发送逻辑
    // await this.manager.sendMessage(channelId, message);
  }
}
```

### 示例 4: 通道健康检查

```typescript
// src/channel-health.ts
import { ChannelManager } from './channel-manager';

export interface HealthReport {
  channelId: string;
  status: string;
  issues: string[];
  recommendations: string[];
}

export class ChannelHealth {
  private manager = new ChannelManager();

  /**
   * 执行健康检查
   */
  async check(): Promise<HealthReport[]> {
    const channels = await this.manager.list();
    const reports: HealthReport[] = [];

    for (const channel of channels) {
      const report = await this.checkChannel(channel.id);
      reports.push(report);
    }

    return reports;
  }

  /**
   * 检查单个通道
   */
  private async checkChannel(channelId: string): Promise<HealthReport> {
    const status = await this.manager.getStatus(channelId);
    const issues: string[] = [];
    const recommendations: string[] = [];

    // 检查连接状态
    if (status[channelId] === 'disconnected') {
      issues.push('Channel is disconnected');
      recommendations.push('Run: openclaw channels logout && openclaw channels add');
    }

    // 检查最后活动时间
    const channels = await this.manager.list();
    const channel = channels.find(c => c.id === channelId);

    if (channel?.lastActivity) {
      const lastActivity = new Date(channel.lastActivity);
      const now = new Date();
      const hoursSinceActivity = (now.getTime() - lastActivity.getTime()) / (1000 * 60 * 60);

      if (hoursSinceActivity > 24) {
        issues.push('No activity in 24+ hours');
        recommendations.push('Check channel logs for errors');
      }
    }

    return {
      channelId,
      status: status[channelId],
      issues,
      recommendations,
    };
  }

  /**
   * 生成健康报告
   */
  async generateReport(): Promise<string> {
    const reports = await this.check();
    let output = '# Channel Health Report\n\n';

    for (const report of reports) {
      output += `## ${report.channelId}\n`;
      output += `Status: ${report.status}\n\n`;

      if (report.issues.length > 0) {
        output += '### Issues:\n';
        report.issues.forEach(issue => {
          output += `- ${issue}\n`;
        });
        output += '\n';
      }

      if (report.recommendations.length > 0) {
        output += '### Recommendations:\n';
        report.recommendations.forEach(rec => {
          output += `- ${rec}\n`;
        });
        output += '\n';
      }
    }

    return output;
  }
}
```

---

## 常见问题

### Q1: WhatsApp 配对失败

**问题**：
```
Error: Failed to pair WhatsApp channel
```

**解决方案**：
```bash
# 1. 清除旧会话
openclaw channels remove whatsapp-1

# 2. 重新添加
openclaw channels add whatsapp

# 3. 确保手机在线并扫码
# 4. 检查日志
openclaw channels logs whatsapp-1
```

---

### Q2: Telegram Bot 无响应

**问题**：
```
Telegram channel connected but not responding
```

**解决方案**：
```bash
# 1. 检查 Bot Token
openclaw config get channels.telegram.token

# 2. 验证 Bot 权限
# 确保 Bot 有发送消息权限

# 3. 重启通道
openclaw channels logout telegram-1
openclaw channels add telegram --token <bot-token>

# 4. 测试发送
openclaw message send --to @username --message "Test"
```

---

### Q3: 通道频繁断开

**问题**：
```
Channel keeps disconnecting
```

**解决方案**：
```bash
# 1. 检查网络连接
ping 8.8.8.8

# 2. 查看日志
openclaw channels logs <channel-id> --tail 200

# 3. 检查 Gateway 状态
openclaw gateway health

# 4. 增加重连间隔
openclaw config set channels.reconnectInterval 30000
```

---

### Q4: 如何批量管理通道？

**解决方案**：
```typescript
// 批量添加通道
const channels = [
  { type: 'whatsapp' },
  { type: 'telegram', token: 'bot-token' },
  { type: 'slack', token: 'slack-token' },
];

const manager = new ChannelManager();

for (const channel of channels) {
  await manager.add(channel.type, channel);
}
```

---

## 最佳实践

### 1. 通道命名规范

```bash
# ✅ 推荐：使用描述性名称
whatsapp-personal
telegram-work-bot
slack-team-notifications

# ❌ 不推荐：使用默认名称
whatsapp-1
telegram-1
```

### 2. 定期健康检查

```bash
# 添加到 crontab
*/30 * * * * openclaw channels status || openclaw gateway restart
```

### 3. 日志轮转

```bash
# 定期清理日志
openclaw channels logs whatsapp-1 --tail 1000 > whatsapp-backup.log
```

### 4. 监控通道状态

```typescript
// 使用 ChannelMonitor 持续监控
const monitor = new ChannelMonitor();
monitor.start(60000); // 每分钟检查一次
```

---

## 性能优化

### 1. 连接池管理

```typescript
// 限制并发连接数
const maxConnections = 5;
const activeChannels = await manager.getConnectedChannels();

if (activeChannels.length >= maxConnections) {
  console.warn('Max connections reached');
}
```

### 2. 消息队列

```typescript
// 使用队列避免消息丢失
const messageQueue = [];

async function sendWithRetry(message: Message, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await router.send(message);
      return;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

### 3. 缓存通道状态

```typescript
// 缓存状态减少 CLI 调用
const statusCache = new Map<string, { status: string; timestamp: number }>();
const CACHE_TTL = 30000; // 30 秒

async function getCachedStatus(channelId: string): Promise<string> {
  const cached = statusCache.get(channelId);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.status;
  }

  const status = await manager.getStatus(channelId);
  statusCache.set(channelId, { status: status[channelId], timestamp: Date.now() });
  return status[channelId];
}
```

---

## 下一步

- 学习 **config 命令** → `03_核心概念_05_config命令.md`
- 学习 **实战场景** → `07_实战代码_场景1_基础命令使用.md`
- 学习 **通道集成** → Phase 3: 多通道消息系统

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [通道文档](https://docs.openclaw.ai/channels)
- [CLI 官方文档](https://docs.openclaw.ai/cli)
- [通道抽象层设计](https://docs.openclaw.ai/architecture/channels)
