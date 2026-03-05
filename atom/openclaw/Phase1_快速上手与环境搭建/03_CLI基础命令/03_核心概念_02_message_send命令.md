# 03_核心概念_02_message_send命令

## 命令概述

`openclaw message send` 用于通过 OpenClaw Gateway 向多个通道发送消息。这是 OpenClaw 最常用的命令之一，支持跨平台消息路由和自动通道选择。

---

## 命令语法

```bash
openclaw message send --to <target> --message <text> [options]
```

---

## 必需参数

### `--to <target>` - 目标地址

指定消息接收者，支持多种格式：

| 格式 | 说明 | 示例 |
|------|------|------|
| 电话号码 | WhatsApp、Telegram | `+1234567890` |
| 用户名 | Telegram、Discord | `@username` |
| 频道 ID | Slack、Discord | `#channel-name` |
| 特殊目标 | 所有通道 | `@all` |
| 邮箱 | Email 通道 | `user@example.com` |

**示例**：
```bash
# 发送到电话号码
openclaw message send --to +1234567890 --message "Hello"

# 发送到用户名
openclaw message send --to @john --message "Hi John"

# 发送到所有通道
openclaw message send --to @all --message "系统通知"
```

---

### `--message <text>` - 消息内容

指定要发送的消息文本。

**支持的格式**：
- 纯文本
- Markdown（部分通道支持）
- 表情符号

**示例**：
```bash
# 纯文本
openclaw message send --to @user --message "Hello World"

# 多行文本（使用引号）
openclaw message send --to @user --message "Line 1
Line 2
Line 3"

# 包含表情符号
openclaw message send --to @user --message "Hello 👋"
```

---

## 可选参数

### `--channel <name>` - 指定通道

强制使用特定通道发送消息。

**示例**：
```bash
# 只通过 Telegram 发送
openclaw message send --to @user --message "Hello" --channel telegram

# 只通过 Slack 发送
openclaw message send --to @user --message "Hello" --channel slack
```

---

### `--deliver <boolean>` - 控制发送

控制是否实际发送消息（默认 `true`）。

**示例**：
```bash
# 测试模式（不实际发送）
openclaw message send --to @user --message "Test" --deliver=false
```

---

## 使用场景

### 场景 1: 单通道消息发送

```bash
# 发送到 WhatsApp
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# 发送到 Telegram
openclaw message send --to @username --message "Hi there"

# 发送到 Slack
openclaw message send --to #general --message "Team update"
```

---

### 场景 2: 多通道广播

```bash
# 发送到所有配置的通道
openclaw message send --to @all --message "系统维护通知：今晚 10 点开始"

# Gateway 自动路由到：
# - WhatsApp
# - Telegram
# - Slack
# - Discord
# - 其他配置的通道
```

---

### 场景 3: 批量消息发送

```bash
#!/bin/bash
# 批量发送脚本

recipients=("+1234567890" "+0987654321" "+1122334455")

for recipient in "${recipients[@]}"; do
  openclaw message send --to "$recipient" --message "批量通知消息"
  sleep 1  # 避免速率限制
done
```

---

### 场景 4: 从文件读取消息

```bash
# 从文件读取消息内容
message=$(cat message.txt)
openclaw message send --to @all --message "$message"

# 或使用管道
cat message.txt | xargs -I {} openclaw message send --to @all --message "{}"
```

---

## TypeScript 集成

### 示例 1: 基础消息发送

```typescript
// src/message-sender.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class MessageSender {
  async send(to: string, message: string): Promise<void> {
    const command = `openclaw message send --to "${to}" --message "${message}"`;

    try {
      const { stdout, stderr } = await execAsync(command);

      if (stderr) {
        throw new Error(`发送失败: ${stderr}`);
      }

      console.log('消息已发送:', stdout);
    } catch (error) {
      console.error('发送错误:', error);
      throw error;
    }
  }

  async broadcast(message: string): Promise<void> {
    await this.send('@all', message);
  }
}

// 使用示例
const sender = new MessageSender();
await sender.send('+1234567890', 'Hello');
await sender.broadcast('系统通知');
```

---

### 示例 2: 批量发送（带重试）

```typescript
// src/batch-sender.ts
import pLimit from 'p-limit';

export class BatchSender {
  private sender = new MessageSender();
  private limit = pLimit(5); // 最多 5 个并发
  private maxRetries = 3;

  async sendBatch(
    recipients: string[],
    message: string
  ): Promise<BatchResult> {
    const results: BatchResult = {
      success: [],
      failed: [],
    };

    const tasks = recipients.map(to =>
      this.limit(async () => {
        try {
          await this.sendWithRetry(to, message);
          results.success.push(to);
        } catch (error) {
          results.failed.push({ to, error: error.message });
        }
      })
    );

    await Promise.all(tasks);
    return results;
  }

  private async sendWithRetry(
    to: string,
    message: string,
    retries = 0
  ): Promise<void> {
    try {
      await this.sender.send(to, message);
    } catch (error) {
      if (retries < this.maxRetries) {
        console.log(`重试 ${retries + 1}/${this.maxRetries}...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (retries + 1)));
        return this.sendWithRetry(to, message, retries + 1);
      }
      throw error;
    }
  }
}

interface BatchResult {
  success: string[];
  failed: Array<{ to: string; error: string }>;
}
```

---

### 示例 3: 消息队列

```typescript
// src/message-queue.ts
export class MessageQueue {
  private queue: QueueItem[] = [];
  private processing = false;
  private sender = new MessageSender();

  enqueue(to: string, message: string, priority = 0): void {
    this.queue.push({ to, message, priority, timestamp: Date.now() });
    this.queue.sort((a, b) => b.priority - a.priority);

    if (!this.processing) {
      this.process();
    }
  }

  private async process(): Promise<void> {
    this.processing = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift()!;

      try {
        await this.sender.send(item.to, item.message);
        console.log(`✅ 已发送: ${item.to}`);
      } catch (error) {
        console.error(`❌ 发送失败: ${item.to}`, error);
        // 可选：重新入队
        if (item.retries < 3) {
          this.queue.push({ ...item, retries: (item.retries || 0) + 1 });
        }
      }

      // 速率限制
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    this.processing = false;
  }
}

interface QueueItem {
  to: string;
  message: string;
  priority: number;
  timestamp: number;
  retries?: number;
}
```

---

## 消息路由机制

### 自动路由规则

Gateway 根据目标地址自动选择通道：

```typescript
// Gateway 内部路由逻辑（简化）
class MessageRouter {
  route(to: string): Channel[] {
    if (to === '@all') {
      return this.getAllChannels();
    }

    if (to.startsWith('+')) {
      // 电话号码 → WhatsApp, Telegram
      return this.getChannelsByType(['whatsapp', 'telegram']);
    }

    if (to.startsWith('@')) {
      // 用户名 → Telegram, Discord
      return this.getChannelsByType(['telegram', 'discord']);
    }

    if (to.startsWith('#')) {
      // 频道 → Slack, Discord
      return this.getChannelsByType(['slack', 'discord']);
    }

    if (to.includes('@') && to.includes('.')) {
      // 邮箱 → Email
      return this.getChannelsByType(['email']);
    }

    // 默认：所有通道
    return this.getAllChannels();
  }
}
```

---

## 常见问题

### Q1: 消息发送失败，如何排查？

**排查步骤**：
```bash
# 1. 检查 Gateway 状态
openclaw gateway status

# 2. 检查通道状态
openclaw channels status

# 3. 查看日志
tail -f ~/.openclaw/logs/gateway.log

# 4. 测试模式发送
openclaw message send --to @test --message "Test" --deliver=false
```

---

### Q2: 如何避免速率限制？

**解决方案**：
```typescript
// 添加延迟
for (const recipient of recipients) {
  await sender.send(recipient, message);
  await new Promise(resolve => setTimeout(resolve, 1000)); // 1 秒延迟
}

// 或使用并发控制
const limit = pLimit(5); // 最多 5 个并发
```

---

### Q3: 如何发送多行消息？

**方法 1: 使用引号**
```bash
openclaw message send --to @user --message "Line 1
Line 2
Line 3"
```

**方法 2: 使用 heredoc**
```bash
openclaw message send --to @user --message "$(cat <<'EOF'
Line 1
Line 2
Line 3
EOF
)"
```

---

## 最佳实践

### 1. 使用环境变量存储常用目标

```bash
# .env
DEFAULT_RECIPIENT="+1234567890"
TEAM_CHANNEL="@all"

# 使用
openclaw message send --to "$DEFAULT_RECIPIENT" --message "Hello"
```

### 2. 记录发送日志

```typescript
async send(to: string, message: string): Promise<void> {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] 发送到 ${to}: ${message}`);

  await execAsync(`openclaw message send --to "${to}" --message "${message}"`);

  // 记录到文件
  fs.appendFileSync('message-log.txt', `${timestamp} | ${to} | ${message}\n`);
}
```

### 3. 错误处理和重试

```typescript
async sendWithRetry(to: string, message: string, maxRetries = 3): Promise<void> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await this.send(to, message);
      return;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

---

## 下一步

- 学习 **agent 命令** → `03_核心概念_03_agent命令.md`
- 学习 **channels 命令** → `03_核心概念_04_channels_status命令.md`
- 学习 **config 命令** → `03_核心概念_05_config命令.md`
- 实战 **高级场景** → `07_实战代码_*.md`

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [CLI 官方文档](https://docs.openclaw.ai/cli)
- [消息路由文档](https://docs.openclaw.ai/routing)
