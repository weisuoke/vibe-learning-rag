# 03_核心概念_05_config命令

## 命令概述

`openclaw config` 是 OpenClaw 的配置管理命令，用于查看、设置和管理 OpenClaw 的各种配置选项。配置存储在 `~/.openclaw/config/settings.json` 中，支持热重载（部分配置需要重启 Gateway）。

---

## 命令语法

```bash
openclaw config <subcommand> [options]
```

---

## 子命令

### 1. `config get` - 获取配置值

**用途**：查看指定配置项的值

**语法**：
```bash
openclaw config get <key>
```

**示例**：
```bash
# 获取默认模型
openclaw config get model

# 获取 Provider
openclaw config get provider

# 获取 Gateway 端口
openclaw config get gateway.port

# 获取所有配置（不指定 key）
openclaw config get
```

**输出示例**：
```bash
$ openclaw config get model
claude-opus-4

$ openclaw config get provider
anthropic

$ openclaw config get
{
  "provider": "anthropic",
  "model": "claude-opus-4",
  "gateway": {
    "port": 18789,
    "verbose": false
  },
  "channels": {
    "whatsapp": {
      "enabled": true
    },
    "telegram": {
      "enabled": true,
      "token": "***"
    }
  }
}
```

---

### 2. `config set` - 设置配置值

**用途**：设置指定配置项的值

**语法**：
```bash
openclaw config set <key> <value>
```

**示例**：
```bash
# 设置默认模型
openclaw config set model claude-opus-4

# 设置 Provider
openclaw config set provider anthropic

# 设置 Gateway 端口
openclaw config set gateway.port 18790

# 设置思考级别
openclaw config set agent.thinking high

# 设置日志级别
openclaw config set logging.level debug
```

**输出示例**：
```bash
$ openclaw config set model claude-opus-4
✅ Configuration updated: model = claude-opus-4
```

---

### 3. `config unset` - 删除配置项

**用途**：删除指定配置项（恢复默认值）

**语法**：
```bash
openclaw config unset <key>
```

**示例**：
```bash
# 删除自定义模型配置
openclaw config unset model

# 删除自定义端口配置
openclaw config unset gateway.port

# 删除通道配置
openclaw config unset channels.telegram.token
```

**输出示例**：
```bash
$ openclaw config unset model
✅ Configuration removed: model (will use default)
```

---

## 配置项详解

### 核心配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `provider` | AI Provider | anthropic | anthropic, openai, google |
| `model` | 默认模型 | claude-opus-4 | claude-opus-4, gpt-4, gemini-pro |
| `workspace` | 工作空间路径 | ~/.openclaw/workspace | /path/to/workspace |

### Gateway 配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `gateway.port` | 监听端口 | 18789 | 18789, 18790 |
| `gateway.verbose` | 详细日志 | false | true, false |
| `gateway.autoStart` | 自动启动 | false | true, false |

### Agent 配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `agent.thinking` | 思考级别 | medium | low, medium, high |
| `agent.maxTokens` | 最大 Token 数 | 4096 | 2048, 4096, 8192 |
| `agent.temperature` | 温度参数 | 0.7 | 0.0 - 1.0 |
| `agent.deliver` | 自动回传 | true | true, false |

### 通道配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `channels.whatsapp.enabled` | 启用 WhatsApp | false | true, false |
| `channels.telegram.enabled` | 启用 Telegram | false | true, false |
| `channels.telegram.token` | Telegram Bot Token | - | bot123:ABC... |
| `channels.slack.enabled` | 启用 Slack | false | true, false |
| `channels.slack.token` | Slack Token | - | xoxb-... |

### 日志配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `logging.level` | 日志级别 | info | debug, info, warn, error |
| `logging.file` | 日志文件路径 | ~/.openclaw/logs/gateway.log | /path/to/log |
| `logging.maxSize` | 日志文件最大大小 | 10MB | 5MB, 10MB, 20MB |

### 安全配置

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `security.dmPolicy` | DM 策略 | pairing | pairing, open, closed |
| `security.allowedUsers` | 允许的用户列表 | [] | ["+1234567890"] |
| `security.blockedUsers` | 阻止的用户列表 | [] | ["+9876543210"] |

---

## 配置文件结构

### 配置文件位置

```bash
~/.openclaw/
├── config/
│   ├── settings.json          # 主配置文件
│   ├── channels.json          # 通道配置
│   ├── credentials.json       # 凭证（加密）
│   └── profiles/              # 配置文件（多环境）
│       ├── dev.json
│       ├── staging.json
│       └── production.json
├── workspace/                 # 工作空间
├── logs/                      # 日志目录
└── sessions/                  # 会话数据
```

### settings.json 示例

```json
{
  "provider": "anthropic",
  "model": "claude-opus-4",
  "workspace": "/Users/username/.openclaw/workspace",
  "gateway": {
    "port": 18789,
    "verbose": false,
    "autoStart": false
  },
  "agent": {
    "thinking": "medium",
    "maxTokens": 4096,
    "temperature": 0.7,
    "deliver": true
  },
  "channels": {
    "whatsapp": {
      "enabled": true
    },
    "telegram": {
      "enabled": true,
      "token": "bot123:ABC..."
    },
    "slack": {
      "enabled": false
    }
  },
  "logging": {
    "level": "info",
    "file": "/Users/username/.openclaw/logs/gateway.log",
    "maxSize": "10MB"
  },
  "security": {
    "dmPolicy": "pairing",
    "allowedUsers": [],
    "blockedUsers": []
  }
}
```

---

## 使用场景

### 场景 1: 初始化配置

```bash
# 设置 Provider 和模型
openclaw config set provider anthropic
openclaw config set model claude-opus-4

# 设置工作空间
openclaw config set workspace ~/my-openclaw-workspace

# 设置 Gateway 端口
openclaw config set gateway.port 18789

# 验证配置
openclaw config get
```

---

### 场景 2: 多环境配置

```bash
# 开发环境
openclaw --profile dev config set model claude-haiku
openclaw --profile dev config set gateway.port 18789

# 测试环境
openclaw --profile staging config set model claude-sonnet
openclaw --profile staging config set gateway.port 18790

# 生产环境
openclaw --profile production config set model claude-opus-4
openclaw --profile production config set gateway.port 18791

# 使用不同环境
openclaw --profile dev gateway
openclaw --profile staging gateway
openclaw --profile production gateway
```

---

### 场景 3: 通道配置

```bash
# 启用 WhatsApp
openclaw config set channels.whatsapp.enabled true

# 配置 Telegram Bot
openclaw config set channels.telegram.enabled true
openclaw config set channels.telegram.token bot123:ABC...

# 配置 Slack
openclaw config set channels.slack.enabled true
openclaw config set channels.slack.token xoxb-...

# 查看通道配置
openclaw config get channels
```

---

### 场景 4: 性能调优

```bash
# 提高 Agent 思考级别
openclaw config set agent.thinking high

# 增加 Token 限制
openclaw config set agent.maxTokens 8192

# 降低温度（更确定性）
openclaw config set agent.temperature 0.3

# 启用详细日志
openclaw config set logging.level debug
openclaw config set gateway.verbose true
```

---

## TypeScript 集成

### 示例 1: 配置管理器

```typescript
// src/config-manager.ts
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';

const execAsync = promisify(exec);

export interface OpenClawConfig {
  provider?: string;
  model?: string;
  workspace?: string;
  gateway?: {
    port?: number;
    verbose?: boolean;
    autoStart?: boolean;
  };
  agent?: {
    thinking?: 'low' | 'medium' | 'high';
    maxTokens?: number;
    temperature?: number;
    deliver?: boolean;
  };
  channels?: Record<string, any>;
  logging?: {
    level?: 'debug' | 'info' | 'warn' | 'error';
    file?: string;
    maxSize?: string;
  };
  security?: {
    dmPolicy?: 'pairing' | 'open' | 'closed';
    allowedUsers?: string[];
    blockedUsers?: string[];
  };
}

export class ConfigManager {
  private configPath: string;

  constructor(profile = 'default') {
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    this.configPath = path.join(
      homeDir,
      profile === 'default' ? '.openclaw' : `.openclaw-${profile}`,
      'config',
      'settings.json'
    );
  }

  /**
   * 获取配置值
   */
  async get(key?: string): Promise<any> {
    if (key) {
      const { stdout } = await execAsync(`openclaw config get ${key}`);
      return stdout.trim();
    }

    // 获取所有配置
    const { stdout } = await execAsync('openclaw config get');
    return JSON.parse(stdout);
  }

  /**
   * 设置配置值
   */
  async set(key: string, value: any): Promise<void> {
    const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
    await execAsync(`openclaw config set ${key} ${valueStr}`);
    console.log(`✅ Set ${key} = ${valueStr}`);
  }

  /**
   * 删除配置项
   */
  async unset(key: string): Promise<void> {
    await execAsync(`openclaw config unset ${key}`);
    console.log(`✅ Unset ${key}`);
  }

  /**
   * 批量设置配置
   */
  async setMultiple(config: Record<string, any>): Promise<void> {
    for (const [key, value] of Object.entries(config)) {
      await this.set(key, value);
    }
  }

  /**
   * 读取配置文件
   */
  async readConfigFile(): Promise<OpenClawConfig> {
    try {
      const content = await fs.readFile(this.configPath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to read config file: ${error.message}`);
    }
  }

  /**
   * 写入配置文件
   */
  async writeConfigFile(config: OpenClawConfig): Promise<void> {
    try {
      await fs.writeFile(
        this.configPath,
        JSON.stringify(config, null, 2),
        'utf-8'
      );
      console.log(`✅ Config file updated: ${this.configPath}`);
    } catch (error) {
      throw new Error(`Failed to write config file: ${error.message}`);
    }
  }

  /**
   * 合并配置
   */
  async mergeConfig(updates: Partial<OpenClawConfig>): Promise<void> {
    const current = await this.readConfigFile();
    const merged = this.deepMerge(current, updates);
    await this.writeConfigFile(merged);
  }

  /**
   * 重置配置
   */
  async reset(): Promise<void> {
    const defaultConfig: OpenClawConfig = {
      provider: 'anthropic',
      model: 'claude-opus-4',
      gateway: {
        port: 18789,
        verbose: false,
        autoStart: false,
      },
      agent: {
        thinking: 'medium',
        maxTokens: 4096,
        temperature: 0.7,
        deliver: true,
      },
      logging: {
        level: 'info',
      },
      security: {
        dmPolicy: 'pairing',
        allowedUsers: [],
        blockedUsers: [],
      },
    };

    await this.writeConfigFile(defaultConfig);
    console.log('✅ Config reset to defaults');
  }

  /**
   * 验证配置
   */
  async validate(): Promise<{ valid: boolean; errors: string[] }> {
    const errors: string[] = [];
    const config = await this.readConfigFile();

    // 验证 Provider
    const validProviders = ['anthropic', 'openai', 'google'];
    if (config.provider && !validProviders.includes(config.provider)) {
      errors.push(`Invalid provider: ${config.provider}`);
    }

    // 验证端口
    if (config.gateway?.port) {
      const port = config.gateway.port;
      if (port < 1024 || port > 65535) {
        errors.push(`Invalid port: ${port} (must be 1024-65535)`);
      }
    }

    // 验证思考级别
    const validThinking = ['low', 'medium', 'high'];
    if (config.agent?.thinking && !validThinking.includes(config.agent.thinking)) {
      errors.push(`Invalid thinking level: ${config.agent.thinking}`);
    }

    // 验证温度
    if (config.agent?.temperature !== undefined) {
      const temp = config.agent.temperature;
      if (temp < 0 || temp > 1) {
        errors.push(`Invalid temperature: ${temp} (must be 0-1)`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * 深度合并对象
   */
  private deepMerge(target: any, source: any): any {
    const output = { ...target };

    for (const key in source) {
      if (source[key] instanceof Object && key in target) {
        output[key] = this.deepMerge(target[key], source[key]);
      } else {
        output[key] = source[key];
      }
    }

    return output;
  }
}
```

### 示例 2: 配置预设

```typescript
// src/config-presets.ts
import { ConfigManager, OpenClawConfig } from './config-manager';

export class ConfigPresets {
  private manager: ConfigManager;

  constructor(profile = 'default') {
    this.manager = new ConfigManager(profile);
  }

  /**
   * 开发环境预设
   */
  async applyDevelopment(): Promise<void> {
    const config: Partial<OpenClawConfig> = {
      model: 'claude-haiku',
      gateway: {
        port: 18789,
        verbose: true,
      },
      agent: {
        thinking: 'low',
        maxTokens: 2048,
      },
      logging: {
        level: 'debug',
      },
    };

    await this.manager.mergeConfig(config);
    console.log('✅ Development preset applied');
  }

  /**
   * 生产环境预设
   */
  async applyProduction(): Promise<void> {
    const config: Partial<OpenClawConfig> = {
      model: 'claude-opus-4',
      gateway: {
        port: 18789,
        verbose: false,
        autoStart: true,
      },
      agent: {
        thinking: 'high',
        maxTokens: 8192,
        temperature: 0.7,
      },
      logging: {
        level: 'info',
      },
      security: {
        dmPolicy: 'pairing',
      },
    };

    await this.manager.mergeConfig(config);
    console.log('✅ Production preset applied');
  }

  /**
   * 性能优化预设
   */
  async applyPerformance(): Promise<void> {
    const config: Partial<OpenClawConfig> = {
      model: 'claude-haiku',
      agent: {
        thinking: 'low',
        maxTokens: 2048,
        temperature: 0.3,
      },
      logging: {
        level: 'warn',
      },
    };

    await this.manager.mergeConfig(config);
    console.log('✅ Performance preset applied');
  }

  /**
   * 高质量预设
   */
  async applyHighQuality(): Promise<void> {
    const config: Partial<OpenClawConfig> = {
      model: 'claude-opus-4',
      agent: {
        thinking: 'high',
        maxTokens: 8192,
        temperature: 0.8,
      },
    };

    await this.manager.mergeConfig(config);
    console.log('✅ High quality preset applied');
  }
}

// 使用示例
const presets = new ConfigPresets();

// 应用开发环境预设
await presets.applyDevelopment();

// 应用生产环境预设
await presets.applyProduction();
```

### 示例 3: 配置监听器

```typescript
// src/config-watcher.ts
import { EventEmitter } from 'events';
import * as fs from 'fs';
import { ConfigManager } from './config-manager';

export class ConfigWatcher extends EventEmitter {
  private manager: ConfigManager;
  private watcher: fs.FSWatcher | null = null;
  private lastConfig: any = null;

  constructor(profile = 'default') {
    super();
    this.manager = new ConfigManager(profile);
  }

  /**
   * 开始监听配置变化
   */
  async start(): Promise<void> {
    const configPath = (this.manager as any).configPath;

    // 读取初始配置
    this.lastConfig = await this.manager.readConfigFile();

    // 监听文件变化
    this.watcher = fs.watch(configPath, async (eventType) => {
      if (eventType === 'change') {
        await this.handleConfigChange();
      }
    });

    console.log('📡 Config watcher started');
  }

  /**
   * 停止监听
   */
  stop(): void {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
      console.log('📡 Config watcher stopped');
    }
  }

  /**
   * 处理配置变化
   */
  private async handleConfigChange(): Promise<void> {
    try {
      const newConfig = await this.manager.readConfigFile();
      const changes = this.detectChanges(this.lastConfig, newConfig);

      if (changes.length > 0) {
        this.emit('config-changed', { changes, newConfig });

        for (const change of changes) {
          this.emit(`config-changed:${change.key}`, {
            oldValue: change.oldValue,
            newValue: change.newValue,
          });
        }
      }

      this.lastConfig = newConfig;
    } catch (error) {
      this.emit('error', error);
    }
  }

  /**
   * 检测配置变化
   */
  private detectChanges(oldConfig: any, newConfig: any, prefix = ''): Array<{
    key: string;
    oldValue: any;
    newValue: any;
  }> {
    const changes: Array<{ key: string; oldValue: any; newValue: any }> = [];

    for (const key in newConfig) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      const oldValue = oldConfig?.[key];
      const newValue = newConfig[key];

      if (typeof newValue === 'object' && newValue !== null) {
        changes.push(...this.detectChanges(oldValue || {}, newValue, fullKey));
      } else if (oldValue !== newValue) {
        changes.push({ key: fullKey, oldValue, newValue });
      }
    }

    return changes;
  }
}

// 使用示例
const watcher = new ConfigWatcher();

watcher.on('config-changed', ({ changes }) => {
  console.log('⚙️  Config changed:', changes);
});

watcher.on('config-changed:model', ({ oldValue, newValue }) => {
  console.log(`🔄 Model changed: ${oldValue} → ${newValue}`);
});

watcher.on('config-changed:gateway.port', ({ oldValue, newValue }) => {
  console.log(`🔄 Port changed: ${oldValue} → ${newValue}`);
  console.log('⚠️  Gateway restart required');
});

await watcher.start();
```

---

## 常见问题

### Q1: 配置修改后需要重启 Gateway 吗？

**部分配置需要重启**：
- `gateway.port`: 需要重启
- `gateway.verbose`: 需要重启
- `channels.*`: 需要重启

**热重载配置**：
- `agent.thinking`: 立即生效
- `agent.temperature`: 立即生效
- `logging.level`: 立即生效

**解决方案**：
```bash
# 修改配置后重启 Gateway
openclaw config set gateway.port 18790
openclaw gateway restart
```

---

### Q2: 如何备份和恢复配置？

**备份配置**：
```bash
# 备份配置文件
cp ~/.openclaw/config/settings.json ~/.openclaw/config/settings.backup.json

# 或使用 TypeScript
const manager = new ConfigManager();
const config = await manager.readConfigFile();
await fs.writeFile('backup.json', JSON.stringify(config, null, 2));
```

**恢复配置**：
```bash
# 恢复配置文件
cp ~/.openclaw/config/settings.backup.json ~/.openclaw/config/settings.json

# 或使用 TypeScript
const backup = JSON.parse(await fs.readFile('backup.json', 'utf-8'));
await manager.writeConfigFile(backup);
```

---

### Q3: 如何在多个环境之间切换？

**解决方案**：
```bash
# 使用 --profile 标志
openclaw --profile dev gateway
openclaw --profile staging gateway
openclaw --profile production gateway

# 或使用环境变量
export OPENCLAW_PROFILE=dev
openclaw gateway
```

---

### Q4: 配置文件损坏怎么办？

**解决方案**：
```bash
# 重置配置
openclaw reset

# 或手动删除配置文件
rm ~/.openclaw/config/settings.json

# 重新运行 onboarding
openclaw onboard
```

---

## 最佳实践

### 1. 使用版本控制管理配置

```bash
# 创建配置仓库
cd ~/.openclaw/config
git init
git add settings.json
git commit -m "Initial config"

# 修改配置后提交
openclaw config set model claude-opus-4
git add settings.json
git commit -m "Update model to claude-opus-4"
```

### 2. 使用环境变量覆盖配置

```bash
# 临时覆盖配置
OPENCLAW_MODEL=claude-haiku openclaw agent --message "Test"

# 永久设置
export OPENCLAW_MODEL=claude-opus-4
export OPENCLAW_GATEWAY_PORT=18789
```

### 3. 定期验证配置

```typescript
// 定期验证配置
const manager = new ConfigManager();
const { valid, errors } = await manager.validate();

if (!valid) {
  console.error('❌ Config validation failed:', errors);
  // 发送告警
}
```

### 4. 使用配置预设

```typescript
// 根据环境应用预设
const presets = new ConfigPresets();

if (process.env.NODE_ENV === 'development') {
  await presets.applyDevelopment();
} else if (process.env.NODE_ENV === 'production') {
  await presets.applyProduction();
}
```

---

## 下一步

- 学习 **实战场景** → `07_实战代码_场景1_基础命令使用.md`
- 学习 **配置管理实战** → `07_实战代码_场景2_配置管理实战.md`
- 学习 **Gateway 管理** → `04_Gateway启动与管理/`

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [配置文档](https://docs.openclaw.ai/configuration)
- [CLI 官方文档](https://docs.openclaw.ai/cli)
- [环境变量](https://docs.openclaw.ai/configuration/environment-variables)
