# OpenClaw 安装与配置 - 实战代码 Part 2: 配置 Gateway

**目标：** 提供完整的 Gateway 配置自动化脚本

本文档包含 Onboarding Wizard 自动化、配置生成和 Gateway 启动的完整代码。

---

## 完整配置自动化脚本

```typescript
/**
 * OpenClaw Gateway 配置自动化脚本
 *
 * 功能：
 * 1. 自动化 Onboarding Wizard
 * 2. 生成配置文件
 * 3. 启动 Gateway
 * 4. 验证 Gateway 状态
 *
 * 使用方法：
 *   ts-node configure-gateway.ts
 */

import { execSync, spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';

// ===== 类型定义 =====

interface GatewayConfig {
  gateway: {
    port: number;
    host: string;
    mode: 'development' | 'production';
    logLevel: 'debug' | 'info' | 'warn' | 'error';
  };
  channels: {
    [key: string]: ChannelConfig;
  };
  cache?: {
    enabled: boolean;
    ttl: number;
    maxSize: string;
  };
  logging?: {
    file: string;
    maxSize: string;
    maxFiles: number;
    format: 'json' | 'text';
  };
}

interface ChannelConfig {
  apiKey: string;
  enabled: boolean;
  model: string;
  maxTokens: number;
  temperature: number;
}

interface OnboardingAnswers {
  claudeApiKey?: string;
  gptApiKey?: string;
  geminiApiKey?: string;
  gatewayPort: number;
  enabledChannels: string[];
}

// ===== 配置目录管理 =====

/**
 * 获取配置目录路径
 */
function getConfigDir(): string {
  return path.join(os.homedir(), '.openclaw');
}

/**
 * 获取配置文件路径
 */
function getConfigPath(): string {
  return path.join(getConfigDir(), 'settings.json');
}

/**
 * 创建配置目录
 */
function ensureConfigDir(): void {
  const configDir = getConfigDir();

  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true, mode: 0o700 });
    console.log(`✅ 创建配置目录: ${configDir}\n`);
  }

  // 创建子目录
  const subdirs = ['sessions', 'cache', 'plugins'];
  for (const subdir of subdirs) {
    const subdirPath = path.join(configDir, subdir);
    if (!fs.existsSync(subdirPath)) {
      fs.mkdirSync(subdirPath, { mode: 0o700 });
    }
  }
}

// ===== 交互式输入 =====

/**
 * 创建 readline 接口
 */
function createReadlineInterface(): readline.Interface {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
}

/**
 * 询问问题
 */
function question(rl: readline.Interface, prompt: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(prompt, (answer) => {
      resolve(answer.trim());
    });
  });
}

/**
 * 询问是否继续（y/n）
 */
async function confirm(rl: readline.Interface, prompt: string): Promise<boolean> {
  const answer = await question(rl, `${prompt} (y/n): `);
  return answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes';
}

// ===== Onboarding Wizard =====

/**
 * 运行 Onboarding Wizard
 */
async function runOnboardingWizard(): Promise<OnboardingAnswers> {
  console.log('🎉 Welcome to OpenClaw!\n');
  console.log('Let\'s set up your AI gateway.\n');

  const rl = createReadlineInterface();
  const answers: OnboardingAnswers = {
    gatewayPort: 3000,
    enabledChannels: []
  };

  try {
    // 1. Claude API Key
    console.log('=== Claude Configuration ===\n');
    const enableClaude = await confirm(rl, 'Enable Claude channel?');

    if (enableClaude) {
      const claudeKey = await question(rl, 'Enter your Claude API Key: ');
      if (claudeKey && claudeKey.startsWith('sk-ant-')) {
        answers.claudeApiKey = claudeKey;
        answers.enabledChannels.push('claude');
        console.log('✅ Claude API Key saved\n');
      } else {
        console.log('⚠️  Invalid Claude API Key format (should start with sk-ant-)\n');
      }
    }

    // 2. GPT API Key
    console.log('=== GPT Configuration ===\n');
    const enableGPT = await confirm(rl, 'Enable GPT channel?');

    if (enableGPT) {
      const gptKey = await question(rl, 'Enter your OpenAI API Key: ');
      if (gptKey && gptKey.startsWith('sk-')) {
        answers.gptApiKey = gptKey;
        answers.enabledChannels.push('gpt');
        console.log('✅ GPT API Key saved\n');
      } else {
        console.log('⚠️  Invalid OpenAI API Key format (should start with sk-)\n');
      }
    }

    // 3. Gemini API Key
    console.log('=== Gemini Configuration ===\n');
    const enableGemini = await confirm(rl, 'Enable Gemini channel?');

    if (enableGemini) {
      const geminiKey = await question(rl, 'Enter your Gemini API Key: ');
      if (geminiKey) {
        answers.geminiApiKey = geminiKey;
        answers.enabledChannels.push('gemini');
        console.log('✅ Gemini API Key saved\n');
      }
    }

    // 4. Gateway Port
    console.log('=== Gateway Configuration ===\n');
    const portInput = await question(rl, 'Gateway port (default: 3000): ');
    if (portInput) {
      const port = parseInt(portInput);
      if (!isNaN(port) && port >= 1024 && port <= 65535) {
        answers.gatewayPort = port;
      } else {
        console.log('⚠️  Invalid port, using default: 3000\n');
      }
    }

    console.log(`✅ Gateway will listen on port ${answers.gatewayPort}\n`);

    return answers;

  } finally {
    rl.close();
  }
}

// ===== 配置生成 =====

/**
 * 生成配置对象
 */
function generateConfig(answers: OnboardingAnswers): GatewayConfig {
  const config: GatewayConfig = {
    gateway: {
      port: answers.gatewayPort,
      host: 'localhost',
      mode: 'development',
      logLevel: 'info'
    },
    channels: {}
  };

  // Claude 通道
  if (answers.claudeApiKey) {
    config.channels.claude = {
      apiKey: answers.claudeApiKey,
      enabled: true,
      model: 'claude-3-5-sonnet-20241022',
      maxTokens: 4096,
      temperature: 0.7
    };
  }

  // GPT 通道
  if (answers.gptApiKey) {
    config.channels.gpt = {
      apiKey: answers.gptApiKey,
      enabled: true,
      model: 'gpt-4',
      maxTokens: 4096,
      temperature: 0.7
    };
  }

  // Gemini 通道
  if (answers.geminiApiKey) {
    config.channels.gemini = {
      apiKey: answers.geminiApiKey,
      enabled: true,
      model: 'gemini-pro',
      maxTokens: 4096,
      temperature: 0.7
    };
  }

  // 缓存配置
  config.cache = {
    enabled: false,
    ttl: 3600,
    maxSize: '100MB'
  };

  // 日志配置
  config.logging = {
    file: 'gateway.log',
    maxSize: '10MB',
    maxFiles: 5,
    format: 'json'
  };

  return config;
}

/**
 * 保存配置文件
 */
function saveConfig(config: GatewayConfig): void {
  const configPath = getConfigPath();
  const configData = JSON.stringify(config, null, 2);

  // 写入配置文件
  fs.writeFileSync(configPath, configData, { mode: 0o600 });

  console.log(`✅ Configuration saved to ${configPath}\n`);
}

// ===== Gateway 启动 =====

/**
 * 检查 Gateway 是否正在运行
 */
function isGatewayRunning(): boolean {
  try {
    execSync('pgrep -f "openclaw gateway"', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * 停止 Gateway
 */
function stopGateway(): void {
  if (!isGatewayRunning()) {
    console.log('⚠️  Gateway is not running\n');
    return;
  }

  try {
    execSync('pkill -f "openclaw gateway"');
    console.log('✅ Gateway stopped\n');
  } catch (error) {
    console.error('❌ Failed to stop Gateway\n');
  }
}

/**
 * 启动 Gateway（前台模式）
 */
function startGatewayForeground(port: number): ChildProcess {
  console.log(`🚀 Starting Gateway on port ${port}...\n`);
  console.log('Press Ctrl+C to stop\n');

  const child = spawn('openclaw', ['gateway', 'start', '--port', port.toString()], {
    stdio: 'inherit'
  });

  child.on('error', (error) => {
    console.error(`❌ Failed to start Gateway: ${error.message}`);
    process.exit(1);
  });

  child.on('exit', (code) => {
    console.log(`\n✅ Gateway stopped (exit code: ${code})`);
    process.exit(code || 0);
  });

  return child;
}

/**
 * 启动 Gateway（守护进程模式）
 */
function startGatewayDaemon(port: number): void {
  console.log(`🚀 Starting Gateway in daemon mode on port ${port}...\n`);

  const logPath = path.join(getConfigDir(), 'gateway.log');
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });

  const child = spawn('openclaw', ['gateway', 'start', '--port', port.toString()], {
    detached: true,
    stdio: ['ignore', logStream, logStream]
  });

  child.unref();

  console.log(`✅ Gateway started in background (PID: ${child.pid})`);
  console.log(`   Log file: ${logPath}`);
  console.log(`   View logs: tail -f ${logPath}`);
  console.log(`   Stop Gateway: openclaw gateway stop\n`);
}

// ===== Gateway 验证 =====

/**
 * 等待 Gateway 启动
 */
async function waitForGateway(port: number, timeout: number = 30000): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    try {
      execSync(`curl -s http://localhost:${port}/health`, { stdio: 'ignore' });
      return true;
    } catch {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  return false;
}

/**
 * 验证 Gateway 状态
 */
async function verifyGateway(port: number): Promise<boolean> {
  console.log('🔍 Verifying Gateway...\n');

  try {
    const response = execSync(`curl -s http://localhost:${port}/health`, {
      encoding: 'utf-8'
    });

    const health = JSON.parse(response);

    if (health.status === 'ok') {
      console.log('✅ Gateway is healthy');
      console.log(`   Version: ${health.version || 'unknown'}`);
      console.log(`   Port: ${port}\n`);
      return true;
    } else {
      console.log('❌ Gateway health check failed\n');
      return false;
    }
  } catch (error) {
    console.log('❌ Cannot connect to Gateway\n');
    return false;
  }
}

// ===== 环境变量配置 =====

/**
 * 从环境变量加载配置
 */
function loadConfigFromEnv(): Partial<OnboardingAnswers> {
  const answers: Partial<OnboardingAnswers> = {};

  if (process.env.CLAUDE_API_KEY) {
    answers.claudeApiKey = process.env.CLAUDE_API_KEY;
    answers.enabledChannels = answers.enabledChannels || [];
    answers.enabledChannels.push('claude');
  }

  if (process.env.OPENAI_API_KEY) {
    answers.gptApiKey = process.env.OPENAI_API_KEY;
    answers.enabledChannels = answers.enabledChannels || [];
    answers.enabledChannels.push('gpt');
  }

  if (process.env.GEMINI_API_KEY) {
    answers.geminiApiKey = process.env.GEMINI_API_KEY;
    answers.enabledChannels = answers.enabledChannels || [];
    answers.enabledChannels.push('gemini');
  }

  if (process.env.OPENCLAW_PORT) {
    answers.gatewayPort = parseInt(process.env.OPENCLAW_PORT);
  }

  return answers;
}

/**
 * 非交互式配置（使用环境变量）
 */
function configureFromEnv(): GatewayConfig | null {
  console.log('🔍 Loading configuration from environment variables...\n');

  const envConfig = loadConfigFromEnv();

  if (!envConfig.claudeApiKey && !envConfig.gptApiKey && !envConfig.geminiApiKey) {
    console.log('⚠️  No API keys found in environment variables\n');
    return null;
  }

  const answers: OnboardingAnswers = {
    claudeApiKey: envConfig.claudeApiKey,
    gptApiKey: envConfig.gptApiKey,
    geminiApiKey: envConfig.geminiApiKey,
    gatewayPort: envConfig.gatewayPort || 3000,
    enabledChannels: envConfig.enabledChannels || []
  };

  console.log('✅ Configuration loaded from environment\n');
  console.log(`   Enabled channels: ${answers.enabledChannels.join(', ')}`);
  console.log(`   Gateway port: ${answers.gatewayPort}\n`);

  return generateConfig(answers);
}

// ===== 主函数 =====

/**
 * 主配置流程
 */
async function main(): Promise<void> {
  console.log('');
  console.log('='.repeat(60));
  console.log('OpenClaw Gateway Configuration');
  console.log('='.repeat(60));
  console.log('');

  try {
    // 1. 确保配置目录存在
    ensureConfigDir();

    // 2. 检查是否使用环境变量配置
    const useEnv = process.argv.includes('--env') || process.argv.includes('-e');
    const daemon = process.argv.includes('--daemon') || process.argv.includes('-d');

    let config: GatewayConfig;

    if (useEnv) {
      // 从环境变量加载配置
      const envConfig = configureFromEnv();
      if (!envConfig) {
        console.log('Please set environment variables or run without --env flag\n');
        process.exit(1);
      }
      config = envConfig;
    } else {
      // 运行交互式 Onboarding Wizard
      const answers = await runOnboardingWizard();

      if (answers.enabledChannels.length === 0) {
        console.log('❌ No channels enabled. Please enable at least one channel.\n');
        process.exit(1);
      }

      // 生成配置
      config = generateConfig(answers);
    }

    // 3. 保存配置
    saveConfig(config);

    // 4. 询问是否启动 Gateway
    if (!useEnv) {
      const rl = createReadlineInterface();
      const shouldStart = await confirm(rl, 'Start Gateway now?');
      rl.close();

      if (!shouldStart) {
        console.log('\nConfiguration complete!');
        console.log('\nTo start Gateway later:');
        console.log('  openclaw gateway start\n');
        return;
      }
    }

    // 5. 停止已运行的 Gateway
    if (isGatewayRunning()) {
      console.log('⚠️  Gateway is already running. Stopping...\n');
      stopGateway();
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // 6. 启动 Gateway
    if (daemon) {
      startGatewayDaemon(config.gateway.port);

      // 等待 Gateway 启动
      console.log('Waiting for Gateway to start...\n');
      const started = await waitForGateway(config.gateway.port);

      if (started) {
        await verifyGateway(config.gateway.port);
      } else {
        console.log('❌ Gateway failed to start within timeout\n');
        process.exit(1);
      }
    } else {
      startGatewayForeground(config.gateway.port);
    }

  } catch (error: any) {
    console.error('');
    console.error('❌ Configuration failed:');
    console.error(`   ${error.message}`);
    console.error('');
    process.exit(1);
  }
}

// 运行主函数
if (require.main === module) {
  main();
}

// 导出函数供其他模块使用
export {
  runOnboardingWizard,
  generateConfig,
  saveConfig,
  startGatewayForeground,
  startGatewayDaemon,
  verifyGateway,
  configureFromEnv
};
```

---

## 使用方法

### 1. 交互式配置

```bash
# 运行配置向导
npx tsx configure-gateway.ts

# 按提示输入：
# - Claude API Key
# - GPT API Key (可选)
# - Gemini API Key (可选)
# - Gateway 端口
```

### 2. 使用环境变量配置

```bash
# 设置环境变量
export CLAUDE_API_KEY=sk-ant-api03-...
export OPENAI_API_KEY=sk-...
export OPENCLAW_PORT=3000

# 运行配置（非交互式）
npx tsx configure-gateway.ts --env --daemon
```

### 3. 作为 npm script

```json
{
  "scripts": {
    "configure": "tsx configure-gateway.ts",
    "configure:env": "tsx configure-gateway.ts --env --daemon"
  }
}
```

---

## 预期输出

### 交互式配置

```
============================================================
OpenClaw Gateway Configuration
============================================================

✅ 创建配置目录: /Users/username/.openclaw

🎉 Welcome to OpenClaw!

Let's set up your AI gateway.

=== Claude Configuration ===

Enable Claude channel? (y/n): y
Enter your Claude API Key: sk-ant-api03-...
✅ Claude API Key saved

=== GPT Configuration ===

Enable GPT channel? (y/n): n

=== Gemini Configuration ===

Enable Gemini channel? (y/n): n

=== Gateway Configuration ===

Gateway port (default: 3000):
✅ Gateway will listen on port 3000

✅ Configuration saved to /Users/username/.openclaw/settings.json

Start Gateway now? (y/n): y

🚀 Starting Gateway on port 3000...

Press Ctrl+C to stop

[Gateway 启动日志...]
```

### 环境变量配置

```
============================================================
OpenClaw Gateway Configuration
============================================================

✅ 创建配置目录: /Users/username/.openclaw

🔍 Loading configuration from environment variables...

✅ Configuration loaded from environment

   Enabled channels: claude, gpt
   Gateway port: 3000

✅ Configuration saved to /Users/username/.openclaw/settings.json

🚀 Starting Gateway in daemon mode on port 3000...

✅ Gateway started in background (PID: 12345)
   Log file: /Users/username/.openclaw/gateway.log
   View logs: tail -f /Users/username/.openclaw/gateway.log
   Stop Gateway: openclaw gateway stop

Waiting for Gateway to start...

🔍 Verifying Gateway...

✅ Gateway is healthy
   Version: 1.2.3
   Port: 3000
```

---

## 实际应用场景

### 场景 1：Docker 容器配置

```dockerfile
# Dockerfile
FROM node:22-alpine

WORKDIR /app

# 安装 OpenClaw
RUN npm install -g openclaw@latest tsx

# 复制配置脚本
COPY configure-gateway.ts .

# 设置环境变量
ENV CLAUDE_API_KEY=sk-ant-api03-...
ENV OPENCLAW_PORT=3000

# 运行配置并启动 Gateway
CMD ["tsx", "configure-gateway.ts", "--env", "--daemon"]
```

### 场景 2：CI/CD 自动化

```yaml
# .github/workflows/deploy-gateway.yml
name: Deploy Gateway

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '22'

      - name: Install OpenClaw
        run: npm install -g openclaw@latest

      - name: Configure Gateway
        env:
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: npx tsx configure-gateway.ts --env

      - name: Start Gateway
        run: openclaw gateway start --daemon

      - name: Verify Gateway
        run: curl http://localhost:3000/health
```

### 场景 3：多环境配置

```bash
# development.env
CLAUDE_API_KEY=sk-ant-api03-dev-...
OPENCLAW_PORT=3000

# production.env
CLAUDE_API_KEY=sk-ant-api03-prod-...
OPENCLAW_PORT=8080

# 使用不同环境
source development.env && npx tsx configure-gateway.ts --env
source production.env && npx tsx configure-gateway.ts --env
```

---

## 参考资料

### 核心引用

1. **OpenClaw Onboarding CLI Reference**
   - URL: https://docs.openclaw.ai/cli/onboard
   - 关键内容: Onboarding wizard 流程、配置项说明
   - 引用章节: Onboarding Wizard 实现

2. **OpenClaw Gateway Configuration**
   - URL: https://docs.openclaw.ai/gateway/configuration
   - 关键内容: Gateway 配置格式、通道配置
   - 引用章节: 配置生成逻辑

3. **Node.js CLI Best Practices**
   - URL: https://github.com/lirantal/nodejs-cli-apps-best-practices
   - 关键内容: 交互式 CLI 设计、配置管理
   - 引用章节: 交互式输入实现

---

*本文档提供完整的 OpenClaw Gateway 配置自动化代码。*
*最后更新：2026-02-22*
