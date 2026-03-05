# 实战代码 1：前台启动 Gateway

## 场景概述

前台启动 Gateway 是开发调试的主要方式，本节提供完整的 TypeScript 代码示例，展示如何在不同场景下前台启动 Gateway。

**参考资料：**
- [OpenClaw CLI Gateway](https://docs.openclaw.ai/cli/gateway) - 前台启动命令
- [Gateway Runbook](https://docs.openclaw.ai/gateway) - 运行手册
- [OpenClaw Architecture](https://ppaolo.substack.com/p/openclaw-system-architecture-overview) - 架构详解

---

## 场景 1：基础前台启动

### 原理

前台启动将 Gateway 进程运行在当前终端，日志实时输出到 stdout/stderr，适合快速测试和开发调试。

### 实现

```bash
# 最简单的前台启动
openclaw gateway

# 等价命令
openclaw gateway run
```

### 完整示例

```typescript
// scripts/start-gateway-foreground.ts
/**
 * 前台启动 Gateway 脚本
 * 用途：开发环境快速启动
 */

import { spawn } from 'child_process';
import * as path from 'path';

interface GatewayOptions {
  port?: number;
  bind?: 'loopback' | 'lan' | 'tailnet';
  verbose?: boolean;
  dev?: boolean;
}

/**
 * 前台启动 Gateway
 */
function startGatewayForeground(options: GatewayOptions = {}) {
  const args = ['gateway'];

  // 添加选项
  if (options.port) {
    args.push('--port', options.port.toString());
  }

  if (options.bind) {
    args.push('--bind', options.bind);
  }

  if (options.verbose) {
    args.push('--verbose');
  }

  if (options.dev) {
    args.push('--dev');
  }

  console.log(`Starting Gateway: openclaw ${args.join(' ')}`);

  // 启动 Gateway 进程
  const gateway = spawn('openclaw', args, {
    stdio: 'inherit', // 继承父进程的 stdio
    shell: true
  });

  // 监听进程事件
  gateway.on('error', (error) => {
    console.error('Failed to start Gateway:', error);
    process.exit(1);
  });

  gateway.on('exit', (code, signal) => {
    if (code !== null) {
      console.log(`Gateway exited with code ${code}`);
    } else if (signal !== null) {
      console.log(`Gateway killed with signal ${signal}`);
    }
    process.exit(code || 0);
  });

  // 处理 Ctrl+C
  process.on('SIGINT', () => {
    console.log('\nStopping Gateway...');
    gateway.kill('SIGINT');
  });

  return gateway;
}

// 使用示例
if (require.main === module) {
  startGatewayForeground({
    port: 18789,
    bind: 'loopback',
    verbose: true,
    dev: true
  });
}

export { startGatewayForeground };
```

### 使用方式

```bash
# 运行脚本
ts-node scripts/start-gateway-foreground.ts

# 或编译后运行
tsc scripts/start-gateway-foreground.ts
node scripts/start-gateway-foreground.js
```

---

## 场景 2：开发模式启动

### 原理

开发模式自动创建开发配置，跳过 onboarding 流程，适合快速迭代开发。

### 实现

```typescript
// scripts/dev-gateway.ts
/**
 * 开发模式启动 Gateway
 * 特性：
 * - 自动创建 dev 配置
 * - 详细日志输出
 * - 热重载支持
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

interface DevGatewayOptions {
  port?: number;
  reset?: boolean;
  wsLog?: 'auto' | 'full' | 'compact';
}

/**
 * 开发模式启动 Gateway
 */
function startDevGateway(options: DevGatewayOptions = {}) {
  const {
    port = 18789,
    reset = false,
    wsLog = 'full'
  } = options;

  const args = [
    'gateway',
    '--dev',
    '--verbose',
    '--port', port.toString(),
    '--ws-log', wsLog
  ];

  if (reset) {
    args.push('--reset');
  }

  console.log('🚀 Starting Gateway in development mode...');
  console.log(`   Port: ${port}`);
  console.log(`   WebSocket Log: ${wsLog}`);
  console.log(`   Reset: ${reset}`);
  console.log('');

  const gateway = spawn('openclaw', args, {
    stdio: 'inherit',
    shell: true,
    env: {
      ...process.env,
      // 开发环境变量
      NODE_ENV: 'development',
      OPENCLAW_PROFILE: 'dev'
    }
  });

  gateway.on('error', (error) => {
    console.error('❌ Failed to start Gateway:', error);
    process.exit(1);
  });

  gateway.on('exit', (code) => {
    if (code === 0) {
      console.log('✅ Gateway stopped gracefully');
    } else {
      console.log(`❌ Gateway exited with code ${code}`);
    }
    process.exit(code || 0);
  });

  // Ctrl+C 处理
  process.on('SIGINT', () => {
    console.log('\n🛑 Stopping Gateway...');
    gateway.kill('SIGINT');
  });

  return gateway;
}

// 使用示例
if (require.main === module) {
  startDevGateway({
    port: 18789,
    reset: process.argv.includes('--reset'),
    wsLog: 'full'
  });
}

export { startDevGateway };
```

---

## 场景 3：调试模式启动

### 原理

调试模式启用所有详细日志，包括 WebSocket 消息、原始流日志等，用于深度调试。

### 实现

```typescript
// scripts/debug-gateway.ts
/**
 * 调试模式启动 Gateway
 * 特性：
 * - 完整 WebSocket 日志
 * - 原始流日志
 * - Claude CLI 日志
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

interface DebugGatewayOptions {
  port?: number;
  rawStream?: boolean;
  rawStreamPath?: string;
  claudeCliLogs?: boolean;
}

/**
 * 调试模式启动 Gateway
 */
function startDebugGateway(options: DebugGatewayOptions = {}) {
  const {
    port = 18789,
    rawStream = true,
    rawStreamPath = `./debug/stream-${Date.now()}.jsonl`,
    claudeCliLogs = false
  } = options;

  // 确保调试目录存在
  if (rawStream && rawStreamPath) {
    const dir = path.dirname(rawStreamPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  const args = [
    'gateway',
    '--verbose',
    '--port', port.toString(),
    '--ws-log', 'full'
  ];

  if (rawStream) {
    args.push('--raw-stream');
    if (rawStreamPath) {
      args.push('--raw-stream-path', rawStreamPath);
    }
  }

  if (claudeCliLogs) {
    args.push('--claude-cli-logs');
  }

  console.log('🔍 Starting Gateway in debug mode...');
  console.log(`   Port: ${port}`);
  console.log(`   Raw Stream: ${rawStream}`);
  if (rawStream && rawStreamPath) {
    console.log(`   Stream Path: ${rawStreamPath}`);
  }
  console.log(`   Claude CLI Logs: ${claudeCliLogs}`);
  console.log('');
  console.log('📝 Debug tips:');
  console.log('   - All WebSocket messages will be logged');
  console.log('   - Raw stream events saved to file');
  console.log('   - Press Ctrl+C to stop');
  console.log('');

  const gateway = spawn('openclaw', args, {
    stdio: 'inherit',
    shell: true
  });

  gateway.on('error', (error) => {
    console.error('❌ Failed to start Gateway:', error);
    process.exit(1);
  });

  gateway.on('exit', (code) => {
    console.log(`\n🛑 Gateway stopped (exit code: ${code})`);
    if (rawStream && rawStreamPath) {
      console.log(`📄 Raw stream log: ${rawStreamPath}`);
    }
    process.exit(code || 0);
  });

  process.on('SIGINT', () => {
    console.log('\n🛑 Stopping Gateway...');
    gateway.kill('SIGINT');
  });

  return gateway;
}

// 使用示例
if (require.main === module) {
  startDebugGateway({
    port: 18789,
    rawStream: true,
    rawStreamPath: `./debug/stream-${Date.now()}.jsonl`,
    claudeCliLogs: false
  });
}

export { startDebugGateway };
```

---

## 场景 4：多终端开发工作流

### 原理

使用多个终端窗口分别运行 Gateway、测试命令和日志监控，提高开发效率。

### 实现

```typescript
// scripts/multi-terminal-dev.ts
/**
 * 多终端开发工作流脚本
 * 自动打开多个终端窗口
 */

import { spawn } from 'child_process';
import * as os from 'os';

interface TerminalCommand {
  title: string;
  command: string;
}

/**
 * 在新终端窗口中执行命令
 */
function openTerminal(cmd: TerminalCommand) {
  const platform = os.platform();

  if (platform === 'darwin') {
    // macOS: 使用 Terminal.app
    const script = `
      tell application "Terminal"
        do script "${cmd.command}"
        set custom title of front window to "${cmd.title}"
        activate
      end tell
    `;
    spawn('osascript', ['-e', script]);
  } else if (platform === 'linux') {
    // Linux: 使用 gnome-terminal 或 xterm
    spawn('gnome-terminal', [
      '--title', cmd.title,
      '--', 'bash', '-c', `${cmd.command}; exec bash`
    ]);
  } else {
    console.log(`Platform ${platform} not supported for multi-terminal`);
    console.log(`Please run manually: ${cmd.command}`);
  }
}

/**
 * 启动多终端开发环境
 */
function startMultiTerminalDev() {
  console.log('🚀 Starting multi-terminal development environment...\n');

  // 终端 1: Gateway
  console.log('📟 Terminal 1: Gateway (verbose mode)');
  openTerminal({
    title: 'OpenClaw Gateway',
    command: 'openclaw gateway --verbose --dev'
  });

  // 等待 2 秒让 Gateway 启动
  setTimeout(() => {
    // 终端 2: 测试命令
    console.log('📟 Terminal 2: Test Commands');
    openTerminal({
      title: 'OpenClaw Test',
      command: 'echo "Test commands ready. Try: openclaw agent --message \\"Hello\\""'
    });

    // 终端 3: 日志监控
    console.log('📟 Terminal 3: Log Monitor');
    openTerminal({
      title: 'OpenClaw Logs',
      command: 'openclaw logs --follow'
    });

    console.log('\n✅ Multi-terminal environment started!');
    console.log('   Terminal 1: Gateway running');
    console.log('   Terminal 2: Test commands');
    console.log('   Terminal 3: Log monitoring');
  }, 2000);
}

// 使用示例
if (require.main === module) {
  startMultiTerminalDev();
}

export { startMultiTerminalDev, openTerminal };
```

---

## 最佳实践

### 1. 开发环境使用详细日志

```bash
openclaw gateway --verbose --dev
```

### 2. 调试问题时启用完整 WebSocket 日志

```bash
openclaw gateway --ws-log full --raw-stream
```

### 3. 使用开发模式快速重置

```bash
openclaw gateway --dev --reset
```

### 4. 多终端提高效率

- Terminal 1: Gateway 前台运行
- Terminal 2: 测试命令
- Terminal 3: 日志监控

### 5. 快速迭代流程

```bash
# 1. 启动 Gateway
openclaw gateway --verbose --dev

# 2. 测试功能
openclaw agent --message "Test"

# 3. 停止 (Ctrl+C)

# 4. 修改代码

# 5. 重新启动 (Up arrow + Enter)
```

---

## 故障排查

### 问题 1：端口已被占用

```bash
# 检查端口
lsof -i :18789

# 强制释放
openclaw gateway --force --verbose
```

### 问题 2：配置验证错误

```bash
# 运行诊断
openclaw doctor

# 使用 dev 模式跳过配置检查
openclaw gateway --dev --verbose
```

### 问题 3：日志不够详细

```bash
# 启用所有调试选项
openclaw gateway --verbose --ws-log full --raw-stream
```

---

## 总结

前台启动 Gateway 是开发调试的核心技能：

- **基础启动**：`openclaw gateway --verbose`
- **开发模式**：`openclaw gateway --dev`
- **调试模式**：`openclaw gateway --ws-log full --raw-stream`
- **多终端工作流**：提高开发效率

掌握这些技巧可以显著提升 OpenClaw 开发体验。
