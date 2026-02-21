# 实战代码 4：Spawn Hooks 实现

完整可运行的 Spawn Hooks 示例，展示如何拦截和修改 bash 命令执行。

---

## 示例概述

本示例展示：
1. 环境变量注入
2. 命令包装和预处理
3. 安全命令过滤
4. 性能监控

**适用场景：**
- 开发环境配置
- 命令安全过滤
- 性能监控
- Docker 容器化执行

---

## 完整代码

```typescript
/**
 * Spawn Hooks 实现扩展
 *
 * 功能：
 * 1. 注入环境变量
 * 2. 命令包装和预处理
 * 3. 安全命令过滤
 * 4. 性能监控
 *
 * 文件：spawn-hooks-impl.ts
 */

import { ExtensionAPI } from '@mariozechner/pi-agent-core';
import fs from 'fs';
import path from 'path';

// ===== 1. 配置 =====

const CONFIG = {
  // 环境变量配置
  env: {
    // 项目根目录
    PROJECT_ROOT: process.cwd(),

    // Node 环境
    NODE_ENV: 'development',

    // 调试模式
    DEBUG: '*',

    // 日志级别
    LOG_LEVEL: 'debug',

    // 自定义 PATH
    customPaths: [
      path.join(process.cwd(), 'node_modules', '.bin'),
      path.join(process.cwd(), 'scripts')
    ]
  },

  // 安全配置
  security: {
    // 危险命令模式
    dangerousPatterns: [
      /rm\s+-rf\s+\//,           // rm -rf /
      /sudo\s+rm/,               // sudo rm
      /chmod\s+777/,             // chmod 777
      /curl.*\|\s*bash/,         // curl | bash
      />\/dev\/sd[a-z]/,         // 直接写入磁盘
      /dd\s+if=/,                // dd 命令
      /mkfs/,                    // 格式化文件系统
      /:\(\)\{.*\}:/             // Fork bomb
    ],

    // 允许的命令白名单（可选）
    allowedCommands: [
      'npm', 'node', 'git', 'ls', 'cat', 'grep',
      'find', 'echo', 'pwd', 'cd', 'mkdir', 'touch',
      'cp', 'mv', 'rm', 'chmod', 'chown'
    ],

    // 是否启用白名单
    enableWhitelist: false
  },

  // 性能监控
  performance: {
    // 是否启用性能监控
    enabled: true,

    // 慢命令阈值（毫秒）
    slowCommandThreshold: 5000
  },

  // 命令包装
  wrapper: {
    // 是否添加 time 前缀
    addTime: false,

    // 是否添加超时控制
    addTimeout: false,

    // 超时时间（秒）
    timeoutSeconds: 300
  }
};

// ===== 2. 状态管理 =====

// 命令执行统计
const commandStats = new Map<string, {
  count: number;
  totalDuration: number;
  avgDuration: number;
  maxDuration: number;
  minDuration: number;
}>();

// 命令执行日志
interface CommandLog {
  command: string;
  cwd: string;
  timestamp: number;
  blocked: boolean;
  reason?: string;
}

const commandLogs: CommandLog[] = [];

// ===== 3. 辅助函数 =====

/**
 * 检查命令是否危险
 */
function isDangerousCommand(command: string): { dangerous: boolean; reason?: string } {
  for (const pattern of CONFIG.security.dangerousPatterns) {
    if (pattern.test(command)) {
      return {
        dangerous: true,
        reason: `Matches dangerous pattern: ${pattern.source}`
      };
    }
  }

  return { dangerous: false };
}

/**
 * 检查命令是否在白名单中
 */
function isCommandAllowed(command: string): boolean {
  if (!CONFIG.security.enableWhitelist) {
    return true;
  }

  const commandName = command.trim().split(/\s+/)[0];

  return CONFIG.security.allowedCommands.includes(commandName);
}

/**
 * 包装命令
 */
function wrapCommand(command: string): string {
  let wrappedCommand = command;

  // 添加 time 前缀
  if (CONFIG.wrapper.addTime) {
    wrappedCommand = `time ${wrappedCommand}`;
  }

  // 添加超时控制
  if (CONFIG.wrapper.addTimeout) {
    wrappedCommand = `timeout ${CONFIG.wrapper.timeoutSeconds} ${wrappedCommand}`;
  }

  return wrappedCommand;
}

/**
 * 记录命令日志
 */
function logCommand(command: string, cwd: string, blocked: boolean, reason?: string) {
  commandLogs.push({
    command,
    cwd,
    timestamp: Date.now(),
    blocked,
    reason
  });

  // 限制日志大小
  if (commandLogs.length > 1000) {
    commandLogs.shift();
  }
}

/**
 * 更新命令统计
 */
function updateCommandStats(command: string, duration: number) {
  const commandName = command.trim().split(/\s+/)[0];

  if (!commandStats.has(commandName)) {
    commandStats.set(commandName, {
      count: 0,
      totalDuration: 0,
      avgDuration: 0,
      maxDuration: 0,
      minDuration: Infinity
    });
  }

  const stats = commandStats.get(commandName)!;

  stats.count++;
  stats.totalDuration += duration;
  stats.avgDuration = stats.totalDuration / stats.count;
  stats.maxDuration = Math.max(stats.maxDuration, duration);
  stats.minDuration = Math.min(stats.minDuration, duration);
}

/**
 * 格式化持续时间
 */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else {
    return `${(ms / 60000).toFixed(2)}min`;
  }
}

// ===== 4. Extension 主函数 =====

export default function spawnHooksImpl(pi: ExtensionAPI) {
  console.log('🔧 Spawn Hooks Implementation Extension loaded');

  // 命令开始时间映射
  const commandStartTimes = new Map<string, number>();

  // ===== 注册 Spawn Hook =====
  pi.registerSpawnHook((command: string, cwd: string, env: Record<string, string>) => {
    console.log('\n🔧 Spawn Hook: Processing command');
    console.log(`  Command: ${command}`);
    console.log(`  CWD: ${cwd}`);

    // ===== 1. 安全检查 =====
    const dangerCheck = isDangerousCommand(command);

    if (dangerCheck.dangerous) {
      console.error(`  ⚠️  BLOCKED: ${dangerCheck.reason}`);
      console.error(`  Command: ${command}`);

      // 记录日志
      logCommand(command, cwd, true, dangerCheck.reason);

      // 返回安全的替代命令
      return {
        command: `echo "⚠️  Command blocked: ${dangerCheck.reason}"`,
        cwd,
        env
      };
    }

    // 白名单检查
    if (!isCommandAllowed(command)) {
      const commandName = command.trim().split(/\s+/)[0];
      console.warn(`  ⚠️  Command not in whitelist: ${commandName}`);

      // 记录日志
      logCommand(command, cwd, true, 'Not in whitelist');

      // 可以选择阻止或允许
      // 这里选择允许但发出警告
    }

    // ===== 2. 环境变量注入 =====
    console.log('  Injecting environment variables...');

    // 注入项目根目录
    env.PROJECT_ROOT = CONFIG.env.PROJECT_ROOT;

    // 注入 Node 环境
    env.NODE_ENV = CONFIG.env.NODE_ENV;

    // 注入调试标志
    env.DEBUG = CONFIG.env.DEBUG;

    // 注入日志级别
    env.LOG_LEVEL = CONFIG.env.LOG_LEVEL;

    // 添加自定义路径到 PATH
    const customPaths = CONFIG.env.customPaths.filter(p => fs.existsSync(p));

    if (customPaths.length > 0) {
      const currentPath = env.PATH || '';
      env.PATH = [...customPaths, currentPath].join(':');
      console.log(`  ✓ Added ${customPaths.length} custom paths to PATH`);
    }

    // 加载 .env 文件（如果存在）
    const envFile = path.join(cwd, '.env');

    if (fs.existsSync(envFile)) {
      try {
        const envContent = fs.readFileSync(envFile, 'utf-8');
        const envVars = envContent
          .split('\n')
          .filter(line => line.trim() && !line.startsWith('#'))
          .reduce((acc, line) => {
            const [key, ...valueParts] = line.split('=');
            if (key && valueParts.length > 0) {
              acc[key.trim()] = valueParts.join('=').trim();
            }
            return acc;
          }, {} as Record<string, string>);

        Object.assign(env, envVars);
        console.log(`  ✓ Loaded ${Object.keys(envVars).length} variables from .env`);
      } catch (error) {
        console.error('  ✗ Failed to load .env file:', error);
      }
    }

    // ===== 3. 命令包装 =====
    let modifiedCommand = command;

    if (CONFIG.wrapper.addTime || CONFIG.wrapper.addTimeout) {
      modifiedCommand = wrapCommand(command);
      console.log(`  ✓ Wrapped command: ${modifiedCommand}`);
    }

    // ===== 4. 性能监控 =====
    if (CONFIG.performance.enabled) {
      // 生成命令 ID
      const commandId = `cmd-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

      // 记录开始时间
      commandStartTimes.set(commandId, Date.now());

      // 包装命令以记录结束时间
      modifiedCommand = `
        ${modifiedCommand}
        EXIT_CODE=$?
        echo "COMMAND_ID:${commandId}:EXIT_CODE:$EXIT_CODE"
        exit $EXIT_CODE
      `.trim();
    }

    // ===== 5. 记录日志 =====
    logCommand(command, cwd, false);

    console.log('  ✓ Spawn Hook processing complete');

    // 返回修改后的参数
    return {
      command: modifiedCommand,
      cwd,
      env
    };
  });

  // ===== 监听工具结果（用于性能监控） =====
  if (CONFIG.performance.enabled) {
    pi.on('tool_result', (tool, result) => {
      if (tool.name === 'bash') {
        // 解析命令 ID 和退出码
        const output = result.output || '';
        const match = output.match(/COMMAND_ID:(\w+-\d+-\w+):EXIT_CODE:(\d+)/);

        if (match) {
          const [, commandId, exitCode] = match;
          const startTime = commandStartTimes.get(commandId);

          if (startTime) {
            const duration = Date.now() - startTime;

            console.log('\n📊 Command Performance:');
            console.log(`  Duration: ${formatDuration(duration)}`);
            console.log(`  Exit code: ${exitCode}`);

            // 检查是否为慢命令
            if (duration > CONFIG.performance.slowCommandThreshold) {
              console.warn(`  ⚠️  Slow command detected (>${formatDuration(CONFIG.performance.slowCommandThreshold)})`);
            }

            // 更新统计
            updateCommandStats(result.args?.command || 'unknown', duration);

            // 清理
            commandStartTimes.delete(commandId);
          }
        }
      }
    });
  }

  // ===== 输出统计信息 =====
  pi.on('session_shutdown', () => {
    console.log('\n🔧 Spawn Hooks: Session Statistics');

    // 输出命令日志统计
    const blockedCommands = commandLogs.filter(log => log.blocked);

    console.log(`\n📋 Command Logs:`);
    console.log(`  Total commands: ${commandLogs.length}`);
    console.log(`  Blocked commands: ${blockedCommands.length}`);

    if (blockedCommands.length > 0) {
      console.log('\n  Blocked commands:');
      blockedCommands.forEach((log, index) => {
        console.log(`  ${index + 1}. ${log.command}`);
        console.log(`     Reason: ${log.reason}`);
      });
    }

    // 输出性能统计
    if (CONFIG.performance.enabled && commandStats.size > 0) {
      console.log('\n📊 Performance Statistics:');

      const sortedStats = Array.from(commandStats.entries())
        .sort((a, b) => b[1].count - a[1].count);

      sortedStats.forEach(([commandName, stats]) => {
        console.log(`\n  ${commandName}:`);
        console.log(`    Count: ${stats.count}`);
        console.log(`    Avg: ${formatDuration(stats.avgDuration)}`);
        console.log(`    Min: ${formatDuration(stats.minDuration)}`);
        console.log(`    Max: ${formatDuration(stats.maxDuration)}`);
      });
    }

    // 清理
    commandLogs.length = 0;
    commandStats.clear();
    commandStartTimes.clear();
  });
}
```

---

## 配置文件示例

### .env 文件

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://localhost:5432/mydb

# Application
APP_PORT=3000
APP_ENV=development
```

---

## 运行输出示例

```
🔧 Spawn Hooks Implementation Extension loaded

🔧 Spawn Hook: Processing command
  Command: npm test
  CWD: /path/to/project
  Injecting environment variables...
  ✓ Added 2 custom paths to PATH
  ✓ Loaded 5 variables from .env
  ✓ Spawn Hook processing complete

📊 Command Performance:
  Duration: 2.34s
  Exit code: 0

🔧 Spawn Hook: Processing command
  Command: git status
  CWD: /path/to/project
  Injecting environment variables...
  ✓ Added 2 custom paths to PATH
  ✓ Loaded 5 variables from .env
  ✓ Spawn Hook processing complete

📊 Command Performance:
  Duration: 0.15s
  Exit code: 0

🔧 Spawn Hook: Processing command
  Command: rm -rf /
  CWD: /path/to/project
  ⚠️  BLOCKED: Matches dangerous pattern: rm\s+-rf\s+\/
  Command: rm -rf /

🔧 Spawn Hooks: Session Statistics

📋 Command Logs:
  Total commands: 3
  Blocked commands: 1

  Blocked commands:
  1. rm -rf /
     Reason: Matches dangerous pattern: rm\s+-rf\s+\/

📊 Performance Statistics:

  npm:
    Count: 1
    Avg: 2.34s
    Min: 2.34s
    Max: 2.34s

  git:
    Count: 1
    Avg: 0.15s
    Min: 0.15s
    Max: 0.15s
```

---

## 代码说明

### 1. 安全检查

```typescript
const dangerCheck = isDangerousCommand(command);

if (dangerCheck.dangerous) {
  console.error(`BLOCKED: ${dangerCheck.reason}`);

  return {
    command: `echo "Command blocked"`,
    cwd,
    env
  };
}
```

**功能：** 检查命令是否匹配危险模式，如果是则阻止执行。

### 2. 环境变量注入

```typescript
env.PROJECT_ROOT = CONFIG.env.PROJECT_ROOT;
env.NODE_ENV = CONFIG.env.NODE_ENV;
env.DEBUG = CONFIG.env.DEBUG;

// 添加自定义路径
env.PATH = [...customPaths, env.PATH].join(':');

// 加载 .env 文件
const envVars = loadEnvFile(cwd);
Object.assign(env, envVars);
```

**功能：** 注入项目环境变量和加载 .env 文件。

### 3. 命令包装

```typescript
let modifiedCommand = command;

if (CONFIG.wrapper.addTime) {
  modifiedCommand = `time ${modifiedCommand}`;
}

if (CONFIG.wrapper.addTimeout) {
  modifiedCommand = `timeout 300 ${modifiedCommand}`;
}
```

**功能：** 为命令添加 time 前缀或超时控制。

### 4. 性能监控

```typescript
const commandId = generateId();
commandStartTimes.set(commandId, Date.now());

modifiedCommand = `
  ${modifiedCommand}
  EXIT_CODE=$?
  echo "COMMAND_ID:${commandId}:EXIT_CODE:$EXIT_CODE"
  exit $EXIT_CODE
`;
```

**功能：** 记录命令执行时间和退出码。

---

## 使用场景

### 场景 1：开发环境配置

```typescript
// 自动注入开发环境变量
env.NODE_ENV = 'development';
env.DEBUG = '*';
env.LOG_LEVEL = 'debug';
```

### 场景 2：安全命令过滤

```typescript
// 阻止危险命令
if (command.includes('rm -rf /')) {
  return { command: 'echo "Blocked"', cwd, env };
}
```

### 场景 3：性能监控

```typescript
// 记录命令执行时间
const duration = Date.now() - startTime;
console.log(`Duration: ${duration}ms`);
```

### 场景 4：Docker 容器化

```typescript
// 在 Docker 容器中执行命令
const dockerCommand = `docker run --rm -v ${cwd}:/workspace node:20 bash -c "${command}"`;
return { command: dockerCommand, cwd, env };
```

---

## 最佳实践

### 1. 始终返回所有参数

```typescript
// ✅ 推荐
return { command, cwd, env };

// ❌ 不推荐
return { env }; // 缺少 command 和 cwd
```

### 2. 避免修改原始对象

```typescript
// ✅ 推荐
const newEnv = { ...env, MY_VAR: 'value' };
return { command, cwd, env: newEnv };

// ⚠️ 可以工作但不推荐
env.MY_VAR = 'value';
return { command, cwd, env };
```

### 3. 处理命令中的特殊字符

```typescript
// ✅ 推荐
const escapedCommand = command.replace(/"/g, '\\"');
const wrappedCommand = `bash -c "${escapedCommand}"`;
```

### 4. 记录 Hook 执行

```typescript
// ✅ 推荐
console.log('[SpawnHook] Processing:', command);
console.log('[SpawnHook] Modified env:', Object.keys(env));
```

---

## 参考资料

**pi-mono 源码：**
- `packages/coding-agent/src/core/tools/bash.ts` - Spawn Hooks 实现
- `packages/coding-agent/examples/extensions/bash-spawn-hook.ts` - Spawn Hook 示例

---

**版本：** v1.0
**最后更新：** 2026-02-21
**维护者：** Claude Code
