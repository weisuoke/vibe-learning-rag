# 核心概念 4：Spawn Hooks

## 一句话定义

**Spawn Hooks 是拦截 bash 命令执行的特殊钩子，允许在命令执行前修改 command、cwd、env，实现命令预处理和环境注入。**

---

## 详细解释

### 什么是 Spawn Hooks？

Spawn Hooks 是 pi-mono 提供的一种特殊钩子，专门用于拦截和修改 bash 工具的命令执行。

**核心特点：**
- **命令拦截**：在 bash 命令执行前拦截
- **参数修改**：可以修改 command、cwd、env
- **环境注入**：可以注入环境变量
- **命令预处理**：可以修改或包装命令

**与普通事件钩子的区别：**

| 特性 | 普通事件钩子 | Spawn Hooks |
|------|------------|-------------|
| 触发时机 | 事件发生时 | bash 命令执行前 |
| 作用 | 监听和响应 | 拦截和修改 |
| 返回值 | 无需返回值 | 必须返回修改后的参数 |
| 用途 | 监控、日志、通知 | 命令预处理、环境注入 |

### spawnHook 接口

**接口定义：**
```typescript
interface SpawnHook {
  (command: string, cwd: string, env: Record<string, string>): {
    command: string;
    cwd: string;
    env: Record<string, string>;
  };
}
```

**参数：**
- `command`：要执行的 bash 命令
- `cwd`：当前工作目录
- `env`：环境变量对象

**返回值：**
- 必须返回包含 `command`、`cwd`、`env` 的对象
- 可以修改任意参数
- 不修改的参数也要返回

**注册方法：**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 修改参数
  env.MY_VAR = 'value';

  // 必须返回
  return { command, cwd, env };
});
```

### command 修改

**用途：** 修改或包装要执行的命令

**示例 1：添加命令前缀**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 为所有命令添加 time 前缀
  const modifiedCommand = `time ${command}`;

  return { command: modifiedCommand, cwd, env };
});
```

**示例 2：命令替换**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 将 npm 替换为 pnpm
  const modifiedCommand = command.replace(/^npm\s/, 'pnpm ');

  return { command: modifiedCommand, cwd, env };
});
```

**示例 3：命令包装**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 在 Docker 容器中执行命令
  const modifiedCommand = `docker run --rm -v ${cwd}:/workspace -w /workspace node:20 bash -c "${command}"`;

  return { command: modifiedCommand, cwd, env };
});
```

**示例 4：危险命令拦截**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 拦截危险命令
  if (command.includes('rm -rf /')) {
    console.error('⚠️  Dangerous command blocked!');
    return { command: 'echo "Command blocked"', cwd, env };
  }

  return { command, cwd, env };
});
```

### cwd 修改

**用途：** 修改命令执行的工作目录

**示例 1：切换到项目根目录**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 始终在项目根目录执行命令
  const projectRoot = '/path/to/project';

  return { command, cwd: projectRoot, env };
});
```

**示例 2：相对路径转换**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 如果命令包含相对路径，转换为绝对路径
  if (command.includes('./')) {
    const absoluteCwd = path.resolve(cwd);
    return { command, cwd: absoluteCwd, env };
  }

  return { command, cwd, env };
});
```

**示例 3：沙箱目录**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 在沙箱目录中执行命令
  const sandboxDir = '/tmp/sandbox';

  return { command, cwd: sandboxDir, env };
});
```

### env 修改

**用途：** 注入或修改环境变量

**示例 1：注入项目环境变量**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 注入项目环境变量
  env.PROJECT_ROOT = '/path/to/project';
  env.NODE_ENV = 'development';
  env.DEBUG = 'true';

  return { command, cwd, env };
});
```

**示例 2：API 密钥注入**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 从安全存储加载 API 密钥
  env.OPENAI_API_KEY = loadApiKey('openai');
  env.ANTHROPIC_API_KEY = loadApiKey('anthropic');

  return { command, cwd, env };
});
```

**示例 3：PATH 修改**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 添加自定义路径到 PATH
  const customPath = '/usr/local/custom/bin';
  env.PATH = `${customPath}:${env.PATH}`;

  return { command, cwd, env };
});
```

**示例 4：代理设置**
```typescript
pi.registerSpawnHook((command, cwd, env) => {
  // 设置代理
  env.HTTP_PROXY = 'http://proxy.example.com:8080';
  env.HTTPS_PROXY = 'http://proxy.example.com:8080';
  env.NO_PROXY = 'localhost,127.0.0.1';

  return { command, cwd, env };
});
```

### 实际应用场景

#### 场景 1：开发环境配置

```typescript
export default function devEnvironment(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 注入开发环境变量
    env.NODE_ENV = 'development';
    env.DEBUG = '*';
    env.LOG_LEVEL = 'debug';

    // 添加本地 node_modules/.bin 到 PATH
    const localBin = path.join(cwd, 'node_modules', '.bin');
    env.PATH = `${localBin}:${env.PATH}`;

    return { command, cwd, env };
  });
}
```

#### 场景 2：命令日志记录

```typescript
export default function commandLogger(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 记录命令执行
    console.log('=== Command Execution ===');
    console.log('Command:', command);
    console.log('CWD:', cwd);
    console.log('Timestamp:', new Date().toISOString());

    // 记录到文件
    logToFile({
      command,
      cwd,
      timestamp: Date.now()
    });

    return { command, cwd, env };
  });
}
```

#### 场景 3：Docker 容器化执行

```typescript
export default function dockerExecution(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 在 Docker 容器中执行命令
    const dockerCommand = `docker run --rm \
      -v ${cwd}:/workspace \
      -w /workspace \
      -e NODE_ENV=${env.NODE_ENV || 'development'} \
      node:20 \
      bash -c "${command.replace(/"/g, '\\"')}"`;

    return { command: dockerCommand, cwd, env };
  });
}
```

#### 场景 4：命令超时控制

```typescript
export default function commandTimeout(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 添加超时控制（使用 timeout 命令）
    const timeoutSeconds = 300; // 5 分钟
    const wrappedCommand = `timeout ${timeoutSeconds} ${command}`;

    return { command: wrappedCommand, cwd, env };
  });
}
```

#### 场景 5：命令重试机制

```typescript
export default function commandRetry(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 添加重试机制（使用 bash 循环）
    const maxRetries = 3;
    const retryCommand = `
      for i in {1..${maxRetries}}; do
        ${command} && break || {
          echo "Attempt $i failed, retrying..."
          sleep 2
        }
      done
    `;

    return { command: retryCommand, cwd, env };
  });
}
```

#### 场景 6：安全沙箱

```typescript
export default function securitySandbox(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 检查危险命令
    const dangerousPatterns = [
      /rm\s+-rf\s+\//,
      /sudo/,
      /chmod\s+777/,
      /curl.*\|\s*bash/
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(command)) {
        console.error('⚠️  Dangerous command blocked:', command);
        return { command: 'echo "Command blocked for security"', cwd, env };
      }
    }

    // 限制可执行的命令
    const allowedCommands = ['npm', 'node', 'git', 'ls', 'cat', 'grep'];
    const commandName = command.split(' ')[0];

    if (!allowedCommands.includes(commandName)) {
      console.warn('⚠️  Command not in whitelist:', commandName);
    }

    return { command, cwd, env };
  });
}
```

### 与其他 hooks 的区别

#### Spawn Hooks vs tool_call 事件

**tool_call 事件：**
- 监听所有工具调用（包括 bash、read、write 等）
- 只能观察，不能修改
- 在工具调用时触发

**Spawn Hooks：**
- 只拦截 bash 命令
- 可以修改命令、目录、环境变量
- 在命令执行前触发

**示例对比：**
```typescript
// tool_call 事件：只能观察
pi.on('tool_call', (tool, args) => {
  if (tool.name === 'bash') {
    console.log('Bash command:', args.command);
    // 无法修改命令
  }
});

// Spawn Hooks：可以修改
pi.registerSpawnHook((command, cwd, env) => {
  console.log('Bash command:', command);
  // 可以修改命令
  return { command: `echo "Modified: ${command}"`, cwd, env };
});
```

#### Spawn Hooks vs before_agent_start

**before_agent_start：**
- 在 Agent 启动前触发
- 可以修改消息列表
- 影响整个 Agent 运行

**Spawn Hooks：**
- 在每次 bash 命令执行前触发
- 可以修改命令参数
- 只影响单个命令执行

**使用场景：**
- `before_agent_start`：注入系统指令、项目上下文
- `Spawn Hooks`：命令预处理、环境注入

### 多个 Spawn Hooks

可以注册多个 Spawn Hooks，它们会按注册顺序执行：

```typescript
// Hook 1：注入环境变量
pi.registerSpawnHook((command, cwd, env) => {
  env.MY_VAR = 'value1';
  return { command, cwd, env };
});

// Hook 2：修改命令
pi.registerSpawnHook((command, cwd, env) => {
  const modifiedCommand = `time ${command}`;
  return { command: modifiedCommand, cwd, env };
});

// Hook 3：记录日志
pi.registerSpawnHook((command, cwd, env) => {
  console.log('Executing:', command);
  return { command, cwd, env };
});

// 执行顺序：Hook 1 → Hook 2 → Hook 3 → 实际执行
```

**注意事项：**
- 每个 Hook 接收的是前一个 Hook 的输出
- 最后一个 Hook 的输出是实际执行的参数
- 如果某个 Hook 抛出错误，后续 Hook 不会执行

---

## 代码示例

### 示例 1：基础环境注入

```typescript
import { ExtensionAPI } from '@mariozechner/pi-agent-core';

export default function basicSpawnHook(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    console.log('=== Spawn Hook ===');
    console.log('Command:', command);
    console.log('CWD:', cwd);

    // 注入环境变量
    env.PROJECT_ROOT = cwd;
    env.NODE_ENV = 'development';

    return { command, cwd, env };
  });
}
```

### 示例 2：命令包装器

```typescript
export default function commandWrapper(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 为所有 npm 命令添加 --silent 标志
    if (command.startsWith('npm ')) {
      const modifiedCommand = command.replace('npm ', 'npm --silent ');
      return { command: modifiedCommand, cwd, env };
    }

    // 为所有 git 命令添加颜色输出
    if (command.startsWith('git ')) {
      env.GIT_COLOR = 'always';
    }

    return { command, cwd, env };
  });
}
```

### 示例 3：完整的开发环境配置

```typescript
import path from 'path';

export default function devEnvironmentSetup(pi: ExtensionAPI) {
  pi.registerSpawnHook((command, cwd, env) => {
    // 1. 设置 Node.js 环境
    env.NODE_ENV = 'development';
    env.DEBUG = '*';

    // 2. 添加本地 node_modules/.bin 到 PATH
    const localBin = path.join(cwd, 'node_modules', '.bin');
    env.PATH = `${localBin}:${env.PATH}`;

    // 3. 设置项目根目录
    env.PROJECT_ROOT = cwd;

    // 4. 加载 .env 文件中的环境变量
    const envFile = path.join(cwd, '.env');
    if (fs.existsSync(envFile)) {
      const envVars = dotenv.parse(fs.readFileSync(envFile));
      Object.assign(env, envVars);
    }

    // 5. 记录命令执行
    console.log(`[${new Date().toISOString()}] Executing: ${command}`);

    return { command, cwd, env };
  });
}
```

### 示例 4：安全命令过滤

```typescript
export default function securityFilter(pi: ExtensionAPI) {
  const dangerousPatterns = [
    { pattern: /rm\s+-rf\s+\//, message: 'Recursive delete from root' },
    { pattern: /sudo/, message: 'Sudo command' },
    { pattern: /chmod\s+777/, message: 'Insecure permissions' },
    { pattern: /curl.*\|\s*bash/, message: 'Pipe to bash' },
    { pattern: />\/dev\/sd[a-z]/, message: 'Direct disk write' }
  ];

  pi.registerSpawnHook((command, cwd, env) => {
    // 检查危险模式
    for (const { pattern, message } of dangerousPatterns) {
      if (pattern.test(command)) {
        console.error(`⚠️  BLOCKED: ${message}`);
        console.error(`Command: ${command}`);

        // 返回安全的替代命令
        return {
          command: `echo "Command blocked: ${message}"`,
          cwd,
          env
        };
      }
    }

    // 命令安全，允许执行
    return { command, cwd, env };
  });
}
```

### 示例 5：命令性能监控

```typescript
export default function performanceMonitor(pi: ExtensionAPI) {
  const commandStats = new Map();

  pi.registerSpawnHook((command, cwd, env) => {
    const commandId = generateId();
    const startTime = Date.now();

    // 记录命令开始
    commandStats.set(commandId, {
      command,
      startTime,
      cwd
    });

    // 包装命令以记录结束时间
    const wrappedCommand = `
      ${command}
      EXIT_CODE=$?
      echo "COMMAND_ID:${commandId}:EXIT_CODE:$EXIT_CODE"
      exit $EXIT_CODE
    `;

    return { command: wrappedCommand, cwd, env };
  });

  // 监听命令结果
  pi.on('tool_result', (tool, result) => {
    if (tool.name === 'bash') {
      // 解析命令 ID 和退出码
      const match = result.output.match(/COMMAND_ID:(\w+):EXIT_CODE:(\d+)/);
      if (match) {
        const [, commandId, exitCode] = match;
        const stats = commandStats.get(commandId);

        if (stats) {
          const duration = Date.now() - stats.startTime;
          console.log('=== Command Stats ===');
          console.log('Command:', stats.command);
          console.log('Duration:', duration, 'ms');
          console.log('Exit code:', exitCode);

          commandStats.delete(commandId);
        }
      }
    }
  });
}
```

---

## 最佳实践

### 1. 始终返回所有参数

```typescript
// ✅ 推荐：返回所有参数
pi.registerSpawnHook((command, cwd, env) => {
  env.MY_VAR = 'value';
  return { command, cwd, env };
});

// ❌ 不推荐：只返回部分参数
pi.registerSpawnHook((command, cwd, env) => {
  env.MY_VAR = 'value';
  return { env }; // 缺少 command 和 cwd
});
```

### 2. 避免修改原始对象

```typescript
// ✅ 推荐：创建新对象
pi.registerSpawnHook((command, cwd, env) => {
  const newEnv = { ...env, MY_VAR: 'value' };
  return { command, cwd, env: newEnv };
});

// ❌ 不推荐：直接修改（虽然可以工作）
pi.registerSpawnHook((command, cwd, env) => {
  env.MY_VAR = 'value'; // 直接修改
  return { command, cwd, env };
});
```

### 3. 处理命令中的特殊字符

```typescript
// ✅ 推荐：正确转义
pi.registerSpawnHook((command, cwd, env) => {
  const escapedCommand = command.replace(/"/g, '\\"');
  const wrappedCommand = `bash -c "${escapedCommand}"`;
  return { command: wrappedCommand, cwd, env };
});
```

### 4. 记录 Hook 执行

```typescript
// ✅ 推荐：记录 Hook 执行
pi.registerSpawnHook((command, cwd, env) => {
  console.log('[SpawnHook] Processing command:', command);

  // 修改参数
  env.MY_VAR = 'value';

  console.log('[SpawnHook] Modified env:', Object.keys(env));

  return { command, cwd, env };
});
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
