# 实战代码 5：多 Gateway 实例管理

## 场景概述

多 Gateway 实例管理通过 profile 机制实现隔离，本节提供完整的 TypeScript 代码示例，展示如何在同一主机上部署和管理多个 Gateway 实例。

**参考资料：**
- [Multiple Gateways - OpenClaw](https://docs.openclaw.ai/gateway/multiple-gateways) - 多实例配置
- [OpenClaw CLI Gateway](https://docs.openclaw.ai/cli/gateway) - CLI 命令
- [OpenClaw Configuration](https://docs.openclaw.ai/gateway/configuration) - 配置管理

---

## 场景 1：开发与生产环境隔离

### 原理

使用不同的 profile 隔离开发和生产环境，每个 profile 有独立的配置、端口、凭证和工作空间。

### 实现

```typescript
// scripts/setup-dev-prod.ts
/**
 * 开发与生产环境隔离脚本
 * 用途：部署 default (生产) 和 dev (开发) 两个 Gateway 实例
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

interface ProfileConfig {
  profile: string;
  port: number;
  description: string;
  bind: 'loopback' | 'lan' | 'tailnet';
  authMode: 'token' | 'password';
}

const PROFILES: ProfileConfig[] = [
  {
    profile: 'default',
    port: 18789,
    description: 'Production Gateway',
    bind: 'loopback',
    authMode: 'token'
  },
  {
    profile: 'dev',
    port: 18790,
    description: 'Development Gateway',
    bind: 'loopback',
    authMode: 'token'
  }
];

/**
 * 部署多个 Gateway 实例
 */
async function setupDevProd(): Promise<void> {
  console.log('🚀 Setting up Development and Production Gateways...\n');

  for (const config of PROFILES) {
    console.log(`📦 Deploying ${config.description}...`);
    console.log(`   Profile: ${config.profile}`);
    console.log(`   Port: ${config.port}`);
    console.log(`   Bind: ${config.bind}`);
    console.log('');

    // 创建配置目录
    const configDir = config.profile === 'default'
      ? path.join(os.homedir(), '.openclaw')
      : path.join(os.homedir(), `.openclaw-${config.profile}`);

    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }

    // 生成配置文件
    const gatewayConfig = {
      gateway: {
        mode: 'local',
        port: config.port,
        bind: config.bind,
        auth: {
          mode: config.authMode,
          token: `\${OPENCLAW_${config.profile.toUpperCase()}_GATEWAY_TOKEN}`
        }
      },
      agents: {
        defaults: {
          workspace: path.join(configDir, 'workspace')
        }
      }
    };

    const configPath = path.join(configDir, 'openclaw.json');
    fs.writeFileSync(configPath, JSON.stringify(gatewayConfig, null, 2));
    console.log(`✅ Configuration written to: ${configPath}`);

    // 安装守护进程
    console.log(`🔧 Installing daemon for ${config.profile}...`);
    await installDaemon(config.profile);

    console.log(`✅ ${config.description} deployed\n`);
  }

  // 验证部署
  console.log('🔍 Verifying deployment...\n');
  await verifyDeployment();

  console.log('✅ All Gateways deployed successfully\n');
  console.log('📊 Gateway Summary:');
  PROFILES.forEach(config => {
    console.log(`   - ${config.description}: port ${config.port} (profile: ${config.profile})`);
  });
}

/**
 * 安装守护进程
 */
async function installDaemon(profile: string): Promise<void> {
  const args = ['onboard', '--install-daemon'];
  if (profile !== 'default') {
    args.push('--profile', profile);
  }

  return new Promise((resolve, reject) => {
    const install = spawn('openclaw', args, {
      stdio: 'inherit',
      shell: true
    });

    install.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Daemon installation failed for ${profile}`));
      }
    });
  });
}

/**
 * 验证部署
 */
async function verifyDeployment(): Promise<void> {
  for (const config of PROFILES) {
    const args = ['gateway', 'status'];
    if (config.profile !== 'default') {
      args.push('--profile', config.profile);
    }

    await new Promise((resolve) => {
      const status = spawn('openclaw', args, {
        stdio: 'pipe',
        shell: true
      });

      status.on('exit', (code) => {
        if (code === 0) {
          console.log(`   ✅ ${config.profile}: running on port ${config.port}`);
        } else {
          console.log(`   ❌ ${config.profile}: not running`);
        }
        resolve(null);
      });
    });
  }
}

// 使用示例
if (require.main === module) {
  setupDevProd().catch((error) => {
    console.error('Setup failed:', error);
    process.exit(1);
  });
}

export { setupDevProd };
```

---

## 场景 2：Gateway 生命周期管理

### 原理

提供统一的接口管理多个 Gateway 实例的启动、停止、重启和状态查询。

### 实现

```typescript
// scripts/gateway-manager.ts
/**
 * Gateway 生命周期管理器
 * 用途：统一管理多个 Gateway 实例
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface GatewayInstance {
  profile: string;
  port: number;
  description: string;
}

class GatewayManager {
  private instances: GatewayInstance[];

  constructor(instances: GatewayInstance[]) {
    this.instances = instances;
  }

  /**
   * 启动所有 Gateway 实例
   */
  async startAll(): Promise<void> {
    console.log('🚀 Starting all Gateways...\n');

    for (const instance of this.instances) {
      console.log(`Starting ${instance.description}...`);
      await this.start(instance.profile);
    }

    console.log('\n✅ All Gateways started');
  }

  /**
   * 停止所有 Gateway 实例
   */
  async stopAll(): Promise<void> {
    console.log('🛑 Stopping all Gateways...\n');

    for (const instance of this.instances) {
      console.log(`Stopping ${instance.description}...`);
      await this.stop(instance.profile);
    }

    console.log('\n✅ All Gateways stopped');
  }

  /**
   * 重启所有 Gateway 实例
   */
  async restartAll(): Promise<void> {
    console.log('🔄 Restarting all Gateways...\n');

    for (const instance of this.instances) {
      console.log(`Restarting ${instance.description}...`);
      await this.restart(instance.profile);
    }

    console.log('\n✅ All Gateways restarted');
  }

  /**
   * 查看所有 Gateway 状态
   */
  async statusAll(): Promise<void> {
    console.log('📊 Gateway Status:\n');

    for (const instance of this.instances) {
      const status = await this.getStatus(instance.profile);
      const statusIcon = status.running ? '✅' : '❌';
      const statusText = status.running ? 'running' : 'stopped';

      console.log(`${statusIcon} ${instance.description} (${instance.profile})`);
      console.log(`   Port: ${instance.port}`);
      console.log(`   Status: ${statusText}`);
      if (status.pid) {
        console.log(`   PID: ${status.pid}`);
      }
      console.log('');
    }
  }

  /**
   * 启动单个 Gateway
   */
  private async start(profile: string): Promise<void> {
    const args = ['gateway', 'start'];
    if (profile !== 'default') {
      args.push('--profile', profile);
    }

    return this.executeCommand(args);
  }

  /**
   * 停止单个 Gateway
   */
  private async stop(profile: string): Promise<void> {
    const args = ['gateway', 'stop'];
    if (profile !== 'default') {
      args.push('--profile', profile);
    }

    return this.executeCommand(args);
  }

  /**
   * 重启单个 Gateway
   */
  private async restart(profile: string): Promise<void> {
    const args = ['gateway', 'restart'];
    if (profile !== 'default') {
      args.push('--profile', profile);
    }

    return this.executeCommand(args);
  }

  /**
   * 获取单个 Gateway 状态
   */
  private async getStatus(profile: string): Promise<{ running: boolean; pid?: number }> {
    const args = ['gateway', 'status', '--json'];
    if (profile !== 'default') {
      args.push('--profile', profile);
    }

    try {
      const { stdout } = await execAsync(`openclaw ${args.join(' ')}`);
      const status = JSON.parse(stdout);

      return {
        running: status.service?.running || false,
        pid: status.service?.pid
      };
    } catch (error) {
      return { running: false };
    }
  }

  /**
   * 执行命令
   */
  private executeCommand(args: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
      const process = spawn('openclaw', args, {
        stdio: 'inherit',
        shell: true
      });

      process.on('exit', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Command failed with code ${code}`));
        }
      });
    });
  }
}

// 使用示例
if (require.main === module) {
  const instances: GatewayInstance[] = [
    { profile: 'default', port: 18789, description: 'Production Gateway' },
    { profile: 'dev', port: 18790, description: 'Development Gateway' },
    { profile: 'rescue', port: 18791, description: 'Rescue Gateway' }
  ];

  const manager = new GatewayManager(instances);

  (async () => {
    const command = process.argv[2];

    switch (command) {
      case 'start':
        await manager.startAll();
        break;
      case 'stop':
        await manager.stopAll();
        break;
      case 'restart':
        await manager.restartAll();
        break;
      case 'status':
        await manager.statusAll();
        break;
      default:
        console.log('Usage: ts-node gateway-manager.ts <start|stop|restart|status>');
        process.exit(1);
    }
  })().catch(console.error);
}

export { GatewayManager };
```

---

## 场景 3：端口冲突检测与解决

### 原理

在启动 Gateway 前检测端口占用情况，自动解决端口冲突。

### 实现

```typescript
// scripts/port-conflict-resolver.ts
/**
 * 端口冲突检测与解决脚本
 * 用途：检测端口占用，自动分配可用端口
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface PortStatus {
  port: number;
  inUse: boolean;
  process?: string;
  pid?: number;
}

/**
 * 检查端口是否被占用
 */
async function checkPort(port: number): Promise<PortStatus> {
  try {
    const { stdout } = await execAsync(`lsof -i :${port} -t`);
    const pid = parseInt(stdout.trim());

    // 获取进程信息
    const { stdout: processInfo } = await execAsync(`ps -p ${pid} -o comm=`);

    return {
      port,
      inUse: true,
      process: processInfo.trim(),
      pid
    };
  } catch (error) {
    return {
      port,
      inUse: false
    };
  }
}

/**
 * 检查多个端口
 */
async function checkPorts(ports: number[]): Promise<PortStatus[]> {
  console.log('🔍 Checking port availability...\n');

  const results: PortStatus[] = [];

  for (const port of ports) {
    const status = await checkPort(port);
    results.push(status);

    if (status.inUse) {
      console.log(`❌ Port ${port}: in use by ${status.process} (PID: ${status.pid})`);
    } else {
      console.log(`✅ Port ${port}: available`);
    }
  }

  console.log('');
  return results;
}

/**
 * 查找可用端口
 */
async function findAvailablePort(startPort: number, count: number = 1): Promise<number[]> {
  const availablePorts: number[] = [];
  let currentPort = startPort;

  while (availablePorts.length < count) {
    const status = await checkPort(currentPort);
    if (!status.inUse) {
      availablePorts.push(currentPort);
    }
    currentPort++;
  }

  return availablePorts;
}

/**
 * 解决端口冲突
 */
async function resolvePortConflicts(
  desiredPorts: number[]
): Promise<Map<number, number>> {
  console.log('🔧 Resolving port conflicts...\n');

  const portMapping = new Map<number, number>();
  const statuses = await checkPorts(desiredPorts);

  for (const status of statuses) {
    if (status.inUse) {
      console.log(`⚠️  Port ${status.port} is in use, finding alternative...`);
      const [alternativePort] = await findAvailablePort(status.port + 1);
      portMapping.set(status.port, alternativePort);
      console.log(`✅ Alternative port found: ${alternativePort}\n`);
    } else {
      portMapping.set(status.port, status.port);
    }
  }

  return portMapping;
}

// 使用示例
if (require.main === module) {
  const desiredPorts = [18789, 18790, 18791];

  (async () => {
    const portMapping = await resolvePortConflicts(desiredPorts);

    console.log('📊 Port Mapping:');
    portMapping.forEach((actualPort, desiredPort) => {
      if (actualPort === desiredPort) {
        console.log(`   ${desiredPort} → ${actualPort} (no change)`);
      } else {
        console.log(`   ${desiredPort} → ${actualPort} (conflict resolved)`);
      }
    });
  })().catch(console.error);
}

export { checkPort, checkPorts, findAvailablePort, resolvePortConflicts };
```

---

## 最佳实践

### 1. Profile 命名规范

```typescript
// 推荐的 profile 命名
const profiles = {
  production: 'default',      // 生产环境
  development: 'dev',         // 开发环境
  staging: 'staging',         // 预发布环境
  rescue: 'rescue',           // 救援实例
  testing: 'test'             // 测试环境
};
```

### 2. 端口分配策略

```typescript
// 端口分配规则
const portAllocation = {
  default: 18789,    // 生产环境
  dev: 18790,        // 开发环境
  staging: 18791,    // 预发布环境
  rescue: 18792,     // 救援实例
  test: 18793        // 测试环境
};
```

### 3. 环境变量隔离

```bash
# ~/.openclaw/.env (生产环境)
OPENCLAW_GATEWAY_TOKEN=production-token
ANTHROPIC_API_KEY=sk-ant-production

# ~/.openclaw-dev/.env (开发环境)
OPENCLAW_GATEWAY_TOKEN=dev-token
ANTHROPIC_API_KEY=sk-ant-dev
```

---

## 故障排查

### 问题 1：端口冲突

```bash
# 检查所有 Gateway 端口
openclaw gateway probe

# 查看端口占用
lsof -i :18789
lsof -i :18790
```

### 问题 2：Profile 未找到

```bash
# 列出所有 profile
ls -la ~/.openclaw*

# 初始化新 profile
openclaw onboard --profile <profile-name>
```

### 问题 3：守护进程冲突

```bash
# macOS: 查看所有 Gateway 守护进程
launchctl list | grep openclaw

# Linux: 查看所有 Gateway 守护进程
systemctl --user list-units | grep openclaw
```

---

## 总结

多 Gateway 实例管理是生产环境的核心能力：

- **Profile 隔离**：独立的配置、端口、凭证
- **生命周期管理**：统一的启动/停止/重启接口
- **端口冲突解决**：自动检测和分配可用端口
- **最佳实践**：命名规范、端口分配、环境变量隔离

掌握这些技巧可以在同一主机上高效管理多个 Gateway 实例。
