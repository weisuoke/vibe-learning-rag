# 03_核心概念_01_gateway命令

## 命令概述

`openclaw gateway` 是 OpenClaw 的核心命令，用于启动和管理 Gateway 控制平面。Gateway 是整个系统的中枢，负责消息路由、Agent 集成、Session 管理和扩展加载。

---

## 命令语法

```bash
openclaw gateway [subcommand] [options]
```

---

## 主命令：启动 Gateway

### 基础用法

```bash
# 启动 Gateway（前台运行）
openclaw gateway

# 指定端口
openclaw gateway --port 18789

# 详细日志
openclaw gateway --verbose

# 开发模式
openclaw gateway --dev
```

### 常用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--port <number>` | 监听端口 | 18789 |
| `--verbose` | 详细日志输出 | false |
| `--dev` | 开发模式（隔离状态） | false |
| `--profile <name>` | 使用指定配置文件 | default |

### 示例

```bash
# 生产环境
openclaw gateway --port 18789

# 开发环境（隔离状态）
openclaw gateway --dev --port 18790

# 使用自定义配置
openclaw gateway --profile staging
```

---

## 子命令

### 1. `gateway status` - 查看状态

**用途**：检查 Gateway 是否运行及其状态信息

**语法**：
```bash
openclaw gateway status [--json]
```

**输出示例**：
```
Gateway Status: Running
Port: 18789
Uptime: 5m 23s
Active Channels: 2
Model: claude-opus-4
PID: 12345
```

**JSON 输出**：
```bash
openclaw gateway status --json
```

```json
{
  "status": "running",
  "port": 18789,
  "uptime": 323,
  "activeChannels": 2,
  "model": "claude-opus-4",
  "pid": 12345
}
```

---

### 2. `gateway health` - 健康检查

**用途**：检查 Gateway 的健康状态

**语法**：
```bash
openclaw gateway health
```

**输出示例**：
```
Health Status: Healthy
Gateway: OK
Channels: 2/2 online
Agent: OK
Memory: 245 MB / 512 MB
```

**健康状态**：
- `Healthy`: 所有组件正常
- `Degraded`: 部分组件异常
- `Unhealthy`: 关键组件异常

---

### 3. `gateway start` - 启动守护进程

**用途**：启动 Gateway 守护进程（后台运行）

**语法**：
```bash
openclaw gateway start
```

**前提条件**：
- 已安装守护进程（`openclaw onboard --install-daemon`）

**输出示例**：
```
Starting Gateway daemon...
Gateway started successfully
PID: 12345
```

---

### 4. `gateway stop` - 停止守护进程

**用途**：停止 Gateway 守护进程

**语法**：
```bash
openclaw gateway stop
```

**输出示例**：
```
Stopping Gateway daemon...
Gateway stopped successfully
```

---

### 5. `gateway restart` - 重启守护进程

**用途**：重启 Gateway 守护进程

**语法**：
```bash
openclaw gateway restart
```

**输出示例**：
```
Restarting Gateway daemon...
Gateway restarted successfully
PID: 12346
```

---

### 6. `gateway run` - 前台运行

**用途**：在前台运行 Gateway（用于调试）

**语法**：
```bash
openclaw gateway run [options]
```

**与主命令的区别**：
- `openclaw gateway`: 默认行为（可能是前台或后台，取决于配置）
- `openclaw gateway run`: 强制前台运行

---

### 7. `gateway probe` - 探测 Gateway

**用途**：探测本地或远程 Gateway

**语法**：
```bash
openclaw gateway probe [url]
```

**示例**：
```bash
# 探测本地 Gateway
openclaw gateway probe

# 探测远程 Gateway
openclaw gateway probe http://remote-host:18789
```

---

### 8. `gateway discover` - 发现网络中的 Gateway

**用途**：发现局域网中的 Gateway 实例

**语法**：
```bash
openclaw gateway discover
```

**输出示例**：
```
Discovering Gateways on network...

Found 2 Gateways:
1. http://192.168.1.100:18789 (local)
2. http://192.168.1.101:18789 (remote)
```

---

## 使用场景

### 场景 1: 开发环境快速启动

```bash
# 启动开发环境 Gateway
openclaw gateway --dev --verbose

# 在另一个终端测试
openclaw --dev message send --to @test --message "Hello"
```

**优势**：
- 隔离开发环境（`~/.openclaw-dev/`）
- 详细日志便于调试
- 不影响生产配置

---

### 场景 2: 生产环境守护进程

```bash
# 安装守护进程
openclaw onboard --install-daemon

# 启动守护进程
openclaw gateway start

# 检查状态
openclaw gateway status

# 查看日志
tail -f ~/.openclaw/logs/gateway.log
```

**优势**：
- 后台运行，不占用终端
- 系统启动时自动启动
- 崩溃自动重启

---

### 场景 3: 多环境管理

```bash
# 开发环境
openclaw --profile dev gateway --port 18789

# 测试环境
openclaw --profile staging gateway --port 18790

# 生产环境
openclaw --profile production gateway --port 18791
```

**优势**：
- 多个环境隔离
- 不同配置独立管理
- 可以同时运行

---

### 场景 4: 健康监控

```bash
#!/bin/bash
# 健康监控脚本

while true; do
  if ! openclaw gateway health > /dev/null 2>&1; then
    echo "Gateway unhealthy, restarting..."
    openclaw gateway restart
  fi
  sleep 60
done
```

---

## TypeScript 集成

### 示例 1: 启动 Gateway

```typescript
// src/gateway-manager.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class GatewayManager {
  async start(port = 18789): Promise<void> {
    console.log(`Starting Gateway on port ${port}...`);

    // 启动 Gateway（后台）
    exec(`openclaw gateway --port ${port} > /dev/null 2>&1 &`);

    // 等待启动
    await this.waitForReady(port);

    console.log('Gateway started successfully');
  }

  async stop(): Promise<void> {
    console.log('Stopping Gateway...');
    await execAsync('openclaw gateway stop');
    console.log('Gateway stopped');
  }

  async restart(): Promise<void> {
    console.log('Restarting Gateway...');
    await execAsync('openclaw gateway restart');
    console.log('Gateway restarted');
  }

  async getStatus(): Promise<GatewayStatus> {
    const { stdout } = await execAsync('openclaw gateway status --json');
    return JSON.parse(stdout);
  }

  async isHealthy(): Promise<boolean> {
    try {
      await execAsync('openclaw gateway health');
      return true;
    } catch {
      return false;
    }
  }

  private async waitForReady(port: number, timeout = 10000): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      try {
        const response = await fetch(`http://localhost:${port}/health`);
        if (response.ok) return;
      } catch {
        // Gateway 还未就绪
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    throw new Error('Gateway failed to start');
  }
}

interface GatewayStatus {
  status: 'running' | 'stopped';
  port: number;
  uptime: number;
  activeChannels: number;
  model: string;
  pid: number;
}
```

### 示例 2: 自动重启机制

```typescript
// src/gateway-monitor.ts
import { EventEmitter } from 'events';
import { GatewayManager } from './gateway-manager';

export class GatewayMonitor extends EventEmitter {
  private manager = new GatewayManager();
  private interval: NodeJS.Timeout | null = null;
  private restartCount = 0;
  private maxRestarts = 3;

  start(checkInterval = 30000): void {
    this.interval = setInterval(async () => {
      const healthy = await this.manager.isHealthy();

      if (!healthy) {
        this.emit('unhealthy');
        await this.handleUnhealthy();
      } else {
        this.emit('healthy');
        this.restartCount = 0; // 重置重启计数
      }
    }, checkInterval);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  private async handleUnhealthy(): Promise<void> {
    if (this.restartCount >= this.maxRestarts) {
      this.emit('max-restarts-reached');
      console.error('Max restarts reached, manual intervention required');
      return;
    }

    this.restartCount++;
    console.log(`Restarting Gateway (attempt ${this.restartCount}/${this.maxRestarts})...`);

    try {
      await this.manager.restart();
      this.emit('restarted');
    } catch (error) {
      this.emit('restart-failed', error);
      console.error('Failed to restart Gateway:', error);
    }
  }
}

// 使用示例
const monitor = new GatewayMonitor();

monitor.on('unhealthy', () => {
  console.log('⚠️  Gateway is unhealthy');
});

monitor.on('restarted', () => {
  console.log('✅ Gateway restarted successfully');
});

monitor.on('max-restarts-reached', () => {
  console.log('❌ Max restarts reached, sending alert...');
  // 发送告警通知
});

monitor.start();
```

---

## 常见问题

### Q1: Gateway 启动失败，端口被占用

**问题**：
```
Error: Port 18789 is already in use
```

**解决方案**：
```bash
# 检查端口占用
lsof -i :18789

# 停止占用进程
kill -9 <PID>

# 或使用其他端口
openclaw gateway --port 18790
```

---

### Q2: Gateway 崩溃后如何自动重启？

**解决方案 1: 使用守护进程**
```bash
openclaw onboard --install-daemon
openclaw gateway start
```

**解决方案 2: 使用 systemd（Linux）**
```ini
# /etc/systemd/system/openclaw-gateway.service
[Unit]
Description=OpenClaw Gateway
After=network.target

[Service]
Type=simple
User=your-user
ExecStart=/usr/local/bin/openclaw gateway --port 18789
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable openclaw-gateway
sudo systemctl start openclaw-gateway
```

---

### Q3: 如何查看 Gateway 日志？

**日志位置**：
```bash
~/.openclaw/logs/gateway.log
```

**查看日志**：
```bash
# 实时查看
tail -f ~/.openclaw/logs/gateway.log

# 查看最近 100 行
tail -n 100 ~/.openclaw/logs/gateway.log

# 搜索错误
grep ERROR ~/.openclaw/logs/gateway.log
```

---

## 最佳实践

### 1. 生产环境使用守护进程

```bash
# ✅ 推荐
openclaw gateway start

# ❌ 不推荐（占用终端）
openclaw gateway
```

### 2. 开发环境使用 --dev 标志

```bash
# ✅ 推荐（隔离环境）
openclaw gateway --dev

# ❌ 不推荐（污染生产配置）
openclaw gateway
```

### 3. 定期健康检查

```bash
# 添加到 crontab
*/5 * * * * openclaw gateway health || openclaw gateway restart
```

### 4. 监控日志大小

```bash
# 日志轮转配置
# /etc/logrotate.d/openclaw
~/.openclaw/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## 下一步

- 学习 **message send 命令** → `03_核心概念_02_message_send命令.md`
- 学习 **agent 命令** → `03_核心概念_03_agent命令.md`
- 学习 **channels 命令** → `03_核心概念_04_channels_status命令.md`
- 学习 **config 命令** → `03_核心概念_05_config命令.md`

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [CLI 官方文档](https://docs.openclaw.ai/cli)
- [Gateway 架构](https://docs.openclaw.ai/architecture/gateway)
