# 核心概念 3：Gateway 配置机制

**本文档深入讲解 Gateway 的配置机制，包括端口、绑定、认证、Tailscale 等核心配置选项。**

---

## 概述

Gateway 是 OpenClaw 的核心组件，负责：
1. **接收请求**：接收来自 CLI、WebUI、Channels 的请求
2. **路由转发**：将请求路由到对应的 Agent
3. **认证授权**：验证请求的合法性
4. **会话管理**：管理对话会话和状态

Gateway 配置决定了如何访问 OpenClaw，是安全和可用性的关键。

---

## Gateway 配置结构

### 配置文件位置

```bash
~/.openclaw/openclaw.json
```

### 完整配置示例

```json5
{
  gateway: {
    // 基础配置
    port: 18789,
    bind: "loopback",  // loopback | lan | auto | custom | tailnet
    customBindHost: "192.168.1.100",  // 仅当 bind=custom 时使用

    // 认证配置
    auth: {
      mode: "token",  // token | password
      token: "oc_1a2b3c4d5e6f7g8h9i0j",
      password: "strong-password-here",
    },

    // Tailscale 配置
    tailscale: {
      mode: "off",  // off | serve | funnel
      resetOnExit: false,
    },

    // 热重载配置
    reload: {
      mode: "hybrid",  // hybrid | hot | restart | off
      debounceMs: 300,
    },

    // 远程 Gateway 配置
    remote: {
      url: "ws://remote-host:18789",
      token: "oc_remote_token",
    },
  },
}
```

---

## 配置选项详解

### 1. Port（端口）

**作用**：Gateway 监听的端口号

**默认值**：`18789`

**有效范围**：`1-65535`

**配置示例**：

```json5
{
  gateway: {
    port: 18789,
  },
}
```

**验证逻辑**：

```typescript
function validatePort(value: string): string | undefined {
  const port = Number(value);
  if (!Number.isFinite(port)) {
    return "Invalid port number";
  }
  if (port < 1 || port > 65535) {
    return "Port must be between 1 and 65535";
  }
  return undefined;
}
```

**常见端口选择**：

| 端口 | 用途 | 冲突风险 |
|------|------|---------|
| **18789** | 默认端口 | 低 |
| **8080** | 常用开发端口 | 高（可能被其他服务占用） |
| **3000** | Node.js 常用端口 | 高 |
| **自定义** | 避免冲突 | 低 |

**端口冲突检测**：

```bash
# 检查端口占用
lsof -i :18789

# 如果被占用，杀死进程或使用其他端口
kill -9 <PID>
```

---

### 2. Bind（绑定地址）

**作用**：Gateway 绑定的网络接口

**可选值**：

| 值 | 含义 | 访问范围 | 安全性 |
|---|------|---------|-------|
| **loopback** | 127.0.0.1 | 仅本机 | 高 |
| **lan** | 0.0.0.0 | 局域网 | 中 |
| **auto** | 自动选择（Loopback → LAN） | 动态 | 中 |
| **custom** | 自定义 IP | 指定网卡 | 取决于配置 |
| **tailnet** | Tailscale IP | Tailnet | 高 |

**配置示例**：

```json5
// Loopback（推荐）
{
  gateway: {
    bind: "loopback",  // 127.0.0.1
  },
}

// LAN（局域网访问）
{
  gateway: {
    bind: "lan",  // 0.0.0.0
  },
}

// Custom（自定义 IP）
{
  gateway: {
    bind: "custom",
    customBindHost: "192.168.1.100",
  },
}

// Tailnet（Tailscale）
{
  gateway: {
    bind: "tailnet",  // Tailscale IP
  },
}
```

**实现逻辑**：

```typescript
function resolveBindAddress(config: GatewayConfig): string {
  switch (config.bind) {
    case "loopback":
      return "127.0.0.1";
    case "lan":
      return "0.0.0.0";
    case "auto":
      // 尝试 Loopback，失败则 LAN
      return isLoopbackAvailable() ? "127.0.0.1" : "0.0.0.0";
    case "custom":
      return config.customBindHost ?? "127.0.0.1";
    case "tailnet":
      return getTailscaleIP();
    default:
      return "127.0.0.1";
  }
}
```

**安全建议**：

1. **本地开发**：使用 `loopback`（最安全）
2. **局域网访问**：使用 `lan` + Token 认证
3. **远程访问**：使用 Tailscale（推荐）或 VPN
4. **公网暴露**：使用 Tailscale Funnel + Password 认证

---

### 3. Auth（认证）

#### 3.1 Token 认证

**特点**：
- 长期有效的随机 Token
- 适合自动化和多设备访问
- 可撤销和重新生成

**配置示例**：

```json5
{
  gateway: {
    auth: {
      mode: "token",
      token: "oc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p",
    },
  },
}
```

**Token 生成**：

```typescript
function randomToken(): string {
  const bytes = crypto.randomBytes(16);
  return `oc_${bytes.toString('hex')}`;
}

// 示例输出：oc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
```

**Token 格式**：
- 前缀：`oc_`
- 长度：32 个十六进制字符（16 字节）
- 熵：128 位（足够安全）

**Token 验证**：

```typescript
function normalizeGatewayTokenInput(input: string): string | undefined {
  if (typeof input !== "string") return undefined;
  const trimmed = input.trim();
  if (trimmed === "") return undefined;

  // 验证格式：oc_ + 32 个十六进制字符
  if (!/^oc_[0-9a-f]{32}$/.test(trimmed)) {
    throw new Error("Invalid token format. Expected: oc_<32 hex chars>");
  }

  return trimmed;
}
```

**使用方式**：

```bash
# HTTP Header
Authorization: Bearer oc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p

# 环境变量
export OPENCLAW_GATEWAY_TOKEN="oc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p"

# CLI 参数
openclaw chat --token "oc_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p"
```

#### 3.2 Password 认证

**特点**：
- 用户友好（易于记忆）
- 适合临时访问和公网暴露
- 使用 HTTP Basic Auth

**配置示例**：

```json5
{
  gateway: {
    auth: {
      mode: "password",
      password: "strong-password-here",
    },
  },
}
```

**密码验证**：

```typescript
function validateGatewayPasswordInput(value: string): string | undefined {
  if (!value || value.trim() === "") {
    return "Password is required";
  }
  if (value.length < 8) {
    return "Password must be at least 8 characters";
  }
  return undefined;
}
```

**使用方式**：

```bash
# HTTP Basic Auth
Authorization: Basic base64(username:password)

# CLI 参数
openclaw chat --password "strong-password-here"

# 浏览器提示
# 访问 http://127.0.0.1:18789 时会弹出认证对话框
```

**安全建议**：

1. **密码强度**：至少 8 个字符，包含大小写字母、数字、特殊字符
2. **定期轮换**：每 90 天更换一次密码
3. **避免泄露**：不要在 URL 中传递密码
4. **HTTPS**：公网暴露时必须使用 HTTPS（Tailscale Funnel 自动提供）

---

### 4. Tailscale 配置

#### 4.1 Tailscale 模式

| 模式 | 含义 | 访问范围 | 适用场景 |
|------|------|---------|---------|
| **off** | 关闭 | 无 | 本地开发 |
| **serve** | 私有网络 | Tailnet 内部 | 团队协作 |
| **funnel** | 公网暴露 | 全球互联网 | 公开服务 |

#### 4.2 Serve 模式（私有网络）

**配置示例**：

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",  // 强制 Loopback
    auth: {
      mode: "token",  // Token 或 Password
      token: "oc_1a2b3c4d5e6f7g8h9i0j",
    },
    tailscale: {
      mode: "serve",
      resetOnExit: false,
    },
  },
}
```

**工作原理**：

```
Tailscale Serve（反向代理）
    ↓
Gateway (127.0.0.1:18789)
    ↓
Agent
```

**访问方式**：

```bash
# Tailnet 内部访问
https://<machine-name>.tailnet-name.ts.net

# 示例
https://openclaw-server.example.ts.net
```

**安全特性**：
- 仅 Tailnet 成员可访问
- 自动 HTTPS（Tailscale 提供证书）
- 无需公网 IP

#### 4.3 Funnel 模式（公网暴露）

**配置示例**：

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",  // 强制 Loopback
    auth: {
      mode: "password",  // 强制 Password
      password: "strong-password-here",
    },
    tailscale: {
      mode: "funnel",
      resetOnExit: true,  // 推荐：退出时清理
    },
  },
}
```

**工作原理**：

```
Internet（全球）
    ↓
Tailscale Funnel（公网入口）
    ↓
Gateway (127.0.0.1:18789)
    ↓
Agent
```

**访问方式**：

```bash
# 公网访问
https://<machine-name>.tailnet-name.ts.net

# 任何人都可以访问（需要密码认证）
```

**安全约束**：

1. **强制 Loopback**：防止同时暴露 LAN + 公网
2. **强制 Password**：Token 可能泄露到 URL
3. **推荐 resetOnExit**：退出时自动清理 Funnel

**实现逻辑**：

```typescript
// Tailscale 强制 Loopback
if (tailscaleMode !== "off" && bind !== "loopback") {
  await prompter.note(
    "Tailscale requires bind=loopback. Adjusting bind to loopback.",
    "Note"
  );
  bind = "loopback";
  customBindHost = undefined;
}

// Funnel 强制 Password
if (tailscaleMode === "funnel" && authMode !== "password") {
  await prompter.note("Tailscale funnel requires password auth.", "Note");
  authMode = "password";
}
```

---

## 配置约束和验证

### 约束规则

| 约束 | 原因 | 实现 |
|------|------|------|
| **Tailscale → Loopback** | 防止同时暴露 LAN + Tailnet | 自动调整 bind |
| **Funnel → Password** | Token 可能泄露到 URL | 自动切换 authMode |
| **Port 范围** | 系统限制 | 验证 1-65535 |
| **Custom Bind → IPv4** | 仅支持 IPv4 | 验证 IP 格式 |

### 验证实现

```typescript
export async function configureGatewayForOnboarding(
  opts: ConfigureGatewayOptions,
): Promise<ConfigureGatewayResult> {
  const { flow, localPort, quickstartGateway, prompter } = opts;
  let { nextConfig } = opts;

  // 1. 端口配置
  const port = flow === "quickstart"
    ? quickstartGateway.port
    : Number.parseInt(await prompter.text({
        message: "Gateway port",
        initialValue: String(localPort),
        validate: (value) => Number.isFinite(Number(value)) ? undefined : "Invalid port",
      }), 10);

  // 2. 绑定地址
  let bind = flow === "quickstart"
    ? quickstartGateway.bind
    : await prompter.select({
        message: "Gateway bind",
        options: [
          { value: "loopback", label: "Loopback (127.0.0.1)" },
          { value: "lan", label: "LAN (0.0.0.0)" },
          { value: "tailnet", label: "Tailnet (Tailscale IP)" },
          { value: "auto", label: "Auto (Loopback → LAN)" },
          { value: "custom", label: "Custom IP" },
        ],
      });

  // 3. 自定义 IP（如果需要）
  let customBindHost = quickstartGateway.customBindHost;
  if (bind === "custom") {
    const needsPrompt = flow !== "quickstart" || !customBindHost;
    if (needsPrompt) {
      const input = await prompter.text({
        message: "Custom IP address",
        placeholder: "192.168.1.100",
        initialValue: customBindHost ?? "",
        validate: validateIPv4AddressInput,
      });
      customBindHost = typeof input === "string" ? input.trim() : undefined;
    }
  }

  // 4. 认证方式
  let authMode = flow === "quickstart"
    ? quickstartGateway.authMode
    : await prompter.select({
        message: "Gateway auth",
        options: [
          { value: "token", label: "Token", hint: "Recommended default" },
          { value: "password", label: "Password" },
        ],
        initialValue: "token",
      });

  // 5. Tailscale 配置
  const tailscaleMode = flow === "quickstart"
    ? quickstartGateway.tailscaleMode
    : await prompter.select({
        message: "Tailscale exposure",
        options: [
          { value: "off", label: "Off" },
          { value: "serve", label: "Serve (private Tailnet)" },
          { value: "funnel", label: "Funnel (public internet)" },
        ],
      });

  // 6. 约束检查：Tailscale → Loopback
  if (tailscaleMode !== "off" && bind !== "loopback") {
    await prompter.note("Tailscale requires bind=loopback. Adjusting bind to loopback.", "Note");
    bind = "loopback";
    customBindHost = undefined;
  }

  // 7. 约束检查：Funnel → Password
  if (tailscaleMode === "funnel" && authMode !== "password") {
    await prompter.note("Tailscale funnel requires password auth.", "Note");
    authMode = "password";
  }

  // 8. Token 生成
  let gatewayToken: string | undefined;
  if (authMode === "token") {
    if (flow === "quickstart") {
      gatewayToken = quickstartGateway.token ?? randomToken();
    } else {
      const tokenInput = await prompter.text({
        message: "Gateway token (blank to generate)",
        placeholder: "Needed for multi-machine or non-loopback access",
        initialValue: quickstartGateway.token ?? "",
      });
      gatewayToken = normalizeGatewayTokenInput(tokenInput) || randomToken();
    }
  }

  // 9. Password 配置
  if (authMode === "password") {
    const password = flow === "quickstart" && quickstartGateway.password
      ? quickstartGateway.password
      : await prompter.text({
          message: "Gateway password",
          validate: validateGatewayPasswordInput,
        });
    nextConfig = {
      ...nextConfig,
      gateway: {
        ...nextConfig.gateway,
        auth: {
          ...nextConfig.gateway?.auth,
          mode: "password",
          password: String(password ?? "").trim(),
        },
      },
    };
  } else if (authMode === "token") {
    nextConfig = {
      ...nextConfig,
      gateway: {
        ...nextConfig.gateway,
        auth: {
          ...nextConfig.gateway?.auth,
          mode: "token",
          token: gatewayToken,
        },
      },
    };
  }

  // 10. 返回配置
  return {
    nextConfig,
    settings: {
      port,
      bind,
      customBindHost,
      authMode,
      gatewayToken,
      tailscaleMode,
      tailscaleResetOnExit,
    },
  };
}
```

---

## 配置热重载

### 热重载模式

```json5
{
  gateway: {
    reload: {
      mode: "hybrid",  // hybrid | hot | restart | off
      debounceMs: 300,
    },
  },
}
```

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| **hybrid** | 热重载安全配置，自动重启关键配置 | 开发环境（推荐） |
| **hot** | 仅热重载安全配置，关键配置需手动重启 | 生产环境 |
| **restart** | 任何配置变更都重启 Gateway | 调试环境 |
| **off** | 禁用文件监听，需手动重启 | 容器环境 |

### 热重载支持

| 配置类别 | 热重载 | 需要重启 |
|---------|-------|---------|
| Channels | ✅ | ❌ |
| Agent & Models | ✅ | ❌ |
| Automation (hooks, cron) | ✅ | ❌ |
| Sessions & Messages | ✅ | ❌ |
| Tools & Media | ✅ | ❌ |
| **Gateway Server** (port, bind, auth) | ❌ | ✅ |
| **Infrastructure** (discovery, plugins) | ❌ | ✅ |

### 实现逻辑

```typescript
async function watchConfigFile() {
  const watcher = fs.watch(configPath, { persistent: true });

  for await (const event of watcher) {
    if (event.eventType === "change") {
      await debounce(async () => {
        const newConfig = await readConfigFile();
        const changes = detectChanges(currentConfig, newConfig);

        if (needsRestart(changes)) {
          if (reloadMode === "hybrid") {
            logger.info("Config change requires restart. Restarting Gateway...");
            await restartGateway();
          } else if (reloadMode === "hot") {
            logger.warn("Config change requires restart. Run: openclaw gateway restart");
          }
        } else {
          logger.info("Hot-reloading config changes...");
          await applyHotReload(changes);
        }

        currentConfig = newConfig;
      }, reloadConfig.debounceMs);
    }
  }
}

function needsRestart(changes: ConfigChanges): boolean {
  const restartRequired = [
    "gateway.port",
    "gateway.bind",
    "gateway.auth",
    "gateway.tailscale",
    "discovery",
    "plugins",
  ];

  return changes.some(change =>
    restartRequired.some(path => change.path.startsWith(path))
  );
}
```

---

## 远程 Gateway 配置

### 配置示例

```json5
{
  gateway: {
    remote: {
      url: "ws://remote-host:18789",
      token: "oc_remote_token",
    },
  },
}
```

### 使用场景

1. **多机部署**：CLI 在本机，Gateway 在服务器
2. **团队协作**：多个用户共享一个 Gateway
3. **资源隔离**：Gateway 和 Agent 分离部署

### 连接验证

```typescript
async function probeGatewayReachable(params: {
  url: string;
  token?: string;
  password?: string;
}): Promise<{ ok: boolean; error?: string }> {
  try {
    const response = await fetch(`${params.url}/health`, {
      headers: {
        ...(params.token && { "Authorization": `Bearer ${params.token}` }),
        ...(params.password && { "Authorization": `Basic ${btoa(`:${params.password}`)}` }),
      },
    });

    if (response.ok) {
      return { ok: true };
    } else {
      return { ok: false, error: `HTTP ${response.status}` };
    }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
```

---

## 最佳实践

### 本地开发

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",
    auth: {
      mode: "token",
      token: "oc_dev_token",
    },
    tailscale: {
      mode: "off",
    },
  },
}
```

### 局域网访问

```json5
{
  gateway: {
    port: 18789,
    bind: "lan",
    auth: {
      mode: "token",
      token: "oc_lan_token",
    },
    tailscale: {
      mode: "off",
    },
  },
}
```

### 远程访问（Tailscale）

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",
    auth: {
      mode: "token",
      token: "oc_tailscale_token",
    },
    tailscale: {
      mode: "serve",
      resetOnExit: false,
    },
  },
}
```

### 公网暴露（Tailscale Funnel）

```json5
{
  gateway: {
    port: 18789,
    bind: "loopback",
    auth: {
      mode: "password",
      password: "strong-password-here",
    },
    tailscale: {
      mode: "funnel",
      resetOnExit: true,
    },
  },
}
```

---

## 安全建议

### 1. 最小权限原则

- **本地开发**：使用 Loopback
- **局域网访问**：使用 LAN + Token
- **远程访问**：使用 Tailscale Serve
- **公网暴露**：使用 Tailscale Funnel + 强密码

### 2. 认证强度

- **Token**：至少 128 位熵（16 字节）
- **Password**：至少 8 个字符，包含大小写字母、数字、特殊字符

### 3. 定期轮换

- **Token**：每 90 天轮换一次
- **Password**：每 90 天更换一次

### 4. 监控和审计

```bash
# 查看 Gateway 日志
openclaw logs gateway

# 查看访问记录
grep "Authorization" ~/.openclaw/logs/gateway.log

# 检测异常访问
grep "401\|403" ~/.openclaw/logs/gateway.log
```

---

## 故障排查

### 问题 1：端口被占用

```bash
# 检查端口占用
lsof -i :18789

# 杀死占用进程
kill -9 <PID>

# 或使用其他端口
openclaw config set gateway.port 18790
openclaw gateway restart
```

### 问题 2：Token 认证失败

```bash
# 检查 Token 格式
openclaw config get gateway.auth.token

# 重新生成 Token
openclaw config set gateway.auth.token "oc_$(openssl rand -hex 16)"
openclaw gateway restart
```

### 问题 3：Tailscale 未启动

```bash
# 检查 Tailscale 状态
tailscale status

# 启动 Tailscale
sudo tailscale up

# 验证 Tailscale IP
tailscale ip
```

---

## 总结

Gateway 配置机制的核心要点：

1. **端口和绑定**：决定访问范围（本机/局域网/远程）
2. **认证方式**：Token（自动化）vs Password（用户友好）
3. **Tailscale 集成**：安全的远程访问和公网暴露
4. **配置约束**：自动强制安全配置（Tailscale → Loopback，Funnel → Password）
5. **热重载**：大部分配置支持热重载，Gateway 服务器设置需要重启

理解这些机制，可以帮助你安全、高效地配置 OpenClaw Gateway。
