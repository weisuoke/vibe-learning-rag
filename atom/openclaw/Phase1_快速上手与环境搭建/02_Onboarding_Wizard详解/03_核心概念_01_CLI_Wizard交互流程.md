# 核心概念 1：CLI Wizard 交互流程

**本文档深入讲解 CLI Wizard 的交互流程、状态管理和输入验证机制。**

---

## 概述

CLI Wizard 是 OpenClaw Onboarding 的核心组件，通过交互式命令行界面引导用户完成配置。它基于 `@clack/prompts` 库实现，提供友好的用户体验和严格的输入验证。

---

## 交互流程架构

### 整体流程图

```
用户运行 openclaw onboard
    ↓
┌─────────────────────────────────────┐
│  1. 初始化 (Initialization)         │
│  - 打印欢迎信息                     │
│  - 读取已有配置                     │
│  - 安全风险确认                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 模式选择 (Mode Selection)       │
│  - QuickStart vs Advanced           │
│  - Local vs Remote                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. 配置流程 (Configuration Flow)   │
│  - Model/Auth                       │
│  - Workspace                        │
│  - Gateway                          │
│  - Channels                         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. 安装流程 (Installation Flow)    │
│  - Daemon 安装                      │
│  - Health Check                     │
│  - Skills 安装                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. 完成 (Completion)               │
│  - 显示配置摘要                     │
│  - 提供下一步建议                   │
└─────────────────────────────────────┘
```

---

## 阶段 1：初始化

### 1.1 欢迎信息

```typescript
// sourcecode/openclaw/src/wizard/onboarding.ts
export async function runOnboardingWizard(
  opts: OnboardOptions,
  runtime: RuntimeEnv,
  prompter: WizardPrompter,
) {
  // 打印欢迎信息
  onboardHelpers.printWizardHeader(runtime);
  await prompter.intro("OpenClaw onboarding");

  // ...
}
```

**输出示例**：

```
┌  OpenClaw onboarding
│
```

### 1.2 安全风险确认

```typescript
async function requireRiskAcknowledgement(params: {
  opts: OnboardOptions;
  prompter: WizardPrompter;
}) {
  if (params.opts.acceptRisk === true) {
    return; // 跳过（非交互模式）
  }

  await params.prompter.note(
    [
      "Security warning — please read.",
      "",
      "OpenClaw is a hobby project and still in beta. Expect sharp edges.",
      "This bot can read files and run actions if tools are enabled.",
      "A bad prompt can trick it into doing unsafe things.",
      // ...
    ].join("\n"),
    "Security",
  );

  const ok = await params.prompter.confirm({
    message: "I understand this is powerful and inherently risky. Continue?",
    initialValue: false,
  });

  if (!ok) {
    throw new WizardCancelledError("risk not accepted");
  }
}
```

**关键设计**：
- 默认值为 `false`（用户必须主动确认）
- 非交互模式可通过 `--accept-risk` 跳过
- 拒绝后抛出 `WizardCancelledError`（优雅退出）

### 1.3 读取已有配置

```typescript
const snapshot = await readConfigFileSnapshot();
let baseConfig: OpenClawConfig = snapshot.valid ? snapshot.config : {};

if (snapshot.exists && !snapshot.valid) {
  // 配置文件存在但无效
  await prompter.note(
    onboardHelpers.summarizeExistingConfig(baseConfig),
    "Invalid config"
  );

  if (snapshot.issues.length > 0) {
    await prompter.note(
      snapshot.issues.map((iss) => `- ${iss.path}: ${iss.message}`).join("\n"),
      "Config issues"
    );
  }

  await prompter.outro(
    `Config invalid. Run \`openclaw doctor\` to repair it, then re-run onboarding.`
  );
  runtime.exit(1);
  return;
}
```

**配置状态处理**：

| 状态 | 行为 |
|------|------|
| 不存在 | 继续向导（全新配置） |
| 存在且有效 | 提供保留/更新/重置选项 |
| 存在但无效 | 提示运行 `openclaw doctor`，退出向导 |

---

## 阶段 2：模式选择

### 2.1 QuickStart vs Advanced

```typescript
const explicitFlow: WizardFlow | undefined = opts.flow;
let flow: WizardFlow =
  explicitFlow ??
  (await prompter.select({
    message: "Onboarding mode",
    options: [
      { value: "quickstart", label: "QuickStart", hint: quickstartHint },
      { value: "advanced", label: "Manual", hint: manualHint },
    ],
    initialValue: "quickstart",
  }));
```

**交互示例**：

```
◆  Onboarding mode
│  ● QuickStart (Configure details later via openclaw configure)
│  ○ Manual (Configure port, network, Tailscale, and auth options)
```

**模式特点**：

| 模式 | 交互次数 | 默认值 | 适用场景 |
|------|---------|-------|---------|
| QuickStart | 最少（3-5 个问题） | 使用安全默认值 | 本地开发、快速体验 |
| Advanced | 完整（10+ 个问题） | 每个选项都询问 | 生产部署、完全控制 |

### 2.2 Local vs Remote

```typescript
const mode =
  opts.mode ??
  (flow === "quickstart"
    ? "local"
    : await prompter.select({
        message: "What do you want to set up?",
        options: [
          {
            value: "local",
            label: "Local gateway (this machine)",
            hint: localProbe.ok
              ? `Gateway reachable (${localUrl})`
              : `No gateway detected (${localUrl})`,
          },
          {
            value: "remote",
            label: "Remote gateway (connect to existing)",
            hint: remoteProbe?.ok
              ? `Gateway reachable (${remoteUrl})`
              : "Configure connection to remote gateway",
          },
        ],
      }));
```

**Gateway 探测**：

```typescript
const localProbe = await onboardHelpers.probeGatewayReachable({
  url: localUrl,
  token: baseConfig.gateway?.auth?.token,
  password: baseConfig.gateway?.auth?.password,
});
```

**探测逻辑**：
- 尝试连接 `/health` 端点
- 使用已有 Token/Password（如果存在）
- 显示探测结果（reachable / not detected）

---

## 阶段 3：配置流程

### 3.1 Model/Auth 配置

```typescript
// 选择 Provider
const authChoice = await prompter.select({
  message: "Choose your model provider",
  options: [
    { value: "anthropic-api-key", label: "Anthropic (recommended)" },
    { value: "openai-api-key", label: "OpenAI" },
    { value: "custom-api-key", label: "Custom Provider" },
  ],
});

// 输入 API 密钥
if (authChoice === "anthropic-api-key") {
  const apiKey = await prompter.text({
    message: "Anthropic API key",
    placeholder: "sk-ant-...",
    validate: (value) => {
      if (!value || value.trim() === "") {
        return "API key is required";
      }
      if (!value.startsWith("sk-ant-")) {
        return "Invalid Anthropic API key format";
      }
      return undefined;
    },
  });

  nextConfig.env = {
    ...nextConfig.env,
    ANTHROPIC_API_KEY: apiKey,
  };
}

// 选择默认模型
const model = await prompter.select({
  message: "Choose your default model",
  options: [
    { value: "claude-sonnet-4-5", label: "claude-sonnet-4-5 (recommended)" },
    { value: "claude-opus-4-6", label: "claude-opus-4-6" },
    { value: "claude-haiku-4", label: "claude-haiku-4" },
  ],
});
```

**输入验证**：
- API 密钥格式验证（`sk-ant-` 前缀）
- 非空验证
- 实时反馈（输入时显示错误）

### 3.2 Workspace 配置

```typescript
const workspaceInput = await prompter.text({
  message: "Agent workspace directory",
  initialValue: baseConfig.agents?.defaults?.workspace ?? DEFAULT_WORKSPACE,
  validate: (value) => {
    if (!value || value.trim() === "") {
      return "Workspace path is required";
    }
    return undefined;
  },
});

const workspaceDir = resolveUserPath(workspaceInput);

// 初始化 Workspace
await ensureWorkspaceExists(workspaceDir);
await seedBootstrapFile(workspaceDir);
```

**Workspace 初始化**：
1. 创建目录（如果不存在）
2. 创建 `.openclaw/` 子目录
3. 生成 `bootstrap.md` 文件
4. 设置权限（仅用户可读写）

### 3.3 Gateway 配置

#### QuickStart 模式

```typescript
if (flow === "quickstart") {
  // 使用默认值
  const settings: GatewayWizardSettings = {
    port: DEFAULT_GATEWAY_PORT,  // 18789
    bind: "loopback",  // 127.0.0.1
    authMode: "token",
    gatewayToken: randomToken(),  // 自动生成
    tailscaleMode: "off",
    tailscaleResetOnExit: false,
  };

  await prompter.note(
    [
      `Gateway port: ${DEFAULT_GATEWAY_PORT}`,
      "Gateway bind: Loopback (127.0.0.1)",
      "Gateway auth: Token (default)",
      "Tailscale exposure: Off",
    ].join("\n"),
    "QuickStart"
  );
}
```

#### Advanced 模式

```typescript
// 端口配置
const port = Number.parseInt(
  await prompter.text({
    message: "Gateway port",
    initialValue: String(localPort),
    validate: (value) =>
      Number.isFinite(Number(value)) ? undefined : "Invalid port",
  }),
  10
);

// 绑定地址
const bind = await prompter.select({
  message: "Gateway bind",
  options: [
    { value: "loopback", label: "Loopback (127.0.0.1)" },
    { value: "lan", label: "LAN (0.0.0.0)" },
    { value: "tailnet", label: "Tailnet (Tailscale IP)" },
    { value: "auto", label: "Auto (Loopback → LAN)" },
    { value: "custom", label: "Custom IP" },
  ],
});

// 自定义 IP（如果选择 custom）
if (bind === "custom") {
  const customBindHost = await prompter.text({
    message: "Custom IP address",
    placeholder: "192.168.1.100",
    validate: validateIPv4AddressInput,
  });
}

// 认证方式
const authMode = await prompter.select({
  message: "Gateway auth",
  options: [
    { value: "token", label: "Token", hint: "Recommended default" },
    { value: "password", label: "Password" },
  ],
  initialValue: "token",
});

// Token 生成
if (authMode === "token") {
  const tokenInput = await prompter.text({
    message: "Gateway token (blank to generate)",
    placeholder: "Needed for multi-machine or non-loopback access",
  });
  gatewayToken = normalizeGatewayTokenInput(tokenInput) || randomToken();
}

// Tailscale 配置
const tailscaleMode = await prompter.select({
  message: "Tailscale exposure",
  options: [
    { value: "off", label: "Off" },
    { value: "serve", label: "Serve (private Tailnet)" },
    { value: "funnel", label: "Funnel (public internet)" },
  ],
});
```

**约束检查**：

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

### 3.4 Channels 配置

```typescript
const channelChoices = await prompter.multiselect({
  message: "Which channels do you want to set up?",
  options: [
    { value: "whatsapp", label: "WhatsApp" },
    { value: "telegram", label: "Telegram" },
    { value: "discord", label: "Discord" },
    { value: "skip", label: "Skip for now (use Dashboard)" },
  ],
});

for (const channel of channelChoices) {
  if (channel === "whatsapp") {
    await setupWhatsAppChannel(prompter, nextConfig);
  } else if (channel === "telegram") {
    await setupTelegramChannel(prompter, nextConfig);
  } else if (channel === "discord") {
    await setupDiscordChannel(prompter, nextConfig);
  }
}
```

**WhatsApp 配置示例**：

```typescript
async function setupWhatsAppChannel(
  prompter: WizardPrompter,
  config: OpenClawConfig
) {
  await prompter.note(
    "WhatsApp requires QR code pairing. You'll scan a QR code with your phone.",
    "WhatsApp"
  );

  const phoneNumber = await prompter.text({
    message: "Your phone number (for allowlist)",
    placeholder: "+1234567890",
    validate: (value) => {
      if (!value.startsWith("+")) {
        return "Phone number must start with +";
      }
      return undefined;
    },
  });

  config.channels = {
    ...config.channels,
    whatsapp: {
      enabled: true,
      dmPolicy: "allowlist",
      allowFrom: [phoneNumber],
    },
  };
}
```

---

## 阶段 4：安装流程

### 4.1 Daemon 安装

```typescript
const installDaemon =
  explicitInstallDaemon ??
  (flow === "quickstart"
    ? true
    : await prompter.confirm({
        message: "Install Gateway service (recommended)",
        initialValue: true,
      }));

if (installDaemon) {
  const daemonRuntime =
    flow === "quickstart"
      ? DEFAULT_GATEWAY_DAEMON_RUNTIME
      : await prompter.select({
          message: "Gateway service runtime",
          options: GATEWAY_DAEMON_RUNTIME_OPTIONS,
        });

  const service = resolveGatewayService();
  const loaded = await service.isLoaded({ env: process.env });

  if (loaded) {
    const action = await prompter.select({
      message: "Gateway service already installed",
      options: [
        { value: "restart", label: "Restart" },
        { value: "reinstall", label: "Reinstall" },
        { value: "skip", label: "Skip" },
      ],
    });
    // 处理选择...
  }

  // 安装服务
  await service.install({
    env: process.env,
    programArguments,
    workingDirectory,
    environment,
  });
}
```

**进度显示**：

```typescript
const progress = prompter.progress("Gateway service");
try {
  progress.update("Preparing Gateway service…");
  // 构建安装计划...

  progress.update("Installing Gateway service…");
  await service.install({...});
} finally {
  progress.stop("Gateway service installed.");
}
```

### 4.2 Health Check

```typescript
if (!opts.skipHealth) {
  const progress = prompter.progress("Health check");

  try {
    progress.update("Starting Gateway...");
    await service.start({ env: process.env });

    progress.update("Waiting for Gateway to become ready...");
    const probe = await waitForGatewayReachable({
      url: gatewayUrl,
      token: settings.gatewayToken,
      timeoutMs: 12000,
    });

    if (probe.ok) {
      progress.stop("Gateway is ready!");
    } else {
      progress.stop("Gateway did not become ready.");
      await prompter.note(
        "Check logs: openclaw logs gateway",
        "Health check failed"
      );
    }
  } catch (err) {
    progress.stop("Health check failed.");
  }
}
```

**轮询逻辑**：

```typescript
async function waitForGatewayReachable(params: {
  url: string;
  token?: string;
  timeoutMs: number;
}): Promise<{ ok: boolean }> {
  const startTime = Date.now();

  while (Date.now() - startTime < params.timeoutMs) {
    try {
      const response = await fetch(`${params.url}/health`, {
        headers: {
          ...(params.token && { "Authorization": `Bearer ${params.token}` }),
        },
      });

      if (response.ok) {
        return { ok: true };
      }
    } catch (err) {
      // Gateway 未启动，继续轮询
    }

    await sleep(1000); // 每秒轮询一次
  }

  return { ok: false };
}
```

### 4.3 Skills 安装

```typescript
const installSkills =
  flow === "quickstart"
    ? true
    : await prompter.confirm({
        message: "Install recommended skills?",
        initialValue: true,
      });

if (installSkills) {
  const progress = prompter.progress("Skills");

  try {
    progress.update("Installing recommended skills...");

    const recommendedSkills = ["web-search", "file-operations", "code-execution"];

    for (const skill of recommendedSkills) {
      await installSkill(skill);
      await prompter.note(`✓ ${skill}`, "Skills");
    }
  } finally {
    progress.stop("Skills installed.");
  }
}
```

---

## 阶段 5：完成

### 5.1 配置摘要

```typescript
await prompter.outro("Setup complete! 🎉");

await prompter.note(
  [
    "Next steps:",
    "  openclaw dashboard    # Open Control UI",
    "  openclaw chat         # Start TUI chat",
    "  openclaw configure    # Adjust settings",
  ].join("\n"),
  "Next steps"
);
```

### 5.2 Shell 补全安装

```typescript
await setupOnboardingShellCompletion({
  flow,
  prompter,
});
```

**补全安装逻辑**：

```typescript
export async function setupOnboardingShellCompletion(params: {
  flow: WizardFlow;
  prompter: WizardPrompter;
}) {
  const completionStatus = await checkShellCompletionStatus(cliName);

  if (!completionStatus.profileInstalled) {
    const shouldInstall =
      params.flow === "quickstart"
        ? true
        : await params.prompter.confirm({
            message: `Enable ${completionStatus.shell} shell completion?`,
            initialValue: true,
          });

    if (shouldInstall) {
      await ensureCompletionCacheExists(cliName);
      await installCompletion(completionStatus.shell, true, cliName);

      await params.prompter.note(
        `Shell completion installed. Restart your shell or run: source ${profileHint}`,
        "Shell completion"
      );
    }
  }
}
```

---

## 状态管理

### 配置状态

```typescript
type ConfigState = {
  baseConfig: OpenClawConfig;  // 已有配置
  nextConfig: OpenClawConfig;  // 新配置
  flow: WizardFlow;            // QuickStart | Advanced
  mode: OnboardMode;           // local | remote
  settings: GatewayWizardSettings;  // Gateway 设置
};
```

### 状态转换

```
初始状态
    ↓
读取已有配置 → baseConfig
    ↓
用户选择模式 → flow, mode
    ↓
配置流程 → nextConfig
    ↓
安装流程 → settings
    ↓
写入配置文件
    ↓
完成状态
```

---

## 输入验证

### 验证函数

```typescript
// 端口验证
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

// IPv4 地址验证
function validateIPv4AddressInput(value: string): string | undefined {
  const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
  if (!ipv4Regex.test(value)) {
    return "Invalid IPv4 address";
  }

  const parts = value.split(".").map(Number);
  if (parts.some(part => part > 255)) {
    return "Invalid IPv4 address (octets must be 0-255)";
  }

  return undefined;
}

// API 密钥验证
function validateAnthropicApiKey(value: string): string | undefined {
  if (!value || value.trim() === "") {
    return "API key is required";
  }
  if (!value.startsWith("sk-ant-")) {
    return "Invalid Anthropic API key format (must start with sk-ant-)";
  }
  return undefined;
}

// Gateway Token 验证
function normalizeGatewayTokenInput(input: string): string | undefined {
  if (typeof input !== "string") return undefined;
  const trimmed = input.trim();
  if (trimmed === "") return undefined;

  if (!/^oc_[0-9a-f]{32}$/.test(trimmed)) {
    throw new Error("Invalid token format. Expected: oc_<32 hex chars>");
  }

  return trimmed;
}
```

### 实时验证

```typescript
const apiKey = await prompter.text({
  message: "Anthropic API key",
  validate: (value) => {
    // 实时验证（用户输入时）
    if (!value || value.trim() === "") {
      return "API key is required";
    }
    if (!value.startsWith("sk-ant-")) {
      return "Invalid format (must start with sk-ant-)";
    }
    return undefined; // 验证通过
  },
});
```

**验证时机**：
- 用户输入时（实时反馈）
- 提交前（最终验证）
- 配置写入前（严格验证）

---

## 错误处理

### 用户取消

```typescript
export class WizardCancelledError extends Error {
  constructor(reason: string) {
    super(`Wizard cancelled: ${reason}`);
    this.name = "WizardCancelledError";
  }
}

// 使用
if (!ok) {
  throw new WizardCancelledError("risk not accepted");
}
```

### 配置无效

```typescript
if (snapshot.exists && !snapshot.valid) {
  await prompter.outro(
    `Config invalid. Run \`openclaw doctor\` to repair it, then re-run onboarding.`
  );
  runtime.exit(1);
  return;
}
```

### 安装失败

```typescript
try {
  await service.install({...});
} catch (err) {
  const installError = err instanceof Error ? err.message : String(err);
  await prompter.note(
    `Gateway service install failed: ${installError}`,
    "Gateway"
  );
  await prompter.note(gatewayInstallErrorHint(), "Gateway");
}
```

---

## 非交互模式

### 参数映射

```typescript
if (opts.nonInteractive) {
  // 跳过所有交互，使用参数或环境变量
  flow = opts.flow ?? "quickstart";
  authChoice = opts.authChoice ?? "anthropic-api-key";
  apiKey = opts.anthropicApiKey ?? process.env.ANTHROPIC_API_KEY;
  port = opts.gatewayPort ?? DEFAULT_GATEWAY_PORT;
  bind = opts.gatewayBind ?? "loopback";
  // ...
}
```

### 必需参数检查

```typescript
if (opts.nonInteractive) {
  if (!apiKey) {
    throw new Error(
      "Missing required parameter: --anthropic-api-key or ANTHROPIC_API_KEY"
    );
  }
  if (!model) {
    throw new Error("Missing required parameter: --anthropic-model");
  }
}
```

---

## 总结

CLI Wizard 的交互流程设计精妙：

1. **渐进式披露**：QuickStart 隐藏复杂选项，Advanced 暴露全部控制
2. **实时验证**：输入时立即反馈错误，提升用户体验
3. **状态管理**：清晰的状态转换，易于理解和维护
4. **错误处理**：优雅的错误处理和恢复机制
5. **非交互支持**：完整的自动化配置能力

理解这些机制，可以帮助你更好地使用和扩展 Onboarding Wizard。
