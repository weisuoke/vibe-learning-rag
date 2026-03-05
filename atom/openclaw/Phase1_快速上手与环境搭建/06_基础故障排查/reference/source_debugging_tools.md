# OpenClaw 调试工具

**来源**: `sourcecode/openclaw/docs/help/debugging.md`

## 运行时调试覆盖

使用 `/debug` 在聊天中设置**仅运行时**配置覆盖（内存，非磁盘）。
`/debug` 默认禁用；使用 `commands.debug: true` 启用。

示例：
```
/debug show
/debug set messages.responsePrefix="[openclaw]"
/debug unset messages.responsePrefix
/debug reset
```

`/debug reset` 清除所有覆盖并返回到磁盘上的配置。

## Gateway 监视模式

快速迭代，在文件监视器下运行网关：

```bash
pnpm gateway:watch
```

映射到：
```bash
node --watch-path src --watch-path tsconfig.json --watch-path package.json --watch-preserve-output scripts/run-node.mjs gateway --force
```

## 开发配置文件 + 开发网关 (--dev)

使用开发配置文件隔离状态并启动安全、一次性的调试设置。有**两个** `--dev` 标志：

- **全局 `--dev`（配置文件）**: 在 `~/.openclaw-dev` 下隔离状态，默认网关端口为 `19001`
- **`gateway --dev`**: 告诉 Gateway 在缺失时自动创建默认配置 + 工作区（并跳过 BOOTSTRAP.md）

推荐流程（开发配置文件 + 开发引导）：

```bash
pnpm gateway:dev
OPENCLAW_PROFILE=dev openclaw tui
```

这会做什么：

1. **配置文件隔离**（全局 `--dev`）
   - `OPENCLAW_PROFILE=dev`
   - `OPENCLAW_STATE_DIR=~/.openclaw-dev`
   - `OPENCLAW_CONFIG_PATH=~/.openclaw-dev/openclaw.json`
   - `OPENCLAW_GATEWAY_PORT=19001`

2. **开发引导**（`gateway --dev`）
   - 如果缺失，写入最小配置（`gateway.mode=local`，绑定回环）
   - 设置 `agent.workspace` 到开发工作区
   - 设置 `agent.skipBootstrap=true`（无 BOOTSTRAP.md）
   - 如果缺失，种子工作区文件：`AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`
   - 默认身份：**C3‑PO**（协议机器人）
   - 在开发模式下跳过通道提供商（`OPENCLAW_SKIP_CHANNELS=1`）

重置流程（全新开始）：

```bash
pnpm gateway:dev:reset
```

## 原始流日志记录 (OpenClaw)

OpenClaw 可以在任何过滤/格式化之前记录**原始助手流**。

通过 CLI 启用：

```bash
pnpm gateway:watch --raw-stream
```

可选路径覆盖：

```bash
pnpm gateway:watch --raw-stream --raw-stream-path ~/.openclaw/logs/raw-stream.jsonl
```

等效环境变量：

```bash
OPENCLAW_RAW_STREAM=1
OPENCLAW_RAW_STREAM_PATH=~/.openclaw/logs/raw-stream.jsonl
```

默认文件：`~/.openclaw/logs/raw-stream.jsonl`

## 原始块日志记录 (pi-mono)

要在解析为块之前捕获**原始 OpenAI 兼容块**，pi-mono 公开了一个单独的记录器：

```bash
PI_RAW_STREAM=1
```

可选路径：

```bash
PI_RAW_STREAM_PATH=~/.pi-mono/logs/raw-openai-completions.jsonl
```

默认文件：`~/.pi-mono/logs/raw-openai-completions.jsonl`

## 安全注意事项

- 原始流日志可以包含完整的提示、工具输出和用户数据
- 保持日志本地并在调试后删除它们
- 如果你分享日志，首先清理机密和 PII

## 核心概念

1. **运行时调试覆盖**: 使用 `/debug` 命令临时修改配置
2. **监视模式**: 文件更改时自动重启网关
3. **开发配置文件**: 隔离的开发环境，不影响生产配置
4. **原始流日志**: 捕获未过滤的模型输出用于调试
5. **安全实践**: 保护敏感数据不被泄露
