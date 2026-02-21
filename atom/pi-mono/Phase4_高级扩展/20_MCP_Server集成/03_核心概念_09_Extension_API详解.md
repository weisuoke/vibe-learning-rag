# 核心概念 09：Extension API 详解

> **深入理解 pi-mono Extension API 的接口设计、事件系统和工具注册机制**

---

## 概述

pi-mono Extension API 提供了强大的扩展能力，允许开发者通过 TypeScript 模块扩展 pi 的行为，实现自定义工具、命令、UI 组件和事件处理。

```
Extension API 核心能力：
├─ 自定义工具 → 注册 LLM 可调用的工具
├─ 事件拦截 → 阻止或修改工具调用、注入上下文
├─ 用户交互 → 对话框、通知、自定义 UI 组件
├─ 命令注册 → 添加斜杠命令（/mycommand）
└─ 状态持久化 → 跨会话保存扩展状态
```

**本质**：Extension API 是 pi-mono 的插件系统，通过事件驱动架构和丰富的 API 接口，让开发者能够深度定制 coding agent 的行为，而无需修改核心代码。

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## ExtensionAPI 接口概览

### 基本结构

扩展是一个导出默认函数的 TypeScript 模块，接收 `ExtensionAPI` 参数：

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

export default function (pi: ExtensionAPI) {
  // 订阅事件
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify("Extension loaded!", "info");
  });

  // 注册工具
  pi.registerTool({
    name: "greet",
    label: "Greet",
    description: "Greet someone by name",
    parameters: Type.Object({
      name: Type.String({ description: "Name to greet" }),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      return {
        content: [{ type: "text", text: `Hello, ${params.name}!` }],
        details: {},
      };
    },
  });

  // 注册命令
  pi.registerCommand("hello", {
    description: "Say hello",
    handler: async (args, ctx) => {
      ctx.ui.notify(`Hello ${args || "world"}!`, "info");
    },
  });
}
```

### 扩展位置

扩展自动从以下位置加载：

| 位置 | 作用域 |
|------|--------|
| `~/.pi/agent/extensions/*.ts` | 全局（所有项目） |
| `~/.pi/agent/extensions/*/index.ts` | 全局（子目录） |
| `.pi/extensions/*.ts` | 项目本地 |
| `.pi/extensions/*/index.ts` | 项目本地（子目录） |

**热重载**：使用 `/reload` 命令重新加载扩展。

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## 事件系统

### 事件生命周期

```
pi 启动
  │
  └─► session_start
      │
      ▼
用户发送提示 ─────────────────────────────────────────┐
  │                                                  │
  ├─► input (可拦截、转换或处理)                      │
  ├─► before_agent_start (可注入消息、修改系统提示)   │
  ├─► agent_start                                    │
  ├─► message_start / message_update / message_end   │
  │                                                  │
  │   ┌─── turn (LLM 调用工具时重复) ───┐            │
  │   │                                 │            │
  │   ├─► turn_start                    │            │
  │   ├─► context (可修改消息)          │            │
  │   │                                 │            │
  │   │   LLM 响应，可能调用工具：       │            │
  │   │     ├─► tool_call (可阻止)      │            │
  │   │     ├─► tool_execution_start    │            │
  │   │     ├─► tool_execution_update   │            │
  │   │     ├─► tool_execution_end      │            │
  │   │     └─► tool_result (可修改)    │            │
  │   │                                 │            │
  │   └─► turn_end                      │            │
  │                                                  │
  └─► agent_end                                      │
                                                     │
用户发送另一个提示 ◄────────────────────────────────────┘

退出 (Ctrl+C, Ctrl+D)
  └─► session_shutdown
```

### 会话事件

**session_start**：会话加载时触发

```typescript
pi.on("session_start", async (_event, ctx) => {
  const sessionFile = ctx.sessionManager.getSessionFile();
  ctx.ui.notify(`Session: ${sessionFile ?? "ephemeral"}`, "info");
});
```

**session_before_switch / session_switch**：切换会话时触发

```typescript
pi.on("session_before_switch", async (event, ctx) => {
  // event.reason - "new" 或 "resume"
  // event.targetSessionFile - 目标会话文件（仅 "resume"）

  if (event.reason === "new") {
    const ok = await ctx.ui.confirm("Clear?", "Delete all messages?");
    if (!ok) return { cancel: true };
  }
});

pi.on("session_switch", async (event, ctx) => {
  // event.reason - "new" 或 "resume"
  // event.previousSessionFile - 之前的会话文件
});
```

**session_shutdown**：退出时触发

```typescript
pi.on("session_shutdown", async (_event, ctx) => {
  // 清理资源、保存状态等
});
```

### 代理事件

**before_agent_start**：用户提交提示后、代理循环前触发

```typescript
pi.on("before_agent_start", async (event, ctx) => {
  // event.prompt - 用户提示文本
  // event.images - 附加图片（如果有）
  // event.systemPrompt - 当前系统提示

  return {
    // 注入持久化消息（存储在会话中，发送给 LLM）
    message: {
      customType: "my-extension",
      content: "Additional context for the LLM",
      display: true,
    },
    // 替换此轮的系统提示（跨扩展链式调用）
    systemPrompt: event.systemPrompt + "\n\nExtra instructions...",
  };
});
```

**agent_start / agent_end**：每个用户提示触发一次

```typescript
pi.on("agent_start", async (_event, ctx) => {});

pi.on("agent_end", async (event, ctx) => {
  // event.messages - 此提示的消息
});
```

**turn_start / turn_end**：每个回合触发（一次 LLM 响应 + 工具调用）

```typescript
pi.on("turn_start", async (event, ctx) => {
  // event.turnIndex, event.timestamp
});

pi.on("turn_end", async (event, ctx) => {
  // event.turnIndex, event.message, event.toolResults
});
```

### 工具事件

**tool_call**：工具执行前触发，**可阻止**

```typescript
import { isToolCallEventType } from "@mariozechner/pi-coding-agent";

pi.on("tool_call", async (event, ctx) => {
  // event.toolName - "bash", "read", "write", "edit" 等
  // event.toolCallId
  // event.input - 工具参数

  // 内置工具：无需类型参数
  if (isToolCallEventType("bash", event)) {
    // event.input 是 { command: string; timeout?: number }
    if (event.input.command.includes("rm -rf")) {
      const ok = await ctx.ui.confirm("Dangerous!", "Allow rm -rf?");
      if (!ok) return { block: true, reason: "Blocked by user" };
    }
  }

  if (isToolCallEventType("read", event)) {
    // event.input 是 { path: string; offset?: number; limit?: number }
    console.log(`Reading: ${event.input.path}`);
  }
});
```

**tool_result**：工具执行后触发，**可修改结果**

```typescript
import { isBashToolResult } from "@mariozechner/pi-coding-agent";

pi.on("tool_result", async (event, ctx) => {
  // event.toolName, event.toolCallId, event.input
  // event.content, event.details, event.isError

  if (isBashToolResult(event)) {
    // event.details 类型为 BashToolDetails
  }

  // 修改结果
  return {
    content: [...],
    details: {...},
    isError: false
  };
});
```

**tool_result 处理器链式调用**：

- 处理器按扩展加载顺序运行
- 每个处理器看到前一个处理器修改后的最新结果
- 处理器可以返回部分补丁（`content`、`details` 或 `isError`）；省略的字段保持当前值

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## 工具注册机制

### 基本工具定义

```typescript
import { Type } from "@sinclair/typebox";
import { StringEnum } from "@mariozechner/pi-ai";

pi.registerTool({
  name: "my_tool",
  label: "My Tool",
  description: "What this tool does",
  parameters: Type.Object({
    action: StringEnum(["list", "add"] as const),
    text: Type.Optional(Type.String()),
  }),

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    // 流式进度更新
    onUpdate?.({ content: [{ type: "text", text: "Working..." }] });

    return {
      content: [{ type: "text", text: "Done" }],
      details: { result: "..." },
    };
  },

  // 可选：自定义渲染
  renderCall(args, theme) {
    return `Calling ${args.action}...`;
  },
  renderResult(result, options, theme) {
    return `Result: ${result.content[0].text}`;
  },
});
```

### 工具执行流程

**每个工具调用的流程**：

1. `tool_call` 事件（可阻止）
2. `tool_execution_start` 事件
3. （可选重复）`tool_execution_update` 事件
4. `tool_execution_end` 事件
5. `tool_result` 事件（可修改）

### 流式更新

```typescript
async execute(toolCallId, params, signal, onUpdate, ctx) {
  onUpdate?.({ content: [{ type: "text", text: "Step 1..." }] });
  // 执行步骤 1
  onUpdate?.({ content: [{ type: "text", text: "Step 2..." }] });
  // 执行步骤 2
  return { content: [{ type: "text", text: "Final" }] };
}
```

### 大结果截断

```typescript
return {
  content: [{ type: "text", text: fullOutput }],
  truncated: true,   // 显示 "Output truncated" 横幅
};
```

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## ExtensionContext

每个处理器接收 `ctx: ExtensionContext`：

### ctx.ui - 用户交互

**对话框**：

```typescript
const choice = await ctx.ui.select("Title", ["Option 1", "Option 2"]);
const ok = await ctx.ui.confirm("Title", "Message?");
const text = await ctx.ui.input("Title", "Placeholder");
const text = await ctx.ui.editor("Title", "Initial text");
ctx.ui.notify("Message", "success|info|warning|error");
```

**状态 / 小部件**：

```typescript
ctx.ui.setStatus("my-ext", "Working...");     // 页脚
ctx.ui.setWidget("my-ext", ["Line 1", "Line 2"]); // 编辑器上方
ctx.ui.setTitle("Custom title");              // 窗口标题
ctx.ui.setEditorText("Replace editor content");
```

**自定义组件**：

```typescript
ctx.ui.custom({
  render: (ctx) => {
    // 返回行或使用 ctx.onKey, ctx.onResize 等
  },
  width: 60,
  height: 20,
});
```

### ctx.sessionManager - 会话管理

只读访问会话状态：

```typescript
ctx.sessionManager.getEntries()       // 所有条目
ctx.sessionManager.getBranch()        // 当前分支
ctx.sessionManager.getLeafId()        // 当前叶子条目 ID
```

### ctx.modelRegistry / ctx.model

访问模型和 API 密钥。

### ctx.isIdle() / ctx.abort() / ctx.hasPendingMessages()

控制流辅助方法。

### ctx.shutdown()

请求优雅关闭 pi：

```typescript
pi.on("tool_call", (event, ctx) => {
  if (isFatal(event.input)) {
    ctx.shutdown();
  }
});
```

### ctx.getContextUsage()

返回活动模型的当前上下文使用情况：

```typescript
const usage = ctx.getContextUsage();
if (usage && usage.tokens > 100_000) {
  // 触发压缩或其他操作
}
```

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## 状态管理

### 通过工具结果详情持久化

扩展应将状态存储在工具结果 `details` 中以支持分支：

```typescript
export default function (pi: ExtensionAPI) {
  let items: string[] = [];

  // 从会话重建状态
  pi.on("session_start", async (_event, ctx) => {
    items = [];
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "message" && entry.message.role === "toolResult") {
        if (entry.message.toolName === "my_tool") {
          items = entry.message.details?.items ?? [];
        }
      }
    }
  });

  pi.registerTool({
    name: "my_tool",
    // ...
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      items.push("new item");
      return {
        content: [{ type: "text", text: `Added item. Total: ${items.length}` }],
        details: { items },
      };
    },
  });
}
```

### 通过 appendEntry 持久化

持久化扩展状态（**不参与 LLM 上下文**）：

```typescript
pi.appendEntry("my-state", { count: 42 });

// 重新加载时恢复
pi.on("session_start", async (_event, ctx) => {
  for (const entry of ctx.sessionManager.getEntries()) {
    if (entry.type === "custom" && entry.customType === "my-state") {
      // 从 entry.data 重建
    }
  }
});
```

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## 命令注册

### 基本命令

```typescript
pi.registerCommand("stats", {
  description: "Show session statistics",
  handler: async (args, ctx) => {
    const count = ctx.sessionManager.getEntries().length;
    ctx.ui.notify(`${count} entries`, "info");
  }
});
```

### 参数自动补全

```typescript
import type { AutocompleteItem } from "@mariozechner/pi-tui";

pi.registerCommand("deploy", {
  description: "Deploy to an environment",
  getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
    const envs = ["dev", "staging", "prod"];
    const items = envs.map((e) => ({ value: e, label: e }));
    const filtered = items.filter((i) => i.value.startsWith(prefix));
    return filtered.length > 0 ? filtered : null;
  },
  handler: async (args, ctx) => {
    ctx.ui.notify(`Deploying: ${args}`, "info");
  },
});
```

### ExtensionCommandContext

命令处理器接收 `ExtensionCommandContext`，扩展了 `ExtensionContext`，增加了会话控制方法：

**ctx.waitForIdle()**：等待代理完成流式传输

```typescript
pi.registerCommand("my-cmd", {
  handler: async (args, ctx) => {
    await ctx.waitForIdle();
    // 代理现在空闲，可以安全修改会话
  },
});
```

**ctx.newSession(options?)**：创建新会话

```typescript
const result = await ctx.newSession({
  parentSession: ctx.sessionManager.getSessionFile(),
  setup: async (sm) => {
    sm.appendMessage({
      role: "user",
      content: [{ type: "text", text: "Context from previous session..." }],
      timestamp: Date.now(),
    });
  },
});

if (result.cancelled) {
  // 扩展取消了新会话
}
```

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## MCP 集成示例

### pi-mcp-adapter

pi-mono 本身不支持 MCP，但可以通过扩展集成。[pi-mcp-adapter](https://github.com/nicobailon/pi-mcp-adapter) 是一个令牌高效的 MCP 适配器扩展。

**为什么需要适配器？**

Mario Zechner 在博客中解释了为什么 pi 不内置 MCP 支持：

> "MCP servers are overkill for most use cases, and they come with significant context overhead. Popular MCP servers like Playwright MCP (21 tools, 13.7k tokens) or Chrome DevTools MCP (26 tools, 18k tokens) dump their entire tool descriptions into your context on every session. That's 7-9% of your context window gone before you even start working."

**适配器解决方案**：

- 一个代理工具（~200 tokens）而不是数百个工具
- 代理按需发现所需内容
- 服务器仅在实际使用时启动
- 元数据缓存使搜索和描述无需实时连接即可工作

**安装**：

```bash
pi install npm:pi-mcp-adapter
```

**配置** (`~/.pi/agent/mcp.json`)：

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest"],
      "lifecycle": "lazy",
      "idleTimeout": 10
    }
  }
}
```

**使用**：

```typescript
// 搜索工具
mcp({ search: "screenshot" })

// 调用工具
mcp({ tool: "chrome_devtools_take_screenshot", args: '{"format": "png"}' })
```

[Source: What I learned building an opinionated and minimal coding agent](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/)
[Source: pi-mcp-adapter GitHub](https://github.com/nicobailon/pi-mcp-adapter)

---

## 最佳实践

### 扩展开发

1. **使用 TypeScript**：扩展通过 [jiti](https://github.com/unjs/jiti) 加载，无需编译
2. **热重载**：使用 `/reload` 命令重新加载扩展
3. **错误处理**：处理器中未捕获的错误 → 记录 + UI 通知
4. **状态管理**：使用工具结果 `details` 或 `appendEntry` 持久化状态
5. **UI 交互**：检查 `ctx.hasUI` 以支持不同模式（交互式、RPC、打印）

### 模式行为

| 模式 | ctx.hasUI | 对话框 | 小部件 | notify |
|------|-----------|--------|--------|--------|
| 交互式 | true | 完整 | 完整 | 完整 |
| RPC | true | 通过协议 | 即发即弃 | 即发即弃 |
| 打印 (`-p`) | false | 无操作 | 无操作 | 无操作 |
| JSON | false | 无操作 | 无操作 | 无操作 |

### 安全考虑

**扩展以完整系统权限运行，可以执行任意代码。仅从信任的来源安装。**

[Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)

---

## 总结

### 核心要点

1. **ExtensionAPI**：通过事件驱动架构扩展 pi 行为
2. **事件系统**：会话事件、代理事件、工具事件、模型事件
3. **工具注册**：使用 TypeBox schema 定义参数，支持流式更新
4. **状态管理**：通过工具结果 details 或 appendEntry 持久化
5. **MCP 集成**：通过适配器扩展实现令牌高效的 MCP 支持

### 关键约束

- ✅ 扩展位置：`~/.pi/agent/extensions/` 或 `.pi/extensions/`
- ✅ 热重载：使用 `/reload` 命令
- ✅ 事件处理器：可阻止或修改工具调用
- ✅ 状态持久化：使用 details 或 appendEntry
- ✅ 安全：扩展以完整系统权限运行

### 下一步

- 阅读 [03_核心概念_10_MCP_Client集成](./03_核心概念_10_MCP_Client集成.md) 了解 MCP 客户端集成
- 阅读 [07_实战代码_05_基础Extension开发](./07_实战代码_05_基础Extension开发.md) 查看完整实现

---

**参考资源**：
- [Source: pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
- [Source: What I learned building an opinionated and minimal coding agent](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/)
- [Source: pi-mcp-adapter GitHub](https://github.com/nicobailon/pi-mcp-adapter)
