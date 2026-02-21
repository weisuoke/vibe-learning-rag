# 实战代码 05：基础 Extension 开发

> **从零构建 pi-mono 扩展，集成 MCP 服务器**

---

## 概述

pi-mono 扩展是 TypeScript 模块，通过 ExtensionAPI 扩展 pi 的行为。本文将实现一个完整的 MCP 扩展，连接到 MCP 服务器并包装其工具。

```
Extension 开发核心：
├─ 扩展结构 → TypeScript 模块
├─ MCP 客户端 → 连接服务器
├─ 工具包装 → MCP 工具转 pi 工具
└─ 事件处理 → session_start + tool_call
```

**本质**：pi-mono 扩展是插件系统，通过事件驱动架构和 ExtensionAPI，让开发者能够深度定制 coding agent 的行为。

---

## 项目初始化

### 创建扩展目录

```bash
# 全局扩展（所有项目）
mkdir -p ~/.pi/agent/extensions/mcp-weather

# 或项目本地扩展
mkdir -p .pi/extensions/mcp-weather

cd ~/.pi/agent/extensions/mcp-weather
```

### 初始化 package.json

```json
{
  "name": "mcp-weather-extension",
  "version": "1.0.0",
  "type": "module",
  "main": "index.ts",
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0"
  },
  "peerDependencies": {
    "@mariozechner/pi-coding-agent": "*"
  }
}
```

---

## 完整扩展实现

### 基础扩展结构

创建 `index.ts`:

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { Type } from "@sinclair/typebox";

export default function (pi: ExtensionAPI) {
  let mcpClient: Client | null = null;
  let transport: StdioClientTransport | null = null;

  // 会话启动时连接 MCP 服务器
  pi.on("session_start", async (_event, ctx) => {
    try {
      // 创建 MCP 客户端
      mcpClient = new Client({
        name: "pi-mcp-weather",
        version: "1.0.0",
      });

      // 创建 stdio 传输
      transport = new StdioClientTransport({
        command: "node",
        args: ["/path/to/weather-mcp-server/build/index.js"],
      });

      // 连接到服务器
      await mcpClient.connect(transport);

      // 获取可用工具
      const toolsResult = await mcpClient.listTools();
      ctx.ui.notify(
        `Connected to MCP server with ${toolsResult.tools.length} tools`,
        "info"
      );

      // 注册每个 MCP 工具为 pi 工具
      for (const mcpTool of toolsResult.tools) {
        registerMCPTool(pi, mcpClient, mcpTool);
      }
    } catch (error) {
      ctx.ui.notify(
        `Failed to connect to MCP server: ${error}`,
        "error"
      );
    }
  });

  // 会话关闭时断开连接
  pi.on("session_shutdown", async (_event, ctx) => {
    if (mcpClient) {
      await mcpClient.close();
      mcpClient = null;
      transport = null;
    }
  });
}

/**
 * 注册 MCP 工具为 pi 工具
 */
function registerMCPTool(
  pi: ExtensionAPI,
  mcpClient: Client,
  mcpTool: any
) {
  // 转换 MCP Schema 到 TypeBox Schema
  const schema = convertMCPSchemaToTypeBox(mcpTool.inputSchema);

  // 注册 pi 工具
  pi.registerTool({
    name: `mcp_${mcpTool.name}`,
    label: mcpTool.title || mcpTool.name,
    description: mcpTool.description,
    parameters: schema,

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      try {
        // 调用 MCP 工具
        const result = await mcpClient.callTool({
          name: mcpTool.name,
          arguments: params,
        });

        // 转换结果格式
        return {
          content: result.content.map((item: any) => ({
            type: item.type,
            text: item.type === "text" ? item.text : undefined,
            data: item.type === "image" ? item.data : undefined,
            mimeType: item.mimeType,
          })),
          details: { mcpResult: result },
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `MCP tool error: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    },
  });
}

/**
 * 转换 MCP Schema 到 TypeBox Schema
 */
function convertMCPSchemaToTypeBox(mcpSchema: any): any {
  if (mcpSchema.type === "object") {
    const properties: Record<string, any> = {};

    for (const [key, value] of Object.entries(mcpSchema.properties || {})) {
      const prop = value as any;
      let typeboxType = convertType(prop);

      // 处理可选字段
      if (!mcpSchema.required?.includes(key)) {
        typeboxType = Type.Optional(typeboxType);
      }

      properties[key] = typeboxType;
    }

    return Type.Object(properties);
  }

  return convertType(mcpSchema);
}

/**
 * 转换单个类型
 */
function convertType(prop: any): any {
  switch (prop.type) {
    case "string":
      return Type.String({
        description: prop.description,
        minLength: prop.minLength,
        maxLength: prop.maxLength,
      });

    case "number":
      return Type.Number({
        description: prop.description,
        minimum: prop.minimum,
        maximum: prop.maximum,
      });

    case "integer":
      return Type.Integer({
        description: prop.description,
        minimum: prop.minimum,
        maximum: prop.maximum,
      });

    case "boolean":
      return Type.Boolean({
        description: prop.description,
      });

    case "array":
      return Type.Array(convertType(prop.items), {
        description: prop.description,
        minItems: prop.minItems,
        maxItems: prop.maxItems,
      });

    case "object":
      return convertMCPSchemaToTypeBox(prop);

    default:
      return Type.Any();
  }
}
```

---

## 高级特性

### 工具调用拦截

```typescript
// 拦截工具调用，添加确认
pi.on("tool_call", async (event, ctx) => {
  if (event.toolName.startsWith("mcp_")) {
    const ok = await ctx.ui.confirm(
      "MCP Tool Call",
      `Allow calling ${event.toolName}?`
    );

    if (!ok) {
      return { block: true, reason: "User denied" };
    }
  }
});
```

### 结果转换

```typescript
// 转换工具结果
pi.on("tool_result", async (event, ctx) => {
  if (event.toolName.startsWith("mcp_")) {
    // 添加额外信息
    return {
      content: [
        ...event.content,
        {
          type: "text",
          text: `\n[MCP Tool: ${event.toolName}]`,
        },
      ],
      details: {
        ...event.details,
        source: "mcp-server",
      },
    };
  }
});
```

### 状态持久化

```typescript
// 保存 MCP 连接状态
pi.on("session_start", async (_event, ctx) => {
  // 从会话恢复状态
  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type === "custom" && entry.customType === "mcp-state") {
      const state = entry.data;
      // 恢复连接
    }
  }
});

// 保存状态
pi.appendEntry("mcp-state", {
  serverPath: "/path/to/server",
  connectedAt: Date.now(),
});
```

### 自定义命令

```typescript
// 注册命令
pi.registerCommand("mcp-status", {
  description: "Show MCP server status",
  handler: async (args, ctx) => {
    if (!mcpClient) {
      ctx.ui.notify("MCP server not connected", "warning");
      return;
    }

    const toolsResult = await mcpClient.listTools();
    ctx.ui.notify(
      `Connected with ${toolsResult.tools.length} tools`,
      "info"
    );
  },
});

// 使用：/mcp-status
```

---

## 测试与调试

### 本地测试

```bash
# 1. 安装依赖
cd ~/.pi/agent/extensions/mcp-weather
npm install

# 2. 启动 pi
pi

# 3. 在 pi 中重载扩展
/reload

# 4. 测试工具
# 扩展会自动注册 MCP 工具
```

### 调试日志

```typescript
// 添加调试日志
pi.on("session_start", async (_event, ctx) => {
  console.error("[MCP Extension] Connecting to server...");

  try {
    await mcpClient.connect(transport);
    console.error("[MCP Extension] Connected successfully");
  } catch (error) {
    console.error("[MCP Extension] Connection failed:", error);
  }
});
```

### 错误处理

```typescript
// 完整的错误处理
async function connectToMCPServer(ctx: any) {
  try {
    mcpClient = new Client({
      name: "pi-mcp-weather",
      version: "1.0.0",
    });

    transport = new StdioClientTransport({
      command: "node",
      args: ["/path/to/server/build/index.js"],
    });

    await mcpClient.connect(transport);

    const toolsResult = await mcpClient.listTools();
    console.error(`[MCP] Connected with ${toolsResult.tools.length} tools`);

    return true;
  } catch (error) {
    console.error("[MCP] Connection error:", error);
    ctx.ui.notify(
      `MCP connection failed: ${error instanceof Error ? error.message : String(error)}`,
      "error"
    );
    return false;
  }
}
```

---

## 最佳实践

### 扩展结构

```typescript
// ✅ 清晰的模块结构
export default function (pi: ExtensionAPI) {
  // 状态管理
  let mcpClient: Client | null = null;

  // 事件处理
  pi.on("session_start", async (_event, ctx) => {
    // 初始化
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    // 清理
  });

  // 工具注册
  // 命令注册
}
```

### 错误处理

```typescript
// ✅ 完整的错误处理
try {
  const result = await mcpClient.callTool({
    name: toolName,
    arguments: params,
  });
  return convertResult(result);
} catch (error) {
  console.error(`Tool execution failed:`, error);
  return {
    content: [{
      type: "text",
      text: `Error: ${error instanceof Error ? error.message : "Unknown error"}`,
    }],
    isError: true,
  };
}
```

### 资源管理

```typescript
// ✅ 优雅关闭
pi.on("session_shutdown", async (_event, ctx) => {
  if (mcpClient) {
    try {
      await mcpClient.close();
      console.error("[MCP] Disconnected");
    } catch (error) {
      console.error("[MCP] Disconnect error:", error);
    } finally {
      mcpClient = null;
      transport = null;
    }
  }
});
```

### 用户体验

```typescript
// ✅ 提供反馈
pi.on("session_start", async (_event, ctx) => {
  ctx.ui.notify("Connecting to MCP server...", "info");

  const connected = await connectToMCPServer(ctx);

  if (connected) {
    ctx.ui.notify("MCP server connected", "success");
  } else {
    ctx.ui.notify("MCP server connection failed", "error");
  }
});
```

---

## 总结

### 核心要点

1. **扩展结构**：导出默认函数，接收 ExtensionAPI
2. **MCP 客户端**：使用 @modelcontextprotocol/sdk 连接服务器
3. **工具包装**：转换 MCP Schema 到 TypeBox Schema
4. **事件处理**：session_start 连接，session_shutdown 断开
5. **错误处理**：完整的 try-catch 和用户通知

### 关键约束

- ✅ 使用 peerDependencies 声明核心包
- ✅ 在 session_start 连接 MCP 服务器
- ✅ 在 session_shutdown 清理资源
- ✅ 转换 MCP 工具为 pi 工具
- ✅ 提供完整的错误处理

### 下一步

- 阅读 [07_实战代码_06_MCP_Client集成](./07_实战代码_06_MCP_Client集成.md) 学习客户端集成
- 阅读 [07_实战代码_07_工具包装与转换](./07_实战代码_07_工具包装与转换.md) 学习工具包装

---

**参考资源**：
- [pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
