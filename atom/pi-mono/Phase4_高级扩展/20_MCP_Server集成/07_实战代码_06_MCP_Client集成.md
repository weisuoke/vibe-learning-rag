# 实战代码 06：MCP Client 集成

> **在 pi-mono 扩展中集成 MCP Client，实现服务器发现和工具调用**

---

## 概述

MCP Client 是连接 MCP Server 的客户端实现，负责管理服务器连接、工具发现、请求处理和生命周期管理。本文实现一个完整的 MCP Client 集成方案。

```
MCP Client 集成核心：
├─ 连接管理 → stdio + HTTP 传输
├─ 工具发现 → listTools + 动态注册
├─ 请求处理 → callTool + 结果转换
└─ 生命周期 → 初始化 + 清理
```

**本质**：MCP Client 是 pi-mono 扩展与 MCP Server 之间的桥梁，通过标准化的协议实现工具发现、调用和结果处理。

---

## 完整 Client 实现

### Client 封装类

创建 `src/mcp-client.ts`:

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { Tool } from "@modelcontextprotocol/sdk/types.js";

export interface MCPClientConfig {
  name: string;
  version: string;
  serverCommand: string;
  serverArgs: string[];
  serverEnv?: Record<string, string>;
}

export class MCPClientManager {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private tools: Tool[] = [];
  private connected = false;

  constructor(private config: MCPClientConfig) {}

  /**
   * 连接到 MCP 服务器
   */
  async connect(): Promise<void> {
    if (this.connected) {
      console.error("[MCP Client] Already connected");
      return;
    }

    try {
      // 创建客户端
      this.client = new Client({
        name: this.config.name,
        version: this.config.version,
      });

      // 创建传输层
      this.transport = new StdioClientTransport({
        command: this.config.serverCommand,
        args: this.config.serverArgs,
        env: this.config.serverEnv,
      });

      // 连接
      await this.client.connect(this.transport);
      this.connected = true;

      // 发现工具
      await this.discoverTools();

      console.error(
        `[MCP Client] Connected to server with ${this.tools.length} tools`
      );
    } catch (error) {
      console.error("[MCP Client] Connection failed:", error);
      throw error;
    }
  }

  /**
   * 发现可用工具
   */
  private async discoverTools(): Promise<void> {
    if (!this.client) {
      throw new Error("Client not initialized");
    }

    const result = await this.client.listTools();
    this.tools = result.tools;
  }

  /**
   * 获取工具列表
   */
  getTools(): Tool[] {
    return this.tools;
  }

  /**
   * 调用工具
   */
  async callTool(name: string, args: any): Promise<any> {
    if (!this.client) {
      throw new Error("Client not connected");
    }

    try {
      const result = await this.client.callTool({
        name,
        arguments: args,
      });

      return result;
    } catch (error) {
      console.error(`[MCP Client] Tool call failed: ${name}`, error);
      throw error;
    }
  }

  /**
   * 断开连接
   */
  async disconnect(): Promise<void> {
    if (!this.connected) {
      return;
    }

    try {
      if (this.client) {
        await this.client.close();
      }

      this.client = null;
      this.transport = null;
      this.tools = [];
      this.connected = false;

      console.error("[MCP Client] Disconnected");
    } catch (error) {
      console.error("[MCP Client] Disconnect error:", error);
    }
  }

  /**
   * 检查连接状态
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * 重新连接
   */
  async reconnect(): Promise<void> {
    await this.disconnect();
    await this.connect();
  }
}
```

### 扩展集成

创建 `index.ts`:

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { MCPClientManager } from "./mcp-client.js";
import { Type } from "@sinclair/typebox";

export default function (pi: ExtensionAPI) {
  const clients = new Map<string, MCPClientManager>();

  // 配置多个 MCP 服务器
  const serverConfigs = [
    {
      id: "weather",
      name: "weather-client",
      version: "1.0.0",
      serverCommand: "node",
      serverArgs: ["/path/to/weather-server/build/index.js"],
    },
    {
      id: "github",
      name: "github-client",
      version: "1.0.0",
      serverCommand: "node",
      serverArgs: ["/path/to/github-server/build/index.js"],
      serverEnv: {
        GITHUB_TOKEN: process.env.GITHUB_TOKEN || "",
      },
    },
  ];

  // 会话启动时连接所有服务器
  pi.on("session_start", async (_event, ctx) => {
    for (const config of serverConfigs) {
      try {
        const client = new MCPClientManager(config);
        await client.connect();

        clients.set(config.id, client);

        // 注册该服务器的所有工具
        const tools = client.getTools();
        for (const tool of tools) {
          registerTool(pi, config.id, client, tool);
        }

        ctx.ui.notify(
          `Connected to ${config.id} server (${tools.length} tools)`,
          "success"
        );
      } catch (error) {
        ctx.ui.notify(
          `Failed to connect to ${config.id}: ${error}`,
          "error"
        );
      }
    }
  });

  // 会话关闭时断开所有连接
  pi.on("session_shutdown", async (_event, ctx) => {
    for (const [id, client] of clients.entries()) {
      await client.disconnect();
      clients.delete(id);
    }
  });

  // 注册管理命令
  pi.registerCommand("mcp-status", {
    description: "Show MCP servers status",
    handler: async (args, ctx) => {
      const status = Array.from(clients.entries()).map(([id, client]) => {
        const tools = client.getTools();
        return `${id}: ${client.isConnected() ? "✓" : "✗"} (${tools.length} tools)`;
      });

      ctx.ui.notify(status.join("\n"), "info");
    },
  });

  pi.registerCommand("mcp-reconnect", {
    description: "Reconnect to MCP servers",
    handler: async (args, ctx) => {
      const serverId = args || "";

      if (serverId) {
        const client = clients.get(serverId);
        if (client) {
          await client.reconnect();
          ctx.ui.notify(`Reconnected to ${serverId}`, "success");
        } else {
          ctx.ui.notify(`Server not found: ${serverId}`, "error");
        }
      } else {
        for (const [id, client] of clients.entries()) {
          await client.reconnect();
        }
        ctx.ui.notify("Reconnected to all servers", "success");
      }
    },
  });
}

/**
 * 注册单个工具
 */
function registerTool(
  pi: ExtensionAPI,
  serverId: string,
  client: MCPClientManager,
  tool: any
) {
  pi.registerTool({
    name: `${serverId}_${tool.name}`,
    label: tool.title || tool.name,
    description: `[${serverId}] ${tool.description}`,
    parameters: convertSchema(tool.inputSchema),

    async execute(toolCallId, params, signal, onUpdate, ctx) {
      try {
        const result = await client.callTool(tool.name, params);

        return {
          content: result.content.map((item: any) => ({
            type: item.type,
            text: item.type === "text" ? item.text : undefined,
            data: item.type === "image" ? item.data : undefined,
            mimeType: item.mimeType,
          })),
          details: {
            server: serverId,
            tool: tool.name,
          },
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Error calling ${serverId}.${tool.name}: ${error}`,
            },
          ],
          isError: true,
        };
      }
    },
  });
}

/**
 * 转换 Schema
 */
function convertSchema(schema: any): any {
  if (!schema || schema.type !== "object") {
    return Type.Object({});
  }

  const properties: Record<string, any> = {};

  for (const [key, value] of Object.entries(schema.properties || {})) {
    const prop = value as any;
    let typeboxType: any;

    switch (prop.type) {
      case "string":
        typeboxType = Type.String({ description: prop.description });
        break;
      case "number":
        typeboxType = Type.Number({ description: prop.description });
        break;
      case "boolean":
        typeboxType = Type.Boolean({ description: prop.description });
        break;
      default:
        typeboxType = Type.Any();
    }

    if (!schema.required?.includes(key)) {
      typeboxType = Type.Optional(typeboxType);
    }

    properties[key] = typeboxType;
  }

  return Type.Object(properties);
}
```

---

## 高级特性

### 连接池管理

```typescript
export class MCPClientPool {
  private clients = new Map<string, MCPClientManager>();

  async addClient(id: string, config: MCPClientConfig): Promise<void> {
    if (this.clients.has(id)) {
      throw new Error(`Client ${id} already exists`);
    }

    const client = new MCPClientManager(config);
    await client.connect();
    this.clients.set(id, client);
  }

  async removeClient(id: string): Promise<void> {
    const client = this.clients.get(id);
    if (client) {
      await client.disconnect();
      this.clients.delete(id);
    }
  }

  getClient(id: string): MCPClientManager | undefined {
    return this.clients.get(id);
  }

  getAllClients(): Map<string, MCPClientManager> {
    return this.clients;
  }

  async disconnectAll(): Promise<void> {
    for (const client of this.clients.values()) {
      await client.disconnect();
    }
    this.clients.clear();
  }
}
```

### 自动重连

```typescript
export class ResilientMCPClient extends MCPClientManager {
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 1000;

  async connect(): Promise<void> {
    try {
      await super.connect();
      this.reconnectAttempts = 0;
    } catch (error) {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.error(
          `[MCP Client] Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`
        );

        await new Promise((resolve) =>
          setTimeout(resolve, this.reconnectDelay * this.reconnectAttempts)
        );

        return this.connect();
      }

      throw error;
    }
  }
}
```

### 工具缓存

```typescript
export class CachedMCPClient extends MCPClientManager {
  private cache = new Map<string, { result: any; timestamp: number }>();
  private cacheTTL = 5 * 60 * 1000; // 5 分钟

  async callTool(name: string, args: any): Promise<any> {
    const cacheKey = `${name}:${JSON.stringify(args)}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      console.error(`[MCP Client] Cache hit: ${cacheKey}`);
      return cached.result;
    }

    const result = await super.callTool(name, args);
    this.cache.set(cacheKey, { result, timestamp: Date.now() });

    return result;
  }
}
```

---

## 测试与调试

### 本地测试

```bash
# 1. 启动 MCP 服务器
cd /path/to/weather-server
npm start &

# 2. 启动 pi
pi

# 3. 测试连接
/mcp-status

# 4. 测试工具调用
# 工具会自动注册为 weather_get_forecast 等
```

### 调试日志

```typescript
// 添加详细日志
export class DebugMCPClient extends MCPClientManager {
  async connect(): Promise<void> {
    console.error("[MCP Client] Connecting...");
    console.error("[MCP Client] Config:", this.config);

    try {
      await super.connect();
      console.error("[MCP Client] Connected successfully");
    } catch (error) {
      console.error("[MCP Client] Connection failed:", error);
      throw error;
    }
  }

  async callTool(name: string, args: any): Promise<any> {
    console.error(`[MCP Client] Calling tool: ${name}`);
    console.error(`[MCP Client] Args:`, args);

    const result = await super.callTool(name, args);

    console.error(`[MCP Client] Result:`, result);
    return result;
  }
}
```

---

## 最佳实践

### 错误处理

```typescript
// ✅ 完整的错误处理
async connect(): Promise<void> {
  try {
    await this.client.connect(this.transport);
    this.connected = true;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("ENOENT")) {
        throw new Error("Server executable not found");
      } else if (error.message.includes("ECONNREFUSED")) {
        throw new Error("Server refused connection");
      }
    }
    throw error;
  }
}
```

### 资源管理

```typescript
// ✅ 优雅关闭
async disconnect(): Promise<void> {
  if (!this.connected) {
    return;
  }

  try {
    if (this.client) {
      await this.client.close();
    }
  } catch (error) {
    console.error("[MCP Client] Disconnect error:", error);
  } finally {
    this.client = null;
    this.transport = null;
    this.connected = false;
  }
}
```

### 超时处理

```typescript
// ✅ 添加超时
async callToolWithTimeout(
  name: string,
  args: any,
  timeout: number = 30000
): Promise<any> {
  return Promise.race([
    this.callTool(name, args),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Tool call timeout")), timeout)
    ),
  ]);
}
```

---

## 总结

### 核心要点

1. **Client 封装**：MCPClientManager 管理连接和工具
2. **连接管理**：stdio 传输 + 自动重连
3. **工具发现**：listTools + 动态注册
4. **生命周期**：session_start 连接，session_shutdown 断开
5. **多服务器**：支持同时连接多个 MCP 服务器

### 关键约束

- ✅ 封装 Client 逻辑到独立类
- ✅ 在 session_start 连接服务器
- ✅ 在 session_shutdown 清理资源
- ✅ 提供重连和状态查询命令
- ✅ 完整的错误处理和日志

### 下一步

- 阅读 [07_实战代码_07_工具包装与转换](./07_实战代码_07_工具包装与转换.md) 学习工具包装
- 阅读 [07_实战代码_08_生产环境部署](./07_实战代码_08_生产环境部署.md) 学习生产部署

---

**参考资源**：
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [pi-mono Extensions Documentation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md)
