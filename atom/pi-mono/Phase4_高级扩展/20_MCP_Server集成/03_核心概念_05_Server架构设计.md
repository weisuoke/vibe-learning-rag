# 核心概念 05：Server 架构设计

> **深入理解 MCP Server 的生命周期、请求处理和状态管理**

---

## 概述

MCP Server 的架构设计遵循严格的生命周期管理和请求处理模式，确保客户端和服务器之间的可靠通信。

```
MCP Server 架构核心：
├─ 生命周期管理 → 初始化、运行、关闭三阶段
├─ 请求处理模式 → stdin/stdout 或 HTTP 传输
├─ 状态管理策略 → 能力协商、会话管理
└─ 可扩展性设计 → 超时控制、错误处理、日志记录
```

**本质**：MCP Server 是一个遵循严格协议的状态机，通过标准化的生命周期和请求处理模式，为 AI 应用提供可靠的工具和资源访问能力。

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

---

## 服务器生命周期

### 三阶段模型

MCP Server 的生命周期分为三个明确的阶段：

```
生命周期阶段：
1. Initialization（初始化）→ 能力协商和协议版本协定
2. Operation（运行）     → 正常的协议通信
3. Shutdown（关闭）      → 优雅地终止连接
```

**关键约束**：
- 初始化阶段**必须**是客户端和服务器之间的第一次交互
- 在初始化完成前，不能进行其他协议通信（ping 和 logging 除外）
- 关闭阶段使用底层传输机制来终止连接

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

### 初始化阶段详解

#### 1. 客户端发起初始化请求

客户端**必须**发送 `initialize` 请求，包含：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": {
        "listChanged": true
      },
      "sampling": {}
    },
    "clientInfo": {
      "name": "ExampleClient",
      "version": "1.0.0"
    }
  }
}
```

**重要约束**：
- `initialize` 请求**不得**作为 JSON-RPC 批处理的一部分
- 这确保了向后兼容性，因为早期协议版本不支持批处理

#### 2. 服务器响应能力信息

服务器**必须**响应自己的能力和信息：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "logging": {},
      "prompts": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "tools": {
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "ExampleServer",
      "version": "1.0.0"
    },
    "instructions": "Optional instructions for the client"
  }
}
```

#### 3. 客户端发送初始化完成通知

成功初始化后，客户端**必须**发送 `initialized` 通知：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

**通信约束**：
- 客户端**不应该**在服务器响应 `initialize` 请求前发送其他请求（ping 除外）
- 服务器**不应该**在收到 `initialized` 通知前发送其他请求（ping 和 logging 除外）

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

### 版本协商机制

**协商流程**：

1. 客户端在 `initialize` 请求中发送它支持的协议版本（**应该**是最新版本）
2. 如果服务器支持该版本，**必须**响应相同版本
3. 否则，服务器**必须**响应它支持的另一个版本（**应该**是最新版本）
4. 如果客户端不支持服务器响应的版本，**应该**断开连接

**示例场景**：

```
场景 1：版本匹配
客户端请求：2025-03-26
服务器响应：2025-03-26
结果：✅ 继续通信

场景 2：版本不匹配
客户端请求：2025-03-26
服务器响应：2024-11-05（服务器不支持 2025-03-26）
结果：❌ 客户端断开连接（如果不支持 2024-11-05）
```

### 能力协商

客户端和服务器能力决定了会话期间可用的可选协议特性。

**关键能力表**：

| 类别 | 能力 | 描述 |
|------|------|------|
| Client | roots | 提供文件系统根目录的能力 |
| Client | sampling | 支持 LLM 采样请求 |
| Client | experimental | 支持非标准实验性特性 |
| Server | prompts | 提供提示模板 |
| Server | resources | 提供可读资源 |
| Server | tools | 暴露可调用工具 |
| Server | logging | 发出结构化日志消息 |
| Server | completions | 支持参数自动补全 |
| Server | experimental | 支持非标准实验性特性 |

**子能力**：

- `listChanged`：支持列表变更通知（用于 prompts、resources、tools）
- `subscribe`：支持订阅单个项目的变更（仅用于 resources）

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

---

## 请求处理模式

### stdin/stdout 通信机制

MCP Server 主要通过**标准输入输出**（stdin/stdout）进行本地通信。

**为什么使用 stdin/stdout？**

- 本地服务器运行在用户机器上，连接到本地数据库或文件
- 不需要网络调用，使用直接的系统级通信
- 更快、更安全、更灵活

**通信流程**：

```
AI 应用（如 Claude）
    ↓ 写入 stdin
MCP Server（读取输入）
    ↓ 执行逻辑（访问数据库、文件系统等）
MCP Server（写入 stdout）
    ↑ 返回结果
AI 应用（读取输出）
```

**实际示例**：

用户问："Do I have a meeting today?"

1. Claude 发现需要调用 MCP Server
2. Claude 写入 stdin：
   ```json
   {
       "method": "calendar",
       "params": {
           "date": "2025-06-16"
       }
   }
   ```
3. MCP Server 读取输入，查询 Google Calendar
4. MCP Server 写入 stdout：
   ```json
   {
       "result": {
           "meetings": [
               {
                   "title": "Team Sync",
                   "time": "4:00 PM"
               }
           ]
       }
   }
   ```
5. Claude 读取输出，转换为自然语言："Yes, you have a meeting 'Team Sync' today at 4 PM."

[Source: How to Build a Custom MCP Server with TypeScript](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/)

### 服务器实例化模式

#### TypeScript 实现

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

// 创建服务器实例
const server = new McpServer({
  name: "weather",
  version: "1.0.0",
});

// 设置传输层
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
```

**关键点**：
- 使用 `McpServer` 类创建服务器实例
- 使用 `StdioServerTransport` 设置 stdin/stdout 传输
- 使用 `console.error()` 输出日志（不能使用 `console.log()`，会污染 stdout）

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

#### Python 实现

```python
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务器
mcp = FastMCP("weather")

def main():
    # 初始化并运行服务器
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

**关键点**：
- 使用 `FastMCP` 类创建服务器实例
- 使用 `transport="stdio"` 指定传输方式
- 更简洁的 API，适合快速开发

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

### 工具注册模式

#### TypeScript 工具注册

```typescript
import { z } from "zod";

server.registerTool({
  name: "get_alerts",
  description: "Get weather alerts for a US state",
  inputSchema: z.object({
    state: z.string().length(2).describe("Two-letter US state code (e.g. CA, NY)"),
  }),
  handler: async ({ state }) => {
    const url = `${NWS_API_BASE}/alerts/active/area/${state}`;
    const data = await makeNWSRequest(url);

    if (!data || !data.features) {
      return {
        content: [
          {
            type: "text",
            text: "Unable to fetch alerts or no alerts found.",
          },
        ],
      };
    }

    const alerts = data.features.map(formatAlert);
    return {
      content: [
        {
          type: "text",
          text: alerts.join("\n---\n"),
        },
      ],
    };
  },
});
```

**关键点**：
- 使用 `registerTool()` 方法注册工具
- 使用 `zod` 进行输入验证
- 返回结构化的 `content` 数组

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

#### Python 工具注册

```python
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)
```

**关键点**：
- 使用 `@mcp.tool()` 装饰器注册工具
- 使用 Python 类型提示进行参数验证
- 直接返回字符串结果

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 状态管理策略

### 会话状态管理

MCP Server 在运行阶段需要管理会话状态：

**状态管理原则**：

1. **尊重协商的协议版本**：只使用双方都支持的协议特性
2. **仅使用协商的能力**：不要尝试使用未成功协商的能力
3. **维护请求上下文**：跟踪请求 ID 和进度通知

**示例：能力检查**

```typescript
// 检查客户端是否支持 sampling
if (clientCapabilities.sampling) {
  // 可以发送 sampling 请求
  await client.sendSamplingRequest({...});
} else {
  // 不支持，使用替代方案
  console.error("Client does not support sampling");
}
```

### 超时管理

实现**应该**为所有发送的请求建立超时，以防止连接挂起和资源耗尽。

**超时策略**：

```typescript
// 基本超时实现
const TIMEOUT_MS = 30000; // 30 秒

async function sendRequestWithTimeout(request: any) {
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Request timeout")), TIMEOUT_MS);
  });

  try {
    return await Promise.race([
      sendRequest(request),
      timeoutPromise
    ]);
  } catch (error) {
    // 发送取消通知
    await sendCancellationNotification(request.id);
    throw error;
  }
}
```

**进度通知处理**：

实现**可以**在收到进度通知时重置超时时钟，因为这表明工作正在进行。但是，实现**应该**始终强制执行最大超时，无论进度通知如何，以限制行为不当的客户端或服务器的影响。

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

### 错误处理策略

实现**应该**准备处理以下错误情况：

1. **协议版本不匹配**
2. **无法协商所需能力**
3. **请求超时**

**初始化错误示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Unsupported protocol version",
    "data": {
      "supported": ["2024-11-05"],
      "requested": "1.0.0"
    }
  }
}
```

**错误处理最佳实践**：

```typescript
try {
  const result = await executeToolCall(toolName, params);
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(result),
      },
    ],
  };
} catch (error) {
  // 返回工具执行错误（不是协议错误）
  return {
    content: [
      {
        type: "text",
        text: `Error: ${error.message}`,
      },
    ],
    isError: true,
  };
}
```

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

---

## 日志记录最佳实践

### stdio 传输的日志约束

**关键规则**：对于基于 stdio 的服务器，**永远不要**写入 stdout。

**为什么？**

- 写入 stdout 会破坏 JSON-RPC 消息
- stdout 是协议通信通道，不是日志通道

**正确的日志方式**：

```python
import sys
import logging

# ❌ 错误（stdio）
print("Processing request")

# ✅ 正确（stdio）
print("Processing request", file=sys.stderr)

# ✅ 正确（stdio）
logging.info("Processing request")
```

```typescript
// ❌ 错误（stdio）
console.log("Processing request");

// ✅ 正确（stdio）
console.error("Processing request");

// ✅ 正确（stdio）
import winston from "winston";
logger.info("Processing request");
```

**HTTP 传输的日志**：

对于基于 HTTP 的服务器，标准输出日志是可以的，因为它不会干扰 HTTP 响应。

[Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

---

## 关闭阶段处理

### stdio 传输关闭

客户端**应该**通过以下方式启动关闭：

1. 首先，关闭子进程（服务器）的输入流
2. 等待服务器退出，或在合理时间内发送 SIGTERM
3. 如果服务器在 SIGTERM 后的合理时间内未退出，发送 SIGKILL

服务器**可以**通过关闭其输出流并退出来启动关闭。

### HTTP 传输关闭

对于 HTTP 传输，关闭通过关闭相关的 HTTP 连接来指示。

[Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)

---

## 可扩展性考虑

### 异步处理模式

**TypeScript 异步实现**：

```typescript
server.registerTool({
  name: "get_forecast",
  description: "Get weather forecast for a location",
  inputSchema: z.object({
    latitude: z.number(),
    longitude: z.number(),
  }),
  handler: async ({ latitude, longitude }) => {
    // 异步获取数据
    const pointsData = await makeNWSRequest(
      `${NWS_API_BASE}/points/${latitude},${longitude}`
    );

    const forecastData = await makeNWSRequest(
      pointsData.properties.forecast
    );

    return {
      content: [
        {
          type: "text",
          text: formatForecast(forecastData),
        },
      ],
    };
  },
});
```

**Python 异步实现**：

```python
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location."""
    # 异步获取数据
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    return format_forecast(forecast_data)
```

### 资源管理

**连接池管理**：

```typescript
import httpx from "httpx";

// 创建 HTTP 客户端池
const httpClient = new httpx.Client({
  timeout: 30000,
  maxConnections: 10,
});

async function makeNWSRequest(url: string) {
  try {
    const response = await httpClient.get(url, {
      headers: {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
      },
    });
    return response.json();
  } catch (error) {
    console.error(`Request failed: ${error.message}`);
    return null;
  }
}
```

---

## 总结

### 核心要点

1. **三阶段生命周期**：初始化、运行、关闭，严格遵循顺序
2. **能力协商机制**：客户端和服务器协商可用特性
3. **stdin/stdout 通信**：本地服务器使用标准输入输出，快速安全
4. **日志记录约束**：stdio 服务器不能写入 stdout，使用 stderr
5. **超时和错误处理**：防止连接挂起，优雅处理错误

### 关键约束

- ✅ 初始化请求不得作为批处理的一部分
- ✅ 客户端必须发送 initialized 通知
- ✅ stdio 服务器不能写入 stdout
- ✅ 实现应该建立请求超时
- ✅ 尊重协商的协议版本和能力

### 下一步

- 阅读 [03_核心概念_06_Tool实现模式](./03_核心概念_06_Tool实现模式.md) 了解工具实现
- 阅读 [07_实战代码_01_简单MCP_Server](./07_实战代码_01_简单MCP_Server.md) 查看完整实现

---

**参考资源**：
- [Source: Lifecycle - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)
- [Source: Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)
- [Source: How to Build a Custom MCP Server with TypeScript](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/)
