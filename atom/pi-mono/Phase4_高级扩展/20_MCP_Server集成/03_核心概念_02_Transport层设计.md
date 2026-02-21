# 核心概念 02：Transport 层设计

> **深入理解 MCP 的传输层机制：stdio 和 Streamable HTTP**

---

## 传输层概述

MCP 协议定义了两种标准传输机制，用于客户端和服务器之间的通信：

1. **stdio**：通过标准输入输出进行通信
2. **Streamable HTTP**：基于 HTTP 的传输（替代了旧的 HTTP+SSE）

**核心要求**：
- 所有 JSON-RPC 消息**必须**使用 UTF-8 编码
- 客户端**应该**尽可能支持 stdio 传输
- 客户端和服务器**可以**实现自定义传输机制

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

---

## stdio 传输

### 工作原理

**stdio** 传输是最简单的传输机制，适用于本地进程通信：

```
客户端进程
    ↓ 启动子进程
MCP Server 子进程
    ↓ stdin（接收消息）
    ↓ stdout（发送消息）
    ↓ stderr（日志输出）
```

### 核心规范

**消息传输**：

- 服务器从标准输入（stdin）读取 JSON-RPC 消息
- 服务器向标准输出（stdout）发送 JSON-RPC 消息
- 消息是独立的 JSON-RPC 请求、通知或响应
- 消息以换行符分隔，**不得**包含嵌入的换行符

**日志输出**：

- 服务器**可以**向标准错误（stderr）写入 UTF-8 字符串用于日志
- 客户端**可以**捕获、转发或忽略服务器的 stderr 输出
- 客户端**不应**假设 stderr 输出表示错误条件

**严格约束**：

- ✅ 服务器**不得**向 stdout 写入任何非有效 MCP 消息的内容
- ✅ 客户端**不得**向服务器的 stdin 写入任何非有效 MCP 消息的内容

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 消息格式示例

```typescript
// 客户端通过 stdin 发送请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
}\n

// 服务器通过 stdout 返回响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [...]
  }
}\n

// 服务器通过 stderr 输出日志
[INFO] Tool list requested\n
```

### 实现示例

**服务器端（TypeScript）**：

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server(
  { name: "my-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 注册处理器
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [/* ... */]
}));

// 连接 stdio 传输
const transport = new StdioServerTransport();
await server.connect(transport);

// 日志输出到 stderr
console.error("[INFO] Server started");  // ✅ 正确
// console.log("[INFO] Server started");  // ❌ 错误：会破坏协议
```

**客户端端（TypeScript）**：

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const client = new Client(
  { name: "my-client", version: "1.0.0" },
  { capabilities: {} }
);

// 连接到 MCP Server（作为子进程）
const transport = new StdioClientTransport({
  command: "node",
  args: ["./dist/server.js"]
});

await client.connect(transport);

// 调用工具
const { tools } = await client.listTools();
```

### 优势与限制

**优势**：

- ✅ 实现简单，无需网络配置
- ✅ 适合本地工具和命令行应用
- ✅ 进程隔离，安全性好
- ✅ 无需处理 HTTP 复杂性

**限制**：

- ❌ 仅适用于本地进程
- ❌ 不支持远程访问
- ❌ 不支持多客户端连接
- ❌ 进程管理复杂度

---

## Streamable HTTP 传输

### 协议演进

**重要变更**：Streamable HTTP 替代了协议版本 2024-11-05 中的 HTTP+SSE 传输。

**为什么改变**：

1. **安全性增强**：更好的 Origin 验证和 DNS 重绑定防护
2. **连接管理**：改进的会话管理和连接恢复
3. **可扩展性**：支持多客户端和无状态部署
4. **简化设计**：单一端点，更清晰的语义

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 核心设计

**单一端点哲学**：

服务器**必须**提供单一 HTTP 端点路径（MCP 端点），支持 POST 和 GET 方法。

```
MCP 端点示例：https://example.com/mcp

支持的方法：
├─ POST：发送消息到服务器
└─ GET：监听服务器消息（SSE 流）
```

### 安全警告

实现 Streamable HTTP 传输时的**强制安全要求**：

1. **Origin 验证**：
   - 服务器**必须**验证所有传入连接的 Origin 头，防止 DNS 重绑定攻击
   - 如果 Origin 头存在但无效，服务器**必须**返回 HTTP 403 Forbidden

2. **本地绑定**：
   - 本地运行时，服务器**应该**仅绑定到 localhost (127.0.0.1)，而不是所有网络接口 (0.0.0.0)

3. **认证**：
   - 服务器**应该**为所有连接实现适当的认证

**安全风险**：没有这些保护，攻击者可以使用 DNS 重绑定从远程网站与本地 MCP 服务器交互。

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 发送消息到服务器（POST）

**核心流程**：

客户端发送的每个 JSON-RPC 消息**必须**是一个新的 HTTP POST 请求。

**请求规范**：

```http
POST /mcp HTTP/1.1
Host: example.com
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Session-Id: <session-id>
MCP-Protocol-Version: 2025-11-25

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": { ... }
}
```

**响应类型**：

1. **JSON-RPC 响应或通知**：
   - 服务器接受：返回 HTTP 202 Accepted，无响应体
   - 服务器拒绝：返回 HTTP 错误状态码（如 400 Bad Request）

2. **JSON-RPC 请求**：
   - 服务器**必须**返回 `Content-Type: text/event-stream`（SSE 流）或 `Content-Type: application/json`（单个 JSON 对象）
   - 客户端**必须**支持这两种情况

**SSE 流式响应**：

当服务器启动 SSE 流时：

```
服务器行为：
├─ 立即发送 SSE 事件（事件 ID + 空数据）以准备重连
├─ 可以在任何时候关闭连接（不终止流）
├─ 应该在关闭前发送 retry 字段
├─ 流应该最终包含 JSON-RPC 响应
├─ 可以在响应前发送请求和通知
└─ 响应后应该终止 SSE 流
```

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 监听服务器消息（GET）

**核心流程**：

客户端**可以**向 MCP 端点发出 HTTP GET 请求，打开 SSE 流以监听服务器消息。

**请求规范**：

```http
GET /mcp HTTP/1.1
Host: example.com
Accept: text/event-stream
MCP-Session-Id: <session-id>
MCP-Protocol-Version: 2025-11-25
```

**响应规范**：

- 服务器**必须**返回 `Content-Type: text/event-stream`，或返回 HTTP 405 Method Not Allowed

**SSE 流行为**：

```
服务器可以：
├─ 发送 JSON-RPC 请求和通知
├─ 这些消息应该与任何并发客户端请求无关
├─ 不得发送 JSON-RPC 响应（除非恢复流）
└─ 随时关闭 SSE 流

客户端可以：
└─ 随时关闭 SSE 流
```

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 会话管理

**会话概念**：

MCP "会话"由客户端和服务器之间逻辑相关的交互组成，从初始化阶段开始。

**会话 ID 机制**：

1. **分配**：
   - 服务器**可以**在初始化时分配会话 ID
   - 通过在 InitializeResult 的 HTTP 响应中包含 `MCP-Session-Id` 头

2. **会话 ID 要求**：
   - **应该**是全局唯一且加密安全的（如 UUID、JWT、加密哈希）
   - **必须**仅包含可见 ASCII 字符（0x21 到 0x7E）
   - 客户端**必须**以安全方式处理会话 ID

3. **使用**：
   - 如果服务器返回会话 ID，客户端**必须**在所有后续 HTTP 请求中包含它
   - 服务器**应该**对没有会话 ID 的请求（初始化除外）返回 HTTP 400 Bad Request

4. **终止**：
   - 服务器**可以**随时终止会话，之后**必须**对包含该会话 ID 的请求返回 HTTP 404 Not Found
   - 客户端收到 HTTP 404 时**必须**通过发送新的 InitializeRequest 开始新会话
   - 客户端**应该**通过向 MCP 端点发送 HTTP DELETE 显式终止会话

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 多连接支持

**并发连接**：

1. 客户端**可以**同时保持多个 SSE 流连接
2. 服务器**必须**仅在一个连接流上发送每个 JSON-RPC 消息
   - **不得**在多个流上广播相同消息
3. 消息丢失风险**可以**通过使流可恢复来缓解

### 可恢复性与重新交付

**恢复机制**：

服务器**可以**为 SSE 事件附加 ID 字段：

```
事件 ID 要求：
├─ 必须在会话内全局唯一
├─ 应该编码足够信息以识别原始流
└─ 用于关联 Last-Event-ID 到正确的流
```

**恢复流程**：

```typescript
// 客户端恢复断开的连接
GET /mcp HTTP/1.1
Host: example.com
Accept: text/event-stream
Last-Event-ID: <last-received-event-id>
MCP-Session-Id: <session-id>
```

**服务器行为**：

- **可以**使用 Last-Event-ID 头重放消息
- **必须**仅重放断开流上的消息
- **不得**重放其他流上的消息

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

### 协议版本头

**强制要求**：

使用 HTTP 时，客户端**必须**在所有后续请求中包含 `MCP-Protocol-Version` HTTP 头：

```http
MCP-Protocol-Version: 2025-11-25
```

**版本协商**：

- 协议版本**应该**是初始化期间协商的版本
- 如果服务器未收到版本头且无法识别版本，**应该**假设协议版本 `2025-03-26`
- 如果服务器收到无效或不支持的版本，**必须**返回 `400 Bad Request`

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

---

## 传输层对比

### stdio vs Streamable HTTP

| 维度 | stdio | Streamable HTTP |
|------|-------|----------------|
| **使用场景** | 本地工具、CLI 应用 | 远程服务、Web 应用 |
| **连接方式** | 子进程 | HTTP 连接 |
| **多客户端** | 不支持 | 支持 |
| **网络访问** | 否 | 是 |
| **实现复杂度** | 低 | 中等 |
| **安全考虑** | 进程隔离 | Origin 验证、认证 |
| **会话管理** | 隐式（进程生命周期） | 显式（Session ID） |
| **流式支持** | 天然支持 | 通过 SSE 支持 |
| **恢复能力** | 不支持 | 支持（Last-Event-ID） |

### 选择建议

**使用 stdio 当**：

- ✅ 构建本地命令行工具
- ✅ 需要简单的进程隔离
- ✅ 不需要远程访问
- ✅ 单客户端场景

**使用 Streamable HTTP 当**：

- ✅ 需要远程访问
- ✅ 支持多客户端连接
- ✅ 需要会话管理
- ✅ 需要连接恢复能力
- ✅ 构建 Web 服务

---

## 实现最佳实践

### stdio 实现

**服务器端**：

```typescript
// ✅ 正确：使用 stderr 输出日志
console.error(`[${new Date().toISOString()}] Server started`);

// ❌ 错误：使用 stdout 会破坏协议
// console.log("Server started");

// ✅ 正确：消息不包含换行符
const message = JSON.stringify({ jsonrpc: "2.0", ... });
process.stdout.write(message + "\n");
```

**客户端端**：

```typescript
// ✅ 正确：捕获 stderr 用于日志
serverProcess.stderr.on("data", (data) => {
  logger.info(`Server log: ${data}`);
});

// ✅ 正确：解析 stdout 为 JSON-RPC 消息
serverProcess.stdout.on("data", (data) => {
  const messages = data.toString().split("\n").filter(Boolean);
  messages.forEach(msg => handleMessage(JSON.parse(msg)));
});
```

### Streamable HTTP 实现

**安全配置**：

```typescript
import express from "express";

const app = express();

// ✅ 正确：验证 Origin
app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin && !isAllowedOrigin(origin)) {
    return res.status(403).json({
      jsonrpc: "2.0",
      error: {
        code: -32600,
        message: "Invalid origin"
      }
    });
  }
  next();
});

// ✅ 正确：仅绑定到 localhost
app.listen(3000, "127.0.0.1", () => {
  console.log("Server listening on localhost:3000");
});
```

**会话管理**：

```typescript
// 服务器端：分配会话 ID
app.post("/mcp", async (req, res) => {
  if (req.body.method === "initialize") {
    const sessionId = crypto.randomUUID();
    sessions.set(sessionId, { initialized: true });

    res.setHeader("MCP-Session-Id", sessionId);
    res.json({
      jsonrpc: "2.0",
      id: req.body.id,
      result: { /* ... */ }
    });
  }
});

// 客户端端：使用会话 ID
const response = await fetch("https://example.com/mcp", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "MCP-Session-Id": sessionId,
    "MCP-Protocol-Version": "2025-11-25"
  },
  body: JSON.stringify({ /* ... */ })
});
```

---

## 向后兼容性

### 支持旧版 HTTP+SSE

**服务器端策略**：

```
支持旧客户端：
├─ 继续托管旧的 SSE 和 POST 端点
├─ 同时提供新的 MCP 端点
└─ 可以合并旧 POST 端点和新 MCP 端点（但增加复杂度）
```

**客户端端策略**：

```
支持旧服务器：
1. 尝试 POST InitializeRequest 到服务器 URL
2. 如果成功 → 使用新的 Streamable HTTP 传输
3. 如果失败（400/404/405）→ 尝试 GET 打开 SSE 流
4. 如果收到 endpoint 事件 → 使用旧的 HTTP+SSE 传输
```

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

---

## 自定义传输

### 扩展性

客户端和服务器**可以**实现额外的自定义传输机制以满足特定需求。

**要求**：

- ✅ **必须**保留 JSON-RPC 消息格式
- ✅ **必须**遵守 MCP 生命周期要求
- ✅ **应该**记录连接建立和消息交换模式

**示例场景**：

- WebSocket 传输（双向实时通信）
- gRPC 传输（高性能 RPC）
- 消息队列传输（异步解耦）
- 自定义协议（特殊网络环境）

[Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

---

## 总结

### 核心要点

1. **两种标准传输**：stdio（本地）和 Streamable HTTP（远程）
2. **stdio 简单高效**：适合本地工具，但仅支持单客户端
3. **Streamable HTTP 功能丰富**：支持多客户端、会话管理、连接恢复
4. **安全至关重要**：Origin 验证、本地绑定、认证
5. **可扩展设计**：支持自定义传输机制

### 关键约束

- ✅ stdio：stdout 仅用于协议，stderr 用于日志
- ✅ HTTP：必须验证 Origin，必须包含协议版本头
- ✅ 会话：使用加密安全的会话 ID
- ✅ 恢复：通过 Last-Event-ID 支持流恢复

### 下一步

- 阅读 [03_核心概念_03_Tools与Resources](./03_核心概念_03_Tools与Resources.md) 了解工具和资源的详细规范
- 阅读 [07_实战代码_01_简单MCP_Server](./07_实战代码_01_简单MCP_Server.md) 查看完整实现示例

---

**参考资源**：
- [Source: Transports - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
