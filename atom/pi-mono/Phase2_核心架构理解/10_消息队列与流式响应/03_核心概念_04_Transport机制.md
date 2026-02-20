# 核心概念 04：Transport 机制

> **核心价值**: Transport 机制决定了 Agent 与用户之间的通信方式，SSE 适合单向流式传输，WebSocket 适合双向实时交互，HTTP Streaming 是 2025 新标准，选择合适的 Transport 是构建高性能 Agent 的关键。

---

## 概述

在 AI Agent 系统中，Transport 机制负责在服务端和客户端之间传输消息。主要有三种 Transport 方式：

1. **SSE (Server-Sent Events)**：单向流式传输，服务端 → 客户端
2. **WebSocket**：双向实时通信，服务端 ↔ 客户端
3. **HTTP Streaming**：2025 新标准，轻量级流式传输

选择合适的 Transport 机制可以显著影响 Agent 的性能、用户体验和系统复杂度。

---

## 1. SSE (Server-Sent Events)

### 1.1 定义

**SSE (Server-Sent Events)** 是一种基于 HTTP 的单向流式传输协议，允许服务端主动向客户端推送数据。

**核心特征**：
- **单向传输**：服务端 → 客户端（客户端不能通过 SSE 发送数据）
- **基于 HTTP**：使用标准 HTTP 协议，无需特殊服务器支持
- **自动重连**：连接断开后自动重连
- **事件流格式**：使用简单的文本格式传输数据
- **轻量级**：比 WebSocket 更轻量，适合单向流式场景

### 1.2 工作原理

```typescript
// SSE 工作流程

// 1. 客户端发起 HTTP 请求
GET /api/stream HTTP/1.1
Accept: text/event-stream

// 2. 服务端返回 SSE 响应
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

// 3. 服务端持续推送事件
data: {"type": "start", "content": "开始生成响应"}

data: {"type": "delta", "content": "这是"}

data: {"type": "delta", "content": "一个"}

data: {"type": "delta", "content": "示例"}

data: {"type": "end", "content": ""}
```

### 1.3 TypeScript 实现

#### 服务端实现（Node.js + Express）

```typescript
import express from 'express';

const app = express();

// SSE 端点
app.get('/api/stream', (req, res) => {
  // 1. 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  // 2. 发送事件的辅助函数
  const sendEvent = (event: string, data: any) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  // 3. 开始流式传输
  sendEvent('start', { message: '开始生成响应' });

  // 4. 模拟逐字生成
  const text = '这是一个流式响应示例';
  let index = 0;

  const interval = setInterval(() => {
    if (index < text.length) {
      sendEvent('delta', { content: text[index] });
      index++;
    } else {
      sendEvent('end', { message: '完成' });
      clearInterval(interval);
      res.end();
    }
  }, 100);

  // 5. 处理客户端断开连接
  req.on('close', () => {
    clearInterval(interval);
    res.end();
  });
});

app.listen(3000, () => {
  console.log('SSE server running on port 3000');
});
```

#### 客户端实现

```typescript
// 浏览器端 SSE 客户端
const eventSource = new EventSource('/api/stream');

// 监听不同类型的事件
eventSource.addEventListener('start', (event) => {
  const data = JSON.parse(event.data);
  console.log('开始:', data.message);
});

eventSource.addEventListener('delta', (event) => {
  const data = JSON.parse(event.data);
  process.stdout.write(data.content); // 逐字显示
});

eventSource.addEventListener('end', (event) => {
  const data = JSON.parse(event.data);
  console.log('\n完成:', data.message);
  eventSource.close();
});

// 监听错误
eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};
```

### 1.4 适用场景

✅ **推荐使用场景**：
- **Agent 响应流**：LLM 逐字生成响应
- **实时通知**：服务端推送通知给客户端
- **日志流**：实时显示日志输出
- **进度更新**：长时间任务的进度推送

❌ **不推荐使用场景**：
- **双向通信**：需要客户端频繁发送数据（用 WebSocket）
- **二进制数据**：SSE 只支持文本（用 WebSocket）
- **低延迟要求**：需要极低延迟（用 WebSocket）

### 1.5 TypeScript/Node.js 类比

**类比：ReadableStream**

```typescript
// SSE 类似于 Node.js 的 ReadableStream
const stream = fs.createReadStream('file.txt');

stream.on('data', (chunk) => {
  console.log('收到数据:', chunk);
});

stream.on('end', () => {
  console.log('流结束');
});
```

### 1.6 日常生活类比

**类比：电台广播**

想象你在听电台广播：
- **SSE** = 电台单向广播，你只能听，不能说话
- 电台持续推送内容，你实时接收

---

## 2. WebSocket

### 2.1 定义

**WebSocket** 是一种基于 TCP 的双向通信协议，允许服务端和客户端之间进行全双工通信。

**核心特征**：
- **双向通信**：服务端 ↔ 客户端（双方都可以主动发送数据）
- **持久连接**：建立连接后保持长期连接
- **低延迟**：比 HTTP 轮询延迟更低
- **支持二进制**：可以传输文本和二进制数据
- **更复杂**：需要特殊的服务器支持

### 2.2 工作原理

```typescript
// WebSocket 工作流程

// 1. 客户端发起 WebSocket 握手
GET /ws HTTP/1.1
Upgrade: websocket
Connection: Upgrade

// 2. 服务端响应升级
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade

// 3. 建立 WebSocket 连接

// 4. 双向通信
Client → Server: {"type": "user", "content": "你好"}
Server → Client: {"type": "assistant", "content": "你好！"}
Client → Server: {"type": "steering", "content": "停止"}
Server → Client: {"type": "ack", "content": "已停止"}
```

### 2.3 TypeScript 实现

#### 服务端实现（Node.js + ws）

```typescript
import WebSocket, { WebSocketServer } from 'ws';

// 创建 WebSocket 服务器
const wss = new WebSocketServer({ port: 8080 });

wss.on('connection', (ws: WebSocket) => {
  console.log('客户端已连接');

  // 监听客户端消息
  ws.on('message', async (data: Buffer) => {
    const message = JSON.parse(data.toString());
    console.log('收到消息:', message);

    // 处理不同类型的消息
    switch (message.type) {
      case 'user':
        // 处理用户消息
        await handleUserMessage(ws, message);
        break;

      case 'steering':
        // 处理 Steering 消息
        await handleSteeringMessage(ws, message);
        break;

      default:
        ws.send(JSON.stringify({
          type: 'error',
          content: `Unknown message type: ${message.type}`
        }));
    }
  });

  // 监听连接关闭
  ws.on('close', () => {
    console.log('客户端已断开');
  });

  // 监听错误
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
});

// 处理用户消息
async function handleUserMessage(ws: WebSocket, message: any) {
  // 1. 发送开始事件
  ws.send(JSON.stringify({ type: 'start' }));

  // 2. 模拟流式响应
  const response = '这是一个流式响应示例';
  for (const char of response) {
    ws.send(JSON.stringify({ type: 'delta', content: char }));
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  // 3. 发送结束事件
  ws.send(JSON.stringify({ type: 'end' }));
}

// 处理 Steering 消息
async function handleSteeringMessage(ws: WebSocket, message: any) {
  // 中断当前流
  ws.send(JSON.stringify({ type: 'interrupted' }));
}

console.log('WebSocket server running on port 8080');
```

#### 客户端实现

```typescript
// Node.js 客户端
import WebSocket from 'ws';

const ws = new WebSocket('ws://localhost:8080');

// 连接建立
ws.on('open', () => {
  console.log('已连接到服务器');

  // 发送用户消息
  ws.send(JSON.stringify({
    type: 'user',
    content: '你好'
  }));
});

// 接收消息
ws.on('message', (data: Buffer) => {
  const message = JSON.parse(data.toString());

  switch (message.type) {
    case 'start':
      console.log('开始生成响应');
      break;

    case 'delta':
      process.stdout.write(message.content);
      break;

    case 'end':
      console.log('\n响应完成');
      break;

    case 'interrupted':
      console.log('响应已中断');
      break;
  }
});

// 连接关闭
ws.on('close', () => {
  console.log('连接已关闭');
});

// 错误处理
ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});

// 5 秒后发送 Steering 消息
setTimeout(() => {
  ws.send(JSON.stringify({
    type: 'steering',
    content: '停止'
  }));
}, 5000);
```

### 2.4 适用场景

✅ **推荐使用场景**：
- **双向实时交互**：聊天应用、协作编辑
- **Web UI**：浏览器端的 Agent 交互
- **低延迟要求**：需要极低延迟的场景
- **二进制数据**：需要传输二进制数据

❌ **不推荐使用场景**：
- **单向流式传输**：只需要服务端推送（用 SSE 更轻量）
- **简单场景**：不需要双向通信（用 SSE 或 HTTP）
- **防火墙限制**：某些网络环境不支持 WebSocket

### 2.5 TypeScript/Node.js 类比

**类比：Duplex Stream**

```typescript
// WebSocket 类似于 Node.js 的 Duplex Stream
const { Duplex } = require('stream');

const duplexStream = new Duplex({
  read(size) {
    // 读取数据（接收）
  },
  write(chunk, encoding, callback) {
    // 写入数据（发送）
    callback();
  }
});
```

### 2.6 日常生活类比

**类比：电话通话**

想象你在打电话：
- **WebSocket** = 电话通话，双方都可以说话和听
- 双向实时通信，低延迟

---

## 3. HTTP Streaming（2025 新标准）

### 3.1 定义

**HTTP Streaming** 是 2025 年提出的新标准，结合了 HTTP 的简单性和流式传输的实时性。

**核心特征**：
- **基于 HTTP**：使用标准 HTTP 协议
- **流式传输**：支持流式数据传输
- **更轻量**：比 SSE 更轻量，比 WebSocket 更简单
- **更好的兼容性**：更好的防火墙和代理兼容性

### 3.2 工作原理

```typescript
// HTTP Streaming 工作流程

// 1. 客户端发起 HTTP 请求
POST /api/chat HTTP/1.1
Content-Type: application/json

{"message": "你好"}

// 2. 服务端返回流式响应
HTTP/1.1 200 OK
Content-Type: application/x-ndjson
Transfer-Encoding: chunked

{"type":"start"}
{"type":"delta","content":"你"}
{"type":"delta","content":"好"}
{"type":"end"}
```

### 3.3 TypeScript 实现

```typescript
// 服务端实现（Node.js + Express）
app.post('/api/chat', async (req, res) => {
  // 设置流式响应头
  res.setHeader('Content-Type', 'application/x-ndjson');
  res.setHeader('Transfer-Encoding', 'chunked');

  // 发送 NDJSON 格式的数据
  const sendChunk = (data: any) => {
    res.write(JSON.stringify(data) + '\n');
  };

  // 流式响应
  sendChunk({ type: 'start' });

  const response = '你好！';
  for (const char of response) {
    sendChunk({ type: 'delta', content: char });
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  sendChunk({ type: 'end' });
  res.end();
});
```

### 3.4 2025-2026 最新实践

> **2025-2026 最新实践**: 根据 Cloudflare MCP 2025 研究，HTTP Streaming 在轻量级场景下优于 SSE，具有更好的兼容性和性能。

**核心优势**：
- **更好的防火墙兼容性**：比 WebSocket 更容易通过防火墙
- **更简单的实现**：比 SSE 更简单，无需特殊事件格式
- **更好的错误处理**：标准 HTTP 错误处理机制

**引用来源**：
- Cloudflare MCP 2025 - HTTP streaming vs SSE
- https://blog.cloudflare.com/mcp-http-streaming-2025

---

## 4. Transport 选择决策树

### 4.1 决策流程

```
需要双向通信？
    ↓
  是 → WebSocket
    ↓
  否 → 需要流式传输？
    ↓
      是 → 2025 新标准可用？
        ↓
          是 → HTTP Streaming（推荐）
          否 → SSE
      ↓
      否 → 普通 HTTP
```

### 4.2 对比表格

| 特性 | SSE | WebSocket | HTTP Streaming |
|------|-----|-----------|----------------|
| **通信方向** | 单向（服务端 → 客户端） | 双向 | 单向（服务端 → 客户端） |
| **协议** | HTTP | WebSocket (TCP) | HTTP |
| **连接方式** | 长连接 | 持久连接 | 长连接 |
| **数据格式** | 文本（事件流） | 文本 + 二进制 | 文本（NDJSON） |
| **自动重连** | 是 | 否（需手动实现） | 否（需手动实现） |
| **防火墙兼容性** | 好 | 一般 | 很好 |
| **实现复杂度** | 简单 | 复杂 | 很简单 |
| **延迟** | 低 | 很低 | 低 |
| **浏览器支持** | 好 | 很好 | 好 |
| **适用场景** | Agent 响应流、通知 | Web UI、双向交互 | 轻量级流式传输 |

### 4.3 性能对比

```typescript
// 场景：传输 1000 条消息

// SSE
// 连接建立: 100ms
// 传输: 1000 × 1ms = 1000ms
// 总耗时: 1100ms

// WebSocket
// 连接建立: 50ms（握手更快）
// 传输: 1000 × 0.5ms = 500ms（更低延迟）
// 总耗时: 550ms

// HTTP Streaming
// 连接建立: 80ms
// 传输: 1000 × 0.8ms = 800ms
// 总耗时: 880ms
```

---

## 5. 2025-2026 最新实践

### 5.1 MCP Servers：HTTP/SSE Transport 标准

> **2025-2026 最新实践**: MCP servers 定义了 HTTP 和 SSE 传输的标准规范。

**核心规范**：
```typescript
// MCP Transport 接口
interface MCPTransport {
  // 传输类型
  type: 'http' | 'sse' | 'websocket';

  // 发送消息
  send(message: Message): Promise<void>;

  // 接收消息
  onMessage(callback: (message: Message) => void): void;

  // 关闭连接
  close(): void;
}
```

**引用来源**：
- MCP servers - HTTP/SSE transport standards
- https://github.com/modelcontextprotocol/servers

### 5.2 MCP 可恢复流提案

> **2025-2026 最新实践**: MCP 可恢复流提案支持长运行任务的流式传输，允许连接断开后恢复。

**核心特性**：
```typescript
// 可恢复流
interface ResumableStream {
  // 流 ID
  streamId: string;

  // 当前位置
  position: number;

  // 恢复流
  resume(position: number): Promise<void>;
}
```

**引用来源**：
- MCP Resumable Streams Proposal
- https://github.com/modelcontextprotocol/proposals/resumable-streams

### 5.3 OptiLLM：SSE 和 WebSocket 支持

> **2025-2026 最新实践**: OptiLLM 提供了 SSE 和 WebSocket 的优化实现，支持多种 LLM 提供商。

**核心架构**：
```typescript
// OptiLLM Transport
interface OptiLLMTransport {
  // 支持的传输类型
  supportedTransports: ['sse', 'websocket'];

  // 自动选择最佳传输
  selectBestTransport(): TransportType;
}
```

**引用来源**：
- OptiLLM - SSE and WebSocket support
- https://github.com/optillm/optillm

---

## 6. 实际应用示例

### 6.1 pi-mono 中的 SSE 实现

```typescript
// packages/pi-coding-agent/src/transport/sse.ts

export class SSETransport {
  private res: Response;

  constructor(res: Response) {
    this.res = res;
    this.setupSSE();
  }

  private setupSSE(): void {
    this.res.setHeader('Content-Type', 'text/event-stream');
    this.res.setHeader('Cache-Control', 'no-cache');
    this.res.setHeader('Connection', 'keep-alive');
  }

  sendEvent(event: string, data: any): void {
    this.res.write(`event: ${event}\n`);
    this.res.write(`data: ${JSON.stringify(data)}\n\n`);
  }

  close(): void {
    this.res.end();
  }
}
```

### 6.2 选择 Transport 的实际考虑

```typescript
// 根据场景选择 Transport
function selectTransport(context: Context): TransportType {
  // 1. Web UI → WebSocket（双向交互）
  if (context.platform === 'web') {
    return 'websocket';
  }

  // 2. CLI → SSE（单向流式）
  if (context.platform === 'cli') {
    return 'sse';
  }

  // 3. API → HTTP Streaming（轻量级）
  if (context.platform === 'api') {
    return 'http-streaming';
  }

  // 默认 SSE
  return 'sse';
}
```

---

## 7. 最佳实践

### 7.1 优先使用 SSE

```typescript
// 推荐：默认使用 SSE
// 原因：简单、轻量、自动重连

const transport = new SSETransport(res);
```

### 7.2 需要双向通信时使用 WebSocket

```typescript
// 只在需要双向通信时使用 WebSocket
if (needsBidirectional) {
  const transport = new WebSocketTransport(ws);
}
```

### 7.3 错误处理和重连

```typescript
// SSE 自动重连
const eventSource = new EventSource('/api/stream');

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  // SSE 会自动重连
};

// WebSocket 需要手动重连
const ws = new WebSocket('ws://localhost:8080');

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // 手动重连
  setTimeout(() => {
    reconnect();
  }, 1000);
};
```

---

## 8. 总结

### 8.1 核心要点

1. **SSE**：单向流式传输，轻量级，适合 Agent 响应流
2. **WebSocket**：双向实时通信，适合 Web UI 和低延迟场景
3. **HTTP Streaming**：2025 新标准，更轻量，更好的兼容性
4. **选择标准**：需要双向 → WebSocket，单向 → SSE/HTTP Streaming
5. **默认推荐**：SSE（简单、轻量、自动重连）

### 8.2 TypeScript/Node.js 类比

- **SSE** = ReadableStream（单向读取）
- **WebSocket** = Duplex Stream（双向读写）

### 8.3 日常生活类比

- **SSE** = 电台广播（单向）
- **WebSocket** = 电话通话（双向）

### 8.4 2025-2026 最新实践

- **Cloudflare MCP 2025**：HTTP Streaming 优于 SSE
- **MCP Servers**：HTTP/SSE 传输标准
- **MCP 可恢复流**：支持长运行任务
- **OptiLLM**：SSE 和 WebSocket 优化实现

### 8.5 学习检查

- [ ] 理解 SSE、WebSocket、HTTP Streaming 的区别
- [ ] 知道何时使用 SSE，何时使用 WebSocket
- [ ] 理解 2025-2026 最新 Transport 标准
- [ ] 能够实现 SSE 和 WebSocket 服务端和客户端
- [ ] 能够根据场景选择合适的 Transport

### 8.6 下一步

- **03_核心概念_05_流式响应架构.md**：学习流式响应的完整架构
- **07_实战代码_04_SSE流式实现.md**：手写 SSE 完整实现
- **07_实战代码_05_WebSocket流式实现.md**：手写 WebSocket 完整实现

---

**版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: Claude Code
