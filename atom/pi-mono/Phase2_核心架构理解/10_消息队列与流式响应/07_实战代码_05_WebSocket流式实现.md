# 实战代码 05：WebSocket 流式实现

> **目标**: 实现完整的 WebSocket 双向通信，包括服务端和客户端。

---

## 完整代码实现

### 服务端实现（Node.js + ws）

```typescript
/**
 * WebSocket 流式传输服务端实现
 * 演示：如何使用 ws 库实现 WebSocket 双向通信
 */

import WebSocket, { WebSocketServer } from 'ws';
import { createServer } from 'http';

// ===== 1. WebSocket Transport 类 =====

/**
 * WebSocket Transport
 * 封装 WebSocket 连接的发送逻辑
 */
export class WebSocketTransport {
  private ws: WebSocket;
  private clientId: string;

  constructor(ws: WebSocket, clientId: string) {
    this.ws = ws;
    this.clientId = clientId;
    this.setupHandlers();
  }

  /**
   * 设置事件处理器
   */
  private setupHandlers(): void {
    this.ws.on('error', (error) => {
      console.error(`WebSocket error for client ${this.clientId}:`, error);
    });

    this.ws.on('close', () => {
      console.log(`WebSocket closed for client ${this.clientId}`);
    });
  }

  /**
   * 发送消息
   */
  send(type: string, data: any): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    }
  }

  /**
   * 发送事件
   */
  sendEvent(event: string, data: any): void {
    this.send('event', { event, data });
  }

  /**
   * 关闭连接
   */
  close(): void {
    this.ws.close();
  }

  /**
   * 检查连接是否打开
   */
  isOpen(): boolean {
    return this.ws.readyState === WebSocket.OPEN;
  }
}

// ===== 2. WebSocket 服务器 =====

/**
 * WebSocket 服务器
 */
export class WSServer {
  private wss: WebSocketServer;
  private clients: Map<string, WebSocketTransport> = new Map();

  constructor(port: number) {
    const server = createServer();
    this.wss = new WebSocketServer({ server });
    this.setupServer();

    server.listen(port, () => {
      console.log(`WebSocket server running on ws://localhost:${port}`);
    });
  }

  /**
   * 设置服务器
   */
  private setupServer(): void {
    this.wss.on('connection', (ws: WebSocket) => {
      const clientId = `client-${Date.now()}`;
      const transport = new WebSocketTransport(ws, clientId);

      // 保存客户端连接
      this.clients.set(clientId, transport);

      // 发送连接成功消息
      transport.send('connected', {
        clientId,
        message: 'Connected to WebSocket server'
      });

      // 监听客户端消息
      ws.on('message', async (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleMessage(transport, message);
        } catch (error) {
          transport.send('error', {
            message: 'Invalid message format'
          });
        }
      });

      // 监听连接关闭
      ws.on('close', () => {
        this.clients.delete(clientId);
      });
    });
  }

  /**
   * 处理客户端消息
   */
  private async handleMessage(
    transport: WebSocketTransport,
    message: any
  ): Promise<void> {
    console.log('Received message:', message);

    switch (message.type) {
      case 'generate':
        await this.handleGenerate(transport, message.data);
        break;

      case 'steering':
        await this.handleSteering(transport, message.data);
        break;

      case 'ping':
        transport.send('pong', { timestamp: Date.now() });
        break;

      default:
        transport.send('error', {
          message: `Unknown message type: ${message.type}`
        });
    }
  }

  /**
   * 处理生成请求
   */
  private async handleGenerate(
    transport: WebSocketTransport,
    data: any
  ): Promise<void> {
    const prompt = data.prompt || 'Hello';

    // 发送 start 事件
    transport.sendEvent('start', {
      prompt,
      timestamp: Date.now()
    });

    // 模拟 LLM 流式生成
    const response = `这是对 "${prompt}" 的响应。我会逐字生成内容，展示 WebSocket 流式传输的效果。`;

    for (let i = 0; i < response.length; i++) {
      // 检查连接是否仍然打开
      if (!transport.isOpen()) {
        console.log('Connection closed, stopping generation');
        break;
      }

      // 发送 delta 事件
      transport.sendEvent('delta', {
        content: response[i],
        index: i
      });

      // 模拟生成延迟
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // 发送 end 事件
    if (transport.isOpen()) {
      transport.sendEvent('end', {
        finishReason: 'stop',
        timestamp: Date.now()
      });
    }
  }

  /**
   * 处理 Steering 请求
   */
  private async handleSteering(
    transport: WebSocketTransport,
    data: any
  ): Promise<void> {
    console.log('Handling steering:', data);

    transport.sendEvent('interrupted', {
      message: 'Current generation interrupted',
      timestamp: Date.now()
    });

    // 处理新请求
    if (data.newPrompt) {
      await this.handleGenerate(transport, { prompt: data.newPrompt });
    }
  }

  /**
   * 广播消息给所有客户端
   */
  broadcast(type: string, data: any): void {
    for (const [clientId, transport] of this.clients) {
      if (transport.isOpen()) {
        transport.send(type, data);
      }
    }
  }

  /**
   * 获取连接数
   */
  getConnectionCount(): number {
    return this.clients.size;
  }
}

// ===== 3. 启动服务器 =====

const server = new WSServer(8080);

// 定期广播服务器状态
setInterval(() => {
  server.broadcast('status', {
    connections: server.getConnectionCount(),
    timestamp: Date.now()
  });
}, 30000);
```

### 客户端实现

```typescript
/**
 * WebSocket 客户端实现
 * 演示：如何使用 WebSocket 进行双向通信
 */

import WebSocket from 'ws';

// ===== 1. WebSocket 客户端类 =====

/**
 * WebSocket 客户端
 */
export class WSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map();

  constructor(url: string) {
    this.url = url;
  }

  /**
   * 连接到 WebSocket 服务器
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to ${this.url}...`);

      this.ws = new WebSocket(this.url);

      // 监听连接打开
      this.ws.on('open', () => {
        console.log('WebSocket connection opened');
        this.reconnectAttempts = 0;
        resolve();
      });

      // 监听消息
      this.ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      });

      // 监听错误
      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      });

      // 监听连接关闭
      this.ws.on('close', () => {
        console.log('WebSocket connection closed');
        this.attemptReconnect();
      });
    });
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(message: any): void {
    const { type, data } = message;

    // 触发对应类型的处理器
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }

    // 处理事件类型消息
    if (type === 'event' && data.event) {
      const eventHandlers = this.messageHandlers.get(`event:${data.event}`);
      if (eventHandlers) {
        eventHandlers.forEach(handler => handler(data.data));
      }
    }
  }

  /**
   * 监听消息类型
   */
  on(type: string, handler: (data: any) => void): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);
  }

  /**
   * 监听事件
   */
  onEvent(event: string, handler: (data: any) => void): void {
    this.on(`event:${event}`, handler);
  }

  /**
   * 发送消息
   */
  send(type: string, data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    } else {
      console.error('WebSocket is not open');
    }
  }

  /**
   * 尝试重连
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.log('Max reconnect attempts reached');
    }
  }

  /**
   * 关闭连接
   */
  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * 检查连接是否打开
   */
  isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// ===== 2. 使用示例 =====

/**
 * 示例 1：基础 WebSocket 连接
 */
async function example1_BasicWebSocket() {
  console.log('=== 示例 1：基础 WebSocket 连接 ===\n');

  const client = new WSClient('ws://localhost:8080');

  // 监听连接成功
  client.on('connected', (data) => {
    console.log('Connected:', data);
  });

  // 连接
  await client.connect();

  // 发送 ping
  client.send('ping', {});

  // 监听 pong
  client.on('pong', (data) => {
    console.log('Received pong:', data);
  });

  // 10 秒后关闭
  setTimeout(() => {
    client.close();
  }, 10000);
}

/**
 * 示例 2：流式生成
 */
async function example2_StreamingGeneration() {
  console.log('=== 示例 2：流式生成 ===\n');

  const client = new WSClient('ws://localhost:8080');

  let buffer = '';

  // 监听 start 事件
  client.onEvent('start', (data) => {
    console.log('Generation started:', data);
  });

  // 监听 delta 事件
  client.onEvent('delta', (data) => {
    buffer += data.content;
    process.stdout.write(data.content);
  });

  // 监听 end 事件
  client.onEvent('end', (data) => {
    console.log('\n\nGeneration completed:', data);
    console.log('Full response:', buffer);
    client.close();
  });

  // 连接并发送生成请求
  await client.connect();
  client.send('generate', { prompt: '你好' });
}

/**
 * 示例 3：Steering 中断
 */
async function example3_SteeringInterrupt() {
  console.log('=== 示例 3：Steering 中断 ===\n');

  const client = new WSClient('ws://localhost:8080');

  // 监听 delta 事件
  let charCount = 0;
  client.onEvent('delta', (data) => {
    process.stdout.write(data.content);
    charCount++;

    // 接收 20 个字符后发送 Steering
    if (charCount === 20) {
      console.log('\n\n[Sending steering message...]');
      client.send('steering', {
        message: '停止当前生成',
        newPrompt: '改成生成另一个内容'
      });
    }
  });

  // 监听 interrupted 事件
  client.onEvent('interrupted', (data) => {
    console.log('\nInterrupted:', data);
  });

  // 连接并发送生成请求
  await client.connect();
  client.send('generate', { prompt: '生成一段长文本' });

  // 30 秒后关闭
  setTimeout(() => {
    client.close();
  }, 30000);
}

/**
 * 示例 4：双向通信
 */
async function example4_BidirectionalCommunication() {
  console.log('=== 示例 4：双向通信 ===\n');

  const client = new WSClient('ws://localhost:8080');

  // 监听服务器状态
  client.on('status', (data) => {
    console.log('Server status:', data);
  });

  // 连接
  await client.connect();

  // 定期发送 ping
  const pingInterval = setInterval(() => {
    if (client.isOpen()) {
      client.send('ping', { timestamp: Date.now() });
    }
  }, 5000);

  // 监听 pong
  client.on('pong', (data) => {
    const latency = Date.now() - data.timestamp;
    console.log(`Pong received (latency: ${latency}ms)`);
  });

  // 30 秒后关闭
  setTimeout(() => {
    clearInterval(pingInterval);
    client.close();
  }, 30000);
}

// ===== 3. 运行所有示例 =====

async function runAllExamples() {
  await example1_BasicWebSocket();
  await new Promise(resolve => setTimeout(resolve, 2000));

  await example2_StreamingGeneration();
  await new Promise(resolve => setTimeout(resolve, 2000));

  await example3_SteeringInterrupt();
  await new Promise(resolve => setTimeout(resolve, 2000));

  await example4_BidirectionalCommunication();
}

// 运行示例
runAllExamples().catch(console.error);
```

---

## 核心要点

1. **双向通信**：客户端和服务端都可以主动发送消息
2. **实时性**：比 SSE 延迟更低，适合实时交互
3. **二进制支持**：可以传输文本和二进制数据
4. **手动重连**：需要手动实现重连逻辑
5. **连接管理**：需要管理连接状态和心跳

---

## 关键实现细节

### 1. 连接状态检查

```typescript
if (ws.readyState === WebSocket.OPEN) {
  ws.send(data);
}
```

### 2. 心跳机制

```typescript
// 客户端定期发送 ping
setInterval(() => {
  client.send('ping', {});
}, 30000);

// 服务端响应 pong
ws.on('message', (data) => {
  if (data.type === 'ping') {
    ws.send(JSON.stringify({ type: 'pong' }));
  }
});
```

### 3. 重连逻辑

```typescript
ws.on('close', () => {
  if (reconnectAttempts < maxReconnectAttempts) {
    setTimeout(() => {
      connect();
    }, reconnectDelay);
  }
});
```

---

## SSE vs WebSocket 对比

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务端 → 客户端） | 双向 |
| 协议 | HTTP | WebSocket (TCP) |
| 自动重连 | 是 | 否（需手动实现） |
| 二进制支持 | 否 | 是 |
| 实现复杂度 | 简单 | 复杂 |
| 延迟 | 低 | 很低 |
| 适用场景 | Agent 响应流 | Web UI、实时交互 |

---

## 扩展练习

1. **添加认证**：使用 JWT 或 API key 认证 WebSocket 连接
2. **添加房间机制**：支持多个客户端加入同一个房间
3. **添加消息确认**：实现消息确认和重传机制
4. **添加压缩**：使用 permessage-deflate 压缩数据

---

**版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: Claude Code
