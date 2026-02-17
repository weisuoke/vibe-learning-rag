# 核心概念02：WebSocket实时通信

> 深入理解WebSocket协议、FastAPI集成、连接管理和实时进度推送

---

## 什么是WebSocket？

### 定义

**WebSocket = 持久连接 + 双向通信 + 低延迟**

WebSocket是一种在单个TCP连接上进行全双工通信的协议，允许服务器主动向客户端推送数据。

### HTTP vs WebSocket

```
HTTP（请求-响应模型）：
客户端 ──请求──> 服务器
客户端 <──响应── 服务器
（每次通信都需要建立新连接）

WebSocket（持久连接）：
客户端 ←─────────> 服务器
       双向实时通信
（连接建立后保持打开状态）
```

**对比表**：

| 特性 | HTTP | WebSocket |
|------|------|-----------|
| **通信方向** | 单向（客户端→服务器） | 双向（客户端↔服务器） |
| **连接** | 短连接（每次请求建立） | 长连接（持久保持） |
| **延迟** | 高（每次请求都有开销） | 低（连接已建立） |
| **服务器推送** | ❌ 不支持 | ✅ 支持 |
| **协议** | HTTP/HTTPS | ws:// / wss:// |
| **适用场景** | 普通API请求 | 实时通信、聊天、推送 |

---

## WebSocket协议基础

### 1. 握手过程

```
1. 客户端发起HTTP升级请求
   GET /ws HTTP/1.1
   Host: localhost:8000
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
   Sec-WebSocket-Version: 13

2. 服务器响应升级
   HTTP/1.1 101 Switching Protocols
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

3. 连接建立，开始双向通信
   客户端 ←─────────> 服务器
```

**关键点**：
- 使用HTTP协议发起握手
- 服务器返回101状态码（Switching Protocols）
- 握手成功后，协议升级为WebSocket

---

### 2. 消息格式

WebSocket支持两种消息类型：
- **文本消息**：UTF-8编码的字符串
- **二进制消息**：任意二进制数据

```python
# 文本消息
await websocket.send_text("Hello, World!")

# JSON消息（文本）
await websocket.send_json({"type": "progress", "value": 50})

# 二进制消息
await websocket.send_bytes(b"\x00\x01\x02\x03")
```

---

## FastAPI WebSocket集成

### 1. 基础WebSocket端点

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 1. 接受连接
    await websocket.accept()

    try:
        while True:
            # 2. 接收消息
            data = await websocket.receive_text()
            print(f"收到消息: {data}")

            # 3. 发送消息
            await websocket.send_text(f"Echo: {data}")

    except WebSocketDisconnect:
        # 4. 连接断开
        print("客户端断开连接")
```

**关键步骤**：
1. `await websocket.accept()`：接受WebSocket连接
2. `await websocket.receive_text()`：接收客户端消息
3. `await websocket.send_text()`：发送消息给客户端
4. `WebSocketDisconnect`：处理连接断开

---

### 2. 前端连接WebSocket

```javascript
// 前端代码（JavaScript）
const ws = new WebSocket('ws://localhost:8000/ws');

// 连接建立
ws.onopen = () => {
    console.log('WebSocket连接已建立');
    ws.send('Hello, Server!');
};

// 接收消息
ws.onmessage = (event) => {
    console.log('收到消息:', event.data);
};

// 连接关闭
ws.onclose = () => {
    console.log('WebSocket连接已关闭');
};

// 连接错误
ws.onerror = (error) => {
    console.error('WebSocket错误:', error);
};

// 发送消息
ws.send('Hello, Server!');

// 关闭连接
ws.close();
```

---

### 3. 接收不同类型的消息

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收任意类型的消息
            message = await websocket.receive()

            if message['type'] == 'websocket.receive':
                # 文本消息
                if 'text' in message:
                    text = message['text']
                    print(f"收到文本: {text}")
                    await websocket.send_text(f"Echo: {text}")

                # 二进制消息
                elif 'bytes' in message:
                    data = message['bytes']
                    print(f"收到二进制数据: {len(data)} bytes")
                    await websocket.send_bytes(data)

            elif message['type'] == 'websocket.disconnect':
                print("客户端断开连接")
                break

    except WebSocketDisconnect:
        print("连接异常断开")
```

---

## 连接管理器

### 1. 为什么需要连接管理器？

**问题**：
- 多个客户端同时连接
- 需要向特定客户端发送消息
- 需要广播消息给所有客户端
- 需要管理连接的生命周期

**解决方案**：连接管理器

---

### 2. 连接管理器实现

```python
# app/websocket_manager.py
from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    def __init__(self):
        # 存储所有活跃连接
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        """接受新连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"客户端 {client_id} 已连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """断开连接"""
        if client_id in self.active_connections:
            self.active_connections.pop(client_id)
            print(f"客户端 {client_id} 已断开，当前连接数: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, client_id: str):
        """发送消息给特定客户端"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """广播消息给所有客户端"""
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"发送消息给 {client_id} 失败: {e}")

    async def broadcast_except(self, message: dict, except_client_id: str):
        """广播消息给除了指定客户端外的所有客户端"""
        for client_id, websocket in self.active_connections.items():
            if client_id != except_client_id:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"发送消息给 {client_id} 失败: {e}")

# 创建全局连接管理器实例
manager = ConnectionManager()
```

---

### 3. 使用连接管理器

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.websocket_manager import manager

app = FastAPI()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # 1. 连接
    await manager.connect(client_id, websocket)

    try:
        while True:
            # 2. 接收消息
            data = await websocket.receive_json()

            # 3. 处理消息
            if data['type'] == 'broadcast':
                # 广播给所有客户端
                await manager.broadcast({
                    "from": client_id,
                    "message": data['message']
                })
            elif data['type'] == 'personal':
                # 发送给特定客户端
                await manager.send_personal_message(
                    {"from": client_id, "message": data['message']},
                    data['to']
                )

    except WebSocketDisconnect:
        # 4. 断开连接
        manager.disconnect(client_id)
        await manager.broadcast({
            "type": "system",
            "message": f"{client_id} 离开了"
        })
```

---

## 任务进度推送

### 1. 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                  任务进度推送系统                        │
│                                                          │
│  ┌──────────┐                                           │
│  │ 客户端   │                                           │
│  └────┬─────┘                                           │
│       │ 1. 建立WebSocket连接                            │
│       ↓                                                  │
│  ┌──────────────────────────────────────────┐          │
│  │         WebSocket端点                     │          │
│  │  /ws/{task_id}                           │          │
│  └──────────────────────────────────────────┘          │
│       │                                                  │
│       │ 2. 订阅任务进度                                 │
│       ↓                                                  │
│  ┌──────────────────────────────────────────┐          │
│  │         连接管理器                        │          │
│  │  存储 task_id → websocket 映射           │          │
│  └──────────────────────────────────────────┘          │
│       ↑                                                  │
│       │ 3. Worker更新进度                               │
│       │                                                  │
│  ┌──────────────────────────────────────────┐          │
│  │         Celery Worker                     │          │
│  │  - 执行任务                               │          │
│  │  - 更新数据库进度                         │          │
│  │  - 通知WebSocket推送                      │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

### 2. 实现任务进度推送

```python
# app/websocket_manager.py
class TaskConnectionManager:
    def __init__(self):
        # task_id → websocket 映射
        self.task_connections: Dict[str, WebSocket] = {}

    async def connect_task(self, task_id: str, websocket: WebSocket):
        """连接到任务进度推送"""
        await websocket.accept()
        self.task_connections[task_id] = websocket
        print(f"客户端订阅任务 {task_id} 的进度")

    def disconnect_task(self, task_id: str):
        """断开任务连接"""
        if task_id in self.task_connections:
            self.task_connections.pop(task_id)
            print(f"客户端取消订阅任务 {task_id}")

    async def send_progress(self, task_id: str, progress: dict):
        """发送进度更新"""
        websocket = self.task_connections.get(task_id)
        if websocket:
            try:
                await websocket.send_json(progress)
            except Exception as e:
                print(f"发送进度失败: {e}")
                self.disconnect_task(task_id)

task_manager = TaskConnectionManager()
```

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.websocket_manager import task_manager
from app.models.task import Task
from app.core.database import SessionLocal

app = FastAPI()

@app.websocket("/ws/tasks/{task_id}")
async def task_progress_websocket(websocket: WebSocket, task_id: str):
    """任务进度推送WebSocket端点"""
    # 1. 连接
    await task_manager.connect_task(task_id, websocket)

    # 2. 查询任务初始状态
    db = SessionLocal()
    task = db.query(Task).filter(Task.task_id == task_id).first()

    if not task:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    # 3. 发送初始状态
    await websocket.send_json({
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress
    })

    try:
        # 4. 保持连接，等待进度更新
        while True:
            # 定期查询数据库（或等待Worker推送）
            await asyncio.sleep(1)

            task = db.query(Task).filter(Task.task_id == task_id).first()

            # 发送进度更新
            await websocket.send_json({
                "task_id": task_id,
                "status": task.status,
                "progress": task.progress
            })

            # 任务完成，关闭连接
            if task.status in ["completed", "failed"]:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.status,
                    "result": task.result
                })
                break

    except WebSocketDisconnect:
        task_manager.disconnect_task(task_id)
    finally:
        db.close()
```

---

### 3. Worker推送进度

```python
# app/tasks.py
from app.celery_app import app
from app.websocket_manager import task_manager
from app.models.task import Task
from app.core.database import SessionLocal

@app.task(bind=True)
def process_documents(self, task_id: int, files: List[str]):
    """处理文档并推送进度"""
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()

    total = len(files)
    for i, file in enumerate(files):
        # 处理文件
        process_file(file)

        # 更新进度
        progress = (i + 1) / total * 100
        task.progress = progress
        task.status = "running"
        db.commit()

        # 推送进度（异步）
        asyncio.run(task_manager.send_progress(task.task_id, {
            "task_id": task.task_id,
            "status": "running",
            "progress": progress,
            "message": f"正在处理第 {i+1}/{total} 个文件"
        }))

    # 任务完成
    task.status = "completed"
    task.progress = 100.0
    db.commit()

    # 推送完成消息
    asyncio.run(task_manager.send_progress(task.task_id, {
        "task_id": task.task_id,
        "status": "completed",
        "progress": 100.0,
        "result": {"files_processed": total}
    }))

    db.close()
```

---

## 心跳机制

### 1. 为什么需要心跳？

**问题**：
- 网络中断时，服务器无法立即知道客户端已断开
- 客户端无法知道服务器是否还在运行
- 长时间无数据传输，连接可能被中间设备（防火墙、负载均衡器）关闭

**解决方案**：心跳机制（Ping/Pong）

---

### 2. 实现心跳机制

```python
# app/main.py
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    # 心跳任务
    async def heartbeat():
        while True:
            try:
                # 每30秒发送一次ping
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    # 启动心跳任务
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()

            # 处理pong响应
            if data.get('type') == 'pong':
                print(f"收到 {client_id} 的pong")
                continue

            # 处理其他消息
            print(f"收到消息: {data}")

    except WebSocketDisconnect:
        print(f"{client_id} 断开连接")
    finally:
        # 取消心跳任务
        heartbeat_task.cancel()
```

```javascript
// 前端心跳响应
const ws = new WebSocket('ws://localhost:8000/ws/client123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'ping') {
        // 响应pong
        ws.send(JSON.stringify({ type: 'pong' }));
    } else {
        // 处理其他消息
        console.log('收到消息:', data);
    }
};
```

---

## 断线重连

### 1. 前端自动重连

```javascript
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 1000;  // 重连间隔（毫秒）
        this.maxReconnectInterval = 30000;  // 最大重连间隔
        this.reconnectAttempts = 0;
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('WebSocket连接已建立');
            this.reconnectAttempts = 0;
            this.reconnectInterval = 1000;
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接已关闭');
            this.reconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
        };
    }

    reconnect() {
        this.reconnectAttempts++;

        // 指数退避
        const delay = Math.min(
            this.reconnectInterval * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectInterval
        );

        console.log(`${delay}ms后尝试重连...`);

        setTimeout(() => {
            console.log(`第${this.reconnectAttempts}次重连尝试`);
            this.connect();
        }, delay);
    }

    send(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.error('WebSocket未连接');
        }
    }

    handleMessage(data) {
        // 处理消息
        console.log('收到消息:', data);
    }

    close() {
        this.ws.close();
    }
}

// 使用
const client = new WebSocketClient('ws://localhost:8000/ws/client123');
client.send({ type: 'message', content: 'Hello' });
```

---

## WebSocket vs SSE对比

### 功能对比

| 特性 | WebSocket | SSE |
|------|-----------|-----|
| **通信方向** | 双向 | 单向（服务器→客户端） |
| **协议** | WebSocket协议 | HTTP |
| **自动重连** | ❌ 需要手动实现 | ✅ 浏览器自动 |
| **实现复杂度** | 高（200行代码） | 低（50行代码） |
| **连接管理** | 需要手动管理 | 浏览器自动 |
| **心跳机制** | 需要手动实现 | 不需要 |
| **消息格式** | 文本/二进制 | 仅文本 |
| **浏览器支持** | 所有现代浏览器 | 所有现代浏览器 |

### 适用场景

**WebSocket适用场景**：
- ✅ 聊天应用（需要双向通信）
- ✅ 实时协作（多人编辑文档）
- ✅ 在线游戏（高频双向通信）
- ✅ 实时交易（股票、加密货币）

**SSE适用场景**：
- ✅ 进度推送（单向推送）
- ✅ 实时通知（服务器推送）
- ✅ 日志流（服务器推送日志）
- ✅ 监控数据（服务器推送指标）

---

## 最佳实践

### 1. 错误处理

```python
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        await websocket.accept()

        while True:
            try:
                data = await websocket.receive_json()
                # 处理消息
                await process_message(data)

            except json.JSONDecodeError:
                # JSON解析错误
                await websocket.send_json({
                    "error": "Invalid JSON format"
                })

            except Exception as e:
                # 其他错误
                await websocket.send_json({
                    "error": str(e)
                })

    except WebSocketDisconnect:
        print(f"{client_id} 断开连接")

    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.close()
```

### 2. 消息验证

```python
from pydantic import BaseModel

class WebSocketMessage(BaseModel):
    type: str
    data: dict

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # 接收并验证消息
            raw_data = await websocket.receive_json()
            message = WebSocketMessage(**raw_data)

            # 处理消息
            if message.type == "chat":
                await handle_chat(message.data)
            elif message.type == "command":
                await handle_command(message.data)

        except ValidationError as e:
            await websocket.send_json({
                "error": "Invalid message format",
                "details": e.errors()
            })
```

### 3. 连接限制

```python
class ConnectionManager:
    def __init__(self, max_connections: int = 1000):
        self.active_connections: Dict[str, WebSocket] = {}
        self.max_connections = max_connections

    async def connect(self, client_id: str, websocket: WebSocket):
        # 检查连接数限制
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            return False

        await websocket.accept()
        self.active_connections[client_id] = websocket
        return True
```

---

## 总结

### WebSocket核心要点

1. **双向通信**：客户端和服务器都可以主动发送消息
2. **持久连接**：连接建立后保持打开状态
3. **低延迟**：无需每次请求都建立连接
4. **需要管理**：连接管理、心跳、重连都需要手动实现

### 何时使用WebSocket

- ✅ 需要双向实时通信
- ✅ 需要低延迟
- ✅ 需要高频通信
- ❌ 只需要单向推送（用SSE更简单）

### 实现检查清单

- [ ] 实现连接管理器
- [ ] 实现心跳机制
- [ ] 实现断线重连
- [ ] 实现错误处理
- [ ] 实现消息验证
- [ ] 实现连接限制

---

**记住**：WebSocket功能强大但复杂，如果只需要单向推送（如进度推送），SSE是更简单的选择。
