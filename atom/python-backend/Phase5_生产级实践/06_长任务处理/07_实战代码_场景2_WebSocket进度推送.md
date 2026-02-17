# 实战代码 - 场景2：WebSocket进度推送

> 完整可运行的WebSocket实时进度推送示例

---

## 场景描述

**需求**：用户提交任务后，通过WebSocket实时接收任务进度更新

**技术栈**：FastAPI + WebSocket + Celery + Redis + PostgreSQL

---

## 完整代码实现

### 1. WebSocket连接管理器

```python
# app/websocket_manager.py
from fastapi import WebSocket
from typing import Dict
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections[task_id] = websocket
        print(f"客户端订阅任务 {task_id}")

    def disconnect(self, task_id: str):
        """断开连接"""
        if task_id in self.active_connections:
            self.active_connections.pop(task_id)
            print(f"客户端取消订阅任务 {task_id}")

    async def send_progress(self, task_id: str, data: dict):
        """发送进度更新"""
        websocket = self.active_connections.get(task_id)
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                print(f"发送失败: {e}")
                self.disconnect(task_id)

manager = ConnectionManager()
```

---

### 2. FastAPI WebSocket端点

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.websocket_manager import manager
from app.models.task import Task
from app.core.database import SessionLocal
import asyncio

app = FastAPI()

@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket进度推送端点"""
    await manager.connect(task_id, websocket)

    db = SessionLocal()

    try:
        # 发送初始状态
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            await websocket.send_json({"error": "Task not found"})
            return

        await websocket.send_json({
            "task_id": task_id,
            "status": task.status.value,
            "progress": task.progress
        })

        # 持续推送进度
        while True:
            await asyncio.sleep(1)

            task = db.query(Task).filter(Task.task_id == task_id).first()

            await websocket.send_json({
                "task_id": task_id,
                "status": task.status.value,
                "progress": task.progress
            })

            # 任务完成，关闭连接
            if task.status in ["completed", "failed"]:
                await websocket.send_json({
                    "type": "complete",
                    "result": task.result
                })
                break

    except WebSocketDisconnect:
        manager.disconnect(task_id)
    finally:
        db.close()
```

---

### 3. Celery任务（推送进度）

```python
# app/tasks.py
from app.celery_app import app
from app.websocket_manager import manager
from app.models.task import Task
from app.core.database import SessionLocal
import asyncio

@app.task(bind=True)
def process_document_with_ws(self, file_path: str, db_task_id: int):
    """处理文档并通过WebSocket推送进度"""
    db = SessionLocal()

    try:
        task = db.query(Task).filter(Task.id == db_task_id).first()
        task.status = "running"
        db.commit()

        # 推送进度
        asyncio.run(manager.send_progress(task.task_id, {
            "status": "running",
            "progress": 0
        }))

        # 步骤1：解析（33%）
        parse_pdf(file_path)
        task.progress = 33.0
        db.commit()

        asyncio.run(manager.send_progress(task.task_id, {
            "status": "running",
            "progress": 33.0,
            "message": "解析完成"
        }))

        # 步骤2：Embedding（66%）
        generate_embedding(content)
        task.progress = 66.0
        db.commit()

        asyncio.run(manager.send_progress(task.task_id, {
            "status": "running",
            "progress": 66.0,
            "message": "Embedding完成"
        }))

        # 步骤3：保存（100%）
        save_to_vectordb(embedding)
        task.progress = 100.0
        task.status = "completed"
        db.commit()

        asyncio.run(manager.send_progress(task.task_id, {
            "status": "completed",
            "progress": 100.0,
            "result": {"status": "success"}
        }))

    finally:
        db.close()
```

---

### 4. 前端WebSocket客户端

```javascript
// 前端代码（JavaScript）
const taskId = 'task-123';
const ws = new WebSocket(`ws://localhost:8000/ws/tasks/${taskId}`);

// 连接建立
ws.onopen = () => {
    console.log('WebSocket连接已建立');
};

// 接收进度更新
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('进度更新:', data);

    // 更新进度条
    document.getElementById('progress').value = data.progress;
    document.getElementById('status').textContent = data.status;

    if (data.message) {
        document.getElementById('message').textContent = data.message;
    }

    // 任务完成
    if (data.type === 'complete') {
        console.log('任务完成:', data.result);
        ws.close();
    }
};

// 连接关闭
ws.onclose = () => {
    console.log('WebSocket连接已关闭');
};

// 错误处理
ws.onerror = (error) => {
    console.error('WebSocket错误:', error);
};
```

---

### 5. 前端自动重连

```javascript
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 1000;
        this.maxReconnectInterval = 30000;
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

    handleMessage(data) {
        console.log('收到消息:', data);

        // 更新UI
        document.getElementById('progress').value = data.progress;
        document.getElementById('status').textContent = data.status;

        if (data.type === 'complete') {
            this.ws.close();
        }
    }

    close() {
        this.ws.close();
    }
}

// 使用
const client = new WebSocketClient(`ws://localhost:8000/ws/tasks/${taskId}`);
```

---

## 运行步骤

```bash
# 1. 启动Redis
redis-server

# 2. 启动Celery Worker
celery -A app.celery_app worker --loglevel=info

# 3. 启动FastAPI
uvicorn app.main:app --reload

# 4. 测试WebSocket连接
# 使用浏览器或WebSocket客户端连接
ws://localhost:8000/ws/tasks/task-123
```

---

## 预期输出

### WebSocket消息流

```json
// 初始状态
{"task_id": "task-123", "status": "pending", "progress": 0}

// 进度更新
{"status": "running", "progress": 33.0, "message": "解析完成"}
{"status": "running", "progress": 66.0, "message": "Embedding完成"}

// 任务完成
{"status": "completed", "progress": 100.0, "type": "complete", "result": {...}}
```

---

## 关键知识点

### 1. WebSocket vs SSE

| 特性 | WebSocket | SSE |
|------|-----------|-----|
| 通信方向 | 双向 | 单向 |
| 自动重连 | ❌ 需要手动实现 | ✅ 浏览器自动 |
| 实现复杂度 | 高 | 低 |

### 2. 连接管理

```python
# 存储活跃连接
active_connections: Dict[str, WebSocket] = {}

# 连接时添加
active_connections[task_id] = websocket

# 断开时移除
active_connections.pop(task_id)
```

### 3. 心跳机制

```python
async def heartbeat():
    while True:
        await asyncio.sleep(30)
        await websocket.send_json({"type": "ping"})
```

---

## 总结

本示例演示了：
- ✅ WebSocket连接管理
- ✅ 实时进度推送
- ✅ 前端自动重连
- ✅ Celery集成

**下一步**：学习任务状态追踪系统（场景3）
