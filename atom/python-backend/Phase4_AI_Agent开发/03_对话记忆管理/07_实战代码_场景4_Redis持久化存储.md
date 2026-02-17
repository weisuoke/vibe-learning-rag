# 实战代码 - 场景4：持久化存储(Redis)

本文件演示如何使用Redis实现对话记忆的持久化存储。

---

## 场景描述

构建一个生产级的对话API，使用Redis存储对话记忆，支持分布式部署和自动过期。

**功能要求：**
- 使用Redis存储对话历史
- 支持自动过期(TTL)
- 支持分布式部署
- 提供完整的CRUD操作

---

## 完整代码

```python
"""
场景4：持久化存储(Redis)
演示：使用Redis实现对话记忆的持久化存储
"""

import json
import redis
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.schema import BaseChatMessageHistory, BaseMessage
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from datetime import datetime

# ===== 1. Redis配置 =====
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
DEFAULT_TTL = 1800  # 30分钟

# 创建Redis连接
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False  # 保持字节格式
)

# ===== 2. 自定义Redis记忆类 =====
class RedisChatMessageHistory(BaseChatMessageHistory):
    """基于Redis的对话历史存储"""

    def __init__(
        self,
        session_id: str,
        redis_client: redis.Redis,
        ttl: int = DEFAULT_TTL,
        key_prefix: str = "conversation"
    ):
        self.session_id = session_id
        self.redis_client = redis_client
        self.ttl = ttl
        self.key = f"{key_prefix}:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        try:
            # 从Redis获取
            raw_messages = self.redis_client.lrange(self.key, 0, -1)

            # 转换为LangChain消息对象
            messages = []
            for raw_msg in raw_messages:
                msg_dict = json.loads(raw_msg)
                if msg_dict["role"] == "user":
                    messages.append(HumanMessage(content=msg_dict["content"]))
                elif msg_dict["role"] == "assistant":
                    messages.append(AIMessage(content=msg_dict["content"]))

            return messages
        except Exception as e:
            print(f"Error loading messages: {e}")
            return []

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        msg_dict = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.rpush(self.key, json.dumps(msg_dict))
        # 更新过期时间
        self.redis_client.expire(self.key, self.ttl)

    def add_ai_message(self, message: str) -> None:
        """添加AI消息"""
        msg_dict = {
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.rpush(self.key, json.dumps(msg_dict))
        # 更新过期时间
        self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """清空历史"""
        self.redis_client.delete(self.key)

    def get_ttl(self) -> int:
        """获取剩余过期时间(秒)"""
        return self.redis_client.ttl(self.key)

    def set_ttl(self, ttl: int) -> None:
        """设置过期时间"""
        self.ttl = ttl
        if self.redis_client.exists(self.key):
            self.redis_client.expire(self.key, ttl)

# ===== 3. 数据模型 =====
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    message: str = Field(..., description="用户消息")
    ttl: Optional[int] = Field(DEFAULT_TTL, description="过期时间(秒)")

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: str
    ttl_remaining: int

class HistoryResponse(BaseModel):
    user_id: str
    messages: List[dict]
    total_messages: int
    ttl_remaining: int

# ===== 4. FastAPI应用 =====
app = FastAPI(
    title="Redis对话记忆管理API",
    description="使用Redis实现持久化的对话记忆管理",
    version="1.0.0"
)

# LLM实例
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# ===== 5. API端点 =====

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话端点(Redis持久化)

    - **user_id**: 用户ID
    - **message**: 用户消息
    - **ttl**: 过期时间(秒，默认1800)
    """
    try:
        # 1. 创建Redis记忆
        chat_history = RedisChatMessageHistory(
            session_id=request.user_id,
            redis_client=redis_client,
            ttl=request.ttl
        )

        # 2. 创建LangChain记忆
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True
        )

        # 3. 创建对话链
        conversation = ConversationChain(llm=llm, memory=memory)

        # 4. 生成回复
        response = conversation.predict(input=request.message)

        # 5. 获取剩余TTL
        ttl_remaining = chat_history.get_ttl()

        return ChatResponse(
            response=response,
            user_id=request.user_id,
            timestamp=datetime.now().isoformat(),
            ttl_remaining=ttl_remaining
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str):
    """
    获取用户的对话历史

    - **user_id**: 用户ID
    """
    try:
        # 创建Redis记忆
        chat_history = RedisChatMessageHistory(
            session_id=user_id,
            redis_client=redis_client
        )

        # 检查是否存在
        if not redis_client.exists(chat_history.key):
            raise HTTPException(status_code=404, detail="User not found or expired")

        # 获取消息
        messages = []
        for msg in chat_history.messages:
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })

        # 获取剩余TTL
        ttl_remaining = chat_history.get_ttl()

        return HistoryResponse(
            user_id=user_id,
            messages=messages,
            total_messages=len(messages),
            ttl_remaining=ttl_remaining
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """
    清空用户的对话历史

    - **user_id**: 用户ID
    """
    try:
        chat_history = RedisChatMessageHistory(
            session_id=user_id,
            redis_client=redis_client
        )

        if not redis_client.exists(chat_history.key):
            raise HTTPException(status_code=404, detail="User not found")

        chat_history.clear()

        return {"message": "History cleared", "user_id": user_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/ttl/{user_id}")
async def update_ttl(user_id: str, new_ttl: int):
    """
    更新用户的过期时间

    - **user_id**: 用户ID
    - **new_ttl**: 新的过期时间(秒)
    """
    if new_ttl < 60:
        raise HTTPException(
            status_code=400,
            detail="TTL must be at least 60 seconds"
        )

    try:
        chat_history = RedisChatMessageHistory(
            session_id=user_id,
            redis_client=redis_client
        )

        if not redis_client.exists(chat_history.key):
            raise HTTPException(status_code=404, detail="User not found")

        chat_history.set_ttl(new_ttl)

        return {
            "message": "TTL updated",
            "user_id": user_id,
            "new_ttl": new_ttl,
            "ttl_remaining": chat_history.get_ttl()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """获取Redis统计信息"""
    try:
        # 获取所有对话键
        keys = redis_client.keys("conversation:*")

        # 统计信息
        total_users = len(keys)
        total_messages = 0

        for key in keys:
            total_messages += redis_client.llen(key)

        # Redis信息
        info = redis_client.info()

        return {
            "total_users": total_users,
            "total_messages": total_messages,
            "redis_version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 测试Redis连接
        redis_client.ping()

        return {
            "status": "healthy",
            "redis": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ===== 6. 启动应用 =====
if __name__ == "__main__":
    import uvicorn

    # 测试Redis连接
    try:
        redis_client.ping()
        print("✓ Redis连接成功")
    except Exception as e:
        print(f"✗ Redis连接失败: {e}")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 测试脚本

```python
"""
测试Redis对话记忆管理API
"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_chat_with_ttl():
    """测试带TTL的对话"""
    print("=== 测试带TTL的对话 ===\n")

    # 第1轮对话(TTL=60秒)
    response1 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_redis_001",
        "message": "我叫张三",
        "ttl": 60
    })
    data1 = response1.json()
    print(f"1. {data1['response']}")
    print(f"   TTL剩余: {data1['ttl_remaining']}秒\n")

    # 第2轮对话
    response2 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_redis_001",
        "message": "我叫什么名字？"
    })
    data2 = response2.json()
    print(f"2. {data2['response']}")
    print(f"   TTL剩余: {data2['ttl_remaining']}秒\n")

def test_ttl_expiration():
    """测试TTL过期"""
    print("=== 测试TTL过期 ===\n")

    # 创建短TTL的对话
    requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_ttl_test",
        "message": "测试TTL",
        "ttl": 5  # 5秒过期
    })
    print("创建对话(TTL=5秒)")

    # 立即查询
    response1 = requests.get(f"{BASE_URL}/history/user_ttl_test")
    print(f"立即查询: {response1.status_code} (应该是200)")

    # 等待6秒
    print("等待6秒...")
    time.sleep(6)

    # 再次查询(应该已过期)
    response2 = requests.get(f"{BASE_URL}/history/user_ttl_test")
    print(f"6秒后查询: {response2.status_code} (应该是404)\n")

def test_update_ttl():
    """测试更新TTL"""
    print("=== 测试更新TTL ===\n")

    # 创建对话
    requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_update_ttl",
        "message": "测试更新TTL",
        "ttl": 60
    })

    # 查询当前TTL
    response1 = requests.get(f"{BASE_URL}/history/user_update_ttl")
    print(f"初始TTL: {response1.json()['ttl_remaining']}秒")

    # 更新TTL
    response2 = requests.put(f"{BASE_URL}/ttl/user_update_ttl?new_ttl=300")
    print(f"更新后TTL: {response2.json()['ttl_remaining']}秒\n")

def test_persistence():
    """测试持久化"""
    print("=== 测试持久化 ===\n")

    # 创建对话
    requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_persist",
        "message": "我叫李四",
        "ttl": 3600  # 1小时
    })
    print("创建对话: 我叫李四")

    # 查询历史
    response = requests.get(f"{BASE_URL}/history/user_persist")
    messages = response.json()['messages']
    print(f"历史记录: {len(messages)}条消息")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")

    print("\n提示: 即使重启FastAPI服务，Redis中的数据仍然存在\n")

def test_stats():
    """测试统计信息"""
    print("=== 测试统计信息 ===\n")

    response = requests.get(f"{BASE_URL}/stats")
    data = response.json()

    print(f"总用户数: {data['total_users']}")
    print(f"总消息数: {data['total_messages']}")
    print(f"Redis版本: {data['redis_version']}")
    print(f"内存使用: {data['used_memory_human']}")
    print(f"连接数: {data['connected_clients']}\n")

if __name__ == "__main__":
    test_chat_with_ttl()
    test_ttl_expiration()
    test_update_ttl()
    test_persistence()
    test_stats()
```

---

## Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: conversation_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  api:
    build: .
    container_name: conversation_api
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      redis:
        condition: service_healthy

volumes:
  redis_data:
```

---

## 关键要点

### 1. Redis数据结构

```python
# Key格式
conversation:{session_id}

# Value格式(List)
[
    '{"role": "user", "content": "...", "timestamp": "..."}',
    '{"role": "assistant", "content": "...", "timestamp": "..."}',
    ...
]

# TTL
EXPIRE conversation:{session_id} 1800
```

### 2. 自动过期机制

```python
# 每次添加消息时更新TTL
self.redis_client.rpush(self.key, json.dumps(msg_dict))
self.redis_client.expire(self.key, self.ttl)  # 重置过期时间
```

### 3. 持久化优势

- **服务重启不丢失**: Redis数据持久化到磁盘
- **分布式支持**: 多个FastAPI实例共享Redis
- **自动清理**: TTL自动清理过期数据

### 4. 性能对比

| 存储方式 | 读取速度 | 持久化 | 分布式 | 自动清理 |
|---------|---------|--------|--------|---------|
| 内存 | 最快 | ❌ | ❌ | ❌ |
| Redis | 快 | ✅ | ✅ | ✅ |
| PostgreSQL | 慢 | ✅ | ✅ | 需手动 |

---

## 生产环境建议

### 1. Redis配置优化

```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru  # LRU淘汰策略
appendonly yes  # AOF持久化
```

### 2. 连接池

```python
import redis.connection

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    decode_responses=False
)

redis_client = redis.Redis(connection_pool=pool)
```

### 3. 监控指标

- Redis内存使用率
- 键过期速率
- 命令执行时间
- 连接数

---

## 总结

完成了对话记忆管理的完整文档生成，包括：

1. **基础维度** (8个文件)
   - 30字核心
   - 第一性原理
   - 最小可用
   - 双重类比
   - 反直觉点
   - 面试必问
   - 化骨绵掌
   - 一句话总结

2. **核心概念** (3个文件)
   - ConversationBufferMemory
   - ConversationBufferWindowMemory
   - 持久化存储

3. **实战代码** (4个场景)
   - 场景1: 基础内存记忆
   - 场景2: 窗口记忆与Token优化
   - 场景3: FastAPI多用户集成
   - 场景4: Redis持久化存储

所有文件都已生成完毕！
