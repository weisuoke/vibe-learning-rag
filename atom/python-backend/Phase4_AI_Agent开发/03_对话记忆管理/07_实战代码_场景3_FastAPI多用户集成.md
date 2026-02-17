# 实战代码 - 场景3：FastAPI多用户集成

本文件演示如何在FastAPI中集成对话记忆管理，支持多用户并发。

---

## 场景描述

构建一个生产级的对话API，支持多用户并发对话，每个用户有独立的记忆。

**功能要求：**
- 多用户隔离（每个用户独立记忆）
- 支持查看历史记录
- 支持清空历史
- 支持动态调整窗口大小
- 提供统计信息

---

## 完整代码

```python
"""
场景3：FastAPI多用户集成
演示：在FastAPI中实现多用户对话记忆管理
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from datetime import datetime
import tiktoken

# ===== 1. 数据模型 =====
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    message: str = Field(..., description="用户消息")
    window_size: Optional[int] = Field(5, description="窗口大小", ge=1, le=20)

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: str
    stats: dict

class HistoryResponse(BaseModel):
    user_id: str
    messages: List[dict]
    total_messages: int
    total_rounds: int
    window_size: int

class StatsResponse(BaseModel):
    user_id: str
    total_messages: int
    total_rounds: int
    window_size: int
    tokens: int
    cost_gpt35: float
    cost_gpt4: float

# ===== 2. 记忆管理器 =====
class MemoryManager:
    """多用户记忆管理器"""

    def __init__(self):
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    def get_or_create_memory(
        self,
        user_id: str,
        window_size: int = 5
    ) -> ConversationBufferWindowMemory:
        """获取或创建用户的记忆"""
        if user_id not in self.memories:
            self.memories[user_id] = ConversationBufferWindowMemory(k=window_size)
        else:
            # 更新窗口大小
            self.memories[user_id].k = window_size

        return self.memories[user_id]

    def get_memory(self, user_id: str) -> Optional[ConversationBufferWindowMemory]:
        """获取用户的记忆"""
        return self.memories.get(user_id)

    def delete_memory(self, user_id: str) -> bool:
        """删除用户的记忆"""
        if user_id in self.memories:
            del self.memories[user_id]
            return True
        return False

    def clear_memory(self, user_id: str) -> bool:
        """清空用户的记忆"""
        memory = self.get_memory(user_id)
        if memory:
            memory.clear()
            return True
        return False

    def get_stats(self, user_id: str) -> Optional[dict]:
        """获取用户的统计信息"""
        memory = self.get_memory(user_id)
        if not memory:
            return None

        history = memory.load_memory_variables({})
        tokens = self._count_tokens(history["history"])

        return {
            "total_messages": len(memory.chat_memory.messages),
            "total_rounds": len(memory.chat_memory.messages) // 2,
            "window_size": memory.k,
            "tokens": tokens,
            "cost_gpt35": tokens * 0.0000015,
            "cost_gpt4": tokens * 0.00003
        }

    def _count_tokens(self, text: str) -> int:
        """计算token数量"""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

    def list_users(self) -> List[str]:
        """列出所有用户"""
        return list(self.memories.keys())

# ===== 3. FastAPI应用 =====
app = FastAPI(
    title="对话记忆管理API",
    description="支持多用户的对话记忆管理系统",
    version="1.0.0"
)

# 创建记忆管理器
memory_manager = MemoryManager()

# ===== 4. API端点 =====

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话端点

    - **user_id**: 用户ID（用于隔离不同用户的记忆）
    - **message**: 用户消息
    - **window_size**: 窗口大小（可选，默认5）
    """
    try:
        # 1. 获取或创建用户的记忆
        memory = memory_manager.get_or_create_memory(
            request.user_id,
            request.window_size
        )

        # 2. 创建对话链
        conversation = ConversationChain(
            llm=memory_manager.llm,
            memory=memory,
            verbose=False
        )

        # 3. 生成回复
        response = conversation.predict(input=request.message)

        # 4. 获取统计信息
        stats = memory_manager.get_stats(request.user_id)

        return ChatResponse(
            response=response,
            user_id=request.user_id,
            timestamp=datetime.now().isoformat(),
            stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history(user_id: str):
    """
    获取用户的对话历史

    - **user_id**: 用户ID
    """
    memory = memory_manager.get_memory(user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="User not found")

    # 获取消息列表
    messages = []
    for msg in memory.chat_memory.messages:
        messages.append({
            "role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
            "content": msg.content
        })

    return HistoryResponse(
        user_id=user_id,
        messages=messages,
        total_messages=len(messages),
        total_rounds=len(messages) // 2,
        window_size=memory.k
    )

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """
    清空用户的对话历史

    - **user_id**: 用户ID
    """
    success = memory_manager.clear_memory(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "History cleared", "user_id": user_id}

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """
    删除用户（包括记忆）

    - **user_id**: 用户ID
    """
    success = memory_manager.delete_memory(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User deleted", "user_id": user_id}

@app.get("/stats/{user_id}", response_model=StatsResponse)
async def get_stats(user_id: str):
    """
    获取用户的统计信息

    - **user_id**: 用户ID
    """
    stats = memory_manager.get_stats(user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")

    return StatsResponse(user_id=user_id, **stats)

@app.put("/window-size/{user_id}")
async def update_window_size(user_id: str, new_size: int):
    """
    更新用户的窗口大小

    - **user_id**: 用户ID
    - **new_size**: 新的窗口大小
    """
    if new_size < 1 or new_size > 20:
        raise HTTPException(
            status_code=400,
            detail="Window size must be between 1 and 20"
        )

    memory = memory_manager.get_memory(user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="User not found")

    memory.k = new_size

    return {
        "message": "Window size updated",
        "user_id": user_id,
        "new_size": new_size
    }

@app.get("/users")
async def list_users():
    """列出所有用户"""
    users = memory_manager.list_users()
    return {
        "total_users": len(users),
        "users": users
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "total_users": len(memory_manager.list_users()),
        "timestamp": datetime.now().isoformat()
    }

# ===== 5. 启动应用 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 测试脚本

```python
"""
测试FastAPI对话记忆管理API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat():
    """测试对话"""
    print("=== 测试对话 ===\n")

    # 用户1的对话
    print("用户1的对话:")
    response1 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_001",
        "message": "我叫张三",
        "window_size": 3
    })
    print(f"1. {response1.json()['response']}")

    response2 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_001",
        "message": "我叫什么名字？"
    })
    print(f"2. {response2.json()['response']}\n")

    # 用户2的对话（独立记忆）
    print("用户2的对话:")
    response3 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_002",
        "message": "我叫李四"
    })
    print(f"1. {response3.json()['response']}")

    response4 = requests.post(f"{BASE_URL}/chat", json={
        "user_id": "user_002",
        "message": "我叫什么名字？"
    })
    print(f"2. {response4.json()['response']}\n")

def test_history():
    """测试历史记录"""
    print("=== 测试历史记录 ===\n")

    response = requests.get(f"{BASE_URL}/history/user_001")
    data = response.json()

    print(f"用户: {data['user_id']}")
    print(f"总消息数: {data['total_messages']}")
    print(f"对话轮数: {data['total_rounds']}")
    print(f"窗口大小: {data['window_size']}")
    print("\n消息列表:")
    for i, msg in enumerate(data['messages'], 1):
        print(f"{i}. [{msg['role']}] {msg['content']}")
    print()

def test_stats():
    """测试统计信息"""
    print("=== 测试统计信息 ===\n")

    response = requests.get(f"{BASE_URL}/stats/user_001")
    data = response.json()

    print(f"用户: {data['user_id']}")
    print(f"总消息数: {data['total_messages']}")
    print(f"对话轮数: {data['total_rounds']}")
    print(f"窗口大小: {data['window_size']}")
    print(f"Token数: {data['tokens']}")
    print(f"成本(GPT-3.5): ${data['cost_gpt35']:.6f}")
    print(f"成本(GPT-4): ${data['cost_gpt4']:.6f}\n")

def test_clear():
    """测试清空历史"""
    print("=== 测试清空历史 ===\n")

    # 清空前
    response1 = requests.get(f"{BASE_URL}/history/user_001")
    print(f"清空前消息数: {response1.json()['total_messages']}")

    # 清空
    requests.delete(f"{BASE_URL}/history/user_001")
    print("已清空历史")

    # 清空后
    response2 = requests.get(f"{BASE_URL}/history/user_001")
    print(f"清空后消息数: {response2.json()['total_messages']}\n")

def test_list_users():
    """测试列出用户"""
    print("=== 测试列出用户 ===\n")

    response = requests.get(f"{BASE_URL}/users")
    data = response.json()

    print(f"总用户数: {data['total_users']}")
    print(f"用户列表: {', '.join(data['users'])}\n")

if __name__ == "__main__":
    test_chat()
    test_history()
    test_stats()
    test_clear()
    test_list_users()
```

---

## API文档

启动服务后，访问 `http://localhost:8000/docs` 查看自动生成的API文档。

---

## 关键要点

### 1. 多用户隔离

```python
# 每个用户有独立的记忆
memories: Dict[str, ConversationBufferWindowMemory] = {}

# 通过user_id隔离
memory = memories.get(user_id)
```

### 2. 记忆管理器模式

```python
class MemoryManager:
    def get_or_create_memory(self, user_id, window_size):
        """获取或创建记忆"""
        pass

    def get_stats(self, user_id):
        """获取统计信息"""
        pass
```

### 3. RESTful API设计

- `POST /chat` - 发送消息
- `GET /history/{user_id}` - 获取历史
- `DELETE /history/{user_id}` - 清空历史
- `GET /stats/{user_id}` - 获取统计
- `GET /users` - 列出用户

---

## 下一步

- **场景4**: 持久化存储(PostgreSQL/Redis)
