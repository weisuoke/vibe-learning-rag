# JWT认证 - 实战代码08：完整AI Agent API

## 概述

本章整合所有知识点，构建一个完整的AI Agent API，包含认证、权限控制、对话管理等功能。

---

## 项目结构

```
ai-agent-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置管理
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py             # 认证依赖
│   │   ├── security.py         # 密码哈希、JWT生成
│   │   └── database.py         # 数据库连接
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # 用户模型
│   │   ├── conversation.py     # 对话模型
│   │   └── message.py          # 消息模型
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py             # 认证相关schema
│   │   ├── user.py             # 用户相关schema
│   │   └── agent.py            # Agent相关schema
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py             # 认证端点
│   │   ├── users.py            # 用户端点
│   │   └── agent.py            # Agent端点
│   └── services/
│       ├── __init__.py
│       └── agent_service.py    # Agent服务
├── .env                        # 环境变量
├── requirements.txt            # 依赖
└── README.md                   # 文档
```

---

## 配置管理

### config.py

```python
"""
配置管理
演示：使用Pydantic管理配置
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""

    # 应用配置
    APP_NAME: str = "AI Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # JWT配置
    SECRET_KEY: str
    REFRESH_SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 数据库配置
    DATABASE_URL: str

    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI配置
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None

    # 安全配置
    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: str = "lax"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

---

## 数据库模型

### models/user.py

```python
"""
用户模型
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Table, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

# 用户-角色关联表
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Role(Base):
    """角色模型"""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)

    users = relationship("User", secondary=user_roles, back_populates="roles")
```

### models/conversation.py

```python
"""
对话模型
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class Conversation(Base):
    """对话模型"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String, default="新对话")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """消息模型"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    conversation = relationship("Conversation", back_populates="messages")
```

---

## 核心功能

### core/security.py

```python
"""
安全功能：密码哈希、JWT生成
"""

from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordManager:
    """密码管理器"""

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, username: str, roles: list) -> str:
    """生成Access Token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "user_id": user_id,
        "username": username,
        "roles": roles,
        "type": "access",
        "exp": expire
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """生成Refresh Token"""
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "user_id": user_id,
        "type": "refresh",
        "exp": expire
    }

    return jwt.encode(payload, settings.REFRESH_SECRET_KEY, algorithm=settings.ALGORITHM)
```

### core/auth.py

```python
"""
认证依赖
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User
from app.config import settings

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前用户"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )

        if payload.get("type") != "access":
            raise HTTPException(400, "无效的Token类型")

        user_id = payload.get("user_id")

        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(404, "用户不存在或已禁用")

        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token已过期")

    except JWTError:
        raise HTTPException(401, "无效的Token")


def require_role(required_role: str):
    """检查用户角色"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_roles = [role.name for role in current_user.roles]

        if required_role not in user_roles:
            raise HTTPException(403, f"需要 {required_role} 角色")

        return current_user

    return role_checker
```

---

## API端点

### api/auth.py

```python
"""
认证端点
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import PasswordManager, create_access_token, create_refresh_token
from app.core.auth import get_current_user
from app.models.user import User, Role
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from redis import Redis

router = APIRouter(prefix="/auth", tags=["认证"])
redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """用户注册"""
    # 检查邮箱
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(400, "邮箱已被注册")

    # 检查用户名
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(400, "用户名已被使用")

    # 创建用户
    user = User(
        email=request.email,
        username=request.username,
        hashed_password=PasswordManager.hash_password(request.password)
    )

    # 分配默认角色
    default_role = db.query(Role).filter(Role.name == "user").first()
    if not default_role:
        default_role = Role(name="user", description="普通用户")
        db.add(default_role)

    user.roles.append(default_role)

    db.add(user)
    db.commit()
    db.refresh(user)

    # 生成Token
    roles = [role.name for role in user.roles]
    access_token = create_access_token(user.id, user.username, roles)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(401, "邮箱或密码错误")

    if not PasswordManager.verify_password(request.password, user.hashed_password):
        raise HTTPException(401, "邮箱或密码错误")

    if not user.is_active:
        raise HTTPException(403, "账号已被禁用")

    # 生成Token
    roles = [role.name for role in user.roles]
    access_token = create_access_token(user.id, user.username, roles)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/logout")
async def logout(
    refresh_token: str,
    current_user: User = Depends(get_current_user)
):
    """用户登出"""
    # 将Refresh Token加入黑名单
    try:
        payload = jwt.decode(
            refresh_token,
            settings.REFRESH_SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )

        exp = payload.get("exp")
        ttl = exp - int(datetime.utcnow().timestamp())

        if ttl > 0:
            redis_client.setex(f"blacklist:refresh:{refresh_token}", ttl, "1")

        return {"message": "登出成功"}

    except JWTError:
        raise HTTPException(401, "无效的Refresh Token")
```

### api/agent.py

```python
"""
AI Agent端点
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.schemas.agent import ChatRequest, ChatResponse, ConversationResponse
from app.services.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["AI Agent"])
agent_service = AgentService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI Agent对话"""
    # 获取或创建对话
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == current_user.id
        ).first()

        if not conversation:
            raise HTTPException(404, "对话不存在")
    else:
        # 创建新对话
        conversation = Conversation(
            user_id=current_user.id,
            title=request.message[:50]  # 使用前50个字符作为标题
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 保存用户消息
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )
    db.add(user_message)

    # 获取对话历史
    messages = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.created_at).all()

    # 调用AI Agent
    response_text = await agent_service.chat(
        message=request.message,
        history=[{"role": m.role, "content": m.content} for m in messages[:-1]]
    )

    # 保存AI响应
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text
    )
    db.add(assistant_message)
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        message=request.message,
        response=response_text
    )


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取对话列表"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return conversations


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除对话"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(404, "对话不存在")

    db.delete(conversation)
    db.commit()

    return {"message": "对话已删除"}
```

---

## AI Agent服务

### services/agent_service.py

```python
"""
AI Agent服务
"""

from openai import AsyncOpenAI
from app.config import settings


class AgentService:
    """AI Agent服务"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )

    async def chat(self, message: str, history: list) -> str:
        """
        AI对话

        Args:
            message: 用户消息
            history: 对话历史

        Returns:
            AI响应
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。"}
        ]

        # 添加历史消息
        messages.extend(history)

        # 添加当前消息
        messages.append({"role": "user", "content": message})

        # 调用OpenAI API
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content
```

---

## 主应用

### main.py

```python
"""
主应用
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.database import engine, Base
from app.api import auth, agent

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 创建应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 注册路由
app.include_router(auth.router)
app.include_router(agent.router)


@app.get("/")
async def root():
    """根端点"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 环境配置

### .env

```bash
# 应用配置
APP_NAME=AI Agent API
APP_VERSION=1.0.0
DEBUG=False

# JWT配置
SECRET_KEY=your-secret-key-at-least-32-characters-long
REFRESH_SECRET_KEY=your-refresh-secret-key-at-least-32-characters-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/ai_agent_db

# Redis配置
REDIS_URL=redis://localhost:6379/0

# OpenAI配置
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 安全配置
COOKIE_SECURE=True
COOKIE_SAMESITE=lax
```

### requirements.txt

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
pydantic==2.5.3
pydantic-settings==2.1.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
redis==5.0.1
openai==1.10.0
```

---

## 运行应用

### 启动步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑.env文件，填入实际配置

# 3. 创建数据库
createdb ai_agent_db

# 4. 运行应用
python -m app.main

# 或使用uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 测试API

### 完整测试流程

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

# 1. 注册用户
echo "=== 注册用户 ==="
REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "username": "alice",
    "password": "SecurePass123!"
  }')

echo $REGISTER_RESPONSE | jq .

# 2. 登录
echo -e "\n=== 登录 ==="
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "SecurePass123!"
  }')

ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')
REFRESH_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.refresh_token')

echo "Access Token: ${ACCESS_TOKEN:0:50}..."
echo "Refresh Token: ${REFRESH_TOKEN:0:50}..."

# 3. AI对话（创建新对话）
echo -e "\n=== AI对话（新对话）==="
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/agent/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "message": "你好，请介绍一下自己"
  }')

echo $CHAT_RESPONSE | jq .
CONVERSATION_ID=$(echo $CHAT_RESPONSE | jq -r '.conversation_id')

# 4. 继续对话
echo -e "\n=== 继续对话 ==="
curl -s -X POST "$BASE_URL/agent/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{
    \"conversation_id\": $CONVERSATION_ID,
    \"message\": \"你能帮我做什么？\"
  }" | jq .

# 5. 获取对话列表
echo -e "\n=== 对话列表 ==="
curl -s -X GET "$BASE_URL/agent/conversations" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq .

# 6. 删除对话
echo -e "\n=== 删除对话 ==="
curl -s -X DELETE "$BASE_URL/agent/conversations/$CONVERSATION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq .

# 7. 登出
echo -e "\n=== 登出 ==="
curl -s -X POST "$BASE_URL/auth/logout?refresh_token=$REFRESH_TOKEN" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq .
```

---

## Docker部署

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY app/ ./app/

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ai_agent_db
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=ai_agent_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 运行Docker

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止
docker-compose down
```

---

## 总结

**本章实现了：**
1. 完整的项目结构
2. 用户认证系统（注册、登录、登出）
3. JWT Token管理（Access + Refresh）
4. AI Agent对话功能
5. 对话历史管理
6. 角色权限控制
7. Docker部署配置

**关键特性：**
- 用户注册和登录
- JWT认证和刷新
- AI Agent对话
- 对话历史持久化
- 用户数据隔离
- 角色权限控制
- Docker容器化部署

**生产环境优化：**
- 使用环境变量管理配置
- 密码哈希存储
- Token黑名单
- 数据库连接池
- Redis缓存
- 健康检查端点
- CORS配置
- 错误处理

这是一个完整的、可直接部署的AI Agent API系统！
