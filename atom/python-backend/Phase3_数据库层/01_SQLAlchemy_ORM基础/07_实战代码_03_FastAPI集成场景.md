# 实战代码场景3：FastAPI 集成场景

完整可运行的 FastAPI + SQLAlchemy 集成示例，演示生产级 API 开发。

---

## 场景说明

**目标：** 实现一个完整的 AI Agent API，包含用户认证、对话管理、消息处理。

**功能：**
- 用户注册和登录
- JWT 认证
- 对话 CRUD 操作
- 消息管理
- 依赖注入管理 Session
- 错误处理

**技术栈：**
- FastAPI
- SQLAlchemy 2.0+
- PostgreSQL
- JWT 认证
- Pydantic 验证

---

## 完整代码

```python
"""
FastAPI + SQLAlchemy 集成示例
演示：完整的 AI Agent API
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
from typing import List, Optional
import uuid
import jwt
from passlib.context import CryptContext

# ===== 1. 配置 =====

DATABASE_URL = "postgresql://user:password@localhost:5432/ai_agent_db"
SECRET_KEY = "your-secret-key-here"  # 生产环境使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 数据库引擎
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# FastAPI 应用
app = FastAPI(title="AI Agent API", version="1.0.0")

# JWT 认证
security = HTTPBearer()


# ===== 2. 数据库模型 =====

class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(100), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """对话模型"""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete='CASCADE'), nullable=False)
    title = Column(String(200), nullable=False)
    is_archived = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    """消息模型"""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# ===== 3. Pydantic 模型 =====

# 用户相关
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# 对话相关
class ConversationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    first_message: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    is_archived: Optional[bool] = None


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


# 消息相关
class MessageCreate(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


# ===== 4. 依赖注入 =====

def get_db():
    """获取数据库 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前登录用户"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    return user


# ===== 5. 工具函数 =====

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """加密密码"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建 JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# ===== 6. 认证端点 =====

@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    # 检查邮箱是否已存在
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # 检查用户名是否已存在
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # 创建用户
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@app.post("/auth/login", response_model=Token)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    user = db.query(User).filter(User.email == login_data.email).first()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    # 创建 token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user


# ===== 7. 对话端点 =====

@app.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建对话"""
    conversation = Conversation(
        user_id=current_user.id,
        title=data.title
    )
    db.add(conversation)
    db.flush()

    # 如果提供了第一条消息，创建消息
    if data.first_message:
        message = Message(
            conversation_id=conversation.id,
            role="user",
            content=data.first_message
        )
        db.add(message)

    db.commit()
    db.refresh(conversation)

    # 添加消息数量
    response = ConversationResponse.from_orm(conversation)
    response.message_count = len(conversation.messages)
    return response


@app.get("/conversations", response_model=List[ConversationResponse])
def get_conversations(
    skip: int = 0,
    limit: int = 10,
    archived: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """查询对话列表"""
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)

    if archived is not None:
        query = query.filter(Conversation.is_archived == archived)

    conversations = query\
        .order_by(Conversation.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    # 添加消息数量
    results = []
    for conv in conversations:
        response = ConversationResponse.from_orm(conv)
        response.message_count = len(conv.messages)
        results.append(response)

    return results


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """查询对话详情"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .filter(Conversation.user_id == current_user.id)\
        .first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    response = ConversationResponse.from_orm(conversation)
    response.message_count = len(conversation.messages)
    return response


@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: uuid.UUID,
    data: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新对话"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .filter(Conversation.user_id == current_user.id)\
        .first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if data.title is not None:
        conversation.title = data.title
    if data.is_archived is not None:
        conversation.is_archived = data.is_archived

    db.commit()
    db.refresh(conversation)

    response = ConversationResponse.from_orm(conversation)
    response.message_count = len(conversation.messages)
    return response


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除对话"""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .filter(Conversation.user_id == current_user.id)\
        .first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()


# ===== 8. 消息端点 =====

@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def create_message(
    conversation_id: uuid.UUID,
    data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建消息"""
    # 验证对话存在且属于当前用户
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .filter(Conversation.user_id == current_user.id)\
        .first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    message = Message(
        conversation_id=conversation_id,
        role=data.role,
        content=data.content
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    return message


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def get_messages(
    conversation_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """查询消息列表"""
    # 验证对话存在且属于当前用户
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .filter(Conversation.user_id == current_user.id)\
        .first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message)\
        .filter(Message.conversation_id == conversation_id)\
        .order_by(Message.created_at)\
        .offset(skip)\
        .limit(limit)\
        .all()

    return messages


# ===== 9. 健康检查 =====

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok"}


# ===== 10. 初始化 =====

@app.on_event("startup")
def startup():
    """应用启动时创建表"""
    Base.metadata.create_all(engine)
    print("✅ 数据库表创建成功")


# ===== 11. 运行 =====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 使用示例

### 1. 启动服务

```bash
# 安装依赖
uv add fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib[bcrypt]

# 运行服务
python main.py

# 或使用 uvicorn
uvicorn main:app --reload
```

### 2. 用户注册

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "username": "alice",
    "password": "password123"
  }'
```

**响应：**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "alice@example.com",
  "username": "alice",
  "is_active": true,
  "created_at": "2026-02-11T12:00:00"
}
```

### 3. 用户登录

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "password": "password123"
  }'
```

**响应：**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 4. 创建对话

```bash
curl -X POST "http://localhost:8000/conversations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "title": "Python 学习",
    "first_message": "如何学习 Python？"
  }'
```

### 5. 查询对话列表

```bash
curl -X GET "http://localhost:8000/conversations?skip=0&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 6. 创建消息

```bash
curl -X POST "http://localhost:8000/conversations/{conversation_id}/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "role": "assistant",
    "content": "推荐先学习 Python 基础..."
  }'
```

---

## 关键技术点

### 1. 依赖注入

```python
def get_db():
    """获取数据库 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 在路由中使用
@app.get("/conversations")
def get_conversations(db: Session = Depends(get_db)):
    ...
```

### 2. JWT 认证

```python
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """获取当前登录用户"""
    token = credentials.credentials
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    return user
```

### 3. Pydantic 验证

```python
class ConversationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    first_message: Optional[str] = None

class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy 2.0+
```

### 4. 错误处理

```python
if not conversation:
    raise HTTPException(status_code=404, detail="Conversation not found")

if not user.is_active:
    raise HTTPException(status_code=403, detail="User is inactive")
```

### 5. 事务管理

```python
@app.post("/conversations")
def create_conversation(data: ConversationCreate, db: Session = Depends(get_db)):
    conversation = Conversation(title=data.title)
    db.add(conversation)
    db.flush()  # 获取 ID

    if data.first_message:
        message = Message(conversation_id=conversation.id, content=data.first_message)
        db.add(message)

    db.commit()  # 一起提交
    return conversation
```

---

## 项目结构建议

```
ai-agent-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用
│   ├── config.py            # 配置
│   ├── database.py          # 数据库配置
│   ├── models/              # SQLAlchemy 模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── conversation.py
│   │   └── message.py
│   ├── schemas/             # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── conversation.py
│   │   └── message.py
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── conversations.py
│   │   └── messages.py
│   ├── core/                # 核心功能
│   │   ├── __init__.py
│   │   ├── security.py      # JWT、密码加密
│   │   └── deps.py          # 依赖注入
│   └── services/            # 业务逻辑
│       ├── __init__.py
│       ├── user.py
│       └── conversation.py
├── tests/                   # 测试
├── .env                     # 环境变量
├── requirements.txt         # 依赖
└── README.md
```

---

## 最佳实践

### 1. 环境变量管理

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. 错误处理中间件

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### 3. CORS 配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
```

### 5. 测试

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_register():
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "username": "test",
        "password": "password123"
    })
    assert response.status_code == 201
```

---

## 扩展功能

1. **分页辅助函数**：统一的分页响应格式
2. **搜索功能**：全文搜索对话和消息
3. **文件上传**：支持上传文档
4. **流式响应**：AI 回复的流式输出
5. **WebSocket**：实时消息推送
6. **限流**：防止 API 滥用
7. **缓存**：Redis 缓存热点数据
8. **监控**：Prometheus + Grafana

---

## 部署

### Docker 部署

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

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
      - DATABASE_URL=postgresql://user:password@db:5432/ai_agent_db
      - SECRET_KEY=your-secret-key
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ai_agent_db
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

**记住：** FastAPI + SQLAlchemy 是构建生产级 API 的最佳组合，依赖注入让代码更清晰、更易测试。

完成 SQLAlchemy ORM 基础的所有文档生成。文档包含：

**基础维度（8个文件）：**
- 30字核心
- 第一性原理
- 最小可用
- 双重类比
- 反直觉点
- 面试必问
- 化骨绵掌
- 一句话总结

**核心概念（3个文件）：**
- Model 定义与映射
- CRUD 操作
- 关系映射

**实战代码（3个文件）：**
- 基础 CRUD 场景
- 关系映射场景
- FastAPI 集成场景

**概览文件（1个）：**
- 完整学习路径和检查清单

所有文档都遵循 CLAUDE_PYTHON_BACKEND.md 的规范，包含完整可运行的代码示例，适合前端工程师转型学习。
