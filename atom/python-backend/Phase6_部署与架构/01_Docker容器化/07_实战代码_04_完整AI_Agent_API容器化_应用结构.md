# 实战代码4：完整 AI Agent API 容器化（应用结构）

## 目标

容器化一个完整的 AI Agent API，包含 FastAPI + PostgreSQL + Redis + LangChain。

---

## 项目结构

```
ai-agent-api/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 应用入口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # 配置管理
│   │   ├── database.py            # 数据库连接
│   │   └── cache.py               # Redis 缓存
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py                # 用户模型
│   │   └── conversation.py        # 对话模型
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py              # 健康检查
│   │   └── agent.py               # Agent API
│   └── services/
│       ├── __init__.py
│       └── agent_service.py       # Agent 服务
├── alembic/
│   ├── versions/
│   └── env.py
├── pyproject.toml
├── uv.lock
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 应用代码

### app/core/config.py

```python
"""配置管理"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """应用配置"""

    # 应用配置
    APP_NAME: str = "AI Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # 数据库配置
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis 配置
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = 50

    # AI 配置
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4"

    # JWT 配置
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
```

### app/core/database.py

```python
"""数据库连接管理"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import get_settings

settings = get_settings()

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,  # 连接前检查
    echo=settings.DEBUG
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

def get_db() -> Session:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### app/core/cache.py

```python
"""Redis 缓存管理"""
import redis
from app.core.config import get_settings

settings = get_settings()

# 创建 Redis 连接池
redis_pool = redis.ConnectionPool.from_url(
    settings.REDIS_URL,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    decode_responses=True
)

# 创建 Redis 客户端
redis_client = redis.Redis(connection_pool=redis_pool)

def get_cache():
    """获取缓存客户端"""
    return redis_client
```

### app/models/user.py

```python
"""用户模型"""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### app/models/conversation.py

```python
"""对话模型"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Conversation(Base):
    """对话表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关系
    user = relationship("User", backref="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    conversation = relationship("Conversation", back_populates="messages")
```

### app/services/agent_service.py

```python
"""Agent 服务"""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.cache import get_cache
from app.models.conversation import Conversation, Message

settings = get_settings()

class AgentService:
    """AI Agent 服务"""

    def __init__(self, db: Session):
        self.db = db
        self.cache = get_cache()
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )

    async def chat(self, conversation_id: int, user_message: str) -> str:
        """处理对话"""
        # 获取对话历史
        conversation = self.db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if not conversation:
            raise ValueError("Conversation not found")

        # 构建消息历史
        messages = []
        for msg in conversation.messages[-10:]:  # 最近10条消息
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # 添加新消息
        messages.append(HumanMessage(content=user_message))

        # 调用 LLM
        response = await self.llm.ainvoke(messages)

        # 保存消息
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_message
        )
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.content
        )
        self.db.add(user_msg)
        self.db.add(assistant_msg)
        self.db.commit()

        return response.content
```

### app/api/health.py

```python
"""健康检查 API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.cache import get_cache

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """健康检查"""
    health_status = {
        "status": "healthy",
        "checks": {}
    }

    # 检查数据库
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"error: {str(e)}"
        raise HTTPException(status_code=503, detail=health_status)

    # 检查 Redis
    try:
        cache = get_cache()
        cache.ping()
        health_status["checks"]["redis"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["redis"] = f"error: {str(e)}"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
```

### app/api/agent.py

```python
"""Agent API"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.agent_service import AgentService

router = APIRouter()

class ChatRequest(BaseModel):
    """对话请求"""
    conversation_id: int
    message: str

class ChatResponse(BaseModel):
    """对话响应"""
    response: str

@router.post("/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """AI Agent 对话"""
    try:
        agent = AgentService(db)
        response = await agent.chat(request.conversation_id, request.message)
        return ChatResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### app/main.py

```python
"""FastAPI 应用入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api import health, agent

settings = get_settings()

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["health"])
app.include_router(agent.router, prefix="/api", tags=["agent"])

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.on_event("startup")
async def startup_event():
    """应用启动"""
    print(f"{settings.APP_NAME} v{settings.APP_VERSION} started")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    print(f"{settings.APP_NAME} shutting down")
```

---

## 数据库迁移

### alembic.ini

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### alembic/env.py

```python
"""Alembic 环境配置"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from app.core.database import Base
from app.core.config import get_settings

# 导入所有模型
from app.models import user, conversation

config = context.config
settings = get_settings()

# 设置数据库 URL
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_online():
    """在线迁移"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
```

---

## pyproject.toml

```toml
[project]
name = "ai-agent-api"
version = "1.0.0"
requires-python = ">=3.13"
dependencies = [
    "fastapi==0.109.0",
    "uvicorn[standard]==0.27.0",
    "pydantic==2.5.3",
    "pydantic-settings==2.1.0",
    "sqlalchemy==2.0.25",
    "alembic==1.13.1",
    "psycopg2-binary==2.9.9",
    "redis==5.0.1",
    "langchain==0.1.0",
    "langchain-openai==0.0.5",
    "python-jose[cryptography]==3.3.0",
    "passlib[bcrypt]==1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.3",
    "pytest-asyncio==0.21.1",
    "httpx==0.26.0",
]
```

---

## 总结

**完整 AI Agent API 应用结构：**
1. ✅ FastAPI 应用入口
2. ✅ 配置管理（Pydantic Settings）
3. ✅ 数据库连接（SQLAlchemy）
4. ✅ Redis 缓存
5. ✅ 数据模型（User、Conversation、Message）
6. ✅ Agent 服务（LangChain）
7. ✅ API 路由（健康检查、Agent 对话）
8. ✅ 数据库迁移（Alembic）

**下一步：** Docker 配置（Dockerfile + docker-compose.yml）

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
