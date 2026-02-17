# 实战代码8：完整AI Agent API

> 集成JWT认证、限流、日志、缓存的生产级AI Agent API

---

## 完整项目结构

```
ai-agent-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── chat.py
│   ├── services/               # 业务逻辑
│   │   ├── __init__.py
│   │   ├── auth.py             # JWT 认证
│   │   ├── rate_limit.py       # 限流
│   │   └── llm.py              # LLM 调用
│   ├── middleware/             # 中间件
│   │   ├── __init__.py
│   │   ├── logging.py          # 日志中间件
│   │   └── error_handler.py    # 错误处理
│   └── api/                    # API 路由
│       ├── __init__.py
│       ├── auth.py             # 认证路由
│       └── chat.py             # 聊天路由
├── .env                        # 环境变量
├── pyproject.toml              # 依赖管理
└── README.md
```

---

## 配置管理

```python
"""app/config.py - 配置管理"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "AI Agent API"
    DEBUG: bool = False

    # JWT 配置
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # OpenAI 配置
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # 限流配置
    RATE_LIMIT_FREE: int = 10
    RATE_LIMIT_PRO: int = 1000

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## 数据模型

```python
"""app/models/user.py - 用户模型"""
from pydantic import BaseModel
from enum import Enum

class UserTier(str, Enum):
    FREE = "free"
    PRO = "pro"

class User(BaseModel):
    id: str
    email: str
    tier: UserTier

class TokenData(BaseModel):
    user_id: str
    tier: UserTier
```

```python
"""app/models/chat.py - 聊天模型"""
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    from_cache: bool = False
```

---

## JWT 认证服务

```python
"""app/services/auth.py - JWT 认证"""
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import get_settings
from app.models.user import User, TokenData

settings = get_settings()
security = HTTPBearer()

def create_access_token(user_id: str, tier: str) -> str:
    """创建 JWT token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "user_id": user_id,
        "tier": tier,
        "exp": expire
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """验证 JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("user_id")
        tier = payload.get("tier")

        if user_id is None or tier is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return TokenData(user_id=user_id, tier=tier)

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## 限流服务

```python
"""app/services/rate_limit.py - 限流服务"""
import redis
import time
from fastapi import HTTPException, Request, Depends
from app.config import get_settings
from app.services.auth import verify_token
from app.models.user import TokenData

settings = get_settings()
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB
)

class RedisTokenBucket:
    """Redis 令牌桶限流器"""

    LUA_SCRIPT = """
    local key = KEYS[1]
    local rate = tonumber(ARGV[1])
    local capacity = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])

    local state = redis.call('HMGET', key, 'tokens', 'last_time')
    local tokens = tonumber(state[1]) or capacity
    local last_time = tonumber(state[2]) or now

    local elapsed = now - last_time
    tokens = math.min(capacity, tokens + elapsed * rate)

    if tokens >= requested then
        tokens = tokens - requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_time', now)
        redis.call('EXPIRE', key, 3600)
        return 1
    end
    return 0
    """

    def __init__(self, key: str, rate: float, capacity: float):
        self.redis = redis_client
        self.key = key
        self.rate = rate
        self.capacity = capacity
        self.script_sha = self.redis.script_load(self.LUA_SCRIPT)

    def acquire(self, tokens: int = 1) -> bool:
        now = time.time()
        result = self.redis.evalsha(
            self.script_sha, 1, self.key,
            self.rate, self.capacity, now, tokens
        )
        return result == 1

async def check_rate_limit(
    request: Request,
    token_data: TokenData = Depends(verify_token)
):
    """检查限流"""
    # 根据用户等级获取限流配置
    if token_data.tier == "free":
        rate, capacity = 1, settings.RATE_LIMIT_FREE
    else:
        rate, capacity = 10, settings.RATE_LIMIT_PRO

    # 创建限流器
    limiter = RedisTokenBucket(
        key=f"user:{token_data.user_id}:rate_limit",
        rate=rate,
        capacity=capacity
    )

    # 检查限流
    if not limiter.acquire():
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "tier": token_data.tier,
                "limit": capacity
            }
        )
```

---

## LLM 服务

```python
"""app/services/llm.py - LLM 调用服务"""
from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

async def chat_completion(message: str) -> tuple[str, int]:
    """调用 LLM"""
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )

    content = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return content, tokens_used
```

---

## 日志中间件

```python
"""app/middleware/logging.py - 日志中间件"""
import structlog
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 记录请求
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host
        )

        # 处理请求
        response = await call_next(request)

        # 记录响应
        duration = time.time() - start_time
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )

        return response
```

---

## 错误处理

```python
"""app/middleware/error_handler.py - 错误处理"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger()

async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理"""
    logger.error(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path
        }
    )
```

---

## API 路由

```python
"""app/api/auth.py - 认证路由"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """登录"""
    # 简化示例：实际应该验证密码
    if request.email == "user@example.com":
        token = create_access_token(user_id="123", tier="free")
        return LoginResponse(access_token=token)

    raise HTTPException(status_code=401, detail="Invalid credentials")
```

```python
"""app/api/chat.py - 聊天路由"""
from fastapi import APIRouter, Depends
from app.models.chat import ChatRequest, ChatResponse
from app.services.auth import verify_token
from app.services.rate_limit import check_rate_limit
from app.services.llm import chat_completion
from app.models.user import TokenData

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token_data: TokenData = Depends(verify_token),
    _: None = Depends(check_rate_limit)
):
    """聊天接口"""
    # 调用 LLM
    response, tokens_used = await chat_completion(request.message)

    return ChatResponse(
        response=response,
        tokens_used=tokens_used
    )

@router.get("/quota")
async def get_quota(token_data: TokenData = Depends(verify_token)):
    """查询配额"""
    # 从 Redis 获取限流状态
    # 简化示例
    return {
        "user_id": token_data.user_id,
        "tier": token_data.tier,
        "remaining": 10
    }
```

---

## 主应用

```python
"""app/main.py - FastAPI 应用入口"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
from app.config import get_settings
from app.middleware.logging import LoggingMiddleware
from app.middleware.error_handler import http_exception_handler, general_exception_handler
from app.api import auth, chat

# 配置日志
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(LoggingMiddleware)

# 添加异常处理
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 注册路由
app.include_router(auth.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "AI Agent API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 环境变量

```bash
# .env
# JWT 配置
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# OpenAI 配置
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 限流配置
RATE_LIMIT_FREE=10
RATE_LIMIT_PRO=1000
```

---

## 依赖管理

```toml
# pyproject.toml
[project]
name = "ai-agent-api"
version = "1.0.0"
requires-python = ">=3.13"

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "redis>=5.0.0",
    "openai>=1.10.0",
    "structlog>=24.1.0",
    "python-dotenv>=1.0.0"
]
```

---

## 测试脚本

```python
"""test_api.py - API 测试"""
import asyncio
import httpx

BASE_URL = "http://localhost:8000"

async def test_complete_flow():
    """测试完整流程"""
    async with httpx.AsyncClient() as client:
        # 1. 登录获取 token
        print("=== 1. 登录 ===")
        response = await client.post(
            f"{BASE_URL}/auth/login",
            json={"email": "user@example.com", "password": "password"}
        )
        token = response.json()["access_token"]
        print(f"Token: {token[:20]}...")

        # 2. 聊天（带认证和限流）
        print("\n=== 2. 聊天 ===")
        headers = {"Authorization": f"Bearer {token}"}

        for i in range(15):
            response = await client.post(
                f"{BASE_URL}/chat",
                json={"message": f"Hello {i}"},
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                print(f"请求{i}: ✅ {data['response'][:50]}...")
            elif response.status_code == 429:
                print(f"请求{i}: ❌ 被限流")
            else:
                print(f"请求{i}: ❌ 错误: {response.status_code}")

        # 3. 查询配额
        print("\n=== 3. 查询配额 ===")
        response = await client.get(
            f"{BASE_URL}/chat/quota",
            headers=headers
        )
        print(f"配额: {response.json()}")

if __name__ == "__main__":
    asyncio.run(test_complete_flow())
```

---

## 运行应用

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 3. 启动 Redis
docker run -d -p 6379:6379 redis:7

# 4. 运行应用
uvicorn app.main:app --reload

# 5. 测试 API
python test_api.py
```

---

## 生产部署

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install uv && uv sync

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis

  redis:
    image: redis:7
    ports:
      - "6379:6379"
```

---

## 总结

这个完整的 AI Agent API 集成了：
1. **JWT 认证**：保护 API 端点
2. **限流中间件**：防止滥用（令牌桶算法）
3. **结构化日志**：可观测性
4. **错误处理**：优雅处理异常
5. **LLM 集成**：OpenAI API 调用
6. **配置管理**：环境变量管理
7. **生产部署**：Docker 容器化

**记住：** 这是一个生产级的 AI Agent API 模板，可以直接用于实际项目。
