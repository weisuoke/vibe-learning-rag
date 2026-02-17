# 核心概念6：FastAPI中间件实现

> 在 FastAPI 中集成限流的三种方式

---

## 概述

FastAPI 提供了多种方式集成限流：
1. **依赖注入**（推荐）
2. **装饰器**
3. **全局中间件**

---

## 方式1：依赖注入（推荐）

### 基本用法

```python
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

async def rate_limit(user_id: str = "anonymous"):
    """限流依赖"""
    limiter = TokenBucket(rate=10, capacity=10)
    if not limiter.acquire():
        raise HTTPException(status_code=429, detail="Too many requests")

@app.post("/chat")
async def chat(message: str, _: None = Depends(rate_limit)):
    return {"response": message}
```

### 优点

- ✅ 代码清晰（限流逻辑独立）
- ✅ 可复用（多个端点共享）
- ✅ 易测试（可以 mock 依赖）
- ✅ 灵活（可以传参数）

---

## 方式2：装饰器

### 基本用法

```python
from functools import wraps

def rate_limit_decorator(rate: float, capacity: float):
    """限流装饰器"""
    def decorator(func):
        limiter = TokenBucket(rate, capacity)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not limiter.acquire():
                raise HTTPException(status_code=429)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/chat")
@rate_limit_decorator(rate=10, capacity=10)
async def chat(message: str):
    return {"response": message}
```

### 优点

- ✅ 简洁（一行代码）
- ✅ 直观（限流配置在路由旁边）

### 缺点

- ❌ 难以测试（装饰器难以 mock）
- ❌ 不够灵活（无法动态调整）

---

## 方式3：全局中间件

### 基本用法

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """全局限流中间件"""

    def __init__(self, app, rate: float, capacity: float):
        super().__init__(app)
        self.limiter = TokenBucket(rate, capacity)

    async def dispatch(self, request: Request, call_next):
        if not self.limiter.acquire():
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests"}
            )

        response = await call_next(request)
        return response

# 添加中间件
app.add_middleware(RateLimitMiddleware, rate=100, capacity=100)
```

### 优点

- ✅ 全局生效（所有端点）
- ✅ 统一管理

### 缺点

- ❌ 不够灵活（无法针对不同端点）
- ❌ 难以测试

---

## 推荐方案：依赖注入 + 工厂函数

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from typing import Annotated
import redis

app = FastAPI()
redis_client = redis.Redis()

def create_rate_limiter(rate: float, capacity: float):
    """限流器工厂函数"""
    async def rate_limit(request: Request):
        # 从请求中获取用户ID
        user_id = request.headers.get("X-User-ID", "anonymous")

        # 创建限流器
        limiter = RedisTokenBucket(
            redis_client=redis_client,
            key=f"user:{user_id}:rate_limit",
            rate=rate,
            capacity=capacity
        )

        # 检查限流
        if not limiter.acquire():
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )

    return rate_limit

# 使用
@app.post("/chat")
async def chat(
    message: str,
    _: None = Depends(create_rate_limiter(rate=10, capacity=100))
):
    return {"response": message}

@app.post("/embedding")
async def embedding(
    text: str,
    _: None = Depends(create_rate_limiter(rate=100, capacity=1000))
):
    return {"embedding": [0.1, 0.2, 0.3]}
```

---

## 多维度限流

```python
class MultiDimensionLimiter:
    """多维度限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_limit(
        self,
        user_id: str,
        ip: str,
        endpoint: str,
        user_tier: str
    ):
        # 1. 全局限流
        global_limiter = RedisTokenBucket(
            self.redis, "global", rate=1000, capacity=1000
        )
        if not global_limiter.acquire():
            raise HTTPException(status_code=503, detail="Service overloaded")

        # 2. IP 限流
        ip_limiter = RedisTokenBucket(
            self.redis, f"ip:{ip}", rate=10, capacity=10
        )
        if not ip_limiter.acquire():
            raise HTTPException(status_code=429, detail="Too many requests from this IP")

        # 3. 用户限流
        limits = {
            "free": {"rate": 1, "capacity": 10},
            "pro": {"rate": 100, "capacity": 1000}
        }
        user_limit = limits[user_tier]
        user_limiter = RedisTokenBucket(
            self.redis, f"user:{user_id}", **user_limit
        )
        if not user_limiter.acquire():
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {user_tier}")

# 使用
limiter = MultiDimensionLimiter(redis_client)

async def rate_limit_dependency(request: Request, user: User = Depends(get_current_user)):
    await limiter.check_limit(
        user_id=user.id,
        ip=request.client.host,
        endpoint=request.url.path,
        user_tier=user.tier
    )

@app.post("/chat")
async def chat(message: str, _: None = Depends(rate_limit_dependency)):
    return {"response": message}
```

---

## 错误处理

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """限流错误处理"""
    if exc.status_code == 429:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "You have sent too many requests. Please try again later.",
                "retry_after": 60,
                "upgrade_url": "/pricing"
            },
            headers={"Retry-After": "60"}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

---

## 监控集成

```python
from prometheus_client import Counter, Histogram

rate_limit_requests = Counter(
    'rate_limit_requests_total',
    'Total rate limit checks',
    ['endpoint', 'result']
)

rate_limit_latency = Histogram(
    'rate_limit_check_duration_seconds',
    'Rate limit check latency'
)

async def rate_limit_with_metrics(request: Request):
    """带监控的限流"""
    endpoint = request.url.path

    with rate_limit_latency.time():
        try:
            await limiter.check_limit(...)
            rate_limit_requests.labels(endpoint, 'allowed').inc()
        except HTTPException:
            rate_limit_requests.labels(endpoint, 'rejected').inc()
            raise
```

---

## 总结

| 方式 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| 依赖注入 | 灵活、易测试 | 略复杂 | ⭐⭐⭐ 推荐 |
| 装饰器 | 简洁 | 难测试 | ⭐⭐ 可选 |
| 全局中间件 | 统一管理 | 不灵活 | ⭐ 不推荐 |

---

**记住：** 推荐使用依赖注入方式，配合工厂函数实现灵活的限流策略。
