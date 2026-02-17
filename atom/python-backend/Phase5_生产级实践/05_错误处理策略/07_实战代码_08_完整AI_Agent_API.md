# 实战代码8：完整AI Agent API

**场景：** 端到端 AI Agent API，集成所有错误处理策略

---

## 完整代码

```python
"""
完整的 AI Agent API
集成：异常分层、重试、熔断、超时、错误监控
"""

import asyncio
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openai import AsyncOpenAI, Timeout, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

# ===== 配置 =====
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()
client = AsyncOpenAI()

# ===== 异常体系 =====
class AppError(Exception):
    def __init__(self, message: str, error_code: str = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.status_code = status_code
        super().__init__(message)

class BusinessError(AppError):
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message, error_code, 422)

class TextTooLongError(BusinessError):
    pass

class LLMTimeoutError(AppError):
    def __init__(self):
        super().__init__("LLM 服务响应超时", "LLM_TIMEOUT", 504)

# ===== 熔断器 =====
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: float = 0.5, timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise AppError("服务暂时不可用（熔断中）", "CIRCUIT_BREAKER_OPEN", 503)

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        total = self.failure_count + self.success_count
        if total >= 5 and self.failure_count / total > self.failure_threshold:
            self.state = CircuitState.OPEN

llm_breaker = CircuitBreaker("LLM")

# ===== LLM 调用 =====
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Timeout, RateLimitError))
)
async def call_llm(prompt: str) -> str:
    try:
        async with asyncio.timeout(30):
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                timeout=25
            )
            return response.choices[0].message.content
    except asyncio.TimeoutError:
        raise LLMTimeoutError()

# ===== 中间件 =====
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# ===== FastAPI 应用 =====
app = FastAPI()
app.add_middleware(RequestIDMiddleware)

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("app_error", request_id=request_id, error_code=exc.error_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "error_code": exc.error_code, "request_id": request_id}
    )

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("unhandled_exception", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "request_id": request_id}
    )

# ===== API 端点 =====
@app.post("/chat")
async def chat(message: str, request: Request):
    request_id = request.state.request_id
    logger.info("chat_start", request_id=request_id)

    # 业务校验
    if len(message) > 2000:
        raise TextTooLongError("输入文本过长，请缩短至 2000 字以内")

    # LLM 调用（带重试、熔断、超时）
    response = await llm_breaker.call(call_llm, message)

    logger.info("chat_success", request_id=request_id)
    return {"response": response, "request_id": request_id}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "circuit_breaker": {
            "llm": {
                "state": llm_breaker.state.value,
                "failure_count": llm_breaker.failure_count,
                "success_count": llm_breaker.success_count
            }
        }
    }
```

---

## 测试

```bash
# 启动服务
uvicorn main:app --reload

# 正常请求
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 文本过长
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "很长的文本..."}'

# 健康检查
curl "http://localhost:8000/health"
```
