# 实战代码02：FastAPI日志集成

## 学习目标

将structlog集成到FastAPI应用中，实现完整的HTTP请求日志记录。

---

## 第一步：基础FastAPI应用

### 示例1：最简单的FastAPI应用

```python
# examples/logging/10_basic_fastapi.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    return {"user_id": user_id, "name": "John"}

# 运行：uvicorn 10_basic_fastapi:app --reload
```

**问题：** 没有任何日志，无法追踪请求

---

## 第二步：添加基础日志

### 示例2：在路由中添加日志

```python
# examples/logging/11_fastapi_with_logging.py
from fastapi import FastAPI
import structlog

# 配置structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

@app.get("/")
async def root():
    logger.info("root_endpoint_called")
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    logger.info("get_user_called", user_id=user_id)
    return {"user_id": user_id, "name": "John"}
```

**输出：**
```
2024-01-15 10:30:45 [info     ] root_endpoint_called
2024-01-15 10:30:46 [info     ] get_user_called        user_id=user_123
```

**问题：** 需要在每个路由中手动添加日志

---

## 第三步：使用中间件自动记录请求

### 示例3：基础日志中间件

```python
# examples/logging/12_fastapi_middleware.py
from fastapi import FastAPI, Request
import structlog
import time

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # 记录请求开始
    start_time = time.time()

    logger.info("request_start",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )

    # 处理请求
    response = await call_next(request)

    # 记录请求结束
    duration_ms = (time.time() - start_time) * 1000

    logger.info("request_end",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms
    )

    return response

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    return {"user_id": user_id, "name": "John"}
```

**输出：**
```
2024-01-15 10:30:45 [info     ] request_start          method=GET path=/ client_ip=127.0.0.1
2024-01-15 10:30:45 [info     ] request_end            method=GET path=/ status_code=200 duration_ms=5.2
```

**优势：** 所有请求自动记录日志

---

## 第四步：替换uvicorn的默认日志

### 示例4：配置uvicorn使用structlog

```python
# examples/logging/13_uvicorn_structlog.py
from fastapi import FastAPI, Request
import structlog
import logging
import time

# 配置structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory()
)

# 配置标准库logging（uvicorn使用）
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO
)

app = FastAPI()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    logger.info("request_start",
        method=request.method,
        path=request.url.path
    )

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    logger.info("request_end",
        status_code=response.status_code,
        duration_ms=duration_ms
    )

    return response

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 运行时配置uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # 禁用uvicorn的默认日志配置
    )
```

---

## 第五步：记录请求和响应体

### 示例5：记录请求体和响应体

```python
# examples/logging/14_log_request_response.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
import time
import json

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    # 读取请求体
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        # 重要：需要重新设置body，否则路由无法读取
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive

    # 记录请求
    logger.info("request_received",
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params),
        body_size=len(body) if body else 0,
        body_preview=body[:100].decode() if body else None  # 只记录前100字节
    )

    # 处理请求
    response = await call_next(request)

    # 记录响应
    duration_ms = (time.time() - start_time) * 1000

    logger.info("response_sent",
        status_code=response.status_code,
        duration_ms=duration_ms
    )

    return response

@app.post("/users")
async def create_user(user: dict):
    return {"id": "user_123", **user}

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    return {"user_id": user_id, "name": "John"}
```

**测试：**
```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "email": "john@example.com"}'
```

**输出：**
```json
{"level": "info", "event": "request_received", "method": "POST", "path": "/users", "body_size": 45, "body_preview": "{\"name\": \"John\", \"email\": \"john@example.com\"}", "timestamp": "..."}
{"level": "info", "event": "response_sent", "status_code": 200, "duration_ms": 5.2, "timestamp": "..."}
```

---

## 第六步：过滤特定路径

### 示例6：跳过健康检查日志

```python
# examples/logging/15_skip_health_check.py
from fastapi import FastAPI, Request
import structlog
import time

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

# 不需要记录日志的路径
SKIP_PATHS = {"/health", "/metrics", "/favicon.ico"}

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # 跳过特定路径
    if request.url.path in SKIP_PATHS:
        return await call_next(request)

    start_time = time.time()

    logger.info("request_start",
        method=request.method,
        path=request.url.path
    )

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    logger.info("request_end",
        status_code=response.status_code,
        duration_ms=duration_ms
    )

    return response

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# 访问 / 会记录日志
# 访问 /health 不会记录日志
```

---

## 第七步：记录异常

### 示例7：捕获和记录异常

```python
# examples/logging/16_exception_logging.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import structlog
import time
import traceback

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    logger.info("request_start",
        method=request.method,
        path=request.url.path
    )

    try:
        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000

        logger.info("request_end",
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        # 记录异常
        logger.error("request_failed",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            duration_ms=duration_ms
        )

        # 返回500错误
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/error")
async def error():
    raise ValueError("Something went wrong")

@app.get("/http-error")
async def http_error():
    raise HTTPException(status_code=404, detail="Not found")
```

**测试：**
```bash
curl http://localhost:8000/error
```

**输出：**
```json
{
  "level": "error",
  "event": "request_failed",
  "error": "Something went wrong",
  "error_type": "ValueError",
  "stack_trace": "Traceback (most recent call last):\n  ...",
  "duration_ms": 2.5,
  "timestamp": "..."
}
```

---

## 第八步：完整的生产级中间件

### 示例8：生产级日志中间件

```python
# examples/logging/17_production_middleware.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
import time
import traceback
from typing import Set

# 配置structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

# 配置
SKIP_PATHS: Set[str] = {"/health", "/metrics"}
LOG_REQUEST_BODY: bool = True
LOG_RESPONSE_BODY: bool = False
MAX_BODY_LOG_SIZE: int = 1000  # 最多记录1000字节

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """生产级日志中间件"""

    # 跳过特定路径
    if request.url.path in SKIP_PATHS:
        return await call_next(request)

    start_time = time.time()

    # 准备请求日志数据
    log_data = {
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params) if request.query_params else None,
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent"),
    }

    # 记录请求体（可选）
    if LOG_REQUEST_BODY and request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                # 只记录前N字节
                body_preview = body[:MAX_BODY_LOG_SIZE].decode("utf-8", errors="ignore")
                log_data["body_size"] = len(body)
                log_data["body_preview"] = body_preview

                # 重新设置body
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
        except Exception as e:
            logger.warning("failed_to_read_request_body", error=str(e))

    # 记录请求开始
    logger.info("request_start", **log_data)

    try:
        # 处理请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 记录请求结束
        logger.info("request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        # 添加响应头
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 记录异常
        logger.error("request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            duration_ms=duration_ms
        )

        # 返回500错误
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
            headers={"X-Response-Time": f"{duration_ms:.2f}ms"}
        )

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/users")
async def create_user(user: dict):
    return {"id": "user_123", **user}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/error")
async def error():
    raise ValueError("Test error")
```

---

## 第九步：添加性能监控

### 示例9：慢请求告警

```python
# examples/logging/18_slow_request_alert.py
from fastapi import FastAPI, Request
import structlog
import time
import asyncio

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

app = FastAPI()
logger = structlog.get_logger()

# 慢请求阈值（毫秒）
SLOW_REQUEST_THRESHOLD_MS = 1000

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    logger.info("request_start",
        method=request.method,
        path=request.url.path
    )

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    # 判断是否为慢请求
    if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
        logger.warning("slow_request",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            threshold_ms=SLOW_REQUEST_THRESHOLD_MS
        )
    else:
        logger.info("request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms
        )

    return response

@app.get("/fast")
async def fast():
    return {"message": "Fast response"}

@app.get("/slow")
async def slow():
    await asyncio.sleep(2)  # 模拟慢请求
    return {"message": "Slow response"}
```

**测试：**
```bash
curl http://localhost:8000/fast  # 正常日志
curl http://localhost:8000/slow  # 慢请求告警
```

---

## 第十步：可复用的日志中间件模块

### 示例10：独立的中间件模块

```python
# middleware/logging.py
"""
FastAPI日志中间件

使用方法：
    from middleware.logging import setup_logging_middleware

    app = FastAPI()
    setup_logging_middleware(app)
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
import time
import traceback
from typing import Set, Optional

logger = structlog.get_logger()

class LoggingMiddlewareConfig:
    """日志中间件配置"""

    def __init__(
        self,
        skip_paths: Optional[Set[str]] = None,
        log_request_body: bool = True,
        log_response_body: bool = False,
        max_body_log_size: int = 1000,
        slow_request_threshold_ms: float = 1000
    ):
        self.skip_paths = skip_paths or {"/health", "/metrics"}
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_log_size = max_body_log_size
        self.slow_request_threshold_ms = slow_request_threshold_ms


async def logging_middleware(
    request: Request,
    call_next,
    config: LoggingMiddlewareConfig
):
    """日志中间件"""

    # 跳过特定路径
    if request.url.path in config.skip_paths:
        return await call_next(request)

    start_time = time.time()

    # 准备请求日志
    log_data = {
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params) if request.query_params else None,
        "client_ip": request.client.host,
    }

    # 记录请求体
    if config.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                body_preview = body[:config.max_body_log_size].decode("utf-8", errors="ignore")
                log_data["body_size"] = len(body)
                log_data["body_preview"] = body_preview

                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
        except Exception as e:
            logger.warning("failed_to_read_request_body", error=str(e))

    # 记录请求开始
    logger.info("request_start", **log_data)

    try:
        # 处理请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 判断是否为慢请求
        if duration_ms > config.slow_request_threshold_ms:
            logger.warning("slow_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                threshold_ms=config.slow_request_threshold_ms
            )
        else:
            logger.info("request_end",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms
            )

        # 添加响应头
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        logger.error("request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            duration_ms=duration_ms
        )

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
            headers={"X-Response-Time": f"{duration_ms:.2f}ms"}
        )


def setup_logging_middleware(
    app: FastAPI,
    config: Optional[LoggingMiddlewareConfig] = None
):
    """
    为FastAPI应用添加日志中间件

    Args:
        app: FastAPI应用实例
        config: 中间件配置，默认使用默认配置
    """
    if config is None:
        config = LoggingMiddlewareConfig()

    @app.middleware("http")
    async def _logging_middleware(request: Request, call_next):
        return await logging_middleware(request, call_next, config)


# 使用示例
if __name__ == "__main__":
    from fastapi import FastAPI
    import structlog

    # 配置structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ]
    )

    # 创建应用
    app = FastAPI()

    # 添加日志中间件
    setup_logging_middleware(app, LoggingMiddlewareConfig(
        skip_paths={"/health"},
        slow_request_threshold_ms=500
    ))

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # 运行
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 总结

### 核心要点

1. **中间件自动记录**
   - 所有请求自动记录日志
   - 不需要在每个路由中手动添加

2. **性能监控**
   - 记录请求耗时
   - 慢请求告警

3. **异常处理**
   - 捕获和记录所有异常
   - 返回统一的错误响应

4. **可配置**
   - 跳过特定路径
   - 控制日志详细程度
   - 自定义阈值

### 最佳实践

1. 使用中间件而不是在路由中手动记录
2. 跳过健康检查等高频路径
3. 记录请求体时限制大小
4. 添加响应时间到响应头
5. 慢请求单独告警

### 下一步

- 【实战代码03】：请求ID追踪
- 【实战代码04】：上下文变量传递
