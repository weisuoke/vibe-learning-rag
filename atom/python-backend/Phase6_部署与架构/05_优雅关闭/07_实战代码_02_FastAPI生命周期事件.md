# 实战代码：FastAPI生命周期事件

> 完整可运行的 FastAPI lifespan 上下文管理器示例

---

## 代码说明

本示例演示如何使用 FastAPI 的 lifespan 上下文管理器管理应用生命周期，包括：
- 使用 lifespan 替代 startup/shutdown 事件
- 初始化和清理数据库连接池
- 初始化和清理 Redis 连接
- 与优雅关闭集成

**运行环境：**
- Python 3.13+
- FastAPI
- SQLAlchemy (async)
- Redis (async)

---

## 完整代码

```python
"""
FastAPI生命周期事件实现
演示：使用 lifespan 上下文管理器管理资源

运行方式：
    python 07_实战代码_02_FastAPI生命周期事件.py

测试方式：
    1. 启动应用
    2. 访问 http://localhost:8000/
    3. 按 Ctrl+C 测试优雅关闭
"""

import signal
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from redis.asyncio import Redis
import uvicorn

# ===== 1. 全局状态 =====
shutdown_event = asyncio.Event()
accepting_requests = True
active_requests = 0

# ===== 2. 信号处理器 =====
def signal_handler(signum, frame):
    """信号处理器"""
    signal_name = signal.Signals(signum).name
    print(f"\n收到信号: {signal_name} ({signum})")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== 3. lifespan 上下文管理器 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    使用 lifespan 上下文管理器的优点：
    1. 启动和关闭逻辑在同一个函数中
    2. 可以使用局部变量
    3. 更符合 Python 的上下文管理器模式
    """
    print("\n" + "="*60)
    print("应用启动")
    print("="*60)

    # ===== 启动阶段 =====

    # 1. 初始化数据库连接池
    print("\n1. 初始化数据库连接池...")
    try:
        app.state.db_engine = create_async_engine(
            # 使用 SQLite 进行演示（实际应用中使用 PostgreSQL）
            "sqlite+aiosqlite:///./test.db",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )

        # 测试连接
        async with app.state.db_engine.connect() as conn:
            await conn.execute("SELECT 1")

        print("✓ 数据库连接池初始化成功")

    except Exception as e:
        print(f"✗ 数据库连接池初始化失败: {e}")
        raise

    # 2. 初始化 Redis 连接
    print("\n2. 初始化 Redis 连接...")
    try:
        app.state.redis_client = Redis.from_url(
            "redis://localhost:6379/0",
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
        )

        # 测试连接
        await app.state.redis_client.ping()

        print("✓ Redis 连接初始化成功")

    except Exception as e:
        print(f"✗ Redis 连接初始化失败: {e}")
        print("  提示：请确保 Redis 服务正在运行")
        # Redis 失败不阻止应用启动
        app.state.redis_client = None

    # 3. 启动关闭监听器
    print("\n3. 启动关闭监听器...")
    asyncio.create_task(shutdown_monitor(app))
    print("✓ 关闭监听器已启动")

    print("\n" + "="*60)
    print("应用启动完成")
    print("="*60 + "\n")

    # ===== 应用运行 =====
    yield  # 在这里应用开始处理请求

    # ===== 关闭阶段 =====
    # 注意：这部分由 shutdown_monitor 触发
    # 这里只是最后的清理工作

    print("\n" + "="*60)
    print("FastAPI lifespan 关闭阶段")
    print("="*60)

    # 清理资源（如果 shutdown_monitor 没有清理）
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        try:
            await app.state.redis_client.close()
            print("✓ Redis 连接已关闭")
        except Exception as e:
            print(f"✗ Redis 关闭失败: {e}")

    if hasattr(app.state, "db_engine") and app.state.db_engine:
        try:
            await app.state.db_engine.dispose()
            print("✓ 数据库连接池已关闭")
        except Exception as e:
            print(f"✗ 数据库关闭失败: {e}")

    print("="*60 + "\n")

# 创建 FastAPI 应用
app = FastAPI(
    title="FastAPI生命周期示例",
    lifespan=lifespan  # 使用 lifespan 上下文管理器
)

# ===== 4. 请求排空中间件 =====
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """请求排空中间件"""
    global active_requests

    if not accepting_requests:
        return JSONResponse(
            status_code=503,
            content={"error": "Server is shutting down"}
        )

    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

# ===== 5. 示例路由 =====
@app.get("/")
async def root(request: Request):
    """根路由"""
    return {
        "message": "Hello World",
        "db_connected": hasattr(request.app.state, "db_engine"),
        "redis_connected": hasattr(request.app.state, "redis_client")
                          and request.app.state.redis_client is not None,
    }

@app.get("/db-test")
async def db_test(request: Request):
    """测试数据库连接"""
    if not hasattr(request.app.state, "db_engine"):
        return {"error": "Database not initialized"}

    try:
        async with request.app.state.db_engine.connect() as conn:
            result = await conn.execute("SELECT 1 as test")
            row = result.fetchone()

        return {
            "status": "success",
            "result": row[0] if row else None
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/redis-test")
async def redis_test(request: Request):
    """测试 Redis 连接"""
    if not hasattr(request.app.state, "redis_client") or not request.app.state.redis_client:
        return {"error": "Redis not initialized"}

    try:
        # 设置值
        await request.app.state.redis_client.set("test_key", "test_value", ex=60)

        # 获取值
        value = await request.app.state.redis_client.get("test_key")

        return {
            "status": "success",
            "value": value
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/pool-status")
async def pool_status(request: Request):
    """获取连接池状态"""
    if not hasattr(request.app.state, "db_engine"):
        return {"error": "Database not initialized"}

    pool = request.app.state.db_engine.pool

    return {
        "database": {
            "pool_size": pool.size(),
            "checkedout": pool.checkedout(),
            "overflow": pool.overflow(),
            "idle": pool.size() - pool.checkedout(),
        },
        "requests": {
            "accepting": accepting_requests,
            "active": active_requests,
        }
    }

# ===== 6. 优雅关闭逻辑 =====
async def wait_for_requests(timeout: int = 30):
    """等待所有请求完成"""
    global accepting_requests
    accepting_requests = False

    print(f"\n[请求排空] 停止接收新请求")
    print(f"[请求排空] 当前活跃请求数: {active_requests}")

    start_time = asyncio.get_event_loop().time()

    while active_requests > 0:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            print(f"[请求排空] 超时（{timeout}秒）")
            break

        await asyncio.sleep(0.1)

    print(f"[请求排空] 完成")

async def cleanup_resources(app: FastAPI):
    """清理资源"""
    print(f"\n[资源清理] 开始清理...")

    # 1. 关闭 Redis
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        try:
            await app.state.redis_client.close()
            await app.state.redis_client.connection_pool.disconnect()
            print("[资源清理] ✓ Redis 已关闭")
        except Exception as e:
            print(f"[资源清理] ✗ Redis 关闭失败: {e}")

    # 2. 关闭数据库
    if hasattr(app.state, "db_engine") and app.state.db_engine:
        try:
            await app.state.db_engine.dispose()
            print("[资源清理] ✓ 数据库已关闭")
        except Exception as e:
            print(f"[资源清理] ✗ 数据库关闭失败: {e}")

    print(f"[资源清理] 完成")

async def graceful_shutdown(app: FastAPI):
    """优雅关闭"""
    print("\n" + "="*60)
    print("开始优雅关闭")
    print("="*60)

    # 1. 等待请求完成
    await wait_for_requests()

    # 2. 清理资源
    await cleanup_resources(app)

    print("\n" + "="*60)
    print("优雅关闭完成")
    print("="*60 + "\n")

    sys.exit(0)

async def shutdown_monitor(app: FastAPI):
    """监听关闭信号"""
    await shutdown_event.wait()
    await graceful_shutdown(app)

# ===== 7. 主函数 =====
def main():
    """主函数"""
    print("="*60)
    print("FastAPI生命周期事件示例")
    print("="*60 + "\n")

    print("提示：")
    print("  - 访问 http://localhost:8000/ 测试基本功能")
    print("  - 访问 http://localhost:8000/db-test 测试数据库")
    print("  - 访问 http://localhost:8000/redis-test 测试 Redis")
    print("  - 访问 http://localhost:8000/pool-status 查看连接池状态")
    print("  - 按 Ctrl+C 测试优雅关闭")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
```

---

## 运行示例

### 1. 启动应用

```bash
python 07_实战代码_02_FastAPI生命周期事件.py
```

**预期输出：**
```
============================================================
FastAPI生命周期示例
============================================================

提示：
  - 访问 http://localhost:8000/ 测试基本功能
  - 访问 http://localhost:8000/db-test 测试数据库
  - 访问 http://localhost:8000/redis-test 测试 Redis
  - 访问 http://localhost:8000/pool-status 查看连接池状态
  - 按 Ctrl+C 测试优雅关闭

============================================================
应用启动
============================================================

1. 初始化数据库连接池...
✓ 数据库连接池初始化成功

2. 初始化 Redis 连接...
✓ Redis 连接初始化成功

3. 启动关闭监听器...
✓ 关闭监听器已启动

============================================================
应用启动完成
============================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2. 测试基本功能

```bash
curl http://localhost:8000/
```

**预期输出：**
```json
{
  "message": "Hello World",
  "db_connected": true,
  "redis_connected": true
}
```

### 3. 测试数据库连接

```bash
curl http://localhost:8000/db-test
```

**预期输出：**
```json
{
  "status": "success",
  "result": 1
}
```

### 4. 测试 Redis 连接

```bash
curl http://localhost:8000/redis-test
```

**预期输出：**
```json
{
  "status": "success",
  "value": "test_value"
}
```

### 5. 查看连接池状态

```bash
curl http://localhost:8000/pool-status
```

**预期输出：**
```json
{
  "database": {
    "pool_size": 10,
    "checkedout": 0,
    "overflow": 0,
    "idle": 10
  },
  "requests": {
    "accepting": true,
    "active": 0
  }
}
```

### 6. 测试优雅关闭

按 `Ctrl+C`：

**预期输出：**
```
^C
收到信号: SIGINT (2)

============================================================
开始优雅关闭
============================================================

[请求排空] 停止接收新请求
[请求排空] 当前活跃请求数: 0
[请求排空] 完成

[资源清理] 开始清理...
[资源清理] ✓ Redis 已关闭
[资源清理] ✓ 数据库已关闭
[资源清理] 完成

============================================================
优雅关闭完成
============================================================

============================================================
FastAPI lifespan 关闭阶段
============================================================
✓ Redis 连接已关闭
✓ 数据库连接池已关闭
============================================================
```

---

## 代码解析

### 1. lifespan 上下文管理器

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    app.state.db_engine = create_async_engine(...)
    app.state.redis_client = Redis.from_url(...)

    yield  # 应用运行

    # 关闭阶段
    await app.state.redis_client.close()
    await app.state.db_engine.dispose()
```

**关键点：**
- 使用 `@asynccontextmanager` 装饰器
- `yield` 前是启动逻辑
- `yield` 后是关闭逻辑
- 使用 `app.state` 存储资源

### 2. 资源存储

```python
# 在 lifespan 中初始化
app.state.db_engine = create_async_engine(...)

# 在路由中使用
@app.get("/")
async def root(request: Request):
    engine = request.app.state.db_engine
```

**关键点：**
- 使用 `app.state` 存储全局资源
- 在路由中通过 `request.app.state` 访问
- 避免使用全局变量

### 3. 错误处理

```python
try:
    app.state.db_engine = create_async_engine(...)
    # 测试连接
    async with app.state.db_engine.connect() as conn:
        await conn.execute("SELECT 1")
    print("✓ 数据库连接成功")
except Exception as e:
    print(f"✗ 数据库连接失败: {e}")
    raise  # 阻止应用启动
```

**关键点：**
- 在启动阶段测试连接
- 连接失败时抛出异常，阻止应用启动
- 对于可选资源（如 Redis），可以不抛出异常

---

## 与 startup/shutdown 事件的对比

### 使用 startup/shutdown（旧方式）

```python
db_engine = None  # 需要全局变量

@app.on_event("startup")
async def startup():
    global db_engine
    db_engine = create_async_engine(...)

@app.on_event("shutdown")
async def shutdown():
    global db_engine
    await db_engine.dispose()
```

### 使用 lifespan（推荐）

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 局部变量，更清晰
    db_engine = create_async_engine(...)
    app.state.db_engine = db_engine

    yield

    await db_engine.dispose()

app = FastAPI(lifespan=lifespan)
```

**优点：**
- 启动和关闭逻辑在同一个函数中
- 可以使用局部变量
- 更符合 Python 的上下文管理器模式
- FastAPI 官方推荐

---

## 扩展练习

### 练习1：添加向量数据库

```python
from pymilvus import connections

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    app.state.db_engine = create_async_engine(...)
    app.state.redis_client = Redis.from_url(...)

    # 连接向量数据库
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    app.state.vector_db_connected = True

    yield

    # 关闭阶段
    if app.state.vector_db_connected:
        connections.disconnect("default")

    await app.state.redis_client.close()
    await app.state.db_engine.dispose()
```

### 练习2：添加 Embedding 模型

```python
from sentence_transformers import SentenceTransformer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段
    app.state.db_engine = create_async_engine(...)
    app.state.redis_client = Redis.from_url(...)

    # 加载 Embedding 模型
    app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    yield

    # 关闭阶段
    del app.state.embedding_model

    await app.state.redis_client.close()
    await app.state.db_engine.dispose()
```

### 练习3：添加健康检查

```python
@app.get("/health")
async def health_check(request: Request):
    """健康检查端点"""
    health = {
        "status": "healthy",
        "database": False,
        "redis": False,
    }

    # 检查数据库
    if hasattr(request.app.state, "db_engine"):
        try:
            async with request.app.state.db_engine.connect() as conn:
                await conn.execute("SELECT 1")
            health["database"] = True
        except:
            health["status"] = "unhealthy"

    # 检查 Redis
    if hasattr(request.app.state, "redis_client") and request.app.state.redis_client:
        try:
            await request.app.state.redis_client.ping()
            health["redis"] = True
        except:
            health["status"] = "unhealthy"

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)
```

---

## 总结

本示例演示了：

1. **lifespan 上下文管理器**：管理应用生命周期
2. **资源初始化**：数据库连接池、Redis 连接
3. **资源清理**：正确关闭所有连接
4. **错误处理**：启动阶段的错误处理
5. **与优雅关闭集成**：配合信号处理实现完整的优雅关闭

**关键要点：**
- 使用 lifespan 替代 startup/shutdown 事件
- 使用 app.state 存储全局资源
- 在启动阶段测试连接
- 在关闭阶段正确清理资源

**下一步：** 学习如何实现数据库连接池的优雅关闭。
