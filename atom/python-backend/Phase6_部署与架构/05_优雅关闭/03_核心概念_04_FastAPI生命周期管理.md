# 核心概念：FastAPI生命周期管理

> 深入理解 FastAPI 的 startup/shutdown 事件和 lifespan 上下文管理器

---

## 什么是生命周期管理？

**生命周期管理（Lifecycle Management）** 是指管理应用从启动到关闭的整个生命周期，包括：
- **启动阶段**：初始化资源（数据库连接、缓存、模型加载）
- **运行阶段**：处理请求
- **关闭阶段**：清理资源（关闭连接、保存状态）

**类比：**
- **前端视角**：类似于 React 的 `useEffect` 钩子，组件挂载时初始化，卸载时清理
- **日常视角**：类似于餐厅的开店和关店流程

---

## FastAPI 生命周期事件

### 1. startup 事件（已废弃，但仍可用）

**定义：** 应用启动时执行的函数

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("应用启动")
    # 初始化资源
    await initialize_database()
    await initialize_redis()
    await load_models()
```

**执行时机：**
- 在应用开始接收请求之前
- 只执行一次
- 按注册顺序执行

**使用场景：**
- 初始化数据库连接池
- 连接 Redis
- 加载 ML 模型
- 启动后台任务

### 2. shutdown 事件（已废弃，但仍可用）

**定义：** 应用关闭时执行的函数

```python
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("应用关闭")
    # 清理资源
    await cleanup_database()
    await cleanup_redis()
    await unload_models()
```

**执行时机：**
- 在应用停止接收请求之后
- 只执行一次
- 按注册顺序执行

**使用场景：**
- 关闭数据库连接池
- 关闭 Redis 连接
- 卸载 ML 模型
- 保存应用状态

---

## lifespan 上下文管理器（推荐）

### 1. 基础用法

**定义：** 使用异步上下文管理器管理应用生命周期

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动阶段
    print("应用启动")
    await initialize_resources()

    yield  # 应用运行

    # 关闭阶段
    print("应用关闭")
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)
```

**为什么推荐 lifespan？**
- 更符合 Python 的上下文管理器模式
- 启动和关闭逻辑在同一个函数中，更清晰
- 可以共享局部变量
- FastAPI 官方推荐的方式

### 2. 完整示例

```python
"""
FastAPI lifespan 完整示例
演示：初始化和清理资源
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from redis.asyncio import Redis

# 全局资源（在 lifespan 中初始化）
db_engine = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global db_engine, redis_client

    # ===== 启动阶段 =====
    print("========== 应用启动 ==========")

    # 1. 初始化数据库
    print("初始化数据库连接池...")
    db_engine = create_async_engine(
        "postgresql+asyncpg://user:pass@localhost/db",
        pool_size=10,
        max_overflow=20,
    )

    # 2. 初始化 Redis
    print("初始化 Redis 连接...")
    redis_client = Redis.from_url("redis://localhost:6379/0")

    # 3. 加载模型
    print("加载 Embedding 模型...")
    # embedding_model = load_model()

    print("========== 应用启动完成 ==========\n")

    # ===== 应用运行 =====
    yield  # 在这里应用开始处理请求

    # ===== 关闭阶段 =====
    print("\n========== 应用关闭 ==========")

    # 1. 关闭 Redis
    print("关闭 Redis 连接...")
    if redis_client:
        await redis_client.close()
        await redis_client.connection_pool.disconnect()

    # 2. 关闭数据库
    print("关闭数据库连接池...")
    if db_engine:
        await db_engine.dispose()

    # 3. 卸载模型
    print("卸载 Embedding 模型...")
    # unload_model()

    print("========== 应用关闭完成 ==========")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

---

## 在 lifespan 中管理状态

### 1. 使用 app.state 共享状态

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """使用 app.state 共享状态"""
    # 启动阶段：初始化资源并存储到 app.state
    app.state.db_engine = create_async_engine(DATABASE_URL)
    app.state.redis_client = Redis.from_url(REDIS_URL)
    app.state.embedding_model = load_model()

    yield

    # 关闭阶段：从 app.state 获取资源并清理
    await app.state.db_engine.dispose()
    await app.state.redis_client.close()
    app.state.embedding_model.unload()

app = FastAPI(lifespan=lifespan)

# 在路由中使用
@app.get("/users")
async def get_users(request: Request):
    # 从 request.app.state 访问资源
    async with request.app.state.db_engine.begin() as conn:
        result = await conn.execute("SELECT * FROM users")
        return result.fetchall()
```

### 2. 使用依赖注入

```python
from fastapi import Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化资源"""
    app.state.db_engine = create_async_engine(DATABASE_URL)
    yield
    await app.state.db_engine.dispose()

app = FastAPI(lifespan=lifespan)

# 依赖注入函数
async def get_db(request: Request):
    """获取数据库会话"""
    async with AsyncSession(request.app.state.db_engine) as session:
        yield session

# 在路由中使用依赖注入
@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute("SELECT * FROM users")
    return result.fetchall()
```

---

## 与优雅关闭的集成

### 1. 基础集成

```python
"""
FastAPI lifespan + 优雅关闭
演示：集成信号处理和资源清理
"""

import signal
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"收到信号 {signum}")
    shutdown_event.set()

# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动阶段
    print("应用启动")
    await initialize_resources()

    # 启动关闭监听器
    asyncio.create_task(shutdown_monitor())

    yield

    # 关闭阶段
    print("应用关闭")
    await cleanup_resources()

async def shutdown_monitor():
    """监听关闭信号"""
    await shutdown_event.wait()
    print("收到关闭信号，开始优雅关闭...")
    # 执行优雅关闭逻辑
    await graceful_shutdown()

app = FastAPI(lifespan=lifespan)
```

### 2. 完整的优雅关闭集成

```python
"""
完整的 FastAPI 优雅关闭实现
演示：lifespan + 信号处理 + 请求排空 + 资源清理
"""

import signal
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException

# ===== 全局状态 =====
shutdown_event = asyncio.Event()
accepting_requests = True
active_requests = 0

# ===== 信号处理 =====
def signal_handler(signum, frame):
    print(f"收到信号 {signum}")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== 生命周期管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动阶段
    print("========== 应用启动 ==========")

    # 初始化资源
    app.state.db_engine = create_async_engine(DATABASE_URL)
    app.state.redis_client = Redis.from_url(REDIS_URL)

    # 启动关闭监听器
    asyncio.create_task(shutdown_monitor())

    print("========== 应用启动完成 ==========\n")

    yield

    # 关闭阶段（由 shutdown_monitor 触发）
    print("\n========== 应用关闭 ==========")

    # 清理资源
    await app.state.db_engine.dispose()
    await app.state.redis_client.close()

    print("========== 应用关闭完成 ==========")

app = FastAPI(lifespan=lifespan)

# ===== 请求排空中间件 =====
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    global active_requests

    if not accepting_requests:
        raise HTTPException(503, "Server is shutting down")

    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

# ===== 优雅关闭逻辑 =====
async def shutdown_monitor():
    """监听关闭信号"""
    await shutdown_event.wait()
    await graceful_shutdown()

async def graceful_shutdown():
    """优雅关闭"""
    global accepting_requests

    print("开始优雅关闭...")

    # 1. 停止接收新请求
    accepting_requests = False

    # 2. 等待现有请求完成
    while active_requests > 0:
        print(f"等待 {active_requests} 个请求完成...")
        await asyncio.sleep(0.1)

    print("所有请求已完成")

    # 3. 触发 FastAPI 的 shutdown 事件
    # （lifespan 的清理部分会自动执行）

    # 4. 退出进程
    import sys
    sys.exit(0)

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

---

## startup/shutdown vs lifespan 对比

| 特性 | startup/shutdown | lifespan |
|------|-----------------|----------|
| **语法** | 装饰器 | 上下文管理器 |
| **状态共享** | 需要全局变量 | 可以使用局部变量 |
| **清晰度** | 分散在两个函数 | 集中在一个函数 |
| **官方推荐** | 已废弃 | ✅ 推荐 |
| **错误处理** | 需要分别处理 | 统一处理 |
| **代码量** | 较多 | 较少 |

**示例对比：**

```python
# ===== 使用 startup/shutdown =====
db_engine = None  # 需要全局变量

@app.on_event("startup")
async def startup():
    global db_engine
    db_engine = create_async_engine(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown():
    global db_engine
    await db_engine.dispose()

# ===== 使用 lifespan =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 局部变量，更清晰
    db_engine = create_async_engine(DATABASE_URL)
    app.state.db_engine = db_engine

    yield

    await db_engine.dispose()

app = FastAPI(lifespan=lifespan)
```

---

## 常见模式

### 1. 资源管理器模式

```python
"""
资源管理器模式
演示：封装资源初始化和清理逻辑
"""

class ResourceManager:
    """资源管理器"""

    def __init__(self):
        self.db_engine = None
        self.redis_client = None

    async def initialize(self):
        """初始化资源"""
        self.db_engine = create_async_engine(DATABASE_URL)
        self.redis_client = Redis.from_url(REDIS_URL)

    async def cleanup(self):
        """清理资源"""
        if self.db_engine:
            await self.db_engine.dispose()
        if self.redis_client:
            await self.redis_client.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """使用资源管理器"""
    # 创建资源管理器
    resource_manager = ResourceManager()
    app.state.resources = resource_manager

    # 初始化
    await resource_manager.initialize()

    yield

    # 清理
    await resource_manager.cleanup()

app = FastAPI(lifespan=lifespan)
```

### 2. 工厂模式

```python
"""
工厂模式
演示：根据配置创建不同的资源
"""

def create_lifespan(config: dict):
    """工厂函数：根据配置创建 lifespan"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 根据配置初始化资源
        if config.get("use_database"):
            app.state.db_engine = create_async_engine(config["database_url"])

        if config.get("use_redis"):
            app.state.redis_client = Redis.from_url(config["redis_url"])

        yield

        # 清理资源
        if hasattr(app.state, "db_engine"):
            await app.state.db_engine.dispose()

        if hasattr(app.state, "redis_client"):
            await app.state.redis_client.close()

    return lifespan

# 使用工厂函数
config = {
    "use_database": True,
    "database_url": "postgresql+asyncpg://...",
    "use_redis": True,
    "redis_url": "redis://localhost:6379/0",
}

app = FastAPI(lifespan=create_lifespan(config))
```

---

## AI Agent 后端的生命周期管理

```python
"""
AI Agent 后端的生命周期管理
演示：初始化和清理 AI 相关资源
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from redis.asyncio import Redis
from pymilvus import connections
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """AI Agent 后端生命周期管理"""
    print("========== AI Agent 后端启动 ==========")

    # 1. 初始化数据库
    print("初始化数据库...")
    app.state.db_engine = create_async_engine(
        "postgresql+asyncpg://user:pass@localhost/db"
    )

    # 2. 初始化 Redis
    print("初始化 Redis...")
    app.state.redis_client = Redis.from_url("redis://localhost:6379/0")

    # 3. 连接向量数据库
    print("连接向量数据库...")
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )

    # 4. 加载 Embedding 模型
    print("加载 Embedding 模型...")
    app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. 初始化 LLM 客户端
    print("初始化 LLM 客户端...")
    app.state.llm_client = AsyncOpenAI()

    print("========== AI Agent 后端启动完成 ==========\n")

    yield

    print("\n========== AI Agent 后端关闭 ==========")

    # 1. 关闭 LLM 客户端
    print("关闭 LLM 客户端...")
    await app.state.llm_client.close()

    # 2. 卸载 Embedding 模型
    print("卸载 Embedding 模型...")
    del app.state.embedding_model

    # 3. 断开向量数据库
    print("断开向量数据库...")
    connections.disconnect("default")

    # 4. 关闭 Redis
    print("关闭 Redis...")
    await app.state.redis_client.close()

    # 5. 关闭数据库
    print("关闭数据库...")
    await app.state.db_engine.dispose()

    print("========== AI Agent 后端关闭完成 ==========")

app = FastAPI(lifespan=lifespan)
```

---

## 错误处理

### 1. 启动阶段错误

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理启动阶段错误"""
    try:
        # 初始化资源
        app.state.db_engine = create_async_engine(DATABASE_URL)
        await app.state.db_engine.connect()  # 测试连接
    except Exception as e:
        print(f"启动失败: {e}")
        # 清理已初始化的资源
        if hasattr(app.state, "db_engine"):
            await app.state.db_engine.dispose()
        raise  # 重新抛出异常，阻止应用启动

    yield

    # 清理资源
    await app.state.db_engine.dispose()
```

### 2. 关闭阶段错误

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理关闭阶段错误"""
    # 初始化资源
    app.state.db_engine = create_async_engine(DATABASE_URL)
    app.state.redis_client = Redis.from_url(REDIS_URL)

    yield

    # 清理资源（捕获异常，确保所有资源都尝试清理）
    errors = []

    try:
        await app.state.db_engine.dispose()
    except Exception as e:
        errors.append(f"数据库清理失败: {e}")

    try:
        await app.state.redis_client.close()
    except Exception as e:
        errors.append(f"Redis 清理失败: {e}")

    if errors:
        print("清理过程中发生错误:")
        for error in errors:
            print(f"  - {error}")
```

---

## 总结

### 核心要点

1. **生命周期管理**：
   - 启动阶段：初始化资源
   - 运行阶段：处理请求
   - 关闭阶段：清理资源

2. **推荐使用 lifespan**：
   - 更清晰的代码结构
   - 更好的状态管理
   - FastAPI 官方推荐

3. **与优雅关闭集成**：
   - 在 lifespan 中启动关闭监听器
   - 在关闭阶段清理资源
   - 使用 app.state 共享状态

4. **错误处理**：
   - 启动阶段错误：阻止应用启动
   - 关闭阶段错误：记录但继续清理

### 检查清单

- [ ] 使用 lifespan 而不是 startup/shutdown
- [ ] 在启动阶段初始化所有资源
- [ ] 在关闭阶段清理所有资源
- [ ] 使用 app.state 共享状态
- [ ] 处理启动和关闭阶段的错误
- [ ] 集成信号处理和优雅关闭
- [ ] 测试生命周期管理

---

**下一步**：学习 **03_核心概念_05_异步任务管理.md**，了解如何管理 BackgroundTasks 和 asyncio.Task。
