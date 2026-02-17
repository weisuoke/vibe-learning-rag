# 核心概念2：FastAPI中的连接池集成

## FastAPI + SQLAlchemy 的连接池架构

**FastAPI 通过依赖注入（Dependency Injection）管理数据库 Session 的生命周期，底层自动使用 SQLAlchemy 的连接池。**

### 完整架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Request 1 (协程)                         │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  get_db() → Session → Connection from Pool     │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Request 2 (协程)                         │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  get_db() → Session → Connection from Pool     │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    SQLAlchemy Engine                         │
├─────────────────────────────────────────────────────────────┤
│                    Connection Pool                           │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                            │
│  │ C │ │ C │ │ C │ │ C │ │ C │  ← pool_size=5              │
│  └───┘ └───┘ └───┘ └───┘ └───┘                            │
├─────────────────────────────────────────────────────────────┤
│                    PostgreSQL Database                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 标准集成模式

### 1. 创建数据库引擎和 SessionLocal

```python
# app/core/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. 从环境变量读取数据库 URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/dbname"
)

# 2. 创建引擎（带连接池）
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # 常驻连接数
    max_overflow=20,           # 临时连接数
    pool_timeout=30,           # 获取连接超时
    pool_recycle=3600,         # 连接回收时间（1小时）
    pool_pre_ping=True,        # 连接前检测
    echo=False,                # 不打印 SQL（生产环境）
)

# 3. 创建 SessionLocal 类
SessionLocal = sessionmaker(
    autocommit=False,          # 不自动提交
    autoflush=False,           # 不自动刷新
    bind=engine,               # 绑定到引擎（连接池）
)

# 4. 创建 Base 类（用于定义模型）
Base = declarative_base()
```

**关键点：**
- `engine` 包含连接池配置
- `SessionLocal` 绑定到 `engine`
- 每次调用 `SessionLocal()` 会从连接池获取连接

---

### 2. 依赖注入函数

```python
# app/core/database.py (续)
from sqlalchemy.orm import Session

def get_db():
    """
    依赖注入函数：为每个请求提供独立的数据库 Session
    """
    db = SessionLocal()  # 从连接池获取连接
    try:
        yield db  # 返回 Session 给路由函数
    finally:
        db.close()  # 归还连接到池中
```

**工作流程：**

```python
# 1. 请求到达
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # ↓
    # 2. FastAPI 调用 get_db()
    db = SessionLocal()  # 从连接池获取连接
    # ↓
    # 3. 执行路由函数
    user = db.query(User).filter(User.id == user_id).first()
    # ↓
    # 4. 请求结束，执行 finally
    db.close()  # 归还连接到池中
    # ↓
    # 5. 返回响应
    return user
```

**前端类比：**
```javascript
// 类似 Express 中间件
app.use((req, res, next) => {
  req.db = pool.getConnection();  // 获取连接
  res.on('finish', () => {
    req.db.release();  // 归还连接
  });
  next();
});
```

---

### 3. 在路由中使用

```python
# app/api/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db)  # 依赖注入
):
    """
    获取用户信息
    """
    # db 已经从连接池获取了连接
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return user
    # 请求结束后，连接自动归还到池中
```

**关键点：**
- `db: Session = Depends(get_db)` 自动注入 Session
- 无需手动管理连接的获取和释放
- 异常情况下也能正确归还连接

---

## 高级集成模式

### 1. 异步 FastAPI + 异步 SQLAlchemy

**适用场景：** 高并发、I/O 密集型应用

```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 1. 创建异步引擎
async_engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost:5432/dbname",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

# 2. 创建异步 SessionLocal
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 3. 异步依赖注入函数
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
```

**在路由中使用：**

```python
from sqlalchemy import select

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """
    异步获取用户信息
    """
    result = await db.execute(
        select(User).filter(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return user
```

**优点：**
- 更高的并发能力（协程级别）
- 更少的线程开销
- 适合 I/O 密集型应用

**缺点：**
- 需要使用异步驱动（asyncpg、aiomysql）
- 代码复杂度略高

---

### 2. 事务管理

**场景：** 多个数据库操作需要原子性

```python
# app/core/database.py
from contextlib import contextmanager

@contextmanager
def get_db_transaction():
    """
    事务管理：自动提交或回滚
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()  # 成功时提交
    except Exception:
        db.rollback()  # 失败时回滚
        raise
    finally:
        db.close()  # 归还连接
```

**在路由中使用：**

```python
@router.post("/transfer")
async def transfer_money(
    from_user_id: int,
    to_user_id: int,
    amount: float,
):
    """
    转账：需要事务保证原子性
    """
    with get_db_transaction() as db:
        # 1. 扣款
        from_user = db.query(User).filter(User.id == from_user_id).first()
        from_user.balance -= amount

        # 2. 加款
        to_user = db.query(User).filter(User.id == to_user_id).first()
        to_user.balance += amount

        # 3. 自动提交（如果没有异常）
        # 4. 如果有异常，自动回滚

    return {"message": "转账成功"}
```

**前端类比：**
```javascript
// 类似数据库事务
async function transferMoney(fromUserId, toUserId, amount) {
  const transaction = await db.transaction();
  try {
    await transaction.update('users', { balance: balance - amount }, { id: fromUserId });
    await transaction.update('users', { balance: balance + amount }, { id: toUserId });
    await transaction.commit();  // 提交
  } catch (error) {
    await transaction.rollback();  // 回滚
    throw error;
  }
}
```

---

### 3. 读写分离

**场景：** 主从复制，读操作使用从库，写操作使用主库

```python
# app/core/database.py

# 主库引擎（写操作）
master_engine = create_engine(
    "postgresql://user:password@master-db:5432/dbname",
    pool_size=10,
    max_overflow=20,
)

# 从库引擎（读操作）
slave_engine = create_engine(
    "postgresql://user:password@slave-db:5432/dbname",
    pool_size=20,  # 读操作更多，连接池更大
    max_overflow=30,
)

# 主库 Session
MasterSessionLocal = sessionmaker(bind=master_engine)

# 从库 Session
SlaveSessionLocal = sessionmaker(bind=slave_engine)

# 依赖注入函数
def get_master_db():
    """主库：用于写操作"""
    db = MasterSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_slave_db():
    """从库：用于读操作"""
    db = SlaveSessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**在路由中使用：**

```python
# 读操作：使用从库
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_slave_db)  # 从库
):
    user = db.query(User).filter(User.id == user_id).first()
    return user

# 写操作：使用主库
@router.post("/users")
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_master_db)  # 主库
):
    user = User(**user_data.dict())
    db.add(user)
    db.commit()
    return user
```

---

## 连接池与请求生命周期

### 请求生命周期详解

```python
# 1. 请求到达
# ↓
# 2. FastAPI 调用依赖注入函数
db = SessionLocal()  # 从连接池获取连接（1-5ms）
# ↓
# 3. 执行路由函数
user = db.query(User).filter(User.id == user_id).first()  # 执行查询（5-20ms）
# ↓
# 4. 路由函数返回
return user
# ↓
# 5. FastAPI 执行 finally 块
db.close()  # 归还连接到池中（<1ms）
# ↓
# 6. 返回响应给客户端
```

**时间分析：**
- 获取连接：1-5ms（从连接池）
- 执行查询：5-20ms（取决于查询复杂度）
- 归还连接：<1ms
- **总耗时：6-26ms**

**如果没有连接池：**
- 建立连接：65-160ms
- 执行查询：5-20ms
- 关闭连接：5-10ms
- **总耗时：75-190ms**

**性能提升：10-30倍**

---

## 常见问题与解决方案

### 问题1：连接池耗尽

**现象：**
```python
# 错误信息
TimeoutError: QueuePool limit of size 10 overflow 20 reached,
connection timed out, timeout 30
```

**原因：**
- 并发请求数超过连接池容量
- 某些请求长时间占用连接

**解决方案：**

```python
# 方案1：增大连接池
engine = create_engine(
    DATABASE_URL,
    pool_size=20,       # 增大常驻连接
    max_overflow=40,    # 增大临时连接
)

# 方案2：减少单个请求的连接占用时间
@router.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # 只查询需要的字段，减少查询时间
    user = db.query(User.id, User.name).filter(User.id == user_id).first()
    return user

# 方案3：使用缓存减少数据库查询
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_cached(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    return user
```

---

### 问题2：连接泄漏

**现象：**
```python
# 连接池中的连接越来越少
# 最终导致连接池耗尽
```

**原因：**
- Session 没有正确关闭
- 异常情况下没有执行 `db.close()`

**解决方案：**

```python
# ❌ 错误：没有使用 try-finally
def get_db_wrong():
    db = SessionLocal()
    yield db
    db.close()  # 如果发生异常，这行不会执行

# ✅ 正确：使用 try-finally
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # 无论是否异常，都会执行

# ✅ 更好：使用上下文管理器
from contextlib import contextmanager

@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 问题3：跨请求共享 Session

**现象：**
```python
# 数据不一致、查询结果错误
```

**原因：**
- 多个请求共享同一个 Session
- Session 缓存了旧数据

**解决方案：**

```python
# ❌ 错误：全局共享 Session
global_session = SessionLocal()

@router.get("/users/{user_id}")
async def get_user(user_id: int):
    user = global_session.query(User).filter(User.id == user_id).first()
    return user

# ✅ 正确：每个请求独立 Session
@router.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    return user
```

---

## 性能优化技巧

### 1. 连接池预热

**目的：** 应用启动时预创建连接，避免首次请求慢

```python
# app/main.py
from fastapi import FastAPI
from app.core.database import engine

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    应用启动时预热连接池
    """
    # 预创建连接
    with engine.connect() as conn:
        conn.execute("SELECT 1")

    print("连接池预热完成")
```

---

### 2. 连接池监控

**目的：** 实时监控连接池状态，及时发现问题

```python
# app/api/health.py
from fastapi import APIRouter
from app.core.database import engine

router = APIRouter()

@router.get("/health/db")
async def check_db_health():
    """
    检查数据库连接池状态
    """
    pool = engine.pool

    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "status": pool.status(),
    }
```

**输出示例：**
```json
{
  "pool_size": 10,
  "checked_in": 5,
  "checked_out": 3,
  "overflow": 2,
  "status": "Pool size: 10  Connections in pool: 5  Current Overflow: 2  Current Checked out connections: 3"
}
```

---

### 3. 慢查询日志

**目的：** 记录慢查询，优化数据库性能

```python
# app/core/database.py
from sqlalchemy import event
import time
import logging

logger = logging.getLogger(__name__)

@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time'].pop()

    # 记录慢查询（超过 100ms）
    if total_time > 0.1:
        logger.warning(f"慢查询 ({total_time:.2f}s): {statement}")
```

---

## 总结

### FastAPI + SQLAlchemy 连接池集成的核心要点

| 要点 | 说明 | 代码示例 |
|------|------|---------|
| **依赖注入** | 使用 `Depends(get_db)` 自动管理 Session | `db: Session = Depends(get_db)` |
| **生命周期** | 每个请求独立 Session，请求结束后自动关闭 | `try: yield db finally: db.close()` |
| **事务管理** | 使用上下文管理器自动提交/回滚 | `with get_db_transaction() as db:` |
| **异常处理** | 使用 try-finally 确保连接归还 | `try: yield db finally: db.close()` |
| **性能优化** | 连接池预热、监控、慢查询日志 | `@app.on_event("startup")` |

### 推荐配置

```python
# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

# 1. 创建引擎
engine = create_engine(
    DATABASE_URL,
    pool_size=20,              # 根据并发量调整
    max_overflow=30,           # pool_size × 1.5
    pool_timeout=30,           # 30 秒超时
    pool_recycle=3600,         # 1 小时回收
    pool_pre_ping=True,        # 必须开启
)

# 2. 创建 SessionLocal
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 依赖注入函数
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 4. 在路由中使用
# @router.get("/users/{user_id}")
# async def get_user(user_id: int, db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.id == user_id).first()
#     return user
```

**记住：** FastAPI 的依赖注入系统自动管理 Session 的生命周期，你只需要配置好连接池参数，剩下的交给 FastAPI 处理。
