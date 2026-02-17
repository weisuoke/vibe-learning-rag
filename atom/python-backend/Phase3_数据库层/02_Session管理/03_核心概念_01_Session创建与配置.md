# 【核心概念1】Session创建与配置

> SessionMaker、Engine、连接参数的详细讲解

---

## 概述

**Session 的创建涉及三个核心组件：**
1. **Engine**：管理数据库连接池
2. **SessionMaker**：Session 工厂，定义配置
3. **Session**：具体的工作单元实例

```
┌─────────────────────────────────────┐
│  Engine（连接池管理器）              │
│  - 管理物理连接                      │
│  - 全局单例                          │
│  - 线程安全                          │
└─────────────────────────────────────┘
              ↓ bind
┌─────────────────────────────────────┐
│  SessionMaker（Session 工厂）        │
│  - 定义 Session 配置                 │
│  - 创建 Session 实例                 │
└─────────────────────────────────────┘
              ↓ create
┌─────────────────────────────────────┐
│  Session（工作单元）                 │
│  - 请求级别                          │
│  - 管理事务和对象                    │
└─────────────────────────────────────┘
```

---

## 1. Engine：连接池管理器

### 1.1 创建 Engine

**Engine 是 SQLAlchemy 的核心，管理数据库连接池。**

```python
from sqlalchemy import create_engine

# 基础创建
engine = create_engine("postgresql://user:password@localhost:5432/mydb")

# 完整配置
engine = create_engine(
    "postgresql://user:password@localhost:5432/mydb",
    # 连接池配置
    pool_size=10,              # 连接池大小
    max_overflow=20,           # 超出后的最大连接数
    pool_timeout=30,           # 获取连接的超时时间（秒）
    pool_recycle=3600,         # 连接回收时间（秒）
    pool_pre_ping=True,        # 连接前检查是否有效
    # 日志配置
    echo=False,                # 是否打印 SQL 语句
    echo_pool=False,           # 是否打印连接池日志
    # 其他配置
    connect_args={
        "connect_timeout": 10,  # 连接超时
        "options": "-c timezone=utc",  # PostgreSQL 特定选项
    }
)
```

### 1.2 连接字符串格式

**不同数据库的连接字符串格式：**

```python
# PostgreSQL
engine = create_engine("postgresql://user:password@localhost:5432/mydb")
engine = create_engine("postgresql+psycopg2://user:password@localhost:5432/mydb")

# MySQL
engine = create_engine("mysql://user:password@localhost:3306/mydb")
engine = create_engine("mysql+pymysql://user:password@localhost:3306/mydb")

# SQLite
engine = create_engine("sqlite:///path/to/database.db")
engine = create_engine("sqlite:///:memory:")  # 内存数据库

# SQL Server
engine = create_engine("mssql+pyodbc://user:password@localhost/mydb")
```

### 1.3 连接池配置详解

**pool_size：连接池大小**

```python
# 默认值：5
engine = create_engine(
    "postgresql://...",
    pool_size=10  # 连接池中保持的连接数
)
```

**作用：**
- 定义连接池中保持的连接数
- 连接可以被复用，避免频繁创建和销毁

**如何选择：**
- 小型应用：5-10
- 中型应用：10-20
- 大型应用：20-50
- 公式：`pool_size = (核心数 * 2) + 磁盘数`

---

**max_overflow：超出后的最大连接数**

```python
engine = create_engine(
    "postgresql://...",
    pool_size=10,
    max_overflow=20  # 超出 pool_size 后，最多再创建20个连接
)
```

**作用：**
- 当连接池满时，可以临时创建额外的连接
- 总连接数 = pool_size + max_overflow

**如何选择：**
- 设置为 pool_size 的 1-2 倍
- 避免设置过大，导致数据库连接数超限

---

**pool_timeout：获取连接的超时时间**

```python
engine = create_engine(
    "postgresql://...",
    pool_timeout=30  # 等待30秒后抛出异常
)
```

**作用：**
- 当连接池满时，等待多久后抛出异常
- 避免无限等待

**如何选择：**
- 快速失败：5-10秒
- 容忍等待：30-60秒

---

**pool_recycle：连接回收时间**

```python
engine = create_engine(
    "postgresql://...",
    pool_recycle=3600  # 1小时后回收连接
)
```

**作用：**
- 定期回收连接，避免连接过期
- 数据库通常有连接超时时间（如 MySQL 的 wait_timeout）

**如何选择：**
- 设置为数据库超时时间的 80%
- MySQL：28800秒（8小时）→ 设置为 7200秒（2小时）
- PostgreSQL：通常不需要设置

---

**pool_pre_ping：连接前检查**

```python
engine = create_engine(
    "postgresql://...",
    pool_pre_ping=True  # 使用连接前先检查是否有效
)
```

**作用：**
- 使用连接前先发送一个简单的查询（如 SELECT 1）
- 如果连接失效，自动重新创建

**如何选择：**
- 推荐开启，避免使用失效的连接
- 性能影响很小

---

### 1.4 Engine 的生命周期

**Engine 是全局单例，整个应用只创建一次。**

```python
# ✅ 正确：全局单例
# database.py
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL, pool_size=10)

# main.py
from database import engine

# 使用 engine
SessionLocal = sessionmaker(bind=engine)
```

```python
# ❌ 错误：每次请求创建 Engine
@app.get("/users")
def get_users():
    engine = create_engine("postgresql://...")  # ❌ 不要这样做
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    return session.query(User).all()
```

**为什么 Engine 是全局单例？**
- Engine 管理连接池，创建成本高
- 连接池需要在整个应用中共享
- Engine 是线程安全的

---

### 1.5 Engine 的监控

**监听 Engine 事件，监控连接池状态。**

```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    print(f"New connection: {dbapi_conn}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    print(f"Connection checked out from pool")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    print(f"Connection returned to pool")

# 查看连接池状态
print(f"Pool size: {engine.pool.size()}")
print(f"Checked out: {engine.pool.checkedout()}")
print(f"Overflow: {engine.pool.overflow()}")
```

---

## 2. SessionMaker：Session 工厂

### 2.1 创建 SessionMaker

**SessionMaker 是 Session 的工厂，定义 Session 的配置。**

```python
from sqlalchemy.orm import sessionmaker

# 基础创建
SessionLocal = sessionmaker(bind=engine)

# 完整配置
SessionLocal = sessionmaker(
    bind=engine,                # 绑定 Engine
    autocommit=False,           # 不自动提交（推荐）
    autoflush=True,             # 查询前自动 flush（推荐）
    expire_on_commit=True,      # commit 后对象过期（推荐）
    class_=Session,             # Session 类（默认）
)
```

### 2.2 SessionMaker 配置详解

**autocommit：是否自动提交**

```python
# autocommit=False（推荐）
SessionLocal = sessionmaker(bind=engine, autocommit=False)

session = SessionLocal()
session.add(User(name='Alice'))
session.commit()  # 显式提交

# autocommit=True（不推荐）
SessionLocal = sessionmaker(bind=engine, autocommit=True)

session = SessionLocal()
with session.begin():  # 需要显式开启事务
    session.add(User(name='Alice'))
    # 自动提交
```

**为什么不推荐 autocommit=True？**
- 失去事务控制
- 无法批量操作
- 性能下降

---

**autoflush：查询前是否自动 flush**

```python
# autoflush=True（推荐）
SessionLocal = sessionmaker(bind=engine, autoflush=True)

session = SessionLocal()
session.add(User(name='Alice'))

# 查询前自动 flush
user = session.query(User).filter_by(name='Alice').first()
# 自动执行 INSERT，然后执行 SELECT

# autoflush=False
SessionLocal = sessionmaker(bind=engine, autoflush=False)

session = SessionLocal()
session.add(User(name='Alice'))

# 查询前不会 flush
user = session.query(User).filter_by(name='Alice').first()
# 只执行 SELECT，查询不到 Alice
```

**为什么推荐 autoflush=True？**
- 确保查询到最新数据
- 避免数据不一致

---

**expire_on_commit：commit 后对象是否过期**

```python
# expire_on_commit=True（推荐）
SessionLocal = sessionmaker(bind=engine, expire_on_commit=True)

session = SessionLocal()
user = User(name='Alice')
session.add(user)
session.commit()

# commit 后对象过期
print(user.name)  # 重新查询数据库

# expire_on_commit=False
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

session = SessionLocal()
user = User(name='Alice')
session.add(user)
session.commit()

# commit 后对象不过期
print(user.name)  # 使用缓存数据
```

**为什么推荐 expire_on_commit=True？**
- 确保数据是最新的
- 避免使用过期数据

---

### 2.3 SessionMaker 的使用

**SessionMaker 是工厂，每次调用创建新的 Session。**

```python
# 创建 SessionMaker
SessionLocal = sessionmaker(bind=engine)

# 创建 Session 实例
session1 = SessionLocal()
session2 = SessionLocal()

print(session1 is session2)  # False（不同的 Session）
```

**SessionMaker 的生命周期：**
- SessionMaker 是全局单例
- 每次调用 SessionLocal() 创建新的 Session

---

## 3. Session：工作单元

### 3.1 创建 Session

**Session 是具体的工作单元实例。**

```python
# 方法1：直接创建
session = SessionLocal()

# 方法2：使用上下文管理器（推荐）
with SessionLocal() as session:
    # 使用 session
    pass
# 自动关闭

# 方法3：使用 begin 上下文管理器
with SessionLocal.begin() as session:
    # 使用 session
    pass
# 自动 commit 和关闭
```

### 3.2 Session 的配置

**Session 可以在创建时覆盖 SessionMaker 的配置。**

```python
# SessionMaker 的默认配置
SessionLocal = sessionmaker(bind=engine, autoflush=True)

# 创建 Session 时覆盖配置
session = SessionLocal(autoflush=False)  # 禁用 autoflush
```

### 3.3 Session 的生命周期

**Session 的生命周期：创建 → 使用 → 提交/回滚 → 关闭**

```python
# 完整的生命周期
session = SessionLocal()  # 1. 创建

try:
    # 2. 使用
    session.add(User(name='Alice'))
    session.query(User).all()

    # 3. 提交
    session.commit()
except Exception as e:
    # 3. 回滚
    session.rollback()
    raise
finally:
    # 4. 关闭
    session.close()
```

**使用上下文管理器简化：**

```python
# 自动管理生命周期
with SessionLocal() as session:
    session.add(User(name='Alice'))
    session.commit()
# 自动关闭
```

---

## 4. 完整示例

### 4.1 基础配置

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 数据库连接字符串
DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"

# 创建 Engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

# 创建 SessionMaker
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=True,
    expire_on_commit=True,
)

# 创建 Base 类
Base = declarative_base()
```

### 4.2 使用环境变量

```python
# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 从环境变量读取配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

# 创建 Engine
engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
)

# 创建 SessionMaker
SessionLocal = sessionmaker(bind=engine)
```

```bash
# .env 文件
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

### 4.3 多数据库配置

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 主数据库
engine_primary = create_engine("postgresql://localhost/primary")
SessionPrimary = sessionmaker(bind=engine_primary)

# 只读副本
engine_replica = create_engine("postgresql://localhost/replica")
SessionReplica = sessionmaker(bind=engine_replica)

# 使用
def get_users():
    # 读操作使用副本
    with SessionReplica() as session:
        return session.query(User).all()

def create_user(name):
    # 写操作使用主库
    with SessionPrimary() as session:
        user = User(name=name)
        session.add(user)
        session.commit()
        return user
```

### 4.4 异步配置

```python
# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 创建异步 Engine
engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/mydb",
    pool_size=10,
    max_overflow=20,
)

# 创建异步 SessionMaker
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 使用
async def get_users():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User))
        return result.scalars().all()
```

---

## 5. 最佳实践

### 5.1 配置推荐

```python
# 生产环境推荐配置
engine = create_engine(
    DATABASE_URL,
    pool_size=20,              # 根据应用规模调整
    max_overflow=40,           # pool_size 的 2 倍
    pool_timeout=30,           # 30秒超时
    pool_recycle=3600,         # 1小时回收
    pool_pre_ping=True,        # 开启连接检查
    echo=False,                # 生产环境关闭 SQL 日志
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,          # 不自动提交
    autoflush=True,            # 查询前自动 flush
    expire_on_commit=True,     # commit 后对象过期
)
```

### 5.2 连接池大小计算

```python
# 公式：pool_size = (核心数 * 2) + 磁盘数
import os

cpu_count = os.cpu_count()
disk_count = 1  # 假设1个磁盘

pool_size = (cpu_count * 2) + disk_count
max_overflow = pool_size * 2

engine = create_engine(
    DATABASE_URL,
    pool_size=pool_size,
    max_overflow=max_overflow,
)
```

### 5.3 健康检查

```python
# 检查数据库连接
def check_database_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# FastAPI 健康检查端点
@app.get("/health")
def health_check():
    return check_database_health()
```

### 5.4 连接池监控

```python
# 监控连接池状态
def get_pool_status():
    return {
        "pool_size": engine.pool.size(),
        "checked_out": engine.pool.checkedout(),
        "overflow": engine.pool.overflow(),
        "total": engine.pool.size() + engine.pool.overflow(),
    }

# FastAPI 监控端点
@app.get("/metrics/pool")
def pool_metrics():
    return get_pool_status()
```

---

## 6. 常见问题

### 问题1：连接池耗尽

**症状：**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 10 overflow 20 reached
```

**原因：**
- Session 没有关闭
- pool_size 设置过小

**解决方案：**
```python
# 1. 确保 Session 关闭
with SessionLocal() as session:
    # 使用 session
    pass
# 自动关闭

# 2. 增加 pool_size
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
)
```

### 问题2：连接过期

**症状：**
```
sqlalchemy.exc.OperationalError: server closed the connection unexpectedly
```

**原因：**
- 连接空闲时间过长，被数据库关闭

**解决方案：**
```python
# 开启 pool_pre_ping 和 pool_recycle
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

### 问题3：连接泄漏

**症状：**
- 连接数持续增长
- 数据库连接数达到上限

**原因：**
- Session 没有关闭

**解决方案：**
```python
# 使用上下文管理器
with SessionLocal() as session:
    # 使用 session
    pass
# 自动关闭

# 或使用 try-finally
session = SessionLocal()
try:
    # 使用 session
    pass
finally:
    session.close()
```

---

## 总结

**Session 创建与配置的核心要点：**

1. **Engine**：管理连接池，全局单例，线程安全
2. **SessionMaker**：Session 工厂，定义配置
3. **Session**：工作单元，请求级别，用完即关

**配置推荐：**
- **pool_size**：20（根据应用规模调整）
- **max_overflow**：pool_size 的 2 倍
- **pool_timeout**：30秒
- **pool_recycle**：3600秒（1小时）
- **pool_pre_ping**：True
- **autocommit**：False
- **autoflush**：True
- **expire_on_commit**：True

**最佳实践：**
- Engine 全局单例
- SessionMaker 全局单例
- Session 请求级别
- 使用上下文管理器
- 监控连接池状态

---

**记住：** Engine 管理连接池，SessionMaker 定义配置，Session 是工作单元。
