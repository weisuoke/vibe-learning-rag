# 核心概念1：SQLAlchemy连接池机制

## 什么是 SQLAlchemy 连接池？

**SQLAlchemy 连接池是 SQLAlchemy 内置的数据库连接管理系统，负责创建、维护、分配和回收数据库连接。**

### 连接池的核心组件

```python
from sqlalchemy import create_engine

# 创建引擎时，SQLAlchemy 自动创建连接池
engine = create_engine(
    "postgresql://user:password@localhost:5432/dbname",
    pool_size=10,              # 连接池大小
    max_overflow=20,           # 溢出连接数
    pool_timeout=30,           # 获取连接超时
    pool_recycle=3600,         # 连接回收时间
    pool_pre_ping=True,        # 连接前检测
)

# 连接池的内部结构
# ┌─────────────────────────────────────┐
# │         SQLAlchemy Engine           │
# ├─────────────────────────────────────┤
# │           Connection Pool           │
# │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    │
# │  │ C │ │ C │ │ C │ │ C │ │ C │    │ ← pool_size=5 (常驻连接)
# │  └───┘ └───┘ └───┘ └───┘ └───┘    │
# │  ┌───┐ ┌───┐ ┌───┐                │
# │  │ O │ │ O │ │ O │                │ ← max_overflow=3 (临时连接)
# │  └───┘ └───┘ └───┘                │
# └─────────────────────────────────────┘
```

---

## SQLAlchemy 的连接池类型

### 1. QueuePool（默认，推荐）

**特点：** 使用队列管理连接，支持连接复用和溢出

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# 显式指定 QueuePool（默认就是这个）
engine = create_engine(
    "postgresql://user:password@localhost:5432/dbname",
    poolclass=QueuePool,       # 连接池类型
    pool_size=10,              # 常驻连接数
    max_overflow=20,           # 临时连接数
    pool_timeout=30,           # 获取连接超时
)
```

**工作原理：**

```python
# QueuePool 的内部实现（简化版）
class QueuePool:
    def __init__(self, pool_size, max_overflow):
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool = Queue(maxsize=pool_size)
        self.overflow = 0

        # 预创建连接
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)

    def get_connection(self, timeout=30):
        try:
            # 1. 尝试从队列获取连接
            return self.pool.get(timeout=timeout)
        except Empty:
            # 2. 队列为空，检查是否可以创建溢出连接
            if self.overflow < self.max_overflow:
                self.overflow += 1
                return self._create_connection()
            else:
                # 3. 达到最大连接数，抛出超时异常
                raise TimeoutError("无法获取连接")

    def return_connection(self, conn):
        if self.overflow > 0:
            # 溢出连接：直接关闭
            conn.close()
            self.overflow -= 1
        else:
            # 常驻连接：放回队列
            self.pool.put(conn)
```

**适用场景：**
- ✅ Web 应用（FastAPI、Flask、Django）
- ✅ 高并发场景
- ✅ 需要连接复用的场景

**前端类比：**
```javascript
// 类似 HTTP 连接池
const agent = new http.Agent({
  keepAlive: true,
  maxSockets: 10,        // 类似 pool_size
  maxFreeSockets: 20,    // 类似 max_overflow
});
```

---

### 2. NullPool（无连接池）

**特点：** 每次都创建新连接，用完立即关闭

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# 使用 NullPool（无连接池）
engine = create_engine(
    "postgresql://user:password@localhost:5432/dbname",
    poolclass=NullPool,  # 禁用连接池
)
```

**工作原理：**

```python
# NullPool 的内部实现（简化版）
class NullPool:
    def get_connection(self):
        # 每次都创建新连接
        return self._create_connection()

    def return_connection(self, conn):
        # 立即关闭连接
        conn.close()
```

**适用场景：**
- ✅ 单次脚本（数据迁移、批处理）
- ✅ 低频访问（每小时几次请求）
- ✅ 调试和测试

**不适用场景：**
- ❌ Web 应用（性能差）
- ❌ 高并发场景（频繁建立连接）

**前端类比：**
```javascript
// 类似每次都创建新的 HTTP 连接
fetch('https://api.example.com/users', {
  keepalive: false,  // 不复用连接
});
```

---

### 3. StaticPool（单连接池）

**特点：** 只有一个连接，所有请求共享

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# 使用 StaticPool（单连接）
engine = create_engine(
    "sqlite:///:memory:",  # 通常用于 SQLite 内存数据库
    poolclass=StaticPool,
)
```

**工作原理：**

```python
# StaticPool 的内部实现（简化版）
class StaticPool:
    def __init__(self):
        self.connection = self._create_connection()

    def get_connection(self):
        # 总是返回同一个连接
        return self.connection

    def return_connection(self, conn):
        # 不关闭连接，继续复用
        pass
```

**适用场景：**
- ✅ SQLite 内存数据库
- ✅ 单线程应用
- ✅ 测试环境

**不适用场景：**
- ❌ 多线程应用（不是线程安全的）
- ❌ PostgreSQL/MySQL（需要多连接）

---

### 4. SingletonThreadPool（线程单例池）

**特点：** 每个线程一个连接

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import SingletonThreadPool

# 使用 SingletonThreadPool
engine = create_engine(
    "sqlite:///database.db",
    poolclass=SingletonThreadPool,
)
```

**工作原理：**

```python
# SingletonThreadPool 的内部实现（简化版）
class SingletonThreadPool:
    def __init__(self):
        self.connections = {}  # {thread_id: connection}

    def get_connection(self):
        thread_id = threading.current_thread().ident
        if thread_id not in self.connections:
            self.connections[thread_id] = self._create_connection()
        return self.connections[thread_id]
```

**适用场景：**
- ✅ SQLite 文件数据库（多线程）
- ✅ 每个线程需要独立连接的场景

---

## 连接池参数详解

### 1. pool_size（常驻连接数）

**定义：** 连接池中始终保持的连接数

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # 始终保持 10 个连接
)
```

**特点：**
- 应用启动时预创建
- 始终保持在连接池中
- 不会被关闭（除非应用关闭）

**配置建议：**
```python
# 低并发（个人项目）
pool_size=5

# 中等并发（小型 API）
pool_size=10

# 高并发（生产环境）
pool_size=20-50
```

**前端类比：**
```javascript
// 类似浏览器的持久连接数
const agent = new http.Agent({
  maxSockets: 10,  // 类似 pool_size
});
```

---

### 2. max_overflow（溢出连接数）

**定义：** 在 pool_size 基础上，最多可以创建的临时连接数

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,       # 常驻连接：10 个
    max_overflow=20,    # 临时连接：最多 20 个
    # 总连接数：10 + 20 = 30 个
)
```

**特点：**
- 只在连接池满时创建
- 使用完毕后立即关闭
- 不会保留在连接池中

**工作流程：**
```python
# 1. 前 10 个请求：使用常驻连接
请求 1-10: 从连接池获取连接 ✅

# 2. 第 11-30 个请求：创建临时连接
请求 11-30: 创建临时连接 ✅

# 3. 第 31 个请求：等待或超时
请求 31: 等待连接释放 ⏳
```

**配置建议：**
```python
# 公式：max_overflow = pool_size × 1.5 - 2
pool_size=10, max_overflow=15-20
pool_size=20, max_overflow=30-40
```

---

### 3. pool_timeout（获取连接超时）

**定义：** 等待获取连接的最长时间（秒）

```python
engine = create_engine(
    DATABASE_URL,
    pool_timeout=30,  # 30 秒内无法获取连接，抛出异常
)
```

**工作流程：**
```python
# 连接池已满，新请求到达
try:
    conn = pool.get_connection(timeout=30)  # 等待 30 秒
except TimeoutError:
    # 30 秒后仍无可用连接
    raise HTTPException(503, "数据库连接池已满")
```

**配置建议：**
```python
# Web API：30 秒（避免请求长时间挂起）
pool_timeout=30

# 批处理：300 秒（可以等待更久）
pool_timeout=300

# 实时系统：5 秒（快速失败）
pool_timeout=5
```

**前端类比：**
```javascript
// 类似 fetch 请求超时
const controller = new AbortController();
setTimeout(() => controller.abort(), 30000);

fetch(url, { signal: controller.signal });
```

---

### 4. pool_recycle（连接回收时间）

**定义：** 连接使用超过指定时间后，自动关闭并重新创建（秒）

```python
engine = create_engine(
    DATABASE_URL,
    pool_recycle=3600,  # 1 小时后回收连接
)
```

**为什么需要回收？**

1. **数据库主动断开连接**
   ```python
   # MySQL 默认 8 小时后断开空闲连接
   # PostgreSQL 默认无限制，但网络设备可能断开

   # 如果不回收，使用失效连接会报错
   # OperationalError: (2006, 'MySQL server has gone away')
   ```

2. **防止连接状态累积**
   ```python
   # 长时间使用的连接可能累积：
   # - 临时表
   # - 会话变量
   # - 锁
   # 定期回收可以清理这些状态
   ```

**配置建议：**
```python
# MySQL：小于 8 小时（28800 秒）
pool_recycle=3600  # 1 小时

# PostgreSQL：根据网络设备超时时间
pool_recycle=3600  # 1 小时

# 如果数据库配置了 wait_timeout
pool_recycle = wait_timeout - 300  # 提前 5 分钟回收
```

**前端类比：**
```javascript
// 类似 JWT Token 过期
const token = localStorage.getItem('token');
if (Date.now() > decoded.exp * 1000) {
  refreshToken();  // 类似 pool_recycle
}
```

---

### 5. pool_pre_ping（连接前检测）

**定义：** 使用连接前先发送一个简单查询，检测连接是否有效

```python
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 使用前先 ping
)
```

**工作原理：**
```python
# pool_pre_ping 的内部实现（简化版）
def get_connection_with_pre_ping(pool):
    conn = pool.get_connection()

    # 发送 SELECT 1 检测连接
    try:
        conn.execute("SELECT 1")
        return conn  # 连接有效
    except Exception:
        # 连接失效，重新创建
        conn.close()
        return pool.create_new_connection()
```

**性能开销：**
```python
# SELECT 1 的开销：<1ms
# 如果连接刚被使用过（<1秒），跳过 ping
# 所以实际开销很小
```

**配置建议：**
```python
# 生产环境：必须开启
pool_pre_ping=True

# 开发环境：可选
pool_pre_ping=False  # 减少日志输出
```

---

## 连接池的生命周期

### 1. 创建阶段

```python
# 应用启动时
engine = create_engine(DATABASE_URL, pool_size=10)

# 连接池初始化
# 1. 预创建 pool_size 个连接
# 2. 放入连接队列
# 3. 等待请求获取
```

### 2. 使用阶段

```python
# 请求到达
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # 1. 从连接池获取连接（1-5ms）
    # 2. 执行查询（5-20ms）
    user = db.query(User).filter(User.id == user_id).first()
    # 3. 归还连接到池中
    return user
```

### 3. 回收阶段

```python
# 连接使用超过 pool_recycle 时间
# 1. 标记连接为"待回收"
# 2. 下次使用时关闭旧连接
# 3. 创建新连接
```

### 4. 关闭阶段

```python
# 应用关闭时
engine.dispose()

# 连接池清理
# 1. 关闭所有常驻连接
# 2. 关闭所有溢出连接
# 3. 释放资源
```

---

## 连接池监控

### 查看连接池状态

```python
# 获取连接池状态
pool_status = engine.pool.status()
print(pool_status)

# 输出示例：
# Pool size: 10  Connections in pool: 5  Current Overflow: 3  Current Checked out connections: 8
```

**状态解读：**
- **Pool size**: 配置的 pool_size
- **Connections in pool**: 当前空闲连接数
- **Current Overflow**: 当前溢出连接数
- **Current Checked out**: 当前正在使用的连接数

### 监控连接池事件

```python
from sqlalchemy import event

# 监听连接创建
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    print(f"新连接创建: {id(dbapi_conn)}")

# 监听连接获取
@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    print(f"连接被获取: {id(dbapi_conn)}")

# 监听连接归还
@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    print(f"连接被归还: {id(dbapi_conn)}")
```

---

## 总结

### SQLAlchemy 连接池的核心特性

| 特性 | 说明 | 配置参数 |
|------|------|---------|
| **连接复用** | 避免频繁建立连接 | `pool_size` |
| **弹性扩容** | 应对突发流量 | `max_overflow` |
| **超时保护** | 防止无限等待 | `pool_timeout` |
| **自动回收** | 防止连接失效 | `pool_recycle` |
| **健康检查** | 使用前检测连接 | `pool_pre_ping` |

### 推荐配置

```python
# 生产环境推荐配置
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,       # 使用 QueuePool（默认）
    pool_size=20,              # 根据并发量调整
    max_overflow=30,           # pool_size × 1.5
    pool_timeout=30,           # 30 秒超时
    pool_recycle=3600,         # 1 小时回收
    pool_pre_ping=True,        # 必须开启
)
```

**记住：** SQLAlchemy 的连接池是自动管理的，你只需要配置参数，连接池会自动处理连接的创建、复用、回收和关闭。
