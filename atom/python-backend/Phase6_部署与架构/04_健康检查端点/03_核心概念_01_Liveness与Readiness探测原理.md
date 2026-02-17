# 核心概念1：Liveness与Readiness探测原理

> Kubernetes 的三种健康探测机制：Liveness、Readiness、Startup

---

## 概述

Kubernetes 提供了三种健康探测机制来管理 Pod 的生命周期：

1. **Liveness Probe（存活探测）**：检测容器是否还在运行
2. **Readiness Probe（就绪探测）**：检测容器是否准备好接收流量
3. **Startup Probe（启动探测）**：检测容器应用是否启动完成

这三种探测机制的设计遵循**关注点分离**原则，各司其职，共同保证服务的可用性。

---

## 1. Liveness Probe（存活探测）

### 1.1 定义

**Liveness Probe 检测容器是否还在运行（进程是否存活）**

### 1.2 失败处理

**连续失败 → Kubernetes 重启 Pod**

这是一个**破坏性操作**，会导致：
- 容器被杀死
- 新容器启动
- 所有内存状态丢失
- 短暂的服务中断

### 1.3 使用场景

**检测以下问题：**

#### 场景1：死锁（Deadlock）

```python
# 示例：死锁导致进程无响应
import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    with lock1:
        time.sleep(0.1)
        with lock2:  # 等待 lock2
            pass

def thread2():
    with lock2:
        time.sleep(0.1)
        with lock1:  # 等待 lock1
            pass

# 两个线程互相等待，进程死锁
# Liveness Probe 超时 → 重启 Pod
```

#### 场景2：内存泄漏导致 OOM

```python
# 示例：内存泄漏
cache = []

@app.get("/data")
async def get_data():
    # 不断往缓存添加数据，从不清理
    cache.append(generate_large_data())
    return {"data": cache[-1]}

# 最终内存耗尽，进程无响应
# Liveness Probe 失败 → 重启 Pod
```

#### 场景3：无限循环

```python
# 示例：无限循环导致进程卡死
@app.get("/process")
async def process():
    while True:
        # 忘记添加退出条件
        do_something()

# 进程卡在无限循环中
# Liveness Probe 超时 → 重启 Pod
```

### 1.4 实现示例

**FastAPI 端点：**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    """
    Liveness Probe 端点
    只要进程能响应就返回 200
    """
    return {"status": "healthy"}
```

**Kubernetes 配置：**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30  # 启动后 30 秒开始检查
  periodSeconds: 10         # 每 10 秒检查一次
  timeoutSeconds: 5         # 超时时间 5 秒
  failureThreshold: 3       # 连续失败 3 次才重启
```

### 1.5 配置参数详解

| 参数 | 说明 | 推荐值 | 原因 |
|------|------|--------|------|
| initialDelaySeconds | 启动后多久开始检查 | 30-60秒 | 给应用足够的启动时间 |
| periodSeconds | 检查间隔 | 10秒 | 不要太频繁，避免影响性能 |
| timeoutSeconds | 单次检查超时 | 5秒 | 足够长，避免误判 |
| failureThreshold | 连续失败多少次才重启 | 3次 | 避免短暂抖动导致重启 |

### 1.6 关键原则

**Liveness Probe 应该：**
- ✅ 简单快速（< 10ms）
- ✅ 不检查依赖服务
- ✅ 只检查进程是否响应
- ❌ 不要检查数据库、Redis 等依赖

**为什么不检查依赖？**

```python
# ❌ 错误：Liveness 检查依赖服务
@app.get("/health")
async def health():
    # 如果数据库断了，Liveness 失败
    await db.execute("SELECT 1")
    return {"status": "healthy"}

# 问题：数据库断了 → Liveness 失败 → 重启 Pod
# 但重启 Pod 解决不了数据库的问题！
# 结果：Pod 不断重启，服务完全不可用

# ✅ 正确：Liveness 只检查进程
@app.get("/health")
async def health():
    # 只要进程能响应就返回 200
    return {"status": "healthy"}

# 数据库断了 → Liveness 通过 → Pod 不重启
# Readiness 失败 → 停止流量 → 等待数据库恢复
```

---

## 2. Readiness Probe（就绪探测）

### 2.1 定义

**Readiness Probe 检测容器是否准备好接收流量**

### 2.2 失败处理

**连续失败 → 从 Service 的 Endpoints 中移除，停止发流量**

这是一个**非破坏性操作**：
- Pod 继续运行
- 不接收新流量
- 等待依赖服务恢复
- 恢复后自动接收流量

### 2.3 使用场景

**检测以下问题：**

#### 场景1：依赖服务不可用

```python
@app.get("/ready")
async def ready():
    """
    Readiness Probe 端点
    检查依赖服务是否可用
    """
    # 检查数据库
    if not await check_database():
        raise HTTPException(503, "Database unavailable")

    # 检查 Redis
    if not await check_redis():
        raise HTTPException(503, "Redis unavailable")

    return {"status": "ready"}
```

#### 场景2：应用启动中

```python
# 应用启动流程
@app.on_event("startup")
async def startup():
    # 1. 连接数据库（需要 5 秒）
    await connect_database()

    # 2. 加载 ML 模型（需要 30 秒）
    await load_ml_model()

    # 3. 预热缓存（需要 10 秒）
    await warmup_cache()

    # 总共需要 45 秒
    # 在此期间，Readiness Probe 应该返回 503
    # 避免负载均衡器把流量发给未准备好的 Pod
```

#### 场景3：服务过载

```python
@app.get("/ready")
async def ready():
    """检查服务是否过载"""
    # 检查当前请求队列长度
    queue_length = await get_queue_length()

    if queue_length > 1000:
        # 队列积压太多，暂时不接收新流量
        raise HTTPException(503, "Service overloaded")

    return {"status": "ready"}
```

### 2.4 实现示例

**FastAPI 端点：**

```python
from fastapi import FastAPI, HTTPException
import time

app = FastAPI()

# 健康检查缓存
health_cache = {
    "last_check": 0,
    "checks": {}
}

CACHE_TTL = 30  # 缓存 30 秒

async def check_database() -> bool:
    """检查数据库连接"""
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False

async def check_redis() -> bool:
    """检查 Redis 连接"""
    try:
        await redis.ping()
        return True
    except Exception:
        return False

async def get_health_status() -> dict:
    """获取健康状态（带缓存）"""
    now = time.time()

    # 如果缓存未过期，直接返回
    if now - health_cache["last_check"] < CACHE_TTL:
        return health_cache

    # 执行实际检查
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
    }

    # 更新缓存
    health_cache.update({
        "last_check": now,
        "checks": checks
    })

    return health_cache

@app.get("/ready")
async def ready():
    """
    Readiness Probe 端点
    检查依赖服务是否可用（带缓存）
    """
    status = await get_health_status()
    checks = status["checks"]

    # 如果任何依赖失败，返回 503
    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks
            }
        )

    return {
        "status": "ready",
        "checks": checks
    }
```

**Kubernetes 配置：**

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5   # 启动后 5 秒开始检查
  periodSeconds: 5         # 每 5 秒检查一次
  timeoutSeconds: 3        # 超时时间 3 秒
  failureThreshold: 2      # 连续失败 2 次停止流量
  successThreshold: 1      # 连续成功 1 次恢复流量
```

### 2.5 配置参数详解

| 参数 | 说明 | 推荐值 | 原因 |
|------|------|--------|------|
| initialDelaySeconds | 启动后多久开始检查 | 5-10秒 | 比 Liveness 更早开始 |
| periodSeconds | 检查间隔 | 5秒 | 比 Liveness 更频繁 |
| timeoutSeconds | 单次检查超时 | 3秒 | 比 Liveness 更短 |
| failureThreshold | 连续失败多少次停止流量 | 2次 | 比 Liveness 更敏感 |
| successThreshold | 连续成功多少次恢复流量 | 1次 | 快速恢复 |

### 2.6 关键原则

**Readiness Probe 应该：**
- ✅ 检查依赖服务（数据库、Redis、外部 API）
- ✅ 使用缓存（避免频繁检查）
- ✅ 快速返回（< 100ms）
- ✅ 区分核心依赖和可选依赖
- ❌ 不要检查太多依赖（影响性能）

---

## 3. Startup Probe（启动探测）

### 3.1 定义

**Startup Probe 检测容器应用是否启动完成**

### 3.2 失败处理

**连续失败 → Kubernetes 重启 Pod**

### 3.3 使用场景

**适用于慢启动的应用：**

#### 场景1：加载大型 ML 模型

```python
@app.on_event("startup")
async def startup():
    # 加载大型 ML 模型（需要 2 分钟）
    global embedding_model
    embedding_model = load_embedding_model()  # 2 分钟

# 如果没有 Startup Probe：
# - Liveness Probe 在 30 秒后开始检查
# - 但模型还在加载中，应用无法响应
# - Liveness Probe 失败 → 重启 Pod
# - 陷入无限重启循环

# 有了 Startup Probe：
# - Startup Probe 给 3 分钟的启动时间
# - 在 Startup Probe 成功前，Liveness 不会执行
# - 模型加载完成后，Startup Probe 成功
# - 然后 Liveness 和 Readiness 开始工作
```

#### 场景2：数据库迁移

```python
@app.on_event("startup")
async def startup():
    # 运行数据库迁移（可能需要几分钟）
    await run_database_migrations()  # 5 分钟

# Startup Probe 给足够的时间完成迁移
```

### 3.4 实现示例

**FastAPI 端点：**

```python
from fastapi import FastAPI

app = FastAPI()

# 启动状态标志
app.state.started = False

@app.on_event("startup")
async def startup():
    # 加载大型模型
    await load_ml_model()  # 2 分钟

    # 标记为已启动
    app.state.started = True

@app.get("/startup")
async def startup_check():
    """
    Startup Probe 端点
    检查应用是否启动完成
    """
    if not app.state.started:
        raise HTTPException(503, "Still starting up")

    return {"status": "started"}
```

**Kubernetes 配置：**

```yaml
startupProbe:
  httpGet:
    path: /startup
    port: 8000
  initialDelaySeconds: 0   # 立即开始检查
  periodSeconds: 10        # 每 10 秒检查一次
  timeoutSeconds: 5        # 超时时间 5 秒
  failureThreshold: 30     # 连续失败 30 次才重启（总共 5 分钟）

livenessProbe:
  httpGet:
    path: /health
    port: 8000
  # Startup Probe 成功后才开始执行
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  # Startup Probe 成功后才开始执行
  periodSeconds: 5
  failureThreshold: 2
```

### 3.5 关键原则

**Startup Probe 应该：**
- ✅ 给足够的启动时间（failureThreshold × periodSeconds）
- ✅ 检查应用是否完全启动
- ✅ 成功后，Liveness 和 Readiness 才开始工作
- ❌ 不要用于快速启动的应用（增加复杂度）

---

## 4. 三种探测的对比

### 4.1 对比表

| 维度 | Liveness | Readiness | Startup |
|------|----------|-----------|---------|
| **检查内容** | 进程是否存活 | 是否准备好接收流量 | 是否启动完成 |
| **失败处理** | 重启 Pod | 停止流量 | 重启 Pod |
| **检查速度** | 极快（< 10ms） | 快速（< 100ms） | 可以慢 |
| **检查依赖** | 不检查 | 检查 | 不检查 |
| **失败阈值** | 3次（30秒） | 2次（10秒） | 30次（5分钟） |
| **使用场景** | 检测死锁、内存泄漏 | 检测依赖服务故障 | 检测慢启动 |

### 4.2 执行顺序

```
Pod 启动
  ↓
Startup Probe 开始执行
  ↓
Startup Probe 成功
  ↓
Liveness Probe 和 Readiness Probe 开始执行
  ↓
Readiness Probe 成功 → 开始接收流量
  ↓
持续运行，定期检查 Liveness 和 Readiness
```

### 4.3 实际场景分析

#### 场景1：数据库连接断了

```
- Liveness Probe → 200 OK（进程还活着）
- Readiness Probe → 503（数据库不可用）
- 结果：Pod 不重启，但停止接收流量
- 等待：数据库恢复后，Readiness 自动通过，恢复流量
```

#### 场景2：进程死锁了

```
- Liveness Probe → 超时（进程无响应）
- Readiness Probe → 超时
- 结果：Liveness 连续失败 3 次 → Kubernetes 重启 Pod
```

#### 场景3：应用启动中

```
- Startup Probe → 503（还在加载模型）
- Liveness Probe → 不执行（等待 Startup 成功）
- Readiness Probe → 不执行（等待 Startup 成功）
- 结果：给足够时间启动，不会被误判为不健康
```

---

## 5. 在 AI Agent 后端中的应用

### 5.1 完整配置示例

**FastAPI 应用：**

```python
from fastapi import FastAPI, HTTPException
import time

app = FastAPI()

# 启动状态
app.state.started = False
app.state.start_time = 0

# 健康检查缓存
health_cache = {"last_check": 0, "checks": {}}

@app.on_event("startup")
async def startup():
    """应用启动"""
    # 1. 连接数据库
    await connect_database()

    # 2. 加载 Embedding 模型（慢）
    await load_embedding_model()  # 1 分钟

    # 3. 预热缓存
    await warmup_cache()

    # 标记为已启动
    app.state.started = True
    app.state.start_time = time.time()

# Startup Probe
@app.get("/startup")
async def startup_check():
    """检查应用是否启动完成"""
    if not app.state.started:
        raise HTTPException(503, "Still starting up")
    return {"status": "started"}

# Liveness Probe
@app.get("/health")
async def health():
    """检查进程是否存活"""
    return {"status": "healthy"}

# Readiness Probe
@app.get("/ready")
async def ready():
    """检查是否准备好接收流量"""
    now = time.time()

    # 使用缓存
    if now - health_cache["last_check"] < 30:
        checks = health_cache["checks"]
    else:
        checks = {
            "database": await check_database(),
            "redis": await check_redis(),
            "llm_api": await check_llm_api(),
        }
        health_cache.update({
            "last_check": now,
            "checks": checks
        })

    # 核心依赖失败 → 不健康
    if not checks["database"]:
        raise HTTPException(503, "Database unavailable")

    # LLM API 失败 → 降级但可用
    if not checks["llm_api"]:
        return {
            "status": "degraded",
            "message": "LLM API unavailable, using fallback",
            "checks": checks
        }

    return {"status": "ready", "checks": checks}
```

**Kubernetes 配置：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: ai-agent-api:latest
        ports:
        - containerPort: 8000

        # Startup Probe：给 2 分钟启动时间
        startupProbe:
          httpGet:
            path: /startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 12  # 12 × 10s = 2 分钟

        # Liveness Probe：检测死锁
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness Probe：检测依赖服务
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
          successThreshold: 1
```

---

## 6. 最佳实践

### 6.1 Liveness Probe 最佳实践

1. **保持简单**：不检查依赖服务
2. **设置合理的 failureThreshold**：避免误重启
3. **给足够的 initialDelaySeconds**：避免启动时被误判

### 6.2 Readiness Probe 最佳实践

1. **使用缓存**：避免频繁检查影响性能
2. **区分核心和可选依赖**：可选依赖失败时降级而非不可用
3. **快速返回**：< 100ms
4. **设置超时**：避免依赖服务慢导致健康检查慢

### 6.3 Startup Probe 最佳实践

1. **只用于慢启动应用**：快速启动的应用不需要
2. **给足够的时间**：failureThreshold × periodSeconds
3. **检查启动完成标志**：不要检查依赖服务

---

## 总结

Kubernetes 的三种健康探测机制各司其职：

1. **Liveness Probe**：检测进程是否存活，失败重启 Pod
2. **Readiness Probe**：检测是否准备好接收流量，失败停止流量
3. **Startup Probe**：检测是否启动完成，给慢启动应用足够时间

在 AI Agent 后端中，合理配置这三种探测可以实现：
- 自动检测和恢复故障（Liveness）
- 避免把流量发给不可用的服务（Readiness）
- 支持慢启动的 ML 模型加载（Startup）

关键是理解每种探测的目的和失败处理方式，避免误配置导致服务不稳定。
