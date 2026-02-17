# 实战代码3：Redis缓存检查

> Redis 连接健康检查的完整实现

---

## 概述

本文提供 Redis 缓存健康检查的完整实现，包括：
- PING 命令检查
- 读写测试
- 性能检查
- 连接信息获取
- 完整的 Redis 健康检查器

---

## 完整代码

```python
"""
Redis 缓存健康检查实现
演示：Redis 连接和性能检查
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import redis.asyncio as redis
import asyncio
import time

# ===== 1. Redis 配置 =====

# Redis 连接 URL
REDIS_URL = "redis://localhost:6379/0"

# 创建 Redis 连接池
redis_pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=10,
    decode_responses=True
)

# 创建 Redis 客户端
redis_client = redis.Redis(connection_pool=redis_pool)

# ===== 2. FastAPI 应用 =====

app = FastAPI(title="Redis Health Check")

# ===== 3. 响应模型 =====

class RedisHealthResponse(BaseModel):
    """Redis 健康检查响应"""
    healthy: bool
    duration_ms: int
    info: Optional[Dict] = None
    error: Optional[str] = None

# ===== 4. PING 命令检查 =====

async def check_redis_ping() -> bool:
    """
    PING 命令检查

    最简单的 Redis 健康检查
    """
    try:
        result = await asyncio.wait_for(
            redis_client.ping(),
            timeout=3.0
        )
        return result
    except Exception as e:
        print(f"❌ Redis PING failed: {e}")
        return False

@app.get("/health/redis/ping")
async def health_redis_ping():
    """Redis PING 检查"""
    start_time = time.time()

    healthy = await check_redis_ping()
    duration_ms = int((time.time() - start_time) * 1000)

    if not healthy:
        raise HTTPException(503, "Redis unavailable")

    return {
        "healthy": healthy,
        "duration_ms": duration_ms
    }

# ===== 5. 读写测试 =====

async def check_redis_readwrite() -> RedisHealthResponse:
    """
    Redis 读写测试

    测试 Redis 的读写功能
    """
    start_time = time.time()

    try:
        # 1. 写入测试数据
        test_key = "health_check:test"
        test_value = "ok"

        await asyncio.wait_for(
            redis_client.set(test_key, test_value, ex=10),  # 10 秒过期
            timeout=3.0
        )

        # 2. 读取测试数据
        value = await asyncio.wait_for(
            redis_client.get(test_key),
            timeout=3.0
        )

        # 3. 验证数据
        if value != test_value:
            return RedisHealthResponse(
                healthy=False,
                duration_ms=int((time.time() - start_time) * 1000),
                error="Redis read/write mismatch"
            )

        # 4. 删除测试数据
        await redis_client.delete(test_key)

        duration_ms = int((time.time() - start_time) * 1000)

        return RedisHealthResponse(
            healthy=True,
            duration_ms=duration_ms
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        return RedisHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            error="Redis operation timeout"
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return RedisHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            error=str(e)
        )

@app.get("/health/redis/readwrite", response_model=RedisHealthResponse)
async def health_redis_readwrite():
    """Redis 读写测试"""
    result = await check_redis_readwrite()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 6. 性能检查 =====

async def check_redis_performance() -> RedisHealthResponse:
    """
    Redis 性能检查

    检查 Redis 响应时间是否正常
    """
    start_time = time.time()

    try:
        # PING 命令
        await asyncio.wait_for(
            redis_client.ping(),
            timeout=3.0
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # 判断性能
        if duration_ms > 100:
            # PING 超过 100ms，性能下降
            print(f"⚠️  Warning: Redis slow ({duration_ms}ms)")
            return RedisHealthResponse(
                healthy=True,
                duration_ms=duration_ms,
                error=f"Slow response: {duration_ms}ms"
            )

        return RedisHealthResponse(
            healthy=True,
            duration_ms=duration_ms
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return RedisHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            error=str(e)
        )

@app.get("/health/redis/performance", response_model=RedisHealthResponse)
async def health_redis_performance():
    """Redis 性能检查"""
    result = await check_redis_performance()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 7. 获取 Redis 信息 =====

async def get_redis_info() -> Dict:
    """
    获取 Redis 信息

    返回 Redis 服务器的详细信息
    """
    try:
        # 获取 Redis INFO
        info = await redis_client.info()

        # 提取关键信息
        return {
            "version": info.get("redis_version", "unknown"),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
        }
    except Exception as e:
        print(f"❌ Failed to get Redis info: {e}")
        return {}

@app.get("/health/redis/info")
async def health_redis_info():
    """获取 Redis 信息"""
    info = await get_redis_info()

    if not info:
        raise HTTPException(503, "Failed to get Redis info")

    return info

# ===== 8. 完整的 Redis 健康检查器 =====

class RedisHealthChecker:
    """完整的 Redis 健康检查器"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache = {
            "last_check": 0,
            "result": None
        }
        self.cache_ttl = 60  # 缓存 60 秒

    async def check(self, use_cache: bool = True) -> RedisHealthResponse:
        """
        执行 Redis 健康检查

        Args:
            use_cache: 是否使用缓存

        Returns:
            RedisHealthResponse: 健康检查结果
        """
        # 检查缓存
        if use_cache and self.cache["result"]:
            now = time.time()
            if now - self.cache["last_check"] < self.cache_ttl:
                print("✅ Using cached Redis health status")
                return self.cache["result"]

        print("🔍 Performing Redis health check...")
        start_time = time.time()

        try:
            # 1. PING 命令
            await asyncio.wait_for(
                self.redis.ping(),
                timeout=3.0
            )

            # 2. 获取 Redis 信息
            info = await self.redis.info()

            # 3. 提取关键信息
            redis_info = {
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            }

            # 4. 判断健康状态
            duration_ms = int((time.time() - start_time) * 1000)
            warnings = []

            # 检查性能
            if duration_ms > 100:
                warnings.append(f"Slow response: {duration_ms}ms")

            # 检查连接数
            if redis_info["connected_clients"] > 100:
                warnings.append(f"High client count: {redis_info['connected_clients']}")

            result = RedisHealthResponse(
                healthy=True,
                duration_ms=duration_ms,
                info=redis_info,
                error="; ".join(warnings) if warnings else None
            )

            # 更新缓存
            self.cache = {
                "last_check": time.time(),
                "result": result
            }

            return result

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            result = RedisHealthResponse(
                healthy=False,
                duration_ms=duration_ms,
                error="Redis timeout"
            )

            # 不缓存失败结果
            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = RedisHealthResponse(
                healthy=False,
                duration_ms=duration_ms,
                error=str(e)
            )

            # 不缓存失败结果
            return result

# 创建全局健康检查器
redis_health_checker = RedisHealthChecker(redis_client)

@app.get("/health/redis/complete", response_model=RedisHealthResponse)
async def health_redis_complete():
    """完整的 Redis 健康检查（带缓存）"""
    result = await redis_health_checker.check(use_cache=True)

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 9. Redis 集群健康检查 =====

async def check_redis_cluster() -> Dict:
    """
    Redis 集群健康检查

    检查 Redis 集群的健康状态
    """
    try:
        # 获取集群信息
        cluster_info = await redis_client.cluster("info")

        # 解析集群信息
        info_dict = {}
        for line in cluster_info.split("\r\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                info_dict[key] = value

        return {
            "cluster_state": info_dict.get("cluster_state", "unknown"),
            "cluster_slots_assigned": info_dict.get("cluster_slots_assigned", "unknown"),
            "cluster_slots_ok": info_dict.get("cluster_slots_ok", "unknown"),
            "cluster_slots_fail": info_dict.get("cluster_slots_fail", "unknown"),
            "cluster_known_nodes": info_dict.get("cluster_known_nodes", "unknown"),
            "cluster_size": info_dict.get("cluster_size", "unknown"),
        }
    except Exception as e:
        print(f"❌ Not a Redis cluster or failed to get cluster info: {e}")
        return {"error": str(e)}

@app.get("/health/redis/cluster")
async def health_redis_cluster():
    """Redis 集群健康检查"""
    cluster_info = await check_redis_cluster()

    if "error" in cluster_info:
        raise HTTPException(503, detail=cluster_info)

    return cluster_info

# ===== 10. 启动和关闭事件 =====

@app.on_event("startup")
async def startup():
    """应用启动"""
    print("🚀 Starting application...")
    print(f"📊 Redis: {REDIS_URL}")

    # 测试 Redis 连接
    try:
        result = await redis_health_checker.check(use_cache=False)
        if result.healthy:
            print("✅ Redis connection successful")
        else:
            print(f"❌ Redis connection failed: {result.error}")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    """应用关闭"""
    print("👋 Shutting down application...")

    # 关闭 Redis 连接
    await redis_client.close()
    await redis_pool.disconnect()
    print("✅ Redis connections closed")

# ===== 11. 运行说明 =====

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("Redis 缓存健康检查实现")
    print("=" * 50)
    print()
    print("端点：")
    print("  /health/redis/ping        - PING 检查")
    print("  /health/redis/readwrite   - 读写测试")
    print("  /health/redis/performance - 性能检查")
    print("  /health/redis/info        - Redis 信息")
    print("  /health/redis/complete    - 完整检查（带缓存）")
    print("  /health/redis/cluster     - 集群检查")
    print()
    print("测试命令：")
    print("  curl http://localhost:8000/health/redis/ping")
    print("  curl http://localhost:8000/health/redis/readwrite")
    print("  curl http://localhost:8000/health/redis/complete")
    print()
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 环境配置

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv add fastapi uvicorn[standard] redis
```

### 2. 启动 Redis

**使用 Docker 启动 Redis：**

```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

**或者使用 docker-compose：**

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

```bash
docker-compose up -d
```

---

## 运行示例

### 1. 启动服务

```bash
python main.py
```

### 2. 测试端点

**PING 检查：**

```bash
curl http://localhost:8000/health/redis/ping
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 2
}
```

**读写测试：**

```bash
curl http://localhost:8000/health/redis/readwrite
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 5,
  "info": null,
  "error": null
}
```

**性能检查：**

```bash
curl http://localhost:8000/health/redis/performance
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 3,
  "info": null,
  "error": null
}
```

**Redis 信息：**

```bash
curl http://localhost:8000/health/redis/info
```

**输出：**

```json
{
  "version": "7.0.15",
  "uptime_seconds": 12345,
  "connected_clients": 2,
  "used_memory_human": "1.23M",
  "used_memory_peak_human": "1.45M",
  "total_commands_processed": 1000,
  "instantaneous_ops_per_sec": 10,
  "keyspace_hits": 500,
  "keyspace_misses": 50
}
```

**完整检查（带缓存）：**

```bash
# 第一次请求（执行实际检查）
curl http://localhost:8000/health/redis/complete

# 立即再次请求（使用缓存）
curl http://localhost:8000/health/redis/complete
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 4,
  "info": {
    "version": "7.0.15",
    "connected_clients": 2,
    "used_memory_human": "1.23M",
    "instantaneous_ops_per_sec": 10
  },
  "error": null
}
```

---

## 扩展示例：监控 Redis 性能

```python
"""
监控 Redis 性能指标
"""

from prometheus_client import Gauge

# 定义 Prometheus 指标
redis_connected_clients = Gauge('redis_connected_clients', 'Redis connected clients')
redis_used_memory_bytes = Gauge('redis_used_memory_bytes', 'Redis used memory in bytes')
redis_ops_per_sec = Gauge('redis_ops_per_sec', 'Redis operations per second')
redis_keyspace_hits = Gauge('redis_keyspace_hits', 'Redis keyspace hits')
redis_keyspace_misses = Gauge('redis_keyspace_misses', 'Redis keyspace misses')

async def monitor_redis_metrics():
    """监控 Redis 性能指标"""
    try:
        info = await redis_client.info()

        # 更新指标
        redis_connected_clients.set(info.get("connected_clients", 0))
        redis_used_memory_bytes.set(info.get("used_memory", 0))
        redis_ops_per_sec.set(info.get("instantaneous_ops_per_sec", 0))
        redis_keyspace_hits.set(info.get("keyspace_hits", 0))
        redis_keyspace_misses.set(info.get("keyspace_misses", 0))

        # 计算缓存命中率
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0

        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            "hit_rate_percent": round(hit_rate, 2)
        }
    except Exception as e:
        print(f"❌ Failed to monitor Redis metrics: {e}")
        return {}

@app.get("/metrics/redis")
async def metrics_redis():
    """Redis 性能指标"""
    return await monitor_redis_metrics()
```

---

## 扩展示例：Redis 连接池监控

```python
"""
监控 Redis 连接池状态
"""

async def monitor_redis_pool():
    """监控 Redis 连接池"""
    pool_stats = {
        "max_connections": redis_pool.max_connections,
        "connection_kwargs": {
            "host": redis_pool.connection_kwargs.get("host", "unknown"),
            "port": redis_pool.connection_kwargs.get("port", 0),
            "db": redis_pool.connection_kwargs.get("db", 0),
        }
    }

    return pool_stats

@app.get("/health/redis/pool")
async def health_redis_pool():
    """Redis 连接池状态"""
    return await monitor_redis_pool()
```

---

## 扩展示例：Redis 慢查询监控

```python
"""
监控 Redis 慢查询
"""

async def get_redis_slowlog(count: int = 10):
    """获取 Redis 慢查询日志"""
    try:
        # 获取慢查询日志
        slowlog = await redis_client.slowlog_get(count)

        # 格式化慢查询
        formatted_slowlog = []
        for entry in slowlog:
            formatted_slowlog.append({
                "id": entry["id"],
                "timestamp": entry["start_time"],
                "duration_us": entry["duration"],
                "command": " ".join(entry["command"]),
            })

        return formatted_slowlog
    except Exception as e:
        print(f"❌ Failed to get Redis slowlog: {e}")
        return []

@app.get("/health/redis/slowlog")
async def health_redis_slowlog():
    """Redis 慢查询日志"""
    slowlog = await get_redis_slowlog(count=10)

    return {
        "count": len(slowlog),
        "slowlog": slowlog
    }
```

---

## 关键要点

### 1. PING vs 读写测试

**PING 命令：**
- 最快（< 5ms）
- 只检查连接
- 不检查读写功能

**读写测试：**
- 较慢（10-20ms）
- 检查完整功能
- 更准确

### 2. 性能阈值

- **正常**：< 10ms
- **警告**：10-100ms
- **慢**：> 100ms

### 3. 缓存策略

```python
# Redis 健康检查缓存 60 秒
# 比数据库缓存时间更长（数据库 30 秒）
cache_ttl = 60
```

### 4. 关键指标

- **connected_clients**：连接数
- **used_memory**：内存使用
- **instantaneous_ops_per_sec**：每秒操作数
- **keyspace_hits/misses**：缓存命中率

### 5. 集群检查

```python
# 检查集群状态
cluster_info = await redis_client.cluster("info")

# 关键字段：
# - cluster_state: ok/fail
# - cluster_slots_ok: 正常的槽位数
# - cluster_slots_fail: 失败的槽位数
```

---

## 在 AI Agent 后端中的应用

### Redis 在 AI Agent 中的作用

**缓存用途：**
- Embedding 向量缓存
- LLM 响应缓存
- 会话状态缓存
- 用户数据缓存

**Redis 故障的影响：**
- 性能下降（无缓存）
- 但服务仍可用（降级运行）
- 不应导致服务完全不可用

### 推荐配置

```python
# Redis 作为可选依赖
@app.get("/ready")
async def ready():
    checks = {
        "database": await check_database(),  # 核心依赖
        "redis": await check_redis(),        # 可选依赖
    }

    # 数据库失败 → 不可用
    if not checks["database"]:
        raise HTTPException(503, "Database unavailable")

    # Redis 失败 → 降级但可用
    if not checks["redis"]:
        return {
            "status": "degraded",
            "message": "Redis unavailable, caching disabled",
            "checks": checks
        }

    return {"status": "healthy", "checks": checks}
```

---

## 总结

Redis 缓存健康检查的关键：

1. **PING 命令**：最简单快速的检查
2. **读写测试**：检查完整功能
3. **性能监控**：监控响应时间
4. **信息获取**：获取 Redis 服务器信息
5. **缓存策略**：缓存 60 秒，避免频繁检查
6. **可选依赖**：Redis 失败时降级而非不可用

在 AI Agent 后端中，Redis 通常是可选依赖，失败时应该降级运行而不是完全不可用。
