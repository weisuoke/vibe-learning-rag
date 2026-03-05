# 实战代码 - 场景2：RedisCache 生产部署

> **场景目标**：掌握生产环境 Redis 缓存配置和最佳实践

---

## 场景说明

### 适用场景
- 生产环境部署
- 分布式应用
- 高并发场景
- 需要持久化的缓存

### 核心特性
- **持久化**：支持 RDB 和 AOF 持久化
- **分布式**：多个服务实例共享缓存
- **高性能**：内存存储，毫秒级响应
- **可靠性**：成熟稳定的 Redis 服务

### 与 InMemoryCache 对比

| 特性 | InMemoryCache | RedisCache |
|------|---------------|------------|
| 持久化 | ❌ 不支持 | ✅ 支持 |
| 分布式 | ❌ 不支持 | ✅ 支持 |
| 性能 | 最快 | 很快 |
| 配置复杂度 | 零配置 | 需要 Redis 服务 |
| 适用环境 | 开发测试 | 生产环境 |

---

## 完整代码

```python
"""
LangChain RedisCache 生产部署实战
演示：生产环境 Redis 缓存配置、监控和优化

环境要求：
- Python 3.13+
- langchain-core
- langchain-openai
- redis
- python-dotenv

Redis 要求：
- Redis 6.0+
- 建议配置持久化（RDB + AOF）
"""

import os
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import redis

from langchain_core.globals import set_llm_cache, get_llm_cache
from langchain_openai import ChatOpenAI

# 注意：RedisCache 在 langchain-community 中
# 如果没有安装，需要：uv add langchain-community
try:
    from langchain_community.cache import RedisCache
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    REDIS_CACHE_AVAILABLE = False
    print("警告：langchain-community 未安装，无法使用 RedisCache")
    print("安装命令：uv add langchain-community")

# 加载环境变量
load_dotenv()

print("=" * 80)
print("场景2：RedisCache 生产部署")
print("=" * 80)

# ===== 1. Redis 连接配置 =====
print("\n【步骤1】配置 Redis 连接")
print("-" * 80)

# Redis 配置（生产环境建议使用环境变量）
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

print(f"Redis 配置:")
print(f"  Host: {REDIS_HOST}")
print(f"  Port: {REDIS_PORT}")
print(f"  DB: {REDIS_DB}")
print(f"  Password: {'***' if REDIS_PASSWORD else 'None'}")

# 创建 Redis 客户端
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,  # 自动解码为字符串
        socket_connect_timeout=5,  # 连接超时
        socket_timeout=5,  # 操作超时
    )

    # 测试连接
    redis_client.ping()
    print("✓ Redis 连接成功")

except redis.ConnectionError as e:
    print(f"✗ Redis 连接失败: {e}")
    print("\n提示：请确保 Redis 服务已启动")
    print("  - Docker: docker run -d -p 6379:6379 redis:latest")
    print("  - macOS: brew services start redis")
    print("  - Linux: sudo systemctl start redis")
    exit(1)

# ===== 2. 启用 RedisCache =====
print("\n【步骤2】启用 RedisCache")
print("-" * 80)

if not REDIS_CACHE_AVAILABLE:
    print("✗ RedisCache 不可用，请安装 langchain-community")
    exit(1)

# 创建并设置 Redis 缓存
cache = RedisCache(redis_=redis_client)
set_llm_cache(cache)

print("✓ RedisCache 已启用")
print(f"✓ 缓存类型: {type(get_llm_cache()).__name__}")

# ===== 3. 基础缓存测试 =====
print("\n【步骤3】基础缓存测试")
print("-" * 80)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100
)

prompt = "What is Redis in one sentence?"

# 首次调用
start_time = time.time()
response1 = llm.invoke(prompt)
first_call_time = time.time() - start_time

print(f"问题: {prompt}")
print(f"回答: {response1.content}")
print(f"⏱️  首次调用: {first_call_time:.3f}s")

# 第二次调用（缓存命中）
start_time = time.time()
response2 = llm.invoke(prompt)
second_call_time = time.time() - start_time

print(f"⏱️  第二次调用: {second_call_time:.3f}s")
print(f"📊 性能提升: {first_call_time / second_call_time:.1f}x")

# ===== 4. 查看 Redis 缓存数据 =====
print("\n【步骤4】查看 Redis 缓存数据")
print("-" * 80)

# 获取所有缓存键
cache_keys = redis_client.keys("*")
print(f"缓存键数量: {len(cache_keys)}")

if cache_keys:
    print(f"\n前3个缓存键:")
    for i, key in enumerate(cache_keys[:3], 1):
        print(f"  {i}. {key[:80]}...")

        # 获取键的类型和 TTL
        key_type = redis_client.type(key)
        ttl = redis_client.ttl(key)
        print(f"     类型: {key_type}, TTL: {ttl if ttl > 0 else '永久'}")

# ===== 5. 分布式缓存共享演示 =====
print("\n【步骤5】分布式缓存共享演示")
print("-" * 80)

print("模拟多个服务实例共享缓存...")

# 创建第二个 LLM 实例（模拟另一个服务）
llm2 = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100
)

# 使用相同的问题
start_time = time.time()
response3 = llm2.invoke(prompt)
third_call_time = time.time() - start_time

print(f"第二个服务实例调用: {third_call_time:.3f}s")
print(f"缓存命中: {'✓ 是' if third_call_time < 0.1 else '✗ 否'}")
print(f"说明: 多个服务实例共享 Redis 缓存")

# ===== 6. 缓存过期时间设置 =====
print("\n【步骤6】缓存过期时间设置")
print("-" * 80)

print("注意：LangChain 的 RedisCache 默认不设置过期时间")
print("生产环境建议手动设置 TTL 避免缓存无限增长")

# 示例：为所有缓存键设置 7 天过期时间
CACHE_TTL = 7 * 24 * 60 * 60  # 7 天（秒）

cache_keys = redis_client.keys("*")
if cache_keys:
    for key in cache_keys:
        redis_client.expire(key, CACHE_TTL)

    print(f"✓ 已为 {len(cache_keys)} 个缓存键设置 TTL: {CACHE_TTL}s (7天)")

# ===== 7. 缓存统计信息 =====
print("\n【步骤7】缓存统计信息")
print("-" * 80)

# Redis INFO 命令获取统计信息
info = redis_client.info()

print("Redis 服务器信息:")
print(f"  版本: {info.get('redis_version', 'N/A')}")
print(f"  运行模式: {info.get('redis_mode', 'N/A')}")
print(f"  已用内存: {info.get('used_memory_human', 'N/A')}")
print(f"  峰值内存: {info.get('used_memory_peak_human', 'N/A')}")
print(f"  连接数: {info.get('connected_clients', 'N/A')}")
print(f"  总键数: {info.get('db0', {}).get('keys', 0) if 'db0' in info else 0}")

# 持久化配置
print("\n持久化配置:")
print(f"  RDB: {'启用' if info.get('rdb_last_save_time', 0) > 0 else '未启用'}")
print(f"  AOF: {'启用' if info.get('aof_enabled', 0) == 1 else '未启用'}")

# ===== 8. 缓存性能测试 =====
print("\n【步骤8】缓存性能测试")
print("-" * 80)

test_questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is TypeScript?",
    "What is Go?",
    "What is Rust?",
]

print("批量测试（首次调用）...")
total_time = 0
for i, q in enumerate(test_questions, 1):
    start_time = time.time()
    llm.invoke(q)
    elapsed = time.time() - start_time
    total_time += elapsed
    print(f"  {i}. {q[:30]}... - {elapsed:.3f}s")

print(f"\n总耗时: {total_time:.3f}s")
print(f"平均耗时: {total_time / len(test_questions):.3f}s")

print("\n批量测试（缓存命中）...")
total_time_cached = 0
for i, q in enumerate(test_questions, 1):
    start_time = time.time()
    llm.invoke(q)
    elapsed = time.time() - start_time
    total_time_cached += elapsed
    print(f"  {i}. {q[:30]}... - {elapsed:.3f}s")

print(f"\n总耗时: {total_time_cached:.3f}s")
print(f"平均耗时: {total_time_cached / len(test_questions):.3f}s")
print(f"性能提升: {total_time / total_time_cached:.1f}x")

# ===== 9. 缓存清理 =====
print("\n【步骤9】缓存清理")
print("-" * 80)

# 获取当前缓存键数量
before_count = len(redis_client.keys("*"))
print(f"清理前缓存键数量: {before_count}")

# 清理缓存（生产环境谨慎使用）
# 方案1：清理所有键
# redis_client.flushdb()

# 方案2：清理特定模式的键
# pattern = "langchain:*"
# keys_to_delete = redis_client.keys(pattern)
# if keys_to_delete:
#     redis_client.delete(*keys_to_delete)

print("提示：生产环境建议使用 TTL 自动过期，而非手动清理")

# ===== 10. 生产环境最佳实践 =====
print("\n【步骤10】生产环境最佳实践")
print("-" * 80)

print("""
1. 持久化配置：
   - 启用 RDB：定期快照备份
   - 启用 AOF：实时持久化
   - 配置示例：
     save 900 1
     save 300 10
     save 60 10000
     appendonly yes
     appendfsync everysec

2. 内存管理：
   - 设置 maxmemory 限制
   - 配置淘汰策略：maxmemory-policy allkeys-lru
   - 监控内存使用

3. 连接池配置：
   - 使用连接池避免频繁创建连接
   - 设置合理的超时时间
   - 配置最大连接数

4. 缓存策略：
   - 设置合理的 TTL（如 7 天）
   - 监控缓存命中率
   - 定期清理过期数据

5. 监控告警：
   - 监控 Redis 内存使用
   - 监控缓存命中率
   - 监控连接数和延迟
   - 设置告警阈值

6. 高可用：
   - 使用 Redis Sentinel 或 Cluster
   - 配置主从复制
   - 定期备份数据
""")

# ===== 总结 =====
print("\n" + "=" * 80)
print("【总结】RedisCache 生产部署要点")
print("=" * 80)

print("""
1. 连接配置：
   - 使用环境变量管理配置
   - 设置连接超时和操作超时
   - 使用连接池提高性能

2. 持久化：
   - RDB + AOF 双重保障
   - 定期备份数据
   - 测试恢复流程

3. 性能优化：
   - 设置合理的 TTL
   - 监控缓存命中率
   - 使用 pipeline 批量操作

4. 分布式支持：
   - 多个服务实例共享缓存
   - 避免缓存雪崩
   - 考虑缓存预热

5. 监控运维：
   - 监控内存、连接、延迟
   - 设置告警阈值
   - 定期检查持久化状态
""")

print("\n✓ 场景2演示完成！")
```

---

## 运行输出示例

```
================================================================================
场景2：RedisCache 生产部署
================================================================================

【步骤1】配置 Redis 连接
--------------------------------------------------------------------------------
Redis 配置:
  Host: localhost
  Port: 6379
  DB: 0
  Password: None
✓ Redis 连接成功

【步骤2】启用 RedisCache
--------------------------------------------------------------------------------
✓ RedisCache 已启用
✓ 缓存类型: RedisCache

【步骤3】基础缓存测试
--------------------------------------------------------------------------------
问题: What is Redis in one sentence?
回答: Redis is an open-source, in-memory data structure store used as a database, cache, and message broker.
⏱️  首次调用: 1.234s
⏱️  第二次调用: 0.005s
📊 性能提升: 246.8x

【步骤4】查看 Redis 缓存数据
--------------------------------------------------------------------------------
缓存键数量: 1

前3个缓存键:
  1. langchain:cache:llm:gpt-4o-mini:0.7:100:What is Redis in one sentence?...
     类型: string, TTL: 永久

【步骤5】分布式缓存共享演示
--------------------------------------------------------------------------------
模拟多个服务实例共享缓存...
第二个服务实例调用: 0.004s
缓存命中: ✓ 是
说明: 多个服务实例共享 Redis 缓存

【步骤6】缓存过期时间设置
--------------------------------------------------------------------------------
注意：LangChain 的 RedisCache 默认不设置过期时间
生产环境建议手动设置 TTL 避免缓存无限增长
✓ 已为 1 个缓存键设置 TTL: 604800s (7天)

【步骤7】缓存统计信息
--------------------------------------------------------------------------------
Redis 服务器信息:
  版本: 7.2.4
  运行模式: standalone
  已用内存: 1.23M
  峰值内存: 1.45M
  连接数: 2
  总键数: 1

持久化配置:
  RDB: 启用
  AOF: 启用

【步骤8】缓存性能测试
--------------------------------------------------------------------------------
批量测试（首次调用）...
  1. What is Python?... - 1.123s
  2. What is JavaScript?... - 1.089s
  3. What is TypeScript?... - 1.156s
  4. What is Go?... - 1.078s
  5. What is Rust?... - 1.134s

总耗时: 5.580s
平均耗时: 1.116s

批量测试（缓存命中）...
  1. What is Python?... - 0.004s
  2. What is JavaScript?... - 0.005s
  3. What is TypeScript?... - 0.004s
  4. What is Go?... - 0.005s
  5. What is Rust?... - 0.004s

总耗时: 0.022s
平均耗时: 0.004s
性能提升: 253.6x

【步骤9】缓存清理
--------------------------------------------------------------------------------
清理前缓存键数量: 6
提示：生产环境建议使用 TTL 自动过期，而非手动清理

【步骤10】生产环境最佳实践
--------------------------------------------------------------------------------

1. 持久化配置：
   - 启用 RDB：定期快照备份
   - 启用 AOF：实时持久化
   - 配置示例：
     save 900 1
     save 300 10
     save 60 10000
     appendonly yes
     appendfsync everysec

2. 内存管理：
   - 设置 maxmemory 限制
   - 配置淘汰策略：maxmemory-policy allkeys-lru
   - 监控内存使用

3. 连接池配置：
   - 使用连接池避免频繁创建连接
   - 设置合理的超时时间
   - 配置最大连接数

4. 缓存策略：
   - 设置合理的 TTL（如 7 天）
   - 监控缓存命中率
   - 定期清理过期数据

5. 监控告警：
   - 监控 Redis 内存使用
   - 监控缓存命中率
   - 监控连接数和延迟
   - 设置告警阈值

6. 高可用：
   - 使用 Redis Sentinel 或 Cluster
   - 配置主从复制
   - 定期备份数据

================================================================================
【总结】RedisCache 生产部署要点
================================================================================

1. 连接配置：
   - 使用环境变量管理配置
   - 设置连接超时和操作超时
   - 使用连接池提高性能

2. 持久化：
   - RDB + AOF 双重保障
   - 定期备份数据
   - 测试恢复流程

3. 性能优化：
   - 设置合理的 TTL
   - 监控缓存命中率
   - 使用 pipeline 批量操作

4. 分布式支持：
   - 多个服务实例共享缓存
   - 避免缓存雪崩
   - 考虑缓存预热

5. 监控运维：
   - 监控内存、连接、延迟
   - 设置告警阈值
   - 定期检查持久化状态

✓ 场景2演示完成！
```

---

## 关键点解释

### 1. Redis 持久化策略

**RDB (Redis Database)**：
- 定期快照备份
- 适合灾难恢复
- 配置示例：`save 900 1`（900秒内至少1个键变化）

**AOF (Append Only File)**：
- 实时记录写操作
- 数据更安全
- 配置示例：`appendonly yes`, `appendfsync everysec`

**推荐配置**：
```conf
# RDB
save 900 1
save 300 10
save 60 10000

# AOF
appendonly yes
appendfsync everysec
```

### 2. 内存管理

**maxmemory 配置**：
```conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

**淘汰策略**：
- `allkeys-lru`：所有键 LRU 淘汰（推荐）
- `volatile-lru`：仅淘汰设置了过期时间的键
- `allkeys-random`：随机淘汰
- `noeviction`：不淘汰，内存满时报错

### 3. 连接池配置

```python
from redis import ConnectionPool

pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,  # 最大连接数
    socket_connect_timeout=5,
    socket_timeout=5,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=pool)
```

### 4. 缓存键设计

**LangChain 缓存键格式**：
```
langchain:cache:llm:{model}:{temperature}:{max_tokens}:{prompt}
```

**注意事项**：
- 键名包含所有配置参数
- 确保配置一致性
- 避免键名过长

### 5. TTL 设置策略

**为什么需要 TTL**：
- 避免缓存无限增长
- 自动清理过期数据
- 减少内存压力

**推荐 TTL**：
- 开发环境：1-3 天
- 生产环境：7-30 天
- 根据业务需求调整

### 6. 监控指标

**关键指标**：
- `used_memory`：已用内存
- `connected_clients`：连接数
- `keyspace_hits`：缓存命中次数
- `keyspace_misses`：缓存未命中次数
- `evicted_keys`：淘汰的键数量

**命中率计算**：
```python
hits = info['keyspace_hits']
misses = info['keyspace_misses']
hit_rate = hits / (hits + misses) * 100
```

---

## 数据来源

本文档基于以下资料编写：

1. **官方文档** (`reference/context7_langchain_cache_01.md`)
   - RedisCache 使用方法
   - 配置参数说明

2. **社区讨论** (`reference/search_cache_reddit_01.md`)
   - 生产环境配置建议
   - 持久化策略讨论
   - 性能优化经验

3. **GitHub Issues** (`reference/search_cache_github_01.md`)
   - 缓存键生成问题
   - 配置参数影响

---

## 下一步学习

完成本场景后，建议继续学习：

1. **场景3：CacheBackedEmbeddings 实战** - Embedding 缓存
2. **场景4：语义缓存实现** - 提升缓存命中率
3. **场景5：缓存性能优化** - 监控和优化策略

---

**版本信息**：
- LangChain 版本：0.3.x (2025+)
- Redis 版本：6.0+
- Python 版本：3.13+
- 最后更新：2026-02-25
