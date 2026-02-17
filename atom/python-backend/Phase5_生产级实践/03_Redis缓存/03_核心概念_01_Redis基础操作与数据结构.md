# 核心概念1：Redis基础操作与数据结构

## 概述

Redis支持5种核心数据结构，每种结构适合不同的使用场景。在AI Agent开发中，选择合适的数据结构可以显著提升性能和内存效率。

---

## 1. String类型：最基础的键值对

### 定义

**String是Redis最简单的数据类型，存储字符串、数字或二进制数据。**

### 核心操作

```python
import redis

client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 1. SET/GET：基础读写
client.set("key", "value")
value = client.get("key")
print(value)  # "value"

# 2. SETEX：设置带过期时间的值
client.setex("session:abc123", 3600, "user_data")  # 1小时后过期

# 3. SETNX：只在key不存在时设置（分布式锁）
success = client.setnx("lock:resource", "locked")
print(success)  # True（第一次）或 False（已存在）

# 4. INCR/DECR：原子递增/递减
client.set("counter", 0)
client.incr("counter")  # 1
client.incr("counter")  # 2
client.decr("counter")  # 1

# 5. MGET/MSET：批量操作
client.mset({"key1": "value1", "key2": "value2", "key3": "value3"})
values = client.mget("key1", "key2", "key3")
print(values)  # ['value1', 'value2', 'value3']

# 6. APPEND：追加字符串
client.set("message", "Hello")
client.append("message", " World")
print(client.get("message"))  # "Hello World"

# 7. GETRANGE：获取子字符串
client.set("text", "Hello World")
substring = client.getrange("text", 0, 4)
print(substring)  # "Hello"
```

### 在AI Agent中的应用

```python
# 1. 缓存LLM响应
def cache_llm_response(prompt: str, response: str):
    cache_key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    client.setex(cache_key, 3600, response)

# 2. 缓存API token
def cache_api_token(user_id: int, token: str):
    client.setex(f"token:{user_id}", 1800, token)  # 30分钟

# 3. 计数器（API调用次数）
def increment_api_calls(user_id: int):
    key = f"api_calls:{user_id}:{datetime.now().strftime('%Y%m%d')}"
    client.incr(key)
    client.expire(key, 86400)  # 24小时后过期

# 4. 分布式锁
def acquire_lock(resource_id: str, timeout: int = 10):
    lock_key = f"lock:{resource_id}"
    acquired = client.setnx(lock_key, "locked")
    if acquired:
        client.expire(lock_key, timeout)
    return acquired
```

---

## 2. Hash类型：存储对象的多个字段

### 定义

**Hash类型存储字段-值对的集合，适合存储对象。**

### 核心操作

```python
# 1. HSET/HGET：设置/获取单个字段
client.hset("user:1001", "name", "John Doe")
client.hset("user:1001", "email", "john@example.com")
name = client.hget("user:1001", "name")
print(name)  # "John Doe"

# 2. HMSET/HMGET：批量设置/获取
client.hset("user:1001", mapping={
    "name": "John Doe",
    "email": "john@example.com",
    "role": "admin"
})
values = client.hmget("user:1001", "name", "email")
print(values)  # ['John Doe', 'john@example.com']

# 3. HGETALL：获取所有字段
user = client.hgetall("user:1001")
print(user)  # {'name': 'John Doe', 'email': 'john@example.com', 'role': 'admin'}

# 4. HEXISTS：检查字段是否存在
exists = client.hexists("user:1001", "name")
print(exists)  # True

# 5. HDEL：删除字段
client.hdel("user:1001", "role")

# 6. HKEYS/HVALS：获取所有字段名/值
keys = client.hkeys("user:1001")
values = client.hvals("user:1001")
print(keys)    # ['name', 'email']
print(values)  # ['John Doe', 'john@example.com']

# 7. HLEN：获取字段数量
count = client.hlen("user:1001")
print(count)  # 2

# 8. HINCRBY：字段值递增
client.hset("stats:user:1001", "api_calls", 0)
client.hincrby("stats:user:1001", "api_calls", 1)
```

### 在AI Agent中的应用

```python
# 1. 缓存用户会话
def cache_user_session(session_id: str, user_data: dict):
    client.hset(f"session:{session_id}", mapping=user_data)
    client.expire(f"session:{session_id}", 1800)  # 30分钟

# 2. 缓存Agent执行状态
def cache_agent_state(agent_id: str, state: dict):
    client.hset(f"agent:{agent_id}", mapping={
        "status": state["status"],
        "current_step": state["current_step"],
        "progress": state["progress"],
        "last_update": datetime.now().isoformat()
    })

# 3. 缓存文档元数据
def cache_document_metadata(doc_id: str, metadata: dict):
    client.hset(f"doc:{doc_id}", mapping={
        "title": metadata["title"],
        "author": metadata["author"],
        "created_at": metadata["created_at"],
        "chunk_count": metadata["chunk_count"]
    })

# 4. 用户统计数据
def update_user_stats(user_id: int, stat_type: str):
    key = f"stats:user:{user_id}"
    client.hincrby(key, stat_type, 1)
    client.expire(key, 86400)  # 24小时
```

---

## 3. List类型：有序列表

### 定义

**List是有序的字符串列表，支持从两端插入和弹出。**

### 核心操作

```python
# 1. LPUSH/RPUSH：从左/右插入
client.lpush("queue", "task1")  # 从左插入
client.rpush("queue", "task2")  # 从右插入

# 2. LPOP/RPOP：从左/右弹出
task = client.lpop("queue")
print(task)  # "task1"

# 3. LRANGE：获取范围内的元素
client.rpush("messages", "msg1", "msg2", "msg3")
messages = client.lrange("messages", 0, -1)  # 获取所有
print(messages)  # ['msg1', 'msg2', 'msg3']

# 4. LLEN：获取列表长度
length = client.llen("messages")
print(length)  # 3

# 5. LINDEX：获取指定位置的元素
msg = client.lindex("messages", 1)
print(msg)  # "msg2"

# 6. LSET：设置指定位置的元素
client.lset("messages", 1, "new_msg2")

# 7. LTRIM：保留指定范围的元素
client.ltrim("messages", 0, 99)  # 只保留前100个

# 8. BLPOP/BRPOP：阻塞式弹出（用于队列）
task = client.blpop("queue", timeout=5)  # 等待5秒
```

### 在AI Agent中的应用

```python
# 1. 任务队列
def enqueue_task(task_data: dict):
    client.rpush("task_queue", json.dumps(task_data))

def dequeue_task() -> Optional[dict]:
    task_json = client.lpop("task_queue")
    return json.loads(task_json) if task_json else None

# 2. 对话历史（最近N条消息）
def add_message(conversation_id: str, message: dict):
    key = f"conversation:{conversation_id}"
    client.rpush(key, json.dumps(message))
    client.ltrim(key, -10, -1)  # 只保留最近10条
    client.expire(key, 3600)

def get_conversation_history(conversation_id: str) -> list:
    key = f"conversation:{conversation_id}"
    messages = client.lrange(key, 0, -1)
    return [json.loads(msg) for msg in messages]

# 3. 日志缓冲区
def log_to_redis(log_entry: str):
    client.rpush("logs", log_entry)
    client.ltrim("logs", -1000, -1)  # 只保留最近1000条

# 4. 实时通知队列
def push_notification(user_id: int, notification: dict):
    key = f"notifications:{user_id}"
    client.lpush(key, json.dumps(notification))
    client.ltrim(key, 0, 49)  # 只保留最近50条
    client.expire(key, 86400)
```

---

## 4. Set类型：无序不重复集合

### 定义

**Set是无序的字符串集合，元素不重复。**

### 核心操作

```python
# 1. SADD：添加元素
client.sadd("tags", "python", "redis", "ai")

# 2. SMEMBERS：获取所有元素
tags = client.smembers("tags")
print(tags)  # {'python', 'redis', 'ai'}

# 3. SISMEMBER：检查元素是否存在
exists = client.sismember("tags", "python")
print(exists)  # True

# 4. SREM：删除元素
client.srem("tags", "redis")

# 5. SCARD：获取集合大小
count = client.scard("tags")
print(count)  # 2

# 6. SINTER：交集
client.sadd("set1", "a", "b", "c")
client.sadd("set2", "b", "c", "d")
intersection = client.sinter("set1", "set2")
print(intersection)  # {'b', 'c'}

# 7. SUNION：并集
union = client.sunion("set1", "set2")
print(union)  # {'a', 'b', 'c', 'd'}

# 8. SDIFF：差集
diff = client.sdiff("set1", "set2")
print(diff)  # {'a'}

# 9. SPOP：随机弹出元素
element = client.spop("tags")

# 10. SRANDMEMBER：随机获取元素（不删除）
element = client.srandmember("tags")
```

### 在AI Agent中的应用

```python
# 1. 用户标签系统
def add_user_tags(user_id: int, tags: list[str]):
    client.sadd(f"user:{user_id}:tags", *tags)

def get_user_tags(user_id: int) -> set:
    return client.smembers(f"user:{user_id}:tags")

# 2. 去重（已处理的文档ID）
def mark_document_processed(doc_id: str):
    client.sadd("processed_docs", doc_id)

def is_document_processed(doc_id: str) -> bool:
    return client.sismember("processed_docs", doc_id)

# 3. 在线用户列表
def user_online(user_id: int):
    client.sadd("online_users", user_id)
    client.expire("online_users", 300)  # 5分钟

def user_offline(user_id: int):
    client.srem("online_users", user_id)

def get_online_users() -> set:
    return client.smembers("online_users")

# 4. 推荐系统（共同兴趣）
def get_common_interests(user1_id: int, user2_id: int) -> set:
    return client.sinter(
        f"user:{user1_id}:interests",
        f"user:{user2_id}:interests"
    )
```

---

## 5. Sorted Set类型：有序集合

### 定义

**Sorted Set是有序的集合，每个元素关联一个分数（score），按分数排序。**

### 核心操作

```python
# 1. ZADD：添加元素（带分数）
client.zadd("leaderboard", {"user1": 100, "user2": 95, "user3": 110})

# 2. ZRANGE：按分数升序获取
users = client.zrange("leaderboard", 0, -1, withscores=True)
print(users)  # [('user2', 95.0), ('user1', 100.0), ('user3', 110.0)]

# 3. ZREVRANGE：按分数降序获取
users = client.zrevrange("leaderboard", 0, -1, withscores=True)
print(users)  # [('user3', 110.0), ('user1', 100.0), ('user2', 95.0)]

# 4. ZSCORE：获取元素的分数
score = client.zscore("leaderboard", "user1")
print(score)  # 100.0

# 5. ZRANK：获取元素的排名（升序）
rank = client.zrank("leaderboard", "user1")
print(rank)  # 1（从0开始）

# 6. ZREVRANK：获取元素的排名（降序）
rank = client.zrevrank("leaderboard", "user1")
print(rank)  # 1

# 7. ZINCRBY：增加元素的分数
client.zincrby("leaderboard", 10, "user1")  # user1的分数+10

# 8. ZREM：删除元素
client.zrem("leaderboard", "user2")

# 9. ZCARD：获取集合大小
count = client.zcard("leaderboard")
print(count)  # 2

# 10. ZCOUNT：统计分数范围内的元素数量
count = client.zcount("leaderboard", 90, 105)
print(count)  # 1

# 11. ZRANGEBYSCORE：按分数范围获取
users = client.zrangebyscore("leaderboard", 90, 105, withscores=True)
```

### 在AI Agent中的应用

```python
# 1. 排行榜
def update_leaderboard(user_id: int, score: int):
    client.zadd("leaderboard", {f"user:{user_id}": score})

def get_top_users(limit: int = 10) -> list:
    return client.zrevrange("leaderboard", 0, limit-1, withscores=True)

# 2. 延迟队列（按时间戳排序）
def schedule_task(task_id: str, execute_at: float):
    client.zadd("delayed_tasks", {task_id: execute_at})

def get_due_tasks() -> list:
    now = time.time()
    tasks = client.zrangebyscore("delayed_tasks", 0, now)
    if tasks:
        client.zrem("delayed_tasks", *tasks)
    return tasks

# 3. 热门文档（按访问次数排序）
def increment_document_views(doc_id: str):
    client.zincrby("hot_documents", 1, doc_id)

def get_hot_documents(limit: int = 10) -> list:
    return client.zrevrange("hot_documents", 0, limit-1, withscores=True)

# 4. 语义缓存（按相似度排序）
def add_semantic_cache(query_hash: str, similarity: float, response: str):
    # 使用Sorted Set存储相似度
    client.zadd(f"semantic:{query_hash}", {response: similarity})
    client.expire(f"semantic:{query_hash}", 3600)

def get_best_match(query_hash: str, threshold: float = 0.9):
    # 获取相似度最高的缓存
    matches = client.zrevrangebyscore(
        f"semantic:{query_hash}",
        "+inf",
        threshold,
        start=0,
        num=1,
        withscores=True
    )
    return matches[0] if matches else None
```

---

## 数据结构选择指南

| 场景 | 推荐数据结构 | 理由 |
|------|-------------|------|
| 缓存LLM响应 | String | 简单键值对，查询快 |
| 缓存用户信息 | Hash | 对象有多个字段 |
| 对话历史 | List | 有序，需要保留最近N条 |
| 已处理文档ID | Set | 去重，快速判断存在 |
| 用户标签 | Set | 无序不重复，支持交并差 |
| 排行榜 | Sorted Set | 需要按分数排序 |
| 延迟任务 | Sorted Set | 按时间戳排序 |
| 任务队列 | List | FIFO队列 |
| 分布式锁 | String | SETNX原子操作 |
| 计数器 | String | INCR原子递增 |

---

## 性能对比

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| String GET/SET | O(1) | 最快 |
| Hash HGET/HSET | O(1) | 单字段操作 |
| Hash HGETALL | O(N) | N为字段数 |
| List LPUSH/RPUSH | O(1) | 两端插入 |
| List LINDEX | O(N) | 需要遍历 |
| Set SADD/SISMEMBER | O(1) | 基于哈希表 |
| Set SINTER | O(N*M) | N为最小集合大小 |
| Sorted Set ZADD | O(log N) | 基于跳表 |
| Sorted Set ZRANGE | O(log N + M) | M为返回元素数 |

---

## 内存优化技巧

### 1. 使用Hash代替多个String

```python
# ❌ 不推荐：每个字段一个key
client.set("user:1001:name", "John")
client.set("user:1001:email", "john@example.com")
client.set("user:1001:role", "admin")
# 内存占用：3个key，每个key有额外开销

# ✅ 推荐：使用Hash
client.hset("user:1001", mapping={
    "name": "John",
    "email": "john@example.com",
    "role": "admin"
})
# 内存占用：1个key，节省内存
```

### 2. 设置合理的TTL

```python
# 避免永久存储，设置TTL自动清理
client.setex("cache:key", 3600, "value")  # 1小时后自动删除
```

### 3. 使用压缩

```python
import gzip
import json

def set_compressed(key: str, data: dict, ttl: int):
    """压缩后存储"""
    json_str = json.dumps(data)
    compressed = gzip.compress(json_str.encode())
    client.setex(key, ttl, compressed)

def get_compressed(key: str) -> dict:
    """解压缩读取"""
    compressed = client.get(key)
    if not compressed:
        return None
    json_str = gzip.decompress(compressed).decode()
    return json.loads(json_str)
```

---

## 总结

1. **String**：最基础，适合简单键值对（LLM响应、token、计数器）
2. **Hash**：存储对象，适合多字段数据（用户信息、Agent状态）
3. **List**：有序列表，适合队列和历史记录（任务队列、对话历史）
4. **Set**：无序不重复，适合去重和集合运算（标签、在线用户）
5. **Sorted Set**：有序集合，适合排序场景（排行榜、延迟队列）

**选择原则：** 根据数据特性和操作需求选择合适的数据结构，平衡性能和内存占用。
