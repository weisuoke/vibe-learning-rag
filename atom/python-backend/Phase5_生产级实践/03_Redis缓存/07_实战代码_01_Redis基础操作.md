# 实战代码1：Redis基础操作

## 完整可运行示例

```python
"""
Redis基础操作实战
演示：String、Hash、List、Set、Sorted Set的基本用法
"""

import redis
import json
from typing import List, Optional
from datetime import datetime

# ===== 1. 连接Redis =====
print("=== 连接Redis ===")

client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # 自动解码为字符串
)

# 测试连接
try:
    client.ping()
    print("✅ Redis连接成功")
except redis.ConnectionError:
    print("❌ Redis连接失败，请确保Redis服务已启动")
    exit(1)

# ===== 2. String类型操作 =====
print("\n=== String类型操作 ===")

# 基础读写
client.set("name", "John Doe")
name = client.get("name")
print(f"GET name: {name}")

# 带过期时间
client.setex("session:abc123", 60, "user_data")
ttl = client.ttl("session:abc123")
print(f"Session TTL: {ttl}秒")

# 原子递增
client.set("counter", 0)
client.incr("counter")
client.incr("counter")
counter = client.get("counter")
print(f"Counter: {counter}")

# 批量操作
client.mset({"key1": "value1", "key2": "value2", "key3": "value3"})
values = client.mget("key1", "key2", "key3")
print(f"MGET: {values}")

# ===== 3. Hash类型操作 =====
print("\n=== Hash类型操作 ===")

# 存储用户对象
user_data = {
    "name": "Alice",
    "email": "alice@example.com",
    "role": "admin",
    "created_at": datetime.now().isoformat()
}

client.hset("user:1001", mapping=user_data)

# 获取单个字段
name = client.hget("user:1001", "name")
print(f"User name: {name}")

# 获取所有字段
user = client.hgetall("user:1001")
print(f"User data: {json.dumps(user, indent=2)}")

# 检查字段是否存在
exists = client.hexists("user:1001", "email")
print(f"Email exists: {exists}")

# 字段计数
field_count = client.hlen("user:1001")
print(f"Field count: {field_count}")

# ===== 4. List类型操作 =====
print("\n=== List类型操作 ===")

# 任务队列
client.delete("task_queue")  # 清空
client.rpush("task_queue", "task1", "task2", "task3")

# 获取队列长度
length = client.llen("task_queue")
print(f"Queue length: {length}")

# 弹出任务
task = client.lpop("task_queue")
print(f"Popped task: {task}")

# 获取所有任务
tasks = client.lrange("task_queue", 0, -1)
print(f"Remaining tasks: {tasks}")

# 对话历史（保留最近5条）
client.delete("conversation:123")
for i in range(10):
    client.rpush("conversation:123", f"message_{i}")
    client.ltrim("conversation:123", -5, -1)  # 只保留最近5条

history = client.lrange("conversation:123", 0, -1)
print(f"Conversation history (last 5): {history}")

# ===== 5. Set类型操作 =====
print("\n=== Set类型操作 ===")

# 用户标签
client.sadd("user:1001:tags", "python", "redis", "ai", "backend")

# 获取所有标签
tags = client.smembers("user:1001:tags")
print(f"User tags: {tags}")

# 检查标签是否存在
has_python = client.sismember("user:1001:tags", "python")
print(f"Has 'python' tag: {has_python}")

# 标签数量
tag_count = client.scard("user:1001:tags")
print(f"Tag count: {tag_count}")

# 集合运算
client.sadd("user:1002:tags", "python", "javascript", "frontend")

# 交集（共同标签）
common_tags = client.sinter("user:1001:tags", "user:1002:tags")
print(f"Common tags: {common_tags}")

# 并集（所有标签）
all_tags = client.sunion("user:1001:tags", "user:1002:tags")
print(f"All tags: {all_tags}")

# 差集（user1独有的标签）
unique_tags = client.sdiff("user:1001:tags", "user:1002:tags")
print(f"User1 unique tags: {unique_tags}")

# ===== 6. Sorted Set类型操作 =====
print("\n=== Sorted Set类型操作 ===")

# 排行榜
client.zadd("leaderboard", {
    "Alice": 100,
    "Bob": 95,
    "Charlie": 110,
    "David": 88
})

# 获取排名（降序）
top_users = client.zrevrange("leaderboard", 0, 2, withscores=True)
print(f"Top 3 users: {top_users}")

# 获取用户分数
alice_score = client.zscore("leaderboard", "Alice")
print(f"Alice's score: {alice_score}")

# 获取用户排名（降序，0开始）
alice_rank = client.zrevrank("leaderboard", "Alice")
print(f"Alice's rank: {alice_rank + 1}")  # +1转换为从1开始

# 增加分数
client.zincrby("leaderboard", 10, "Alice")
new_score = client.zscore("leaderboard", "Alice")
print(f"Alice's new score: {new_score}")

# 按分数范围查询
mid_range = client.zrangebyscore("leaderboard", 90, 105, withscores=True)
print(f"Scores 90-105: {mid_range}")

# ===== 7. 过期时间管理 =====
print("\n=== 过期时间管理 ===")

# 设置过期时间
client.set("temp_key", "temp_value")
client.expire("temp_key", 10)  # 10秒后过期

# 查看剩余时间
ttl = client.ttl("temp_key")
print(f"Temp key TTL: {ttl}秒")

# 取消过期
client.persist("temp_key")
ttl = client.ttl("temp_key")
print(f"After persist, TTL: {ttl}")  # -1表示永不过期

# ===== 8. 键管理 =====
print("\n=== 键管理 ===")

# 检查键是否存在
exists = client.exists("name")
print(f"Key 'name' exists: {exists}")

# 删除键
client.delete("temp_key")

# 获取键的类型
key_type = client.type("user:1001")
print(f"Key type: {key_type}")

# 模式匹配（谨慎使用，生产环境可能很慢）
keys = client.keys("user:*")
print(f"Keys matching 'user:*': {keys[:5]}")  # 只显示前5个

# ===== 9. 清理 =====
print("\n=== 清理测试数据 ===")

# 删除所有测试键
test_keys = [
    "name", "session:abc123", "counter",
    "key1", "key2", "key3",
    "user:1001", "user:1002:tags",
    "task_queue", "conversation:123",
    "user:1001:tags", "leaderboard"
]

for key in test_keys:
    client.delete(key)

print("✅ 测试数据已清理")
```

## 运行输出示例

```
=== 连接Redis ===
✅ Redis连接成功

=== String类型操作 ===
GET name: John Doe
Session TTL: 60秒
Counter: 2
MGET: ['value1', 'value2', 'value3']

=== Hash类型操作 ===
User name: Alice
User data: {
  "name": "Alice",
  "email": "alice@example.com",
  "role": "admin",
  "created_at": "2026-02-12T14:30:00.123456"
}
Email exists: True
Field count: 4

=== List类型操作 ===
Queue length: 3
Popped task: task1
Remaining tasks: ['task2', 'task3']
Conversation history (last 5): ['message_5', 'message_6', 'message_7', 'message_8', 'message_9']

=== Set类型操作 ===
User tags: {'python', 'redis', 'ai', 'backend'}
Has 'python' tag: True
Tag count: 4
Common tags: {'python'}
All tags: {'python', 'redis', 'ai', 'backend', 'javascript', 'frontend'}
User1 unique tags: {'redis', 'ai', 'backend'}

=== Sorted Set类型操作 ===
Top 3 users: [('Charlie', 110.0), ('Alice', 100.0), ('Bob', 95.0)]
Alice's score: 100.0
Alice's rank: 2
Alice's new score: 110.0
Scores 90-105: [('Bob', 95.0)]

=== 过期时间管理 ===
Temp key TTL: 10秒
After persist, TTL: -1

=== 键管理 ===
Key 'name' exists: 1
Key type: hash
Keys matching 'user:*': ['user:1001']

=== 清理测试数据 ===
✅ 测试数据已清理
```

## 在AI Agent中的应用

```python
"""
AI Agent中的Redis应用示例
"""

import redis
import json
from datetime import datetime

client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 1. 缓存用户会话
def cache_user_session(session_id: str, user_data: dict):
    """缓存用户会话（30分钟）"""
    client.hset(f"session:{session_id}", mapping=user_data)
    client.expire(f"session:{session_id}", 1800)

# 2. 记录对话历史
def add_conversation_message(conversation_id: str, message: dict):
    """添加对话消息（保留最近20条）"""
    client.rpush(f"conv:{conversation_id}", json.dumps(message))
    client.ltrim(f"conv:{conversation_id}", -20, -1)
    client.expire(f"conv:{conversation_id}", 3600)

# 3. 用户标签系统
def add_user_interests(user_id: int, interests: List[str]):
    """添加用户兴趣标签"""
    client.sadd(f"user:{user_id}:interests", *interests)

# 4. 热门文档排行
def increment_document_views(doc_id: str):
    """增加文档浏览次数"""
    client.zincrby("hot_docs", 1, doc_id)

def get_hot_documents(limit: int = 10):
    """获取热门文档"""
    return client.zrevrange("hot_docs", 0, limit-1, withscores=True)

# 5. API调用计数
def increment_api_calls(user_id: int):
    """记录API调用次数（按天）"""
    key = f"api_calls:{user_id}:{datetime.now().strftime('%Y%m%d')}"
    client.incr(key)
    client.expire(key, 86400)

# 使用示例
cache_user_session("abc123", {"user_id": 1001, "name": "Alice"})
add_conversation_message("conv_001", {"role": "user", "content": "Hello"})
add_user_interests(1001, ["python", "ai", "redis"])
increment_document_views("doc_123")
increment_api_calls(1001)

print("✅ AI Agent Redis操作完成")
```

## 学习检查清单

- [ ] 理解5种数据结构的特点和使用场景
- [ ] 掌握String的基本操作（SET/GET/INCR）
- [ ] 掌握Hash的对象存储（HSET/HGET/HGETALL）
- [ ] 掌握List的队列操作（LPUSH/RPOP/LTRIM）
- [ ] 掌握Set的集合运算（SADD/SINTER/SUNION）
- [ ] 掌握Sorted Set的排序功能（ZADD/ZRANGE/ZINCRBY）
- [ ] 理解TTL和过期机制
- [ ] 能够在AI Agent项目中应用Redis

## 下一步

学习如何使用Redis缓存LLM响应，降低API调用成本。
