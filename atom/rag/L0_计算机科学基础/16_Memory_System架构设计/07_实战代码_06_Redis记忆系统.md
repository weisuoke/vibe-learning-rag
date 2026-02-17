# 实战代码：Redis记忆系统

> **使用 Redis 构建高性能的 AI Agent 记忆系统**

---

## 学习目标

- 掌握 Redis 的数据结构和命令
- 实现短期和长期记忆管理
- 理解语义缓存的 Redis 实现
- 构建生产级的记忆系统

---

## 基础实现

### Redis 连接配置

```python
import redis
from typing import Optional, List, Dict
import json

class RedisMemorySystem:
    """基于 Redis 的记忆系统"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        初始化 Redis 连接

        Args:
            host: Redis 主机
            port: Redis 端口
            db: 数据库编号
            password: 密码
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # 保持二进制模式
        )

        # 测试连接
        try:
            self.redis.ping()
            print("✅ Redis 连接成功")
        except redis.ConnectionError as e:
            print(f"❌ Redis 连接失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        self.redis.close()
```

---

## 短期记忆（Session Storage）

### 对话历史管理

```python
class SessionMemory:
    """会话级短期记忆"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        ttl: int = 3600
    ) -> None:
        """
        添加对话消息

        Args:
            user_id: 用户 ID
            role: 角色（user/assistant）
            content: 消息内容
            ttl: 过期时间（秒）
        """
        key = f"session:{user_id}:messages"
        message = json.dumps({"role": role, "content": content})

        # 使用 List 存储消息
        self.redis.lpush(key, message)

        # 只保留最近 100 条
        self.redis.ltrim(key, 0, 99)

        # 设置过期时间
        self.redis.expire(key, ttl)

    def get_recent_messages(
        self,
        user_id: str,
        n: int = 10
    ) -> List[Dict]:
        """
        获取最近消息

        Args:
            user_id: 用户 ID
            n: 消息数量

        Returns:
            消息列表
        """
        key = f"session:{user_id}:messages"
        messages = self.redis.lrange(key, 0, n - 1)

        return [json.loads(msg.decode()) for msg in messages]

    def clear_session(self, user_id: str) -> None:
        """清空会话"""
        key = f"session:{user_id}:messages"
        self.redis.delete(key)

# 使用示例
redis_client = redis.Redis(host='localhost', port=6379)
session = SessionMemory(redis_client)

# 添加消息
session.add_message("user_123", "user", "你好")
session.add_message("user_123", "assistant", "你好！")

# 获取最近消息
recent = session.get_recent_messages("user_123", n=10)
print(recent)
```

---

## 长期记忆（Persistent Storage）

### 用户偏好管理

```python
class LongTermMemory:
    """长期记忆管理"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def save_preference(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> None:
        """
        保存用户偏好

        Args:
            user_id: 用户 ID
            key: 偏好键
            value: 偏好值
        """
        redis_key = f"user:{user_id}:preferences"
        self.redis.hset(redis_key, key, json.dumps(value))

    def get_preference(
        self,
        user_id: str,
        key: str
    ) -> Optional[any]:
        """
        获取用户偏好

        Args:
            user_id: 用户 ID
            key: 偏好键

        Returns:
            偏好值
        """
        redis_key = f"user:{user_id}:preferences"
        value = self.redis.hget(redis_key, key)

        if value:
            return json.loads(value.decode())
        return None

    def get_all_preferences(
        self,
        user_id: str
    ) -> Dict:
        """
        获取所有偏好

        Args:
            user_id: 用户 ID

        Returns:
            偏好字典
        """
        redis_key = f"user:{user_id}:preferences"
        prefs = self.redis.hgetall(redis_key)

        return {
            k.decode(): json.loads(v.decode())
            for k, v in prefs.items()
        }

    def delete_preference(
        self,
        user_id: str,
        key: str
    ) -> bool:
        """
        删除偏好

        Args:
            user_id: 用户 ID
            key: 偏好键

        Returns:
            是否删除成功
        """
        redis_key = f"user:{user_id}:preferences"
        return self.redis.hdel(redis_key, key) > 0

# 使用示例
long_term = LongTermMemory(redis_client)

# 保存偏好
long_term.save_preference("user_123", "language", "zh-CN")
long_term.save_preference("user_123", "theme", "dark")

# 获取偏好
lang = long_term.get_preference("user_123", "language")
all_prefs = long_term.get_all_preferences("user_123")

print(f"语言: {lang}")
print(f"所有偏好: {all_prefs}")
```

---

## 语义缓存

### 基于 Redis 的语义缓存

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

class RedisSemanticCache:
    """Redis 语义缓存"""

    def __init__(
        self,
        redis_client: redis.Redis,
        threshold: float = 0.95,
        ttl: int = 86400
    ):
        """
        初始化

        Args:
            redis_client: Redis 客户端
            threshold: 相似度阈值
            ttl: 过期时间（秒）
        """
        self.redis = redis_client
        self.threshold = threshold
        self.ttl = ttl

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本向量"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, question: str) -> Optional[str]:
        """
        查询缓存

        Args:
            question: 问题

        Returns:
            答案，如果未命中返回 None
        """
        query_emb = self._get_embedding(question)

        # 获取所有缓存的键
        keys = self.redis.keys("semantic:*")
        if not keys:
            return None

        best_match = None
        best_similarity = 0.0

        for key in keys:
            # 获取缓存数据
            data = self.redis.hgetall(key)
            if not data:
                continue

            cached_emb = np.frombuffer(
                data[b"embedding"],
                dtype=np.float32
            )
            cached_answer = data[b"answer"].decode()

            # 计算相似度
            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_answer

        if best_similarity >= self.threshold:
            print(f"✅ 语义缓存命中 (相似度: {best_similarity:.3f})")
            return best_match

        return None

    def put(self, question: str, answer: str) -> None:
        """
        存入缓存

        Args:
            question: 问题
            answer: 答案
        """
        embedding = self._get_embedding(question)
        key = f"semantic:{hash(question)}"

        # 存储到 Redis
        self.redis.hset(key, mapping={
            "question": question,
            "answer": answer,
            "embedding": embedding.astype(np.float32).tobytes()
        })

        # 设置过期时间
        self.redis.expire(key, self.ttl)

    def clear(self) -> int:
        """
        清空缓存

        Returns:
            删除的数量
        """
        keys = self.redis.keys("semantic:*")
        if keys:
            return self.redis.delete(*keys)
        return 0

# 使用示例
semantic_cache = RedisSemanticCache(redis_client, threshold=0.95)

# 缓存问答
semantic_cache.put("什么是 RAG？", "RAG 是检索增强生成...")

# 查询缓存
answer = semantic_cache.get("RAG 是什么？")  # ✅ 命中
print(answer)
```

---

## 完整的记忆系统

### 集成短期、长期和语义缓存

```python
class UnifiedMemorySystem:
    """统一的记忆系统"""

    def __init__(self, redis_client: redis.Redis):
        """
        初始化

        Args:
            redis_client: Redis 客户端
        """
        self.redis = redis_client
        self.session = SessionMemory(redis_client)
        self.long_term = LongTermMemory(redis_client)
        self.semantic = RedisSemanticCache(redis_client)

    # ========== 对话管理 ==========

    def add_conversation(
        self,
        user_id: str,
        role: str,
        content: str
    ) -> None:
        """
        添加对话消息

        Args:
            user_id: 用户 ID
            role: 角色
            content: 内容
        """
        self.session.add_message(user_id, role, content)

    def get_conversation_history(
        self,
        user_id: str,
        n: int = 10
    ) -> List[Dict]:
        """
        获取对话历史

        Args:
            user_id: 用户 ID
            n: 消息数量

        Returns:
            消息列表
        """
        return self.session.get_recent_messages(user_id, n)

    # ========== 偏好管理 ==========

    def save_user_preference(
        self,
        user_id: str,
        key: str,
        value: any
    ) -> None:
        """保存用户偏好"""
        self.long_term.save_preference(user_id, key, value)

    def get_user_preference(
        self,
        user_id: str,
        key: str
    ) -> Optional[any]:
        """获取用户偏好"""
        return self.long_term.get_preference(user_id, key)

    # ========== 语义缓存 ==========

    def cache_qa(self, question: str, answer: str) -> None:
        """缓存问答对"""
        self.semantic.put(question, answer)

    def get_cached_answer(self, question: str) -> Optional[str]:
        """获取缓存的答案"""
        return self.semantic.get(question)

    # ========== 统计信息 ==========

    def get_user_stats(self, user_id: str) -> Dict:
        """
        获取用户统计

        Args:
            user_id: 用户 ID

        Returns:
            统计信息
        """
        # 对话数量
        conv_key = f"session:{user_id}:messages"
        conv_count = self.redis.llen(conv_key)

        # 偏好数量
        pref_key = f"user:{user_id}:preferences"
        pref_count = self.redis.hlen(pref_key)

        return {
            "conversations": conv_count,
            "preferences": pref_count
        }

# 使用示例
memory = UnifiedMemorySystem(redis_client)

# 对话管理
memory.add_conversation("user_123", "user", "你好")
memory.add_conversation("user_123", "assistant", "你好！")

# 偏好管理
memory.save_user_preference("user_123", "language", "zh-CN")

# 语义缓存
memory.cache_qa("什么是 RAG？", "RAG 是检索增强生成...")

# 统计
stats = memory.get_user_stats("user_123")
print(f"用户统计: {stats}")
```

---

## AI Agent 集成

### 带记忆的对话 Agent

```python
from openai import OpenAI

client = OpenAI()

class ConversationalAgent:
    """带记忆的对话 Agent"""

    def __init__(self, memory: UnifiedMemorySystem):
        """
        初始化

        Args:
            memory: 记忆系统
        """
        self.memory = memory

    def chat(self, user_id: str, message: str) -> str:
        """
        对话

        Args:
            user_id: 用户 ID
            message: 用户消息

        Returns:
            AI 回复
        """
        # 1. 检查语义缓存
        cached_answer = self.memory.get_cached_answer(message)
        if cached_answer:
            print("✅ 使用缓存答案")
            return cached_answer

        # 2. 获取对话历史
        history = self.memory.get_conversation_history(user_id, n=10)

        # 3. 获取用户偏好
        language = self.memory.get_user_preference(user_id, "language") or "zh-CN"

        # 4. 构建上下文
        messages = [
            {"role": "system", "content": f"用户语言偏好: {language}"}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        # 5. 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        answer = response.choices[0].message.content

        # 6. 保存对话
        self.memory.add_conversation(user_id, "user", message)
        self.memory.add_conversation(user_id, "assistant", answer)

        # 7. 缓存问答
        self.memory.cache_qa(message, answer)

        return answer

# 使用示例
agent = ConversationalAgent(memory)

# 第一次对话
response1 = agent.chat("user_123", "什么是 RAG？")
print(response1)

# 第二次对话（有上下文）
response2 = agent.chat("user_123", "能举个例子吗？")
print(response2)

# 第三次对话（相似问题，使用缓存）
response3 = agent.chat("user_456", "RAG 是什么？")  # ✅ 缓存命中
print(response3)
```

---

## 性能优化

### 连接池

```python
from redis.connection import ConnectionPool

class OptimizedRedisMemory:
    """优化的 Redis 记忆系统"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        max_connections: int = 50
    ):
        """
        初始化（使用连接池）

        Args:
            host: Redis 主机
            port: Redis 端口
            max_connections: 最大连接数
        """
        self.pool = ConnectionPool(
            host=host,
            port=port,
            max_connections=max_connections,
            decode_responses=False
        )
        self.redis = redis.Redis(connection_pool=self.pool)

    def close(self):
        """关闭连接池"""
        self.pool.disconnect()
```

### Pipeline 批量操作

```python
def batch_save_messages(
    redis_client: redis.Redis,
    user_id: str,
    messages: List[Dict]
) -> None:
    """
    批量保存消息（使用 Pipeline）

    Args:
        redis_client: Redis 客户端
        user_id: 用户 ID
        messages: 消息列表
    """
    key = f"session:{user_id}:messages"

    # 使用 Pipeline 批量操作
    pipe = redis_client.pipeline()

    for msg in messages:
        pipe.lpush(key, json.dumps(msg))

    pipe.ltrim(key, 0, 99)
    pipe.expire(key, 3600)

    # 执行所有命令
    pipe.execute()

    print(f"✅ 批量保存: {len(messages)} 条消息")

# 使用示例
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "什么是 RAG？"}
]

batch_save_messages(redis_client, "user_123", messages)
```

---

## 监控和统计

### 缓存命中率监控

```python
class MonitoredMemorySystem(UnifiedMemorySystem):
    """带监控的记忆系统"""

    def __init__(self, redis_client: redis.Redis):
        super().__init__(redis_client)
        self.stats = {
            "semantic_hits": 0,
            "semantic_misses": 0,
            "session_hits": 0,
            "session_misses": 0
        }

    def get_cached_answer(self, question: str) -> Optional[str]:
        """获取缓存答案（记录统计）"""
        answer = super().get_cached_answer(question)

        if answer:
            self.stats["semantic_hits"] += 1
        else:
            self.stats["semantic_misses"] += 1

        return answer

    def get_conversation_history(
        self,
        user_id: str,
        n: int = 10
    ) -> List[Dict]:
        """获取对话历史（记录统计）"""
        history = super().get_conversation_history(user_id, n)

        if history:
            self.stats["session_hits"] += 1
        else:
            self.stats["session_misses"] += 1

        return history

    def get_stats(self) -> Dict:
        """获取统计信息"""
        semantic_total = (
            self.stats["semantic_hits"] + self.stats["semantic_misses"]
        )
        session_total = (
            self.stats["session_hits"] + self.stats["session_misses"]
        )

        return {
            **self.stats,
            "semantic_hit_rate": (
                self.stats["semantic_hits"] / semantic_total
                if semantic_total > 0 else 0
            ),
            "session_hit_rate": (
                self.stats["session_hits"] / session_total
                if session_total > 0 else 0
            )
        }

    def print_report(self) -> None:
        """打印报告"""
        stats = self.get_stats()
        print(f"\n=== 记忆系统报告 ===")
        print(f"语义缓存命中率: {stats['semantic_hit_rate']:.2%}")
        print(f"会话缓存命中率: {stats['session_hit_rate']:.2%}")
        print(f"语义缓存命中: {stats['semantic_hits']}")
        print(f"语义缓存未命中: {stats['semantic_misses']}")

# 使用示例
monitored_memory = MonitoredMemorySystem(redis_client)

# 模拟使用
for i in range(100):
    monitored_memory.get_cached_answer(f"问题 {i % 10}")

monitored_memory.print_report()
```

---

## 数据持久化

### RDB 和 AOF 配置

```python
def configure_redis_persistence(redis_client: redis.Redis) -> None:
    """
    配置 Redis 持久化

    Args:
        redis_client: Redis 客户端
    """
    # 配置 RDB（快照）
    redis_client.config_set('save', '900 1 300 10 60 10000')

    # 配置 AOF（追加文件）
    redis_client.config_set('appendonly', 'yes')
    redis_client.config_set('appendfsync', 'everysec')

    print("✅ Redis 持久化配置完成")

# 使用示例
configure_redis_persistence(redis_client)
```

### 手动保存

```python
def manual_save(redis_client: redis.Redis) -> None:
    """
    手动保存 Redis 数据

    Args:
        redis_client: Redis 客户端
    """
    # 同步保存（阻塞）
    redis_client.save()
    print("✅ 同步保存完成")

def background_save(redis_client: redis.Redis) -> None:
    """
    后台保存 Redis 数据

    Args:
        redis_client: Redis 客户端
    """
    # 异步保存（非阻塞）
    redis_client.bgsave()
    print("✅ 后台保存已启动")

# 使用示例
background_save(redis_client)
```

---

## 完整测试套件

```python
import unittest

class TestRedisMemorySystem(unittest.TestCase):
    """Redis 记忆系统测试"""

    def setUp(self):
        """测试前准备"""
        self.redis = redis.Redis(host='localhost', port=6379, db=15)
        self.redis.flushdb()  # 清空测试数据库
        self.memory = UnifiedMemorySystem(self.redis)

    def tearDown(self):
        """测试后清理"""
        self.redis.flushdb()
        self.redis.close()

    def test_session_memory(self):
        """测试会话记忆"""
        self.memory.add_conversation("test_user", "user", "你好")
        self.memory.add_conversation("test_user", "assistant", "你好！")

        history = self.memory.get_conversation_history("test_user")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "你好！")

    def test_long_term_memory(self):
        """测试长期记忆"""
        self.memory.save_user_preference("test_user", "language", "zh-CN")
        lang = self.memory.get_user_preference("test_user", "language")

        self.assertEqual(lang, "zh-CN")

    def test_semantic_cache(self):
        """测试语义缓存"""
        self.memory.cache_qa("什么是 RAG？", "RAG 是检索增强生成")
        answer = self.memory.get_cached_answer("什么是 RAG？")

        self.assertIsNotNone(answer)
        self.assertEqual(answer, "RAG 是检索增强生成")

if __name__ == "__main__":
    unittest.main()
```

---

## 总结

### 关键要点

1. **数据结构选择**：List、Hash、String
2. **过期时间**：自动清理旧数据
3. **连接池**：提升并发性能
4. **Pipeline**：批量操作
5. **持久化**：RDB + AOF

### 最佳实践

- 使用连接池管理连接
- 设置合理的过期时间
- 使用 Pipeline 批量操作
- 配置持久化策略
- 监控缓存命中率

---

## 参考资源

- [Redis Documentation](https://redis.io/docs/)
- [Redis AI Agent Memory](https://redis.io/blog/build-smarter-ai-agents-manage-short-term-and-long-term-memory-with-redis)
- [redis-py Documentation](https://redis-py.readthedocs.io/)
- [Redis Persistence](https://redis.io/docs/management/persistence/)
