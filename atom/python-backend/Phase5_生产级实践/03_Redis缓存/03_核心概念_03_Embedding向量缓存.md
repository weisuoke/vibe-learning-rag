# 核心概念3：Embedding向量缓存

## 概述

Embedding API调用虽然比LLM便宜，但在高频场景下仍会产生可观成本。缓存Embedding向量可以显著降低成本和延迟。

---

## 1. 为什么需要缓存Embedding？

### 成本分析

```python
# Embedding API成本（OpenAI text-embedding-3-small）
# 价格：$0.00002 / 1K tokens

# 场景：语义缓存系统，每天10000个查询
daily_queries = 10000
avg_tokens_per_query = 50  # 平均每个查询50 tokens

# 没有缓存的成本
cost_per_query = (avg_tokens_per_query / 1000) * 0.00002
daily_cost_no_cache = daily_queries * cost_per_query
monthly_cost_no_cache = daily_cost_no_cache * 30

print(f"每日成本（无缓存）: ${daily_cost_no_cache:.4f}")
print(f"每月成本（无缓存）: ${monthly_cost_no_cache:.2f}")

# 有缓存的成本（假设80%命中率）
hit_rate = 0.8
api_calls_with_cache = daily_queries * (1 - hit_rate)
daily_cost_with_cache = api_calls_with_cache * cost_per_query
monthly_cost_with_cache = daily_cost_with_cache * 30

print(f"每日成本（有缓存）: ${daily_cost_with_cache:.4f}")
print(f"每月成本（有缓存）: ${monthly_cost_with_cache:.2f}")
print(f"每月节省: ${monthly_cost_no_cache - monthly_cost_with_cache:.2f}")
```

**输出：**
```
每日成本（无缓存）: $0.0100
每月成本（无缓存）: $0.30
每日成本（有缓存）: $0.0020
每月成本（有缓存）: $0.06
每月节省: $0.24
```

---

## 2. Embedding缓存实现

### 基础实现

```python
import hashlib
import json
from typing import List, Optional
import redis
from openai import OpenAI

class EmbeddingCache:
    """Embedding向量缓存"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.openai_client = OpenAI()

    def _generate_cache_key(self, text: str, model: str) -> str:
        """生成缓存key"""
        content = f"{model}:{text}"
        return f"emb:{hashlib.md5(content.encode()).hexdigest()}"

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """获取Embedding（带缓存）"""
        cache_key = self._generate_cache_key(text, model)

        # 1. 查询缓存
        cached = self.redis.get(cache_key)
        if cached:
            print(f"✅ Embedding缓存命中")
            return json.loads(cached)

        # 2. 调用API
        print(f"🤖 调用Embedding API")
        response = self.openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding

        # 3. 缓存结果（24小时）
        self.redis.setex(cache_key, 86400, json.dumps(embedding))

        return embedding

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """批量获取Embedding"""
        results = []
        uncached_texts = []
        uncached_indices = []

        # 1. 检查缓存
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, model)
            cached = self.redis.get(cache_key)

            if cached:
                results.append(json.loads(cached))
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 2. 批量调用API（未命中的）
        if uncached_texts:
            print(f"🤖 批量调用API: {len(uncached_texts)}个文本")
            response = self.openai_client.embeddings.create(
                model=model,
                input=uncached_texts
            )

            # 3. 缓存并填充结果
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                original_index = uncached_indices[i]
                text = uncached_texts[i]

                # 缓存
                cache_key = self._generate_cache_key(text, model)
                self.redis.setex(cache_key, 86400, json.dumps(embedding))

                # 填充结果
                results[original_index] = embedding

        print(f"✅ 缓存命中: {len(texts) - len(uncached_texts)}/{len(texts)}")
        return results

# 使用示例
cache = EmbeddingCache(redis_client)

# 单个文本
embedding = cache.get_embedding("Python is a programming language")

# 批量文本
texts = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Rust is a systems programming language"
]
embeddings = cache.get_embeddings_batch(texts)
```

---

## 3. 使用Hash存储向量

### 为什么使用Hash？

```python
# String方式：每个向量一个key
redis_client.setex("emb:abc123", 86400, json.dumps(embedding))
# 问题：大量key，内存开销大

# Hash方式：所有向量存在一个Hash中
redis_client.hset("embeddings", "abc123", json.dumps(embedding))
# 优点：减少key数量，降低内存开销
```

### Hash实现

```python
class HashEmbeddingCache:
    """使用Hash存储Embedding"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.openai_client = OpenAI()
        self.hash_key = "embeddings"  # 统一的Hash key

    def _generate_field_name(self, text: str, model: str) -> str:
        """生成Hash字段名"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """获取Embedding"""
        field_name = self._generate_field_name(text, model)

        # 1. 查询Hash
        cached = self.redis.hget(self.hash_key, field_name)
        if cached:
            print(f"✅ Hash缓存命中")
            return json.loads(cached)

        # 2. 调用API
        print(f"🤖 调用Embedding API")
        response = self.openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding

        # 3. 存入Hash
        self.redis.hset(self.hash_key, field_name, json.dumps(embedding))

        # 4. 设置Hash的过期时间（可选）
        # 注意：Hash的TTL是整个Hash，不是单个字段
        self.redis.expire(self.hash_key, 86400)

        return embedding

    def get_cache_size(self) -> int:
        """获取缓存的向量数量"""
        return self.redis.hlen(self.hash_key)

    def clear_cache(self):
        """清空缓存"""
        self.redis.delete(self.hash_key)
```

---

## 4. 内存优化：压缩存储

### 向量压缩

```python
import numpy as np
import struct

class CompressedEmbeddingCache:
    """压缩存储Embedding"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.openai_client = OpenAI()

    def _compress_embedding(self, embedding: List[float]) -> bytes:
        """压缩向量（float32）"""
        # 将float64转为float32，减少50%内存
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()

    def _decompress_embedding(self, data: bytes) -> List[float]:
        """解压缩向量"""
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.tolist()

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """获取Embedding（压缩存储）"""
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        # 1. 查询缓存
        cached = self.redis.get(cache_key)
        if cached:
            print(f"✅ 缓存命中（压缩）")
            return self._decompress_embedding(cached)

        # 2. 调用API
        print(f"🤖 调用Embedding API")
        response = self.openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding

        # 3. 压缩并缓存
        compressed = self._compress_embedding(embedding)
        self.redis.setex(cache_key, 86400, compressed)

        print(f"💾 压缩率: {len(json.dumps(embedding)) / len(compressed):.2f}x")

        return embedding

# 内存对比
embedding = [0.1] * 1536  # text-embedding-3-small的维度

# JSON存储
json_size = len(json.dumps(embedding))
print(f"JSON大小: {json_size} bytes")  # ~12KB

# 压缩存储（float32）
compressed_size = len(np.array(embedding, dtype=np.float32).tobytes())
print(f"压缩大小: {compressed_size} bytes")  # ~6KB

print(f"压缩率: {json_size / compressed_size:.2f}x")  # ~2x
```

---

## 5. 混合缓存：内存 + Redis

### 两层缓存

```python
from collections import OrderedDict

class TieredEmbeddingCache:
    """两层缓存：内存 + Redis"""

    def __init__(self, redis_client: redis.Redis, memory_size: int = 1000):
        self.redis = redis_client
        self.openai_client = OpenAI()
        self.memory_cache = OrderedDict()  # LRU缓存
        self.memory_size = memory_size

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """两层缓存获取"""
        cache_key = f"{model}:{text}"

        # 1. 查询内存缓存（最快）
        if cache_key in self.memory_cache:
            print(f"🎯 内存缓存命中")
            self.memory_cache.move_to_end(cache_key)  # LRU更新
            return self.memory_cache[cache_key]

        # 2. 查询Redis缓存
        redis_key = f"emb:{hashlib.md5(cache_key.encode()).hexdigest()}"
        cached = self.redis.get(redis_key)
        if cached:
            print(f"✅ Redis缓存命中")
            embedding = json.loads(cached)
            self._set_memory_cache(cache_key, embedding)
            return embedding

        # 3. 调用API
        print(f"🤖 调用Embedding API")
        response = self.openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding

        # 4. 同时缓存到内存和Redis
        self._set_memory_cache(cache_key, embedding)
        self.redis.setex(redis_key, 86400, json.dumps(embedding))

        return embedding

    def _set_memory_cache(self, key: str, embedding: List[float]):
        """设置内存缓存（LRU淘汰）"""
        if len(self.memory_cache) >= self.memory_size:
            self.memory_cache.popitem(last=False)  # 删除最旧的
        self.memory_cache[key] = embedding
```

---

## 6. 在语义缓存中的应用

### 完整示例

```python
class SemanticCache:
    """语义缓存（使用Embedding缓存）"""

    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_cache: EmbeddingCache
    ):
        self.redis = redis_client
        self.embedding_cache = embedding_cache

    def add_cache(
        self,
        query: str,
        response: str,
        ttl: int = 3600
    ):
        """添加语义缓存"""
        # 1. 获取query的Embedding（带缓存）
        query_embedding = self.embedding_cache.get_embedding(query)

        # 2. 存储到Redis Hash
        cache_data = {
            "query": query,
            "response": response,
            "embedding": query_embedding
        }

        cache_id = hashlib.md5(query.encode()).hexdigest()
        self.redis.hset(
            "semantic_cache",
            cache_id,
            json.dumps(cache_data)
        )
        self.redis.expire("semantic_cache", ttl)

    def lookup(
        self,
        query: str,
        threshold: float = 0.9
    ) -> Optional[str]:
        """查询语义缓存"""
        # 1. 获取query的Embedding（带缓存）
        query_embedding = self.embedding_cache.get_embedding(query)

        # 2. 获取所有缓存
        cached_items = self.redis.hgetall("semantic_cache")

        # 3. 计算相似度
        best_match = None
        best_score = 0.0

        for cache_id, cache_data_json in cached_items.items():
            cache_data = json.loads(cache_data_json)
            cached_embedding = cache_data["embedding"]

            similarity = self._cosine_similarity(
                query_embedding,
                cached_embedding
            )

            if similarity > best_score:
                best_score = similarity
                best_match = cache_data

        # 4. 判断是否命中
        if best_score >= threshold:
            print(f"🎯 语义缓存命中，相似度={best_score:.3f}")
            return best_match["response"]

        print(f"❌ 语义缓存未命中，最高相似度={best_score:.3f}")
        return None

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """计算余弦相似度"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 总结

1. **基础缓存**：使用String类型存储JSON格式的向量
2. **Hash存储**：减少key数量，降低内存开销
3. **压缩存储**：使用float32代替float64，节省50%内存
4. **两层缓存**：内存 + Redis，提升查询速度
5. **批量处理**：批量获取Embedding，减少API调用次数
6. **语义缓存集成**：Embedding缓存是语义缓存的基础

**记住：** Embedding缓存的TTL可以设置较长（24小时），因为相同文本的Embedding是确定的。
