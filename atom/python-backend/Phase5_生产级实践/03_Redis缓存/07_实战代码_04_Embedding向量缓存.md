# 实战代码4：Embedding向量缓存

## 完整可运行示例

```python
"""
Embedding向量缓存实战
演示：缓存Embedding向量，降低API调用成本
"""

import redis
import hashlib
import json
import time
import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
openai_client = OpenAI()

# ===== 1. 基础Embedding缓存 =====
print("=== 基础Embedding缓存 ===\n")

class EmbeddingCache:
    """Embedding缓存"""

    def __init__(self, redis_client: redis.Redis, openai_client: OpenAI):
        self.redis = redis_client
        self.openai = openai_client
        self.stats = {"hits": 0, "misses": 0}

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """获取Embedding（带缓存）"""
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        # 查询缓存
        cached = self.redis.get(cache_key)
        if cached:
            self.stats["hits"] += 1
            print(f"✅ Embedding缓存命中")
            return json.loads(cached)

        # 调用API
        self.stats["misses"] += 1
        print(f"🤖 调用Embedding API")
        start_time = time.time()

        response = self.openai.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding
        api_time = time.time() - start_time

        print(f"⏱️ API调用耗时: {api_time:.2f}秒")

        # 缓存（24小时）
        self.redis.setex(cache_key, 86400, json.dumps(embedding))

        return embedding

# 测试
cache = EmbeddingCache(redis_client, openai_client)

texts = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Python is a programming language",  # 重复
]

for text in texts:
    print(f"\n文本: {text}")
    embedding = cache.get_embedding(text)
    print(f"向量维度: {len(embedding)}")

print(f"\n统计: 命中={cache.stats['hits']}, 未命中={cache.stats['misses']}")

# ===== 2. 批量Embedding缓存 =====
print("\n=== 批量Embedding缓存 ===\n")

class BatchEmbeddingCache:
    """批量Embedding缓存"""

    def __init__(self, redis_client: redis.Redis, openai_client: OpenAI):
        self.redis = redis_client
        self.openai = openai_client

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """批量获取Embedding"""
        results = []
        uncached_texts = []
        uncached_indices = []

        # 检查缓存
        for i, text in enumerate(texts):
            cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
            cached = self.redis.get(cache_key)

            if cached:
                results.append(json.loads(cached))
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量调用API
        if uncached_texts:
            print(f"🤖 批量调用API: {len(uncached_texts)}个文本")
            response = self.openai.embeddings.create(
                model=model,
                input=uncached_texts
            )

            # 缓存并填充结果
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                original_index = uncached_indices[i]
                text = uncached_texts[i]

                # 缓存
                cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
                self.redis.setex(cache_key, 86400, json.dumps(embedding))

                # 填充结果
                results[original_index] = embedding

        print(f"✅ 缓存命中: {len(texts) - len(uncached_texts)}/{len(texts)}")
        return results

# 测试批量缓存
batch_cache = BatchEmbeddingCache(redis_client, openai_client)

batch_texts = [
    "Python programming",
    "JavaScript development",
    "Rust systems programming",
    "Python programming",  # 重复
    "Go concurrency",
]

embeddings = batch_cache.get_embeddings_batch(batch_texts)
print(f"获取了{len(embeddings)}个向量")

# ===== 3. 压缩存储 =====
print("\n=== 压缩存储 ===\n")

class CompressedEmbeddingCache:
    """压缩Embedding缓存"""

    def __init__(self, redis_client: redis.Redis, openai_client: OpenAI):
        self.redis = redis_client
        self.openai = openai_client

    def _compress(self, embedding: List[float]) -> bytes:
        """压缩向量（float32）"""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _decompress(self, data: bytes) -> List[float]:
        """解压缩向量"""
        return np.frombuffer(data, dtype=np.float32).tolist()

    def get_embedding(self, text: str) -> List[float]:
        """获取Embedding（压缩存储）"""
        cache_key = f"emb_compressed:{hashlib.md5(text.encode()).hexdigest()}"

        cached = self.redis.get(cache_key)
        if cached:
            print("✅ 压缩缓存命中")
            return self._decompress(cached.encode('latin1'))

        print("🤖 调用API")
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding

        # 压缩并缓存
        compressed = self._compress(embedding)
        self.redis.setex(cache_key, 86400, compressed.decode('latin1'))

        # 对比大小
        json_size = len(json.dumps(embedding))
        compressed_size = len(compressed)
        print(f"💾 压缩率: {json_size / compressed_size:.2f}x")

        return embedding

# 测试压缩
compressed_cache = CompressedEmbeddingCache(redis_client, openai_client)
embedding = compressed_cache.get_embedding("Test compression")

# ===== 4. 清理 =====
print("\n=== 清理测试数据 ===")
keys = redis_client.keys("emb:*") + redis_client.keys("emb_compressed:*")
if keys:
    redis_client.delete(*keys)
    print(f"✅ 已删除{len(keys)}个缓存")
```

## 学习检查清单

- [ ] 实现基础Embedding缓存
- [ ] 实现批量Embedding缓存
- [ ] 使用压缩存储节省内存
- [ ] 理解Embedding缓存的价值（降低成本）
- [ ] 在语义缓存中应用Embedding缓存
