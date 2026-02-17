# 核心概念 2: pymilvus 基础

深入理解 pymilvus 2.6+ SDK 的核心概念、API 设计和使用方法。

---

## 概述

pymilvus 是 Milvus 官方 Python 客户端库,提供 Pythonic API 用于连接和操作 Milvus 向量数据库。

---

## pymilvus 2.6+ 架构

### API 演进

```
pymilvus 历史版本:
├── v1.x (2019-2020)
│   └── 基础 API,功能有限
├── v2.0-2.3 (2021-2022)
│   └── connections + Collection API
└── v2.4+ (2023-2026)
    └── MilvusClient 简化 API (推荐)
```

### 两种 API 风格

**旧版 API (connections + Collection)**:

```python
from pymilvus import connections, Collection

# 连接
connections.connect(host="localhost", port="19530")

# 操作 Collection
collection = Collection("test")
collection.load()
results = collection.search(...)
```

**新版 API (MilvusClient)** (推荐):

```python
from pymilvus import MilvusClient

# 连接
client = MilvusClient(uri="http://localhost:19530")

# 操作 Collection
results = client.search(
    collection_name="test",
    data=[[0.1, 0.2, ...]],
    limit=10
)
```

**对比**:

| 特性 | 旧版 API | 新版 API (MilvusClient) |
|------|---------|------------------------|
| **连接方式** | connections.connect() | MilvusClient() |
| **代码行数** | 多 | 少 |
| **易用性** | 复杂 | 简单 |
| **功能完整性** | 完整 | 完整 |
| **推荐度** | ❌ 不推荐 | ✅ 推荐 |

---

## MilvusClient 核心 API

### 1. 连接管理

#### 基础连接

```python
from pymilvus import MilvusClient

# 最简单的连接
client = MilvusClient(uri="http://localhost:19530")
```

#### 完整连接参数

```python
client = MilvusClient(
    uri="http://localhost:19530",  # Milvus 服务地址
    token="username:password",     # 认证凭证 (可选)
    timeout=10.0,                  # 连接超时 (秒)
    pool_size=10,                  # 连接池大小
    secure=False,                  # 是否使用 TLS
    server_name="",                # TLS 服务器名称
    db_name="default"              # 数据库名称 (2.6+)
)
```

**参数详解**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `uri` | str | 必需 | Milvus 服务地址 |
| `token` | str | None | 认证凭证 (格式: username:password) |
| `timeout` | float | 10.0 | 连接超时时间 (秒) |
| `pool_size` | int | 10 | 连接池大小 |
| `secure` | bool | False | 是否使用 TLS 加密 |
| `server_name` | str | "" | TLS 服务器名称 |
| `db_name` | str | "default" | 数据库名称 |

#### 连接验证

```python
# 方法 1: 列出 Collection
collections = client.list_collections()
print(f"连接成功,Collection 数量: {len(collections)}")

# 方法 2: 捕获异常
try:
    client.list_collections()
    print("连接成功")
except Exception as e:
    print(f"连接失败: {e}")
```

### 2. Collection 管理

#### 创建 Collection

```python
# 简化创建 (自动生成 Schema)
client.create_collection(
    collection_name="quick_setup",
    dimension=128,                # 向量维度
    metric_type="COSINE",         # 相似度度量 (COSINE, L2, IP)
    auto_id=True,                 # 自动生成 ID
    primary_field_name="id",      # 主键字段名
    vector_field_name="vector"    # 向量字段名
)
```

**metric_type 说明**:

| 度量类型 | 说明 | 取值范围 | 适用场景 |
|---------|------|---------|---------|
| **COSINE** | 余弦相似度 | [-1, 1] | 文本、语义检索 |
| **L2** | 欧氏距离 | [0, ∞) | 图像、通用场景 |
| **IP** | 内积 | (-∞, ∞) | 推荐系统 |

#### 自定义 Schema 创建

```python
from pymilvus import DataType

# 定义 Schema
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 启用动态字段
)

# 添加字段
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True
)

schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=128
)

schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=512
)

# 创建 Collection
client.create_collection(
    collection_name="custom_setup",
    schema=schema
)
```

**DataType 支持**:

| 数据类型 | 说明 | Python 类型 |
|---------|------|------------|
| `INT8` | 8 位整数 | int |
| `INT16` | 16 位整数 | int |
| `INT32` | 32 位整数 | int |
| `INT64` | 64 位整数 | int |
| `FLOAT` | 32 位浮点数 | float |
| `DOUBLE` | 64 位浮点数 | float |
| `VARCHAR` | 变长字符串 | str |
| `BOOL` | 布尔值 | bool |
| `JSON` | JSON 对象 | dict |
| `ARRAY` | 数组 | list |
| `FLOAT_VECTOR` | 浮点向量 | list[float] |
| `BINARY_VECTOR` | 二进制向量 | bytes |

#### 查看 Collection

```python
# 列出所有 Collection
collections = client.list_collections()
print(collections)

# 检查 Collection 是否存在
exists = client.has_collection("test")
print(f"Collection 'test' exists: {exists}")

# 获取 Collection 详情
info = client.describe_collection("test")
print(info)
```

#### 删除 Collection

```python
# 删除 Collection
client.drop_collection("test")
```

### 3. 数据操作

#### 插入数据

```python
# 准备数据
data = [
    {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, ...],  # 128 维向量
        "text": "这是第一条数据"
    },
    {
        "id": 2,
        "vector": [0.4, 0.5, 0.6, ...],
        "text": "这是第二条数据"
    }
]

# 插入数据
result = client.insert(
    collection_name="test",
    data=data
)

print(f"插入成功,ID: {result['ids']}")
```

#### 批量插入

```python
# 批量插入 (推荐)
batch_size = 1000
data = [
    {
        "id": i,
        "vector": [random.random() for _ in range(128)],
        "text": f"数据 {i}"
    }
    for i in range(10000)
]

# 分批插入
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    client.insert(collection_name="test", data=batch)
    print(f"已插入 {i+len(batch)}/{len(data)}")
```

#### Upsert (插入或更新)

```python
# Upsert 数据
data = [
    {
        "id": 1,
        "vector": [0.7, 0.8, 0.9, ...],
        "text": "更新后的数据"
    }
]

client.upsert(
    collection_name="test",
    data=data
)
```

#### 删除数据

```python
# 按 ID 删除
client.delete(
    collection_name="test",
    ids=[1, 2, 3]
)

# 按条件删除
client.delete(
    collection_name="test",
    filter="id > 100"
)
```

### 4. 向量检索

#### 基础检索

```python
# 准备查询向量
query_vector = [0.1, 0.2, 0.3, ...]  # 128 维

# 向量检索
results = client.search(
    collection_name="test",
    data=[query_vector],
    limit=10,                    # 返回 Top-10
    output_fields=["id", "text"] # 返回字段
)

# 处理结果
for hit in results[0]:
    print(f"ID: {hit['id']}, 相似度: {hit['distance']}, 文本: {hit['entity']['text']}")
```

#### 高级检索

```python
# 带标量过滤的检索
results = client.search(
    collection_name="test",
    data=[query_vector],
    limit=10,
    filter="id > 100 and text like '%关键词%'",  # 标量过滤
    output_fields=["id", "text", "score"]
)

# 多向量检索
query_vectors = [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
]

results = client.search(
    collection_name="test",
    data=query_vectors,
    limit=10
)

# results[0] 是第一个查询向量的结果
# results[1] 是第二个查询向量的结果
```

#### 检索参数

```python
# 自定义检索参数
results = client.search(
    collection_name="test",
    data=[query_vector],
    limit=10,
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 10}  # 索引参数
    },
    output_fields=["id", "text"],
    consistency_level="Strong"  # 一致性级别
)
```

**consistency_level 说明**:

| 一致性级别 | 说明 | 延迟 | 适用场景 |
|-----------|------|------|---------|
| **Strong** | 强一致性 | 高 | 金融、交易 |
| **Bounded** | 有界一致性 | 中 | 通用场景 |
| **Eventually** | 最终一致性 | 低 | 高并发场景 |

### 5. 标量查询

#### Query (不涉及向量)

```python
# 按 ID 查询
results = client.query(
    collection_name="test",
    ids=[1, 2, 3],
    output_fields=["id", "text"]
)

# 按条件查询
results = client.query(
    collection_name="test",
    filter="id > 100 and id < 200",
    output_fields=["id", "text"],
    limit=10
)
```

### 6. 索引管理

#### 创建索引

```python
# 创建向量索引
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",           # 索引类型
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

client.create_index(
    collection_name="test",
    index_params=index_params
)
```

**常用索引类型**:

| 索引类型 | 说明 | 性能 | 内存 | 适用场景 |
|---------|------|------|------|---------|
| **FLAT** | 暴力搜索 | 慢 | 低 | 小数据集 |
| **IVF_FLAT** | 倒排索引 | 中 | 中 | 中等数据集 |
| **HNSW** | 图索引 | 快 | 高 | 大数据集 |
| **DISKANN** | 磁盘索引 | 快 | 低 | 超大数据集 |

#### 查看索引

```python
# 查看索引
indexes = client.list_indexes("test")
print(indexes)

# 查看索引详情
index_info = client.describe_index(
    collection_name="test",
    index_name="vector"
)
print(index_info)
```

#### 删除索引

```python
# 删除索引
client.drop_index(
    collection_name="test",
    index_name="vector"
)
```

---

## Embedding Functions (2.6 新特性)

### 内置向量化

pymilvus 2.6+ 支持内置 Embedding Functions,无需手动调用 Embedding 模型。

#### 使用 SentenceTransformer

```python
from pymilvus import MilvusClient
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

# 初始化 Embedding Function
ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

# 创建 Client
client = MilvusClient(uri="http://localhost:19530")

# 自动向量化并插入
data = [
    {"text": "人工智能的应用"},
    {"text": "机器学习的实践"}
]

client.insert(
    collection_name="test",
    data=data,
    embedding_function=ef  # 自动向量化
)

# 自动向量化并检索
results = client.search(
    collection_name="test",
    data=["AI 的应用场景"],
    embedding_function=ef,  # 自动向量化
    limit=10
)
```

#### 使用 OpenAI Embedding

```python
from pymilvus.model.dense import OpenAIEmbeddingFunction

# 初始化 OpenAI Embedding
ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key="your_api_key"
)

# 使用方式同上
client.insert(
    collection_name="test",
    data=data,
    embedding_function=ef
)
```

---

## 异步 API (2.6 增强)

### 异步操作

```python
import asyncio
from pymilvus import MilvusClient

async def async_search():
    client = MilvusClient(uri="http://localhost:19530")

    # 异步检索
    results = await client.search_async(
        collection_name="test",
        data=[[0.1, 0.2, ...]],
        limit=10
    )

    return results

# 运行异步任务
results = asyncio.run(async_search())
```

### 并发检索

```python
async def concurrent_search():
    client = MilvusClient(uri="http://localhost:19530")

    # 并发执行多个检索
    tasks = [
        client.search_async(
            collection_name="test",
            data=[[0.1, 0.2, ...]],
            limit=10
        )
        for _ in range(10)
    ]

    results = await asyncio.gather(*tasks)
    return results

# 运行并发任务
results = asyncio.run(concurrent_search())
```

---

## 连接池管理

### 连接池原理

```
连接池机制:

1. 创建连接池
   ├── 初始化 pool_size 个连接
   └── 连接保持活跃状态

2. 使用连接
   ├── 从池中获取空闲连接
   ├── 执行操作
   └── 归还连接到池中

3. 连接复用
   ├── 避免频繁创建/销毁连接
   └── 提升性能
```

### 连接池配置

```python
# 配置连接池
client = MilvusClient(
    uri="http://localhost:19530",
    pool_size=20,           # 连接池大小
    timeout=30.0,           # 连接超时
    max_idle_time=300       # 最大空闲时间 (秒)
)
```

### 连接池监控

```python
# 获取连接池状态 (需要自定义实现)
def get_pool_stats(client):
    # pymilvus 2.6 暂不支持直接获取连接池状态
    # 可以通过日志或监控工具查看
    pass
```

---

## 错误处理

### 常见异常

```python
from pymilvus import MilvusClient
from pymilvus.exceptions import (
    MilvusException,
    ConnectionNotExistException,
    CollectionNotExistException
)

client = MilvusClient(uri="http://localhost:19530")

try:
    # 操作 Collection
    results = client.search(
        collection_name="test",
        data=[[0.1, 0.2, ...]],
        limit=10
    )
except ConnectionNotExistException:
    print("连接不存在")
except CollectionNotExistException:
    print("Collection 不存在")
except MilvusException as e:
    print(f"Milvus 错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 重试机制

```python
import time

def retry_operation(func, max_retries=3, delay=1.0):
    """重试机制"""
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            print(f"重试 {i+1}/{max_retries}: {e}")
            time.sleep(delay)

# 使用重试机制
results = retry_operation(
    lambda: client.search(
        collection_name="test",
        data=[[0.1, 0.2, ...]],
        limit=10
    )
)
```

---

## 最佳实践

### 1. 连接管理

```python
# ✅ 推荐: 单例模式
class MilvusClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MilvusClient(uri="http://localhost:19530")
        return cls._instance

# 使用单例
client = MilvusClientSingleton.get_instance()

# ❌ 不推荐: 频繁创建客户端
for i in range(100):
    client = MilvusClient(uri="http://localhost:19530")  # 浪费资源
    client.search(...)
```

### 2. 批量操作

```python
# ✅ 推荐: 批量插入
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    client.insert(collection_name="test", data=batch)

# ❌ 不推荐: 逐条插入
for item in data:
    client.insert(collection_name="test", data=[item])  # 性能差
```

### 3. 异常处理

```python
# ✅ 推荐: 细粒度异常处理
try:
    results = client.search(...)
except CollectionNotExistException:
    # 创建 Collection
    client.create_collection(...)
    results = client.search(...)
except Exception as e:
    # 记录日志
    logger.error(f"检索失败: {e}")
    raise

# ❌ 不推荐: 忽略异常
try:
    results = client.search(...)
except:
    pass  # 忽略异常
```

### 4. 资源清理

```python
# ✅ 推荐: 使用上下文管理器
class MilvusClientContext:
    def __enter__(self):
        self.client = MilvusClient(uri="http://localhost:19530")
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        pass

with MilvusClientContext() as client:
    results = client.search(...)

# ❌ 不推荐: 不清理资源
client = MilvusClient(uri="http://localhost:19530")
results = client.search(...)
# 没有清理资源
```

---

## 总结

### 核心要点

1. **MilvusClient API**: pymilvus 2.6+ 推荐使用 MilvusClient 简化 API
2. **连接管理**: 使用单例模式,避免频繁创建客户端
3. **数据操作**: 支持插入、Upsert、删除、查询、检索
4. **Embedding Functions**: 内置向量化,简化 RAG 开发
5. **异步 API**: 支持异步操作,提升并发性能
6. **连接池**: 自动管理连接池,提升性能
7. **错误处理**: 细粒度异常处理,提升稳定性

### 下一步

- 阅读 **03_核心概念_3_健康检查.md** 学习健康检查方法
- 阅读 **07_实战代码_场景3_连接管理.md** 学习连接管理实战
- 阅读 **07_实战代码_场景4_端到端RAG.md** 学习 RAG 集成

---

**参考文献**:
- pymilvus Documentation: https://milvus.io/docs/install-pymilvus.md
- MilvusClient API: https://milvus.io/docs/connect-to-milvus-server.md
- Embedding Functions: https://milvus.io/docs/embedding-function-overview.md
