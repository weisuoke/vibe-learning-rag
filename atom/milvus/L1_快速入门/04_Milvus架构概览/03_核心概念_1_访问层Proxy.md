# 核心概念 1：访问层 (Proxy)

## 1. 什么是 Proxy

**定义：** Proxy 是 Milvus 的请求入口和流量管理中心，所有客户端请求都必须经过 Proxy 层。

**在架构中的位置：**
```
客户端 (pymilvus/SDK)
    ↓
【Proxy 层】← 你在这里
    ↓
协调服务层 (RootCoord/DataCoord/QueryCoord)
    ↓
工作节点层 (DataNode/QueryNode/IndexNode)
    ↓
存储层 (MinIO/S3 + etcd + Pulsar)
```

**核心特点：**
- **无状态设计**：不存储任何数据，可水平扩展
- **统一入口**：所有请求的唯一入口点
- **智能路由**：根据请求类型分发到不同后端组件
- **负载均衡**：在多个 QueryNode 间分配查询负载
- **连接管理**：维护客户端连接池

---

## 2. 核心职责（5 项）

### 2.1 请求路由 (Request Routing)

**职责：** 根据请求类型，将请求路由到正确的后端组件。

**路由规则：**
```
写入请求 (insert/delete/upsert)
    → Proxy → DataCoord → DataNode

查询请求 (search/query)
    → Proxy → QueryNode → 返回结果

管理请求 (create_collection/create_index)
    → Proxy → RootCoord

DDL 请求 (drop_collection/alter)
    → Proxy → RootCoord
```

**代码示例：**
```python
from pymilvus import connections, Collection

# 连接到 Proxy (默认端口 19530)
connections.connect(
    alias="default",
    host="localhost",
    port="19530"  # Proxy 监听端口
)

# 所有操作都通过 Proxy
collection = Collection("my_collection")

# 写入请求 → Proxy → DataCoord → DataNode
collection.insert([
    [1, 2, 3],
    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
])

# 查询请求 → Proxy → QueryNode
results = collection.search(
    data=[[0.1, 0.2]],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10
)
```

---

### 2.2 负载均衡 (Load Balancing)

**职责：** 在多个 QueryNode 之间分配查询请求，避免单点过载。

**负载均衡策略：**
1. **轮询 (Round Robin)**：按顺序分配请求
2. **最少连接 (Least Connections)**：分配给连接数最少的节点
3. **一致性哈希 (Consistent Hashing)**：根据请求特征分配

**场景示例：**
```
假设有 3 个 QueryNode：
QueryNode-1: 负载 30%
QueryNode-2: 负载 50%
QueryNode-3: 负载 20%

新查询请求到达 Proxy：
→ Proxy 选择 QueryNode-3 (负载最低)
→ 将请求转发到 QueryNode-3
→ 返回结果给客户端
```

**配置示例：**
```yaml
# milvus.yaml
proxy:
  grpc:
    serverMaxRecvSize: 536870912  # 512 MB
    serverMaxSendSize: 536870912
  timeTickInterval: 200  # ms
  msgStream:
    timeTick:
      bufSize: 512
```

---

### 2.3 连接管理 (Connection Management)

**职责：** 维护客户端连接池，管理连接生命周期。

**连接池机制：**
```python
from pymilvus import connections

# 创建连接池
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    pool_size=10,  # 连接池大小
    timeout=30     # 连接超时时间
)

# 连接池自动管理连接复用
# 多个请求共享连接池中的连接
for i in range(100):
    collection.search(...)  # 自动从连接池获取连接
```

**连接状态管理：**
```python
# 检查连接状态
print(connections.has_connection("default"))  # True

# 列出所有连接
print(connections.list_connections())  # ['default']

# 断开连接
connections.disconnect("default")

# 移除连接
connections.remove_connection("default")
```

**连接池优势：**
- **减少开销**：避免频繁创建/销毁连接
- **提高性能**：连接复用提升吞吐量
- **资源控制**：限制并发连接数

---

### 2.4 请求验证 (Request Validation)

**职责：** 验证请求格式、参数合法性、权限检查。

**验证内容：**
1. **格式验证**：检查请求格式是否符合 API 规范
2. **参数验证**：检查参数范围、类型是否合法
3. **权限验证**：检查用户是否有操作权限
4. **资源验证**：检查 Collection/Partition 是否存在

**验证示例：**
```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

# 1. 格式验证：字段类型必须匹配
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128)
])

collection = Collection("test", schema)

# ❌ 错误：维度不匹配 (Proxy 会拒绝)
try:
    collection.insert([
        [1, 2, 3],
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # 维度 2，期望 128
    ])
except Exception as e:
    print(f"Proxy 验证失败: {e}")
    # 输出: dimension mismatch: expected 128, got 2

# ✅ 正确：维度匹配
collection.insert([
    [1, 2, 3],
    [[0.1] * 128, [0.2] * 128, [0.3] * 128]
])

# 2. 参数验证：limit 必须 > 0
try:
    collection.search(
        data=[[0.1] * 128],
        anns_field="embedding",
        param={"metric_type": "L2"},
        limit=0  # ❌ 非法参数
    )
except Exception as e:
    print(f"Proxy 验证失败: {e}")

# 3. 资源验证：Collection 必须存在
try:
    Collection("non_existent_collection")
except Exception as e:
    print(f"Proxy 验证失败: {e}")
```

---

### 2.5 结果聚合 (Result Aggregation)

**职责：** 合并来自多个 QueryNode 的查询结果，返回统一结果。

**聚合场景：**
```
查询请求: Top-10 相似向量

Proxy 将请求分发到 3 个 QueryNode：
QueryNode-1 返回: [id=1(0.1), id=5(0.3), id=9(0.5), ...]
QueryNode-2 返回: [id=2(0.15), id=6(0.35), id=10(0.55), ...]
QueryNode-3 返回: [id=3(0.2), id=7(0.4), id=11(0.6), ...]

Proxy 聚合结果:
1. 合并所有结果 (30 个)
2. 按距离排序
3. 取 Top-10
4. 返回给客户端: [id=1(0.1), id=2(0.15), id=3(0.2), ...]
```

**聚合算法：**
```python
# 伪代码：Proxy 内部聚合逻辑
def aggregate_search_results(results_from_nodes, limit):
    """
    聚合多个 QueryNode 的搜索结果
    """
    all_results = []

    # 1. 合并所有结果
    for node_results in results_from_nodes:
        all_results.extend(node_results)

    # 2. 按距离排序
    all_results.sort(key=lambda x: x.distance)

    # 3. 取 Top-K
    return all_results[:limit]

# 客户端视角：透明的聚合过程
results = collection.search(
    data=[[0.1] * 128],
    anns_field="embedding",
    param={"metric_type": "L2"},
    limit=10  # Proxy 自动聚合多个 QueryNode 的结果
)

# 结果已经是聚合后的 Top-10
for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance}")
```

---

## 3. 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端层                              │
│  pymilvus SDK / Java SDK / Go SDK / RESTful API             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      【Proxy 层】                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Proxy-1    │  │  Proxy-2    │  │  Proxy-3    │         │
│  │  (无状态)   │  │  (无状态)   │  │  (无状态)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
│  核心功能：                                                   │
│  • 请求路由      • 负载均衡      • 连接管理                  │
│  • 请求验证      • 结果聚合                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RootCoord   │  │  DataCoord   │  │ QueryCoord   │
│  (元数据管理) │  │  (数据协调)  │  │  (查询协调)  │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                   ↓                   ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   etcd       │  │  DataNode    │  │  QueryNode   │
│  (元数据存储) │  │  (数据写入)  │  │  (数据查询)  │
└──────────────┘  └──────────────┘  └──────────────┘
                        ↓                   ↓
                ┌──────────────┐  ┌──────────────┐
                │  MinIO/S3    │  │  IndexNode   │
                │  (对象存储)  │  │  (索引构建)  │
                └──────────────┘  └──────────────┘
```

---

## 4. 请求流程详解

### 4.1 写入请求流程

```
客户端: collection.insert(data)
    ↓
Proxy:
    1. 验证请求格式
    2. 验证 Collection 是否存在
    3. 验证数据维度是否匹配
    4. 生成时间戳 (Timestamp)
    5. 将请求发送到 DataCoord
    ↓
DataCoord:
    1. 分配 Segment
    2. 选择 DataNode
    3. 返回写入位置给 Proxy
    ↓
Proxy:
    1. 将数据发送到指定 DataNode
    2. 等待写入确认
    3. 返回成功响应给客户端
    ↓
客户端: 收到插入成功响应
```

**代码示例：**
```python
from pymilvus import Collection
import time

collection = Collection("my_collection")

# 写入流程
start = time.time()
mr = collection.insert([
    [1, 2, 3, 4, 5],
    [[0.1] * 128, [0.2] * 128, [0.3] * 128, [0.4] * 128, [0.5] * 128]
])
print(f"插入耗时: {time.time() - start:.3f}s")
print(f"插入 ID: {mr.primary_keys}")

# Proxy 自动处理：
# 1. 验证维度 (128)
# 2. 联系 DataCoord 分配 Segment
# 3. 转发数据到 DataNode
# 4. 返回插入的主键 ID
```

---

### 4.2 查询请求流程

```
客户端: collection.search(query_vector)
    ↓
Proxy:
    1. 验证请求参数
    2. 验证 Collection 是否已加载
    3. 从 QueryCoord 获取 QueryNode 列表
    4. 根据负载均衡策略选择 QueryNode
    5. 将查询请求分发到多个 QueryNode
    ↓
QueryNode-1, QueryNode-2, QueryNode-3:
    1. 在内存中执行向量搜索
    2. 返回 Top-K 结果给 Proxy
    ↓
Proxy:
    1. 聚合所有 QueryNode 的结果
    2. 全局排序取 Top-K
    3. 返回最终结果给客户端
    ↓
客户端: 收到搜索结果
```

**代码示例：**
```python
import time

# 查询流程
start = time.time()
results = collection.search(
    data=[[0.1] * 128],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10
)
print(f"查询耗时: {time.time() - start:.3f}s")

# Proxy 自动处理：
# 1. 验证 Collection 已加载
# 2. 分发请求到多个 QueryNode
# 3. 聚合结果并排序
# 4. 返回 Top-10

for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance:.4f}")
```

---

### 4.3 管理请求流程

```
客户端: create_collection(schema)
    ↓
Proxy:
    1. 验证 Schema 格式
    2. 验证 Collection 名称是否合法
    3. 将请求转发到 RootCoord
    ↓
RootCoord:
    1. 检查 Collection 是否已存在
    2. 分配 Collection ID
    3. 将元数据写入 etcd
    4. 返回成功响应给 Proxy
    ↓
Proxy:
    1. 返回成功响应给客户端
    ↓
客户端: 收到创建成功响应
```

**代码示例：**
```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

# 管理请求：创建 Collection
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128)
])

collection = Collection("my_new_collection", schema)

# Proxy 自动处理：
# 1. 验证 Schema 格式
# 2. 转发到 RootCoord
# 3. RootCoord 写入元数据到 etcd
# 4. 返回成功响应

print(f"Collection 创建成功: {collection.name}")
```

---

## 5. 在 RAG 中的应用

### 5.1 高并发场景

**场景：** 多用户同时查询知识库

```python
from pymilvus import connections, Collection
import concurrent.futures

# 连接到 Proxy
connections.connect(host="localhost", port="19530")
collection = Collection("knowledge_base")

def search_knowledge(query_text, query_embedding):
    """单次知识库查询"""
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=5,
        output_fields=["text"]
    )
    return results[0]

# 模拟 100 个并发查询
queries = [f"Query {i}" for i in range(100)]
embeddings = [[0.1] * 128 for _ in range(100)]

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(search_knowledge, q, e)
        for q, e in zip(queries, embeddings)
    ]

    results = [f.result() for f in futures]

# Proxy 自动处理：
# 1. 管理 20 个并发连接
# 2. 负载均衡到多个 QueryNode
# 3. 聚合结果返回
```

---

### 5.2 连接池管理

**场景：** RAG 服务长期运行，需要高效连接管理

```python
from pymilvus import connections, Collection
from contextlib import contextmanager

class MilvusConnectionPool:
    """Milvus 连接池管理器"""

    def __init__(self, host="localhost", port="19530", pool_size=10):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self._initialized = False

    def initialize(self):
        """初始化连接池"""
        if not self._initialized:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                pool_size=self.pool_size
            )
            self._initialized = True

    @contextmanager
    def get_collection(self, name):
        """获取 Collection（自动使用连接池）"""
        self.initialize()
        collection = Collection(name)
        try:
            yield collection
        finally:
            pass  # 连接自动归还到池中

# 使用连接池
pool = MilvusConnectionPool(pool_size=20)

# RAG 查询函数
def rag_search(query_embedding):
    with pool.get_collection("knowledge_base") as collection:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP"},
            limit=5
        )
        return results[0]

# 多次查询自动复用连接
for i in range(1000):
    results = rag_search([0.1] * 128)
```

---

### 5.3 请求限流

**场景：** 防止恶意请求或突发流量压垮系统

```python
from pymilvus import Collection
import time
from collections import deque

class RateLimiter:
    """简单的令牌桶限流器"""

    def __init__(self, rate=10, capacity=20):
        """
        rate: 每秒生成令牌数
        capacity: 令牌桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    def acquire(self):
        """获取令牌（阻塞直到获取成功）"""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            time.sleep(0.1)

# 使用限流器
limiter = RateLimiter(rate=10, capacity=20)  # 每秒最多 10 次查询
collection = Collection("knowledge_base")

def rate_limited_search(query_embedding):
    """限流的搜索函数"""
    limiter.acquire()  # 获取令牌

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP"},
        limit=5
    )
    return results[0]

# 即使有大量请求，也会被限流到每秒 10 次
for i in range(100):
    results = rate_limited_search([0.1] * 128)
    print(f"Query {i} completed")
```

---

## 6. 性能考虑

### 6.1 Proxy 是无状态的

**特点：**
- Proxy 不存储任何数据
- 所有状态信息都在 etcd 或其他组件中
- 可以随时启动/停止 Proxy 实例

**优势：**
- **水平扩展**：可以部署多个 Proxy 实例
- **高可用**：单个 Proxy 故障不影响系统
- **负载均衡**：客户端可以连接到任意 Proxy

---

### 6.2 生产环境建议

**部署建议：**
```yaml
# 生产环境配置
proxy:
  replicas: 3  # 部署 3 个 Proxy 实例
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"
    requests:
      cpu: "2"
      memory: "4Gi"

  # 负载均衡配置
  service:
    type: LoadBalancer
    port: 19530
```

**监控指标：**
```python
# 关键监控指标
metrics = {
    "proxy_request_rate": "每秒请求数",
    "proxy_request_latency": "请求延迟 (P50/P95/P99)",
    "proxy_connection_count": "活跃连接数",
    "proxy_error_rate": "错误率",
    "proxy_throughput": "吞吐量 (MB/s)"
}
```

**性能优化：**
```python
# 1. 调整连接池大小
connections.connect(
    host="localhost",
    port="19530",
    pool_size=50  # 根据并发量调整
)

# 2. 批量操作减少请求次数
collection.insert([
    list(range(10000)),  # 一次插入 10000 条
    [[0.1] * 128 for _ in range(10000)]
])

# 3. 异步查询提高吞吐量
import asyncio
from pymilvus import Collection

async def async_search(collection, embedding):
    return collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP"},
        limit=5
    )

# 并发执行多个查询
async def batch_search():
    collection = Collection("knowledge_base")
    tasks = [
        async_search(collection, [0.1] * 128)
        for _ in range(100)
    ]
    results = await asyncio.gather(*tasks)
    return results

# asyncio.run(batch_search())
```

---

## 7. 常见问题

### Q1: Proxy 挂了怎么办？

**答：**
- 如果只有一个 Proxy：服务不可用，需要重启
- 如果有多个 Proxy：客户端自动连接到其他 Proxy
- 生产环境建议：部署至少 3 个 Proxy 实例

### Q2: Proxy 会成为性能瓶颈吗？

**答：**
- Proxy 本身很轻量，主要做转发和聚合
- 如果成为瓶颈，可以水平扩展 Proxy 实例
- 监控 Proxy 的 CPU/内存/网络使用率

### Q3: 如何连接到多个 Proxy？

**答：**
```python
# 方式 1：使用负载均衡器
connections.connect(
    host="load-balancer.example.com",  # 负载均衡器地址
    port="19530"
)

# 方式 2：客户端轮询
proxies = [
    ("proxy1.example.com", "19530"),
    ("proxy2.example.com", "19530"),
    ("proxy3.example.com", "19530")
]

import random
host, port = random.choice(proxies)
connections.connect(host=host, port=port)
```

---

## 8. 总结

**Proxy 的核心价值：**
1. **统一入口**：所有请求的唯一入口点
2. **智能路由**：根据请求类型分发到正确组件
3. **负载均衡**：在多个 QueryNode 间分配负载
4. **连接管理**：高效管理客户端连接
5. **结果聚合**：合并多个节点的查询结果

**在 RAG 中的应用：**
- 高并发查询场景
- 连接池管理
- 请求限流和熔断

**生产环境建议：**
- 部署多个 Proxy 实例（至少 3 个）
- 使用负载均衡器分发流量
- 监控 Proxy 性能指标
- 根据负载动态扩缩容

---

**下一步：** 学习协调服务层 (RootCoord/DataCoord/QueryCoord) 的职责和工作原理。
