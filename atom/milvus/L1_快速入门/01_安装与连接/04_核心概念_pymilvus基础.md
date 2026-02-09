# 04_核心概念_pymilvus基础

> 深入理解 pymilvus SDK 的使用方法和最佳实践

---

## 1. pymilvus SDK 概述

### 1.1 什么是 pymilvus

**pymilvus 是 Milvus 的官方 Python 客户端库**，提供了与 Milvus 服务器交互的完整 API。

```python
from pymilvus import connections, Collection

# pymilvus 的本质：gRPC 客户端封装
# 它不存储数据，只是发送请求到 Milvus 服务器
```

**核心功能：**

| 功能模块 | 说明 | 主要类/函数 |
|----------|------|-------------|
| **连接管理** | 建立和管理与 Milvus 的连接 | `connections` |
| **Collection 操作** | 创建、删除、查询 Collection | `Collection`, `CollectionSchema` |
| **数据操作** | 插入、删除、查询向量数据 | `collection.insert()`, `collection.search()` |
| **索引管理** | 创建和管理向量索引 | `collection.create_index()` |
| **分区管理** | 创建和管理分区 | `Partition` |
| **工具函数** | 服务器信息、健康检查等 | `utility` |

---

### 1.2 安装方式

**使用 uv（推荐）：**

```bash
# 安装 pymilvus
uv add pymilvus

# 查看已安装版本
uv pip list | grep pymilvus
```

**使用 pip：**

```bash
# 安装最新版本
pip install pymilvus

# 安装指定版本
pip install pymilvus==2.4.0

# 升级到最新版本
pip install --upgrade pymilvus
```

**验证安装：**

```python
import pymilvus

print(f"pymilvus 版本: {pymilvus.__version__}")
# 输出示例：pymilvus 版本: 2.4.0
```

---

### 1.3 版本兼容性

**pymilvus 与 Milvus 服务器的版本对应关系：**

| pymilvus 版本 | Milvus 服务器版本 | 兼容性 |
|---------------|-------------------|--------|
| 2.4.x | 2.4.x | ✅ 完全兼容 |
| 2.4.x | 2.3.x | ✅ 向后兼容 |
| 2.3.x | 2.4.x | ⚠️ 部分功能不可用 |
| 2.3.x | 2.3.x | ✅ 完全兼容 |
| 2.2.x | 2.3.x+ | ❌ 不兼容 |

**最佳实践：**
- 保持 pymilvus 和 Milvus 服务器版本一致
- 升级 Milvus 服务器后，同步升级 pymilvus
- 使用 `utility.get_server_version()` 检查服务器版本

```python
from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")
server_version = utility.get_server_version()
print(f"Milvus 服务器版本: {server_version}")
```

---

## 2. 连接管理

### 2.1 connections.connect() 详解

**基本用法：**

```python
from pymilvus import connections

# 建立连接
connections.connect(
    alias="default",           # 连接别名
    host="localhost",          # Milvus 服务器地址
    port="19530"               # gRPC 端口
)

print("✅ 连接成功！")
```

**完整参数列表：**

```python
connections.connect(
    alias="default",           # 连接别名（必需）
    host="localhost",          # 服务器地址（必需）
    port="19530",              # 端口（必需）
    user="",                   # 用户名（可选，启用认证时需要）
    password="",               # 密码（可选，启用认证时需要）
    db_name="",                # 数据库名称（可选，默认为 default）
    token="",                  # Token 认证（可选）
    timeout=None,              # 连接超时（秒）
    secure=False,              # 是否使用 TLS/SSL
    server_pem_path="",        # TLS 证书路径
    server_name="",            # TLS 服务器名称
    client_pem_path="",        # 客户端证书路径
    client_key_path="",        # 客户端密钥路径
    ca_pem_path=""             # CA 证书路径
)
```

---

### 2.2 连接参数详解

#### 2.2.1 alias（连接别名）

**作用：** 为连接指定一个唯一标识符，用于在多个连接之间切换。

```python
# 创建多个连接
connections.connect("dev", host="localhost", port="19530")
connections.connect("prod", host="milvus.prod.com", port="19530")

# 使用不同的连接
from pymilvus import Collection

# 使用 dev 连接
collection_dev = Collection("my_collection", using="dev")

# 使用 prod 连接
collection_prod = Collection("my_collection", using="prod")
```

**默认别名：**
- 如果不指定 `using` 参数，默认使用 `"default"` 别名

---

#### 2.2.2 host 和 port

**host：** Milvus 服务器的地址

```python
# 本地开发
connections.connect("default", host="localhost", port="19530")

# 远程服务器
connections.connect("default", host="192.168.1.100", port="19530")

# 域名
connections.connect("default", host="milvus.example.com", port="19530")
```

**port：** gRPC 端口（默认 19530）

```python
# 标准端口
connections.connect("default", host="localhost", port="19530")

# 自定义端口（如果 Docker 映射到其他端口）
connections.connect("default", host="localhost", port="19531")
```

---

#### 2.2.3 user 和 password（认证）

**启用认证后的连接：**

```python
# Milvus 启用认证后，需要提供用户名和密码
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    user="admin",
    password="your_password"
)
```

**注意：**
- Milvus Standalone 默认不启用认证
- 生产环境建议启用认证
- 密码不要硬编码在代码中，使用环境变量

```python
import os
from dotenv import load_dotenv

load_dotenv()

connections.connect(
    alias="default",
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=os.getenv("MILVUS_PORT", "19530"),
    user=os.getenv("MILVUS_USER", ""),
    password=os.getenv("MILVUS_PASSWORD", "")
)
```

---

#### 2.2.4 timeout（超时设置）

**作用：** 设置连接超时时间（秒）

```python
# 设置 10 秒超时
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    timeout=10
)
```

**使用场景：**
- 网络不稳定时增加超时时间
- 快速失败场景减少超时时间

---

### 2.3 连接别名（alias）机制

**为什么需要连接别名？**

在实际应用中，可能需要连接到多个 Milvus 实例：
- 开发环境、测试环境、生产环境
- 不同的数据中心
- 主从复制的多个节点

**多连接示例：**

```python
from pymilvus import connections, Collection

# 连接到开发环境
connections.connect(
    alias="dev",
    host="localhost",
    port="19530"
)

# 连接到生产环境
connections.connect(
    alias="prod",
    host="milvus.prod.com",
    port="19530",
    user="admin",
    password="prod_password"
)

# 在开发环境操作
dev_collection = Collection("test_collection", using="dev")
print(f"Dev 环境数据量: {dev_collection.num_entities}")

# 在生产环境操作
prod_collection = Collection("prod_collection", using="prod")
print(f"Prod 环境数据量: {prod_collection.num_entities}")

# 断开连接
connections.disconnect("dev")
connections.disconnect("prod")
```

---

### 2.4 断开连接与资源释放

**为什么需要断开连接？**

```python
# 每个连接占用资源：
# - 客户端：TCP socket + 内存缓冲区
# - 服务器：连接状态 + 线程/协程
```

**断开单个连接：**

```python
from pymilvus import connections

# 建立连接
connections.connect("default", host="localhost", port="19530")

# 使用连接...

# 断开连接
connections.disconnect("default")
```

**断开所有连接：**

```python
# 断开所有连接
connections.disconnect("default")
connections.disconnect("dev")
connections.disconnect("prod")

# 或者使用循环
for alias in ["default", "dev", "prod"]:
    if connections.has_connection(alias):
        connections.disconnect(alias)
```

**检查连接状态：**

```python
# 检查连接是否存在
if connections.has_connection("default"):
    print("连接存在")
else:
    print("连接不存在")

# 列出所有连接
all_connections = connections.list_connections()
print(f"所有连接: {all_connections}")
```

---

## 3. 客户端对象模型

### 3.1 Collection 对象

**Collection 是 Milvus 中最核心的概念**，类似于关系型数据库中的表。

**创建 Collection：**

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 1. 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]

# 2. 创建 Schema
schema = CollectionSchema(fields, description="My first collection")

# 3. 创建 Collection
collection = Collection(name="my_collection", schema=schema)

print(f"Collection 创建成功: {collection.name}")
```

**获取已存在的 Collection：**

```python
# 方式1：直接创建对象
collection = Collection("my_collection")

# 方式2：检查是否存在
from pymilvus import utility

if utility.has_collection("my_collection"):
    collection = Collection("my_collection")
    print(f"Collection 存在，数据量: {collection.num_entities}")
else:
    print("Collection 不存在")
```

---

### 3.2 Partition 对象

**Partition 是 Collection 的逻辑分区**，用于数据隔离和查询优化。

**创建 Partition：**

```python
from pymilvus import Collection, Partition

collection = Collection("my_collection")

# 创建分区
partition = collection.create_partition("partition_2024")

print(f"Partition 创建成功: {partition.name}")
```

**使用 Partition：**

```python
# 获取 Partition 对象
partition = Partition(collection, "partition_2024")

# 插入数据到分区
data = [
    [1, 2, 3],  # id
    ["text1", "text2", "text3"],  # text
    [[0.1]*128, [0.2]*128, [0.3]*128]  # embedding
]
partition.insert(data)

# 在分区中检索
results = partition.search(
    data=[[0.1]*128],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10
)
```

---

### 3.3 Index 对象

**Index 是加速向量检索的关键**。

**创建索引：**

```python
from pymilvus import Collection

collection = Collection("my_collection")

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print("索引创建成功")
```

**查看索引信息：**

```python
# 获取索引信息
index = collection.index()
print(f"索引类型: {index.params['index_type']}")
print(f"度量类型: {index.params['metric_type']}")
```

---

### 3.4 对象生命周期管理

**Collection 的生命周期：**

```python
from pymilvus import Collection, utility

# 1. 创建 Collection
collection = Collection("my_collection", schema=schema)

# 2. 使用 Collection
collection.insert(data)
collection.create_index("embedding", index_params)
collection.load()
results = collection.search(...)

# 3. 释放 Collection（从内存中卸载）
collection.release()

# 4. 删除 Collection
utility.drop_collection("my_collection")
```

**load 和 release 的作用：**

```python
# load：将 Collection 加载到内存，才能执行检索
collection.load()

# 检索操作
results = collection.search(...)

# release：从内存中卸载，释放资源
collection.release()

# 注意：release 后无法检索，需要重新 load
```

---

## 4. 异常处理

### 4.1 常见异常类型

**pymilvus 的异常层次结构：**

```python
from pymilvus import MilvusException

# 所有 pymilvus 异常的基类
try:
    connections.connect("default", host="invalid_host", port="19530")
except MilvusException as e:
    print(f"Milvus 异常: {e}")
```

**常见异常：**

| 异常类型 | 说明 | 常见原因 |
|----------|------|----------|
| `MilvusException` | 基础异常类 | 所有 Milvus 相关错误 |
| `ConnectionNotExistException` | 连接不存在 | 未建立连接或连接已断开 |
| `CollectionNotExistException` | Collection 不存在 | Collection 未创建 |
| `PartitionNotExistException` | Partition 不存在 | Partition 未创建 |
| `IndexNotExistException` | 索引不存在 | 索引未创建 |
| `ParamError` | 参数错误 | 参数类型或值不正确 |

---

### 4.2 连接超时处理

**连接超时示例：**

```python
from pymilvus import connections, MilvusException
import time

try:
    # 设置较短的超时时间
    connections.connect(
        alias="default",
        host="192.168.1.100",  # 假设这是一个慢速服务器
        port="19530",
        timeout=5  # 5 秒超时
    )
except MilvusException as e:
    print(f"连接超时: {e}")
    # 处理超时逻辑
```

---

### 4.3 重试机制

**实现自动重试：**

```python
from pymilvus import connections, MilvusException
import time

def connect_with_retry(max_retries=3, retry_delay=2):
    """带重试的连接函数"""
    for attempt in range(max_retries):
        try:
            connections.connect(
                alias="default",
                host="localhost",
                port="19530",
                timeout=10
            )
            print("✅ 连接成功")
            return True
        except MilvusException as e:
            print(f"❌ 连接失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("❌ 达到最大重试次数，连接失败")
                return False

# 使用
connect_with_retry()
```

---

### 4.4 错误码对照表

**常见错误码：**

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| `1` | 连接失败 | 检查 Milvus 是否启动，端口是否正确 |
| `5` | Collection 不存在 | 先创建 Collection |
| `15` | 索引不存在 | 先创建索引 |
| `65535` | 未知错误 | 查看详细错误信息 |

**获取错误码：**

```python
from pymilvus import MilvusException

try:
    # 某个操作
    pass
except MilvusException as e:
    print(f"错误码: {e.code}")
    print(f"错误信息: {e.message}")
```

---

## 5. 在 RAG 中的应用

### 5.1 向量存储客户端封装

**封装 Milvus 客户端：**

```python
from pymilvus import connections, Collection, utility
from typing import List, Dict, Any

class MilvusClient:
    """Milvus 客户端封装"""

    def __init__(self, host="localhost", port="19530", alias="default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    def _connect(self):
        """建立连接"""
        connections.connect(
            alias=self.alias,
            host=self.host,
            port=self.port
        )
        print(f"✅ 连接到 Milvus: {self.host}:{self.port}")

    def get_collection(self, name: str) -> Collection:
        """获取 Collection"""
        if not utility.has_collection(name):
            raise ValueError(f"Collection '{name}' 不存在")
        return Collection(name, using=self.alias)

    def insert_vectors(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> List[int]:
        """插入向量数据"""
        collection = self.get_collection(collection_name)
        data = [texts, embeddings]
        result = collection.insert(data)
        collection.flush()
        return result.primary_keys

    def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索向量"""
        collection = self.get_collection(collection_name)

        # 确保 Collection 已加载
        if not utility.load_state(collection_name).get("state") == "Loaded":
            collection.load()

        # 执行检索
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text"]
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text")
                })

        return formatted_results

    def close(self):
        """关闭连接"""
        connections.disconnect(self.alias)
        print("✅ 连接已关闭")

# 使用示例
client = MilvusClient()

# 插入数据
texts = ["Milvus is a vector database", "RAG uses vector search"]
embeddings = [[0.1]*128, [0.2]*128]
ids = client.insert_vectors("my_collection", texts, embeddings)
print(f"插入了 {len(ids)} 条数据")

# 检索数据
query_embedding = [0.15]*128
results = client.search_vectors("my_collection", query_embedding, top_k=2)
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Text: {result['text']}")

client.close()
```

---

### 5.2 连接池设计模式

**为什么需要连接池？**

在高并发场景下，频繁创建和销毁连接会导致：
- 性能下降（TCP 三次握手开销）
- 资源浪费（连接建立和销毁的 CPU 开销）
- 服务器压力（大量连接请求）

**连接池实现：**

```python
import queue
import threading
from pymilvus import connections

class MilvusConnectionPool:
    """Milvus 连接池"""

    def __init__(self, size=10, host="localhost", port="19530"):
        self.size = size
        self.host = host
        self.port = port
        self.pool = queue.Queue(maxsize=size)
        self.lock = threading.Lock()
        self._init_pool()

    def _init_pool(self):
        """初始化连接池"""
        for i in range(self.size):
            alias = f"conn_{i}"
            connections.connect(alias, host=self.host, port=self.port)
            self.pool.put(alias)
        print(f"✅ 连接池初始化完成，大小: {self.size}")

    def get_connection(self, timeout=None):
        """获取连接（阻塞）"""
        try:
            alias = self.pool.get(timeout=timeout)
            return alias
        except queue.Empty:
            raise TimeoutError("连接池已满，无法获取连接")

    def return_connection(self, alias):
        """归还连接"""
        self.pool.put(alias)

    def close_all(self):
        """关闭所有连接"""
        while not self.pool.empty():
            alias = self.pool.get()
            connections.disconnect(alias)
        print("✅ 所有连接已关闭")

# 使用示例
pool = MilvusConnectionPool(size=5)

def worker(worker_id):
    """工作线程"""
    alias = pool.get_connection()
    try:
        from pymilvus import Collection
        collection = Collection("my_collection", using=alias)
        print(f"Worker {worker_id} 使用连接 {alias}，数据量: {collection.num_entities}")
    finally:
        pool.return_connection(alias)

# 启动多个工作线程
threads = []
for i in range(20):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

pool.close_all()
```

---

### 5.3 与 LangChain/LlamaIndex 集成

**LangChain 集成示例：**

```python
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# 1. 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. 创建 Embedding
embeddings = OpenAIEmbeddings()

# 4. 创建 Milvus 向量存储
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="langchain_docs",
    connection_args={"host": "localhost", "port": "19530"}
)

# 5. 添加文档
vector_store.add_documents(chunks)

# 6. 检索
query = "What is Milvus?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print(f"内容: {doc.page_content}")
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 理解 pymilvus 的作用和架构
- [ ] 安装和配置 pymilvus
- [ ] 使用 connections.connect() 建立连接
- [ ] 理解连接别名机制
- [ ] 正确断开连接和释放资源
- [ ] 使用 Collection、Partition、Index 对象
- [ ] 处理常见的异常情况
- [ ] 实现连接重试机制
- [ ] 封装 Milvus 客户端
- [ ] 设计和实现连接池
- [ ] 与 LangChain/LlamaIndex 集成

---

## 下一步学习

- **健康检查**（05_核心概念_健康检查.md）
  - 学习如何监控 Milvus 服务状态

- **实战代码**（09-12_实战代码_场景*.md）
  - 动手实践完整的连接管理流程

---

**记住：** pymilvus 是连接 Milvus 的桥梁，掌握连接管理和异常处理是构建稳定 RAG 系统的关键！
