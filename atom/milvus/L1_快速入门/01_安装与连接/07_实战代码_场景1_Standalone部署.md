# 实战代码 场景1: Standalone部署

完整的 Milvus 2.6 Standalone 部署实战,从零到生产环境的完整流程。

---

## 场景概述

**目标**: 在本地或服务器上部署 Milvus 2.6 Standalone,并验证服务可用性。

**适用场景**:
- 开发环境搭建
- 测试环境部署
- 中小规模生产环境 (< 1 亿向量)

**时间投入**: 10-15 分钟

---

## 前置准备

### 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Linux/macOS/Windows | Linux (Ubuntu 20.04+) |
| **Docker** | 20.10+ | 最新版本 |
| **Docker Compose** | 2.0+ | 最新版本 |
| **CPU** | 2 核 | 4 核+ |
| **内存** | 4GB | 8GB+ |
| **磁盘** | 20GB | 100GB+ SSD |

### 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER

# macOS
# 下载并安装 Docker Desktop
# https://www.docker.com/products/docker-desktop

# 验证安装
docker --version
docker compose version
```

---

## 完整部署流程

### 步骤 1: 创建项目目录

```bash
# 创建项目目录
mkdir -p ~/milvus-standalone
cd ~/milvus-standalone

# 创建数据目录 (可选,Docker 会自动创建)
mkdir -p volumes/{milvus,etcd,minio}
```

### 步骤 2: 下载 Docker Compose 配置

```bash
# 下载 Milvus 2.6.11 官方配置
wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 或使用 curl
curl -L https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -o docker-compose.yml

# 查看配置文件
cat docker-compose.yml
```

### 步骤 3: 启动 Milvus

```bash
# 启动所有服务
docker compose up -d

# 输出示例:
# [+] Running 3/3
#  ✔ Container milvus-etcd        Started
#  ✔ Container milvus-minio       Started
#  ✔ Container milvus-standalone  Started
```

### 步骤 4: 验证服务状态

```bash
# 检查容器状态
docker compose ps

# 输出示例:
# NAME                COMMAND                  SERVICE      STATUS       PORTS
# milvus-etcd         etcd -advertise-client…  etcd         Up 30 seconds  2379/tcp, 2380/tcp
# milvus-minio        /usr/bin/docker-entryp…  minio        Up 30 seconds  9000/tcp
# milvus-standalone   /tini -- milvus run st…  standalone   Up 30 seconds  0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp

# 查看日志
docker compose logs -f milvus-standalone

# 等待看到这行日志:
# "Milvus Proxy successfully started"
```

### 步骤 5: 健康检查

```bash
# HTTP healthz 检查
curl http://localhost:9091/healthz

# 输出: OK

# 访问 WebUI
open http://127.0.0.1:9091/webui/
```

---

## Python 客户端验证

### 安装 pymilvus

```bash
# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装 pymilvus
pip install pymilvus>=2.6.0

# 验证安装
python -c "import pymilvus; print(pymilvus.__version__)"
```

### 基础连接测试

创建文件 `test_connection.py`:

```python
#!/usr/bin/env python3
"""
Milvus 2.6 连接测试脚本
"""

from pymilvus import MilvusClient
import sys

def test_connection():
    """测试 Milvus 连接"""
    print("=" * 60)
    print("Milvus 2.6 连接测试")
    print("=" * 60)
    print()

    try:
        # 连接到 Milvus
        print("[1/3] 连接到 Milvus...")
        client = MilvusClient(uri="http://localhost:19530")
        print("✅ 连接成功")
        print()

        # 列出 Collection
        print("[2/3] 列出 Collection...")
        collections = client.list_collections()
        print(f"✅ Collection 数量: {len(collections)}")
        if collections:
            print(f"   Collection 列表: {', '.join(collections)}")
        print()

        # 创建测试 Collection
        print("[3/3] 创建测试 Collection...")
        test_collection = "test_connection"

        # 如果存在则删除
        if client.has_collection(test_collection):
            client.drop_collection(test_collection)

        # 创建 Collection
        client.create_collection(
            collection_name=test_collection,
            dimension=128,
            metric_type="COSINE"
        )
        print(f"✅ Collection '{test_collection}' 创建成功")

        # 插入测试数据
        import random
        test_data = [{
            "id": i,
            "vector": [random.random() for _ in range(128)],
            "text": f"测试数据 {i}"
        } for i in range(10)]

        client.insert(collection_name=test_collection, data=test_data)
        print(f"✅ 插入 {len(test_data)} 条测试数据")

        # 查询测试数据
        results = client.query(
            collection_name=test_collection,
            filter="id >= 0",
            output_fields=["id", "text"],
            limit=5
        )
        print(f"✅ 查询成功,返回 {len(results)} 条数据")

        # 清理测试 Collection
        client.drop_collection(test_collection)
        print(f"✅ 清理测试 Collection")
        print()

        print("=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
```

运行测试:

```bash
# 运行连接测试
python test_connection.py

# 输出示例:
# ============================================================
# Milvus 2.6 连接测试
# ============================================================
#
# [1/3] 连接到 Milvus...
# ✅ 连接成功
#
# [2/3] 列出 Collection...
# ✅ Collection 数量: 0
#
# [3/3] 创建测试 Collection...
# ✅ Collection 'test_connection' 创建成功
# ✅ 插入 10 条测试数据
# ✅ 查询成功,返回 5 条数据
# ✅ 清理测试 Collection
#
# ============================================================
# 🎉 所有测试通过!
# ============================================================
```

---

## 完整示例: 向量检索

创建文件 `vector_search_demo.py`:

```python
#!/usr/bin/env python3
"""
Milvus 2.6 向量检索完整示例
"""

from pymilvus import MilvusClient
import random
import time

def vector_search_demo():
    """向量检索完整示例"""
    print("=" * 60)
    print("Milvus 2.6 向量检索示例")
    print("=" * 60)
    print()

    # 1. 连接到 Milvus
    print("[1/6] 连接到 Milvus...")
    client = MilvusClient(uri="http://localhost:19530")
    print("✅ 连接成功")
    print()

    # 2. 创建 Collection
    print("[2/6] 创建 Collection...")
    collection_name = "vector_search_demo"

    # 如果存在则删除
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 创建 Collection (128 维向量)
    client.create_collection(
        collection_name=collection_name,
        dimension=128,
        metric_type="COSINE",
        auto_id=False
    )
    print(f"✅ Collection '{collection_name}' 创建成功")
    print()

    # 3. 插入数据
    print("[3/6] 插入数据...")
    num_entities = 1000
    data = [
        {
            "id": i,
            "vector": [random.random() for _ in range(128)],
            "text": f"这是第 {i} 条数据",
            "category": f"类别_{i % 10}",
            "score": random.randint(1, 100)
        }
        for i in range(num_entities)
    ]

    # 批量插入
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        client.insert(collection_name=collection_name, data=batch)
        print(f"   已插入 {min(i+batch_size, len(data))}/{len(data)}")

    print(f"✅ 插入 {num_entities} 条数据完成")
    print()

    # 4. 创建索引
    print("[4/6] 创建索引...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    print("✅ 索引创建成功")
    print()

    # 5. 向量检索
    print("[5/6] 向量检索...")

    # 生成查询向量
    query_vector = [random.random() for _ in range(128)]

    # 基础检索
    print("   5.1 基础检索 (Top-5):")
    start = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        output_fields=["id", "text", "category", "score"]
    )
    elapsed = time.time() - start

    for i, hit in enumerate(results[0], 1):
        print(f"      {i}. ID: {hit['id']}, 相似度: {hit['distance']:.4f}, "
              f"类别: {hit['entity']['category']}, 分数: {hit['entity']['score']}")
    print(f"   ⏱️  检索耗时: {elapsed*1000:.2f}ms")
    print()

    # 带标量过滤的检索
    print("   5.2 带标量过滤的检索 (score > 50):")
    start = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        filter="score > 50",
        output_fields=["id", "text", "category", "score"]
    )
    elapsed = time.time() - start

    for i, hit in enumerate(results[0], 1):
        print(f"      {i}. ID: {hit['id']}, 相似度: {hit['distance']:.4f}, "
              f"分数: {hit['entity']['score']}")
    print(f"   ⏱️  检索耗时: {elapsed*1000:.2f}ms")
    print()

    # 6. 清理
    print("[6/6] 清理...")
    client.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' 已删除")
    print()

    print("=" * 60)
    print("🎉 向量检索示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    vector_search_demo()
```

运行示例:

```bash
python vector_search_demo.py

# 输出示例:
# ============================================================
# Milvus 2.6 向量检索示例
# ============================================================
#
# [1/6] 连接到 Milvus...
# ✅ 连接成功
#
# [2/6] 创建 Collection...
# ✅ Collection 'vector_search_demo' 创建成功
#
# [3/6] 插入数据...
#    已插入 100/1000
#    已插入 200/1000
#    ...
#    已插入 1000/1000
# ✅ 插入 1000 条数据完成
#
# [4/6] 创建索引...
# ✅ 索引创建成功
#
# [5/6] 向量检索...
#    5.1 基础检索 (Top-5):
#       1. ID: 42, 相似度: 0.9234, 类别: 类别_2, 分数: 87
#       2. ID: 17, 相似度: 0.9156, 类别: 类别_7, 分数: 65
#       3. ID: 89, 相似度: 0.9087, 类别: 类别_9, 分数: 92
#       4. ID: 5, 相似度: 0.9012, 类别: 类别_5, 分数: 43
#       5. ID: 73, 相似度: 0.8945, 类别: 类别_3, 分数: 78
#    ⏱️  检索耗时: 12.34ms
#
#    5.2 带标量过滤的检索 (score > 50):
#       1. ID: 42, 相似度: 0.9234, 分数: 87
#       2. ID: 17, 相似度: 0.9156, 分数: 65
#       3. ID: 89, 相似度: 0.9087, 分数: 92
#       4. ID: 73, 相似度: 0.8945, 分数: 78
#       5. ID: 91, 相似度: 0.8876, 分数: 56
#    ⏱️  检索耗时: 15.67ms
#
# [6/6] 清理...
# ✅ Collection 'vector_search_demo' 已删除
#
# ============================================================
# 🎉 向量检索示例完成!
# ============================================================
```

---

## 常见问题排查

### 问题 1: 容器启动失败

**症状**:
```bash
docker compose up -d
# Error: Cannot start service standalone: driver failed programming external connectivity
```

**解决方案**:
```bash
# 1. 检查端口占用
lsof -i :19530
lsof -i :9091

# 2. 停止占用端口的进程
kill -9 <PID>

# 3. 或修改端口映射
# 编辑 docker-compose.yml
# ports:
#   - "19531:19530"
#   - "9092:9091"

# 4. 重新启动
docker compose up -d
```

### 问题 2: 连接超时

**症状**:
```python
client = MilvusClient(uri="http://localhost:19530")
# Error: Connection timeout
```

**解决方案**:
```bash
# 1. 检查容器状态
docker compose ps

# 2. 查看日志
docker compose logs milvus-standalone | tail -50

# 3. 等待服务完全启动 (10-30 秒)
sleep 30

# 4. 重新尝试连接
python test_connection.py
```

### 问题 3: 内存不足

**症状**:
```bash
docker compose logs milvus-standalone
# Error: OOM (Out of Memory)
```

**解决方案**:
```yaml
# 编辑 docker-compose.yml
services:
  standalone:
    deploy:
      resources:
        limits:
          memory: 4G  # 增加内存限制
```

### 问题 4: 磁盘空间不足

**症状**:
```bash
docker compose logs milvus-standalone
# Error: No space left on device
```

**解决方案**:
```bash
# 1. 检查磁盘空间
df -h

# 2. 清理 Docker 缓存
docker system prune -a

# 3. 清理 Milvus 数据 (谨慎!)
docker compose down
rm -rf volumes/

# 4. 重新启动
docker compose up -d
```

---

## 停止和清理

### 停止服务

```bash
# 停止所有服务
docker compose down

# 输出示例:
# [+] Running 3/3
#  ✔ Container milvus-standalone  Removed
#  ✔ Container milvus-minio       Removed
#  ✔ Container milvus-etcd        Removed
```

### 清理数据

```bash
# 停止并删除 volumes (数据会丢失!)
docker compose down -v

# 或手动删除 volumes 目录
rm -rf volumes/
```

### 完全清理

```bash
# 停止服务
docker compose down -v

# 删除镜像
docker rmi milvusdb/milvus:v2.6.11
docker rmi quay.io/coreos/etcd:v3.5.5
docker rmi minio/minio:RELEASE.2023-03-20T20-16-18Z

# 删除项目目录
cd ..
rm -rf milvus-standalone
```

---

## 性能测试

创建文件 `performance_test.py`:

```python
#!/usr/bin/env python3
"""
Milvus 2.6 性能测试脚本
"""

from pymilvus import MilvusClient
import random
import time

def performance_test():
    """性能测试"""
    print("=" * 60)
    print("Milvus 2.6 性能测试")
    print("=" * 60)
    print()

    client = MilvusClient(uri="http://localhost:19530")
    collection_name = "performance_test"

    # 创建 Collection
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=128,
        metric_type="COSINE"
    )

    # 测试 1: 插入性能
    print("[1/3] 插入性能测试...")
    num_entities = 10000
    data = [
        {
            "id": i,
            "vector": [random.random() for _ in range(128)]
        }
        for i in range(num_entities)
    ]

    start = time.time()
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        client.insert(collection_name=collection_name, data=batch)
    elapsed = time.time() - start

    print(f"✅ 插入 {num_entities} 条数据")
    print(f"   耗时: {elapsed:.2f}s")
    print(f"   吞吐量: {num_entities/elapsed:.2f} 条/秒")
    print()

    # 测试 2: 创建索引性能
    print("[2/3] 索引创建性能测试...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )

    start = time.time()
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    elapsed = time.time() - start

    print(f"✅ 索引创建完成")
    print(f"   耗时: {elapsed:.2f}s")
    print()

    # 测试 3: 检索性能
    print("[3/3] 检索性能测试...")
    query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]

    start = time.time()
    for query_vector in query_vectors:
        client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=10
        )
    elapsed = time.time() - start

    print(f"✅ 完成 {len(query_vectors)} 次检索")
    print(f"   耗时: {elapsed:.2f}s")
    print(f"   平均延迟: {elapsed/len(query_vectors)*1000:.2f}ms")
    print(f"   QPS: {len(query_vectors)/elapsed:.2f}")
    print()

    # 清理
    client.drop_collection(collection_name)

    print("=" * 60)
    print("🎉 性能测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    performance_test()
```

运行性能测试:

```bash
python performance_test.py

# 输出示例:
# ============================================================
# Milvus 2.6 性能测试
# ============================================================
#
# [1/3] 插入性能测试...
# ✅ 插入 10000 条数据
#    耗时: 2.34s
#    吞吐量: 4273.50 条/秒
#
# [2/3] 索引创建性能测试...
# ✅ 索引创建完成
#    耗时: 5.67s
#
# [3/3] 检索性能测试...
# ✅ 完成 100 次检索
#    耗时: 1.23s
#    平均延迟: 12.30ms
#    QPS: 81.30
#
# ============================================================
# 🎉 性能测试完成!
# ============================================================
```

---

## 总结

### 核心步骤回顾

1. **下载配置**: `wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml`
2. **启动服务**: `docker compose up -d`
3. **验证连接**: `python test_connection.py`
4. **向量检索**: `python vector_search_demo.py`
5. **性能测试**: `python performance_test.py`

### 关键命令

```bash
# 启动
docker compose up -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f milvus-standalone

# 停止
docker compose down

# 清理数据
docker compose down -v
```

### 下一步

- 阅读 **07_实战代码_场景2_Compose部署.md** 学习生产环境配置
- 阅读 **07_实战代码_场景3_连接管理.md** 学习连接管理实战
- 阅读 **07_实战代码_场景4_端到端RAG.md** 学习 RAG 系统集成

---

**参考文献**:
- Milvus 2.6 Installation: https://milvus.io/docs/install_standalone-docker-compose.md
- pymilvus Quickstart: https://milvus.io/docs/quickstart.md
- Docker Compose Documentation: https://docs.docker.com/compose/
