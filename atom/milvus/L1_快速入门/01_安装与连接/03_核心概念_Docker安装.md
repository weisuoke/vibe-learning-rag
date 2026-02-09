# 03_核心概念_Docker安装

> 深入理解如何使用 Docker 部署 Milvus 向量数据库

---

## 1. 什么是 Docker 部署 Milvus

### 1.1 Docker 容器化概念

**Docker 是什么？**

Docker 是一个容器化平台，可以将应用程序及其所有依赖打包到一个标准化的单元中。

```
传统部署：
应用程序 → 依赖库 → 操作系统 → 硬件
（每一层都需要手动配置）

Docker 部署：
Docker 镜像（应用 + 依赖 + 环境）→ Docker 引擎 → 操作系统 → 硬件
（一键启动，环境隔离）
```

**核心概念：**

| 概念 | 说明 | 类比 |
|------|------|------|
| **镜像 (Image)** | 应用程序的模板 | 软件安装包 |
| **容器 (Container)** | 镜像的运行实例 | 运行中的程序 |
| **仓库 (Registry)** | 存储镜像的地方 | 应用商店 |
| **Docker Hub** | 官方镜像仓库 | GitHub for Docker |

---

### 1.2 为什么 Milvus 推荐 Docker 部署

**Milvus 的架构复杂性：**

Milvus Standalone 模式包含 3 个核心组件：

```
┌─────────────────────────────────────┐
│         Milvus Standalone           │
├─────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │  etcd   │  │  MinIO  │  │Milvus│ │
│  │(元数据) │  │(对象存储)│  │(查询)│ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
```

**手动安装的挑战：**

1. **依赖管理复杂**
   ```bash
   # 需要分别安装和配置
   - etcd: 分布式键值存储
   - MinIO: 对象存储服务
   - Milvus: 向量数据库本体
   ```

2. **配置繁琐**
   ```yaml
   # 需要配置三者之间的通信
   etcd:
     endpoints: ["localhost:2379"]

   minio:
     address: localhost
     port: 9000

   milvus:
     etcd: localhost:2379
     minio: localhost:9000
   ```

3. **版本兼容性问题**
   - etcd 版本必须 >= 3.5.0
   - MinIO 版本必须兼容 S3 API
   - Milvus 版本与依赖版本需要匹配

**Docker 的优势：**

```bash
# 一行命令解决所有问题
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest

# Docker 镜像已经包含：
# ✅ etcd（正确版本）
# ✅ MinIO（正确版本）
# ✅ Milvus（正确版本）
# ✅ 所有配置（已优化）
```

---

### 1.3 Standalone vs Cluster 模式对比

| 特性 | Standalone 模式 | Cluster 模式 |
|------|-----------------|--------------|
| **部署方式** | 单个 Docker 容器 | 多个容器/K8s |
| **适用场景** | 开发、测试、小规模 | 生产、大规模 |
| **数据量** | < 100万向量 | 亿级向量 |
| **QPS** | < 1000 | > 10000 |
| **高可用** | ❌ 单点故障 | ✅ 多副本 |
| **水平扩展** | ❌ 不支持 | ✅ 支持 |
| **资源需求** | 4GB RAM | 16GB+ RAM |
| **启动时间** | < 30秒 | 2-5分钟 |

**选择建议：**

```python
# 开发环境：Standalone
if environment == "development":
    use_standalone()  # 快速启动，资源占用少

# 测试环境：Standalone
elif environment == "testing":
    use_standalone()  # 与开发环境一致

# 生产环境：Cluster
elif environment == "production":
    if data_size < 1_000_000 and qps < 1000:
        use_standalone()  # 小规模可以用 Standalone
    else:
        use_cluster()  # 大规模必须用 Cluster
```

---

## 2. Docker 安装步骤

### 2.1 前置条件检查

**系统要求：**

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Linux/macOS/Windows | Linux |
| **Docker 版本** | 19.03+ | 最新稳定版 |
| **CPU** | 2 核 | 4 核+ |
| **内存** | 4GB | 8GB+ |
| **磁盘** | 10GB | 50GB+ SSD |

**检查 Docker 是否安装：**

```bash
# 检查 Docker 版本
docker --version
# 输出示例：Docker version 24.0.7, build afdd53b

# 检查 Docker 是否运行
docker ps
# 如果能看到表头，说明 Docker 正常运行

# 检查 Docker 信息
docker info | grep "Server Version"
# 输出示例：Server Version: 24.0.7
```

**如果 Docker 未安装：**

```bash
# macOS（使用 Homebrew）
brew install --cask docker

# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# CentOS/RHEL
sudo yum install -y docker-ce docker-ce-cli containerd.io

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker
```

---

### 2.2 拉取 Milvus 镜像

**查看可用版本：**

```bash
# 在 Docker Hub 查看 Milvus 镜像
# https://hub.docker.com/r/milvusdb/milvus/tags

# 拉取最新稳定版
docker pull milvusdb/milvus:latest

# 拉取指定版本（推荐）
docker pull milvusdb/milvus:v2.4.0

# 查看已下载的镜像
docker images | grep milvus
```

**镜像大小：**

```
milvusdb/milvus:latest   ~1.5GB
（包含 etcd + MinIO + Milvus）
```

---

### 2.3 启动 Standalone 容器

**基础启动命令：**

```bash
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `-d` | 后台运行（detached mode） |
| `--name milvus-standalone` | 容器名称 |
| `-p 19530:19530` | 映射 gRPC 端口（客户端连接） |
| `-p 9091:9091` | 映射 Metrics 端口（监控） |
| `milvusdb/milvus:latest` | 镜像名称和标签 |

**验证启动成功：**

```bash
# 查看容器状态
docker ps | grep milvus
# 输出示例：
# CONTAINER ID   IMAGE                    STATUS         PORTS
# abc123def456   milvusdb/milvus:latest   Up 30 seconds  0.0.0.0:19530->19530/tcp

# 查看容器日志
docker logs milvus-standalone
# 看到 "Milvus Proxy successfully started" 表示启动成功

# 查看容器资源使用
docker stats milvus-standalone --no-stream
```

---

### 2.4 端口映射说明

**Milvus 使用的端口：**

| 端口 | 协议 | 用途 | 是否必需 |
|------|------|------|----------|
| **19530** | gRPC | 客户端连接 | ✅ 必需 |
| **9091** | HTTP | Prometheus Metrics | ⚠️ 推荐 |
| **2379** | HTTP | etcd 客户端 | ❌ 内部使用 |
| **9000** | HTTP | MinIO API | ❌ 内部使用 |

**端口映射格式：**

```bash
-p <宿主机端口>:<容器端口>

# 示例1：标准映射
-p 19530:19530  # 外部访问 localhost:19530 → 容器内 19530

# 示例2：自定义映射
-p 8080:19530   # 外部访问 localhost:8080 → 容器内 19530

# 示例3：只监听本地
-p 127.0.0.1:19530:19530  # 只允许本机访问
```

**端口冲突解决：**

```bash
# 检查端口是否被占用
lsof -i :19530
# 或
netstat -an | grep 19530

# 如果端口被占用，使用其他端口
docker run -d \
  --name milvus-standalone \
  -p 19531:19530 \  # 使用 19531 代替 19530
  -p 9092:9091 \
  milvusdb/milvus:latest

# Python 连接时使用新端口
from pymilvus import connections
connections.connect("default", host="localhost", port="19531")
```

---

### 2.5 数据持久化配置

**为什么需要数据持久化？**

```
默认情况：
容器删除 → 数据丢失 ❌

持久化后：
容器删除 → 数据保留 ✅
```

**使用 Docker Volumes：**

```bash
# 创建数据卷
docker volume create milvus-data

# 启动容器并挂载数据卷
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v milvus-data:/var/lib/milvus \
  milvusdb/milvus:latest

# 查看数据卷
docker volume ls
docker volume inspect milvus-data
```

**使用本地目录：**

```bash
# 创建本地目录
mkdir -p ~/milvus/data

# 启动容器并挂载本地目录
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v ~/milvus/data:/var/lib/milvus \
  milvusdb/milvus:latest

# 数据存储在 ~/milvus/data 目录
ls ~/milvus/data
```

**数据持久化验证：**

```python
# 1. 插入数据
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields)
collection = Collection("test_persistence", schema)

import random
vectors = [[random.random() for _ in range(128)] for _ in range(100)]
collection.insert([vectors])
collection.flush()

print(f"插入了 {collection.num_entities} 条数据")

# 2. 停止并删除容器
# docker stop milvus-standalone
# docker rm milvus-standalone

# 3. 重新启动容器（使用相同的数据卷）
# docker run -d --name milvus-standalone -p 19530:19530 -v milvus-data:/var/lib/milvus milvusdb/milvus:latest

# 4. 验证数据是否还在
connections.connect("default", host="localhost", port="19530")
collection = Collection("test_persistence")
print(f"数据还在！共 {collection.num_entities} 条")  # 应该还是 100
```

---

## 3. Docker Compose 部署

### 3.1 为什么使用 Docker Compose

**Docker Compose 的优势：**

| 特性 | docker run | docker-compose |
|------|------------|----------------|
| **配置管理** | 命令行参数 | YAML 文件 |
| **多容器** | 手动启动 | 一键启动 |
| **网络** | 手动配置 | 自动创建 |
| **依赖** | 手动管理 | 自动处理 |
| **版本控制** | ❌ | ✅ |
| **团队协作** | ❌ | ✅ |

---

### 3.2 docker-compose.yml 完整配置

**创建配置文件：**

```bash
# 创建项目目录
mkdir -p ~/milvus-docker
cd ~/milvus-docker

# 创建 docker-compose.yml
cat > docker-compose.yml <<'EOF'
version: '3.8'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus

volumes:
  etcd:
  minio:
  milvus:
EOF
```

---

### 3.3 环境变量配置

**创建 .env 文件：**

```bash
cat > .env <<'EOF'
# 数据存储目录
DOCKER_VOLUME_DIRECTORY=./volumes

# Milvus 配置
MILVUS_VERSION=v2.4.0

# MinIO 配置
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# etcd 配置
ETCD_VERSION=v3.5.5
EOF
```

**配置说明：**

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DOCKER_VOLUME_DIRECTORY` | 数据存储目录 | `./volumes` |
| `MILVUS_VERSION` | Milvus 版本 | `v2.4.0` |
| `MINIO_ACCESS_KEY` | MinIO 访问密钥 | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO 密钥 | `minioadmin` |

---

### 3.4 启动和管理

**启动服务：**

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f standalone

# 查看特定服务日志
docker-compose logs -f minio
```

**停止和清理：**

```bash
# 停止服务
docker-compose stop

# 停止并删除容器
docker-compose down

# 停止并删除容器和数据卷
docker-compose down -v

# 重启服务
docker-compose restart
```

**更新服务：**

```bash
# 拉取最新镜像
docker-compose pull

# 重新创建容器
docker-compose up -d --force-recreate
```

---

### 3.5 网络配置

**Docker Compose 自动创建网络：**

```bash
# 查看网络
docker network ls | grep milvus

# 查看网络详情
docker network inspect milvus
```

**服务间通信：**

```yaml
# 在 docker-compose.yml 中，服务可以通过服务名访问
standalone:
  environment:
    ETCD_ENDPOINTS: etcd:2379      # 通过服务名 "etcd" 访问
    MINIO_ADDRESS: minio:9000      # 通过服务名 "minio" 访问
```

**外部访问：**

```python
# Python 客户端通过宿主机端口访问
from pymilvus import connections

connections.connect(
    "default",
    host="localhost",  # 宿主机地址
    port="19530"       # 映射的端口
)
```

---

## 4. 常见问题排查

### 4.1 端口冲突

**问题现象：**

```bash
docker: Error response from daemon: driver failed programming external connectivity on endpoint milvus-standalone: Bind for 0.0.0.0:19530 failed: port is already allocated.
```

**解决方案：**

```bash
# 1. 查找占用端口的进程
lsof -i :19530
# 或
netstat -tulpn | grep 19530

# 2. 停止占用端口的进程
kill -9 <PID>

# 3. 或者使用其他端口
docker run -d \
  --name milvus-standalone \
  -p 19531:19530 \  # 使用 19531
  milvusdb/milvus:latest
```

---

### 4.2 权限问题

**问题现象：**

```bash
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**解决方案：**

```bash
# 方案1：将用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 方案2：使用 sudo
sudo docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest

# 方案3：修改 Docker socket 权限（不推荐）
sudo chmod 666 /var/run/docker.sock
```

---

### 4.3 内存不足

**问题现象：**

```bash
# 容器频繁重启
docker ps -a | grep milvus
# STATUS: Restarting (137) 2 seconds ago

# 查看日志
docker logs milvus-standalone
# OOMKilled: Out of memory
```

**解决方案：**

```bash
# 1. 检查系统内存
free -h

# 2. 限制容器内存使用
docker run -d \
  --name milvus-standalone \
  --memory="4g" \
  --memory-swap="4g" \
  -p 19530:19530 \
  milvusdb/milvus:latest

# 3. 在 docker-compose.yml 中配置
services:
  standalone:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

---

### 4.4 容器启动失败

**问题现象：**

```bash
docker ps -a | grep milvus
# STATUS: Exited (1) 10 seconds ago
```

**排查步骤：**

```bash
# 1. 查看详细日志
docker logs milvus-standalone

# 2. 检查容器配置
docker inspect milvus-standalone

# 3. 进入容器调试
docker exec -it milvus-standalone /bin/bash

# 4. 检查依赖服务
docker-compose ps  # 确保 etcd 和 minio 正常运行

# 5. 重新创建容器
docker rm milvus-standalone
docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest
```

---

## 5. 在 RAG 中的应用

### 5.1 开发环境快速搭建

**场景：** 本地开发 RAG 应用

```bash
# 1. 启动 Milvus
docker run -d \
  --name milvus-dev \
  -p 19530:19530 \
  -v ~/milvus-dev:/var/lib/milvus \
  milvusdb/milvus:latest

# 2. Python 开发环境
cat > rag_dev.py <<'EOF'
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from openai import OpenAI

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 创建 Collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields, description="RAG documents")
collection = Collection("rag_docs", schema)

# 插入文档
client = OpenAI()
texts = ["Milvus is a vector database", "RAG uses vector search"]
embeddings = [
    client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    for text in texts
]
collection.insert([texts, embeddings])

print("✅ RAG 开发环境就绪！")
EOF

python rag_dev.py
```

---

### 5.2 多环境隔离

**场景：** 开发、测试、生产环境隔离

```bash
# 开发环境
docker run -d \
  --name milvus-dev \
  -p 19530:19530 \
  -v ~/milvus-dev:/var/lib/milvus \
  milvusdb/milvus:latest

# 测试环境
docker run -d \
  --name milvus-test \
  -p 19531:19530 \
  -v ~/milvus-test:/var/lib/milvus \
  milvusdb/milvus:latest

# 生产环境（使用 docker-compose）
cd ~/milvus-prod
docker-compose up -d
```

**Python 配置：**

```python
import os

# 根据环境变量选择 Milvus 实例
env = os.getenv("ENV", "dev")

milvus_config = {
    "dev": {"host": "localhost", "port": "19530"},
    "test": {"host": "localhost", "port": "19531"},
    "prod": {"host": "milvus.prod.com", "port": "19530"}
}

from pymilvus import connections
connections.connect("default", **milvus_config[env])
```

---

### 5.3 与其他 RAG 组件的容器化集成

**完整 RAG 系统的 docker-compose.yml：**

```yaml
version: '3.8'

services:
  # Milvus 向量数据库
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus-data:/var/lib/milvus

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # PostgreSQL 元数据存储
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # RAG API 服务
  rag-api:
    build: ./rag-api
    ports:
      - "8000:8000"
    environment:
      MILVUS_HOST: milvus
      MILVUS_PORT: 19530
      REDIS_HOST: redis
      POSTGRES_HOST: postgres
    depends_on:
      - milvus
      - redis
      - postgres

volumes:
  milvus-data:
  redis-data:
  postgres-data:
```

**启动完整 RAG 系统：**

```bash
docker-compose up -d

# 验证所有服务
docker-compose ps

# 测试 RAG API
curl http://localhost:8000/health
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 理解 Docker 容器化的基本概念
- [ ] 解释为什么 Milvus 推荐使用 Docker 部署
- [ ] 区分 Standalone 和 Cluster 模式的使用场景
- [ ] 使用 docker run 启动 Milvus Standalone
- [ ] 配置端口映射和数据持久化
- [ ] 编写 docker-compose.yml 配置文件
- [ ] 排查常见的 Docker 部署问题
- [ ] 在 RAG 系统中集成 Milvus 容器

---

## 下一步学习

- **pymilvus 基础**（04_核心概念_pymilvus基础.md）
  - 学习如何使用 Python SDK 连接 Milvus

- **健康检查**（05_核心概念_健康检查.md）
  - 学习如何监控 Milvus 服务状态

- **实战代码**（09-12_实战代码_场景*.md）
  - 动手实践完整的部署流程

---

**记住：** Docker 部署是 Milvus 的推荐方式，掌握 Docker 基础操作是使用 Milvus 的第一步！
