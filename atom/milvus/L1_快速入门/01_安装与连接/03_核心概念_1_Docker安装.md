# 核心概念 1: Docker 安装

深入理解 Milvus 2.6 的 Docker 部署机制和配置细节。

---

## 概述

Docker 安装是 Milvus 2.6 最推荐的部署方式,通过 Docker Compose 可以一键启动所有必需的服务组件。

---

## Docker Compose 架构

### 服务组件

Milvus 2.6 Standalone 包含 3 个核心服务:

```yaml
services:
  etcd:          # 元数据存储
  minio:         # 对象存储
  standalone:    # Milvus 核心服务
```

**组件关系图**:

```
┌─────────────────────────────────────────────────────────┐
│                    Milvus Standalone                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Milvus Core (standalone container)              │  │
│  │  ├── Proxy (接收客户端请求)                       │  │
│  │  ├── Query Node (查询执行)                        │  │
│  │  ├── Data Node (数据写入)                         │  │
│  │  ├── Index Node (索引构建)                        │  │
│  │  ├── Streaming Node (流式处理) [2.6 新增]        │  │
│  │  └── Woodpecker WAL (预写日志) [2.6 新增]        │  │
│  └──────────────────────────────────────────────────┘  │
│                          ↓ ↑                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │  etcd (元数据存储)                                │  │
│  │  ├── Collection Schema                           │  │
│  │  ├── Index 配置                                   │  │
│  │  └── 系统配置                                     │  │
│  └──────────────────────────────────────────────────┘  │
│                          ↓ ↑                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │  MinIO (对象存储)                                 │  │
│  │  ├── 向量数据                                     │  │
│  │  ├── 索引文件                                     │  │
│  │  └── 日志文件                                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## docker-compose.yml 详解

### 完整配置文件

```yaml
version: '3.5'

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
    image: milvusdb/milvus:v2.6.11
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: minioadmin
      MINIO_SECRET_ACCESS_KEY: minioadmin
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
```

### 配置详解

#### 1. etcd 配置

```yaml
etcd:
  image: quay.io/coreos/etcd:v3.5.5  # etcd 版本
  environment:
    # 自动压缩模式 (按修订版本)
    - ETCD_AUTO_COMPACTION_MODE=revision
    # 保留最近 1000 个修订版本
    - ETCD_AUTO_COMPACTION_RETENTION=1000
    # 后端数据库大小限制 (4GB)
    - ETCD_QUOTA_BACKEND_BYTES=4294967296
    # 快照计数 (每 50000 次事务创建快照)
    - ETCD_SNAPSHOT_COUNT=50000
  volumes:
    # 数据持久化目录
    - ./volumes/etcd:/etcd
  command: |
    etcd \
      -advertise-client-urls=http://127.0.0.1:2379 \
      -listen-client-urls http://0.0.0.0:2379 \
      --data-dir /etcd
```

**关键参数说明**:

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `ETCD_AUTO_COMPACTION_MODE` | 自动压缩模式 | - | `revision` |
| `ETCD_AUTO_COMPACTION_RETENTION` | 保留修订版本数 | - | `1000` |
| `ETCD_QUOTA_BACKEND_BYTES` | 数据库大小限制 | 2GB | 4GB |
| `ETCD_SNAPSHOT_COUNT` | 快照间隔 | 10000 | 50000 |

#### 2. MinIO 配置

```yaml
minio:
  image: minio/minio:RELEASE.2023-03-20T20-16-18Z  # MinIO 版本
  environment:
    # 访问密钥 (默认: minioadmin)
    MINIO_ACCESS_KEY: minioadmin
    # 密钥 (默认: minioadmin)
    MINIO_SECRET_KEY: minioadmin
  ports:
    # API 端口
    - "9000:9000"
    # 控制台端口
    - "9001:9001"
  volumes:
    # 数据持久化目录
    - ./volumes/minio:/minio_data
  command: minio server /minio_data --console-address ":9001"
```

**关键参数说明**:

| 参数 | 说明 | 默认值 | 生产环境建议 |
|------|------|--------|-------------|
| `MINIO_ACCESS_KEY` | 访问密钥 | `minioadmin` | 修改为强密码 |
| `MINIO_SECRET_KEY` | 密钥 | `minioadmin` | 修改为强密码 |
| `--console-address` | 控制台地址 | `:9001` | 保持默认 |

#### 3. Milvus Standalone 配置

```yaml
standalone:
  image: milvusdb/milvus:v2.6.11  # Milvus 2.6.11 版本
  command: ["milvus", "run", "standalone"]
  environment:
    # etcd 连接地址
    ETCD_ENDPOINTS: etcd:2379
    # MinIO 连接地址
    MINIO_ADDRESS: minio:9000
    # MinIO 访问密钥
    MINIO_ACCESS_KEY_ID: minioadmin
    # MinIO 密钥
    MINIO_SECRET_ACCESS_KEY: minioadmin
  volumes:
    # 数据持久化目录
    - ./volumes/milvus:/var/lib/milvus
  ports:
    # gRPC API 端口
    - "19530:19530"
    # WebUI 端口
    - "9091:9091"
  depends_on:
    - "etcd"
    - "minio"
```

**关键参数说明**:

| 参数 | 说明 | 默认值 | 用途 |
|------|------|--------|------|
| `ETCD_ENDPOINTS` | etcd 地址 | `etcd:2379` | 元数据存储 |
| `MINIO_ADDRESS` | MinIO 地址 | `minio:9000` | 对象存储 |
| `19530` | gRPC 端口 | - | 客户端连接 |
| `9091` | WebUI 端口 | - | 可视化管理 |

---

## 端口映射

### 端口列表

| 服务 | 容器端口 | 宿主机端口 | 用途 |
|------|---------|-----------|------|
| **Milvus** | 19530 | 19530 | gRPC API (客户端连接) |
| **Milvus** | 9091 | 9091 | WebUI (可视化管理) |
| **MinIO** | 9000 | 9000 | S3 API (对象存储) |
| **MinIO** | 9001 | 9001 | 控制台 (管理界面) |
| **etcd** | 2379 | - | 客户端 API (内部使用) |
| **etcd** | 2380 | - | 对等通信 (内部使用) |

### 端口修改示例

```yaml
# 修改 Milvus 端口 (避免冲突)
standalone:
  ports:
    - "19531:19530"  # 外部端口改为 19531
    - "9092:9091"    # 外部端口改为 9092
```

```python
# Python 代码使用修改后的端口
client = MilvusClient(uri="http://localhost:19531")
```

---

## 数据持久化

### Volumes 目录结构

```bash
volumes/
├── etcd/           # etcd 数据
│   ├── member/     # 成员数据
│   └── wal/        # 预写日志
├── minio/          # MinIO 数据
│   ├── .minio.sys/ # 系统配置
│   └── milvus/     # Milvus 数据桶
│       ├── delta_log/  # 增量日志
│       ├── insert_log/ # 插入日志
│       └── stats_log/  # 统计日志
└── milvus/         # Milvus 数据
    ├── wal/        # Woodpecker WAL
    ├── rdb/        # RocksDB 数据
    └── logs/       # 日志文件
```

### 数据持久化机制

**写入流程**:

```
1. 客户端写入请求
   ↓
2. Milvus Proxy 接收
   ↓
3. Woodpecker WAL 记录 (./volumes/milvus/wal/)
   ↓
4. Data Node 处理
   ↓
5. MinIO 持久化 (./volumes/minio/milvus/)
   ↓
6. etcd 更新元数据 (./volumes/etcd/)
```

**恢复流程**:

```
1. 容器重启
   ↓
2. 挂载 volumes 目录
   ↓
3. 读取 Woodpecker WAL
   ↓
4. 回放未提交的操作
   ↓
5. 从 MinIO 加载数据
   ↓
6. 从 etcd 加载元数据
   ↓
7. 服务就绪
```

---

## 健康检查

### 健康检查配置

```yaml
standalone:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
    interval: 30s        # 检查间隔
    start_period: 90s    # 启动宽限期
    timeout: 20s         # 超时时间
    retries: 3           # 重试次数
```

### 手动健康检查

```bash
# 方法 1: 使用 curl
curl http://localhost:9091/healthz

# 输出示例:
# OK

# 方法 2: 使用 docker inspect
docker inspect --format='{{.State.Health.Status}}' milvus-standalone

# 输出示例:
# healthy

# 方法 3: 使用 docker compose ps
docker compose ps

# 输出示例:
# NAME                COMMAND                  SERVICE      STATUS       PORTS
# milvus-standalone   /tini -- milvus run st…  standalone   Up (healthy) 0.0.0.0:19530->19530/tcp
```

---

## 环境变量

### 常用环境变量

```yaml
standalone:
  environment:
    # === 基础配置 ===
    ETCD_ENDPOINTS: etcd:2379
    MINIO_ADDRESS: minio:9000

    # === 认证配置 ===
    MINIO_ACCESS_KEY_ID: minioadmin
    MINIO_SECRET_ACCESS_KEY: minioadmin

    # === 性能配置 ===
    # 查询节点内存限制 (字节)
    QUERY_NODE_MEMORY_LIMIT: 8589934592  # 8GB

    # 数据节点内存限制 (字节)
    DATA_NODE_MEMORY_LIMIT: 4294967296   # 4GB

    # 索引节点内存限制 (字节)
    INDEX_NODE_MEMORY_LIMIT: 4294967296  # 4GB

    # === 日志配置 ===
    # 日志级别 (debug, info, warn, error)
    LOG_LEVEL: info

    # 日志格式 (text, json)
    LOG_FORMAT: text

    # === 其他配置 ===
    # 时区
    TZ: Asia/Shanghai
```

### 生产环境配置示例

```yaml
standalone:
  environment:
    # 基础配置
    ETCD_ENDPOINTS: etcd:2379
    MINIO_ADDRESS: minio:9000
    MINIO_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY}  # 从环境变量读取
    MINIO_SECRET_ACCESS_KEY: ${MINIO_SECRET_KEY}

    # 性能优化
    QUERY_NODE_MEMORY_LIMIT: 17179869184  # 16GB
    DATA_NODE_MEMORY_LIMIT: 8589934592    # 8GB
    INDEX_NODE_MEMORY_LIMIT: 8589934592   # 8GB

    # 日志配置
    LOG_LEVEL: warn
    LOG_FORMAT: json

    # 时区
    TZ: Asia/Shanghai
```

---

## 网络配置

### 默认网络

```yaml
networks:
  default:
    name: milvus
```

**网络特性**:
- 所有容器在同一网络中
- 容器间可以通过服务名通信
- 例如: `etcd:2379`, `minio:9000`

### 自定义网络

```yaml
networks:
  milvus-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  standalone:
    networks:
      - milvus-network
```

---

## 资源限制

### CPU 和内存限制

```yaml
standalone:
  deploy:
    resources:
      limits:
        cpus: '8'      # 最多使用 8 核 CPU
        memory: 16G    # 最多使用 16GB 内存
      reservations:
        cpus: '4'      # 预留 4 核 CPU
        memory: 8G     # 预留 8GB 内存
```

### 推荐配置

| 规模 | CPU 限制 | 内存限制 | CPU 预留 | 内存预留 |
|------|---------|---------|---------|---------|
| 小型 | 4 核 | 8GB | 2 核 | 4GB |
| 中型 | 8 核 | 16GB | 4 核 | 8GB |
| 大型 | 16 核 | 32GB | 8 核 | 16GB |

---

## 启动和停止

### 启动服务

```bash
# 启动所有服务
docker compose up -d

# 启动特定服务
docker compose up -d standalone

# 查看启动日志
docker compose logs -f standalone
```

### 停止服务

```bash
# 停止所有服务
docker compose down

# 停止并删除 volumes (数据会丢失!)
docker compose down -v

# 停止特定服务
docker compose stop standalone
```

### 重启服务

```bash
# 重启所有服务
docker compose restart

# 重启特定服务
docker compose restart standalone
```

---

## 日志管理

### 查看日志

```bash
# 查看所有服务日志
docker compose logs

# 查看特定服务日志
docker compose logs standalone

# 实时查看日志
docker compose logs -f standalone

# 查看最近 100 行日志
docker compose logs --tail=100 standalone
```

### 日志配置

```yaml
standalone:
  logging:
    driver: "json-file"
    options:
      max-size: "100m"   # 单个日志文件最大 100MB
      max-file: "10"     # 最多保留 10 个日志文件
```

---

## 升级和迁移

### 升级 Milvus 版本

```bash
# 1. 停止服务
docker compose down

# 2. 备份数据
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz volumes/

# 3. 修改 docker-compose.yml
# image: milvusdb/milvus:v2.6.11 → image: milvusdb/milvus:v2.6.12

# 4. 拉取新镜像
docker compose pull

# 5. 启动服务
docker compose up -d

# 6. 验证升级
docker compose logs -f standalone
```

### 迁移到其他机器

```bash
# 源机器
# 1. 停止服务
docker compose down

# 2. 打包数据
tar -czf milvus-data.tar.gz volumes/ docker-compose.yml

# 3. 传输到目标机器
scp milvus-data.tar.gz user@target-host:/path/to/milvus/

# 目标机器
# 4. 解压数据
tar -xzf milvus-data.tar.gz

# 5. 启动服务
docker compose up -d
```

---

## 故障排查

### 常见问题

#### 1. 容器启动失败

```bash
# 查看容器状态
docker compose ps

# 查看错误日志
docker compose logs standalone

# 常见原因:
# - 端口被占用
# - 内存不足
# - 磁盘空间不足
```

#### 2. 连接超时

```bash
# 检查容器是否运行
docker compose ps

# 检查端口是否开放
netstat -an | grep 19530

# 检查防火墙
sudo iptables -L | grep 19530
```

#### 3. 数据丢失

```bash
# 检查 volumes 是否挂载
docker inspect milvus-standalone | grep Mounts

# 检查 volumes 目录权限
ls -la volumes/

# 恢复数据
tar -xzf milvus-backup-20260216.tar.gz
docker compose up -d
```

---

## 最佳实践

### 1. 生产环境配置

```yaml
# docker-compose-prod.yml
version: '3.5'

services:
  standalone:
    image: milvusdb/milvus:v2.6.11
    restart: always  # 自动重启
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
    volumes:
      - /data/milvus:/var/lib/milvus  # 使用独立磁盘
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      QUERY_NODE_MEMORY_LIMIT: 8589934592
      LOG_LEVEL: warn
      LOG_FORMAT: json
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
```

### 2. 定期备份

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/milvus"
DATE=$(date +%Y%m%d_%H%M%S)

# 停止服务
docker compose down

# 备份数据
tar -czf $BACKUP_DIR/milvus-$DATE.tar.gz volumes/

# 启动服务
docker compose up -d

# 保留最近 7 天的备份
find $BACKUP_DIR -name "milvus-*.tar.gz" -mtime +7 -delete
```

### 3. 监控和告警

```yaml
# 使用 Prometheus 监控
standalone:
  ports:
    - "9091:9091"  # Prometheus metrics endpoint
```

---

## 总结

### 核心要点

1. **Docker Compose 架构**: 3 个核心服务 (etcd, MinIO, Milvus)
2. **端口映射**: 19530 (gRPC), 9091 (WebUI)
3. **数据持久化**: volumes 目录自动创建和挂载
4. **健康检查**: 自动检测服务状态
5. **环境变量**: 灵活配置性能和日志
6. **资源限制**: 控制 CPU 和内存使用
7. **日志管理**: 自动轮转和清理
8. **升级迁移**: 简单的备份和恢复流程

### 下一步

- 阅读 **03_核心概念_2_pymilvus基础.md** 学习 Python SDK
- 阅读 **07_实战代码_场景1_Standalone部署.md** 动手实践
- 阅读 **07_实战代码_场景2_Compose部署.md** 学习生产环境配置

---

**参考文献**:
- Milvus 2.6 Installation: https://milvus.io/docs/install_standalone-docker-compose.md
- Docker Compose Documentation: https://docs.docker.com/compose/
- Milvus 2.6 Configuration: https://milvus.io/docs/configure-docker.md
