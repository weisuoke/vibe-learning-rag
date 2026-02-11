# 核心概念1：Docker Compose 配置详解

> 深入理解 docker-compose.yml 的结构、服务定义、网络和卷管理

---

## 概述

Docker Compose 使用 YAML 文件定义多容器应用。理解配置文件的结构是掌握 Docker 部署的基础。

---

## 1. Docker Compose 文件的基本结构

### 1.1 顶层配置项

```yaml
version: '3.8'          # Compose 文件版本

services:               # 服务定义（必需）
  service1:
  service2:

networks:               # 网络定义（可选）
  network1:

volumes:                # 卷定义（可选）
  volume1:

configs:                # 配置文件（可选，Swarm 模式）
  config1:

secrets:                # 密钥（可选，Swarm 模式）
  secret1:
```

**版本说明**：
- `3.8`：最新的 Compose 文件格式版本
- 不同版本支持不同的功能
- 推荐使用 3.8 或更高版本

---

## 2. 服务定义（Services）

### 2.1 基本服务配置

```yaml
services:
  milvus:
    # 镜像配置
    image: milvusdb/milvus:v2.4.0        # 使用的镜像
    # 或者使用 build
    # build:
    #   context: ./milvus
    #   dockerfile: Dockerfile

    # 容器名称
    container_name: milvus-standalone    # 自定义容器名

    # 启动命令
    command: ["milvus", "run", "standalone"]

    # 环境变量
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      # 或使用 key=value 格式
      # - ETCD_ENDPOINTS=etcd:2379

    # 端口映射
    ports:
      - "19530:19530"    # 宿主机:容器
      - "9091:9091"

    # 卷挂载
    volumes:
      - milvus_data:/var/lib/milvus
      - ./configs:/milvus/configs:ro  # :ro 表示只读

    # 依赖关系
    depends_on:
      - etcd
      - minio

    # 重启策略
    restart: unless-stopped

    # 网络配置
    networks:
      - milvus

    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 2.2 镜像配置详解

**使用预构建镜像**：

```yaml
services:
  milvus:
    image: milvusdb/milvus:v2.4.0
    # 格式：[registry/][namespace/]image[:tag|@digest]
```

**从 Dockerfile 构建**：

```yaml
services:
  milvus:
    build:
      context: ./milvus              # 构建上下文目录
      dockerfile: Dockerfile.prod    # Dockerfile 文件名
      args:                          # 构建参数
        VERSION: 2.4.0
        BUILD_DATE: 2024-01-01
      target: production             # 多阶段构建的目标阶段
      cache_from:                    # 缓存来源
        - milvusdb/milvus:v2.3.0
```

**在 RAG 开发中的应用**：
- 开发环境：使用官方镜像快速启动
- 生产环境：基于官方镜像构建自定义镜像，添加监控和配置

### 2.3 环境变量配置

**三种配置方式**：

```yaml
services:
  milvus:
    # 方式1：键值对格式
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      LOG_LEVEL: info

    # 方式2：数组格式
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000

    # 方式3：从文件加载
    env_file:
      - .env
      - .env.production
```

**环境变量优先级**：
```
命令行 -e > docker-compose.yml environment > env_file > Dockerfile ENV
```

**常用 Milvus 环境变量**：

```yaml
environment:
  # 依赖服务配置
  ETCD_ENDPOINTS: etcd:2379
  MINIO_ADDRESS: minio:9000
  PULSAR_ADDRESS: pulsar://pulsar:6650

  # 日志配置
  LOG_LEVEL: info                    # debug, info, warn, error
  LOG_FILE: /var/log/milvus/milvus.log

  # 性能配置
  CACHE_SIZE: 8                      # GB
  MAX_DEGREE: 64                     # 索引构建并行度

  # 数据一致性
  CONSISTENCY_LEVEL: Bounded         # Strong, Bounded, Session, Eventually
```

**在 RAG 开发中的应用**：

```yaml
# 开发环境
environment:
  LOG_LEVEL: debug
  CACHE_SIZE: 2

# 生产环境
environment:
  LOG_LEVEL: warn
  CACHE_SIZE: 16
```

### 2.4 端口映射详解

**基本格式**：

```yaml
ports:
  - "宿主机端口:容器端口"
  - "宿主机IP:宿主机端口:容器端口"
  - "宿主机端口:容器端口/协议"
```

**示例**：

```yaml
services:
  milvus:
    ports:
      # 基本映射
      - "19530:19530"                # Milvus gRPC 端口

      # 指定宿主机 IP
      - "127.0.0.1:19530:19530"      # 只允许本地访问

      # 指定协议
      - "19530:19530/tcp"

      # 动态端口（宿主机随机端口）
      - "19530"

      # 端口范围
      - "9091-9093:9091-9093"
```

**expose vs ports**：

```yaml
services:
  milvus:
    # expose：只在容器间可见，不映射到宿主机
    expose:
      - "19530"

    # ports：映射到宿主机，外部可访问
    ports:
      - "19530:19530"
```

**在 RAG 开发中的应用**：

```yaml
# 开发环境：映射所有端口，方便调试
ports:
  - "19530:19530"    # gRPC
  - "9091:9091"      # Metrics

# 生产环境：只映射必要端口，增强安全性
ports:
  - "127.0.0.1:19530:19530"  # 只允许本地访问
expose:
  - "9091"                    # Metrics 只在内部网络可见
```

### 2.5 卷挂载详解

**三种卷类型**：

```yaml
services:
  milvus:
    volumes:
      # 1. Named Volume（推荐）
      - milvus_data:/var/lib/milvus

      # 2. Bind Mount（绝对路径或相对路径）
      - ./data:/var/lib/milvus
      - /opt/milvus/data:/var/lib/milvus

      # 3. tmpfs（临时文件系统，存储在内存中）
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 1000000000  # 1GB
```

**卷挂载选项**：

```yaml
volumes:
  # 只读挂载
  - milvus_data:/var/lib/milvus:ro

  # 读写挂载（默认）
  - milvus_data:/var/lib/milvus:rw

  # 长格式（更多选项）
  - type: volume
    source: milvus_data
    target: /var/lib/milvus
    read_only: false
    volume:
      nocopy: true  # 不复制容器中的数据到 volume
```

**在 RAG 开发中的应用**：

```yaml
services:
  milvus:
    volumes:
      # 数据持久化（Named Volume）
      - milvus_data:/var/lib/milvus

      # 配置文件（Bind Mount，只读）
      - ./configs/milvus.yaml:/milvus/configs/milvus.yaml:ro

      # 日志输出（Bind Mount，方便查看）
      - ./logs:/var/log/milvus

      # 临时文件（tmpfs，提高性能）
      - type: tmpfs
        target: /tmp
```

### 2.6 依赖关系配置

**基本依赖**：

```yaml
services:
  milvus:
    depends_on:
      - etcd
      - minio
    # Milvus 会在 etcd 和 minio 启动后启动
```

**等待服务就绪**：

```yaml
services:
  etcd:
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 5s
      timeout: 3s
      retries: 5

  milvus:
    depends_on:
      etcd:
        condition: service_healthy  # 等待 etcd 健康检查通过
      minio:
        condition: service_started  # 等待 minio 启动（默认）
```

**依赖条件**：
- `service_started`：服务启动（默认）
- `service_healthy`：健康检查通过
- `service_completed_successfully`：服务成功退出

**在 RAG 开发中的应用**：

```yaml
services:
  etcd:
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s

  minio:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s

  milvus:
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    # 确保 Milvus 在依赖服务就绪后才启动
```

### 2.7 重启策略

```yaml
services:
  milvus:
    restart: unless-stopped
```

**重启策略选项**：
- `no`：不自动重启（默认）
- `always`：总是重启
- `on-failure`：失败时重启
- `unless-stopped`：除非手动停止，否则总是重启（推荐）

**在 RAG 开发中的应用**：

```yaml
# 开发环境：不自动重启，方便调试
restart: "no"

# 生产环境：自动重启，提高可用性
restart: unless-stopped
```

### 2.8 资源限制

```yaml
services:
  milvus:
    deploy:
      resources:
        limits:
          cpus: '4'           # 最多使用 4 个 CPU
          memory: 8G          # 最多使用 8GB 内存
        reservations:
          cpus: '2'           # 保证至少 2 个 CPU
          memory: 4G          # 保证至少 4GB 内存
```

**在 RAG 开发中的应用**：

```yaml
# 开发环境：较小的资源限制
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G

# 生产环境：更大的资源限制
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
    reservations:
      cpus: '4'
      memory: 8G
```

---

## 3. 网络配置（Networks）

### 3.1 默认网络

```yaml
services:
  milvus:
    # 不指定 networks，使用默认网络
  etcd:
    # 所有服务在同一个默认网络中
```

**默认网络特点**：
- 自动创建，名称为 `<项目名>_default`
- 所有服务可以通过服务名互相访问
- 使用 bridge 驱动

### 3.2 自定义网络

```yaml
networks:
  milvus:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 1500
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  milvus:
    networks:
      - milvus
  etcd:
    networks:
      - milvus
```

**网络驱动类型**：
- `bridge`：桥接网络（默认，推荐）
- `host`：使用宿主机网络（性能最好，但失去隔离性）
- `overlay`：跨主机网络（Swarm 模式）
- `none`：无网络

### 3.3 多网络配置

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

services:
  proxy:
    networks:
      - frontend
      - backend

  milvus:
    networks:
      - backend

  etcd:
    networks:
      - backend
```

**在 RAG 开发中的应用**：

```yaml
networks:
  # 前端网络：API 网关和应用
  frontend:
    driver: bridge

  # 后端网络：Milvus 和依赖服务
  backend:
    driver: bridge
    internal: true  # 内部网络，不能访问外部

services:
  api_gateway:
    networks:
      - frontend
      - backend

  milvus:
    networks:
      - backend

  etcd:
    networks:
      - backend
```

### 3.4 网络别名

```yaml
services:
  milvus:
    networks:
      milvus:
        aliases:
          - milvus-server
          - vector-db
```

**用途**：
- 同一个服务可以有多个网络别名
- 其他服务可以通过任意别名访问

---

## 4. 卷配置（Volumes）

### 4.1 Named Volumes

```yaml
volumes:
  milvus_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/milvus/data

  etcd_data:
    driver: local

  minio_data:
    driver: local
```

**卷驱动类型**：
- `local`：本地存储（默认）
- `nfs`：NFS 网络存储
- 第三方驱动：如 `rexray/ebs`（AWS EBS）

### 4.2 外部卷

```yaml
volumes:
  milvus_data:
    external: true  # 使用已存在的卷
```

**用途**：
- 在多个 Compose 项目间共享卷
- 使用预先创建的卷

### 4.3 卷标签

```yaml
volumes:
  milvus_data:
    labels:
      com.example.description: "Milvus vector data"
      com.example.department: "AI"
```

**在 RAG 开发中的应用**：

```yaml
volumes:
  milvus_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/ssd/milvus  # 使用 SSD 存储
    labels:
      project: rag-system
      environment: production

  etcd_data:
    driver: local
    labels:
      project: rag-system
      environment: production

  minio_data:
    driver: local
    labels:
      project: rag-system
      environment: production
```

---

## 5. 完整的生产级配置示例

### 5.1 单机部署配置

```yaml
version: '3.8'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - milvus
    restart: unless-stopped

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: milvus-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - milvus
    restart: unless-stopped

  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: minioadmin
      MINIO_SECRET_ACCESS_KEY: minioadmin
      LOG_LEVEL: info
      CACHE_SIZE: 8
    volumes:
      - milvus_data:/var/lib/milvus
      - ./configs/milvus.yaml:/milvus/configs/milvus.yaml:ro
      - ./logs:/var/log/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3
      start_period: 90s
    networks:
      - milvus
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

networks:
  milvus:
    driver: bridge

volumes:
  etcd_data:
    driver: local
  minio_data:
    driver: local
  milvus_data:
    driver: local
```

### 5.2 配置文件说明

**关键配置点**：

1. **健康检查**：所有服务都配置了 healthcheck
2. **依赖管理**：Milvus 等待 etcd 和 minio 健康后启动
3. **资源限制**：限制 Milvus 的 CPU 和内存使用
4. **数据持久化**：使用 Named Volumes
5. **日志管理**：将日志输出到宿主机目录
6. **重启策略**：使用 unless-stopped

---

## 6. 常用配置模式

### 6.1 开发环境配置

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  milvus:
    image: milvusdb/milvus:v2.4.0
    environment:
      LOG_LEVEL: debug
      CACHE_SIZE: 2
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ./data:/var/lib/milvus
      - ./configs:/milvus/configs
    restart: "no"
```

**特点**：
- 调试日志级别
- 较小的资源配置
- 使用 Bind Mount 方便查看数据
- 不自动重启

### 6.2 生产环境配置

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  milvus:
    image: milvusdb/milvus:v2.4.0
    environment:
      LOG_LEVEL: warn
      CACHE_SIZE: 16
    ports:
      - "127.0.0.1:19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus
      - ./configs/milvus.yaml:/milvus/configs/milvus.yaml:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

**特点**：
- 警告日志级别
- 更大的资源配置
- 使用 Named Volume
- 只允许本地访问
- 自动重启

### 6.3 使用不同配置文件

```bash
# 开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 生产环境
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 7. 配置验证和调试

### 7.1 验证配置文件

```bash
# 验证语法
docker-compose config

# 查看合并后的配置
docker-compose -f docker-compose.yml -f docker-compose.prod.yml config

# 验证服务定义
docker-compose config --services

# 验证卷定义
docker-compose config --volumes
```

### 7.2 调试技巧

```bash
# 查看服务日志
docker-compose logs milvus

# 查看环境变量
docker-compose exec milvus env

# 查看网络配置
docker network inspect <network_name>

# 查看卷配置
docker volume inspect <volume_name>

# 进入容器调试
docker-compose exec milvus bash
```

---

## 8. 在 RAG 开发中的最佳实践

### 8.1 配置文件组织

```
project/
├── docker-compose.yml          # 基础配置
├── docker-compose.dev.yml      # 开发环境覆盖
├── docker-compose.prod.yml     # 生产环境覆盖
├── .env                        # 环境变量
├── configs/
│   └── milvus.yaml            # Milvus 配置文件
└── data/                       # 数据目录（开发环境）
```

### 8.2 环境变量管理

```bash
# .env
MILVUS_VERSION=v2.4.0
ETCD_VERSION=v3.5.5
MINIO_VERSION=RELEASE.2023-03-20T20-16-18Z

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

LOG_LEVEL=info
CACHE_SIZE=8
```

```yaml
# docker-compose.yml
services:
  milvus:
    image: milvusdb/milvus:${MILVUS_VERSION}
    environment:
      LOG_LEVEL: ${LOG_LEVEL}
      CACHE_SIZE: ${CACHE_SIZE}
```

### 8.3 配置模板化

```yaml
# 使用 YAML 锚点和别名
x-common-healthcheck: &common-healthcheck
  interval: 30s
  timeout: 20s
  retries: 3

services:
  etcd:
    healthcheck:
      <<: *common-healthcheck
      test: ["CMD", "etcdctl", "endpoint", "health"]

  minio:
    healthcheck:
      <<: *common-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
```

---

## 关键要点

1. **服务定义是核心**：理解 image、environment、ports、volumes、depends_on
2. **健康检查很重要**：确保服务真正就绪后再启动依赖服务
3. **网络隔离提高安全性**：使用自定义网络和内部网络
4. **卷管理保证数据持久化**：使用 Named Volumes 而非 Bind Mounts
5. **配置分层管理**：基础配置 + 环境特定配置
6. **资源限制防止滥用**：在生产环境中配置 CPU 和内存限制
7. **重启策略提高可用性**：使用 unless-stopped
8. **环境变量集中管理**：使用 .env 文件

---

## 下一步学习

完成本节后，建议学习：
- **核心概念2**：Milvus 容器化架构
- **核心概念3**：数据持久化与卷管理
- **实战代码**：完整的部署示例
