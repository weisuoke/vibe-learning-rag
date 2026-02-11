# 核心概念2：Milvus 容器化架构

> 深入理解 Milvus 的单机部署和分布式部署架构，以及容器间的通信机制

---

## 概述

Milvus 提供两种部署模式：
- **Standalone（单机模式）**：所有组件在一个进程中，适合开发和小规模生产
- **Distributed（分布式模式）**：每个组件独立部署，适合大规模生产

---

## 1. Milvus 架构概览

### 1.1 核心组件

Milvus 采用微服务架构，包含以下核心组件：

```
┌─────────────────────────────────────────────────┐
│              Milvus 架构                         │
├─────────────────────────────────────────────────┤
│  接入层                                          │
│  ┌──────────┐                                   │
│  │  Proxy   │  ← 客户端请求入口                 │
│  └──────────┘                                   │
├─────────────────────────────────────────────────┤
│  协调层（Coordinator）                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │RootCoord │ │DataCoord │ │QueryCoord│        │
│  │          │ │          │ │          │        │
│  │IndexCoord│ │          │ │          │        │
│  └──────────┘ └──────────┘ └──────────┘        │
├─────────────────────────────────────────────────┤
│  工作层（Worker）                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │DataNode  │ │QueryNode │ │IndexNode │        │
│  └──────────┘ └──────────┘ └──────────┘        │
├─────────────────────────────────────────────────┤
│  存储层                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  etcd    │ │  MinIO   │ │  Pulsar  │        │
│  │(元数据)  │ │(对象存储)│ │(消息队列)│        │
│  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────┘
```

### 1.2 组件职责

| 组件 | 职责 | 容器化方式 |
|------|------|-----------|
| **Proxy** | 接收客户端请求，路由到对应的 Coordinator | 独立容器 |
| **RootCoord** | 管理 Collection、Partition 等元数据 | 独立容器 |
| **DataCoord** | 管理数据节点，协调数据持久化 | 独立容器 |
| **QueryCoord** | 管理查询节点，协调查询任务 | 独立容器 |
| **IndexCoord** | 管理索引构建任务 | 独立容器 |
| **DataNode** | 执行数据插入和持久化 | 独立容器（可多副本）|
| **QueryNode** | 执行向量检索 | 独立容器（可多副本）|
| **IndexNode** | 执行索引构建 | 独立容器（可多副本）|
| **etcd** | 存储元数据 | 独立容器 |
| **MinIO** | 存储向量数据和索引文件 | 独立容器 |
| **Pulsar** | 消息队列（分布式模式） | 独立容器 |

---

## 2. 单机部署架构（Standalone）

### 2.1 架构图

```
┌─────────────────────────────────────────────────┐
│  Docker Compose 环境                             │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │  Milvus Standalone 容器                    │ │
│  │                                            │ │
│  │  ┌──────────────────────────────────────┐ │ │
│  │  │  Milvus 进程（All-in-One）           │ │ │
│  │  │                                      │ │ │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐ │ │ │
│  │  │  │ Proxy  │  │RootCrd │  │DataCrd │ │ │ │
│  │  │  └────────┘  └────────┘  └────────┘ │ │ │
│  │  │                                      │ │ │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐ │ │ │
│  │  │  │QueryCrd│  │IndexCrd│  │DataNode│ │ │ │
│  │  │  └────────┘  └────────┘  └────────┘ │ │ │
│  │  │                                      │ │ │
│  │  │  ┌────────┐  ┌────────┐            │ │ │
│  │  │  │QryNode │  │IdxNode │            │ │ │
│  │  │  └────────┘  └────────┘            │ │ │
│  │  └──────────────────────────────────────┘ │ │
│  │                                            │ │
│  │  端口映射：                                 │ │
│  │  - 19530:19530 (gRPC)                     │ │
│  │  - 9091:9091 (Metrics)                    │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────┐  ┌────────────────┐        │
│  │  etcd 容器     │  │  MinIO 容器    │        │
│  │                │  │                │        │
│  │  端口：2379    │  │  端口：9000    │        │
│  └────────────────┘  └────────────────┘        │
│         ↑                    ↑                  │
│         └────────┬───────────┘                  │
│                  │                              │
│         Docker 网络（milvus）                   │
└─────────────────────────────────────────────────┘
```

### 2.2 Docker Compose 配置

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
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - milvus

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: milvus-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    networks:
      - milvus

  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    networks:
      - milvus

networks:
  milvus:
    driver: bridge

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

### 2.3 单机模式特点

**优点**：
- ✅ 部署简单，只需 3 个容器
- ✅ 资源占用小（2-4GB 内存）
- ✅ 适合开发和测试
- ✅ 适合小规模生产（QPS < 1000）

**缺点**：
- ❌ 无法水平扩展
- ❌ 单点故障风险
- ❌ 性能受限于单机资源

**适用场景**：
- 本地开发环境
- 小规模 RAG 应用（< 100万向量）
- 原型验证和测试

---

## 3. 分布式部署架构（Distributed）

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│  Docker Compose 环境（分布式）                               │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Proxy   │  │  Proxy   │  │RootCoord │  │DataCoord │   │
│  │ 容器 1   │  │ 容器 2   │  │  容器    │  │  容器    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓             ↓              ↓             ↓          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │QueryCoord│  │IndexCoord│  │DataNode 1│  │DataNode 2│   │
│  │  容器    │  │  容器    │  │  容器    │  │  容器    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓             ↓              ↓             ↓          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │QueryNode1│  │QueryNode2│  │IndexNode1│  │IndexNode2│   │
│  │  容器    │  │  容器    │  │  容器    │  │  容器    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │etcd Cluster│  │Pulsar Clstr│  │MinIO Clstr │           │
│  │  (3节点)   │  │  (3节点)   │  │  (4节点)   │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                                                              │
│         Docker 网络（milvus）                                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Docker Compose 配置（简化版）

```yaml
version: '3.8'

services:
  # 存储层
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    # ... etcd 配置

  pulsar:
    image: apachepulsar/pulsar:2.10.0
    command: bin/pulsar standalone
    # ... pulsar 配置

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    # ... minio 配置

  # 协调层
  rootcoord:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "rootcoord"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      PULSAR_ADDRESS: pulsar://pulsar:6650
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - pulsar
      - minio

  datacoord:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "datacoord"]
    # ... 类似配置

  querycoord:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "querycoord"]
    # ... 类似配置

  indexcoord:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "indexcoord"]
    # ... 类似配置

  # 工作层
  datanode:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "datanode"]
    deploy:
      replicas: 2  # 2 个副本
    # ... 配置

  querynode:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "querynode"]
    deploy:
      replicas: 2  # 2 个副本
    # ... 配置

  indexnode:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "indexnode"]
    deploy:
      replicas: 2  # 2 个副本
    # ... 配置

  # 接入层
  proxy:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "proxy"]
    ports:
      - "19530:19530"
    deploy:
      replicas: 2  # 2 个副本
    depends_on:
      - rootcoord
      - datacoord
      - querycoord
      - indexcoord
```

### 3.3 分布式模式特点

**优点**：
- ✅ 可水平扩展（增加 Worker 节点）
- ✅ 高可用（组件多副本）
- ✅ 性能强（分布式计算）
- ✅ 适合大规模生产（QPS > 10000）

**缺点**：
- ❌ 部署复杂（10+ 个容器）
- ❌ 资源占用大（16GB+ 内存）
- ❌ 运维成本高

**适用场景**：
- 大规模 RAG 应用（> 1000万向量）
- 高并发场景（QPS > 1000）
- 需要高可用的生产环境

---

## 4. 容器间通信机制

### 4.1 服务发现

Docker Compose 提供内置的 DNS 服务发现：

```yaml
services:
  milvus:
    environment:
      ETCD_ENDPOINTS: etcd:2379  # 通过服务名访问
      MINIO_ADDRESS: minio:9000
```

**原理**：
```
Milvus 容器内部：
1. 解析 "etcd" → Docker DNS
2. Docker DNS 返回 etcd 容器的 IP
3. Milvus 连接到 etcd 容器
```

### 4.2 网络隔离

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # 内部网络，不能访问外部

services:
  proxy:
    networks:
      - frontend  # 可以被外部访问
      - backend   # 可以访问后端服务

  milvus:
    networks:
      - backend   # 只能在内部网络访问
```

### 4.3 通信流程

**客户端请求流程**：

```
客户端
  ↓ gRPC (19530)
Proxy 容器
  ↓ 内部网络
RootCoord 容器（获取 Collection 信息）
  ↓ 内部网络
QueryCoord 容器（分配查询任务）
  ↓ 内部网络
QueryNode 容器（执行向量检索）
  ↓ 内部网络
MinIO 容器（读取向量数据）
  ↓ 内部网络
QueryNode 容器（返回结果）
  ↓ 内部网络
Proxy 容器
  ↓ gRPC
客户端
```

---

## 5. 单机 vs 分布式对比

| 维度 | 单机模式 | 分布式模式 |
|------|---------|-----------|
| **容器数量** | 3 个 | 10+ 个 |
| **内存占用** | 2-4GB | 16GB+ |
| **部署复杂度** | 简单 | 复杂 |
| **扩展性** | 无法扩展 | 可水平扩展 |
| **高可用** | 单点故障 | 多副本容错 |
| **性能** | 单机性能 | 分布式性能 |
| **适用场景** | 开发/测试/小规模 | 大规模生产 |
| **QPS** | < 1000 | > 10000 |
| **向量规模** | < 100万 | > 1000万 |

---

## 6. 在 RAG 开发中的应用

### 6.1 开发环境：单机模式

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    # ... 最小配置

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    # ... 最小配置

  milvus:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
    # ... 开发配置
```

**优势**：
- 快速启动（30 秒）
- 资源占用小
- 方便调试

### 6.2 生产环境：分布式模式

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # 完整的分布式配置
  # 包含多副本、资源限制、健康检查
```

**优势**：
- 高性能（支持 10000+ QPS）
- 高可用（组件多副本）
- 可扩展（增加 Worker 节点）

### 6.3 迁移路径

```
开发阶段：单机模式
    ↓
测试阶段：单机模式（生产配置）
    ↓
小规模生产：单机模式
    ↓
大规模生产：分布式模式
```

---

## 7. 容器化最佳实践

### 7.1 资源配置

```yaml
services:
  milvus:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### 7.2 健康检查

```yaml
services:
  milvus:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
```

### 7.3 日志管理

```yaml
services:
  milvus:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
```

### 7.4 网络优化

```yaml
networks:
  milvus:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 9000  # Jumbo Frames
```

---

## 关键要点

1. **单机模式适合开发和小规模生产**
   - 3 个容器（Milvus + etcd + MinIO）
   - 所有组件在一个 Milvus 进程中

2. **分布式模式适合大规模生产**
   - 10+ 个容器（每个组件独立）
   - 可水平扩展和高可用

3. **容器间通过 Docker 网络通信**
   - 使用服务名进行服务发现
   - 内部网络隔离提高安全性

4. **从单机到分布式的迁移路径清晰**
   - 开发阶段使用单机模式
   - 生产阶段根据规模选择模式

5. **容器化提供一致的部署体验**
   - 相同的配置文件
   - 相同的管理命令
   - 环境一致性

---

## 下一步学习

完成本节后，建议学习：
- **核心概念3**：数据持久化与卷管理
- **实战代码**：单机部署和分布式部署示例
