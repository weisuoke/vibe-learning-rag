# 实战代码 场景2: Compose部署

生产环境级别的 Milvus 2.6 Docker Compose 部署配置和优化实战。

---

## 场景概述

**目标**: 配置生产环境级别的 Milvus 2.6 Standalone 部署,包括资源限制、监控、备份和高可用配置。

**适用场景**:
- 生产环境部署
- 中小规模应用 (< 1 亿向量)
- 需要稳定性和可维护性

**与场景1的区别**:
- 场景1: 开发/测试环境,快速启动
- 场景2: 生产环境,完整配置

---

## 生产环境配置清单

### 必需配置

| 配置项 | 开发环境 | 生产环境 |
|--------|---------|---------|
| **资源限制** | ❌ 无 | ✅ CPU/内存限制 |
| **自动重启** | ❌ 无 | ✅ restart: always |
| **日志管理** | ❌ 无限制 | ✅ 日志轮转 |
| **数据持久化** | ✅ volumes | ✅ 独立磁盘 |
| **监控** | ❌ 无 | ✅ Prometheus |
| **备份** | ❌ 无 | ✅ 定时备份 |
| **认证** | ❌ 无 | ✅ 启用认证 |

---

## 完整生产环境配置

### docker-compose-prod.yml

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
      - /data/milvus/etcd:/etcd  # 使用独立磁盘
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: always  # 自动重启
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    networks:
      - milvus

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      # 生产环境: 使用强密码
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin}
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - /data/milvus/minio:/minio_data  # 使用独立磁盘
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    networks:
      - milvus

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.6.11
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      # 基础配置
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD:-minioadmin}

      # 性能配置
      QUERY_NODE_MEMORY_LIMIT: 17179869184  # 16GB
      DATA_NODE_MEMORY_LIMIT: 8589934592    # 8GB
      INDEX_NODE_MEMORY_LIMIT: 8589934592   # 8GB

      # 日志配置
      LOG_LEVEL: warn
      LOG_FORMAT: json

      # 认证配置 (可选)
      COMMON_SECURITY_AUTHORIZATIONENABLED: "false"  # 生产环境建议设为 true

      # 时区
      TZ: Asia/Shanghai
    volumes:
      - /data/milvus/standalone:/var/lib/milvus  # 使用独立磁盘
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
    restart: always
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
    networks:
      - milvus

networks:
  milvus:
    driver: bridge

volumes:
  etcd:
  minio:
  milvus:
```

### 环境变量配置 (.env)

```bash
# .env 文件 (生产环境)

# MinIO 认证
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_strong_password_here

# Milvus 认证 (如果启用)
MILVUS_USER=root
MILVUS_PASSWORD=your_milvus_password_here

# 数据目录
DATA_DIR=/data/milvus

# 备份目录
BACKUP_DIR=/backup/milvus

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

---

## 资源配置详解

### CPU 和内存配置

```yaml
# 小型部署 (< 1000 万向量)
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G

# 中型部署 (1000-5000 万向量)
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
    reservations:
      cpus: '4'
      memory: 8G

# 大型部署 (5000 万-1 亿向量)
deploy:
  resources:
    limits:
      cpus: '16'
      memory: 32G
    reservations:
      cpus: '8'
      memory: 16G
```

### 磁盘配置

```yaml
# 推荐使用独立磁盘
volumes:
  # SSD 磁盘 (推荐)
  - /data/milvus/standalone:/var/lib/milvus

  # 或使用 Docker volumes
  - milvus-data:/var/lib/milvus

# 磁盘空间建议
# - 小型: 100GB SSD
# - 中型: 500GB SSD
# - 大型: 1TB+ SSD
```

---

## 监控配置

### Prometheus + Grafana

创建 `docker-compose-monitoring.yml`:

```yaml
version: '3.5'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: milvus-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: always
    networks:
      - milvus

  grafana:
    image: grafana/grafana:latest
    container_name: milvus-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    restart: always
    depends_on:
      - prometheus
    networks:
      - milvus

networks:
  milvus:
    external: true

volumes:
  prometheus-data:
  grafana-data:
```

### prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus-standalone:9091']
        labels:
          instance: 'milvus-standalone'

  - job_name: 'etcd'
    static_configs:
      - targets: ['milvus-etcd:2379']
        labels:
          instance: 'milvus-etcd'

  - job_name: 'minio'
    static_configs:
      - targets: ['milvus-minio:9000']
        labels:
          instance: 'milvus-minio'
```

### 启动监控

```bash
# 启动 Milvus
docker compose -f docker-compose-prod.yml up -d

# 启动监控
docker compose -f docker-compose-monitoring.yml up -d

# 访问 Grafana
open http://localhost:3000
# 默认用户名: admin
# 默认密码: admin (或 .env 中配置的密码)
```

---

## 备份策略

### 自动备份脚本

创建 `backup.sh`:

```bash
#!/bin/bash
# Milvus 自动备份脚本

set -e

# 配置
BACKUP_DIR="${BACKUP_DIR:-/backup/milvus}"
DATA_DIR="${DATA_DIR:-/data/milvus}"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/milvus-$DATE.tar.gz"

# 创建备份目录
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "Milvus 备份开始: $DATE"
echo "=========================================="

# 1. 停止 Milvus (可选,建议在低峰期执行)
echo "[1/5] 停止 Milvus..."
docker compose -f docker-compose-prod.yml down

# 2. 备份数据
echo "[2/5] 备份数据..."
tar -czf "$BACKUP_FILE" -C "$DATA_DIR" .
echo "备份文件: $BACKUP_FILE"
echo "备份大小: $(du -h $BACKUP_FILE | cut -f1)"

# 3. 启动 Milvus
echo "[3/5] 启动 Milvus..."
docker compose -f docker-compose-prod.yml up -d

# 4. 等待服务就绪
echo "[4/5] 等待服务就绪..."
sleep 30

# 5. 验证服务
echo "[5/5] 验证服务..."
if curl -sf http://localhost:9091/healthz > /dev/null; then
    echo "✅ Milvus 服务正常"
else
    echo "❌ Milvus 服务异常"
    exit 1
fi

# 6. 清理旧备份
echo "清理旧备份 (保留 $RETENTION_DAYS 天)..."
find "$BACKUP_DIR" -name "milvus-*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "=========================================="
echo "备份完成: $DATE"
echo "=========================================="
```

### 定时备份 (crontab)

```bash
# 编辑 crontab
crontab -e

# 添加定时任务 (每天凌晨 2 点备份)
0 2 * * * /path/to/backup.sh >> /var/log/milvus_backup.log 2>&1
```

### 恢复数据

创建 `restore.sh`:

```bash
#!/bin/bash
# Milvus 数据恢复脚本

set -e

# 配置
BACKUP_FILE="$1"
DATA_DIR="${DATA_DIR:-/data/milvus}"

if [ -z "$BACKUP_FILE" ]; then
    echo "用法: $0 <backup_file>"
    echo "示例: $0 /backup/milvus/milvus-20260216_020000.tar.gz"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "错误: 备份文件不存在: $BACKUP_FILE"
    exit 1
fi

echo "=========================================="
echo "Milvus 数据恢复"
echo "=========================================="
echo "备份文件: $BACKUP_FILE"
echo "数据目录: $DATA_DIR"
echo ""
read -p "确认恢复? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "取消恢复"
    exit 0
fi

# 1. 停止 Milvus
echo "[1/4] 停止 Milvus..."
docker compose -f docker-compose-prod.yml down

# 2. 备份当前数据 (以防万一)
echo "[2/4] 备份当前数据..."
CURRENT_BACKUP="$DATA_DIR.backup.$(date +%Y%m%d_%H%M%S)"
mv "$DATA_DIR" "$CURRENT_BACKUP"
echo "当前数据已备份到: $CURRENT_BACKUP"

# 3. 恢复数据
echo "[3/4] 恢复数据..."
mkdir -p "$DATA_DIR"
tar -xzf "$BACKUP_FILE" -C "$DATA_DIR"
echo "数据恢复完成"

# 4. 启动 Milvus
echo "[4/4] 启动 Milvus..."
docker compose -f docker-compose-prod.yml up -d

echo "=========================================="
echo "恢复完成"
echo "=========================================="
echo "如果恢复失败,可以从以下位置恢复原数据:"
echo "$CURRENT_BACKUP"
```

---

## 高可用配置

### 负载均衡 (Nginx)

创建 `nginx.conf`:

```nginx
upstream milvus_backend {
    # 如果有多个 Milvus 实例
    server milvus-standalone-1:19530 max_fails=3 fail_timeout=30s;
    # server milvus-standalone-2:19530 max_fails=3 fail_timeout=30s;
}

server {
    listen 19530;

    location / {
        proxy_pass http://milvus_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # 超时配置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # 健康检查
        proxy_next_upstream error timeout http_500 http_502 http_503;
    }
}
```

### 添加 Nginx 到 docker-compose

```yaml
# docker-compose-prod.yml
services:
  nginx:
    image: nginx:latest
    container_name: milvus-nginx
    ports:
      - "19530:19530"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - standalone
    restart: always
    networks:
      - milvus
```

---

## 安全配置

### 启用认证

```yaml
# docker-compose-prod.yml
services:
  standalone:
    environment:
      # 启用认证
      COMMON_SECURITY_AUTHORIZATIONENABLED: "true"
      # 设置超级用户
      COMMON_SECURITY_SUPERUSERS: "root"
```

### 初始化认证

```python
from pymilvus import MilvusClient

# 首次连接 (使用默认密码)
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"  # 默认密码
)

# 修改密码
# 注意: pymilvus 2.6 暂不支持直接修改密码
# 需要通过 Milvus CLI 或 API 修改
```

### 防火墙配置

```bash
# Ubuntu/Debian
sudo ufw allow 19530/tcp  # Milvus gRPC
sudo ufw allow 9091/tcp   # Milvus WebUI
sudo ufw allow 9090/tcp   # Prometheus
sudo ufw allow 3000/tcp   # Grafana

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=19530/tcp
sudo firewall-cmd --permanent --add-port=9091/tcp
sudo firewall-cmd --permanent --add-port=9090/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload
```

---

## 性能优化

### 内存优化

```yaml
# docker-compose-prod.yml
services:
  standalone:
    environment:
      # 查询节点内存限制
      QUERY_NODE_MEMORY_LIMIT: 17179869184  # 16GB

      # 数据节点内存限制
      DATA_NODE_MEMORY_LIMIT: 8589934592    # 8GB

      # 索引节点内存限制
      INDEX_NODE_MEMORY_LIMIT: 8589934592   # 8GB

      # 缓存配置
      CACHE_SIZE: 4294967296  # 4GB
```

### 并发优化

```yaml
# docker-compose-prod.yml
services:
  standalone:
    environment:
      # 查询节点并发数
      QUERY_NODE_NUM_PARALLEL_TASKS: 8

      # 数据节点并发数
      DATA_NODE_NUM_PARALLEL_TASKS: 4

      # 索引节点并发数
      INDEX_NODE_NUM_PARALLEL_TASKS: 4
```

---

## 部署流程

### 1. 准备环境

```bash
# 创建项目目录
mkdir -p ~/milvus-production
cd ~/milvus-production

# 创建数据目录
sudo mkdir -p /data/milvus/{standalone,etcd,minio}
sudo mkdir -p /backup/milvus

# 设置权限
sudo chown -R $USER:$USER /data/milvus
sudo chown -R $USER:$USER /backup/milvus
```

### 2. 配置文件

```bash
# 复制配置文件
cp docker-compose-prod.yml .
cp .env.example .env
cp backup.sh .
cp restore.sh .

# 编辑 .env 文件
vim .env

# 设置执行权限
chmod +x backup.sh restore.sh
```

### 3. 启动服务

```bash
# 启动 Milvus
docker compose -f docker-compose-prod.yml up -d

# 查看日志
docker compose -f docker-compose-prod.yml logs -f

# 等待服务就绪
sleep 30

# 验证服务
curl http://localhost:9091/healthz
```

### 4. 配置监控

```bash
# 启动监控
docker compose -f docker-compose-monitoring.yml up -d

# 访问 Grafana
open http://localhost:3000
```

### 5. 配置备份

```bash
# 测试备份
./backup.sh

# 配置定时备份
crontab -e
# 添加: 0 2 * * * /path/to/backup.sh >> /var/log/milvus_backup.log 2>&1
```

---

## 健康检查脚本

创建 `health_check.sh`:

```bash
#!/bin/bash
# Milvus 健康检查脚本

set -e

echo "=========================================="
echo "Milvus 健康检查"
echo "=========================================="

# 1. 检查容器状态
echo "[1/5] 检查容器状态..."
if docker compose -f docker-compose-prod.yml ps | grep -q "Up"; then
    echo "✅ 容器运行正常"
else
    echo "❌ 容器未运行"
    exit 1
fi

# 2. 检查 HTTP healthz
echo "[2/5] 检查 HTTP healthz..."
if curl -sf http://localhost:9091/healthz > /dev/null; then
    echo "✅ HTTP healthz 正常"
else
    echo "❌ HTTP healthz 异常"
    exit 1
fi

# 3. 检查 gRPC 连接
echo "[3/5] 检查 gRPC 连接..."
if python3 -c "from pymilvus import MilvusClient; MilvusClient('http://localhost:19530').list_collections()" 2>/dev/null; then
    echo "✅ gRPC 连接正常"
else
    echo "❌ gRPC 连接失败"
    exit 1
fi

# 4. 检查磁盘空间
echo "[4/5] 检查磁盘空间..."
DISK_USAGE=$(df -h /data/milvus | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo "✅ 磁盘空间充足 ($DISK_USAGE%)"
else
    echo "⚠️  磁盘空间不足 ($DISK_USAGE%)"
fi

# 5. 检查内存使用
echo "[5/5] 检查内存使用..."
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" milvus-standalone | sed 's/%//')
if (( $(echo "$MEMORY_USAGE < 90" | bc -l) )); then
    echo "✅ 内存使用正常 ($MEMORY_USAGE%)"
else
    echo "⚠️  内存使用过高 ($MEMORY_USAGE%)"
fi

echo "=========================================="
echo "健康检查完成"
echo "=========================================="
```

---

## 故障排查

### 常见问题

#### 1. 容器频繁重启

```bash
# 查看日志
docker compose -f docker-compose-prod.yml logs --tail=100 milvus-standalone

# 检查资源限制
docker stats milvus-standalone

# 增加资源限制
# 编辑 docker-compose-prod.yml
# deploy.resources.limits.memory: 32G
```

#### 2. 性能下降

```bash
# 检查 Prometheus 指标
open http://localhost:9090

# 检查慢查询
docker compose -f docker-compose-prod.yml logs milvus-standalone | grep "slow"

# 优化索引
# 参考 07_实战代码_场景1_Standalone部署.md
```

#### 3. 磁盘空间不足

```bash
# 检查磁盘使用
df -h /data/milvus

# 清理旧数据
docker compose -f docker-compose-prod.yml exec milvus-standalone milvus compact

# 扩展磁盘
# 参考云服务商文档
```

---

## 总结

### 生产环境配置要点

1. **资源限制**: CPU/内存限制,防止资源耗尽
2. **自动重启**: restart: always,确保高可用
3. **日志管理**: 日志轮转,防止磁盘占满
4. **数据持久化**: 独立磁盘,SSD 优先
5. **监控**: Prometheus + Grafana
6. **备份**: 定时备份,快速恢复
7. **安全**: 认证、防火墙、强密码
8. **性能优化**: 内存、并发、缓存配置

### 核心命令

```bash
# 启动
docker compose -f docker-compose-prod.yml up -d

# 查看状态
docker compose -f docker-compose-prod.yml ps

# 查看日志
docker compose -f docker-compose-prod.yml logs -f

# 备份
./backup.sh

# 恢复
./restore.sh /backup/milvus/milvus-20260216_020000.tar.gz

# 健康检查
./health_check.sh

# 停止
docker compose -f docker-compose-prod.yml down
```

### 下一步

- 阅读 **07_实战代码_场景3_连接管理.md** 学习连接管理实战
- 阅读 **07_实战代码_场景4_端到端RAG.md** 学习 RAG 系统集成
- 阅读 **09_化骨绵掌.md** 深入理解原理

---

**参考文献**:
- Milvus Production Deployment: https://milvus.io/docs/configure-docker.md
- Docker Compose Best Practices: https://docs.docker.com/compose/production/
- Prometheus Monitoring: https://prometheus.io/docs/introduction/overview/
