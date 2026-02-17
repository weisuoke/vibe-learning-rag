# 实战代码3：生产级 Dockerfile（完整示例）

## 完整的生产级配置

包含安全扫描、性能优化、监控集成的完整示例。

---

## 完整 Dockerfile

```dockerfile
# ===== 构建阶段 =====
FROM python:3.13-slim AS builder

# 设置构建参数
ARG PYTHON_VERSION=3.13
ARG APP_ENV=production

WORKDIR /app

# 更新系统包并安装构建依赖
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen --no-dev

# ===== 运行阶段 =====
FROM python:3.13-slim

# 元数据标签
LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="AI Agent API - Production Ready"

# 更新系统包并安装运行时依赖
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建非 root 用户和必要的目录
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/uploads /app/cache && \
    chown -R appuser:appuser /app

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /app/.venv /app/.venv

# 复制应用代码
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser alembic.ini .

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=info \
    WORKERS=4

# 切换到非 root 用户
USER appuser

# 声明端口
EXPOSE 8000

# 声明数据卷
VOLUME ["/app/logs", "/app/uploads"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
```

---

## docker-compose.yml（生产级）

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        APP_ENV: production
    image: ai-agent-api:${VERSION:-latest}
    container_name: ai-agent-api
    restart: unless-stopped
    ports:
      - "${API_PORT:-8000}:8000"
    networks:
      - aiagent-network
    volumes:
      - api_logs:/app/logs
      - api_uploads:/app/uploads
    environment:
      # 数据库配置
      - DATABASE_URL=${DATABASE_URL}
      - DB_POOL_SIZE=${DB_POOL_SIZE:-10}
      - DB_MAX_OVERFLOW=${DB_MAX_OVERFLOW:-20}

      # Redis 配置
      - REDIS_URL=${REDIS_URL}
      - REDIS_MAX_CONNECTIONS=${REDIS_MAX_CONNECTIONS:-50}

      # AI 配置
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4}

      # 应用配置
      - APP_ENV=production
      - LOG_LEVEL=${LOG_LEVEL:-info}
      - WORKERS=${WORKERS:-4}
      - SECRET_KEY=${SECRET_KEY}

      # 监控配置
      - SENTRY_DSN=${SENTRY_DSN}
      - PROMETHEUS_ENABLED=${PROMETHEUS_ENABLED:-true}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '1'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  db:
    image: postgres:14-alpine
    container_name: aiagent-db
    restart: unless-stopped
    networks:
      - aiagent-network
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    environment:
      - POSTGRES_USER=${DB_USER:-aiagent}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME:-aiagent}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-aiagent}"]
      interval: 10s
      timeout: 3s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  redis:
    image: redis:7-alpine
    container_name: aiagent-redis
    restart: unless-stopped
    networks:
      - aiagent-network
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  aiagent-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  api_logs:
    driver: local
  api_uploads:
    driver: local
```

---

## .env.example

```bash
# 应用配置
VERSION=1.0.0
API_PORT=8000
APP_ENV=production
LOG_LEVEL=info
WORKERS=4
SECRET_KEY=your-secret-key-here

# 数据库配置
DATABASE_URL=postgresql://aiagent:secret@db:5432/aiagent
DB_USER=aiagent
DB_PASSWORD=secret
DB_NAME=aiagent
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis 配置
REDIS_URL=redis://redis:6379/0
REDIS_MAX_CONNECTIONS=50

# AI 配置
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# 监控配置
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_ENABLED=true
```

---

## 部署脚本

### deploy.sh

```bash
#!/bin/bash
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== AI Agent API 部署脚本 ===${NC}"

# 检查 .env 文件
if [ ! -f .env ]; then
    echo -e "${RED}错误: .env 文件不存在${NC}"
    echo "请复制 .env.example 并配置环境变量"
    exit 1
fi

# 加载环境变量
source .env

# 构建镜像
echo -e "${YELLOW}1. 构建 Docker 镜像...${NC}"
docker-compose build --no-cache

# 运行安全扫描
echo -e "${YELLOW}2. 运行安全扫描...${NC}"
trivy image ai-agent-api:${VERSION:-latest} --severity HIGH,CRITICAL

# 停止旧容器
echo -e "${YELLOW}3. 停止旧容器...${NC}"
docker-compose down

# 启动新容器
echo -e "${YELLOW}4. 启动新容器...${NC}"
docker-compose up -d

# 等待服务启动
echo -e "${YELLOW}5. 等待服务启动...${NC}"
sleep 10

# 检查健康状态
echo -e "${YELLOW}6. 检查健康状态...${NC}"
if curl -f http://localhost:${API_PORT:-8000}/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 服务启动成功${NC}"
else
    echo -e "${RED}✗ 服务启动失败${NC}"
    docker-compose logs api
    exit 1
fi

# 运行数据库迁移
echo -e "${YELLOW}7. 运行数据库迁移...${NC}"
docker-compose exec api alembic upgrade head

echo -e "${GREEN}=== 部署完成 ===${NC}"
echo "API 地址: http://localhost:${API_PORT:-8000}"
echo "API 文档: http://localhost:${API_PORT:-8000}/docs"
echo "健康检查: http://localhost:${API_PORT:-8000}/health"
```

---

## 监控和日志

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看 API 日志
docker-compose logs -f api

# 查看最近 100 行日志
docker-compose logs --tail=100 api
```

### 查看资源使用

```bash
# 查看所有容器资源使用
docker stats

# 查看 API 容器资源使用
docker stats ai-agent-api
```

### 查看健康状态

```bash
# 查看所有容器状态
docker-compose ps

# 查看 API 健康检查详情
docker inspect --format='{{json .State.Health}}' ai-agent-api | jq
```

---

## 备份和恢复

### 备份数据库

```bash
# 备份数据库
docker-compose exec db pg_dump -U aiagent aiagent > backup_$(date +%Y%m%d_%H%M%S).sql

# 备份到容器外
docker-compose exec db pg_dump -U aiagent aiagent | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

### 恢复数据库

```bash
# 恢复数据库
docker-compose exec -T db psql -U aiagent aiagent < backup.sql

# 从压缩文件恢复
gunzip < backup.sql.gz | docker-compose exec -T db psql -U aiagent aiagent
```

---

## 总结

**生产级配置的完整要素：**
1. ✅ 多阶段构建优化镜像体积
2. ✅ 非 root 用户运行
3. ✅ 健康检查和自动重启
4. ✅ 资源限制（CPU、内存）
5. ✅ 日志管理（限制大小和数量）
6. ✅ 环境变量配置
7. ✅ 数据持久化（Volume）
8. ✅ 网络隔离
9. ✅ 安全扫描
10. ✅ 部署脚本自动化

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
