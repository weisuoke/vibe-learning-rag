# 实战代码4：完整 AI Agent API 容器化（Docker 配置）

## Dockerfile

```dockerfile
# ===== 构建阶段 =====
FROM python:3.13-slim AS builder

WORKDIR /app

# 更新系统包
RUN apt-get update && \
    apt-get upgrade -y && \
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

# 安装运行时依赖
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir -p /app/logs && \
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
    PYTHONDONTWRITEBYTECODE=1

# 切换到非 root 用户
USER appuser

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: ai-agent-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    networks:
      - aiagent-network
    volumes:
      - api_logs:/app/logs
    environment:
      - DATABASE_URL=postgresql://aiagent:${DB_PASSWORD}@db:5432/aiagent
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4}
      - SECRET_KEY=${SECRET_KEY}
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s

  db:
    image: postgres:14-alpine
    container_name: aiagent-db
    restart: unless-stopped
    networks:
      - aiagent-network
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=aiagent
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=aiagent
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aiagent"]
      interval: 10s
      timeout: 3s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: aiagent-redis
    restart: unless-stopped
    networks:
      - aiagent-network
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
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

networks:
  aiagent-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  api_logs:
```

---

## .env.example

```bash
# 数据库配置
DB_PASSWORD=your-secure-password

# AI 配置
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# JWT 配置
SECRET_KEY=your-secret-key-here
```

---

## 启动和测试

### 1. 准备环境

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入真实的配置
vim .env
```

### 2. 构建和启动

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 3. 运行数据库迁移

```bash
# 创建迁移
docker-compose exec api alembic revision --autogenerate -m "Initial migration"

# 执行迁移
docker-compose exec api alembic upgrade head
```

### 4. 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 创建用户（示例）
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"secret"}'

# Agent 对话（示例）
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":1,"message":"Hello!"}'
```

### 5. 查看资源使用

```bash
# 查看容器状态
docker-compose ps

# 查看资源使用
docker stats
```

---

## 停止和清理

```bash
# 停止所有服务
docker-compose down

# 停止并删除 Volume（数据会丢失）
docker-compose down -v

# 删除镜像
docker rmi ai-agent-api
```

---

## 总结

**完整 AI Agent API 容器化配置：**
1. ✅ 多阶段 Dockerfile（优化镜像体积）
2. ✅ docker-compose.yml（编排 API + PostgreSQL + Redis）
3. ✅ 健康检查（所有服务）
4. ✅ 资源限制（CPU、内存）
5. ✅ 数据持久化（Volume）
6. ✅ 网络隔离（自定义网络）
7. ✅ 环境变量配置
8. ✅ 非 root 用户运行
9. ✅ 自动重启策略

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
