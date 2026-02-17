# 实战代码3：生产级 Dockerfile（基础配置）

## 目标

编写生产级 Dockerfile，包含安全配置、健康检查、资源限制等最佳实践。

---

## 生产级 vs 开发级对比

| 特性 | 开发级 | 生产级 |
|-----|-------|-------|
| 用户权限 | root | 非 root ✅ |
| 健康检查 | 无 | 有 ✅ |
| 镜像体积 | 1GB | 200MB ✅ |
| 安全扫描 | 无 | 有 ✅ |
| 环境变量 | 硬编码 | 运行时注入 ✅ |
| 日志配置 | 无 | 结构化日志 ✅ |

---

## 基础配置要素

### 1. 非 root 用户

```dockerfile
# 创建用户组和用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置文件所有者
COPY --chown=appuser:appuser app/ app/

# 切换到非 root 用户
USER appuser
```

### 2. 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### 3. 环境变量

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
```

### 4. 优雅关闭

```dockerfile
# 使用 Exec 形式（确保进程 PID 为 1）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 完整生产级 Dockerfile

```dockerfile
# ===== 构建阶段 =====
FROM python:3.13-slim AS builder

WORKDIR /app

# 更新系统包（修复已知漏洞）
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装构建工具
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖（不包含开发依赖）
RUN uv sync --frozen --no-dev

# ===== 运行阶段 =====
FROM python:3.13-slim

# 更新系统包
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

# 复制应用代码并设置所有者
COPY --chown=appuser:appuser app/ app/

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info

# 切换到非 root 用户
USER appuser

# 声明端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令（使用 Exec 形式）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 配置说明

### 多阶段构建

**Builder 阶段：**
- 更新系统包修复漏洞
- 安装构建工具（uv）
- 安装运行时依赖

**Runtime 阶段：**
- 全新的基础镜像
- 只复制必要的依赖和代码
- 不包含构建工具

### 安全配置

**1. 非 root 用户**
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

**2. 更新系统包**
```dockerfile
RUN apt-get update && apt-get upgrade -y
```

**3. 最小权限**
```dockerfile
COPY --chown=appuser:appuser app/ app/
```

### 健康检查配置

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**参数说明：**
- `--interval=30s`: 每 30 秒检查一次
- `--timeout=3s`: 超时时间 3 秒
- `--start-period=5s`: 启动宽限期 5 秒
- `--retries=3`: 失败 3 次后标记为不健康

### 环境变量配置

```dockerfile
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info
```

**说明：**
- `PYTHONUNBUFFERED=1`: 不缓冲输出（实时查看日志）
- `PYTHONDONTWRITEBYTECODE=1`: 不生成 .pyc 文件
- `LOG_LEVEL=info`: 日志级别

### 启动命令配置

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**参数说明：**
- `--host 0.0.0.0`: 监听所有网络接口
- `--port 8000`: 监听端口
- `--workers 4`: 4 个工作进程（根据 CPU 核心数调整）

---

## 构建和运行

### 构建镜像

```bash
docker build -t ai-agent-api:prod .
```

### 运行容器

```bash
docker run -d \
  --name api \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/aiagent \
  -e REDIS_URL=redis://redis:6379/0 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  --restart unless-stopped \
  --memory=1g \
  --cpus=2 \
  ai-agent-api:prod
```

### 查看健康状态

```bash
# 查看容器状态
docker ps
# CONTAINER ID   STATUS
# abc123         Up 5 minutes (healthy)

# 查看健康检查日志
docker inspect --format='{{json .State.Health}}' api | jq
```

---

## 安全扫描

### 使用 Trivy 扫描镜像

```bash
# 安装 Trivy
brew install trivy

# 扫描镜像
trivy image ai-agent-api:prod

# 只显示高危和严重漏洞
trivy image --severity HIGH,CRITICAL ai-agent-api:prod
```

---

## 总结

**生产级 Dockerfile 的关键要素：**
1. ✅ 多阶段构建（减小镜像体积）
2. ✅ 非 root 用户运行
3. ✅ 健康检查配置
4. ✅ 更新系统包（修复漏洞）
5. ✅ 环境变量配置
6. ✅ 优雅关闭（Exec 形式）
7. ✅ 资源限制（--memory, --cpus）

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
