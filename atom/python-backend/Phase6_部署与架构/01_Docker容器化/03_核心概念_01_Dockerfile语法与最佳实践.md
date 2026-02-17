# 核心概念1：Dockerfile 语法与最佳实践

## 概述

Dockerfile 是构建 Docker 镜像的指令脚本，定义了如何从基础镜像构建应用镜像。

**类比：** 前端的 webpack.config.js / 日常的装修图纸

---

## 1. FROM - 指定基础镜像

**作用：** 指定构建的基础镜像，必须是 Dockerfile 的第一条指令（除了 ARG）。

### 基础语法

```dockerfile
FROM <image>[:<tag>] [AS <name>]
```

### 常用基础镜像

```dockerfile
# Python 官方镜像
FROM python:3.13              # 完整版（1GB）
FROM python:3.13-slim         # 精简版（200MB）✅ 推荐
FROM python:3.13-alpine       # 最小版（150MB）⚠️ 兼容性问题

# 多阶段构建
FROM python:3.13-slim AS builder  # 构建阶段
FROM python:3.13-slim             # 运行阶段
```

### 最佳实践

**1. 使用官方镜像**
```dockerfile
# ✅ 推荐：官方镜像
FROM python:3.13-slim

# ❌ 不推荐：非官方镜像
FROM some-random-user/python:3.13
```

**2. 固定版本标签**
```dockerfile
# ✅ 推荐：固定版本
FROM python:3.13.1-slim

# ❌ 不推荐：latest 标签（版本不确定）
FROM python:latest
```

**3. 选择合适的变体**
```dockerfile
# AI Agent API 推荐：slim 版本
FROM python:3.13-slim
# 优势：体积小、兼容性好、构建快
```

### 实际应用

```dockerfile
# AI Agent API 基础镜像
FROM python:3.13-slim AS builder
WORKDIR /app
# ... 构建步骤

FROM python:3.13-slim
WORKDIR /app
# ... 运行步骤
```

---

## 2. WORKDIR - 设置工作目录

**作用：** 设置后续指令的工作目录，如果目录不存在会自动创建。

### 基础语法

```dockerfile
WORKDIR <path>
```

### 示例

```dockerfile
FROM python:3.13-slim
WORKDIR /app  # 设置工作目录为 /app

# 后续的 COPY、RUN、CMD 都在 /app 目录下执行
COPY requirements.txt .  # 复制到 /app/requirements.txt
RUN pip install -r requirements.txt  # 在 /app 目录下执行
```

### 最佳实践

**1. 使用绝对路径**
```dockerfile
# ✅ 推荐：绝对路径
WORKDIR /app

# ❌ 不推荐：相对路径（容易混淆）
WORKDIR app
```

**2. 避免使用 cd**
```dockerfile
# ✅ 推荐：使用 WORKDIR
WORKDIR /app
RUN pip install -r requirements.txt

# ❌ 不推荐：使用 cd（不会持久化）
RUN cd /app && pip install -r requirements.txt
```

**3. 多次使用 WORKDIR**
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app/src  # 切换到子目录
COPY src/ .
```

---

## 3. COPY - 复制文件到镜像

**作用：** 从构建上下文复制文件或目录到镜像。

### 基础语法

```dockerfile
COPY [--chown=<user>:<group>] <src>... <dest>
```

### 示例

```dockerfile
# 复制单个文件
COPY requirements.txt /app/

# 复制多个文件
COPY requirements.txt pyproject.toml /app/

# 复制目录
COPY app/ /app/

# 复制所有文件
COPY . /app/

# 设置文件所有者
COPY --chown=appuser:appuser app/ /app/
```

### 最佳实践

**1. 利用层缓存**
```dockerfile
# ✅ 推荐：先复制依赖文件，再复制代码
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# 代码修改后，依赖层被缓存，不需要重新安装

# ❌ 不推荐：一次性复制所有文件
COPY . .
RUN pip install -r requirements.txt
# 代码修改后，依赖层失效，需要重新安装
```

**2. 使用 .dockerignore**
```dockerfile
# .dockerignore
__pycache__/
*.pyc
.git/
.env
.venv/
node_modules/
```

```dockerfile
# Dockerfile
COPY . /app/
# 会自动忽略 .dockerignore 中的文件
```

**3. 复制特定文件**
```dockerfile
# ✅ 推荐：只复制需要的文件
COPY app/ /app/app/
COPY requirements.txt /app/

# ❌ 不推荐：复制所有文件（包含不必要的文件）
COPY . /app/
```

---

## 4. ADD - 复制文件（支持 URL 和自动解压）

**作用：** 类似 COPY，但支持 URL 和自动解压 tar 文件。

### 基础语法

```dockerfile
ADD <src>... <dest>
```

### 示例

```dockerfile
# 复制本地文件（与 COPY 相同）
ADD requirements.txt /app/

# 从 URL 下载文件
ADD https://example.com/file.tar.gz /app/

# 自动解压 tar 文件
ADD archive.tar.gz /app/
# 会自动解压到 /app/ 目录
```

### COPY vs ADD

| 特性 | COPY | ADD |
|-----|------|-----|
| 复制本地文件 | ✅ | ✅ |
| 从 URL 下载 | ❌ | ✅ |
| 自动解压 tar | ❌ | ✅ |
| 推荐使用 | ✅ | ⚠️ |

### 最佳实践

**推荐使用 COPY，除非需要 ADD 的特殊功能**

```dockerfile
# ✅ 推荐：使用 COPY
COPY requirements.txt /app/

# ⚠️ 只在需要时使用 ADD
ADD https://example.com/model.tar.gz /app/models/
```

---

## 5. RUN - 执行命令（构建时）

**作用：** 在镜像构建时执行命令，每个 RUN 指令创建一个新的镜像层。

### 基础语法

```dockerfile
# Shell 形式
RUN <command>

# Exec 形式
RUN ["executable", "param1", "param2"]
```

### 示例

```dockerfile
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装系统包
RUN apt-get update && apt-get install -y curl

# 创建目录
RUN mkdir -p /app/logs

# 多条命令
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

### 最佳实践

**1. 合并 RUN 指令减少层数**
```dockerfile
# ✅ 推荐：合并多条命令
RUN apt-get update && \
    apt-get install -y curl vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ❌ 不推荐：多个 RUN 指令（增加层数）
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get clean
```

**2. 清理缓存**
```dockerfile
# ✅ 推荐：清理 apt 缓存
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ✅ 推荐：清理 pip 缓存
RUN pip install --no-cache-dir -r requirements.txt
```

**3. 使用 && 连接命令**
```dockerfile
# ✅ 推荐：使用 && 确保前一条命令成功
RUN apt-get update && apt-get install -y curl

# ❌ 不推荐：使用 ; 分隔（前一条失败也会继续）
RUN apt-get update ; apt-get install -y curl
```

---

## 6. CMD - 启动命令（运行时）

**作用：** 指定容器启动时执行的命令，一个 Dockerfile 只能有一个 CMD。

### 基础语法

```dockerfile
# Exec 形式（推荐）
CMD ["executable", "param1", "param2"]

# Shell 形式
CMD command param1 param2
```

### 示例

```dockerfile
# FastAPI 应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 使用 shell 形式
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000

# 使用环境变量
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

### CMD vs ENTRYPOINT

| 特性 | CMD | ENTRYPOINT |
|-----|-----|-----------|
| 可被 docker run 覆盖 | ✅ | ❌ |
| 推荐使用场景 | 默认命令 | 固定命令 |

### 最佳实践

**1. 使用 Exec 形式**
```dockerfile
# ✅ 推荐：Exec 形式（不启动 shell）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]

# ❌ 不推荐：Shell 形式（启动 shell，PID 不是 1）
CMD uvicorn app.main:app --host 0.0.0.0
```

**2. 监听 0.0.0.0**
```dockerfile
# ✅ 推荐：监听所有网络接口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]

# ❌ 不推荐：只监听 localhost（宿主机无法访问）
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1"]
```

---

## 7. ENTRYPOINT - 固定入口点

**作用：** 指定容器启动时的固定入口点，不会被 docker run 覆盖。

### 基础语法

```dockerfile
ENTRYPOINT ["executable", "param1", "param2"]
```

### 示例

```dockerfile
# 固定入口点
ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

# docker run 时可以覆盖 CMD
docker run my-app --host 0.0.0.0 --port 9000
# 实际执行：uvicorn app.main:app --host 0.0.0.0 --port 9000
```

### ENTRYPOINT + CMD 组合

```dockerfile
# ENTRYPOINT 定义固定命令
ENTRYPOINT ["python", "app.py"]

# CMD 定义默认参数
CMD ["--mode", "production"]

# docker run my-app
# 执行：python app.py --mode production

# docker run my-app --mode development
# 执行：python app.py --mode development
```

### 最佳实践

**使用 CMD 而非 ENTRYPOINT（除非需要固定入口点）**

```dockerfile
# ✅ 推荐：使用 CMD（灵活）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]

# ⚠️ 只在需要时使用 ENTRYPOINT
ENTRYPOINT ["python", "app.py"]
CMD ["--mode", "production"]
```

---

## 8. ENV - 设置环境变量

**作用：** 设置环境变量，在构建和运行时都可用。

### 基础语法

```dockerfile
ENV <key>=<value> ...
```

### 示例

```dockerfile
# 设置单个环境变量
ENV PYTHONUNBUFFERED=1

# 设置多个环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info

# 设置 PATH
ENV PATH="/app/.venv/bin:$PATH"
```

### 最佳实践

**1. 设置 Python 环境变量**
```dockerfile
# ✅ 推荐：设置 Python 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# PYTHONUNBUFFERED=1: 不缓冲输出（实时查看日志）
# PYTHONDONTWRITEBYTECODE=1: 不生成 .pyc 文件
```

**2. 不要硬编码敏感信息**
```dockerfile
# ❌ 错误：硬编码 API Key
ENV OPENAI_API_KEY=sk-1234567890

# ✅ 正确：运行时传递
# docker run -e OPENAI_API_KEY=${OPENAI_API_KEY} my-app
```

---

## 9. ARG - 构建参数

**作用：** 定义构建时的参数，只在构建时可用。

### 基础语法

```dockerfile
ARG <name>[=<default value>]
```

### 示例

```dockerfile
# 定义构建参数
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ARG APP_ENV=production
RUN echo "Building for ${APP_ENV}"
```

### ARG vs ENV

| 特性 | ARG | ENV |
|-----|-----|-----|
| 构建时可用 | ✅ | ✅ |
| 运行时可用 | ❌ | ✅ |
| 可被 docker build 覆盖 | ✅ | ❌ |

### 最佳实践

```dockerfile
# 使用 ARG 定义构建参数
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# 使用 ENV 定义运行时环境变量
ENV PYTHONUNBUFFERED=1

# 构建时覆盖 ARG
docker build --build-arg PYTHON_VERSION=3.12 -t my-app .
```

---

## 10. EXPOSE - 声明端口

**作用：** 声明容器监听的端口（仅文档作用，不实际映射）。

### 基础语法

```dockerfile
EXPOSE <port> [<port>/<protocol>...]
```

### 示例

```dockerfile
# 声明 HTTP 端口
EXPOSE 8000

# 声明多个端口
EXPOSE 8000 5678

# 声明 TCP/UDP 端口
EXPOSE 8000/tcp
EXPOSE 53/udp
```

### 最佳实践

```dockerfile
# ✅ 推荐：声明应用监听的端口
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 运行时映射端口
docker run -p 8000:8000 my-app
```

---

## 11. USER - 切换用户

**作用：** 切换后续指令的执行用户。

### 基础语法

```dockerfile
USER <user>[:<group>]
```

### 示例

```dockerfile
# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 切换到非 root 用户
USER appuser

# 后续指令以 appuser 身份执行
WORKDIR /app
CMD ["uvicorn", "app.main:app"]
```

### 最佳实践

**生产环境使用非 root 用户**

```dockerfile
FROM python:3.13-slim

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制文件并设置所有者
COPY --chown=appuser:appuser . .

# 切换到非 root 用户
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

---

## 12. .dockerignore - 忽略文件

**作用：** 指定构建时忽略的文件和目录，减小构建上下文。

### 示例

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.env
.venv
venv/
.git/
.gitignore
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
node_modules/
.DS_Store
```

### 最佳实践

```dockerfile
# Dockerfile
COPY . /app/
# 会自动忽略 .dockerignore 中的文件
```

---

## 13. 层缓存机制

**原理：** Docker 构建镜像时，每条指令创建一个层，如果指令和上下文没有变化，会复用缓存。

### 示例

```dockerfile
# 第一次构建
FROM python:3.13-slim
COPY requirements.txt .
RUN pip install -r requirements.txt  # 执行，创建层 A
COPY . .                             # 执行，创建层 B
# 构建时间：2分钟

# 修改代码后第二次构建
FROM python:3.13-slim
COPY requirements.txt .
RUN pip install -r requirements.txt  # 缓存命中，复用层 A
COPY . .                             # 重新执行，创建层 C
# 构建时间：5秒
```

### 最佳实践

**1. 先复制依赖文件，再复制代码**
```dockerfile
# ✅ 推荐：利用缓存
COPY requirements.txt .
RUN pip install -r requirements.txt  # 依赖不变时，缓存命中
COPY . .                             # 代码修改，只重新执行这一步

# ❌ 不推荐：缓存失效
COPY . .                             # 代码修改，缓存失效
RUN pip install -r requirements.txt  # 每次都重新安装依赖
```

**2. 合并不常变化的指令**
```dockerfile
# ✅ 推荐：合并系统包安装
RUN apt-get update && \
    apt-get install -y curl vim && \
    apt-get clean
# 系统包不常变化，缓存命中率高
```

---

## 14. 指令顺序优化

**原则：** 不常变化的指令放前面，常变化的指令放后面。

### 最佳实践

```dockerfile
# ✅ 推荐：优化指令顺序
FROM python:3.13-slim                # 1. 基础镜像（几乎不变）
WORKDIR /app                         # 2. 工作目录（不变）
COPY requirements.txt .              # 3. 依赖文件（偶尔变化）
RUN pip install -r requirements.txt  # 4. 安装依赖（偶尔变化）
COPY . .                             # 5. 应用代码（经常变化）
CMD ["uvicorn", "app.main:app"]      # 6. 启动命令（不变）
```

---

## 完整示例：AI Agent API Dockerfile

```dockerfile
# ===== 构建阶段 =====
FROM python:3.13-slim AS builder

# 设置工作目录
WORKDIR /app

# 安装构建工具
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen --no-dev

# ===== 运行阶段 =====
FROM python:3.13-slim

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /app/.venv /app/.venv

# 复制应用代码
COPY --chown=appuser:appuser app/ app/

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 切换到非 root 用户
USER appuser

# 声明端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 总结

**Dockerfile 核心指令：**

| 指令 | 作用 | 使用时机 |
|-----|------|---------|
| FROM | 指定基础镜像 | 必须，第一条指令 |
| WORKDIR | 设置工作目录 | 推荐使用 |
| COPY | 复制文件 | 常用 |
| ADD | 复制文件（支持 URL） | 特殊场景 |
| RUN | 执行命令（构建时） | 安装依赖 |
| CMD | 启动命令（运行时） | 必须 |
| ENTRYPOINT | 固定入口点 | 特殊场景 |
| ENV | 设置环境变量 | 推荐使用 |
| ARG | 构建参数 | 可选 |
| EXPOSE | 声明端口 | 推荐使用 |
| USER | 切换用户 | 生产环境必须 |

**最佳实践总结：**
1. 使用官方基础镜像（python:3.13-slim）
2. 固定版本标签
3. 利用层缓存（先复制依赖文件）
4. 合并 RUN 指令减少层数
5. 清理缓存（--no-cache-dir）
6. 使用 .dockerignore 减小构建上下文
7. 生产环境使用非 root 用户
8. 不要硬编码敏感信息
9. 优化指令顺序（不常变化的放前面）
10. 使用多阶段构建（下一节）

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
