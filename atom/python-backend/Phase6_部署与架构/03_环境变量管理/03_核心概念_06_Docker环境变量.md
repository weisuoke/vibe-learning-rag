# 核心概念6：Docker 环境变量

## 一句话定义

**Docker 环境变量是在容器启动时注入到容器内部的配置参数，让同一个镜像可以在不同环境中使用不同的配置而无需重新构建镜像。**

---

## 为什么需要 Docker 环境变量？

### 问题场景

在没有环境变量之前，你需要为每个环境构建不同的镜像：

```dockerfile
# ❌ 错误：配置硬编码在镜像中
FROM python:3.13
COPY . /app
WORKDIR /app

# 硬编码配置
ENV DATABASE_URL=postgresql://localhost:5432/mydb
ENV DEBUG=True

CMD ["uvicorn", "app.main:app"]
```

**问题：**
- 开发、测试、生产环境需要不同的镜像
- 修改配置需要重新构建镜像
- 敏感信息（密钥）被打包到镜像中
- 镜像无法复用

### Docker 环境变量的解决方案

```dockerfile
# ✅ 正确：配置通过环境变量注入
FROM python:3.13
COPY . /app
WORKDIR /app

# 不硬编码配置
CMD ["uvicorn", "app.main:app"]
```

```bash
# 启动时注入环境变量
docker run -e DATABASE_URL=postgresql://db:5432/mydb \
           -e DEBUG=False \
           my-app
```

**好处：**
- 同一个镜像可以在不同环境中使用
- 修改配置不需要重新构建镜像
- 敏感信息不打包到镜像中
- 镜像可以复用

---

## Docker 环境变量的注入方式

### 方式1：docker run -e

**最简单的方式，适合少量环境变量**

```bash
# 单个环境变量
docker run -e DATABASE_URL=postgresql://db:5432/mydb my-app

# 多个环境变量
docker run \
  -e DATABASE_URL=postgresql://db:5432/mydb \
  -e OPENAI_API_KEY=sk-xxx \
  -e DEBUG=False \
  my-app
```

---

### 方式2：docker run --env-file

**从文件读取环境变量，适合大量环境变量**

```bash
# .env.prod
DATABASE_URL=postgresql://prod-db:5432/prod_db
OPENAI_API_KEY=sk-prod-xxx
DEBUG=False
SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx
```

```bash
# 从文件加载环境变量
docker run --env-file .env.prod my-app
```

---

### 方式3：Dockerfile ENV

**在 Dockerfile 中设置默认值**

```dockerfile
FROM python:3.13
COPY . /app
WORKDIR /app

# 设置默认环境变量
ENV DEBUG=False
ENV LOG_LEVEL=INFO
ENV PORT=8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

**注意：** Dockerfile 中的 ENV 会被 `docker run -e` 覆盖。

---

### 方式4：docker-compose environment

**在 docker-compose.yml 中直接定义**

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
      - OPENAI_API_KEY=sk-xxx
      - DEBUG=False
    ports:
      - "8000:8000"
```

---

### 方式5：docker-compose env_file

**从文件读取环境变量**

```yaml
# docker-compose.yml
services:
  api:
    build: .
    env_file:
      - .env.prod
    ports:
      - "8000:8000"
```

---

### 方式6：docker-compose 变量替换

**从宿主机环境变量读取**

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=${DEBUG:-False}  # 默认值
    ports:
      - "8000:8000"
```

```bash
# 宿主机上设置环境变量
export DATABASE_URL=postgresql://prod-db:5432/prod_db
export OPENAI_API_KEY=sk-prod-xxx

# 启动容器
docker-compose up -d
```

---

## Docker 环境变量的优先级

### 优先级规则

**从高到低：**

```
1. docker run -e（最高优先级）
   ↓
2. docker-compose environment
   ↓
3. docker-compose env_file
   ↓
4. Dockerfile ENV
   ↓
5. 应用默认值（最低优先级）
```

### 优先级验证

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DEBUG=False  # 优先级：中等
    env_file:
      - .env  # 优先级：较低
```

```dockerfile
# Dockerfile
ENV DEBUG=True  # 优先级：最低
```

```bash
# docker run -e（优先级：最高）
docker-compose run -e DEBUG=True api

# 最终 DEBUG=True（docker run -e 覆盖所有）
```

---

## 实际应用示例

### 示例1：FastAPI 应用的 Docker 部署

**Dockerfile：**

```dockerfile
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=False
    ports:
      - "8000:8000"
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**启动：**

```bash
# 设置宿主机环境变量
export OPENAI_API_KEY=sk-xxx

# 启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f api
```

---

### 示例2：多环境部署

**docker-compose.dev.yml（开发环境）：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - ENV=dev
      - DEBUG=True
      - LOG_LEVEL=DEBUG
    env_file:
      - .env.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app  # 挂载代码，支持热重载
    command: uvicorn app.main:app --host 0.0.0.0 --reload
```

**docker-compose.prod.yml（生产环境）：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - ENV=prod
      - DEBUG=False
      - LOG_LEVEL=WARNING
    env_file:
      - .env.prod
    ports:
      - "80:8000"
    restart: always
    # 不挂载代码
```

**使用：**

```bash
# 开发环境
docker-compose -f docker-compose.dev.yml up

# 生产环境
docker-compose -f docker-compose.prod.yml up -d
```

---

### 示例3：数据库连接配置

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      # 使用服务名作为主机名
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**注意：** 在 Docker 网络中，服务名（如 `db`、`redis`）可以作为主机名使用。

---

### 示例4：敏感信息管理

**使用 Docker Secrets（Swarm 模式）：**

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: my-api
    secrets:
      - openai_api_key
      - secret_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
      - SECRET_KEY_FILE=/run/secrets/secret_key

secrets:
  openai_api_key:
    external: true
  secret_key:
    external: true
```

```python
# app/config.py
import os

def read_secret(secret_name: str) -> str:
    """从 Docker Secret 读取密钥"""
    secret_file = os.getenv(f"{secret_name.upper()}_FILE")
    if secret_file and os.path.exists(secret_file):
        with open(secret_file) as f:
            return f.read().strip()
    return os.getenv(secret_name.upper(), "")

OPENAI_API_KEY = read_secret("openai_api_key")
SECRET_KEY = read_secret("secret_key")
```

---

## Docker 环境变量的最佳实践

### 实践1：不要在 Dockerfile 中硬编码敏感信息

```dockerfile
# ❌ 错误：硬编码密钥
ENV OPENAI_API_KEY=sk-xxx

# ✅ 正确：通过环境变量注入
# 不在 Dockerfile 中设置敏感信息
```

---

### 实践2：使用 .env.example 作为模板

```bash
# .env.example（提交到 git）
DATABASE_URL=postgresql://localhost:5432/dbname
OPENAI_API_KEY=your_key_here
SECRET_KEY=your-secret-key-here
DEBUG=False
```

```bash
# .env.prod（不提交到 git）
DATABASE_URL=postgresql://prod-db:5432/prod_db
OPENAI_API_KEY=sk-prod-xxx
SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx
DEBUG=False
```

---

### 实践3：使用 docker-compose 变量替换

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=${DEBUG:-False}  # 默认值
```

```bash
# 从宿主机环境变量读取
export DATABASE_URL=postgresql://prod-db:5432/prod_db
export OPENAI_API_KEY=sk-prod-xxx

docker-compose up -d
```

---

### 实践4：验证环境变量

```python
# app/main.py
from fastapi import FastAPI
import os

app = FastAPI()

@app.on_event("startup")
async def startup():
    # 验证必需的环境变量
    required_vars = ["DATABASE_URL", "OPENAI_API_KEY", "SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"缺少必需的环境变量: {', '.join(missing_vars)}")

    print("✅ 环境变量验证通过")
```

---

### 实践5：使用健康检查

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

```python
# app/main.py
@app.get("/health")
def health_check():
    # 检查环境变量是否正确
    if not os.getenv("DATABASE_URL"):
        return {"status": "unhealthy", "reason": "DATABASE_URL not set"}

    return {"status": "healthy"}
```

---

## Docker 环境变量的高级特性

### 特性1：多阶段构建

```dockerfile
# 构建阶段
FROM python:3.13 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY . .

# 不设置环境变量，由运行时注入
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

---

### 特性2：构建时参数（ARG）

```dockerfile
# 构建时参数
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV}

WORKDIR /app
COPY . .

CMD ["uvicorn", "app.main:app"]
```

```bash
# 构建时传递参数
docker build --build-arg PYTHON_VERSION=3.13 --build-arg APP_ENV=dev -t my-app .
```

**注意：** ARG 只在构建时有效，ENV 在运行时有效。

---

### 特性3：环境变量插值

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@db:5432/${DB_NAME}
      - API_URL=https://${DOMAIN}/api
```

```bash
# 设置宿主机环境变量
export DB_USER=admin
export DB_PASSWORD=password
export DB_NAME=mydb
export DOMAIN=example.com

docker-compose up -d
```

---

### 特性4：条件环境变量

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DEBUG=${DEBUG:-False}  # 默认 False
      - LOG_LEVEL=${LOG_LEVEL:-INFO}  # 默认 INFO
      - PORT=${PORT:-8000}  # 默认 8000
```

---

## 常见问题

### Q1: Docker 容器中的环境变量会同步到宿主机吗？

**A:** 不会。容器的环境变量是隔离的。

```bash
# 容器内
docker exec -it api bash
echo $DATABASE_URL  # postgresql://db:5432/mydb

# 宿主机
echo $DATABASE_URL  # 空（容器环境变量不会同步到宿主机）
```

---

### Q2: 如何在容器中查看环境变量？

**A:** 使用 `docker exec` 或 `docker inspect`。

```bash
# 方式1：进入容器查看
docker exec -it api bash
env | grep DATABASE_URL

# 方式2：使用 docker inspect
docker inspect api | grep -A 10 "Env"

# 方式3：使用 docker-compose
docker-compose exec api env
```

---

### Q3: 如何在 Docker 中使用 .env 文件？

**A:** 使用 `env_file` 或 `--env-file`。

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env.prod
```

```bash
# docker run
docker run --env-file .env.prod my-app
```

---

### Q4: Docker 环境变量和 python-dotenv 冲突吗？

**A:** 不冲突。Docker 环境变量优先级更高。

```python
# app.py
from dotenv import load_dotenv
import os

# 加载 .env 文件（不覆盖 Docker 环境变量）
load_dotenv()

# Docker 环境变量优先
DATABASE_URL = os.getenv("DATABASE_URL")
```

---

## 在 AI Agent 后端中的应用

### 完整示例

**Dockerfile：**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - ENV=prod
      - DEBUG=False
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:password@db:5432/agent_db
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    restart: always

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agent_db
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**app/config.py：**

```python
import os
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Environment(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"

class Settings(BaseSettings):
    env: Environment = Environment.PROD
    debug: bool = False
    log_level: str = "INFO"

    database_url: str
    redis_url: str

    openai_api_key: str
    openai_model: str = "gpt-4"

    secret_key: str

    @field_validator("debug")
    @classmethod
    def validate_debug(cls, v: bool, info) -> bool:
        if info.data.get("env") == Environment.PROD and v:
            raise ValueError("生产环境不能启用调试模式")
        return v

    class Config:
        # Docker 环境变量优先，不需要 env_file
        pass

settings = Settings()

# 启动时打印配置
print(f"✅ 环境: {settings.env}")
print(f"✅ 数据库: {settings.database_url}")
print(f"✅ Redis: {settings.redis_url}")
print(f"✅ 调试模式: {settings.debug}")
```

**部署：**

```bash
# 设置宿主机环境变量
export OPENAI_API_KEY=sk-prod-xxx
export SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx

# 启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 查看环境变量
docker-compose exec api env | grep -E "DATABASE_URL|OPENAI_API_KEY"
```

---

## 总结

**Docker 环境变量的核心价值：**

1. **镜像复用**：同一个镜像可以在不同环境中使用
2. **配置灵活**：修改配置不需要重新构建镜像
3. **安全**：敏感信息不打包到镜像中
4. **标准化**：遵循 12-Factor App 原则

**注入方式：**
- `docker run -e`：命令行注入
- `docker run --env-file`：从文件注入
- `docker-compose environment`：直接定义
- `docker-compose env_file`：从文件注入
- `docker-compose 变量替换`：从宿主机读取

**优先级规则：**
```
docker run -e > docker-compose environment > docker-compose env_file > Dockerfile ENV > 应用默认值
```

**最佳实践：**
- 不在 Dockerfile 中硬编码敏感信息
- 使用 .env.example 作为模板
- 使用 docker-compose 变量替换
- 验证环境变量
- 使用健康检查

**实际应用：**
- 开发环境：docker-compose.dev.yml
- 生产环境：docker-compose.prod.yml
- 数据库连接：使用服务名作为主机名
- 敏感信息：使用 Docker Secrets 或宿主机环境变量
