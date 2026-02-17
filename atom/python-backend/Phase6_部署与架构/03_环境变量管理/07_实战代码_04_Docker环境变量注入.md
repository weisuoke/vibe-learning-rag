# 实战代码4：Docker 环境变量注入

## 场景说明

演示如何在 Docker 容器中注入环境变量，包括 Dockerfile、docker-compose、多环境部署等完整场景。

---

## 完整代码示例

### 示例1：基础 Docker 环境变量

**Dockerfile：**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建和运行：**

```bash
# 构建镜像
docker build -t my-api .

# 运行容器（注入环境变量）
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://db:5432/mydb \
  -e OPENAI_API_KEY=sk-xxx \
  -e DEBUG=False \
  my-api
```

---

### 示例2：使用 --env-file

**.env.prod：**

```bash
DATABASE_URL=postgresql://prod-db:5432/prod_db
OPENAI_API_KEY=sk-prod-xxx
SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx
DEBUG=False
LOG_LEVEL=WARNING
```

**运行容器：**

```bash
docker run -p 8000:8000 --env-file .env.prod my-api
```

---

### 示例3：docker-compose 基础配置

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

**使用：**

```bash
# 设置宿主机环境变量
export OPENAI_API_KEY=sk-xxx

# 启动容器
docker-compose up -d
```

---

### 示例4：docker-compose 使用 env_file

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    env_file:
      - .env.prod
    ports:
      - "8000:8000"
```

**.env.prod：**

```bash
DATABASE_URL=postgresql://db:5432/mydb
OPENAI_API_KEY=sk-prod-xxx
DEBUG=False
```

---

### 示例5：多环境 docker-compose

**docker-compose.dev.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - ENV=dev
      - DEBUG=True
    env_file:
      - .env.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    command: uvicorn app.main:app --host 0.0.0.0 --reload
```

**docker-compose.prod.yml：**

```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - ENV=prod
      - DEBUG=False
    env_file:
      - .env.prod
    ports:
      - "80:8000"
    restart: always
```

**使用：**

```bash
# 开发环境
docker-compose -f docker-compose.dev.yml up

# 生产环境
docker-compose -f docker-compose.prod.yml up -d
```

---

### 示例6：完整的 FastAPI + PostgreSQL + Redis

**项目结构：**

```
project/
├── Dockerfile
├── docker-compose.yml
├── .env.prod
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── config.py
└── README.md
```

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

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

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
      - DATABASE_URL=postgresql://user:password@db:5432/agent_db
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: always

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agent_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

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
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "prod"
    database_url: str
    redis_url: str
    openai_api_key: str
    secret_key: str

settings = Settings()
```

**app/main.py：**

```python
from fastapi import FastAPI
from app.config import settings

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy", "env": settings.env}

@app.get("/")
def read_root():
    return {"message": "AI Agent API", "env": settings.env}
```

**部署：**

```bash
# 设置环境变量
export OPENAI_API_KEY=sk-prod-xxx
export SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx

# 启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 健康检查
curl http://localhost:8000/health
```

---

## 运行说明

### 1. 构建镜像

```bash
docker build -t my-api .
```

### 2. 运行容器

```bash
# 方式1：命令行注入
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://db:5432/mydb \
  -e OPENAI_API_KEY=sk-xxx \
  my-api

# 方式2：使用 env-file
docker run -p 8000:8000 --env-file .env.prod my-api

# 方式3：使用 docker-compose
docker-compose up -d
```

### 3. 验证环境变量

```bash
# 进入容器
docker exec -it <container_id> bash

# 查看环境变量
env | grep DATABASE_URL
```

---

## 总结

**Docker 环境变量注入的核心方法：**

1. **docker run -e**：命令行注入
2. **docker run --env-file**：从文件注入
3. **docker-compose environment**：直接定义
4. **docker-compose env_file**：从文件注入
5. **docker-compose 变量替换**：从宿主机读取

**最佳实践：**
- 不在 Dockerfile 中硬编码敏感信息
- 使用 docker-compose 管理多容器应用
- 使用健康检查确保服务就绪
- 生产环境使用 restart: always
- 使用 depends_on 管理服务依赖
