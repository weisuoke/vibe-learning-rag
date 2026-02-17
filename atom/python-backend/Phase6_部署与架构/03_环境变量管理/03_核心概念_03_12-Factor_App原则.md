# 核心概念3：12-Factor App 原则

## 一句话定义

**12-Factor App 是一套构建现代云原生应用的方法论，其中第三条原则"配置"要求将配置存储在环境变量中，实现配置与代码的严格分离。**

---

## 什么是 12-Factor App？

**12-Factor App** 是 Heroku 团队在 2011 年提出的一套构建 SaaS 应用的最佳实践，包含 12 条原则：

1. **代码库（Codebase）**：一份代码，多份部署
2. **依赖（Dependencies）**：显式声明依赖
3. **配置（Config）**：在环境中存储配置 ⭐
4. **后端服务（Backing Services）**：把后端服务当作附加资源
5. **构建、发布、运行（Build, Release, Run）**：严格分离构建和运行
6. **进程（Processes）**：以无状态进程运行应用
7. **端口绑定（Port Binding）**：通过端口绑定提供服务
8. **并发（Concurrency）**：通过进程模型进行扩展
9. **易处理（Disposability）**：快速启动和优雅关闭
10. **开发环境与生产环境等价（Dev/Prod Parity）**：尽可能保持一致
11. **日志（Logs）**：把日志当作事件流
12. **管理进程（Admin Processes）**：后台管理任务作为一次性进程

**本节重点：第三条原则 - 配置**

---

## 第三条原则：配置

### 核心思想

**配置是指在不同部署环境（开发、测试、生产）中会变化的内容，应该存储在环境变量中，而不是代码中。**

### 什么是配置？

**配置包括：**
- 数据库连接信息（URL、用户名、密码）
- 外部服务凭证（API 密钥、访问令牌）
- 部署环境特定的值（域名、端口、调试开关）

**不是配置：**
- 应用的内部配置（路由定义、业务逻辑）
- 常量（不会因环境而变化）

### 为什么不能把配置写在代码中？

**问题1：安全风险**

```python
# ❌ 错误：配置硬编码在代码中
DATABASE_URL = "postgresql://admin:password123@prod-db:5432/mydb"
OPENAI_API_KEY = "sk-xxx"

# 代码提交到 git，密钥泄露！
```

**问题2：环境耦合**

```python
# ❌ 错误：不同环境需要修改代码
if ENV == "dev":
    DATABASE_URL = "postgresql://localhost:5432/dev_db"
elif ENV == "prod":
    DATABASE_URL = "postgresql://prod-db:5432/prod_db"

# 每次部署都要修改代码
```

**问题3：难以维护**

```python
# ❌ 错误：配置分散在多个文件
# database.py
DB_HOST = "localhost"

# api.py
API_KEY = "sk-xxx"

# main.py
DEBUG = True

# 配置分散，难以管理
```

---

## 12-Factor App 的配置原则

### 原则1：配置与代码严格分离

**定义：** 配置不应该出现在代码中，应该存储在环境变量中。

**判断标准：** 代码能否开源？如果代码中包含密钥，就不能开源。

```python
# ✅ 正确：配置与代码分离
import os

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 代码可以安全地开源
```

---

### 原则2：环境变量是唯一的配置来源

**定义：** 不使用配置文件（如 `config.yaml`、`settings.ini`），只使用环境变量。

**为什么？**
- 环境变量是语言和操作系统无关的标准
- 不会意外提交到版本控制
- 易于在不同平台（Docker、Kubernetes、云平台）中使用

```python
# ❌ 错误：使用配置文件
import yaml

with open("config.yaml") as f:
    config = yaml.load(f)

DATABASE_URL = config["database"]["url"]

# ✅ 正确：使用环境变量
import os

DATABASE_URL = os.getenv("DATABASE_URL")
```

**注意：** `.env` 文件是本地开发的便利工具，不违反 12-Factor 原则，因为它最终也是加载到环境变量中。

---

### 原则3：不同环境使用相同的代码

**定义：** 开发、测试、生产环境运行完全相同的代码，只有配置不同。

```python
# ✅ 正确：相同的代码，不同的配置
import os
from sqlalchemy import create_engine

# 代码完全相同
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# 开发环境：DATABASE_URL=postgresql://localhost:5432/dev_db
# 生产环境：DATABASE_URL=postgresql://prod-db:5432/prod_db
```

---

### 原则4：配置不分组

**定义：** 不要为不同环境创建配置组（如 `development`、`production`），而是直接使用环境变量。

```python
# ❌ 错误：配置分组
config = {
    "development": {
        "database_url": "postgresql://localhost:5432/dev_db",
        "debug": True
    },
    "production": {
        "database_url": "postgresql://prod-db:5432/prod_db",
        "debug": False
    }
}

env = os.getenv("ENV", "development")
DATABASE_URL = config[env]["database_url"]

# ✅ 正确：直接使用环境变量
DATABASE_URL = os.getenv("DATABASE_URL")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```

**为什么？** 配置分组会导致配置数量随环境数量线性增长，难以维护。

---

## 12-Factor App 在 Python 后端中的实现

### 实现1：使用 python-dotenv

```python
# .env（本地开发）
DATABASE_URL=postgresql://localhost:5432/dev_db
OPENAI_API_KEY=sk-dev-xxx
DEBUG=True

# app.py
from dotenv import load_dotenv
import os

# 加载 .env 文件到环境变量
load_dotenv()

# 从环境变量读取配置
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```

---

### 实现2：使用 Pydantic Settings

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()

# main.py
from config import settings

print(f"数据库: {settings.database_url}")
print(f"调试模式: {settings.debug}")
```

---

### 实现3：Docker 部署

```dockerfile
# Dockerfile
FROM python:3.13
COPY . /app
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=False
```

```bash
# 部署时设置环境变量
export OPENAI_API_KEY=sk-prod-xxx
docker-compose up -d
```

---

### 实现4：Kubernetes 部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      containers:
      - name: api
        image: my-api:latest
        env:
        - name: DATABASE_URL
          value: "postgresql://db:5432/mydb"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        - name: DEBUG
          value: "False"
```

---

## 12-Factor App 的好处

### 好处1：安全

**配置不提交到版本控制，避免密钥泄露。**

```bash
# .gitignore
.env
.env.local
.env.*.local

# 只提交配置模板
.env.example  # ✅
```

---

### 好处2：灵活

**不修改代码，只改环境变量，就能切换环境。**

```bash
# 开发环境
export DATABASE_URL=postgresql://localhost:5432/dev_db
python app.py

# 生产环境
export DATABASE_URL=postgresql://prod-db:5432/prod_db
python app.py
```

---

### 好处3：可移植

**相同的代码可以在任何平台运行（本地、Docker、Kubernetes、云平台）。**

```bash
# 本地
export DATABASE_URL=postgresql://localhost:5432/mydb
python app.py

# Docker
docker run -e DATABASE_URL=postgresql://db:5432/mydb my-api

# Kubernetes
kubectl set env deployment/api DATABASE_URL=postgresql://db:5432/mydb
```

---

### 好处4：易于扩展

**添加新环境（如 staging）不需要修改代码，只需要设置环境变量。**

```bash
# 新增 staging 环境
export DATABASE_URL=postgresql://staging-db:5432/staging_db
export OPENAI_API_KEY=sk-staging-xxx
python app.py
```

---

## 常见误区

### 误区1：12-Factor App 禁止使用 .env 文件 ❌

**错误理解：** 12-Factor App 要求只使用环境变量，所以不能用 .env 文件。

**正确理解：** .env 文件是本地开发的便利工具，最终也是加载到环境变量中，不违反 12-Factor 原则。

```python
# ✅ 正确：.env 文件用于本地开发
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 到环境变量

# 生产环境不使用 .env，直接设置系统环境变量
```

---

### 误区2：所有配置都应该是环境变量 ❌

**错误理解：** 应用的所有配置都应该放在环境变量中。

**正确理解：** 只有**会因环境而变化**的配置才应该是环境变量。

```python
# ✅ 环境变量（会因环境而变化）
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ✅ 代码中的常量（不会因环境而变化）
MAX_RETRIES = 3
TIMEOUT = 30
DEFAULT_PAGE_SIZE = 20
```

---

### 误区3：配置分组更灵活 ❌

**错误理解：** 使用配置分组（如 `config.development`、`config.production`）更灵活。

**正确理解：** 配置分组会导致配置数量线性增长，难以维护。

```python
# ❌ 错误：配置分组
config = {
    "development": {...},
    "test": {...},
    "staging": {...},
    "production": {...}
}

# ✅ 正确：直接使用环境变量
DATABASE_URL = os.getenv("DATABASE_URL")
```

---

## 12-Factor App 与其他配置方式的对比

| 配置方式 | 12-Factor App | 配置文件 | 配置分组 |
|---------|---------------|----------|----------|
| 配置来源 | 环境变量 | YAML/JSON/INI | 代码中的字典 |
| 安全性 | ✅ 高 | ❌ 低（易泄露） | ❌ 低（硬编码） |
| 可移植性 | ✅ 高 | ⚠️ 中等 | ❌ 低 |
| 易用性 | ✅ 高 | ⚠️ 中等 | ✅ 高 |
| 维护性 | ✅ 高 | ⚠️ 中等 | ❌ 低 |
| 云原生 | ✅ 完全支持 | ⚠️ 部分支持 | ❌ 不支持 |

---

## 实际应用场景

### 场景1：本地开发

```bash
# .env（不提交到 git）
DATABASE_URL=postgresql://localhost:5432/dev_db
OPENAI_API_KEY=sk-dev-xxx
DEBUG=True
```

```python
# app.py
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
```

---

### 场景2：Docker 部署

```yaml
# docker-compose.yml
services:
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

```bash
# 部署
export OPENAI_API_KEY=sk-prod-xxx
docker-compose up -d
```

---

### 场景3：Kubernetes 部署

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
data:
  DATABASE_URL: "postgresql://db:5432/mydb"
  DEBUG: "False"
```

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: api
        envFrom:
        - configMapRef:
            name: api-config
        - secretRef:
            name: api-secrets
```

---

### 场景4：云平台部署

**AWS Elastic Beanstalk：**

```bash
# 设置环境变量
eb setenv DATABASE_URL=postgresql://... OPENAI_API_KEY=sk-xxx
```

**Heroku：**

```bash
# 设置环境变量
heroku config:set DATABASE_URL=postgresql://...
heroku config:set OPENAI_API_KEY=sk-xxx
```

**Google Cloud Run：**

```bash
# 部署时设置环境变量
gcloud run deploy api \
  --set-env-vars DATABASE_URL=postgresql://... \
  --set-env-vars OPENAI_API_KEY=sk-xxx
```

---

## 12-Factor App 的扩展原则

### 扩展1：配置验证

**在应用启动时验证配置，而不是运行时。**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str  # 必需字段
    openai_api_key: str  # 必需字段
    debug: bool = False

    class Config:
        env_file = ".env"

# 启动时验证，缺少配置立即报错
settings = Settings()
```

---

### 扩展2：配置文档化

**使用类型注解和文档字符串说明配置。**

```python
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL",
        examples=["postgresql://user:password@localhost:5432/mydb"]
    )

    debug: bool = Field(
        False,
        description="Enable debug mode (only for development)"
    )
```

---

### 扩展3：配置分层

**支持多层配置覆盖：系统环境变量 > .env 文件 > 默认值。**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False  # 默认值（优先级最低）

    class Config:
        env_file = ".env"  # .env 文件（优先级中等）

# 系统环境变量（优先级最高）
# export DEBUG=True
```

---

## 在 AI Agent 后端中的应用

### 完整示例

```python
# config.py
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "AI Agent API"
    debug: bool = False

    # 数据库配置
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL"
    )

    # LLM 配置
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key"
    )
    openai_model: str = "gpt-4"

    # 安全配置
    secret_key: str = Field(
        ...,
        min_length=32,
        description="JWT secret key"
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            raise ValueError("Only PostgreSQL is supported")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

```bash
# .env.example（提交到 git）
DATABASE_URL=postgresql://localhost:5432/agent_db
OPENAI_API_KEY=your_key_here
SECRET_KEY=your-secret-key-at-least-32-characters-long
DEBUG=False
```

```bash
# .env（不提交到 git）
DATABASE_URL=postgresql://localhost:5432/agent_db
OPENAI_API_KEY=sk-xxx
SECRET_KEY=generated-secret-key-xxxxxxxxxxxxx
DEBUG=True
```

---

## 总结

**12-Factor App 配置原则的核心：**

1. **配置与代码严格分离** - 配置不应该出现在代码中
2. **环境变量是唯一的配置来源** - 不使用配置文件
3. **不同环境使用相同的代码** - 只有配置不同
4. **配置不分组** - 直接使用环境变量，不创建配置组

**好处：**
- ✅ 安全：配置不提交到版本控制
- ✅ 灵活：不修改代码就能切换环境
- ✅ 可移植：相同的代码可以在任何平台运行
- ✅ 易于扩展：添加新环境不需要修改代码

**实现方式：**
- 本地开发：python-dotenv + .env 文件
- 类型安全：Pydantic Settings
- Docker 部署：docker-compose 环境变量
- Kubernetes 部署：ConfigMap + Secret
- 云平台部署：平台提供的环境变量管理

**判断标准：**
- 代码能否开源？如果包含密钥，就违反了 12-Factor 原则
- 部署时是否需要修改代码？如果需要，就违反了 12-Factor 原则
- 配置是否会因环境而变化？如果不会，就不应该是环境变量
