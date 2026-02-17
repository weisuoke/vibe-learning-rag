# 核心概念2：Pydantic Settings 配置类

## 一句话定义

**Pydantic Settings 是基于 Pydantic 的类型安全配置管理库，自动从环境变量加载配置并验证类型，让配置管理像定义数据模型一样简单。**

---

## 为什么需要 Pydantic Settings？

### python-dotenv 的局限性

使用 `python-dotenv` + `os.getenv()` 时，你会遇到这些问题：

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 问题1：所有值都是字符串，需要手动转换
debug = os.getenv("DEBUG", "False")
if debug.lower() == "true":  # 容易出错
    debug = True

port = int(os.getenv("PORT", "8000"))  # 可能抛异常

# 问题2：没有类型提示，IDE 无法自动补全
database_url = os.getenv("DATABASE_URL")  # 类型是 str | None

# 问题3：没有必需字段验证，运行时才发现问题
api_key = os.getenv("OPENAI_API_KEY")  # 可能是 None
client = OpenAI(api_key=api_key)  # 运行时报错

# 问题4：配置分散在代码各处，难以维护
```

### Pydantic Settings 的解决方案

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 自动类型转换
    debug: bool = False
    port: int = 8000

    # 必需字段（缺少时启动失败）
    database_url: str
    openai_api_key: str

    # 可选字段
    redis_url: str | None = None

    class Config:
        env_file = ".env"

# 自动加载、验证、转换
settings = Settings()

# IDE 有完整的类型提示
if settings.debug:  # IDE 知道这是 bool
    print(f"数据库: {settings.database_url}")  # IDE 知道这是 str
```

---

## 核心功能

### 功能1：自动类型转换

**支持的类型：**

```python
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # 基础类型
    debug: bool = False          # "true"/"false" → bool
    port: int = 8000             # "8000" → int
    timeout: float = 30.5        # "30.5" → float
    name: str = "app"            # 字符串

    # 复杂类型
    allowed_hosts: List[str] = ["localhost"]  # "host1,host2" → ["host1", "host2"]
    database_config: Dict[str, str] = {}      # JSON 字符串 → dict

    class Config:
        env_file = ".env"
```

**环境变量：**

```bash
# .env
DEBUG=true
PORT=8000
TIMEOUT=30.5
ALLOWED_HOSTS=localhost,127.0.0.1,example.com
DATABASE_CONFIG={"host": "localhost", "port": "5432"}
```

**自动转换：**

```python
settings = Settings()

print(settings.debug)          # True (bool)
print(settings.port)           # 8000 (int)
print(settings.timeout)        # 30.5 (float)
print(settings.allowed_hosts)  # ["localhost", "127.0.0.1", "example.com"]
print(settings.database_config)  # {"host": "localhost", "port": "5432"}
```

---

### 功能2：必需字段验证

**启动时验证：**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 必需字段（没有默认值）
    database_url: str
    openai_api_key: str
    secret_key: str

    # 可选字段（有默认值）
    debug: bool = False

    class Config:
        env_file = ".env"

# 如果缺少必需字段，启动时立即报错
try:
    settings = Settings()
except Exception as e:
    print(f"配置错误: {e}")
    # ValidationError: 1 validation error for Settings
    # database_url
    #   Field required [type=missing, input_value={}, input_type=dict]
```

**好处：** 在应用启动时就发现配置问题，而不是运行时才报错。

---

### 功能3：字段验证器

**自定义验证逻辑：**

```python
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    openai_api_key: str
    port: int = 8000

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key is too short")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1024 or v > 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v

    class Config:
        env_file = ".env"

# 验证失败时报错
settings = Settings()  # 如果 API key 格式错误，启动失败
```

---

### 功能4：环境变量名映射

**自动映射：**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Python 字段名（snake_case）
    database_url: str
    openai_api_key: str
    max_connections: int = 10

    class Config:
        env_file = ".env"

# 自动映射到环境变量名（SCREAMING_SNAKE_CASE）
# database_url → DATABASE_URL
# openai_api_key → OPENAI_API_KEY
# max_connections → MAX_CONNECTIONS
```

**自定义映射：**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # 使用 Field 自定义环境变量名
    api_key: str = Field(alias="CUSTOM_API_KEY")
    db_url: str = Field(alias="DB_CONNECTION_STRING")

# 环境变量
# CUSTOM_API_KEY=sk-xxx
# DB_CONNECTION_STRING=postgresql://...
```

---

### 功能5：嵌套配置

**嵌套模型：**

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    database: str = "mydb"

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class Settings(BaseSettings):
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# 环境变量（使用 __ 分隔）
# DATABASE__HOST=prod-db.example.com
# DATABASE__PORT=5432
# REDIS__HOST=redis.example.com
```

**使用：**

```python
settings = Settings()

print(settings.database.host)  # prod-db.example.com
print(settings.redis.host)     # redis.example.com
```

---

### 功能6：配置优先级

**优先级规则：**

1. 系统环境变量（最高）
2. .env 文件
3. 默认值（最低）

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False  # 默认值（优先级最低）

    class Config:
        env_file = ".env"  # .env 文件（优先级中等）

# .env 文件
# DEBUG=True

# 系统环境变量（优先级最高）
# export DEBUG=False

settings = Settings()
print(settings.debug)  # False（系统环境变量覆盖了 .env 和默认值）
```

---

### 功能7：多环境配置

**根据环境加载不同配置：**

```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    debug: bool = False

    class Config:
        # 根据 ENV 环境变量选择配置文件
        env_file = f".env.{os.getenv('ENV', 'dev')}"

# 使用
# ENV=dev python app.py   → 加载 .env.dev
# ENV=test python app.py  → 加载 .env.test
# ENV=prod python app.py  → 加载 .env.prod
```

---

### 功能8：配置导出

**导出为字典：**

```python
settings = Settings()

# 导出为字典
config_dict = settings.model_dump()
print(config_dict)
# {
#     'database_url': 'postgresql://...',
#     'debug': True,
#     'port': 8000
# }

# 导出为 JSON
config_json = settings.model_dump_json()
print(config_json)
# {"database_url": "postgresql://...", "debug": true, "port": 8000}
```

---

## 实际应用示例

### 示例1：FastAPI 应用配置

```python
# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "AI Agent API"
    debug: bool = False

    # 数据库配置
    database_url: str = Field(..., description="PostgreSQL connection URL")
    database_pool_size: int = 10

    # LLM 配置
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"

    # 安全配置
    secret_key: str = Field(..., description="JWT secret key")
    access_token_expire_minutes: int = 30

    # Redis 配置（可选）
    redis_url: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局单例
settings = Settings()
```

```python
# app/main.py
from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

@app.get("/config")
def get_config():
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "openai_model": settings.openai_model
    }
```

---

### 示例2：数据库连接配置

```python
# app/config.py
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            raise ValueError("Only PostgreSQL is supported")
        return v

class Settings(BaseSettings):
    database: DatabaseConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# .env
# DATABASE__URL=postgresql://localhost:5432/mydb
# DATABASE__POOL_SIZE=20

settings = Settings()
```

```python
# app/database.py
from sqlalchemy import create_engine
from app.config import settings

engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_timeout=settings.database.pool_timeout
)
```

---

### 示例3：多环境配置

```python
# app/config.py
import os
from pydantic_settings import BaseSettings
from enum import Enum

class Environment(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"

class Settings(BaseSettings):
    env: Environment = Environment.DEV
    database_url: str
    debug: bool = False

    class Config:
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # 根据环境选择配置文件
            env = os.getenv("ENV", "dev")
            return (
                init_settings,
                env_settings,
                file_secret_settings(f".env.{env}"),
            )

settings = Settings()

# 根据环境调整行为
if settings.env == Environment.PROD:
    print("生产环境：禁用调试模式")
elif settings.env == Environment.DEV:
    print("开发环境：启用调试模式")
```

---

### 示例4：配置验证和日志

```python
# app/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    debug: bool = False

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if "localhost" in v:
            logger.warning("使用本地数据库，不适合生产环境")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if v.startswith("sk-test"):
            logger.warning("使用测试 API 密钥")
        return v

    class Config:
        env_file = ".env"

settings = Settings()

# 启动时打印配置（隐藏敏感信息）
logger.info(f"数据库: {settings.database_url}")
logger.info(f"API 密钥: {settings.openai_api_key[:10]}...")
logger.info(f"调试模式: {settings.debug}")
```

---

## 高级特性

### 特性1：Secrets 文件支持

**从文件读取密钥：**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        secrets_dir="/run/secrets"  # Docker secrets 目录
    )

    database_url: str
    openai_api_key: str

# Docker secrets
# /run/secrets/database_url
# /run/secrets/openai_api_key
```

---

### 特性2：配置来源优先级自定义

```python
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

class Settings(BaseSettings):
    database_url: str

    class Config:
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ):
            # 自定义优先级
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            )
```

---

### 特性3：配置缓存

```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# 使用缓存的配置
settings = get_settings()
```

---

## 常见问题

### Q1: Pydantic Settings 和 python-dotenv 可以一起用吗？

**A:** 可以，但通常不需要。Pydantic Settings 可以直接加载 .env 文件。

```python
# 方式1：只用 Pydantic Settings（推荐）
class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"

# 方式2：python-dotenv + Pydantic Settings
from dotenv import load_dotenv

load_dotenv()  # 加载到系统环境变量

class Settings(BaseSettings):
    database_url: str  # 从系统环境变量读取
```

---

### Q2: 如何处理敏感信息？

**A:** 不要在日志中打印敏感信息。

```python
class Settings(BaseSettings):
    openai_api_key: str

    def __repr__(self):
        # 隐藏敏感信息
        return f"Settings(openai_api_key={self.openai_api_key[:10]}...)"

settings = Settings()
print(settings)  # Settings(openai_api_key=sk-xxx...)
```

---

### Q3: 如何在测试中覆盖配置？

**A:** 使用依赖注入或环境变量。

```python
# 方式1：依赖注入
def get_settings():
    return Settings()

# 测试时覆盖
def test_api():
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url="postgresql://test:5432/test_db"
    )

# 方式2：环境变量
import os
os.environ["DATABASE_URL"] = "postgresql://test:5432/test_db"
settings = Settings()
```

---

## 最佳实践

### 1. 使用全局单例

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"

# 全局单例
settings = Settings()
```

### 2. 分组配置

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    url: str
    pool_size: int = 10

class LLMConfig(BaseModel):
    api_key: str
    model: str = "gpt-4"

class Settings(BaseSettings):
    database: DatabaseConfig
    llm: LLMConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

### 3. 添加文档字符串

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
        description="Enable debug mode"
    )
```

### 4. 验证配置

```python
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1024 or v > 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v
```

---

## 在 AI Agent 后端中的应用

### 完整配置示例

```python
# app/config.py
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional

class DatabaseConfig(BaseModel):
    url: str
    pool_size: int = 10
    max_overflow: int = 20

class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

class RedisConfig(BaseModel):
    url: Optional[str] = None
    ttl: int = 3600

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "AI Agent API"
    debug: bool = False

    # 数据库配置
    database: DatabaseConfig

    # LLM 配置
    llm: LLMConfig

    # Redis 配置
    redis: RedisConfig = RedisConfig()

    # 安全配置
    secret_key: str = Field(..., min_length=32)
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

# 全局单例
settings = Settings()
```

```bash
# .env
APP_NAME=AI Agent API
DEBUG=True

DATABASE__URL=postgresql://localhost:5432/agent_db
DATABASE__POOL_SIZE=20

LLM__API_KEY=sk-xxx
LLM__MODEL=gpt-4
LLM__TEMPERATURE=0.7

REDIS__URL=redis://localhost:6379/0

SECRET_KEY=your-secret-key-at-least-32-characters-long
```

---

## 总结

**Pydantic Settings 的核心价值：**

1. **类型安全**：自动类型转换和验证
2. **启动时验证**：缺少必需字段时立即报错
3. **IDE 支持**：完整的类型提示和自动补全
4. **集中管理**：所有配置在一个类中
5. **灵活验证**：自定义验证逻辑
6. **嵌套配置**：支持复杂的配置结构
7. **优先级规则**：系统环境变量 > .env > 默认值

**与 python-dotenv 的对比：**

| 特性 | python-dotenv | Pydantic Settings |
|------|---------------|-------------------|
| 类型转换 | 手动 | 自动 |
| 类型验证 | 无 | 自动 |
| 必需字段 | 无 | 自动验证 |
| IDE 提示 | 无 | 完整支持 |
| 嵌套配置 | 不支持 | 支持 |
| 自定义验证 | 手动 | 装饰器 |
| 配置导出 | 无 | 支持 |

**推荐使用场景：**
- ✅ FastAPI 应用（官方推荐）
- ✅ 需要类型安全的项目
- ✅ 复杂的配置结构
- ✅ 团队协作项目
- ✅ 生产环境应用
