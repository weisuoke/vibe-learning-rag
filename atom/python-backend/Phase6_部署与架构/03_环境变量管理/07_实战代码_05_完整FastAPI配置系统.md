# 实战代码5：完整 FastAPI 配置系统

## 场景说明

演示一个生产级的 FastAPI 应用配置系统，包括多环境支持、类型验证、敏感信息保护、配置导出等完整功能。

---

## 完整项目结构

```
project/
├── .env
├── .env.dev
├── .env.test
├── .env.prod
├── .env.example
├── .gitignore
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   └── api/
│       └── routes.py
└── README.md
```

---

## 完整代码

### app/config.py

```python
"""
配置管理模块
生产级的配置系统
"""
import os
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import sys

class Environment(str, Enum):
    """环境枚举"""
    DEV = "dev"
    TEST = "test"
    PROD = "prod"

def load_env_files():
    """加载多层配置文件"""
    env = os.getenv("ENV", "dev")

    # 1. 加载基础配置
    if Path(".env").exists():
        load_dotenv(".env")
        print(f"✅ 加载基础配置: .env")

    # 2. 加载环境配置
    env_file = f".env.{env}"
    if Path(env_file).exists():
        load_dotenv(env_file, override=True)
        print(f"✅ 加载环境配置: {env_file}")
    else:
        print(f"❌ 错误：配置文件不存在: {env_file}")
        sys.exit(1)

    # 3. 加载本地配置
    if Path(".env.local").exists():
        load_dotenv(".env.local", override=True)
        print(f"✅ 加载本地配置: .env.local")

# 启动时加载配置
load_env_files()

class DatabaseConfig(BaseModel):
    """数据库配置"""
    url: str = Field(..., description="数据库连接字符串")
    pool_size: int = Field(10, ge=1, le=100)
    max_overflow: int = Field(20, ge=0, le=100)
    pool_timeout: int = Field(30, ge=1, le=300)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            raise ValueError("只支持 PostgreSQL 数据库")
        return v

class LLMConfig(BaseModel):
    """LLM 配置"""
    api_key: SecretStr = Field(..., description="API 密钥")
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=100000)

class Settings(BaseSettings):
    """应用配置"""
    # 环境标识
    env: Environment = Environment.DEV

    # 应用配置
    app_name: str = "AI Agent API"
    debug: bool = False
    log_level: str = "INFO"

    # 数据库配置
    database: DatabaseConfig

    # LLM 配置
    llm: LLMConfig

    # 安全配置
    secret_key: SecretStr = Field(..., min_length=32)

    # Redis 配置（可选）
    redis_url: str | None = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"日志级别必须是 {', '.join(valid_levels)} 之一")
        return v

    @model_validator(mode='after')
    def validate_production_config(self):
        """验证生产环境配置"""
        if self.env == Environment.PROD:
            if self.debug:
                raise ValueError("生产环境不能启用调试模式")
            if "localhost" in self.database.url:
                raise ValueError("生产环境不能使用本地数据库")
        return self

    class Config:
        env_nested_delimiter = "__"

# 加载配置
try:
    settings = Settings()
    print(f"\n✅ 配置验证通过")
    print(f"   环境: {settings.env}")
    print(f"   应用名称: {settings.app_name}")
    print(f"   调试模式: {settings.debug}")
except Exception as e:
    print(f"\n❌ 配置错误: {e}")
    sys.exit(1)
```

### app/main.py

```python
"""
FastAPI 应用入口
"""
from fastapi import FastAPI
from app.config import settings
import logging

# 配置日志
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

@app.on_event("startup")
async def startup():
    """应用启动事件"""
    logger.info("🚀 启动应用")
    logger.info(f"📍 环境: {settings.env}")
    logger.info(f"🗄️  数据库: {settings.database.url}")
    logger.info(f"🤖 LLM 模型: {settings.llm.model}")
    logger.info(f"🐛 调试模式: {settings.debug}")

@app.get("/")
def read_root():
    """根路径"""
    return {
        "app_name": settings.app_name,
        "env": settings.env,
        "debug": settings.debug
    }

@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "env": settings.env
    }

@app.get("/config")
def get_config():
    """获取配置（隐藏敏感信息）"""
    return {
        "app_name": settings.app_name,
        "env": settings.env,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "database_url": settings.database.url.split("@")[1] if "@" in settings.database.url else "未设置",
        "llm_model": settings.llm.model,
        "redis_url": settings.redis_url or "未配置"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower()
    )
```

---

## 配置文件

### .env（基础配置）

```bash
# 应用配置
APP_NAME=AI Agent API
LOG_LEVEL=INFO

# 数据库配置
DATABASE__POOL_SIZE=10
DATABASE__MAX_OVERFLOW=20
DATABASE__POOL_TIMEOUT=30

# LLM 配置
LLM__BASE_URL=https://api.openai.com/v1
LLM__MODEL=gpt-4
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=2000
```

### .env.dev（开发环境）

```bash
ENV=dev
DEBUG=True
LOG_LEVEL=DEBUG

DATABASE__URL=postgresql://localhost:5432/dev_db
LLM__API_KEY=sk-dev-xxx
SECRET_KEY=dev-secret-key-xxxxxxxxxxxxx
REDIS_URL=redis://localhost:6379/0
```

### .env.prod（生产环境）

```bash
ENV=prod
DEBUG=False
LOG_LEVEL=WARNING

DATABASE__URL=postgresql://prod-db.example.com:5432/prod_db
LLM__API_KEY=sk-prod-xxxxxxxxxxxxx
SECRET_KEY=prod-secret-key-xxxxxxxxxxxxx
REDIS_URL=redis://prod-redis.example.com:6379/0
```

### .env.example（配置模板）

```bash
ENV=dev
DEBUG=True
LOG_LEVEL=INFO

DATABASE__URL=postgresql://localhost:5432/dbname
DATABASE__POOL_SIZE=10

LLM__API_KEY=your_openai_api_key_here
LLM__MODEL=gpt-4

SECRET_KEY=your-secret-key-at-least-32-characters-long
REDIS_URL=redis://localhost:6379/0
```

---

## 运行说明

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv
```

### 2. 初始化配置

```bash
cp .env.example .env.dev
vim .env.dev  # 填入真实配置
```

### 3. 运行应用

```bash
# 开发环境
ENV=dev python -m app.main

# 或使用 uvicorn
ENV=dev uvicorn app.main:app --reload
```

### 4. 测试接口

```bash
# 健康检查
curl http://localhost:8000/health

# 获取配置
curl http://localhost:8000/config
```

---

## 总结

这个完整的 FastAPI 配置系统包含：

1. **多层配置加载**：基础 + 环境 + 本地
2. **类型安全**：Pydantic Settings 自动验证
3. **嵌套配置**：数据库、LLM 等模块化配置
4. **敏感信息保护**：SecretStr 自动隐藏
5. **环境验证**：生产环境的特殊验证规则
6. **配置导出**：隐藏敏感信息的配置接口
7. **日志集成**：根据配置设置日志级别
8. **健康检查**：监控应用状态

**最佳实践：**
- 配置集中管理
- 启动时验证
- 环境隔离
- 敏感信息保护
- 文档化配置
