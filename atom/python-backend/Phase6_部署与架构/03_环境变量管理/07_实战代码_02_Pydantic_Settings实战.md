# 实战代码2：Pydantic Settings 实战

## 场景说明

演示如何使用 Pydantic Settings 构建类型安全的配置系统，包括自动验证、类型转换、嵌套配置、自定义验证器等高级特性。

---

## 完整代码示例

### 示例1：基础 Pydantic Settings

```python
"""
基础 Pydantic Settings 使用
演示：类型安全的配置管理
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 必需字段（没有默认值）
    database_url: str
    openai_api_key: str

    # 可选字段（有默认值）
    debug: bool = False
    port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

# 加载配置（自动从环境变量和 .env 文件读取）
try:
    settings = Settings()
    print("✅ 配置加载成功")
    print(f"数据库: {settings.database_url}")
    print(f"端口: {settings.port}")
    print(f"调试模式: {settings.debug}")
except Exception as e:
    print(f"❌ 配置错误: {e}")
```

**.env 文件：**

```bash
DATABASE_URL=postgresql://localhost:5432/mydb
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
DEBUG=True
PORT=8000
LOG_LEVEL=DEBUG
```

**运行输出：**

```
✅ 配置加载成功
数据库: postgresql://localhost:5432/mydb
端口: 8000
调试模式: True
```

---

### 示例2：自动类型转换

```python
"""
自动类型转换
演示：Pydantic 自动将字符串转换为正确的类型
"""
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 基础类型
    debug: bool = False          # "true" → True
    port: int = 8000             # "8000" → 8000
    timeout: float = 30.5        # "30.5" → 30.5

    # 列表类型（逗号分隔）
    allowed_hosts: List[str] = ["localhost"]

    class Config:
        env_file = ".env"

settings = Settings()

print("=== 类型转换结果 ===")
print(f"debug (bool): {settings.debug} - {type(settings.debug)}")
print(f"port (int): {settings.port} - {type(settings.port)}")
print(f"timeout (float): {settings.timeout} - {type(settings.timeout)}")
print(f"allowed_hosts (list): {settings.allowed_hosts} - {type(settings.allowed_hosts)}")
```

**.env 文件：**

```bash
DEBUG=true
PORT=8000
TIMEOUT=30.5
ALLOWED_HOSTS=localhost,127.0.0.1,example.com
```

**运行输出：**

```
=== 类型转换结果 ===
debug (bool): True - <class 'bool'>
port (int): 8000 - <class 'int'>
timeout (float): 30.5 - <class 'float'>
allowed_hosts (list): ['localhost', '127.0.0.1', 'example.com'] - <class 'list'>
```

---

### 示例3：字段验证器

```python
"""
字段验证器
演示：使用 field_validator 自定义验证逻辑
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    port: int = 8000
    log_level: str = "INFO"

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证 API 密钥格式"""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API 密钥必须以 'sk-' 开头")
        if len(v) < 20:
            raise ValueError("OpenAI API 密钥长度不足")
        print(f"✅ API 密钥验证通过: {v[:10]}...")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """验证数据库连接字符串"""
        if not v.startswith("postgresql://"):
            raise ValueError("只支持 PostgreSQL 数据库")
        print(f"✅ 数据库 URL 验证通过")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """验证端口号范围"""
        if v < 1024 or v > 65535:
            raise ValueError(f"端口号必须在 1024-65535 之间，当前值: {v}")
        print(f"✅ 端口号验证通过: {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"日志级别必须是 {', '.join(valid_levels)} 之一")
        print(f"✅ 日志级别验证通过: {v}")
        return v

    class Config:
        env_file = ".env"

# 加载配置（自动验证）
try:
    settings = Settings()
    print("\n✅ 所有配置验证通过")
except Exception as e:
    print(f"\n❌ 配置验证失败: {e}")
```

---

### 示例4：跨字段验证

```python
"""
跨字段验证
演示：使用 model_validator 验证多个字段之间的关系
"""
from pydantic import model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "dev"
    debug: bool = False
    database_url: str
    secret_key: str

    @model_validator(mode='after')
    def validate_production_config(self):
        """验证生产环境配置"""
        if self.env == "prod":
            # 生产环境不能启用调试模式
            if self.debug:
                raise ValueError("生产环境不能启用调试模式")

            # 生产环境不能使用本地数据库
            if "localhost" in self.database_url or "127.0.0.1" in self.database_url:
                raise ValueError("生产环境不能使用本地数据库")

            # 生产环境密钥长度必须足够
            if len(self.secret_key) < 32:
                raise ValueError("生产环境密钥长度必须至少 32 字符")

            print("✅ 生产环境配置验证通过")

        return self

    class Config:
        env_file = ".env"

# 测试开发环境
print("=== 测试开发环境 ===")
try:
    settings_dev = Settings(
        env="dev",
        debug=True,
        database_url="postgresql://localhost:5432/dev_db",
        secret_key="dev-key"
    )
    print("✅ 开发环境配置通过")
except Exception as e:
    print(f"❌ {e}")

# 测试生产环境（错误配置）
print("\n=== 测试生产环境（错误配置）===")
try:
    settings_prod = Settings(
        env="prod",
        debug=True,  # 错误：生产环境不能启用调试
        database_url="postgresql://localhost:5432/prod_db",
        secret_key="short-key"
    )
except Exception as e:
    print(f"❌ {e}")

# 测试生产环境（正确配置）
print("\n=== 测试生产环境（正确配置）===")
try:
    settings_prod = Settings(
        env="prod",
        debug=False,
        database_url="postgresql://prod-db.example.com:5432/prod_db",
        secret_key="prod-secret-key-xxxxxxxxxxxxx"
    )
    print("✅ 生产环境配置通过")
except Exception as e:
    print(f"❌ {e}")
```

---

### 示例5：嵌套配置

```python
"""
嵌套配置
演示：使用嵌套模型组织复杂配置
"""
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    database: str = "mydb"
    pool_size: int = 10

    @property
    def url(self) -> str:
        """生成数据库连接字符串"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseModel):
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def url(self) -> str:
        """生成 Redis 连接字符串"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class LLMConfig(BaseModel):
    """LLM 配置"""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000

class Settings(BaseSettings):
    """应用配置"""
    app_name: str = "AI Agent API"
    debug: bool = False

    # 嵌套配置
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    llm: LLMConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"  # 使用 __ 分隔嵌套字段

# 使用
settings = Settings()

print("=== 应用配置 ===")
print(f"应用名称: {settings.app_name}")
print(f"调试模式: {settings.debug}")

print("\n=== 数据库配置 ===")
print(f"主机: {settings.database.host}")
print(f"端口: {settings.database.port}")
print(f"连接字符串: {settings.database.url}")

print("\n=== Redis 配置 ===")
print(f"主机: {settings.redis.host}")
print(f"端口: {settings.redis.port}")
print(f"连接字符串: {settings.redis.url}")

print("\n=== LLM 配置 ===")
print(f"模型: {settings.llm.model}")
print(f"温度: {settings.llm.temperature}")
print(f"API 密钥: {settings.llm.api_key[:10]}...")
```

**.env 文件：**

```bash
APP_NAME=AI Agent API
DEBUG=True

# 数据库配置（使用 __ 分隔）
DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__USERNAME=admin
DATABASE__PASSWORD=password123
DATABASE__DATABASE=agent_db
DATABASE__POOL_SIZE=20

# Redis 配置
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__DB=0

# LLM 配置
LLM__API_KEY=sk-proj-xxxxxxxxxxxxx
LLM__MODEL=gpt-4
LLM__TEMPERATURE=0.7
```

---

### 示例6：使用 SecretStr 隐藏敏感信息

```python
"""
使用 SecretStr 隐藏敏感信息
演示：在日志和打印中自动隐藏敏感信息
"""
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 使用 SecretStr 隐藏敏感信息
    openai_api_key: SecretStr
    database_password: SecretStr
    secret_key: SecretStr

    # 普通字段
    database_url: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()

# 打印配置（敏感信息被隐藏）
print("=== 配置信息 ===")
print(f"数据库 URL: {settings.database_url}")
print(f"调试模式: {settings.debug}")
print(f"API 密钥: {settings.openai_api_key}")  # 自动隐藏
print(f"数据库密码: {settings.database_password}")  # 自动隐藏
print(f"密钥: {settings.secret_key}")  # 自动隐藏

# 获取真实值
print("\n=== 真实值 ===")
real_api_key = settings.openai_api_key.get_secret_value()
print(f"API 密钥: {real_api_key[:10]}...")

# 打印整个配置对象
print("\n=== 配置对象 ===")
print(settings)
```

**运行输出：**

```
=== 配置信息 ===
数据库 URL: postgresql://localhost:5432/mydb
调试模式: True
API 密钥: SecretStr('**********')
数据库密码: SecretStr('**********')
密钥: SecretStr('**********')

=== 真实值 ===
API 密钥: sk-proj-xx...

=== 配置对象 ===
Settings(
    openai_api_key=SecretStr('**********'),
    database_password=SecretStr('**********'),
    secret_key=SecretStr('**********'),
    database_url='postgresql://localhost:5432/mydb',
    debug=True
)
```

---

### 示例7：配置导出

```python
"""
配置导出
演示：将配置导出为字典或 JSON
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    debug: bool = False
    port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()

# 导出为字典
config_dict = settings.model_dump()
print("=== 导出为字典 ===")
for key, value in config_dict.items():
    if "key" in key.lower() or "password" in key.lower():
        print(f"{key}: {value[:10]}...")
    else:
        print(f"{key}: {value}")

# 导出为 JSON
config_json = settings.model_dump_json(indent=2)
print("\n=== 导出为 JSON ===")
print(config_json)

# 排除敏感字段
config_safe = settings.model_dump(exclude={"openai_api_key"})
print("\n=== 排除敏感字段 ===")
print(config_safe)
```

---

### 示例8：FastAPI 依赖注入

```python
"""
FastAPI 依赖注入
演示：在 FastAPI 中使用 Pydantic Settings
"""
from functools import lru_cache
from fastapi import FastAPI, Depends
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AI Agent API"
    debug: bool = False
    database_url: str
    openai_api_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """获取配置（缓存）"""
    return Settings()

# 创建 FastAPI 应用
app = FastAPI()

@app.get("/")
def read_root(settings: Settings = Depends(get_settings)):
    """根路径"""
    return {
        "app_name": settings.app_name,
        "debug": settings.debug
    }

@app.get("/config")
def get_config(settings: Settings = Depends(get_settings)):
    """获取配置"""
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "database_url": settings.database_url.split("@")[1] if "@" in settings.database_url else "未设置"
    }

@app.get("/health")
def health_check(settings: Settings = Depends(get_settings)):
    """健康检查"""
    return {
        "status": "healthy",
        "app_name": settings.app_name
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=settings.debug)
```

---

### 示例9：多环境配置

```python
"""
多环境配置
演示：根据环境加载不同的配置
"""
import os
from enum import Enum
from pydantic_settings import BaseSettings

class Environment(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"

class Settings(BaseSettings):
    env: Environment = Environment.DEV
    debug: bool = False
    database_url: str
    openai_api_key: str

    class Config:
        # 根据环境变量选择配置文件
        env_file = f".env.{os.getenv('ENV', 'dev')}"

# 使用
settings = Settings()

print(f"✅ 当前环境: {settings.env}")
print(f"✅ 调试模式: {settings.debug}")
print(f"✅ 数据库: {settings.database_url}")

# 根据环境调整行为
if settings.env == Environment.DEV:
    print("🔧 开发环境：启用详细日志")
elif settings.env == Environment.PROD:
    print("🚀 生产环境：启用性能优化")
```

---

### 示例10：完整的配置系统

```python
"""
完整的配置系统
演示：生产级的配置管理系统
"""
import os
from enum import Enum
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings

class Environment(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"

class DatabaseConfig(BaseModel):
    """数据库配置"""
    url: str = Field(..., description="数据库连接字符串")
    pool_size: int = Field(10, ge=1, le=100, description="连接池大小")
    max_overflow: int = Field(20, ge=0, le=100, description="最大溢出连接数")
    pool_timeout: int = Field(30, ge=1, le=300, description="连接池超时（秒）")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith("postgresql://"):
            raise ValueError("只支持 PostgreSQL 数据库")
        return v

class LLMConfig(BaseModel):
    """LLM 配置"""
    api_key: SecretStr = Field(..., description="API 密钥")
    base_url: str = Field("https://api.openai.com/v1", description="API 端点")
    model: str = Field("gpt-4", description="模型名称")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(2000, ge=1, le=100000, description="最大 token 数")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        key = v.get_secret_value()
        if not key.startswith("sk-"):
            raise ValueError("OpenAI API 密钥格式错误")
        return v

class Settings(BaseSettings):
    """应用配置"""
    # 环境标识
    env: Environment = Environment.DEV

    # 应用配置
    app_name: str = Field("AI Agent API", description="应用名称")
    debug: bool = Field(False, description="调试模式")
    log_level: str = Field("INFO", description="日志级别")

    # 数据库配置
    database: DatabaseConfig

    # LLM 配置
    llm: LLMConfig

    # 安全配置
    secret_key: SecretStr = Field(..., min_length=32, description="JWT 密钥")

    # Redis 配置（可选）
    redis_url: str | None = Field(None, description="Redis 连接字符串")

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
        env_file = f".env.{os.getenv('ENV', 'dev')}"
        env_nested_delimiter = "__"

    def __repr__(self):
        """隐藏敏感信息"""
        return (
            f"Settings("
            f"env={self.env}, "
            f"debug={self.debug}, "
            f"database_url=*****, "
            f"llm_api_key=*****, "
            f"secret_key=*****"
            f")"
        )

# 启动时加载配置
try:
    settings = Settings()
    print("✅ 配置加载成功")
    print(f"   环境: {settings.env}")
    print(f"   应用名称: {settings.app_name}")
    print(f"   调试模式: {settings.debug}")
    print(f"   日志级别: {settings.log_level}")
    print(f"   数据库: {settings.database.url}")
    print(f"   LLM 模型: {settings.llm.model}")
    print(f"   Redis: {settings.redis_url or '未配置'}")
except Exception as e:
    print(f"❌ 配置错误: {e}")
    import sys
    sys.exit(1)
```

**.env.dev 文件：**

```bash
ENV=dev
DEBUG=True
LOG_LEVEL=DEBUG

DATABASE__URL=postgresql://localhost:5432/dev_db
DATABASE__POOL_SIZE=10

LLM__API_KEY=sk-dev-xxxxxxxxxxxxx
LLM__MODEL=gpt-3.5-turbo
LLM__TEMPERATURE=0.7

SECRET_KEY=dev-secret-key-xxxxxxxxxxxxx

REDIS_URL=redis://localhost:6379/0
```

---

## 运行说明

### 1. 安装依赖

```bash
pip install pydantic pydantic-settings python-dotenv
```

### 2. 创建配置文件

```bash
# 创建 .env 文件
cat > .env << EOF
DATABASE_URL=postgresql://localhost:5432/mydb
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
DEBUG=True
PORT=8000
SECRET_KEY=your-secret-key-at-least-32-characters-long
EOF
```

### 3. 运行示例

```bash
# 运行基础示例
python example1.py

# 运行 FastAPI 示例
python example8.py
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
load_dotenv()

class Settings(BaseSettings):
    database_url: str  # 从系统环境变量读取
```

### Q2: 如何在测试中覆盖配置？

**A:** 使用依赖注入或直接传递参数。

```python
# 测试时覆盖配置
def test_api():
    test_settings = Settings(
        database_url="postgresql://test:5432/test_db",
        openai_api_key="sk-test-xxx"
    )
    # 使用 test_settings
```

### Q3: 如何处理可选配置？

**A:** 使用 `Optional` 或 `None` 默认值。

```python
from typing import Optional

class Settings(BaseSettings):
    redis_url: Optional[str] = None  # 可选配置
    cache_ttl: int = 3600  # 有默认值
```

---

## 总结

**Pydantic Settings 的核心优势：**

1. **类型安全**：自动类型转换和验证
2. **启动时验证**：缺少必需字段时立即报错
3. **IDE 支持**：完整的类型提示和自动补全
4. **自定义验证**：field_validator 和 model_validator
5. **嵌套配置**：支持复杂的配置结构
6. **敏感信息保护**：SecretStr 自动隐藏敏感信息

**最佳实践：**
- 使用 Field 添加描述和验证规则
- 使用 SecretStr 保护敏感信息
- 使用 field_validator 自定义验证逻辑
- 使用 model_validator 跨字段验证
- 使用嵌套模型组织复杂配置
- 在 FastAPI 中使用依赖注入
- 使用 lru_cache 缓存配置对象
