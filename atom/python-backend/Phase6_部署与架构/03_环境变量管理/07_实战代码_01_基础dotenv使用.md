# 实战代码1：基础 dotenv 使用

## 场景说明

演示如何使用 `python-dotenv` 加载 `.env` 文件，读取环境变量，并在 FastAPI 应用中使用。

---

## 完整代码示例

### 示例1：最简单的 dotenv 使用

```python
"""
最简单的 dotenv 使用示例
演示：加载 .env 文件并读取环境变量
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取环境变量
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 打印配置
print("=== 配置信息 ===")
print(f"数据库: {DATABASE_URL}")
print(f"API 密钥: {OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "未设置")
print(f"调试模式: {DEBUG}")
```

**.env 文件：**

```bash
# .env
DATABASE_URL=postgresql://localhost:5432/mydb
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
DEBUG=True
```

**运行输出：**

```
=== 配置信息 ===
数据库: postgresql://localhost:5432/mydb
API 密钥: sk-proj-xx...
调试模式: True
```

---

### 示例2：指定 .env 文件路径

```python
"""
指定 .env 文件路径
演示：加载不同路径的 .env 文件
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# 方式1：相对路径
load_dotenv(".env.dev")

# 方式2：绝对路径
env_path = Path(__file__).parent / ".env.dev"
load_dotenv(env_path)

# 方式3：自动查找（向上查找父目录）
from dotenv import find_dotenv
load_dotenv(find_dotenv())

# 读取配置
DATABASE_URL = os.getenv("DATABASE_URL")
print(f"数据库: {DATABASE_URL}")
```

---

### 示例3：覆盖已有环境变量

```python
"""
覆盖已有环境变量
演示：.env 文件和系统环境变量的优先级
"""
import os
from dotenv import load_dotenv

# 设置系统环境变量
os.environ["DEBUG"] = "False"

# 默认不覆盖系统环境变量
load_dotenv()
print(f"DEBUG (不覆盖): {os.getenv('DEBUG')}")  # False

# 强制覆盖系统环境变量
load_dotenv(override=True)
print(f"DEBUG (覆盖): {os.getenv('DEBUG')}")  # True（来自 .env 文件）
```

**.env 文件：**

```bash
DEBUG=True
```

**运行输出：**

```
DEBUG (不覆盖): False
DEBUG (覆盖): True
```

---

### 示例4：读取为字典（不加载到环境变量）

```python
"""
读取为字典
演示：使用 dotenv_values() 读取配置为字典
"""
from dotenv import dotenv_values

# 读取为字典（不加载到系统环境变量）
config = dotenv_values(".env")

print("=== 配置字典 ===")
for key, value in config.items():
    if "KEY" in key or "PASSWORD" in key:
        print(f"{key}: {value[:10]}...")
    else:
        print(f"{key}: {value}")

# 使用配置
DATABASE_URL = config.get("DATABASE_URL")
OPENAI_API_KEY = config.get("OPENAI_API_KEY")

print(f"\n数据库: {DATABASE_URL}")
print(f"API 密钥: {OPENAI_API_KEY[:10]}...")
```

---

### 示例5：合并多个配置文件

```python
"""
合并多个配置文件
演示：从多个 .env 文件读取并合并配置
"""
from dotenv import dotenv_values

# 读取多个配置文件
base_config = dotenv_values(".env")
dev_config = dotenv_values(".env.dev")
local_config = dotenv_values(".env.local")

# 合并配置（后者覆盖前者）
config = {**base_config, **dev_config, **local_config}

print("=== 合并后的配置 ===")
for key, value in config.items():
    print(f"{key}: {value}")
```

---

### 示例6：验证必需的环境变量

```python
"""
验证必需的环境变量
演示：检查必需的配置是否存在
"""
import os
import sys
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 必需的环境变量
REQUIRED_VARS = [
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "SECRET_KEY"
]

# 验证配置
missing_vars = []
for var in REQUIRED_VARS:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"❌ 错误：缺少必需的环境变量: {', '.join(missing_vars)}")
    print("\n请在 .env 文件中设置以下变量：")
    for var in missing_vars:
        print(f"  {var}=your_value_here")
    sys.exit(1)

print("✅ 所有必需的环境变量都已设置")
```

---

### 示例7：类型转换

```python
"""
类型转换
演示：将环境变量转换为正确的类型
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 字符串（默认）
DATABASE_URL = os.getenv("DATABASE_URL")

# 布尔值
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes", "on")

# 整数
PORT = int(os.getenv("PORT", "8000"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))

# 浮点数
TIMEOUT = float(os.getenv("TIMEOUT", "30.5"))

# 列表（逗号分隔）
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost").split(",")

print("=== 类型转换后的配置 ===")
print(f"DATABASE_URL (str): {DATABASE_URL}")
print(f"DEBUG (bool): {DEBUG}")
print(f"PORT (int): {PORT}")
print(f"MAX_CONNECTIONS (int): {MAX_CONNECTIONS}")
print(f"TIMEOUT (float): {TIMEOUT}")
print(f"ALLOWED_HOSTS (list): {ALLOWED_HOSTS}")
```

**.env 文件：**

```bash
DATABASE_URL=postgresql://localhost:5432/mydb
DEBUG=True
PORT=8000
MAX_CONNECTIONS=100
TIMEOUT=30.5
ALLOWED_HOSTS=localhost,127.0.0.1,example.com
```

**运行输出：**

```
=== 类型转换后的配置 ===
DATABASE_URL (str): postgresql://localhost:5432/mydb
DEBUG (bool): True
PORT (int): 8000
MAX_CONNECTIONS (int): 100
TIMEOUT (float): 30.5
ALLOWED_HOSTS (list): ['localhost', '127.0.0.1', 'example.com']
```

---

### 示例8：FastAPI 应用中使用 dotenv

```python
"""
FastAPI 应用中使用 dotenv
演示：在 FastAPI 应用中加载和使用环境变量
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI

# 加载环境变量
load_dotenv()

# 读取配置
APP_NAME = os.getenv("APP_NAME", "My API")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 创建 FastAPI 应用
app = FastAPI(
    title=APP_NAME,
    debug=DEBUG
)

@app.on_event("startup")
async def startup():
    """应用启动时打印配置"""
    print("🚀 启动应用")
    print(f"📍 应用名称: {APP_NAME}")
    print(f"🐛 调试模式: {DEBUG}")
    print(f"🗄️  数据库: {DATABASE_URL}")
    print(f"🤖 API 密钥: {OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "未设置")

@app.get("/")
def read_root():
    """根路径"""
    return {
        "app_name": APP_NAME,
        "debug": DEBUG
    }

@app.get("/config")
def get_config():
    """获取配置信息（隐藏敏感信息）"""
    return {
        "app_name": APP_NAME,
        "debug": DEBUG,
        "database_url": DATABASE_URL.split("@")[1] if DATABASE_URL and "@" in DATABASE_URL else "未设置",
        "openai_api_key": f"{OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "未设置"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**.env 文件：**

```bash
APP_NAME=AI Agent API
DEBUG=True
PORT=8000
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

**运行：**

```bash
python app.py
```

**输出：**

```
🚀 启动应用
📍 应用名称: AI Agent API
🐛 调试模式: True
🗄️  数据库: postgresql://user:password@localhost:5432/mydb
🤖 API 密钥: sk-proj-xx...
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### 示例9：多环境配置

```python
"""
多环境配置
演示：根据环境变量加载不同的配置文件
"""
import os
from dotenv import load_dotenv

# 检测环境
env = os.getenv("ENV", "dev")

# 加载对应的配置文件
env_file = f".env.{env}"
load_dotenv(env_file)

print(f"✅ 当前环境: {env}")
print(f"✅ 加载配置文件: {env_file}")

# 读取配置
DATABASE_URL = os.getenv("DATABASE_URL")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

print(f"🗄️  数据库: {DATABASE_URL}")
print(f"🐛 调试模式: {DEBUG}")
```

**配置文件：**

```bash
# .env.dev
DATABASE_URL=postgresql://localhost:5432/dev_db
DEBUG=True

# .env.test
DATABASE_URL=postgresql://test-db:5432/test_db
DEBUG=False

# .env.prod
DATABASE_URL=postgresql://prod-db:5432/prod_db
DEBUG=False
```

**使用：**

```bash
# 开发环境
ENV=dev python app.py

# 测试环境
ENV=test python app.py

# 生产环境
ENV=prod python app.py
```

---

### 示例10：动态修改 .env 文件

```python
"""
动态修改 .env 文件
演示：使用 set_key() 和 get_key() 动态修改配置
"""
from dotenv import set_key, get_key, load_dotenv

env_file = ".env"

# 读取单个键
api_key = get_key(env_file, "OPENAI_API_KEY")
print(f"当前 API 密钥: {api_key[:10]}..." if api_key else "未设置")

# 设置或更新键
set_key(env_file, "OPENAI_API_KEY", "sk-new-key-xxxxxxxxxxxxx")
print("✅ 已更新 API 密钥")

# 添加新键
set_key(env_file, "NEW_CONFIG", "new_value")
print("✅ 已添加新配置")

# 重新加载配置
load_dotenv(env_file, override=True)

# 验证修改
new_api_key = get_key(env_file, "OPENAI_API_KEY")
new_config = get_key(env_file, "NEW_CONFIG")

print(f"\n新的 API 密钥: {new_api_key[:10]}...")
print(f"新的配置: {new_config}")
```

---

## 完整项目示例

### 项目结构

```
project/
├── .env
├── .env.example
├── .gitignore
├── app.py
└── config.py
```

### config.py

```python
"""
配置模块
集中管理所有配置
"""
import os
import sys
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 必需的环境变量
REQUIRED_VARS = [
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "SECRET_KEY"
]

# 验证配置
def validate_config():
    """验证必需的环境变量"""
    missing_vars = []
    for var in REQUIRED_VARS:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ 错误：缺少必需的环境变量: {', '.join(missing_vars)}")
        print("\n请复制 .env.example 为 .env 并填入真实的值")
        sys.exit(1)

# 启动时验证
validate_config()

# 应用配置
APP_NAME = os.getenv("APP_NAME", "My API")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", "8000"))

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL")

# LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# 安全配置
SECRET_KEY = os.getenv("SECRET_KEY")

print("✅ 配置加载成功")
```

### app.py

```python
"""
FastAPI 应用
"""
from fastapi import FastAPI
from config import APP_NAME, DEBUG, PORT, DATABASE_URL, OPENAI_API_KEY

app = FastAPI(
    title=APP_NAME,
    debug=DEBUG
)

@app.on_event("startup")
async def startup():
    print("🚀 启动应用")
    print(f"📍 应用名称: {APP_NAME}")
    print(f"🐛 调试模式: {DEBUG}")
    print(f"🗄️  数据库: {DATABASE_URL}")
    print(f"🤖 API 密钥: {OPENAI_API_KEY[:10]}...")

@app.get("/")
def read_root():
    return {
        "app_name": APP_NAME,
        "debug": DEBUG
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

### .env.example

```bash
# 应用配置
APP_NAME=My API
DEBUG=True
PORT=8000

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

# LLM 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# 安全配置
SECRET_KEY=your-secret-key-at-least-32-characters-long
```

### .gitignore

```bash
# 环境变量文件
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
.venv/
```

---

## 运行说明

### 1. 安装依赖

```bash
pip install python-dotenv fastapi uvicorn
```

### 2. 创建配置文件

```bash
cp .env.example .env
```

### 3. 编辑 .env 文件

```bash
# 填入真实的配置
vim .env
```

### 4. 运行应用

```bash
python app.py
```

---

## 常见问题

### Q1: .env 文件不生效？

**A:** 确保在导入其他模块前调用 `load_dotenv()`。

```python
# ✅ 正确：先加载环境变量
from dotenv import load_dotenv
load_dotenv()

from config import settings

# ❌ 错误：后加载环境变量
from config import settings

from dotenv import load_dotenv
load_dotenv()  # 太晚了，settings 已经加载
```

### Q2: 如何在 Docker 中使用 .env 文件？

**A:** 使用 `--env-file` 参数。

```bash
docker run --env-file .env my-app
```

### Q3: 如何处理多行值？

**A:** 使用引号包裹。

```bash
# .env
PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"
```

---

## 总结

**python-dotenv 的核心用法：**

1. **基础加载**：`load_dotenv()`
2. **指定路径**：`load_dotenv(".env.dev")`
3. **强制覆盖**：`load_dotenv(override=True)`
4. **读取为字典**：`dotenv_values(".env")`
5. **自动查找**：`load_dotenv(find_dotenv())`
6. **动态修改**：`set_key()` 和 `get_key()`

**最佳实践：**
- 在应用启动时立即加载
- 验证必需的环境变量
- 使用 .env.example 作为模板
- .env 文件不提交到 git
- 类型转换要处理默认值
