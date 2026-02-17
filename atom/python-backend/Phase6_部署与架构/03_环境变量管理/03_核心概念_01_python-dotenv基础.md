# 核心概念1：python-dotenv 基础

## 一句话定义

**python-dotenv 是一个将 `.env` 文件中的键值对加载到系统环境变量的 Python 库，让你可以用文件管理配置而不是在命令行中逐个设置。**

---

## 为什么需要 python-dotenv？

### 问题场景

在没有 python-dotenv 之前，你需要这样设置环境变量：

```bash
# 每次启动应用前都要设置
export DATABASE_URL=postgresql://localhost:5432/mydb
export OPENAI_API_KEY=sk-xxx
export SECRET_KEY=your-secret-key
export DEBUG=True

# 然后启动应用
python app.py
```

**问题：**
- 每次启动都要手动设置，容易遗漏
- 配置分散在命令行历史中，难以管理
- 团队协作时，每个人都要手动配置
- 配置容易出错（拼写错误、值错误）

### python-dotenv 的解决方案

```bash
# .env 文件（一次配置，永久使用）
DATABASE_URL=postgresql://localhost:5432/mydb
OPENAI_API_KEY=sk-xxx
SECRET_KEY=your-secret-key
DEBUG=True
```

```python
# app.py
from dotenv import load_dotenv
import os

# 一行代码加载所有配置
load_dotenv()

# 使用配置
db_url = os.getenv("DATABASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
```

---

## 核心功能

### 功能1：加载 .env 文件

**基础用法：**

```python
from dotenv import load_dotenv
import os

# 加载当前目录的 .env 文件
load_dotenv()

# 读取环境变量
database_url = os.getenv("DATABASE_URL")
print(database_url)  # postgresql://localhost:5432/mydb
```

**指定文件路径：**

```python
from dotenv import load_dotenv

# 加载指定路径的 .env 文件
load_dotenv(".env.dev")
load_dotenv("/path/to/.env")
```

**覆盖已有环境变量：**

```python
# 默认不覆盖已有的系统环境变量
load_dotenv()  # 系统环境变量优先

# 强制覆盖
load_dotenv(override=True)  # .env 文件优先
```

---

### 功能2：dotenv_values() - 读取为字典

**不加载到系统环境变量，只读取为字典：**

```python
from dotenv import dotenv_values

# 读取为字典
config = dotenv_values(".env")

print(config)
# {
#     'DATABASE_URL': 'postgresql://localhost:5432/mydb',
#     'OPENAI_API_KEY': 'sk-xxx',
#     'DEBUG': 'True'
# }

# 使用配置
database_url = config["DATABASE_URL"]
```

**用途：** 当你不想污染系统环境变量，只想读取配置时使用。

---

### 功能3：find_dotenv() - 自动查找 .env 文件

**自动向上查找 .env 文件：**

```python
from dotenv import load_dotenv, find_dotenv

# 从当前目录向上查找 .env 文件
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

print(f"找到的 .env 文件: {dotenv_path}")
```

**查找规则：**
- 从当前目录开始
- 向上查找父目录
- 找到第一个 `.env` 文件就停止

**用途：** 在子目录中运行脚本时，自动找到项目根目录的 `.env` 文件。

---

### 功能4：set_key() 和 get_key() - 动态修改 .env

**读取单个键：**

```python
from dotenv import get_key

# 读取单个键
api_key = get_key(".env", "OPENAI_API_KEY")
print(api_key)  # sk-xxx
```

**设置单个键：**

```python
from dotenv import set_key

# 设置或更新键值
set_key(".env", "NEW_KEY", "new_value")

# 如果键不存在，会添加
# 如果键已存在，会更新
```

**用途：** 动态修改配置文件，如在安装脚本中自动生成配置。

---

## .env 文件格式

### 基础格式

```bash
# 注释以 # 开头
# 键值对格式：KEY=VALUE

# 字符串（不需要引号）
DATABASE_URL=postgresql://localhost:5432/mydb
API_KEY=sk-xxx

# 布尔值（字符串形式）
DEBUG=True
ENABLE_CACHE=False

# 数字（字符串形式）
PORT=8000
MAX_CONNECTIONS=100

# 空值
OPTIONAL_KEY=
```

### 引号规则

```bash
# 不需要引号
NAME=John

# 单引号（保留原样）
MESSAGE='Hello World'

# 双引号（支持转义）
MESSAGE="Hello\nWorld"

# 包含空格（需要引号）
MESSAGE="Hello World"
MESSAGE='Hello World'

# 包含特殊字符
PASSWORD="p@ssw0rd!"
```

### 变量引用

```bash
# 引用其他变量
BASE_URL=http://localhost
API_URL=${BASE_URL}/api

# 默认值
DATABASE_URL=${DATABASE_URL:-postgresql://localhost:5432/mydb}
```

**注意：** python-dotenv 默认不支持变量引用，需要手动处理或使用其他库。

---

## 实际应用示例

### 示例1：FastAPI 应用配置

```python
# .env
DATABASE_URL=postgresql://localhost:5432/mydb
OPENAI_API_KEY=sk-xxx
SECRET_KEY=your-secret-key
DEBUG=True

# app/main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

app = FastAPI(
    debug=os.getenv("DEBUG", "False").lower() == "true"
)

@app.get("/")
def read_root():
    return {
        "database": os.getenv("DATABASE_URL"),
        "debug": os.getenv("DEBUG")
    }
```

---

### 示例2：多环境配置

```python
# config.py
import os
from dotenv import load_dotenv

# 根据环境变量选择配置文件
env = os.getenv("ENV", "dev")
dotenv_path = f".env.{env}"

print(f"加载配置文件: {dotenv_path}")
load_dotenv(dotenv_path)

# 读取配置
DATABASE_URL = os.getenv("DATABASE_URL")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

print(f"数据库: {DATABASE_URL}")
print(f"调试模式: {DEBUG}")
```

```bash
# 使用不同环境
ENV=dev python app.py    # 加载 .env.dev
ENV=test python app.py   # 加载 .env.test
ENV=prod python app.py   # 加载 .env.prod
```

---

### 示例3：配置验证

```python
from dotenv import load_dotenv
import os
import sys

# 加载环境变量
load_dotenv()

# 必需的配置
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
    print(f"错误：缺少必需的环境变量: {', '.join(missing_vars)}")
    sys.exit(1)

print("配置验证通过！")
```

---

### 示例4：开发工具集成

```python
# scripts/setup.py
"""
项目初始化脚本
"""
from dotenv import set_key
import secrets

# 创建 .env 文件
env_file = ".env"

# 生成随机密钥
secret_key = secrets.token_urlsafe(32)

# 设置默认配置
set_key(env_file, "DATABASE_URL", "postgresql://localhost:5432/mydb")
set_key(env_file, "SECRET_KEY", secret_key)
set_key(env_file, "DEBUG", "True")

print(f"✅ 已创建 {env_file}")
print(f"✅ 已生成随机密钥: {secret_key[:10]}...")
```

---

## 常见问题

### Q1: load_dotenv() 会覆盖已有的环境变量吗？

**A:** 默认不会。系统环境变量优先级更高。

```python
# 系统环境变量
export DEBUG=False

# .env 文件
DEBUG=True

# Python 代码
load_dotenv()
print(os.getenv("DEBUG"))  # False（系统环境变量优先）

# 强制覆盖
load_dotenv(override=True)
print(os.getenv("DEBUG"))  # True（.env 文件覆盖）
```

---

### Q2: .env 文件中的值都是字符串吗？

**A:** 是的。所有值都是字符串，需要手动转换。

```python
# .env
DEBUG=True
PORT=8000

# Python 代码
debug = os.getenv("DEBUG")  # "True"（字符串）
port = os.getenv("PORT")    # "8000"（字符串）

# 需要手动转换
debug = os.getenv("DEBUG", "False").lower() == "true"  # 布尔值
port = int(os.getenv("PORT", "8000"))  # 整数
```

---

### Q3: 如何在 .env 文件中使用多行值？

**A:** 使用引号包裹。

```bash
# .env
PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"
```

---

### Q4: load_dotenv() 应该在哪里调用？

**A:** 在应用启动的最早阶段，通常在 `main.py` 或 `config.py` 的顶部。

```python
# main.py
from dotenv import load_dotenv

# 第一件事：加载环境变量
load_dotenv()

# 然后导入其他模块（它们可能依赖环境变量）
from app.config import settings
from app.database import engine
```

---

## 最佳实践

### 1. .env 文件不要提交到 git

```bash
# .gitignore
.env
.env.local
.env.*.local
```

### 2. 提供 .env.example 作为模板

```bash
# .env.example（提交到 git）
DATABASE_URL=postgresql://localhost:5432/dbname
OPENAI_API_KEY=your_key_here
SECRET_KEY=your-secret-key-here
DEBUG=False
```

### 3. 在应用启动时加载

```python
# main.py
from dotenv import load_dotenv

# 应用启动时立即加载
load_dotenv()

# 后续代码可以使用环境变量
```

### 4. 使用 find_dotenv() 自动查找

```python
from dotenv import load_dotenv, find_dotenv

# 自动查找 .env 文件
load_dotenv(find_dotenv())
```

### 5. 验证必需的环境变量

```python
from dotenv import load_dotenv
import os

load_dotenv()

# 验证必需的配置
assert os.getenv("DATABASE_URL"), "DATABASE_URL is required"
assert os.getenv("SECRET_KEY"), "SECRET_KEY is required"
```

---

## 与其他工具的对比

| 特性 | python-dotenv | os.environ | configparser |
|------|---------------|------------|--------------|
| 配置格式 | KEY=VALUE | 系统环境变量 | INI 格式 |
| 类型转换 | 手动 | 手动 | 手动 |
| 嵌套配置 | 不支持 | 不支持 | 支持 |
| 变量引用 | 不支持 | 不支持 | 不支持 |
| 12-Factor App | ✅ 符合 | ✅ 符合 | ❌ 不符合 |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 在 AI Agent 后端中的应用

### 场景1：LLM API 配置

```bash
# .env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

```python
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL"),
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

### 场景2：数据库连接

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost:5432/agent_db
DATABASE_POOL_SIZE=10
```

```python
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(
    os.getenv("DATABASE_URL"),
    pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10"))
)
```

---

### 场景3：向量数据库配置

```bash
# .env
CHROMA_HOST=localhost
CHROMA_PORT=8000
EMBEDDING_MODEL=text-embedding-3-small
```

```python
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST"),
    port=int(os.getenv("CHROMA_PORT"))
)
```

---

## 总结

**python-dotenv 的核心价值：**

1. **简化配置管理**：用文件管理配置，不用手动设置环境变量
2. **符合 12-Factor App**：配置与代码分离
3. **易于使用**：一行代码加载所有配置
4. **团队协作友好**：每个人有自己的 `.env` 文件
5. **安全**：`.env` 不提交到 git，保护敏感信息

**关键 API：**
- `load_dotenv()`：加载 .env 文件到系统环境变量
- `dotenv_values()`：读取 .env 文件为字典
- `find_dotenv()`：自动查找 .env 文件
- `set_key()` / `get_key()`：动态修改 .env 文件

**最佳实践：**
- .env 不提交到 git
- 提供 .env.example 作为模板
- 在应用启动时立即加载
- 验证必需的环境变量
- 使用 Pydantic Settings 进行类型验证（下一个核心概念）
