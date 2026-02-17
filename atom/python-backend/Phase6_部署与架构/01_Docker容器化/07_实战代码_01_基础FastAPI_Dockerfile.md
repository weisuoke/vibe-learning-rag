# 实战代码1：基础 FastAPI Dockerfile

## 目标

从零编写一个简单的 FastAPI 应用 Dockerfile，理解容器化的基本流程。

---

## 项目结构

```
my-agent-api/
├── app/
│   ├── __init__.py
│   └── main.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

---

## 步骤1：创建 FastAPI 应用

### app/__init__.py

```python
# 空文件，标记 app 为 Python 包
```

### app/main.py

```python
"""
基础 FastAPI 应用
演示：Docker 容器化的最小示例
"""

from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="AI Agent API",
    description="基础 FastAPI 应用示例",
    version="1.0.0"
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Hello from Docker!",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "ai-agent-api",
        "version": "1.0.0"
    }

@app.get("/api/info")
async def info():
    """API 信息"""
    return {
        "name": "AI Agent API",
        "description": "基础 FastAPI 应用",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "根路径"},
            {"path": "/health", "method": "GET", "description": "健康检查"},
            {"path": "/api/info", "method": "GET", "description": "API 信息"},
            {"path": "/docs", "method": "GET", "description": "API 文档"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 步骤2：创建依赖文件

### requirements.txt

```txt
# Web 框架
fastapi==0.109.0

# ASGI 服务器
uvicorn[standard]==0.27.0

# 数据验证
pydantic==2.5.3
```

---

## 步骤3：创建 Dockerfile

### Dockerfile

```dockerfile
# 使用官方 Python 基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 声明端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**指令说明：**

1. **FROM python:3.13-slim**
   - 使用 Python 3.13 精简版基础镜像
   - 体积约 200MB

2. **WORKDIR /app**
   - 设置工作目录为 /app
   - 后续指令都在此目录执行

3. **COPY requirements.txt .**
   - 先复制依赖文件
   - 利用 Docker 层缓存

4. **RUN pip install --no-cache-dir -r requirements.txt**
   - 安装依赖
   - --no-cache-dir 减小镜像体积

5. **COPY . .**
   - 复制应用代码
   - 代码修改后，只重新执行这一步

6. **EXPOSE 8000**
   - 声明容器监听 8000 端口
   - 仅文档作用

7. **CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]**
   - 启动 FastAPI 应用
   - --host 0.0.0.0 监听所有网络接口

---

## 步骤4：创建 .dockerignore

### .dockerignore

```
# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# 虚拟环境
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# 测试
.pytest_cache/
.coverage
htmlcov/

# 其他
.DS_Store
*.log
```

---

## 步骤5：构建镜像

```bash
# 进入项目目录
cd my-agent-api

# 构建镜像
docker build -t my-agent-api:v1.0 .

# 输出示例
[+] Building 45.2s (10/10) FINISHED
 => [1/5] FROM python:3.13-slim
 => [2/5] WORKDIR /app
 => [3/5] COPY requirements.txt .
 => [4/5] RUN pip install --no-cache-dir -r requirements.txt
 => [5/5] COPY . .
 => exporting to image
 => => naming to docker.io/library/my-agent-api:v1.0

# 查看镜像
docker images my-agent-api
REPOSITORY      TAG    SIZE
my-agent-api    v1.0   250MB
```

---

## 步骤6：运行容器

```bash
# 运行容器
docker run -d \
  --name my-api \
  -p 8000:8000 \
  my-agent-api:v1.0

# 输出容器 ID
abc123def456...

# 查看运行中的容器
docker ps
CONTAINER ID   IMAGE              STATUS         PORTS
abc123         my-agent-api:v1.0  Up 5 seconds   0.0.0.0:8000->8000/tcp
```

---

## 步骤7：测试应用

### 测试根路径

```bash
curl http://localhost:8000

# 输出
{
  "message": "Hello from Docker!",
  "timestamp": "2026-02-12T10:00:00.123456"
}
```

### 测试健康检查

```bash
curl http://localhost:8000/health

# 输出
{
  "status": "healthy",
  "service": "ai-agent-api",
  "version": "1.0.0"
}
```

### 测试 API 信息

```bash
curl http://localhost:8000/api/info

# 输出
{
  "name": "AI Agent API",
  "description": "基础 FastAPI 应用",
  "endpoints": [
    {"path": "/", "method": "GET", "description": "根路径"},
    {"path": "/health", "method": "GET", "description": "健康检查"},
    {"path": "/api/info", "method": "GET", "description": "API 信息"},
    {"path": "/docs", "method": "GET", "description": "API 文档"}
  ]
}
```

### 访问 API 文档

```bash
# 在浏览器中打开
open http://localhost:8000/docs
```

---

## 步骤8：查看日志

```bash
# 查看容器日志
docker logs my-api

# 输出
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

# 实时查看日志
docker logs -f my-api
```

---

## 步骤9：进入容器调试

```bash
# 进入容器
docker exec -it my-api bash

# 容器内操作
root@abc123:/app# ls
app  requirements.txt

root@abc123:/app# python -c "import fastapi; print(fastapi.__version__)"
0.109.0

root@abc123:/app# exit
```

---

## 步骤10：停止和删除容器

```bash
# 停止容器
docker stop my-api

# 删除容器
docker rm my-api

# 或者一条命令
docker rm -f my-api
```

---

## 常见问题

### Q1: 为什么要先复制 requirements.txt？

**答：** 利用 Docker 层缓存。

```dockerfile
# ✅ 推荐：先复制依赖文件
COPY requirements.txt .
RUN pip install -r requirements.txt  # 依赖不变时，缓存命中
COPY . .  # 代码修改，只重新执行这一步

# ❌ 不推荐：一次性复制所有文件
COPY . .
RUN pip install -r requirements.txt  # 代码修改，依赖重新安装
```

### Q2: 为什么要用 --no-cache-dir？

**答：** 减小镜像体积。

```bash
# 不使用 --no-cache-dir
RUN pip install -r requirements.txt
# 镜像体积：350MB（包含 pip 缓存）

# 使用 --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt
# 镜像体积：250MB（不包含 pip 缓存）
```

### Q3: 为什么要用 --host 0.0.0.0？

**答：** 容器内的服务需要监听所有网络接口。

```bash
# ❌ 错误：只监听 localhost
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1"]
# 宿主机无法访问容器服务

# ✅ 正确：监听所有网络接口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
# 宿主机可以通过端口映射访问
```

### Q4: 如何修改代码并重新构建？

```bash
# 1. 修改代码
echo "print('Updated')" >> app/main.py

# 2. 重新构建镜像
docker build -t my-agent-api:v1.1 .

# 3. 停止旧容器
docker stop my-api

# 4. 运行新容器
docker run -d --name my-api -p 8000:8000 my-agent-api:v1.1
```

### Q5: 如何查看镜像构建历史？

```bash
# 查看镜像层
docker history my-agent-api:v1.0

# 输出
IMAGE          CREATED        SIZE      COMMENT
abc123         2 minutes ago  5MB       CMD ["uvicorn"...]
def456         2 minutes ago  10MB      COPY . .
ghi789         2 minutes ago  50MB      RUN pip install...
jkl012         2 minutes ago  1KB       COPY requirements.txt
mno345         2 minutes ago  0B        WORKDIR /app
pqr678         3 days ago     200MB     FROM python:3.13-slim
```

---

## 完整操作流程

```bash
# 1. 创建项目目录
mkdir my-agent-api && cd my-agent-api

# 2. 创建文件
mkdir app
touch app/__init__.py
cat > app/main.py << 'EOF'
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Docker!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF

cat > requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
EOF

cat > Dockerfile << 'EOF'
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > .dockerignore << 'EOF'
__pycache__/
*.pyc
.venv/
.git/
EOF

# 3. 构建镜像
docker build -t my-agent-api:v1.0 .

# 4. 运行容器
docker run -d --name my-api -p 8000:8000 my-agent-api:v1.0

# 5. 测试
curl http://localhost:8000
curl http://localhost:8000/health

# 6. 查看日志
docker logs my-api

# 7. 停止和删除
docker rm -f my-api
```

---

## 总结

**基础 Dockerfile 的核心要素：**
1. ✅ 选择合适的基础镜像（python:3.13-slim）
2. ✅ 设置工作目录（WORKDIR）
3. ✅ 先复制依赖文件，利用缓存
4. ✅ 安装依赖（--no-cache-dir）
5. ✅ 复制应用代码
6. ✅ 声明端口（EXPOSE）
7. ✅ 设置启动命令（CMD）

**下一步：**
- 学习多阶段构建优化镜像体积
- 添加健康检查
- 使用非 root 用户
- 配置环境变量

---

**版本：** v1.0
**最后更新：** 2026-02-12
**适用于：** Python 3.13+ / FastAPI / AI Agent 后端开发
