# 实战代码 - 场景1：Celery基础任务队列

> 完整可运行的Celery任务队列示例，演示文档处理任务

---

## 场景描述

**需求**：用户上传PDF文档，后台异步处理（解析、生成Embedding、保存到向量库）

**技术栈**：
- FastAPI：Web框架
- Celery：任务队列
- Redis：消息代理
- PostgreSQL：任务状态存储

---

## 项目结构

```
celery-basic-demo/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI应用
│   ├── celery_app.py        # Celery配置
│   ├── tasks.py             # Celery任务
│   ├── models/
│   │   ├── __init__.py
│   │   └── task.py          # 任务模型
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # 配置
│   │   └── database.py      # 数据库连接
│   └── services/
│       ├── __init__.py
│       └── document.py      # 文档处理服务
├── .env                     # 环境变量
├── pyproject.toml           # 依赖配置
└── README.md
```

---

## 完整代码实现

### 1. 环境配置

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost:5432/celery_demo
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=your_openai_key_here
```

```toml
# pyproject.toml
[project]
name = "celery-basic-demo"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
]
```

---

### 2. 配置文件

```python
# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 数据库配置
    DATABASE_URL: str

    # Redis配置
    REDIS_URL: str

    # OpenAI配置
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
```

---

### 3. 数据库配置

```python
# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# 创建数据库引擎
engine = create_engine(settings.DATABASE_URL)

# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建Base类
Base = declarative_base()

def get_db():
    """获取数据库Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 4. 任务模型

```python
# app/models/task.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum
from sqlalchemy.sql import func
from app.core.database import Base
import enum

class TaskStatus(str, enum.Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(Base):
    """任务模型"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True, nullable=False)
    task_type = Column(String, nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

---

### 5. Celery配置

```python
# app/celery_app.py
from celery import Celery
from app.core.config import settings

# 创建Celery应用
app = Celery(
    'tasks',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# 配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    result_expires=3600,
)
```

---

### 6. 文档处理服务

```python
# app/services/document.py
import time
from typing import List
from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def parse_pdf(file_path: str) -> str:
    """解析PDF文档"""
    print(f"解析PDF: {file_path}")
    time.sleep(2)  # 模拟耗时操作
    return f"Content of {file_path}"

def generate_embedding(text: str) -> List[float]:
    """生成Embedding"""
    print(f"生成Embedding: {text[:50]}...")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

def save_to_vectordb(file_path: str, embedding: List[float]):
    """保存到向量库"""
    print(f"保存到向量库: {file_path}")
    time.sleep(1)  # 模拟耗时操作
```

---

### 7. Celery任务

```python
# app/tasks.py
from app.celery_app import app
from app.models.task import Task, TaskStatus
from app.core.database import SessionLocal
from app.services.document import parse_pdf, generate_embedding, save_to_vectordb
import json

@app.task(bind=True, max_retries=3)
def process_document(self, file_path: str, db_task_id: int):
    """
    处理文档任务

    Args:
        file_path: 文档路径
        db_task_id: 数据库任务ID
    """
    db = SessionLocal()

    try:
        # 获取任务记录
        task = db.query(Task).filter(Task.id == db_task_id).first()

        # 更新状态为运行中
        task.status = TaskStatus.RUNNING
        task.progress = 0.0
        db.commit()

        # 步骤1：解析PDF（33%）
        print(f"[Task {self.request.id}] 步骤1：解析PDF")
        content = parse_pdf(file_path)
        task.progress = 33.0
        db.commit()

        # 步骤2：生成Embedding（66%）
        print(f"[Task {self.request.id}] 步骤2：生成Embedding")
        embedding = generate_embedding(content)
        task.progress = 66.0
        db.commit()

        # 步骤3：保存到向量库（100%）
        print(f"[Task {self.request.id}] 步骤3：保存到向量库")
        save_to_vectordb(file_path, embedding)
        task.progress = 100.0
        task.status = TaskStatus.COMPLETED
        task.result = json.dumps({
            "file": file_path,
            "embedding_dim": len(embedding),
            "content_length": len(content)
        })
        db.commit()

        print(f"[Task {self.request.id}] 任务完成")
        return {"status": "success"}

    except Exception as e:
        # 更新状态为失败
        task.status = TaskStatus.FAILED
        task.error = str(e)
        db.commit()

        print(f"[Task {self.request.id}] 任务失败: {e}")

        # 重试
        raise self.retry(exc=e, countdown=60)

    finally:
        db.close()
```

---

### 8. FastAPI应用

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db, engine, Base
from app.models.task import Task, TaskStatus
from app.tasks import process_document
from celery.result import AsyncResult
import os

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Celery基础任务队列示例")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    上传文件并异步处理

    Returns:
        task_id: 任务ID
        status: 任务状态
    """
    # 1. 保存文件
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. 创建任务记录
    db_task = Task(
        task_id="",  # 稍后更新
        task_type="document_process",
        status=TaskStatus.PENDING
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # 3. 异步调用Celery任务
    celery_result = process_document.delay(file_path, db_task.id)

    # 4. 更新Celery任务ID
    db_task.task_id = celery_result.id
    db.commit()

    return {
        "task_id": celery_result.id,
        "status": "processing",
        "message": "文件已上传，正在后台处理"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """
    查询任务状态

    Args:
        task_id: Celery任务ID

    Returns:
        任务状态信息
    """
    # 1. 从数据库查询任务
    task = db.query(Task).filter(Task.task_id == task_id).first()

    if not task:
        return {"error": "Task not found"}

    # 2. 从Celery查询任务状态
    celery_result = AsyncResult(task_id, app=process_document.app)

    return {
        "task_id": task_id,
        "status": task.status.value,
        "progress": task.progress,
        "result": task.result,
        "error": task.error,
        "celery_status": celery_result.status,
        "created_at": task.created_at.isoformat() if task.created_at else None
    }

@app.get("/tasks")
async def list_tasks(
    status: str = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    查询任务列表

    Args:
        status: 任务状态过滤
        limit: 返回数量限制

    Returns:
        任务列表
    """
    query = db.query(Task)

    if status:
        query = query.filter(Task.status == status)

    tasks = query.order_by(Task.created_at.desc()).limit(limit).all()

    return {
        "total": len(tasks),
        "tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat() if task.created_at else None
            }
            for task in tasks
        ]
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}
```

---

## 运行步骤

### 1. 安装依赖

```bash
# 安装Python依赖
uv sync

# 启动Redis
redis-server

# 启动PostgreSQL
# 确保PostgreSQL正在运行
```

---

### 2. 初始化数据库

```bash
# 创建数据库
createdb celery_demo

# 数据库表会在FastAPI启动时自动创建
```

---

### 3. 启动服务

```bash
# Terminal 1: 启动Celery Worker
celery -A app.celery_app worker --loglevel=info

# Terminal 2: 启动FastAPI
uvicorn app.main:app --reload --port 8000
```

---

### 4. 测试API

```bash
# 1. 上传文件
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test.pdf"

# 响应示例：
# {
#   "task_id": "abc123...",
#   "status": "processing",
#   "message": "文件已上传，正在后台处理"
# }

# 2. 查询任务状态
curl "http://localhost:8000/tasks/abc123..."

# 响应示例：
# {
#   "task_id": "abc123...",
#   "status": "running",
#   "progress": 66.0,
#   "result": null,
#   "error": null,
#   "celery_status": "STARTED"
# }

# 3. 查询任务列表
curl "http://localhost:8000/tasks?status=completed&limit=10"

# 响应示例：
# {
#   "total": 5,
#   "tasks": [...]
# }
```

---

## 预期输出

### Celery Worker日志

```
[2026-02-12 10:00:00,000: INFO/MainProcess] Connected to redis://localhost:6379/0
[2026-02-12 10:00:00,100: INFO/MainProcess] celery@hostname ready.

[2026-02-12 10:01:00,000: INFO/MainProcess] Task app.tasks.process_document[abc123...] received
[2026-02-12 10:01:00,100: INFO/ForkPoolWorker-1] [Task abc123...] 步骤1：解析PDF
[2026-02-12 10:01:02,200: INFO/ForkPoolWorker-1] 解析PDF: /tmp/test.pdf
[2026-02-12 10:01:02,300: INFO/ForkPoolWorker-1] [Task abc123...] 步骤2：生成Embedding
[2026-02-12 10:01:03,400: INFO/ForkPoolWorker-1] 生成Embedding: Content of /tmp/test.pdf...
[2026-02-12 10:01:04,500: INFO/ForkPoolWorker-1] [Task abc123...] 步骤3：保存到向量库
[2026-02-12 10:01:05,600: INFO/ForkPoolWorker-1] 保存到向量库: /tmp/test.pdf
[2026-02-12 10:01:06,700: INFO/ForkPoolWorker-1] [Task abc123...] 任务完成
[2026-02-12 10:01:06,800: INFO/ForkPoolWorker-1] Task app.tasks.process_document[abc123...] succeeded in 6.8s
```

---

## 关键知识点

### 1. Celery任务定义

```python
@app.task(bind=True, max_retries=3)
def process_document(self, file_path: str, db_task_id: int):
    # bind=True: 可以访问self（任务实例）
    # max_retries=3: 最多重试3次
    pass
```

### 2. 任务调用

```python
# 异步调用（推荐）
result = process_document.delay(file_path, db_task_id)
print(result.id)  # 任务ID

# 同步调用（阻塞，仅用于测试）
result = process_document(file_path, db_task_id)
```

### 3. 任务状态查询

```python
from celery.result import AsyncResult

result = AsyncResult(task_id, app=process_document.app)
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE
print(result.result)  # 任务结果
```

### 4. 进度更新

```python
# 在任务中更新数据库进度
task.progress = 50.0
db.commit()

# 客户端轮询查询进度
GET /tasks/{task_id}
```

---

## 常见问题

### Q1: Celery Worker无法连接Redis

**问题**：
```
[ERROR/MainProcess] consumer: Cannot connect to redis://localhost:6379/0
```

**解决**：
```bash
# 检查Redis是否运行
redis-cli ping

# 如果未运行，启动Redis
redis-server
```

---

### Q2: 任务一直处于PENDING状态

**问题**：任务提交后一直是PENDING，不执行

**原因**：Celery Worker未启动或未监听正确的队列

**解决**：
```bash
# 启动Worker
celery -A app.celery_app worker --loglevel=info

# 检查Worker是否运行
celery -A app.celery_app inspect active
```

---

### Q3: 数据库连接错误

**问题**：
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**解决**：
```bash
# 检查PostgreSQL是否运行
pg_isready

# 检查.env中的DATABASE_URL是否正确
cat .env | grep DATABASE_URL
```

---

## 扩展练习

### 练习1：添加任务取消功能

```python
@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """取消任务"""
    from celery.result import AsyncResult

    # 撤销Celery任务
    result = AsyncResult(task_id, app=process_document.app)
    result.revoke(terminate=True)

    # 更新数据库状态
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if task:
        task.status = TaskStatus.FAILED
        task.error = "Task canceled by user"
        db.commit()

    return {"status": "canceled"}
```

### 练习2：添加批量处理

```python
@app.task(bind=True)
def process_documents_batch(self, file_paths: List[str], db_task_id: int):
    """批量处理文档"""
    total = len(file_paths)

    for i, file_path in enumerate(file_paths):
        # 处理单个文件
        process_file(file_path)

        # 更新进度
        progress = (i + 1) / total * 100
        update_task_progress(db_task_id, progress)

    return {"files_processed": total}
```

### 练习3：添加定时任务

```python
# app/celery_app.py
from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-old-tasks': {
        'task': 'app.tasks.cleanup_old_tasks',
        'schedule': crontab(hour=2, minute=0),  # 每天凌晨2点
    },
}

# app/tasks.py
@app.task
def cleanup_old_tasks():
    """清理30天前的已完成任务"""
    from datetime import datetime, timedelta

    db = SessionLocal()
    cutoff_date = datetime.utcnow() - timedelta(days=30)

    db.query(Task).filter(
        Task.status == TaskStatus.COMPLETED,
        Task.created_at < cutoff_date
    ).delete()

    db.commit()
    db.close()
```

---

## 总结

本示例演示了：
- ✅ Celery基础配置
- ✅ 任务定义和调用
- ✅ 任务状态管理（数据库）
- ✅ 进度追踪
- ✅ 错误处理和重试
- ✅ FastAPI集成

**下一步**：学习WebSocket实时进度推送（场景2）
