# 核心概念01：任务队列基础（Celery & ARQ）

> 深入理解任务队列的工作原理、Celery和ARQ的使用

---

## 什么是任务队列？

### 定义

**任务队列 = 持久化队列（存储任务）+ Worker进程（执行任务）+ 消息代理（分发任务）**

### 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                    任务队列系统                          │
│                                                          │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐     │
│  │  生产者   │ ───> │ 消息代理  │ ───> │  Worker  │     │
│  │(FastAPI) │      │ (Redis)  │      │  进程    │     │
│  └──────────┘      └──────────┘      └──────────┘     │
│       │                  │                  │           │
│       │ 1.提交任务       │ 2.存储任务       │ 3.执行任务│
│       ↓                  ↓                  ↓           │
│  返回task_id        持久化队列          更新状态       │
└─────────────────────────────────────────────────────────┘
```

**三个关键角色**：
1. **生产者（Producer）**：提交任务（FastAPI应用）
2. **消息代理（Broker）**：存储和分发任务（Redis/RabbitMQ）
3. **消费者（Consumer）**：执行任务（Worker进程）

---

## Celery：功能全面的分布式任务队列

### 1. Celery架构

```
┌─────────────────────────────────────────────────────────┐
│                    Celery 架构                           │
│                                                          │
│  ┌──────────┐                                           │
│  │ FastAPI  │                                           │
│  │  应用    │                                           │
│  └────┬─────┘                                           │
│       │ task.delay()                                    │
│       ↓                                                  │
│  ┌──────────────────────────────────────────┐          │
│  │         Broker (Redis/RabbitMQ)          │          │
│  │  ┌────────┐  ┌────────┐  ┌────────┐    │          │
│  │  │ Queue1 │  │ Queue2 │  │ Queue3 │    │          │
│  │  └────────┘  └────────┘  └────────┘    │          │
│  └──────────────────────────────────────────┘          │
│       │              │              │                   │
│       ↓              ↓              ↓                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│  │ Worker1 │   │ Worker2 │   │ Worker3 │             │
│  └─────────┘   └─────────┘   └─────────┘             │
│       │              │              │                   │
│       ↓              ↓              ↓                   │
│  ┌──────────────────────────────────────────┐          │
│  │         Backend (Redis/Database)         │          │
│  │         存储任务结果和状态                │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

**关键概念**：
- **Broker**：消息队列，存储待执行的任务
- **Worker**：独立进程，从队列中取任务并执行
- **Backend**：结果存储，保存任务执行结果和状态

---

### 2. Celery安装与配置

#### 安装依赖

```bash
# 安装Celery和Redis
uv add celery redis

# 启动Redis（作为Broker和Backend）
redis-server
```

#### 基础配置

```python
# app/celery_app.py
from celery import Celery

# 创建Celery应用
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',      # 消息队列
    backend='redis://localhost:6379/1'      # 结果存储
)

# 配置选项
app.conf.update(
    # 任务序列化格式
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # 时区设置
    timezone='Asia/Shanghai',
    enable_utc=True,

    # 任务结果过期时间（秒）
    result_expires=3600,

    # Worker并发数
    worker_concurrency=4,

    # 任务确认机制
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
```

**配置说明**：
- `broker`：消息队列地址（Redis/RabbitMQ）
- `backend`：结果存储地址（Redis/Database）
- `task_serializer`：任务序列化格式（json/pickle）
- `result_expires`：结果过期时间（秒）
- `worker_concurrency`：Worker并发数

---

### 3. 定义和调用任务

#### 定义任务

```python
# app/tasks.py
from app.celery_app import app
import time

# 简单任务
@app.task
def add(x: int, y: int) -> int:
    """简单的加法任务"""
    return x + y

# 耗时任务
@app.task
def process_document(file_path: str) -> dict:
    """处理文档（耗时操作）"""
    print(f"开始处理: {file_path}")

    # 模拟耗时操作
    time.sleep(60)

    print(f"处理完成: {file_path}")
    return {
        "file": file_path,
        "status": "completed",
        "word_count": 1000
    }

# 带参数的任务
@app.task
def send_email(to: str, subject: str, body: str) -> dict:
    """发送邮件"""
    print(f"发送邮件到: {to}")
    # 实际的邮件发送逻辑
    return {"status": "sent", "to": to}
```

#### 调用任务

```python
# 同步调用（阻塞，等待结果）
result = add(2, 3)
print(result)  # 5

# 异步调用（非阻塞，立即返回）
result = add.delay(2, 3)
print(f"任务ID: {result.id}")
print(f"任务状态: {result.status}")  # PENDING

# 等待结果
print(f"任务结果: {result.get()}")  # 5

# 带参数的异步调用
result = process_document.delay('/path/to/file.pdf')
print(f"任务ID: {result.id}")
```

**调用方式对比**：

| 调用方式 | 语法 | 阻塞 | 返回值 | 适用场景 |
|---------|------|------|--------|---------|
| **同步调用** | `add(2, 3)` | ✅ 阻塞 | 任务结果 | 测试、调试 |
| **异步调用** | `add.delay(2, 3)` | ❌ 非阻塞 | AsyncResult对象 | 生产环境 |
| **apply_async** | `add.apply_async((2, 3))` | ❌ 非阻塞 | AsyncResult对象 | 需要高级选项 |

---

### 4. 查询任务状态和结果

```python
from celery.result import AsyncResult

# 提交任务
result = process_document.delay('/path/to/file.pdf')
task_id = result.id

# 查询任务状态
print(f"任务状态: {result.status}")
# PENDING: 待处理
# STARTED: 执行中
# SUCCESS: 成功
# FAILURE: 失败
# RETRY: 重试中

# 检查任务是否完成
if result.ready():
    print("任务已完成")
else:
    print("任务执行中")

# 检查任务是否成功
if result.successful():
    print("任务成功")
    print(f"结果: {result.result}")
else:
    print("任务失败或未完成")

# 获取任务结果（阻塞等待）
try:
    result_data = result.get(timeout=10)  # 最多等待10秒
    print(f"任务结果: {result_data}")
except TimeoutError:
    print("任务超时")

# 通过task_id查询任务
result = AsyncResult(task_id, app=app)
print(f"任务状态: {result.status}")
```

---

### 5. 启动Celery Worker

```bash
# 基础启动
celery -A app.celery_app worker --loglevel=info

# 指定并发数
celery -A app.celery_app worker --concurrency=4

# 指定队列
celery -A app.celery_app worker -Q default,high_priority

# 后台运行
celery -A app.celery_app worker --detach

# 查看Worker状态
celery -A app.celery_app inspect active
celery -A app.celery_app inspect stats
```

**启动参数**：
- `-A`：Celery应用路径
- `--loglevel`：日志级别（debug/info/warning/error）
- `--concurrency`：并发数（默认为CPU核心数）
- `-Q`：监听的队列名称
- `--detach`：后台运行

---

### 6. 在FastAPI中集成Celery

```python
# app/main.py
from fastapi import FastAPI, UploadFile
from app.tasks import process_document
from app.models.task import Task
from app.core.database import SessionLocal

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile):
    """上传文件并异步处理"""
    # 1. 保存文件
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. 创建任务记录
    db = SessionLocal()
    db_task = Task(
        task_type="document_process",
        status="pending"
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # 3. 异步调用Celery任务
    celery_result = process_document.delay(file_path, db_task.id)

    # 4. 更新Celery任务ID
    db_task.task_id = celery_result.id
    db.commit()
    db.close()

    # 5. 返回任务ID
    return {
        "task_id": celery_result.id,
        "status": "processing",
        "message": "任务已提交，请稍后查询结果"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    from celery.result import AsyncResult

    # 查询Celery任务状态
    result = AsyncResult(task_id, app=process_document.app)

    # 查询数据库任务记录
    db = SessionLocal()
    db_task = db.query(Task).filter(Task.task_id == task_id).first()
    db.close()

    if not db_task:
        return {"error": "Task not found"}

    return {
        "task_id": task_id,
        "status": result.status,
        "progress": db_task.progress,
        "result": result.result if result.ready() else None
    }
```

---

## ARQ：轻量级异步任务队列

### 1. ARQ简介

**ARQ vs Celery**：

| 特性 | ARQ | Celery |
|------|-----|--------|
| **配置复杂度** | 低（50行代码） | 高（100+行代码） |
| **异步支持** | ✅ 原生async/await | ✅ 支持但不原生 |
| **定时任务** | ✅ | ✅ |
| **重试机制** | ✅ | ✅ |
| **监控工具** | ❌ | ✅ Flower |
| **消息代理** | 仅Redis | Redis/RabbitMQ/... |
| **学习曲线** | 平缓 | 陡峭 |
| **适用场景** | 中小型项目 | 大型项目 |

**ARQ的优势**：
- 原生支持async/await
- 配置简单
- 代码量少
- 适合FastAPI异步应用

---

### 2. ARQ安装与配置

```bash
# 安装ARQ
uv add arq redis
```

```python
# app/arq_worker.py
from arq import create_pool
from arq.connections import RedisSettings
import asyncio

# 定义异步任务
async def process_document(ctx, file_path: str) -> dict:
    """处理文档（异步）"""
    print(f"开始处理: {file_path}")

    # 模拟耗时操作
    await asyncio.sleep(60)

    print(f"处理完成: {file_path}")
    return {
        "file": file_path,
        "status": "completed"
    }

async def send_email(ctx, to: str, subject: str) -> dict:
    """发送邮件（异步）"""
    print(f"发送邮件到: {to}")
    await asyncio.sleep(5)
    return {"status": "sent", "to": to}

# Worker配置
class WorkerSettings:
    functions = [process_document, send_email]
    redis_settings = RedisSettings(host='localhost', port=6379)

    # 任务超时
    job_timeout = 300  # 5分钟

    # 重试次数
    max_tries = 3

    # Worker并发数
    max_jobs = 10
```

---

### 3. 在FastAPI中使用ARQ

```python
# app/main.py
from fastapi import FastAPI, UploadFile
from arq import create_pool
from arq.connections import RedisSettings

app = FastAPI()

# 创建Redis连接池
@app.on_event("startup")
async def startup():
    app.state.arq = await create_pool(RedisSettings())

@app.on_event("shutdown")
async def shutdown():
    await app.state.arq.close()

@app.post("/upload")
async def upload_file(file: UploadFile):
    """上传文件并异步处理"""
    # 保存文件
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 提交ARQ任务
    job = await app.state.arq.enqueue_job(
        'process_document',
        file_path
    )

    return {
        "job_id": job.job_id,
        "status": "processing"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """查询任务状态"""
    from arq.jobs import Job

    job = Job(job_id, app.state.arq)
    info = await job.info()

    return {
        "job_id": job_id,
        "status": info.status if info else "not_found",
        "result": info.result if info and info.success else None
    }
```

---

### 4. 启动ARQ Worker

```bash
# 启动Worker
arq app.arq_worker.WorkerSettings

# 指定并发数
arq app.arq_worker.WorkerSettings --max-jobs 10

# 查看Worker状态
arq app.arq_worker.WorkerSettings --check
```

---

## Celery vs ARQ：如何选择？

### 决策树

```
项目规模？
├─ 小型（<100任务/天）
│  └─ 用 Redis Queue（极简）
│
├─ 中型（100-10000任务/天）
│  └─ 需要异步支持？
│      ├─ 是 → 用 ARQ（原生async/await）
│      └─ 否 → 用 Celery（功能全面）
│
└─ 大型（>10000任务/天）
   └─ 用 Celery（可扩展、有监控）
```

### 实际案例对比

#### 场景1：AI Agent文档处理（1000个任务/天）

**推荐：ARQ**

```python
# ARQ实现（简单）
async def process_document(ctx, file_path: str):
    # 解析PDF
    content = await parse_pdf(file_path)

    # 生成Embedding
    embedding = await generate_embedding(content)

    # 保存到向量库
    await save_to_vectordb(embedding)

    return {"status": "done"}

# 调用
job = await redis.enqueue_job('process_document', '/path/to/file.pdf')
```

**优势**：
- 原生async/await，与FastAPI无缝集成
- 配置简单，50行代码
- 性能好（异步I/O）

---

#### 场景2：大规模批量处理（100000个任务/天）

**推荐：Celery**

```python
# Celery实现（功能全面）
@app.task(
    bind=True,
    max_retries=3,
    time_limit=300
)
def process_document(self, file_path: str):
    # 处理文档
    result = parse_and_embed(file_path)
    return result

# 调用
result = process_document.delay('/path/to/file.pdf')
```

**优势**：
- 功能全面（重试、定时、监控）
- 可扩展（多Worker、多队列）
- 有监控工具（Flower）

---

## 高级特性

### 1. 任务优先级

```python
# Celery：使用多个队列
@app.task(queue='high_priority')
def urgent_task():
    pass

@app.task(queue='low_priority')
def normal_task():
    pass

# 启动Worker监听不同队列
# celery -A app.celery_app worker -Q high_priority,low_priority
```

### 2. 定时任务

```python
# Celery Beat：定时任务调度器
from celery.schedules import crontab

app.conf.beat_schedule = {
    # 每天凌晨1点执行
    'update-knowledge-base': {
        'task': 'app.tasks.update_knowledge_base',
        'schedule': crontab(hour=1, minute=0),
    },
    # 每5分钟执行
    'cleanup-temp-files': {
        'task': 'app.tasks.cleanup_temp_files',
        'schedule': 300.0,  # 秒
    },
}

# 启动Beat调度器
# celery -A app.celery_app beat
```

### 3. 任务链（Chain）

```python
# 任务链：按顺序执行多个任务
from celery import chain

# 定义任务
@app.task
def parse_pdf(file_path: str) -> str:
    return "parsed_content"

@app.task
def generate_embedding(content: str) -> list:
    return [0.1, 0.2, 0.3]

@app.task
def save_to_db(embedding: list) -> dict:
    return {"status": "saved"}

# 创建任务链
workflow = chain(
    parse_pdf.s('/path/to/file.pdf'),
    generate_embedding.s(),
    save_to_db.s()
)

# 执行任务链
result = workflow.apply_async()
```

### 4. 任务组（Group）

```python
# 任务组：并行执行多个任务
from celery import group

# 并行处理多个文件
job = group(
    process_document.s('/path/to/file1.pdf'),
    process_document.s('/path/to/file2.pdf'),
    process_document.s('/path/to/file3.pdf'),
)

result = job.apply_async()

# 等待所有任务完成
results = result.get()
```

---

## 监控与调试

### 1. Flower：Celery监控工具

```bash
# 安装Flower
uv add flower

# 启动Flower
celery -A app.celery_app flower

# 访问Web界面
# http://localhost:5555
```

**Flower功能**：
- 实时监控Worker状态
- 查看任务执行历史
- 查看任务队列长度
- 手动重试失败任务
- 查看任务执行时间统计

### 2. 日志配置

```python
# Celery日志配置
app.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s',
)
```

---

## 最佳实践

### 1. 任务幂等性

```python
# ✅ 幂等任务：多次执行结果相同
@app.task
def update_user_status(user_id: int, status: str):
    user = get_user(user_id)
    user.status = status  # 多次执行结果相同
    save_user(user)

# ❌ 非幂等任务：多次执行结果不同
@app.task
def increment_counter(user_id: int):
    user = get_user(user_id)
    user.counter += 1  # 多次执行结果不同
    save_user(user)
```

### 2. 避免大对象传递

```python
# ❌ 错误：传递大对象
@app.task
def process_data(large_data: dict):  # large_data可能有10MB
    # 处理数据
    pass

# ✅ 正确：传递ID，在任务中查询
@app.task
def process_data(data_id: int):
    large_data = get_data_from_db(data_id)
    # 处理数据
    pass
```

### 3. 合理设置超时

```python
@app.task(
    time_limit=300,      # 硬超时：5分钟
    soft_time_limit=270  # 软超时：4.5分钟
)
def process_document(file_path: str):
    # 任务逻辑
    pass
```

---

## 总结

### Celery适用场景
- ✅ 大型项目（>10000任务/天）
- ✅ 需要定时任务
- ✅ 需要监控工具
- ✅ 需要多种消息代理

### ARQ适用场景
- ✅ 中小型项目（100-10000任务/天）
- ✅ FastAPI异步应用
- ✅ 需要简单配置
- ✅ 原生async/await

### 选择建议
- 小项目：Redis Queue
- 中项目：ARQ
- 大项目：Celery

---

**记住**：任务队列的核心是**解耦**，让HTTP请求立即返回，任务在后台异步执行。
