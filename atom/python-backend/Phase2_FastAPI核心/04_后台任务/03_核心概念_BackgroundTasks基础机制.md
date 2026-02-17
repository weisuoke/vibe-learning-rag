# 核心概念1：BackgroundTasks 基础机制

> 深入理解 BackgroundTasks 的工作原理、API 使用和生命周期

---

## 1. BackgroundTasks 是什么？

**定义：** BackgroundTasks 是 FastAPI 内置的后台任务管理器，用于在 HTTP 响应返回后执行函数。

```python
from fastapi import BackgroundTasks

# BackgroundTasks 的核心接口
class BackgroundTasks:
    def add_task(
        self,
        func: Callable[..., Any],  # 任务函数
        *args: Any,                # 位置参数
        **kwargs: Any              # 关键字参数
    ) -> None:
        """添加任务到队列"""
```

**关键特性：**
- 进程内执行（与 FastAPI 应用在同一进程）
- 顺序执行（按添加顺序）
- 与请求生命周期绑定
- 支持同步和异步函数

---

## 2. 工作原理

### 2.1 执行流程

```
┌─────────────────────────────────────────────────────────┐
│ 1. 请求到达                                              │
│    ↓                                                     │
│ 2. 路由函数执行                                          │
│    - 注入 BackgroundTasks 实例                           │
│    - 执行业务逻辑                                        │
│    ↓                                                     │
│ 3. 添加后台任务                                          │
│    - background_tasks.add_task(func, args, kwargs)       │
│    - 任务添加到内部队列                                  │
│    ↓                                                     │
│ 4. 返回响应                                              │
│    - 构建 HTTP 响应                                      │
│    - 发送给客户端                                        │
│    ↓                                                     │
│ 5. 执行后台任务（响应已返回）                            │
│    - 从队列中取出任务                                    │
│    - 按顺序执行                                          │
│    ↓                                                     │
│ 6. 关闭连接                                              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 代码示例

```python
from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def task1():
    print("任务1开始")
    time.sleep(2)
    print("任务1完成")

def task2():
    print("任务2开始")
    time.sleep(1)
    print("任务2完成")

@app.post("/demo")
async def demo(background_tasks: BackgroundTasks):
    print("路由函数开始")

    # 添加任务
    background_tasks.add_task(task1)
    background_tasks.add_task(task2)

    print("返回响应")
    return {"status": "ok"}

# 执行顺序：
# 1. "路由函数开始"
# 2. "返回响应"
# 3. 响应发送给客户端
# 4. "任务1开始"
# 5. 等待2秒
# 6. "任务1完成"
# 7. "任务2开始"
# 8. 等待1秒
# 9. "任务2完成"
```

---

## 3. 依赖注入机制

### 3.1 如何获取 BackgroundTasks

**方式1：路由函数参数注入**

```python
@app.post("/action")
async def action(background_tasks: BackgroundTasks):
    background_tasks.add_task(some_task)
    return {"status": "ok"}
```

**方式2：依赖函数中注入**

```python
from fastapi import Depends

def get_background_tasks(background_tasks: BackgroundTasks):
    """依赖函数"""
    return background_tasks

@app.post("/action")
async def action(bg: BackgroundTasks = Depends(get_background_tasks)):
    bg.add_task(some_task)
    return {"status": "ok"}
```

**方式3：在依赖中直接使用**

```python
def process_with_background(
    data: dict,
    background_tasks: BackgroundTasks
):
    """依赖函数中使用后台任务"""
    background_tasks.add_task(log_data, data)
    return data

@app.post("/process")
async def process(
    result: dict = Depends(process_with_background)
):
    return result
```

### 3.2 BackgroundTasks 的作用域

**重要：** 每个请求都有独立的 BackgroundTasks 实例

```python
# 错误理解：以为 BackgroundTasks 是全局单例
# 正确理解：每个请求有自己的 BackgroundTasks

@app.post("/action1")
async def action1(bg: BackgroundTasks):
    bg.add_task(task1)  # 只影响这个请求
    return {"status": "ok"}

@app.post("/action2")
async def action2(bg: BackgroundTasks):
    bg.add_task(task2)  # 不同的 BackgroundTasks 实例
    return {"status": "ok"}
```

---

## 4. add_task() 方法详解

### 4.1 基础用法

```python
# 无参数任务
background_tasks.add_task(simple_task)

# 位置参数
background_tasks.add_task(task_with_args, arg1, arg2)

# 关键字参数
background_tasks.add_task(task_with_kwargs, key1=value1, key2=value2)

# 混合参数
background_tasks.add_task(
    complex_task,
    arg1,              # 位置参数
    arg2,
    key1=value1,       # 关键字参数
    key2=value2
)
```

### 4.2 参数传递时机

**重要：** 参数在任务执行时传递，不是添加时

```python
def task(value: int):
    print(f"任务执行，值为: {value}")

@app.post("/demo")
async def demo(background_tasks: BackgroundTasks):
    value = 10
    background_tasks.add_task(task, value)  # 传递 10

    value = 20  # 修改变量

    return {"status": "ok"}
    # 任务执行时，value 仍然是 10（不是 20）
```

**原因：** `add_task()` 会立即捕获参数的值（对于不可变类型）

### 4.3 传递可变对象

```python
def task(data: list):
    print(f"任务执行，数据: {data}")

@app.post("/demo")
async def demo(background_tasks: BackgroundTasks):
    data = [1, 2, 3]
    background_tasks.add_task(task, data)

    data.append(4)  # 修改列表

    return {"status": "ok"}
    # 任务执行时，data 是 [1, 2, 3, 4]（包含修改）
```

**注意：** 可变对象（list、dict）的修改会影响后台任务

**最佳实践：** 传递不可变数据或复制数据

```python
# ✅ 推荐：传递不可变数据
background_tasks.add_task(task, user_id)  # int 是不可变的

# ✅ 推荐：复制可变数据
background_tasks.add_task(task, data.copy())

# ❌ 避免：传递可变对象的引用
background_tasks.add_task(task, data)  # 可能被修改
```

---

## 5. 同步 vs 异步任务

### 5.1 同步任务

```python
import time

def sync_task(name: str):
    """同步任务：阻塞执行"""
    print(f"同步任务 {name} 开始")
    time.sleep(2)  # 阻塞2秒
    print(f"同步任务 {name} 完成")

@app.post("/sync")
async def sync_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(sync_task, "A")
    background_tasks.add_task(sync_task, "B")
    return {"status": "ok"}

# 执行：
# 1. 响应返回
# 2. "同步任务 A 开始"
# 3. 阻塞2秒
# 4. "同步任务 A 完成"
# 5. "同步任务 B 开始"
# 6. 阻塞2秒
# 7. "同步任务 B 完成"
# 总时间：4秒
```

### 5.2 异步任务

```python
import asyncio

async def async_task(name: str):
    """异步任务：不阻塞"""
    print(f"异步任务 {name} 开始")
    await asyncio.sleep(2)  # 不阻塞
    print(f"异步任务 {name} 完成")

@app.post("/async")
async def async_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(async_task, "A")
    background_tasks.add_task(async_task, "B")
    return {"status": "ok"}

# 执行：
# 1. 响应返回
# 2. "异步任务 A 开始"
# 3. await asyncio.sleep(2)
# 4. "异步任务 A 完成"
# 5. "异步任务 B 开始"
# 6. await asyncio.sleep(2)
# 7. "异步任务 B 完成"
# 总时间：4秒（仍然是顺序执行）
```

**注意：** 即使是异步任务，BackgroundTasks 也是顺序执行的！

### 5.3 并发执行异步任务

如果需要并发执行，在任务内部使用 `asyncio.gather()`：

```python
async def concurrent_tasks():
    """在任务内部并发执行"""
    await asyncio.gather(
        async_task("A"),
        async_task("B")
    )

@app.post("/concurrent")
async def concurrent_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(concurrent_tasks)
    return {"status": "ok"}

# 执行：
# 1. 响应返回
# 2. "异步任务 A 开始" 和 "异步任务 B 开始"（同时）
# 3. 等待2秒
# 4. "异步任务 A 完成" 和 "异步任务 B 完成"（同时）
# 总时间：2秒
```

---

## 6. 任务执行顺序

### 6.1 顺序执行

```python
def task1():
    print("任务1")
    time.sleep(1)

def task2():
    print("任务2")
    time.sleep(1)

def task3():
    print("任务3")
    time.sleep(1)

@app.post("/order")
async def order_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(task1)
    background_tasks.add_task(task2)
    background_tasks.add_task(task3)
    return {"status": "ok"}

# 执行顺序：task1 → task2 → task3
# 总时间：3秒
```

### 6.2 任务之间的依赖

```python
# 场景：任务2依赖任务1的结果

# ❌ 错误：无法传递任务1的结果给任务2
def task1():
    return "result1"

def task2(result):
    print(f"使用结果: {result}")

@app.post("/wrong")
async def wrong_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(task1)
    background_tasks.add_task(task2, ???)  # 无法获取 task1 的结果
    return {"status": "ok"}

# ✅ 正确：在一个任务中完成所有依赖操作
def combined_task():
    result1 = task1()
    task2(result1)

@app.post("/correct")
async def correct_demo(background_tasks: BackgroundTasks):
    background_tasks.add_task(combined_task)
    return {"status": "ok"}
```

---

## 7. 生命周期和资源管理

### 7.1 任务与请求的关系

```python
@app.post("/demo")
async def demo(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)  # 请求作用域
):
    user = User(email="test@example.com")
    db.add(user)
    db.commit()

    # ❌ 错误：db 在任务执行时可能已关闭
    background_tasks.add_task(send_email, user, db)

    return {"user_id": user.id}
```

**问题：** `db` 是请求作用域的资源，响应返回后会被清理

**解决方案：** 任务自己管理资源

```python
def send_email(user_id: int):
    # 在任务内部创建新的数据库连接
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        # 发送邮件
    finally:
        db.close()

@app.post("/demo")
async def demo(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    user = User(email="test@example.com")
    db.add(user)
    db.commit()

    # ✅ 正确：只传递 user_id
    background_tasks.add_task(send_email, user.id)

    return {"user_id": user.id}
```

### 7.2 任务的超时和取消

**重要：** BackgroundTasks 没有内置的超时和取消机制

```python
# 如果需要超时控制，需要在任务内部实现
import asyncio

async def task_with_timeout():
    try:
        await asyncio.wait_for(
            long_running_operation(),
            timeout=30.0  # 30秒超时
        )
    except asyncio.TimeoutError:
        print("任务超时")
```

---

## 8. 在 AI Agent 中的应用

### 8.1 文档处理

```python
from fastapi import UploadFile

async def process_document(file_path: str, user_id: int):
    """后台处理文档"""
    # 1. 解析文档
    text = extract_text(file_path)

    # 2. 生成 Embedding
    embedding = await generate_embedding(text)

    # 3. 存储到向量数据库
    await store_to_vectordb(user_id, embedding)

    # 4. 清理临时文件
    os.remove(file_path)

@app.post("/upload")
async def upload_document(
    file: UploadFile,
    user_id: int,
    background_tasks: BackgroundTasks
):
    # 保存文件
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 后台处理
    background_tasks.add_task(process_document, file_path, user_id)

    return {"message": "文档上传成功，正在处理"}
```

### 8.2 对话日志记录

```python
async def log_conversation(
    user_id: int,
    message: str,
    response: str
):
    """记录对话到数据库"""
    db = SessionLocal()
    try:
        conversation = Conversation(
            user_id=user_id,
            message=message,
            response=response,
            timestamp=datetime.now()
        )
        db.add(conversation)
        db.commit()
    finally:
        db.close()

@app.post("/chat")
async def chat(
    user_id: int,
    message: str,
    background_tasks: BackgroundTasks
):
    # 生成回复
    response = await generate_response(message)

    # 后台记录日志
    background_tasks.add_task(
        log_conversation,
        user_id,
        message,
        response
    )

    return {"response": response}
```

---

## 9. 最佳实践

### 9.1 任务设计原则

1. **自包含**：任务不依赖请求上下文
2. **幂等性**：任务可以安全地重复执行
3. **快速失败**：尽早检测错误
4. **资源清理**：确保资源正确释放

### 9.2 代码示例

```python
async def well_designed_task(user_id: int, action: str):
    """设计良好的后台任务"""
    # 1. 参数验证（快速失败）
    if not user_id or not action:
        logger.error("无效参数")
        return

    # 2. 资源管理（自包含）
    db = SessionLocal()
    try:
        # 3. 幂等性检查
        existing = db.query(Action).filter(
            Action.user_id == user_id,
            Action.action == action
        ).first()

        if existing:
            logger.info("操作已存在，跳过")
            return

        # 4. 执行操作
        new_action = Action(user_id=user_id, action=action)
        db.add(new_action)
        db.commit()

    except Exception as e:
        logger.error(f"任务失败: {e}")
        db.rollback()
    finally:
        # 5. 资源清理
        db.close()
```

---

## 10. 总结

**BackgroundTasks 的核心特性：**
- 进程内执行，无需额外服务
- 顺序执行，按添加顺序
- 与请求生命周期绑定
- 支持同步和异步函数

**适用场景：**
- 短时间任务（< 30秒）
- 失败可接受的任务
- 简单的异步操作

**关键原则：**
- 任务应该自包含
- 传递数据，不传递资源
- 添加错误处理和日志
- 考虑任务的幂等性

---

**版本：** v1.0
**最后更新：** 2026-02-11
