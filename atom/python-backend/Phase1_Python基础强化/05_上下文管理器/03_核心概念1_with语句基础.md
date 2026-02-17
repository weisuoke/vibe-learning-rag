# 核心概念1：with 语句基础

## 概述

**`with` 语句是 Python 的上下文管理协议的语法糖，用于自动管理资源的获取和释放。**

---

## 1. with 语句的基本语法

### 1.1 最简单的形式

```python
with expression as variable:
    # 使用 variable
    pass
# variable 自动清理
```

**执行流程：**
1. 计算 `expression`，得到一个上下文管理器对象
2. 调用对象的 `__enter__()` 方法，返回值赋给 `variable`
3. 执行 `with` 块内的代码
4. 调用对象的 `__exit__()` 方法，无论是否发生异常

### 1.2 不使用 as 子句

```python
# 如果不需要使用返回值，可以省略 as 子句
with expression:
    # 执行代码
    pass
```

**示例：**
```python
# 计时器不需要返回值
@contextmanager
def timer():
    start = time.time()
    yield
    print(f"Time: {time.time() - start:.2f}s")

with timer():
    # 执行耗时操作
    expensive_computation()
```

---

## 2. with 语句的等价形式

### 2.1 与 try-finally 的对比

**with 语句：**
```python
with open("file.txt") as f:
    data = f.read()
    process(data)
```

**等价的 try-finally：**
```python
f = open("file.txt")
try:
    data = f.read()
    process(data)
finally:
    f.close()
```

### 2.2 完整的等价转换

**with 语句：**
```python
with expression as variable:
    suite
```

**等价代码：**
```python
manager = expression
variable = manager.__enter__()
try:
    suite
except:
    # 如果发生异常
    if not manager.__exit__(*sys.exc_info()):
        raise  # __exit__ 返回 False，重新抛出异常
else:
    # 如果没有异常
    manager.__exit__(None, None, None)
```

**关键点：**
- `__enter__()` 在进入 `with` 块前调用
- `__exit__()` 在退出 `with` 块时调用，无论是否异常
- `__exit__()` 接收异常信息（如果有）
- `__exit__()` 返回 `True` 可以抑制异常

---

## 3. 多个上下文管理器

### 3.1 嵌套写法

```python
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        for line in infile:
            outfile.write(line.upper())
```

**执行顺序：**
1. 打开 `input.txt`
2. 打开 `output.txt`
3. 处理数据
4. 关闭 `output.txt`
5. 关闭 `input.txt`

### 3.2 一行写法（推荐）

```python
# Python 3.1+ 支持
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    for line in infile:
        outfile.write(line.upper())
```

**优点：**
- 代码更简洁
- 减少缩进层级
- 语义更清晰

### 3.3 多个不同类型的上下文管理器

```python
# 同时管理数据库和缓存
with get_db() as db, get_redis() as cache:
    # 先查缓存
    cached = cache.get(key)
    if cached:
        return cached

    # 缓存未命中，查数据库
    data = db.query(Model).first()
    cache.set(key, data)
    return data
```

---

## 4. with 语句的执行时机

### 4.1 __enter__ 的执行时机

```python
class Demo:
    def __enter__(self):
        print("__enter__ called")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__ called")
        return False

print("Before with")
with Demo() as d:
    print("Inside with")
print("After with")

# 输出：
# Before with
# __enter__ called
# Inside with
# __exit__ called
# After with
```

**关键点：**
- `__enter__` 在进入 `with` 块之前立即执行
- `__enter__` 的返回值赋给 `as` 后的变量
- `with` 块内的代码在 `__enter__` 之后执行

### 4.2 __exit__ 的执行时机

```python
class Demo:
    def __enter__(self):
        print("Enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exit: exc_type={exc_type}")
        return False

# 正常退出
with Demo():
    print("Normal execution")
# 输出：
# Enter
# Normal execution
# Exit: exc_type=None

# 异常退出
try:
    with Demo():
        print("Before exception")
        raise ValueError("Error!")
except ValueError:
    print("Exception caught")
# 输出：
# Enter
# Before exception
# Exit: exc_type=<class 'ValueError'>
# Exception caught
```

**关键点：**
- `__exit__` 总是会执行，即使发生异常
- 如果没有异常，`exc_type` 为 `None`
- 如果有异常，`exc_type` 是异常类型

---

## 5. with 语句的异常处理

### 5.1 异常传播机制

```python
class ExceptionDemo:
    def __enter__(self):
        print("Enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exit: {exc_type.__name__ if exc_type else 'No exception'}")
        return False  # 不抑制异常

# 测试异常传播
try:
    with ExceptionDemo():
        print("Raising exception")
        raise ValueError("Test error")
        print("This won't execute")
except ValueError as e:
    print(f"Caught: {e}")

# 输出：
# Enter
# Raising exception
# Exit: ValueError
# Caught: Test error
```

### 5.2 抑制异常

```python
class SuppressException:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ValueError:
            print(f"Suppressed: {exc_val}")
            return True  # 抑制 ValueError
        return False  # 其他异常继续传播

# ValueError 被抑制
with SuppressException():
    raise ValueError("This will be suppressed")
print("Program continues")

# TypeError 不被抑制
try:
    with SuppressException():
        raise TypeError("This will propagate")
except TypeError:
    print("TypeError caught")
```

### 5.3 __enter__ 中的异常

```python
class EnterException:
    def __enter__(self):
        print("Enter: raising exception")
        raise RuntimeError("Error in __enter__")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exit called")  # 不会执行
        return False

try:
    with EnterException():
        print("This won't execute")
except RuntimeError:
    print("Exception from __enter__")

# 输出：
# Enter: raising exception
# Exception from __enter__
```

**关键点：**
- 如果 `__enter__` 抛出异常，`__exit__` 不会被调用
- `with` 块内的代码不会执行
- 异常直接传播到外层

---

## 6. with 语句的实际应用

### 6.1 文件操作

```python
# 读取文件
with open("data.txt", "r") as f:
    content = f.read()
    print(content)
# 文件自动关闭

# 写入文件
with open("output.txt", "w") as f:
    f.write("Hello, World!\n")
    f.write("Second line\n")
# 文件自动关闭并刷新缓冲区

# 追加文件
with open("log.txt", "a") as f:
    f.write(f"[{datetime.now()}] Log entry\n")
```

### 6.2 数据库连接

```python
# SQLite 示例
import sqlite3

with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
# 连接自动关闭

# PostgreSQL 示例（使用 psycopg2）
import psycopg2

with psycopg2.connect("dbname=mydb user=postgres") as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
# cursor 和 conn 都自动关闭
```

### 6.3 锁管理

```python
import threading

lock = threading.Lock()

# 使用 with 语句自动获取和释放锁
with lock:
    # 临界区代码
    shared_resource += 1
# 锁自动释放

# 等价于：
lock.acquire()
try:
    shared_resource += 1
finally:
    lock.release()
```

### 6.4 临时目录

```python
import tempfile
import os

# 创建临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    # 在临时目录中工作
    filepath = os.path.join(tmpdir, "temp.txt")
    with open(filepath, "w") as f:
        f.write("Temporary data")

    # 使用临时文件
    process_file(filepath)
# 临时目录自动删除
```

---

## 7. with 语句的高级用法

### 7.1 条件上下文管理器

```python
from contextlib import nullcontext

def process_data(data, use_cache=False):
    # 根据条件选择是否使用缓存
    cache_context = get_cache() if use_cache else nullcontext()

    with cache_context as cache:
        if cache:
            cached = cache.get("data")
            if cached:
                return cached

        result = expensive_computation(data)

        if cache:
            cache.set("data", result)

        return result
```

### 7.2 动态上下文管理器

```python
def get_context_manager(resource_type):
    """根据类型返回不同的上下文管理器"""
    if resource_type == "file":
        return open("data.txt")
    elif resource_type == "db":
        return get_db()
    elif resource_type == "cache":
        return get_cache()

# 动态选择
resource_type = "db"
with get_context_manager(resource_type) as resource:
    # 使用资源
    pass
```

### 7.3 上下文管理器链

```python
from contextlib import ExitStack

def process_multiple_files(filenames):
    """同时打开多个文件"""
    with ExitStack() as stack:
        # 动态添加上下文管理器
        files = [stack.enter_context(open(fname)) for fname in filenames]

        # 处理所有文件
        for f in files:
            process(f.read())
    # 所有文件自动关闭
```

---

## 8. with 语句的性能考虑

### 8.1 开销分析

```python
import time

# 测试 with 语句的开销
def test_with_overhead():
    iterations = 1000000

    # 使用 with 语句
    start = time.time()
    for _ in range(iterations):
        with open("test.txt", "w") as f:
            f.write("test")
    with_time = time.time() - start

    # 手动管理
    start = time.time()
    for _ in range(iterations):
        f = open("test.txt", "w")
        try:
            f.write("test")
        finally:
            f.close()
    manual_time = time.time() - start

    print(f"With statement: {with_time:.2f}s")
    print(f"Manual: {manual_time:.2f}s")
    print(f"Overhead: {(with_time - manual_time) / manual_time * 100:.2f}%")
```

**结论：**
- `with` 语句的开销非常小（通常 < 5%）
- 代码可读性和安全性的提升远超过微小的性能损失
- 在生产环境中应该优先使用 `with` 语句

### 8.2 避免不必要的上下文管理器

```python
# ❌ 不必要的上下文管理器
with some_context():
    pass  # 什么都不做

# ✅ 只在需要时使用
if need_context:
    with some_context():
        do_work()
else:
    do_work()
```

---

## 9. with 语句的常见错误

### 9.1 忘记 as 子句

```python
# ❌ 错误：忘记 as 子句
with open("file.txt"):
    data = ???  # 无法访问文件对象

# ✅ 正确：使用 as 子句
with open("file.txt") as f:
    data = f.read()
```

### 9.2 在 with 块外使用资源

```python
# ❌ 错误：在 with 块外使用
with open("file.txt") as f:
    pass

data = f.read()  # ValueError: I/O operation on closed file

# ✅ 正确：在 with 块内使用
with open("file.txt") as f:
    data = f.read()
```

### 9.3 混淆 with 和 if

```python
# ❌ 错误：把 with 当成 if
with some_condition:  # SyntaxError
    pass

# ✅ 正确：with 需要上下文管理器
with some_context_manager():
    pass
```

---

## 10. 在 AI Agent 后端中的应用

### 10.1 FastAPI 依赖注入

```python
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    """数据库会话依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db)  # 自动注入
):
    user = db.query(User).filter(User.id == user_id).first()
    return user
# db 自动关闭
```

### 10.2 请求上下文

```python
from contextvars import ContextVar

request_id_var = ContextVar("request_id")

@contextmanager
def request_context(request_id):
    """请求上下文管理器"""
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)

@app.get("/api/data")
async def get_data(request_id: str):
    with request_context(request_id):
        # 在整个请求处理过程中可以访问 request_id
        logger.info(f"[{request_id_var.get()}] Processing request")
        return process_request()
```

### 10.3 AI 模型推理

```python
@contextmanager
def load_model(model_name):
    """AI 模型加载上下文管理器"""
    print(f"Loading model: {model_name}")
    model = load_model_from_disk(model_name)
    model.to("cuda")  # 移到 GPU

    try:
        yield model
    finally:
        print(f"Unloading model: {model_name}")
        model.to("cpu")  # 释放 GPU 内存
        del model

@app.post("/predict")
async def predict(data: PredictRequest):
    with load_model("gpt-3.5") as model:
        result = model.predict(data.text)
        return result
    # 模型自动卸载
```

---

## 总结

### with 语句的核心特性

| 特性 | 说明 |
|------|------|
| **语法糖** | 简化 try-finally 模式 |
| **自动清理** | 保证资源释放 |
| **异常安全** | 即使异常也能清理 |
| **可组合** | 支持多个上下文管理器 |
| **零开销** | 性能损失可忽略 |

### 使用建议

1. ✅ **优先使用 with**：任何需要清理的资源都应该用 `with`
2. ✅ **一行多个**：多个上下文管理器用逗号分隔
3. ✅ **明确范围**：资源的生命周期应该清晰可见
4. ❌ **避免嵌套过深**：超过3层嵌套考虑重构
5. ❌ **不要在外部使用**：不要在 `with` 块外使用资源

---

**版本：** v1.0
**最后更新：** 2026-02-11
