# 核心概念3：contextlib 模块

## 概述

**`contextlib` 是 Python 标准库提供的上下文管理器工具模块，提供了创建和使用上下文管理器的便捷方法，最常用的是 `@contextmanager` 装饰器。**

---

## 1. @contextmanager 装饰器

### 1.1 基本用法

**核心思想：** 用生成器函数 + 装饰器快速创建上下文管理器，无需定义类

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    # 1. __enter__ 阶段：资源获取
    print("Setup")
    resource = acquire_resource()

    try:
        # 2. yield：提供资源给 with 块
        yield resource
    finally:
        # 3. __exit__ 阶段：资源释放
        print("Cleanup")
        release_resource(resource)

# 使用
with my_context() as res:
    use_resource(res)
```

**等价的类实现：**
```python
class MyContext:
    def __enter__(self):
        print("Setup")
        self.resource = acquire_resource()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleanup")
        release_resource(self.resource)
        return False
```

### 1.2 执行流程详解

```python
@contextmanager
def demo():
    print("1. Before yield (__enter__)")
    yield "resource"
    print("3. After yield (__exit__)")

with demo() as res:
    print(f"2. Inside with: {res}")

# 输出：
# 1. Before yield (__enter__)
# 2. Inside with: resource
# 3. After yield (__exit__)
```

**关键点：**
- `yield` 前的代码 = `__enter__` 方法
- `yield` 的值 = `__enter__` 的返回值
- `yield` 后的代码 = `__exit__` 方法
- `finally` 确保清理代码总是执行

### 1.3 异常处理

```python
@contextmanager
def safe_context():
    print("Enter")
    try:
        yield
    except ValueError as e:
        print(f"Caught ValueError: {e}")
        # 不重新抛出 = 抑制异常
    except Exception as e:
        print(f"Caught other exception: {e}")
        raise  # 重新抛出 = 不抑制
    finally:
        print("Cleanup")

# ValueError 被抑制
with safe_context():
    raise ValueError("Test")
print("Program continues")

# TypeError 不被抑制
try:
    with safe_context():
        raise TypeError("Test")
except TypeError:
    print("TypeError propagated")
```

**异常处理规则：**
- 在 `try` 块中捕获异常 = 抑制异常
- 重新 `raise` = 不抑制异常
- `finally` 总是执行

### 1.4 实际应用示例

**示例1：数据库事务**
```python
@contextmanager
def transaction(db):
    """数据库事务上下文管理器"""
    print("BEGIN TRANSACTION")
    try:
        yield db
        print("COMMIT")
        db.commit()
    except Exception as e:
        print(f"ROLLBACK: {e}")
        db.rollback()
        raise

# 使用
with transaction(db):
    db.execute("INSERT INTO users ...")
    db.execute("UPDATE accounts ...")
```

**示例2：临时目录切换**
```python
import os

@contextmanager
def cd(path):
    """临时切换工作目录"""
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

# 使用
with cd("/tmp"):
    # 在 /tmp 目录下工作
    os.system("ls")
# 自动回到原目录
```

**示例3：性能计时**
```python
import time

@contextmanager
def timer(name):
    """性能计时器"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

# 使用
with timer("Database query"):
    result = db.query(User).all()
```

**示例4：临时配置修改**
```python
@contextmanager
def temp_config(**kwargs):
    """临时修改配置"""
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = config.get(key)
        config[key] = value

    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                config.pop(key, None)
            else:
                config[key] = old_value

# 使用
with temp_config(DEBUG=True, LOG_LEVEL="debug"):
    run_tests()
# 配置自动恢复
```

---

## 2. contextlib.closing

### 2.1 基本用法

**作用：** 自动调用对象的 `close()` 方法

```python
from contextlib import closing
from urllib.request import urlopen

# 自动关闭 URL 连接
with closing(urlopen("https://example.com")) as page:
    content = page.read()
# page.close() 自动调用
```

**等价实现：**
```python
@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()
```

### 2.2 适用场景

**任何有 `close()` 方法的对象：**
```python
# 网络连接
with closing(socket.socket()) as sock:
    sock.connect(("localhost", 8080))

# 数据库游标
with closing(conn.cursor()) as cursor:
    cursor.execute("SELECT * FROM users")

# 自定义对象
class MyResource:
    def close(self):
        print("Resource closed")

with closing(MyResource()) as res:
    # 使用资源
    pass
```

### 2.3 与 with 语句的对比

```python
# ❌ 不支持上下文管理器的对象
page = urlopen("https://example.com")
try:
    content = page.read()
finally:
    page.close()

# ✅ 使用 closing 包装
with closing(urlopen("https://example.com")) as page:
    content = page.read()

# ✅ 如果对象本身支持上下文管理器，直接用 with
with open("file.txt") as f:
    content = f.read()
```

---

## 3. contextlib.suppress

### 3.1 基本用法

**作用：** 抑制指定的异常类型

```python
from contextlib import suppress

# 抑制 FileNotFoundError
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")
# 如果文件不存在，不会抛出异常

# 等价于：
try:
    os.remove("nonexistent_file.txt")
except FileNotFoundError:
    pass
```

### 3.2 抑制多个异常

```python
# 抑制多种异常
with suppress(FileNotFoundError, PermissionError):
    os.remove("file.txt")

# 等价于：
try:
    os.remove("file.txt")
except (FileNotFoundError, PermissionError):
    pass
```

### 3.3 实际应用场景

**场景1：清理临时文件**
```python
# 删除临时文件，不存在也不报错
with suppress(FileNotFoundError):
    os.remove("/tmp/temp_file.txt")
```

**场景2：尝试导入可选依赖**
```python
# 尝试导入可选库
with suppress(ImportError):
    import optional_library
    USE_OPTIONAL = True
else:
    USE_OPTIONAL = False
```

**场景3：字典操作**
```python
# 删除字典键，不存在也不报错
with suppress(KeyError):
    del my_dict["nonexistent_key"]

# 等价于：
my_dict.pop("nonexistent_key", None)
```

### 3.4 注意事项

```python
# ❌ 不要滥用 suppress
with suppress(Exception):
    # 抑制所有异常是危险的
    risky_operation()

# ✅ 只抑制预期的异常
with suppress(ValueError, KeyError):
    # 只抑制特定的预期异常
    safe_operation()
```

---

## 4. contextlib.redirect_stdout / redirect_stderr

### 4.1 redirect_stdout

**作用：** 重定向标准输出

```python
from contextlib import redirect_stdout
import io

# 捕获 print 输出
f = io.StringIO()
with redirect_stdout(f):
    print("Hello, World!")
    print("Second line")

output = f.getvalue()
print(f"Captured: {output}")
# 输出：Captured: Hello, World!\nSecond line\n
```

### 4.2 redirect_stderr

**作用：** 重定向标准错误

```python
from contextlib import redirect_stderr
import sys

# 捕获错误输出
f = io.StringIO()
with redirect_stderr(f):
    sys.stderr.write("Error message\n")

error_output = f.getvalue()
print(f"Captured error: {error_output}")
```

### 4.3 实际应用场景

**场景1：测试输出**
```python
def test_print_function():
    """测试函数的输出"""
    f = io.StringIO()
    with redirect_stdout(f):
        my_function()

    output = f.getvalue()
    assert "Expected output" in output
```

**场景2：静默第三方库**
```python
# 静默第三方库的输出
with redirect_stdout(io.StringIO()):
    noisy_library.process()  # 不会打印任何内容
```

**场景3：日志重定向**
```python
# 将输出重定向到文件
with open("output.log", "w") as f:
    with redirect_stdout(f):
        print("This goes to file")
        run_analysis()
```

---

## 5. contextlib.ExitStack

### 5.1 基本用法

**作用：** 动态管理多个上下文管理器

```python
from contextlib import ExitStack

# 动态打开多个文件
filenames = ["file1.txt", "file2.txt", "file3.txt"]

with ExitStack() as stack:
    files = [stack.enter_context(open(fname)) for fname in filenames]

    # 处理所有文件
    for f in files:
        print(f.read())
# 所有文件自动关闭
```

### 5.2 条件上下文管理器

```python
def process_data(use_cache=False):
    with ExitStack() as stack:
        # 根据条件添加上下文管理器
        if use_cache:
            cache = stack.enter_context(get_cache())
        else:
            cache = None

        # 总是需要数据库
        db = stack.enter_context(get_db())

        # 处理数据
        if cache:
            cached = cache.get("data")
            if cached:
                return cached

        result = db.query(Data).all()

        if cache:
            cache.set("data", result)

        return result
```

### 5.3 注册清理回调

```python
with ExitStack() as stack:
    # 注册清理函数
    stack.callback(print, "Cleanup 1")
    stack.callback(print, "Cleanup 2")

    # 执行操作
    print("Main operation")

# 输出：
# Main operation
# Cleanup 2
# Cleanup 1
# （注意：回调按 LIFO 顺序执行）
```

### 5.4 实际应用场景

**场景1：批量文件处理**
```python
def merge_files(input_files, output_file):
    """合并多个文件"""
    with ExitStack() as stack:
        # 打开所有输入文件
        infiles = [stack.enter_context(open(f)) for f in input_files]

        # 打开输出文件
        outfile = stack.enter_context(open(output_file, "w"))

        # 合并内容
        for infile in infiles:
            outfile.write(infile.read())
```

**场景2：资源池管理**
```python
def process_with_resources(resource_count):
    """动态获取多个资源"""
    with ExitStack() as stack:
        resources = []
        for i in range(resource_count):
            res = stack.enter_context(acquire_resource(i))
            resources.append(res)

        # 使用所有资源
        for res in resources:
            res.process()
```

**场景3：清理多个临时文件**
```python
def create_temp_files(count):
    """创建多个临时文件"""
    with ExitStack() as stack:
        temp_files = []
        for i in range(count):
            f = tempfile.NamedTemporaryFile(delete=False)
            temp_files.append(f.name)
            f.close()

            # 注册清理函数
            stack.callback(os.unlink, f.name)

        # 使用临时文件
        yield temp_files
    # 所有临时文件自动删除
```

---

## 6. contextlib.nullcontext

### 6.1 基本用法

**作用：** 提供一个"什么都不做"的上下文管理器

```python
from contextlib import nullcontext

def process_data(use_lock=False):
    # 根据条件选择是否使用锁
    lock = threading.Lock() if use_lock else nullcontext()

    with lock:
        # 处理数据
        shared_data.append(item)

# use_lock=True: 使用真实的锁
# use_lock=False: 使用 nullcontext，不加锁
```

### 6.2 条件上下文管理器

```python
def save_data(data, use_transaction=False):
    """根据条件决定是否使用事务"""
    ctx = transaction(db) if use_transaction else nullcontext(db)

    with ctx as session:
        session.add(data)
        # 如果 use_transaction=True，自动提交/回滚
        # 如果 use_transaction=False，不做任何事
```

### 6.3 实际应用场景

**场景1：可选的性能监控**
```python
def process(enable_timing=False):
    timer_ctx = timer("process") if enable_timing else nullcontext()

    with timer_ctx:
        # 执行操作
        expensive_computation()
```

**场景2：可选的日志上下文**
```python
def api_call(request_id=None):
    log_ctx = log_context(request_id) if request_id else nullcontext()

    with log_ctx:
        # 处理请求
        return handle_request()
```

---

## 7. contextlib.asynccontextmanager

### 7.1 基本用法

**作用：** 创建异步上下文管理器

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def async_resource():
    # 异步获取资源
    print("Acquiring resource")
    await asyncio.sleep(0.1)
    resource = "async_resource"

    try:
        yield resource
    finally:
        # 异步释放资源
        print("Releasing resource")
        await asyncio.sleep(0.1)

# 使用
async def main():
    async with async_resource() as res:
        print(f"Using {res}")

asyncio.run(main())
```

### 7.2 异步数据库连接

```python
@asynccontextmanager
async def get_async_db():
    """异步数据库连接"""
    conn = await asyncpg.connect("postgresql://...")
    try:
        yield conn
    finally:
        await conn.close()

# 使用
async def fetch_users():
    async with get_async_db() as conn:
        users = await conn.fetch("SELECT * FROM users")
        return users
```

### 7.3 FastAPI 中的应用

```python
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理"""
    # 启动时
    print("Starting up")
    await init_database()
    await init_cache()

    yield  # 应用运行期间

    # 关闭时
    print("Shutting down")
    await close_database()
    await close_cache()

app = FastAPI(lifespan=lifespan)
```

---

## 8. contextlib 的高级用法

### 8.1 嵌套上下文管理器

```python
@contextmanager
def outer():
    print("Outer enter")
    with inner():
        yield
    print("Outer exit")

@contextmanager
def inner():
    print("Inner enter")
    yield
    print("Inner exit")

with outer():
    print("Main")

# 输出：
# Outer enter
# Inner enter
# Main
# Inner exit
# Outer exit
```

### 8.2 组合多个装饰器

```python
@contextmanager
def timer_and_log(name):
    """组合计时和日志"""
    start = time.time()
    logger.info(f"[{name}] Started")

    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"[{name}] Completed in {elapsed:.2f}s")

with timer_and_log("Database query"):
    result = db.query(User).all()
```

### 8.3 可重入上下文管理器

```python
from threading import RLock

@contextmanager
def reentrant_lock():
    """可重入锁"""
    lock = RLock()
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# 可以嵌套使用
with reentrant_lock():
    with reentrant_lock():
        # 不会死锁
        pass
```

---

## 9. contextlib 最佳实践

### 9.1 选择合适的工具

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| 简单的资源管理 | `@contextmanager` | 代码简洁 |
| 对象有 `close()` 方法 | `closing` | 专门为此设计 |
| 抑制特定异常 | `suppress` | 语义清晰 |
| 动态数量的资源 | `ExitStack` | 灵活管理 |
| 条件上下文管理 | `nullcontext` | 避免 if-else |
| 异步资源管理 | `@asynccontextmanager` | 支持 async/await |

### 9.2 代码示例对比

**❌ 不推荐：手动实现类**
```python
class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False
```

**✅ 推荐：使用 @contextmanager**
```python
@contextmanager
def file_manager(filename):
    f = open(filename)
    try:
        yield f
    finally:
        f.close()
```

### 9.3 常见错误

**错误1：忘记 try-finally**
```python
# ❌ 错误：没有 finally
@contextmanager
def bad_context():
    resource = acquire()
    yield resource
    release(resource)  # 异常时不会执行

# ✅ 正确：使用 try-finally
@contextmanager
def good_context():
    resource = acquire()
    try:
        yield resource
    finally:
        release(resource)  # 总是执行
```

**错误2：在 yield 后抛出异常**
```python
# ❌ 错误：覆盖原始异常
@contextmanager
def bad_cleanup():
    try:
        yield
    finally:
        raise RuntimeError("Cleanup error")  # 覆盖原始异常

# ✅ 正确：捕获清理异常
@contextmanager
def good_cleanup():
    try:
        yield
    finally:
        try:
            cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
```

---

## 10. 在 AI Agent 后端中的应用

### 10.1 FastAPI 数据库会话

```python
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

@asynccontextmanager
async def get_db_session():
    """异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# 在 FastAPI 中使用
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with get_db_session() as db:
        user = await db.get(User, user_id)
        return user
```

### 10.2 AI 模型推理上下文

```python
@contextmanager
def load_model(model_name):
    """AI 模型加载上下文"""
    print(f"Loading {model_name}")
    model = load_model_from_disk(model_name)
    model.to("cuda")

    try:
        yield model
    finally:
        print(f"Unloading {model_name}")
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

# 使用
@app.post("/predict")
async def predict(request: PredictRequest):
    with load_model("gpt-3.5") as model:
        result = model.predict(request.text)
        return result
```

### 10.3 请求追踪

```python
from contextvars import ContextVar

request_id_var = ContextVar("request_id")

@contextmanager
def request_context(request_id):
    """请求上下文"""
    token = request_id_var.set(request_id)
    logger.info(f"[{request_id}] Request started")

    try:
        yield
    finally:
        logger.info(f"[{request_id}] Request finished")
        request_id_var.reset(token)

# 在 FastAPI 中使用
@app.middleware("http")
async def add_request_context(request, call_next):
    request_id = str(uuid.uuid4())
    with request_context(request_id):
        response = await call_next(request)
        return response
```

### 10.4 缓存管理

```python
@contextmanager
def cache_context(key, ttl=3600):
    """缓存上下文管理器"""
    # 尝试从缓存获取
    cached = cache.get(key)
    if cached:
        yield cached
        return

    # 缓存未命中，计算结果
    result = None

    def set_result(value):
        nonlocal result
        result = value

    yield set_result

    # 保存到缓存
    if result is not None:
        cache.set(key, result, ttl)

# 使用
@app.get("/expensive-data")
async def get_expensive_data():
    with cache_context("expensive_data") as cached_or_setter:
        if callable(cached_or_setter):
            # 缓存未命中，计算结果
            result = expensive_computation()
            cached_or_setter(result)
            return result
        else:
            # 缓存命中
            return cached_or_setter
```

---

## 总结

### contextlib 工具对比

| 工具 | 用途 | 适用场景 |
|------|------|---------|
| `@contextmanager` | 创建上下文管理器 | 大多数场景 |
| `closing` | 自动调用 close() | 有 close() 方法的对象 |
| `suppress` | 抑制异常 | 预期的异常 |
| `redirect_stdout/stderr` | 重定向输出 | 测试、日志 |
| `ExitStack` | 动态管理多个上下文 | 数量不确定的资源 |
| `nullcontext` | 条件上下文管理 | 可选的上下文 |
| `@asynccontextmanager` | 异步上下文管理器 | 异步资源 |

### 最佳实践

1. ✅ **优先使用 @contextmanager**：比手动实现类更简洁
2. ✅ **总是使用 try-finally**：确保清理代码执行
3. ✅ **选择合适的工具**：根据场景选择最合适的 contextlib 工具
4. ✅ **异步用 @asynccontextmanager**：异步资源管理
5. ❌ **避免在 finally 中抛出异常**：会覆盖原始异常
6. ❌ **不要滥用 suppress**：只抑制预期的异常

---

**版本：** v1.0
**最后更新：** 2026-02-11
