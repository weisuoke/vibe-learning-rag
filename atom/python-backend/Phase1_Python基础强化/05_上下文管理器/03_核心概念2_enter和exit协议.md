# 核心概念2：__enter__ 和 __exit__ 协议

## 概述

**上下文管理器协议由 `__enter__` 和 `__exit__` 两个特殊方法定义，任何实现了这两个方法的对象都可以用 `with` 语句管理。**

---

## 1. 协议定义

### 1.1 完整的协议签名

```python
class ContextManager:
    def __enter__(self):
        """
        进入 with 块时调用
        返回值：赋给 as 后的变量
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出 with 块时调用
        参数：
            exc_type: 异常类型（如果有异常）
            exc_val: 异常实例（如果有异常）
            exc_tb: 异常追踪信息（如果有异常）
        返回值：
            True: 抑制异常
            False: 传播异常
        """
        return False
```

### 1.2 协议的最小实现

```python
class MinimalContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

# 可以用 with 语句
with MinimalContext():
    print("Inside context")
```

---

## 2. __enter__ 方法详解

### 2.1 基本用法

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """打开文件并返回文件对象"""
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file  # 返回值赋给 as 后的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭文件"""
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        return False

# 使用
with FileManager("data.txt", "w") as f:
    f.write("Hello, World!")  # f 是 __enter__ 的返回值
```

### 2.2 返回值的选择

**返回 self（管理器本身）：**
```python
class DatabaseConnection:
    def __init__(self, db_url):
        self.db_url = db_url
        self.conn = None

    def __enter__(self):
        self.conn = connect(self.db_url)
        return self  # 返回管理器本身

    def query(self, sql):
        return self.conn.execute(sql)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        return False

# 使用管理器的方法
with DatabaseConnection("postgresql://...") as db:
    result = db.query("SELECT * FROM users")
```

**返回被管理的资源：**
```python
class FileManager:
    def __enter__(self):
        self.file = open("data.txt")
        return self.file  # 返回文件对象

with FileManager() as f:
    data = f.read()  # 直接使用文件对象
```

**返回多个值（元组）：**
```python
class MultiResource:
    def __enter__(self):
        self.db = get_db()
        self.cache = get_cache()
        return self.db, self.cache  # 返回元组

with MultiResource() as (db, cache):
    user = db.query(User).first()
    cache.set("user", user)
```

**不返回值（None）：**
```python
class Timer:
    def __enter__(self):
        self.start = time.time()
        return None  # 或者不写 return

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"Time: {elapsed:.2f}s")
        return False

# 不需要 as 子句
with Timer():
    expensive_operation()
```

### 2.3 __enter__ 中的初始化逻辑

```python
class DatabaseSession:
    def __init__(self, db_url):
        self.db_url = db_url
        self.session = None
        self.transaction = None

    def __enter__(self):
        """建立连接并开始事务"""
        # 1. 建立连接
        self.session = create_session(self.db_url)

        # 2. 开始事务
        self.transaction = self.session.begin()

        # 3. 设置会话参数
        self.session.execute("SET timezone = 'UTC'")

        # 4. 返回会话对象
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.transaction.commit()
        else:
            self.transaction.rollback()
        self.session.close()
        return False
```

### 2.4 __enter__ 中的异常处理

```python
class SafeConnection:
    def __enter__(self):
        try:
            self.conn = connect_to_database()
            return self.conn
        except ConnectionError as e:
            # __enter__ 中的异常会直接传播
            # __exit__ 不会被调用
            print(f"Failed to connect: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果 __enter__ 失败，这里不会执行
        if self.conn:
            self.conn.close()
        return False
```

**关键点：**
- 如果 `__enter__` 抛出异常，`__exit__` 不会被调用
- `with` 块内的代码不会执行
- 异常直接传播到外层

---

## 3. __exit__ 方法详解

### 3.1 参数详解

```python
class ExitDemo:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exc_type: 异常类型（如 ValueError, TypeError）
        exc_val: 异常实例（异常对象）
        exc_tb: 异常追踪信息（traceback 对象）
        """
        if exc_type is None:
            print("No exception occurred")
        else:
            print(f"Exception type: {exc_type}")
            print(f"Exception value: {exc_val}")
            print(f"Exception traceback: {exc_tb}")

        return False  # 不抑制异常

# 测试无异常情况
with ExitDemo():
    print("Normal execution")
# 输出：No exception occurred

# 测试有异常情况
try:
    with ExitDemo():
        raise ValueError("Test error")
except ValueError:
    pass
# 输出：
# Exception type: <class 'ValueError'>
# Exception value: Test error
# Exception traceback: <traceback object>
```

### 3.2 返回值的含义

**返回 False（默认）：异常继续传播**
```python
class PropagateException:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleaning up...")
        return False  # 或者不写 return（默认 None，等价于 False）

try:
    with PropagateException():
        raise ValueError("Error!")
except ValueError as e:
    print(f"Caught: {e}")

# 输出：
# Cleaning up...
# Caught: Error!
```

**返回 True：抑制异常**
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

### 3.3 清理逻辑的实现

```python
class ResourceManager:
    def __init__(self):
        self.resources = []

    def __enter__(self):
        # 获取多个资源
        self.resources.append(acquire_resource_1())
        self.resources.append(acquire_resource_2())
        self.resources.append(acquire_resource_3())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 按相反顺序释放资源
        for resource in reversed(self.resources):
            try:
                release_resource(resource)
            except Exception as e:
                # 记录清理错误，但不影响其他资源的释放
                print(f"Error releasing resource: {e}")

        # 不抑制原始异常
        return False
```

### 3.4 异常处理策略

**策略1：记录异常但不抑制**
```python
class LoggingContext:
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Exception in context: {exc_val}")
        return False  # 让异常继续传播
```

**策略2：转换异常类型**
```python
class ExceptionTransformer:
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ValueError:
            # 转换为自定义异常
            raise CustomError(f"Invalid value: {exc_val}") from exc_val
        return False
```

**策略3：选择性抑制**
```python
class SelectiveSuppression:
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 只抑制预期的异常
        if exc_type in (ValueError, KeyError):
            print(f"Handled expected error: {exc_val}")
            return True
        # 其他异常继续传播
        return False
```

**策略4：重试机制**
```python
class RetryContext:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.attempt = 0

    def __enter__(self):
        self.attempt += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.attempt < self.max_retries:
            print(f"Retry {self.attempt}/{self.max_retries}")
            return True  # 抑制异常，允许重试
        return False
```

---

## 4. 协议的完整生命周期

### 4.1 正常执行流程

```python
class LifecycleDemo:
    def __init__(self, name):
        self.name = name
        print(f"1. __init__: {self.name}")

    def __enter__(self):
        print(f"2. __enter__: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"4. __exit__: {self.name}")
        return False

print("0. Before with")
with LifecycleDemo("test") as ctx:
    print(f"3. Inside with: {ctx.name}")
print("5. After with")

# 输出：
# 0. Before with
# 1. __init__: test
# 2. __enter__: test
# 3. Inside with: test
# 4. __exit__: test
# 5. After with
```

### 4.2 异常执行流程

```python
class ExceptionLifecycle:
    def __enter__(self):
        print("1. __enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"3. __exit__: exc_type={exc_type}")
        return False

try:
    with ExceptionLifecycle():
        print("2. Inside with")
        raise ValueError("Error!")
        print("This won't execute")
except ValueError:
    print("4. Exception caught")

# 输出：
# 1. __enter__
# 2. Inside with
# 3. __exit__: exc_type=<class 'ValueError'>
# 4. Exception caught
```

### 4.3 嵌套上下文管理器的执行顺序

```python
class Nested:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"Enter: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exit: {self.name}")
        return False

with Nested("outer"):
    with Nested("inner"):
        print("Inside both")

# 输出：
# Enter: outer
# Enter: inner
# Inside both
# Exit: inner
# Exit: outer
```

**执行顺序：**
1. 外层 `__enter__`
2. 内层 `__enter__`
3. `with` 块代码
4. 内层 `__exit__`
5. 外层 `__exit__`

---

## 5. 实际应用案例

### 5.1 数据库事务管理

```python
class Transaction:
    def __init__(self, db):
        self.db = db
        self.transaction = None

    def __enter__(self):
        """开始事务"""
        print("BEGIN TRANSACTION")
        self.transaction = self.db.begin()
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        """提交或回滚事务"""
        if exc_type is None:
            print("COMMIT")
            self.transaction.commit()
        else:
            print(f"ROLLBACK (due to {exc_type.__name__})")
            self.transaction.rollback()
        return False  # 不抑制异常

# 使用
with Transaction(db) as session:
    session.execute("INSERT INTO users ...")
    session.execute("UPDATE accounts ...")
# 自动提交

# 异常时自动回滚
try:
    with Transaction(db) as session:
        session.execute("INSERT INTO users ...")
        raise ValueError("Error!")
except ValueError:
    pass
# 自动回滚
```

### 5.2 文件锁管理

```python
import fcntl

class FileLock:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        """获取文件锁"""
        self.file = open(self.filename, "a")
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        print(f"Lock acquired: {self.filename}")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """释放文件锁"""
        if self.file:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            print(f"Lock released: {self.filename}")
        return False

# 使用
with FileLock("data.txt") as f:
    f.write("Protected write\n")
# 锁自动释放
```

### 5.3 临时环境变量

```python
import os

class TempEnv:
    def __init__(self, **env_vars):
        self.env_vars = env_vars
        self.old_values = {}

    def __enter__(self):
        """设置临时环境变量"""
        for key, value in self.env_vars.items():
            self.old_values[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """恢复原环境变量"""
        for key, old_value in self.old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        return False

# 使用
with TempEnv(DEBUG="true", LOG_LEVEL="debug"):
    # 在这个范围内环境变量被修改
    run_tests()
# 环境变量自动恢复
```

### 5.4 性能监控

```python
import time

class PerformanceMonitor:
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """记录开始时间"""
        self.start_time = time.time()
        print(f"[{self.operation_name}] Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """计算并报告执行时间"""
        elapsed = time.time() - self.start_time

        if exc_type is None:
            print(f"[{self.operation_name}] Completed in {elapsed:.2f}s")
        else:
            print(f"[{self.operation_name}] Failed after {elapsed:.2f}s")

        return False

# 使用
with PerformanceMonitor("Database query"):
    result = db.query(User).all()
```

### 5.5 FastAPI 数据库会话

```python
from sqlalchemy.orm import Session

class DatabaseSession:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.session = None

    def __enter__(self):
        """创建数据库会话"""
        self.session = self.session_factory()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭会话"""
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()
        return False

# 在 FastAPI 中使用
def get_db():
    with DatabaseSession(SessionLocal) as session:
        yield session

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```

---

## 6. 协议的高级技巧

### 6.1 可重入上下文管理器

```python
import threading

class ReentrantLock:
    def __init__(self):
        self.lock = threading.RLock()  # 可重入锁
        self.count = 0

    def __enter__(self):
        self.lock.acquire()
        self.count += 1
        print(f"Lock acquired (count: {self.count})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count -= 1
        print(f"Lock released (count: {self.count})")
        self.lock.release()
        return False

# 可以嵌套使用
lock = ReentrantLock()
with lock:
    print("Outer context")
    with lock:
        print("Inner context")
# 正常工作
```

### 6.2 异步上下文管理器

```python
class AsyncContextManager:
    async def __aenter__(self):
        """异步进入"""
        print("Async enter")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出"""
        print("Async exit")
        await asyncio.sleep(0.1)
        return False

# 使用 async with
async def main():
    async with AsyncContextManager():
        print("Inside async context")
```

### 6.3 条件上下文管理器

```python
class ConditionalContext:
    def __init__(self, condition):
        self.condition = condition
        self.active = False

    def __enter__(self):
        if self.condition:
            self.active = True
            print("Context activated")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            print("Context deactivated")
        return False

# 根据条件决定是否激活
with ConditionalContext(condition=True):
    print("This will be wrapped")

with ConditionalContext(condition=False):
    print("This won't be wrapped")
```

---

## 7. 常见错误和陷阱

### 7.1 忘记返回值

```python
# ❌ 错误：__enter__ 没有返回值
class BadContext:
    def __enter__(self):
        self.resource = acquire_resource()
        # 忘记 return

    def __exit__(self, exc_type, exc_val, exc_tb):
        release_resource(self.resource)
        return False

with BadContext() as ctx:
    print(ctx)  # None

# ✅ 正确：返回资源或 self
class GoodContext:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource  # 或 return self
```

### 7.2 __exit__ 中抛出异常

```python
# ❌ 危险：__exit__ 中抛出新异常
class DangerousContext:
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("Error in __exit__")
        # 会覆盖原始异常

# ✅ 正确：捕获 __exit__ 中的异常
class SafeContext:
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        return False
```

### 7.3 错误的异常抑制

```python
# ❌ 错误：总是返回 True
class BadSuppression:
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # 抑制所有异常，包括 KeyboardInterrupt

# ✅ 正确：选择性抑制
class GoodSuppression:
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type in (ValueError, KeyError):
            return True
        return False
```

---

## 总结

### 协议核心要点

| 方法 | 调用时机 | 返回值 | 作用 |
|------|---------|--------|------|
| `__enter__` | 进入 with 块前 | 任意对象 | 资源获取和初始化 |
| `__exit__` | 退出 with 块时 | True/False | 资源释放和异常处理 |

### 最佳实践

1. ✅ **__enter__ 返回有用的对象**：返回被管理的资源或 self
2. ✅ **__exit__ 总是清理**：无论是否异常都要清理资源
3. ✅ **默认不抑制异常**：返回 False 让异常传播
4. ✅ **选择性抑制**：只抑制预期的异常类型
5. ❌ **避免 __exit__ 抛出异常**：会覆盖原始异常
6. ❌ **不要忘记返回值**：__exit__ 必须返回 True 或 False

---

**版本：** v1.0
**最后更新：** 2026-02-11
