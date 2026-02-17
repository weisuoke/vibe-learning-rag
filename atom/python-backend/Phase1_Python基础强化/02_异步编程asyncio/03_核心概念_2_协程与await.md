# 核心概念2：协程与 await

> 理解 Python 异步编程的核心机制

---

## 什么是协程？

**一句话定义：** 协程是一个可以暂停和恢复执行的函数，是 asyncio 实现并发的基础单元。

---

## 直觉理解

### 类比：看书时接电话

**普通函数（不可暂停）：**
```
1. 开始看书
2. 必须看完整本书
3. 不能中断
4. 看完才能做其他事
```

**协程（可暂停）：**
```
1. 开始看书（第1页）
2. 电话响了（await 触发）
3. 放下书，记住页码（暂停）
4. 接电话（处理其他任务）
5. 电话结束
6. 拿起书，从记住的页码继续看（恢复）
7. 看完书
```

**协程就是"可以放下书去接电话，然后继续看书的能力"**

---

## 协程的定义和创建

### 1. 定义协程函数

```python
import asyncio

# 使用 async def 定义协程函数
async def fetch_user(user_id: int):
    print(f"开始获取用户 {user_id}")
    await asyncio.sleep(0.1)  # 模拟 I/O 操作
    print(f"完成获取用户 {user_id}")
    return {"id": user_id, "name": "Alice"}
```

**关键点：**
- `async def` 定义协程函数
- 协程函数内可以使用 `await`
- 协程函数返回一个协程对象

### 2. 调用协程函数

```python
# 调用协程函数，返回协程对象（不执行）
coro = fetch_user(1)
print(type(coro))  # <class 'coroutine'>

# 协程对象必须被 await 或传给事件循环才会执行
result = await fetch_user(1)  # 在异步函数中
# 或
asyncio.run(fetch_user(1))  # 在同步代码中
```

**重要：** 调用协程函数不会执行函数体，只是创建协程对象！

### 3. 协程的三种状态

```python
import asyncio
import inspect

async def example():
    await asyncio.sleep(0.1)
    return "done"

# 状态1：创建（Created）
coro = example()
print(inspect.getcoroutinestate(coro))  # CORO_CREATED

# 状态2：运行中（Running）
# 在 await 时，协程处于运行状态

# 状态3：暂停（Suspended）
# 在 await 等待时，协程处于暂停状态

# 状态4：关闭（Closed）
# 协程执行完毕或被关闭
```

---

## await 的工作原理

### 1. await 的本质

**await 做了三件事：**
1. **暂停当前协程**：保存当前执行状态
2. **让出控制权**：将控制权交还给事件循环
3. **等待完成**：等待被 await 的对象完成，然后恢复执行

```python
import asyncio

async def main():
    print("1. 开始")

    # await 暂停当前协程，让出控制权
    result = await asyncio.sleep(1)

    print("2. 继续")  # 1秒后恢复执行
    return result

asyncio.run(main())
```

**执行流程：**
```
时间 0.0s: 打印"1. 开始"
时间 0.0s: 遇到 await，暂停 main 协程
时间 0.0s: 事件循环注册 1 秒后的回调
时间 0.0s-1.0s: 事件循环可以处理其他任务
时间 1.0s: 回调触发，恢复 main 协程
时间 1.0s: 打印"2. 继续"
```

### 2. await 的语法规则

**只能 await 可等待对象（Awaitable）：**
```python
# ✅ 可以 await 的对象
await coroutine()        # 协程对象
await asyncio.sleep(1)   # 协程对象
await task               # Task 对象
await future             # Future 对象

# ❌ 不能 await 的对象
await 123                # 错误：int 不是可等待对象
await "hello"            # 错误：str 不是可等待对象
await sync_function()    # 错误：普通函数返回值不是可等待对象
```

**只能在 async def 函数内使用 await：**
```python
# ❌ 错误：在普通函数中使用 await
def sync_function():
    result = await async_function()  # SyntaxError

# ✅ 正确：在协程函数中使用 await
async def async_function():
    result = await another_async_function()
```

### 3. await 的链式调用

```python
import asyncio

async def level_3():
    print("Level 3: 开始")
    await asyncio.sleep(0.1)
    print("Level 3: 完成")
    return "L3"

async def level_2():
    print("Level 2: 开始")
    result = await level_3()  # 等待 level_3 完成
    print(f"Level 2: 收到 {result}")
    return "L2"

async def level_1():
    print("Level 1: 开始")
    result = await level_2()  # 等待 level_2 完成
    print(f"Level 1: 收到 {result}")
    return "L1"

asyncio.run(level_1())
```

**输出：**
```
Level 1: 开始
Level 2: 开始
Level 3: 开始
Level 3: 完成
Level 2: 收到 L3
Level 1: 收到 L2
```

**执行流程：**
```
level_1 开始 → await level_2
  level_2 开始 → await level_3
    level_3 开始 → await sleep(0.1)
      暂停，等待 0.1 秒
    level_3 恢复 → 返回 "L3"
  level_2 恢复 → 返回 "L2"
level_1 恢复 → 返回 "L1"
```

---

## 协程的底层实现

### 1. 协程基于生成器

Python 的协程是基于生成器（Generator）实现的：

```python
# 生成器（Generator）
def generator():
    print("开始")
    yield 1  # 暂停，返回 1
    print("继续")
    yield 2  # 暂停，返回 2
    print("结束")

gen = generator()
print(next(gen))  # 开始，输出 1
print(next(gen))  # 继续，输出 2

# 协程（Coroutine）类似生成器
async def coroutine():
    print("开始")
    await asyncio.sleep(0)  # 类似 yield
    print("继续")
    await asyncio.sleep(0)  # 类似 yield
    print("结束")
```

**相似性：**
- 都可以暂停和恢复
- 都保存执行状态
- 都可以返回值

**差异：**
- 生成器用 `yield`，协程用 `await`
- 生成器手动迭代，协程由事件循环调度
- 生成器用于惰性求值，协程用于异步 I/O

### 2. await 的底层机制

```python
# await 的简化实现
class Awaitable:
    def __await__(self):
        # 返回一个迭代器
        yield self  # 暂停，将控制权交给事件循环
        return self.result  # 恢复后返回结果

# 使用
async def example():
    result = await Awaitable()
    # 等价于
    # result = yield from Awaitable().__await__()
```

**关键点：**
- `await` 本质上是 `yield from`
- `__await__()` 方法返回一个迭代器
- 事件循环通过迭代器控制协程的暂停和恢复

### 3. 协程的执行模型

```python
# 简化的协程执行模型
class Coroutine:
    def __init__(self, gen):
        self.gen = gen  # 生成器对象

    def send(self, value):
        # 恢复协程执行
        try:
            return self.gen.send(value)
        except StopIteration as e:
            return e.value  # 协程返回值

# 事件循环执行协程
def run_coroutine(coro):
    try:
        # 启动协程
        coro.send(None)
    except StopIteration as e:
        return e.value
```

---

## 协程的高级用法

### 1. 协程装饰器

```python
import asyncio
from functools import wraps

def async_timer(func):
    """测量异步函数执行时间的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.2f}秒")
        return result
    return wrapper

@async_timer
async def fetch_data():
    await asyncio.sleep(1)
    return "data"

asyncio.run(fetch_data())
# 输出: fetch_data 耗时: 1.00秒
```

### 2. 协程上下文管理器

```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("获取资源")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")
        await asyncio.sleep(0.1)

    async def query(self):
        print("查询数据")
        return "data"

async def main():
    # 使用 async with
    async with AsyncResource() as resource:
        result = await resource.query()
        print(result)

asyncio.run(main())
```

**输出：**
```
获取资源
查询数据
data
释放资源
```

**应用场景：**
- 异步数据库连接
- 异步文件操作
- 异步网络连接

### 3. 异步生成器

```python
import asyncio

async def async_range(n):
    """异步生成器"""
    for i in range(n):
        await asyncio.sleep(0.1)  # 模拟异步操作
        yield i

async def main():
    # 使用 async for 迭代异步生成器
    async for i in async_range(5):
        print(i)

asyncio.run(main())
```

**输出：**
```
0
1
2
3
4
```

**应用场景：**
- AI 流式输出
- 大文件逐行读取
- 实时数据流处理

---

## 协程与普通函数的对比

| 特性 | 普通函数 | 协程函数 |
|------|---------|---------|
| **定义** | `def func():` | `async def func():` |
| **调用** | `func()` 立即执行 | `func()` 返回协程对象 |
| **执行** | 同步执行，阻塞 | 异步执行，可暂停 |
| **返回** | 直接返回值 | 需要 `await` 获取返回值 |
| **暂停** | 不能暂停 | 可以 `await` 暂停 |
| **并发** | 需要多线程/多进程 | 单线程并发 |
| **适用场景** | CPU 密集型 | I/O 密集型 |

---

## 在 AI Agent 开发中的应用

### 1. 异步 API 调用

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_response(prompt: str):
    """异步调用 LLM API"""
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def main():
    # 并发调用多个 API
    responses = await asyncio.gather(
        generate_response("什么是 Python?"),
        generate_response("什么是 asyncio?"),
        generate_response("什么是协程?")
    )
    for response in responses:
        print(response)

asyncio.run(main())
```

### 2. 异步数据库查询

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine("postgresql+asyncpg://...")
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)

async def get_user(user_id: int):
    """异步查询用户"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

async def main():
    user = await get_user(1)
    print(user)

asyncio.run(main())
```

### 3. 流式响应

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_stream(prompt: str):
    """异步生成器：流式输出"""
    async for chunk in client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

@app.get("/stream")
async def stream_response(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/plain"
    )
```

---

## 常见陷阱

### 陷阱1：忘记 await

```python
# ❌ 错误：忘记 await
async def bad():
    result = fetch_data()  # 返回协程对象，不执行
    print(result)  # <coroutine object fetch_data at 0x...>

# ✅ 正确：使用 await
async def good():
    result = await fetch_data()  # 执行并获取返回值
    print(result)  # 实际数据
```

### 陷阱2：在同步函数中使用 await

```python
# ❌ 错误：在同步函数中使用 await
def bad():
    result = await fetch_data()  # SyntaxError

# ✅ 正确：在协程函数中使用 await
async def good():
    result = await fetch_data()
```

### 陷阱3：混用同步和异步 I/O

```python
import asyncio
import requests  # 同步 HTTP 库

# ❌ 错误：在异步函数中使用同步 I/O
async def bad():
    response = requests.get(url)  # 阻塞事件循环！
    return response.json()

# ✅ 正确：使用异步 HTTP 库
import httpx

async def good():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### 陷阱4：协程未被 await 导致警告

```python
import asyncio

async def fetch_data():
    return "data"

async def main():
    # ❌ 错误：创建协程但不 await
    fetch_data()  # RuntimeWarning: coroutine 'fetch_data' was never awaited

    # ✅ 正确：await 协程
    result = await fetch_data()

asyncio.run(main())
```

---

## 学习检查

完成本节后，你应该能够：

- [ ] 理解协程的定义和特性
- [ ] 使用 `async def` 定义协程函数
- [ ] 理解 `await` 的工作原理
- [ ] 知道协程的三种状态
- [ ] 使用异步上下文管理器
- [ ] 使用异步生成器
- [ ] 避免常见的协程陷阱
- [ ] 在 AI Agent 开发中正确使用协程

---

## 下一步

- **继续学习**：阅读【核心概念3：并发执行】
- **实战练习**：完成【实战代码】中的协程示例
- **深入理解**：阅读【化骨绵掌】了解更多细节
