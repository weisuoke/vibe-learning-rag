# 核心概念02: Python异步生成器

> 深入理解 async def + yield 的异步生成器机制

---

## 概述

**异步生成器** 是 Python 3.6+ 引入的特性,结合了异步编程和生成器的优势,是实现流式输出的核心技术。

**本节目标:**
- 理解异步生成器的语法和原理
- 掌握异步迭代器协议
- 对比同步生成器和异步生成器
- 手写异步生成器实现流式输出

---

## 1. 生成器基础回顾

### 1.1 普通函数 vs 生成器

```python
# 普通函数: 一次性返回所有数据
def get_numbers():
    result = []
    for i in range(5):
        result.append(i)
    return result  # 一次性返回 [0, 1, 2, 3, 4]

# 使用
numbers = get_numbers()
print(numbers)  # [0, 1, 2, 3, 4]
```

```python
# 生成器: 逐个返回数据
def generate_numbers():
    for i in range(5):
        yield i  # 逐个返回

# 使用
for num in generate_numbers():
    print(num)  # 0, 1, 2, 3, 4 (逐个输出)
```

**核心区别:**
- 普通函数: `return` 一次性返回
- 生成器: `yield` 逐个返回

### 1.2 生成器的优势

```python
"""
内存占用对比
"""

import sys

# 普通函数: 占用大量内存
def get_large_list():
    return [i for i in range(1000000)]  # 生成100万个数字

# 生成器: 占用极少内存
def generate_large_list():
    for i in range(1000000):
        yield i

# 内存占用对比
list_data = get_large_list()
print(f"列表内存: {sys.getsizeof(list_data)} bytes")  # ~8MB

gen_data = generate_large_list()
print(f"生成器内存: {sys.getsizeof(gen_data)} bytes")  # ~128 bytes
```

**生成器的三大优势:**
1. **内存优化**: 不需要一次性存储所有数据
2. **惰性求值**: 只在需要时生成数据
3. **无限序列**: 可以表示无限长的数据序列

---

## 2. 异步生成器详解

### 2.1 什么是异步生成器?

**异步生成器 = `async def` + `yield`**

```python
import asyncio

# 同步生成器
def sync_generate():
    for i in range(5):
        yield i

# 异步生成器
async def async_generate():
    for i in range(5):
        yield i
```

**核心区别:**
- 同步生成器: 阻塞式,一个生成器运行时其他任务等待
- 异步生成器: 非阻塞式,可以同时运行多个生成器

### 2.2 异步生成器的语法

```python
"""
异步生成器的完整语法
"""

import asyncio

async def async_generator():
    """异步生成器函数"""
    for i in range(5):
        # 可以使用 await 执行异步操作
        await asyncio.sleep(0.5)
        # 使用 yield 返回数据
        yield i

# 使用异步生成器
async def main():
    # 使用 async for 迭代
    async for num in async_generator():
        print(num)

# 运行
asyncio.run(main())
```

**关键点:**
1. 使用 `async def` 定义异步函数
2. 使用 `yield` 返回数据
3. 可以在函数内使用 `await`
4. 使用 `async for` 迭代

### 2.3 异步生成器 vs 同步生成器

```python
"""
同步生成器 vs 异步生成器对比
"""

import asyncio
import time

# 同步生成器
def sync_generate():
    for i in range(3):
        time.sleep(1)  # 阻塞1秒
        yield i

# 异步生成器
async def async_generate():
    for i in range(3):
        await asyncio.sleep(1)  # 非阻塞等待1秒
        yield i

# 测试同步生成器
def test_sync():
    start = time.time()
    for num in sync_generate():
        print(f"同步: {num}")
    print(f"同步总时间: {time.time() - start:.2f}s")

# 测试异步生成器
async def test_async():
    start = time.time()
    async for num in async_generate():
        print(f"异步: {num}")
    print(f"异步总时间: {time.time() - start:.2f}s")

# 运行
test_sync()  # 输出: 同步总时间: 3.00s
asyncio.run(test_async())  # 输出: 异步总时间: 3.00s
```

**单个生成器的性能相同,但并发场景下异步生成器更优:**

```python
"""
并发场景对比
"""

import asyncio
import time

async def async_generate(name):
    for i in range(3):
        await asyncio.sleep(1)
        yield f"{name}: {i}"

# 并发运行多个异步生成器
async def test_concurrent():
    start = time.time()

    # 同时运行3个异步生成器
    tasks = []
    for name in ['A', 'B', 'C']:
        async def consume(name):
            async for item in async_generate(name):
                print(item)
        tasks.append(consume(name))

    await asyncio.gather(*tasks)
    print(f"并发总时间: {time.time() - start:.2f}s")

asyncio.run(test_concurrent())
# 输出: 并发总时间: 3.00s (而不是 9.00s)
```

---

## 3. 异步迭代器协议

### 3.1 迭代器协议

**同步迭代器协议:**

```python
class SyncIterator:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            self.current += 1
            return self.current
        raise StopIteration

# 使用
for num in SyncIterator(5):
    print(num)  # 1, 2, 3, 4, 5
```

**异步迭代器协议:**

```python
class AsyncIterator:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current < self.n:
            await asyncio.sleep(0.5)
            self.current += 1
            return self.current
        raise StopAsyncIteration

# 使用
async def main():
    async for num in AsyncIterator(5):
        print(num)  # 1, 2, 3, 4, 5

asyncio.run(main())
```

**对比:**

| 协议 | 同步迭代器 | 异步迭代器 |
|------|-----------|-----------|
| 迭代方法 | `__iter__()` | `__aiter__()` |
| 下一个元素 | `__next__()` | `__anext__()` |
| 停止迭代 | `StopIteration` | `StopAsyncIteration` |
| 使用方式 | `for x in iter` | `async for x in iter` |

### 3.2 异步生成器的内部实现

```python
"""
异步生成器的等价实现
"""

import asyncio

# 异步生成器 (简洁)
async def async_gen():
    for i in range(3):
        await asyncio.sleep(0.5)
        yield i

# 等价的异步迭代器实现 (详细)
class AsyncGenEquivalent:
    def __init__(self):
        self.current = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current < 3:
            await asyncio.sleep(0.5)
            value = self.current
            self.current += 1
            return value
        raise StopAsyncIteration

# 两者等价
async def main():
    # 使用异步生成器
    async for num in async_gen():
        print(num)

    # 使用异步迭代器
    async for num in AsyncGenEquivalent():
        print(num)

asyncio.run(main())
```

---

## 4. 异步生成器的高级用法

### 4.1 异步生成器表达式

```python
"""
异步生成器表达式
"""

import asyncio

# 同步生成器表达式
sync_gen = (i for i in range(5))

# 异步生成器表达式
async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async def main():
    # 使用异步生成器表达式
    async_gen = (i * 2 async for i in async_range(5))

    async for num in async_gen():
        print(num)  # 0, 2, 4, 6, 8

asyncio.run(main())
```

### 4.2 异步生成器的 send() 和 throw()

```python
"""
异步生成器的双向通信
"""

import asyncio

async def async_gen_with_send():
    value = None
    for i in range(5):
        # 接收外部发送的值
        value = yield i
        if value is not None:
            print(f"收到: {value}")

async def main():
    gen = async_gen_with_send()

    # 启动生成器
    print(await gen.asend(None))  # 0

    # 发送值给生成器
    print(await gen.asend("Hello"))  # 收到: Hello, 输出: 1
    print(await gen.asend("World"))  # 收到: World, 输出: 2

asyncio.run(main())
```

### 4.3 异步生成器的清理

```python
"""
异步生成器的资源清理
"""

import asyncio

async def async_gen_with_cleanup():
    try:
        for i in range(5):
            await asyncio.sleep(0.5)
            yield i
    finally:
        # 清理资源
        print("清理资源")

async def main():
    gen = async_gen_with_cleanup()

    # 只迭代2次就停止
    async for i, num in enumerate(gen):
        print(num)
        if i == 1:
            break  # 触发 finally 块

asyncio.run(main())
# 输出: 0, 1, 清理资源
```

---

## 5. 异步生成器在流式输出中的应用

### 5.1 基础流式输出

```python
"""
使用异步生成器实现流式输出
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream_endpoint():
    """流式端点"""
    async def generate():
        for i in range(10):
            # 模拟数据生成
            await asyncio.sleep(0.5)
            # SSE 格式
            yield f"data: Message {i}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.2 LangChain 流式输出

```python
"""
LangChain 的 astream() 返回异步生成器
"""

from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
llm = ChatOpenAI()

@app.post("/chat-stream")
async def chat_stream(message: str):
    """LangChain 流式聊天"""
    async def generate():
        # llm.astream() 返回异步生成器
        async for chunk in llm.astream(message):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.3 复杂流式输出

```python
"""
复杂的流式输出场景
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

@app.get("/complex-stream")
async def complex_stream():
    """复杂流式输出"""
    async def generate():
        try:
            # 1. 发送初始化事件
            yield f"event: init\ndata: {json.dumps({'status': 'started'})}\n\n"

            # 2. 流式发送数据
            for i in range(10):
                # 模拟异步操作
                await asyncio.sleep(0.5)

                # 发送进度
                progress = {
                    "current": i + 1,
                    "total": 10,
                    "percentage": (i + 1) / 10 * 100
                }
                yield f"event: progress\ndata: {json.dumps(progress)}\n\n"

                # 发送数据
                data = {"id": i, "message": f"Message {i}"}
                yield f"data: {json.dumps(data)}\n\n"

            # 3. 发送完成事件
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            # 4. 发送错误事件
            error = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(error)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 6. 异步生成器的性能优化

### 6.1 缓冲区优化

```python
"""
使用缓冲区减少网络开销
"""

import asyncio

async def buffered_generate(buffer_size=10):
    """带缓冲区的异步生成器"""
    buffer = []

    for i in range(100):
        await asyncio.sleep(0.01)
        buffer.append(f"Message {i}")

        # 缓冲区满了,批量发送
        if len(buffer) >= buffer_size:
            yield "\n".join(buffer)
            buffer.clear()

    # 发送剩余数据
    if buffer:
        yield "\n".join(buffer)
```

### 6.2 背压处理

```python
"""
处理背压问题
"""

import asyncio
from asyncio import Queue

async def generate_with_backpressure():
    """带背压处理的异步生成器"""
    queue = Queue(maxsize=10)  # 限制队列大小

    async def producer():
        """生产者"""
        for i in range(100):
            await queue.put(f"Message {i}")
            await asyncio.sleep(0.01)
        await queue.put(None)  # 结束信号

    async def consumer():
        """消费者"""
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    # 启动生产者
    asyncio.create_task(producer())

    # 返回消费者
    async for item in consumer():
        yield item
```

---

## 7. 常见问题和陷阱

### 问题1: 忘记使用 async for

```python
# ❌ 错误: 使用普通 for
async def wrong():
    async def gen():
        yield 1
        yield 2

    for item in gen():  # TypeError
        print(item)

# ✅ 正确: 使用 async for
async def correct():
    async def gen():
        yield 1
        yield 2

    async for item in gen():
        print(item)
```

### 问题2: 在同步函数中使用异步生成器

```python
# ❌ 错误: 在同步函数中使用
def wrong():
    async def gen():
        yield 1

    async for item in gen():  # SyntaxError
        print(item)

# ✅ 正确: 在异步函数中使用
async def correct():
    async def gen():
        yield 1

    async for item in gen():
        print(item)
```

### 问题3: 忘记 await 异步操作

```python
# ❌ 错误: 忘记 await
async def wrong():
    async def gen():
        asyncio.sleep(1)  # 忘记 await
        yield 1

# ✅ 正确: 使用 await
async def correct():
    async def gen():
        await asyncio.sleep(1)
        yield 1
```

---

## 8. 实战示例

### 示例1: 文件流式读取

```python
"""
异步流式读取大文件
"""

import aiofiles

async def read_file_stream(file_path):
    """异步流式读取文件"""
    async with aiofiles.open(file_path, 'r') as f:
        async for line in f:
            yield line.strip()

# 使用
async def main():
    async for line in read_file_stream('large_file.txt'):
        print(line)
```

### 示例2: 数据库流式查询

```python
"""
异步流式查询数据库
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def query_stream(session: AsyncSession, model, batch_size=100):
    """异步流式查询"""
    offset = 0

    while True:
        # 分批查询
        stmt = select(model).offset(offset).limit(batch_size)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        if not rows:
            break

        # 逐行返回
        for row in rows:
            yield row

        offset += batch_size
```

### 示例3: API 流式调用

```python
"""
异步流式调用外部 API
"""

import aiohttp

async def api_stream(url, params):
    """异步流式调用 API"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            async for line in response.content:
                yield line.decode('utf-8')
```

---

## 总结

**异步生成器的核心要点:**

1. **语法**: `async def` + `yield`
2. **使用**: `async for` 迭代
3. **优势**: 非阻塞,支持并发
4. **应用**: 流式输出,大文件处理,数据库查询

**与同步生成器的区别:**
- 同步生成器: 阻塞式,适合 CPU 密集型
- 异步生成器: 非阻塞式,适合 I/O 密集型

**在流式输出中的作用:**
- FastAPI StreamingResponse 接受异步生成器
- LangChain astream() 返回异步生成器
- 实现边生成边发送的流式输出

**下一步:**

理解了异步生成器后,可以学习:
- FastAPI StreamingResponse 的使用
- LangChain 流式 API 的集成
- 生产环境的错误处理和优化

---

**记住:** 异步生成器是流式输出的核心技术,掌握它是实现高性能流式输出的关键。
