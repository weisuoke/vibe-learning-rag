# 核心概念1：StreamingResponse 基础机制

> 理解 FastAPI 流式响应的 API 使用、生命周期和 HTTP 协议基础

---

## 1. StreamingResponse 是什么？

**StreamingResponse** 是 FastAPI 提供的响应类，用于发送流式数据。

```python
from fastapi.responses import StreamingResponse

# 基础用法
@app.get("/stream")
async def stream_endpoint():
    async def generate():
        yield "数据1"
        yield "数据2"
        yield "数据3"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
```

**核心特点：**
- 接收一个生成器（generator）或异步生成器（async generator）
- 逐步发送数据，不需要缓存完整响应
- 使用 HTTP Chunked Transfer Encoding

---

## 2. StreamingResponse 的参数

### 2.1 必需参数：content

**content**：生成器函数或异步生成器函数

```python
# 同步生成器
def sync_generator():
    yield "数据1"
    yield "数据2"

# 异步生成器（推荐）
async def async_generator():
    yield "数据1"
    await asyncio.sleep(0.1)
    yield "数据2"

# 使用
@app.get("/sync")
def sync_endpoint():
    return StreamingResponse(sync_generator())

@app.get("/async")
async def async_endpoint():
    return StreamingResponse(async_generator())
```

**推荐使用异步生成器：**
- ✅ 支持异步操作（数据库查询、HTTP 请求）
- ✅ 不阻塞事件循环
- ✅ 更好的并发性能

### 2.2 可选参数：media_type

**media_type**：响应的 MIME 类型

```python
# 纯文本
StreamingResponse(generate(), media_type="text/plain")

# JSON 流
StreamingResponse(generate(), media_type="application/json")

# Server-Sent Events
StreamingResponse(generate(), media_type="text/event-stream")

# 二进制数据
StreamingResponse(generate(), media_type="application/octet-stream")

# 视频流
StreamingResponse(generate(), media_type="video/mp4")
```

**常用 MIME 类型：**
| 类型 | 用途 |
|------|------|
| `text/plain` | 纯文本流 |
| `text/event-stream` | Server-Sent Events (SSE) |
| `application/json` | JSON 流 |
| `application/octet-stream` | 二进制数据流 |
| `video/mp4` | 视频流 |
| `audio/mpeg` | 音频流 |

### 2.3 可选参数：status_code

**status_code**：HTTP 状态码（默认 200）

```python
# 成功响应（默认）
StreamingResponse(generate(), status_code=200)

# 部分内容
StreamingResponse(generate(), status_code=206)

# 自定义状态码
StreamingResponse(generate(), status_code=201)
```

### 2.4 可选参数：headers

**headers**：自定义响应头

```python
StreamingResponse(
    generate(),
    headers={
        "X-Custom-Header": "value",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
)
```

**常用响应头：**
```python
# SSE 专用响应头
headers = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
}

# 文件下载
headers = {
    "Content-Disposition": "attachment; filename=file.zip"
}
```

---

## 3. StreamingResponse 的生命周期

### 3.1 完整生命周期

```python
async def stream_with_lifecycle():
    print("1. 生成器创建")

    try:
        print("2. 开始生成数据")
        yield "数据1"
        print("3. 数据1已发送")

        yield "数据2"
        print("4. 数据2已发送")

        yield "数据3"
        print("5. 数据3已发送")

        print("6. 正常结束")
    except GeneratorExit:
        print("7. 客户端提前断开连接")
    finally:
        print("8. 清理资源")

@app.get("/lifecycle")
async def lifecycle_endpoint():
    return StreamingResponse(stream_with_lifecycle())
```

**执行流程：**
```
客户端请求 → FastAPI 调用路由函数 → 创建 StreamingResponse
    ↓
开始生成数据（第一个 yield）
    ↓
发送数据块到客户端
    ↓
继续生成数据（第二个 yield）
    ↓
发送数据块到客户端
    ↓
...
    ↓
生成器结束或客户端断开
    ↓
执行 finally 清理资源
```

### 3.2 客户端断开连接的处理

```python
async def handle_disconnect():
    try:
        for i in range(100):
            yield f"数据 {i}\n"
            await asyncio.sleep(0.1)
    except GeneratorExit:
        # 客户端断开连接时触发
        print("客户端断开连接，停止生成数据")
        # 清理资源
        await cleanup_resources()
```

**何时触发 GeneratorExit？**
- 客户端关闭连接
- 客户端刷新页面
- 网络中断
- 客户端超时

### 3.3 资源清理最佳实践

```python
async def stream_with_cleanup():
    # 获取资源
    db = await get_db_connection()
    file = open("data.txt", "r")

    try:
        # 生成数据
        async for row in db.query("SELECT * FROM data"):
            yield f"{row}\n"

        for line in file:
            yield line
    finally:
        # 确保资源释放（无论是否正常结束）
        await db.close()
        file.close()
        print("资源已清理")
```

---

## 4. HTTP 协议层面的实现

### 4.1 Chunked Transfer Encoding

**传统 HTTP 响应：**
```http
HTTP/1.1 200 OK
Content-Length: 100
Content-Type: text/plain

[100字节的完整数据]
```

**流式 HTTP 响应：**
```http
HTTP/1.1 200 OK
Transfer-Encoding: chunked
Content-Type: text/plain

A\r\n
[10字节数据]\r\n
14\r\n
[20字节数据]\r\n
0\r\n
\r\n
```

**Chunked 编码格式：**
```
[块大小（十六进制）]\r\n
[块数据]\r\n
[块大小]\r\n
[块数据]\r\n
...
0\r\n
\r\n
```

### 4.2 FastAPI 自动处理 Chunked 编码

```python
async def generate():
    yield "Hello"  # FastAPI 自动编码为 chunk
    yield " "
    yield "World"

# FastAPI 自动生成的 HTTP 响应：
# 5\r\n
# Hello\r\n
# 1\r\n
#  \r\n
# 5\r\n
# World\r\n
# 0\r\n
# \r\n
```

**你不需要手动处理 Chunked 编码，FastAPI 会自动处理！**

---

## 5. 同步生成器 vs 异步生成器

### 5.1 同步生成器

```python
def sync_generator():
    """同步生成器：不能执行异步操作"""
    for i in range(10):
        # ❌ 不能使用 await
        # await asyncio.sleep(0.1)  # 语法错误

        # ✅ 只能使用同步操作
        time.sleep(0.1)
        yield f"数据 {i}"

@app.get("/sync")
def sync_endpoint():
    return StreamingResponse(sync_generator())
```

**适用场景：**
- 读取本地文件
- 处理内存中的数据
- 不需要异步操作的场景

**缺点：**
- 阻塞事件循环
- 影响并发性能
- 不能调用异步 API

### 5.2 异步生成器（推荐）

```python
async def async_generator():
    """异步生成器：可以执行异步操作"""
    for i in range(10):
        # ✅ 可以使用 await
        await asyncio.sleep(0.1)

        # ✅ 可以调用异步 API
        data = await fetch_data_from_db(i)

        yield f"数据 {i}: {data}"

@app.get("/async")
async def async_endpoint():
    return StreamingResponse(async_generator())
```

**适用场景：**
- 调用异步 API（数据库、HTTP 请求）
- 需要异步等待的场景
- 高并发场景

**优点：**
- 不阻塞事件循环
- 支持异步操作
- 更好的并发性能

### 5.3 性能对比

```python
import time
import asyncio

# 同步生成器：阻塞事件循环
def sync_generator():
    for i in range(5):
        time.sleep(1)  # 阻塞1秒
        yield f"数据 {i}"
# 总耗时：5秒，期间无法处理其他请求

# 异步生成器：不阻塞事件循环
async def async_generator():
    for i in range(5):
        await asyncio.sleep(1)  # 异步等待1秒
        yield f"数据 {i}"
# 总耗时：5秒，但期间可以处理其他请求
```

**并发测试：**
```python
# 同步生成器：10个并发请求需要 50秒（串行）
# 异步生成器：10个并发请求需要 5秒（并行）
```

---

## 6. 实战示例：完整的流式响应

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator

app = FastAPI()

async def generate_data(count: int) -> AsyncGenerator[str, None]:
    """
    异步生成器：生成指定数量的数据

    Args:
        count: 数据数量

    Yields:
        str: 数据字符串
    """
    # 验证参数
    if count <= 0:
        raise ValueError("count 必须大于 0")

    try:
        for i in range(count):
            # 模拟异步操作（如数据库查询）
            await asyncio.sleep(0.1)

            # 生成数据
            data = f"数据 {i + 1}/{count}\n"
            yield data

            # 日志记录
            print(f"已发送: {data.strip()}")

    except GeneratorExit:
        # 客户端断开连接
        print(f"客户端断开连接，已发送 {i + 1}/{count} 条数据")
    finally:
        # 清理资源
        print("生成器结束，资源已清理")

@app.get("/stream/{count}")
async def stream_endpoint(count: int):
    """
    流式响应端点

    Args:
        count: 要生成的数据数量

    Returns:
        StreamingResponse: 流式响应
    """
    # 参数验证
    if count <= 0 or count > 1000:
        raise HTTPException(
            status_code=400,
            detail="count 必须在 1-1000 之间"
        )

    # 返回流式响应
    return StreamingResponse(
        generate_data(count),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff"
        }
    )

# 测试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**测试命令：**
```bash
# 使用 curl 测试
curl http://localhost:8000/stream/10

# 输出：
# 数据 1/10
# 数据 2/10
# ...
# 数据 10/10
```

---

## 7. 常见问题

### Q1: 何时使用同步生成器，何时使用异步生成器？

**使用同步生成器：**
- 只读取本地文件
- 处理内存中的数据
- 不需要任何异步操作

**使用异步生成器：**
- 调用数据库
- 调用 HTTP API
- 需要异步等待
- 高并发场景

**推荐：** 默认使用异步生成器

### Q2: StreamingResponse 会自动关闭生成器吗？

**会！** FastAPI 会自动处理生成器的生命周期：
- 正常结束时自动关闭
- 客户端断开时触发 GeneratorExit
- finally 块总是会执行

### Q3: 如何在流式响应中处理错误？

```python
async def generate_with_error_handling():
    try:
        for i in range(10):
            if i == 5:
                raise ValueError("模拟错误")
            yield f"数据 {i}\n"
    except ValueError as e:
        # 发送错误信息
        yield f"错误: {e}\n"
    finally:
        yield "流结束\n"
```

**注意：** 一旦开始发送数据，就无法修改 HTTP 状态码！

### Q4: 流式响应支持 CORS 吗？

**支持！** 使用 FastAPI 的 CORS 中间件：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 总结

**StreamingResponse 的核心要点：**

1. **API 使用**：接收生成器，指定 media_type
2. **生命周期**：创建 → 生成 → 发送 → 清理
3. **HTTP 协议**：使用 Chunked Transfer Encoding
4. **异步优先**：推荐使用异步生成器
5. **资源管理**：使用 try-finally 确保清理

**记住：** StreamingResponse 是 FastAPI 流式响应的基础，掌握它就掌握了流式响应的核心！
