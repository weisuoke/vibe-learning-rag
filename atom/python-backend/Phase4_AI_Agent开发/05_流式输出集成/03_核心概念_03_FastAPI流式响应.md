# 核心概念03: FastAPI流式响应

> 深入理解 StreamingResponse 的原理和使用

---

## 概述

**StreamingResponse** 是 FastAPI 提供的流式响应类,用于将异步生成器的输出转换为 HTTP 流式响应。它是连接 Python 后端和前端的桥梁。

**本节目标:**
- 理解 StreamingResponse 的工作原理
- 掌握 StreamingResponse 的配置选项
- 学习如何处理流式响应中的错误
- 实现生产级的流式端点

---

## 1. StreamingResponse 基础

### 1.1 什么是 StreamingResponse?

**StreamingResponse** 是 FastAPI 的响应类,专门用于流式传输数据。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream_endpoint():
    """基础流式端点"""
    async def generate():
        for i in range(5):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**核心特点:**
1. 接受异步生成器或同步生成器
2. 自动处理流式传输
3. 支持自定义响应头
4. 支持 SSE 格式

### 1.2 StreamingResponse 的参数

```python
StreamingResponse(
    content,              # 生成器(必需)
    status_code=200,      # HTTP 状态码
    headers=None,         # 自定义响应头
    media_type=None,      # Content-Type
    background=None       # 后台任务
)
```

**参数详解:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.background import BackgroundTask

app = FastAPI()

@app.get("/stream-full")
async def stream_full():
    """完整参数示例"""
    async def generate():
        for i in range(5):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.5)

    def cleanup():
        """后台清理任务"""
        print("清理资源")

    return StreamingResponse(
        content=generate(),
        status_code=200,
        headers={
            "Cache-Control": "no-cache",
            "X-Custom-Header": "value"
        },
        media_type="text/event-stream",
        background=BackgroundTask(cleanup)
    )
```

---

## 2. StreamingResponse vs 普通响应

### 2.1 普通响应

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/normal")
async def normal_response():
    """普通响应"""
    # 等待所有数据生成完成
    data = []
    for i in range(5):
        await asyncio.sleep(0.5)
        data.append(i)

    # 一次性返回
    return {"data": data}
```

**特点:**
- 等待所有数据生成完成
- 一次性返回 JSON
- 客户端等待时间长

### 2.2 流式响应

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_response():
    """流式响应"""
    async def generate():
        for i in range(5):
            await asyncio.sleep(0.5)
            # 边生成边发送
            yield f"data: {i}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**特点:**
- 边生成边发送
- 客户端立即收到第一条数据
- 用户体验好

### 2.3 对比表格

| 特性 | 普通响应 | 流式响应 |
|------|----------|----------|
| 数据发送 | 一次性 | 分批 |
| 客户端等待 | 等待完整响应 | 立即收到数据 |
| 内存占用 | 需要缓存完整数据 | 边生成边发送 |
| 适用场景 | 短数据、结构化数据 | 长数据、AI生成 |
| 错误处理 | 简单(返回错误状态码) | 复杂(需要发送错误事件) |

---

## 3. StreamingResponse 的工作原理

### 3.1 内部实现

```python
"""
StreamingResponse 的简化实现
"""

class StreamingResponse:
    def __init__(self, content, media_type=None):
        self.content = content
        self.media_type = media_type

    async def __call__(self, scope, receive, send):
        """ASGI 应用接口"""
        # 1. 发送响应头
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [
                [b'content-type', self.media_type.encode()],
            ],
        })

        # 2. 流式发送响应体
        async for chunk in self.content:
            await send({
                'type': 'http.response.body',
                'body': chunk.encode(),
                'more_body': True,  # 还有更多数据
            })

        # 3. 发送结束信号
        await send({
            'type': 'http.response.body',
            'body': b'',
            'more_body': False,  # 没有更多数据
        })
```

**关键点:**
1. 先发送响应头(status + headers)
2. 逐块发送响应体(body)
3. 发送结束信号(more_body=False)

### 3.2 与 ASGI 的关系

```python
"""
ASGI 应用的流式响应
"""

async def asgi_app(scope, receive, send):
    """ASGI 应用"""
    if scope['type'] == 'http':
        # 发送响应头
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [[b'content-type', b'text/event-stream']],
        })

        # 流式发送数据
        for i in range(5):
            await asyncio.sleep(0.5)
            await send({
                'type': 'http.response.body',
                'body': f"data: {i}\n\n".encode(),
                'more_body': True,
            })

        # 结束响应
        await send({
            'type': 'http.response.body',
            'body': b'',
            'more_body': False,
        })
```

**StreamingResponse 简化了 ASGI 的使用:**
- 不需要手动处理 ASGI 协议
- 自动处理响应头和响应体
- 自动处理结束信号

---

## 4. StreamingResponse 的高级用法

### 4.1 自定义响应头

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream-headers")
async def stream_with_headers():
    """自定义响应头"""
    async def generate():
        for i in range(5):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            "Access-Control-Allow-Origin": "*",  # CORS
        }
    )
```

**常用响应头:**
- `Cache-Control: no-cache` - 禁止缓存
- `Connection: keep-alive` - 保持连接
- `X-Accel-Buffering: no` - 禁用 Nginx 缓冲
- `Access-Control-Allow-Origin: *` - 允许跨域

### 4.2 后台任务

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.background import BackgroundTask

app = FastAPI()

@app.get("/stream-background")
async def stream_with_background():
    """带后台任务的流式响应"""
    async def generate():
        for i in range(5):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.5)

    def cleanup():
        """流式响应结束后执行"""
        print("清理资源")
        # 关闭数据库连接
        # 删除临时文件
        # 记录日志

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        background=BackgroundTask(cleanup)
    )
```

### 4.3 条件流式响应

```python
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

@app.get("/conditional-stream")
async def conditional_stream(stream: bool = Query(False)):
    """根据参数决定是否流式响应"""
    data = []
    for i in range(5):
        await asyncio.sleep(0.5)
        data.append(i)

    if stream:
        # 流式响应
        async def generate():
            for item in data:
                yield f"data: {item}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    else:
        # 普通响应
        return JSONResponse({"data": data})
```

---

## 5. 错误处理

### 5.1 基础错误处理

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.get("/stream-error")
async def stream_with_error():
    """带错误处理的流式响应"""
    async def generate():
        try:
            for i in range(10):
                # 模拟错误
                if i == 5:
                    raise Exception("模拟错误")

                yield f"data: {i}\n\n"
                await asyncio.sleep(0.5)

        except Exception as e:
            # 发送错误事件
            error_data = {
                "error": str(e),
                "code": "GENERATION_ERROR"
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.2 完整的错误处理

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import traceback

app = FastAPI()

@app.get("/stream-robust")
async def stream_robust():
    """生产级错误处理"""
    async def generate():
        try:
            # 发送初始化事件
            yield f"event: init\ndata: {json.dumps({'status': 'started'})}\n\n"

            # 流式生成数据
            for i in range(10):
                # 模拟可能出错的操作
                if i == 5:
                    raise ValueError("数据验证失败")

                data = {"id": i, "message": f"Message {i}"}
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.5)

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except ValueError as e:
            # 业务错误
            error_data = {
                "error": str(e),
                "code": "VALIDATION_ERROR",
                "recoverable": True
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        except Exception as e:
            # 系统错误
            error_data = {
                "error": str(e),
                "code": "SYSTEM_ERROR",
                "recoverable": False,
                "traceback": traceback.format_exc()
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        finally:
            # 清理资源
            print("清理资源")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.3 超时处理

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/stream-timeout")
async def stream_with_timeout():
    """带超时处理的流式响应"""
    async def generate():
        try:
            for i in range(10):
                # 设置超时
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(0.5),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: Timeout\n\n"
                    break

                yield f"data: {i}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 6. 性能优化

### 6.1 缓冲区优化

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream-buffered")
async def stream_buffered():
    """带缓冲区的流式响应"""
    async def generate():
        buffer = []
        buffer_size = 10

        for i in range(100):
            buffer.append(f"Message {i}")

            # 缓冲区满了,批量发送
            if len(buffer) >= buffer_size:
                yield f"data: {json.dumps(buffer)}\n\n"
                buffer.clear()

            await asyncio.sleep(0.01)

        # 发送剩余数据
        if buffer:
            yield f"data: {json.dumps(buffer)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 6.2 心跳机制

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

@app.get("/stream-heartbeat")
async def stream_heartbeat():
    """带心跳的流式响应"""
    async def generate():
        last_heartbeat = time.time()
        heartbeat_interval = 30  # 30秒

        for i in range(100):
            # 发送数据
            yield f"data: {i}\n\n"
            await asyncio.sleep(1)

            # 检查是否需要发送心跳
            if time.time() - last_heartbeat > heartbeat_interval:
                yield ": heartbeat\n\n"
                last_heartbeat = time.time()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 6.3 压缩

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import gzip

app = FastAPI()

@app.get("/stream-compressed")
async def stream_compressed():
    """压缩的流式响应"""
    async def generate():
        for i in range(10):
            data = f"data: {'x' * 1000}\n\n"  # 大量数据
            # 压缩数据
            compressed = gzip.compress(data.encode())
            yield compressed
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Content-Encoding": "gzip"
        }
    )
```

---

## 7. 实战示例

### 示例1: LangChain 集成

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI()

@app.post("/chat-stream")
async def chat_stream(message: str):
    """LangChain 流式聊天"""
    async def generate():
        try:
            async for chunk in llm.astream(message):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 示例2: 进度条

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.get("/progress")
async def progress_stream():
    """进度条流式更新"""
    async def generate():
        total = 100

        for i in range(total + 1):
            await asyncio.sleep(0.1)

            progress = {
                "current": i,
                "total": total,
                "percentage": i / total * 100,
                "message": f"Processing {i}/{total}"
            }
            yield f"data: {json.dumps(progress)}\n\n"

        yield f"event: done\ndata: Completed\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 示例3: 日志流式显示

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/logs")
async def logs_stream():
    """日志流式显示"""
    async def generate():
        # 模拟读取日志文件
        log_file = "/var/log/app.log"

        try:
            with open(log_file) as f:
                # 跳到文件末尾
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {line}\n\n"
                    else:
                        await asyncio.sleep(0.1)

        except FileNotFoundError:
            yield f"event: error\ndata: Log file not found\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 8. 常见问题

### 问题1: 响应被缓冲

**问题:** Nginx 等代理服务器缓冲响应,导致延迟

**解决方案:**

```python
return StreamingResponse(
    generate(),
    media_type="text/event-stream",
    headers={
        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        "Cache-Control": "no-cache"
    }
)
```

```nginx
# Nginx 配置
location /stream {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_cache off;
}
```

### 问题2: 连接被关闭

**问题:** 客户端或代理服务器关闭空闲连接

**解决方案:** 添加心跳

```python
async def generate():
    last_heartbeat = time.time()

    for i in range(100):
        yield f"data: {i}\n\n"

        # 每30秒发送心跳
        if time.time() - last_heartbeat > 30:
            yield ": heartbeat\n\n"
            last_heartbeat = time.time()
```

### 问题3: 内存泄漏

**问题:** 生成器没有正确清理资源

**解决方案:** 使用 try-finally

```python
async def generate():
    resource = acquire_resource()
    try:
        for i in range(100):
            yield f"data: {i}\n\n"
    finally:
        release_resource(resource)
```

---

## 总结

**StreamingResponse 的核心要点:**

1. **基础用法**: 接受异步生成器,返回流式响应
2. **响应头**: 设置 `text/event-stream` 和其他必要头
3. **错误处理**: 使用 SSE 的 `event: error` 机制
4. **性能优化**: 缓冲区、心跳、压缩

**与普通响应的区别:**
- 普通响应: 一次性返回,简单但等待时间长
- 流式响应: 分批返回,复杂但用户体验好

**在流式输出中的作用:**
- 连接异步生成器和 HTTP 响应
- 自动处理 ASGI 协议
- 简化流式输出的实现

**下一步:**

理解了 StreamingResponse 后,可以学习:
- LangChain 流式 API 的集成
- 流式输出粒度控制
- 生产环境的错误处理和优化

---

**记住:** StreamingResponse 是 FastAPI 流式输出的核心,掌握它是实现流式输出的关键。
