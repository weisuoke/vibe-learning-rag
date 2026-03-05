# 实战代码：FastAPI流式回调集成

## 场景说明

在生产环境中，我们经常需要通过 API 提供 LLM 服务，并实现流式响应以提升用户体验。流式响应可以让用户实时看到 LLM 的生成过程，而不是等待完整响应后才显示。

**核心挑战：**
- 如何在异步 FastAPI 中实现流式响应
- 如何使用回调处理器捕获 LLM 的每个 token
- 如何通过队列管理异步数据流
- 如何处理 Agent 的中间步骤（只返回最终答案）

**在 RAG 中的应用场景：**
- RAG 问答系统的流式响应
- 智能客服的实时对话
- 文档问答的渐进式展示
- 多轮对话的流畅体验

---

## 核心概念

### 1. AsyncIteratorCallbackHandler

LangChain 提供的异步迭代器回调处理器，专门用于流式输出。

**核心机制：**
- 使用 `asyncio.Queue` 管理 token 流
- 实现 `aiter()` 方法提供异步迭代器
- 通过 `done` 事件标记生成完成

**前端类比：** 类似于 Server-Sent Events (SSE)，服务器持续推送数据到客户端
**日常生活类比：** 像直播，内容实时生成并传输，而不是录播后一次性播放

### 2. FastAPI StreamingResponse

FastAPI 的流式响应类，支持异步生成器。

**关键特性：**
- `media_type="text/event-stream"` - 使用 SSE 协议
- 接收异步生成器函数
- 自动处理连接管理

### 3. 异步队列管理

使用 `asyncio.Queue` 在回调处理器和生成器之间传递数据。

**工作流程：**
```
LLM 生成 token → 回调处理器 → Queue.put() → 生成器 → Queue.get() → 客户端
```

---

## 完整代码

### 方案1：使用 AsyncIteratorCallbackHandler（推荐）

```python
"""
FastAPI 流式回调集成 - 方案1
使用 LangChain 内置的 AsyncIteratorCallbackHandler
演示：实现流式 RAG 问答 API
"""

import os
import asyncio
from typing import AsyncIterable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 创建 FastAPI 应用
app = FastAPI(title="RAG Streaming API")


# ===== 1. 定义请求和响应模型 =====

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    temperature: float = 0.7
    model: str = "gpt-3.5-turbo"


# ===== 2. 流式消息生成器 =====

async def generate_stream(message: str, temperature: float = 0.7, model: str = "gpt-3.5-turbo") -> AsyncIterable[str]:
    """
    生成流式响应的异步生成器

    Args:
        message: 用户消息
        temperature: 温度参数
        model: 模型名称

    Yields:
        str: 流式 token（SSE 格式）
    """
    # 创建回调处理器
    callback = AsyncIteratorCallbackHandler()

    # 初始化 LLM（传入回调处理器）
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True,  # 启用流式输出
        callbacks=[callback]
    )

    # 定义包装函数，用于捕获异常
    async def wrap_done(fn: asyncio.Task, event: asyncio.Event):
        """
        包装异步任务，确保完成时设置事件

        Args:
            fn: 异步任务
            event: 完成事件
        """
        try:
            await fn
        except Exception as e:
            print(f"Error in LLM generation: {e}")
        finally:
            # 标记生成完成
            event.set()

    # 创建后台任务
    task = asyncio.create_task(
        wrap_done(
            llm.agenerate(messages=[[HumanMessage(content=message)]]),
            callback.done
        )
    )

    # 异步迭代 token
    async for token in callback.aiter():
        # 使用 SSE 格式返回
        yield f"data: {token}\n\n"

    # 等待任务完成
    await task


# ===== 3. API 端点 =====

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    流式聊天端点

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse: 流式响应
    """
    return StreamingResponse(
        generate_stream(
            message=request.message,
            temperature=request.temperature,
            model=request.model
        ),
        media_type="text/event-stream"
    )


# ===== 4. RAG 流式端点 =====

async def generate_rag_stream(query: str) -> AsyncIterable[str]:
    """
    RAG 流式生成器

    Args:
        query: 用户查询

    Yields:
        str: 流式 token（SSE 格式）
    """
    # 创建回调处理器
    callback = AsyncIteratorCallbackHandler()

    # 初始化 LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
        callbacks=[callback]
    )

    # 模拟检索相关文档
    retrieved_docs = [
        "CallbackHandler 是 LangChain 的回调系统，用于监控和追踪 LLM 调用。",
        "通过自定义 CallbackHandler，可以记录日志、统计 token 用量、实现流式输出等。"
    ]

    # 构建增强 prompt
    context = "\n".join(retrieved_docs)
    augmented_prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请用简洁的语言回答。"""

    # 定义包装函数
    async def wrap_done(fn: asyncio.Task, event: asyncio.Event):
        try:
            await fn
        except Exception as e:
            print(f"Error in RAG generation: {e}")
        finally:
            event.set()

    # 创建后台任务
    task = asyncio.create_task(
        wrap_done(
            llm.agenerate(messages=[[HumanMessage(content=augmented_prompt)]]),
            callback.done
        )
    )

    # 异步迭代 token
    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task


@app.post("/rag/stream")
async def stream_rag(request: ChatRequest):
    """
    RAG 流式端点

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse: 流式响应
    """
    return StreamingResponse(
        generate_rag_stream(query=request.message),
        media_type="text/event-stream"
    )


# ===== 5. 主函数 =====

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### 方案2：自定义 AsyncCallbackHandler

```python
"""
FastAPI 流式回调集成 - 方案2
自定义 AsyncCallbackHandler 实现更灵活的控制
演示：实现 Agent 流式输出（只返回最终答案）
"""

import os
import sys
from typing import Any, Callable, Union, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.types import Send

load_dotenv()

app = FastAPI(title="Custom Streaming API")

# ===== 1. 自定义异步回调处理器 =====

Sender = Callable[[Union[str, bytes]], Awaitable[None]]


class CustomAsyncStreamCallbackHandler(AsyncCallbackHandler):
    """
    自定义异步流式回调处理器
    支持过滤中间步骤，只返回最终答案
    """

    def __init__(self, send: Sender):
        """
        初始化回调处理器

        Args:
            send: 发送函数
        """
        super().__init__()
        self.send = send
        self.content = ""
        self.final_answer = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        LLM 生成新 token 时触发

        Args:
            token: 新生成的 token
            **kwargs: 额外参数
        """
        # 累积内容
        self.content += token

        # 检测是否进入最终答案阶段
        if "Final Answer" in self.content or "最终答案" in self.content:
            self.final_answer = True
            self.content = ""  # 清空缓冲区

        # 只发送最终答案的 token
        if self.final_answer:
            # 过滤掉 JSON 格式的标记
            if '"action_input": "' in self.content:
                if token not in ["}",  '"']:
                    await self.send(f"data: {token}\n\n")
            else:
                await self.send(f"data: {token}\n\n")


# ===== 2. 自定义 StreamingResponse =====

class CustomStreamingResponse(StreamingResponse):
    """
    自定义流式响应类
    支持更灵活的流式控制
    """

    def __init__(
        self,
        generate: Callable[[Sender], Awaitable[None]],
        status_code: int = 200,
        media_type: str = "text/event-stream",
    ) -> None:
        super().__init__(content=iter(()), status_code=status_code, media_type=media_type)
        self.generate = generate

    async def stream_response(self, send: Send) -> None:
        """
        流式响应处理

        Args:
            send: ASGI send 函数
        """
        # 发送响应头
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        # 定义发送函数
        async def send_chunk(chunk: Union[str, bytes]):
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        # 生成并发送内容
        await self.generate(send_chunk)

        # 发送结束标记
        await send({"type": "http.response.body", "body": b"", "more_body": False})


# ===== 3. 消息生成函数 =====

def create_message_generator(message: str) -> Callable[[Sender], Awaitable[None]]:
    """
    创建消息生成器

    Args:
        message: 用户消息

    Returns:
        Callable: 异步生成器函数
    """
    async def generate(send: Sender):
        # 创建自定义回调处理器
        callback = CustomAsyncStreamCallbackHandler(send)

        # 初始化 LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True,
            callbacks=[callback]
        )

        # 生成响应
        await llm.agenerate(messages=[[HumanMessage(content=message)]])

    return generate


# ===== 4. API 端点 =====

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}


@app.post("/chat/custom-stream")
async def custom_stream_chat(request: ChatRequest):
    """
    自定义流式聊天端点

    Args:
        request: 聊天请求

    Returns:
        CustomStreamingResponse: 自定义流式响应
    """
    return CustomStreamingResponse(
        create_message_generator(request.message)
    )


# ===== 5. 主函数 =====

if __name__ == "__main__":
    print("Starting Custom Streaming API server...")
    print("API Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

---

## 客户端测试代码

```python
"""
客户端测试代码
演示：如何调用流式 API
"""

import requests


def test_stream_api():
    """测试流式 API"""
    print("=== 测试流式 API ===\n")

    url = "http://localhost:8000/chat/stream"
    data = {
        "message": "请详细解释什么是 RAG？",
        "temperature": 0.7,
        "model": "gpt-3.5-turbo"
    }

    # 发送流式请求
    with requests.post(url, json=data, stream=True) as response:
        print("响应开始：\n")

        # 逐行读取流式响应
        for line in response.iter_lines():
            if line:
                # 解码并去除 "data: " 前缀
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    token = decoded_line[6:]  # 去除 "data: "
                    print(token, end="", flush=True)

        print("\n\n响应结束")


def test_rag_stream_api():
    """测试 RAG 流式 API"""
    print("\n=== 测试 RAG 流式 API ===\n")

    url = "http://localhost:8000/rag/stream"
    data = {
        "message": "LangChain 的 CallbackHandler 有什么作用？"
    }

    with requests.post(url, json=data, stream=True) as response:
        print("RAG 响应开始：\n")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    token = decoded_line[6:]
                    print(token, end="", flush=True)

        print("\n\nRAG 响应结束")


if __name__ == "__main__":
    # 测试基础流式 API
    test_stream_api()

    # 测试 RAG 流式 API
    test_rag_stream_api()
```

---

## 运行输出示例

### 启动服务器

```bash
$ python fastapi_streaming.py

Starting FastAPI server...
API Docs: http://localhost:8000/docs
Health Check: http://localhost:8000/health
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 客户端测试输出

```
=== 测试流式 API ===

响应开始：

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，它通过从外部知识库检索相关信息来增强大语言模型的回答质量。

具体来说，RAG 的工作流程包括：

1. 检索阶段：根据用户查询，从向量数据库中检索相关文档
2. 增强阶段：将检索到的文档作为上下文注入到 prompt 中
3. 生成阶段：LLM 基于增强后的 prompt 生成答案

RAG 的优势在于可以让 LLM 访问最新的、特定领域的知识，而无需重新训练模型。

响应结束

=== 测试 RAG 流式 API ===

RAG 响应开始：

CallbackHandler 是 LangChain 的回调系统，主要用于监控和追踪 LLM 调用。通过自定义 CallbackHandler，开发者可以实现日志记录、token 用量统计、流式输出等功能。

RAG 响应结束
```

---

## 代码解析

### 1. AsyncIteratorCallbackHandler 工作原理

```python
callback = AsyncIteratorCallbackHandler()

# 内部实现（简化版）
class AsyncIteratorCallbackHandler:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_llm_new_token(self, token: str, **kwargs):
        # 将 token 放入队列
        await self.queue.put(token)

    async def aiter(self):
        # 异步迭代器
        while not self.done.is_set():
            try:
                token = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.1
                )
                yield token
            except asyncio.TimeoutError:
                continue
```

**关键点：**
- 使用 `asyncio.Queue` 作为缓冲区
- `on_llm_new_token` 将 token 放入队列
- `aiter()` 从队列中取出 token 并 yield

### 2. 异步任务包装

```python
async def wrap_done(fn: asyncio.Task, event: asyncio.Event):
    try:
        await fn
    except Exception as e:
        print(f"Error: {e}")
    finally:
        event.set()  # 确保事件被设置
```

**作用：**
- 捕获异常，防止任务崩溃
- 确保 `done` 事件被设置，避免死锁
- 提供错误日志

### 3. SSE 格式

```python
yield f"data: {token}\n\n"
```

**SSE 协议要求：**
- 每条消息以 `data: ` 开头
- 以 `\n\n` 结尾
- 客户端通过 `EventSource` 或流式请求接收

---

## 在 RAG 中的应用

### 1. 实时文档问答

```python
async def rag_qa_stream(query: str, docs: list[str]) -> AsyncIterable[str]:
    """RAG 问答流式生成"""
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(streaming=True, callbacks=[callback])

    # 构建增强 prompt
    context = "\n".join(docs)
    prompt = f"基于以下文档回答：\n{context}\n\n问题：{query}"

    # 异步生成
    task = asyncio.create_task(
        llm.agenerate(messages=[[HumanMessage(content=prompt)]])
    )

    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task
```

### 2. 多轮对话流式响应

```python
async def conversational_rag_stream(
    query: str,
    chat_history: list[dict]
) -> AsyncIterable[str]:
    """对话式 RAG 流式生成"""
    # 检索相关文档
    docs = retriever.get_relevant_documents(query)

    # 构建对话历史
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 添加当前查询
    context = "\n".join([doc.page_content for doc in docs])
    messages.append(HumanMessage(content=f"上下文：{context}\n\n问题：{query}"))

    # 流式生成
    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(streaming=True, callbacks=[callback])

    task = asyncio.create_task(llm.agenerate(messages=[messages]))

    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task
```

---

## 常见问题

### Q1: 如何处理连接中断？

**A:** 使用 try-except 捕获异常：

```python
async def generate_stream(message: str) -> AsyncIterable[str]:
    try:
        callback = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(streaming=True, callbacks=[callback])

        task = asyncio.create_task(
            llm.agenerate(messages=[[HumanMessage(content=message)]])
        )

        async for token in callback.aiter():
            yield f"data: {token}\n\n"

        await task

    except asyncio.CancelledError:
        print("Client disconnected")
        task.cancel()
    except Exception as e:
        print(f"Error: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"
```

### Q2: 如何限制流式响应的速率？

**A:** 使用 `asyncio.sleep()` 控制速率：

```python
async for token in callback.aiter():
    yield f"data: {token}\n\n"
    await asyncio.sleep(0.01)  # 限制速率
```

### Q3: 如何在流式响应中添加元数据？

**A:** 使用 JSON 格式传输：

```python
import json

async for token in callback.aiter():
    data = {
        "type": "token",
        "content": token,
        "timestamp": time.time()
    }
    yield f"data: {json.dumps(data)}\n\n"
```

### Q4: 如何实现超时控制？

**A:** 使用 `asyncio.wait_for()`：

```python
try:
    await asyncio.wait_for(task, timeout=30.0)
except asyncio.TimeoutError:
    yield "data: [TIMEOUT] Request timeout\n\n"
```

---

## 性能优化

### 1. 连接池管理

```python
from langchain_openai import ChatOpenAI

# 复用 LLM 实例
llm_pool = {}

def get_llm(model: str):
    if model not in llm_pool:
        llm_pool[model] = ChatOpenAI(model=model)
    return llm_pool[model]
```

### 2. 并发控制

```python
from asyncio import Semaphore

# 限制并发请求数
semaphore = Semaphore(10)

async def generate_stream(message: str):
    async with semaphore:
        # 流式生成逻辑
        pass
```

### 3. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query: str):
    # 缓存常见查询的响应
    pass
```

---

## 扩展练习

1. **添加进度指示**：在流式响应中添加生成进度
2. **实现重试机制**：当 LLM 调用失败时自动重试
3. **多模型切换**：支持在流式响应中动态切换模型
4. **WebSocket 集成**：使用 WebSocket 替代 SSE
5. **流式日志记录**：同时记录流式响应到日志文件

---

## 参考资源

- [FastAPI StreamingResponse 文档](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [LangChain AsyncIteratorCallbackHandler 源码](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/streaming_aiter.py)
- [Server-Sent Events (SSE) 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Pinecone LangChain Streaming 教程](https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/)
