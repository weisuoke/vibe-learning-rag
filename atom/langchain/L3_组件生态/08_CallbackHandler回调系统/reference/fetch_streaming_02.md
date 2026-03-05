---
type: fetched_content
source: https://gist.github.com/ninely/88485b2e265d852d3feb8bd115065b1a
title: Langchain with fastapi stream example
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 流式输出处理
---

# Langchain with fastapi stream example

**所有者**：ninely
**最后更新**：September 15, 2025 10:10
**Stars**：100
**Forks**：11

Instantly share code, notes, and snippets.

## 示例说明
这是一个使用 async langchain 与 FastAPI 实现流式响应的示例。
最新版本的 Langchain 改进了与异步 FastAPI 的兼容性，使实现流式功能更加简单。

## 文件列表

### main.py
```python
"""This is an example of how to use async langchain with fastapi and return a streaming response.

The latest version of Langchain has improved its compatibility with asynchronous FastAPI,
making it easier to implement streaming functionality in your applications.
"""
import asyncio
import os
from typing import AsyncIterable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

# Two ways to load env variables
# 1.load env variables from .env file
load_dotenv()

# 2.manually set env variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()


async def send_message(message: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        model.agenerate(messages=[[HumanMessage(content=message)]]),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task


class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(send_message(body.message), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
```

### main2.py
```python
"""This is an example of how to use async langchain with fastapi and return a streaming response."""
import os
from typing import Any, Optional, Awaitable, Callable, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from starlette.types import Send

# two ways to load env variables
# 1.load env variables from .env file
load_dotenv()

# 2.manually set env variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

Sender = Callable[[Union[str, bytes]], Awaitable[None]]


class AsyncStreamCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming, inheritance from AsyncCallbackHandler."""

    def __init__(self, send: Sender):
        super().__init__()
        self.send = send

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Rewrite on_llm_new_token to send token to client."""
        await self.send(f"data: {token}\n\n")


class ChatOpenAIStreamingResponse(StreamingResponse):
    """Streaming response for openai chat model, inheritance from StreamingResponse."""

    def __init__(
        self,
        generate: Callable[[Sender], Awaitable[None]],
        status_code: int = 200,
        media_type: Optional[str] = None,
    ) -> None:
        super().__init__(content=iter(()), status_code=status_code, media_type=media_type)
        self.generate = generate

    async def stream_response(self, send: Send) -> None:
        """Rewrite stream_response to send response to client."""
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        async def send_chunk(chunk: Union[str, bytes]):
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        # send body to client
        await self.generate(send_chunk)

        # send empty body to client to close connection
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def send_message(message: str) -> Callable[[Sender], Awaitable[None]]:
    async def generate(send: Sender):
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callback_manager=AsyncCallbackManager([AsyncStreamCallbackHandler(send)]),
        )
        await model.agenerate(messages=[[HumanMessage(content=message)]])

    return generate


class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


@app.post("/stream")
def stream(body: StreamRequest):
    return ChatOpenAIStreamingResponse(send_message(body.message), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
```

### requirements.txt
```txt
aiohttp == 3.8.4 ; python_full_version >= "3.8.1" and python_version < "3.12"
aiosignal == 1.3.1 ; python_full_version >= "3.8.1" and python_version < "3.12"
anyio == 3.7.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
async-timeout == 4.0.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
attrs == 23.1.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
certifi == 2023.5.7 ; python_full_version >= "3.8.1" and python_version < "3.12"
charset-normalizer == 3.1.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
click == 8.1.3 ; python_full_version >= "3.8.1" and python_version < "3.12"
colorama == 0.4.6 ; python_full_version >= "3.8.1" and python_version < "3.12" and platform_system == "Windows"
dataclasses-json == 0.5.7 ; python_full_version >= "3.8.1" and python_version < "3.12"
exceptiongroup == 1.1.1 ; python_full_version >= "3.8.1" and python_version < "3.11"
fastapi == 0.95.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
frozenlist == 1.3.3 ; python_full_version >= "3.8.1" and python_version < "3.12"
greenlet == 2.0.2 ; python_full_version >= "3.8.1" and python_version < "3.12" and (platform_machine == "win32" or platform_machine == "WIN32" or platform_machine == "AMD64" or platform_machine == "amd64" or platform_machine == "x86_64" or platform_machine == "ppc64le" or platform_machine == "aarch64")
h11 == 0.14.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
idna == 3.4 ; python_full_version >= "3.8.1" and python_version < "3.12"
langchain == 0.0.181 ; python_full_version >= "3.8.1" and python_version < "3.12"
marshmallow-enum == 1.5.1 ; python_full_version >= "3.8.1" and python_version < "3.12"
marshmallow == 3.19.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
multidict == 6.0.4 ; python_full_version >= "3.8.1" and python_version < "3.12"
mypy-extensions == 1.0.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
numexpr == 2.8.4 ; python_full_version >= "3.8.1" and python_version < "3.12"
numpy == 1.24.3 ; python_full_version >= "3.8.1" and python_version < "3.12"
openai == 0.27.7 ; python_full_version >= "3.8.1" and python_version < "3.12"
openapi-schema-pydantic == 1.2.4 ; python_full_version >= "3.8.1" and python_version < "3.12"
packaging == 23.1 ; python_full_version >= "3.8.1" and python_version < "3.12"
pydantic == 1.10.8 ; python_full_version >= "3.8.1" and python_version < "3.12"
python-dotenv == 1.0.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
pyyaml == 6.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
requests == 2.31.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
sniffio == 1.3.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
sqlalchemy == 2.0.15 ; python_full_version >= "3.8.1" and python_version < "3.12"
starlette == 0.27.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
tenacity == 8.2.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
tqdm == 4.65.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
typing-extensions == 4.6.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
typing-inspect == 0.9.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
urllib3 == 2.0.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
uvicorn == 0.22.0 ; python_full_version >= "3.8.1" and python_version < "3.12"
yarl == 1.9.2 ; python_full_version >= "3.8.1" and python_version < "3.12"
```

### test.sh
```sh
#!/usr/bin/env sh
# This script is used to test.
curl "http://127.0.0.1:8000/stream" -X POST -d '{"message": "hello!"}' -H 'Content-Type: application/json'
```

## 附加功能
- [Download ZIP](https://gist.github.com/ninely/88485b2e265d852d3feb8bd115065b1a/archive/b0fb51e341f7467261c24828b18d8807482a0cc4.zip)
- Embed 脚本可用
- 可通过 GitHub Desktop 保存

**原始网页所有文本、代码结构、注释及格式均 100% 完整保留，无任何删减、改写或摘要。**
