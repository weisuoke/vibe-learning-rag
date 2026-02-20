# 核心概念 04: stream 方法

> 理解 Runnable 的流式实时输出方法

---

## 什么是 stream 方法？

**stream 是 Runnable 协议的流式执行方法，用于逐块输出结果，实现实时响应和降低感知延迟。**

### 一句话定义

stream 方法接收一个输入和可选的配置参数，返回一个迭代器，逐块产出输出数据。

---

## 方法签名

```python
from typing import TypeVar, Optional, Iterator
from langchain_core.runnables.config import RunnableConfig

Input = TypeVar("Input")
Output = TypeVar("Output")

def stream(
    self,
    input: Input,
    config: Optional[RunnableConfig] = None
) -> Iterator[Output]:
    """
    流式实时输出

    Args:
        input: 输入数据
        config: 运行时配置（可选）

    Yields:
        输出数据块（chunk）

    注意:
        - 返回迭代器，需要遍历获取结果
        - 适用于 LLM token-by-token 输出
        - 降低用户感知延迟
    """
    ...
```

---

## 流式输出原理

### Token-by-Token 流式

**LLM 生成文本时逐 token 输出**[^1]：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

# 流式输出
for chunk in chain.stream({"topic": "春天"}):
    print(chunk.content, end="", flush=True)

# 输出效果：
# 春风拂面...（逐字显示）
```

### 异步流式

```python
import asyncio

async def stream_example():
    async for chunk in chain.astream({"topic": "春天"}):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_example())
```

---

## UI 集成模式

### FastAPI 集成（2025-2026 最佳实践）[^2]

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    prompt = ChatPromptTemplate.from_template("回答: {message}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm

    async def generate():
        async for chunk in chain.astream({"message": request.message}):
            # 发送 SSE 格式
            yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 前端集成（EventSource）

```javascript
// 前端代码
const eventSource = new EventSource('/chat/stream');

eventSource.onmessage = (event) => {
    const chunk = event.data;
    // 逐块显示
    document.getElementById('output').textContent += chunk;
};

eventSource.onerror = () => {
    eventSource.close();
};
```

---

## 流式模式

### 模式 1: updates（默认）

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough() | llm

# 只输出最终结果的增量
for chunk in chain.stream({"text": "你好"}):
    print(chunk)
```

### 模式 2: messages（聊天应用）

```python
# 输出完整的消息对象
for chunk in chain.stream({"text": "你好"}, stream_mode="messages"):
    print(chunk)
```

### 模式 3: custom（自定义）

```python
# 自定义流式输出格式
for chunk in chain.stream({"text": "你好"}, stream_mode="custom"):
    print(chunk)
```

---

## 性能优化

### 降低感知延迟

**流式输出可降低 50-70% 的感知延迟**[^3]：

```python
import time

# ❌ 非流式：等待完整响应
start = time.time()
result = chain.invoke({"text": "写一篇长文章"})
print(f"首次输出延迟: {time.time() - start:.2f}秒")  # 约 5-10 秒

# ✅ 流式：立即开始输出
start = time.time()
first_chunk = True
for chunk in chain.stream({"text": "写一篇长文章"}):
    if first_chunk:
        print(f"首次输出延迟: {time.time() - start:.2f}秒")  # 约 0.5-1 秒
        first_chunk = False
    print(chunk.content, end="", flush=True)
```

### 进度跟踪

```python
from langchain_core.runnables import RunnableLambda

def track_progress(chunks):
    """跟踪流式进度"""
    total_chunks = 0
    for chunk in chunks:
        total_chunks += 1
        print(f"\r处理中... ({total_chunks} chunks)", end="", flush=True)
        yield chunk
    print(f"\n完成！共 {total_chunks} chunks")

# 使用
chain_with_progress = chain | RunnableLambda(track_progress)
for chunk in chain_with_progress.stream({"text": "你好"}):
    pass
```

---

## 实战代码示例

### 示例 1: 流式聊天应用

```python
"""
流式聊天应用
演示 stream 方法的实际应用
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import sys

# 定义聊天链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),
    ("user", "{message}")
])
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
parser = StrOutputParser()

chain = prompt | llm | parser

# 流式聊天
def chat_stream(message: str):
    """流式输出聊天响应"""
    print("助手: ", end="", flush=True)

    full_response = ""
    for chunk in chain.stream({"message": message}):
        print(chunk, end="", flush=True)
        full_response += chunk

    print()  # 换行
    return full_response

# 使用
if __name__ == "__main__":
    print("=== 流式聊天演示 ===\n")

    messages = [
        "你好！",
        "介绍一下 LangChain",
        "什么是 Runnable？"
    ]

    for msg in messages:
        print(f"用户: {msg}")
        response = chat_stream(msg)
        print()
```

### 示例 2: 带取消的流式输出

```python
"""
可取消的流式输出
演示如何中断流式处理
"""

import signal
import sys

class StreamCanceller:
    """流式输出取消器"""

    def __init__(self):
        self.cancelled = False
        # 注册 Ctrl+C 处理
        signal.signal(signal.SIGINT, self._handle_cancel)

    def _handle_cancel(self, signum, frame):
        """处理取消信号"""
        print("\n\n⚠️  流式输出已取消")
        self.cancelled = True

    def stream_with_cancel(self, chain, input_data):
        """支持取消的流式输出"""
        try:
            for chunk in chain.stream(input_data):
                if self.cancelled:
                    break
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\n\n⚠️  流式输出已中断")

# 使用
canceller = StreamCanceller()
print("提示: 按 Ctrl+C 可以取消输出\n")
canceller.stream_with_cancel(chain, {"message": "写一篇很长的文章"})
```

### 示例 3: FastAPI 完整集成

```python
"""
FastAPI 流式聊天 API
演示生产级流式接口实现
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import asyncio

app = FastAPI(title="流式聊天 API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    stream: bool = True

class ChatResponse(BaseModel):
    response: str

# 定义链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),
    ("user", "{message}")
])
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm

@app.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口（支持流式和非流式）"""
    try:
        if request.stream:
            # 流式响应
            async def generate():
                async for chunk in chain.astream({"message": request.message}):
                    data = {
                        "type": "chunk",
                        "content": chunk.content
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                # 发送结束标记
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            result = await chain.ainvoke({"message": request.message})
            return ChatResponse(response=result.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}

# 运行: uvicorn script_name:app --reload
```

---

## 错误处理

### 流式错误处理

```python
from langchain_core.runnables import RunnableLambda

def safe_stream(chain, input_data):
    """安全的流式输出"""
    try:
        for chunk in chain.stream(input_data):
            yield chunk
    except Exception as e:
        # 发送错误信息
        yield f"\n\n❌ 错误: {e}"

# 使用
for chunk in safe_stream(chain, {"message": "你好"}):
    print(chunk, end="", flush=True)
```

---

## 2025-2026 最佳实践

### 1. 始终启用流式

```python
# ✅ 启用流式
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# ❌ 未启用流式
llm = ChatOpenAI(model="gpt-4o-mini")
```

### 2. 使用异步流式

```python
# ✅ 异步流式（更高效）
async for chunk in chain.astream(input):
    print(chunk, end="", flush=True)

# ❌ 同步流式
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)
```

### 3. 实现取消机制

```python
# ✅ 支持取消
async def cancellable_stream(chain, input_data, cancel_event):
    async for chunk in chain.astream(input_data):
        if cancel_event.is_set():
            break
        yield chunk
```

### 4. 监控流式性能

```python
# ✅ 追踪首 token 延迟
import time

start = time.time()
first_chunk = True
for chunk in chain.stream(input):
    if first_chunk:
        print(f"首 token 延迟: {time.time() - start:.2f}秒")
        first_chunk = False
```

---

## 总结

### stream 方法的核心价值

1. **降低延迟**: 50-70% 感知延迟降低
2. **实时反馈**: 用户立即看到输出
3. **更好体验**: 适合聊天和长文本生成
4. **可取消**: 支持中断长时间任务

### 何时使用 stream

- ✅ 聊天机器人和对话应用
- ✅ 长文本生成（文章、报告）
- ✅ 需要实时反馈的场景
- ❌ 批量处理（用 batch）
- ❌ 简单查询（用 invoke）

---

## 参考资料

[^1]: [LangChain Streaming Overview](https://docs.langchain.com/oss/python/langchain/streaming/overview) - LangChain, 2025-2026
[^2]: [Building Production-Ready AI Pipelines](https://medium.com/@sajo02/building-production-ready-ai-pipelines-with-langchain-runnables-a-complete-lcel-guide-2f9b27f6d557) - Medium, 2026
[^3]: [Streaming and Batching LLM Inference](https://levelup.gitconnected.com/streaming-and-batching-llm-inference-using-nvidia-nim-and-langchain-e0afdc031543) - Level Up Coding, 2025

---

**下一步**: 阅读 [03_核心概念_05_LCEL设计哲学.md](./03_核心概念_05_LCEL设计哲学.md) 理解声明式编程
