# 实战代码02: LangChain Token流式

> 实现 ChatGPT 式的打字机效果

---

## 概述

本节实现 LangChain 的 Token 级流式输出,逐字显示 LLM 生成的内容,实现打字机效果。

**学习目标:**
- 掌握 LangChain astream() 的使用
- 实现 Token 级流式输出
- 集成到 FastAPI
- 前端实现打字机效果

---

## 1. 基础 Token 流式输出

```python
"""
基础 Token 流式输出
文件: examples/streaming/langchain_token_basic.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import os

app = FastAPI()

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.post("/chat-token")
async def chat_token(message: str):
    """Token 级流式聊天"""
    async def generate():
        # 使用 astream() 逐 Token 生成
        async for chunk in llm.astream(message):
            if chunk.content:
                # 转换为 SSE 格式
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 运行: uvicorn examples.streaming.langchain_token_basic:app --reload
# 测试: curl -X POST "http://localhost:8000/chat-token?message=讲个笑话"
```

---

## 2. 完整的 Token 流式实现

```python
"""
完整的 Token 流式实现
文件: examples/streaming/langchain_token_complete.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import time

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    streaming=True  # 启用流式输出
)

@app.post("/chat")
async def chat(message: str, system_prompt: str = "你是一个有用的助手"):
    """完整的 Token 流式聊天"""
    async def generate():
        try:
            # 1. 发送开始事件
            start_data = {
                "status": "started",
                "timestamp": time.time()
            }
            yield f"event: start\ndata: {json.dumps(start_data)}\n\n"

            # 2. 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]

            # 3. Token 流式生成
            full_response = ""
            token_count = 0
            start_time = time.time()

            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    token_count += 1

                    # 发送 Token
                    token_data = {
                        "token": chunk.content,
                        "token_count": token_count
                    }
                    yield f"data: {json.dumps(token_data)}\n\n"

            # 4. 发送完成事件
            duration = time.time() - start_time
            done_data = {
                "status": "completed",
                "full_response": full_response,
                "token_count": token_count,
                "duration": duration,
                "tokens_per_second": token_count / duration if duration > 0 else 0
            }
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        except Exception as e:
            # 发送错误事件
            error_data = {
                "error": str(e),
                "type": "generation_error"
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. 多轮对话 Token 流式

```python
"""
多轮对话 Token 流式
文件: examples/streaming/langchain_token_conversation.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()
llm = ChatOpenAI(model="gpt-3.5-turbo")

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system_prompt: str = "你是一个有用的助手"

@app.post("/chat-conversation")
async def chat_conversation(request: ChatRequest):
    """多轮对话 Token 流式"""
    async def generate():
        try:
            # 构建消息历史
            chat_history = [SystemMessage(content=request.system_prompt)]

            for msg in request.messages:
                if msg.role == "user":
                    chat_history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    chat_history.append(AIMessage(content=msg.content))

            # Token 流式生成
            async for chunk in llm.astream(chat_history):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"

            yield f"event: done\ndata: Completed\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**测试:**

```bash
curl -X POST "http://localhost:8000/chat-conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好!有什么可以帮助你的吗?"},
      {"role": "user", "content": "讲个笑话"}
    ]
  }'
```

---

## 4. 前端实现打字机效果

### 4.1 React 实现

```javascript
/**
 * React Token 流式聊天
 * 文件: examples/streaming/frontend/ReactTokenChat.jsx
 */

import React, { useState, useRef, useEffect } from 'react';

function TokenStreamChat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [currentMessage, setCurrentMessage] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const eventSourceRef = useRef(null);

    const sendMessage = async () => {
        if (!input.trim() || isStreaming) return;

        // 添加用户消息
        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setCurrentMessage('');
        setIsStreaming(true);

        // 创建 EventSource
        const eventSource = new EventSource(
            `http://localhost:8000/chat?message=${encodeURIComponent(input)}`
        );
        eventSourceRef.current = eventSource;

        // 监听开始事件
        eventSource.addEventListener('start', (event) => {
            console.log('开始生成');
        });

        // 监听 Token
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setCurrentMessage(prev => prev + data.token);
        };

        // 监听完成事件
        eventSource.addEventListener('done', (event) => {
            const data = JSON.parse(event.data);
            console.log('完成:', data);

            // 保存完整消息
            const assistantMessage = {
                role: 'assistant',
                content: data.full_response
            };
            setMessages(prev => [...prev, assistantMessage]);
            setCurrentMessage('');
            setIsStreaming(false);
            eventSource.close();
        });

        // 监听错误
        eventSource.addEventListener('error', (event) => {
            const data = JSON.parse(event.data);
            console.error('错误:', data);
            setIsStreaming(false);
            eventSource.close();
        });
    };

    // 组件卸载时关闭连接
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, []);

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>Token Stream Chat</h1>

            {/* 消息列表 */}
            <div style={{
                border: '1px solid #ccc',
                padding: '20px',
                minHeight: '400px',
                marginBottom: '20px',
                overflowY: 'auto'
            }}>
                {messages.map((msg, index) => (
                    <div key={index} style={{
                        marginBottom: '15px',
                        padding: '10px',
                        background: msg.role === 'user' ? '#e3f2fd' : '#f5f5f5',
                        borderRadius: '8px'
                    }}>
                        <strong>{msg.role === 'user' ? '你' : 'AI'}:</strong>
                        <div style={{ marginTop: '5px' }}>{msg.content}</div>
                    </div>
                ))}

                {/* 当前正在生成的消息 */}
                {currentMessage && (
                    <div style={{
                        marginBottom: '15px',
                        padding: '10px',
                        background: '#f5f5f5',
                        borderRadius: '8px'
                    }}>
                        <strong>AI:</strong>
                        <div style={{ marginTop: '5px' }}>
                            {currentMessage}
                            <span className="cursor">▋</span>
                        </div>
                    </div>
                )}
            </div>

            {/* 输入框 */}
            <div style={{ display: 'flex', gap: '10px' }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="输入消息..."
                    disabled={isStreaming}
                    style={{
                        flex: 1,
                        padding: '10px',
                        fontSize: '16px',
                        border: '1px solid #ccc',
                        borderRadius: '4px'
                    }}
                />
                <button
                    onClick={sendMessage}
                    disabled={isStreaming}
                    style={{
                        padding: '10px 20px',
                        fontSize: '16px',
                        background: isStreaming ? '#ccc' : '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: isStreaming ? 'not-allowed' : 'pointer'
                    }}
                >
                    {isStreaming ? '生成中...' : '发送'}
                </button>
            </div>

            <style>{`
                @keyframes blink {
                    0%, 50% { opacity: 1; }
                    51%, 100% { opacity: 0; }
                }
                .cursor {
                    animation: blink 1s infinite;
                }
            `}</style>
        </div>
    );
}

export default TokenStreamChat;
```

### 4.2 Vue 实现

```vue
<!--
Vue Token 流式聊天
文件: examples/streaming/frontend/VueTokenChat.vue
-->

<template>
  <div class="token-chat">
    <h1>Token Stream Chat</h1>

    <!-- 消息列表 -->
    <div class="messages">
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message', msg.role]"
      >
        <strong>{{ msg.role === 'user' ? '你' : 'AI' }}:</strong>
        <div class="content">{{ msg.content }}</div>
      </div>

      <!-- 当前正在生成的消息 -->
      <div v-if="currentMessage" class="message assistant">
        <strong>AI:</strong>
        <div class="content">
          {{ currentMessage }}
          <span class="cursor">▋</span>
        </div>
      </div>
    </div>

    <!-- 输入框 -->
    <div class="input-area">
      <input
        v-model="input"
        @keypress.enter="sendMessage"
        :disabled="isStreaming"
        placeholder="输入消息..."
      />
      <button @click="sendMessage" :disabled="isStreaming">
        {{ isStreaming ? '生成中...' : '发送' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue';

const messages = ref([]);
const input = ref('');
const currentMessage = ref('');
const isStreaming = ref(false);
let eventSource = null;

const sendMessage = () => {
  if (!input.value.trim() || isStreaming.value) return;

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: input.value
  });

  const userInput = input.value;
  input.value = '';
  currentMessage.value = '';
  isStreaming.value = true;

  // 创建 EventSource
  eventSource = new EventSource(
    `http://localhost:8000/chat?message=${encodeURIComponent(userInput)}`
  );

  // 监听开始事件
  eventSource.addEventListener('start', () => {
    console.log('开始生成');
  });

  // 监听 Token
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    currentMessage.value += data.token;
  };

  // 监听完成事件
  eventSource.addEventListener('done', (event) => {
    const data = JSON.parse(event.data);
    console.log('完成:', data);

    // 保存完整消息
    messages.value.push({
      role: 'assistant',
      content: data.full_response
    });

    currentMessage.value = '';
    isStreaming.value = false;
    eventSource.close();
  });

  // 监听错误
  eventSource.addEventListener('error', (event) => {
    console.error('错误');
    isStreaming.value = false;
    eventSource.close();
  });
};

// 组件卸载时关闭连接
onUnmounted(() => {
  if (eventSource) {
    eventSource.close();
  }
});
</script>

<style scoped>
.token-chat {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.messages {
  border: 1px solid #ccc;
  padding: 20px;
  min-height: 400px;
  margin-bottom: 20px;
  overflow-y: auto;
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  border-radius: 8px;
}

.message.user {
  background: #e3f2fd;
}

.message.assistant {
  background: #f5f5f5;
}

.content {
  margin-top: 5px;
}

.cursor {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.input-area {
  display: flex;
  gap: 10px;
}

.input-area input {
  flex: 1;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.input-area button {
  padding: 10px 20px;
  font-size: 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-area button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
```

---

## 5. 性能优化

### 5.1 缓冲区优化

```python
"""
使用缓冲区减少网络开销
"""

@app.post("/chat-buffered")
async def chat_buffered(message: str, buffer_size: int = 3):
    """带缓冲区的 Token 流式"""
    async def generate():
        buffer = []

        async for chunk in llm.astream(message):
            if chunk.content:
                buffer.append(chunk.content)

                # 缓冲区满了,批量发送
                if len(buffer) >= buffer_size:
                    yield f"data: {''.join(buffer)}\n\n"
                    buffer.clear()

        # 发送剩余数据
        if buffer:
            yield f"data: {''.join(buffer)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.2 超时处理

```python
"""
超时处理
"""

import asyncio

@app.post("/chat-timeout")
async def chat_timeout(message: str, timeout: int = 30):
    """带超时的 Token 流式"""
    async def generate():
        try:
            async with asyncio.timeout(timeout):
                async for chunk in llm.astream(message):
                    if chunk.content:
                        yield f"data: {chunk.content}\n\n"

        except asyncio.TimeoutError:
            yield f"event: error\ndata: Timeout after {timeout}s\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 6. 测试

### 6.1 单元测试

```python
"""
Token 流式单元测试
文件: tests/test_token_stream.py
"""

import pytest
from httpx import AsyncClient
from examples.streaming.langchain_token_complete import app

@pytest.mark.asyncio
async def test_token_stream():
    """测试 Token 流式输出"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat?message=讲个笑话"
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"

            tokens = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    tokens.append(line)

            # 验证收到了 Token
            assert len(tokens) > 0
```

### 6.2 性能测试

```python
"""
Token 流式性能测试
"""

import asyncio
import time

async def benchmark_token_stream():
    """性能测试"""
    async with AsyncClient(app=app, base_url="http://localhost:8000") as client:
        start_time = time.time()
        token_count = 0

        async with client.stream("POST", "/chat?message=写一首诗") as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    token_count += 1

        duration = time.time() - start_time
        print(f"Token 数量: {token_count}")
        print(f"总时间: {duration:.2f}s")
        print(f"Token/秒: {token_count / duration:.2f}")

asyncio.run(benchmark_token_stream())
```

---

## 总结

**本节要点:**

1. **LangChain astream()**: 逐 Token 返回 LLM 输出
2. **FastAPI 集成**: 使用 StreamingResponse 封装
3. **打字机效果**: 前端逐字显示,实现流畅体验
4. **多轮对话**: 支持对话历史
5. **性能优化**: 缓冲区、超时处理

**关键代码:**
```python
async for chunk in llm.astream(message):
    if chunk.content:
        yield f"data: {chunk.content}\n\n"
```

**下一步:**

掌握了 Token 流式后,可以学习:
- Chunk 流式输出
- RAG 流式问答
- Agent 流式执行

---

**记住:** Token 流式输出是实现 ChatGPT 式打字机效果的关键,掌握它是构建现代 AI 聊天应用的基础。
