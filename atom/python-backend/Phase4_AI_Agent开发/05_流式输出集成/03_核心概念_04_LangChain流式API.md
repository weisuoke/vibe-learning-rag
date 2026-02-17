# 核心概念04: LangChain流式API

> 深入理解 LangChain 的 astream() 系列流式输出方法

---

## 概述

**LangChain** 提供了完整的流式输出 API,包括 `astream()`、`astream_log()`、`astream_events()` 等方法,用于实现 LLM 的流式输出。

**本节目标:**
- 掌握 LangChain 的三种流式 API
- 理解 Token 流式和 Chunk 流式的区别
- 学习如何集成 LangChain 流式输出到 FastAPI
- 实现生产级的 LangChain 流式端点

---

## 1. LangChain 流式 API 概览

### 1.1 三种流式方法

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

# 1. astream() - 基础流式输出
async for chunk in llm.astream("讲个笑话"):
    print(chunk.content, end="")

# 2. astream_log() - 带日志的流式输出
async for chunk in llm.astream_log("讲个笑话"):
    print(chunk)

# 3. astream_events() - 带事件的流式输出
async for event in llm.astream_events("讲个笑话", version="v1"):
    print(event)
```

**对比:**

| 方法 | 返回内容 | 适用场景 |
|------|----------|----------|
| `astream()` | Token/Chunk | 基础流式输出 |
| `astream_log()` | 日志+Token | 调试和监控 |
| `astream_events()` | 事件流 | 复杂流程可视化 |

### 1.2 astream() 详解

```python
"""
astream() 基础用法
"""

from langchain_openai import ChatOpenAI
import asyncio

async def basic_astream():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 流式生成
    async for chunk in llm.astream("写一首诗"):
        # chunk 是 AIMessageChunk 对象
        print(f"类型: {type(chunk)}")
        print(f"内容: {chunk.content}")
        print(f"---")

asyncio.run(basic_astream())
```

**输出:**
```
类型: <class 'langchain_core.messages.ai.AIMessageChunk'>
内容: 春
---
类型: <class 'langchain_core.messages.ai.AIMessageChunk'>
内容: 风
---
...
```

---

## 2. Token 流式输出

### 2.1 什么是 Token 流式?

**Token 流式** = 逐 Token 返回 LLM 输出

```python
"""
Token 流式输出示例
"""

from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
llm = ChatOpenAI()

@app.post("/chat-token-stream")
async def chat_token_stream(message: str):
    """Token 级流式输出"""
    async def generate():
        async for chunk in llm.astream(message):
            if chunk.content:
                # 逐 Token 发送
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**特点:**
- 实时性最好 (立即看到第一个字)
- 网络开销大 (每个 Token 一次传输)
- 适合聊天机器人 (打字机效果)

### 2.2 Token 流式的完整示例

```python
"""
完整的 Token 流式输出示例
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

app = FastAPI()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

@app.post("/chat-token")
async def chat_token(message: str):
    """Token 级流式聊天"""
    async def generate():
        try:
            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({'status': 'started'})}\n\n"

            # 流式生成
            full_response = ""
            async for chunk in llm.astream([HumanMessage(content=message)]):
                if chunk.content:
                    full_response += chunk.content
                    # 发送 Token
                    yield f"data: {chunk.content}\n\n"

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'full_response': full_response})}\n\n"

        except Exception as e:
            # 发送错误事件
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. Chunk 流式输出

### 3.1 什么是 Chunk 流式?

**Chunk 流式** = 逐 Chunk 返回 LCEL 链的中间结果

```python
"""
Chunk 流式输出示例
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 构建 LCEL 链
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
llm = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | llm | parser

# Chunk 流式输出
async for chunk in chain.astream({"topic": "程序员"}):
    print(chunk, end="")
```

**特点:**
- 可以看到链的中间步骤
- 网络开销适中
- 适合 RAG 问答 (先显示检索结果,再显示生成内容)

### 3.2 Chunk 流式的完整示例

```python
"""
完整的 Chunk 流式输出示例
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

app = FastAPI()

@app.post("/chat-chunk")
async def chat_chunk(topic: str):
    """Chunk 级流式输出"""
    async def generate():
        try:
            # 构建链
            prompt = ChatPromptTemplate.from_template(
                "请详细解释{topic},包括定义、原理、应用场景"
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            parser = StrOutputParser()
            chain = prompt | llm | parser

            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({'topic': topic})}\n\n"

            # 流式生成
            async for chunk in chain.astream({"topic": topic}):
                # 发送 Chunk
                yield f"data: {chunk}\n\n"

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 4. astream_events() 高级用法

### 4.1 什么是 astream_events()?

**astream_events()** 返回链执行过程中的所有事件,包括:
- LLM 调用开始/结束
- Tool 调用开始/结束
- Chain 执行开始/结束
- Token 生成

```python
"""
astream_events() 基础用法
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

async for event in llm.astream_events("讲个笑话", version="v1"):
    print(f"事件类型: {event['event']}")
    print(f"事件名称: {event['name']}")
    print(f"事件数据: {event['data']}")
    print("---")
```

**输出:**
```
事件类型: on_llm_start
事件名称: ChatOpenAI
事件数据: {'input': '讲个笑话'}
---
事件类型: on_llm_stream
事件名称: ChatOpenAI
事件数据: {'chunk': AIMessageChunk(content='为')}
---
事件类型: on_llm_stream
事件名称: ChatOpenAI
事件数据: {'chunk': AIMessageChunk(content='什么')}
---
...
事件类型: on_llm_end
事件名称: ChatOpenAI
事件数据: {'output': AIMessage(content='为什么程序员喜欢黑色？因为黑色显瘦！')}
---
```

### 4.2 astream_events() 完整示例

```python
"""
使用 astream_events() 实现详细的流式输出
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

app = FastAPI()

@app.post("/chat-events")
async def chat_events(message: str):
    """基于事件的流式输出"""
    async def generate():
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo")

            # 流式生成事件
            async for event in llm.astream_events(message, version="v1"):
                event_type = event['event']
                event_name = event['name']
                event_data = event['data']

                # 处理不同类型的事件
                if event_type == "on_llm_start":
                    # LLM 开始生成
                    yield f"event: llm_start\ndata: {json.dumps({'status': 'started'})}\n\n"

                elif event_type == "on_llm_stream":
                    # LLM 流式输出
                    chunk = event_data.get('chunk')
                    if chunk and chunk.content:
                        yield f"data: {chunk.content}\n\n"

                elif event_type == "on_llm_end":
                    # LLM 生成结束
                    output = event_data.get('output')
                    yield f"event: llm_end\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 5. RAG 流式问答

### 5.1 RAG 流式输出的挑战

**挑战:**
1. 需要先显示检索结果
2. 再显示生成内容
3. 两者都要流式输出

**解决方案:** 使用 astream_events() 区分不同阶段

### 5.2 RAG 流式问答示例

```python
"""
RAG 流式问答完整示例
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

app = FastAPI()

# 初始化组件
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")

@app.post("/rag-stream")
async def rag_stream(question: str):
    """RAG 流式问答"""
    async def generate():
        try:
            # 1. 检索阶段
            yield f"event: retrieval_start\ndata: {json.dumps({'status': 'retrieving'})}\n\n"

            docs = await retriever.ainvoke(question)

            # 发送检索结果
            docs_data = [{"content": doc.page_content, "source": doc.metadata} for doc in docs]
            yield f"event: retrieval_done\ndata: {json.dumps({'docs': docs_data})}\n\n"

            # 2. 生成阶段
            yield f"event: generation_start\ndata: {json.dumps({'status': 'generating'})}\n\n"

            # 构建 RAG 链
            prompt = ChatPromptTemplate.from_template(
                "根据以下文档回答问题:\n\n{context}\n\n问题: {question}"
            )

            context = "\n\n".join([doc.page_content for doc in docs])

            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 流式生成答案
            async for chunk in chain.astream(question):
                yield f"data: {chunk}\n\n"

            # 3. 完成
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**前端接收:**

```javascript
const eventSource = new EventSource('/rag-stream?question=什么是AI');

// 监听检索开始
eventSource.addEventListener('retrieval_start', (event) => {
    console.log('开始检索...');
});

// 监听检索完成
eventSource.addEventListener('retrieval_done', (event) => {
    const data = JSON.parse(event.data);
    console.log('检索到文档:', data.docs);
    // 显示检索结果
});

// 监听生成开始
eventSource.addEventListener('generation_start', (event) => {
    console.log('开始生成答案...');
});

// 监听生成内容
eventSource.onmessage = (event) => {
    console.log('生成内容:', event.data);
    // 逐字显示答案
};

// 监听完成
eventSource.addEventListener('done', (event) => {
    console.log('完成');
    eventSource.close();
});
```

---

## 6. Agent 流式执行

### 6.1 Agent 流式输出的挑战

**挑战:**
1. 需要显示 Agent 的思考过程
2. 需要显示 Tool 调用
3. 需要显示最终答案

**解决方案:** 使用 astream_events() 捕获所有事件

### 6.2 Agent 流式执行示例

```python
"""
Agent 流式执行完整示例
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import json

app = FastAPI()

# 定义工具
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

@tool
def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算错误"

# 创建 Agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [search, calculator]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

@app.post("/agent-stream")
async def agent_stream(question: str):
    """Agent 流式执行"""
    async def generate():
        try:
            # 使用 astream_events 捕获所有事件
            async for event in agent_executor.astream_events(
                {"input": question},
                version="v1"
            ):
                event_type = event['event']
                event_name = event['name']
                event_data = event['data']

                # Agent 开始思考
                if event_type == "on_chain_start" and "Agent" in event_name:
                    yield f"event: agent_start\ndata: {json.dumps({'status': 'thinking'})}\n\n"

                # Tool 调用开始
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get('input')
                    yield f"event: tool_start\ndata: {json.dumps({'tool': tool_name, 'input': tool_input})}\n\n"

                # Tool 调用结束
                elif event_type == "on_tool_end":
                    tool_output = event_data.get('output')
                    yield f"event: tool_end\ndata: {json.dumps({'output': tool_output})}\n\n"

                # LLM 流式输出
                elif event_type == "on_llm_stream":
                    chunk = event_data.get('chunk')
                    if chunk and chunk.content:
                        yield f"data: {chunk.content}\n\n"

                # Agent 执行结束
                elif event_type == "on_chain_end" and "Agent" in event_name:
                    output = event_data.get('output')
                    yield f"event: agent_end\ndata: {json.dumps({'output': output})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**前端接收:**

```javascript
const eventSource = new EventSource('/agent-stream?question=计算123+456');

// 监听 Agent 开始思考
eventSource.addEventListener('agent_start', (event) => {
    console.log('Agent 开始思考...');
});

// 监听 Tool 调用开始
eventSource.addEventListener('tool_start', (event) => {
    const data = JSON.parse(event.data);
    console.log(`调用工具: ${data.tool}, 输入: ${data.input}`);
});

// 监听 Tool 调用结束
eventSource.addEventListener('tool_end', (event) => {
    const data = JSON.parse(event.data);
    console.log(`工具输出: ${data.output}`);
});

// 监听生成内容
eventSource.onmessage = (event) => {
    console.log('生成内容:', event.data);
};

// 监听 Agent 执行结束
eventSource.addEventListener('agent_end', (event) => {
    const data = JSON.parse(event.data);
    console.log('Agent 执行完成:', data.output);
    eventSource.close();
});
```

---

## 7. 流式输出的性能优化

### 7.1 批量发送

```python
"""
批量发送优化
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI()

@app.post("/chat-batched")
async def chat_batched(message: str):
    """批量发送优化"""
    async def generate():
        buffer = []
        buffer_size = 5  # 每5个 Token 发送一次

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

### 7.2 超时处理

```python
"""
超时处理
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import asyncio

app = FastAPI()
llm = ChatOpenAI()

@app.post("/chat-timeout")
async def chat_timeout(message: str):
    """带超时处理的流式输出"""
    async def generate():
        try:
            # 设置总超时时间
            async with asyncio.timeout(30):  # 30秒超时
                async for chunk in llm.astream(message):
                    if chunk.content:
                        yield f"data: {chunk.content}\n\n"

        except asyncio.TimeoutError:
            yield f"event: error\ndata: Timeout\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 8. 常见问题

### 问题1: 如何获取完整响应?

```python
"""
在流式输出的同时获取完整响应
"""

async def generate():
    full_response = ""

    async for chunk in llm.astream(message):
        if chunk.content:
            full_response += chunk.content
            yield f"data: {chunk.content}\n\n"

    # 发送完整响应
    yield f"event: done\ndata: {json.dumps({'full_response': full_response})}\n\n"
```

### 问题2: 如何处理多轮对话?

```python
"""
多轮对话的流式输出
"""

from langchain_core.messages import HumanMessage, AIMessage

@app.post("/chat-multi-turn")
async def chat_multi_turn(messages: list):
    """多轮对话流式输出"""
    async def generate():
        # 构建消息历史
        chat_history = []
        for msg in messages:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            else:
                chat_history.append(AIMessage(content=msg['content']))

        # 流式生成
        async for chunk in llm.astream(chat_history):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 总结

**LangChain 流式 API 的核心要点:**

1. **三种方法**: astream()、astream_log()、astream_events()
2. **Token 流式**: 逐 Token 返回,实时性最好
3. **Chunk 流式**: 逐 Chunk 返回,可见中间步骤
4. **事件流式**: 捕获所有事件,适合复杂流程

**适用场景:**
- Token 流式: 聊天机器人
- Chunk 流式: RAG 问答
- 事件流式: Agent 执行

**集成 FastAPI:**
- 使用 StreamingResponse
- 转换为 SSE 格式
- 处理错误和超时

**下一步:**

理解了 LangChain 流式 API 后,可以学习:
- 流式输出粒度控制
- 错误处理与重连
- 生产环境优化

---

**记住:** LangChain 流式 API 是实现 AI 流式输出的核心,掌握它是构建生产级 AI Agent 的关键。
