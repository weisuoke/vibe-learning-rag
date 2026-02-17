# 实战代码03: LangChain Chunk流式

> 实现 LCEL 链的 Chunk 级流式输出,显示中间步骤

---

## 概述

本节实现 LangChain LCEL 链的 Chunk 级流式输出,可以看到链的中间步骤,适合 RAG 问答等场景。

**学习目标:**
- 掌握 LCEL 链的 astream() 使用
- 实现 Chunk 级流式输出
- 显示链的中间步骤
- 优化用户体验

---

## 1. 基础 Chunk 流式输出

```python
"""
基础 Chunk 流式输出
文件: examples/streaming/langchain_chunk_basic.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

@app.post("/explain")
async def explain(topic: str):
    """Chunk 级流式解释"""
    async def generate():
        # 构建 LCEL 链
        prompt = ChatPromptTemplate.from_template(
            "详细解释{topic},分3段讲解:\n1. 定义\n2. 原理\n3. 应用"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        parser = StrOutputParser()

        chain = prompt | llm | parser

        # Chunk 流式输出
        async for chunk in chain.astream({"topic": topic}):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 运行: uvicorn examples.streaming.langchain_chunk_basic:app --reload
# 测试: curl -X POST "http://localhost:8000/explain?topic=AI"
```

---

## 2. 带段落检测的 Chunk 流式

```python
"""
带段落检测的 Chunk 流式
文件: examples/streaming/langchain_chunk_paragraph.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

app = FastAPI()

@app.post("/explain-paragraphs")
async def explain_paragraphs(topic: str):
    """按段落流式输出"""
    async def generate():
        try:
            # 构建链
            prompt = ChatPromptTemplate.from_template(
                "详细解释{topic},分3段讲解,每段之间用空行分隔"
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = prompt | llm

            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({'topic': topic})}\n\n"

            # 流式生成,检测段落
            buffer = ""
            paragraph_count = 0

            async for chunk in chain.astream({"topic": topic}):
                if chunk.content:
                    buffer += chunk.content

                    # 检测段落结束 (双换行)
                    if "\n\n" in buffer:
                        parts = buffer.split("\n\n")

                        # 发送完整段落
                        for part in parts[:-1]:
                            if part.strip():
                                paragraph_count += 1
                                paragraph_data = {
                                    "paragraph_id": paragraph_count,
                                    "content": part.strip()
                                }
                                yield f"event: paragraph\ndata: {json.dumps(paragraph_data)}\n\n"

                        # 保留未完成的部分
                        buffer = parts[-1]

            # 发送剩余内容
            if buffer.strip():
                paragraph_count += 1
                paragraph_data = {
                    "paragraph_id": paragraph_count,
                    "content": buffer.strip()
                }
                yield f"event: paragraph\ndata: {json.dumps(paragraph_data)}\n\n"

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'paragraph_count': paragraph_count})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. 多步骤 Chunk 流式

```python
"""
多步骤 Chunk 流式
文件: examples/streaming/langchain_chunk_multistep.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json

app = FastAPI()

@app.post("/analyze")
async def analyze(text: str):
    """多步骤分析"""
    async def generate():
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo")

            # 步骤1: 摘要
            yield f"event: step\ndata: {json.dumps({'step': 'summary', 'status': 'started'})}\n\n"

            summary_prompt = ChatPromptTemplate.from_template("总结以下文本:\n{text}")
            summary_chain = summary_prompt | llm

            summary = ""
            async for chunk in summary_chain.astream({"text": text}):
                if chunk.content:
                    summary += chunk.content
                    yield f"data: {chunk.content}\n\n"

            yield f"event: step\ndata: {json.dumps({'step': 'summary', 'status': 'completed', 'result': summary})}\n\n"

            # 步骤2: 关键词提取
            yield f"event: step\ndata: {json.dumps({'step': 'keywords', 'status': 'started'})}\n\n"

            keywords_prompt = ChatPromptTemplate.from_template("提取以下文本的关键词:\n{text}")
            keywords_chain = keywords_prompt | llm

            keywords = ""
            async for chunk in keywords_chain.astream({"text": text}):
                if chunk.content:
                    keywords += chunk.content
                    yield f"data: {chunk.content}\n\n"

            yield f"event: step\ndata: {json.dumps({'step': 'keywords', 'status': 'completed', 'result': keywords})}\n\n"

            # 步骤3: 情感分析
            yield f"event: step\ndata: {json.dumps({'step': 'sentiment', 'status': 'started'})}\n\n"

            sentiment_prompt = ChatPromptTemplate.from_template("分析以下文本的情感:\n{text}")
            sentiment_chain = sentiment_prompt | llm

            sentiment = ""
            async for chunk in sentiment_chain.astream({"text": text}):
                if chunk.content:
                    sentiment += chunk.content
                    yield f"data: {chunk.content}\n\n"

            yield f"event: step\ndata: {json.dumps({'step': 'sentiment', 'status': 'completed', 'result': sentiment})}\n\n"

            # 完成
            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 4. 前端实现

### 4.1 React 实现

```javascript
/**
 * React Chunk 流式客户端
 * 文件: examples/streaming/frontend/ReactChunkStream.jsx
 */

import React, { useState } from 'react';

function ChunkStreamClient() {
    const [topic, setTopic] = useState('');
    const [paragraphs, setParagraphs] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [currentParagraph, setCurrentParagraph] = useState('');

    const startStream = () => {
        if (!topic.trim() || isStreaming) return;

        setParagraphs([]);
        setCurrentParagraph('');
        setIsStreaming(true);

        const eventSource = new EventSource(
            `http://localhost:8000/explain-paragraphs?topic=${encodeURIComponent(topic)}`
        );

        // 监听开始事件
        eventSource.addEventListener('start', (event) => {
            console.log('开始生成');
        });

        // 监听段落事件
        eventSource.addEventListener('paragraph', (event) => {
            const data = JSON.parse(event.data);
            setParagraphs(prev => [...prev, data]);
        });

        // 监听完成事件
        eventSource.addEventListener('done', (event) => {
            const data = JSON.parse(event.data);
            console.log('完成:', data);
            setIsStreaming(false);
            eventSource.close();
        });

        // 监听错误
        eventSource.addEventListener('error', (event) => {
            console.error('错误');
            setIsStreaming(false);
            eventSource.close();
        });
    };

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>Chunk Stream - 段落流式输出</h1>

            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
                <input
                    type="text"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && startStream()}
                    placeholder="输入主题..."
                    disabled={isStreaming}
                    style={{ flex: 1, padding: '10px', fontSize: '16px' }}
                />
                <button
                    onClick={startStream}
                    disabled={isStreaming}
                    style={{ padding: '10px 20px', fontSize: '16px' }}
                >
                    {isStreaming ? '生成中...' : '开始'}
                </button>
            </div>

            <div style={{ border: '1px solid #ccc', padding: '20px', minHeight: '400px' }}>
                {paragraphs.map((para, index) => (
                    <div
                        key={index}
                        style={{
                            marginBottom: '20px',
                            padding: '15px',
                            background: '#f9f9f9',
                            borderLeft: '4px solid #007bff',
                            animation: 'fadeIn 0.5s'
                        }}
                    >
                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                            段落 {para.paragraph_id}
                        </div>
                        <div style={{ lineHeight: '1.6' }}>{para.content}</div>
                    </div>
                ))}
            </div>

            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
}

export default ChunkStreamClient;
```

### 4.2 多步骤流式前端

```javascript
/**
 * 多步骤流式前端
 * 文件: examples/streaming/frontend/ReactMultiStep.jsx
 */

import React, { useState } from 'react';

function MultiStepStream() {
    const [text, setText] = useState('');
    const [steps, setSteps] = useState({});
    const [currentStep, setCurrentStep] = useState(null);
    const [currentContent, setCurrentContent] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);

    const startAnalysis = () => {
        if (!text.trim() || isStreaming) return;

        setSteps({});
        setCurrentStep(null);
        setCurrentContent('');
        setIsStreaming(true);

        const eventSource = new EventSource(
            `http://localhost:8000/analyze?text=${encodeURIComponent(text)}`
        );

        // 监听步骤事件
        eventSource.addEventListener('step', (event) => {
            const data = JSON.parse(event.data);

            if (data.status === 'started') {
                setCurrentStep(data.step);
                setCurrentContent('');
            } else if (data.status === 'completed') {
                setSteps(prev => ({
                    ...prev,
                    [data.step]: data.result
                }));
                setCurrentStep(null);
                setCurrentContent('');
            }
        });

        // 监听内容
        eventSource.onmessage = (event) => {
            setCurrentContent(prev => prev + event.data);
        };

        // 监听完成
        eventSource.addEventListener('done', () => {
            setIsStreaming(false);
            eventSource.close();
        });

        // 监听错误
        eventSource.addEventListener('error', () => {
            setIsStreaming(false);
            eventSource.close();
        });
    };

    const stepNames = {
        summary: '摘要',
        keywords: '关键词',
        sentiment: '情感分析'
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1000px', margin: '0 auto' }}>
            <h1>多步骤分析</h1>

            <div style={{ marginBottom: '20px' }}>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="输入要分析的文本..."
                    disabled={isStreaming}
                    style={{
                        width: '100%',
                        minHeight: '100px',
                        padding: '10px',
                        fontSize: '16px'
                    }}
                />
                <button
                    onClick={startAnalysis}
                    disabled={isStreaming}
                    style={{ marginTop: '10px', padding: '10px 20px', fontSize: '16px' }}
                >
                    {isStreaming ? '分析中...' : '开始分析'}
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
                {['summary', 'keywords', 'sentiment'].map(step => (
                    <div
                        key={step}
                        style={{
                            border: '1px solid #ccc',
                            padding: '15px',
                            borderRadius: '8px',
                            background: currentStep === step ? '#e3f2fd' : 'white'
                        }}
                    >
                        <h3>{stepNames[step]}</h3>
                        <div style={{ minHeight: '100px', lineHeight: '1.6' }}>
                            {currentStep === step ? (
                                <div>
                                    {currentContent}
                                    <span className="cursor">▋</span>
                                </div>
                            ) : steps[step] ? (
                                steps[step]
                            ) : (
                                <div style={{ color: '#999' }}>等待中...</div>
                            )}
                        </div>
                    </div>
                ))}
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

export default MultiStepStream;
```

---

## 5. 性能优化

### 5.1 智能缓冲

```python
"""
智能缓冲优化
"""

@app.post("/explain-buffered")
async def explain_buffered(topic: str):
    """智能缓冲的 Chunk 流式"""
    async def generate():
        prompt = ChatPromptTemplate.from_template("详细解释{topic}")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        chain = prompt | llm

        buffer = ""
        last_send_time = time.time()
        buffer_timeout = 0.5  # 500ms 超时

        async for chunk in chain.astream({"topic": topic}):
            if chunk.content:
                buffer += chunk.content

                # 条件1: 检测到句子结束
                if any(buffer.endswith(p) for p in ['. ', '。', '! ', '！', '? ', '？']):
                    yield f"data: {buffer}\n\n"
                    buffer = ""
                    last_send_time = time.time()

                # 条件2: 缓冲区超时
                elif time.time() - last_send_time > buffer_timeout:
                    if buffer:
                        yield f"data: {buffer}\n\n"
                        buffer = ""
                        last_send_time = time.time()

        # 发送剩余内容
        if buffer:
            yield f"data: {buffer}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 5.2 并行处理

```python
"""
并行处理多个步骤
"""

import asyncio

@app.post("/analyze-parallel")
async def analyze_parallel(text: str):
    """并行分析"""
    async def generate():
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # 定义所有步骤
        steps = {
            'summary': ChatPromptTemplate.from_template("总结:\n{text}"),
            'keywords': ChatPromptTemplate.from_template("关键词:\n{text}"),
            'sentiment': ChatPromptTemplate.from_template("情感:\n{text}")
        }

        # 并行执行所有步骤
        async def process_step(step_name, prompt):
            chain = prompt | llm
            result = ""
            async for chunk in chain.astream({"text": text}):
                if chunk.content:
                    result += chunk.content
            return step_name, result

        # 并行运行
        tasks = [process_step(name, prompt) for name, prompt in steps.items()]
        results = await asyncio.gather(*tasks)

        # 发送结果
        for step_name, result in results:
            yield f"event: {step_name}\ndata: {json.dumps({'result': result})}\n\n"

        yield f"event: done\ndata: Completed\n\n"

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
Chunk 流式单元测试
文件: tests/test_chunk_stream.py
"""

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_chunk_stream():
    """测试 Chunk 流式输出"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/explain-paragraphs?topic=AI"
        ) as response:
            assert response.status_code == 200

            paragraphs = []
            async for line in response.aiter_lines():
                if line.startswith("event: paragraph"):
                    # 下一行是数据
                    continue
                elif line.startswith("data:"):
                    data = json.loads(line[6:])
                    if 'paragraph_id' in data:
                        paragraphs.append(data)

            # 验证收到了段落
            assert len(paragraphs) > 0
```

---

## 总结

**本节要点:**

1. **Chunk 流式**: 逐段落/逐句返回,适合长文本
2. **段落检测**: 通过双换行检测段落边界
3. **多步骤流式**: 显示处理的不同阶段
4. **智能缓冲**: 根据句子边界和超时发送
5. **并行处理**: 同时执行多个步骤

**关键代码:**
```python
async for chunk in chain.astream({"topic": topic}):
    buffer += chunk.content
    if "\n\n" in buffer:
        # 发送完整段落
        yield f"data: {paragraph}\n\n"
```

**下一步:**

掌握了 Chunk 流式后,可以学习:
- RAG 流式问答
- Agent 流式执行
- 前端集成示例

---

**记住:** Chunk 流式输出适合需要显示中间步骤的场景,如 RAG 问答、多步骤分析等。
