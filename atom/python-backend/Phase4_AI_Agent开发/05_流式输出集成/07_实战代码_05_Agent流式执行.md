# 实战代码05: Agent流式执行

> 实现 Agent 工具调用的流式可视化,显示思考过程和执行步骤

---

## 概述

本节实现 Agent 流式执行,使用 astream_events() 捕获 Agent 的思考过程、工具调用和最终答案,实现完整的流程可视化。

**学习目标:**
- 掌握 Agent 流式执行的实现
- 使用 astream_events() 捕获所有事件
- 实现思考过程可视化
- 优化用户体验

---

## 1. 基础 Agent 流式执行

```python
"""
基础 Agent 流式执行
文件: examples/streaming/agent_stream_basic.py
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
    return f"搜索结果: {query} 的相关信息..."

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
    ("system", "你是一个有用的助手,可以使用工具来回答问题"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/agent-execute")
async def agent_execute(question: str):
    """基础 Agent 流式执行"""
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

                # 发送事件
                yield f"event: {event_type}\ndata: {json.dumps({
                    'name': event_name,
                    'data': event_data
                })}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 2. 完整的 Agent 流式执行

```python
"""
完整的 Agent 流式执行
文件: examples/streaming/agent_stream_complete.py
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pydantic import BaseModel
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

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网信息"""
    # 模拟搜索延迟
    import time
    time.sleep(1)
    return f"搜索到关于'{query}'的信息: 这是一个示例搜索结果..."

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    # 模拟天气查询
    return f"{city}的天气: 晴天, 温度25°C"

# 创建 Agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [search, calculator, get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。你可以使用以下工具来回答问题:\n- search: 搜索信息\n- calculator: 计算数学表达式\n- get_weather: 获取天气信息"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class AgentRequest(BaseModel):
    question: str

@app.post("/agent-stream")
async def agent_stream(request: AgentRequest):
    """完整的 Agent 流式执行"""
    async def generate():
        start_time = time.time()

        try:
            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({
                'question': request.question,
                'timestamp': start_time
            })}\n\n"

            # 跟踪状态
            current_tool = None
            thinking_content = ""

            # 使用 astream_events 捕获所有事件
            async for event in agent_executor.astream_events(
                {"input": request.question},
                version="v1"
            ):
                event_type = event['event']
                event_name = event['name']
                event_data = event['data']

                # Agent 开始思考
                if event_type == "on_chain_start" and "Agent" in event_name:
                    yield f"event: agent_thinking\ndata: {json.dumps({
                        'status': 'thinking'
                    })}\n\n"

                # LLM 流式输出 (Agent 的思考过程)
                elif event_type == "on_llm_stream":
                    chunk = event_data.get('chunk')
                    if chunk and hasattr(chunk, 'content') and chunk.content:
                        thinking_content += chunk.content
                        yield f"event: thinking\ndata: {json.dumps({
                            'content': chunk.content
                        })}\n\n"

                # Tool 调用开始
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get('input', {})
                    current_tool = tool_name

                    yield f"event: tool_start\ndata: {json.dumps({
                        'tool': tool_name,
                        'input': tool_input
                    })}\n\n"

                # Tool 调用结束
                elif event_type == "on_tool_end":
                    tool_output = event_data.get('output')

                    yield f"event: tool_end\ndata: {json.dumps({
                        'tool': current_tool,
                        'output': tool_output
                    })}\n\n"

                    current_tool = None

                # Agent 执行结束
                elif event_type == "on_chain_end" and "Agent" in event_name:
                    output = event_data.get('output', {})
                    final_answer = output.get('output', '')

                    duration = time.time() - start_time

                    yield f"event: done\ndata: {json.dumps({
                        'status': 'completed',
                        'answer': final_answer,
                        'duration': duration
                    })}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. 带步骤追踪的 Agent 流式

```python
"""
带步骤追踪的 Agent 流式
文件: examples/streaming/agent_stream_steps.py
"""

@app.post("/agent-stream-steps")
async def agent_stream_steps(question: str):
    """带步骤追踪的 Agent 流式执行"""
    async def generate():
        try:
            step_count = 0
            steps = []

            async for event in agent_executor.astream_events(
                {"input": question},
                version="v1"
            ):
                event_type = event['event']
                event_name = event['name']
                event_data = event['data']

                # 记录步骤
                if event_type == "on_tool_start":
                    step_count += 1
                    step = {
                        "step_id": step_count,
                        "type": "tool_call",
                        "tool": event_name,
                        "input": event_data.get('input'),
                        "status": "running"
                    }
                    steps.append(step)

                    yield f"event: step_start\ndata: {json.dumps(step)}\n\n"

                elif event_type == "on_tool_end":
                    # 更新最后一个步骤
                    if steps:
                        steps[-1]["status"] = "completed"
                        steps[-1]["output"] = event_data.get('output')

                        yield f"event: step_end\ndata: {json.dumps(steps[-1])}\n\n"

                elif event_type == "on_chain_end" and "Agent" in event_name:
                    # 发送所有步骤摘要
                    yield f"event: steps_summary\ndata: {json.dumps({
                        'total_steps': len(steps),
                        'steps': steps
                    })}\n\n"

                    # 发送最终答案
                    output = event_data.get('output', {})
                    yield f"event: done\ndata: {json.dumps({
                        'answer': output.get('output', '')
                    })}\n\n"

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
 * React Agent 流式执行
 * 文件: examples/streaming/frontend/ReactAgentStream.jsx
 */

import React, { useState } from 'react';

function AgentStreamExecutor() {
    const [question, setQuestion] = useState('');
    const [steps, setSteps] = useState([]);
    const [thinking, setThinking] = useState('');
    const [answer, setAnswer] = useState('');
    const [isExecuting, setIsExecuting] = useState(false);
    const [currentTool, setCurrentTool] = useState(null);

    const executeAgent = async () => {
        if (!question.trim() || isExecuting) return;

        // 重置状态
        setSteps([]);
        setThinking('');
        setAnswer('');
        setCurrentTool(null);
        setIsExecuting(true);

        try {
            const response = await fetch('http://localhost:8000/agent-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            const readStream = () => {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        setIsExecuting(false);
                        return;
                    }

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    let currentEvent = null;
                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            currentEvent = line.slice(7);
                        } else if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            handleEvent(currentEvent, data);
                        }
                    }

                    readStream();
                });
            };

            readStream();
        } catch (error) {
            console.error('Error:', error);
            setIsExecuting(false);
        }
    };

    const handleEvent = (eventType, data) => {
        switch (eventType) {
            case 'agent_thinking':
                setThinking('');
                break;

            case 'thinking':
                setThinking(prev => prev + data.content);
                break;

            case 'tool_start':
                setCurrentTool({
                    name: data.tool,
                    input: data.input,
                    status: 'running'
                });
                setSteps(prev => [...prev, {
                    tool: data.tool,
                    input: data.input,
                    status: 'running'
                }]);
                break;

            case 'tool_end':
                setCurrentTool(null);
                setSteps(prev => {
                    const newSteps = [...prev];
                    if (newSteps.length > 0) {
                        newSteps[newSteps.length - 1] = {
                            ...newSteps[newSteps.length - 1],
                            output: data.output,
                            status: 'completed'
                        };
                    }
                    return newSteps;
                });
                break;

            case 'done':
                setAnswer(data.answer);
                break;

            case 'error':
                console.error('Agent error:', data.error);
                break;
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h1>Agent 流式执行</h1>

            {/* 输入区域 */}
            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && executeAgent()}
                    placeholder="输入问题..."
                    disabled={isExecuting}
                    style={{ flex: 1, padding: '10px', fontSize: '16px' }}
                />
                <button
                    onClick={executeAgent}
                    disabled={isExecuting}
                    style={{ padding: '10px 20px', fontSize: '16px' }}
                >
                    {isExecuting ? '执行中...' : '执行'}
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* 执行步骤 */}
                <div>
                    <h3>执行步骤</h3>

                    {/* 思考过程 */}
                    {thinking && (
                        <div style={{
                            marginBottom: '15px',
                            padding: '15px',
                            background: '#fff3cd',
                            borderRadius: '8px'
                        }}>
                            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                                🤔 思考中...
                            </div>
                            <div style={{ fontSize: '14px' }}>{thinking}</div>
                        </div>
                    )}

                    {/* 工具调用步骤 */}
                    {steps.map((step, index) => (
                        <div
                            key={index}
                            style={{
                                marginBottom: '15px',
                                padding: '15px',
                                background: step.status === 'running' ? '#e3f2fd' : '#f5f5f5',
                                borderRadius: '8px',
                                borderLeft: `4px solid ${step.status === 'running' ? '#2196f3' : '#4caf50'}`
                            }}
                        >
                            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                                {step.status === 'running' ? '⚙️' : '✅'} 步骤 {index + 1}: {step.tool}
                            </div>
                            <div style={{ fontSize: '14px', marginBottom: '5px' }}>
                                <strong>输入:</strong> {JSON.stringify(step.input)}
                            </div>
                            {step.output && (
                                <div style={{ fontSize: '14px' }}>
                                    <strong>输出:</strong> {step.output}
                                </div>
                            )}
                        </div>
                    ))}

                    {/* 当前工具调用 */}
                    {currentTool && (
                        <div style={{
                            padding: '15px',
                            background: '#e3f2fd',
                            borderRadius: '8px',
                            animation: 'pulse 1.5s infinite'
                        }}>
                            <div style={{ fontWeight: 'bold' }}>
                                ⚙️ 正在调用: {currentTool.name}
                            </div>
                        </div>
                    )}
                </div>

                {/* 最终答案 */}
                <div>
                    <h3>最终答案</h3>
                    <div style={{
                        padding: '20px',
                        background: 'white',
                        border: '1px solid #ccc',
                        borderRadius: '8px',
                        minHeight: '300px',
                        lineHeight: '1.6'
                    }}>
                        {answer || (isExecuting ? '等待 Agent 完成...' : '等待执行')}
                    </div>
                </div>
            </div>

            <style>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
            `}</style>
        </div>
    );
}

export default AgentStreamExecutor;
```

---

## 5. 复杂 Agent 示例

```python
"""
复杂 Agent 示例 - 多工具协作
文件: examples/streaming/agent_stream_complex.py
"""

from langchain.agents import Tool

# 定义更多工具
@tool
def web_search(query: str) -> str:
    """搜索网页信息"""
    return f"网页搜索结果: {query}"

@tool
def database_query(sql: str) -> str:
    """查询数据库"""
    return f"数据库查询结果: {sql}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    return f"邮件已发送到 {to}"

@tool
def create_calendar_event(title: str, date: str) -> str:
    """创建日历事件"""
    return f"已创建事件: {title} 在 {date}"

# 创建复杂 Agent
complex_tools = [web_search, database_query, calculator, send_email, create_calendar_event]

complex_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个强大的AI助手,可以使用多个工具来完成复杂任务。

可用工具:
- web_search: 搜索网页信息
- database_query: 查询数据库
- calculator: 执行数学计算
- send_email: 发送邮件
- create_calendar_event: 创建日历事件

请根据用户的问题,合理使用这些工具来完成任务。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

complex_agent = create_openai_functions_agent(llm, complex_tools, complex_prompt)
complex_agent_executor = AgentExecutor(
    agent=complex_agent,
    tools=complex_tools,
    verbose=True,
    max_iterations=10
)

@app.post("/agent-complex")
async def agent_complex(question: str):
    """复杂 Agent 流式执行"""
    async def generate():
        try:
            iteration_count = 0

            async for event in complex_agent_executor.astream_events(
                {"input": question},
                version="v1"
            ):
                event_type = event['event']
                event_name = event['name']
                event_data = event['data']

                # 跟踪迭代次数
                if event_type == "on_chain_start" and "Agent" in event_name:
                    iteration_count += 1
                    yield f"event: iteration\ndata: {json.dumps({
                        'iteration': iteration_count
                    })}\n\n"

                # 工具调用
                elif event_type == "on_tool_start":
                    yield f"event: tool_start\ndata: {json.dumps({
                        'iteration': iteration_count,
                        'tool': event_name,
                        'input': event_data.get('input')
                    })}\n\n"

                elif event_type == "on_tool_end":
                    yield f"event: tool_end\ndata: {json.dumps({
                        'iteration': iteration_count,
                        'tool': event_name,
                        'output': event_data.get('output')
                    })}\n\n"

                # 最终答案
                elif event_type == "on_chain_end" and "AgentExecutor" in event_name:
                    output = event_data.get('output', {})
                    yield f"event: done\ndata: {json.dumps({
                        'answer': output.get('output', ''),
                        'total_iterations': iteration_count
                    })}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

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
Agent 流式执行单元测试
文件: tests/test_agent_stream.py
"""

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_agent_stream():
    """测试 Agent 流式执行"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/agent-stream",
            json={"question": "计算 123 + 456"}
        ) as response:
            assert response.status_code == 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    events.append(line[7:])

            # 验证事件顺序
            assert "start" in events
            assert "tool_start" in events
            assert "tool_end" in events
            assert "done" in events
```

---

## 总结

**本节要点:**

1. **Agent 流式执行**: 使用 astream_events() 捕获所有事件
2. **思考过程可视化**: 显示 Agent 的思考和决策过程
3. **工具调用追踪**: 实时显示工具调用的输入和输出
4. **步骤追踪**: 记录和显示 Agent 的执行步骤
5. **复杂场景**: 支持多工具协作和多轮迭代

**关键代码:**
```python
async for event in agent_executor.astream_events({"input": question}, version="v1"):
    if event['event'] == "on_tool_start":
        yield f"event: tool_start\ndata: {json.dumps(event['data'])}\n\n"
```

**下一步:**

掌握了 Agent 流式执行后,可以学习:
- 前端集成示例
- 错误处理与重试
- 性能优化

---

**记住:** Agent 流式执行是 AI Agent 可视化的关键,让用户看到 Agent 的思考和执行过程,大幅提升用户体验和信任度。
