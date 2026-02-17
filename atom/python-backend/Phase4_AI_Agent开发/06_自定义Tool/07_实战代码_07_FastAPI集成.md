# 自定义Tool - 实战代码07：FastAPI集成

> 完整可运行的FastAPI集成示例，将自定义Tool集成到Web API

---

## 概述

将自定义Tool集成到FastAPI，提供HTTP API接口。本文提供完整的生产级实现。

**技术要点：**
- FastAPI异步端点
- 依赖注入
- 流式响应
- 错误处理
- CORS配置

---

## 完整示例：Agent API服务

```python
"""
Agent API服务
演示：将自定义Tool集成到FastAPI
"""

import os
import asyncio
import asyncpg
import httpx
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# ===== 全局资源 =====
app = FastAPI(title="AI Agent API", version="1.0.0")
db_pool = None
http_client = None
agent_executor = None

# ===== CORS配置 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 生命周期管理 =====
@app.on_event("startup")
async def startup():
    """应用启动时初始化资源"""
    global db_pool, http_client, agent_executor

    # 初始化数据库连接池
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=5,
        max_size=20
    )

    # 初始化HTTP客户端
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=10),
        timeout=httpx.Timeout(10.0)
    )

    # 初始化Agent
    agent_executor = await create_agent()

    print("✅ 应用启动成功")

@app.on_event("shutdown")
async def shutdown():
    """应用关闭时清理资源"""
    global db_pool, http_client

    if db_pool:
        await db_pool.close()
    if http_client:
        await http_client.aclose()

    print("✅ 应用关闭完成")

# ===== 依赖注入 =====
async def get_db_pool():
    """获取数据库连接池"""
    if db_pool is None:
        raise HTTPException(status_code=500, detail="数据库未初始化")
    return db_pool

async def get_http_client():
    """获取HTTP客户端"""
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP客户端未初始化")
    return http_client

# ===== 定义Tools =====
@tool
async def get_order_details(order_id: str) -> str:
    """获取订单详细信息"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT order_id, status, total_amount FROM orders WHERE order_id = $1",
                order_id
            )
            if not row:
                return f"订单{order_id}不存在"
            return f"订单{row['order_id']}：状态{row['status']}，金额¥{row['total_amount']}"
    except Exception as e:
        return f"查询失败：{str(e)}"

@tool
async def get_weather(city: str) -> str:
    """获取天气信息"""
    try:
        # 模拟天气查询
        return f"{city}的天气：晴天，25°C"
    except Exception as e:
        return f"查询失败：{str(e)}"

# ===== 创建Agent =====
async def create_agent():
    """创建Agent执行器"""
    tools = [get_order_details, get_weather]

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，可以查询订单和天气信息。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )

# ===== API模型 =====
class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(
        description="用户问题",
        min_length=1,
        max_length=500
    )

class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(description="回答")
    success: bool = Field(description="是否成功")

# ===== API端点 =====
@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI Agent API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "database": "connected" if db_pool else "disconnected",
        "agent": "ready" if agent_executor else "not ready"
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    """Agent问答（非流式）"""
    try:
        if not agent_executor:
            raise HTTPException(status_code=500, detail="Agent未初始化")

        # 调用Agent
        result = await agent_executor.ainvoke({"input": request.question})

        return QueryResponse(
            answer=result["output"],
            success=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 流式响应 =====
class StreamingCallbackHandler(BaseCallbackHandler):
    """流式回调处理器"""

    def __init__(self):
        self.tokens = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        """接收新token"""
        await self.tokens.put(token)

    async def on_llm_end(self, response, **kwargs):
        """LLM结束"""
        await self.tokens.put(None)  # 结束标记

@app.post("/ask/stream")
async def ask_agent_stream(request: QueryRequest):
    """Agent问答（流式）"""

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # 创建流式回调
            callback = StreamingCallbackHandler()

            # 创建带回调的Agent
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                streaming=True,
                callbacks=[callback]
            )

            tools = [get_order_details, get_weather]
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools)

            # 启动Agent（异步）
            task = asyncio.create_task(
                executor.ainvoke({"input": request.question})
            )

            # 流式输出tokens
            while True:
                token = await callback.tokens.get()
                if token is None:
                    break
                yield f"data: {token}\n\n"

            # 等待Agent完成
            await task

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# ===== 直接调用Tool端点 =====
@app.get("/tools/order/{order_id}")
async def get_order(order_id: str):
    """直接查询订单"""
    try:
        result = await get_order_details.ainvoke({"order_id": order_id})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/weather/{city}")
async def get_city_weather(city: str):
    """直接查询天气"""
    try:
        result = await get_weather.ainvoke({"city": city})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 运行服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

## 测试API

### 1. 启动服务

```bash
# 运行服务
python main.py

# 或使用uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 测试端点

```bash
# 健康检查
curl http://localhost:8000/health

# 非流式问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "北京今天天气怎么样？"}'

# 流式问答
curl -N http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "北京今天天气怎么样？"}'

# 直接调用Tool
curl http://localhost:8000/tools/weather/北京
curl http://localhost:8000/tools/order/ORD-123456
```

---

## 前端集成示例

### JavaScript/TypeScript

```typescript
// 非流式请求
async function askAgent(question: string) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  const data = await response.json();
  return data.answer;
}

// 流式请求
async function askAgentStream(question: string, onToken: (token: string) => void) {
  const response = await fetch('http://localhost:8000/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const token = line.slice(6);
        if (token === '[DONE]') return;
        if (token.startsWith('[ERROR]')) throw new Error(token);
        onToken(token);
      }
    }
  }
}

// 使用示例
askAgent("北京天气怎么样？").then(answer => console.log(answer));

askAgentStream("北京天气怎么样？", token => {
  process.stdout.write(token);
});
```

---

## 最佳实践

### 1. 资源管理

```python
# ✅ 在startup/shutdown中管理资源
@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(...)

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()
```

### 2. 依赖注入

```python
# ✅ 使用依赖注入
async def get_db_pool():
    return db_pool

@app.post("/query")
async def query(pool = Depends(get_db_pool)):
    async with pool.acquire() as conn:
        ...
```

### 3. 错误处理

```python
# ✅ 统一错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )
```

### 4. 流式响应

```python
# ✅ 使用StreamingResponse
async def generate():
    for token in tokens:
        yield f"data: {token}\n\n"

return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 总结

FastAPI集成的核心：
1. **资源管理**：startup/shutdown管理连接池
2. **依赖注入**：复用资源
3. **异步端点**：async def
4. **流式响应**：StreamingResponse

**记住：** FastAPI天然支持异步，充分利用异步Tool的优势！
