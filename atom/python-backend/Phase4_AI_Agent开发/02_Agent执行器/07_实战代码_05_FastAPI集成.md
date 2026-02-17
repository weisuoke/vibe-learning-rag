# Agent执行器 - 实战代码5：FastAPI集成

> 将 Agent 集成到 FastAPI 端点

---

## 学习目标

- 将 Agent 集成到 FastAPI
- 实现流式返回
- 处理异步执行
- 实现后台任务

---

## 示例1：基础集成

```python
"""
基础 FastAPI 集成
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

app = FastAPI()

# 定义工具
@tool
def query_order(order_id: str) -> str:
    """查询订单信息"""
    return f"订单 {order_id} 已发货"

# 创建 Agent（全局变量）
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [query_order]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 请求模型
class Question(BaseModel):
    question: str

# API 端点
@app.post("/agent")
async def run_agent(question: Question):
    try:
        result = await executor.ainvoke({"input": question.question})
        return {"answer": result["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 示例2：流式输出

```python
"""
流式输出
实时显示 Agent 执行过程
"""

from fastapi.responses import StreamingResponse
import json

@app.post("/agent/stream")
async def run_agent_stream(question: Question):
    """流式返回 Agent 执行过程"""

    async def generate():
        try:
            # 使用 astream_events 获取流式输出
            async for event in executor.astream_events(
                {"input": question.question},
                version="v1"
            ):
                # 发送事件
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 示例3：后台任务

```python
"""
后台任务
处理长时间运行的 Agent
"""

from fastapi import BackgroundTasks
from uuid import uuid4

# 任务存储
tasks = {}

@app.post("/agent/async")
async def run_agent_async(question: Question, background_tasks: BackgroundTasks):
    """异步运行 Agent"""

    task_id = str(uuid4())
    tasks[task_id] = {"status": "pending", "result": None}

    async def run_task():
        try:
            result = await executor.ainvoke({"input": question.question})
            tasks[task_id] = {
                "status": "completed",
                "result": result["output"]
            }
        except Exception as e:
            tasks[task_id] = {
                "status": "failed",
                "error": str(e)
            }

    background_tasks.add_task(run_task)

    return {"task_id": task_id}

@app.get("/agent/async/{task_id}")
async def get_task_result(task_id: str):
    """获取任务结果"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    return tasks[task_id]
```

---

## 示例4：完整生产实现

```python
"""
生产级 FastAPI 集成
包含超时、错误处理、监控
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger(__name__)

# 创建 Agent（全局变量）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    verbose=False,
)

class Question(BaseModel):
    question: str
    timeout: float = 30.0

@app.post("/agent")
async def run_agent(question: Question):
    """运行 Agent"""
    try:
        # 1. 超时控制
        result = await asyncio.wait_for(
            executor.ainvoke({"input": question.question}),
            timeout=question.timeout
        )

        # 2. 记录日志
        iterations = len(result.get('intermediate_steps', []))
        logger.info(f"问题：{question.question}")
        logger.info(f"迭代次数：{iterations}")

        # 3. 告警
        if iterations >= 10:
            logger.warning(f"达到最大迭代次数")

        # 4. 返回结果
        return {
            "answer": result["output"],
            "iterations": iterations,
        }

    except asyncio.TimeoutError:
        logger.error(f"超时：{question.question}")
        raise HTTPException(status_code=504, detail="执行超时")

    except Exception as e:
        logger.error(f"失败：{str(e)}")
        raise HTTPException(status_code=500, detail="执行失败")

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
```

---

## 最佳实践

### 1. 全局 Agent

```python
# 在应用启动时创建 Agent
executor = AgentExecutor(agent=agent, tools=tools)

# 在端点中使用
@app.post("/agent")
async def run_agent(question: Question):
    result = await executor.ainvoke({"input": question.question})
    return {"answer": result["output"]}
```

### 2. 异步调用

```python
# 使用 ainvoke 而不是 invoke
result = await executor.ainvoke({"input": question})
```

### 3. 错误处理

```python
try:
    result = await executor.ainvoke({"input": question})
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

### 4. 超时控制

```python
result = await asyncio.wait_for(
    executor.ainvoke({"input": question}),
    timeout=30.0
)
```

---

## 学习检查清单

- [ ] 将 Agent 集成到 FastAPI
- [ ] 实现基础 API 端点
- [ ] 实现流式输出
- [ ] 实现后台任务
- [ ] 添加超时控制
- [ ] 添加错误处理
- [ ] 添加健康检查

---

**版本：** v1.0
**最后更新：** 2026-02-12
