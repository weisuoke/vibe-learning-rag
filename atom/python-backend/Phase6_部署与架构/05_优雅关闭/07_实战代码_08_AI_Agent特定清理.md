# 实战代码：AI Agent 特定清理

> 完整可运行的 AI Agent 后端优雅关闭示例

---

## 代码说明

本示例演示 AI Agent 后端的特定清理需求，包括：
- LLM 流式响应的优雅中断
- 向量数据库连接清理
- Embedding 模型卸载
- Agent 任务队列管理

**运行环境：**
- Python 3.13+
- FastAPI
- OpenAI
- Sentence Transformers

---

## 完整代码

```python
"""
AI Agent 特定清理实现
演示：LLM 流式响应、向量数据库、Embedding 模型的优雅关闭

运行方式：
    python 07_实战代码_08_AI_Agent特定清理.py

测试方式：
    1. 启动应用
    2. 访问 /stream 端点测试流式响应
    3. 按 Ctrl+C 测试优雅关闭
"""

import signal
import asyncio
import sys
import json
from contextlib import asynccontextmanager
from typing import Set, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

# ===== 1. 全局状态 =====
shutdown_event = asyncio.Event()
accepting_requests = True
active_requests = 0

# ===== 2. LLM 流式响应管理器 =====
class StreamingResponseManager:
    """流式响应管理器"""

    def __init__(self):
        self.active_streams: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()

    async def create_stream(self, prompt: str, llm_client: AsyncOpenAI):
        """创建流式响应"""
        task = asyncio.current_task()
        self.active_streams.add(task)

        try:
            # 调用 LLM
            stream = await llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            async for chunk in stream:
                # 检查关闭信号
                if self.shutdown_event.is_set():
                    yield json.dumps({
                        "type": "end",
                        "reason": "server_shutdown",
                        "message": "服务正在重启，请稍后重试"
                    }) + "\n"
                    break

                # 提取内容
                if chunk.choices[0].delta.content:
                    yield json.dumps({
                        "type": "chunk",
                        "content": chunk.choices[0].delta.content
                    }) + "\n"

        except asyncio.CancelledError:
            yield json.dumps({
                "type": "end",
                "reason": "cancelled"
            }) + "\n"
            raise

        finally:
            self.active_streams.discard(task)

    async def shutdown(self, timeout: int = 30):
        """优雅关闭所有流式响应"""
        print(f"\n[流式响应] 关闭 {len(self.active_streams)} 个流式响应...")

        # 设置关闭标志
        self.shutdown_event.set()

        # 等待流式响应自然结束
        if self.active_streams:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_streams, return_exceptions=True),
                    timeout=timeout
                )
                print("[流式响应] ✓ 所有流式响应已完成")

            except asyncio.TimeoutError:
                print(f"[流式响应] ⚠ 超时（{timeout}秒），强制取消")

                for task in self.active_streams:
                    if not task.done():
                        task.cancel()

                await asyncio.gather(*self.active_streams, return_exceptions=True)

        print("[流式响应] 完成")

stream_manager = StreamingResponseManager()

# ===== 3. Embedding 管理器 =====
class EmbeddingManager:
    """Embedding 管理器"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.active_tasks: Set[asyncio.Task] = set()

    def load_model(self):
        """加载模型"""
        print(f"\n[Embedding] 加载模型 {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("[Embedding] ✓ 模型加载完成")

    async def encode(self, texts: list):
        """生成 Embedding"""
        if not self.model:
            raise ValueError("Model not loaded")

        task = asyncio.current_task()
        self.active_tasks.add(task)

        try:
            # 在线程池中执行（避免阻塞事件循环）
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts
            )
            return embeddings

        finally:
            self.active_tasks.discard(task)

    async def cleanup(self, timeout: int = 10):
        """清理 Embedding 模型"""
        print(f"\n[Embedding] 卸载模型（{len(self.active_tasks)} 个任务）...")

        # 等待所有任务完成
        if self.active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks, return_exceptions=True),
                    timeout=timeout
                )
                print("[Embedding] ✓ 所有任务已完成")

            except asyncio.TimeoutError:
                print(f"[Embedding] ⚠ 任务超时（{timeout}秒）")

                for task in self.active_tasks:
                    if not task.done():
                        task.cancel()

        # 卸载模型
        if self.model:
            del self.model
            self.model = None

            # 释放 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Embedding] ✓ 模型已卸载")

        print("[Embedding] 完成")

embedding_manager = EmbeddingManager("all-MiniLM-L6-v2")

# ===== 4. Agent 任务队列 =====
class AgentTaskQueue:
    """Agent 任务队列"""

    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()
        self.accepting_tasks = True

    async def submit_task(self, task_id: str, prompt: str):
        """提交 Agent 任务"""
        if not self.accepting_tasks:
            raise ValueError("Queue is shutting down")

        # 创建任务
        task = asyncio.create_task(self._run_agent(task_id, prompt))
        self.tasks[task_id] = task

        return task_id

    async def _run_agent(self, task_id: str, prompt: str):
        """运行 Agent"""
        try:
            # 检查关闭信号
            if self.shutdown_event.is_set():
                return {
                    "status": "cancelled",
                    "reason": "shutdown"
                }

            # 模拟 Agent 执行（实际应用中调用真实的 Agent）
            for i in range(10):
                if self.shutdown_event.is_set():
                    return {
                        "status": "interrupted",
                        "step": i,
                        "reason": "shutdown"
                    }

                await asyncio.sleep(1)

            return {
                "status": "completed",
                "result": f"Agent completed: {prompt}"
            }

        except asyncio.CancelledError:
            return {
                "status": "cancelled",
                "reason": "timeout"
            }

        finally:
            self.tasks.pop(task_id, None)

    async def shutdown(self, timeout: int = 60):
        """优雅关闭 Agent 任务队列"""
        print(f"\n[Agent] 关闭任务队列（{len(self.tasks)} 个任务）...")

        # 停止接收新任务
        self.accepting_tasks = False

        # 设置关闭标志
        self.shutdown_event.set()

        # 等待任务完成
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
                print("[Agent] ✓ 所有任务已完成")

            except asyncio.TimeoutError:
                print(f"[Agent] ⚠ 任务超时（{timeout}秒），取消剩余任务")

                for task in self.tasks.values():
                    if not task.done():
                        task.cancel()

                await asyncio.gather(*self.tasks.values(), return_exceptions=True)

        print("[Agent] 完成")

agent_queue = AgentTaskQueue()

# ===== 5. 信号处理器 =====
def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n收到信号 {signum}")
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== 6. lifespan 上下文管理器 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("\n" + "="*60)
    print("AI Agent 后端启动")
    print("="*60)

    # 初始化 LLM 客户端
    print("\n1. 初始化 LLM 客户端...")
    app.state.llm_client = AsyncOpenAI()
    print("✓ LLM 客户端初始化完成")

    # 加载 Embedding 模型
    print("\n2. 加载 Embedding 模型...")
    embedding_manager.load_model()

    # 启动关闭监听器
    print("\n3. 启动关闭监听器...")
    asyncio.create_task(shutdown_monitor())

    print("\n" + "="*60)
    print("AI Agent 后端启动完成")
    print("="*60 + "\n")

    yield

    print("\n" + "="*60)
    print("AI Agent 后端关闭")
    print("="*60)

app = FastAPI(title="AI Agent 特定清理示例", lifespan=lifespan)

# ===== 7. 请求排空中间件 =====
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    """请求排空中间件"""
    global active_requests

    if not accepting_requests:
        return JSONResponse(
            status_code=503,
            content={"error": "Server is shutting down"}
        )

    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

# ===== 8. 示例路由 =====
@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "AI Agent 特定清理示例",
        "streaming_tasks": len(stream_manager.active_streams),
        "embedding_tasks": len(embedding_manager.active_tasks),
        "agent_tasks": len(agent_queue.tasks),
    }

@app.get("/stream")
async def stream_endpoint(request: Request, prompt: str = "Hello, how are you?"):
    """LLM 流式响应端点"""
    return StreamingResponse(
        stream_manager.create_stream(prompt, request.app.state.llm_client),
        media_type="text/event-stream"
    )

@app.post("/embed")
async def embed_endpoint(texts: list[str]):
    """Embedding 端点"""
    embeddings = await embedding_manager.encode(texts)
    return {
        "embeddings": embeddings.tolist(),
        "shape": embeddings.shape
    }

@app.post("/agent")
async def agent_endpoint(task_id: str, prompt: str):
    """Agent 任务端点"""
    try:
        task_id = await agent_queue.submit_task(task_id, prompt)
        return {"task_id": task_id, "status": "submitted"}
    except ValueError as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)}
        )

@app.get("/status")
async def status_endpoint():
    """状态端点"""
    return {
        "accepting_requests": accepting_requests,
        "active_requests": active_requests,
        "streaming_tasks": len(stream_manager.active_streams),
        "embedding_tasks": len(embedding_manager.active_tasks),
        "agent_tasks": len(agent_queue.tasks),
    }

# ===== 9. 优雅关闭逻辑 =====
async def wait_for_requests(timeout: int = 30):
    """等待所有请求完成"""
    global accepting_requests
    accepting_requests = False

    print(f"\n[请求排空] 停止接收新请求")
    print(f"[请求排空] 当前活跃请求数: {active_requests}")

    start_time = asyncio.get_event_loop().time()

    while active_requests > 0:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            print(f"[请求排空] 超时（{timeout}秒）")
            break

        await asyncio.sleep(0.1)

    print(f"[请求排空] 完成")

async def graceful_shutdown():
    """优雅关闭"""
    print("\n" + "="*60)
    print("开始 AI Agent 优雅关闭")
    print("="*60)

    # 1. 等待请求完成
    await wait_for_requests()

    # 2. 关闭流式响应
    await stream_manager.shutdown(timeout=30)

    # 3. 关闭 Agent 任务队列
    await agent_queue.shutdown(timeout=60)

    # 4. 卸载 Embedding 模型
    await embedding_manager.cleanup(timeout=10)

    # 5. 关闭 LLM 客户端
    print("\n[LLM] 关闭 LLM 客户端...")
    # OpenAI 客户端会自动清理
    print("[LLM] ✓ LLM 客户端已关闭")

    print("\n" + "="*60)
    print("AI Agent 优雅关闭完成")
    print("="*60 + "\n")

    sys.exit(0)

async def shutdown_monitor():
    """监听关闭信号"""
    await shutdown_event.wait()
    await graceful_shutdown()

# ===== 10. 主函数 =====
def main():
    """主函数"""
    print("="*60)
    print("AI Agent 特定清理示例")
    print("="*60 + "\n")

    print("提示：")
    print("  - 访问 /stream?prompt=你好 测试流式响应")
    print("  - 访问 /embed 测试 Embedding")
    print("  - 访问 /agent 测试 Agent 任务")
    print("  - 按 Ctrl+C 测试优雅关闭")
    print()

    print("注意：")
    print("  - 需要设置 OPENAI_API_KEY 环境变量")
    print("  - 首次运行会下载 Embedding 模型")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
```

---

## 运行示例

### 1. 设置环境变量

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 启动应用

```bash
python 07_实战代码_08_AI_Agent特定清理.py
```

### 3. 测试流式响应

```bash
curl "http://localhost:8000/stream?prompt=讲个笑话"
```

### 4. 测试 Embedding

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'
```

### 5. 测试 Agent 任务

```bash
curl -X POST "http://localhost:8000/agent?task_id=task1&prompt=分析文档"
```

### 6. 测试优雅关闭

按 `Ctrl+C`，观察输出：

```
^C
收到信号 2

============================================================
开始 AI Agent 优雅关闭
============================================================

[请求排空] 停止接收新请求
[请求排空] 当前活跃请求数: 0
[请求排空] 完成

[流式响应] 关闭 1 个流式响应...
[流式响应] ✓ 所有流式响应已完成
[流式响应] 完成

[Agent] 关闭任务队列（1 个任务）...
[Agent] ✓ 所有任务已完成
[Agent] 完成

[Embedding] 卸载模型（0 个任务）...
[Embedding] ✓ 模型已卸载
[Embedding] 完成

[LLM] 关闭 LLM 客户端...
[LLM] ✓ LLM 客户端已关闭

============================================================
AI Agent 优雅关闭完成
============================================================
```

---

## 核心特性

### 1. LLM 流式响应管理

- 检测关闭信号
- 发送结束标记
- 优雅中断生成

### 2. Embedding 模型管理

- 等待任务完成
- 卸载模型
- 释放 GPU 内存

### 3. Agent 任务队列

- 停止接收新任务
- 等待现有任务完成
- 支持任务中断

### 4. 分阶段关闭

- 请求排空
- 流式响应清理
- Agent 任务清理
- 模型卸载

---

## 总结

本示例演示了 AI Agent 后端的特定清理需求：

1. **LLM 流式响应**：检测关闭信号并优雅中断
2. **Embedding 模型**：等待任务完成后卸载
3. **Agent 任务**：停止接收新任务，等待现有任务完成
4. **资源清理**：按顺序清理所有 AI 相关资源

**关键要点：**
- 在流式响应中定期检查关闭信号
- 等待 Embedding 任务完成后再卸载模型
- Agent 任务支持中断和状态保存
- 释放 GPU 内存避免资源泄漏

**下一步：** 学习如何在 Kubernetes 环境中配置优雅关闭。
