# 核心概念：AI Agent 特定清理

> 深入理解 AI Agent 后端的特殊清理需求：LLM 流式响应、向量数据库、Embedding 模型

---

## 什么是 AI Agent 特定清理？

**AI Agent 特定清理** 是指在 AI Agent 后端优雅关闭时，需要处理的特殊资源和任务，这些资源和任务在传统 Web 应用中不存在。

**核心资源：**
1. **LLM 流式响应**：正在进行的 LLM 生成任务
2. **向量数据库连接**：Milvus、Qdrant、Pinecone 等
3. **Embedding 模型**：加载在内存中的模型
4. **Agent 任务队列**：正在执行的 Agent 任务
5. **LLM 客户端**：OpenAI、Anthropic 等客户端

**类比：**
- **前端视角**：类似于清理 WebSocket 连接、取消正在进行的 API 请求
- **日常视角**：类似于关闭正在播放的视频、保存正在编辑的文档

---

## LLM 流式响应的优雅中断

### 1. 问题场景

```python
# LLM 流式响应正在进行
@app.get("/stream")
async def stream_llm():
    async def generate():
        async for chunk in llm.astream("分析这份文档..."):
            yield chunk  # 正在输出...

    return StreamingResponse(generate())

# 问题：
# 1. 服务收到 SIGTERM 信号
# 2. 流式响应被强制中断
# 3. 用户看到一半的内容
# 4. 没有结束标记
# 5. LLM API 已扣费但结果不完整
```

### 2. 解决方案：检测关闭信号

```python
"""
LLM 流式响应的优雅中断
演示：检测关闭信号并发送结束标记
"""

import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
shutdown_event = asyncio.Event()

@app.get("/stream")
async def stream_llm(prompt: str):
    """LLM 流式响应"""
    async def generate():
        try:
            async for chunk in llm.astream(prompt):
                # 检查关闭信号
                if shutdown_event.is_set():
                    # 发送结束标记
                    yield json.dumps({
                        "type": "end",
                        "reason": "server_shutdown",
                        "message": "服务正在重启，请稍后重试"
                    }) + "\n"
                    break

                # 正常输出
                yield json.dumps({
                    "type": "chunk",
                    "content": chunk
                }) + "\n"

        except asyncio.CancelledError:
            # 任务被取消
            yield json.dumps({
                "type": "end",
                "reason": "cancelled"
            }) + "\n"
            raise

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 3. 保存已生成的内容

```python
"""
保存已生成的内容
演示：在中断时保存部分结果
"""

async def stream_with_state_saving(
    prompt: str,
    user_id: str,
    conversation_id: str
):
    """带状态保存的流式响应"""
    generated_content = []

    try:
        async for chunk in llm.astream(prompt):
            # 检查关闭信号
            if shutdown_event.is_set():
                # 保存已生成的内容
                await save_partial_response(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content="".join(generated_content),
                    status="interrupted"
                )

                yield {
                    "type": "end",
                    "reason": "shutdown",
                    "saved": True
                }
                break

            # 记录内容
            generated_content.append(chunk)

            yield {
                "type": "chunk",
                "content": chunk
            }

    except asyncio.CancelledError:
        # 保存状态
        await save_partial_response(
            user_id=user_id,
            conversation_id=conversation_id,
            content="".join(generated_content),
            status="cancelled"
        )
        raise

async def save_partial_response(
    user_id: str,
    conversation_id: str,
    content: str,
    status: str
):
    """保存部分响应"""
    await db.execute(
        """
        INSERT INTO partial_responses
        (user_id, conversation_id, content, status, created_at)
        VALUES (:user_id, :conversation_id, :content, :status, NOW())
        """,
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "content": content,
            "status": status,
        }
    )
```

### 4. 管理所有流式响应

```python
"""
流式响应管理器
演示：跟踪和管理所有流式响应
"""

from typing import Set
import asyncio

class StreamingResponseManager:
    """流式响应管理器"""

    def __init__(self):
        self.active_streams: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()

    async def create_stream(self, prompt: str):
        """创建流式响应"""
        task = asyncio.current_task()
        self.active_streams.add(task)

        try:
            async for chunk in llm.astream(prompt):
                if self.shutdown_event.is_set():
                    yield {"type": "end", "reason": "shutdown"}
                    break

                yield {"type": "chunk", "content": chunk}

        finally:
            self.active_streams.discard(task)

    async def shutdown(self, timeout: int = 30):
        """优雅关闭所有流式响应"""
        print(f"关闭 {len(self.active_streams)} 个流式响应...")

        # 1. 设置关闭标志
        self.shutdown_event.set()

        # 2. 等待流式响应自然结束
        if self.active_streams:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_streams, return_exceptions=True),
                    timeout=timeout
                )
                print("所有流式响应已完成")

            except asyncio.TimeoutError:
                print(f"流式响应超时（{timeout}秒），强制取消")

                # 3. 强制取消
                for task in self.active_streams:
                    if not task.done():
                        task.cancel()

                await asyncio.gather(*self.active_streams, return_exceptions=True)

        print("流式响应管理器已关闭")

# 使用示例
stream_manager = StreamingResponseManager()

@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        stream_manager.create_stream(prompt),
        media_type="text/event-stream"
    )

# 优雅关闭时
await stream_manager.shutdown()
```

---

## 向量数据库连接清理

### 1. Milvus 连接清理

```python
"""
Milvus 向量数据库清理
演示：正确关闭 Milvus 连接
"""

from pymilvus import connections, Collection

# 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

async def cleanup_milvus():
    """清理 Milvus 连接"""
    print("关闭 Milvus 连接...")

    # 1. 等待所有查询完成
    # Milvus 客户端会自动等待

    # 2. 断开连接
    connections.disconnect("default")

    print("Milvus 连接已关闭")
```

### 2. Qdrant 连接清理

```python
"""
Qdrant 向量数据库清理
演示：正确关闭 Qdrant 连接
"""

from qdrant_client import QdrantClient
from qdrant_client.http import AsyncQdrantClient

# 创建客户端
qdrant_client = AsyncQdrantClient(
    host="localhost",
    port=6333
)

async def cleanup_qdrant():
    """清理 Qdrant 连接"""
    print("关闭 Qdrant 连接...")

    # 关闭客户端
    await qdrant_client.close()

    print("Qdrant 连接已关闭")
```

### 3. 等待向量查询完成

```python
"""
等待向量查询完成
演示：确保所有查询完成后再关闭
"""

class VectorDBManager:
    """向量数据库管理器"""

    def __init__(self):
        self.active_queries: Set[asyncio.Task] = set()
        self.client = None

    async def search(self, query_vector: list, top_k: int = 10):
        """向量检索"""
        task = asyncio.current_task()
        self.active_queries.add(task)

        try:
            results = await self.client.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=top_k
            )
            return results

        finally:
            self.active_queries.discard(task)

    async def cleanup(self, timeout: int = 10):
        """清理向量数据库连接"""
        print(f"关闭向量数据库（{len(self.active_queries)} 个查询）...")

        # 1. 等待所有查询完成
        if self.active_queries:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_queries, return_exceptions=True),
                    timeout=timeout
                )
                print("所有查询已完成")

            except asyncio.TimeoutError:
                print(f"查询超时（{timeout}秒），取消剩余查询")

                for task in self.active_queries:
                    if not task.done():
                        task.cancel()

        # 2. 关闭连接
        await self.client.close()

        print("向量数据库连接已关闭")
```

---

## Embedding 模型卸载

### 1. SentenceTransformer 模型卸载

```python
"""
SentenceTransformer 模型卸载
演示：正确卸载 Embedding 模型
"""

from sentence_transformers import SentenceTransformer
import torch

# 加载模型
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def cleanup_embedding_model():
    """卸载 Embedding 模型"""
    print("卸载 Embedding 模型...")

    # 1. 删除模型对象
    global embedding_model
    del embedding_model

    # 2. 释放 GPU 内存（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Embedding 模型已卸载")
```

### 2. 等待 Embedding 任务完成

```python
"""
等待 Embedding 任务完成
演示：确保所有 Embedding 任务完成后再卸载模型
"""

class EmbeddingManager:
    """Embedding 管理器"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.active_tasks: Set[asyncio.Task] = set()

    async def encode(self, texts: list):
        """生成 Embedding"""
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
        print(f"卸载 Embedding 模型（{len(self.active_tasks)} 个任务）...")

        # 1. 等待所有任务完成
        if self.active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks, return_exceptions=True),
                    timeout=timeout
                )
                print("所有 Embedding 任务已完成")

            except asyncio.TimeoutError:
                print(f"Embedding 任务超时（{timeout}秒）")

                for task in self.active_tasks:
                    if not task.done():
                        task.cancel()

        # 2. 卸载模型
        del self.model

        # 3. 释放 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Embedding 模型已卸载")
```

---

## Agent 任务队列清理

### 1. Agent 任务管理

```python
"""
Agent 任务队列管理
演示：管理 Agent 任务的优雅关闭
"""

from typing import Dict
import asyncio

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

            # 运行 Agent（可能需要数十秒）
            result = await agent.run(prompt)

            return {
                "status": "completed",
                "result": result
            }

        except asyncio.CancelledError:
            # 保存任务状态
            await self.save_task_state(task_id, "cancelled")

            return {
                "status": "cancelled",
                "reason": "timeout"
            }

        finally:
            # 移除任务
            self.tasks.pop(task_id, None)

    async def save_task_state(self, task_id: str, status: str):
        """保存任务状态"""
        await db.execute(
            """
            UPDATE agent_tasks
            SET status = :status, updated_at = NOW()
            WHERE task_id = :task_id
            """,
            {"task_id": task_id, "status": status}
        )

    async def shutdown(self, timeout: int = 60):
        """优雅关闭 Agent 任务队列"""
        print(f"关闭 Agent 任务队列（{len(self.tasks)} 个任务）...")

        # 1. 停止接收新任务
        self.accepting_tasks = False

        # 2. 设置关闭标志
        self.shutdown_event.set()

        # 3. 等待任务完成
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
                print("所有 Agent 任务已完成")

            except asyncio.TimeoutError:
                print(f"Agent 任务超时（{timeout}秒），取消剩余任务")

                # 4. 取消未完成的任务
                for task_id, task in self.tasks.items():
                    if not task.done():
                        await self.save_task_state(task_id, "timeout")
                        task.cancel()

                await asyncio.gather(*self.tasks.values(), return_exceptions=True)

        print("Agent 任务队列已关闭")
```

---

## LLM 客户端清理

### 1. OpenAI 客户端清理

```python
"""
OpenAI 客户端清理
演示：正确关闭 OpenAI 客户端
"""

from openai import AsyncOpenAI

# 创建客户端
openai_client = AsyncOpenAI()

async def cleanup_openai_client():
    """清理 OpenAI 客户端"""
    print("关闭 OpenAI 客户端...")

    # OpenAI 客户端基于 httpx
    await openai_client.close()

    print("OpenAI 客户端已关闭")
```

### 2. Anthropic 客户端清理

```python
"""
Anthropic 客户端清理
演示：正确关闭 Anthropic 客户端
"""

from anthropic import AsyncAnthropic

# 创建客户端
anthropic_client = AsyncAnthropic()

async def cleanup_anthropic_client():
    """清理 Anthropic 客户端"""
    print("关闭 Anthropic 客户端...")

    # Anthropic 客户端基于 httpx
    await anthropic_client.close()

    print("Anthropic 客户端已关闭")
```

---

## 完整示例：AI Agent 后端优雅关闭

```python
"""
AI Agent 后端完整优雅关闭
演示：处理所有 AI Agent 特定资源
"""

import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine
from redis.asyncio import Redis
from pymilvus import connections
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

class AIAgentResourceManager:
    """AI Agent 资源管理器"""

    def __init__(self):
        # 通用资源
        self.db_engine: Optional[AsyncEngine] = None
        self.redis_client: Optional[Redis] = None

        # AI 特定资源
        self.vector_db_connected = False
        self.embedding_model: Optional[SentenceTransformer] = None
        self.llm_client: Optional[AsyncOpenAI] = None

        # 任务管理
        self.stream_manager = StreamingResponseManager()
        self.agent_queue = AgentTaskQueue()
        self.embedding_manager: Optional[EmbeddingManager] = None

    async def initialize(self):
        """初始化所有资源"""
        print("========== 初始化 AI Agent 资源 ==========")

        # 1. 数据库
        print("初始化数据库...")
        self.db_engine = create_async_engine(DATABASE_URL)

        # 2. Redis
        print("初始化 Redis...")
        self.redis_client = Redis.from_url(REDIS_URL)

        # 3. 向量数据库
        print("连接向量数据库...")
        connections.connect(alias="default", host="localhost", port="19530")
        self.vector_db_connected = True

        # 4. Embedding 模型
        print("加载 Embedding 模型...")
        self.embedding_manager = EmbeddingManager("all-MiniLM-L6-v2")

        # 5. LLM 客户端
        print("初始化 LLM 客户端...")
        self.llm_client = AsyncOpenAI()

        print("========== AI Agent 资源初始化完成 ==========\n")

    async def cleanup(self, timeout: int = 60):
        """优雅关闭所有资源"""
        print("\n========== 开始 AI Agent 资源清理 ==========")

        start_time = asyncio.get_event_loop().time()

        def remaining_time():
            elapsed = asyncio.get_event_loop().time() - start_time
            return max(0, timeout - elapsed)

        # 1. 关闭流式响应（30秒）
        stage_timeout = min(30, remaining_time())
        print(f"\n1. 关闭流式响应（超时 {stage_timeout}秒）...")
        try:
            await asyncio.wait_for(
                self.stream_manager.shutdown(),
                timeout=stage_timeout
            )
            print("✓ 流式响应已关闭")
        except asyncio.TimeoutError:
            print("✗ 流式响应关闭超时")

        # 2. 关闭 Agent 任务队列（60秒）
        stage_timeout = min(60, remaining_time())
        print(f"\n2. 关闭 Agent 任务队列（超时 {stage_timeout}秒）...")
        try:
            await asyncio.wait_for(
                self.agent_queue.shutdown(),
                timeout=stage_timeout
            )
            print("✓ Agent 任务队列已关闭")
        except asyncio.TimeoutError:
            print("✗ Agent 任务队列关闭超时")

        # 3. 关闭 LLM 客户端（5秒）
        stage_timeout = min(5, remaining_time())
        print(f"\n3. 关闭 LLM 客户端（超时 {stage_timeout}秒）...")
        try:
            if self.llm_client:
                await asyncio.wait_for(
                    self.llm_client.close(),
                    timeout=stage_timeout
                )
            print("✓ LLM 客户端已关闭")
        except asyncio.TimeoutError:
            print("✗ LLM 客户端关闭超时")

        # 4. 卸载 Embedding 模型（10秒）
        stage_timeout = min(10, remaining_time())
        print(f"\n4. 卸载 Embedding 模型（超时 {stage_timeout}秒）...")
        try:
            if self.embedding_manager:
                await asyncio.wait_for(
                    self.embedding_manager.cleanup(),
                    timeout=stage_timeout
                )
            print("✓ Embedding 模型已卸载")
        except asyncio.TimeoutError:
            print("✗ Embedding 模型卸载超时")

        # 5. 关闭向量数据库（5秒）
        print(f"\n5. 关闭向量数据库...")
        try:
            if self.vector_db_connected:
                connections.disconnect("default")
            print("✓ 向量数据库已关闭")
        except Exception as e:
            print(f"✗ 向量数据库关闭失败: {e}")

        # 6. 关闭 Redis（5秒）
        stage_timeout = min(5, remaining_time())
        print(f"\n6. 关闭 Redis（超时 {stage_timeout}秒）...")
        try:
            if self.redis_client:
                await asyncio.wait_for(
                    self.redis_client.close(),
                    timeout=stage_timeout
                )
                await self.redis_client.connection_pool.disconnect()
            print("✓ Redis 已关闭")
        except asyncio.TimeoutError:
            print("✗ Redis 关闭超时")

        # 7. 关闭数据库（10秒）
        stage_timeout = min(10, remaining_time())
        print(f"\n7. 关闭数据库（超时 {stage_timeout}秒）...")
        try:
            if self.db_engine:
                await asyncio.wait_for(
                    self.db_engine.dispose(),
                    timeout=stage_timeout
                )
            print("✓ 数据库已关闭")
        except asyncio.TimeoutError:
            print("✗ 数据库关闭超时")

        print("\n========== AI Agent 资源清理完成 ==========")

# ===== 使用示例 =====
resource_manager = AIAgentResourceManager()

# 初始化
await resource_manager.initialize()

# 优雅关闭
await resource_manager.cleanup(timeout=120)
```

---

## 监控和告警

### 1. 监控 AI 资源状态

```python
def get_ai_resource_status():
    """获取 AI 资源状态"""
    return {
        "streaming_responses": len(stream_manager.active_streams),
        "agent_tasks": len(agent_queue.tasks),
        "embedding_tasks": len(embedding_manager.active_tasks),
        "vector_db_connected": vector_db_connected,
        "llm_client_connected": llm_client is not None,
    }
```

### 2. 记录清理日志

```python
import structlog

logger = structlog.get_logger()

async def cleanup_with_logging():
    """带日志的资源清理"""
    logger.info("开始 AI Agent 资源清理")

    # 记录清理前状态
    status = get_ai_resource_status()
    logger.info("清理前状态", status=status)

    # 执行清理
    await resource_manager.cleanup()

    # 记录清理后状态
    logger.info("AI Agent 资源清理完成")
```

---

## 总结

### 核心要点

1. **LLM 流式响应**：
   - 检测关闭信号
   - 发送结束标记
   - 保存已生成的内容

2. **向量数据库**：
   - 等待查询完成
   - 正确断开连接
   - 处理不同的向量数据库

3. **Embedding 模型**：
   - 等待任务完成
   - 卸载模型
   - 释放 GPU 内存

4. **Agent 任务队列**：
   - 停止接收新任务
   - 等待现有任务完成
   - 保存未完成任务状态

5. **LLM 客户端**：
   - 关闭 HTTP 连接
   - 释放资源

### 检查清单

- [ ] 实现流式响应的优雅中断
- [ ] 发送结束标记给客户端
- [ ] 保存已生成的内容
- [ ] 等待向量查询完成
- [ ] 关闭向量数据库连接
- [ ] 等待 Embedding 任务完成
- [ ] 卸载 Embedding 模型
- [ ] 释放 GPU 内存
- [ ] 停止接收新 Agent 任务
- [ ] 等待 Agent 任务完成
- [ ] 保存未完成任务状态
- [ ] 关闭 LLM 客户端

---

**下一步**：学习实战代码文件，了解如何在实际项目中实现优雅关闭。
