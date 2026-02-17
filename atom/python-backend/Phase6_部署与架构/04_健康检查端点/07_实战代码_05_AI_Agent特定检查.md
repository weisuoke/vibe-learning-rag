# 实战代码5：AI Agent特定检查

> LLM API、向量数据库、Embedding模型、Agent任务队列的健康检查

---

## 概述

本文提供 AI Agent 特定的健康检查实现，包括：
- LLM API 可用性检查（OpenAI/Anthropic）
- 向量数据库连接检查（pgvector）
- Embedding 模型加载状态检查
- Agent 任务队列健康检查
- RAG 系统端到端健康检查

---

## 完整代码

```python
"""
AI Agent 特定健康检查实现
演示：LLM API、向量数据库、Embedding 模型、任务队列检查
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import asyncio
import time
import os
from datetime import datetime

# ===== 1. 依赖导入 =====

# LLM 客户端
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# 数据库
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

# Redis
import redis.asyncio as redis

# Embedding 模型（可选）
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ===== 2. 配置 =====

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Anthropic 配置
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/dbname")

# Redis 配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ===== 3. 客户端初始化 =====

# OpenAI 客户端
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Anthropic 客户端
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# 数据库引擎
engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=20)

# Redis 客户端
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Embedding 模型（全局变量）
embedding_model = None

# ===== 4. FastAPI 应用 =====

app = FastAPI(title="AI Agent Health Check")

# ===== 5. 响应模型 =====

class LLMHealthResponse(BaseModel):
    """LLM API 健康检查响应"""
    healthy: bool
    duration_ms: int
    provider: str
    model: Optional[str] = None
    error: Optional[str] = None

class VectorDBHealthResponse(BaseModel):
    """向量数据库健康检查响应"""
    healthy: bool
    duration_ms: int
    extension_installed: bool
    table_exists: bool
    error: Optional[str] = None

class EmbeddingHealthResponse(BaseModel):
    """Embedding 模型健康检查响应"""
    healthy: bool
    loaded: bool
    model_name: Optional[str] = None
    error: Optional[str] = None

class RAGHealthResponse(BaseModel):
    """RAG 系统健康检查响应"""
    healthy: bool
    duration_ms: int
    checks: Dict[str, bool]
    error: Optional[str] = None

# ===== 6. LLM API 健康检查 =====

async def check_openai_api() -> LLMHealthResponse:
    """
    检查 OpenAI API 可用性

    使用最小的请求来检查 API 是否可用
    """
    start_time = time.time()

    try:
        # 发送最小的请求（1 token）
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1
            ),
            timeout=10.0
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return LLMHealthResponse(
            healthy=True,
            duration_ms=duration_ms,
            provider="openai",
            model=response.model
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        return LLMHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            provider="openai",
            error="Request timeout"
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return LLMHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            provider="openai",
            error=str(e)
        )

async def check_anthropic_api() -> LLMHealthResponse:
    """
    检查 Anthropic API 可用性

    使用最小的请求来检查 API 是否可用
    """
    start_time = time.time()

    try:
        # 发送最小的请求（1 token）
        response = await asyncio.wait_for(
            anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            ),
            timeout=10.0
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return LLMHealthResponse(
            healthy=True,
            duration_ms=duration_ms,
            provider="anthropic",
            model=response.model
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        return LLMHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            provider="anthropic",
            error="Request timeout"
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return LLMHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            provider="anthropic",
            error=str(e)
        )

@app.get("/health/llm/openai", response_model=LLMHealthResponse)
async def health_llm_openai():
    """OpenAI API 健康检查"""
    result = await check_openai_api()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

@app.get("/health/llm/anthropic", response_model=LLMHealthResponse)
async def health_llm_anthropic():
    """Anthropic API 健康检查"""
    result = await check_anthropic_api()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 7. 向量数据库健康检查 =====

async def check_vector_db() -> VectorDBHealthResponse:
    """
    检查向量数据库（pgvector）

    检查 pgvector 扩展是否安装，embeddings 表是否存在
    """
    start_time = time.time()

    try:
        async with AsyncSession(engine) as session:
            # 1. 检查 pgvector 扩展是否安装
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM pg_extension
                    WHERE extname = 'vector'
                )
            """))
            extension_installed = result.scalar()

            # 2. 检查 embeddings 表是否存在
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'embeddings'
                )
            """))
            table_exists = result.scalar()

            # 3. 如果表存在，执行简单的向量查询
            if table_exists:
                await asyncio.wait_for(
                    session.execute(text("""
                        SELECT id FROM embeddings
                        ORDER BY embedding <-> '[0,0,0]'::vector
                        LIMIT 1
                    """)),
                    timeout=3.0
                )

        duration_ms = int((time.time() - start_time) * 1000)

        return VectorDBHealthResponse(
            healthy=extension_installed and table_exists,
            duration_ms=duration_ms,
            extension_installed=extension_installed,
            table_exists=table_exists
        )

    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        return VectorDBHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            extension_installed=False,
            table_exists=False,
            error="Query timeout"
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return VectorDBHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            extension_installed=False,
            table_exists=False,
            error=str(e)
        )

@app.get("/health/vector_db", response_model=VectorDBHealthResponse)
async def health_vector_db():
    """向量数据库健康检查"""
    result = await check_vector_db()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 8. Embedding 模型健康检查 =====

async def check_embedding_model() -> EmbeddingHealthResponse:
    """
    检查 Embedding 模型是否已加载

    检查模型是否可用，并测试编码功能
    """
    global embedding_model

    try:
        # 检查模型是否已加载
        if embedding_model is None:
            return EmbeddingHealthResponse(
                healthy=False,
                loaded=False,
                error="Embedding model not loaded"
            )

        # 测试模型编码功能
        test_text = "test"
        embedding = embedding_model.encode(test_text)

        return EmbeddingHealthResponse(
            healthy=True,
            loaded=True,
            model_name=embedding_model.get_sentence_embedding_dimension() if hasattr(embedding_model, 'get_sentence_embedding_dimension') else None
        )

    except Exception as e:
        return EmbeddingHealthResponse(
            healthy=False,
            loaded=embedding_model is not None,
            error=str(e)
        )

@app.get("/health/embedding", response_model=EmbeddingHealthResponse)
async def health_embedding():
    """Embedding 模型健康检查"""
    result = await check_embedding_model()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 9. Agent 任务队列健康检查 =====

async def check_agent_task_queue() -> Dict:
    """
    检查 Agent 任务队列

    检查 Redis 队列长度，判断是否积压
    """
    try:
        # 检查任务队列长度
        queue_length = await redis_client.llen("agent_tasks")

        # 判断是否积压
        if queue_length > 1000:
            return {
                "healthy": False,
                "queue_length": queue_length,
                "error": "Task queue backlog"
            }

        return {
            "healthy": True,
            "queue_length": queue_length
        }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }

@app.get("/health/agent/queue")
async def health_agent_queue():
    """Agent 任务队列健康检查"""
    result = await check_agent_task_queue()

    if not result["healthy"]:
        raise HTTPException(503, detail=result)

    return result

# ===== 10. RAG 系统端到端健康检查 =====

async def check_rag_system() -> RAGHealthResponse:
    """
    RAG 系统端到端健康检查

    检查 RAG 系统的所有组件
    """
    start_time = time.time()

    try:
        # 并发检查所有组件
        results = await asyncio.gather(
            check_openai_api(),
            check_vector_db(),
            check_embedding_model(),
            return_exceptions=True
        )

        # 解析结果
        checks = {
            "llm_api": results[0].healthy if not isinstance(results[0], Exception) else False,
            "vector_db": results[1].healthy if not isinstance(results[1], Exception) else False,
            "embedding_model": results[2].healthy if not isinstance(results[2], Exception) else False,
        }

        duration_ms = int((time.time() - start_time) * 1000)

        # 判断整体健康状态
        healthy = all(checks.values())

        return RAGHealthResponse(
            healthy=healthy,
            duration_ms=duration_ms,
            checks=checks
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return RAGHealthResponse(
            healthy=False,
            duration_ms=duration_ms,
            checks={},
            error=str(e)
        )

@app.get("/health/rag", response_model=RAGHealthResponse)
async def health_rag():
    """RAG 系统端到端健康检查"""
    result = await check_rag_system()

    if not result.healthy:
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 11. 完整的 AI Agent 健康检查 =====

class AIAgentHealthResponse(BaseModel):
    """AI Agent 健康检查响应"""
    status: str  # healthy, degraded, unhealthy
    duration_ms: int
    checks: Dict[str, Dict]
    message: Optional[str] = None

async def check_ai_agent_complete() -> AIAgentHealthResponse:
    """
    完整的 AI Agent 健康检查

    检查所有 AI Agent 相关组件
    """
    start_time = time.time()

    try:
        # 并发检查所有组件
        results = await asyncio.gather(
            check_openai_api(),
            check_vector_db(),
            check_embedding_model(),
            check_agent_task_queue(),
            return_exceptions=True
        )

        # 解析结果
        checks = {
            "llm_api": {
                "healthy": results[0].healthy if not isinstance(results[0], Exception) else False,
                "duration_ms": results[0].duration_ms if not isinstance(results[0], Exception) else 0,
                "error": results[0].error if not isinstance(results[0], Exception) else str(results[0])
            },
            "vector_db": {
                "healthy": results[1].healthy if not isinstance(results[1], Exception) else False,
                "duration_ms": results[1].duration_ms if not isinstance(results[1], Exception) else 0,
                "error": results[1].error if not isinstance(results[1], Exception) else str(results[1])
            },
            "embedding_model": {
                "healthy": results[2].healthy if not isinstance(results[2], Exception) else False,
                "loaded": results[2].loaded if not isinstance(results[2], Exception) else False,
                "error": results[2].error if not isinstance(results[2], Exception) else str(results[2])
            },
            "task_queue": results[3] if not isinstance(results[3], Exception) else {"healthy": False, "error": str(results[3])}
        }

        duration_ms = int((time.time() - start_time) * 1000)

        # 判断整体状态
        all_healthy = all(check["healthy"] for check in checks.values())
        core_healthy = checks["llm_api"]["healthy"] and checks["vector_db"]["healthy"]

        if all_healthy:
            status = "healthy"
            message = "All AI Agent components are healthy"
        elif core_healthy:
            status = "degraded"
            failed = [k for k, v in checks.items() if not v["healthy"]]
            message = f"Running in degraded mode: {', '.join(failed)} unavailable"
        else:
            status = "unhealthy"
            message = "Core AI Agent components are unhealthy"

        return AIAgentHealthResponse(
            status=status,
            duration_ms=duration_ms,
            checks=checks,
            message=message
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return AIAgentHealthResponse(
            status="unhealthy",
            duration_ms=duration_ms,
            checks={},
            message=str(e)
        )

@app.get("/health/ai_agent", response_model=AIAgentHealthResponse)
async def health_ai_agent():
    """完整的 AI Agent 健康检查"""
    result = await check_ai_agent_complete()

    if result.status == "unhealthy":
        raise HTTPException(503, detail=result.dict())

    return result

# ===== 12. 启动和关闭事件 =====

@app.on_event("startup")
async def startup():
    """应用启动"""
    global embedding_model

    print("🚀 Starting AI Agent API...")

    # 加载 Embedding 模型（如果可用）
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print("📦 Loading Embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Failed to load Embedding model: {e}")
    else:
        print("⚠️  sentence-transformers not available")

    # 测试 AI Agent 健康检查
    try:
        result = await check_ai_agent_complete()
        print(f"📊 AI Agent status: {result.status}")
    except Exception as e:
        print(f"❌ AI Agent health check failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    """应用关闭"""
    print("👋 Shutting down AI Agent API...")

    # 关闭连接
    await engine.dispose()
    await redis_client.close()

    print("✅ Connections closed")

# ===== 13. 运行说明 =====

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("AI Agent 特定健康检查实现")
    print("=" * 50)
    print()
    print("端点：")
    print("  /health/llm/openai     - OpenAI API 检查")
    print("  /health/llm/anthropic  - Anthropic API 检查")
    print("  /health/vector_db      - 向量数据库检查")
    print("  /health/embedding      - Embedding 模型检查")
    print("  /health/agent/queue    - Agent 任务队列检查")
    print("  /health/rag            - RAG 系统端到端检查")
    print("  /health/ai_agent       - 完整的 AI Agent 检查")
    print()
    print("环境变量：")
    print("  OPENAI_API_KEY         - OpenAI API 密钥")
    print("  ANTHROPIC_API_KEY      - Anthropic API 密钥")
    print("  DATABASE_URL           - 数据库连接 URL")
    print("  REDIS_URL              - Redis 连接 URL")
    print()
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 环境配置

### 1. 安装依赖

```bash
# 使用 uv 安装依赖
uv add fastapi uvicorn[standard] \
  openai anthropic \
  sqlalchemy[asyncio] asyncpg \
  redis sentence-transformers
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# LLM API
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

ANTHROPIC_API_KEY=sk-ant-...

# 数据库
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0
```

---

## 运行示例

### 1. 启动服务

```bash
python main.py
```

### 2. 测试端点

**OpenAI API 检查：**

```bash
curl http://localhost:8000/health/llm/openai
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 1500,
  "provider": "openai",
  "model": "gpt-3.5-turbo-0125",
  "error": null
}
```

**向量数据库检查：**

```bash
curl http://localhost:8000/health/vector_db
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 50,
  "extension_installed": true,
  "table_exists": true,
  "error": null
}
```

**Embedding 模型检查：**

```bash
curl http://localhost:8000/health/embedding
```

**输出：**

```json
{
  "healthy": true,
  "loaded": true,
  "model_name": "384",
  "error": null
}
```

**RAG 系统检查：**

```bash
curl http://localhost:8000/health/rag
```

**输出：**

```json
{
  "healthy": true,
  "duration_ms": 1600,
  "checks": {
    "llm_api": true,
    "vector_db": true,
    "embedding_model": true
  },
  "error": null
}
```

**完整的 AI Agent 检查：**

```bash
curl http://localhost:8000/health/ai_agent
```

**输出：**

```json
{
  "status": "healthy",
  "duration_ms": 1650,
  "checks": {
    "llm_api": {
      "healthy": true,
      "duration_ms": 1500,
      "error": null
    },
    "vector_db": {
      "healthy": true,
      "duration_ms": 50,
      "error": null
    },
    "embedding_model": {
      "healthy": true,
      "loaded": true,
      "error": null
    },
    "task_queue": {
      "healthy": true,
      "queue_length": 0
    }
  },
  "message": "All AI Agent components are healthy"
}
```

---

## 关键要点

### 1. LLM API 检查策略

**最小请求：**
- 使用最小的 token 数（1 token）
- 超时时间 10 秒（LLM API 较慢）
- 缓存时间 5 分钟（避免频繁调用）

### 2. 向量数据库检查

**检查内容：**
- pgvector 扩展是否安装
- embeddings 表是否存在
- 简单的向量查询是否正常

### 3. Embedding 模型检查

**检查内容：**
- 模型是否已加载
- 模型编码功能是否正常

### 4. 任务队列检查

**检查内容：**
- 队列长度是否正常
- 是否有积压（> 1000）

### 5. 三态模型

- **healthy**：所有组件正常
- **degraded**：核心组件正常，可选组件失败
- **unhealthy**：核心组件失败

---

## 在生产环境中的应用

### 推荐配置

```python
@app.get("/ready")
async def ready():
    """生产环境就绪检查"""
    result = await check_ai_agent_complete()

    # 核心组件失败 → 不可用
    if result.status == "unhealthy":
        raise HTTPException(503, detail=result.dict())

    # 降级模式 → 可用但警告
    if result.status == "degraded":
        return {
            "status": "degraded",
            "message": result.message,
            "checks": result.checks
        }

    return {
        "status": "healthy",
        "checks": result.checks
    }
```

---

## 总结

AI Agent 特定健康检查的关键：

1. **LLM API**：使用最小请求，长超时，长缓存
2. **向量数据库**：检查扩展和表，执行简单查询
3. **Embedding 模型**：检查加载状态和编码功能
4. **任务队列**：检查队列长度，避免积压
5. **RAG 系统**：端到端检查所有组件
6. **三态模型**：健康、降级、不健康

在 AI Agent 后端中，合理的健康检查可以确保 RAG 系统的可用性和可靠性。
