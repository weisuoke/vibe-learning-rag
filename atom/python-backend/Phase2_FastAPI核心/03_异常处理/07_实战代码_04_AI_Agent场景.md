# 实战代码4：AI Agent 场景

> 完整可运行的 AI Agent 异常处理示例，处理 LLM 调用失败等场景

---

## 示例概述

本示例演示：
1. 处理 OpenAI API 调用失败（超时、限流、余额不足）
2. 处理向量数据库异常
3. 处理文档解析失败
4. 实现重试机制
5. 流式响应的异常处理

---

## 完整代码

```python
"""
FastAPI AI Agent 异常处理示例
演示：如何处理 AI Agent 开发中的常见异常
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, AsyncGenerator
import logging
import uuid
import asyncio
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent 异常处理示例")

# ===== 自定义异常类 =====

class AIServiceError(Exception):
    """AI 服务异常基类"""
    def __init__(self, message: str, code: str, retry_after: int = None):
        self.message = message
        self.code = code
        self.retry_after = retry_after
        super().__init__(self.message)

class LLMAPIError(AIServiceError):
    """LLM API 调用异常"""
    pass

class LLMRateLimitError(AIServiceError):
    """LLM API 限流异常"""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="请求过于频繁，请稍后重试",
            code="LLM_RATE_LIMIT",
            retry_after=retry_after
        )

class LLMTimeoutError(AIServiceError):
    """LLM API 超时异常"""
    def __init__(self):
        super().__init__(
            message="AI 服务响应超时",
            code="LLM_TIMEOUT"
        )

class VectorDBError(AIServiceError):
    """向量数据库异常"""
    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            message=f"向量数据库操作失败: {operation}",
            code="VECTOR_DB_ERROR"
        )

class DocumentParseError(AIServiceError):
    """文档解析异常"""
    def __init__(self, file_name: str, error: str):
        self.file_name = file_name
        self.error = error
        super().__init__(
            message=f"文档解析失败: {file_name} - {error}",
            code="DOCUMENT_PARSE_ERROR"
        )

# ===== 异常处理器 =====

@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_handler(request: Request, exc: LLMRateLimitError):
    """处理 LLM 限流异常"""
    logger.warning(f"LLM 限流: {exc.message}")
    return JSONResponse(
        status_code=429,
        content={
            "error": exc.message,
            "code": exc.code,
            "retry_after": exc.retry_after
        },
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(request: Request, exc: LLMTimeoutError):
    """处理 LLM 超时异常"""
    logger.error(f"LLM 超时: {exc.message}")
    return JSONResponse(
        status_code=504,
        content={
            "error": exc.message,
            "code": exc.code
        }
    )

@app.exception_handler(VectorDBError)
async def vector_db_error_handler(request: Request, exc: VectorDBError):
    """处理向量数据库异常"""
    logger.error(f"向量数据库错误: {exc.message}")
    return JSONResponse(
        status_code=503,
        content={
            "error": exc.message,
            "code": exc.code,
            "operation": exc.operation
        }
    )

@app.exception_handler(DocumentParseError)
async def document_parse_error_handler(request: Request, exc: DocumentParseError):
    """处理文档解析异常"""
    logger.error(f"文档解析错误: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "code": exc.code,
            "file_name": exc.file_name,
            "details": exc.error
        }
    )

@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    """处理所有 AI 服务异常（兜底）"""
    logger.error(f"AI 服务错误: {exc.message}")
    return JSONResponse(
        status_code=503,
        content={
            "error": exc.message,
            "code": exc.code
        }
    )

# ===== 模拟的 AI 服务 =====

class MockLLMService:
    """模拟的 LLM 服务"""

    def __init__(self):
        self.request_count = 0
        self.rate_limit = 5  # 每分钟最多 5 次请求

    async def generate(self, prompt: str, timeout: int = 30) -> str:
        """生成文本"""
        self.request_count += 1

        # 模拟限流
        if self.request_count > self.rate_limit:
            raise LLMRateLimitError(retry_after=60)

        # 模拟超时
        if "slow" in prompt.lower():
            await asyncio.sleep(timeout + 1)
            raise LLMTimeoutError()

        # 模拟正常响应
        await asyncio.sleep(0.5)
        return f"AI 回复: {prompt}"

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        self.request_count += 1

        # 模拟限流
        if self.request_count > self.rate_limit:
            raise LLMRateLimitError(retry_after=60)

        # 模拟流式响应
        words = f"AI 回复: {prompt}".split()
        for word in words:
            await asyncio.sleep(0.1)
            yield word + " "

class MockVectorDB:
    """模拟的向量数据库"""

    def __init__(self):
        self.documents = {}
        self.is_connected = True

    async def search(self, query: str, top_k: int = 5) -> list:
        """搜索相似文档"""
        if not self.is_connected:
            raise VectorDBError("search")

        # 模拟搜索
        await asyncio.sleep(0.2)
        return [
            {"id": 1, "content": "文档1", "score": 0.9},
            {"id": 2, "content": "文档2", "score": 0.8},
        ]

    async def insert(self, doc_id: str, content: str, embedding: list) -> None:
        """插入文档"""
        if not self.is_connected:
            raise VectorDBError("insert")

        # 模拟插入
        await asyncio.sleep(0.1)
        self.documents[doc_id] = {"content": content, "embedding": embedding}

class MockDocumentParser:
    """模拟的文档解析器"""

    async def parse(self, file_name: str, content: bytes) -> str:
        """解析文档"""
        # 模拟解析失败
        if file_name.endswith(".corrupted"):
            raise DocumentParseError(file_name, "文件损坏")

        # 模拟解析成功
        await asyncio.sleep(0.1)
        return f"解析后的文本: {file_name}"

# 创建服务实例
llm_service = MockLLMService()
vector_db = MockVectorDB()
document_parser = MockDocumentParser()

# ===== Pydantic 模型 =====

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class DocumentUpload(BaseModel):
    file_name: str
    content: str

# ===== 路由示例 =====

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "FastAPI AI Agent 异常处理示例",
        "endpoints": {
            "chat": "/chat",
            "search": "/search",
            "upload": "/upload"
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """AI 对话（非流式）"""
    logger.info(f"收到对话请求: {request.message}")

    try:
        # 调用 LLM 服务
        response = await llm_service.generate(request.message, timeout=30)

        return {
            "message": request.message,
            "reply": response,
            "timestamp": datetime.utcnow().isoformat()
        }

    except LLMRateLimitError:
        # 限流异常会被异常处理器捕获
        raise

    except LLMTimeoutError:
        # 超时异常会被异常处理器捕获
        raise

    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail="对话服务暂时不可用")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """AI 对话（流式）"""
    logger.info(f"收到流式对话请求: {request.message}")

    async def generate():
        try:
            async for chunk in llm_service.generate_stream(request.message):
                yield chunk

        except LLMRateLimitError as e:
            # 流式响应中的异常处理
            error_message = f"data: {{'error': '{e.message}', 'code': '{e.code}'}}\n\n"
            yield error_message

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            error_message = f"data: {{'error': '对话服务暂时不可用'}}\n\n"
            yield error_message

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/search")
async def search_documents(request: SearchRequest):
    """搜索文档"""
    logger.info(f"搜索文档: {request.query}")

    try:
        # 搜索向量数据库
        results = await vector_db.search(request.query, request.top_k)

        return {
            "query": request.query,
            "results": results,
            "total": len(results)
        }

    except VectorDBError:
        # 向量数据库异常会被异常处理器捕获
        raise

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail="搜索服务暂时不可用")

@app.post("/upload")
async def upload_document(doc: DocumentUpload):
    """上传文档"""
    logger.info(f"上传文档: {doc.file_name}")

    try:
        # 1. 解析文档
        parsed_text = await document_parser.parse(doc.file_name, doc.content.encode())

        # 2. 生成 Embedding（模拟）
        embedding = [0.1, 0.2, 0.3]  # 实际应该调用 Embedding API

        # 3. 插入向量数据库
        doc_id = str(uuid.uuid4())
        await vector_db.insert(doc_id, parsed_text, embedding)

        return {
            "message": "文档上传成功",
            "doc_id": doc_id,
            "file_name": doc.file_name
        }

    except DocumentParseError:
        # 文档解析异常会被异常处理器捕获
        raise

    except VectorDBError:
        # 向量数据库异常会被异常处理器捕获
        raise

    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail="上传服务暂时不可用")

@app.post("/chat/retry")
async def chat_with_retry(request: ChatRequest, max_retries: int = 3):
    """带重试机制的 AI 对话"""
    logger.info(f"收到对话请求（带重试）: {request.message}")

    for attempt in range(max_retries):
        try:
            # 调用 LLM 服务
            response = await llm_service.generate(request.message, timeout=30)

            return {
                "message": request.message,
                "reply": response,
                "attempts": attempt + 1,
                "timestamp": datetime.utcnow().isoformat()
            }

        except LLMTimeoutError:
            logger.warning(f"LLM 超时，重试 {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                # 最后一次重试失败，抛出异常
                raise

            # 等待后重试
            await asyncio.sleep(2 ** attempt)  # 指数退避

        except LLMRateLimitError:
            # 限流不重试，直接抛出
            raise

        except Exception as e:
            logger.error(f"对话失败: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="对话服务暂时不可用")

            await asyncio.sleep(1)

@app.get("/error/rate-limit")
async def simulate_rate_limit():
    """模拟限流错误"""
    raise LLMRateLimitError(retry_after=60)

@app.get("/error/timeout")
async def simulate_timeout():
    """模拟超时错误"""
    raise LLMTimeoutError()

@app.get("/error/vector-db")
async def simulate_vector_db_error():
    """模拟向量数据库错误"""
    vector_db.is_connected = False
    try:
        await vector_db.search("test")
    finally:
        vector_db.is_connected = True

@app.get("/error/document-parse")
async def simulate_document_parse_error():
    """模拟文档解析错误"""
    await document_parser.parse("test.corrupted", b"content")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "services": {
            "llm": "ok",
            "vector_db": "ok" if vector_db.is_connected else "error",
            "document_parser": "ok"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("FastAPI AI Agent 异常处理示例")
    print("=" * 50)
    print("\n启动服务器...")
    print("访问 http://localhost:8000/docs 查看 API 文档\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 运行示例

### 1. 运行服务器

```bash
python examples/fastapi_ai_agent_errors.py
```

---

### 2. 测试各种异常场景

#### 场景1：正常对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

**响应：**
```json
{
  "message": "你好",
  "reply": "AI 回复: 你好",
  "timestamp": "2026-02-11T09:30:00.000Z"
}
```

---

#### 场景2：LLM 限流

```bash
# 连续发送多次请求触发限流
for i in {1..10}; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "测试'$i'"}'
done
```

**响应（第6次请求）：**
```json
{
  "error": "请求过于频繁，请稍后重试",
  "code": "LLM_RATE_LIMIT",
  "retry_after": 60
}
```

**响应头：**
```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```

---

#### 场景3：LLM 超时

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "slow request"}'
```

**响应：**
```json
{
  "error": "AI 服务响应超时",
  "code": "LLM_TIMEOUT"
}
```

---

#### 场景4：向量数据库错误

```bash
curl http://localhost:8000/error/vector-db
```

**响应：**
```json
{
  "error": "向量数据库操作失败: search",
  "code": "VECTOR_DB_ERROR",
  "operation": "search"
}
```

---

#### 场景5：文档解析错误

```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "test.corrupted",
    "content": "test content"
  }'
```

**响应：**
```json
{
  "error": "文档解析失败: test.corrupted - 文件损坏",
  "code": "DOCUMENT_PARSE_ERROR",
  "file_name": "test.corrupted",
  "details": "文件损坏"
}
```

---

#### 场景6：带重试机制的对话

```bash
curl -X POST "http://localhost:8000/chat/retry?max_retries=3" \
  -H "Content-Type: application/json" \
  -d '{"message": "测试重试"}'
```

**响应：**
```json
{
  "message": "测试重试",
  "reply": "AI 回复: 测试重试",
  "attempts": 1,
  "timestamp": "2026-02-11T09:30:00.000Z"
}
```

---

#### 场景7：流式对话

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "stream": true}'
```

**响应（流式）：**
```
AI 回复: 你好
```

---

## 关键知识点

### 1. AI 服务异常的分类

```python
class AIServiceError(Exception):
    """AI 服务异常基类"""
    pass

class LLMRateLimitError(AIServiceError):
    """LLM API 限流异常"""
    pass

class LLMTimeoutError(AIServiceError):
    """LLM API 超时异常"""
    pass

class VectorDBError(AIServiceError):
    """向量数据库异常"""
    pass

class DocumentParseError(AIServiceError):
    """文档解析异常"""
    pass
```

---

### 2. 限流异常的处理

```python
@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_handler(request: Request, exc: LLMRateLimitError):
    """处理 LLM 限流异常"""
    return JSONResponse(
        status_code=429,
        content={
            "error": exc.message,
            "code": exc.code,
            "retry_after": exc.retry_after
        },
        headers={"Retry-After": str(exc.retry_after)}  # 告诉客户端何时重试
    )
```

---

### 3. 重试机制

```python
async def chat_with_retry(request: ChatRequest, max_retries: int = 3):
    """带重试机制的 AI 对话"""
    for attempt in range(max_retries):
        try:
            response = await llm_service.generate(request.message)
            return response

        except LLMTimeoutError:
            if attempt == max_retries - 1:
                raise  # 最后一次重试失败，抛出异常

            # 指数退避
            await asyncio.sleep(2 ** attempt)

        except LLMRateLimitError:
            # 限流不重试，直接抛出
            raise
```

---

### 4. 流式响应的异常处理

```python
async def chat_stream(request: ChatRequest):
    """AI 对话（流式）"""
    async def generate():
        try:
            async for chunk in llm_service.generate_stream(request.message):
                yield chunk

        except LLMRateLimitError as e:
            # 流式响应中的异常处理
            error_message = f"data: {{'error': '{e.message}'}}\n\n"
            yield error_message

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 最佳实践

### 1. 区分可重试和不可重试的异常

```python
# ✅ 好：区分可重试和不可重试的异常
try:
    response = await llm_service.generate(message)
except LLMTimeoutError:
    # 超时可以重试
    await asyncio.sleep(2)
    response = await llm_service.generate(message)
except LLMRateLimitError:
    # 限流不应该重试，应该等待
    raise
```

---

### 2. 使用指数退避

```python
# ✅ 好：使用指数退避
for attempt in range(max_retries):
    try:
        response = await llm_service.generate(message)
        break
    except LLMTimeoutError:
        await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s...
```

---

### 3. 添加 Retry-After 响应头

```python
# ✅ 好：告诉客户端何时重试
return JSONResponse(
    status_code=429,
    content={"error": "请求过多"},
    headers={"Retry-After": "60"}  # 60秒后重试
)
```

---

### 4. 记录详细的错误日志

```python
# ✅ 好：记录详细日志
logger.error(
    f"LLM 调用失败: {exc}",
    extra={
        "request_id": request_id,
        "model": "gpt-4",
        "prompt_length": len(prompt)
    }
)
```

---

## 小结

AI Agent 异常处理的核心要点：

1. **分类处理**：区分限流、超时、数据库错误等不同类型
2. **重试机制**：对可重试的异常使用指数退避
3. **限流处理**：返回 429 状态码和 Retry-After 响应头
4. **流式响应**：在流式响应中也要处理异常
5. **详细日志**：记录所有 AI 服务调用的详细信息

**下一步：** 学习面试中关于异常处理的常见问题。
