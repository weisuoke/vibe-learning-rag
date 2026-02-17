# RAG检索链 - 实战代码6：FastAPI集成

> 将 RAG 系统集成到 FastAPI 服务中

---

## 完整代码

```python
"""
FastAPI RAG 服务
演示：完整的 RAG API 端点、流式输出、错误处理
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

# ===== FastAPI 应用 =====
app = FastAPI(title="RAG API", version="1.0.0")

# ===== 全局变量 =====
vectorstore = None
qa_chain = None

# ===== Pydantic 模型 =====
class QueryRequest(BaseModel):
    question: str
    k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

# ===== 初始化 =====
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化向量库和 RAG 链"""
    global vectorstore, qa_chain

    # 初始化向量库
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db_api"
    )

    # 添加示例文档（如果向量库为空）
    if vectorstore._collection.count() == 0:
        documents = [
            "Python 是一门编程语言，由 Guido van Rossum 创建。",
            "FastAPI 是一个现代、快速的 Web 框架。",
            "RAG 是检索增强生成技术。"
        ]
        vectorstore.add_texts(documents)

    # 初始化 RAG 链
    llm = ChatOpenAI(temperature=0)
    prompt = PromptTemplate(
        template="""
基于以下文档回答问题。如果文档中没有答案，说"我不知道"。

文档：
{context}

问题：{question}

答案：
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    print("✓ RAG 服务初始化完成")

# ===== API 端点 =====

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG 查询端点"""
    try:
        # 调用 RAG 链
        result = await qa_chain.ainvoke({"query": request.question})

        # 构造响应
        return QueryResponse(
            question=request.question,
            answer=result["result"],
            sources=[
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """流式输出端点"""
    async def generate():
        try:
            # 流式回调
            class StreamCallback(BaseCallbackHandler):
                def on_llm_new_token(self, token: str, **kwargs):
                    return token

            # 创建流式 LLM
            streaming_llm = ChatOpenAI(
                temperature=0,
                streaming=True,
                callbacks=[StreamCallback()]
            )

            # 检索文档
            docs = vectorstore.similarity_search(request.question, k=request.k)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 构造 Prompt
            prompt = f"""
基于以下文档回答问题：

{context}

问题：{request.question}

答案：
"""

            # 流式生成
            async for chunk in streaming_llm.astream(prompt):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/documents")
async def add_document(request: DocumentRequest):
    """添加文档端点"""
    try:
        vectorstore.add_texts(
            texts=[request.content],
            metadatas=[request.metadata] if request.metadata else None
        )
        return {"status": "success", "message": "文档已添加"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/count")
async def get_document_count():
    """获取文档数量"""
    count = vectorstore._collection.count()
    return {"count": count}

@app.delete("/documents")
async def clear_documents():
    """清空所有文档"""
    try:
        vectorstore.delete_collection()
        return {"status": "success", "message": "所有文档已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 运行服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 使用示例

### 1. 启动服务

```bash
# 方式1：直接运行
python rag_api.py

# 方式2：使用 uvicorn
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

### 2. 测试 API

```bash
# 健康检查
curl http://localhost:8000/

# 查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 Python？", "k": 2}'

# 流式查询
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 FastAPI？"}' \
  --no-buffer

# 添加文档
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "LangChain 是一个 LLM 应用开发框架", "metadata": {"source": "doc.txt"}}'

# 获取文档数量
curl http://localhost:8000/documents/count

# 清空文档
curl -X DELETE http://localhost:8000/documents
```

### 3. Python 客户端

```python
import requests

# 查询
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "什么是 RAG？", "k": 3}
)
result = response.json()
print(f"答案: {result['answer']}")
print(f"来源: {len(result['sources'])} 个文档")

# 流式查询
response = requests.post(
    "http://localhost:8000/query/stream",
    json={"question": "解释 FastAPI"},
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

---

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 关键技术点

### 1. 应用启动初始化

```python
@app.on_event("startup")
async def startup_event():
    # 初始化向量库和 RAG 链
    # 只在应用启动时执行一次
    pass
```

### 2. 异步端点

```python
@app.post("/query")
async def query(request: QueryRequest):
    # 使用 ainvoke 异步调用
    result = await qa_chain.ainvoke({"query": request.question})
    return result
```

### 3. 流式输出

```python
async def generate():
    async for chunk in streaming_llm.astream(prompt):
        yield f"data: {chunk.content}\n\n"

return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4. 错误处理

```python
try:
    result = await qa_chain.ainvoke({"query": request.question})
    return result
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

---

## 生产环境优化

### 1. 添加认证

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/query")
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # 验证 token
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    # ...
```

### 2. 添加限流

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query_request: QueryRequest):
    # ...
```

### 3. 添加缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(question: str):
    return qa_chain.invoke({"query": question})
```

### 4. 添加日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/query")
async def query(request: QueryRequest):
    logger.info(f"收到查询: {request.question}")
    result = await qa_chain.ainvoke({"query": request.question})
    logger.info(f"返回答案: {result['result'][:50]}...")
    return result
```

---

## 部署

### Docker 部署

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t rag-api .

# 运行容器
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key rag-api
```

---

## 总结

完整的 RAG API 服务包含：
1. ✅ 同步查询端点
2. ✅ 流式输出端点
3. ✅ 文档管理端点
4. ✅ 错误处理
5. ✅ 自动文档生成

**下一步**：学习 [09_化骨绵掌](./09_化骨绵掌.md)，深入理解 RAG 的所有细节。
