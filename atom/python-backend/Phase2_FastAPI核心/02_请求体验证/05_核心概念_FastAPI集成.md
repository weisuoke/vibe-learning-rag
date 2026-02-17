# 核心概念3：FastAPI 集成

> 路由参数、响应模型、错误处理 - FastAPI 如何使用 Pydantic 验证

---

## 概述

FastAPI 通过类型注解自动集成 Pydantic 验证，提供了零配置的请求体验证、响应模型过滤和错误处理。本文详细讲解 FastAPI 如何使用 Pydantic 模型。

---

## 1. 请求体验证

### 1.1 基础用法

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class User(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: str
    age: int = Field(ge=0, le=150)

@app.post("/users")
async def create_user(user: User):
    # FastAPI 自动：
    # 1. 解析请求体（JSON → dict）
    # 2. 验证数据（调用 Pydantic）
    # 3. 创建 User 实例
    # 4. 传递给路由函数
    return {"username": user.username, "age": user.age}
```

### 1.2 多个请求体参数

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str

class Item(BaseModel):
    name: str
    price: float

@app.post("/users/{user_id}/items")
async def create_item(
    user_id: int,
    user: User,
    item: Item,
    importance: int = Body(...)  # 单个字段
):
    return {
        "user_id": user_id,
        "user": user,
        "item": item,
        "importance": importance
    }

# 请求体格式：
# {
#   "user": {"username": "alice", "email": "alice@example.com"},
#   "item": {"name": "laptop", "price": 999.99},
#   "importance": 5
# }
```

### 1.3 嵌入单个模型

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str

@app.post("/users")
async def create_user(user: User = Body(embed=True)):
    return user

# 默认请求体格式（embed=False）：
# {"username": "alice", "email": "alice@example.com"}

# embed=True 的请求体格式：
# {"user": {"username": "alice", "email": "alice@example.com"}}
```

---

## 2. 路径参数和查询参数验证

### 2.1 路径参数

```python
from fastapi import FastAPI, Path
from pydantic import BaseModel

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., ge=1, description="用户ID")
):
    return {"user_id": user_id}

@app.get("/items/{item_id}")
async def get_item(
    item_id: str = Path(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9-]+$')
):
    return {"item_id": item_id}
```

### 2.2 查询参数

```python
from fastapi import FastAPI, Query
from typing import Optional, List

app = FastAPI()

@app.get("/items")
async def list_items(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
    q: Optional[str] = Query(default=None, min_length=3, max_length=50),
    tags: List[str] = Query(default=[])
):
    return {
        "skip": skip,
        "limit": limit,
        "q": q,
        "tags": tags
    }

# 请求示例：
# GET /items?skip=0&limit=20&q=laptop&tags=electronics&tags=computers
```

### 2.3 使用 Pydantic 模型作为查询参数

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

app = FastAPI()

class Pagination(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

class SearchParams(BaseModel):
    q: Optional[str] = Field(default=None, min_length=3)
    tags: List[str] = Field(default_factory=list)

@app.get("/items")
async def list_items(
    pagination: Pagination = Depends(),
    search: SearchParams = Depends()
):
    return {
        "pagination": pagination,
        "search": search
    }

# 请求示例：
# GET /items?skip=0&limit=20&q=laptop&tags=electronics
```

---

## 3. 响应模型

### 3.1 基础用法

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserIn(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    username: str
    email: str

@app.post("/users", response_model=UserOut)
async def create_user(user: UserIn):
    # 即使返回 UserIn（包含 password），
    # FastAPI 也会自动过滤，只返回 UserOut 的字段
    return user

# 实际响应：
# {"username": "alice", "email": "alice@example.com"}
# password 被自动过滤
```

### 3.2 响应模型配置

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    bio: Optional[str] = None
    age: Optional[int] = None

@app.get(
    "/users/{user_id}",
    response_model=User,
    response_model_exclude_none=True,  # 过滤 None 值
    response_model_exclude_unset=True  # 过滤未设置的字段
)
async def get_user(user_id: int):
    return {
        "username": "alice",
        "email": "alice@example.com",
        "bio": None,
        "age": None
    }

# 实际响应（None 值被过滤）：
# {"username": "alice", "email": "alice@example.com"}
```

### 3.3 响应模型列表

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class User(BaseModel):
    username: str
    email: str

@app.get("/users", response_model=List[User])
async def list_users():
    return [
        {"username": "alice", "email": "alice@example.com"},
        {"username": "bob", "email": "bob@example.com"}
    ]
```

### 3.4 部分响应

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    password: str
    bio: str

@app.get(
    "/users/{user_id}",
    response_model=User,
    response_model_exclude={"password"}  # 排除特定字段
)
async def get_user(user_id: int):
    return {
        "username": "alice",
        "email": "alice@example.com",
        "password": "secret",
        "bio": "Hello"
    }

# 实际响应：
# {"username": "alice", "email": "alice@example.com", "bio": "Hello"}

@app.get(
    "/users/{user_id}/public",
    response_model=User,
    response_model_include={"username", "bio"}  # 只包含特定字段
)
async def get_user_public(user_id: int):
    return {
        "username": "alice",
        "email": "alice@example.com",
        "password": "secret",
        "bio": "Hello"
    }

# 实际响应：
# {"username": "alice", "bio": "Hello"}
```

---

## 4. 验证错误处理

### 4.1 默认错误响应

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class User(BaseModel):
    username: str = Field(min_length=3)
    age: int = Field(ge=0, le=150)

@app.post("/users")
async def create_user(user: User):
    return user

# 错误请求：
# POST /users
# {"username": "ab", "age": 200}

# 自动错误响应（422）：
# {
#   "detail": [
#     {
#       "loc": ["body", "username"],
#       "msg": "ensure this value has at least 3 characters",
#       "type": "value_error.any_str.min_length",
#       "ctx": {"limit_value": 3}
#     },
#     {
#       "loc": ["body", "age"],
#       "msg": "ensure this value is less than or equal to 150",
#       "type": "value_error.number.not_le",
#       "ctx": {"limit_value": 150}
#     }
#   ]
# }
```

### 4.2 自定义错误处理

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 自定义错误格式
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": errors}
    )

class User(BaseModel):
    username: str
    age: int

@app.post("/users")
async def create_user(user: User):
    return user

# 错误请求：
# POST /users
# {"username": "alice", "age": "not-a-number"}

# 自定义错误响应（422）：
# {
#   "errors": [
#     {
#       "field": "body.age",
#       "message": "value is not a valid integer",
#       "type": "type_error.integer"
#     }
#   ]
# }
```

### 4.3 中文错误消息

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

app = FastAPI()

class User(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(ge=0, le=150)

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('用户名只能包含字母和数字')
        return v

    @validator('age')
    def age_valid(cls, v):
        if v < 0:
            raise ValueError('年龄不能为负数')
        if v > 150:
            raise ValueError('年龄不能超过150岁')
        return v

@app.post("/users")
async def create_user(user: User):
    return user
```

---

## 5. 依赖注入中的验证

### 5.1 可复用的依赖

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class Pagination(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

def get_pagination(
    skip: int = 0,
    limit: int = 10
) -> Pagination:
    return Pagination(skip=skip, limit=limit)

@app.get("/users")
async def list_users(pagination: Pagination = Depends(get_pagination)):
    return {
        "skip": pagination.skip,
        "limit": pagination.limit
    }

@app.get("/items")
async def list_items(pagination: Pagination = Depends(get_pagination)):
    return {
        "skip": pagination.skip,
        "limit": pagination.limit
    }
```

### 5.2 验证依赖

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    username: str

async def get_current_user(authorization: str = Header(...)) -> User:
    # 验证 token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    token = authorization[7:]
    # 这里应该验证 token 并从数据库获取用户
    # 简化示例
    if token != "valid-token":
        raise HTTPException(status_code=401, detail="Invalid token")

    return User(id=1, username="alice")

@app.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/items")
async def create_item(
    item_name: str,
    current_user: User = Depends(get_current_user)
):
    return {
        "item_name": item_name,
        "owner": current_user.username
    }
```

---

## 6. 在 AI Agent 开发中的应用

### 6.1 RAG 查询端点

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI()

class RAGQuery(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class Document(BaseModel):
    content: str
    score: float
    metadata: dict

class RAGResponse(BaseModel):
    answer: str
    documents: List[Document]
    confidence: float

@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    # 执行 RAG 查询
    documents = await search_documents(
        query.question,
        top_k=query.top_k,
        threshold=query.threshold
    )

    answer = await generate_answer(query.question, documents)

    return RAGResponse(
        answer=answer,
        documents=documents,
        confidence=0.95
    )
```

### 6.2 Agent 对话端点

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

app = FastAPI()

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_items=1)
    model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    stream: bool = False

class ChatResponse(BaseModel):
    message: Message
    usage: dict

@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 调用 LLM
    response = await llm.chat(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    return ChatResponse(
        message=Message(role="assistant", content=response.content),
        usage=response.usage
    )
```

### 6.3 工具调用端点

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Any

app = FastAPI()

class ToolCall(BaseModel):
    tool_name: str = Field(min_length=1)
    parameters: Dict[str, Any]

    @validator('tool_name')
    def tool_exists(cls, v):
        available_tools = ["search", "calculate", "weather"]
        if v not in available_tools:
            raise ValueError(f'tool must be one of {available_tools}')
        return v

class ToolResponse(BaseModel):
    result: Any
    success: bool
    error: Optional[str] = None

@app.post("/agent/tool", response_model=ToolResponse)
async def execute_tool(tool_call: ToolCall):
    try:
        # 执行工具
        result = await execute_tool_function(
            tool_call.tool_name,
            tool_call.parameters
        )
        return ToolResponse(result=result, success=True)
    except Exception as e:
        return ToolResponse(result=None, success=False, error=str(e))
```

---

## 7. 最佳实践

### 7.1 输入输出模型分离

```python
# ✅ 好的做法
class UserIn(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

@app.post("/users", response_model=UserOut)
async def create_user(user: UserIn):
    # 创建用户逻辑
    return created_user
```

### 7.2 使用依赖注入复用验证

```python
# ✅ 好的做法
def get_pagination(skip: int = 0, limit: int = 10) -> Pagination:
    return Pagination(skip=skip, limit=limit)

@app.get("/users")
async def list_users(pagination: Pagination = Depends(get_pagination)):
    ...

@app.get("/items")
async def list_items(pagination: Pagination = Depends(get_pagination)):
    ...
```

### 7.3 响应模型保护敏感信息

```python
# ✅ 好的做法
@app.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int):
    user = await db.get_user(user_id)
    return user  # password 自动过滤
```

---

## 8. 核心要点总结

1. **请求体验证** - 通过类型注解自动触发
2. **路径和查询参数** - 使用 Path 和 Query 添加约束
3. **响应模型** - 自动过滤和验证响应数据
4. **错误处理** - 自动返回 422 错误，可自定义格式
5. **依赖注入** - 复用验证逻辑

---

## 9. 学习检查清单

- [ ] 在路由中使用 Pydantic 模型验证请求体
- [ ] 使用 Path 和 Query 验证路径和查询参数
- [ ] 使用 response_model 过滤响应数据
- [ ] 自定义验证错误处理
- [ ] 使用依赖注入复用验证逻辑
- [ ] 在 AI Agent API 中应用请求体验证

---

**版本：** v1.0
**最后更新：** 2026-02-11
