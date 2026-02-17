# 实战代码1：FastAPI 类型注解

完整的 FastAPI 项目中类型注解的实战应用。

---

## 概述

本文档展示如何在 FastAPI 项目中充分利用类型注解实现自动验证、文档生成和类型安全。

---

## 1. 基础路由类型注解

### 1.1 路径参数

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., ge=1, description="用户ID，必须大于0")
):
    """获取用户信息"""
    return {"user_id": user_id, "name": "Alice"}

# 访问：GET /users/123
# FastAPI 自动：
# - 验证 user_id 是整数
# - 验证 user_id >= 1
# - 生成 API 文档
```

### 1.2 查询参数

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=100),  # 必填
    limit: int = Query(10, ge=1, le=100),  # 可选，默认10
    offset: int = Query(0, ge=0),  # 可选，默认0
    category: Optional[str] = Query(None)  # 可选，无默认值
):
    """搜索接口"""
    return {
        "query": q,
        "limit": limit,
        "offset": offset,
        "category": category
    }

# 访问：GET /search?q=python&limit=20&category=tutorial
```

### 1.3 请求体

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(..., ge=0, le=150)

@app.post("/users")
async def create_user(user: CreateUserRequest):
    """创建用户"""
    return {"message": "User created", "user": user}

# 请求体：
# {
#   "name": "Alice",
#   "email": "alice@example.com",
#   "age": 25
# }
```

---

## 2. 响应模型

### 2.1 基础响应模型

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int) -> User:
    """获取用户，返回 User 模型"""
    return User(id=user_id, name="Alice", email="alice@example.com")

# FastAPI 自动：
# - 验证返回值符合 User 模型
# - 生成响应文档
# - 过滤掉不在模型中的字段
```

### 2.2 列表响应

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class User(BaseModel):
    id: int
    name: str

@app.get("/users", response_model=List[User])
async def list_users() -> List[User]:
    """获取用户列表"""
    return [
        User(id=1, name="Alice"),
        User(id=2, name="Bob")
    ]
```

### 2.3 嵌套响应模型

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    id: int
    name: str
    email: str
    address: Optional[Address] = None

class UserListResponse(BaseModel):
    total: int
    users: List[User]

@app.get("/users", response_model=UserListResponse)
async def list_users() -> UserListResponse:
    """获取用户列表（带分页信息）"""
    users = [
        User(
            id=1,
            name="Alice",
            email="alice@example.com",
            address=Address(street="123 Main St", city="NYC", country="USA")
        ),
        User(id=2, name="Bob", email="bob@example.com")
    ]
    return UserListResponse(total=len(users), users=users)
```

---

## 3. 依赖注入类型注解

### 3.1 基础依赖

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

# 依赖函数
async def get_token(token: Optional[str] = None) -> str:
    """获取并验证 token"""
    if token is None:
        raise HTTPException(status_code=401, detail="Token required")
    return token

@app.get("/protected")
async def protected_route(token: str = Depends(get_token)):
    """受保护的路由"""
    return {"message": "Access granted", "token": token}
```

### 3.2 类依赖

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

class AuthService:
    """认证服务"""

    def __init__(self, token: Optional[str] = None):
        self.token = token

    def verify(self) -> str:
        """验证 token"""
        if self.token is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return self.token

@app.get("/me")
async def get_current_user(auth: AuthService = Depends()):
    """获取当前用户"""
    token = auth.verify()
    return {"token": token}
```

### 3.3 多层依赖

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    is_admin: bool

# 第一层：获取 token
async def get_token(token: str) -> str:
    if not token:
        raise HTTPException(status_code=401)
    return token

# 第二层：获取用户
async def get_current_user(token: str = Depends(get_token)) -> User:
    # 根据 token 获取用户
    return User(id=1, name="Alice", is_admin=False)

# 第三层：验证管理员
async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin required")
    return user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(get_admin_user)
):
    """删除用户（仅管理员）"""
    return {"message": f"User {user_id} deleted by {admin.name}"}
```

---

## 4. 完整的 CRUD 示例

### 4.1 数据模型定义

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# 基础模型
class UserBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: Optional[int] = Field(None, ge=0, le=150)

# 创建请求
class CreateUserRequest(UserBase):
    password: str = Field(..., min_length=8)

# 更新请求
class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[str] = Field(None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: Optional[int] = Field(None, ge=0, le=150)

# 响应模型
class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # 允许从 ORM 对象创建
```

### 4.2 CRUD 路由

```python
from fastapi import FastAPI, HTTPException, status
from typing import List

app = FastAPI()

# 模拟数据库
users_db: dict[int, UserResponse] = {}
next_id = 1

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: CreateUserRequest) -> UserResponse:
    """创建用户"""
    global next_id
    now = datetime.now()
    new_user = UserResponse(
        id=next_id,
        name=user.name,
        email=user.email,
        age=user.age,
        created_at=now,
        updated_at=now
    )
    users_db[next_id] = new_user
    next_id += 1
    return new_user

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[UserResponse]:
    """获取用户列表"""
    all_users = list(users_db.values())
    return all_users[offset:offset + limit]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int) -> UserResponse:
    """获取单个用户"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UpdateUserRequest
) -> UserResponse:
    """更新用户"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user = users_db[user_id]
    update_data = user_update.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(user, field, value)

    user.updated_at = datetime.now()
    return user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int) -> None:
    """删除用户"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
```

---

## 5. AI Agent API 示例

### 5.1 聊天接口

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal

app = FastAPI()

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1)
    model: Literal["gpt-4", "gpt-3.5-turbo", "claude-3"] = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)

class ChatResponse(BaseModel):
    message: str
    tokens_used: int
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """AI 聊天接口"""
    # 模拟 LLM 调用
    last_message = request.messages[-1].content
    reply = f"Echo: {last_message}"

    return ChatResponse(
        message=reply,
        tokens_used=len(reply),
        model=request.model
    )
```

### 5.2 流式响应

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI()

class StreamChatRequest(BaseModel):
    message: str
    model: str = "gpt-4"

async def generate_stream(message: str):
    """生成流式响应"""
    words = message.split()
    for word in words:
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def stream_chat(request: StreamChatRequest):
    """流式聊天接口"""
    return StreamingResponse(
        generate_stream(request.message),
        media_type="text/event-stream"
    )
```

### 5.3 工具调用

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Literal

app = FastAPI()

class ToolCall(BaseModel):
    tool_name: Literal["search", "calculator", "database"]
    parameters: Dict[str, Any]

class AgentRequest(BaseModel):
    prompt: str
    tools: List[Literal["search", "calculator", "database"]]

class AgentResponse(BaseModel):
    message: str
    tool_calls: List[ToolCall]
    final_answer: str

@app.post("/agent/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest) -> AgentResponse:
    """执行 AI Agent"""
    # 模拟 Agent 执行
    tool_calls = [
        ToolCall(
            tool_name="search",
            parameters={"query": request.prompt}
        )
    ]

    return AgentResponse(
        message="Executing agent...",
        tool_calls=tool_calls,
        final_answer=f"Result for: {request.prompt}"
    )
```

---

## 6. 错误处理

### 6.1 自定义异常

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """统一错误处理"""
    return ErrorResponse(
        error="HTTP Error",
        detail=exc.detail,
        code=str(exc.status_code)
    )

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """获取用户"""
    if user_id < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    if user_id > 1000:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"user_id": user_id}
```

### 6.2 验证错误处理

```python
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """处理验证错误"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"errors": errors}
    )

class User(BaseModel):
    name: str
    age: int

@app.post("/users")
async def create_user(user: User):
    return user

# 请求：{"name": "Alice", "age": "not a number"}
# 响应：
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

---

## 7. 完整项目示例

### 7.1 项目结构

```
app/
├── __init__.py
├── main.py              # FastAPI 应用
├── models/              # Pydantic 模型
│   ├── __init__.py
│   ├── user.py
│   └── chat.py
├── api/                 # API 路由
│   ├── __init__.py
│   ├── users.py
│   └── chat.py
├── services/            # 业务逻辑
│   ├── __init__.py
│   ├── user_service.py
│   └── chat_service.py
└── core/                # 核心配置
    ├── __init__.py
    ├── config.py
    └── dependencies.py
```

### 7.2 模型定义（app/models/user.py）

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """用户基础模型"""
    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

class CreateUserRequest(UserBase):
    """创建用户请求"""
    password: str = Field(..., min_length=8)

class UpdateUserRequest(BaseModel):
    """更新用户请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[str] = None

class UserResponse(UserBase):
    """用户响应"""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
```

### 7.3 服务层（app/services/user_service.py）

```python
from typing import List, Optional
from app.models.user import CreateUserRequest, UpdateUserRequest, UserResponse
from datetime import datetime

class UserService:
    """用户服务"""

    def __init__(self):
        self._users: dict[int, UserResponse] = {}
        self._next_id = 1

    def create_user(self, user: CreateUserRequest) -> UserResponse:
        """创建用户"""
        new_user = UserResponse(
            id=self._next_id,
            name=user.name,
            email=user.email,
            created_at=datetime.now()
        )
        self._users[self._next_id] = new_user
        self._next_id += 1
        return new_user

    def get_user(self, user_id: int) -> Optional[UserResponse]:
        """获取用户"""
        return self._users.get(user_id)

    def list_users(self, limit: int, offset: int) -> List[UserResponse]:
        """获取用户列表"""
        all_users = list(self._users.values())
        return all_users[offset:offset + limit]

    def update_user(
        self,
        user_id: int,
        user_update: UpdateUserRequest
    ) -> Optional[UserResponse]:
        """更新用户"""
        user = self._users.get(user_id)
        if user is None:
            return None

        update_data = user_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        return user

    def delete_user(self, user_id: int) -> bool:
        """删除用户"""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
```

### 7.4 API 路由（app/api/users.py）

```python
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from app.models.user import (
    CreateUserRequest,
    UpdateUserRequest,
    UserResponse
)
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])

def get_user_service() -> UserService:
    """依赖注入：获取用户服务"""
    return UserService()

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: CreateUserRequest,
    service: UserService = Depends(get_user_service)
) -> UserResponse:
    """创建用户"""
    return service.create_user(user)

@router.get("", response_model=List[UserResponse])
async def list_users(
    limit: int = 10,
    offset: int = 0,
    service: UserService = Depends(get_user_service)
) -> List[UserResponse]:
    """获取用户列表"""
    return service.list_users(limit, offset)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
) -> UserResponse:
    """获取单个用户"""
    user = service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UpdateUserRequest,
    service: UserService = Depends(get_user_service)
) -> UserResponse:
    """更新用户"""
    user = service.update_user(user_id, user_update)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
) -> None:
    """删除用户"""
    if not service.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
```

### 7.5 主应用（app/main.py）

```python
from fastapi import FastAPI
from app.api import users

app = FastAPI(
    title="User Management API",
    description="用户管理 API 示例",
    version="1.0.0"
)

# 注册路由
app.include_router(users.router)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Welcome to User Management API"}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# 运行：uvicorn app.main:app --reload
# 文档：http://localhost:8000/docs
```

---

## 总结

### 类型注解在 FastAPI 中的价值

1. **自动验证**：请求参数、请求体、响应自动验证
2. **自动文档**：生成交互式 API 文档（Swagger UI）
3. **类型安全**：mypy 静态检查，IDE 智能提示
4. **错误提示**：详细的验证错误信息
5. **代码可读性**：类型注解即文档

### 最佳实践

1. **所有路由都加类型注解**：参数、返回值
2. **使用 Pydantic 模型**：请求体和响应
3. **使用 Field 验证**：添加约束和描述
4. **使用 response_model**：确保响应格式正确
5. **依赖注入类型化**：清晰的依赖关系

### 下一步

学习完 FastAPI 类型注解后，继续学习：
- 数据模型定义最佳实践
- 类型检查工作流集成
- 生产环境配置
