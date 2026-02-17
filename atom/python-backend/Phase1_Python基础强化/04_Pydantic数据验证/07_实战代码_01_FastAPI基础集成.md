# 实战代码1：FastAPI 基础集成

完整的 FastAPI + Pydantic 基础集成示例，展示请求验证和响应序列化。

---

## 场景说明

构建一个用户管理 API，包含创建、查询、更新用户的功能，展示 Pydantic 在 FastAPI 中的基础用法。

---

## 完整代码

```python
"""
FastAPI + Pydantic 基础集成示例
演示：用户管理 API
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import Optional, List
from datetime import datetime

# ===== 1. 创建 FastAPI 应用 =====
app = FastAPI(
    title="User Management API",
    description="用户管理 API 示例",
    version="1.0.0"
)

# ===== 2. 定义 Pydantic 模型 =====

class UserBase(BaseModel):
    """用户基础模型"""
    name: str = Field(..., min_length=1, max_length=50, example="Alice")
    email: EmailStr = Field(..., example="alice@example.com")
    age: int = Field(..., ge=0, le=150, example=25)


class CreateUserRequest(UserBase):
    """创建用户请求模型"""
    password: str = Field(..., min_length=8, example="password123")


class UpdateUserRequest(BaseModel):
    """更新用户请求模型（所有字段可选）"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=0, le=150)


class UserResponse(UserBase):
    """用户响应模型"""
    id: int = Field(..., example=1)
    created_at: datetime = Field(..., example="2026-02-11T06:20:29.095Z")
    is_active: bool = Field(True, example=True)

    model_config = ConfigDict(
        # 允许从 ORM 对象创建
        from_attributes=True
    )


class UserListResponse(BaseModel):
    """用户列表响应模型"""
    total: int
    users: List[UserResponse]


class MessageResponse(BaseModel):
    """通用消息响应"""
    message: str


# ===== 3. 模拟数据库 =====

# 模拟数据库（实际项目中使用真实数据库）
fake_db: List[dict] = []
next_id = 1


# ===== 4. API 端点 =====

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建用户",
    description="创建一个新用户"
)
async def create_user(request: CreateUserRequest):
    """
    创建用户

    FastAPI 自动：
    - 解析请求体 JSON
    - 用 Pydantic 验证数据
    - 转换为 CreateUserRequest 对象
    - 验证失败自动返回 422 错误
    """
    global next_id

    # 检查邮箱是否已存在
    if any(user["email"] == request.email for user in fake_db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )

    # 创建用户（实际项目中保存到数据库）
    user_data = {
        "id": next_id,
        "name": request.name,
        "email": request.email,
        "age": request.age,
        "password": request.password,  # 实际项目中应该加密
        "created_at": datetime.now(),
        "is_active": True
    }
    fake_db.append(user_data)
    next_id += 1

    # 返回响应（自动排除 password 字段）
    return UserResponse(**user_data)


@app.get(
    "/users",
    response_model=UserListResponse,
    summary="获取用户列表",
    description="获取所有用户列表"
)
async def list_users(
    skip: int = Field(0, ge=0, description="跳过的记录数"),
    limit: int = Field(10, ge=1, le=100, description="返回的记录数")
):
    """
    获取用户列表

    查询参数自动验证：
    - skip >= 0
    - limit 范围 1-100
    """
    # 分页查询
    users = fake_db[skip : skip + limit]

    return UserListResponse(
        total=len(fake_db),
        users=[UserResponse(**user) for user in users]
    )


@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="获取用户详情",
    description="根据 ID 获取用户详情"
)
async def get_user(user_id: int = Field(..., ge=1, description="用户 ID")):
    """
    获取用户详情

    路径参数自动验证：
    - user_id >= 1
    """
    # 查找用户
    user = next((u for u in fake_db if u["id"] == user_id), None)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    return UserResponse(**user)


@app.put(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="更新用户",
    description="更新用户信息"
)
async def update_user(
    user_id: int = Field(..., ge=1, description="用户 ID"),
    request: UpdateUserRequest = None
):
    """
    更新用户

    只更新提供的字段（部分更新）
    """
    # 查找用户
    user = next((u for u in fake_db if u["id"] == user_id), None)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # 更新字段（只更新提供的字段）
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        user[field] = value

    return UserResponse(**user)


@app.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    summary="删除用户",
    description="删除指定用户"
)
async def delete_user(user_id: int = Field(..., ge=1, description="用户 ID")):
    """
    删除用户
    """
    global fake_db

    # 查找用户索引
    user_index = next(
        (i for i, u in enumerate(fake_db) if u["id"] == user_id),
        None
    )

    if user_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # 删除用户
    fake_db.pop(user_index)

    return MessageResponse(message=f"User {user_id} deleted successfully")


# ===== 5. 健康检查端点 =====

@app.get(
    "/health",
    response_model=MessageResponse,
    summary="健康检查",
    description="检查 API 是否正常运行"
)
async def health_check():
    """健康检查"""
    return MessageResponse(message="OK")


# ===== 6. 运行应用 =====

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("FastAPI + Pydantic 基础集成示例")
    print("=" * 50)
    print("\n启动服务器...")
    print("API 文档: http://localhost:8000/docs")
    print("ReDoc 文档: http://localhost:8000/redoc")
    print("\n按 Ctrl+C 停止服务器\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 运行示例

### 1. 启动服务器

```bash
# 安装依赖
uv add fastapi uvicorn[standard] pydantic[email]

# 运行服务器
python examples/fastapi_basic.py
```

### 2. 访问 API 文档

打开浏览器访问：http://localhost:8000/docs

你会看到自动生成的 Swagger UI 文档，包含：
- 所有 API 端点
- 请求/响应模型
- 字段验证规则
- 示例数据

### 3. 测试 API

**创建用户：**

```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice",
    "email": "alice@example.com",
    "age": 25,
    "password": "password123"
  }'
```

**响应：**

```json
{
  "id": 1,
  "name": "Alice",
  "email": "alice@example.com",
  "age": 25,
  "created_at": "2026-02-11T06:20:29.095Z",
  "is_active": true
}
```

**获取用户列表：**

```bash
curl "http://localhost:8000/users?skip=0&limit=10"
```

**响应：**

```json
{
  "total": 1,
  "users": [
    {
      "id": 1,
      "name": "Alice",
      "email": "alice@example.com",
      "age": 25,
      "created_at": "2026-02-11T06:20:29.095Z",
      "is_active": true
    }
  ]
}
```

**获取用户详情：**

```bash
curl "http://localhost:8000/users/1"
```

**更新用户：**

```bash
curl -X PUT "http://localhost:8000/users/1" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Smith",
    "age": 26
  }'
```

**删除用户：**

```bash
curl -X DELETE "http://localhost:8000/users/1"
```

---

## 验证失败示例

### 1. 字段缺失

**请求：**

```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice"
  }'
```

**响应（422 错误）：**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "email"],
      "msg": "Field required"
    },
    {
      "type": "missing",
      "loc": ["body", "age"],
      "msg": "Field required"
    },
    {
      "type": "missing",
      "loc": ["body", "password"],
      "msg": "Field required"
    }
  ]
}
```

### 2. 类型错误

**请求：**

```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice",
    "email": "alice@example.com",
    "age": "invalid",
    "password": "password123"
  }'
```

**响应（422 错误）：**

```json
{
  "detail": [
    {
      "type": "int_parsing",
      "loc": ["body", "age"],
      "msg": "Input should be a valid integer"
    }
  ]
}
```

### 3. 验证规则失败

**请求：**

```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "",
    "email": "invalid-email",
    "age": 200,
    "password": "123"
  }'
```

**响应（422 错误）：**

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "name"],
      "msg": "String should have at least 1 character"
    },
    {
      "type": "value_error",
      "loc": ["body", "email"],
      "msg": "value is not a valid email address"
    },
    {
      "type": "less_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be less than or equal to 150"
    },
    {
      "type": "string_too_short",
      "loc": ["body", "password"],
      "msg": "String should have at least 8 characters"
    }
  ]
}
```

---

## 核心要点

### 1. 自动验证

FastAPI 自动使用 Pydantic 验证：
- ✅ 请求体（POST/PUT）
- ✅ 查询参数（GET）
- ✅ 路径参数（GET/PUT/DELETE）
- ✅ 响应数据

### 2. 自动文档

Pydantic 模型自动生成：
- ✅ OpenAPI Schema
- ✅ Swagger UI（/docs）
- ✅ ReDoc（/redoc）
- ✅ 字段描述和示例

### 3. 自动序列化

Pydantic 自动：
- ✅ 转换 datetime 为 ISO 8601 字符串
- ✅ 排除未定义的字段（如 password）
- ✅ 处理嵌套模型

### 4. 类型安全

- ✅ IDE 智能提示
- ✅ mypy 静态检查
- ✅ 运行时验证

---

## 最佳实践

### 1. 模型分层

```python
# 基础模型（共享字段）
class UserBase(BaseModel):
    name: str
    email: EmailStr

# 请求模型（包含敏感字段）
class CreateUserRequest(UserBase):
    password: str

# 响应模型（排除敏感字段）
class UserResponse(UserBase):
    id: int
    created_at: datetime
```

### 2. 字段验证

```python
class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: EmailStr  # 自动验证邮箱格式
```

### 3. 响应模型

```python
@app.post("/users", response_model=UserResponse)
async def create_user(request: CreateUserRequest):
    # response_model 自动：
    # - 验证返回数据
    # - 排除未定义字段
    # - 序列化为 JSON
    return user_data
```

### 4. 错误处理

```python
if not user:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )
```

---

## 扩展阅读

- FastAPI 文档：https://fastapi.tiangolo.com/
- Pydantic 文档：https://docs.pydantic.dev/
- OpenAPI 规范：https://swagger.io/specification/

---

**版本：** v1.0
**最后更新：** 2026-02-11
