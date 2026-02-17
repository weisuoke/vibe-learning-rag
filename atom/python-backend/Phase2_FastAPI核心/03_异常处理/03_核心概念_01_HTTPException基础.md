# 核心概念1：HTTPException 基础

> FastAPI 内置的异常类，用于快速抛出 HTTP 错误

---

## 一句话定义

**HTTPException 是 FastAPI 提供的内置异常类，通过 `raise HTTPException(status_code, detail)` 快速抛出 HTTP 错误，自动转换为标准的 JSON 响应。**

---

## 为什么需要 HTTPException？

### 问题：手动返回错误响应太繁琐

```python
# ❌ 手动返回错误（不推荐）
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.get_user(user_id)
    if not user:
        # 需要手动创建 JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": "User not found"}
        )
    return user
```

**问题：**
- 每次都要手动创建 `JSONResponse`
- 容易忘记设置正确的状态码
- 代码不够简洁
- 无法被异常处理器统一捕获

### 解决方案：使用 HTTPException

```python
# ✅ 使用 HTTPException（推荐）
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.get_user(user_id)
    if not user:
        # 直接抛出异常
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

**优势：**
- 代码简洁，一行搞定
- 自动设置 HTTP 状态码
- 自动转换为 JSON 响应
- 可以被异常处理器统一捕获
- 符合 Python 异常处理习惯

---

## HTTPException 的基本用法

### 1. 最简单的用法

```python
from fastapi import HTTPException

# 只指定状态码和错误信息
raise HTTPException(status_code=404, detail="资源不存在")
```

**响应：**
```json
{
  "detail": "资源不存在"
}
```

**HTTP 状态码：** 404

---

### 2. 添加自定义响应头

```python
# 添加自定义响应头（如 WWW-Authenticate）
raise HTTPException(
    status_code=401,
    detail="未授权",
    headers={"WWW-Authenticate": "Bearer"}
)
```

**响应：**
```json
{
  "detail": "未授权"
}
```

**HTTP 响应头：**
```
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
Content-Type: application/json
```

---

### 3. 详细的错误信息

```python
# detail 可以是字符串或字典
raise HTTPException(
    status_code=400,
    detail={
        "error": "参数验证失败",
        "field": "email",
        "message": "邮箱格式不正确"
    }
)
```

**响应：**
```json
{
  "detail": {
    "error": "参数验证失败",
    "field": "email",
    "message": "邮箱格式不正确"
  }
}
```

---

## 常用的 HTTP 状态码

### 客户端错误（4xx）

| 状态码 | 含义 | 使用场景 | 示例 |
|--------|------|----------|------|
| **400** | Bad Request | 请求参数错误 | 缺少必填字段、格式错误 |
| **401** | Unauthorized | 未认证 | 未登录、Token 过期 |
| **403** | Forbidden | 无权限 | 已登录但无权访问资源 |
| **404** | Not Found | 资源不存在 | 用户不存在、文章不存在 |
| **409** | Conflict | 资源冲突 | 用户名已存在、重复提交 |
| **422** | Unprocessable Entity | 语义错误 | 数据验证失败（Pydantic） |
| **429** | Too Many Requests | 请求过多 | 超过限流阈值 |

### 服务器错误（5xx）

| 状态码 | 含义 | 使用场景 | 示例 |
|--------|------|----------|------|
| **500** | Internal Server Error | 服务器内部错误 | 未捕获的异常 |
| **502** | Bad Gateway | 网关错误 | 上游服务返回错误 |
| **503** | Service Unavailable | 服务不可用 | 数据库连接失败、第三方 API 超时 |
| **504** | Gateway Timeout | 网关超时 | 上游服务超时 |

---

## 实战示例

### 示例1：用户不存在（404）

```python
from fastapi import FastAPI, HTTPException
from typing import Optional

app = FastAPI()

# 模拟数据库
users_db = {
    1: {"id": 1, "name": "Alice"},
    2: {"id": 2, "name": "Bob"}
}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """获取用户信息"""
    user = users_db.get(user_id)

    if not user:
        # 用户不存在 → 404
        raise HTTPException(
            status_code=404,
            detail=f"用户 {user_id} 不存在"
        )

    return user
```

**测试：**
```bash
# 存在的用户
curl http://localhost:8000/users/1
# 响应: {"id": 1, "name": "Alice"}

# 不存在的用户
curl http://localhost:8000/users/999
# 响应: {"detail": "用户 999 不存在"}
# 状态码: 404
```

---

### 示例2：未授权访问（401）

```python
from fastapi import FastAPI, HTTPException, Header
from typing import Optional

app = FastAPI()

@app.get("/admin/users")
async def list_users(authorization: Optional[str] = Header(None)):
    """管理员查看用户列表（需要认证）"""

    # 检查是否提供了 Token
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="未提供认证信息",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 检查 Token 是否有效
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="认证格式错误，应为 'Bearer <token>'"
        )

    token = authorization.replace("Bearer ", "")
    if token != "secret-token":
        raise HTTPException(
            status_code=401,
            detail="Token 无效或已过期"
        )

    # 认证通过，返回用户列表
    return {"users": ["Alice", "Bob", "Charlie"]}
```

**测试：**
```bash
# 未提供 Token
curl http://localhost:8000/admin/users
# 响应: {"detail": "未提供认证信息"}
# 状态码: 401

# 提供正确的 Token
curl -H "Authorization: Bearer secret-token" http://localhost:8000/admin/users
# 响应: {"users": ["Alice", "Bob", "Charlie"]}
# 状态码: 200
```

---

### 示例3：权限不足（403）

```python
from fastapi import FastAPI, HTTPException, Depends
from typing import Optional

app = FastAPI()

# 模拟当前用户
def get_current_user():
    return {"id": 1, "name": "Alice", "role": "user"}

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """删除用户（仅管理员）"""

    # 检查是否是管理员
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="权限不足，仅管理员可以删除用户"
        )

    # 执行删除操作
    return {"message": f"用户 {user_id} 已删除"}
```

**测试：**
```bash
curl -X DELETE http://localhost:8000/users/2
# 响应: {"detail": "权限不足，仅管理员可以删除用户"}
# 状态码: 403
```

---

### 示例4：资源冲突（409）

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 模拟数据库
users_db = {
    "alice@example.com": {"name": "Alice", "email": "alice@example.com"}
}

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users")
async def create_user(user: UserCreate):
    """创建用户"""

    # 检查邮箱是否已存在
    if user.email in users_db:
        raise HTTPException(
            status_code=409,
            detail=f"邮箱 {user.email} 已被注册"
        )

    # 创建用户
    users_db[user.email] = user.dict()
    return {"message": "用户创建成功", "user": user}
```

**测试：**
```bash
# 创建新用户
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob", "email": "bob@example.com"}'
# 响应: {"message": "用户创建成功", ...}

# 重复创建（邮箱已存在）
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'
# 响应: {"detail": "邮箱 alice@example.com 已被注册"}
# 状态码: 409
```

---

### 示例5：请求过多（429）

```python
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timedelta

app = FastAPI()

# 简单的限流器（生产环境应该用 Redis）
request_counts = {}

@app.get("/api/data")
async def get_data(request: Request):
    """获取数据（限流：每分钟最多 10 次请求）"""

    # 获取客户端 IP
    client_ip = request.client.host

    # 检查请求次数
    now = datetime.now()
    if client_ip in request_counts:
        count, last_reset = request_counts[client_ip]

        # 如果超过 1 分钟，重置计数
        if now - last_reset > timedelta(minutes=1):
            request_counts[client_ip] = (1, now)
        else:
            # 检查是否超过限制
            if count >= 10:
                raise HTTPException(
                    status_code=429,
                    detail="请求过于频繁，请稍后重试"
                )
            request_counts[client_ip] = (count + 1, last_reset)
    else:
        request_counts[client_ip] = (1, now)

    return {"data": "some data"}
```

---

## 与 Express 的对比

### Express 错误处理

```javascript
// Express: 手动设置状态码和响应
app.get('/users/:id', (req, res) => {
  const user = db.getUser(req.params.id);

  if (!user) {
    // 方式1：直接返回错误响应
    return res.status(404).json({
      error: 'User not found'
    });

    // 方式2：抛出错误（需要错误中间件）
    throw new Error('User not found');
  }

  res.json(user);
});

// 错误中间件
app.use((err, req, res, next) => {
  res.status(err.status || 500).json({
    error: err.message
  });
});
```

### FastAPI 错误处理

```python
# FastAPI: 直接抛出 HTTPException
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.get_user(user_id)

    if not user:
        # 直接抛出异常，自动处理
        raise HTTPException(status_code=404, detail="User not found")

    return user
```

**对比：**

| 特性 | Express | FastAPI |
|------|---------|---------|
| 抛出错误 | `throw new Error()` | `raise HTTPException()` |
| 设置状态码 | `res.status(404)` | `status_code=404` |
| 返回 JSON | `res.json({...})` | 自动转换 |
| 错误中间件 | 需要手动定义 | 内置处理 |
| 代码简洁度 | 中等 | 高 |

---

## 在 AI Agent 开发中的应用

### 场景1：LLM API 调用失败

```python
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError

app = FastAPI()
client = OpenAI()

@app.post("/chat")
async def chat(message: str):
    """AI 对话接口"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}]
        )
        return {"reply": response.choices[0].message.content}

    except OpenAIError as e:
        # OpenAI API 错误 → 503
        raise HTTPException(
            status_code=503,
            detail=f"AI 服务暂时不可用: {str(e)}"
        )
```

---

### 场景2：文档不存在

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

# 模拟向量数据库
documents_db = {
    "doc1": {"id": "doc1", "content": "FastAPI 教程"},
    "doc2": {"id": "doc2", "content": "Python 基础"}
}

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取文档"""
    doc = documents_db.get(doc_id)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"文档 {doc_id} 不存在"
        )

    return doc
```

---

### 场景3：Embedding 生成失败

```python
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError

app = FastAPI()
client = OpenAI()

@app.post("/embeddings")
async def create_embedding(text: str):
    """生成文本 Embedding"""

    # 检查文本长度
    if len(text) > 8000:
        raise HTTPException(
            status_code=400,
            detail="文本过长，最多支持 8000 个字符"
        )

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return {"embedding": response.data[0].embedding}

    except OpenAIError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding 生成失败: {str(e)}"
        )
```

---

## HTTPException 的内部实现

### 源码解析

```python
# FastAPI 内部实现（简化版）
class HTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
```

**关键点：**
1. `HTTPException` 继承自 Python 的 `Exception`
2. 包含三个属性：`status_code`、`detail`、`headers`
3. FastAPI 会自动捕获 `HTTPException` 并转换为 JSON 响应

---

### FastAPI 如何处理 HTTPException

```python
# FastAPI 内部流程（简化）
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

# FastAPI 内部会这样处理：
try:
    result = await get_user(user_id)
    return JSONResponse(content=result)
except HTTPException as exc:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers
    )
```

---

## 最佳实践

### 1. 使用明确的状态码

```python
# ✅ 好：状态码明确
raise HTTPException(status_code=404, detail="用户不存在")
raise HTTPException(status_code=401, detail="未授权")
raise HTTPException(status_code=403, detail="权限不足")

# ❌ 差：都用 400
raise HTTPException(status_code=400, detail="用户不存在")  # 应该是 404
raise HTTPException(status_code=400, detail="未授权")      # 应该是 401
```

---

### 2. 提供有用的错误信息

```python
# ✅ 好：错误信息清晰
raise HTTPException(
    status_code=404,
    detail=f"用户 {user_id} 不存在，请检查用户 ID 是否正确"
)

# ❌ 差：错误信息模糊
raise HTTPException(status_code=404, detail="Not found")
```

---

### 3. 避免泄露敏感信息

```python
# ✅ 好：不泄露内部错误
try:
    result = db.query(...)
except Exception as e:
    logger.error(f"数据库错误: {e}")  # 记录详细日志
    raise HTTPException(
        status_code=500,
        detail="服务器内部错误"  # 用户只看到通用错误
    )

# ❌ 差：泄露数据库信息
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"数据库错误: {str(e)}"  # 暴露了数据库结构
    )
```

---

### 4. 使用自定义响应头

```python
# ✅ 好：添加有用的响应头
raise HTTPException(
    status_code=401,
    detail="Token 已过期",
    headers={"WWW-Authenticate": "Bearer"}  # 告诉客户端需要 Bearer Token
)

# ✅ 好：添加重试信息
raise HTTPException(
    status_code=429,
    detail="请求过多",
    headers={"Retry-After": "60"}  # 告诉客户端 60 秒后重试
)
```

---

## 常见问题

### Q1: HTTPException 和 Python 的 Exception 有什么区别？

**A:**
- `Exception` 是 Python 的通用异常，FastAPI 会捕获并返回 500 错误
- `HTTPException` 是 FastAPI 专用异常，可以指定 HTTP 状态码和响应内容

```python
# Python Exception → 500 错误
raise Exception("出错了")  # 返回 500

# HTTPException → 自定义状态码
raise HTTPException(status_code=404, detail="出错了")  # 返回 404
```

---

### Q2: 什么时候用 HTTPException，什么时候用自定义异常？

**A:**
- **HTTPException**：简单场景，直接抛错
- **自定义异常**：复杂业务逻辑，需要统一处理

```python
# 简单场景：直接用 HTTPException
if not user:
    raise HTTPException(status_code=404, detail="用户不存在")

# 复杂场景：自定义异常（下一节详细讲解）
class UserNotFoundError(Exception):
    pass

if not user:
    raise UserNotFoundError(f"用户 {user_id} 不存在")
```

---

### Q3: HTTPException 可以被 try/except 捕获吗？

**A:** 可以，但通常不需要。FastAPI 会自动捕获并处理。

```python
# 通常不需要手动捕获
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user

# 特殊情况：需要在抛出前做额外处理
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    try:
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
    except HTTPException as e:
        logger.warning(f"用户 {user_id} 不存在")
        raise  # 重新抛出，让 FastAPI 处理
```

---

## 小结

**HTTPException 是 FastAPI 异常处理的基础：**

1. **简单易用**：一行代码抛出 HTTP 错误
2. **自动处理**：FastAPI 自动转换为 JSON 响应
3. **状态码明确**：支持所有 HTTP 状态码
4. **可扩展**：支持自定义响应头和详细错误信息
5. **符合习惯**：类似 Python 的 `raise Exception()`

**下一步：** 学习如何创建自定义异常处理器，实现更精细的错误处理逻辑。

---

**相关文档：**
- [FastAPI 官方文档 - 异常处理](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [HTTP 状态码完整列表](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Status)
