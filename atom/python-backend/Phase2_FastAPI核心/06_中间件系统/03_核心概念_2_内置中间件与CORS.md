# 核心概念 2：内置中间件与 CORS

本文详细讲解 FastAPI 提供的内置中间件，重点介绍 CORS 中间件的配置和使用。

---

## 1. FastAPI 内置中间件概览

FastAPI 基于 Starlette 构建，提供了多个开箱即用的内置中间件。

### 1.1 内置中间件列表

| 中间件 | 用途 | 典型场景 |
|--------|------|----------|
| **CORSMiddleware** | 处理跨域请求 | 前端调用后端 API |
| **GZipMiddleware** | 压缩响应内容 | 减少传输数据量 |
| **TrustedHostMiddleware** | 验证 Host 头 | 防止 Host 头攻击 |
| **HTTPSRedirectMiddleware** | 重定向到 HTTPS | 强制使用 HTTPS |
| **SessionMiddleware** | 会话管理 | 存储用户会话数据 |

### 1.2 使用方式

所有内置中间件都使用 `app.add_middleware()` 方法添加。

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

# 添加内置中间件
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["example.com"])
app.add_middleware(HTTPSRedirectMiddleware)
```

---

## 2. CORSMiddleware 详解

### 2.1 什么是 CORS？

**CORS（Cross-Origin Resource Sharing，跨域资源共享）** 是一种浏览器安全机制，用于控制跨域请求。

**为什么需要 CORS？**

浏览器的**同源策略**（Same-Origin Policy）规定：
- 只有同源（协议、域名、端口都相同）的页面才能互相访问资源
- 不同源的页面默认不能互相访问

**示例：**

```
前端：http://localhost:3000
后端：http://localhost:8000

这是不同源（端口不同），浏览器会阻止前端调用后端 API
```

**CORS 的作用：**

CORS 允许服务器明确告诉浏览器："我允许来自 http://localhost:3000 的请求"。

### 2.2 CORS 的工作原理

**简单请求：**

```
1. 浏览器发送请求
   GET /api/users HTTP/1.1
   Origin: http://localhost:3000

2. 服务器返回响应
   HTTP/1.1 200 OK
   Access-Control-Allow-Origin: http://localhost:3000

3. 浏览器检查响应头
   - 如果 Access-Control-Allow-Origin 包含请求的 Origin，允许访问
   - 否则，阻止访问
```

**预检请求（Preflight Request）：**

对于复杂请求（如 POST、PUT、DELETE，或自定义请求头），浏览器会先发送一个 OPTIONS 请求。

```
1. 浏览器发送预检请求
   OPTIONS /api/users HTTP/1.1
   Origin: http://localhost:3000
   Access-Control-Request-Method: POST
   Access-Control-Request-Headers: Content-Type, Authorization

2. 服务器返回预检响应
   HTTP/1.1 200 OK
   Access-Control-Allow-Origin: http://localhost:3000
   Access-Control-Allow-Methods: POST, GET, OPTIONS
   Access-Control-Allow-Headers: Content-Type, Authorization

3. 浏览器检查预检响应
   - 如果允许，发送实际请求
   - 否则，阻止请求
```

### 2.3 CORSMiddleware 配置参数

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许的源
    allow_credentials=True,                    # 允许携带 Cookie
    allow_methods=["*"],                       # 允许的 HTTP 方法
    allow_headers=["*"],                       # 允许的请求头
    expose_headers=["X-Custom-Header"],        # 暴露的响应头
    max_age=600,                               # 预检请求缓存时间（秒）
)
```

**参数详解：**

#### allow_origins

**作用：** 指定允许的源（域名）

**选项：**
- `["*"]`：允许所有源（不推荐用于生产环境）
- `["http://localhost:3000"]`：只允许指定的源
- `["http://localhost:3000", "https://example.com"]`：允许多个源

**示例：**

```python
# 开发环境：允许所有源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

# 生产环境：只允许指定的源
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://example.com",
        "https://www.example.com",
        "https://app.example.com",
    ],
)
```

#### allow_credentials

**作用：** 是否允许携带 Cookie 和认证信息

**选项：**
- `True`：允许携带 Cookie
- `False`：不允许携带 Cookie（默认）

**注意：** 如果设置为 `True`，`allow_origins` 不能是 `["*"]`，必须指定具体的源。

```python
# ❌ 错误：allow_credentials=True 时不能使用 allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # 会报错
)

# ✅ 正确：指定具体的源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
)
```

#### allow_methods

**作用：** 指定允许的 HTTP 方法

**选项：**
- `["*"]`：允许所有方法
- `["GET", "POST"]`：只允许指定的方法

**示例：**

```python
# 允许所有方法
app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
)

# 只允许读操作
app.add_middleware(
    CORSMiddleware,
    allow_methods=["GET", "HEAD", "OPTIONS"],
)

# 允许常用方法
app.add_middleware(
    CORSMiddleware,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)
```

#### allow_headers

**作用：** 指定允许的请求头

**选项：**
- `["*"]`：允许所有请求头
- `["Content-Type", "Authorization"]`：只允许指定的请求头

**示例：**

```python
# 允许所有请求头
app.add_middleware(
    CORSMiddleware,
    allow_headers=["*"],
)

# 只允许常用请求头
app.add_middleware(
    CORSMiddleware,
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
    ],
)
```

#### expose_headers

**作用：** 指定哪些响应头可以被前端 JavaScript 访问

**默认情况下，前端只能访问以下响应头：**
- Cache-Control
- Content-Language
- Content-Type
- Expires
- Last-Modified
- Pragma

**如果你想让前端访问自定义响应头，需要在 `expose_headers` 中指定：**

```python
app.add_middleware(
    CORSMiddleware,
    expose_headers=[
        "X-Request-ID",
        "X-Process-Time",
        "X-Rate-Limit-Remaining",
    ],
)

@app.get("/api/data")
async def get_data():
    response = JSONResponse({"data": "..."})
    response.headers["X-Request-ID"] = "req_123"
    response.headers["X-Process-Time"] = "0.123"
    return response

# 前端可以访问这些响应头
# const requestId = response.headers.get('X-Request-ID');
```

#### max_age

**作用：** 预检请求的缓存时间（秒）

**说明：** 浏览器会缓存预检请求的结果，在缓存期内不会再次发送预检请求。

```python
app.add_middleware(
    CORSMiddleware,
    max_age=600,  # 缓存 10 分钟
)
```

### 2.4 CORS 配置示例

#### 示例1：开发环境配置

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 开发环境：宽松配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=False,  # 不允许携带 Cookie
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)
```

#### 示例2：生产环境配置

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# 生产环境：严格配置
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # 只允许指定的源
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 只允许常用方法
    allow_headers=["Content-Type", "Authorization"],  # 只允许必要的请求头
    expose_headers=["X-Request-ID"],  # 暴露自定义响应头
    max_age=600,  # 缓存预检请求 10 分钟
)
```

#### 示例3：AI Agent 后端配置

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# AI Agent 后端：允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # 本地开发
        "https://app.example.com",  # 生产环境
    ],
    allow_credentials=True,  # 允许携带 JWT
    allow_methods=["GET", "POST", "OPTIONS"],  # Agent API 只需要这些方法
    allow_headers=["Content-Type", "Authorization"],  # 只需要这两个请求头
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"],  # 暴露请求 ID 和限流信息
    max_age=3600,  # 缓存 1 小时
)
```

### 2.5 CORS 调试技巧

#### 问题1：前端报错 "CORS policy: No 'Access-Control-Allow-Origin' header"

**原因：** 后端没有配置 CORS 中间件，或者 `allow_origins` 没有包含前端的源。

**解决：**

```python
# 检查 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 确保包含前端的源
)
```

#### 问题2：前端报错 "CORS policy: Credentials flag is 'true', but 'Access-Control-Allow-Credentials' header is ''"

**原因：** 前端请求携带了 Cookie（`credentials: 'include'`），但后端没有设置 `allow_credentials=True`。

**解决：**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,  # 允许携带 Cookie
)
```

#### 问题3：前端报错 "CORS policy: Request header field authorization is not allowed"

**原因：** 前端请求包含了 `Authorization` 请求头，但后端没有在 `allow_headers` 中允许。

**解决：**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_headers=["Content-Type", "Authorization"],  # 允许 Authorization 请求头
)
```

#### 问题4：OPTIONS 请求返回 404

**原因：** FastAPI 没有自动处理 OPTIONS 请求。

**解决：** 添加 CORS 中间件后，FastAPI 会自动处理 OPTIONS 请求。

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],  # 确保包含 OPTIONS
)
```

---

## 3. GZipMiddleware 详解

### 3.1 什么是 GZip 压缩？

**GZip** 是一种数据压缩算法，可以减少传输的数据量，提高响应速度。

**工作原理：**

```
1. 客户端发送请求
   GET /api/data HTTP/1.1
   Accept-Encoding: gzip

2. 服务器压缩响应
   HTTP/1.1 200 OK
   Content-Encoding: gzip
   Content-Length: 1234  (压缩后的大小)

3. 客户端解压响应
   浏览器自动解压 gzip 数据
```

### 3.2 GZipMiddleware 配置

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # 只压缩大于 1000 字节的响应
)
```

**参数说明：**

- `minimum_size`：最小压缩大小（字节）
  - 小于这个大小的响应不会被压缩
  - 默认值：1000
  - 推荐值：500-2000

**为什么需要 minimum_size？**

因为压缩本身也有开销，对于很小的响应，压缩反而会增加大小和耗时。

### 3.3 GZip 压缩效果

**示例：**

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/api/large-data")
async def get_large_data():
    # 返回大量数据
    return {
        "users": [
            {"id": i, "name": f"User {i}", "email": f"user{i}@example.com"}
            for i in range(1000)
        ]
    }

# 不压缩：约 50KB
# 压缩后：约 5KB（压缩率 90%）
```

### 3.4 何时使用 GZip？

**适合压缩：**
- JSON 响应（压缩率高）
- HTML 页面（压缩率高）
- 文本文件（压缩率高）

**不适合压缩：**
- 图片（已经压缩过）
- 视频（已经压缩过）
- 小文件（压缩开销大于收益）

---

## 4. TrustedHostMiddleware 详解

### 4.1 什么是 Host 头攻击？

**Host 头攻击** 是一种安全漏洞，攻击者通过伪造 Host 头来欺骗服务器。

**示例：**

```
正常请求：
GET /api/data HTTP/1.1
Host: example.com

恶意请求：
GET /api/data HTTP/1.1
Host: evil.com

如果服务器使用 Host 头生成链接，可能会生成指向 evil.com 的链接
```

### 4.2 TrustedHostMiddleware 配置

```python
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "example.com",
        "www.example.com",
        "*.example.com",  # 通配符
    ],
)
```

**参数说明：**

- `allowed_hosts`：允许的主机名列表
  - 可以使用通配符 `*`
  - 如果请求的 Host 不在列表中，返回 400 错误

### 4.3 何时使用 TrustedHostMiddleware？

**推荐使用：**
- 生产环境
- 公开的 API
- 使用 Host 头生成链接的应用

**不需要使用：**
- 本地开发
- 内网应用
- 不使用 Host 头的应用

---

## 5. HTTPSRedirectMiddleware 详解

### 5.1 什么是 HTTPS 重定向？

**HTTPS 重定向** 是将 HTTP 请求自动重定向到 HTTPS。

**工作原理：**

```
1. 客户端发送 HTTP 请求
   GET http://example.com/api/data

2. 服务器返回 307 重定向
   HTTP/1.1 307 Temporary Redirect
   Location: https://example.com/api/data

3. 客户端发送 HTTPS 请求
   GET https://example.com/api/data
```

### 5.2 HTTPSRedirectMiddleware 配置

```python
from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

app.add_middleware(HTTPSRedirectMiddleware)
```

**注意：** 这个中间件会将所有 HTTP 请求重定向到 HTTPS，确保你的服务器支持 HTTPS。

### 5.3 何时使用 HTTPSRedirectMiddleware？

**推荐使用：**
- 生产环境
- 公开的 API
- 需要保护用户隐私的应用

**不需要使用：**
- 本地开发（localhost 不需要 HTTPS）
- 内网应用
- 已经在负载均衡器层面处理 HTTPS 的应用

---

## 6. SessionMiddleware 详解

### 6.1 什么是会话（Session）？

**会话** 是一种在服务器端存储用户状态的机制。

**工作原理：**

```
1. 用户登录
   POST /login
   → 服务器生成 session_id，存储用户信息
   → 返回 Set-Cookie: session=session_id

2. 后续请求
   GET /api/data
   Cookie: session=session_id
   → 服务器根据 session_id 获取用户信息
```

### 6.2 SessionMiddleware 配置

```python
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key-here",  # 用于加密 session
    session_cookie="session",  # Cookie 名称
    max_age=3600,  # 会话过期时间（秒）
    same_site="lax",  # SameSite 属性
    https_only=False,  # 是否只在 HTTPS 下发送
)

@app.get("/login")
async def login(request: Request):
    # 设置 session
    request.session["user_id"] = "user_123"
    request.session["username"] = "Alice"
    return {"message": "Logged in"}

@app.get("/profile")
async def get_profile(request: Request):
    # 读取 session
    user_id = request.session.get("user_id")
    username = request.session.get("username")

    if not user_id:
        return {"error": "Not logged in"}

    return {"user_id": user_id, "username": username}

@app.get("/logout")
async def logout(request: Request):
    # 清除 session
    request.session.clear()
    return {"message": "Logged out"}
```

### 6.3 Session vs JWT

| 特性 | Session | JWT |
|------|---------|-----|
| **存储位置** | 服务器端 | 客户端 |
| **状态** | 有状态 | 无状态 |
| **扩展性** | 需要共享存储 | 天然支持分布式 |
| **安全性** | 服务器控制 | 客户端可见（需加密） |
| **性能** | 需要查询存储 | 无需查询 |
| **适用场景** | 传统 Web 应用 | API、微服务 |

**AI Agent 后端推荐使用 JWT，不推荐使用 Session。**

---

## 7. 内置中间件的组合使用

### 7.1 推荐的中间件组合

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

# 1. HTTPS 重定向（最外层）
app.add_middleware(HTTPSRedirectMiddleware)

# 2. 可信主机验证
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"],
)

# 3. CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. GZip 压缩（最内层）
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 7.2 中间件顺序的考虑

**原则：**
1. **HTTPS 重定向**：最外层，确保所有请求都是 HTTPS
2. **可信主机验证**：外层，尽早拒绝恶意请求
3. **CORS 配置**：中层，处理跨域请求
4. **GZip 压缩**：最内层，压缩最终的响应

---

## 8. 在 AI Agent 后端中的应用

### 8.1 AI Agent 后端的中间件配置

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import os

app = FastAPI()

# 1. CORS：允许前端跨域调用
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # 允许携带 JWT
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"],
    max_age=3600,
)

# 2. GZip：压缩大响应（如长文本生成）
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 业务路由
@app.post("/api/agent/chat")
async def chat(message: str):
    # 调用 LLM API，生成长文本
    response = await call_llm(message)
    return {"response": response}  # 自动压缩
```

### 8.2 为什么 AI Agent 后端需要这些中间件？

**CORS：**
- 前端（React/Vue）需要跨域调用后端 API
- 必须配置 CORS，否则浏览器会阻止请求

**GZip：**
- LLM 生成的文本通常很长（几千字）
- 压缩可以减少传输时间，提升用户体验

**不需要 Session：**
- AI Agent 后端通常是无状态的
- 使用 JWT 认证，不需要 Session

---

## 总结

**内置中间件与 CORS 的核心要点：**

1. **CORS 是必需的**：前端调用后端 API 必须配置 CORS
2. **CORS 配置要严格**：生产环境不要使用 `allow_origins=["*"]`
3. **GZip 可以提升性能**：压缩大响应，减少传输时间
4. **TrustedHost 提升安全性**：防止 Host 头攻击
5. **HTTPSRedirect 保护隐私**：强制使用 HTTPS
6. **Session 不适合 API**：AI Agent 后端推荐使用 JWT

**下一步学习：**
- 核心概念 3：自定义中间件开发
- 实战代码：各种场景的中间件实现
