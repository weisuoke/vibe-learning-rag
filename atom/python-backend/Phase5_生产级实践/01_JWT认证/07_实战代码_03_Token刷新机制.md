# JWT认证 - 实战代码03：Token刷新机制

## 概述

本章实现 Access Token + Refresh Token 双Token机制，实现无感知的Token刷新。

---

## 双Token设计

### 为什么需要两个Token？

**问题：单Token的困境**
- Access Token 过期时间长（7天）→ 安全风险高
- Access Token 过期时间短（15分钟）→ 用户体验差

**解决：双Token机制**
- Access Token：短期（15分钟），频繁使用
- Refresh Token：长期（7天），只用于刷新

---

## 生成双Token

### Token生成函数

```python
"""
双Token生成
演示：生成 Access Token 和 Refresh Token
"""

from jose import jwt
from datetime import datetime, timedelta
import os

# 配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY", "your-refresh-secret-key")
ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_access_token(user_id: int, username: str) -> str:
    """生成 Access Token（短期）"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "user_id": user_id,
        "username": username,
        "type": "access",  # Token 类型
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """生成 Refresh Token（长期）"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "user_id": user_id,
        "type": "refresh",  # Token 类型
        "exp": expire
    }

    # 使用不同的密钥
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)


# 使用示例
access_token = create_access_token(user_id=123, username="alice")
refresh_token = create_refresh_token(user_id=123)

print(f"Access Token (15分钟): {access_token[:50]}...")
print(f"Refresh Token (7天): {refresh_token[:50]}...")
```

---

## 登录返回双Token

### 修改登录端点

```python
"""
登录返回双Token
演示：登录 → 返回 Access Token + Refresh Token
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

app = FastAPI()


class LoginRequest(BaseModel):
    """登录请求"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """登录端点：返回双Token"""
    # 1. 查找用户
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(401, "邮箱或密码错误")

    # 2. 验证密码
    if not PasswordManager.verify_password(request.password, user.hashed_password):
        raise HTTPException(401, "邮箱或密码错误")

    # 3. 生成双Token
    access_token = create_access_token(user.id, user.username)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )
```

---

## Token刷新端点

### 刷新实现

```python
"""
Token刷新端点
演示：用 Refresh Token 换取新的双Token
"""

from jose import jwt, JWTError


class RefreshRequest(BaseModel):
    """刷新请求"""
    refresh_token: str


@app.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    """刷新Token端点"""
    try:
        # 1. 验证 Refresh Token
        payload = jwt.decode(
            request.refresh_token,
            REFRESH_SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # 2. 检查Token类型
        if payload.get("type") != "refresh":
            raise HTTPException(400, "无效的Token类型")

        user_id = payload.get("user_id")

        # 3. 查找用户
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(404, "用户不存在")

        # 4. 检查用户状态
        if not user.is_active:
            raise HTTPException(403, "账号已被禁用")

        # 5. 生成新的双Token
        new_access_token = create_access_token(user.id, user.username)
        new_refresh_token = create_refresh_token(user.id)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh Token已过期，请重新登录")

    except JWTError:
        raise HTTPException(401, "无效的Refresh Token")
```

---

## 前端自动刷新

### JavaScript实现

```javascript
/**
 * 前端自动刷新Token
 * 演示：拦截401错误 → 自动刷新 → 重试请求
 */

// 存储Token
let accessToken = localStorage.getItem('access_token');
let refreshToken = localStorage.getItem('refresh_token');

// Axios拦截器
axios.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;

    // 如果是401错误且未重试过
    if (error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // 用Refresh Token刷新
        const response = await axios.post('/refresh', {
          refresh_token: refreshToken
        });

        // 保存新Token
        const { access_token, refresh_token } = response.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);

        accessToken = access_token;
        refreshToken = refresh_token;

        // 更新原请求的Token
        originalRequest.headers['Authorization'] = `Bearer ${access_token}`;

        // 重试原请求
        return axios(originalRequest);

      } catch (refreshError) {
        // Refresh Token也过期了，跳转到登录页
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// 使用示例
async function callAPI() {
  try {
    const response = await axios.get('/api/data', {
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    return response.data;
  } catch (error) {
    console.error('API调用失败:', error);
  }
}
```

---

## Token黑名单

### Redis黑名单实现

```python
"""
Token黑名单
演示：登出时将Refresh Token加入黑名单
"""

from redis import Redis
from datetime import datetime

redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)


@app.post("/logout")
async def logout(
    request: RefreshRequest,
    current_user: dict = Depends(get_current_user)
):
    """登出端点：将Refresh Token加入黑名单"""
    try:
        # 1. 解析Refresh Token
        payload = jwt.decode(
            request.refresh_token,
            REFRESH_SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # 2. 计算剩余有效时间
        exp = payload.get("exp")
        now = int(datetime.utcnow().timestamp())
        ttl = exp - now

        if ttl > 0:
            # 3. 加入黑名单（只需存到过期时间）
            redis_client.setex(
                f"blacklist:refresh:{request.refresh_token}",
                ttl,
                "1"
            )

        return {"message": "登出成功"}

    except JWTError:
        raise HTTPException(401, "无效的Refresh Token")


# 刷新时检查黑名单
@app.post("/refresh")
async def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    """刷新Token（检查黑名单）"""
    # 1. 检查黑名单
    if redis_client.exists(f"blacklist:refresh:{request.refresh_token}"):
        raise HTTPException(401, "Token已被撤销，请重新登录")

    # 2. 验证并刷新...
    # （其余代码同上）
```

---

## 完整示例

### 完整的Token刷新系统

```python
"""
完整的Token刷新系统
演示：登录 → 使用 → 刷新 → 登出
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from redis import Redis
import os

app = FastAPI()
security = HTTPBearer()

# 配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY", "your-refresh-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Redis
redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)


# ===== Token生成 =====

def create_access_token(user_id: int, username: str) -> str:
    """生成Access Token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "username": username,
        "type": "access",
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """生成Refresh Token"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "user_id": user_id,
        "type": "refresh",
        "exp": expire
    }
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)


# ===== 数据模型 =====

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ===== 依赖函数 =====

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> dict:
    """获取当前用户（验证Access Token）"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        if payload.get("type") != "access":
            raise HTTPException(400, "无效的Token类型")

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token已过期")

    except JWTError:
        raise HTTPException(401, "无效的Token")


# ===== 路由端点 =====

@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """登录：返回双Token"""
    # 查找用户
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(401, "邮箱或密码错误")

    # 验证密码
    if not PasswordManager.verify_password(request.password, user.hashed_password):
        raise HTTPException(401, "邮箱或密码错误")

    # 生成双Token
    access_token = create_access_token(user.id, user.username)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@app.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest, db: Session = Depends(get_db)):
    """刷新：用Refresh Token换取新Token"""
    # 检查黑名单
    if redis_client.exists(f"blacklist:refresh:{request.refresh_token}"):
        raise HTTPException(401, "Token已被撤销")

    try:
        # 验证Refresh Token
        payload = jwt.decode(
            request.refresh_token,
            REFRESH_SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        if payload.get("type") != "refresh":
            raise HTTPException(400, "无效的Token类型")

        user_id = payload.get("user_id")

        # 查找用户
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(401, "用户不存在或已禁用")

        # 生成新Token
        new_access_token = create_access_token(user.id, user.username)
        new_refresh_token = create_refresh_token(user.id)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh Token已过期")

    except JWTError:
        raise HTTPException(401, "无效的Refresh Token")


@app.post("/logout")
async def logout(
    request: RefreshRequest,
    current_user: dict = Depends(get_current_user)
):
    """登出：将Refresh Token加入黑名单"""
    try:
        payload = jwt.decode(
            request.refresh_token,
            REFRESH_SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        exp = payload.get("exp")
        ttl = exp - int(datetime.utcnow().timestamp())

        if ttl > 0:
            redis_client.setex(
                f"blacklist:refresh:{request.refresh_token}",
                ttl,
                "1"
            )

        return {"message": "登出成功"}

    except JWTError:
        raise HTTPException(401, "无效的Refresh Token")


@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """受保护的端点"""
    return {
        "message": f"欢迎，{current_user['username']}",
        "user_id": current_user["user_id"]
    }
```

---

## 测试流程

### 完整测试

```bash
# 1. 登录获取双Token
RESPONSE=$(curl -s -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "password": "SecurePass123!"}')

ACCESS_TOKEN=$(echo $RESPONSE | jq -r '.access_token')
REFRESH_TOKEN=$(echo $RESPONSE | jq -r '.refresh_token')

echo "Access Token: ${ACCESS_TOKEN:0:50}..."
echo "Refresh Token: ${REFRESH_TOKEN:0:50}..."

# 2. 使用Access Token访问API
curl -X GET "http://localhost:8000/protected" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 3. 等待Access Token过期（15分钟后）或手动测试刷新
curl -X POST "http://localhost:8000/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}"

# 4. 登出
curl -X POST "http://localhost:8000/logout" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}"

# 5. 尝试用已登出的Refresh Token刷新（应该失败）
curl -X POST "http://localhost:8000/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}"
```

---

## 在AI Agent API中的应用

### AI Agent对话端点

```python
@app.post("/agent/chat")
async def chat(
    message: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI Agent对话（需要Access Token）"""
    user_id = current_user["user_id"]
    username = current_user["username"]

    # 调用AI Agent
    response = f"AI回复给{username}: {message}"

    return {
        "user_id": user_id,
        "message": message,
        "response": response
    }
```

**前端使用：**
```javascript
// 调用AI Agent（自动刷新Token）
async function chatWithAgent(message) {
  try {
    const response = await axios.post('/agent/chat',
      { message },
      {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    // 拦截器会自动刷新Token并重试
    console.error('对话失败:', error);
  }
}
```

---

## 总结

**本章实现了：**
1. 双Token生成（Access + Refresh）
2. 登录返回双Token
3. Token刷新端点
4. 前端自动刷新
5. Token黑名单（Redis）
6. 完整的刷新系统

**关键点：**
- Access Token短期（15分钟）
- Refresh Token长期（7天）
- 使用不同的密钥
- Refresh Token使用黑名单
- 前端自动刷新无感知

**下一步：**
- 学习依赖注入保护路由
- 学习RBAC权限控制
- 学习生产级最佳实践
