# 【核心概念3】FastAPI依赖注入集成

> get_db、中间件、异常处理的详细讲解

---

## 概述

**FastAPI 的依赖注入系统是管理 Session 生命周期的最佳方式。**

```
┌─────────────────────────────────────┐
│  FastAPI 请求                        │
├─────────────────────────────────────┤
│  1. 依赖注入：get_db()               │
│     ├─ 创建 Session                  │
│     └─ yield Session                 │
├─────────────────────────────────────┤
│  2. 路由处理：使用 Session           │
│     ├─ 查询数据                      │
│     ├─ 修改数据                      │
│     └─ 提交事务                      │
├─────────────────────────────────────┤
│  3. 依赖清理：finally                │
│     └─ 关闭 Session                  │
└─────────────────────────────────────┘
```

**核心优势：**
- 自动管理 Session 生命周期
- 异常安全（自动关闭 Session）
- 代码复用（多个路由共享）
- 测试友好（可以替换依赖）

---

## 1. 基础依赖注入

### 1.1 创建 get_db 函数

**get_db 是一个生成器函数，用于创建和管理 Session。**

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**工作原理：**

```
1. 请求到达
   ↓
2. FastAPI 调用 get_db()
   ├─ 创建 Session
   └─ yield Session
   ↓
3. 路由处理函数接收 Session
   ├─ 使用 Session
   └─ 返回响应
   ↓
4. FastAPI 继续执行 get_db()
   └─ finally: 关闭 Session
   ↓
5. 响应返回给客户端
```

---

### 1.2 在路由中使用依赖注入

**使用 Depends(get_db) 注入 Session。**

```python
# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db

app = FastAPI()

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

**类型注解：**
- `db: Session`：告诉 IDE 这是一个 Session 对象
- `= Depends(get_db)`：告诉 FastAPI 使用 get_db 函数注入

---

### 1.3 依赖注入的执行顺序

**FastAPI 会按照依赖关系的顺序执行。**

```python
from fastapi import Depends

def get_db():
    print("1. Creating session")
    db = SessionLocal()
    try:
        yield db
        print("4. Closing session")
    finally:
        db.close()
        print("5. Session closed")

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    print("2. Using session")
    users = db.query(User).all()
    print("3. Returning response")
    return users

# 执行顺序：
# 1. Creating session
# 2. Using session
# 3. Returning response
# 4. Closing session
# 5. Session closed
```

---

## 2. 异常处理

### 2.1 自动回滚

**当路由处理函数抛出异常时，Session 会自动关闭，但不会自动回滚。**

```python
# ❌ 错误：异常时不会自动回滚
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()  # 如果这里抛出异常，数据可能部分提交
    return db_user
```

**正确做法：在 get_db 中处理异常回滚。**

```python
# ✅ 正确：异常时自动回滚
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    return db_user
```

---

### 2.2 HTTPException 处理

**HTTPException 不会触发 except 块，需要单独处理。**

```python
from fastapi import HTTPException

def get_db():
    db = SessionLocal()
    try:
        yield db
    except HTTPException:
        # HTTPException 不需要回滚
        raise
    except Exception:
        # 其他异常需要回滚
        db.rollback()
        raise
    finally:
        db.close()

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

### 2.3 完整的异常处理

**推荐的 get_db 实现，处理所有异常情况。**

```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
        # 如果没有异常，正常结束
    except HTTPException:
        # HTTPException 不需要回滚，直接抛出
        raise
    except Exception as e:
        # 其他异常需要回滚
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        # 无论如何都要关闭 Session
        db.close()
```

---

## 3. 事务管理模式

### 3.1 手动提交模式（推荐）

**在路由处理函数中手动调用 commit。**

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()  # 手动提交
    db.refresh(db_user)
    return db_user
```

**优点：**
- 灵活控制事务边界
- 可以在提交前进行验证
- 适合复杂的业务逻辑

**缺点：**
- 容易忘记 commit
- 代码重复

---

### 3.2 自动提交模式

**在 get_db 中自动提交事务。**

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()  # 自动提交
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    # 不需要手动 commit
    return db_user
```

**优点：**
- 不需要手动 commit
- 代码简洁

**缺点：**
- 失去对事务边界的控制
- 不适合复杂的业务逻辑

---

### 3.3 混合模式

**默认自动提交，但允许手动控制。**

```python
def get_db(auto_commit: bool = True):
    db = SessionLocal()
    try:
        yield db
        if auto_commit:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# 自动提交
@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    return db_user

# 手动提交
@app.post("/users/batch")
def create_users(users: list[UserCreate], db: Session = Depends(lambda: get_db(auto_commit=False))):
    for user in users:
        db_user = User(**user.dict())
        db.add(db_user)
    db.commit()  # 手动提交
    return {"count": len(users)}
```

---

## 4. 依赖注入的高级用法

### 4.1 嵌套依赖

**依赖可以依赖其他依赖。**

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(db: Session = Depends(get_db)):
    # 从请求头获取 token
    token = request.headers.get("Authorization")
    user = db.query(User).filter_by(token=token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
```

**执行顺序：**
```
1. FastAPI 调用 get_db()
   ├─ 创建 Session
   └─ yield Session

2. FastAPI 调用 get_current_user(db)
   ├─ 接收 Session
   ├─ 查询用户
   └─ 返回 User

3. FastAPI 调用 get_me(current_user)
   ├─ 接收 User
   └─ 返回响应

4. FastAPI 清理 get_db()
   └─ 关闭 Session
```

---

### 4.2 依赖缓存

**同一个请求中，依赖只会执行一次。**

```python
def get_db():
    print("Creating session")
    db = SessionLocal()
    try:
        yield db
    finally:
        print("Closing session")
        db.close()

@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    db1: Session = Depends(get_db),
    db2: Session = Depends(get_db),
):
    print(db1 is db2)  # True（同一个 Session）
    return db1.query(User).filter_by(id=user_id).first()

# 输出：
# Creating session
# True
# Closing session
```

**禁用缓存：**

```python
from fastapi import Depends

@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    db1: Session = Depends(get_db, use_cache=False),
    db2: Session = Depends(get_db, use_cache=False),
):
    print(db1 is db2)  # False（不同的 Session）
    return db1.query(User).filter_by(id=user_id).first()
```

---

### 4.3 类依赖

**使用类作为依赖。**

```python
from fastapi import Depends

class DatabaseSession:
    def __init__(self):
        self.db = SessionLocal()

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        self.db.close()

def get_db():
    with DatabaseSession() as db:
        yield db

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

---

## 5. 中间件集成

### 5.1 Session 中间件

**使用中间件管理 Session 生命周期。**

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class DatabaseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 创建 Session
        db = SessionLocal()
        request.state.db = db

        try:
            # 处理请求
            response = await call_next(request)

            # 提交事务
            db.commit()
            return response

        except Exception as e:
            # 回滚事务
            db.rollback()
            raise

        finally:
            # 关闭 Session
            db.close()

# 注册中间件
app.add_middleware(DatabaseMiddleware)

# 在路由中使用
@app.get("/users")
def get_users(request: Request):
    db = request.state.db
    return db.query(User).all()
```

**优点：**
- 自动管理 Session 生命周期
- 不需要在每个路由中注入依赖

**缺点：**
- 失去对事务边界的控制
- 不适合复杂的业务逻辑
- 不推荐使用（依赖注入更灵活）

---

### 5.2 事务中间件

**使用中间件自动管理事务。**

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TransactionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 只对写操作启用事务
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            db = SessionLocal()
            request.state.db = db

            try:
                response = await call_next(request)
                db.commit()
                return response
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            # 读操作不需要事务
            return await call_next(request)

app.add_middleware(TransactionMiddleware)
```

---

## 6. 异步支持

### 6.1 异步 Session

**使用 AsyncSession 支持异步操作。**

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 创建异步引擎
engine = create_async_engine("postgresql+asyncpg://user:password@localhost/mydb")

# 创建异步 SessionMaker
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 异步依赖注入
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# 异步路由
@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

---

### 6.2 异步事务管理

**使用 async with 管理异步事务。**

```python
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@app.post("/users")
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    # 自动提交
    return db_user
```

---

## 7. 测试集成

### 7.1 测试依赖覆盖

**在测试中替换 get_db 依赖。**

```python
# test_main.py
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建测试数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine)

# 创建测试依赖
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# 覆盖依赖
app.dependency_overrides[get_db] = override_get_db

# 测试
client = TestClient(app)

def test_create_user():
    response = client.post("/users", json={"name": "Alice"})
    assert response.status_code == 200
    assert response.json()["name"] == "Alice"
```

---

### 7.2 测试事务回滚

**在测试中自动回滚事务。**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db():
    # 创建测试数据库
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(bind=engine)

    # 创建 Session
    db = TestingSessionLocal()

    # 开始事务
    db.begin()

    yield db

    # 回滚事务
    db.rollback()
    db.close()

def test_create_user(db):
    user = User(name="Alice")
    db.add(user)
    db.commit()

    # 验证
    assert db.query(User).count() == 1

    # 测试结束后自动回滚
```

---

## 8. 最佳实践

### 8.1 推荐的 get_db 实现

```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def get_db():
    """
    数据库 Session 依赖注入

    自动管理 Session 生命周期：
    - 创建 Session
    - 异常时回滚
    - 最终关闭 Session
    """
    db = SessionLocal()
    try:
        yield db
    except HTTPException:
        # HTTPException 不需要回滚
        raise
    except Exception as e:
        # 其他异常需要回滚
        logger.error(f"Database error: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        # 无论如何都要关闭 Session
        db.close()
```

### 8.2 推荐的路由实现

```python
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """创建用户"""
    # 检查邮箱是否已存在
    existing_user = db.query(User).filter_by(email=user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    # 创建用户
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """获取用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### 8.3 推荐的项目结构

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用
│   ├── database.py          # 数据库配置
│   ├── models/              # 数据模型
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/             # Pydantic 模型
│   │   ├── __init__.py
│   │   └── user.py
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   └── users.py
│   └── dependencies.py      # 依赖注入
├── tests/
│   ├── __init__.py
│   └── test_users.py
└── .env
```

```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# app/dependencies.py
from sqlalchemy.orm import Session
from app.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# app/api/users.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.dependencies import get_db

router = APIRouter()

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

# app/main.py
from fastapi import FastAPI
from app.api import users

app = FastAPI()
app.include_router(users.router)
```

---

## 9. 常见问题

### 问题1：Session 没有关闭

**症状：** 连接池耗尽

**原因：** get_db 中没有 finally 块

**解决方案：**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # 确保关闭
```

### 问题2：事务没有提交

**症状：** 数据没有保存

**原因：** 忘记调用 commit

**解决方案：**
```python
@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()  # 必须 commit
    return db_user
```

### 问题3：异常时数据部分提交

**症状：** 数据不一致

**原因：** 异常时没有回滚

**解决方案：**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()  # 异常时回滚
        raise
    finally:
        db.close()
```

---

## 总结

**FastAPI 依赖注入集成的核心要点：**

1. **get_db 函数**：使用生成器函数管理 Session 生命周期
2. **异常处理**：在 get_db 中处理异常回滚
3. **事务管理**：手动提交模式（推荐）或自动提交模式
4. **嵌套依赖**：依赖可以依赖其他依赖
5. **依赖缓存**：同一个请求中依赖只执行一次
6. **中间件**：不推荐使用中间件管理 Session
7. **异步支持**：使用 AsyncSession 支持异步操作
8. **测试集成**：使用 dependency_overrides 替换依赖

**最佳实践：**
- 使用依赖注入管理 Session
- 在 get_db 中处理异常回滚
- 在路由中手动 commit
- 使用 HTTPException 返回错误
- 测试时覆盖依赖

---

**记住：** 依赖注入是管理 Session 生命周期的最佳方式，自动处理创建、使用、关闭。
