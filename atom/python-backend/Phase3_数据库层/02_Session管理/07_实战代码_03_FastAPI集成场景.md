# 【实战代码3】FastAPI集成场景

> 依赖注入、中间件、异步Session的完整示例

---

## 场景1：基础依赖注入

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, email={self.email})>"


Base.metadata.create_all(bind=engine)


def get_db():
    """数据库 Session 依赖注入"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# schemas.py
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """创建用户的请求模型"""
    name: str
    email: EmailStr


class UserResponse(BaseModel):
    """用户响应模型"""
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True


# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, User
from schemas import UserCreate, UserResponse

app = FastAPI(title="User API")


@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """创建用户"""
    # 检查邮箱是否已存在
    existing_user = db.query(User).filter_by(email=user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    # 创建用户
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@app.get("/users", response_model=list[UserResponse])
def get_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """获取用户列表"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """获取单个用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserCreate, db: Session = Depends(get_db)):
    """更新用户"""
    db_user = db.query(User).filter_by(id=user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.name = user.name
    db_user.email = user.email
    db.commit()
    db.refresh(db_user)

    return db_user


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """删除用户"""
    db_user = db.query(User).filter_by(id=user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(db_user)
    db.commit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 测试

```bash
# 启动服务
python main.py

# 创建用户
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# 获取用户列表
curl "http://localhost:8000/users"

# 获取单个用户
curl "http://localhost:8000/users/1"

# 更新用户
curl -X PUT "http://localhost:8000/users/1" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Updated", "email": "alice@example.com"}'

# 删除用户
curl -X DELETE "http://localhost:8000/users/1"
```

---

## 场景2：嵌套依赖注入

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    token = Column(String, unique=True, index=True)


Base.metadata.create_all(bind=engine)


def get_db():
    """数据库 Session 依赖"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# dependencies.py
from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from database import get_db, User


def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    """获取当前用户（依赖 get_db）"""
    # 解析 token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")

    # 查询用户
    user = db.query(User).filter_by(token=token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user


def get_admin_user(current_user: User = Depends(get_current_user)):
    """获取管理员用户（依赖 get_current_user）"""
    if current_user.name != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return current_user


# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db, User
from dependencies import get_current_user, get_admin_user

app = FastAPI()


@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
    }


@app.get("/admin/users")
def get_all_users(
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """管理员获取所有用户"""
    users = db.query(User).all()
    return users


@app.post("/admin/users/{user_id}/delete")
def delete_user_by_admin(
    user_id: int,
    admin_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """管理员删除用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": f"User {user_id} deleted by {admin_user.name}"}


if __name__ == "__main__":
    import uvicorn

    # 创建测试用户
    with SessionLocal() as session:
        admin = User(name="admin", email="admin@example.com", token="admin-token")
        user = User(name="alice", email="alice@example.com", token="user-token")
        session.add_all([admin, user])
        session.commit()

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 测试

```bash
# 获取当前用户信息
curl "http://localhost:8000/me" \
  -H "Authorization: Bearer user-token"

# 管理员获取所有用户
curl "http://localhost:8000/admin/users" \
  -H "Authorization: Bearer admin-token"

# 普通用户访问管理员接口（失败）
curl "http://localhost:8000/admin/users" \
  -H "Authorization: Bearer user-token"

# 管理员删除用户
curl -X POST "http://localhost:8000/admin/users/2/delete" \
  -H "Authorization: Bearer admin-token"
```

---

## 场景3：异步Session

### 完整代码

```python
# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/mydb"

# 创建异步引擎
engine = create_async_engine(DATABASE_URL, echo=True)

# 创建异步 SessionMaker
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)


async def init_db():
    """初始化数据库"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """异步数据库 Session 依赖"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# schemas.py
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    email: EmailStr


class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True


# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, User, init_db
from schemas import UserCreate, UserResponse

app = FastAPI()


@app.on_event("startup")
async def startup():
    """启动时初始化数据库"""
    await init_db()


@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """创建用户（异步）"""
    # 检查邮箱是否已存在
    result = await db.execute(select(User).filter_by(email=user.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    # 创建用户
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user


@app.get("/users", response_model=list[UserResponse])
async def get_users(skip: int = 0, limit: int = 10, db: AsyncSession = Depends(get_db)):
    """获取用户列表（异步）"""
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """获取单个用户（异步）"""
    result = await db.execute(select(User).filter_by(id=user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user: UserCreate, db: AsyncSession = Depends(get_db)):
    """更新用户（异步）"""
    result = await db.execute(select(User).filter_by(id=user_id))
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.name = user.name
    db_user.email = user.email
    await db.commit()
    await db.refresh(db_user)

    return db_user


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """删除用户（异步）"""
    result = await db.execute(select(User).filter_by(id=user_id))
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(db_user)
    await db.commit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 安装依赖

```bash
# 安装异步驱动
uv add asyncpg
```

---

## 场景4：完整的CRUD服务

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    """数据库 Session 依赖"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# schemas.py
from pydantic import BaseModel, EmailStr
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模型"""
    name: str
    email: EmailStr


class UserCreate(UserBase):
    """创建用户"""
    pass


class UserUpdate(BaseModel):
    """更新用户"""
    name: str | None = None
    email: EmailStr | None = None


class UserResponse(UserBase):
    """用户响应"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    """分页响应"""
    total: int
    page: int
    page_size: int
    items: list[UserResponse]


# services/user_service.py
from sqlalchemy.orm import Session
from database import User
from schemas import UserCreate, UserUpdate


class UserService:
    """用户服务"""

    @staticmethod
    def create_user(db: Session, user: UserCreate) -> User:
        """创建用户"""
        db_user = User(name=user.name, email=user.email)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def get_user(db: Session, user_id: int) -> User | None:
        """获取用户"""
        return db.query(User).filter_by(id=user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> User | None:
        """根据邮箱获取用户"""
        return db.query(User).filter_by(email=email).first()

    @staticmethod
    def get_users(db: Session, skip: int = 0, limit: int = 10) -> list[User]:
        """获取用户列表"""
        return db.query(User).offset(skip).limit(limit).all()

    @staticmethod
    def count_users(db: Session) -> int:
        """统计用户数量"""
        return db.query(User).count()

    @staticmethod
    def update_user(db: Session, user_id: int, user: UserUpdate) -> User | None:
        """更新用户"""
        db_user = db.query(User).filter_by(id=user_id).first()
        if not db_user:
            return None

        if user.name is not None:
            db_user.name = user.name
        if user.email is not None:
            db_user.email = user.email

        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        """删除用户"""
        db_user = db.query(User).filter_by(id=user_id).first()
        if not db_user:
            return False

        db.delete(db_user)
        db.commit()
        return True


# api/users.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from schemas import UserCreate, UserUpdate, UserResponse, PaginatedResponse
from services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """创建用户"""
    # 检查邮箱是否已存在
    existing_user = UserService.get_user_by_email(db, user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    return UserService.create_user(db, user)


@router.get("", response_model=PaginatedResponse)
def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """获取用户列表（分页）"""
    skip = (page - 1) * page_size
    users = UserService.get_users(db, skip=skip, limit=page_size)
    total = UserService.count_users(db)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": users,
    }


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """获取单个用户"""
    user = UserService.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    """更新用户"""
    db_user = UserService.update_user(db, user_id, user)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """删除用户"""
    success = UserService.delete_user(db, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")


# main.py
from fastapi import FastAPI
from api.users import router as users_router

app = FastAPI(
    title="User Management API",
    description="Complete CRUD API with FastAPI and SQLAlchemy",
    version="1.0.0",
)

app.include_router(users_router)


@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 项目结构

```
project/
├── database.py          # 数据库配置
├── schemas.py           # Pydantic 模型
├── services/
│   └── user_service.py  # 业务逻辑
├── api/
│   └── users.py         # API 路由
└── main.py              # 应用入口
```

### 测试

```bash
# 创建用户
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'

# 获取用户列表（分页）
curl "http://localhost:8000/users?page=1&page_size=10"

# 获取单个用户
curl "http://localhost:8000/users/1"

# 更新用户
curl -X PUT "http://localhost:8000/users/1" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Updated"}'

# 删除用户
curl -X DELETE "http://localhost:8000/users/1"

# 健康检查
curl "http://localhost:8000/health"
```

---

## 场景5：错误处理和日志

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)


Base.metadata.create_all(bind=engine)


def get_db():
    """数据库 Session 依赖（带日志）"""
    db = SessionLocal()
    logger.info("Database session created")

    try:
        yield db
        logger.info("Database session completed successfully")
    except Exception as e:
        logger.error(f"Database error: {e}", exc_info=True)
        db.rollback()
        logger.info("Database session rolled back")
        raise
    finally:
        db.close()
        logger.info("Database session closed")


# main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, OperationalError
from database import get_db, User
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


# 全局异常处理器
@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    """处理数据库完整性错误"""
    logger.error(f"Integrity error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": "Database integrity error. Duplicate or invalid data."}
    )


@app.exception_handler(OperationalError)
async def operational_error_handler(request: Request, exc: OperationalError):
    """处理数据库操作错误"""
    logger.error(f"Operational error: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Database connection error. Please try again later."}
    )


@app.post("/users")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    """创建用户（带错误处理）"""
    logger.info(f"Creating user: name={name}, email={email}")

    try:
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"User created successfully: id={user.id}")
        return user

    except IntegrityError as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=400, detail="Email already exists")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    """获取用户（带日志）"""
    logger.info(f"Fetching user: id={user_id}")

    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        logger.warning(f"User not found: id={user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"User found: {user}")
    return user


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 运行结果

```
2026-02-11 12:00:00 - database - INFO - Database session created
2026-02-11 12:00:00 - __main__ - INFO - Creating user: name=Alice, email=alice@example.com
2026-02-11 12:00:00 - __main__ - INFO - User created successfully: id=1
2026-02-11 12:00:00 - database - INFO - Database session completed successfully
2026-02-11 12:00:00 - database - INFO - Database session closed

2026-02-11 12:00:01 - database - INFO - Database session created
2026-02-11 12:00:01 - __main__ - INFO - Fetching user: id=1
2026-02-11 12:00:01 - __main__ - INFO - User found: <User(id=1, name=Alice, email=alice@example.com)>
2026-02-11 12:00:01 - database - INFO - Database session completed successfully
2026-02-11 12:00:01 - database - INFO - Database session closed
```

---

## 总结

**FastAPI 集成的核心要点：**

1. **依赖注入**：使用 Depends(get_db) 管理 Session
2. **嵌套依赖**：依赖可以依赖其他依赖
3. **异步支持**：使用 AsyncSession 支持异步操作
4. **服务层**：将业务逻辑封装到服务层
5. **错误处理**：使用全局异常处理器和日志

**最佳实践：**
- 使用依赖注入管理 Session
- 在 get_db 中处理异常回滚
- 使用服务层封装业务逻辑
- 添加日志记录
- 使用全局异常处理器

---

**记住：** FastAPI 的依赖注入是管理 Session 的最佳方式，自动处理创建、使用、关闭。
