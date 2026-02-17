# 实战代码 04 - FastAPI 端点测试

## 项目结构

```
project/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── users.py
│   │   ├── posts.py
│   │   └── auth.py
│   ├── models/
│   │   └── schemas.py
│   └── dependencies.py
├── tests/
│   ├── conftest.py
│   ├── test_users_api.py
│   ├── test_posts_api.py
│   └── test_auth_api.py
└── pytest.ini
```

---

## FastAPI 应用

### 主应用

```python
# app/main.py
"""FastAPI 主应用"""
from fastapi import FastAPI
from app.api import users, posts, auth

app = FastAPI(title="Test API", version="1.0.0")

# 注册路由
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(posts.router, prefix="/api/posts", tags=["posts"])

@app.get("/")
def read_root():
    """根路径"""
    return {"message": "Welcome to Test API"}

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy"}
```

### 数据模型

```python
# app/models/schemas.py
"""Pydantic 模型"""
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    """创建用户请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    """用户响应"""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class PostCreate(BaseModel):
    """创建文章请求"""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)

class PostResponse(BaseModel):
    """文章响应"""
    id: int
    title: str
    content: str
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str
```

### 用户 API

```python
# app/api/users.py
"""用户 API"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List
from app.models.schemas import UserCreate, UserResponse
from app.models.user import User
from app.dependencies import get_db

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """创建用户"""
    # 检查用户名是否存在
    existing = db.query(User).filter_by(username=user_data.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # 创建用户
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=f"hashed_{user_data.password}"  # 简化示例
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.get("/", response_model=List[UserResponse])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """列出用户"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """获取用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """更新用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user.email = user_data.email
    db.commit()
    db.refresh(user)
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """删除用户"""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db.delete(user)
    db.commit()
```

---

## 测试配置

```python
# tests/conftest.py
"""测试配置"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base
from app.dependencies import get_db

# 测试数据库
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    """测试数据库引擎"""
    test_engine = create_engine(TEST_DATABASE_URL, echo=False)
    Base.metadata.create_all(test_engine)
    yield test_engine
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()

@pytest.fixture(scope="function")
def db_session(engine):
    """数据库会话"""
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    """测试客户端"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
```

---

## 用户 API 测试

```python
# tests/test_users_api.py
"""用户 API 测试"""
import pytest
from fastapi import status


class TestCreateUser:
    """创建用户测试"""

    def test_create_user_success(self, client):
        """测试：创建用户成功"""
        response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["username"] == "alice"
        assert data["email"] == "alice@example.com"
        assert "id" in data
        assert data["is_active"] is True

    def test_create_user_duplicate_username(self, client):
        """测试：重复用户名"""
        # 创建第一个用户
        client.post("/api/users/", json={
            "username": "alice",
            "email": "alice1@example.com",
            "password": "password123"
        })

        # 尝试创建相同用户名
        response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice2@example.com",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]

    def test_create_user_invalid_email(self, client):
        """测试：无效邮箱"""
        response = client.post("/api/users/", json={
            "username": "alice",
            "email": "invalid-email",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_user_short_username(self, client):
        """测试：用户名过短"""
        response = client.post("/api/users/", json={
            "username": "ab",
            "email": "test@example.com",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_user_short_password(self, client):
        """测试：密码过短"""
        response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "12345"
        })

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.parametrize("username,email,password,expected_status", [
        ("alice", "alice@example.com", "password123", 201),
        ("ab", "test@example.com", "password123", 422),  # 用户名过短
        ("alice", "invalid", "password123", 422),  # 无效邮箱
        ("alice", "alice@example.com", "12345", 422),  # 密码过短
    ])
    def test_create_user_validation(
        self, client, username, email, password, expected_status
    ):
        """测试：创建用户验证（参数化）"""
        response = client.post("/api/users/", json={
            "username": username,
            "email": email,
            "password": password
        })
        assert response.status_code == expected_status


class TestListUsers:
    """列出用户测试"""

    def test_list_users_empty(self, client):
        """测试：空列表"""
        response = client.get("/api/users/")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_list_users_multiple(self, client):
        """测试：多个用户"""
        # 创建用户
        for i in range(3):
            client.post("/api/users/", json={
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "password": "password123"
            })

        # 列出用户
        response = client.get("/api/users/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 3

    def test_list_users_pagination(self, client):
        """测试：分页"""
        # 创建10个用户
        for i in range(10):
            client.post("/api/users/", json={
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "password": "password123"
            })

        # 第一页
        response = client.get("/api/users/?skip=0&limit=5")
        assert len(response.json()) == 5

        # 第二页
        response = client.get("/api/users/?skip=5&limit=5")
        assert len(response.json()) == 5


class TestGetUser:
    """获取用户测试"""

    def test_get_user_success(self, client):
        """测试：获取用户成功"""
        # 创建用户
        create_response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })
        user_id = create_response.json()["id"]

        # 获取用户
        response = client.get(f"/api/users/{user_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == "alice"

    def test_get_user_not_found(self, client):
        """测试：用户不存在"""
        response = client.get("/api/users/99999")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


class TestUpdateUser:
    """更新用户测试"""

    def test_update_user_success(self, client):
        """测试：更新用户成功"""
        # 创建用户
        create_response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })
        user_id = create_response.json()["id"]

        # 更新用户
        response = client.put(f"/api/users/{user_id}", json={
            "username": "alice",
            "email": "newemail@example.com",
            "password": "newpassword123"
        })

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "newemail@example.com"

    def test_update_user_not_found(self, client):
        """测试：更新不存在的用户"""
        response = client.put("/api/users/99999", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteUser:
    """删除用户测试"""

    def test_delete_user_success(self, client):
        """测试：删除用户成功"""
        # 创建用户
        create_response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })
        user_id = create_response.json()["id"]

        # 删除用户
        response = client.delete(f"/api/users/{user_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # 验证已删除
        get_response = client.get(f"/api/users/{user_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_user_not_found(self, client):
        """测试：删除不存在的用户"""
        response = client.delete("/api/users/99999")

        assert response.status_code == status.HTTP_404_NOT_FOUND
```

---

## 文章 API 测试

```python
# app/api/posts.py
"""文章 API"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List
from app.models.schemas import PostCreate, PostResponse
from app.models.post import Post
from app.dependencies import get_db

router = APIRouter()

@router.post("/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post_data: PostCreate, user_id: int, db: Session = Depends(get_db)):
    """创建文章"""
    post = Post(
        title=post_data.title,
        content=post_data.content,
        user_id=user_id
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post

@router.get("/", response_model=List[PostResponse])
def list_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """列出文章"""
    posts = db.query(Post).offset(skip).limit(limit).all()
    return posts

@router.get("/{post_id}", response_model=PostResponse)
def get_post(post_id: int, db: Session = Depends(get_db)):
    """获取文章"""
    post = db.query(Post).filter_by(id=post_id).first()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    return post


# tests/test_posts_api.py
"""文章 API 测试"""
import pytest
from fastapi import status


@pytest.fixture
def sample_user(client):
    """创建示例用户"""
    response = client.post("/api/users/", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    })
    return response.json()


class TestCreatePost:
    """创建文章测试"""

    def test_create_post_success(self, client, sample_user):
        """测试：创建文章成功"""
        response = client.post(
            f"/api/posts/?user_id={sample_user['id']}",
            json={
                "title": "My First Post",
                "content": "This is the content."
            }
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "My First Post"
        assert data["content"] == "This is the content."
        assert data["user_id"] == sample_user["id"]

    def test_create_post_empty_title(self, client, sample_user):
        """测试：空标题"""
        response = client.post(
            f"/api/posts/?user_id={sample_user['id']}",
            json={
                "title": "",
                "content": "Content"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_post_empty_content(self, client, sample_user):
        """测试：空内容"""
        response = client.post(
            f"/api/posts/?user_id={sample_user['id']}",
            json={
                "title": "Title",
                "content": ""
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestListPosts:
    """列出文章测试"""

    def test_list_posts_empty(self, client):
        """测试：空列表"""
        response = client.get("/api/posts/")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_list_posts_multiple(self, client, sample_user):
        """测试：多篇文章"""
        # 创建文章
        for i in range(3):
            client.post(
                f"/api/posts/?user_id={sample_user['id']}",
                json={
                    "title": f"Post {i}",
                    "content": f"Content {i}"
                }
            )

        # 列出文章
        response = client.get("/api/posts/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 3


class TestGetPost:
    """获取文章测试"""

    def test_get_post_success(self, client, sample_user):
        """测试：获取文章成功"""
        # 创建文章
        create_response = client.post(
            f"/api/posts/?user_id={sample_user['id']}",
            json={
                "title": "Test Post",
                "content": "Test Content"
            }
        )
        post_id = create_response.json()["id"]

        # 获取文章
        response = client.get(f"/api/posts/{post_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == post_id
        assert data["title"] == "Test Post"

    def test_get_post_not_found(self, client):
        """测试：文章不存在"""
        response = client.get("/api/posts/99999")

        assert response.status_code == status.HTTP_404_NOT_FOUND
```

---

## 根路径和健康检查测试

```python
# tests/test_main.py
"""主应用测试"""
from fastapi import status


def test_read_root(client):
    """测试：根路径"""
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Welcome to Test API"}


def test_health_check(client):
    """测试：健康检查"""
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}


def test_openapi_schema(client):
    """测试：OpenAPI 文档"""
    response = client.get("/openapi.json")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "Test API"
```

---

## 查询参数测试

```python
# tests/test_query_params.py
"""查询参数测试"""
import pytest
from fastapi import status


class TestQueryParameters:
    """查询参数测试"""

    def test_pagination_default(self, client, sample_user):
        """测试：默认分页参数"""
        # 创建10个文章
        for i in range(10):
            client.post(
                f"/api/posts/?user_id={sample_user['id']}",
                json={"title": f"Post {i}", "content": f"Content {i}"}
            )

        # 不传分页参数
        response = client.get("/api/posts/")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 10

    def test_pagination_custom(self, client, sample_user):
        """测试：自定义分页参数"""
        # 创建10个文章
        for i in range(10):
            client.post(
                f"/api/posts/?user_id={sample_user['id']}",
                json={"title": f"Post {i}", "content": f"Content {i}"}
            )

        # 自定义分页
        response = client.get("/api/posts/?skip=2&limit=3")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 3

    def test_invalid_query_params(self, client):
        """测试：无效查询参数"""
        response = client.get("/api/posts/?skip=invalid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
```

---

## 请求头测试

```python
# tests/test_headers.py
"""请求头测试"""
from fastapi import status


def test_content_type_json(client):
    """测试：JSON Content-Type"""
    response = client.post("/api/users/", json={
        "username": "alice",
        "email": "alice@example.com",
        "password": "password123"
    })

    assert response.status_code == status.HTTP_201_CREATED
    assert response.headers["content-type"] == "application/json"


def test_custom_headers(client):
    """测试：自定义请求头"""
    response = client.get(
        "/api/users/",
        headers={"X-Custom-Header": "test-value"}
    )

    assert response.status_code == status.HTTP_200_OK


def test_accept_header(client):
    """测试：Accept 头"""
    response = client.get(
        "/api/users/",
        headers={"Accept": "application/json"}
    )

    assert response.status_code == status.HTTP_200_OK
    assert "application/json" in response.headers["content-type"]
```

---

## 错误处理测试

```python
# tests/test_error_handling.py
"""错误处理测试"""
from fastapi import status


class TestErrorHandling:
    """错误处理测试"""

    def test_404_not_found(self, client):
        """测试：404 错误"""
        response = client.get("/api/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_422_validation_error(self, client):
        """测试：422 验证错误"""
        response = client.post("/api/users/", json={
            "username": "ab",  # 太短
            "email": "invalid",  # 无效邮箱
            "password": "12345"  # 太短
        })

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_400_bad_request(self, client):
        """测试：400 错误"""
        # 创建用户
        client.post("/api/users/", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123"
        })

        # 尝试创建重复用户
        response = client.post("/api/users/", json={
            "username": "alice",
            "email": "alice2@example.com",
            "password": "password123"
        })

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_method_not_allowed(self, client):
        """测试：405 方法不允许"""
        response = client.patch("/api/users/")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
```

---

## 运行测试

```bash
# 运行所有 API 测试
pytest tests/test_*_api.py -v

# 运行特定测试类
pytest tests/test_users_api.py::TestCreateUser -v

# 运行特定测试函数
pytest tests/test_users_api.py::TestCreateUser::test_create_user_success -v

# 显示详细输出
pytest tests/ -v -s

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html

# 并行运行
pytest tests/ -n auto
```

---

## 总结

### 核心要点

1. **TestClient**：模拟 HTTP 请求，不需要启动服务器
2. **依赖注入覆盖**：使用 `app.dependency_overrides` 替换依赖
3. **完整测试**：测试所有 HTTP 方法（GET、POST、PUT、DELETE）
4. **验证测试**：测试请求验证和错误处理
5. **状态码检查**：验证正确的 HTTP 状态码

### 最佳实践

- 使用 fixture 准备测试数据
- 测试成功和失败场景
- 测试边界条件和验证
- 测试错误响应格式
- 使用参数化减少重复代码
