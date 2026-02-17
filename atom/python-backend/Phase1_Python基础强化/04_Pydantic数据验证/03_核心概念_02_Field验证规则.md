# 核心概念2：Field 验证规则

Field 是 Pydantic 提供的强大验证工具，让你能够声明式地定义字段约束。

---

## 什么是 Field？

**Field 是 Pydantic 的字段定义函数，用于添加验证规则、默认值、文档等元数据。**

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., description="用户邮箱")

# Field 自动验证
user = User(name="Alice", age=25, email="alice@example.com")
```

**核心功能：**
1. **验证约束**：长度、范围、格式等
2. **默认值**：静态默认值或工厂函数
3. **文档说明**：description、example
4. **别名**：字段别名映射

---

## 1. 基础用法

### 1.1 必填字段

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    # 三种等价写法（必填）
    name1: str                    # 没有默认值 = 必填
    name2: str = Field(...)       # 显式标记为必填
    name3: str = Field()          # Pydantic 2.0+ 也是必填

# 必填字段必须提供
user = User(name1="Alice", name2="Bob", name3="Charlie")
```

**`...` 是什么？**

`...` 是 Python 的 Ellipsis 字面量，在 Pydantic 中表示必填字段。

### 1.2 可选字段

```python
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    name: str
    # 可选字段（有默认值）
    email: Optional[str] = Field(None)
    phone: Optional[str] = Field(default=None)
    age: int = Field(default=0)

# 可选字段可以不传
user = User(name="Alice")
print(user.email)  # None
print(user.age)    # 0
```

### 1.3 默认工厂函数

```python
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class User(BaseModel):
    name: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

# 每个实例都有独立的默认值
user1 = User(name="Alice")
user2 = User(name="Bob")

user1.tags.append("admin")
print(user1.tags)  # ['admin']
print(user2.tags)  # []（独立的列表）
```

---

## 2. 数值验证

### 2.1 整数验证

```python
from pydantic import BaseModel, Field, ValidationError

class Product(BaseModel):
    # gt: greater than (>)
    price: int = Field(..., gt=0)

    # ge: greater than or equal (>=)
    stock: int = Field(..., ge=0)

    # lt: less than (<)
    discount: int = Field(..., lt=100)

    # le: less than or equal (<=)
    rating: int = Field(..., le=5)

    # 组合使用
    age: int = Field(..., ge=0, le=150)

# 验证通过
product = Product(price=100, stock=50, discount=20, rating=5, age=25)

# 验证失败
try:
    product = Product(price=0, stock=-1, discount=100, rating=6, age=200)
except ValidationError as e:
    print(e)
    # price: Input should be greater than 0
    # stock: Input should be greater than or equal to 0
    # discount: Input should be less than 100
    # rating: Input should be less than or equal to 5
    # age: Input should be less than or equal to 150
```

### 2.2 浮点数验证

```python
from pydantic import BaseModel, Field

class Measurement(BaseModel):
    temperature: float = Field(..., ge=-273.15, le=1000)  # 绝对零度到1000度
    humidity: float = Field(..., ge=0, le=100)  # 0-100%
    pressure: float = Field(..., gt=0)  # 必须大于0

# 验证通过
measurement = Measurement(temperature=25.5, humidity=60.0, pressure=101.3)
```

### 2.3 multiple_of（倍数验证）

```python
from pydantic import BaseModel, Field

class Order(BaseModel):
    # 必须是5的倍数
    quantity: int = Field(..., multiple_of=5)

    # 必须是0.5的倍数
    price: float = Field(..., multiple_of=0.5)

# 验证通过
order = Order(quantity=10, price=9.5)

# 验证失败
try:
    order = Order(quantity=7, price=9.3)
except ValidationError as e:
    print(e)
    # quantity: Input should be a multiple of 5
    # price: Input should be a multiple of 0.5
```

---

## 3. 字符串验证

### 3.1 长度验证

```python
from pydantic import BaseModel, Field, ValidationError

class User(BaseModel):
    # 最小长度
    username: str = Field(..., min_length=3)

    # 最大长度
    bio: str = Field(..., max_length=500)

    # 长度范围
    password: str = Field(..., min_length=8, max_length=128)

    # 精确长度（min_length = max_length）
    code: str = Field(..., min_length=6, max_length=6)

# 验证通过
user = User(
    username="alice",
    bio="Hello",
    password="password123",
    code="ABC123"
)

# 验证失败
try:
    user = User(
        username="ab",  # 太短
        bio="x" * 501,  # 太长
        password="123",  # 太短
        code="12345"  # 长度不对
    )
except ValidationError as e:
    print(e)
```

### 3.2 正则表达式验证

```python
from pydantic import BaseModel, Field, ValidationError

class User(BaseModel):
    # 邮箱格式
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    # 手机号格式（中国）
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$")

    # 用户名格式（字母数字下划线）
    username: str = Field(..., pattern=r"^[a-zA-Z0-9_]+$")

# 验证通过
user = User(
    email="alice@example.com",
    phone="13800138000",
    username="alice_123"
)

# 验证失败
try:
    user = User(
        email="invalid-email",
        phone="12345",
        username="alice@123"  # 包含非法字符
    )
except ValidationError as e:
    print(e)
```

### 3.3 字符串转换

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    # 自动去除首尾空格
    name: str = Field(..., strip_whitespace=True)

    # 转换为小写
    email: str = Field(..., to_lower=True)

    # 转换为大写
    code: str = Field(..., to_upper=True)

# 自动转换
user = User(
    name="  Alice  ",
    email="ALICE@EXAMPLE.COM",
    code="abc123"
)

print(user.name)   # "Alice"（去除空格）
print(user.email)  # "alice@example.com"（转小写）
print(user.code)   # "ABC123"（转大写）
```

---

## 4. 容器验证

### 4.1 列表验证

```python
from pydantic import BaseModel, Field, ValidationError
from typing import List

class Team(BaseModel):
    # 最小元素数量
    members: List[str] = Field(..., min_length=1)

    # 最大元素数量
    tags: List[str] = Field(..., max_length=10)

    # 元素数量范围
    scores: List[int] = Field(..., min_length=3, max_length=5)

# 验证通过
team = Team(
    members=["Alice", "Bob"],
    tags=["python", "fastapi"],
    scores=[85, 90, 95]
)

# 验证失败
try:
    team = Team(
        members=[],  # 太少
        tags=["tag"] * 11,  # 太多
        scores=[85, 90]  # 太少
    )
except ValidationError as e:
    print(e)
```

### 4.2 字典验证

```python
from pydantic import BaseModel, Field
from typing import Dict

class Config(BaseModel):
    # 字典最小键数量
    settings: Dict[str, str] = Field(..., min_length=1)

    # 字典最大键数量
    metadata: Dict[str, int] = Field(..., max_length=10)

# 验证通过
config = Config(
    settings={"key1": "value1"},
    metadata={"count": 10}
)
```

### 4.3 集合验证

```python
from pydantic import BaseModel, Field
from typing import Set

class User(BaseModel):
    # 集合元素数量
    permissions: Set[str] = Field(..., min_length=1, max_length=20)

# 验证通过
user = User(permissions={"read", "write", "delete"})
```

---

## 5. 文档和元数据

### 5.1 description（字段描述）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="用户姓名")
    age: int = Field(..., description="用户年龄，范围0-150", ge=0, le=150)
    email: str = Field(..., description="用户邮箱地址")

# FastAPI 会在 API 文档中显示这些描述
```

### 5.2 example（示例值）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., example="Alice")
    age: int = Field(..., example=25)
    email: str = Field(..., example="alice@example.com")

# FastAPI 会在 API 文档中显示这些示例
```

### 5.3 title（字段标题）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., title="用户名")
    age: int = Field(..., title="年龄")

# 用于生成更友好的文档标题
```

### 5.4 组合使用

```python
from pydantic import BaseModel, Field

class CreateUserRequest(BaseModel):
    name: str = Field(
        ...,
        title="用户名",
        description="用户的显示名称",
        example="Alice",
        min_length=1,
        max_length=50
    )

    age: int = Field(
        ...,
        title="年龄",
        description="用户年龄，必须在0-150之间",
        example=25,
        ge=0,
        le=150
    )

    email: str = Field(
        ...,
        title="邮箱",
        description="用户的邮箱地址",
        example="alice@example.com",
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"
    )
```

---

## 6. 字段别名

### 6.1 alias（别名）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    # Python 中用 user_name，JSON 中用 userName
    user_name: str = Field(..., alias="userName")
    user_age: int = Field(..., alias="userAge")

# 使用别名创建
user = User(userName="Alice", userAge=25)

# 访问时用 Python 字段名
print(user.user_name)  # Alice
print(user.user_age)   # 25

# 序列化时用别名
print(user.model_dump(by_alias=True))
# {'userName': 'Alice', 'userAge': 25}
```

### 6.2 populate_by_name（同时支持别名和字段名）

```python
from pydantic import BaseModel, Field, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_name: str = Field(..., alias="userName")

# 可以用别名创建
user1 = User(userName="Alice")

# 也可以用字段名创建
user2 = User(user_name="Bob")

# 两种方式都有效
print(user1.user_name)  # Alice
print(user2.user_name)  # Bob
```

---

## 7. 在 FastAPI 中使用

### 7.1 请求体验证

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, example="Alice")
    age: int = Field(..., ge=0, le=150, example=25)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", example="alice@example.com")

@app.post("/users")
async def create_user(request: CreateUserRequest):
    # FastAPI 自动验证：
    # - name 长度 1-50
    # - age 范围 0-150
    # - email 格式正确
    return {"message": f"User {request.name} created"}
```

### 7.2 查询参数验证

```python
from fastapi import FastAPI, Query
from pydantic import Field

app = FastAPI()

@app.get("/users")
async def list_users(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量")
):
    # FastAPI 自动验证查询参数
    return {"page": page, "page_size": page_size}
```

### 7.3 路径参数验证

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., ge=1, description="用户ID")
):
    # FastAPI 自动验证路径参数
    return {"user_id": user_id}
```

---

## 8. 在 AI Agent 中的应用

### 8.1 LLM 参数验证

```python
from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    model: Literal["gpt-4", "claude-3", "gpt-3.5-turbo"] = Field(
        "gpt-3.5-turbo",
        description="LLM 模型名称"
    )

    temperature: float = Field(
        0.7,
        ge=0,
        le=2,
        description="温度参数，控制输出随机性"
    )

    max_tokens: int = Field(
        1000,
        ge=1,
        le=4096,
        description="最大生成 token 数"
    )

    top_p: float = Field(
        1.0,
        ge=0,
        le=1,
        description="核采样参数"
    )

# 自动验证所有参数范围
request = ChatRequest(
    model="gpt-4",
    temperature=0.8,
    max_tokens=2000
)
```

### 8.2 RAG 检索参数验证

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="检索查询文本"
    )

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="返回结果数量"
    )

    score_threshold: float = Field(
        0.7,
        ge=0,
        le=1,
        description="相似度阈值"
    )

    filters: Optional[List[str]] = Field(
        None,
        max_length=10,
        description="过滤条件"
    )

# 自动验证所有参数
request = RAGQueryRequest(
    query="什么是 Pydantic？",
    top_k=10,
    score_threshold=0.8
)
```

### 8.3 Agent 工具参数验证

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchTool(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    engine: Literal["google", "bing", "duckduckgo"] = Field("google")
    max_results: int = Field(10, ge=1, le=50)

class CalculatorTool(BaseModel):
    expression: str = Field(..., pattern=r"^[\d\+\-\*/\(\)\s]+$")

class DatabaseTool(BaseModel):
    sql: str = Field(..., max_length=1000)
    timeout: int = Field(30, ge=1, le=300)

# Agent 调用工具时自动验证参数
search = SearchTool(query="Python tutorial", max_results=20)
```

---

## 9. 高级用法

### 9.1 json_schema_extra（自定义 JSON Schema）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(
        ...,
        json_schema_extra={
            "format": "name",
            "x-custom-field": "value"
        }
    )

# 自定义 JSON Schema 元数据
```

### 9.2 repr（控制字符串表示）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    password: str = Field(..., repr=False)  # 不在 repr 中显示

user = User(name="Alice", password="secret")
print(user)
# User(name='Alice')（password 不显示）
```

### 9.3 frozen（字段不可变）

```python
from pydantic import BaseModel, Field, ValidationError

class User(BaseModel):
    name: str
    id: int = Field(..., frozen=True)  # 不可修改

user = User(name="Alice", id=1)

# 可以修改 name
user.name = "Bob"

# 不能修改 id
try:
    user.id = 2
except ValidationError as e:
    print("id 字段不可修改")
```

### 9.4 exclude（排除字段）

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    age: int
    password: str = Field(..., exclude=True)  # 序列化时排除

user = User(name="Alice", age=25, password="secret")

# 序列化时自动排除 password
print(user.model_dump())
# {'name': 'Alice', 'age': 25}
```

---

## 10. 常见验证规则速查

### 数值验证

| 参数 | 说明 | 示例 |
|------|------|------|
| `gt` | 大于 (>) | `Field(..., gt=0)` |
| `ge` | 大于等于 (>=) | `Field(..., ge=0)` |
| `lt` | 小于 (<) | `Field(..., lt=100)` |
| `le` | 小于等于 (<=) | `Field(..., le=100)` |
| `multiple_of` | 倍数 | `Field(..., multiple_of=5)` |

### 字符串验证

| 参数 | 说明 | 示例 |
|------|------|------|
| `min_length` | 最小长度 | `Field(..., min_length=1)` |
| `max_length` | 最大长度 | `Field(..., max_length=50)` |
| `pattern` | 正则表达式 | `Field(..., pattern=r"^\d+$")` |
| `strip_whitespace` | 去除空格 | `Field(..., strip_whitespace=True)` |
| `to_lower` | 转小写 | `Field(..., to_lower=True)` |
| `to_upper` | 转大写 | `Field(..., to_upper=True)` |

### 容器验证

| 参数 | 说明 | 示例 |
|------|------|------|
| `min_length` | 最小元素数 | `Field(..., min_length=1)` |
| `max_length` | 最大元素数 | `Field(..., max_length=10)` |

### 元数据

| 参数 | 说明 | 示例 |
|------|------|------|
| `description` | 字段描述 | `Field(..., description="用户名")` |
| `example` | 示例值 | `Field(..., example="Alice")` |
| `title` | 字段标题 | `Field(..., title="用户名")` |
| `alias` | 字段别名 | `Field(..., alias="userName")` |

### 其他

| 参数 | 说明 | 示例 |
|------|------|------|
| `default` | 默认值 | `Field(default=0)` |
| `default_factory` | 默认工厂 | `Field(default_factory=list)` |
| `repr` | 是否在 repr 中显示 | `Field(..., repr=False)` |
| `frozen` | 字段不可变 | `Field(..., frozen=True)` |
| `exclude` | 序列化时排除 | `Field(..., exclude=True)` |

---

## 总结

### Field 的核心价值

1. **声明式验证**：在字段定义处声明验证规则
2. **丰富的约束**：数值、字符串、容器等多种验证
3. **自动文档**：description、example 自动生成文档
4. **灵活配置**：别名、排除、不可变等高级功能

### 最佳实践

1. **合理的约束**：不要过度验证，只验证必要的约束
2. **清晰的文档**：为公共 API 添加 description 和 example
3. **一致的命名**：使用 alias 统一前后端字段命名
4. **安全的默认值**：使用 default_factory 避免可变默认值陷阱

### 下一步

- 学习自定义验证器（更复杂的验证逻辑）
- 学习模型继承和复用
- 学习与数据库 ORM 集成

---

**版本：** v1.0
**最后更新：** 2026-02-11
