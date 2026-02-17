# 核心概念1：Pydantic 模型基础

> BaseModel、Field、类型系统 - Pydantic 的核心构建块

---

## 概述

Pydantic 模型是请求体验证的基础，它通过 Python 的类型注解系统提供了声明式的数据验证。本文详细讲解 Pydantic 的三个核心组件：BaseModel、Field 和类型系统。

---

## 1. BaseModel - 数据模型的基类

### 1.1 什么是 BaseModel？

**BaseModel 是 Pydantic 所有数据模型的基类，它提供了自动验证、序列化和反序列化的能力。**

```python
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int

# 创建实例时自动验证
user = User(username="alice", email="alice@example.com", age=25)
print(user.username)  # alice
print(user.age)       # 25
```

### 1.2 BaseModel 的核心功能

#### 功能1：自动类型验证

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    username: str
    age: int

# ✅ 正确的类型
user1 = User(username="alice", age=25)

# ✅ 自动类型转换
user2 = User(username="bob", age="30")  # "30" → 30
print(user2.age)  # 30 (int)

# ❌ 无法转换的类型 → 抛出异常
try:
    user3 = User(username="charlie", age="not-a-number")
except ValidationError as e:
    print(e)
    # ValidationError: value is not a valid integer
```

#### 功能2：必需字段检查

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    username: str  # 必需字段
    email: str     # 必需字段
    age: int       # 必需字段

# ❌ 缺少必需字段 → 抛出异常
try:
    user = User(username="alice")  # 缺少 email 和 age
except ValidationError as e:
    print(e.json())
    # [
    #   {"loc": ["email"], "msg": "field required", "type": "value_error.missing"},
    #   {"loc": ["age"], "msg": "field required", "type": "value_error.missing"}
    # ]
```

#### 功能3：数据序列化

```python
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int

user = User(username="alice", email="alice@example.com", age=25)

# 转换为字典
print(user.dict())
# {'username': 'alice', 'email': 'alice@example.com', 'age': 25}

# 转换为 JSON 字符串
print(user.json())
# {"username":"alice","email":"alice@example.com","age":25}

# 转换为 JSON（格式化）
print(user.json(indent=2))
# {
#   "username": "alice",
#   "email": "alice@example.com",
#   "age": 25
# }
```

#### 功能4：数据反序列化

```python
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int

# 从字典创建
user1 = User(**{"username": "alice", "email": "alice@example.com", "age": 25})

# 从 JSON 字符串创建
json_str = '{"username":"bob","email":"bob@example.com","age":30}'
user2 = User.parse_raw(json_str)

# 从 JSON 文件创建
# user3 = User.parse_file("user.json")
```

### 1.3 BaseModel 的配置选项

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        # 允许额外字段（默认 False）
        extra='allow',  # 'allow' | 'ignore' | 'forbid'

        # 验证赋值（默认 False）
        validate_assignment=True,

        # 使用枚举值而不是枚举对象
        use_enum_values=True,

        # 字段别名生成器
        alias_generator=lambda field_name: field_name.upper(),

        # 允许从 ORM 对象创建
        from_attributes=True
    )

    username: str
    email: str
    age: int

# extra='allow' - 允许额外字段
user = User(username="alice", email="alice@example.com", age=25, extra_field="value")
print(user.dict())
# {'username': 'alice', 'email': 'alice@example.com', 'age': 25, 'extra_field': 'value'}

# validate_assignment=True - 赋值时验证
user.age = "30"  # 自动转换为 int
user.age = "not-a-number"  # 抛出 ValidationError
```

### 1.4 在 FastAPI 中使用 BaseModel

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    age: int

@app.post("/users")
async def create_user(user: User):
    # FastAPI 自动：
    # 1. 解析请求体（JSON → dict）
    # 2. 验证数据（调用 User 的验证逻辑）
    # 3. 创建 User 实例
    # 4. 传递给路由函数
    return {"username": user.username, "age": user.age}
```

---

## 2. Field - 字段定义和约束

### 2.1 什么是 Field？

**Field 是 Pydantic 用来定义字段元数据和约束的函数，它提供了比类型注解更丰富的验证能力。**

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str = Field(..., description="用户邮箱")
    age: int = Field(..., ge=0, le=150, description="用户年龄")
```

### 2.2 Field 的参数详解

#### 参数1：default 和 default_factory

```python
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class User(BaseModel):
    # 使用 ... 表示必需字段
    username: str = Field(...)

    # 使用 default 设置默认值
    bio: str = Field(default="")
    is_active: bool = Field(default=True)

    # 使用 default_factory 设置动态默认值
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

# 创建实例
user = User(username="alice")
print(user.bio)         # ""
print(user.is_active)   # True
print(user.created_at)  # 当前时间
print(user.tags)        # []
```

**为什么需要 default_factory？**

```python
# ❌ 错误：可变默认值
class User(BaseModel):
    tags: List[str] = []  # 所有实例共享同一个列表！

user1 = User()
user1.tags.append("tag1")

user2 = User()
print(user2.tags)  # ['tag1'] - 意外！

# ✅ 正确：使用 default_factory
class User(BaseModel):
    tags: List[str] = Field(default_factory=list)

user1 = User()
user1.tags.append("tag1")

user2 = User()
print(user2.tags)  # [] - 正确！
```

#### 参数2：字符串约束

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    # 长度约束
    username: str = Field(min_length=3, max_length=20)

    # 正则表达式约束
    phone: str = Field(pattern=r'^\d{3}-\d{4}$')

    # 去除首尾空格
    bio: str = Field(strip_whitespace=True)

    # 转换为小写
    email: str = Field(to_lower=True)

# 示例
user = User(
    username="alice",
    phone="123-4567",
    bio="  Hello World  ",  # 自动去除空格
    email="ALICE@EXAMPLE.COM"  # 自动转为小写
)
print(user.bio)    # "Hello World"
print(user.email)  # "alice@example.com"
```

#### 参数3：数字约束

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    # 大于（greater than）
    price: float = Field(gt=0)

    # 大于等于（greater or equal）
    quantity: int = Field(ge=0)

    # 小于（less than）
    discount: float = Field(lt=1.0)

    # 小于等于（less or equal）
    rating: float = Field(le=5.0)

    # 组合约束
    stock: int = Field(ge=0, le=10000)

# 示例
product = Product(
    price=99.99,
    quantity=10,
    discount=0.2,
    rating=4.5,
    stock=100
)
```

#### 参数4：集合约束

```python
from pydantic import BaseModel, Field
from typing import List, Set

class Post(BaseModel):
    # 列表长度约束
    tags: List[str] = Field(min_items=1, max_items=10)

    # 集合长度约束
    categories: Set[str] = Field(min_items=1)

    # 唯一性约束（自动去重）
    unique_tags: Set[str] = Field(default_factory=set)

# 示例
post = Post(
    tags=["python", "fastapi"],
    categories={"tech", "programming"}
)
```

#### 参数5：元数据

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(
        ...,
        title="用户名",
        description="用户的唯一标识符，3-20个字符",
        example="alice123"
    )

    age: int = Field(
        ...,
        title="年龄",
        description="用户年龄，必须在0-150之间",
        example=25,
        ge=0,
        le=150
    )

# 这些元数据会出现在 FastAPI 的 OpenAPI 文档中
```

### 2.3 Field 的高级用法

#### 用法1：字段别名

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(..., alias="user_name")
    email: str = Field(..., alias="email_address")

# 使用别名创建实例
user = User(user_name="alice", email_address="alice@example.com")

# 访问时使用原始字段名
print(user.username)  # alice
print(user.email)     # alice@example.com

# 序列化时使用别名
print(user.dict(by_alias=True))
# {'user_name': 'alice', 'email_address': 'alice@example.com'}
```

#### 用法2：排除字段

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str
    email: str
    password: str = Field(..., exclude=True)  # 序列化时排除

user = User(username="alice", email="alice@example.com", password="secret")

# 序列化时自动排除 password
print(user.dict())
# {'username': 'alice', 'email': 'alice@example.com'}

# 可以手动包含
print(user.dict(exclude_unset=False))
# {'username': 'alice', 'email': 'alice@example.com'}
```

#### 用法3：常量字段

```python
from pydantic import BaseModel, Field
from typing import Literal

class User(BaseModel):
    username: str
    user_type: Literal["admin", "user"] = Field(default="user")

# 只能是 "admin" 或 "user"
user1 = User(username="alice", user_type="admin")  # ✅
user2 = User(username="bob", user_type="guest")    # ❌ ValidationError
```

---

## 3. 类型系统

### 3.1 基础类型

```python
from pydantic import BaseModel

class Example(BaseModel):
    # 字符串
    name: str

    # 整数
    age: int

    # 浮点数
    price: float

    # 布尔值
    is_active: bool

    # 字节
    data: bytes
```

### 3.2 容器类型

```python
from pydantic import BaseModel
from typing import List, Set, Dict, Tuple

class Example(BaseModel):
    # 列表
    tags: List[str]

    # 集合（自动去重）
    categories: Set[str]

    # 字典
    metadata: Dict[str, str]

    # 元组
    coordinates: Tuple[float, float]

    # 嵌套容器
    matrix: List[List[int]]
```

### 3.3 可选类型

```python
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    username: str
    email: str

    # 可选字段（可以是 int 或 None）
    age: Optional[int] = None

    # 可选字段（有默认值）
    bio: Optional[str] = ""

# 创建实例
user1 = User(username="alice", email="alice@example.com")
print(user1.age)  # None

user2 = User(username="bob", email="bob@example.com", age=30)
print(user2.age)  # 30
```

### 3.4 联合类型

```python
from pydantic import BaseModel
from typing import Union

class Example(BaseModel):
    # 可以是 int 或 str
    value: Union[int, str]

    # 可以是 int、float 或 str
    number: Union[int, float, str]

# 创建实例
ex1 = Example(value=123, number=456)
ex2 = Example(value="hello", number="789")
ex3 = Example(value=123, number=45.6)
```

### 3.5 特殊类型

```python
from pydantic import BaseModel, EmailStr, HttpUrl, UUID4
from datetime import datetime, date
from uuid import UUID

class User(BaseModel):
    # 邮箱（自动验证格式）
    email: EmailStr

    # URL（自动验证格式）
    website: HttpUrl

    # UUID
    id: UUID4

    # 日期时间
    created_at: datetime

    # 日期
    birth_date: date

# 创建实例
from uuid import uuid4

user = User(
    email="alice@example.com",
    website="https://example.com",
    id=uuid4(),
    created_at="2024-01-01T00:00:00",
    birth_date="1990-01-01"
)
```

### 3.6 嵌套模型

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    username: str
    email: str
    address: Address  # 嵌套模型
    addresses: List[Address]  # 嵌套模型列表

# 创建实例
user = User(
    username="alice",
    email="alice@example.com",
    address={
        "street": "123 Main St",
        "city": "NYC",
        "country": "USA"
    },
    addresses=[
        {"street": "123 Main St", "city": "NYC", "country": "USA"},
        {"street": "456 Oak Ave", "city": "LA", "country": "USA"}
    ]
)
```

### 3.7 递归模型

```python
from pydantic import BaseModel
from typing import List, Optional

class Category(BaseModel):
    name: str
    subcategories: Optional[List['Category']] = None

# 更新前向引用
Category.model_rebuild()

# 创建实例
category = Category(
    name="Electronics",
    subcategories=[
        Category(name="Computers", subcategories=[
            Category(name="Laptops"),
            Category(name="Desktops")
        ]),
        Category(name="Phones")
    ]
)
```

---

## 4. 在 AI Agent 开发中的应用

### 4.1 RAG 查询参数模型

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class RAGQuery(BaseModel):
    """RAG 查询参数"""

    # 必需字段
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题"
    )

    # 可选字段（有默认值）
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回结果数量"
    )

    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="相似度阈值"
    )

    # 可选字段（默认 None）
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="过滤条件"
    )

    rerank: bool = Field(
        default=False,
        description="是否重排序"
    )
```

### 4.2 Agent 配置模型

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class AgentConfig(BaseModel):
    """Agent 配置"""

    # 使用 Literal 限制可选值
    model: Literal["gpt-4", "gpt-3.5-turbo"] = Field(
        default="gpt-4",
        description="LLM 模型"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="生成温度"
    )

    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4000,
        description="最大 token 数"
    )

    stream: bool = Field(
        default=False,
        description="是否流式输出"
    )

    system_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="系统提示词"
    )
```

### 4.3 工具调用参数模型

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ToolCall(BaseModel):
    """工具调用参数"""

    tool_name: str = Field(
        ...,
        min_length=1,
        description="工具名称"
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数"
    )

class AgentResponse(BaseModel):
    """Agent 响应"""

    text: str = Field(
        ...,
        description="生成的文本"
    )

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="工具调用列表"
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="置信度"
    )
```

---

## 5. 最佳实践

### 5.1 使用类型注解

```python
# ✅ 好的做法
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    username: str
    tags: List[str]
    age: Optional[int] = None

# ❌ 不好的做法
class User(BaseModel):
    username = "default"  # 没有类型注解
    tags = []             # 没有类型注解
```

### 5.2 使用 Field 添加约束

```python
# ✅ 好的做法
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(ge=0, le=150)

# ❌ 不好的做法
class User(BaseModel):
    username: str  # 没有约束，可能太短或太长
    age: int       # 没有约束，可能是负数
```

### 5.3 使用 default_factory 处理可变默认值

```python
# ✅ 好的做法
from pydantic import BaseModel, Field
from typing import List

class User(BaseModel):
    tags: List[str] = Field(default_factory=list)

# ❌ 不好的做法
class User(BaseModel):
    tags: List[str] = []  # 所有实例共享同一个列表
```

### 5.4 使用描述和示例

```python
# ✅ 好的做法
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=20,
        description="用户名，3-20个字符",
        example="alice123"
    )

# 这些信息会出现在 FastAPI 的 OpenAPI 文档中
```

---

## 6. 核心要点总结

1. **BaseModel** - 所有 Pydantic 模型的基类，提供自动验证和序列化
2. **Field** - 定义字段约束和元数据，比类型注解更强大
3. **类型系统** - 支持基础类型、容器类型、可选类型、联合类型、特殊类型
4. **嵌套模型** - 支持递归验证复杂数据结构
5. **配置选项** - 通过 model_config 自定义模型行为

---

## 7. 学习检查清单

完成本节后，你应该能够：

- [ ] 理解 BaseModel 的核心功能
- [ ] 使用 Field 定义字段约束
- [ ] 掌握 Pydantic 的类型系统
- [ ] 定义嵌套模型和递归模型
- [ ] 配置模型行为（extra、validate_assignment 等）
- [ ] 在 FastAPI 中使用 Pydantic 模型

---

**版本：** v1.0
**最后更新：** 2026-02-11
