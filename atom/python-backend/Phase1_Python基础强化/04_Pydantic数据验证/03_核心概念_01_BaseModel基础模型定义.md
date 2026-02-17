# 核心概念1：BaseModel 基础模型定义

Pydantic 的核心是 BaseModel，理解它是掌握 Pydantic 的关键。

---

## 什么是 BaseModel？

**BaseModel 是 Pydantic 的基础类，所有数据模型都继承自它。**

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# 创建实例时自动验证
user = User(name="Alice", age=25)
```

**核心功能：**
1. **自动验证**：根据类型注解验证数据
2. **自动转换**：尝试将数据转换为目标类型
3. **数据访问**：通过属性访问数据
4. **序列化**：转换为字典或 JSON

---

## 1. 定义模型

### 1.1 基础定义

```python
from pydantic import BaseModel

class User(BaseModel):
    # 必填字段
    name: str
    age: int
    email: str

# 创建实例
user = User(name="Alice", age=25, email="alice@example.com")

# 访问字段
print(user.name)   # Alice
print(user.age)    # 25
print(user.email)  # alice@example.com
```

### 1.2 可选字段

```python
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None  # 可选字段
    phone: Optional[str] = None

# 可选字段可以不传
user1 = User(name="Alice", age=25)
print(user1.email)  # None

# 也可以传值
user2 = User(name="Bob", age=30, email="bob@example.com")
print(user2.email)  # bob@example.com
```

### 1.3 默认值

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    is_active: bool = True  # 默认值
    role: str = "user"      # 默认值

# 使用默认值
user1 = User(name="Alice", age=25)
print(user1.is_active)  # True
print(user1.role)       # user

# 覆盖默认值
user2 = User(name="Bob", age=30, is_active=False, role="admin")
print(user2.is_active)  # False
print(user2.role)       # admin
```

### 1.4 默认工厂函数

```python
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class User(BaseModel):
    name: str
    tags: List[str] = Field(default_factory=list)  # 默认空列表
    created_at: datetime = Field(default_factory=datetime.now)  # 默认当前时间

# 每个实例都有独立的默认值
user1 = User(name="Alice")
user2 = User(name="Bob")

user1.tags.append("admin")
print(user1.tags)  # ['admin']
print(user2.tags)  # []（独立的列表）

print(user1.created_at)  # 创建时间
print(user2.created_at)  # 不同的创建时间
```

**为什么需要 default_factory？**

```python
# ❌ 错误：所有实例共享同一个列表
class User(BaseModel):
    name: str
    tags: List[str] = []  # 危险！

user1 = User(name="Alice")
user2 = User(name="Bob")
user1.tags.append("admin")
print(user2.tags)  # ['admin']（被污染了！）

# ✅ 正确：每个实例有独立的列表
class User(BaseModel):
    name: str
    tags: List[str] = Field(default_factory=list)

user1 = User(name="Alice")
user2 = User(name="Bob")
user1.tags.append("admin")
print(user2.tags)  # []（独立的列表）
```

---

## 2. 数据验证

### 2.1 自动类型验证

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

# 类型正确：通过
user = User(name="Alice", age=25)

# 类型错误：抛出异常
try:
    user = User(name="Alice", age="invalid")
except ValidationError as e:
    print(e)
    # ValidationError: 1 validation error for User
    # age
    #   Input should be a valid integer
```

### 2.2 自动类型转换

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    is_active: bool

# 自动转换
user = User(
    name="Alice",
    age="25",        # str → int
    is_active="true" # str → bool
)

print(user.age)        # 25 (int)
print(user.is_active)  # True (bool)
print(type(user.age))  # <class 'int'>
```

### 2.3 必填字段验证

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

# 缺少必填字段：抛出异常
try:
    user = User(name="Alice")  # 缺少 age
except ValidationError as e:
    print(e)
    # ValidationError: 1 validation error for User
    # age
    #   Field required
```

---

## 3. 数据访问

### 3.1 属性访问

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=25)

# 读取属性
print(user.name)  # Alice
print(user.age)   # 25

# 修改属性
user.age = 26
print(user.age)  # 26

# 修改时会重新验证
user.age = "30"  # 自动转换为 30
print(user.age)  # 30 (int)
```

### 3.2 字典访问

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=25)

# 转换为字典
user_dict = user.model_dump()
print(user_dict)  # {'name': 'Alice', 'age': 25}

# 从字典创建
data = {"name": "Bob", "age": 30}
user = User(**data)
print(user.name)  # Bob
```

### 3.3 JSON 序列化

```python
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    name: str
    age: int
    created_at: datetime

user = User(name="Alice", age=25, created_at=datetime.now())

# 转换为 JSON 字符串
json_str = user.model_dump_json()
print(json_str)
# {"name":"Alice","age":25,"created_at":"2026-02-11T06:20:29.095Z"}

# 从 JSON 字符串创建
user2 = User.model_validate_json(json_str)
print(user2.name)  # Alice
```

---

## 4. 嵌套模型

### 4.1 单层嵌套

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    address: Address  # 嵌套模型

# 创建嵌套数据
user = User(
    name="Alice",
    age=25,
    address={
        "street": "123 Main St",
        "city": "New York",
        "country": "USA"
    }
)

# 访问嵌套数据
print(user.address.city)  # New York
print(user.address.country)  # USA
```

### 4.2 列表嵌套

```python
from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    order_id: str
    items: List[Item]  # 列表嵌套

# 创建列表嵌套数据
order = Order(
    order_id="ORD-001",
    items=[
        {"name": "Apple", "price": 1.5},
        {"name": "Banana", "price": 0.8}
    ]
)

# 访问列表数据
for item in order.items:
    print(f"{item.name}: ${item.price}")
# Apple: $1.5
# Banana: $0.8
```

### 4.3 多层嵌套

```python
from pydantic import BaseModel
from typing import List

class Tag(BaseModel):
    name: str
    color: str

class Post(BaseModel):
    title: str
    content: str
    tags: List[Tag]

class User(BaseModel):
    name: str
    posts: List[Post]

# 创建多层嵌套数据
user = User(
    name="Alice",
    posts=[
        {
            "title": "First Post",
            "content": "Hello World",
            "tags": [
                {"name": "python", "color": "blue"},
                {"name": "tutorial", "color": "green"}
            ]
        }
    ]
)

# 访问多层嵌套数据
print(user.posts[0].tags[0].name)  # python
```

---

## 5. 模型配置

### 5.1 ConfigDict 配置

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        # 不可变（frozen）
        frozen=True,
        # 严格模式（不自动转换类型）
        strict=False,
        # 允许额外字段
        extra='allow',
        # 字段别名
        populate_by_name=True
    )

    name: str
    age: int
```

### 5.2 frozen（不可变）

```python
from pydantic import BaseModel, ConfigDict, ValidationError

class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str
    base_url: str

config = Config(api_key="sk-xxx", base_url="https://api.example.com")

# 尝试修改：报错
try:
    config.api_key = "new-key"
except ValidationError as e:
    print("配置是不可变的")
    # ValidationError: Instance is frozen
```

### 5.3 extra（额外字段处理）

```python
from pydantic import BaseModel, ConfigDict

# 忽略额外字段（默认）
class User1(BaseModel):
    model_config = ConfigDict(extra='ignore')
    name: str
    age: int

user1 = User1(name="Alice", age=25, unknown="value")
print(user1.model_dump())  # {'name': 'Alice', 'age': 25}

# 允许额外字段
class User2(BaseModel):
    model_config = ConfigDict(extra='allow')
    name: str
    age: int

user2 = User2(name="Alice", age=25, unknown="value")
print(user2.model_dump())  # {'name': 'Alice', 'age': 25, 'unknown': 'value'}

# 禁止额外字段（报错）
class User3(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    age: int

try:
    user3 = User3(name="Alice", age=25, unknown="value")
except ValidationError as e:
    print("不允许额外字段")
```

### 5.4 strict（严格模式）

```python
from pydantic import BaseModel, ConfigDict, ValidationError

# 非严格模式（默认，自动转换）
class User1(BaseModel):
    model_config = ConfigDict(strict=False)
    age: int

user1 = User1(age="25")  # ✅ 自动转换
print(user1.age)  # 25 (int)

# 严格模式（不自动转换）
class User2(BaseModel):
    model_config = ConfigDict(strict=True)
    age: int

try:
    user2 = User2(age="25")  # ❌ 报错
except ValidationError as e:
    print("严格模式不允许类型转换")
```

---

## 6. 模型方法

### 6.1 model_dump()

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    password: str

user = User(name="Alice", age=25, password="secret")

# 转换为字典
user_dict = user.model_dump()
print(user_dict)
# {'name': 'Alice', 'age': 25, 'password': 'secret'}

# 排除字段
user_dict = user.model_dump(exclude={'password'})
print(user_dict)
# {'name': 'Alice', 'age': 25}

# 只包含指定字段
user_dict = user.model_dump(include={'name', 'age'})
print(user_dict)
# {'name': 'Alice', 'age': 25}
```

### 6.2 model_dump_json()

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=25)

# 转换为 JSON 字符串
json_str = user.model_dump_json()
print(json_str)  # {"name":"Alice","age":25}

# 格式化输出
json_str = user.model_dump_json(indent=2)
print(json_str)
# {
#   "name": "Alice",
#   "age": 25
# }
```

### 6.3 model_validate()

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# 从字典创建
data = {"name": "Alice", "age": 25}
user = User.model_validate(data)
print(user.name)  # Alice

# 等价于
user = User(**data)
```

### 6.4 model_validate_json()

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# 从 JSON 字符串创建
json_str = '{"name": "Alice", "age": 25}'
user = User.model_validate_json(json_str)
print(user.name)  # Alice
```

### 6.5 model_copy()

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user1 = User(name="Alice", age=25)

# 复制模型
user2 = user1.model_copy()
print(user2.name)  # Alice

# 复制并更新字段
user3 = user1.model_copy(update={"age": 30})
print(user3.age)  # 30
print(user1.age)  # 25（原对象不变）
```

---

## 7. 在 FastAPI 中使用

### 7.1 请求体验证

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str
    age: int
    email: str

@app.post("/users")
async def create_user(request: CreateUserRequest):
    # FastAPI 自动验证请求体
    # request 已经是 CreateUserRequest 对象
    return {
        "message": f"User {request.name} created",
        "user": request.model_dump()
    }
```

### 7.2 响应模型

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    # 返回字典或 Pydantic 对象
    return {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "password": "secret"  # 不会返回（不在 UserResponse 中）
    }
```

### 7.3 嵌套模型

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str

@app.post("/chat")
async def chat(request: ChatRequest):
    # 自动验证嵌套结构
    for message in request.messages:
        print(f"{message.role}: {message.content}")

    return {"response": "Hello!"}
```

---

## 8. 在 AI Agent 中的应用

### 8.1 LLM 请求模型

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    model: Literal["gpt-4", "claude-3", "gpt-3.5-turbo"]
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, le=4096)
    stream: bool = False

# 使用
request = ChatRequest(
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    model="gpt-4"
)
```

### 8.2 RAG 检索模型

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    document_id: str
    content: str
    score: float = Field(..., ge=0, le=1)
    metadata: Dict[str, Any]

class RAGQueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
```

### 8.3 Agent 配置模型

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)  # 配置不可变

    model: Literal["gpt-4", "claude-3"]
    temperature: float = Field(0.7, ge=0, le=2)
    max_retries: int = Field(3, ge=0, le=10)
    timeout: int = Field(30, ge=1, le=300)

# 创建配置
config = AgentConfig(
    model="gpt-4",
    temperature=0.8,
    max_retries=5
)

# 配置不可修改
# config.temperature = 0.9  # ❌ 报错
```

---

## 总结

### BaseModel 的核心价值

1. **声明式定义**：用类型注解定义数据结构
2. **自动验证**：创建实例时自动验证数据
3. **自动转换**：尝试将数据转换为目标类型
4. **嵌套支持**：支持任意层级的嵌套结构
5. **序列化**：轻松转换为字典或 JSON
6. **FastAPI 集成**：无缝集成到 FastAPI 中

### 最佳实践

1. **必填 vs 可选**：明确区分必填和可选字段
2. **默认值**：使用 `default_factory` 避免可变默认值陷阱
3. **嵌套模型**：复杂数据结构用嵌套模型，不要用字典
4. **配置选项**：根据场景选择合适的配置（frozen, strict, extra）
5. **序列化控制**：使用 `exclude`/`include` 控制输出字段

### 下一步

- 学习 Field 验证规则（更细粒度的验证）
- 学习自定义验证器（复杂验证逻辑）
- 学习模型继承和复用

---

**版本：** v1.0
**最后更新：** 2026-02-11
