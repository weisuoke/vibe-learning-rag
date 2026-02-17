# 核心概念1：Model 定义与映射

深入理解 SQLAlchemy 的声明式模型系统。

---

## 1. 什么是声明式模型？

**声明式模型（Declarative Model）** 是 SQLAlchemy 推荐的定义数据模型的方式，通过 Python 类来声明数据库表结构。

**核心思想：**
- 用 Python 类表示数据库表
- 用类属性表示表的列
- 用类型注解和 Column 定义列的类型和约束

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

# 创建基类
Base = declarative_base()

# 定义模型
class User(Base):
    __tablename__ = "users"  # 表名

    id = Column(Integer, primary_key=True)  # 主键
    name = Column(String(50), nullable=False)  # 非空字符串
    email = Column(String(100), unique=True)  # 唯一邮箱
```

**类比前端：** 就像 TypeScript 的 interface，定义数据结构

---

## 2. Base 类的作用

### 2.1 什么是 Base？

**Base** 是所有模型类的基类，提供了 ORM 的核心功能。

```python
from sqlalchemy.orm import declarative_base

# 创建 Base 类
Base = declarative_base()

# 所有模型都继承 Base
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
```

### 2.2 Base 提供的功能

**1. 元数据管理（MetaData）**

```python
# Base 维护了所有表的元数据
print(Base.metadata.tables.keys())
# dict_keys(['users', 'conversations'])

# 创建所有表
Base.metadata.create_all(engine)

# 删除所有表
Base.metadata.drop_all(engine)
```

**2. 查询接口**

```python
# Base 提供了查询接口（需要配置 Session）
from sqlalchemy.orm import Session

session = Session(engine)
users = session.query(User).all()
```

**3. 表映射**

```python
# Base 自动将类映射到表
User.__table__  # Table('users', ...)
User.__tablename__  # 'users'
User.__mapper__  # Mapper[User(users)]
```

### 2.3 在 AI Agent 开发中的应用

```python
# 定义 AI Agent 的数据模型
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200))

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20))  # "user" or "assistant"
    content = Column(Text)

# 创建所有表
Base.metadata.create_all(engine)
```

---

## 3. Column 类型系统

### 3.1 常用列类型

| SQLAlchemy 类型 | Python 类型 | PostgreSQL 类型 | 说明 |
|----------------|-------------|----------------|------|
| `Integer` | `int` | `INTEGER` | 整数 |
| `String(n)` | `str` | `VARCHAR(n)` | 可变长度字符串 |
| `Text` | `str` | `TEXT` | 无限长度文本 |
| `Boolean` | `bool` | `BOOLEAN` | 布尔值 |
| `Float` | `float` | `REAL` | 浮点数 |
| `DateTime` | `datetime` | `TIMESTAMP` | 日期时间 |
| `Date` | `date` | `DATE` | 日期 |
| `Time` | `time` | `TIME` | 时间 |
| `JSON` | `dict` | `JSON` | JSON 数据 |
| `ARRAY` | `list` | `ARRAY` | 数组（PostgreSQL） |

### 3.2 列类型示例

```python
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, JSON
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    # 整数主键
    id = Column(Integer, primary_key=True)

    # 字符串（限制长度）
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)

    # 文本（无限长度）
    bio = Column(Text)

    # 布尔值
    is_active = Column(Boolean, default=True)

    # 日期时间
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # JSON
    settings = Column(JSON, default={})
```

### 3.3 PostgreSQL 特有类型

```python
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB
import uuid

class Document(Base):
    __tablename__ = "documents"

    # UUID 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # JSONB（比 JSON 更高效）
    metadata = Column(JSONB)

    # 数组
    tags = Column(ARRAY(String))

    # 向量（需要 pgvector 扩展）
    from pgvector.sqlalchemy import Vector
    embedding = Column(Vector(1536))  # OpenAI embedding 维度
```

### 3.4 在 AI Agent 开发中的应用

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid

class Conversation(Base):
    __tablename__ = "conversations"

    # UUID 主键（更安全）
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 外键
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # 字符串
    title = Column(String(200), nullable=False)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # JSON 元数据
    metadata = Column(JSON, default={})

    # 标签数组
    tags = Column(ARRAY(String), default=[])

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))

    # 角色（user/assistant/system）
    role = Column(String(20), nullable=False)

    # 消息内容
    content = Column(Text, nullable=False)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # Token 统计
    token_count = Column(Integer)

    # 元数据（模型、温度等）
    metadata = Column(JSON, default={})
```

---

## 4. 列约束

### 4.1 常用约束

```python
class User(Base):
    __tablename__ = "users"

    # 主键约束
    id = Column(Integer, primary_key=True)

    # 非空约束
    name = Column(String(50), nullable=False)

    # 唯一约束
    email = Column(String(100), unique=True)

    # 默认值
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 自动更新
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # 索引
    username = Column(String(50), index=True)

    # 检查约束
    from sqlalchemy import CheckConstraint
    age = Column(Integer, CheckConstraint('age >= 0'))
```

### 4.2 复合约束

```python
from sqlalchemy import UniqueConstraint, Index, CheckConstraint

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(100))
    age = Column(Integer)

    # 复合唯一约束
    __table_args__ = (
        UniqueConstraint('first_name', 'last_name', name='uq_full_name'),

        # 复合索引
        Index('idx_name_email', 'first_name', 'email'),

        # 检查约束
        CheckConstraint('age >= 18', name='check_adult'),
    )
```

### 4.3 在 AI Agent 开发中的应用

```python
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)  # 索引，方便按时间查询
    is_archived = Column(Boolean, default=False, index=True)  # 索引，方便筛选

    # 复合索引：按用户和时间查询
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
    )
```

---

## 5. 自动生成值

### 5.1 默认值

```python
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    # Python 函数作为默认值
    id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 固定值作为默认值
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default="user")

    # 数据库级别的默认值
    from sqlalchemy import text
    status = Column(String(20), server_default=text("'pending'"))
```

### 5.2 自增主键

```python
class User(Base):
    __tablename__ = "users"

    # 自增主键（PostgreSQL 使用 SERIAL）
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 或者使用 Sequence
    from sqlalchemy import Sequence
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
```

### 5.3 自动更新时间戳

```python
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)

    # 创建时间（只设置一次）
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # 更新时间（每次更新都会自动更新）
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### 5.4 在 AI Agent 开发中的应用

```python
class Message(Base):
    __tablename__ = "messages"

    # UUID 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 外键
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))

    # 角色（默认 user）
    role = Column(String(20), default="user")

    # 内容
    content = Column(Text, nullable=False)

    # 自动时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Token 统计（默认 0）
    token_count = Column(Integer, default=0)

    # 元数据（默认空字典）
    metadata = Column(JSON, default=dict)
```

---

## 6. 表配置选项

### 6.1 __table_args__

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(50))

    # 表级别配置
    __table_args__ = {
        'mysql_engine': 'InnoDB',  # MySQL 引擎
        'mysql_charset': 'utf8mb4',  # 字符集
        'postgresql_partition_by': 'RANGE (created_at)',  # PostgreSQL 分区
        'comment': '用户表',  # 表注释
    }
```

### 6.2 表继承

```python
# 单表继承
class Person(Base):
    __tablename__ = "people"

    id = Column(Integer, primary_key=True)
    type = Column(String(50))  # 区分子类
    name = Column(String(50))

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'person'
    }

class Employee(Person):
    employee_id = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'employee'
    }

class Customer(Person):
    customer_id = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'customer'
    }
```

---

## 7. 完整示例：AI Agent 数据模型

```python
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """用户模型"""
    __tablename__ = "users"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 基本信息
    email = Column(String(100), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # 状态
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)

    # 元数据
    metadata = Column(JSON, default=dict)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"

class Conversation(Base):
    """对话模型"""
    __tablename__ = "conversations"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 外键
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # 基本信息
    title = Column(String(200), nullable=False)

    # 状态
    is_archived = Column(Boolean, default=False, index=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 标签
    tags = Column(ARRAY(String), default=list)

    # 元数据
    metadata = Column(JSON, default=dict)

    # 复合索引
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"

class Message(Base):
    """消息模型"""
    __tablename__ = "messages"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 外键
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)

    # 消息内容
    role = Column(String(20), nullable=False)  # user/assistant/system
    content = Column(Text, nullable=False)

    # Token 统计
    token_count = Column(Integer, default=0)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # 元数据（模型、温度等）
    metadata = Column(JSON, default=dict)

    # 复合索引
    __table_args__ = (
        Index('idx_conversation_created', 'conversation_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"
```

---

## 8. 最佳实践

### 8.1 使用 UUID 作为主键

```python
# ✅ 推荐：UUID 主键
id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

# ❌ 不推荐：自增 ID（容易被猜测）
id = Column(Integer, primary_key=True, autoincrement=True)
```

### 8.2 添加时间戳

```python
# ✅ 推荐：添加创建和更新时间
created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### 8.3 添加索引

```python
# ✅ 推荐：为常用查询字段添加索引
email = Column(String(100), unique=True, index=True)
created_at = Column(DateTime, default=datetime.utcnow, index=True)
```

### 8.4 使用 __repr__

```python
# ✅ 推荐：添加 __repr__ 方便调试
def __repr__(self):
    return f"<User(id={self.id}, email={self.email})>"
```

### 8.5 使用类型注解

```python
from typing import Optional
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id: UUID
    email: str
    created_at: datetime
    updated_at: Optional[datetime]

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

---

## 总结

**Model 定义与映射的核心要点：**

1. **声明式模型**：用 Python 类定义数据库表
2. **Base 类**：所有模型的基类，提供 ORM 功能
3. **Column 类型**：丰富的类型系统，支持 PostgreSQL 特有类型
4. **列约束**：主键、唯一、非空、默认值、索引等
5. **自动生成值**：UUID、时间戳、自增 ID
6. **表配置**：__table_args__、__mapper_args__
7. **最佳实践**：UUID 主键、时间戳、索引、__repr__

**在 AI Agent 开发中的应用：**
- 用 UUID 作为主键，更安全
- 添加时间戳，方便追踪
- 使用 JSON 存储元数据，灵活扩展
- 添加索引，优化查询性能
- 使用 PostgreSQL 特有类型（ARRAY、JSONB、Vector）

---

**记住：** Model 定义是 ORM 的基础，定义好数据模型，后续的 CRUD 操作和关系映射才能顺利进行。
