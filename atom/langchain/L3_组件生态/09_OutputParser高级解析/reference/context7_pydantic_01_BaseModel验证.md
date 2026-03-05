---
type: context7_documentation
library: pydantic
version: latest (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: OutputParser高级解析
context7_query: BaseModel Field validation v1 v2 model_validate parse_obj
---

# Context7 文档：Pydantic（BaseModel 与验证）

## 文档来源
- 库名称：Pydantic
- 版本：latest (2026-02-17)
- 官方文档链接：https://github.com/pydantic/pydantic
- Context7 库 ID：/pydantic/pydantic
- 总文档片段：793
- Stars：23,455
- 信任分数：9.6/10
- 基准分数：82.6/100

## 关键信息提取

### 1. Pydantic 模型验证和序列化

**来源**：https://github.com/pydantic/pydantic/blob/main/docs/internals/architecture.md

**示例代码**：

```python
from pydantic import BaseModel

class Model(BaseModel):
    foo: int

model = Model.model_validate({'foo': 1})
dumped = model.model_dump()
```

**关键点**：
- `model_validate()` 方法使用 `pydantic-core` 的 `SchemaValidator` 验证输入数据
- `model_dump()` 方法使用 `pydantic-core` 的 `SchemaSerializer` 将模型实例转换为 Python 字典
- 遵循模型的核心 schema

### 2. 验证任意类实例（SQLAlchemy ORM）

**来源**：https://github.com/pydantic/pydantic/blob/main/docs/concepts/models.md

**示例代码**：

```python
from typing import Annotated
from sqlalchemy import ARRAY, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pydantic import BaseModel, ConfigDict, StringConstraints

class Base(DeclarativeBase):
    pass

class CompanyOrm(Base):
    __tablename__ = 'companies'

    id: Mapped[int] = mapped_column(primary_key=True, nullable=False)
    public_key: Mapped[str] = mapped_column(
        String(20), index=True, nullable=False, unique=True
    )
    domains: Mapped[list[str]] = mapped_column(ARRAY(String(255)))

class CompanyModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    public_key: Annotated[str, StringConstraints(max_length=20)]
    domains: list[Annotated[str, StringConstraints(max_length=255)]]

co_orm = CompanyOrm(
    id=123,
    public_key='foobar',
    domains=['example.com', 'foobar.com'],
)
print(co_orm)
#> <__main__.CompanyOrm object at 0x0123456789ab>

co_model = CompanyModel.model_validate(co_orm)
print(co_model)
#> id=123 public_key='foobar' domains=['example.com', 'foobar.com']
```

**关键点**：
- 使用 `model_validate()` 和 `from_attributes=True` 配置验证任意对象
- 特别适用于 ORM 集成
- 通过映射属性到 Pydantic BaseModel 并应用字段约束

### 3. 从对象属性验证 Pydantic 模型（ORM 模式）

**来源**：https://context7.com/pydantic/pydantic/llms.txt

**示例代码**：

```python
class ORMUser:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class UserModel(BaseModel):
    name: str
    age: int

orm_user = ORMUser("Eve", 28)
user_model = UserModel.model_validate(orm_user, from_attributes=True)
print(user_model)
```

**关键点**：
- 通过设置 `from_attributes=True` 在 `model_validate` 中，Pydantic 可以从对象的属性中提取数据
- 不需要字典结构
- 特别适用于与 ORM 或其他面向对象数据源的无缝集成

### 4. Pydantic 模型验证成功示例

**来源**：https://github.com/pydantic/pydantic/blob/main/docs/index.md

**关键点**：
- 定义 Pydantic `BaseModel` 与各种字段类型
- 包括必需的整数、带默认值的字符串、可选的日期时间、具有特定键值类型约束的字典
- 展示 Pydantic 的自动类型强制转换（例如，字符串到日期时间、字节到字符串、字符串到整数）
- 从外部数据创建实例、访问属性、使用 `model_dump()` 转换回字典

## 与 OutputParser 的关系

### 1. Pydantic v2 的核心方法

**model_validate()**：
- Pydantic v2 的主要验证方法
- 替代 Pydantic v1 的 `parse_obj()`
- 使用 `pydantic-core` 的 `SchemaValidator`

**model_dump()**：
- Pydantic v2 的序列化方法
- 替代 Pydantic v1 的 `dict()`
- 使用 `pydantic-core` 的 `SchemaSerializer`

### 2. Pydantic v1 vs v2 对比

| 特性 | Pydantic v1 | Pydantic v2 |
|------|-------------|-------------|
| 验证方法 | `parse_obj()` | `model_validate()` |
| 序列化方法 | `dict()` | `model_dump()` |
| Schema 生成 | `schema()` | `model_json_schema()` |
| ORM 模式 | `Config.orm_mode = True` | `ConfigDict(from_attributes=True)` |
| 性能 | 较慢 | 更快（使用 Rust 核心） |

### 3. PydanticOutputParser 的兼容性

**从源码分析中**（source_outputparser_02_JSON解析器.md）：

```python
def _parse_obj(self, obj: dict) -> TBaseModel:
    try:
        if issubclass(self.pydantic_object, pydantic.BaseModel):
            return self.pydantic_object.model_validate(obj)  # Pydantic v2
        if issubclass(self.pydantic_object, pydantic.v1.BaseModel):
            return self.pydantic_object.parse_obj(obj)  # Pydantic v1
        msg = f"Unsupported model version for PydanticOutputParser: \
                    {self.pydantic_object.__class__}"
        raise OutputParserException(msg)
    except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
        raise self._parser_exception(e, obj) from e
```

**关键发现**：
- LangChain 的 `PydanticOutputParser` 同时支持 Pydantic v1 和 v2
- 自动检测 Pydantic 版本
- 使用相应的验证方法

### 4. ORM 模式的应用

**在 OutputParser 中的使用**：

```python
from pydantic import BaseModel, ConfigDict

class Person(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: int

# 从 ORM 对象验证
parser = PydanticOutputParser(pydantic_object=Person)
result = parser.parse(orm_object)
```

**用途**：
- 从 ORM 对象（如 SQLAlchemy 模型）创建 Pydantic 模型
- 从任意对象属性提取数据
- 适用于数据库集成场景

## 实战建议

### 1. 使用 Pydantic v2（推荐）

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

# Pydantic v2 验证
person = Person.model_validate({"name": "Alice", "age": 30})

# Pydantic v2 序列化
data = person.model_dump()
```

### 2. 兼容 Pydantic v1

```python
from pydantic.v1 import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

# Pydantic v1 验证
person = Person.parse_obj({"name": "Alice", "age": 30})

# Pydantic v1 序列化
data = person.dict()
```

### 3. ORM 集成

```python
from pydantic import BaseModel, ConfigDict

class PersonModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: int

# 从 ORM 对象验证
person = PersonModel.model_validate(orm_user, from_attributes=True)
```

### 4. 在 PydanticOutputParser 中使用

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)

# 自动检测 Pydantic 版本并使用相应的验证方法
result = parser.parse('{"name": "Alice", "age": 30}')
```

## 总结

Context7 文档主要关注：
1. **Pydantic v2 核心方法**：`model_validate()` 和 `model_dump()`
2. **ORM 集成**：`from_attributes=True` 配置
3. **类型强制转换**：自动类型转换和验证
4. **性能优化**：使用 Rust 核心（`pydantic-core`）

**与 OutputParser 的关系**：
- `PydanticOutputParser` 同时支持 Pydantic v1 和 v2
- 自动检测版本并使用相应的方法
- 支持 ORM 模式用于数据库集成

**下一步**：进入 Grok-mcp 网络搜索阶段，搜索社区资料和实践案例。
