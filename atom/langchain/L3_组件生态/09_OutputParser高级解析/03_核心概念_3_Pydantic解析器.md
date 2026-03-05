# 核心概念 3：Pydantic 解析器（PydanticOutputParser）

> 本文档详细讲解 LangChain 中的 Pydantic 解析器，包括 Pydantic v1 vs v2 兼容性、格式指令生成、ORM 模式集成等核心特性。

---

## 文档元数据

**知识点**：OutputParser 高级解析 - Pydantic 解析器
**层级**：L3_组件生态
**依赖知识**：Runnable 接口、LCEL 表达式、Pydantic 基础、JsonOutputParser
**预计学习时间**：30 分钟
**难度等级**：⭐⭐⭐⭐

---

## 一、Pydantic 解析器概述

### 1.1 什么是 PydanticOutputParser？

PydanticOutputParser 是 LangChain 中用于将 LLM 输出解析为 Pydantic 模型的解析器。它继承自 JsonOutputParser，在 JSON 解析的基础上增加了严格的类型验证和模型验证功能。

**核心价值**：
- 提供严格的类型验证（基于 Pydantic 模型）
- 自动生成格式指令（注入到 Prompt）
- 支持 Pydantic v1 和 v2（自动检测）
- 支持 ORM 模式集成（从对象属性验证）
- 提供详细的验证错误信息

### 1.2 与 JsonOutputParser 的关系

```python
class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a Pydantic model."""

    pydantic_object: Annotated[type[TBaseModel], SkipValidation()]
    """The Pydantic model to parse."""
```

**关键特性**：
1. **继承自 JsonOutputParser**：复用 JSON 解析逻辑
2. **必须提供 pydantic_object**：Pydantic 模型类
3. **支持 Pydantic v1 和 v2**：自动检测版本

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:19-23)

---

## 二、Pydantic v1 vs v2 兼容性

### 2.1 版本自动检测机制

PydanticOutputParser 通过 `_parse_obj()` 方法自动检测 Pydantic 版本：

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
- **Pydantic v2**：使用 `model_validate()` 方法
- **Pydantic v1**：使用 `parse_obj()` 方法
- **自动检测**：通过 `issubclass()` 判断版本
- **错误处理**：捕获两个版本的 `ValidationError`

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:25-35)

### 2.2 Pydantic v1 vs v2 对比

| 特性 | Pydantic v1 | Pydantic v2 |
|------|-------------|-------------|
| 验证方法 | `parse_obj()` | `model_validate()` |
| 序列化方法 | `dict()` | `model_dump()` |
| Schema 生成 | `schema()` | `model_json_schema()` |
| ORM 模式 | `Config.orm_mode = True` | `ConfigDict(from_attributes=True)` |
| 性能 | 较慢 | 更快（使用 Rust 核心） |
| 导入方式 | `from pydantic import BaseModel` | `from pydantic import BaseModel` |
| v1 兼容 | - | `from pydantic.v1 import BaseModel` |

**来源**：`reference/context7_pydantic_01_BaseModel验证.md` (145-153)

### 2.3 使用 Pydantic v2（推荐）

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age", ge=0, le=150)
    city: str = Field(description="The person's city")

parser = PydanticOutputParser(pydantic_object=Person)

# 自动使用 model_validate() 方法
result = parser.parse('{"name": "Alice", "age": 30, "city": "New York"}')
print(result)
# 输出: Person(name='Alice', age=30, city='New York')
```

### 2.4 兼容 Pydantic v1

```python
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)

# 自动使用 parse_obj() 方法
result = parser.parse('{"name": "Alice", "age": 30}')
print(result)
# 输出: Person(name='Alice', age=30)
```

**来源**：`reference/context7_pydantic_01_BaseModel验证.md` (204-232)

---

## 三、格式指令生成

### 3.1 get_format_instructions() 方法

PydanticOutputParser 可以自动生成格式指令，注入到 Prompt 中：

```python
def get_format_instructions(self) -> str:
    """Return the format instructions for the JSON output.

    Returns:
        The format instructions for the JSON output.
    """
    # Copy schema to avoid altering original Pydantic schema.
    schema = dict(self._get_schema(self.pydantic_object).items())
    # ... (省略具体实现)
```

**功能**：
- 从 Pydantic 模型生成 JSON Schema
- 生成格式化指令（可以注入到 Prompt 中）
- 告诉 LLM 如何生成符合模型的 JSON

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:93-100)

### 3.2 Schema 提取

```python
@staticmethod
def _get_schema(pydantic_object: type[TBaseModel]) -> dict[str, Any]:
    if issubclass(pydantic_object, pydantic.BaseModel):
        return pydantic_object.model_json_schema()  # Pydantic v2
    return pydantic_object.schema()  # Pydantic v1
```

**关键发现**：
- **Pydantic v2**：使用 `model_json_schema()`
- **Pydantic v1**：使用 `schema()`
- **返回标准 JSON Schema**

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:54-58)

### 3.3 实战示例

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The person's city")

parser = PydanticOutputParser(pydantic_object=Person)

# 获取格式指令
format_instructions = parser.get_format_instructions()
print(format_instructions)
# 输出:
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
#
# Here is the output schema:
# ```
# {"properties": {"name": {"description": "The person's name", "title": "Name", "type": "string"}, "age": {"description": "The person's age", "title": "Age", "type": "integer"}, "city": {"description": "The person's city", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}
# ```

# 注入到 Prompt
prompt = PromptTemplate(
    template="Extract the person's information from the text.\n{format_instructions}\n\nText: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | model | parser

result = chain.invoke({"text": "Alice is 30 years old and lives in New York."})
print(result)
# 输出: Person(name='Alice', age=30, city='New York')
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:296-315)

---

## 四、ORM 模式集成

### 4.1 什么是 ORM 模式？

ORM 模式允许 Pydantic 模型从对象的属性中提取数据，而不是从字典中提取。这对于与 ORM（如 SQLAlchemy）集成非常有用。

**Pydantic v2 配置**：
```python
from pydantic import BaseModel, ConfigDict

class PersonModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: int
```

**Pydantic v1 配置**：
```python
from pydantic import BaseModel

class PersonModel(BaseModel):
    class Config:
        orm_mode = True

    name: str
    age: int
```

**来源**：`reference/context7_pydantic_01_BaseModel验证.md` (46-93)

### 4.2 从 ORM 对象验证

```python
from pydantic import BaseModel, ConfigDict

class ORMUser:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class UserModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: int

orm_user = ORMUser("Eve", 28)
user_model = UserModel.model_validate(orm_user, from_attributes=True)
print(user_model)
# 输出: UserModel(name='Eve', age=28)
```

**关键点**：
- 通过设置 `from_attributes=True`，Pydantic 可以从对象的属性中提取数据
- 不需要字典结构
- 特别适用于与 ORM 或其他面向对象数据源的无缝集成

**来源**：`reference/context7_pydantic_01_BaseModel验证.md` (94-119)

### 4.3 SQLAlchemy ORM 集成

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

co_model = CompanyModel.model_validate(co_orm)
print(co_model)
# 输出: CompanyModel(id=123, public_key='foobar', domains=['example.com', 'foobar.com'])
```

**来源**：`reference/context7_pydantic_01_BaseModel验证.md` (46-93)

---

## 五、错误处理与验证

### 5.1 验证错误处理

```python
def _parser_exception(
    self, e: Exception, json_object: dict
) -> OutputParserException:
    json_string = json.dumps(json_object, ensure_ascii=False)
    name = self.pydantic_object.__name__
    msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
    return OutputParserException(msg, llm_output=json_string)
```

**关键特性**：
- 包含模型名称（`self.pydantic_object.__name__`）
- 包含原始 JSON 字符串（`json_string`）
- 包含验证错误信息（`e`）
- 使用 `ensure_ascii=False` 保留 Unicode 字符

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:37-43)

### 5.2 实战示例

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age", ge=0, le=150)

parser = PydanticOutputParser(pydantic_object=Person)

try:
    result = parser.parse('{"name": "Alice", "age": 200}')
except OutputParserException as e:
    print(f"Error: {e}")
    print(f"LLM output: {e.llm_output}")
    # 输出:
    # Error: Failed to parse Person from completion {"name": "Alice", "age": 200}. Got: 1 validation error for Person
    # age
    #   Input should be less than or equal to 150 [type=less_than_equal, input_value=200, input_type=int]
    # LLM output: {"name": "Alice", "age": 200}
```

---

## 六、实战代码示例

### 6.1 基础 Pydantic 解析

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The person's city")

parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="Extract info: {text}\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | model | parser

result = chain.invoke({"text": "Alice is 30 years old and lives in New York."})
print(result)
# 输出: Person(name='Alice', age=30, city='New York')
```

### 6.2 复杂模型验证

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age", ge=0, le=150)
    addresses: List[Address] = Field(description="List of addresses")

parser = PydanticOutputParser(pydantic_object=Person)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = model | parser

result = chain.invoke("Extract: Alice, 30, lives at 123 Main St, New York, USA and 456 Oak Ave, Boston, USA")
print(result)
# 输出: Person(name='Alice', age=30, addresses=[Address(street='123 Main St', city='New York', country='USA'), Address(street='456 Oak Ave', city='Boston', country='USA')])
```

### 6.3 流式解析

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    city: str = Field(description="Person's city")

parser = PydanticOutputParser(pydantic_object=Person)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = model | parser

# 流式解析
for chunk in chain.stream("Extract: Alice is 30 years old and lives in New York."):
    print(chunk)
    # 输出:
    # None (部分解析失败时)
    # Person(name='Alice', age=30, city='New York') (完整解析成功时)
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (pydantic.py:55-80)

---

## 七、最佳实践

### 7.1 何时使用 PydanticOutputParser

**推荐场景**：
- ✅ 需要严格的类型验证
- ✅ 需要复杂的数据结构（嵌套模型）
- ✅ 需要字段约束（如 `ge`, `le`, `max_length`）
- ✅ 需要自动生成格式指令
- ✅ 需要与 ORM 集成

**不推荐场景**：
- ❌ 模型支持原生结构化输出（使用 `with_structured_output()` 更好）
- ❌ 动态或未知 schema（使用 `JsonOutputParser` 更灵活）
- ❌ 简单场景（使用 `SimpleJsonOutputParser` 更简单）

**来源**：`reference/context7_langchain_02_Pydantic模型.md` (163-181)

### 7.2 Prompt 工程技巧

**包含格式指令**：
```python
parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

prompt = f"""
Extract person information from the text.

{format_instructions}

Text: Alice is 30 years old and lives in New York.
"""
```

**提供示例输出**：
```python
prompt = """
Extract person information from the text.

Example output:
{{"name": "John", "age": 25, "city": "Boston"}}

Text: Alice is 30 years old and lives in New York.
"""
```

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (108-116)

### 7.3 模型选择建议

**推荐模型**：
- ✅ GPT-4 / GPT-4o：输出最稳定
- ✅ GPT-3.5-turbo：性价比高
- ✅ Claude 3.5 Sonnet：输出质量高

**需要注意的模型**：
- ⚠️ Llama 3.1：可能需要更多 Prompt 工程
- ⚠️ 开源模型：建议使用 OutputFixingParser 提高鲁棒性

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (119-125)

---

## 八、社区最佳实践

### 8.1 Reddit 社区共识

根据 Reddit 社区讨论，以下是 PydanticOutputParser 的最佳实践：

1. **PydanticOutputParser 是首选**：当需要严格的类型验证时
2. **包含格式指令**：可以显著提高成功率
3. **模型选择影响大**：更强大的模型（如 GPT-4）输出更稳定
4. **错误处理很重要**：使用 OutputFixingParser 和重试机制

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (118-125)

### 8.2 一致格式输出

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser

class Item(BaseModel):
    name: str = Field(description="Item name")
    price: float = Field(description="Item price")

class ItemList(BaseModel):
    items: List[Item] = Field(description="List of exactly 10 items", min_items=10, max_items=10)

parser = PydanticOutputParser(pydantic_object=ItemList)

# 确保输出固定数量的项目（如10个）
```

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (53-59)

---

## 九、总结

### 9.1 核心要点

1. **PydanticOutputParser 提供严格的类型验证**（基于 Pydantic 模型）
2. **支持 Pydantic v1 和 v2**（自动检测版本）
3. **自动生成格式指令**（注入到 Prompt）
4. **支持 ORM 模式集成**（从对象属性验证）
5. **提供详细的验证错误信息**

### 9.2 选择指南

| 场景 | 推荐解析器 |
|------|-----------|
| 需要严格类型验证 | PydanticOutputParser |
| 需要复杂数据结构 | PydanticOutputParser |
| 需要字段约束 | PydanticOutputParser |
| 需要 ORM 集成 | PydanticOutputParser |
| 动态 schema | JsonOutputParser |
| 简单场景 | SimpleJsonOutputParser |
| 模型支持原生结构化输出 | with_structured_output() |

### 9.3 下一步学习

- 学习 **OutputFixingParser**：了解如何自动修复解析错误
- 学习 **RetryOutputParser**：了解如何实现重试机制
- 学习 **自定义 OutputParser**：了解如何开发自己的解析器

---

## 参考资料

1. **源码分析**：`reference/source_outputparser_02_JSON解析器.md`
2. **Context7 文档**：`reference/context7_langchain_02_Pydantic模型.md`
3. **Pydantic 文档**：`reference/context7_pydantic_01_BaseModel验证.md`
4. **Reddit 最佳实践**：`reference/search_outputparser_01_Reddit最佳实践.md`
5. **LangChain 官方文档**：https://docs.langchain.com/oss/python/langchain/models

---

**版本**：v1.0
**最后更新**：2026-02-26
**维护者**：Claude Code
