# 实战代码 - 场景2：Pydantic 模型验证

> **数据来源**：
> - `reference/source_outputparser_02_JSON解析器.md` - PydanticOutputParser 源码分析
> - `reference/context7_pydantic_01_BaseModel验证.md` - Pydantic 验证文档

---

## 场景概述

PydanticOutputParser 是 LangChain 中用于类型安全输出验证的解析器，它结合了 JSON 解析和 Pydantic 模型验证。

**核心特性**：
- 自动类型验证和转换
- 生成格式指令（注入到 Prompt）
- 支持 Pydantic v1 和 v2
- 支持复杂嵌套模型
- 支持 ORM 模式集成

**适用场景**：
- 需要严格的类型验证
- 需要复杂的数据结构
- 需要自动生成格式指令
- 需要与 ORM 集成

---

## 示例 1：基础 PydanticOutputParser 使用

**目标**：使用 PydanticOutputParser 验证 LLM 输出

```python
"""示例 1：基础 PydanticOutputParser 使用"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 1. 定义 Pydantic 模型
class Person(BaseModel):
    """人物信息模型"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    occupation: str = Field(description="职业")
    email: str = Field(description="邮箱地址")

# 2. 创建 Parser
parser = PydanticOutputParser(pydantic_object=Person)

# 3. 获取格式指令
format_instructions = parser.get_format_instructions()
print("格式指令：")
print(format_instructions[:200] + "...\n")

# 4. 创建 Prompt
prompt = PromptTemplate(
    template="""提取以下文本中的人物信息：

{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
)

# 5. 构建链
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | parser

# 6. 执行
text = "张三是一位 30 岁的软件工程师，他的邮箱是 zhangsan@example.com"

try:
    result = chain.invoke({"text": text})
    print(f"解析成功！")
    print(f"类型: {type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")
    print(f"邮箱: {result.email}")
except Exception as e:
    print(f"解析失败: {e}")

"""
预期输出：
格式指令：
The output should be formatted as a JSON instance that conforms to the JSON schema below...

解析成功！
类型: <class '__main__.Person'>
姓名: 张三
年龄: 30
职业: 软件工程师
邮箱: zhangsan@example.com
"""
```

**关键点**：
- `Field()` 定义字段约束（ge=0, le=150）
- `get_format_instructions()` 自动生成格式指令
- 返回 Pydantic 对象，可以直接访问属性
- 自动类型验证（age 必须是整数）

---

## 示例 2：Pydantic v1 vs v2 对比

**目标**：理解 Pydantic v1 和 v2 的差异

```python
"""示例 2：Pydantic v1 vs v2 对比"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# Pydantic v2 模型（推荐）
from pydantic import BaseModel, Field

class PersonV2(BaseModel):
    """Pydantic v2 模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

# Pydantic v1 模型（兼容）
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1

class PersonV1(BaseModelV1):
    """Pydantic v1 模型"""
    name: str = FieldV1(description="姓名")
    age: int = FieldV1(description="年龄")

# 测试两个版本
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=" * 60)
print("Pydantic v2 测试")
print("=" * 60)

parser_v2 = PydanticOutputParser(pydantic_object=PersonV2)
chain_v2 = llm | parser_v2

try:
    result_v2 = chain_v2.invoke('{"name": "Alice", "age": 30}')
    print(f"类型: {type(result_v2)}")
    print(f"结果: {result_v2}")
    print(f"验证方法: model_validate()")
except Exception as e:
    print(f"失败: {e}")

print("\n" + "=" * 60)
print("Pydantic v1 测试")
print("=" * 60)

parser_v1 = PydanticOutputParser(pydantic_object=PersonV1)
chain_v1 = llm | parser_v1

try:
    result_v1 = chain_v1.invoke('{"name": "Bob", "age": 25}')
    print(f"类型: {type(result_v1)}")
    print(f"结果: {result_v1}")
    print(f"验证方法: parse_obj()")
except Exception as e:
    print(f"失败: {e}")

"""
关键差异：

| 特性 | Pydantic v1 | Pydantic v2 |
|------|-------------|-------------|
| 验证方法 | parse_obj() | model_validate() |
| 序列化方法 | dict() | model_dump() |
| Schema 生成 | schema() | model_json_schema() |
| 性能 | 较慢 | 更快（Rust 核心） |
| 导入 | from pydantic.v1 | from pydantic |

PydanticOutputParser 自动检测版本并使用相应的方法。
"""
```

---

## 示例 3：复杂嵌套模型验证

**目标**：处理复杂的嵌套数据结构

```python
"""示例 3：复杂嵌套模型验证"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# 定义嵌套模型
class Address(BaseModel):
    """地址模型"""
    street: str = Field(description="街道")
    city: str = Field(description="城市")
    country: str = Field(description="国家")

class Company(BaseModel):
    """公司模型"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="行业")

class Employee(BaseModel):
    """员工模型（嵌套）"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    address: Address = Field(description="地址信息")
    company: Company = Field(description="公司信息")
    skills: List[str] = Field(description="技能列表")
    email: Optional[str] = Field(default=None, description="邮箱（可选）")

# 创建 Parser
parser = PydanticOutputParser(pydantic_object=Employee)

# 创建 Prompt
prompt = PromptTemplate(
    template="""提取以下员工信息：

{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 构建链
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | parser

# 测试
text = """
李明，28岁，住在中国北京市朝阳区建国路1号。
他在阿里巴巴科技公司工作，该公司属于互联网行业。
他擅长 Python、机器学习和数据分析。
"""

try:
    result = chain.invoke({"text": text})
    print("解析成功！")
    print(f"\n姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"地址: {result.address.city}, {result.address.country}")
    print(f"公司: {result.company.name} ({result.company.industry})")
    print(f"技能: {', '.join(result.skills)}")
    print(f"邮箱: {result.email or '未提供'}")
except Exception as e:
    print(f"解析失败: {e}")

"""
预期输出：
解析成功！

姓名: 李明
年龄: 28
地址: 北京市, 中国
公司: 阿里巴巴科技公司 (互联网)
技能: Python, 机器学习, 数据分析
邮箱: 未提供
"""
```

**关键点**：
- 支持嵌套模型（Address, Company）
- 支持列表类型（List[str]）
- 支持可选字段（Optional[str]）
- 自动验证所有嵌套结构

---

## 示例 4：ORM 模式集成

**目标**：从 ORM 对象或任意对象属性验证

```python
"""示例 4：ORM 模式集成"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

# 模拟 ORM 对象
class UserORM:
    """模拟数据库 ORM 对象"""
    def __init__(self, id, username, email, is_active):
        self.id = id
        self.username = username
        self.email = email
        self.is_active = is_active

# 定义 Pydantic 模型（启用 ORM 模式）
class UserModel(BaseModel):
    """用户模型（支持 ORM）"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    email: str = Field(description="邮箱")
    is_active: bool = Field(description="是否激活")

# 创建 ORM 对象
orm_user = UserORM(
    id=123,
    username="alice",
    email="alice@example.com",
    is_active=True
)

print("ORM 对象:")
print(f"  类型: {type(orm_user)}")
print(f"  用户名: {orm_user.username}")

# 使用 Pydantic 验证
try:
    # 方法1：直接使用 model_validate
    user_model = UserModel.model_validate(orm_user)
    print("\n验证成功！")
    print(f"  类型: {type(user_model)}")
    print(f"  用户名: {user_model.username}")
    print(f"  邮箱: {user_model.email}")

    # 方法2：使用 PydanticOutputParser（从 JSON）
    parser = PydanticOutputParser(pydantic_object=UserModel)
    json_str = '{"id": 456, "username": "bob", "email": "bob@example.com", "is_active": false}'
    user_from_json = parser.parse(json_str)
    print(f"\n从 JSON 解析:")
    print(f"  用户名: {user_from_json.username}")
    print(f"  激活状态: {user_from_json.is_active}")

except Exception as e:
    print(f"验证失败: {e}")

"""
预期输出：
ORM 对象:
  类型: <class '__main__.UserORM'>
  用户名: alice

验证成功！
  类型: <class '__main__.UserModel'>
  用户名: alice
  邮箱: alice@example.com

从 JSON 解析:
  用户名: bob
  激活状态: False
"""
```

**关键点**：
- `ConfigDict(from_attributes=True)` 启用 ORM 模式
- 可以从任意对象属性提取数据
- 适用于 SQLAlchemy、Django ORM 等
- 自动类型转换和验证

---

## 示例 5：错误处理与调试

**目标**：处理验证错误和调试

```python
"""示例 5：错误处理与调试"""

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# 定义严格的模型
class StrictPerson(BaseModel):
    """严格验证的人物模型"""
    name: str = Field(description="姓名", min_length=2, max_length=50)
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱", pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

parser = PydanticOutputParser(pydantic_object=StrictPerson)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = llm | parser

# 测试用例
test_cases = [
    ('{"name": "A", "age": 30, "email": "a@b.com"}', "姓名太短"),
    ('{"name": "Alice", "age": 200, "email": "alice@example.com"}', "年龄超出范围"),
    ('{"name": "Bob", "age": 25, "email": "invalid-email"}', "邮箱格式错误"),
    ('{"name": "Charlie", "age": 30, "email": "charlie@example.com"}', "正常数据"),
]

print("=" * 60)
print("错误处理测试")
print("=" * 60)

for json_str, description in test_cases:
    print(f"\n测试: {description}")
    print(f"输入: {json_str}")

    try:
        result = parser.parse(json_str)
        print(f"✓ 验证成功: {result.name}, {result.age}, {result.email}")
    except OutputParserException as e:
        print(f"✗ 解析失败")
        print(f"  错误: {str(e)[:100]}...")
    except ValidationError as e:
        print(f"✗ 验证失败")
        print(f"  错误: {e.errors()[0]['msg']}")

"""
预期输出：
============================================================
错误处理测试
============================================================

测试: 姓名太短
输入: {"name": "A", "age": 30, "email": "a@b.com"}
✗ 验证失败
  错误: String should have at least 2 characters

测试: 年龄超出范围
输入: {"name": "Alice", "age": 200, "email": "alice@example.com"}
✗ 验证失败
  错误: Input should be less than or equal to 150

测试: 邮箱格式错误
输入: {"name": "Bob", "age": 25, "email": "invalid-email"}
✗ 验证失败
  错误: String should match pattern '^[\w\.-]+@[\w\.-]+\.\w+$'

测试: 正常数据
输入: {"name": "Charlie", "age": 30, "email": "charlie@example.com"}
✓ 验证成功: Charlie, 30, charlie@example.com
"""
```

---

## 常见错误与解决方案

### 错误 1：ValidationError

**原因**：数据不符合 Pydantic 模型约束

**解决方案**：
```python
# 1. 使用更宽松的约束
class Person(BaseModel):
    age: int = Field(ge=0)  # 只验证非负

# 2. 使用 Optional 字段
from typing import Optional
class Person(BaseModel):
    email: Optional[str] = None  # 可选字段

# 3. 自定义验证器
from pydantic import field_validator
class Person(BaseModel):
    age: int

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('年龄必须在 0-150 之间')
        return v
```

### 错误 2：格式指令未生效

**原因**：LLM 未遵循格式指令

**解决方案**：
```python
# 1. 在 Prompt 中强调格式要求
prompt = PromptTemplate(
    template="""严格按照以下格式返回 JSON：

{format_instructions}

输入：{text}

重要：必须返回有效的 JSON 格式！
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 2. 使用更强大的模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### 错误 3：ORM 模式不工作

**原因**：未启用 `from_attributes`

**解决方案**：
```python
# 确保配置正确
class UserModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # 必须设置

    id: int
    username: str
```

---

## 最佳实践

### 1. 模型设计

```python
# ✓ 好的设计
class Person(BaseModel):
    """清晰的文档字符串"""
    name: str = Field(description="姓名", min_length=1)
    age: int = Field(description="年龄", ge=0, le=150)
    email: Optional[str] = Field(default=None, description="邮箱")

# ✗ 不好的设计
class Person(BaseModel):
    name: str  # 缺少描述和约束
    age: int
    email: str  # 应该是可选的
```

### 2. 错误处理

```python
# 始终捕获异常
try:
    result = chain.invoke(inputs)
except OutputParserException as e:
    logger.error(f"解析失败: {e.llm_output}")
except ValidationError as e:
    logger.error(f"验证失败: {e.errors()}")
```

### 3. 性能优化

```python
# 使用更便宜的模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 批量处理
results = chain.batch(inputs)
```

---

## 总结

PydanticOutputParser 的核心优势：
1. **类型安全**：自动验证和转换
2. **格式指令**：自动生成 Prompt 指令
3. **版本兼容**：支持 Pydantic v1 和 v2
4. **ORM 集成**：支持从对象属性提取

**何时使用**：
- ✓ 需要严格的类型验证
- ✓ 需要复杂的嵌套结构
- ✓ 需要自动生成格式指令
- ✓ 需要与 ORM 集成

**何时不使用**：
- ✗ Schema 不固定（使用 JsonOutputParser）
- ✗ 模型支持原生结构化输出（使用 with_structured_output()）
- ✗ 不需要验证（使用 StrOutputParser）
