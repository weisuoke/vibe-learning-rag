# 核心概念 3：PydanticOutputParser（传统方法）

## 什么是 PydanticOutputParser？

**PydanticOutputParser 是 LangChain 的传统结构化输出解析器，通过在 Prompt 中注入格式指令，引导 LLM 返回符合 Pydantic 模型的 JSON，然后在应用层解析和验证。**

2025-2026 年主要用作**兼容方案**，当模型不支持原生结构化输出时使用。

---

## 1. 工作原理

### 1.1 三步流程

```
1. 生成格式指令（get_format_instructions）
   ↓
2. 注入到 Prompt 中
   ↓
3. LLM 返回 JSON 字符串
   ↓
4. 解析并验证（parse）
```

### 1.2 代码示例

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄", ge=0, le=150)

# 1. 创建解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 2. 获取格式指令
format_instructions = parser.get_format_instructions()
print(format_instructions)
# 输出：The output should be formatted as a JSON instance...

# 3. 注入到 Prompt
prompt = PromptTemplate(
    template="提取人物信息：{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

# 4. 构建链
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | parser

# 5. 调用
result = chain.invoke({"text": "Alice is 25 years old"})
print(result)  # Person(name='Alice', age=25)
```

---

## 2. Pydantic V2 集成（2025 标准）

### 2.1 字段定义

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Product(BaseModel):
    # 基础字段
    name: str = Field(description="产品名称")
    price: float = Field(description="价格", gt=0)

    # 可选字段
    description: Optional[str] = Field(None, description="产品描述")

    # 列表字段
    tags: List[str] = Field(default_factory=list, description="标签")

    # 约束字段
    stock: int = Field(description="库存", ge=0)
    rating: float = Field(description="评分", ge=0, le=5)
```

### 2.2 嵌套模型

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address  # 嵌套模型
    employees: List[Person]  # 列表嵌套
```

### 2.3 验证器

```python
from pydantic import BaseModel, Field, field_validator

class Person(BaseModel):
    name: str
    email: str
    age: int

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('无效的邮箱地址')
        return v

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('年龄必须在 0-150 之间')
        return v
```

---

## 3. 格式指令生成

### 3.1 默认格式指令

```python
parser = PydanticOutputParser(pydantic_object=Person)
instructions = parser.get_format_instructions()

# 输出示例：
"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"name": {"title": "Name", "description": "人名", "type": "string"}, "age": {"title": "Age", "description": "年龄", "type": "integer", "minimum": 0, "maximum": 150}}, "required": ["name", "age"]}
```
"""
```

### 3.2 自定义格式指令

```python
class CustomParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        """自定义格式指令"""
        schema = self.pydantic_object.schema()
        return f"""
请严格按照以下 JSON 格式返回结果：

{json.dumps(schema, indent=2, ensure_ascii=False)}

注意：
1. 必须是有效的 JSON
2. 所有必填字段都要包含
3. 字段类型必须正确
"""
```

---

## 4. 错误处理

### 4.1 常见错误类型

```python
from pydantic import ValidationError
from langchain.schema import OutputParserException

# 错误 1：JSON 解析失败
try:
    result = parser.parse("This is not JSON")
except OutputParserException as e:
    print(f"JSON 解析失败: {e}")

# 错误 2：字段缺失
try:
    result = parser.parse('{"name": "Alice"}')  # 缺少 age
except ValidationError as e:
    print(f"字段缺失: {e}")

# 错误 3：类型错误
try:
    result = parser.parse('{"name": "Alice", "age": "twenty"}')
except ValidationError as e:
    print(f"类型错误: {e}")

# 错误 4：约束违反
try:
    result = parser.parse('{"name": "Alice", "age": -5}')
except ValidationError as e:
    print(f"约束违反: {e}")
```

### 4.2 容错策略

```python
def safe_parse(parser, text, default=None):
    """安全解析（带默认值）"""
    try:
        return parser.parse(text)
    except Exception as e:
        logger.error(f"解析失败: {e}")
        return default

# 使用
result = safe_parse(parser, llm_output, default=None)
if result:
    process(result)
```

---

## 5. 与 with_structured_output() 对比

### 5.1 何时使用 PydanticOutputParser

✅ **模型不支持原生结构化输出**
- GPT-3.5-turbo（旧版本）
- 开源模型（Llama 2, Mistral）
- 自部署模型

✅ **需要兼容旧代码**
- 已有 PydanticOutputParser 实现
- 不想重构现有系统

✅ **特殊格式需求**
- 需要自定义格式指令
- 需要特殊的后处理逻辑

### 5.2 何时使用 with_structured_output()

✅ **模型支持原生结构化输出**（OpenAI GPT-4o、Anthropic Claude 3.5+）
✅ **新项目**（2025+ 推荐）
✅ **需要高可靠性**（生产环境）
✅ **成本敏感**（减少 token 消耗）

---

## 6. 实际应用示例

### 6.1 批量数据提取

```python
from typing import List

class Contact(BaseModel):
    name: str
    phone: str
    email: str

parser = PydanticOutputParser(pydantic_object=Contact)
prompt = PromptTemplate(
    template="提取联系信息：{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

# 批量处理
texts = [
    "张三，电话 13800138000，邮箱 zhangsan@example.com",
    "李四，电话 13900139000，邮箱 lisi@example.com"
]

contacts = chain.batch([{"text": t} for t in texts])
for contact in contacts:
    print(contact)
```

### 6.2 复杂嵌套结构

```python
class Employee(BaseModel):
    name: str
    position: str
    salary: float

class Department(BaseModel):
    name: str
    manager: Employee
    employees: List[Employee]

parser = PydanticOutputParser(pydantic_object=Department)
# ... 使用 parser 提取部门信息
```

---

## 7. 性能优化

### 7.1 缓存格式指令

```python
# ❌ 差：每次都生成格式指令
for text in texts:
    parser = PydanticOutputParser(pydantic_object=Person)
    instructions = parser.get_format_instructions()
    # ...

# ✅ 好：复用格式指令
parser = PydanticOutputParser(pydantic_object=Person)
instructions = parser.get_format_instructions()
for text in texts:
    # 使用缓存的 instructions
    pass
```

### 7.2 批量处理

```python
# 使用 batch 而不是循环
results = chain.batch([{"text": t} for t in texts])
```

---

## 8. 迁移到 with_structured_output()

### 8.1 迁移步骤

```python
# 旧代码（PydanticOutputParser）
parser = PydanticOutputParser(pydantic_object=Person)
prompt = PromptTemplate(
    template="Extract: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | llm | parser

# 新代码（with_structured_output）
llm = ChatOpenAI(model="gpt-4o")  # 确保模型支持
structured_llm = llm.with_structured_output(Person)
prompt = PromptTemplate(
    template="Extract: {text}",  # 移除格式指令
    input_variables=["text"]
)
chain = prompt | structured_llm
```

### 8.2 兼容性检查

```python
def create_structured_chain(llm, model_class):
    """创建结构化链（自动选择方法）"""
    if hasattr(llm, 'with_structured_output'):
        # 尝试使用现代方法
        try:
            return llm.with_structured_output(model_class)
        except:
            pass

    # 降级到传统方法
    parser = PydanticOutputParser(pydantic_object=model_class)
    prompt = PromptTemplate(
        template="{input}\n\n{format_instructions}",
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser
```

---

## 9. 总结

### 核心要点

1. **传统方法**：通过 Prompt 注入格式指令
2. **Pydantic V2**：2025 标准，支持丰富的验证
3. **兼容方案**：用于不支持原生结构化输出的模型
4. **错误处理**：需要显式捕获 ValidationError
5. **迁移路径**：优先迁移到 with_structured_output()

### 何时使用

- ✅ 模型不支持原生结构化输出
- ✅ 需要兼容旧代码
- ❌ 新项目（优先用 with_structured_output）

---

**记住**：PydanticOutputParser 是可靠的兼容方案，但 2025+ 新项目应优先使用 with_structured_output()。
