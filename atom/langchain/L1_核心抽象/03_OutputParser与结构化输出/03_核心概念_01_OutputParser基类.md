# 核心概念 1：OutputParser 基类

## 什么是 OutputParser 基类？

**OutputParser 基类是 LangChain 中所有输出解析器的抽象协议，定义了将 LLM 字符串输出转换为结构化对象的标准接口。**

---

## 1. 协议定义

### 1.1 核心方法

OutputParser 基于 `Runnable` 协议，必须实现以下方法：

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import TypeVar, Generic

T = TypeVar('T')

class BaseOutputParser(Runnable[str, T], Generic[T]):
    """输出解析器基类"""

    def parse(self, text: str) -> T:
        """
        解析 LLM 输出文本

        Args:
            text: LLM 返回的字符串

        Returns:
            解析后的结构化对象

        Raises:
            OutputParserException: 解析失败时抛出
        """
        raise NotImplementedError

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> T:
        """
        带 Prompt 上下文的解析（用于错误修复）

        Args:
            completion: LLM 返回的字符串
            prompt: 原始 Prompt

        Returns:
            解析后的结构化对象
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """
        获取格式指令（注入到 Prompt 中）

        Returns:
            格式指令字符串
        """
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """解析器类型标识"""
        raise NotImplementedError
```

### 1.2 Runnable 协议集成

OutputParser 继承自 `Runnable`，自动获得以下方法：

```python
# invoke: 单次调用
result = parser.invoke("LLM output text")

# batch: 批量调用
results = parser.batch(["text1", "text2", "text3"])

# stream: 流式调用（如果支持）
for chunk in parser.stream("LLM output text"):
    print(chunk)

# ainvoke: 异步单次调用
result = await parser.ainvoke("LLM output text")

# abatch: 异步批量调用
results = await parser.abatch(["text1", "text2", "text3"])
```

---

## 2. 手写实现：理解内部机制

### 2.1 最简单的 OutputParser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict
import json

class SimpleJsonParser(BaseOutputParser[Dict]):
    """最简单的 JSON 解析器"""

    def parse(self, text: str) -> Dict:
        """解析 JSON 字符串"""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON: {e}")

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return "请以 JSON 格式返回结果"

    @property
    def _type(self) -> str:
        return "simple_json"

# 使用示例
parser = SimpleJsonParser()
result = parser.parse('{"name": "Alice", "age": 25}')
print(result)  # {'name': 'Alice', 'age': 25}
```

### 2.2 带验证的 OutputParser

```python
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int

class ValidatingJsonParser(BaseOutputParser[Person]):
    """带 Pydantic 验证的解析器"""

    def __init__(self, pydantic_object: type[BaseModel]):
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> Person:
        """解析并验证"""
        try:
            # 1. 解析 JSON
            data = json.loads(text)

            # 2. Pydantic 验证
            return self.pydantic_object(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}")
        except ValidationError as e:
            raise ValueError(f"数据验证失败: {e}")

    def get_format_instructions(self) -> str:
        """生成格式指令"""
        schema = self.pydantic_object.schema()
        return f"请返回符合以下 schema 的 JSON:\n{json.dumps(schema, indent=2)}"

    @property
    def _type(self) -> str:
        return "validating_json"

# 使用示例
parser = ValidatingJsonParser(pydantic_object=Person)
result = parser.parse('{"name": "Alice", "age": 25}')
print(result)  # Person(name='Alice', age=25)
print(type(result))  # <class '__main__.Person'>
```

### 2.3 带错误处理的 OutputParser

```python
class RobustJsonParser(BaseOutputParser[Dict]):
    """容错的 JSON 解析器"""

    def parse(self, text: str) -> Dict:
        """尝试多种方式解析"""
        # 策略 1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略 2：提取 JSON 代码块
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 策略 3：查找第一个 { 到最后一个 }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        # 所有策略都失败
        raise ValueError(f"无法从文本中提取 JSON: {text[:100]}...")

    def get_format_instructions(self) -> str:
        return "请返回 JSON 格式的结果（可以用 ```json 代码块包裹）"

    @property
    def _type(self) -> str:
        return "robust_json"

# 使用示例
parser = RobustJsonParser()

# 测试不同格式
texts = [
    '{"name": "Alice"}',  # 纯 JSON
    '```json\n{"name": "Bob"}\n```',  # 代码块
    'Here is the result: {"name": "Charlie"} as requested',  # 混合文本
]

for text in texts:
    result = parser.parse(text)
    print(result)
```

---

## 3. 官方 OutputParser 使用

### 3.1 PydanticOutputParser（官方实现）

```python
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")

# 创建解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 获取格式指令
format_instructions = parser.get_format_instructions()
print(format_instructions)
# 输出:
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema...

# 创建 Prompt（注入格式指令）
prompt = PromptTemplate(
    template="提取人物信息：{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

# 构建链
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | parser

# 调用
result = chain.invoke({"text": "Alice is 25 years old, email: alice@example.com"})
print(result)
# Person(name='Alice', age=25, email='alice@example.com')
```

### 3.2 JsonOutputParser（官方实现）

```python
from langchain.output_parsers import JsonOutputParser

# 创建解析器（无 schema）
parser = JsonOutputParser()

# 使用
chain = prompt | llm | parser
result = chain.invoke({"text": "Extract person info: Bob, 30"})
print(result)  # {'name': 'Bob', 'age': 30}
print(type(result))  # <class 'dict'>
```

---

## 4. 2025-2026 更新：get_output_schema() 方法

### 4.1 新增方法

LangChain 1.0+ 新增了 `get_output_schema()` 方法：

```python
from langchain_core.output_parsers import BaseOutputParser

class ModernOutputParser(BaseOutputParser[Person]):
    """2025+ 标准解析器"""

    def get_output_schema(self) -> dict:
        """
        返回输出 schema（用于模型原生工具调用）

        Returns:
            JSON Schema 字典
        """
        return self.pydantic_object.schema()

    # ... 其他方法
```

### 4.2 与 with_structured_output() 的关系

```python
# 传统方式：手动注入格式指令
parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()
prompt = PromptTemplate(
    template="Extract: {text}\n\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions}
)
chain = prompt | llm | parser

# 现代方式：利用 get_output_schema()
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)
# 内部调用 get_output_schema() 获取 schema
# 使用模型原生工具调用能力
result = structured_llm.invoke("Extract: Alice, 25")
```

---

## 5. 子类化 vs 使用内置解析器

### 5.1 何时子类化 OutputParser？

**适合子类化的场景**：

1. **特殊格式解析**：LLM 返回非标准格式
   ```python
   class CustomFormatParser(BaseOutputParser[Dict]):
       """解析自定义格式：name=Alice|age=25"""

       def parse(self, text: str) -> Dict:
           pairs = text.split('|')
           result = {}
           for pair in pairs:
               key, value = pair.split('=')
               result[key] = value
           return result
   ```

2. **复杂后处理逻辑**：需要额外的数据转换
   ```python
   class DateNormalizingParser(BaseOutputParser[Dict]):
       """解析并标准化日期格式"""

       def parse(self, text: str) -> Dict:
           data = json.loads(text)
           # 标准化日期格式
           if 'date' in data:
               data['date'] = self._normalize_date(data['date'])
           return data

       def _normalize_date(self, date_str: str) -> str:
           # 复杂的日期解析逻辑
           pass
   ```

3. **特定领域需求**：行业特定的验证规则
   ```python
   class MedicalRecordParser(BaseOutputParser[MedicalRecord]):
       """医疗记录解析器（带领域验证）"""

       def parse(self, text: str) -> MedicalRecord:
           data = json.loads(text)
           record = MedicalRecord(**data)
           # 医疗领域特定验证
           self._validate_medical_codes(record)
           return record
   ```

**不适合子类化的场景**：

1. **标准 JSON + Pydantic**：直接用 `PydanticOutputParser`
2. **简单 JSON**：直接用 `JsonOutputParser`
3. **2025+ 模型**：直接用 `with_structured_output()`

### 5.2 对比表

| 方案 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **子类化 OutputParser** | 特殊格式、复杂逻辑 | 完全控制、灵活 | 需要编写代码 |
| **PydanticOutputParser** | 标准 JSON + 验证 | 开箱即用、类型安全 | 需要格式指令 |
| **JsonOutputParser** | 简单 JSON | 简单快速 | 无验证 |
| **with_structured_output** | 2025+ 模型 | 最可靠、最简单 | 需要模型支持 |

---

## 6. 在实际应用中的使用

### 6.1 RAG 元数据提取

```python
from pydantic import BaseModel
from typing import List

class DocumentMetadata(BaseModel):
    title: str
    author: str
    publish_date: str
    keywords: List[str]
    summary: str

# 使用 OutputParser 提取元数据
parser = PydanticOutputParser(pydantic_object=DocumentMetadata)
prompt = PromptTemplate(
    template="从以下文档中提取元数据：\n\n{document}\n\n{format_instructions}",
    input_variables=["document"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

# 批量处理文档
documents = [doc1, doc2, doc3]
metadatas = chain.batch([{"document": doc} for doc in documents])

# 存入向量数据库
for doc, metadata in zip(documents, metadatas):
    vector_store.add_documents(
        documents=[doc],
        metadatas=[metadata.dict()]
    )
```

### 6.2 Agent 工具响应格式化

```python
class ToolResponse(BaseModel):
    success: bool
    data: Dict
    error: Optional[str] = None

# Agent 工具返回标准化响应
parser = PydanticOutputParser(pydantic_object=ToolResponse)

def format_tool_response(raw_response: str) -> ToolResponse:
    """格式化工具响应"""
    return parser.parse(raw_response)

# 在 Agent 中使用
tool_output = call_external_api()
formatted = format_tool_response(tool_output)

if formatted.success:
    agent.process_data(formatted.data)
else:
    agent.handle_error(formatted.error)
```

### 6.3 批量数据处理

```python
# 批量提取结构化信息
texts = [
    "Alice, 25, alice@example.com",
    "Bob, 30, bob@example.com",
    "Charlie, 35, charlie@example.com"
]

parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | llm | parser

# 并行批量处理
results = chain.batch([{"text": text} for text in texts])

# 保存到数据库
for person in results:
    database.save(person)
```

---

## 7. 最佳实践

### 7.1 错误处理

```python
from pydantic import ValidationError
from langchain.schema import OutputParserException

def safe_parse(parser: BaseOutputParser, text: str) -> Optional[T]:
    """安全解析（带错误处理）"""
    try:
        return parser.parse(text)
    except OutputParserException as e:
        logger.error(f"解析失败: {e}")
        return None
    except ValidationError as e:
        logger.error(f"验证失败: {e}")
        return None
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return None
```

### 7.2 性能优化

```python
# 批量处理时复用 parser 实例
parser = PydanticOutputParser(pydantic_object=Person)

# ✅ 好：复用 parser
results = [parser.parse(text) for text in texts]

# ❌ 差：每次创建新 parser
results = [PydanticOutputParser(pydantic_object=Person).parse(text) for text in texts]
```

### 7.3 类型提示

```python
from typing import TypeVar, Generic

T = TypeVar('T')

def create_parser(model: type[T]) -> BaseOutputParser[T]:
    """创建类型化的解析器"""
    return PydanticOutputParser(pydantic_object=model)

# 使用时有类型提示
parser = create_parser(Person)
result = parser.parse(text)  # IDE 知道 result 是 Person 类型
```

---

## 8. 总结

### 核心要点

1. **OutputParser 是协议**：定义了标准接口，所有解析器都遵循
2. **基于 Runnable**：自动获得 invoke/batch/stream 等方法
3. **三个核心方法**：parse()、get_format_instructions()、get_output_schema()
4. **子类化场景**：特殊格式、复杂逻辑、领域特定需求
5. **2025+ 趋势**：优先用 with_structured_output()，OutputParser 作为兼容方案

### 何时使用

- ✅ **需要兼容旧模型**：使用 PydanticOutputParser
- ✅ **特殊格式解析**：子类化 BaseOutputParser
- ✅ **简单 JSON**：使用 JsonOutputParser
- ❌ **2025+ 模型**：直接用 with_structured_output()

---

**记住**：OutputParser 基类定义了"字符串 → 对象"的标准协议，理解这个协议是掌握所有解析器的基础。
