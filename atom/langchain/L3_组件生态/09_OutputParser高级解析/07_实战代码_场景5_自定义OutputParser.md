# 实战代码 - 场景5：自定义 OutputParser 开发

> **数据来源**：
> - reference/source_outputparser_01_基础架构.md（源码分析）
> - reference/search_outputparser_01_Reddit最佳实践.md（社区实践）

---

## 场景概述

当内置的 OutputParser 无法满足需求时，可以通过继承 `BaseOutputParser` 创建自定义解析器。适用于：
- 特殊格式的输出（如自定义标记语言）
- 复杂的验证逻辑
- 多步骤解析流程
- 与特定 LLM 的输出格式适配

---

## 示例1：基础自定义解析器 - 键值对解析器

**场景**：解析 LLM 输出的键值对格式（`key: value`）

```python
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class KeyValueParser(BaseOutputParser[dict]):
    """解析键值对格式的输出

    输入格式：
    name: John
    age: 30
    city: San Francisco

    输出：{"name": "John", "age": "30", "city": "San Francisco"}
    """

    def parse(self, text: str) -> dict:
        """解析文本为字典

        Args:
            text: 待解析的文本

        Returns:
            解析后的字典

        Raises:
            OutputParserException: 解析失败时抛出
        """
        result = {}
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否包含冒号
            if ':' not in line:
                raise OutputParserException(
                    f"Invalid line format: {line}. Expected 'key: value'"
                )

            # 分割键值对
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()

        return result

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return (
            "Please format your response as key-value pairs, "
            "one per line, in the format:\n"
            "key: value\n"
            "Example:\n"
            "name: John\n"
            "age: 30"
        )

# 使用示例
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 初始化
model = ChatOpenAI(model="gpt-4o-mini")
parser = KeyValueParser()

# 创建 Prompt
prompt = PromptTemplate(
    template="""Extract the following information about the person:
{format_instructions}

Text: {text}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 构建链
chain = prompt | model | parser

# 执行
result = chain.invoke({
    "text": "John is 30 years old and lives in San Francisco"
})

print(result)
# 输出: {'name': 'John', 'age': '30', 'city': 'San Francisco'}
```

**关键点**：
- 继承 `BaseOutputParser[dict]`，泛型指定返回类型
- 实现 `parse()` 方法处理核心解析逻辑
- 实现 `get_format_instructions()` 提供格式指令
- 使用 `OutputParserException` 处理解析错误

---

## 示例2：带验证的解析器 - 邮箱提取器

**场景**：提取并验证邮箱地址

```python
import re
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class EmailParser(BaseOutputParser[List[str]]):
    """提取并验证邮箱地址"""

    # 邮箱正则表达式
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def parse(self, text: str) -> List[str]:
        """提取所有有效的邮箱地址

        Args:
            text: 待解析的文本

        Returns:
            邮箱地址列表

        Raises:
            OutputParserException: 未找到邮箱时抛出
        """
        # 提取所有匹配的邮箱
        emails = re.findall(self.EMAIL_PATTERN, text)

        if not emails:
            raise OutputParserException(
                f"No valid email addresses found in: {text}"
            )

        # 去重并排序
        return sorted(set(emails))

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return (
            "Please provide email addresses in standard format.\n"
            "Example: john@example.com, jane@company.org"
        )

# 使用示例
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
parser = EmailParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract all email addresses from the text."),
    ("system", "{format_instructions}"),
    ("user", "{text}")
])

chain = prompt | model | parser

# 测试
result = chain.invoke({
    "text": "Contact John at john@example.com or Jane at jane@company.org",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# 输出: ['jane@company.org', 'john@example.com']
```

**关键点**：
- 使用正则表达式进行格式验证
- 返回 `List[str]` 类型
- 自动去重和排序
- 未找到邮箱时抛出异常

---

## 示例3：流式解析器 - 累积 JSON 解析器

**场景**：支持流式输出的 JSON 解析器

```python
from typing import Optional
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.utils.json import parse_partial_json

class StreamingJsonParser(BaseCumulativeTransformOutputParser[dict]):
    """支持流式解析的 JSON 解析器"""

    def parse_result(self, result: list, *, partial: bool = False) -> dict:
        """解析生成结果

        Args:
            result: 生成结果列表
            partial: 是否为部分结果

        Returns:
            解析后的字典
        """
        # 获取文本内容
        text = result[0].text if result else ""

        # 移除 Markdown 代码块标记
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 部分解析
        if partial:
            return parse_partial_json(text)

        # 完整解析
        import json
        return json.loads(text)

    def parse(self, text: str) -> dict:
        """解析文本（非流式）"""
        import json

        # 移除 Markdown 代码块
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return "Please respond with valid JSON format."

# 使用示例 - 流式解析
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
parser = StreamingJsonParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract information as JSON."),
    ("user", "{text}")
])

chain = prompt | model | parser

# 流式输出
print("Streaming output:")
for chunk in chain.stream({"text": "John is 30 years old"}):
    print(chunk)
    # 输出: 逐步累积的 JSON 对象

# 完整输出
result = chain.invoke({"text": "John is 30 years old"})
print("\nFinal result:", result)
# 输出: {'name': 'John', 'age': 30}
```

**关键点**：
- 继承 `BaseCumulativeTransformOutputParser` 支持流式
- 实现 `parse_result()` 方法处理部分结果
- 使用 `parse_partial_json()` 解析不完整的 JSON
- 同时支持流式和非流式调用

---

## 示例4：多步骤解析器 - 结构化文本解析器

**场景**：解析包含标题、段落和列表的结构化文本

```python
from typing import Dict, List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class StructuredTextParser(BaseOutputParser[Dict[str, any]]):
    """解析结构化文本（标题 + 段落 + 列表）"""

    def parse(self, text: str) -> Dict[str, any]:
        """解析结构化文本

        返回格式：
        {
            "title": str,
            "paragraphs": List[str],
            "items": List[str]
        }
        """
        result = {
            "title": "",
            "paragraphs": [],
            "items": []
        }

        lines = text.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题（以 # 开头）
            if line.startswith('# '):
                result["title"] = line[2:].strip()
                current_section = "title"

            # 检测列表项（以 - 或 * 开头）
            elif line.startswith(('- ', '* ')):
                result["items"].append(line[2:].strip())
                current_section = "items"

            # 普通段落
            else:
                if current_section != "title":
                    result["paragraphs"].append(line)
                    current_section = "paragraphs"

        # 验证必须有标题
        if not result["title"]:
            raise OutputParserException(
                "No title found. Text must start with '# Title'"
            )

        return result

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return """Please format your response as:
# Title
Paragraph text here.
- List item 1
- List item 2"""

# 使用示例
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
parser = StructuredTextParser()

prompt = PromptTemplate(
    template="""Summarize the following text in structured format:
{format_instructions}

Text: {text}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({
    "text": "Python is a programming language. It's easy to learn and powerful."
})

print(result)
# 输出: {
#     "title": "Python Programming",
#     "paragraphs": ["Python is easy to learn and powerful."],
#     "items": ["Easy to learn", "Powerful", "Versatile"]
# }
```

**关键点**：
- 多步骤解析逻辑（标题、段落、列表）
- 状态跟踪（`current_section`）
- 验证必需字段
- 返回复杂的嵌套结构

---

## 示例5：生产级解析器 - 带重试和日志的解析器

**场景**：生产环境中需要错误处理和日志记录

```python
import logging
from typing import Optional
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionParser(BaseOutputParser[dict]):
    """生产级解析器 - 带重试和日志"""

    max_retries: int = 3

    def parse(self, text: str) -> dict:
        """解析文本，带重试机制"""
        import json

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Parsing attempt {attempt + 1}/{self.max_retries}")

                # 清理文本
                cleaned_text = self._clean_text(text)

                # 解析 JSON
                result = json.loads(cleaned_text)

                # 验证结果
                self._validate_result(result)

                logger.info("Parsing successful")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise OutputParserException(
                        f"Failed to parse JSON after {self.max_retries} attempts: {e}"
                    )

            except ValueError as e:
                logger.error(f"Validation error: {e}")
                raise OutputParserException(f"Validation failed: {e}")

        raise OutputParserException("Unexpected error in parsing")

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        text = text.strip()

        # 移除 Markdown 代码块
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _validate_result(self, result: dict) -> None:
        """验证结果"""
        required_keys = ["name", "value"]

        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")

    def get_format_instructions(self) -> str:
        """返回格式指令"""
        return """Please respond with JSON containing:
{
    "name": "string",
    "value": "string"
}"""

# 使用示例
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
parser = ProductionParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract information as JSON."),
    ("system", "{format_instructions}"),
    ("user", "{text}")
])

chain = prompt | model | parser

try:
    result = chain.invoke({
        "text": "The product name is iPhone and its value is $999",
        "format_instructions": parser.get_format_instructions()
    })
    print("Success:", result)
except OutputParserException as e:
    print("Error:", e)
```

**关键点**：
- 重试机制（`max_retries`）
- 日志记录（`logging`）
- 文本清理（`_clean_text`）
- 结果验证（`_validate_result`）
- 异常处理和错误信息

---

## 常见错误与解决方案

### 错误1：忘记实现 `get_format_instructions()`

```python
# ❌ 错误
class MyParser(BaseOutputParser[dict]):
    def parse(self, text: str) -> dict:
        return {"result": text}
    # 缺少 get_format_instructions()

# ✅ 正确
class MyParser(BaseOutputParser[dict]):
    def parse(self, text: str) -> dict:
        return {"result": text}

    def get_format_instructions(self) -> str:
        return "Please respond with text."
```

### 错误2：未指定泛型类型

```python
# ❌ 错误
class MyParser(BaseOutputParser):  # 缺少泛型
    pass

# ✅ 正确
class MyParser(BaseOutputParser[dict]):  # 指定返回类型
    pass
```

### 错误3：未处理异常

```python
# ❌ 错误
def parse(self, text: str) -> dict:
    return json.loads(text)  # 可能抛出 JSONDecodeError

# ✅ 正确
def parse(self, text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Invalid JSON: {e}")
```

---

## 最佳实践

1. **明确返回类型**：使用泛型指定 `BaseOutputParser[T]`
2. **提供格式指令**：实现 `get_format_instructions()` 帮助 LLM 生成正确格式
3. **错误处理**：使用 `OutputParserException` 统一异常类型
4. **文本清理**：处理 Markdown 代码块等常见格式
5. **验证结果**：确保解析结果符合预期结构
6. **日志记录**：生产环境中记录解析过程
7. **重试机制**：处理临时性解析失败

---

## 何时使用自定义 OutputParser

**适用场景**：
- 特殊格式的输出（非 JSON/XML）
- 复杂的验证逻辑
- 多步骤解析流程
- 需要流式支持
- 与特定 LLM 的输出格式适配

**不适用场景**：
- 标准 JSON 格式（使用 `JsonOutputParser`）
- Pydantic 模型验证（使用 `PydanticOutputParser`）
- 简单字符串输出（使用 `StrOutputParser`）

---

**数据来源总结**：
- 基础架构设计：源码分析（source_outputparser_01_基础架构.md）
- 最佳实践：Reddit 社区讨论（search_outputparser_01_Reddit最佳实践.md）
- 错误处理：社区共识和生产经验
