# 核心概念 2：JSON 解析器（JsonOutputParser & SimpleJsonOutputParser）

> 本文档详细讲解 LangChain 中的 JSON 解析器，包括 JsonOutputParser 和 SimpleJsonOutputParser 的原理、实现和最佳实践。

---

## 文档元数据

**知识点**：OutputParser 高级解析 - JSON 解析器
**层级**：L3_组件生态
**依赖知识**：Runnable 接口、LCEL 表达式、Pydantic 基础
**预计学习时间**：30 分钟
**难度等级**：⭐⭐⭐

---

## 一、JSON 解析器概述

### 1.1 什么是 JSON 解析器？

JSON 解析器是 LangChain 中用于将 LLM 输出的文本解析为 JSON 对象的工具。它是最可靠的结构化输出解析器之一，特别适用于不支持原生 Function Calling 的模型。

**核心价值**：
- 将非结构化的 LLM 文本输出转换为结构化的 JSON 数据
- 支持流式解析（逐步构建 JSON 对象）
- 自动处理 Markdown 代码块
- 支持 JSON Patch diff 模式（减少数据传输）

### 1.2 两种 JSON 解析器对比

| 特性 | JsonOutputParser | SimpleJsonOutputParser |
|------|------------------|------------------------|
| 继承关系 | BaseCumulativeTransformOutputParser | BaseOutputParser |
| 流式支持 | ✅ 完整支持 | ❌ 不支持 |
| JSON Patch | ✅ 支持 diff 模式 | ❌ 不支持 |
| Pydantic 验证 | ✅ 可选 | ❌ 不支持 |
| 部分解析 | ✅ 支持 partial=True | ❌ 不支持 |
| Markdown 处理 | ✅ 自动提取 | ✅ 自动提取 |
| 使用场景 | 生产级应用、流式场景 | 简单场景、快速原型 |

**来源**：`reference/source_outputparser_02_JSON解析器.md`

---

## 二、JsonOutputParser 深度解析

### 2.1 类定义与继承关系

```python
class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    Probably the most reliable output parser for getting structured data that does *not*
    use function calling.

    When used in streaming mode, it will yield partial JSON objects containing all the
    keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields `JSONPatch` operations describing
    the difference between the previous and the current object.
    """

    pydantic_object: Annotated[type[TBaseModel] | None, SkipValidation()] = None
    """The Pydantic object to use for validation.

    If `None`, no validation is performed.
    """
```

**关键特性**：
1. **继承自 BaseCumulativeTransformOutputParser**：支持累积式流式解析
2. **可选的 Pydantic 验证**：通过 `pydantic_object` 参数启用类型验证
3. **JSON Patch 支持**：通过 `diff` 参数启用增量更新模式

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:31-42)

### 2.2 核心解析方法：parse_result()

```python
@override
def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
    """Parse the result of an LLM call to a JSON object.

    Args:
        result: The result of the LLM call.
        partial: Whether to parse partial JSON objects.

            If `True`, the output will be a JSON object containing all the keys that
            have been returned so far.

            If `False`, the output will be the full JSON object.

    Returns:
        The parsed JSON object.

    Raises:
        OutputParserException: If the output is not valid JSON.
    """
    text = result[0].text
    text = text.strip()
    if partial:
        try:
            return parse_json_markdown(text)
        except JSONDecodeError:
            return None
    else:
        try:
            return parse_json_markdown(text)
        except JSONDecodeError as e:
            msg = f"Invalid json output: {text}"
            raise OutputParserException(msg, llm_output=text) from e
```

**关键发现**：
1. **partial=True 模式**：返回 `None` 而不是抛出异常（用于流式解析）
2. **partial=False 模式**：抛出 `OutputParserException`（包含原始输出）
3. **使用 parse_json_markdown()**：自动从 Markdown 代码块中提取 JSON

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:61-91)

### 2.3 Markdown 代码块处理

JsonOutputParser 使用 `parse_json_markdown()` 工具函数，可以自动从 Markdown 代码块中提取 JSON：

**示例输入**：
```markdown
Here's the result:
```json
{"name": "Alice", "age": 30}
```
```

**解析结果**：
```python
{"name": "Alice", "age": 30}
```

**工具函数**：
```python
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)
```

**功能说明**：
- `parse_json_markdown()`：从 Markdown 代码块中提取 JSON
- `parse_partial_json()`：解析不完整的 JSON（用于流式）
- `parse_and_check_json_markdown()`：解析并验证 JSON

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:19-23, 116-138)

### 2.4 JSON Patch 支持（流式 diff 模式）

JsonOutputParser 支持 JSON Patch 模式，在流式场景下只返回增量变化：

```python
@override
def _diff(self, prev: Any | None, next: Any) -> Any:
    return jsonpatch.make_patch(prev, next).patch
```

**用途**：
- 在流式模式下，返回 JSON Patch 操作而不是完整对象
- 减少数据传输量
- 适用于大型 JSON 对象的增量更新

**依赖库**：
```python
import jsonpatch  # type: ignore[import-untyped]
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:51-52, 94-113)

---

## 三、流式解析机制

### 3.1 累积式流式解析

JsonOutputParser 继承自 `BaseCumulativeTransformOutputParser`，支持累积式流式解析：

**工作原理**：
1. LLM 逐块输出文本
2. 解析器累积所有已接收的文本
3. 尝试解析为 JSON（使用 `partial=True`）
4. 如果解析成功，返回部分 JSON 对象
5. 如果解析失败，返回 `None`（不抛出异常）

**示例**：
```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser

# 流式解析
for chunk in chain.stream("Return a JSON with name, age, and city"):
    print(chunk)
    # 输出：
    # {"name": "Alice"}
    # {"name": "Alice", "age": 30}
    # {"name": "Alice", "age": 30, "city": "New York"}
```

### 3.2 JSON Patch diff 模式

启用 `diff=True` 后，解析器返回 JSON Patch 操作：

```python
parser = JsonOutputParser(diff=True)

for patch in chain.stream("Return a large JSON"):
    print(patch)
    # 输出：
    # [{"op": "add", "path": "/name", "value": "Alice"}]
    # [{"op": "add", "path": "/age", "value": 30}]
    # [{"op": "add", "path": "/city", "value": "New York"}]
```

**JSON Patch 操作类型**：
- `add`：添加新字段
- `remove`：删除字段
- `replace`：替换字段值
- `move`：移动字段
- `copy`：复制字段
- `test`：测试字段值

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:94-113)

---

## 四、SimpleJsonOutputParser

### 4.1 简化版 JSON 解析器

SimpleJsonOutputParser 是 JsonOutputParser 的简化版本，不支持流式解析和 JSON Patch：

**特点**：
- 继承自 `BaseOutputParser`（不支持流式）
- 只支持基础的 JSON 解析
- 自动处理 Markdown 代码块
- 适用于简单场景和快速原型

**使用场景**：
- 简单的 JSON 解析需求
- 不需要流式输出
- 快速原型开发
- 学习和测试

### 4.2 基础用法

```python
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = SimpleJsonOutputParser()

chain = model | parser

result = chain.invoke("Return a JSON with name and age")
print(result)
# 输出: {"name": "Alice", "age": 30}
```

---

## 五、实战代码示例

### 5.1 基础 JSON 解析

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 初始化模型和解析器
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# 创建 Prompt
prompt = PromptTemplate(
    template="Extract the person's information from the text.\n{format_instructions}\n\nText: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 构建链
chain = prompt | model | parser

# 执行
result = chain.invoke({"text": "Alice is 30 years old and lives in New York."})
print(result)
# 输出: {"name": "Alice", "age": 30, "city": "New York"}
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:278-292)

### 5.2 流式 JSON 解析

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

chain = model | parser

# 流式解析
print("Streaming JSON parsing:")
for chunk in chain.stream("Return a JSON with name, age, city, and occupation"):
    print(chunk)
    # 输出：
    # {"name": "Alice"}
    # {"name": "Alice", "age": 30}
    # {"name": "Alice", "age": 30, "city": "New York"}
    # {"name": "Alice", "age": 30, "city": "New York", "occupation": "Engineer"}
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:318-324)

### 5.3 JSON Patch diff 模式

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser(diff=True)

chain = model | parser

# JSON Patch 模式
print("JSON Patch diff mode:")
for patch in chain.stream("Return a JSON with name, age, city, and occupation"):
    print(patch)
    # 输出：
    # [{"op": "add", "path": "/name", "value": "Alice"}]
    # [{"op": "add", "path": "/age", "value": 30}]
    # [{"op": "add", "path": "/city", "value": "New York"}]
    # [{"op": "add", "path": "/occupation", "value": "Engineer"}]
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:326-334)

### 5.4 处理 Markdown 代码块

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# Prompt 要求 LLM 返回 Markdown 代码块
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Always return JSON in a markdown code block."),
    ("user", "Extract person info from: {text}")
])

chain = prompt | model | parser

result = chain.invoke({"text": "Bob is 25 years old."})
print(result)
# 输出: {"name": "Bob", "age": 25}
# 即使 LLM 返回的是 Markdown 代码块，解析器也能正确提取 JSON
```

### 5.5 可选的 Pydantic 验证

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 定义 Pydantic 模型
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age", ge=0, le=150)
    city: str = Field(description="The person's city")

# 使用 Pydantic 验证
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser(pydantic_object=Person)

chain = model | parser

try:
    result = chain.invoke("Extract: Alice is 30 years old and lives in New York.")
    print(result)
    # 输出: {"name": "Alice", "age": 30, "city": "New York"}
except Exception as e:
    print(f"Validation error: {e}")
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:31-42)

---

## 六、错误处理

### 6.1 OutputParserException

当 JSON 解析失败时，JsonOutputParser 会抛出 `OutputParserException`：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

parser = JsonOutputParser()

try:
    result = parser.parse("This is not valid JSON")
except OutputParserException as e:
    print(f"Error: {e}")
    print(f"LLM output: {e.llm_output}")
    # 输出:
    # Error: Invalid json output: This is not valid JSON
    # LLM output: This is not valid JSON
```

**异常信息包含**：
- 错误消息（`msg`）
- 原始 LLM 输出（`llm_output`）

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:82-86)

### 6.2 部分解析模式（partial=True）

在流式场景下，使用 `partial=True` 避免抛出异常：

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# 部分解析模式
result = parser.parse_result([Generation(text='{"name": "Alice"')], partial=True)
print(result)
# 输出: None (不抛出异常)

# 完整解析模式
try:
    result = parser.parse_result([Generation(text='{"name": "Alice"')], partial=False)
except OutputParserException as e:
    print(f"Error: {e}")
    # 输出: Error: Invalid json output: {"name": "Alice"
```

**来源**：`reference/source_outputparser_02_JSON解析器.md` (json.py:76-86)

---

## 七、与 LCEL 的集成

### 7.1 管道操作符（|）

JsonOutputParser 实现了 Runnable 协议，可以使用 `|` 操作符与其他组件组合：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI()
parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template("Extract info: {text}")

# LCEL 链式组合
chain = prompt | model | parser

result = chain.invoke({"text": "Alice is 30 years old."})
print(result)
# 输出: {"name": "Alice", "age": 30}
```

### 7.2 批处理（batch）

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser

# 批处理
results = chain.batch([
    "Extract: Alice is 30 years old.",
    "Extract: Bob is 25 years old.",
])
print(results)
# 输出: [
#   {"name": "Alice", "age": 30},
#   {"name": "Bob", "age": 25}
# ]
```

### 7.3 异步支持（async）

```python
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser

async def main():
    result = await chain.ainvoke("Extract: Alice is 30 years old.")
    print(result)
    # 输出: {"name": "Alice", "age": 30}

asyncio.run(main())
```

---

## 八、最佳实践

### 8.1 何时使用 JsonOutputParser

**推荐场景**：
- ✅ 需要流式 JSON 解析
- ✅ 需要 JSON Patch diff 模式
- ✅ 需要可选的 Pydantic 验证
- ✅ 生产级应用
- ✅ 模型不支持原生 Function Calling

**不推荐场景**：
- ❌ 模型支持原生结构化输出（使用 `with_structured_output()` 更好）
- ❌ 需要严格的类型验证（使用 `PydanticOutputParser` 更好）
- ❌ 简单场景（使用 `SimpleJsonOutputParser` 更简单）

**来源**：`reference/context7_langchain_02_Pydantic模型.md` (163-181)

### 8.2 Prompt 工程技巧

**包含格式指令**：
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
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

### 8.3 模型选择建议

**推荐模型**：
- ✅ GPT-4 / GPT-4o：输出最稳定
- ✅ GPT-3.5-turbo：性价比高
- ✅ Claude 3.5 Sonnet：输出质量高

**需要注意的模型**：
- ⚠️ Llama 3.1：可能需要更多 Prompt 工程
- ⚠️ 开源模型：建议使用 OutputFixingParser 提高鲁棒性

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (119-125)

### 8.4 错误处理策略

**使用 OutputFixingParser**：
```python
from langchain_core.output_parsers import JsonOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI

base_parser = JsonOutputParser()
model = ChatOpenAI()

# 自动修复解析错误
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=model)

chain = model | fixing_parser

result = chain.invoke("Extract: Alice is 30 years old.")
print(result)
```

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (78-88)

---

## 九、性能考虑

### 9.1 部分解析开销

`parse_partial_json()` 比完整解析慢，因为需要处理不完整的 JSON：

**优化建议**：
- 只在流式场景下使用部分解析
- 非流式场景使用 `partial=False`

### 9.2 JSON Patch 开销

`jsonpatch.make_patch()` 需要计算差异，有一定开销：

**优化建议**：
- 只在需要减少数据传输时使用 diff 模式
- 小型 JSON 对象不需要使用 diff 模式

### 9.3 Markdown 解析开销

`parse_json_markdown()` 需要正则表达式匹配：

**优化建议**：
- 如果确定 LLM 不会返回 Markdown 代码块，可以直接使用 `json.loads()`
- 但通常开销很小，不需要优化

**来源**：`reference/source_outputparser_02_JSON解析器.md` (345-356)

---

## 十、社区最佳实践

### 10.1 Reddit 社区共识

根据 Reddit 社区讨论，以下是 JSON 解析器的最佳实践：

1. **JsonOutputParser 更灵活**：适用于动态或未知 schema
2. **包含格式指令**：可以显著提高成功率
3. **模型选择影响大**：更强大的模型（如 GPT-4）输出更稳定
4. **错误处理很重要**：使用 OutputFixingParser 和重试机制

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (118-125)

### 10.2 无 Schema 解析

JsonOutputParser 可以不指定 schema，直接解析 LLM 输出的 JSON：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()  # 不指定 schema

chain = model | parser

result = chain.invoke("Return any JSON you want")
print(result)
# LLM 可以返回任意结构的 JSON
```

**适用场景**：
- 动态结构的 JSON
- 不确定输出格式
- 快速原型开发

**来源**：`reference/search_outputparser_01_Reddit最佳实践.md` (70-77)

---

## 十一、总结

### 11.1 核心要点

1. **JsonOutputParser 是最可靠的结构化输出解析器**（不使用 Function Calling）
2. **支持流式解析**：通过 `BaseCumulativeTransformOutputParser` 实现
3. **自动处理 Markdown 代码块**：使用 `parse_json_markdown()`
4. **支持 JSON Patch diff 模式**：减少数据传输量
5. **可选的 Pydantic 验证**：通过 `pydantic_object` 参数启用

### 11.2 选择指南

| 场景 | 推荐解析器 |
|------|-----------|
| 需要流式解析 | JsonOutputParser |
| 需要 JSON Patch | JsonOutputParser |
| 需要严格类型验证 | PydanticOutputParser |
| 简单场景 | SimpleJsonOutputParser |
| 模型支持原生结构化输出 | with_structured_output() |

### 11.3 下一步学习

- 学习 **PydanticOutputParser**：了解如何使用 Pydantic 模型进行严格的类型验证
- 学习 **OutputFixingParser**：了解如何自动修复解析错误
- 学习 **自定义 OutputParser**：了解如何开发自己的解析器

---

## 参考资料

1. **源码分析**：`reference/source_outputparser_02_JSON解析器.md`
2. **Context7 文档**：`reference/context7_langchain_02_Pydantic模型.md`
3. **Reddit 最佳实践**：`reference/search_outputparser_01_Reddit最佳实践.md`
4. **LangChain 官方文档**：https://docs.langchain.com/oss/python/langchain/models

---

**版本**：v1.0
**最后更新**：2026-02-26
**维护者**：Claude Code
