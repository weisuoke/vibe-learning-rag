---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/output_parsers
analyzed_files:
  - json.py
  - pydantic.py
analyzed_at: 2026-02-26
knowledge_point: OutputParser高级解析
---

# 源码分析：JSON 解析器（JsonOutputParser & PydanticOutputParser）

## 分析的文件

- `langchain_core/output_parsers/json.py` - JSON 解析器实现
- `langchain_core/output_parsers/pydantic.py` - Pydantic 模型验证解析器

## 关键发现

### 1. JsonOutputParser - 核心 JSON 解析器

**类定义**（json.py:31-42）：

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
- 继承自 `BaseCumulativeTransformOutputParser`（支持流式解析）
- 可选的 Pydantic 验证（`pydantic_object` 参数）
- 支持 JSON Patch diff 模式

### 2. 核心解析方法

**parse_result() 方法**（json.py:61-91）：

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
- 使用 `parse_json_markdown()` 工具函数（可以从 Markdown 代码块中提取 JSON）
- `partial=True` 时返回 `None` 而不是抛出异常（用于流式解析）
- `partial=False` 时抛出 `OutputParserException`（包含原始输出）

### 3. JSON Patch 支持（流式 diff 模式）

**_diff() 方法**（json.py:51-52）：

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

### 4. Markdown 代码块解析

**工具函数**（json.py:19-23）：

```python
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)
```

**功能**：
- `parse_json_markdown()`：从 Markdown 代码块中提取 JSON
- `parse_partial_json()`：解析不完整的 JSON（用于流式）
- `parse_and_check_json_markdown()`：解析并验证 JSON

**示例**：
```markdown
```json
{"name": "Alice", "age": 30}
```
```

会被正确解析为 JSON 对象。

### 5. PydanticOutputParser - Pydantic 模型验证

**类定义**（pydantic.py:19-23）：

```python
class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a Pydantic model."""

    pydantic_object: Annotated[type[TBaseModel], SkipValidation()]
    """The Pydantic model to parse."""
```

**关键特性**：
- 继承自 `JsonOutputParser`（复用 JSON 解析逻辑）
- 必须提供 `pydantic_object`（Pydantic 模型类）
- 支持 Pydantic v1 和 v2

### 6. Pydantic 模型验证

**_parse_obj() 方法**（pydantic.py:25-35）：

```python
def _parse_obj(self, obj: dict) -> TBaseModel:
    try:
        if issubclass(self.pydantic_object, pydantic.BaseModel):
            return self.pydantic_object.model_validate(obj)
        if issubclass(self.pydantic_object, pydantic.v1.BaseModel):
            return self.pydantic_object.parse_obj(obj)
        msg = f"Unsupported model version for PydanticOutputParser: \
                    {self.pydantic_object.__class__}"
        raise OutputParserException(msg)
    except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
        raise self._parser_exception(e, obj) from e
```

**关键发现**：
- 自动检测 Pydantic 版本（v1 vs v2）
- Pydantic v2：使用 `model_validate()`
- Pydantic v1：使用 `parse_obj()`
- 捕获验证错误并转换为 `OutputParserException`

### 7. 错误处理

**_parser_exception() 方法**（pydantic.py:37-43）：

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

### 8. parse_result() 重写

**PydanticOutputParser.parse_result()**（pydantic.py:55-80）：

```python
def parse_result(
    self, result: list[Generation], *, partial: bool = False
) -> TBaseModel | None:
    """Parse the result of an LLM call to a Pydantic object.

    Args:
        result: The result of the LLM call.
        partial: Whether to parse partial JSON objects.

            If `True`, the output will be a JSON object containing all the keys that
            have been returned so far.

    Raises:
        OutputParserException: If the result is not valid JSON or does not conform
            to the Pydantic model.

    Returns:
        The parsed Pydantic object.
    """
    try:
        json_object = super().parse_result(result)
        return self._parse_obj(json_object)
    except OutputParserException:
        if partial:
            return None
        raise
```

**关键发现**：
- 先调用父类的 `parse_result()`（JSON 解析）
- 然后调用 `_parse_obj()`（Pydantic 验证）
- `partial=True` 时返回 `None` 而不是抛出异常

### 9. 格式指令生成

**get_format_instructions() 方法**（pydantic.py:93-100）：

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

### 10. Schema 提取

**_get_schema() 静态方法**（json.py:54-58）：

```python
@staticmethod
def _get_schema(pydantic_object: type[TBaseModel]) -> dict[str, Any]:
    if issubclass(pydantic_object, pydantic.BaseModel):
        return pydantic_object.model_json_schema()
    return pydantic_object.schema()
```

**关键发现**：
- Pydantic v2：使用 `model_json_schema()`
- Pydantic v1：使用 `schema()`
- 返回标准 JSON Schema

## 使用示例

### JsonOutputParser 基础用法

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser

result = chain.invoke("Return a JSON with name and age")
# 输出: {"name": "Alice", "age": 30}
```

### PydanticOutputParser 用法

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)

# 获取格式指令
format_instructions = parser.get_format_instructions()

# 注入到 Prompt
prompt = f"Return a person's info.\n{format_instructions}"

chain = model | parser
result = chain.invoke(prompt)
# 输出: Person(name="Alice", age=30)
```

### 流式解析

```python
# 流式 JSON 解析
for chunk in chain.stream("Return a large JSON"):
    print(chunk)  # 输出部分 JSON 对象
```

### JSON Patch diff 模式

```python
# 启用 diff 模式
parser = JsonOutputParser(diff=True)

for patch in chain.stream("Return a large JSON"):
    print(patch)  # 输出 JSON Patch 操作
```

## 架构设计亮点

1. **Markdown 友好**：自动从 Markdown 代码块中提取 JSON
2. **流式支持**：通过 `partial` 参数和 `parse_partial_json()` 支持流式解析
3. **JSON Patch**：减少流式传输的数据量
4. **Pydantic 兼容**：同时支持 Pydantic v1 和 v2
5. **格式指令**：自动生成 Prompt 指令
6. **错误友好**：详细的错误信息，包含原始输出

## 性能考虑

1. **部分解析开销**：`parse_partial_json()` 比完整解析慢
2. **JSON Patch 开销**：`jsonpatch.make_patch()` 需要计算差异
3. **Pydantic 验证开销**：复杂模型验证可能较慢

## 总结

JsonOutputParser 和 PydanticOutputParser 是 LangChain 中最可靠的结构化输出解析器：
- **JsonOutputParser**：灵活的 JSON 解析，支持流式和 diff 模式
- **PydanticOutputParser**：类型安全的模型验证，支持格式指令生成
- **共同特点**：Markdown 友好、流式支持、错误友好
