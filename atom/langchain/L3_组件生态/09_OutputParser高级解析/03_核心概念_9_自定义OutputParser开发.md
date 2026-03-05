# 核心概念 9：自定义 OutputParser 开发

> **资料来源**：
> - reference/source_outputparser_01_基础架构.md（源码分析）
> - reference/search_outputparser_01_Reddit最佳实践.md（社区最佳实践）

---

## 概述

自定义 OutputParser 开发是 LangChain 中的高级技能，允许你创建专门的解析器来处理特殊格式的 LLM 输出。当内置解析器无法满足需求时，自定义解析器提供了完全的灵活性和控制力。

**核心能力**：
- 继承 `BaseOutputParser` 基类
- 实现 `parse_result()` 方法（核心解析逻辑）
- 实现 `get_format_instructions()` 方法（生成格式指令）
- 实现异步方法（`aparse_result()`）
- 支持流式解析（可选）

**适用场景**：
- 处理特殊格式的输出（如自定义标记语言）
- 需要复杂的验证逻辑
- 需要与现有系统集成
- 需要特殊的错误处理

---

## 1. 为什么需要自定义 OutputParser？

### 1.1 内置解析器的局限性

**内置解析器覆盖的场景**：
- `JsonOutputParser` - 标准 JSON
- `PydanticOutputParser` - Pydantic 模型验证
- `StrOutputParser` - 纯文本
- `XMLOutputParser` - XML 格式

**无法覆盖的场景**：
- 自定义标记语言（如 `[ANSWER]...[/ANSWER]`）
- 复杂的多步解析逻辑
- 需要外部 API 调用的验证
- 特殊的错误恢复策略
- 与遗留系统的格式兼容

### 1.2 自定义解析器的优势

**完全控制**：
- 自定义解析逻辑
- 自定义错误处理
- 自定义格式指令

**灵活性**：
- 支持任意输出格式
- 支持复杂的验证规则
- 支持多步解析流程

**可扩展性**：
- 易于维护和更新
- 可以组合多个解析器
- 可以继承和扩展

### 1.3 实际应用场景

**场景 1：解析特殊标记**
```
输入：
[THOUGHT]我需要先搜索相关信息[/THOUGHT]
[ACTION]search("LangChain")[/ACTION]
[ANSWER]LangChain 是一个 AI 应用开发框架[/ANSWER]

需求：提取 THOUGHT、ACTION、ANSWER 三个部分
```

**场景 2：多步验证**
```
输入：
{
  "user_id": "12345",
  "action": "transfer",
  "amount": 1000
}

需求：
1. 验证 JSON 格式
2. 验证 user_id 是否存在（调用 API）
3. 验证 amount 是否在限额内
4. 返回验证后的对象
```

**场景 3：兼容遗留系统**
```
输入：
STATUS: SUCCESS
CODE: 200
DATA: {"result": "ok"}

需求：解析为标准 JSON 格式
```

---

## 2. BaseOutputParser 基类架构

### 2.1 类层次结构

**三层抽象设计**（来源：source_outputparser_01_基础架构.md）：

```python
BaseLLMOutputParser (ABC, Generic[T])
    ↓
BaseGenerationOutputParser (BaseLLMOutputParser, RunnableSerializable)
    ↓
BaseOutputParser (BaseGenerationOutputParser)
```

**关键特性**：
- `BaseLLMOutputParser`：最基础的抽象类，定义 `parse_result()` 方法
- `BaseGenerationOutputParser`：继承 `RunnableSerializable`，使所有 OutputParser 都是 Runnable
- `BaseOutputParser`：更高级的抽象，提供完整的解析功能

### 2.2 必须实现的方法

**核心方法**：

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation

class CustomOutputParser(BaseOutputParser[T]):
    """自定义输出解析器"""

    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False
    ) -> T:
        """解析 LLM 输出

        Args:
            result: LLM 生成的结果列表
            partial: 是否支持部分解析（流式场景）

        Returns:
            解析后的结果
        """
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        """生成格式指令

        Returns:
            格式指令字符串，用于注入到 Prompt 中
        """
        raise NotImplementedError
```

### 2.3 可选实现的方法

**异步方法**：

```python
async def aparse_result(
    self,
    result: list[Generation],
    *,
    partial: bool = False
) -> T:
    """异步解析 LLM 输出

    默认实现：在线程池中运行同步方法
    """
    return await run_in_executor(None, self.parse_result, result, partial=partial)
```

**流式解析方法**（继承 `BaseTransformOutputParser`）：

```python
def parse(self, text: str) -> T:
    """解析单个文本块"""
    raise NotImplementedError

def transform(self, input: Iterator[str]) -> Iterator[T]:
    """转换输入流为输出流"""
    for chunk in input:
        yield self.parse(chunk)
```

---

## 3. 实现 parse_result() 方法

### 3.1 方法签名

```python
def parse_result(
    self,
    result: list[Generation],
    *,
    partial: bool = False
) -> T:
    """解析 LLM 输出

    Args:
        result: LLM 生成的结果列表
            - 通常只有一个元素（单个生成结果）
            - 可能有多个元素（多个候选结果）
        partial: 是否支持部分解析
            - True: 流式场景，输入可能不完整
            - False: 完整输入

    Returns:
        解析后的结果（类型 T）

    Raises:
        OutputParserException: 解析失败时抛出
    """
```

### 3.2 基础实现模式

**模式 1：提取文本并解析**

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

class TagOutputParser(BaseOutputParser[dict]):
    """解析标记格式的输出"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        # 1. 提取文本
        if not result:
            raise OutputParserException("Empty result")

        text = result[0].text

        # 2. 解析逻辑
        try:
            # 提取标记内容
            thought = self._extract_tag(text, "THOUGHT")
            action = self._extract_tag(text, "ACTION")
            answer = self._extract_tag(text, "ANSWER")

            return {
                "thought": thought,
                "action": action,
                "answer": answer
            }
        except Exception as e:
            raise OutputParserException(f"Failed to parse: {e}")

    def _extract_tag(self, text: str, tag: str) -> str:
        """提取标记内容"""
        import re
        pattern = f"\\[{tag}\\](.*?)\\[/{tag}\\]"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def get_format_instructions(self) -> str:
        return """请按照以下格式输出：
[THOUGHT]你的思考过程[/THOUGHT]
[ACTION]你要执行的动作[/ACTION]
[ANSWER]最终答案[/ANSWER]
"""
```

**模式 2：多步验证**

```python
class ValidatedOutputParser(BaseOutputParser[dict]):
    """带验证的输出解析器"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text

        # 步骤 1：解析 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Invalid JSON: {e}")

        # 步骤 2：验证必需字段
        required_fields = ["user_id", "action", "amount"]
        for field in required_fields:
            if field not in data:
                raise OutputParserException(f"Missing field: {field}")

        # 步骤 3：验证数据类型
        if not isinstance(data["amount"], (int, float)):
            raise OutputParserException("amount must be a number")

        # 步骤 4：业务逻辑验证
        if data["amount"] < 0:
            raise OutputParserException("amount must be positive")

        if data["amount"] > 10000:
            raise OutputParserException("amount exceeds limit")

        return data

    def get_format_instructions(self) -> str:
        return """请输出 JSON 格式，包含以下字段：
- user_id: 用户 ID（字符串）
- action: 操作类型（字符串）
- amount: 金额（数字，0-10000）
"""
```

### 3.3 处理部分解析（流式场景）

```python
class StreamingTagParser(BaseOutputParser[dict]):
    """支持流式解析的标记解析器"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text

        if partial:
            # 部分解析：尽可能提取已完成的标记
            return self._parse_partial(text)
        else:
            # 完整解析：要求所有标记都存在
            return self._parse_complete(text)

    def _parse_partial(self, text: str) -> dict:
        """部分解析：提取已完成的标记"""
        result = {}

        # 尝试提取每个标记
        for tag in ["THOUGHT", "ACTION", "ANSWER"]:
            try:
                content = self._extract_tag(text, tag)
                if content:
                    result[tag.lower()] = content
            except:
                pass  # 忽略未完成的标记

        return result

    def _parse_complete(self, text: str) -> dict:
        """完整解析：要求所有标记都存在"""
        result = {}

        for tag in ["THOUGHT", "ACTION", "ANSWER"]:
            content = self._extract_tag(text, tag)
            if not content:
                raise OutputParserException(f"Missing tag: {tag}")
            result[tag.lower()] = content

        return result

    def _extract_tag(self, text: str, tag: str) -> str:
        import re
        pattern = f"\\[{tag}\\](.*?)\\[/{tag}\\]"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def get_format_instructions(self) -> str:
        return """请按照以下格式输出：
[THOUGHT]你的思考过程[/THOUGHT]
[ACTION]你要执行的动作[/ACTION]
[ANSWER]最终答案[/ANSWER]
"""
```

---

## 4. 实现 get_format_instructions() 方法

### 4.1 方法作用

`get_format_instructions()` 方法生成格式指令，用于注入到 Prompt 中，告诉 LLM 如何格式化输出。

**工作流程**：
```
Prompt Template → 注入格式指令 → LLM → 输出 → OutputParser 解析
```

### 4.2 基础实现

```python
class CustomParser(BaseOutputParser[dict]):
    def get_format_instructions(self) -> str:
        return """请按照以下格式输出：

格式：
[SECTION_NAME]
内容
[/SECTION_NAME]

示例：
[THOUGHT]
我需要先分析问题
[/THOUGHT]
[ANSWER]
最终答案是 42
[/ANSWER]
"""
```

### 4.3 动态生成格式指令

```python
from pydantic import BaseModel, Field

class StructuredParser(BaseOutputParser[dict]):
    """基于 Pydantic 模型生成格式指令"""

    schema: type[BaseModel]

    def get_format_instructions(self) -> str:
        # 从 Pydantic 模型生成格式指令
        schema_dict = self.schema.model_json_schema()

        instructions = "请输出 JSON 格式，包含以下字段：\n\n"

        for field_name, field_info in schema_dict.get("properties", {}).items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            required = field_name in schema_dict.get("required", [])

            instructions += f"- {field_name}: {field_desc} "
            instructions += f"(类型: {field_type}"
            if required:
                instructions += ", 必需"
            instructions += ")\n"

        return instructions

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text
        try:
            data = json.loads(text)
            # 使用 Pydantic 验证
            validated = self.schema(**data)
            return validated.model_dump()
        except Exception as e:
            raise OutputParserException(f"Validation failed: {e}")
```

### 4.4 在 Prompt 中使用格式指令

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 创建解析器
parser = TagOutputParser()

# 创建 Prompt（注入格式指令）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。{format_instructions}"),
    ("human", "{input}")
])

# 创建链
model = ChatOpenAI()
chain = prompt | model | parser

# 调用（自动注入格式指令）
result = chain.invoke({
    "input": "什么是 LangChain？",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# 输出: {
#     "thought": "我需要解释 LangChain 的概念",
#     "action": "explain",
#     "answer": "LangChain 是一个 AI 应用开发框架"
# }
```

---

## 5. 实现异步方法

### 5.1 为什么需要异步方法？

**异步方法的优势**：
- 提高并发性能
- 支持异步 LLM 调用
- 避免阻塞主线程
- 更好的资源利用

**默认实现**（来源：source_outputparser_01_基础架构.md）：

```python
async def aparse_result(
    self,
    result: list[Generation],
    *,
    partial: bool = False
) -> T:
    """异步解析 LLM 输出

    默认实现：在线程池中运行同步方法
    """
    return await run_in_executor(None, self.parse_result, result, partial=partial)
```

### 5.2 何时需要自定义异步方法？

**场景 1：需要异步 I/O 操作**
```python
class AsyncValidatedParser(BaseOutputParser[dict]):
    """需要异步 API 调用的解析器"""

    async def aparse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text
        data = json.loads(text)

        # 异步验证用户 ID
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.example.com/users/{data['user_id']}") as resp:
                if resp.status != 200:
                    raise OutputParserException("Invalid user_id")

        return data

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        # 同步版本
        text = result[0].text
        data = json.loads(text)

        # 同步验证（使用 requests）
        import requests
        resp = requests.get(f"https://api.example.com/users/{data['user_id']}")
        if resp.status_code != 200:
            raise OutputParserException("Invalid user_id")

        return data

    def get_format_instructions(self) -> str:
        return "请输出 JSON 格式，包含 user_id 字段"
```

**场景 2：需要并发处理**
```python
class ParallelValidationParser(BaseOutputParser[dict]):
    """并发验证多个字段"""

    async def aparse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        text = result[0].text
        data = json.loads(text)

        # 并发验证多个字段
        tasks = [
            self._validate_user(data["user_id"]),
            self._validate_product(data["product_id"]),
            self._validate_amount(data["amount"])
        ]

        await asyncio.gather(*tasks)

        return data

    async def _validate_user(self, user_id: str):
        # 异步验证用户
        pass

    async def _validate_product(self, product_id: str):
        # 异步验证产品
        pass

    async def _validate_amount(self, amount: float):
        # 异步验证金额
        pass

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> dict:
        # 同步版本（顺序验证）
        text = result[0].text
        data = json.loads(text)

        self._validate_user_sync(data["user_id"])
        self._validate_product_sync(data["product_id"])
        self._validate_amount_sync(data["amount"])

        return data

    def get_format_instructions(self) -> str:
        return "请输出 JSON 格式，包含 user_id, product_id, amount 字段"
```

### 5.3 异步方法使用示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建异步链
model = ChatOpenAI()
parser = AsyncValidatedParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "{format_instructions}"),
    ("human", "{input}")
])

chain = prompt | model | parser

# 异步调用
async def main():
    result = await chain.ainvoke({
        "input": "创建一个订单",
        "format_instructions": parser.get_format_instructions()
    })
    print(result)

# 运行
import asyncio
asyncio.run(main())
```

---

## 6. 完整示例：生产级自定义解析器

### 6.1 需求分析

**场景**：解析 Agent 的思考过程

**输入格式**：
```
[THOUGHT]
我需要先搜索相关信息
[/THOUGHT]

[ACTION]
search("LangChain")
[/ACTION]

[OBSERVATION]
LangChain 是一个 AI 应用开发框架
[/OBSERVATION]

[ANSWER]
LangChain 是一个用于构建 AI 应用的框架
[/ANSWER]
```

**输出格式**：
```python
{
    "thought": "我需要先搜索相关信息",
    "action": "search",
    "action_input": "LangChain",
    "observation": "LangChain 是一个 AI 应用开发框架",
    "answer": "LangChain 是一个用于构建 AI 应用的框架"
}
```

### 6.2 完整实现

```python
import re
import json
from typing import Any
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.exceptions import OutputParserException

class AgentThoughtParser(BaseOutputParser[dict[str, Any]]):
    """解析 Agent 思考过程的自定义解析器

    支持：
    - 提取 THOUGHT、ACTION、OBSERVATION、ANSWER 标记
    - 解析 ACTION 中的函数调用
    - 部分解析（流式场景）
    - 错误处理和验证
    """

    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False
    ) -> dict[str, Any]:
        """解析 LLM 输出"""
        if not result:
            raise OutputParserException("Empty result")

        text = result[0].text

        if partial:
            return self._parse_partial(text)
        else:
            return self._parse_complete(text)

    def _parse_partial(self, text: str) -> dict[str, Any]:
        """部分解析：提取已完成的标记"""
        result = {}

        # 提取 THOUGHT
        thought = self._extract_tag(text, "THOUGHT")
        if thought:
            result["thought"] = thought

        # 提取 ACTION
        action = self._extract_tag(text, "ACTION")
        if action:
            action_parsed = self._parse_action(action)
            result.update(action_parsed)

        # 提取 OBSERVATION
        observation = self._extract_tag(text, "OBSERVATION")
        if observation:
            result["observation"] = observation

        # 提取 ANSWER
        answer = self._extract_tag(text, "ANSWER")
        if answer:
            result["answer"] = answer

        return result

    def _parse_complete(self, text: str) -> dict[str, Any]:
        """完整解析：要求所有标记都存在"""
        result = {}

        # 提取 THOUGHT（必需）
        thought = self._extract_tag(text, "THOUGHT")
        if not thought:
            raise OutputParserException("Missing THOUGHT tag")
        result["thought"] = thought

        # 提取 ACTION（必需）
        action = self._extract_tag(text, "ACTION")
        if not action:
            raise OutputParserException("Missing ACTION tag")
        action_parsed = self._parse_action(action)
        result.update(action_parsed)

        # 提取 OBSERVATION（可选）
        observation = self._extract_tag(text, "OBSERVATION")
        if observation:
            result["observation"] = observation

        # 提取 ANSWER（必需）
        answer = self._extract_tag(text, "ANSWER")
        if not answer:
            raise OutputParserException("Missing ANSWER tag")
        result["answer"] = answer

        return result

    def _extract_tag(self, text: str, tag: str) -> str:
        """提取标记内容"""
        pattern = f"\\[{tag}\\](.*?)\\[/{tag}\\]"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_action(self, action: str) -> dict[str, Any]:
        """解析 ACTION 中的函数调用

        示例：
        - search("LangChain") → {"action": "search", "action_input": "LangChain"}
        - calculate(10 + 20) → {"action": "calculate", "action_input": "10 + 20"}
        """
        # 匹配函数调用格式：function_name(arguments)
        pattern = r"(\w+)\((.*?)\)"
        match = re.search(pattern, action)

        if match:
            action_name = match.group(1)
            action_input = match.group(2).strip()

            # 移除引号
            if action_input.startswith('"') and action_input.endswith('"'):
                action_input = action_input[1:-1]
            elif action_input.startswith("'") and action_input.endswith("'"):
                action_input = action_input[1:-1]

            return {
                "action": action_name,
                "action_input": action_input
            }
        else:
            # 如果不是函数调用格式，直接返回原始文本
            return {
                "action": action,
                "action_input": ""
            }

    def get_format_instructions(self) -> str:
        """生成格式指令"""
        return """请按照以下格式输出：

[THOUGHT]
你的思考过程（为什么需要执行这个动作）
[/THOUGHT]

[ACTION]
function_name("arguments")
[/ACTION]

[OBSERVATION]
执行结果（如果有）
[/OBSERVATION]

[ANSWER]
最终答案
[/ANSWER]

示例：
[THOUGHT]
我需要搜索 LangChain 的相关信息
[/THOUGHT]

[ACTION]
search("LangChain")
[/ACTION]

[OBSERVATION]
LangChain 是一个 AI 应用开发框架
[/OBSERVATION]

[ANSWER]
LangChain 是一个用于构建 AI 应用的框架
[/ANSWER]
"""

    @property
    def _type(self) -> str:
        """返回解析器类型"""
        return "agent_thought"
```

### 6.3 使用示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建解析器
parser = AgentThoughtParser()

# 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 AI 助手。{format_instructions}"),
    ("human", "{input}")
])

# 创建链
model = ChatOpenAI(model="gpt-4")
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "input": "什么是 LangChain？",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# 输出: {
#     "thought": "我需要解释 LangChain 的概念",
#     "action": "explain",
#     "action_input": "LangChain",
#     "answer": "LangChain 是一个 AI 应用开发框架"
# }
```

### 6.4 流式解析示例

```python
# 流式调用
for chunk in chain.stream({
    "input": "什么是 LangChain？",
    "format_instructions": parser.get_format_instructions()
}):
    print(chunk)
    # 输出: {"thought": "我需要解释 LangChain 的概念"}
    # 输出: {"thought": "...", "action": "explain", "action_input": "LangChain"}
    # 输出: {"thought": "...", "action": "...", "answer": "..."}
```

---

## 7. 最佳实践

### 7.1 设计原则

**1. 单一职责**
```python
# ❌ 不好：解析器做太多事情
class BadParser(BaseOutputParser):
    def parse_result(self, result, *, partial=False):
        text = result[0].text
        data = json.loads(text)
        # 验证
        self._validate(data)
        # 转换
        self._transform(data)
        # 保存到数据库
        self._save_to_db(data)
        return data

# ✅ 好：解析器只负责解析
class GoodParser(BaseOutputParser):
    def parse_result(self, result, *, partial=False):
        text = result[0].text
        data = json.loads(text)
        return data
```

**2. 清晰的错误消息**
```python
# ❌ 不好：错误消息不清晰
raise OutputParserException("Parse failed")

# ✅ 好：错误消息清晰
raise OutputParserException(
    f"Failed to parse JSON: {e}\n"
    f"Input text: {text[:100]}...\n"
    f"Expected format: {self.get_format_instructions()}"
)
```

**3. 支持部分解析**
```python
# ✅ 好：支持流式场景
def parse_result(self, result, *, partial=False):
    if partial:
        return self._parse_partial(result[0].text)
    else:
        return self._parse_complete(result[0].text)
```

### 7.2 性能优化

**1. 缓存正则表达式**
```python
class OptimizedParser(BaseOutputParser):
    def __init__(self):
        # 缓存编译后的正则表达式
        self._tag_pattern = re.compile(r"\[(\w+)\](.*?)\[/\1\]", re.DOTALL)

    def _extract_tags(self, text: str) -> dict:
        return {
            match.group(1): match.group(2).strip()
            for match in self._tag_pattern.finditer(text)
        }
```

**2. 避免重复解析**
```python
class CachedParser(BaseOutputParser):
    def __init__(self):
        self._cache = {}

    def parse_result(self, result, *, partial=False):
        text = result[0].text
        cache_key = hash(text)

        if cache_key in self._cache:
            return self._cache[cache_key]

        parsed = self._parse(text)
        self._cache[cache_key] = parsed
        return parsed
```

### 7.3 测试策略

**1. 单元测试**
```python
import pytest
from langchain_core.outputs import Generation

def test_agent_thought_parser():
    parser = AgentThoughtParser()

    # 测试完整解析
    text = """
    [THOUGHT]
    我需要搜索
    [/THOUGHT]

    [ACTION]
    search("test")
    [/ACTION]

    [ANSWER]
    结果
    [/ANSWER]
    """

    result = parser.parse_result([Generation(text=text)])

    assert result["thought"] == "我需要搜索"
    assert result["action"] == "search"
    assert result["action_input"] == "test"
    assert result["answer"] == "结果"

def test_partial_parsing():
    parser = AgentThoughtParser()

    # 测试部分解析
    text = "[THOUGHT]\n我需要搜索\n[/THOUGHT]"

    result = parser.parse_result([Generation(text=text)], partial=True)

    assert result["thought"] == "我需要搜索"
    assert "action" not in result

def test_error_handling():
    parser = AgentThoughtParser()

    # 测试错误处理
    text = "Invalid format"

    with pytest.raises(OutputParserException):
        parser.parse_result([Generation(text=text)])
```

---

## 8. 常见问题

### 8.1 解析失败

**问题**：`OutputParserException: Failed to parse`

**原因**：
- LLM 输出格式不符合预期
- 正则表达式匹配失败
- JSON 格式错误

**解决方案**：
```python
# 1. 改进格式指令
def get_format_instructions(self) -> str:
    return """请严格按照以下格式输出（不要添加额外的文本）：

[THOUGHT]
你的思考
[/THOUGHT]

[ANSWER]
你的答案
[/ANSWER]

示例：
[THOUGHT]
我需要分析问题
[/THOUGHT]

[ANSWER]
答案是 42
[/ANSWER]
"""

# 2. 添加容错逻辑
def _extract_tag(self, text: str, tag: str) -> str:
    # 尝试多种模式
    patterns = [
        f"\\[{tag}\\](.*?)\\[/{tag}\\]",  # 标准格式
        f"\\[{tag}\\]\\s*(.*?)\\s*\\[/{tag}\\]",  # 允许空格
        f"<{tag}>(.*?)</{tag}>",  # XML 格式
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""
```

### 8.2 流式解析不完整

**问题**：流式场景下解析结果不完整

**解决方案**：
```python
# 使用 BaseCumulativeTransformOutputParser
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser

class StreamingParser(BaseCumulativeTransformOutputParser[dict]):
    """支持流式解析的解析器"""

    def parse_result(self, result, *, partial=False):
        # 累积解析逻辑
        pass
```

### 8.3 性能问题

**问题**：解析速度慢

**解决方案**：
```python
# 1. 使用更高效的解析方法
# ❌ 慢：多次正则匹配
def _extract_tags_slow(self, text: str) -> dict:
    return {
        "thought": self._extract_tag(text, "THOUGHT"),
        "action": self._extract_tag(text, "ACTION"),
        "answer": self._extract_tag(text, "ANSWER"),
    }

# ✅ 快：一次匹配所有标记
def _extract_tags_fast(self, text: str) -> dict:
    pattern = r"\[(\w+)\](.*?)\[/\1\]"
    return {
        match.group(1).lower(): match.group(2).strip()
        for match in re.finditer(pattern, text, re.DOTALL)
    }
```

---

## 9. 与其他 OutputParser 的对比

### 9.1 功能对比

| 解析器 | 灵活性 | 复杂度 | 性能 | 适用场景 |
|--------|--------|--------|------|----------|
| StrOutputParser | 低 | 低 | 高 | 纯文本 |
| JsonOutputParser | 中 | 低 | 高 | 标准 JSON |
| PydanticOutputParser | 中 | 中 | 中 | Pydantic 验证 |
| 自定义 OutputParser | 高 | 高 | 中 | 特殊格式 |

### 9.2 选择指南

**使用内置解析器**：
- 输出格式是标准格式（JSON、XML、纯文本）
- 不需要复杂的验证逻辑
- 追求开发速度

**使用自定义解析器**：
- 输出格式是自定义格式
- 需要复杂的验证逻辑
- 需要与现有系统集成
- 需要特殊的错误处理

---

## 10. 总结

### 10.1 核心要点

1. **继承 BaseOutputParser**：所有自定义解析器都继承自 `BaseOutputParser`
2. **实现核心方法**：
   - `parse_result()`：核心解析逻辑
   - `get_format_instructions()`：生成格式指令
   - `aparse_result()`：异步解析（可选）
3. **支持部分解析**：通过 `partial` 参数支持流式场景
4. **错误处理**：使用 `OutputParserException` 处理解析错误
5. **Runnable 集成**：自动集成到 LCEL 中

### 10.2 最佳实践总结

1. **设计原则**：单一职责、清晰的错误消息、支持部分解析
2. **性能优化**：缓存正则表达式、避免重复解析
3. **测试策略**：单元测试、部分解析测试、错误处理测试
4. **文档完善**：清晰的格式指令、示例输出

### 10.3 适用场景

**适合使用自定义 OutputParser 的场景**：
- 处理特殊格式的输出
- 需要复杂的验证逻辑
- 需要与现有系统集成
- 需要特殊的错误处理

**不适合的场景**：
- 标准格式输出（使用内置解析器）
- 简单的文本提取（使用 `StrOutputParser`）
- 标准 JSON 解析（使用 `JsonOutputParser`）

### 10.4 与 AI Agent 开发的关系

自定义 OutputParser 是构建复杂 AI Agent 系统的关键组件：
- **Agent 思考过程解析**：解析 Agent 的推理步骤
- **工具调用解析**：解析 Agent 的工具调用指令
- **多步推理**：支持复杂的多步推理流程
- **错误恢复**：提供灵活的错误处理机制

---

**参考资料**：
- reference/source_outputparser_01_基础架构.md（源码分析）
- reference/search_outputparser_01_Reddit最佳实践.md（社区最佳实践）
- LangChain 官方文档：https://python.langchain.com/docs/modules/model_io/output_parsers/
- Python 正则表达式文档：https://docs.python.org/3/library/re.html

