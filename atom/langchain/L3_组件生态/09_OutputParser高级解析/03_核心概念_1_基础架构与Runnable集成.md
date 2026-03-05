# 核心概念1：OutputParser 基础架构与 Runnable 集成

## 概述

OutputParser 是 LangChain 中负责将 LLM 文本输出转换为结构化数据的核心组件。本文深入解析 OutputParser 的基础架构设计、Runnable 协议集成以及与 LCEL 的无缝集成机制。

**核心特性：**
- 三层抽象设计（BaseLLMOutputParser → BaseGenerationOutputParser → BaseOutputParser）
- 统一的 Runnable 协议（所有 OutputParser 都是 Runnable）
- 类型安全（泛型 T）
- 异步优先（所有方法都有异步版本）
- 17种具体实现（JSON、Pydantic、List、XML 等）

---

## 1. 基类层次结构

### 1.1 三层抽象设计

LangChain 的 OutputParser 采用三层抽象设计，每一层都有明确的职责：

```
BaseLLMOutputParser (最基础抽象)
    ↓ 继承
BaseGenerationOutputParser (集成 Runnable)
    ↓ 继承
BaseOutputParser (完整功能)
    ↓ 继承
具体实现（JsonOutputParser, PydanticOutputParser, ...）
```

### 1.2 BaseLLMOutputParser - 最基础抽象

**定义**（来源：`langchain_core/output_parsers/base.py:20-50`）：

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

T = TypeVar("T")

class BaseLLMOutputParser(ABC, Generic[T]):
    """Base class to parse the output of an LLM call.
    
    Output parsers help structure language model responses.
    """
    
    @abstractmethod
    def parse_result(
        self, 
        result: List[Generation], 
        *, 
        partial: bool = False
    ) -> T:
        """Parse a list of candidate model Generations into a specific format.
        
        Args:
            result: A list of Generations to be parsed.
                The Generations are assumed to be different candidate outputs
                for a single model input.
            partial: Whether to parse the output as a partial result.
                This is useful for parsers that can parse partial results.
        
        Returns:
            Structured output.
        """
```

**关键特性：**
- 使用泛型 `T` 确保类型安全
- 核心方法 `parse_result()` 接受 `List[Generation]` 作为输入
- `partial` 参数支持部分解析（用于流式场景）

### 1.3 BaseGenerationOutputParser - 集成 Runnable

**定义**（来源：`langchain_core/output_parsers/base.py:70-100`）：

```python
from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import BaseMessage

class BaseGenerationOutputParser(
    BaseLLMOutputParser, 
    RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call.
    
    This class extends BaseLLMOutputParser and integrates with the Runnable protocol.
    """
    
    @property
    def InputType(self) -> Any:
        """Return the input type for the parser."""
        return str | BaseMessage
    
    @property
    def OutputType(self) -> type[T]:
        """Return the output type for the parser."""
        return cast("type[T]", T)
    
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Invoke the parser on an input.
        
        Args:
            input: The input to parse (string or BaseMessage).
            config: Optional configuration for the invocation.
        
        Returns:
            Parsed output of type T.
        """
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [Generation(text=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
```

**关键特性：**
- 继承 `RunnableSerializable`，使所有 OutputParser 都是 Runnable
- 实现 `invoke()` 方法，支持在 LCEL 中使用
- 支持两种输入类型：`str` 和 `BaseMessage`
- 通过 `_call_with_config()` 实现配置传递和可观测性（LangSmith 追踪）

### 1.4 BaseOutputParser - 完整功能

**定义**（来源：`langchain_core/output_parsers/base.py:150-200`）：

```python
class BaseOutputParser(BaseGenerationOutputParser[T]):
    """Base class to parse the output of an LLM call to a specific format.
    
    This class provides additional convenience methods beyond BaseGenerationOutputParser.
    """
    
    def parse(self, text: str) -> T:
        """Parse a single string model output into some structure.
        
        Args:
            text: String output of a language model.
        
        Returns:
            Structured output.
        """
        return self.parse_result([Generation(text=text)])
    
    async def aparse(self, text: str) -> T:
        """Async version of parse.
        
        Args:
            text: String output of a language model.
        
        Returns:
            Structured output.
        """
        return await self.aparse_result([Generation(text=text)])
    
    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted.
        
        Returns:
            Format instructions to be injected into the prompt.
        """
        raise NotImplementedError(
            f"get_format_instructions is not implemented for {self.__class__.__name__}"
        )
```

**关键特性：**
- 提供便捷方法 `parse(text: str)` 用于解析单个字符串
- 提供异步版本 `aparse(text: str)`
- 提供 `get_format_instructions()` 方法生成格式指令（注入到 Prompt）

---

## 2. Runnable 协议集成

### 2.1 为什么需要 Runnable 协议？

**问题**：LangChain 使用 LCEL（LangChain Expression Language）构建链式调用，需要统一的接口。

**解决方案**：所有组件（Model、Parser、Retriever 等）都实现 Runnable 协议。

**Runnable 协议的核心方法：**
```python
class Runnable(ABC):
    def invoke(self, input: Input) -> Output:
        """同步调用"""
        
    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""
        
    def stream(self, input: Input) -> Iterator[Output]:
        """流式调用"""
        
    async def astream(self, input: Input) -> AsyncIterator[Output]:
        """异步流式调用"""
        
    def batch(self, inputs: List[Input]) -> List[Output]:
        """批量调用"""
```

### 2.2 OutputParser 如何实现 Runnable 协议

**关键代码**（来源：`langchain_core/output_parsers/base.py:70-100`）：

```python
class BaseGenerationOutputParser(
    BaseLLMOutputParser, 
    RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call."""
    
    @property
    def InputType(self) -> Any:
        """Return the input type for the parser."""
        return str | BaseMessage
    
    @property
    def OutputType(self) -> type[T]:
        """Return the output type for the parser."""
        return cast("type[T]", T)
    
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Invoke the parser on an input."""
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        else:
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [Generation(text=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
```

**关键点：**
1. **类型声明**：`RunnableSerializable[LanguageModelOutput, T]` 声明输入输出类型
2. **invoke() 实现**：将输入转换为 `Generation` 对象，调用 `parse_result()`
3. **配置传递**：通过 `_call_with_config()` 传递配置和追踪信息

### 2.3 可观测性集成

**_call_with_config() 的作用：**
```python
def _call_with_config(
    self,
    func: Callable,
    input: Input,
    config: RunnableConfig | None,
    run_type: str,
) -> Output:
    """Call function with config and tracing.
    
    This method:
    1. Merges config with default config
    2. Starts a LangSmith trace
    3. Calls the function
    4. Ends the trace
    5. Returns the result
    """
```

**好处：**
- 自动集成 LangSmith 追踪
- 支持配置传递（如 `max_concurrency`）
- 统一的错误处理

---

## 3. 核心方法详解

### 3.1 parse_result() - 核心解析方法

**签名：**
```python
@abstractmethod
def parse_result(
    self, 
    result: List[Generation], 
    *, 
    partial: bool = False
) -> T:
    """Parse a list of candidate model Generations into a specific format."""
```

**参数说明：**
- `result: List[Generation]`：LLM 生成的候选输出列表
  - 支持多个候选输出（`n > 1`）
  - 通常只使用第一个：`result[0].text`
- `partial: bool = False`：是否为部分解析
  - `False`：完整解析（默认）
  - `True`：部分解析（用于流式场景）

**示例实现**（JsonOutputParser）：
```python
class JsonOutputParser(BaseOutputParser[Dict[str, Any]]):
    """Parse JSON output from LLM."""
    
    def parse_result(
        self, 
        result: List[Generation], 
        *, 
        partial: bool = False
    ) -> Dict[str, Any]:
        """Parse JSON from the first generation."""
        text = result[0].text
        
        # 移除 Markdown 代码块
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # 解析 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if partial:
                # 部分解析：返回空字典
                return {}
            else:
                # 完整解析：抛出异常
                raise OutputParserException(
                    f"Invalid JSON: {e}\nText: {text}"
                )
```

### 3.2 parse() - 便捷方法

**签名：**
```python
def parse(self, text: str) -> T:
    """Parse a single string model output into some structure."""
    return self.parse_result([Generation(text=text)])
```

**用途：**
- 简化单个字符串的解析
- 内部调用 `parse_result()`

**示例：**
```python
parser = JsonOutputParser()

# 方式1：使用 parse()
result = parser.parse('{"name": "Alice", "age": 30}')

# 方式2：使用 parse_result()
result = parser.parse_result([Generation(text='{"name": "Alice", "age": 30}')])

# 两者等价
```

### 3.3 get_format_instructions() - 格式指令生成

**签名：**
```python
def get_format_instructions(self) -> str:
    """Instructions on how the LLM output should be formatted."""
```

**用途：**
- 生成格式指令，注入到 Prompt 中
- 引导 LLM 输出符合预期格式的内容

**示例**（PydanticOutputParser）：
```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

print(format_instructions)
# 输出：
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
# 
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
# 
# Here is the output schema:
# ```
# {"properties": {"name": {"title": "Name", "description": "The person's name", "type": "string"}, "age": {"title": "Age", "description": "The person's age", "type": "integer"}}, "required": ["name", "age"]}
# ```
```

**在 Prompt 中使用：**
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Extract person info from: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
result = chain.invoke({"text": "Alice is 30 years old"})
# 输出: Person(name="Alice", age=30)
```

### 3.4 异步方法

**所有方法都有异步版本：**
```python
class BaseOutputParser(BaseGenerationOutputParser[T]):
    async def aparse_result(
        self, 
        result: List[Generation], 
        *, 
        partial: bool = False
    ) -> T:
        """Async version of parse_result."""
        return await run_in_executor(
            None, 
            self.parse_result, 
            result, 
            partial=partial
        )
    
    async def aparse(self, text: str) -> T:
        """Async version of parse."""
        return await self.aparse_result([Generation(text=text)])
```

**使用示例：**
```python
import asyncio

async def main():
    parser = JsonOutputParser()
    result = await parser.aparse('{"name": "Alice", "age": 30}')
    print(result)

asyncio.run(main())
```

---

## 4. 与 LCEL 的集成

### 4.1 LCEL 管道操作符

**基本用法：**
```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
parser = StrOutputParser()

# LCEL 管道：model | parser
chain = model | parser

# 调用
result = chain.invoke("Tell me a joke")
print(result)  # 输出: "Why did the chicken cross the road? ..."
```

**工作原理：**
1. `model.invoke()` 返回 `AIMessage` 对象
2. `parser.invoke()` 接受 `AIMessage`，提取 `content` 字段
3. 返回字符串

### 4.2 复杂链式调用

**示例：Prompt → Model → Parser**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_template(
    "Extract person info from: {text}\nReturn JSON with 'name' and 'age' fields."
)
model = ChatOpenAI()
parser = JsonOutputParser()

# 三段式链
chain = prompt | model | parser

# 调用
result = chain.invoke({"text": "Alice is 30 years old"})
print(result)  # 输出: {"name": "Alice", "age": 30}
```

### 4.3 流式调用

**示例：**
```python
from langchain_core.output_parsers import StrOutputParser

chain = model | StrOutputParser()

# 流式输出
for chunk in chain.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

**工作原理：**
1. `model.stream()` 返回 `AIMessageChunk` 迭代器
2. `parser.stream()` 逐块处理，提取 `content` 字段
3. 逐步输出字符串

### 4.4 批量调用

**示例：**
```python
chain = model | JsonOutputParser()

# 批量调用
inputs = [
    "Extract: Alice is 30",
    "Extract: Bob is 25",
    "Extract: Charlie is 35"
]

results = chain.batch(inputs)
print(results)
# 输出: [
#     {"name": "Alice", "age": 30},
#     {"name": "Bob", "age": 25},
#     {"name": "Charlie", "age": 35}
# ]
```

---

## 5. 所有可用的 OutputParser 类型

**从源码提取**（来源：`langchain_core/output_parsers/__init__.py:52-70`）：

### 5.1 基类（5个）
- `BaseLLMOutputParser`：最基础抽象
- `BaseGenerationOutputParser`：集成 Runnable
- `BaseOutputParser`：完整功能
- `BaseTransformOutputParser`：转换解析器（逐块）
- `BaseCumulativeTransformOutputParser`：累积转换解析器（流式）

### 5.2 JSON 系列（3个）
- `JsonOutputParser`：标准 JSON 解析
- `SimpleJsonOutputParser`：简化 JSON 解析
- `PydanticOutputParser`：Pydantic 模型验证

### 5.3 列表系列（4个）
- `ListOutputParser`：抽象列表解析器
- `CommaSeparatedListOutputParser`：逗号分隔列表
- `MarkdownListOutputParser`：Markdown 列表
- `NumberedListOutputParser`：编号列表

### 5.4 OpenAI Tools 系列（3个）
- `JsonOutputToolsParser`：JSON 工具解析器
- `JsonOutputKeyToolsParser`：JSON 工具键解析器
- `PydanticToolsParser`：Pydantic 工具解析器

### 5.5 其他格式（2个）
- `StrOutputParser`：字符串输出
- `XMLOutputParser`：XML 解析

**总计：17种 OutputParser 类型**

---

## 6. 架构设计亮点

### 6.1 统一接口

**所有 OutputParser 都实现相同的接口：**
```python
# 所有 Parser 都可以这样使用
parser = AnyOutputParser()
result = parser.parse(text)
result = parser.invoke(text)
result = await parser.aparse(text)
```

### 6.2 类型安全

**使用泛型确保类型安全：**
```python
class JsonOutputParser(BaseOutputParser[Dict[str, Any]]):
    """返回类型是 Dict[str, Any]"""

class PydanticOutputParser(BaseOutputParser[T]):
    """返回类型是 Pydantic 模型 T"""

class StrOutputParser(BaseOutputParser[str]):
    """返回类型是 str"""
```

### 6.3 可扩展性

**易于自定义新的 OutputParser：**
```python
class CustomParser(BaseOutputParser[MyType]):
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> MyType:
        # 自定义解析逻辑
        text = result[0].text
        return MyType.from_text(text)
    
    def get_format_instructions(self) -> str:
        return "Custom format instructions"
```

### 6.4 流式支持

**通过 `partial` 参数支持流式解析：**
```python
def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
    if partial:
        # 部分解析逻辑（容错）
        return self._parse_partial(result[0].text)
    else:
        # 完整解析逻辑（严格）
        return self._parse_complete(result[0].text)
```

---

## 7. 与现代方法的对比

### 7.1 OutputParser vs with_structured_output()

**现代方法（2025+）：**
```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Alice is 30 years old")
# 输出: Person(name="Alice", age=30)
```

**传统方法（OutputParser）：**
```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

prompt = f"Extract: Alice is 30 years old\n{format_instructions}"
chain = llm | parser
result = chain.invoke(prompt)
# 输出: Person(name="Alice", age=30)
```

### 7.2 对比表

| 特性 | with_structured_output() | OutputParser |
|------|--------------------------|--------------|
| 实现方式 | LLM 原生支持（Function Calling） | 手动解析 LLM 输出 |
| 错误处理 | 自动（LLM 内置） | 手动（解析时检测） |
| 流式支持 | 部分支持 | 完全支持 |
| 适用模型 | 支持结构化输出的模型 | 所有模型 |
| 性能 | 更快（原生支持） | 较慢（需要解析） |
| 灵活性 | 受限于 LLM 能力 | 完全可控 |

### 7.3 何时使用 OutputParser

**场景1：模型不支持原生结构化输出**
```python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="gpt2")  # 不支持 Function Calling
parser = JsonOutputParser()
chain = llm | parser  # 必须使用 OutputParser
```

**场景2：需要自定义解析逻辑**
```python
class CustomParser(BaseOutputParser):
    def parse(self, text: str):
        # 自定义逻辑：提取 Markdown 代码块中的 JSON
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        return json.loads(match.group(1))
```

**场景3：需要额外验证**
```python
class ValidatedParser(PydanticOutputParser):
    def parse(self, text: str):
        result = super().parse(text)
        if result.age < 0:
            raise ValueError("年龄不能为负数")
        return result
```

**场景4：需要完整流式支持**
```python
parser = JsonOutputParser()
for chunk in (model | parser).stream(input):
    print(chunk)  # 逐步输出解析结果
```

---

## 8. 总结

### 8.1 核心要点

1. **三层抽象设计**：BaseLLMOutputParser → BaseGenerationOutputParser → BaseOutputParser
2. **Runnable 协议统一**：所有 OutputParser 都是 Runnable，可以在 LCEL 中无缝使用
3. **类型安全**：使用泛型 `T` 确保类型正确
4. **流式支持**：通过 `partial` 参数支持部分解析
5. **异步优先**：所有方法都有异步版本
6. **可观测性**：通过 `_call_with_config()` 集成 LangSmith 追踪

### 8.2 架构优势

- **统一接口**：所有 Parser 都实现相同的方法
- **可扩展性**：易于自定义新的 Parser
- **可组合性**：可以在 LCEL 中自由组合
- **现代化**：与 LLM 原生结构化输出共存

### 8.3 使用建议

1. **优先使用原生结构化输出**（`with_structured_output()`）
2. **回退到 OutputParser**（当模型不支持或需要自定义逻辑时）
3. **利用 LCEL**（`model | parser` 简洁高效）
4. **注意流式场景**（使用 `partial=True` 参数）

---

## 参考来源

### 源码分析
- `langchain_core/output_parsers/base.py` - OutputParser 基类定义
- `langchain_core/output_parsers/__init__.py` - 所有 OutputParser 类型的导出

### Context7 官方文档
- LangChain 官方文档 - 结构化输出与错误处理
- LangChain 官方文档 - Pydantic 模型与结构化输出

### 设计理念
- Runnable 协议：统一接口设计
- LCEL：声明式链式调用
- 类型安全：泛型设计模式
