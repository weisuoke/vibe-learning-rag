---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/output_parsers
analyzed_files:
  - base.py
  - __init__.py
analyzed_at: 2026-02-26
knowledge_point: OutputParser高级解析
---

# 源码分析：OutputParser 基础架构与 Runnable 集成

## 分析的文件

- `langchain_core/output_parsers/base.py` - OutputParser 基类定义
- `langchain_core/output_parsers/__init__.py` - 所有 OutputParser 类型的导出

## 关键发现

### 1. 基类层次结构

**三层抽象设计**：

```python
BaseLLMOutputParser (ABC, Generic[T])
    ↓
BaseGenerationOutputParser (BaseLLMOutputParser, RunnableSerializable)
    ↓
BaseOutputParser (BaseGenerationOutputParser)
```

**核心特点**：
- `BaseLLMOutputParser`：最基础的抽象类，定义 `parse_result()` 方法
- `BaseGenerationOutputParser`：继承 `RunnableSerializable`，使所有 OutputParser 都是 Runnable
- `BaseOutputParser`：更高级的抽象，提供完整的解析功能

### 2. Runnable 协议集成

**关键代码**（base.py:70-100）：

```python
class BaseGenerationOutputParser(
    BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call."""

    @property
    @override
    def InputType(self) -> Any:
        """Return the input type for the parser."""
        return str | AnyMessage

    @property
    @override
    def OutputType(self) -> type[T]:
        """Return the output type for the parser."""
        return cast("type[T]", T)

    @override
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
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
                lambda inner_input: self.parse_result([Generation(text=inner_input)]),
                input,
                config,
                run_type="parser",
            )
```

**关键发现**：
- 所有 OutputParser 都实现了 `invoke()` 方法，可以在 LCEL 中使用
- 支持两种输入类型：`str` 和 `BaseMessage`
- 通过 `_call_with_config()` 实现配置传递和可观测性

### 3. 核心方法

**parse_result() - 核心解析方法**：

```python
@abstractmethod
def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
    """Parse a list of candidate model `Generation` objects into a specific format.

    Args:
        result: A list of `Generation` to be parsed.
            The `Generation` objects are assumed to be different candidate outputs
            for a single model input.
        partial: Whether to parse the output as a partial result.
            This is useful for parsers that can parse partial results.

    Returns:
        Structured output.
    """
```

**关键特性**：
- `partial` 参数支持部分解析（用于流式场景）
- 接受 `list[Generation]` 作为输入（支持多个候选输出）
- 返回泛型 `T`（类型安全）

**异步支持**：

```python
async def aparse_result(
    self, result: list[Generation], *, partial: bool = False
) -> T:
    """Parse a list of candidate model `Generation` objects into a specific format.

    Returns:
        Structured output.
    """
    return await run_in_executor(None, self.parse_result, result, partial=partial)
```

### 4. 所有可用的 OutputParser 类型

**从 __init__.py 提取**（__init__.py:52-70）：

```python
__all__ = [
    "BaseCumulativeTransformOutputParser",  # 累积转换解析器（流式）
    "BaseGenerationOutputParser",           # 基础生成解析器
    "BaseLLMOutputParser",                  # 基础 LLM 解析器
    "BaseOutputParser",                     # 基础输出解析器
    "BaseTransformOutputParser",            # 转换解析器
    "CommaSeparatedListOutputParser",       # 逗号分隔列表
    "JsonOutputKeyToolsParser",             # JSON 工具键解析器
    "JsonOutputParser",                     # JSON 解析器
    "JsonOutputToolsParser",                # JSON 工具解析器
    "ListOutputParser",                     # 列表解析器
    "MarkdownListOutputParser",             # Markdown 列表
    "NumberedListOutputParser",             # 编号列表
    "PydanticOutputParser",                 # Pydantic 模型验证
    "PydanticToolsParser",                  # Pydantic 工具解析器
    "SimpleJsonOutputParser",               # 简化 JSON 解析器
    "StrOutputParser",                      # 字符串输出
    "XMLOutputParser",                      # XML 解析器
]
```

**分类统计**：
- **基类**：5个（BaseLLMOutputParser, BaseGenerationOutputParser, BaseOutputParser, BaseTransformOutputParser, BaseCumulativeTransformOutputParser）
- **JSON 系列**：3个（JsonOutputParser, SimpleJsonOutputParser, PydanticOutputParser）
- **列表系列**：4个（ListOutputParser, CommaSeparatedListOutputParser, MarkdownListOutputParser, NumberedListOutputParser）
- **OpenAI Tools 系列**：3个（JsonOutputToolsParser, JsonOutputKeyToolsParser, PydanticToolsParser）
- **其他格式**：2个（StrOutputParser, XMLOutputParser）

### 5. 动态导入机制

**懒加载设计**（__init__.py:93-101）：

```python
def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result
```

**优点**：
- 减少初始加载时间
- 按需导入，节省内存
- 支持类型检查（通过 TYPE_CHECKING）

## 架构设计亮点

1. **Runnable 协议统一**：所有 OutputParser 都是 Runnable，可以无缝集成到 LCEL 中
2. **类型安全**：使用泛型 `T` 确保类型安全
3. **流式支持**：通过 `partial` 参数和 `BaseCumulativeTransformOutputParser` 支持流式解析
4. **异步优先**：所有方法都有异步版本
5. **可观测性**：通过 `_call_with_config()` 集成 LangSmith 追踪

## 与 LCEL 的集成方式

**典型用法**：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = StrOutputParser()

# LCEL 管道
chain = model | parser

# 调用
result = chain.invoke("Tell me a joke")
```

**流式用法**：

```python
# 流式输出
for chunk in chain.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## 重要注释

**从 __init__.py 的文档字符串**（__init__.py:1-17）：

```python
"""`OutputParser` classes parse the output of an LLM call into structured data.

!!! tip "Structured output"

    Output parsers emerged as an early solution to the challenge of obtaining structured
    output from LLMs.

    Today, most LLMs support [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)
    natively. In such cases, using output parsers may be unnecessary, and you should
    leverage the model's built-in capabilities for structured output. Refer to the
    [documentation of your chosen model](https://docs.langchain.com/oss/python/integrations/providers/overview)
    for guidance on how to achieve structured output directly.

    Output parsers remain valuable when working with models that do not support
    structured output natively, or when you require additional processing or validation
    of the model's output beyond its inherent capabilities.
"""
```

**关键信息**：
- OutputParser 是早期解决方案，用于从 LLM 获取结构化输出
- 现代 LLM（2025+）大多支持原生结构化输出
- OutputParser 仍然有价值：
  - 不支持结构化输出的模型
  - 需要额外处理或验证的场景

## 总结

LangChain 的 OutputParser 系统通过三层抽象设计，实现了：
1. **统一接口**：所有解析器都实现 Runnable 协议
2. **类型安全**：泛型设计确保类型正确
3. **流式支持**：部分解析和累积转换
4. **可扩展性**：易于自定义新的解析器
5. **现代化**：与 LLM 原生结构化输出共存
