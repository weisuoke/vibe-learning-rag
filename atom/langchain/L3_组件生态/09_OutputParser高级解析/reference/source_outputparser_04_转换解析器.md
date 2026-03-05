---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/output_parsers
analyzed_files:
  - transform.py
analyzed_at: 2026-02-26
knowledge_point: OutputParser高级解析
---

# 源码分析：转换解析器（BaseTransformOutputParser & BaseCumulativeTransformOutputParser）

## 分析的文件

- `langchain_core/output_parsers/transform.py` - 转换解析器基类

## 关键发现

### 1. BaseTransformOutputParser - 流式解析基类

**类定义**（transform.py:28-96）：

```python
class BaseTransformOutputParser(BaseOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    def _transform(
        self,
        input: Iterator[str | BaseMessage],
    ) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(
        self,
        input: AsyncIterator[str | BaseMessage],
    ) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield await run_in_executor(
                    None, self.parse_result, [ChatGeneration(message=chunk)]
                )
            else:
                yield await run_in_executor(
                    None, self.parse_result, [Generation(text=chunk)]
                )
```

**关键特性**：
- 继承自 `BaseOutputParser`
- 提供 `_transform()` 和 `_atransform()` 方法
- 默认实现：逐块调用 `parse_result()`
- 支持 `str` 和 `BaseMessage` 两种输入类型

### 2. transform() 方法 - 公共接口

**同步版本**（transform.py:55-74）：

```python
@override
def transform(
    self,
    input: Iterator[str | BaseMessage],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Iterator[T]:
    """Transform the input into the output format.

    Args:
        input: The input to transform.
        config: The configuration to use for the transformation.
        **kwargs: Additional keyword arguments.

    Yields:
        The transformed output.
    """
    yield from self._transform_stream_with_config(
        input, self._transform, config, run_type="parser"
    )
```

**关键发现**：
- 使用 `_transform_stream_with_config()` 包装
- 传递 `run_type="parser"` 用于可观测性
- 支持配置传递（`config`）

### 3. atransform() 方法 - 异步接口

**异步版本**（transform.py:76-96）：

```python
@override
async def atransform(
    self,
    input: AsyncIterator[str | BaseMessage],
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> AsyncIterator[T]:
    """Async transform the input into the output format.

    Args:
        input: The input to transform.
        config: The configuration to use for the transformation.
        **kwargs: Additional keyword arguments.

    Yields:
        The transformed output.
    """
    async for chunk in self._atransform_stream_with_config(
        input, self._atransform, config, run_type="parser"
    ):
        yield chunk
```

**关键发现**：
- 使用 `_atransform_stream_with_config()` 包装
- 完整的异步流式支持
- 与同步版本逻辑一致

### 4. BaseCumulativeTransformOutputParser - 累积转换解析器

**类定义**（transform.py:99-100）：

```python
class BaseCumulativeTransformOutputParser(BaseTransformOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""
```

**关键特性**：
- 继承自 `BaseTransformOutputParser`
- 专门用于累积转换（如 JSON 解析）
- 需要子类实现 `_diff()` 方法（用于 diff 模式）

**从 json.py 中的使用**：

```python
class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch
```

### 5. 流式解析的两种模式

**模式1：逐块解析（BaseTransformOutputParser）**

```python
class StrOutputParser(BaseTransformOutputParser[str]):
    def parse(self, text: str) -> str:
        return text

# 每个块直接输出
for chunk in parser.transform(stream):
    print(chunk)  # "Hello", " ", "world", "!"
```

**模式2：累积解析（BaseCumulativeTransformOutputParser）**

```python
class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    def parse_result(self, result, *, partial=False):
        # 累积解析，返回部分 JSON 对象
        return parse_partial_json(text)

# 累积输出
for chunk in parser.transform(stream):
    print(chunk)  # {}, {"name": "Alice"}, {"name": "Alice", "age": 30}
```

### 6. 与 Runnable 的集成

**_transform_stream_with_config() 方法**：

这个方法来自 `RunnableSerializable` 基类，提供：
- **配置传递**：将 `config` 传递给解析器
- **可观测性**：通过 `run_type="parser"` 集成 LangSmith 追踪
- **错误处理**：统一的错误处理机制
- **上下文管理**：管理执行上下文

### 7. 输入类型处理

**支持两种输入类型**：

1. **字符串（str）**：
   ```python
   if isinstance(chunk, BaseMessage):
       # 处理消息
   else:
       # 处理字符串
       yield self.parse_result([Generation(text=chunk)])
   ```

2. **消息（BaseMessage）**：
   ```python
   if isinstance(chunk, BaseMessage):
       yield self.parse_result([ChatGeneration(message=chunk)])
   ```

**为什么需要两种类型？**
- **字符串**：来自 LLM 的原始文本输出
- **消息**：来自 ChatModel 的结构化消息（包含角色、内容等）

### 8. 异步执行器

**run_in_executor() 使用**（transform.py:47-49）：

```python
yield await run_in_executor(
    None, self.parse_result, [ChatGeneration(message=chunk)]
)
```

**用途**：
- 在异步上下文中执行同步方法
- `None` 表示使用默认执行器（线程池）
- 避免阻塞事件循环

## 流式解析工作流程

### 场景：流式 JSON 解析

**输入流**：
```
"{\"name\"" → ": \"Alice\"" → ", \"age\": " → "30}"
```

**BaseTransformOutputParser（逐块）**：
```
chunk1: "{\"name\""        → 解析失败 → 不输出
chunk2: ": \"Alice\""      → 解析失败 → 不输出
chunk3: ", \"age\": "      → 解析失败 → 不输出
chunk4: "30}"              → 解析失败 → 不输出
```

**BaseCumulativeTransformOutputParser（累积）**：
```
buffer1: "{\"name\""                    → 部分解析 → {}
buffer2: "{\"name\": \"Alice\""         → 部分解析 → {"name": "Alice"}
buffer3: "{\"name\": \"Alice\", \"age\": " → 部分解析 → {"name": "Alice"}
buffer4: "{\"name\": \"Alice\", \"age\": 30}" → 完整解析 → {"name": "Alice", "age": 30}
```

### 关键区别

| 特性 | BaseTransformOutputParser | BaseCumulativeTransformOutputParser |
|------|---------------------------|-------------------------------------|
| 解析方式 | 逐块独立解析 | 累积缓冲区解析 |
| 输出时机 | 每个块都输出 | 累积到可解析时输出 |
| 适用场景 | 简单转换（如字符串） | 复杂解析（如 JSON） |
| 内存占用 | 低（无缓冲区） | 高（需要缓冲区） |
| 延迟 | 低（立即输出） | 中（需要累积） |

## 使用示例

### BaseTransformOutputParser 示例

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = StrOutputParser()

chain = model | parser

# 流式输出
for chunk in chain.stream("Tell me a story"):
    print(chunk, end="", flush=True)
    # 输出: "Once", " ", "upon", " ", "a", " ", "time", ...
```

### BaseCumulativeTransformOutputParser 示例

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser

# 流式输出（累积）
for chunk in chain.stream("Return a JSON with name and age"):
    print(chunk)
    # 输出: {}, {"name": "Alice"}, {"name": "Alice", "age": 30}
```

### 自定义转换解析器

```python
from langchain_core.output_parsers import BaseTransformOutputParser

class UpperCaseParser(BaseTransformOutputParser[str]):
    """Convert output to uppercase."""

    def parse(self, text: str) -> str:
        return text.upper()

# 使用
parser = UpperCaseParser()
for chunk in parser.transform(["hello", " ", "world"]):
    print(chunk)  # "HELLO", " ", "WORLD"
```

### 自定义累积解析器

```python
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser

class WordCountParser(BaseCumulativeTransformOutputParser[int]):
    """Count words in accumulated text."""

    def parse_result(self, result, *, partial=False):
        text = result[0].text
        return len(text.split())

# 使用
parser = WordCountParser()
for chunk in parser.transform(["hello", " world", " from", " AI"]):
    print(chunk)  # 1, 2, 3, 4
```

## 架构设计亮点

1. **分层设计**：
   - `BaseTransformOutputParser`：基础流式解析
   - `BaseCumulativeTransformOutputParser`：累积流式解析

2. **灵活扩展**：
   - 子类只需实现 `parse()` 或 `parse_result()`
   - 自动获得流式和异步支持

3. **Runnable 集成**：
   - 通过 `_transform_stream_with_config()` 集成
   - 自动获得配置传递和可观测性

4. **类型安全**：
   - 泛型 `T` 确保类型正确
   - 支持 `str` 和 `BaseMessage` 两种输入

5. **异步优先**：
   - 完整的异步流式支持
   - 使用 `run_in_executor()` 桥接同步方法

## 性能考虑

1. **BaseTransformOutputParser**：
   - 低延迟（立即输出）
   - 低内存（无缓冲区）
   - 适合简单转换

2. **BaseCumulativeTransformOutputParser**：
   - 中等延迟（需要累积）
   - 高内存（需要缓冲区）
   - 适合复杂解析

## 总结

转换解析器通过分层设计，实现了：
- **流式优先**：所有解析器都支持流式解析
- **灵活扩展**：易于实现自定义解析器
- **Runnable 集成**：无缝集成到 LCEL 中
- **异步支持**：完整的异步流式解析
- **两种模式**：逐块解析 vs 累积解析
