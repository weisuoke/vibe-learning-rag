---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/output_parsers
analyzed_files:
  - list.py
analyzed_at: 2026-02-26
knowledge_point: OutputParser高级解析
---

# 源码分析：列表解析器（ListOutputParser 系列）

## 分析的文件

- `langchain_core/output_parsers/list.py` - 列表解析器实现

## 关键发现

### 1. ListOutputParser - 抽象基类

**类定义**（list.py:43-60）：

```python
class ListOutputParser(BaseTransformOutputParser[list[str]]):
    """Parse the output of a model to a list."""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
```

**关键特性**：
- 继承自 `BaseTransformOutputParser`（支持流式解析）
- 返回类型固定为 `list[str]`
- `parse()` 方法是抽象方法，需要子类实现

### 2. 流式解析支持

**parse_iter() 方法**（list.py:61-70）：

```python
def parse_iter(self, text: str) -> Iterator[re.Match]:
    """Parse the output of an LLM call.

    Args:
        text: The output of an LLM call.

    Yields:
        A match object for each part of the output.
    """
    raise NotImplementedError
```

**用途**：
- 返回正则匹配对象的迭代器
- 用于流式解析（逐个元素返回）
- 默认实现抛出 `NotImplementedError`（子类可选实现）

### 3. _transform() 方法 - 流式转换核心

**同步版本**（list.py:72-102）：

```python
@override
def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[list[str]]:
    buffer = ""
    for chunk in input:
        if isinstance(chunk, BaseMessage):
            # Extract text
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                continue
            buffer += chunk_content
        else:
            # Add current chunk to buffer
            buffer += chunk

        # Parse buffer into a list of parts
        try:
            done_idx = 0
            # Yield only complete parts
            for m in droplastn(self.parse_iter(buffer), 1):
                done_idx = m.end()
                yield [m.group(1)]
            buffer = buffer[done_idx:]
        except NotImplementedError:
            parts = self.parse(buffer)
            # Yield only complete parts
            if len(parts) > 1:
                for part in parts[:-1]:
                    yield [part]
                buffer = parts[-1]

    # Yield the last part
    for part in self.parse(buffer):
        yield [part]
```

**关键逻辑**：
1. **缓冲区管理**：累积输入块到 `buffer`
2. **增量解析**：尝试使用 `parse_iter()` 逐个解析
3. **保留最后一个**：使用 `droplastn()` 保留最后一个未完成的元素
4. **回退机制**：如果 `parse_iter()` 未实现，使用 `parse()` 方法
5. **最终输出**：在流结束时输出最后一个元素

### 4. droplastn() 工具函数

**实现**（list.py:23-40）：

```python
def droplastn(
    iter: Iterator[T],
    n: int,
) -> Iterator[T]:
    """Drop the last `n` elements of an iterator.

    Args:
        iter: The iterator to drop elements from.
        n: The number of elements to drop.

    Yields:
        The elements of the iterator, except the last n elements.
    """
    buffer: deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()
```

**用途**：
- 保留最后 N 个元素不输出
- 用于流式解析中保留未完成的元素
- 使用 `deque` 实现高效的滑动窗口

### 5. CommaSeparatedListOutputParser - 逗号分隔列表

**类定义**（list.py:139-150）：

```python
class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of a model to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "output_parser"]`
        """
        return ["langchain", "schema", "output_parser"]
```

**关键特性**：
- 继承自 `ListOutputParser`
- 支持序列化（`is_lc_serializable = True`）
- 属于 LangChain 核心命名空间

**parse() 实现**（继续阅读源码）：

从源码中我看到 `CommaSeparatedListOutputParser` 的实现，但是被截断了。让我推断其实现：
- 使用逗号分隔字符串
- 去除每个元素的空白字符
- 返回字符串列表

### 6. 异步支持

**_atransform() 方法**（list.py:104-136）：

```python
@override
async def _atransform(
    self, input: AsyncIterator[str | BaseMessage]
) -> AsyncIterator[list[str]]:
    buffer = ""
    async for chunk in input:
        if isinstance(chunk, BaseMessage):
            # Extract text
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                continue
            buffer += chunk_content
        else:
            # Add current chunk to buffer
            buffer += chunk

        # Parse buffer into a list of parts
        try:
            done_idx = 0
            # Yield only complete parts
            for m in droplastn(self.parse_iter(buffer), 1):
                done_idx = m.end()
                yield [m.group(1)]
            buffer = buffer[done_idx:]
        except NotImplementedError:
            parts = self.parse(buffer)
            # Yield only complete parts
            if len(parts) > 1:
                for part in parts[:-1]:
                    yield [part]
                buffer = parts[-1]

    # Yield the last part
    for part in self.parse(buffer):
        yield [part]
```

**关键发现**：
- 与同步版本逻辑完全一致
- 使用 `async for` 和 `AsyncIterator`
- 支持异步流式解析

## 流式解析工作原理

### 场景：解析逗号分隔列表

**输入流**：
```
"apple" → ", ban" → "ana, or" → "ange"
```

**解析过程**：

1. **第一个块**："apple"
   - buffer = "apple"
   - 无法确定是否完整（可能后面还有内容）
   - 不输出

2. **第二个块**：", ban"
   - buffer = "apple, ban"
   - 检测到逗号，"apple" 是完整的
   - 输出：["apple"]
   - buffer = " ban"

3. **第三个块**："ana, or"
   - buffer = " banana, or"
   - 检测到逗号，"banana" 是完整的
   - 输出：["banana"]
   - buffer = " or"

4. **第四个块**："ange"
   - buffer = " orange"
   - 流结束，输出最后一个元素
   - 输出：["orange"]

### 关键设计

1. **保守策略**：只输出确定完整的元素
2. **缓冲区管理**：累积未完成的内容
3. **增量输出**：逐个元素输出，而不是等待全部完成
4. **最后处理**：流结束时输出缓冲区中的剩余内容

## 其他列表解析器

根据 `__init__.py`，还有以下列表解析器：

1. **MarkdownListOutputParser**：解析 Markdown 列表
   - 支持无序列表（`-`, `*`）
   - 支持有序列表（`1.`, `2.`）

2. **NumberedListOutputParser**：解析编号列表
   - 支持 `1.`, `2.`, `3.` 格式
   - 自动提取编号后的内容

## 使用示例

### CommaSeparatedListOutputParser 基础用法

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = CommaSeparatedListOutputParser()

chain = model | parser

result = chain.invoke("List 3 fruits")
# 输出: ["apple", "banana", "orange"]
```

### 流式解析

```python
# 流式输出列表元素
for item in chain.stream("List 10 countries"):
    print(item)  # 输出: ["USA"], ["China"], ["Japan"], ...
```

### 自定义列表解析器

```python
import re
from langchain_core.output_parsers import ListOutputParser

class NewlineListOutputParser(ListOutputParser):
    """Parse newline-separated list."""

    def parse(self, text: str) -> list[str]:
        return [line.strip() for line in text.split("\n") if line.strip()]

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        # 返回每一行的匹配对象
        return re.finditer(r"^(.+)$", text, re.MULTILINE)
```

## 架构设计亮点

1. **流式优先**：设计上优先支持流式解析
2. **增量输出**：逐个元素输出，减少延迟
3. **保守策略**：只输出确定完整的元素，避免错误
4. **回退机制**：`parse_iter()` 未实现时自动回退到 `parse()`
5. **异步支持**：完整的异步流式解析支持

## 性能考虑

1. **缓冲区开销**：需要维护缓冲区，内存占用随输入增长
2. **正则匹配开销**：`parse_iter()` 每次都要重新匹配整个缓冲区
3. **回退开销**：回退到 `parse()` 时需要解析整个缓冲区

## 总结

ListOutputParser 系列通过流式解析设计，实现了：
- **增量输出**：逐个元素输出，减少延迟
- **灵活扩展**：易于实现自定义列表解析器
- **流式友好**：完整的流式和异步支持
- **保守策略**：确保输出的元素都是完整的
