---
type: source_code_analysis
source: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/base.py
analyzed_files:
  - sourcecode/langchain/libs/text-splitters/langchain_text_splitters/base.py
analyzed_at: 2026-02-25
knowledge_point: 04_TextSplitter文本分块
---

# 源码分析：TextSplitter 基类架构

## 分析的文件
- `sourcecode/langchain/libs/text-splitters/langchain_text_splitters/base.py` - TextSplitter 基类和核心接口

## 关键发现

### 1. TextSplitter 基类设计

**继承关系**：
```python
class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""
```

- 继承自 `BaseDocumentTransformer`（文档转换器基类）
- 使用 ABC（抽象基类）定义接口规范

### 2. 核心参数

```python
def __init__(
    self,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
    keep_separator: bool | Literal["start", "end"] = False,
    add_start_index: bool = False,
    strip_whitespace: bool = True,
) -> None:
```

**参数说明**：
- `chunk_size`: 最大块大小（默认 4000）
- `chunk_overlap`: 块之间的重叠字符数（默认 200）
- `length_function`: 长度计算函数（默认 `len`，可替换为 token 计数）
- `keep_separator`: 是否保留分隔符及其位置（False/"start"/"end"）
- `add_start_index`: 是否在元数据中添加块的起始索引
- `strip_whitespace`: 是否去除文档首尾空白

**参数验证**：
```python
if chunk_size <= 0:
    raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
if chunk_overlap < 0:
    raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
if chunk_overlap > chunk_size:
    raise ValueError("Got a larger chunk overlap than chunk size")
```

### 3. 核心方法

#### 3.1 抽象方法：split_text()

```python
@abstractmethod
def split_text(self, text: str) -> list[str]:
    """Split text into multiple components."""
```

- 所有子类必须实现此方法
- 输入：文本字符串
- 输出：文本块列表

#### 3.2 create_documents()

```python
def create_documents(
    self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]:
```

**功能**：
- 将文本列表转换为 Document 对象列表
- 支持为每个文本关联元数据
- 自动调用 `split_text()` 进行分块
- 可选添加 `start_index` 到元数据

**实现细节**：
```python
for i, text in enumerate(texts):
    index = 0
    previous_chunk_len = 0
    for chunk in self.split_text(text):
        metadata = copy.deepcopy(metadatas_[i])
        if self._add_start_index:
            offset = index + previous_chunk_len - self._chunk_overlap
            index = text.find(chunk, max(0, offset))
            metadata["start_index"] = index
            previous_chunk_len = len(chunk)
        new_doc = Document(page_content=chunk, metadata=metadata)
        documents.append(new_doc)
```

#### 3.3 split_documents()

```python
def split_documents(self, documents: Iterable[Document]) -> list[Document]:
```

**功能**：
- 分割 Document 对象列表
- 提取文本和元数据
- 调用 `create_documents()` 处理

#### 3.4 _merge_splits() - 核心算法

```python
def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
```

**功能**：将小块合并成中等大小的块，同时处理重叠

**算法逻辑**：
1. 遍历所有小块
2. 累加当前块的长度
3. 如果超过 `chunk_size`，保存当前块
4. 处理重叠：从当前块开头移除元素，直到满足 `chunk_overlap` 约束
5. 继续添加新块

**关键代码**：
```python
for d in splits:
    len_ = self._length_function(d)
    if (total + len_ + separator_len > self._chunk_size):
        if len(current_doc) > 0:
            doc = self._join_docs(current_doc, separator)
            if doc is not None:
                docs.append(doc)
            # 处理重叠
            while total > self._chunk_overlap or (...):
                total -= self._length_function(current_doc[0]) + separator_len
                current_doc = current_doc[1:]
    current_doc.append(d)
    total += len_ + separator_len
```

**警告机制**：
```python
if total > self._chunk_size:
    logger.warning(
        "Created a chunk of size %d, which is longer than the specified %d",
        total, self._chunk_size,
    )
```

### 4. 工厂方法

#### 4.1 from_tiktoken_encoder()

```python
@classmethod
def from_tiktoken_encoder(
    cls,
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | AbstractSet[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self:
```

**功能**：
- 使用 tiktoken 编码器计算长度
- 支持按模型名称或编码名称创建
- 返回使用 token 计数的 TextSplitter 实例

**实现**：
```python
if model_name is not None:
    enc = tiktoken.encoding_for_model(model_name)
else:
    enc = tiktoken.get_encoding(encoding_name)

def _tiktoken_encoder(text: str) -> int:
    return len(enc.encode(text, allowed_special=..., disallowed_special=...))

return cls(length_function=_tiktoken_encoder, **kwargs)
```

#### 4.2 from_huggingface_tokenizer()

```python
@classmethod
def from_huggingface_tokenizer(
    cls, tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter:
```

**功能**：
- 使用 HuggingFace tokenizer 计算长度
- 支持任何 PreTrainedTokenizerBase 实例

**实现**：
```python
def _huggingface_tokenizer_length(text: str) -> int:
    return len(tokenizer.tokenize(text))

return cls(length_function=_huggingface_tokenizer_length, **kwargs)
```

### 5. TokenTextSplitter 类

```python
class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""
```

**特点**：
- 专门用于基于 token 的分块
- 使用 tiktoken 编码器
- 实现了 `split_text()` 方法

**核心方法**：
```python
def split_text(self, text: str) -> list[str]:
    def _encode(_text: str) -> list[int]:
        return self._tokenizer.encode(
            _text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    tokenizer = Tokenizer(
        chunk_overlap=self._chunk_overlap,
        tokens_per_chunk=self._chunk_size,
        decode=self._tokenizer.decode,
        encode=_encode,
    )

    return split_text_on_tokens(text=text, tokenizer=tokenizer)
```

### 6. split_text_on_tokens() 函数

```python
def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
```

**功能**：
- 基于 token 的分块算法
- 处理 token 级别的重叠

**算法**：
```python
splits: list[str] = []
input_ids = tokenizer.encode(text)
start_idx = 0

while start_idx < len(input_ids):
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    decoded = tokenizer.decode(chunk_ids)
    if decoded:
        splits.append(decoded)
    if cur_idx == len(input_ids):
        break
    start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
```

### 7. Language 枚举

```python
class Language(str, Enum):
    """Enum of the programming languages."""
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    PYTHON = "python"
    # ... 20+ 种语言
```

**用途**：
- 为 RecursiveCharacterTextSplitter 提供语言特定的分隔符
- 支持代码分块的语法感知

### 8. Tokenizer 数据类

```python
@dataclass(frozen=True)
class Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[list[int]], str]
    encode: Callable[[str], list[int]]
```

**特点**：
- 不可变数据类（frozen=True）
- 封装编码/解码函数
- 用于 token 级别的分块

## 设计模式

### 1. 模板方法模式
- `TextSplitter` 定义算法骨架（`_merge_splits`）
- 子类实现具体的 `split_text()` 方法

### 2. 策略模式
- `length_function` 参数允许替换长度计算策略
- 支持字符计数、token 计数等不同策略

### 3. 工厂方法模式
- `from_tiktoken_encoder()` 和 `from_huggingface_tokenizer()` 提供便捷的创建方式

## 关键设计决策

### 1. 为什么使用 length_function 参数？
- **灵活性**：支持不同的长度计算方式（字符、token、自定义）
- **可扩展性**：无需修改基类即可支持新的计量方式
- **性能**：避免在基类中硬编码特定的计算逻辑

### 2. 为什么 chunk_overlap 很重要？
- **上下文连续性**：避免在句子或段落中间切断
- **检索质量**：重叠部分提高检索召回率
- **语义完整性**：确保每个块包含足够的上下文

### 3. 为什么需要 keep_separator？
- **结构保留**：某些场景需要保留分隔符（如代码、Markdown）
- **灵活性**：支持将分隔符放在块的开头或结尾
- **语义理解**：分隔符本身可能包含重要信息

### 4. 为什么使用抽象基类？
- **接口规范**：强制子类实现 `split_text()` 方法
- **多态性**：所有 TextSplitter 可以互换使用
- **类型安全**：提供明确的类型提示

## 性能考虑

### 1. 内存效率
- 使用生成器模式（`split_text()` 返回列表，但可以流式处理）
- `_merge_splits()` 逐步构建块，避免一次性加载所有数据

### 2. 时间复杂度
- `_merge_splits()`: O(n)，其中 n 是小块数量
- `split_text()`: 取决于具体实现

### 3. 优化建议
- 对于大文件，考虑使用流式处理
- 缓存 tokenizer 实例，避免重复创建
- 使用批处理提高 token 编码效率

## 与 LangChain 生态集成

### 1. BaseDocumentTransformer
- TextSplitter 实现了 `transform_documents()` 方法
- 可以与其他文档转换器组合使用

### 2. Document 对象
- 输出标准的 `Document` 对象
- 包含 `page_content` 和 `metadata`
- 与 LangChain 的其他组件无缝集成

### 3. 元数据管理
- 支持保留原始文档的元数据
- 可选添加 `start_index` 用于溯源
- 元数据深拷贝避免意外修改

## 常见使用模式

### 1. 基本使用
```python
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)
```

### 2. 创建 Document 对象
```python
docs = splitter.create_documents([text1, text2], metadatas=[meta1, meta2])
```

### 3. 分割现有 Document
```python
split_docs = splitter.split_documents(documents)
```

### 4. 使用 token 计数
```python
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,
    chunk_overlap=200
)
```

## 潜在问题和注意事项

### 1. 块大小超限
- 如果单个小块超过 `chunk_size`，会触发警告
- 需要调整分隔符或 `chunk_size` 参数

### 2. 重叠处理
- `chunk_overlap` 必须小于 `chunk_size`
- 重叠过大会导致大量重复内容

### 3. 分隔符选择
- 不同的分隔符会影响分块质量
- 需要根据文本类型选择合适的分隔符

### 4. 元数据管理
- 元数据会被深拷贝，注意内存使用
- `start_index` 计算依赖于 `text.find()`，可能不准确

## 扩展点

### 1. 自定义 length_function
```python
def custom_length(text: str) -> int:
    # 自定义长度计算逻辑
    return len(text.split())

splitter = CharacterTextSplitter(length_function=custom_length)
```

### 2. 自定义 TextSplitter 子类
```python
class MyTextSplitter(TextSplitter):
    def split_text(self, text: str) -> list[str]:
        # 自定义分块逻辑
        return custom_split_logic(text)
```

### 3. 自定义元数据处理
```python
# 在 create_documents() 后添加自定义元数据
docs = splitter.create_documents(texts)
for doc in docs:
    doc.metadata["custom_field"] = compute_custom_value(doc.page_content)
```
