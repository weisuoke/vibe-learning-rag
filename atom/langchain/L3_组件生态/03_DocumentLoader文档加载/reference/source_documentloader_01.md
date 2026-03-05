---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/document_loaders/base.py
  - libs/core/langchain_core/document_loaders/blob_loaders.py
  - libs/core/langchain_core/documents/base.py
  - libs/core/langchain_core/document_loaders/langsmith.py
analyzed_at: 2026-02-24
knowledge_point: 03_DocumentLoader文档加载
---

# 源码分析：DocumentLoader 核心架构

## 分析的文件

- `libs/core/langchain_core/document_loaders/base.py` - BaseLoader 和 BaseBlobParser 抽象接口
- `libs/core/langchain_core/document_loaders/blob_loaders.py` - BlobLoader 抽象接口
- `libs/core/langchain_core/documents/base.py` - Document 和 Blob 数据结构
- `libs/core/langchain_core/document_loaders/langsmith.py` - LangSmithLoader 实现示例

## 关键发现

### 1. 三层抽象架构

LangChain 的 DocumentLoader 采用了清晰的三层抽象设计:

```
Blob (原始数据) → BlobLoader (加载) → BaseBlobParser (解析) → Document (结构化文档)
                                    ↓
                              BaseLoader (组合加载器)
```

**设计意图**:
- **Blob**: 表示原始数据(bytes/str),可以来自内存或文件
- **BlobLoader**: 负责懒加载原始数据流
- **BaseBlobParser**: 负责将 Blob 解析为 Document
- **BaseLoader**: 可以直接实现,也可以组合 BlobLoader + BlobParser

### 2. BaseLoader 接口设计

```python
class BaseLoader(ABC):
    """Interface for document loader."""

    def load(self) -> list[Document]:
        """Load data into Document objects."""
        return list(self.lazy_load())

    async def aload(self) -> list[Document]:
        """Async load data into Document objects."""
        return [document async for document in self.alazy_load()]

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Document."""
        # 子类应该实现这个方法

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy loader for Document."""
        # 默认实现:在线程池中运行 lazy_load

    def load_and_split(self, text_splitter: TextSplitter | None = None) -> list[Document]:
        """Load Document and split into chunks."""
        # 集成 TextSplitter
```

**关键设计决策**:

1. **懒加载优先**: `lazy_load()` 是核心方法,`load()` 只是便利方法
2. **不要覆盖 load()**: 文档明确警告不要覆盖 `load()`,应该实现 `lazy_load()`
3. **异步支持**: 提供 `aload()` 和 `alazy_load()` 异步版本
4. **集成 TextSplitter**: `load_and_split()` 方法直接集成文本分块功能

### 3. Blob 数据结构

```python
class Blob(BaseMedia):
    """Raw data abstraction for document loading."""

    data: bytes | str | None = None
    """Raw data associated with the Blob."""

    mimetype: str | None = None
    """MIME type, not to be confused with a file extension."""

    encoding: str = "utf-8"
    """Encoding to use if decoding the bytes into a string."""

    path: PathLike | None = None
    """Location where the original content was found."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content."""
```

**关键特性**:

1. **懒加载支持**: `data` 可以为 None,通过 `path` 懒加载
2. **MIME 类型**: 支持 MIME 类型识别
3. **编码处理**: 默认 UTF-8,可自定义
4. **元数据**: 支持任意元数据

**工厂方法**:

```python
# 从文件路径创建(懒加载)
blob = Blob.from_path("path/to/file.txt")

# 从内存数据创建
blob = Blob.from_data("Hello, world!")
```

**读取方法**:

```python
# 读取为字符串
content = blob.as_string()

# 读取为字节
bytes_data = blob.as_bytes()

# 读取为字节流(上下文管理器)
with blob.as_bytes_io() as f:
    data = f.read()
```

### 4. Document 数据结构

```python
class Document(BaseMedia):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content."""

    id: str | None = Field(default=None)
    """An optional identifier for the document."""
```

**关键特性**:

1. **简单结构**: 只有 `page_content` 和 `metadata`
2. **可选 ID**: 支持唯一标识符(未来可能成为必需)
3. **序列化支持**: 实现了 LangChain 的序列化协议

**与 Blob 的区别**:

- **Blob**: 原始数据(bytes/str),用于加载阶段
- **Document**: 结构化文本,用于检索和处理阶段

### 5. BlobLoader 接口

```python
class BlobLoader(ABC):
    """Abstract interface for blob loaders implementation."""

    @abstractmethod
    def yield_blobs(self) -> Iterator[Blob]:
        """A lazy loader for raw data represented by LangChain's Blob object."""
```

**设计意图**:

- 专注于加载原始数据
- 返回 Blob 流(懒加载)
- 可以从文件系统、网络、数据库等加载

### 6. BaseBlobParser 接口

```python
class BaseBlobParser(ABC):
    """Abstract interface for blob parsers."""

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface."""

    def parse(self, blob: Blob) -> list[Document]:
        """Eagerly parse the blob into a Document or list of Document objects."""
        return list(self.lazy_parse(blob))
```

**设计意图**:

- 专注于解析 Blob 为 Document
- 支持一个 Blob 解析为多个 Document(如 PDF 的多页)
- 懒加载优先

### 7. LangSmithLoader 实现示例

```python
class LangSmithLoader(BaseLoader):
    """Load LangSmith Dataset examples as Document objects."""

    def __init__(
        self,
        *,
        dataset_id: uuid.UUID | str | None = None,
        dataset_name: str | None = None,
        content_key: str = "",
        format_content: Callable[..., str] | None = None,
        client: LangSmithClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        # 初始化 LangSmith 客户端
        self._client = client or LangSmithClient(**client_kwargs)
        self.content_key = list(content_key.split(".")) if content_key else []
        self.format_content = format_content or _stringify
        # ... 其他参数

    @override
    def lazy_load(self) -> Iterator[Document]:
        """实现懒加载逻辑"""
        for example in self._client.list_examples(
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            # ... 其他参数
        ):
            # 提取内容
            content: Any = example.inputs
            for key in self.content_key:
                content = content[key]
            content_str = self.format_content(content)

            # 构建元数据
            metadata = pydantic_to_dict(example)

            # 生成 Document
            yield Document(content_str, metadata=metadata)
```

**实现要点**:

1. **只实现 lazy_load()**: 不需要实现 `load()`,基类会自动处理
2. **使用生成器**: `yield` 而不是 `return`,实现真正的懒加载
3. **元数据处理**: 将原始数据转换为字典存储在 metadata 中
4. **内容格式化**: 支持自定义内容格式化函数

### 8. load_and_split 集成

```python
def load_and_split(
    self, text_splitter: TextSplitter | None = None
) -> list[Document]:
    """Load Document and split into chunks."""
    if text_splitter is None:
        if not _HAS_TEXT_SPLITTERS:
            msg = (
                "Unable to import from langchain_text_splitters. Please specify "
                "text_splitter or install langchain_text_splitters with "
                "`pip install -U langchain-text-splitters`."
            )
            raise ImportError(msg)
        text_splitter_ = RecursiveCharacterTextSplitter()
    else:
        text_splitter_ = text_splitter
    docs = self.load()
    return text_splitter_.split_documents(docs)
```

**设计意图**:

- 提供一站式加载和分块功能
- 默认使用 `RecursiveCharacterTextSplitter`
- 支持自定义 TextSplitter

**警告**: 文档标记为 "deprecated",建议分开调用

## 架构设计优势

### 1. 职责分离

- **BlobLoader**: 只负责加载原始数据
- **BaseBlobParser**: 只负责解析数据
- **BaseLoader**: 可以组合上述两者,也可以直接实现

**好处**: 可以复用 Parser,例如同一个 PDF Parser 可以用于不同的 BlobLoader

### 2. 懒加载优先

所有接口都强调懒加载:
- `lazy_load()` 而不是 `load()`
- `yield_blobs()` 而不是 `get_blobs()`
- `lazy_parse()` 而不是 `parse()`

**好处**: 避免大文件一次性加载到内存,支持流式处理

### 3. 异步支持

提供异步版本的所有方法:
- `aload()` / `alazy_load()`

**好处**: 支持高并发场景,提升性能

### 4. 元数据优先

Document 和 Blob 都支持元数据:
- 文件路径
- MIME 类型
- 自定义元数据

**好处**: 支持混合检索、过滤、溯源等高级功能

## 常见实现模式

### 模式 1: 直接实现 BaseLoader

适用于简单场景,直接从数据源加载:

```python
class MyLoader(BaseLoader):
    def lazy_load(self) -> Iterator[Document]:
        # 直接从数据源加载并生成 Document
        for item in data_source:
            yield Document(page_content=item.text, metadata=item.meta)
```

### 模式 2: 组合 BlobLoader + BaseBlobParser

适用于复杂场景,分离加载和解析:

```python
class MyBlobLoader(BlobLoader):
    def yield_blobs(self) -> Iterator[Blob]:
        # 只负责加载原始数据
        for file_path in self.file_paths:
            yield Blob.from_path(file_path)

class MyBlobParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        # 只负责解析 Blob
        content = blob.as_string()
        # 解析逻辑...
        yield Document(page_content=parsed_content, metadata=blob.metadata)

# 使用时组合
loader = MyBlobLoader(...)
parser = MyBlobParser(...)
for blob in loader.yield_blobs():
    for doc in parser.lazy_parse(blob):
        # 处理 Document
```

## 性能优化建议

1. **使用懒加载**: 优先实现 `lazy_load()` 而不是 `load()`
2. **使用异步**: 对于 I/O 密集型操作,使用 `alazy_load()`
3. **批量处理**: 在 `lazy_load()` 中使用批量 API
4. **元数据最小化**: 只保留必要的元数据,避免内存浪费

## 总结

LangChain 的 DocumentLoader 架构设计体现了以下原则:

1. **职责分离**: BlobLoader(加载) + BlobParser(解析) + BaseLoader(组合)
2. **懒加载优先**: 所有接口都支持流式处理
3. **异步支持**: 提供异步版本提升性能
4. **元数据优先**: 支持混合检索和溯源
5. **可组合性**: 可以灵活组合不同的加载器和解析器

这种设计使得 LangChain 能够支持各种数据源和格式,同时保持代码的简洁和可维护性。
