# 核心概念 1: Document 与 Blob 的区别

> 理解 RAG 数据流的两个层次

---

## 核心问题

在 LangChain 的 DocumentLoader 架构中,为什么需要 **Blob** 和 **Document** 两个数据结构?它们有什么区别?

---

## 数据结构对比

### Blob - 原始数据层

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
- `data` 可以是 `bytes` 或 `str` 或 `None`(懒加载)
- 支持 MIME 类型识别
- 支持编码配置
- 可以通过 `path` 懒加载

### Document - 结构化文本层

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
- `page_content` 必须是 `str`(纯文本)
- 支持唯一标识符 `id`
- 实现了 LangChain 的序列化协议

---

## 核心区别

### 1. 数据类型

**Blob**:
```python
# 可以是二进制数据
blob = Blob(data=b'\x89PNG\r\n\x1a\n...')  # PNG 图片

# 可以是文本数据
blob = Blob(data="Hello, world!")

# 可以是 None(懒加载)
blob = Blob(path="file.pdf", data=None)
```

**Document**:
```python
# 只能是文本
doc = Document(page_content="Hello, world!")

# 不能是二进制
# doc = Document(page_content=b'...')  # ❌ 类型错误
```

### 2. 使用阶段

**Blob**: 加载阶段
```
文件系统/网络/数据库
    ↓ BlobLoader
Blob (原始数据)
    ↓ BaseBlobParser
Document (结构化文本)
```

**Document**: 处理阶段
```
Document
    ↓ TextSplitter
Document 分块
    ↓ Embedding
向量化
    ↓ VectorStore
存储和检索
```

### 3. 职责范围

**Blob**:
- 表示原始数据
- 处理编码问题
- 支持懒加载
- 识别 MIME 类型

**Document**:
- 表示结构化文本
- 用于检索和处理
- 支持序列化
- 用于 RAG 管道

---

## 为什么需要两个层次?

### 原因 1: 分离关注点

**问题**: 如果只有 Document,如何处理二进制数据?

```python
# 假设只有 Document
class Document:
    content: str | bytes  # 混合类型

# 问题:
# 1. 如何知道是文本还是二进制?
# 2. 如何处理编码?
# 3. 如何懒加载?
```

**解决**: 分离 Blob(原始数据) 和 Document(文本数据)

```python
# Blob 处理原始数据
blob = Blob.from_path("file.pdf")  # 二进制数据

# Parser 转换为文本
parser = PDFParser()
docs = parser.lazy_parse(blob)  # 文本数据
```

### 原因 2: 支持懒加载

**Blob 的懒加载**:
```python
# 创建 Blob 时不加载数据
blob = Blob.from_path("large_file.pdf")  # data=None

# 需要时才加载
content = blob.as_bytes()  # 此时才读取文件
```

**Document 不需要懒加载**:
```python
# Document 已经是文本,直接使用
doc = Document(page_content="...")
print(doc.page_content)  # 直接访问
```

### 原因 3: 复用解析逻辑

**Blob 可以来自不同来源**:
```python
# 从本地文件
blob1 = Blob.from_path("file.pdf")

# 从网络
blob2 = Blob.from_data(requests.get(url).content)

# 从 S3
blob3 = Blob.from_data(s3_client.download(key))

# 统一的 Parser
parser = PDFParser()
docs1 = parser.lazy_parse(blob1)
docs2 = parser.lazy_parse(blob2)
docs3 = parser.lazy_parse(blob3)
```

---

## 实际应用

### 场景 1: PDF 文档加载

```python
from langchain_core.documents import Blob
from langchain_community.document_loaders.parsers import PyPDFParser

# 步骤 1: 创建 Blob(原始数据)
blob = Blob.from_path("report.pdf")
print(f"MIME 类型: {blob.mimetype}")  # application/pdf
print(f"数据类型: {type(blob.data)}")  # None (懒加载)

# 步骤 2: 解析为 Document(文本数据)
parser = PyPDFParser()
docs = list(parser.lazy_parse(blob))

# 步骤 3: 使用 Document
for doc in docs:
    print(f"页面内容: {doc.page_content[:100]}...")
    print(f"元数据: {doc.metadata}")
```

### 场景 2: 图片 OCR

```python
# Blob 可以表示图片(二进制)
blob = Blob.from_path("image.png")
print(f"MIME 类型: {blob.mimetype}")  # image/png

# OCR Parser 将图片转换为文本
ocr_parser = OCRParser()
docs = list(ocr_parser.lazy_parse(blob))

# Document 包含 OCR 提取的文本
for doc in docs:
    print(f"OCR 文本: {doc.page_content}")
```

### 场景 3: 网页抓取

```python
import requests

# 步骤 1: 抓取 HTML(原始数据)
html_content = requests.get("https://example.com").text
blob = Blob.from_data(
    data=html_content,
    mimetype="text/html",
    metadata={"source": "https://example.com"}
)

# 步骤 2: 解析 HTML 为文本
html_parser = BSHTMLParser()
docs = list(html_parser.lazy_parse(blob))

# 步骤 3: 使用纯文本
for doc in docs:
    print(f"网页文本: {doc.page_content[:100]}...")
```

---

## 类比理解

### 前端开发类比

**Blob** 就像前端的 **Blob 对象**:

```javascript
// 前端 Blob - 原始数据
const blob = new Blob([data], { type: 'application/pdf' })

// 读取为文本
const text = await blob.text()

// 读取为二进制
const arrayBuffer = await blob.arrayBuffer()
```

**Document** 就像前端的 **解析后的数据**:

```javascript
// 解析后的数据 - 结构化
const parsedData = {
  content: "Hello, world!",
  metadata: { source: "file.pdf" }
}
```

### 日常生活类比

**Blob** 就像 **包裹**:
- 可以是任何东西(书、衣服、电子产品)
- 需要打开才知道内容
- 有标签(MIME 类型)

**Document** 就像 **打开后的书**:
- 已经是可读的文本
- 可以直接阅读
- 有页码和章节(元数据)

---

## 工厂方法

### Blob 的创建方法

**从文件路径创建(懒加载)**:
```python
blob = Blob.from_path(
    "file.txt",
    encoding="utf-8",
    mime_type="text/plain",
    metadata={"source": "local"}
)
```

**从内存数据创建**:
```python
blob = Blob.from_data(
    data="Hello, world!",
    encoding="utf-8",
    mime_type="text/plain",
    metadata={"source": "memory"}
)
```

### Blob 的读取方法

**读取为字符串**:
```python
content = blob.as_string()  # str
```

**读取为字节**:
```python
bytes_data = blob.as_bytes()  # bytes
```

**读取为字节流**:
```python
with blob.as_bytes_io() as f:
    data = f.read()  # 流式读取
```

### Document 的创建方法

**直接创建**:
```python
doc = Document(
    page_content="Hello, world!",
    metadata={"source": "file.txt", "page": 1},
    id="doc-001"
)
```

---

## 常见误区

### ❌ 误区 1: Blob 和 Document 可以互换

**错误**:
```python
# 试图用 Document 表示二进制数据
doc = Document(page_content=b'\x89PNG...')  # ❌ 类型错误
```

**正确**:
```python
# 用 Blob 表示二进制数据
blob = Blob(data=b'\x89PNG...')

# 用 Document 表示文本数据
doc = Document(page_content="Hello, world!")
```

### ❌ 误区 2: Blob 只能表示文件

**错误**: 认为 Blob 只能从文件创建

**正确**: Blob 可以从任何数据源创建
```python
# 从文件
blob1 = Blob.from_path("file.txt")

# 从内存
blob2 = Blob.from_data("Hello")

# 从网络
blob3 = Blob.from_data(requests.get(url).content)
```

### ❌ 误区 3: Document 必须有 id

**错误**: 认为 Document 必须提供 id

**正确**: id 是可选的
```python
# 没有 id
doc1 = Document(page_content="Hello")

# 有 id
doc2 = Document(page_content="Hello", id="doc-001")
```

---

## 设计优势

### 1. 类型安全

```python
# Blob 可以是任何类型
blob: Blob = Blob(data=b'...')  # bytes
blob: Blob = Blob(data="...")   # str

# Document 只能是文本
doc: Document = Document(page_content="...")  # str
# doc: Document = Document(page_content=b'...')  # ❌ 类型错误
```

### 2. 职责清晰

```python
# BlobLoader 只负责加载原始数据
class MyBlobLoader(BlobLoader):
    def yield_blobs(self) -> Iterator[Blob]:
        # 只返回 Blob
        yield Blob.from_path("file.pdf")

# BaseBlobParser 只负责解析
class MyParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        # 只返回 Document
        yield Document(page_content=parse(blob))
```

### 3. 可组合性

```python
# 不同的 Loader 可以用同一个 Parser
local_loader = LocalBlobLoader(...)
s3_loader = S3BlobLoader(...)
http_loader = HTTPBlobLoader(...)

pdf_parser = PDFParser()

# 组合使用
for blob in local_loader.yield_blobs():
    for doc in pdf_parser.lazy_parse(blob):
        process(doc)

for blob in s3_loader.yield_blobs():
    for doc in pdf_parser.lazy_parse(blob):
        process(doc)
```

---

## 总结

### 核心区别

| 特性 | Blob | Document |
|------|------|----------|
| 数据类型 | bytes/str/None | str |
| 使用阶段 | 加载阶段 | 处理阶段 |
| 职责 | 原始数据表示 | 结构化文本表示 |
| 懒加载 | 支持 | 不需要 |
| MIME 类型 | 支持 | 不支持 |
| 序列化 | 不支持 | 支持 |

### 设计原则

1. **分离关注点**: Blob 处理原始数据,Document 处理文本
2. **类型安全**: Blob 可以是任何类型,Document 只能是文本
3. **可组合性**: BlobLoader + BlobParser 可以灵活组合

---

## 下一步

理解了 Document 与 Blob 的区别后,建议:

1. **03_核心概念_2_BaseLoader接口设计.md** - 理解加载器接口
2. **03_核心概念_3_BlobLoader与BlobParser分离.md** - 理解职责分离
3. **实战代码系列** - 实践 Blob 和 Document 的使用

---

**数据来源**:
- [来源: reference/source_documentloader_01.md | LangChain 源码分析]
- [来源: sourcecode/langchain/libs/core/langchain_core/documents/base.py | Blob 和 Document 数据结构]
