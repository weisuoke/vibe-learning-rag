# 实战代码 - 场景 8: 自定义 Loader 实现

> 掌握如何实现自己的 DocumentLoader

---

## 概述

虽然 LangChain 提供了丰富的文档加载器，但在实际项目中，我们经常需要处理特殊的数据源或格式。理解如何实现自定义 Loader 是掌握 LangChain 的关键技能。

**本场景涵盖**:
- BaseLoader 接口实现
- 最小可用实现
- 完整功能实现
- 异步 Loader 实现
- API 数据源 Loader
- 数据库 Loader
- 流式 Loader
- 生产级 Loader

---

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install langchain langchain-community langchain-core

# 数据库支持
pip install sqlalchemy psycopg2-binary

# API 支持
pip install requests aiohttp
```

### 导入模块

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, AsyncIterator, List, Optional
import asyncio
from pathlib import Path
```

---

## 场景 1: 最小可用实现

### 只实现 lazy_load()

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator

class SimpleTextLoader(BaseLoader):
    """最简单的文本加载器"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """只需要实现这一个方法"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # 跳过空行
                    yield Document(
                        page_content=line.strip(),
                        metadata={
                            "source": self.file_path,
                            "line": line_num
                        }
                    )

# 使用
loader = SimpleTextLoader("example.txt")

# 自动获得所有方法
documents = loader.load()  # 自动可用
print(f"加载了 {len(documents)} 个文档")

# 懒加载
for doc in loader.lazy_load():
    print(f"行 {doc.metadata['line']}: {doc.page_content[:50]}")
```

**关键点**:
- 只需实现 `lazy_load()` 方法
- 其他方法（load、aload、alazy_load）自动可用
- 使用生成器逐个返回文档

---

## 场景 2: 完整功能实现

### 添加错误处理和元数据

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTextLoader(BaseLoader):
    """增强的文本加载器，带错误处理和丰富元数据"""

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        skip_empty_lines: bool = True
    ):
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.skip_empty_lines = skip_empty_lines

        # 验证文件存在
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

    def lazy_load(self) -> Iterator[Document]:
        """懒加载文档，带错误处理"""
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    # 跳过空行
                    if self.skip_empty_lines and not line.strip():
                        continue

                    # 创建文档
                    doc = Document(
                        page_content=line.strip(),
                        metadata=self._create_metadata(line_num)
                    )

                    yield doc

        except UnicodeDecodeError as e:
            logger.error(f"编码错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            raise

    def _create_metadata(self, line_num: int) -> dict:
        """创建丰富的元数据"""
        return {
            "source": str(self.file_path),
            "line": line_num,
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
            "encoding": self.encoding
        }

# 使用
loader = EnhancedTextLoader("example.txt", encoding="utf-8")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"元数据: {documents[0].metadata}")
```

---

## 场景 3: 异步 Loader 实现

### 实现真正的异步加载

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, AsyncIterator
import aiofiles
import asyncio

class AsyncTextLoader(BaseLoader):
    """异步文本加载器"""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        """同步版本"""
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    yield Document(
                        page_content=line.strip(),
                        metadata={"source": self.file_path, "line": line_num}
                    )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """真正的异步版本"""
        async with aiofiles.open(self.file_path, 'r', encoding=self.encoding) as f:
            line_num = 0
            async for line in f:
                line_num += 1
                if line.strip():
                    yield Document(
                        page_content=line.strip(),
                        metadata={"source": self.file_path, "line": line_num}
                    )

# 使用
async def main():
    loader = AsyncTextLoader("example.txt")

    # 异步懒加载
    async for doc in loader.alazy_load():
        print(f"行 {doc.metadata['line']}: {doc.page_content[:50]}")

    # 异步加载所有文档
    documents = await loader.aload()
    print(f"异步加载了 {len(documents)} 个文档")

asyncio.run(main())
```

---

## 场景 4: API 数据源 Loader

### 从 REST API 加载数据

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, AsyncIterator
import requests
import aiohttp
import asyncio

class APILoader(BaseLoader):
    """从 REST API 加载数据"""

    def __init__(
        self,
        api_url: str,
        headers: dict = None,
        params: dict = None
    ):
        self.api_url = api_url
        self.headers = headers or {}
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        """同步加载 API 数据"""
        response = requests.get(
            self.api_url,
            headers=self.headers,
            params=self.params
        )
        response.raise_for_status()

        data = response.json()

        # 假设 API 返回一个列表
        if isinstance(data, list):
            for idx, item in enumerate(data):
                yield self._create_document(item, idx)
        else:
            yield self._create_document(data, 0)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """异步加载 API 数据"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url,
                headers=self.headers,
                params=self.params
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # 假设 API 返回一个列表
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        yield self._create_document(item, idx)
                else:
                    yield self._create_document(data, 0)

    def _create_document(self, item: dict, index: int) -> Document:
        """将 API 响应转换为 Document"""
        # 提取文本内容
        content = item.get("content") or item.get("text") or str(item)

        # 创建元数据
        metadata = {
            "source": self.api_url,
            "index": index,
            **{k: v for k, v in item.items() if k not in ["content", "text"]}
        }

        return Document(page_content=content, metadata=metadata)

# 使用
loader = APILoader(
    api_url="https://api.example.com/posts",
    headers={"Authorization": "Bearer token"}
)

documents = loader.load()
print(f"从 API 加载了 {len(documents)} 个文档")
```

---

## 场景 5: 数据库 Loader

### 从数据库加载数据

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
from sqlalchemy import create_engine, text

class DatabaseLoader(BaseLoader):
    """从数据库加载数据"""

    def __init__(
        self,
        connection_string: str,
        query: str,
        content_column: str,
        metadata_columns: list = None
    ):
        self.connection_string = connection_string
        self.query = query
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []

    def lazy_load(self) -> Iterator[Document]:
        """懒加载数据库记录"""
        engine = create_engine(self.connection_string)

        with engine.connect() as connection:
            result = connection.execute(text(self.query))

            for row in result:
                # 提取内容
                content = row[self.content_column]

                # 提取元数据
                metadata = {
                    col: row[col]
                    for col in self.metadata_columns
                    if col in row.keys()
                }
                metadata["source"] = "database"

                yield Document(
                    page_content=str(content),
                    metadata=metadata
                )

# 使用
loader = DatabaseLoader(
    connection_string="postgresql://user:password@localhost/dbname",
    query="SELECT id, title, content, created_at FROM articles WHERE published = true",
    content_column="content",
    metadata_columns=["id", "title", "created_at"]
)

documents = loader.load()
print(f"从数据库加载了 {len(documents)} 个文档")
```

---

## 场景 6: 流式 Loader

### 处理大文件的流式加载

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
import json

class StreamingJSONLoader(BaseLoader):
    """流式加载大型 JSON 文件"""

    def __init__(
        self,
        file_path: str,
        content_key: str,
        chunk_size: int = 1000
    ):
        self.file_path = file_path
        self.content_key = content_key
        self.chunk_size = chunk_size

    def lazy_load(self) -> Iterator[Document]:
        """流式加载 JSON 数组"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过开头的 '['
            f.read(1)

            buffer = ""
            bracket_count = 0
            item_index = 0

            for char in iter(lambda: f.read(1), ''):
                buffer += char

                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1

                    # 完整的 JSON 对象
                    if bracket_count == 0:
                        try:
                            item = json.loads(buffer.strip().rstrip(','))
                            yield self._create_document(item, item_index)
                            item_index += 1
                            buffer = ""
                        except json.JSONDecodeError:
                            pass

    def _create_document(self, item: dict, index: int) -> Document:
        """将 JSON 对象转换为 Document"""
        content = item.get(self.content_key, str(item))

        metadata = {
            "source": self.file_path,
            "index": index,
            **{k: v for k, v in item.items() if k != self.content_key}
        }

        return Document(page_content=content, metadata=metadata)

# 使用
loader = StreamingJSONLoader(
    file_path="large_data.json",
    content_key="text"
)

# 流式处理，不会一次性加载所有数据到内存
for doc in loader.lazy_load():
    print(f"处理文档 {doc.metadata['index']}")
```

---

## 场景 7: 分块 Loader

### 自动分块的 Loader

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Iterator

class ChunkingLoader(BaseLoader):
    """自动分块的加载器"""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def lazy_load(self) -> Iterator[Document]:
        """加载并自动分块"""
        # 读取整个文件
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 创建原始文档
        doc = Document(
            page_content=content,
            metadata={"source": self.file_path}
        )

        # 分块
        chunks = self.text_splitter.split_documents([doc])

        # 逐个返回分块
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)
            yield chunk

# 使用
loader = ChunkingLoader("large_document.txt", chunk_size=1000)
chunks = loader.load()

print(f"文档被分成 {len(chunks)} 个块")
for chunk in chunks[:3]:
    print(f"块 {chunk.metadata['chunk_index']}: {chunk.page_content[:50]}...")
```

---

## 场景 8: 生产级 Loader

### 完整的生产级实现

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator, AsyncIterator, Optional
import logging
from pathlib import Path
import hashlib
import aiofiles
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionLoader(BaseLoader):
    """生产级文档加载器"""

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        skip_empty_lines: bool = True,
        add_checksum: bool = True,
        cache_enabled: bool = False
    ):
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.skip_empty_lines = skip_empty_lines
        self.add_checksum = add_checksum
        self.cache_enabled = cache_enabled
        self._cache = {}

        # 验证
        self._validate()

    def _validate(self):
        """验证输入参数"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"不是文件: {self.file_path}")

        if self.file_path.stat().st_size == 0:
            logger.warning(f"文件为空: {self.file_path}")

    def lazy_load(self) -> Iterator[Document]:
        """同步懒加载"""
        # 检查缓存
        if self.cache_enabled and self.file_path in self._cache:
            logger.info(f"从缓存加载: {self.file_path}")
            yield from self._cache[self.file_path]
            return

        documents = []

        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    if self.skip_empty_lines and not line.strip():
                        continue

                    doc = self._create_document(line.strip(), line_num)
                    documents.append(doc)
                    yield doc

            # 缓存
            if self.cache_enabled:
                self._cache[self.file_path] = documents

        except UnicodeDecodeError as e:
            logger.error(f"编码错误 ({self.encoding}): {e}")
            raise
        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise

    async def alazy_load(self) -> AsyncIterator[Document]:
        """异步懒加载"""
        try:
            async with aiofiles.open(
                self.file_path,
                'r',
                encoding=self.encoding
            ) as f:
                line_num = 0
                async for line in f:
                    line_num += 1

                    if self.skip_empty_lines and not line.strip():
                        continue

                    doc = self._create_document(line.strip(), line_num)
                    yield doc

        except Exception as e:
            logger.error(f"异步加载失败: {e}")
            raise

    def _create_document(self, content: str, line_num: int) -> Document:
        """创建文档，带完整元数据"""
        metadata = {
            "source": str(self.file_path),
            "line": line_num,
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
            "encoding": self.encoding
        }

        # 添加校验和
        if self.add_checksum:
            metadata["checksum"] = self._calculate_checksum(content)

        return Document(page_content=content, metadata=metadata)

    def _calculate_checksum(self, content: str) -> str:
        """计算内容的 MD5 校验和"""
        return hashlib.md5(content.encode()).hexdigest()

# 使用
loader = ProductionLoader(
    file_path="example.txt",
    encoding="utf-8",
    skip_empty_lines=True,
    add_checksum=True,
    cache_enabled=True
)

# 同步加载
documents = loader.load()
print(f"加载了 {len(documents)} 个文档")
print(f"元数据: {documents[0].metadata}")

# 异步加载
async def main():
    async for doc in loader.alazy_load():
        print(f"行 {doc.metadata['line']}: {doc.page_content[:50]}")

asyncio.run(main())
```

---

## 总结

### 实现要点

| 方法 | 必需 | 说明 |
|------|------|------|
| lazy_load() | 是 | 核心方法，必须实现 |
| load() | 否 | 自动可用，不要覆盖 |
| alazy_load() | 否 | 可选，实现真正的异步 |
| aload() | 否 | 自动可用，不要覆盖 |

### 最佳实践

1. **只实现 lazy_load()**:
   - 这是唯一必需的方法
   - 其他方法自动可用
   - 使用生成器逐个返回文档

2. **添加错误处理**:
   - 验证输入参数
   - 捕获并记录异常
   - 提供有意义的错误信息

3. **丰富元数据**:
   - 保留数据来源信息
   - 添加时间戳
   - 包含数据特征（大小、类型等）

4. **实现异步版本**:
   - 覆盖 `alazy_load()` 实现真正的异步
   - 使用 `aiofiles` 或 `aiohttp`
   - 支持并发处理

5. **性能优化**:
   - 使用懒加载节省内存
   - 实现缓存机制
   - 支持流式处理

---

## 数据来源

- [来源: reference/source_documentloader_01.md | LangChain 源码分析]
- [来源: 03_核心概念_2_BaseLoader接口设计.md | BaseLoader 接口设计]
- [来源: reference/fetch_1_8_b1b3cc_02.md | LangChain-OpenTutorial GitHub 教程]
