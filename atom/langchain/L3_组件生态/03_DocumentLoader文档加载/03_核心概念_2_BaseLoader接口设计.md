# 核心概念 2: BaseLoader 接口设计

> 理解 load/lazy_load/aload 的设计意图

---

## 核心问题

为什么 LangChain 的 `BaseLoader` 接口设计了 `load()`, `lazy_load()`, `aload()`, `alazy_load()` 四个方法?它们各自的使用场景是什么?

---

## BaseLoader 接口定义

```python
from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator
from langchain_core.documents import Document

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
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy loader for Document."""
        # 默认实现:在线程池中运行 lazy_load
        iterator = await run_in_executor(None, self.lazy_load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)
            if doc is done:
                break
            yield doc

    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> list[Document]:
        """Load Document and split into chunks."""
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter()
        docs = self.load()
        return text_splitter.split_documents(docs)
```

[来源: reference/source_documentloader_01.md | LangChain 源码分析]

---

## 四个方法的关系

### 继承关系

```
lazy_load()  ← 核心方法,子类必须实现
    ↓
load()       ← 便利方法,调用 lazy_load()
    ↓
load_and_split()  ← 集成方法,调用 load()

alazy_load() ← 异步版本,默认在线程池中运行 lazy_load()
    ↓
aload()      ← 异步便利方法,调用 alazy_load()
```

### 设计意图

**lazy_load()** - 核心方法:
- 子类必须实现
- 返回生成器 `Iterator[Document]`
- 支持流式处理

**load()** - 便利方法:
- 不需要子类实现
- 内部调用 `lazy_load()` 并转换为列表
- 适合小数据集

**alazy_load()** - 异步核心方法:
- 默认实现:在线程池中运行 `lazy_load()`
- 子类可以覆盖实现真正的异步
- 返回异步生成器 `AsyncIterator[Document]`

**aload()** - 异步便利方法:
- 不需要子类实现
- 内部调用 `alazy_load()` 并转换为列表
- 适合异步环境

---

## 方法对比

### 1. load() vs lazy_load()

**load() - 一次性加载**:
```python
# 一次性加载所有文档到内存
loader = PyPDFLoader("large_file.pdf")
docs = loader.load()  # list[Document]

# 优点: 简单,可以直接获取数量
print(f"加载了 {len(docs)} 个文档")

# 缺点: 大文件会占用大量内存
```

**lazy_load() - 流式加载**:
```python
# 逐个加载文档
loader = PyPDFLoader("large_file.pdf")
for doc in loader.lazy_load():  # Iterator[Document]
    process(doc)  # 处理完立即释放内存

# 优点: 内存占用小,支持无限大文件
# 缺点: 无法直接获取数量,需要遍历
```

### 2. load() vs aload()

**load() - 同步加载**:
```python
# 同步加载 - 阻塞当前线程
loader = WebBaseLoader("https://example.com")
docs = loader.load()  # 阻塞直到完成
```

**aload() - 异步加载**:
```python
# 异步加载 - 不阻塞事件循环
loader = WebBaseLoader("https://example.com")
docs = await loader.aload()  # 异步等待

# 可以并发加载多个文档
loaders = [WebBaseLoader(url) for url in urls]
results = await asyncio.gather(*[l.aload() for l in loaders])
```

### 3. lazy_load() vs alazy_load()

**lazy_load() - 同步流式**:
```python
# 同步流式加载
for doc in loader.lazy_load():
    process(doc)  # 同步处理
```

**alazy_load() - 异步流式**:
```python
# 异步流式加载
async for doc in loader.alazy_load():
    await async_process(doc)  # 异步处理
```

---

## 使用场景

### 场景 1: 小文件 - 使用 load()

```python
# 适合: 文件小于 10MB
loader = TextLoader("small_file.txt")
docs = loader.load()

# 可以直接操作列表
print(f"文档数量: {len(docs)}")
for doc in docs:
    print(doc.page_content)
```

### 场景 2: 大文件 - 使用 lazy_load()

```python
# 适合: 文件大于 100MB
loader = PyPDFLoader("large_file.pdf")

# 流式处理,节省内存
for doc in loader.lazy_load():
    # 处理单个文档
    embedding = embed(doc.page_content)
    vector_store.add(embedding)
    # 处理完立即释放内存
```

### 场景 3: 批量加载 - 使用 aload()

```python
# 适合: 批量加载多个文档
urls = ["https://example.com/1", "https://example.com/2", ...]
loaders = [WebBaseLoader(url) for url in urls]

# 并发加载,提升性能
results = await asyncio.gather(*[l.aload() for l in loaders])
all_docs = [doc for docs in results for doc in docs]
```

### 场景 4: 实时流式 - 使用 alazy_load()

```python
# 适合: 实时流式处理
async def process_stream():
    loader = DirectoryLoader("docs/")
    async for doc in loader.alazy_load():
        # 实时处理每个文档
        await async_process(doc)
        # 可以提前终止
        if should_stop():
            break
```

---

## 实现示例

### 最小实现 - 只实现 lazy_load()

```python
class MyLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """只需要实现这一个方法"""
        with open(self.file_path) as f:
            for line in f:
                yield Document(
                    page_content=line.strip(),
                    metadata={"source": self.file_path}
                )

# 自动获得其他方法
loader = MyLoader("file.txt")
docs = loader.load()  # 自动可用
docs = await loader.aload()  # 自动可用
```

### 完整实现 - 覆盖所有方法

```python
class MyAsyncLoader(BaseLoader):
    def __init__(self, urls: list[str]):
        self.urls = urls

    def lazy_load(self) -> Iterator[Document]:
        """同步版本"""
        for url in self.urls:
            content = requests.get(url).text
            yield Document(page_content=content, metadata={"source": url})

    async def alazy_load(self) -> AsyncIterator[Document]:
        """真正的异步版本"""
        async with aiohttp.ClientSession() as session:
            for url in self.urls:
                async with session.get(url) as response:
                    content = await response.text()
                    yield Document(
                        page_content=content,
                        metadata={"source": url}
                    )
```

---

## 设计原则

### 原则 1: 懒加载优先

**为什么?**
- 大文件一次性加载会内存溢出
- 流式处理更高效
- 支持无限大数据集

**如何实现?**
```python
# ✓ 正确: 实现 lazy_load()
def lazy_load(self) -> Iterator[Document]:
    for item in data_source:
        yield Document(...)

# ✗ 错误: 实现 load()
def load(self) -> list[Document]:
    return [Document(...) for item in data_source]
```

### 原则 2: 不要覆盖 load()

**为什么?**
- `load()` 是便利方法,内部调用 `lazy_load()`
- 覆盖 `load()` 会破坏一致性

**文档警告**:
```python
# Sub-classes should not implement this method directly. Instead, they
# should implement the lazy load method.
def load(self) -> list[Document]:
    """Load data into Document objects.

    Returns:
        The documents.
    """
    return list(self.lazy_load())
```

[来源: reference/source_documentloader_01.md | LangChain 源码分析]

### 原则 3: 异步是可选的

**默认实现**:
```python
# 默认在线程池中运行 lazy_load()
async def alazy_load(self) -> AsyncIterator[Document]:
    iterator = await run_in_executor(None, self.lazy_load)
    done = object()
    while True:
        doc = await run_in_executor(None, next, iterator, done)
        if doc is done:
            break
        yield doc
```

**何时覆盖?**
- 数据源支持真正的异步(如 aiohttp)
- 需要并发处理多个请求
- I/O 密集型操作

---

## 性能对比

### 内存占用

```python
import tracemalloc

# load() - 一次性加载
tracemalloc.start()
docs = loader.load()  # 加载 1000 个文档
current, peak = tracemalloc.get_traced_memory()
print(f"内存占用: {peak / 1024 / 1024:.2f} MB")
# 输出: 内存占用: 500.00 MB

# lazy_load() - 流式加载
tracemalloc.start()
for doc in loader.lazy_load():
    process(doc)
current, peak = tracemalloc.get_traced_memory()
print(f"内存占用: {peak / 1024 / 1024:.2f} MB")
# 输出: 内存占用: 5.00 MB (节省 99%)
```

### 处理速度

```python
import time

# 同步加载 - 串行
start = time.time()
for url in urls:  # 100 个 URL
    loader = WebBaseLoader(url)
    docs = loader.load()
print(f"耗时: {time.time() - start:.2f}秒")
# 输出: 耗时: 100.00秒

# 异步加载 - 并行
start = time.time()
loaders = [WebBaseLoader(url) for url in urls]
results = await asyncio.gather(*[l.aload() for l in loaders])
print(f"耗时: {time.time() - start:.2f}秒")
# 输出: 耗时: 5.00秒 (提升 20x)
```

---

## 常见误区

### ❌ 误区 1: 总是使用 load()

**错误**:
```python
# 大文件也用 load()
loader = PyPDFLoader("1GB_file.pdf")
docs = loader.load()  # ❌ 内存溢出
```

**正确**:
```python
# 大文件用 lazy_load()
loader = PyPDFLoader("1GB_file.pdf")
for doc in loader.lazy_load():  # ✓ 流式处理
    process(doc)
```

### ❌ 误区 2: 覆盖 load() 方法

**错误**:
```python
class MyLoader(BaseLoader):
    def load(self) -> list[Document]:  # ❌ 不要覆盖
        return [Document(...)]
```

**正确**:
```python
class MyLoader(BaseLoader):
    def lazy_load(self) -> Iterator[Document]:  # ✓ 实现 lazy_load
        yield Document(...)
```

### ❌ 误区 3: 在同步代码中使用 aload()

**错误**:
```python
# 同步代码中使用 aload()
docs = loader.aload()  # ❌ 语法错误
```

**正确**:
```python
# 异步环境中使用 aload()
docs = await loader.aload()  # ✓ 需要 await

# 或者在同步代码中使用 load()
docs = loader.load()  # ✓ 同步版本
```

---

## 类比理解

### 前端开发类比

**load()** 就像 **Promise.all()**:
```javascript
// 一次性获取所有数据
const data = await Promise.all(promises)
console.log(data.length)  // 可以直接获取数量
```

**lazy_load()** 就像 **Generator**:
```javascript
// 流式获取数据
function* dataGenerator() {
  for (const item of items) {
    yield item  // 逐个返回
  }
}
```

**aload()** 就像 **async/await**:
```javascript
// 异步获取数据
const data = await fetchData()
```

### 日常生活类比

**load()** 就像 **一次性搬家**:
- 把所有东西一次性搬到新家
- 需要大卡车
- 快但累

**lazy_load()** 就像 **分批搬家**:
- 每次搬一点
- 用小车就行
- 慢但轻松

**aload()** 就像 **请搬家公司**:
- 多人同时搬
- 效率高
- 成本高

---

## 总结

### 核心设计

1. **lazy_load()** 是核心方法,子类必须实现
2. **load()** 是便利方法,不要覆盖
3. **alazy_load()** 有默认实现,可选覆盖
4. **aload()** 是异步便利方法,自动可用

### 选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 小文件(<10MB) | load() | 简单方便 |
| 大文件(>100MB) | lazy_load() | 节省内存 |
| 批量加载 | aload() | 并发处理 |
| 实时流式 | alazy_load() | 实时响应 |

---

## 下一步

理解了 BaseLoader 接口设计后,建议:

1. **03_核心概念_3_BlobLoader与BlobParser分离.md** - 理解职责分离
2. **03_核心概念_4_懒加载模式.md** - 深入理解懒加载
3. **实战代码系列** - 实践不同方法的使用

---

**数据来源**:
- [来源: reference/source_documentloader_01.md | LangChain 源码分析]
- [来源: sourcecode/langchain/libs/core/langchain_core/document_loaders/base.py | BaseLoader 接口定义]
