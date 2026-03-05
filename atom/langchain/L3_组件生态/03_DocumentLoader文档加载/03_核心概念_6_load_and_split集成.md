# 核心概念 6: load_and_split 集成

> 理解加载与分割的一体化设计

---

## 核心问题

为什么 LangChain 的 DocumentLoader 提供 `load_and_split()` 方法?它如何简化文档处理流程?

---

## 什么是 load_and_split?

### 定义

**load_and_split()**: 一个便利方法,将文档加载和文本分割两个步骤合并为一个操作。

**对比**:

```python
# 分步操作 - 手动分割
loader = PyPDFLoader("file.pdf")
docs = loader.load()  # 步骤 1: 加载
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)  # 步骤 2: 分割

# 一体化操作 - 自动分割
loader = PyPDFLoader("file.pdf")
chunks = loader.load_and_split()  # 一步完成
```

---

## 为什么需要 load_and_split?

### 问题 1: 重复代码

**场景**: 每次加载文档都要手动分割

```python
# 重复的分割代码
loader1 = PyPDFLoader("file1.pdf")
docs1 = loader1.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks1 = splitter.split_documents(docs1)

loader2 = TextLoader("file2.txt")
docs2 = loader2.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks2 = splitter.split_documents(docs2)

# 每次都要重复这个模式
```

**load_and_split 解决**:

```python
# 简化的代码
loader1 = PyPDFLoader("file1.pdf")
chunks1 = loader1.load_and_split()

loader2 = TextLoader("file2.txt")
chunks2 = loader2.load_and_split()

# 一行代码完成加载和分割
```

### 问题 2: 默认配置不统一

**场景**: 不同地方使用不同的分割配置

```python
# 配置不统一
splitter1 = RecursiveCharacterTextSplitter(chunk_size=500)
chunks1 = splitter1.split_documents(docs1)

splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks2 = splitter2.split_documents(docs2)

# 导致分块大小不一致
```

**load_and_split 解决**:

```python
# 统一的默认配置
chunks1 = loader1.load_and_split()  # 使用默认配置
chunks2 = loader2.load_and_split()  # 使用相同的默认配置

# 或者统一自定义配置
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks1 = loader1.load_and_split(text_splitter=splitter)
chunks2 = loader2.load_and_split(text_splitter=splitter)
```

### 问题 3: RAG 管道冗长

**场景**: RAG 管道中的重复步骤

```python
# 冗长的 RAG 管道
loader = DirectoryLoader("docs/")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# 5 行代码才能完成文档入库
```

**load_and_split 简化**:

```python
# 简化的 RAG 管道
loader = DirectoryLoader("docs/")
chunks = loader.load_and_split()
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# 4 行代码完成文档入库
```

---

## load_and_split 的实现

### BaseLoader 中的实现

```python
class BaseLoader(ABC):
    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> list[Document]:
        """加载文档并分割成块

        Args:
            text_splitter: 可选的文本分割器
                          如果为 None,使用默认的 RecursiveCharacterTextSplitter

        Returns:
            分割后的 Document 列表
        """
        if text_splitter is None:
            # 默认使用 RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter()

        # 加载文档
        docs = self.load()

        # 分割文档
        return text_splitter.split_documents(docs)
```

[来源: reference/source_documentloader_01.md | LangChain 源码分析]

### 设计特点

**1. 默认分割器**:

```python
# 不指定分割器 - 使用默认配置
chunks = loader.load_and_split()

# 等价于
splitter = RecursiveCharacterTextSplitter()
docs = loader.load()
chunks = splitter.split_documents(docs)
```

**2. 自定义分割器**:

```python
# 指定分割器 - 使用自定义配置
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = loader.load_and_split(text_splitter=splitter)
```

**3. 返回分割后的文档**:

```python
# 返回的是分割后的 Document 列表
chunks = loader.load_and_split()
print(f"分割后的块数: {len(chunks)}")

# 每个块都是一个 Document
for chunk in chunks:
    print(f"块内容: {chunk.page_content[:100]}...")
    print(f"元数据: {chunk.metadata}")
```

---

## 实际应用

### 场景 1: 快速原型开发

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 快速构建 RAG 原型
loader = PyPDFLoader("document.pdf")
chunks = loader.load_and_split()  # 一行完成加载和分割

# 向量化并存储
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# 检索
query = "什么是 RAG?"
results = vector_store.similarity_search(query)
```

### 场景 2: 批量文档处理

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 批量处理目录中的所有文档
loader = DirectoryLoader("docs/", glob="**/*.txt")

# 使用自定义分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 一次性加载和分割所有文档
chunks = loader.load_and_split(text_splitter=splitter)

print(f"处理了 {len(chunks)} 个文档块")
```

### 场景 3: 不同格式统一处理

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 统一的分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 处理不同格式的文档
loaders = [
    PyPDFLoader("file1.pdf"),
    TextLoader("file2.txt"),
    UnstructuredWordDocumentLoader("file3.docx")
]

all_chunks = []
for loader in loaders:
    chunks = loader.load_and_split(text_splitter=splitter)
    all_chunks.extend(chunks)

print(f"总共 {len(all_chunks)} 个文档块")
```

### 场景 4: 与 LCEL 集成

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 加载和分割网页
loader = WebBaseLoader("https://example.com/article")
chunks = loader.load_and_split()

# 构建向量存储
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# 构建 RAG 链
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever()
)

# 查询
result = qa_chain.invoke({"query": "文章的主要观点是什么?"})
print(result["result"])
```

---

## 与 TextSplitter 的集成

### 默认分割器

```python
# 默认使用 RecursiveCharacterTextSplitter
chunks = loader.load_and_split()

# 默认配置
# - chunk_size: 4000
# - chunk_overlap: 200
# - separators: ["\n\n", "\n", " ", ""]
```

### 自定义分割器

**1. RecursiveCharacterTextSplitter**:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = loader.load_and_split(text_splitter=splitter)
```

**2. CharacterTextSplitter**:

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator="\n"
)

chunks = loader.load_and_split(text_splitter=splitter)
```

**3. TokenTextSplitter**:

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,  # Token 数量
    chunk_overlap=50
)

chunks = loader.load_and_split(text_splitter=splitter)
```

**4. MarkdownTextSplitter**:

```python
from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = loader.load_and_split(text_splitter=splitter)
```

---

## 类比理解

### 前端开发类比

**load_and_split** 就像 **fetch + parse 的组合**:

```javascript
// 分步操作
const response = await fetch(url)
const data = await response.json()

// 一体化操作
const data = await fetch(url).then(r => r.json())
```

### 日常生活类比

**load_and_split** 就像 **买菜 + 切菜的组合**:

**分步操作**:
- 去菜市场买菜
- 回家洗菜
- 切菜

**一体化操作**:
- 买切好的净菜(一步到位)

---

## 常见误区

### ❌ 误区 1: 总是使用 load_and_split

**错误**: 认为总是应该使用 load_and_split

**正确**: 根据场景选择

```python
# ✓ 适合: 需要分割的场景
chunks = loader.load_and_split()

# ✗ 不适合: 不需要分割的场景
docs = loader.load()  # 直接使用完整文档
```

### ❌ 误区 2: 忽略分割器配置

**错误**: 总是使用默认分割器

**正确**: 根据文档类型选择合适的分割器

```python
# ✗ 错误: Markdown 文档使用默认分割器
chunks = loader.load_and_split()

# ✓ 正确: Markdown 文档使用 MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size=1000)
chunks = loader.load_and_split(text_splitter=splitter)
```

### ❌ 误区 3: 分块大小不合理

**错误**: 使用过大或过小的分块

**正确**: 根据应用场景调整

```python
# ✗ 过小: 丢失上下文
splitter = RecursiveCharacterTextSplitter(chunk_size=100)
chunks = loader.load_and_split(text_splitter=splitter)

# ✗ 过大: 超过 LLM 上下文窗口
splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
chunks = loader.load_and_split(text_splitter=splitter)

# ✓ 合理: 根据 LLM 上下文窗口调整
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 适合大多数 LLM
    chunk_overlap=200  # 保留上下文
)
chunks = loader.load_and_split(text_splitter=splitter)
```

---

## 设计优势

### 1. 简化代码

```python
# 之前: 3 行代码
docs = loader.load()
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(docs)

# 现在: 1 行代码
chunks = loader.load_and_split()
```

### 2. 统一接口

```python
# 所有 Loader 都支持 load_and_split
chunks1 = PyPDFLoader("file.pdf").load_and_split()
chunks2 = TextLoader("file.txt").load_and_split()
chunks3 = WebBaseLoader("https://example.com").load_and_split()
```

### 3. 默认配置

```python
# 提供合理的默认配置
chunks = loader.load_and_split()

# 无需每次都指定分割器
```

---

## 最佳实践

### 1. 根据文档类型选择分割器

```python
# Markdown 文档
splitter = MarkdownTextSplitter(chunk_size=1000)
chunks = loader.load_and_split(text_splitter=splitter)

# 代码文件
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000
)
chunks = loader.load_and_split(text_splitter=splitter)

# 普通文本
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = loader.load_and_split(text_splitter=splitter)
```

### 2. 根据 LLM 调整分块大小

```python
# GPT-3.5 (4K 上下文)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 保守估计
    chunk_overlap=50
)

# GPT-4 (8K 上下文)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Claude (100K 上下文)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=500
)

chunks = loader.load_and_split(text_splitter=splitter)
```

### 3. 保留元数据

```python
# load_and_split 会保留原始文档的元数据
chunks = loader.load_and_split()

for chunk in chunks:
    print(f"来源: {chunk.metadata['source']}")
    print(f"页码: {chunk.metadata.get('page', 'N/A')}")
    print(f"内容: {chunk.page_content[:100]}...")
```

### 4. 批量处理优化

```python
# 批量处理时复用分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

all_chunks = []
for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    chunks = loader.load_and_split(text_splitter=splitter)
    all_chunks.extend(chunks)
```

---

## 性能考虑

### 内存占用

```python
# load_and_split 会一次性加载所有文档
chunks = loader.load_and_split()  # 内存占用: 所有文档 + 所有分块

# 对于大文件,考虑使用 lazy_load
for doc in loader.lazy_load():
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents([doc])
    process(chunks)  # 逐个处理,释放内存
```

### 处理速度

```python
import time

# 测试 load_and_split 性能
start = time.time()
chunks = loader.load_and_split()
print(f"耗时: {time.time() - start:.2f}秒")
print(f"分块数: {len(chunks)}")

# 分块大小影响性能
# - 分块越小,分块数越多,处理越慢
# - 分块越大,分块数越少,处理越快
```

---

## 与其他方法的对比

### load() vs load_and_split()

```python
# load() - 返回完整文档
docs = loader.load()
print(f"文档数: {len(docs)}")  # 可能只有 1 个

# load_and_split() - 返回分割后的文档
chunks = loader.load_and_split()
print(f"分块数: {len(chunks)}")  # 可能有 100 个
```

### lazy_load() vs load_and_split()

```python
# lazy_load() - 流式加载,不分割
for doc in loader.lazy_load():
    process(doc)  # 处理完整文档

# load_and_split() - 一次性加载和分割
chunks = loader.load_and_split()
for chunk in chunks:
    process(chunk)  # 处理分块
```

---

## 总结

### 核心设计

1. **便利方法**: 将加载和分割合并为一个操作
2. **默认配置**: 提供合理的默认分割器
3. **灵活定制**: 支持自定义分割器

### 设计优势

| 优势 | 说明 |
|------|------|
| 简化代码 | 1 行代码完成 2 个步骤 |
| 统一接口 | 所有 Loader 都支持 |
| 默认配置 | 无需每次指定分割器 |
| 灵活定制 | 支持自定义分割器 |

### 使用场景

| 场景 | 推荐方法 |
|------|----------|
| 快速原型 | load_and_split() |
| 批量处理 | load_and_split() |
| 大文件处理 | lazy_load() + 手动分割 |
| 不需要分割 | load() |

---

## 下一步

理解了 load_and_split 集成后,建议:

1. **03_核心概念_7_元数据管理策略.md** - 理解元数据管理
2. **实战代码_场景1_基础文本文件加载.md** - 实践 load_and_split
3. **实战代码_场景6_批量文档加载.md** - 批量处理实战

---

**数据来源**:
- [来源: reference/source_documentloader_01.md | LangChain 源码分析]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
