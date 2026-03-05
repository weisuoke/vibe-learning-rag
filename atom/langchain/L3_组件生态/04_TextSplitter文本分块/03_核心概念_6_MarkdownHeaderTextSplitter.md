# 核心概念 6：MarkdownHeaderTextSplitter

> 保留 Markdown 层级的专用分块器

---

## 概述

MarkdownHeaderTextSplitter 是 LangChain 中专门用于处理 Markdown 文档的分块器。与通用的文本分块器不同，它基于 Markdown 标题（#, ##, ###等）进行分块，并保留文档的层级结构信息到元数据中。这对于需要保留技术文档、博客文章、README 文件等 Markdown 格式内容的 RAG 应用至关重要。

**核心特点**：
- 基于 Markdown 标题（#-######）分块
- 保留完整的标题层级元数据
- 代码块感知（不会在代码块中错误识别标题）
- 支持自定义标题模式
- 不继承自 TextSplitter（独立实现）
- 使用正则表达式解析 Markdown

---

## 为什么需要 MarkdownHeaderTextSplitter？

### 问题背景

在 RAG 开发中处理 Markdown 内容时，我们经常遇到以下问题：

1. **结构信息丢失**：使用通用分块器会丢失 Markdown 的层级结构
   - 标题和内容的关系丢失
   - 无法知道某段文本属于哪个章节
   - 检索时缺少上下文信息

2. **代码块误识别**：代码块中的 # 被错误识别为标题
   ```python
   # 这是代码注释，不是标题
   def hello():
       pass
   ```

3. **元数据缺失**：无法保留文档的结构化信息
   - 标题层级关系丢失
   - 无法按章节过滤检索结果
   - 难以溯源到原始文档位置

### MarkdownHeaderTextSplitter 的解决方案

MarkdownHeaderTextSplitter 通过识别 Markdown 标题，自动分块并保留结构信息：

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = """
# 用户指南

欢迎使用我们的产品。

## 快速开始

按照以下步骤开始使用。

### 安装

运行以下命令安装：

\`\`\`bash
pip install package
\`\`\`
"""

headers_to_split_on = [
    ("#", "标题1"),
    ("##", "标题2"),
    ("###", "标题3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown)

# 每个块都包含完整的标题层级信息
for doc in docs:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print()
```

**输出**：
```
内容: 欢迎使用我们的产品。
元数据: {'标题1': '用户指南'}

内容: 按照以下步骤开始使用。
元数据: {'标题1': '用户指南', '标题2': '快速开始'}

内容: 运行以下命令安装：

```bash
pip install package
```
元数据: {'标题1': '用户指南', '标题2': '快速开始', '标题3': '安装'}
```

---

## 核心参数

### 1. headers_to_split_on（标题列表）

**类型**：`list[tuple[str, str]]`
**必需**：是

**格式**：`[(separator, name), ...]`

**说明**：
- 指定要跟踪的 Markdown 标题
- 每个元组包含：标题分隔符（如 "#"）和元数据键名（如 "Header 1"）
- 标题分隔符必须是有效的 Markdown 标题语法（#-######）
- 元数据键名可以自定义

**示例**：
```python
# 基本配置
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

# 中文配置
headers_to_split_on = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题")
]

# 只跟踪部分标题
headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section")
]
```

### 2. return_each_line（返回模式）

**类型**：`bool`
**默认值**：`False`

**说明**：
- `False`：聚合相同层级的内容（推荐）
- `True`：逐行返回

**示例**：
```python
# 聚合模式（默认）
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    return_each_line=False
)

# 逐行模式
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    return_each_line=True
)
```

### 3. strip_headers（去除标题）

**类型**：`bool`
**默认值**：`True`

**说明**：
- `True`：从内容中去除标题（推荐）
- `False`：保留标题在内容中

**示例**：
```python
# 去除标题（默认）
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    strip_headers=True
)

# 保留标题
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    strip_headers=False
)
```

---

## 使用方法

### 方法 1：从 Markdown 字符串分块

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = """
# Introduction

Welcome to the introduction.

## Background

Some background details.

### Details

More details here.
"""

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown)

for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### 方法 2：从文件加载

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 读取 Markdown 文件
with open("README.md", "r", encoding="utf-8") as f:
    markdown = f.read()

headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
    ("###", "Subsection")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown)

for doc in docs:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### 方法 3：与 RecursiveCharacterTextSplitter 组合

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 第一步：按 Markdown 标题分块
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
)
md_docs = md_splitter.split_text(markdown)

# 第二步：进一步分块（如果块太大）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(md_docs)

print(f"Markdown 分块后: {len(md_docs)} 个块")
print(f"最终分块后: {len(final_docs)} 个块")
```

---

## 实战示例

### 示例 1：技术文档分块

**场景**：处理技术文档的 Markdown 内容

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = """
# API 文档

本文档介绍 API 的使用方法。

## 认证

使用 API Key 进行认证。

### 获取 API Key

在控制台中生成 API Key：

\`\`\`bash
curl -X POST https://api.example.com/keys
\`\`\`

## 端点

以下是可用的 API 端点。

### GET /users

获取用户列表。

\`\`\`python
import requests
response = requests.get("https://api.example.com/users")
\`\`\`

### POST /users

创建新用户。
"""

headers_to_split_on = [
    ("#", "文档"),
    ("##", "章节"),
    ("###", "小节")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown)

print(f"总共分块: {len(docs)} 个\n")

for i, doc in enumerate(docs):
    print(f"块 {i+1}:")
    print(f"  内容: {doc.page_content[:50]}...")
    print(f"  元数据: {doc.metadata}")
    print()
```

### 示例 2：README 文件处理

**场景**：处理 GitHub README 文件

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 读取 README 文件
with open("README.md", "r", encoding="utf-8") as f:
    markdown = f.read()

# 2. 分块
headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
    ("###", "Topic")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown)

print(f"从 README 提取了 {len(docs)} 个文档块")

# 3. 向量化和存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. 检索
query = "如何安装？"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n结果 {i+1}:")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"来源: {doc.metadata}")
```

### 示例 3：博客文章分块

**场景**：处理博客文章的 Markdown 内容

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown = """
# 深入理解 RAG 技术

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。

## 什么是 RAG？

RAG 通过检索相关文档来增强 LLM 的生成能力。

### 核心组件

RAG 包含三个核心组件：

1. 文档加载器
2. 向量存储
3. 生成模型

## 如何实现 RAG？

实现 RAG 需要以下步骤。

### 步骤 1：文档加载

首先加载文档：

\`\`\`python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("document.txt")
docs = loader.load()
\`\`\`

### 步骤 2：向量化

然后进行向量化。
"""

headers_to_split_on = [
    ("#", "文章"),
    ("##", "章节"),
    ("###", "小节")
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 保留标题
)
docs = splitter.split_text(markdown)

for doc in docs:
    print(f"内容: {doc.page_content[:80]}...")
    print(f"元数据: {doc.metadata}")
    print()
```

### 示例 4：多文件批量处理

**场景**：批量处理多个 Markdown 文件

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os

markdown_files = [
    "docs/guide.md",
    "docs/api.md",
    "docs/faq.md"
]

headers_to_split_on = [
    ("#", "Page"),
    ("##", "Section")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

all_docs = []

for file_path in markdown_files:
    try:
        # 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            markdown = f.read()

        # 分块
        docs = splitter.split_text(markdown)

        # 添加文件路径到元数据
        for doc in docs:
            doc.metadata["source_file"] = file_path

        all_docs.extend(docs)
        print(f"✓ 处理完成: {file_path} ({len(docs)} 个块)")
    except Exception as e:
        print(f"✗ 处理失败: {file_path} - {e}")

print(f"\n总共提取了 {len(all_docs)} 个文档块")
```

### 示例 5：RAG 问答系统

**场景**：构建基于 Markdown 文档的问答系统

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# 1. 加载和分块
with open("docs/guide.md", "r", encoding="utf-8") as f:
    markdown = f.read()

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Title"), ("##", "Section"), ("###", "Topic")]
)
md_docs = md_splitter.split_text(markdown)

# 2. 进一步分块（控制块大小）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(md_docs)

print(f"总共 {len(docs)} 个文档块")

# 3. 构建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. 创建 QA 链
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. 问答
questions = [
    "产品的主要功能是什么？",
    "如何进行安装？",
    "有哪些 API 端点？"
]

for question in questions:
    answer = qa_chain.run(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
```

---

## 适用场景

### 1. 技术文档处理

**场景**：处理 API 文档、用户手册、技术指南

**为什么使用 MarkdownHeaderTextSplitter**：
- 保留文档的章节结构
- 便于按章节检索
- 元数据包含完整的导航路径

### 2. GitHub README 处理

**场景**：处理 GitHub 仓库的 README 文件

**为什么使用 MarkdownHeaderTextSplitter**：
- README 通常使用 Markdown 格式
- 保留项目文档的结构
- 便于构建项目知识库

### 3. 博客文章处理

**场景**：处理博客文章、教程、笔记

**为什么使用 MarkdownHeaderTextSplitter**：
- 博客通常使用 Markdown 编写
- 保留文章的章节结构
- 便于按主题检索

### 4. 代码文档处理

**场景**：处理代码注释、文档字符串

**为什么使用 MarkdownHeaderTextSplitter**：
- 代码块感知，不会误识别注释
- 保留文档的层级结构
- 适合代码库文档化

---

## 常见问题

### Q1：MarkdownHeaderTextSplitter 和 HTMLHeaderTextSplitter 有什么区别？

**A**：

| 特性 | MarkdownHeaderTextSplitter | HTMLHeaderTextSplitter |
|------|----------------------------|------------------------|
| 输入格式 | Markdown | HTML |
| 解析库 | 正则表达式 | BeautifulSoup |
| 标题识别 | Markdown 语法（#, ##） | HTML 标签（h1-h6） |
| 代码块处理 | 手动检测 | 自动 |
| URL 加载 | 不支持 | 支持 |

**推荐**：
- Markdown 内容 → MarkdownHeaderTextSplitter
- HTML 内容 → HTMLHeaderTextSplitter

### Q2：如何处理代码块中的 # 符号？

**A**：

MarkdownHeaderTextSplitter 会自动检测代码块，避免误识别：

```python
markdown = """
# 标题

这是正常的标题。

\`\`\`python
# 这是代码注释，不会被识别为标题
def hello():
    pass
\`\`\`
"""

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1")])
docs = splitter.split_text(markdown)

# 代码块中的 # 不会被识别为标题
```

### Q3：如何与 RecursiveCharacterTextSplitter 组合使用？

**A**：

**推荐流程**：
1. 先用 MarkdownHeaderTextSplitter 按标题分块
2. 再用 RecursiveCharacterTextSplitter 控制块大小

```python
# 第一步：按标题分块
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)
md_docs = md_splitter.split_text(markdown)

# 第二步：进一步分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(md_docs)
```

**注意**：使用 `split_documents()` 而非 `split_text()`，以保留元数据。

### Q4：如何处理没有标题的 Markdown 内容？

**A**：

如果 Markdown 内容没有标题，MarkdownHeaderTextSplitter 会将所有内容作为一个块返回：

```python
markdown = """
这是一段没有标题的内容。

这是另一段内容。
"""

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1")])
docs = splitter.split_text(markdown)

print(f"块数量: {len(docs)}")  # 1 个块
```

**解决方案**：
- 使用 RecursiveCharacterTextSplitter 进一步分块
- 或者预处理 Markdown，添加标题

---

## 与 RAG 开发的联系

### 1. 技术文档知识库

在 RAG 应用中，MarkdownHeaderTextSplitter 帮助构建基于技术文档的知识库：

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Markdown 文件列表
markdown_files = [
    "docs/guide.md",
    "docs/api.md",
    "docs/faq.md"
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Page"), ("##", "Section")]
)

all_docs = []
for file_path in markdown_files:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown = f.read()
    docs = splitter.split_text(markdown)
    for doc in docs:
        doc.metadata["source"] = file_path
    all_docs.extend(docs)

# 构建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(all_docs, embeddings)

print(f"知识库包含 {len(all_docs)} 个文档块")
```

### 2. 结构化检索

MarkdownHeaderTextSplitter 保留的元数据支持结构化检索：

```python
# 按章节过滤检索
results = vectorstore.similarity_search(
    query="如何安装？",
    k=5,
    filter={"Section": "快速开始"}
)

# 按页面过滤检索
results = vectorstore.similarity_search(
    query="API 端点",
    k=5,
    filter={"Page": "API 文档"}
)
```

### 3. 溯源到原始文档

元数据包含完整的标题层级，便于溯源：

```python
query = "如何获取 API Key？"
results = vectorstore.similarity_search(query, k=1)

doc = results[0]
print(f"内容: {doc.page_content}")
print(f"来源: {doc.metadata}")

# 输出:
# 内容: 在控制台中生成 API Key...
# 来源: {'Page': 'API 文档', 'Section': '认证', 'Topic': '获取 API Key', 'source': 'docs/api.md'}
```

---

## 最佳实践

### 1. 选择合适的标题层级

```python
# ✓ 推荐：根据文档结构选择
headers_to_split_on = [
    ("#", "Page"),      # 页面标题
    ("##", "Section"),  # 主要章节
    ("###", "Topic")    # 子主题
]

# ✗ 不推荐：跟踪过多层级
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
    ("#####", "H5"),
    ("######", "H6")
]
```

### 2. 与 RecursiveCharacterTextSplitter 组合

```python
# ✓ 推荐：先按标题分块，再控制大小
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)
md_docs = md_splitter.split_text(markdown)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(md_docs)

# ✗ 不推荐：只用 MarkdownHeaderTextSplitter（块可能过大）
docs = md_splitter.split_text(markdown)
```

### 3. 添加源信息到元数据

```python
# ✓ 推荐：添加文件路径到元数据
docs = splitter.split_text(markdown)
for doc in docs:
    doc.metadata["source_file"] = file_path
    doc.metadata["processed_at"] = datetime.now().isoformat()

# ✗ 不推荐：不保留源信息
docs = splitter.split_text(markdown)
```

### 4. 处理编码问题

```python
# ✓ 推荐：指定编码
with open("README.md", "r", encoding="utf-8") as f:
    markdown = f.read()

# ✗ 不推荐：不指定编码（可能导致乱码）
with open("README.md", "r") as f:
    markdown = f.read()
```

---

## 数据来源

- [来源: reference/source_textsplitter_03_markdown.md | MarkdownHeaderTextSplitter 源码分析]
- [来源: Markdown 规范 | 标题语法]
