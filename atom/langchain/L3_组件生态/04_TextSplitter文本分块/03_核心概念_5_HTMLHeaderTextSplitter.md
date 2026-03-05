# 核心概念 5：HTMLHeaderTextSplitter

> 保留 HTML 结构的专用分块器

---

## 概述

HTMLHeaderTextSplitter 是 LangChain 中专门用于处理 HTML 文档的分块器。与通用的文本分块器不同，它基于 HTML 标题标签（h1, h2, h3等）进行分块，并保留文档的层级结构信息到元数据中。这对于需要保留网页结构、技术文档或 HTML 格式内容的 RAG 应用至关重要。

**核心特点**：
- 基于 HTML 标题标签（h1-h6）分块
- 保留完整的标题层级元数据
- 不继承自 TextSplitter（独立实现）
- 使用 BeautifulSoup 解析 HTML
- 支持从 URL 直接加载和分块
- 自动处理 HTML 标签和格式

---

## 为什么需要 HTMLHeaderTextSplitter？

### 问题背景

在 RAG 开发中处理 HTML 内容时，我们经常遇到以下问题：

1. **结构信息丢失**：使用通用分块器会丢失 HTML 的层级结构
   - 标题和内容的关系丢失
   - 无法知道某段文本属于哪个章节
   - 检索时缺少上下文信息

2. **网页内容提取困难**：直接处理 HTML 标签很复杂
   - 需要手动解析 HTML
   - 标签、样式、脚本混杂
   - 难以提取纯文本内容

3. **元数据缺失**：无法保留文档的结构化信息
   - 标题层级关系丢失
   - 无法按章节过滤检索结果
   - 难以溯源到原始文档位置

### HTMLHeaderTextSplitter 的解决方案

HTMLHeaderTextSplitter 通过识别 HTML 标题标签，自动分块并保留结构信息：

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<html>
    <body>
        <h1>用户指南</h1>
        <p>欢迎使用我们的产品。</p>

        <h2>快速开始</h2>
        <p>按照以下步骤开始使用。</p>

        <h3>安装</h3>
        <p>运行 pip install package。</p>
    </body>
</html>
"""

headers_to_split_on = [
    ("h1", "标题1"),
    ("h2", "标题2"),
    ("h3", "标题3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html)

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

内容: 运行 pip install package。
元数据: {'标题1': '用户指南', '标题2': '快速开始', '标题3': '安装'}
```

---

## 核心参数

### 1. headers_to_split_on（标题列表）

**类型**：`list[tuple[str, str]]`
**必需**：是

**格式**：`[(tag, name), ...]`

**说明**：
- 指定要跟踪的 HTML 标题标签
- 每个元组包含：标签名（如 "h1"）和元数据键名（如 "Header 1"）
- 标签名必须是有效的 HTML 标题标签（h1-h6）
- 元数据键名可以自定义

**示例**：
```python
# 基本配置
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3")
]

# 中文配置
headers_to_split_on = [
    ("h1", "一级标题"),
    ("h2", "二级标题"),
    ("h3", "三级标题")
]

# 只跟踪部分标题
headers_to_split_on = [
    ("h1", "Title"),
    ("h2", "Section")
]
```

### 2. return_each_element（返回模式）

**类型**：`bool`
**默认值**：`False`

**说明**：
- `False`：聚合相同层级的内容（推荐）
- `True`：每个 HTML 元素单独返回

**示例**：
```python
# 聚合模式（默认）
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2")],
    return_each_element=False
)

# 逐元素模式
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2")],
    return_each_element=True
)
```

---

## 使用方法

### 方法 1：从 HTML 字符串分块

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<html>
    <body>
        <h1>Introduction</h1>
        <p>Welcome to the introduction.</p>
        <h2>Background</h2>
        <p>Some background details.</p>
    </body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html)

for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### 方法 2：从 URL 直接加载

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Title"),
    ("h2", "Section"),
    ("h3", "Subsection")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 从 URL 加载并分块
docs = splitter.split_text_from_url("https://example.com/docs")

for doc in docs:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### 方法 3：与 RecursiveCharacterTextSplitter 组合

```python
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

# 第一步：按 HTML 标题分块
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
)
html_docs = html_splitter.split_text(html)

# 第二步：进一步分块（如果块太大）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(html_docs)

print(f"HTML 分块后: {len(html_docs)} 个块")
print(f"最终分块后: {len(final_docs)} 个块")
```

---

## 实战示例

### 示例 1：技术文档分块

**场景**：处理技术文档网站的 HTML 内容

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<html>
    <body>
        <h1>API 文档</h1>
        <p>本文档介绍 API 的使用方法。</p>

        <h2>认证</h2>
        <p>使用 API Key 进行认证。</p>
        <h3>获取 API Key</h3>
        <p>在控制台中生成 API Key。</p>

        <h2>端点</h2>
        <p>以下是可用的 API 端点。</p>
        <h3>GET /users</h3>
        <p>获取用户列表。</p>
        <h3>POST /users</h3>
        <p>创建新用户。</p>
    </body>
</html>
"""

headers_to_split_on = [
    ("h1", "文档"),
    ("h2", "章节"),
    ("h3", "小节")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html)

print(f"总共分块: {len(docs)} 个\n")

for i, doc in enumerate(docs):
    print(f"块 {i+1}:")
    print(f"  内容: {doc.page_content[:50]}...")
    print(f"  元数据: {doc.metadata}")
    print()
```

**输出**：
```
总共分块: 5 个

块 1:
  内容: 本文档介绍 API 的使用方法。...
  元数据: {'文档': 'API 文档'}

块 2:
  内容: 使用 API Key 进行认证。...
  元数据: {'文档': 'API 文档', '章节': '认证'}

块 3:
  内容: 在控制台中生成 API Key。...
  元数据: {'文档': 'API 文档', '章节': '认证', '小节': '获取 API Key'}

块 4:
  内容: 以下是可用的 API 端点。...
  元数据: {'文档': 'API 文档', '章节': '端点'}

块 5:
  内容: 获取用户列表。...
  元数据: {'文档': 'API 文档', '章节': '端点', '小节': 'GET /users'}
```

### 示例 2：网页内容爬取

**场景**：从网站爬取内容并构建 RAG 系统

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 从 URL 加载并分块
headers_to_split_on = [
    ("h1", "Title"),
    ("h2", "Section"),
    ("h3", "Subsection")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text_from_url("https://docs.example.com")

print(f"从网页提取了 {len(docs)} 个文档块")

# 2. 向量化和存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. 检索
query = "如何进行认证？"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n结果 {i+1}:")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"来源: {doc.metadata}")
```

### 示例 3：多页面批量处理

**场景**：批量处理多个网页

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
import requests

urls = [
    "https://docs.example.com/guide",
    "https://docs.example.com/api",
    "https://docs.example.com/faq"
]

headers_to_split_on = [
    ("h1", "Page"),
    ("h2", "Section")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

all_docs = []

for url in urls:
    try:
        # 获取 HTML 内容
        response = requests.get(url)
        html = response.text

        # 分块
        docs = splitter.split_text(html)

        # 添加 URL 到元数据
        for doc in docs:
            doc.metadata["source_url"] = url

        all_docs.extend(docs)
        print(f"✓ 处理完成: {url} ({len(docs)} 个块)")
    except Exception as e:
        print(f"✗ 处理失败: {url} - {e}")

print(f"\n总共提取了 {len(all_docs)} 个文档块")
```

### 示例 4：保留特定标题层级

**场景**：只关注特定层级的标题

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<html>
    <body>
        <h1>产品文档</h1>
        <h2>功能介绍</h2>
        <p>产品的主要功能。</p>
        <h3>功能 A</h3>
        <p>功能 A 的详细说明。</p>
        <h4>使用方法</h4>
        <p>如何使用功能 A。</p>
    </body>
</html>
"""

# 只跟踪 h1 和 h2
headers_to_split_on = [
    ("h1", "产品"),
    ("h2", "功能")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html)

for doc in docs:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print()
```

### 示例 5：RAG 问答系统

**场景**：构建基于网页内容的问答系统

```python
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# 1. 加载和分块
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Title"), ("h2", "Section"), ("h3", "Topic")]
)
html_docs = html_splitter.split_text_from_url("https://docs.example.com")

# 2. 进一步分块（控制块大小）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(html_docs)

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

**为什么使用 HTMLHeaderTextSplitter**：
- 保留文档的章节结构
- 便于按章节检索
- 元数据包含完整的导航路径

### 2. 网页内容爬取

**场景**：从网站爬取内容构建知识库

**为什么使用 HTMLHeaderTextSplitter**：
- 自动解析 HTML 标签
- 提取纯文本内容
- 保留页面结构信息

### 3. 在线文档 RAG

**场景**：构建基于在线文档的问答系统

**为什么使用 HTMLHeaderTextSplitter**：
- 直接从 URL 加载
- 保留文档层级
- 便于溯源到原始页面

### 4. 多语言文档处理

**场景**：处理多语言的 HTML 文档

**为什么使用 HTMLHeaderTextSplitter**：
- 语言无关的标签识别
- 支持任意语言的内容
- 元数据键名可自定义

---

## 常见问题

### Q1：HTMLHeaderTextSplitter 和 MarkdownHeaderTextSplitter 有什么区别？

**A**：

| 特性 | HTMLHeaderTextSplitter | MarkdownHeaderTextSplitter |
|------|------------------------|----------------------------|
| 输入格式 | HTML | Markdown |
| 解析库 | BeautifulSoup | 正则表达式 |
| 标题识别 | HTML 标签（h1-h6） | Markdown 语法（#, ##） |
| 代码块处理 | 自动 | 手动检测 |
| URL 加载 | 支持 | 不支持 |

**推荐**：
- HTML 内容 → HTMLHeaderTextSplitter
- Markdown 内容 → MarkdownHeaderTextSplitter

### Q2：如何处理没有标题的 HTML 内容？

**A**：

如果 HTML 内容没有标题标签，HTMLHeaderTextSplitter 会将所有内容作为一个块返回：

```python
html = """
<html>
    <body>
        <p>这是一段没有标题的内容。</p>
        <p>这是另一段内容。</p>
    </body>
</html>
"""

splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "H1")])
docs = splitter.split_text(html)

print(f"块数量: {len(docs)}")  # 1 个块
```

**解决方案**：
- 使用 RecursiveCharacterTextSplitter 进一步分块
- 或者预处理 HTML，添加标题标签

### Q3：如何与 RecursiveCharacterTextSplitter 组合使用？

**A**：

**推荐流程**：
1. 先用 HTMLHeaderTextSplitter 按标题分块
2. 再用 RecursiveCharacterTextSplitter 控制块大小

```python
# 第一步：按标题分块
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2")]
)
html_docs = html_splitter.split_text(html)

# 第二步：进一步分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(html_docs)
```

**注意**：使用 `split_documents()` 而非 `split_text()`，以保留元数据。

### Q4：如何处理复杂的 HTML 结构？

**A**：

对于包含表格、列表、嵌套标签的复杂 HTML：

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<html>
    <body>
        <h1>产品列表</h1>
        <table>
            <tr><td>产品 A</td><td>$100</td></tr>
            <tr><td>产品 B</td><td>$200</td></tr>
        </table>

        <h2>详细信息</h2>
        <ul>
            <li>特性 1</li>
            <li>特性 2</li>
        </ul>
    </body>
</html>
"""

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2")]
)
docs = splitter.split_text(html)

# BeautifulSoup 会自动处理表格和列表
for doc in docs:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

---

## 与 RAG 开发的联系

### 1. 网页知识库构建

在 RAG 应用中，HTMLHeaderTextSplitter 帮助构建基于网页的知识库：

```python
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 网站 URL 列表
urls = [
    "https://docs.example.com/guide",
    "https://docs.example.com/api",
    "https://docs.example.com/faq"
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "Page"), ("h2", "Section")]
)

all_docs = []
for url in urls:
    docs = splitter.split_text_from_url(url)
    for doc in docs:
        doc.metadata["source"] = url
    all_docs.extend(docs)

# 构建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(all_docs, embeddings)

print(f"知识库包含 {len(all_docs)} 个文档块")
```

### 2. 结构化检索

HTMLHeaderTextSplitter 保留的元数据支持结构化检索：

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

### 3. 溯源到原始页面

元数据包含完整的标题层级，便于溯源：

```python
query = "如何获取 API Key？"
results = vectorstore.similarity_search(query, k=1)

doc = results[0]
print(f"内容: {doc.page_content}")
print(f"来源: {doc.metadata}")

# 输出:
# 内容: 在控制台中生成 API Key。
# 来源: {'Page': 'API 文档', 'Section': '认证', 'Subsection': '获取 API Key', 'source': 'https://docs.example.com/api'}
```

---

## 最佳实践

### 1. 选择合适的标题层级

```python
# ✓ 推荐：根据文档结构选择
headers_to_split_on = [
    ("h1", "Page"),      # 页面标题
    ("h2", "Section"),   # 主要章节
    ("h3", "Topic")      # 子主题
]

# ✗ 不推荐：跟踪过多层级
headers_to_split_on = [
    ("h1", "H1"),
    ("h2", "H2"),
    ("h3", "H3"),
    ("h4", "H4"),
    ("h5", "H5"),
    ("h6", "H6")
]
```

### 2. 与 RecursiveCharacterTextSplitter 组合

```python
# ✓ 推荐：先按标题分块，再控制大小
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "H1"), ("h2", "H2")]
)
html_docs = html_splitter.split_text(html)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(html_docs)

# ✗ 不推荐：只用 HTMLHeaderTextSplitter（块可能过大）
docs = html_splitter.split_text(html)
```

### 3. 添加源信息到元数据

```python
# ✓ 推荐：添加 URL 到元数据
docs = splitter.split_text_from_url(url)
for doc in docs:
    doc.metadata["source_url"] = url
    doc.metadata["crawled_at"] = datetime.now().isoformat()

# ✗ 不推荐：不保留源信息
docs = splitter.split_text_from_url(url)
```

### 4. 处理错误和异常

```python
# ✓ 推荐：处理网络错误
import requests

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    html = response.text
    docs = splitter.split_text(html)
except requests.RequestException as e:
    print(f"加载失败: {url} - {e}")
    docs = []

# ✗ 不推荐：不处理错误
docs = splitter.split_text_from_url(url)
```

---

## 数据来源

- [来源: reference/source_textsplitter_04_html.md | HTMLHeaderTextSplitter 源码分析]
- [来源: BeautifulSoup 官方文档 | HTML 解析原理]
