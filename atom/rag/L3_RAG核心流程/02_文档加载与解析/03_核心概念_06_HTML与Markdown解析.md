# 核心概念6：HTML与Markdown解析

> 掌握网页内容和Markdown文档的提取与清洗

---

## 为什么HTML与Markdown解析重要？

**HTML**和**Markdown**是互联网和技术文档中最常见的格式：
- **HTML**：网页、博客、在线文档、API文档
- **Markdown**：GitHub README、技术博客、文档站点

**在RAG系统中，正确解析这些格式能够获取大量高质量的在线知识。**

---

## 1. HTML解析基础

### 1.1 HTML结构理解

HTML文档包含大量噪声：
- 导航栏、侧边栏、页脚
- 广告、脚本、样式
- 元数据、注释

**核心挑战：** 从噪声中提取有价值的内容

```html
<!-- 典型的HTML结构 -->
<html>
  <head>
    <title>RAG技术指南</title>
    <script>/* 脚本代码 */</script>
    <style>/* 样式代码 */</style>
  </head>
  <body>
    <nav><!-- 导航栏 --></nav>
    <aside><!-- 侧边栏 --></aside>
    <article>
      <h1>核心内容标题</h1>
      <p>这是有价值的正文...</p>
    </article>
    <footer><!-- 页脚 --></footer>
  </body>
</html>
```

---

## 2. 使用BeautifulSoup解析HTML

### 2.1 基础文本提取

```python
"""
BeautifulSoup基础使用
适用场景：提取HTML网页的文本内容
"""

from bs4 import BeautifulSoup
from langchain.schema import Document

def load_html_basic(html_content: str, source: str = "unknown") -> Document:
    """基础HTML解析"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取所有文本
    text = soup.get_text()

    # 创建Document
    document = Document(
        page_content=text,
        metadata={
            "source": source,
            "format": "html"
        }
    )

    return document

# 使用示例
html = """
<html>
  <body>
    <h1>RAG技术指南</h1>
    <p>RAG系统通过检索增强生成...</p>
  </body>
</html>
"""

doc = load_html_basic(html, "example.html")
print(doc.page_content)
```

### 2.2 智能内容提取

```python
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_html_smart(html_content: str, source: str = "unknown") -> Document:
    """智能HTML解析：只提取核心内容"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. 移除噪声标签
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
        tag.decompose()

    # 2. 查找主要内容区域
    main_content = (
        soup.find('article') or
        soup.find('main') or
        soup.find('div', class_='content') or
        soup.find('div', id='content') or
        soup.body
    )

    if not main_content:
        return Document(page_content="", metadata={"source": source})

    # 3. 提取文本
    text = main_content.get_text(separator='\n', strip=True)

    # 4. 清理多余空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = '\n'.join(lines)

    # 5. 提取元数据
    title = soup.find('title')
    h1 = soup.find('h1')

    metadata = {
        "source": source,
        "format": "html",
        "title": title.text if title else (h1.text if h1 else ""),
        "content_length": len(clean_text)
    }

    return Document(page_content=clean_text, metadata=metadata)

# 使用示例
html = """
<html>
  <head><title>RAG技术指南</title></head>
  <body>
    <nav>导航栏...</nav>
    <article>
      <h1>RAG核心流程</h1>
      <p>RAG系统包含三个核心阶段...</p>
      <h2>1. 文档加载</h2>
      <p>文档加载是第一步...</p>
    </article>
    <footer>版权信息...</footer>
  </body>
</html>
"""

doc = load_html_smart(html, "rag_guide.html")
print(doc.page_content)
print(doc.metadata)
```

### 2.3 保留结构信息

```python
from bs4 import BeautifulSoup
from langchain.schema import Document
from typing import List

def load_html_with_structure(html_content: str, source: str) -> List[Document]:
    """提取HTML并保留结构信息"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # 移除噪声
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    documents = []

    # 查找主要内容
    main_content = soup.find('article') or soup.find('main') or soup.body

    if not main_content:
        return documents

    # 按标题分段
    current_section = {"heading": "", "level": 0, "content": []}

    for element in main_content.descendants:
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # 保存上一个section
            if current_section['content']:
                doc = Document(
                    page_content='\n'.join(current_section['content']),
                    metadata={
                        "source": source,
                        "format": "html",
                        "heading": current_section['heading'],
                        "heading_level": current_section['level']
                    }
                )
                documents.append(doc)

            # 开始新section
            level = int(element.name[1])
            current_section = {
                "heading": element.get_text(strip=True),
                "level": level,
                "content": []
            }
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                current_section['content'].append(text)

    # 保存最后一个section
    if current_section['content']:
        doc = Document(
            page_content='\n'.join(current_section['content']),
            metadata={
                "source": source,
                "format": "html",
                "heading": current_section['heading'],
                "heading_level": current_section['level']
            }
        )
        documents.append(doc)

    return documents

# 使用示例
html = """
<article>
  <h1>RAG技术指南</h1>
  <p>RAG系统介绍...</p>

  <h2>核心流程</h2>
  <p>RAG包含三个阶段...</p>

  <h2>实践案例</h2>
  <p>以下是实际案例...</p>
</article>
"""

docs = load_html_with_structure(html, "rag_guide.html")
for doc in docs:
    print(f"标题: {doc.metadata['heading']} (级别: {doc.metadata['heading_level']})")
    print(f"内容: {doc.page_content[:100]}...\n")
```

---

## 3. 网页抓取与加载

### 3.1 使用requests抓取网页

```python
"""
网页抓取基础
适用场景：从URL加载HTML内容
"""

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_webpage(url: str) -> Document:
    """从URL加载网页内容"""
    # 1. 发送HTTP请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # 2. 解析HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # 3. 移除噪声
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    # 4. 提取主要内容
    main_content = soup.find('article') or soup.find('main') or soup.body
    text = main_content.get_text(separator='\n', strip=True) if main_content else ""

    # 5. 清理文本
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = '\n'.join(lines)

    # 6. 提取元数据
    title = soup.find('title')
    metadata = {
        "source": url,
        "format": "html",
        "title": title.text if title else "",
        "url": url,
        "status_code": response.status_code
    }

    return Document(page_content=clean_text, metadata=metadata)

# 使用示例
# doc = load_webpage("https://example.com/article")
# print(doc.page_content[:200])
```

### 3.2 使用LangChain的WebBaseLoader

```python
from langchain.document_loaders import WebBaseLoader

# 加载单个网页
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

print(f"加载了 {len(docs)} 个文档")
print(f"内容: {docs[0].page_content[:200]}...")
```

---

## 4. Markdown解析

### 4.1 基础Markdown解析

```python
"""
Markdown基础解析
适用场景：GitHub README、技术文档
"""

from langchain.schema import Document

def load_markdown_basic(file_path: str) -> Document:
    """基础Markdown加载"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Document(
        page_content=content,
        metadata={
            "source": file_path,
            "format": "markdown"
        }
    )

# 使用示例
doc = load_markdown_basic("README.md")
print(doc.page_content[:200])
```

### 4.2 按标题分段

```python
import re
from langchain.schema import Document
from typing import List

def load_markdown_by_sections(file_path: str) -> List[Document]:
    """按Markdown标题分段加载"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = []
    current_section = {"heading": "", "level": 0, "content": []}

    for line in content.split('\n'):
        # 检测标题
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            # 保存上一个section
            if current_section['content']:
                doc = Document(
                    page_content='\n'.join(current_section['content']),
                    metadata={
                        "source": file_path,
                        "format": "markdown",
                        "heading": current_section['heading'],
                        "heading_level": current_section['level']
                    }
                )
                documents.append(doc)

            # 开始新section
            level = len(match.group(1))
            heading = match.group(2)
            current_section = {
                "heading": heading,
                "level": level,
                "content": []
            }
        else:
            if line.strip():
                current_section['content'].append(line)

    # 保存最后一个section
    if current_section['content']:
        doc = Document(
            page_content='\n'.join(current_section['content']),
            metadata={
                "source": file_path,
                "format": "markdown",
                "heading": current_section['heading'],
                "heading_level": current_section['level']
            }
        )
        documents.append(doc)

    return documents

# 使用示例
docs = load_markdown_by_sections("guide.md")
for doc in docs:
    print(f"标题: {doc.metadata['heading']} (级别: {doc.metadata['heading_level']})")
    print(f"内容: {doc.page_content[:100]}...\n")
```

### 4.3 Markdown转HTML

```python
import markdown
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_markdown_as_html(file_path: str) -> Document:
    """将Markdown转换为HTML后解析"""
    # 1. 读取Markdown
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 2. 转换为HTML
    html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])

    # 3. 解析HTML
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    return Document(
        page_content=text,
        metadata={
            "source": file_path,
            "format": "markdown",
            "converted_to": "html"
        }
    )
```

---

## 5. 高级内容提取

### 5.1 使用readability提取主要内容

```python
"""
使用readability-lxml提取网页主要内容
适用场景：复杂网页的内容提取
"""

from readability import Document as ReadabilityDocument
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_html_with_readability(html_content: str, source: str) -> Document:
    """使用readability提取主要内容"""
    # 1. 使用readability提取
    doc = ReadabilityDocument(html_content)

    # 2. 获取标题和内容
    title = doc.title()
    content_html = doc.summary()

    # 3. 解析HTML
    soup = BeautifulSoup(content_html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    # 4. 清理文本
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = '\n'.join(lines)

    return Document(
        page_content=clean_text,
        metadata={
            "source": source,
            "format": "html",
            "title": title,
            "extraction_method": "readability"
        }
    )

# 使用示例
html = """
<html>
  <body>
    <nav>导航...</nav>
    <article>
      <h1>核心内容</h1>
      <p>这是主要内容...</p>
    </article>
    <footer>页脚...</footer>
  </body>
</html>
"""

# doc = load_html_with_readability(html, "example.html")
# print(doc.page_content)
```

### 5.2 使用trafilatura提取

```python
"""
使用trafilatura提取网页内容
适用场景：新闻文章、博客文章
"""

import trafilatura
from langchain.schema import Document

def load_html_with_trafilatura(html_content: str, source: str) -> Document:
    """使用trafilatura提取内容"""
    # 提取文本
    text = trafilatura.extract(html_content)

    # 提取元数据
    metadata_dict = trafilatura.extract_metadata(html_content)

    metadata = {
        "source": source,
        "format": "html",
        "extraction_method": "trafilatura"
    }

    if metadata_dict:
        metadata.update({
            "title": metadata_dict.title or "",
            "author": metadata_dict.author or "",
            "date": metadata_dict.date or "",
            "url": metadata_dict.url or ""
        })

    return Document(
        page_content=text or "",
        metadata=metadata
    )

# 使用示例
# doc = load_html_with_trafilatura(html, "article.html")
# print(doc.page_content)
```

---

## 6. 统一HTML/Markdown加载器

### 6.1 通用加载器

```python
from typing import List
from langchain.schema import Document
from bs4 import BeautifulSoup
import os

class UnifiedWebLoader:
    """统一的HTML/Markdown加载器"""

    def load(self, source: str) -> List[Document]:
        """
        根据来源类型自动选择加载方法
        source可以是：
        - 文件路径 (.html, .md)
        - URL (http://, https://)
        """
        if source.startswith('http://') or source.startswith('https://'):
            return self._load_from_url(source)
        elif source.endswith('.html') or source.endswith('.htm'):
            return self._load_html_file(source)
        elif source.endswith('.md') or source.endswith('.markdown'):
            return self._load_markdown_file(source)
        else:
            raise ValueError(f"不支持的来源类型: {source}")

    def _load_from_url(self, url: str) -> List[Document]:
        """从URL加载"""
        import requests

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        return self._parse_html(response.content, url)

    def _load_html_file(self, file_path: str) -> List[Document]:
        """从HTML文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        return self._parse_html(html_content, file_path)

    def _load_markdown_file(self, file_path: str) -> List[Document]:
        """从Markdown文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按标题分段
        import re
        documents = []
        current_section = {"heading": "", "level": 0, "content": []}

        for line in content.split('\n'):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                if current_section['content']:
                    doc = Document(
                        page_content='\n'.join(current_section['content']),
                        metadata={
                            "source": file_path,
                            "format": "markdown",
                            "heading": current_section['heading'],
                            "heading_level": current_section['level']
                        }
                    )
                    documents.append(doc)

                level = len(match.group(1))
                heading = match.group(2)
                current_section = {"heading": heading, "level": level, "content": []}
            else:
                if line.strip():
                    current_section['content'].append(line)

        if current_section['content']:
            doc = Document(
                page_content='\n'.join(current_section['content']),
                metadata={
                    "source": file_path,
                    "format": "markdown",
                    "heading": current_section['heading'],
                    "heading_level": current_section['level']
                }
            )
            documents.append(doc)

        return documents

    def _parse_html(self, html_content, source: str) -> List[Document]:
        """解析HTML内容"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # 移除噪声
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            tag.decompose()

        # 提取主要内容
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_='content') or
            soup.body
        )

        if not main_content:
            return []

        # 提取文本
        text = main_content.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)

        # 提取元数据
        title = soup.find('title')
        metadata = {
            "source": source,
            "format": "html",
            "title": title.text if title else ""
        }

        return [Document(page_content=clean_text, metadata=metadata)]

# 使用示例
loader = UnifiedWebLoader()

# 加载HTML文件
docs = loader.load("article.html")

# 加载Markdown文件
docs = loader.load("README.md")

# 加载网页
# docs = loader.load("https://example.com/article")

print(f"加载了 {len(docs)} 个文档")
```

---

## 7. 在RAG中的应用

### 7.1 构建在线文档知识库

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 1. 加载多个网页
loader = UnifiedWebLoader()
urls = [
    "https://docs.python.org/3/tutorial/",
    "https://docs.langchain.com/docs/",
    "https://platform.openai.com/docs/"
]

all_docs = []
for url in urls:
    try:
        docs = loader.load(url)
        all_docs.extend(docs)
        print(f"✅ 加载成功: {url}")
    except Exception as e:
        print(f"❌ 加载失败: {url} - {e}")

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(all_docs)

# 3. 向量化并存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./web_kb"
)

print(f"在线文档知识库构建完成，包含 {len(chunks)} 个文本块")
```

---

## 总结

**HTML与Markdown解析的核心要点：**

1. **HTML解析**: 使用BeautifulSoup，重点是去除噪声
2. **内容提取**: 优先提取`<article>`、`<main>`等主要内容区域
3. **Markdown解析**: 按标题分段，保留文档结构
4. **网页抓取**: 使用requests + BeautifulSoup或LangChain的WebBaseLoader
5. **高级工具**: readability、trafilatura用于复杂网页

**在RAG中的最佳实践:**
- 移除脚本、样式、导航等噪声标签
- 保留文档结构（标题层级）
- 提取元数据（标题、URL、日期）
- 按章节分段创建Document

---

## 参考来源

> **参考来源：**
> - [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) (2025)
> - [LangChain WebBaseLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/web_base) (2025)
> - [Python Markdown](https://python-markdown.github.io/) (2025)
> - [trafilatura](https://trafilatura.readthedocs.io/) - 网页内容提取 (2025)

---

**版本：** v1.0
**最后更新：** 2026-02-15
**下一步：** 阅读 [03_核心概念_07_代码仓库加载.md](./03_核心概念_07_代码仓库加载.md)
