# 实战代码 - 场景 5: HTML 与 Markdown 加载

> 掌握网页内容和 Markdown 文档的加载方法

---

## 概述

HTML 和 Markdown 是互联网和技术文档中最常见的格式。LangChain 提供了多种加载器来处理这些格式，支持从本地文件、网页 URL、甚至动态网页中提取内容。

**本场景涵盖**:
- HTML 文件加载
- 网页内容抓取
- Markdown 文档加载
- 内容清洗与提取
- 批量网页处理
- RAG 管道集成

---

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install langchain langchain-community langchain-openai

# HTML 解析库
pip install beautifulsoup4 lxml html2text

# 网页抓取
pip install requests aiohttp

# Markdown 解析
pip install unstructured markdown
```

### 导入模块

```python
from langchain_community.document_loaders import (
    BSHTMLLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from pathlib import Path
```

---

## 场景 1: HTML 文件加载

### 方法 1: BSHTMLLoader（BeautifulSoup）

```python
from langchain_community.document_loaders import BSHTMLLoader

# 加载 HTML 文件
loader = BSHTMLLoader("page.html")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"内容预览: {documents[0].page_content[:200]}")
print(f"元数据: {documents[0].metadata}")
```

**特点**:
- 使用 BeautifulSoup 解析
- 自动提取纯文本
- 保留基本结构
- 速度快，适合简单 HTML

### 方法 2: UnstructuredHTMLLoader（保留结构）

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

# 加载 HTML 文件（保留结构）
loader = UnstructuredHTMLLoader(
    "page.html",
    mode="elements"  # 保留文档结构
)
documents = loader.load()

# 查看文档结构
for doc in documents[:3]:
    print(f"类型: {doc.metadata.get('category', 'unknown')}")
    print(f"内容: {doc.page_content[:100]}")
    print("---")
```

**特点**:
- 保留 HTML 结构（标题、段落、列表）
- 提取更多元数据
- 支持复杂布局
- 适合结构化内容

---

## 场景 2: 网页内容抓取

### 基础网页加载

```python
from langchain_community.document_loaders import WebBaseLoader

# 从 URL 加载网页
loader = WebBaseLoader("https://example.com")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"标题: {documents[0].metadata.get('title', 'N/A')}")
print(f"内容预览: {documents[0].page_content[:200]}")
```

### 自定义 BeautifulSoup 解析

```python
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

# 自定义解析器 - 只提取特定内容
loader = WebBaseLoader(
    "https://example.com",
    bs_kwargs={
        "parse_only": BeautifulSoup.SoupStrainer(
            name=["article", "main", "div"],
            attrs={"class": ["content", "post-body"]}
        )
    }
)
documents = loader.load()

print(f"提取的内容: {documents[0].page_content[:200]}")
```

### 批量网页加载

```python
from langchain_community.document_loaders import WebBaseLoader

# 批量加载多个网页
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

loader = WebBaseLoader(urls)
documents = loader.load()

print(f"加载了 {len(documents)} 个网页")
for doc in documents:
    print(f"- {doc.metadata.get('source', 'unknown')}")
```

---

## 场景 3: Markdown 文档加载

### 基础 Markdown 加载

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 加载 Markdown 文件
loader = UnstructuredMarkdownLoader("README.md")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"内容预览: {documents[0].page_content[:200]}")
```

### 保留 Markdown 结构

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 加载 Markdown 文件（保留结构）
loader = UnstructuredMarkdownLoader(
    "README.md",
    mode="elements"  # 保留文档结构
)
documents = loader.load()

# 查看文档结构
for doc in documents[:5]:
    category = doc.metadata.get('category', 'unknown')
    content = doc.page_content[:50]
    print(f"{category}: {content}")
```

### 批量 Markdown 加载

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

# 批量加载目录中的所有 Markdown 文件
loader = DirectoryLoader(
    "./docs",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True
)

documents = loader.load()
print(f"加载了 {len(documents)} 个 Markdown 文档")
```

---

## 场景 4: 内容清洗与提取

### 移除 HTML 标签

```python
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def clean_html_content(html_content: str) -> str:
    """清洗 HTML 内容，移除标签和脚本"""
    soup = BeautifulSoup(html_content, "html.parser")

    # 移除 script 和 style 标签
    for script in soup(["script", "style"]):
        script.decompose()

    # 提取纯文本
    text = soup.get_text()

    # 清理空白
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

# 使用
from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("page.html")
documents = loader.load()

# 清洗内容
for doc in documents:
    doc.page_content = clean_html_content(doc.page_content)

print(f"清洗后的内容: {documents[0].page_content[:200]}")
```

### 提取特定元素

```python
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

def extract_article_content(url: str) -> list[Document]:
    """提取网页中的文章内容"""
    loader = WebBaseLoader(url)
    documents = loader.load()

    if not documents:
        return []

    html_content = documents[0].page_content
    soup = BeautifulSoup(html_content, "html.parser")

    # 提取标题
    title = soup.find("h1")
    title_text = title.get_text() if title else "Untitled"

    # 提取正文
    article = soup.find("article") or soup.find("main")
    if article:
        content = article.get_text(separator="\n", strip=True)
    else:
        content = soup.get_text(separator="\n", strip=True)

    # 创建新文档
    doc = Document(
        page_content=content,
        metadata={
            "source": url,
            "title": title_text,
            "type": "article"
        }
    )

    return [doc]

# 使用
documents = extract_article_content("https://example.com/article")
print(f"标题: {documents[0].metadata['title']}")
print(f"内容: {documents[0].page_content[:200]}")
```

---

## 场景 5: 异步批量网页抓取

### 异步加载多个网页

```python
import asyncio
from langchain_community.document_loaders import WebBaseLoader

async def async_load_urls(urls: list[str]) -> list[Document]:
    """异步加载多个网页"""
    loader = WebBaseLoader(urls)
    documents = await loader.aload()
    return documents

# 使用
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

documents = asyncio.run(async_load_urls(urls))
print(f"异步加载了 {len(documents)} 个网页")
```

### 并发控制

```python
import asyncio
import aiohttp
from langchain_core.documents import Document

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Document:
    """异步抓取单个 URL"""
    try:
        async with session.get(url, timeout=10) as response:
            html = await response.text()
            return Document(
                page_content=html,
                metadata={"source": url, "status": response.status}
            )
    except Exception as e:
        print(f"抓取 {url} 失败: {e}")
        return None

async def batch_fetch_urls(urls: list[str], max_concurrent: int = 5) -> list[Document]:
    """批量异步抓取，控制并发数"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(session, url):
        async with semaphore:
            return await fetch_url(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    # 过滤失败的请求
    return [doc for doc in results if doc is not None]

# 使用
urls = [f"https://example.com/page{i}" for i in range(1, 11)]
documents = asyncio.run(batch_fetch_urls(urls, max_concurrent=5))
print(f"成功抓取 {len(documents)}/{len(urls)} 个网页")
```

---

## 场景 6: RAG 管道集成

### 完整的网页 RAG 系统

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader

# 1. 批量加载网页
urls = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://docs.python.org/3/library/index.html"
]

loader = WebBaseLoader(urls)
documents = loader.load()
print(f"加载了 {len(documents)} 个网页")

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"分块后: {len(chunks)} 个 chunks")

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="web_docs"
)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. 创建 QA 链
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. 查询
query = "Python 中如何处理异常？"
result = qa_chain.invoke({"query": query})

print(f"问题: {query}")
print(f"答案: {result['result']}")
print(f"\n来源:")
for doc in result['source_documents']:
    print(f"- {doc.metadata.get('source', 'unknown')}")
```

---

## 场景 7: Markdown 文档 RAG

### 技术文档问答系统

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains import RetrievalQA

# 1. 加载所有 Markdown 文档
loader = DirectoryLoader(
    "./docs",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True
)
documents = loader.load()
print(f"加载了 {len(documents)} 个 Markdown 文档")

# 2. 使用 Markdown 专用分块器
text_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"分块后: {len(chunks)} 个 chunks")

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="markdown_docs"
)

# 4. 创建 QA 链
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 5. 查询
query = "如何安装这个项目？"
result = qa_chain.invoke({"query": query})

print(f"问题: {query}")
print(f"答案: {result['result']}")
```

---

## 场景 8: 错误处理与容错

### 健壮的网页加载

```python
import logging
from typing import Optional
from langchain_community.document_loaders import WebBaseLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_load_url(url: str, timeout: int = 10) -> Optional[list[Document]]:
    """安全加载网页，带错误处理"""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        logger.info(f"成功加载 {url}: {len(documents)} 个文档")
        return documents
    except Exception as e:
        logger.error(f"加载 {url} 失败: {e}")
        return None

def batch_load_urls_with_retry(
    urls: list[str],
    max_retries: int = 3
) -> list[Document]:
    """批量加载网页，自动重试"""
    all_documents = []
    failed_urls = []

    for url in urls:
        documents = None
        for attempt in range(max_retries):
            documents = safe_load_url(url)
            if documents:
                all_documents.extend(documents)
                break
            logger.warning(f"重试 {url} ({attempt + 1}/{max_retries})")

        if not documents:
            failed_urls.append(url)

    print(f"成功: {len(urls) - len(failed_urls)}/{len(urls)}")
    if failed_urls:
        print(f"失败的 URL: {failed_urls}")

    return all_documents

# 使用
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://invalid-url.com"  # 这个会失败
]

documents = batch_load_urls_with_retry(urls, max_retries=3)
print(f"总共加载了 {len(documents)} 个文档")
```

---

## 总结

### HTML/Markdown 加载器对比

| 加载器 | 适用格式 | 优点 | 缺点 | 推荐场景 |
|--------|----------|------|------|----------|
| BSHTMLLoader | HTML | 快速、简单 | 只提取纯文本 | 简单网页 |
| UnstructuredHTMLLoader | HTML | 保留结构、功能强大 | 依赖多、速度慢 | 复杂网页 |
| WebBaseLoader | URL | 直接抓取网页 | 需要网络连接 | 在线内容 |
| UnstructuredMarkdownLoader | Markdown | 保留 Markdown 结构 | 需要额外配置 | 技术文档 |

### 最佳实践

1. **选择合适的加载器**:
   - 简单 HTML → BSHTMLLoader
   - 复杂 HTML → UnstructuredHTMLLoader
   - 网页抓取 → WebBaseLoader
   - Markdown → UnstructuredMarkdownLoader

2. **内容清洗**:
   - 移除 script 和 style 标签
   - 提取特定元素（article、main）
   - 清理多余空白

3. **批量处理**:
   - 使用异步加载提升性能
   - 控制并发数避免被封禁
   - 添加错误处理和重试机制

4. **RAG 集成**:
   - 使用专用分块器（MarkdownTextSplitter）
   - 保留元数据（URL、标题）
   - 合理设置 chunk_size

---

## 数据来源

- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/fetch_1_8_b1b3cc_02.md | LangChain-OpenTutorial GitHub 教程]
- [来源: 03_核心概念_2_BaseLoader接口设计.md | BaseLoader 接口设计]
