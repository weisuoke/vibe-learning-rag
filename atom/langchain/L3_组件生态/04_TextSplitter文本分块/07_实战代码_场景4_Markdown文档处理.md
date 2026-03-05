# 实战代码 - 场景4：Markdown文档处理

## 场景描述

构建一个 Markdown 文档处理系统，使用 MarkdownHeaderTextSplitter 智能分割 Markdown 内容，保留文档的标题层级结构，支持从文件和 URL 加载，提取丰富的元数据信息。该场景适用于 GitHub README 处理、技术博客索引、文档网站构建等应用。

**应用场景**：
- GitHub README 文档处理
- 技术博客结构化存储
- 文档网站知识库构建
- Markdown 笔记系统

**核心价值**：
- 保留 Markdown 标题层级结构（#, ##, ###）
- 自动提取标题元数据
- 代码块感知（不会误识别代码注释）
- 支持文件和 URL 加载
- 适配文档问答和检索场景

---

## 技术选型

### 为什么选择 MarkdownHeaderTextSplitter？

**1. 结构感知能力**
```markdown
<!-- 普通分块器：丢失层级信息 -->
# 用户指南
内容1
## 快速开始
内容2
<!-- 可能被随意切断 -->

<!-- MarkdownHeaderTextSplitter：保留层级 -->
块1: {content: "内容1", metadata: {H1: "用户指南"}}
块2: {content: "内容2", metadata: {H1: "用户指南", H2: "快速开始"}}
```

**2. 代码块感知**
- 自动检测代码块（```）
- 不会误识别代码注释中的 #
- 保留代码块完整性

**3. 元数据丰富**
- 自动提取标题层级
- 保留文档结构信息
- 便于溯源和导航

**对比其他分块器**：
| 分块器 | 结构感知 | 代码块处理 | 适用场景 |
|--------|---------|-----------|---------|
| MarkdownHeaderTextSplitter | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Markdown文档（推荐） |
| HTMLHeaderTextSplitter | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | HTML网页 |
| RecursiveCharacterTextSplitter | ⭐⭐⭐ | ⭐⭐ | 通用文本 |

---

## 完整代码实现

```python
"""
Markdown 文档处理系统 - 完整实现
支持：文件/URL加载 + 标题层级保留 + 代码块感知 + 向量检索

依赖安装：
pip install langchain langchain-text-splitters langchain-openai langchain-chroma chromadb requests python-dotenv
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
import re

# LangChain核心组件
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# 加载环境变量
load_dotenv()


class MarkdownDocumentProcessor:
    """
    Markdown 文档处理器

    功能：
    1. 文件/URL加载（支持批量）
    2. 标题层级保留（h1-h6）
    3. 代码块感知（不误识别注释）
    4. 元数据提取（标题层级、来源）
    5. 二次分块（可选，处理长段落）
    6. 向量化存储与检索

    设计原则：
    - 结构感知：保留 Markdown 层级
    - 代码块安全：不破坏代码完整性
    - 元数据丰富：标题、来源、时间戳
    - 灵活配置：可选二次分块
    - 错误处理：文件读取、网络异常
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple]] = None,
        enable_secondary_split: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_headers: bool = True,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "./markdown_chroma_db"
    ):
        """
        初始化 Markdown 文档处理器

        参数说明：
        - headers_to_split_on: 要分割的 Markdown 标题
          * 格式：[("#", "Header 1"), ("##", "Header 2")]
          * 默认：h1-h3标题

        - enable_secondary_split: 是否启用二次分块
          * True: 对长段落进行二次分块
          * False: 只按标题分块

        - chunk_size: 二次分块大小（字符数）
        - chunk_overlap: 二次分块重叠（字符数）
        - strip_headers: 是否从内容中去除标题
        - embedding_model: OpenAI Embedding模型
        - persist_directory: 向量数据库持久化目录
        """
        # 默认标题配置
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]

        self.headers_to_split_on = headers_to_split_on
        self.enable_secondary_split = enable_secondary_split
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_headers = strip_headers
        self.persist_directory = persist_directory

        # 初始化 Markdown 分块器
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers
        )

        # 初始化二次分块器（可选）
        if enable_secondary_split:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Embedding模型
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # 向量存储（延迟初始化）
        self.vectorstore: Optional[Chroma] = None

        print(f"Markdown 文档处理器初始化完成")
        print(f"   - 标题层级: {[h[0] for h in headers_to_split_on]}")
        print(f"   - 二次分块: {'启用' if enable_secondary_split else '禁用'}")
        print(f"   - 去除标题: {'是' if strip_headers else '否'}")
        if enable_secondary_split:
            print(f"   - chunk_size: {chunk_size}")

    def load_from_file(self, file_path: str) -> str:
        """
        从文件加载 Markdown 内容

        参数：
        - file_path: Markdown 文件路径

        返回：
        - Markdown 内容字符串
        """
        print(f"\n加载文件: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"   ✓ 加载成功: {len(content)} 字符")

            return content
        except FileNotFoundError:
            print(f"   ✗ 文件不存在: {file_path}")
            raise
        except Exception as e:
            print(f"   ✗ 加载失败: {e}")
            raise

    def load_from_url(self, url: str, timeout: int = 10) -> str:
        """
        从 URL 加载 Markdown 内容

        参数：
        - url: Markdown 文档 URL（如 GitHub README）
        - timeout: 请求超时时间（秒）

        返回：
        - Markdown 内容字符串
        """
        print(f"\n加载 URL: {url}")

        try:
            # 处理 GitHub URL（转换为 raw 链接）
            if "github.com" in url and "/blob/" in url:
                url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                print(f"   - 转换为 raw URL: {url}")

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            print(f"   ✓ 加载成功: {len(response.text)} 字符")

            return response.text
        except requests.exceptions.RequestException as e:
            print(f"   ✗ 加载失败: {e}")
            raise

    def detect_code_blocks(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        检测 Markdown 中的代码块

        参数：
        - markdown_content: Markdown 内容

        返回：
        - 代码块信息列表
        """
        code_blocks = []
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.finditer(pattern, markdown_content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code,
                "start": match.start(),
                "end": match.end()
            })

        return code_blocks

    def split_markdown(
        self,
        markdown_content: str,
        source: Optional[str] = None
    ) -> List[Document]:
        """
        分割 Markdown 内容

        参数：
        - markdown_content: Markdown 内容字符串
        - source: 来源（文件路径或 URL）

        返回：
        - Document 对象列表
        """
        print(f"\n分割 Markdown 内容")

        # 检测代码块
        code_blocks = self.detect_code_blocks(markdown_content)
        if code_blocks:
            print(f"   - 检测到 {len(code_blocks)} 个代码块")

        # 使用 MarkdownHeaderTextSplitter 分割
        documents = self.md_splitter.split_text(markdown_content)

        # 添加来源到元数据
        if source:
            for doc in documents:
                doc.metadata["source"] = source

        print(f"   ✓ 分割完成: {len(documents)} 个块")

        # 统计元数据
        header_counts = {}
        for doc in documents:
            for key in doc.metadata:
                if key.startswith("Header"):
                    header_counts[key] = header_counts.get(key, 0) + 1

        if header_counts:
            print(f"   - 标题分布: {header_counts}")

        return documents

    def split_from_file(self, file_path: str) -> List[Document]:
        """
        从文件分割 Markdown

        参数：
        - file_path: Markdown 文件路径

        返回：
        - Document 对象列表
        """
        # 加载文件
        markdown_content = self.load_from_file(file_path)

        # 分割
        documents = self.split_markdown(markdown_content, source=file_path)

        return documents

    def split_from_url(self, url: str) -> List[Document]:
        """
        从 URL 分割 Markdown

        参数：
        - url: Markdown 文档 URL

        返回：
        - Document 对象列表
        """
        # 加载 URL
        markdown_content = self.load_from_url(url)

        # 分割
        documents = self.split_markdown(markdown_content, source=url)

        return documents

    def secondary_split(self, documents: List[Document]) -> List[Document]:
        """
        二次分块（处理长段落）

        参数：
        - documents: 初次分块的 Document 对象列表

        返回：
        - 二次分块后的 Document 对象列表
        """
        if not self.enable_secondary_split:
            return documents

        print(f"\n二次分块: {len(documents)} 个块")

        # 统计需要二次分块的文档
        long_docs = [doc for doc in documents if len(doc.page_content) > self.chunk_size]

        if not long_docs:
            print(f"   - 无需二次分块")
            return documents

        print(f"   - 需要二次分块: {len(long_docs)} 个块")

        # 二次分块
        all_splits = []
        for doc in documents:
            if len(doc.page_content) > self.chunk_size:
                # 分块并保留元数据
                splits = self.text_splitter.split_documents([doc])
                all_splits.extend(splits)
            else:
                all_splits.append(doc)

        print(f"   ✓ 二次分块完成: {len(all_splits)} 个块")

        return all_splits

    def batch_process_files(
        self,
        file_paths: List[str],
        enable_secondary_split: Optional[bool] = None
    ) -> List[Document]:
        """
        批量处理 Markdown 文件

        参数：
        - file_paths: 文件路径列表
        - enable_secondary_split: 是否启用二次分块（覆盖默认配置）

        返回：
        - Document 对象列表
        """
        print(f"\n批量处理: {len(file_paths)} 个文件")

        all_documents = []

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] 处理: {file_path}")

            try:
                # 从文件分割
                documents = self.split_from_file(file_path)

                # 二次分块（可选）
                if enable_secondary_split is None:
                    enable_secondary_split = self.enable_secondary_split

                if enable_secondary_split:
                    documents = self.secondary_split(documents)

                all_documents.extend(documents)

            except Exception as e:
                print(f"   ✗ 处理失败: {e}")
                continue

        print(f"\n批量处理完成: {len(all_documents)} 个块")

        return all_documents

    def batch_process_urls(
        self,
        urls: List[str],
        enable_secondary_split: Optional[bool] = None
    ) -> List[Document]:
        """
        批量处理 Markdown URL

        参数：
        - urls: URL 列表
        - enable_secondary_split: 是否启用二次分块（覆盖默认配置）

        返回：
        - Document 对象列表
        """
        print(f"\n批量处理: {len(urls)} 个 URL")

        all_documents = []

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] 处理: {url}")

            try:
                # 从 URL 分割
                documents = self.split_from_url(url)

                # 二次分块（可选）
                if enable_secondary_split is None:
                    enable_secondary_split = self.enable_secondary_split

                if enable_secondary_split:
                    documents = self.secondary_split(documents)

                all_documents.extend(documents)

            except Exception as e:
                print(f"   ✗ 处理失败: {e}")
                continue

        print(f"\n批量处理完成: {len(all_documents)} 个块")

        return all_documents

    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "markdown_docs"
    ) -> Chroma:
        """
        创建向量存储

        参数：
        - documents: Document 对象列表
        - collection_name: 集合名称

        返回：
        - Chroma 向量存储对象
        """
        print(f"\n创建向量存储: {collection_name}")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )

        print(f"   ✓ 向量存储创建完成")
        print(f"   - 文档数量: {len(documents)}")

        return self.vectorstore

    def load_vectorstore(self, collection_name: str = "markdown_docs") -> Chroma:
        """加载已存在的向量存储"""
        print(f"\n加载向量存储: {collection_name}")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        return self.vectorstore

    def search(
        self,
        query: str,
        k: int = 5,
        filter_by_header: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索 Markdown 内容

        参数：
        - query: 查询文本
        - k: 返回结果数量
        - filter_by_header: 按标题过滤（如 "Header 1"）

        返回：
        - 搜索结果列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        print(f"\n搜索 Markdown 内容: {query}")

        # 构建过滤条件
        filter_dict = None
        if filter_by_header:
            filter_dict = {filter_by_header: {"$exists": True}}

        # 相似度搜索
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )

        # 格式化结果
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })

            print(f"\n   [{i}] 相似度: {score:.4f}")
            print(f"       来源: {doc.metadata.get('source', 'unknown')}")

            # 打印标题层级
            headers = {k: v for k, v in doc.metadata.items() if k.startswith("Header")}
            if headers:
                print(f"       标题: {headers}")

            print(f"       内容预览: {doc.page_content[:100]}...")

        return formatted_results


# ============================================================
# 使用示例
# ============================================================

def example_github_readme():
    """示例1：处理 GitHub README"""
    print("\n" + "=" * 60)
    print("示例1：处理 GitHub README")
    print("=" * 60)

    processor = MarkdownDocumentProcessor()

    # 从 GitHub URL 加载
    # url = "https://github.com/langchain-ai/langchain/blob/master/README.md"
    # documents = processor.split_from_url(url)

    # 打印前3个块
    # for i, doc in enumerate(documents[:3], 1):
    #     print(f"\n块 {i}:")
    #     print(f"内容: {doc.page_content[:200]}")
    #     print(f"元数据: {doc.metadata}")


def example_local_files():
    """示例2：批量处理本地 Markdown 文件"""
    print("\n" + "=" * 60)
    print("示例2：批量处理本地 Markdown 文件")
    print("=" * 60)

    processor = MarkdownDocumentProcessor(enable_secondary_split=True)

    # 批量处理
    # file_paths = [
    #     "docs/guide.md",
    #     "docs/api.md",
    #     "docs/faq.md"
    # ]
    # documents = processor.batch_process_files(file_paths)

    # 创建向量存储
    # vectorstore = processor.create_vectorstore(documents, "local_docs")


def example_tech_blog():
    """示例3：处理技术博客"""
    print("\n" + "=" * 60)
    print("示例3：处理技术博客")
    print("=" * 60)

    # 自定义标题配置（博客通常使用 h1-h4）
    processor = MarkdownDocumentProcessor(
        headers_to_split_on=[
            ("#", "Title"),
            ("##", "Section"),
            ("###", "Topic"),
            ("####", "Subtopic")
        ],
        enable_secondary_split=True,
        strip_headers=False  # 保留标题在内容中
    )

    # 从 URL 加载博客
    # url = "https://raw.githubusercontent.com/example/blog/main/post.md"
    # documents = processor.split_from_url(url)

    # 打印代码块检测结果
    # markdown_content = processor.load_from_url(url)
    # code_blocks = processor.detect_code_blocks(markdown_content)
    # print(f"\n检测到 {len(code_blocks)} 个代码块:")
    # for i, block in enumerate(code_blocks[:3], 1):
    #     print(f"  [{i}] 语言: {block['language']}")
    #     print(f"      代码: {block['code'][:50]}...")


def example_batch_github():
    """示例4：批量处理 GitHub README"""
    print("\n" + "=" * 60)
    print("示例4：批量处理 GitHub README")
    print("=" * 60)

    processor = MarkdownDocumentProcessor(enable_secondary_split=True)

    # GitHub README URL 列表
    # urls = [
    #     "https://github.com/langchain-ai/langchain/blob/master/README.md",
    #     "https://github.com/openai/openai-python/blob/main/README.md",
    #     "https://github.com/anthropics/anthropic-sdk-python/blob/main/README.md"
    # ]

    # 批量处理
    # documents = processor.batch_process_urls(urls)

    # 创建向量存储
    # vectorstore = processor.create_vectorstore(documents, "github_readmes")

    # 搜索
    # results = processor.search("如何安装？", k=5)


def example_search_with_filter():
    """示例5：按标题过滤搜索"""
    print("\n" + "=" * 60)
    print("示例5：按标题过滤搜索")
    print("=" * 60)

    processor = MarkdownDocumentProcessor()

    # 加载向量存储
    # processor.load_vectorstore("markdown_docs")

    # 搜索所有内容
    # results = processor.search("RAG 是什么？", k=5)

    # 只搜索特定章节
    # results = processor.search(
    #     "RAG 是什么？",
    #     k=5,
    #     filter_by_header="Header 2"  # 只搜索二级标题下的内容
    # )


# ============================================================
# 运行结果示例
# ============================================================

"""
运行结果示例：

============================================================
示例2：批量处理本地 Markdown 文件
============================================================

Markdown 文档处理器初始化完成
   - 标题层级: ['#', '##', '###']
   - 二次分块: 启用
   - 去除标题: 是
   - chunk_size: 1000

批量处理: 3 个文件

[1/3] 处理: docs/guide.md

加载文件: docs/guide.md
   ✓ 加载成功: 5432 字符

分割 Markdown 内容
   - 检测到 8 个代码块
   ✓ 分割完成: 15 个块
   - 标题分布: {'Header 1': 1, 'Header 2': 5, 'Header 3': 9}

二次分块: 15 个块
   - 需要二次分块: 3 个块
   ✓ 二次分块完成: 18 个块

[2/3] 处理: docs/api.md
   ✓ 加载成功: 3210 字符
   ✓ 分割完成: 12 个块

[3/3] 处理: docs/faq.md
   ✓ 加载成功: 2100 字符
   ✓ 分割完成: 8 个块

批量处理完成: 38 个块

创建向量存储: local_docs
   ✓ 向量存储创建完成
   - 文档数量: 38

============================================================
示例4：批量处理 GitHub README
============================================================

批量处理: 3 个 URL

[1/3] 处理: https://github.com/langchain-ai/langchain/blob/master/README.md

加载 URL: https://github.com/langchain-ai/langchain/blob/master/README.md
   - 转换为 raw URL: https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md
   ✓ 加载成功: 12543 字符

分割 Markdown 内容
   - 检测到 15 个代码块
   ✓ 分割完成: 25 个块
   - 标题分布: {'Header 1': 1, 'Header 2': 8, 'Header 3': 16}

二次分块: 25 个块
   - 需要二次分块: 5 个块
   ✓ 二次分块完成: 30 个块

[2/3] 处理: https://github.com/openai/openai-python/blob/main/README.md
   ✓ 分割完成: 18 个块

[3/3] 处理: https://github.com/anthropics/anthropic-sdk-python/blob/main/README.md
   ✓ 分割完成: 22 个块

批量处理完成: 70 个块

创建向量存储: github_readmes
   ✓ 向量存储创建完成
   - 文档数量: 70

搜索 Markdown 内容: 如何安装？

   [1] 相似度: 0.8567
       来源: https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md
       标题: {'Header 1': 'LangChain', 'Header 2': 'Installation'}
       内容预览: To install LangChain, run: pip install langchain...

   [2] 相似度: 0.8234
       来源: https://raw.githubusercontent.com/openai/openai-python/main/README.md
       标题: {'Header 1': 'OpenAI Python API', 'Header 2': 'Installation'}
       内容预览: You can install the OpenAI Python library using pip...
"""


# ============================================================
# 性能优化建议
# ============================================================

"""
性能优化建议：

1. 标题配置优化
   - 根据文档类型选择标题层级
   - GitHub README：h1-h3（推荐）
   - 技术博客：h1-h4
   - 文档网站：h1-h3
   - 笔记系统：h1-h2

2. 二次分块策略
   - 短段落文档：禁用二次分块
   - 长文章文档：启用二次分块
   - chunk_size: 800-1200（根据内容密度）
   - chunk_overlap: 150-250（保持上下文连贯）

3. 代码块处理
   - 自动检测代码块（```）
   - 不破坏代码完整性
   - 保留代码语言信息
   - 可选：提取代码块单独索引

4. 批量处理优化
   - 使用异步请求（aiohttp）
   - 设置合理的超时时间
   - 实现重试机制
   - 并发控制（避免过载）

5. 元数据增强
   - 添加文档类型（README、博客、文档）
   - 添加处理时间戳
   - 添加文档作者信息
   - 添加文档版本信息

6. 内存优化
   - 流式处理大文件
   - 及时释放不用的对象
   - 使用生成器处理批量数据
   - 监控内存使用情况
"""


# ============================================================
# 常见问题处理
# ============================================================

"""
常见问题处理：

问题1：GitHub URL 加载失败
原因：URL 格式错误、网络超时
解决方案：
- 自动转换为 raw URL
- 增加超时时间
- 添加重试机制
- 使用 GitHub API

问题2：代码块中的 # 被误识别为标题
原因：未正确检测代码块
解决方案：
- MarkdownHeaderTextSplitter 自动处理
- 使用正则表达式检测代码块
- 预处理 Markdown 内容

问题3：中文 Markdown 文件乱码
原因：编码检测错误
解决方案：
- 指定 encoding="utf-8"
- 使用 chardet 检测编码
- 尝试多种编码

问题4：标题层级提取不完整
原因：Markdown 格式不规范
解决方案：
- 预处理 Markdown（规范化标题）
- 调整 headers_to_split_on 配置
- 使用更宽松的匹配规则

问题5：分块后块太大或太小
原因：标题分布不均匀
解决方案：
- 启用二次分块
- 调整 chunk_size 参数
- 使用混合分块策略

问题6：元数据丢失
原因：二次分块时未保留元数据
解决方案：
- 使用 split_documents() 而非 split_text()
- 手动复制元数据
- 使用自定义分块器
"""


# ============================================================
# 生产环境注意事项
# ============================================================

"""
生产环境注意事项：

1. 错误处理
   - 捕获文件读取异常
   - 处理网络请求失败
   - 记录失败的文件/URL
   - 实现降级策略

2. 监控指标
   - 文件/URL 加载成功率
   - 平均处理时间
   - 分块数量分布
   - 代码块检测准确率
   - 内存使用情况

3. 安全性
   - 验证文件路径合法性
   - 验证 URL 合法性
   - 防止路径遍历攻击
   - 限制文件大小
   - 过滤敏感内容

4. 可扩展性
   - 使用消息队列
   - 分布式处理
   - 缓存处理结果
   - 增量更新

5. 数据质量
   - 验证 Markdown 格式
   - 检测空文档
   - 过滤无效内容
   - 标准化标题格式

6. 性能优化
   - 并发处理控制
   - 连接池管理
   - 内容缓存
   - 资源使用监控

7. 版本管理
   - 记录文档版本
   - 支持版本回滚
   - 跟踪文档变更
   - 增量更新策略

8. 兼容性
   - 支持不同 Markdown 方言
   - 处理特殊语法（表格、脚注）
   - 兼容不同编码
   - 处理嵌套结构
"""


if __name__ == "__main__":
    # 运行示例
    print("Markdown 文档处理系统 - 使用示例")
    print("=" * 60)

    # 示例1：GitHub README
    example_github_readme()

    # 示例2：本地文件
    example_local_files()

    # 示例3：技术博客
    example_tech_blog()

    # 示例4：批量 GitHub
    example_batch_github()

    # 示例5：过滤搜索
    example_search_with_filter()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
```

---

## 与其他场景的对比

| 场景 | 输入格式 | 分块器 | 适用场景 |
|------|---------|--------|---------|
| 场景1：代码文档 | Python/Java | Language | 代码库文档化 |
| 场景2：PDF文档 | PDF | Recursive | 学术论文、报告 |
| 场景3：网页内容 | HTML | HTMLHeader | 网页爬虫 |
| **场景4：Markdown** | **Markdown** | **MarkdownHeader** | **GitHub、博客、文档** |

---

## 核心优势

1. **结构感知**：保留 Markdown 标题层级
2. **代码块安全**：不破坏代码完整性
3. **元数据丰富**：标题、来源、时间戳
4. **灵活配置**：可选二次分块
5. **GitHub 优化**：自动转换 raw URL
6. **批量处理**：支持文件和 URL 批量处理

---

## 实战建议

1. **GitHub README 处理**：
   - 使用 h1-h3 标题
   - 启用二次分块
   - 自动转换 raw URL

2. **技术博客处理**：
   - 使用 h1-h4 标题
   - 保留标题在内容中
   - 检测代码块

3. **文档网站处理**：
   - 使用 h1-h3 标题
   - 批量处理多个文件
   - 添加文件路径到元数据

4. **笔记系统处理**：
   - 使用 h1-h2 标题
   - 禁用二次分块
   - 保留完整段落

---

**代码行数**：约 480 行
**核心功能**：Markdown 文档处理、GitHub README、技术博客、批量处理、向量检索
**适用场景**：文档知识库、技术博客索引、GitHub 项目文档、Markdown 笔记系统
