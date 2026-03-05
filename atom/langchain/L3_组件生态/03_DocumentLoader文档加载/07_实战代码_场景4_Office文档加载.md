# 实战代码 - 场景 4: Office 文档加载

> 掌握 Word、Excel、PowerPoint 文档的加载方法

---

## 概述

Office 文档（Word、Excel、PowerPoint）是企业和学术环境中最常见的文档格式。LangChain 提供了多种加载器来处理这些格式，每种加载器都有其特定的优势和适用场景。

**本场景涵盖**:
- Word 文档加载（.docx）
- Excel 表格加载（.xlsx）
- PowerPoint 演示文稿加载（.pptx）
- 统一 Office 文档加载
- 批量 Office 文档处理
- RAG 管道集成

---

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install langchain langchain-community langchain-openai

# Office 文档解析库
pip install unstructured python-docx openpyxl python-pptx

# 可选：增强解析能力
pip install unstructured[local-inference]
```

### 导入模块

```python
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    DirectoryLoader
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
```

---

## 场景 1: Word 文档加载

### 方法 1: Docx2txtLoader（简单快速）

```python
from langchain_community.document_loaders import Docx2txtLoader

# 加载 Word 文档
loader = Docx2txtLoader("document.docx")
documents = loader.load()

print(f"加载了 {len(documents)} 个文档")
print(f"内容预览: {documents[0].page_content[:200]}")
print(f"元数据: {documents[0].metadata}")
```

**特点**:
- 速度快，依赖少
- 只提取纯文本
- 不保留格式信息
- 适合简单文档

### 方法 2: UnstructuredFileLoader（功能强大）

```python
from langchain_community.document_loaders import UnstructuredFileLoader

# 加载 Word 文档（保留结构）
loader = UnstructuredFileLoader(
    "document.docx",
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
- 保留文档结构（标题、段落、列表）
- 提取更多元数据
- 支持复杂格式
- 适合结构化文档

---

## 场景 2: Excel 表格加载

### 基础加载

```python
from langchain_community.document_loaders import UnstructuredExcelLoader

# 加载 Excel 文件
loader = UnstructuredExcelLoader(
    "data.xlsx",
    mode="elements"  # 保留表格结构
)
documents = loader.load()

print(f"加载了 {len(documents)} 个元素")
for doc in documents[:3]:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print("---")
```

### 指定工作表

```python
# 只加载特定工作表
loader = UnstructuredExcelLoader(
    "data.xlsx",
    mode="elements"
)
documents = loader.load()

# 过滤特定工作表的内容
sheet_name = "Sheet1"
filtered_docs = [
    doc for doc in documents
    if doc.metadata.get("sheet_name") == sheet_name
]
```

### 处理多个工作表

```python
import pandas as pd
from langchain_core.documents import Document

def load_excel_sheets(file_path: str) -> list[Document]:
    """加载 Excel 的所有工作表"""
    documents = []

    # 读取所有工作表
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 转换为文本
        content = df.to_string(index=False)

        # 创建 Document
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "sheet_name": sheet_name,
                "rows": len(df),
                "columns": len(df.columns)
            }
        )
        documents.append(doc)

    return documents

# 使用
documents = load_excel_sheets("data.xlsx")
print(f"加载了 {len(documents)} 个工作表")
```

---

## 场景 3: PowerPoint 演示文稿加载

### 基础加载

```python
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# 加载 PowerPoint 文件
loader = UnstructuredPowerPointLoader(
    "presentation.pptx",
    mode="elements"  # 保留幻灯片结构
)
documents = loader.load()

print(f"加载了 {len(documents)} 个元素")
for doc in documents[:3]:
    print(f"类型: {doc.metadata.get('category', 'unknown')}")
    print(f"内容: {doc.page_content[:100]}")
    print("---")
```

### 按幻灯片分组

```python
def group_by_slide(documents: list[Document]) -> dict[int, list[Document]]:
    """按幻灯片编号分组"""
    slides = {}

    for doc in documents:
        slide_num = doc.metadata.get("page_number", 0)
        if slide_num not in slides:
            slides[slide_num] = []
        slides[slide_num].append(doc)

    return slides

# 使用
slides = group_by_slide(documents)
print(f"共有 {len(slides)} 张幻灯片")

# 查看第一张幻灯片的内容
if 1 in slides:
    print("第一张幻灯片内容:")
    for doc in slides[1]:
        print(f"- {doc.page_content[:50]}")
```

---

## 场景 4: 统一 Office 文档加载

### 自动识别文件类型

```python
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader

def load_office_document(file_path: str) -> list[Document]:
    """自动识别并加载 Office 文档"""
    suffix = Path(file_path).suffix.lower()

    # 根据文件类型选择加载器
    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif suffix == ".xlsx":
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif suffix == ".pptx":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    else:
        # 使用通用加载器
        loader = UnstructuredFileLoader(file_path, mode="elements")

    return loader.load()

# 使用
files = ["document.docx", "data.xlsx", "presentation.pptx"]
all_documents = []

for file in files:
    docs = load_office_document(file)
    all_documents.extend(docs)
    print(f"从 {file} 加载了 {len(docs)} 个文档")

print(f"总共加载了 {len(all_documents)} 个文档")
```

---

## 场景 5: 批量 Office 文档加载

### 使用 DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

# 批量加载目录中的所有 Office 文档
loader = DirectoryLoader(
    "./office_docs",
    glob="**/*.{docx,xlsx,pptx}",  # 匹配所有 Office 文件
    loader_cls=UnstructuredFileLoader,
    loader_kwargs={"mode": "elements"},
    show_progress=True,
    use_multithreading=True  # 并行加载
)

documents = loader.load()
print(f"批量加载了 {len(documents)} 个文档")
```

### 自定义批量加载

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_load_office_files(directory: str, max_workers: int = 4) -> list[Document]:
    """并行批量加载 Office 文档"""
    # 查找所有 Office 文件
    office_files = []
    for ext in [".docx", ".xlsx", ".pptx"]:
        office_files.extend(Path(directory).rglob(f"*{ext}"))

    print(f"找到 {len(office_files)} 个 Office 文件")

    all_documents = []

    # 并行加载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_office_document, str(file)): file
            for file in office_files
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                docs = future.result()
                all_documents.extend(docs)
                print(f"✓ 加载 {file.name}: {len(docs)} 个文档")
            except Exception as e:
                print(f"✗ 加载 {file.name} 失败: {e}")

    return all_documents

# 使用
documents = batch_load_office_files("./office_docs", max_workers=4)
print(f"总共加载了 {len(documents)} 个文档")
```

---

## 场景 6: RAG 管道集成

### 完整的 Office 文档 RAG 系统

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. 批量加载 Office 文档
documents = batch_load_office_files("./office_docs")
print(f"加载了 {len(documents)} 个文档")

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"分块后: {len(chunks)} 个 chunks")

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="office_docs"
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
query = "文档中提到的主要观点是什么？"
result = qa_chain.invoke({"query": query})

print(f"问题: {query}")
print(f"答案: {result['result']}")
print(f"\n来源文档:")
for doc in result['source_documents']:
    print(f"- {doc.metadata.get('source', 'unknown')}")
```

---

## 场景 7: 元数据增强

### 提取文档属性

```python
from docx import Document as DocxDocument
from datetime import datetime

def extract_word_metadata(file_path: str) -> dict:
    """提取 Word 文档的元数据"""
    doc = DocxDocument(file_path)
    core_properties = doc.core_properties

    return {
        "title": core_properties.title or "",
        "author": core_properties.author or "",
        "created": core_properties.created.isoformat() if core_properties.created else "",
        "modified": core_properties.modified.isoformat() if core_properties.modified else "",
        "subject": core_properties.subject or "",
        "keywords": core_properties.keywords or "",
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables)
    }

# 加载文档并增强元数据
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
documents = loader.load()

# 增强元数据
file_metadata = extract_word_metadata("document.docx")
for doc in documents:
    doc.metadata.update(file_metadata)

print(f"增强后的元数据: {documents[0].metadata}")
```

---

## 场景 8: 错误处理与容错

### 健壮的加载函数

```python
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_load_office_document(
    file_path: str,
    fallback_to_text: bool = True
) -> Optional[list[Document]]:
    """安全加载 Office 文档，带错误处理"""
    try:
        # 尝试使用专用加载器
        documents = load_office_document(file_path)
        logger.info(f"成功加载 {file_path}: {len(documents)} 个文档")
        return documents

    except Exception as e:
        logger.error(f"加载 {file_path} 失败: {e}")

        if fallback_to_text:
            try:
                # 回退到通用文本加载器
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                logger.info(f"使用文本加载器成功加载 {file_path}")
                return documents
            except Exception as e2:
                logger.error(f"文本加载器也失败: {e2}")

        return None

# 批量加载，跳过失败的文件
def batch_load_with_error_handling(directory: str) -> list[Document]:
    """批量加载，自动跳过失败的文件"""
    office_files = []
    for ext in [".docx", ".xlsx", ".pptx"]:
        office_files.extend(Path(directory).rglob(f"*{ext}"))

    all_documents = []
    failed_files = []

    for file in office_files:
        docs = safe_load_office_document(str(file))
        if docs:
            all_documents.extend(docs)
        else:
            failed_files.append(str(file))

    print(f"成功: {len(office_files) - len(failed_files)}/{len(office_files)}")
    if failed_files:
        print(f"失败的文件: {failed_files}")

    return all_documents

# 使用
documents = batch_load_with_error_handling("./office_docs")
```

---

## 总结

### Office 文档加载器对比

| 加载器 | 适用格式 | 优点 | 缺点 | 推荐场景 |
|--------|----------|------|------|----------|
| Docx2txtLoader | .docx | 快速、简单 | 只提取纯文本 | 简单文档 |
| UnstructuredFileLoader | .docx/.xlsx/.pptx | 保留结构、功能强大 | 依赖多、速度慢 | 复杂文档 |
| UnstructuredExcelLoader | .xlsx | 保留表格结构 | 需要额外配置 | 表格数据 |
| UnstructuredPowerPointLoader | .pptx | 保留幻灯片结构 | 需要额外配置 | 演示文稿 |

### 最佳实践

1. **选择合适的加载器**:
   - 简单文档 → Docx2txtLoader
   - 复杂文档 → UnstructuredFileLoader
   - 表格数据 → UnstructuredExcelLoader + pandas

2. **批量处理**:
   - 使用 DirectoryLoader 批量加载
   - 使用多线程提升性能
   - 添加错误处理和日志

3. **元数据管理**:
   - 提取文档属性（作者、创建时间等）
   - 保留文件路径和类型
   - 添加自定义元数据

4. **RAG 集成**:
   - 合理分块（chunk_size=1000）
   - 使用混合检索
   - 保留来源信息

---

## 数据来源

- [来源: reference/fetch_1_8_b1b3cc_02.md | LangChain-OpenTutorial GitHub 教程]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: 03_核心概念_2_BaseLoader接口设计.md | BaseLoader 接口设计]
