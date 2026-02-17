# 实战代码2：Office文档批量处理

> 批量处理DOCX/XLSX文档并提取元数据的完整示例

---

## 场景描述

演示如何批量加载和处理Office文档（Word、Excel、PowerPoint），提取内容和元数据，并构建知识库。

**适用场景：**
- 企业文档管理
- 知识库构建
- 文档内容检索

---

## 依赖库

```bash
# 安装依赖
uv add python-docx openpyxl python-pptx pandas langchain langchain-openai chromadb python-dotenv
```

---

## 完整代码

```python
"""
Office文档批量处理示例
演示：批量加载DOCX/XLSX/PPTX文档并提取元数据

依赖库：
- python-docx: Word文档处理
- openpyxl: Excel文档处理
- python-pptx: PowerPoint文档处理
- pandas: 数据处理
- langchain: 文档处理框架
- chromadb: 向量数据库

参考来源：
- python-docx Documentation (2025): https://python-docx.readthedocs.io/
- openpyxl Documentation (2025): https://openpyxl.readthedocs.io/
- pandas Documentation (2025): https://pandas.pydata.org/docs/
"""

import os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()

# ===== 1. Word文档批量加载 =====
print("=== 1. Word文档批量加载 ===")

from docx import Document as DocxDocument

def load_docx_batch(file_paths: List[str]) -> List[Document]:
    """批量加载Word文档"""
    all_documents = []

    for file_path in file_paths:
        try:
            print(f"\n加载: {file_path}")

            # 打开Word文档
            doc = DocxDocument(file_path)

            # 提取段落文本
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n".join(paragraphs)

            # 提取元数据
            core_props = doc.core_properties
            metadata = {
                "source": file_path,
                "format": "docx",
                "title": core_props.title or os.path.basename(file_path),
                "author": core_props.author or "",
                "created": core_props.created.isoformat() if core_props.created else "",
                "modified": core_props.modified.isoformat() if core_props.modified else "",
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "file_size": os.path.getsize(file_path)
            }

            # 创建Document
            document = Document(page_content=full_text, metadata=metadata)
            all_documents.append(document)

            print(f"  ✅ 成功: {len(paragraphs)} 段落, {len(doc.tables)} 表格")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    return all_documents

# 测试Word批量加载
# docx_files = ["doc1.docx", "doc2.docx", "doc3.docx"]
# docs = load_docx_batch(docx_files)
# print(f"\n总共加载 {len(docs)} 个Word文档")

# ===== 2. Excel文档批量加载 =====
print("\n=== 2. Excel文档批量加载 ===")

import pandas as pd

def load_xlsx_batch(file_paths: List[str]) -> List[Document]:
    """批量加载Excel文档"""
    all_documents = []

    for file_path in file_paths:
        try:
            print(f"\n加载: {file_path}")

            # 读取所有工作表
            excel_file = pd.ExcelFile(file_path)

            for sheet_name in excel_file.sheet_names:
                # 读取工作表
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # 清理空行和空列
                df = df.dropna(how='all').dropna(axis=1, how='all')

                if len(df) == 0:
                    continue

                # 生成文本表示
                text_lines = []
                text_lines.append(f"工作表: {sheet_name}")
                text_lines.append(f"列名: {', '.join(df.columns)}")
                text_lines.append("\n数据预览:")
                text_lines.append(df.head(10).to_string(index=False))

                # 数值列统计
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    text_lines.append("\n数值统计:")
                    text_lines.append(df[numeric_cols].describe().to_string())

                full_text = "\n".join(text_lines)

                # 元数据
                metadata = {
                    "source": file_path,
                    "format": "xlsx",
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns),
                    "numeric_columns": list(numeric_cols),
                    "file_size": os.path.getsize(file_path)
                }

                document = Document(page_content=full_text, metadata=metadata)
                all_documents.append(document)

                print(f"  ✅ 工作表 '{sheet_name}': {len(df)} 行 × {len(df.columns)} 列")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    return all_documents

# 测试Excel批量加载
# xlsx_files = ["data1.xlsx", "data2.xlsx"]
# docs = load_xlsx_batch(xlsx_files)
# print(f"\n总共加载 {len(docs)} 个Excel工作表")

# ===== 3. PowerPoint文档批量加载 =====
print("\n=== 3. PowerPoint文档批量加载 ===")

from pptx import Presentation

def load_pptx_batch(file_paths: List[str]) -> List[Document]:
    """批量加载PowerPoint文档"""
    all_documents = []

    for file_path in file_paths:
        try:
            print(f"\n加载: {file_path}")

            prs = Presentation(file_path)

            for slide_idx, slide in enumerate(prs.slides):
                # 提取幻灯片文本
                text_content = []

                # 提取标题
                if slide.shapes.title:
                    text_content.append(f"标题: {slide.shapes.title.text}")

                # 提取其他文本
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        if shape != slide.shapes.title:
                            text_content.append(shape.text)

                full_text = "\n".join(text_content)

                # 元数据
                metadata = {
                    "source": file_path,
                    "format": "pptx",
                    "slide_number": slide_idx + 1,
                    "total_slides": len(prs.slides),
                    "shape_count": len(slide.shapes),
                    "file_size": os.path.getsize(file_path)
                }

                document = Document(page_content=full_text, metadata=metadata)
                all_documents.append(document)

            print(f"  ✅ 成功: {len(prs.slides)} 张幻灯片")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    return all_documents

# 测试PowerPoint批量加载
# pptx_files = ["presentation1.pptx", "presentation2.pptx"]
# docs = load_pptx_batch(pptx_files)
# print(f"\n总共加载 {len(docs)} 张幻灯片")

# ===== 4. 统一Office文档加载器 =====
print("\n=== 4. 统一Office文档加载器 ===")

class UnifiedOfficeLoader:
    """统一的Office文档批量加载器"""

    def __init__(self):
        self.supported_formats = {
            '.docx': self._load_docx,
            '.xlsx': self._load_xlsx,
            '.pptx': self._load_pptx
        }

    def load_directory(self, directory: str) -> List[Document]:
        """加载目录下所有Office文档"""
        all_documents = []
        file_count = {"docx": 0, "xlsx": 0, "pptx": 0, "failed": 0}

        print(f"扫描目录: {directory}")

        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()

                if ext in self.supported_formats:
                    file_path = os.path.join(root, file)

                    try:
                        docs = self.supported_formats[ext](file_path)
                        all_documents.extend(docs)
                        file_count[ext[1:]] += 1
                        print(f"✅ {file}: {len(docs)} 个文档")
                    except Exception as e:
                        file_count["failed"] += 1
                        print(f"❌ {file}: {e}")

        print(f"\n加载统计:")
        print(f"  Word文档: {file_count['docx']}")
        print(f"  Excel文档: {file_count['xlsx']}")
        print(f"  PowerPoint文档: {file_count['pptx']}")
        print(f"  失败: {file_count['failed']}")
        print(f"  总文档数: {len(all_documents)}")

        return all_documents

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载Word文档"""
        return load_docx_batch([file_path])

    def _load_xlsx(self, file_path: str) -> List[Document]:
        """加载Excel文档"""
        return load_xlsx_batch([file_path])

    def _load_pptx(self, file_path: str) -> List[Document]:
        """加载PowerPoint文档"""
        return load_pptx_batch([file_path])

# 测试统一加载器
# loader = UnifiedOfficeLoader()
# docs = loader.load_directory("./documents")

# ===== 5. 元数据增强 =====
print("\n=== 5. 元数据增强 ===")

def enhance_metadata(documents: List[Document]) -> List[Document]:
    """增强文档元数据"""
    for doc in documents:
        # 添加加载时间
        doc.metadata["loaded_at"] = datetime.now().isoformat()

        # 添加内容统计
        doc.metadata["content_length"] = len(doc.page_content)
        doc.metadata["word_count"] = len(doc.page_content.split())

        # 添加语言检测（简单版本）
        chinese_chars = sum(1 for c in doc.page_content if '\u4e00' <= c <= '\u9fff')
        total_chars = len(doc.page_content)
        doc.metadata["language"] = "zh" if chinese_chars / total_chars > 0.1 else "en"

        # 添加文档类型标签
        if doc.metadata["format"] == "docx":
            doc.metadata["doc_type"] = "文本文档"
        elif doc.metadata["format"] == "xlsx":
            doc.metadata["doc_type"] = "数据表格"
        elif doc.metadata["format"] == "pptx":
            doc.metadata["doc_type"] = "演示文稿"

    return documents

# ===== 6. RAG应用：构建Office文档知识库 =====
print("\n=== 6. RAG应用：构建Office文档知识库 ===")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def build_office_knowledge_base(directory: str, output_dir: str = "./office_kb"):
    """从Office文档构建知识库"""

    print(f"=== 构建Office文档知识库 ===")
    print(f"源目录: {directory}")
    print(f"输出目录: {output_dir}\n")

    # 1. 批量加载Office文档
    print("步骤1: 批量加载文档")
    loader = UnifiedOfficeLoader()
    documents = loader.load_directory(directory)

    if not documents:
        print("❌ 没有加载到任何文档")
        return None

    # 2. 增强元数据
    print("\n步骤2: 增强元数据")
    documents = enhance_metadata(documents)
    print(f"✅ 元数据增强完成")

    # 3. 文本分块
    print("\n步骤3: 文本分块")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 分块完成: {len(chunks)} 个文本块")

    # 4. 向量化并存储
    print("\n步骤4: 向量化并存储")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=output_dir
        )
        print(f"✅ 知识库构建成功")

        # 5. 测试检索
        print("\n步骤5: 测试检索")
        test_queries = [
            "文档的主要内容是什么？",
            "有哪些数据统计？",
            "演示文稿的主题是什么？"
        ]

        for query in test_queries:
            results = vectorstore.similarity_search(query, k=2)
            print(f"\n查询: {query}")
            print(f"找到 {len(results)} 个相关文档:")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. {doc.metadata['source']} ({doc.metadata['doc_type']})")
                print(f"     内容: {doc.page_content[:100]}...")

        return vectorstore

    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return None

# 测试构建知识库
# vectorstore = build_office_knowledge_base("./documents")

# ===== 7. 文档统计分析 =====
print("\n=== 7. 文档统计分析 ===")

def analyze_documents(documents: List[Document]) -> Dict:
    """分析文档统计信息"""
    stats = {
        "total_documents": len(documents),
        "by_format": {},
        "by_language": {},
        "total_words": 0,
        "total_size": 0,
        "largest_doc": None,
        "smallest_doc": None
    }

    for doc in documents:
        # 按格式统计
        fmt = doc.metadata.get("format", "unknown")
        stats["by_format"][fmt] = stats["by_format"].get(fmt, 0) + 1

        # 按语言统计
        lang = doc.metadata.get("language", "unknown")
        stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

        # 总词数
        stats["total_words"] += doc.metadata.get("word_count", 0)

        # 总大小
        stats["total_size"] += doc.metadata.get("file_size", 0)

        # 最大/最小文档
        word_count = doc.metadata.get("word_count", 0)
        if stats["largest_doc"] is None or word_count > stats["largest_doc"]["word_count"]:
            stats["largest_doc"] = {
                "source": doc.metadata["source"],
                "word_count": word_count
            }
        if stats["smallest_doc"] is None or word_count < stats["smallest_doc"]["word_count"]:
            stats["smallest_doc"] = {
                "source": doc.metadata["source"],
                "word_count": word_count
            }

    # 打印统计报告
    print("=== 文档统计报告 ===")
    print(f"\n总文档数: {stats['total_documents']}")

    print(f"\n按格式分布:")
    for fmt, count in stats["by_format"].items():
        print(f"  {fmt}: {count} ({count/stats['total_documents']*100:.1f}%)")

    print(f"\n按语言分布:")
    for lang, count in stats["by_language"].items():
        print(f"  {lang}: {count} ({count/stats['total_documents']*100:.1f}%)")

    print(f"\n总词数: {stats['total_words']:,}")
    print(f"总大小: {stats['total_size']/1024/1024:.2f} MB")

    if stats["largest_doc"]:
        print(f"\n最大文档: {stats['largest_doc']['source']}")
        print(f"  词数: {stats['largest_doc']['word_count']:,}")

    if stats["smallest_doc"]:
        print(f"\n最小文档: {stats['smallest_doc']['source']}")
        print(f"  词数: {stats['smallest_doc']['word_count']:,}")

    return stats

# 测试文档分析
# stats = analyze_documents(documents)

# ===== 8. 导出功能 =====
print("\n=== 8. 导出功能 ===")

import json

def export_metadata(documents: List[Document], output_file: str):
    """导出文档元数据到JSON"""
    metadata_list = []

    for doc in documents:
        metadata_list.append({
            "source": doc.metadata.get("source"),
            "format": doc.metadata.get("format"),
            "title": doc.metadata.get("title"),
            "author": doc.metadata.get("author"),
            "word_count": doc.metadata.get("word_count"),
            "file_size": doc.metadata.get("file_size"),
            "language": doc.metadata.get("language"),
            "loaded_at": doc.metadata.get("loaded_at")
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"✅ 元数据已导出到: {output_file}")

# 测试导出
# export_metadata(documents, "metadata.json")

print("\n=== 示例代码执行完成 ===")
print("\n使用说明:")
print("1. 准备Office文档目录")
print("2. 取消注释相应的测试代码")
print("3. 运行脚本查看效果")
print("\n功能:")
print("- 批量加载Word/Excel/PowerPoint文档")
print("- 自动提取元数据")
print("- 构建可检索的知识库")
print("- 生成统计分析报告")
```

---

## 运行输出示例

```
=== 1. Word文档批量加载 ===

加载: reports/2025_q1.docx
  ✅ 成功: 45 段落, 3 表格

加载: reports/2025_q2.docx
  ✅ 成功: 52 段落, 2 表格

总共加载 2 个Word文档

=== 2. Excel文档批量加载 ===

加载: data/sales_2025.xlsx
  ✅ 工作表 'Q1': 120 行 × 8 列
  ✅ 工作表 'Q2': 135 行 × 8 列

总共加载 2 个Excel工作表

=== 文档统计报告 ===

总文档数: 10

按格式分布:
  docx: 4 (40.0%)
  xlsx: 3 (30.0%)
  pptx: 3 (30.0%)

按语言分布:
  zh: 8 (80.0%)
  en: 2 (20.0%)

总词数: 25,430
总大小: 5.23 MB
```

---

## 关键要点

### 1. 元数据的重要性

```python
# 完整的元数据支持高级检索
metadata = {
    "source": "report.docx",
    "format": "docx",
    "title": "2025年度报告",
    "author": "张三",
    "created": "2025-01-15",
    "department": "技术部",  # 自定义元数据
    "category": "年度报告"
}

# 按元数据过滤检索
results = vectorstore.similarity_search(
    query,
    filter={"department": "技术部", "format": "docx"}
)
```

### 2. Excel数据处理策略

- **小表格**：直接转换为文本
- **大表格**：只保留摘要和统计信息
- **数值列**：提取统计信息（均值、最大值、最小值）

### 3. 批量处理最佳实践

- 使用try-except处理单个文件失败
- 记录失败文件便于后续处理
- 提供进度反馈
- 生成处理报告

---

## 常见问题

### Q1: 如何处理大型Excel文件？

```python
# 使用chunksize分批读取
for chunk in pd.read_excel("large.xlsx", chunksize=1000):
    process_chunk(chunk)
```

### Q2: 如何提取Word文档中的表格？

```python
from docx import Document

doc = Document("file.docx")
for table in doc.tables:
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        print(cells)
```

### Q3: 如何处理密码保护的Office文档？

需要使用专门的库如`msoffcrypto-tool`先解密。

---

## 扩展阅读

- [python-docx Documentation](https://python-docx.readthedocs.io/) (2025)
- [openpyxl Documentation](https://openpyxl.readthedocs.io/) (2025)
- [pandas Excel I/O](https://pandas.pydata.org/docs/user_guide/io.html#excel-files) (2025)

---

**版本：** v1.0
**最后更新：** 2026-02-15
**下一步：** 阅读 [07_实战代码_03_HTML内容提取.md](./07_实战代码_03_HTML内容提取.md)
