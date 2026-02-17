# 核心概念5：Office文档处理

> 掌握DOCX、XLSX、PPTX等Office文档的加载和解析

---

## 为什么Office文档处理重要？

Office文档（Word、Excel、PowerPoint）是企业环境中最常见的文档格式：
- **DOCX**：技术文档、报告、合同
- **XLSX**：数据表格、统计报告
- **PPTX**：演示文稿、培训材料

**在RAG系统中，正确处理Office文档直接影响知识库的完整性。**

---

## 1. Office文档格式基础

### 1.1 Office Open XML格式

现代Office文档（.docx, .xlsx, .pptx）基于**Office Open XML**标准：
- 本质是**ZIP压缩包**
- 包含XML文件和媒体资源
- 结构化存储

```python
import zipfile

# 查看DOCX文件结构
with zipfile.ZipFile("document.docx", 'r') as zip_ref:
    print("DOCX文件内容:")
    for file in zip_ref.namelist():
        print(f"  {file}")

# 输出示例:
# word/document.xml        # 主文档内容
# word/styles.xml          # 样式定义
# word/media/image1.png    # 嵌入的图片
# [Content_Types].xml      # 内容类型定义
```

---

## 2. DOCX文档处理

### 2.1 使用python-docx

**python-docx**是处理Word文档的标准库。

```python
"""
python-docx基础使用
适用场景：提取Word文档的文本和结构
"""

from docx import Document
from langchain.schema import Document as LCDocument

def load_docx_with_python_docx(file_path: str) -> list[LCDocument]:
    """使用python-docx加载Word文档"""
    doc = Document(file_path)
    documents = []

    # 提取段落
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # 跳过空段落
            full_text.append(para.text)

    # 创建LangChain Document
    document = LCDocument(
        page_content="\n".join(full_text),
        metadata={
            "source": file_path,
            "format": "docx",
            "paragraph_count": len(doc.paragraphs)
        }
    )
    documents.append(document)

    return documents

# 使用示例
docs = load_docx_with_python_docx("report.docx")
print(f"提取了 {len(docs[0].page_content)} 个字符")
```

### 2.2 提取结构信息

```python
from docx import Document

def extract_docx_structure(file_path: str):
    """提取Word文档的结构信息"""
    doc = Document(file_path)

    print("=== 文档结构 ===\n")

    # 1. 提取标题
    print("标题:")
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            level = para.style.name.replace('Heading ', '')
            indent = "  " * (int(level) - 1)
            print(f"{indent}{para.text}")

    # 2. 提取表格
    print(f"\n表格数量: {len(doc.tables)}")
    for i, table in enumerate(doc.tables):
        print(f"\n表格 {i+1}:")
        print(f"  行数: {len(table.rows)}")
        print(f"  列数: {len(table.columns)}")

        # 显示表格内容（前3行）
        for row_idx, row in enumerate(table.rows[:3]):
            cells = [cell.text for cell in row.cells]
            print(f"  {cells}")

    # 3. 提取图片
    print(f"\n图片数量: {len(doc.inline_shapes)}")

# 使用示例
extract_docx_structure("report.docx")
```

### 2.3 保留格式信息

```python
from docx import Document
from langchain.schema import Document as LCDocument

def load_docx_with_formatting(file_path: str) -> list[LCDocument]:
    """加载Word文档并保留格式信息"""
    doc = Document(file_path)
    documents = []

    for para_idx, para in enumerate(doc.paragraphs):
        if not para.text.strip():
            continue

        # 提取格式信息
        metadata = {
            "source": file_path,
            "format": "docx",
            "paragraph_index": para_idx,
            "style": para.style.name,
            "is_heading": para.style.name.startswith('Heading'),
            "alignment": str(para.alignment) if para.alignment else None
        }

        # 提取字体信息（如果有）
        if para.runs:
            first_run = para.runs[0]
            metadata["font_name"] = first_run.font.name
            metadata["font_size"] = first_run.font.size.pt if first_run.font.size else None
            metadata["is_bold"] = first_run.font.bold
            metadata["is_italic"] = first_run.font.italic

        document = LCDocument(
            page_content=para.text,
            metadata=metadata
        )
        documents.append(document)

    return documents

# 使用示例
docs = load_docx_with_formatting("report.docx")

# 查看标题
headings = [doc for doc in docs if doc.metadata.get('is_heading')]
print("文档标题:")
for heading in headings:
    print(f"  {heading.metadata['style']}: {heading.page_content}")
```

### 2.4 提取表格数据

```python
from docx import Document
import pandas as pd

def extract_tables_from_docx(file_path: str) -> list[pd.DataFrame]:
    """提取Word文档中的所有表格"""
    doc = Document(file_path)
    tables_data = []

    for table_idx, table in enumerate(doc.tables):
        # 提取表格数据
        data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            data.append(row_data)

        # 转换为DataFrame
        if data:
            # 假设第一行是表头
            df = pd.DataFrame(data[1:], columns=data[0])
            df.attrs['table_index'] = table_idx
            tables_data.append(df)

            print(f"\n表格 {table_idx + 1}:")
            print(df.head())

    return tables_data

# 使用示例
tables = extract_tables_from_docx("report_with_tables.docx")
print(f"\n提取了 {len(tables)} 个表格")
```

---

## 3. XLSX文档处理

### 3.1 使用openpyxl

**openpyxl**是处理Excel文档的标准库。

```python
"""
openpyxl基础使用
适用场景：提取Excel表格数据
"""

from openpyxl import load_workbook
from langchain.schema import Document
import pandas as pd

def load_xlsx_with_openpyxl(file_path: str) -> list[Document]:
    """使用openpyxl加载Excel文档"""
    workbook = load_workbook(file_path, data_only=True)
    documents = []

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]

        # 提取所有数据
        data = []
        for row in sheet.iter_rows(values_only=True):
            data.append(row)

        # 转换为文本
        text_lines = []
        for row in data:
            row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
            if row_text.strip():
                text_lines.append(row_text)

        # 创建Document
        document = Document(
            page_content="\n".join(text_lines),
            metadata={
                "source": file_path,
                "format": "xlsx",
                "sheet_name": sheet_name,
                "row_count": sheet.max_row,
                "column_count": sheet.max_column
            }
        )
        documents.append(document)

    return documents

# 使用示例
docs = load_xlsx_with_openpyxl("data.xlsx")
print(f"提取了 {len(docs)} 个工作表")
for doc in docs:
    print(f"  {doc.metadata['sheet_name']}: {doc.metadata['row_count']} 行")
```

### 3.2 使用pandas（推荐）

```python
import pandas as pd
from langchain.schema import Document

def load_xlsx_with_pandas(file_path: str) -> list[Document]:
    """使用pandas加载Excel文档（推荐）"""
    # 读取所有工作表
    excel_file = pd.ExcelFile(file_path)
    documents = []

    for sheet_name in excel_file.sheet_names:
        # 读取工作表
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 转换为文本（保留表格结构）
        text = df.to_string(index=False)

        # 创建Document
        document = Document(
            page_content=text,
            metadata={
                "source": file_path,
                "format": "xlsx",
                "sheet_name": sheet_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns)
            }
        )
        documents.append(document)

    return documents

# 使用示例
docs = load_xlsx_with_pandas("sales_data.xlsx")
print(f"提取了 {len(docs)} 个工作表")
```

### 3.3 智能表格解析

```python
import pandas as pd
from langchain.schema import Document

def smart_load_xlsx(file_path: str) -> list[Document]:
    """智能加载Excel：自动识别数据区域"""
    excel_file = pd.ExcelFile(file_path)
    documents = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 1. 清理空行和空列
        df = df.dropna(how='all')  # 删除全空行
        df = df.dropna(axis=1, how='all')  # 删除全空列

        # 2. 生成摘要
        summary = f"工作表: {sheet_name}\n"
        summary += f"数据维度: {len(df)} 行 × {len(df.columns)} 列\n"
        summary += f"列名: {', '.join(df.columns)}\n\n"

        # 3. 生成数据描述
        if len(df) > 0:
            summary += "数据预览:\n"
            summary += df.head(5).to_string(index=False)

            # 数值列的统计信息
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary += "\n\n数值列统计:\n"
                summary += df[numeric_cols].describe().to_string()

        # 创建Document
        document = Document(
            page_content=summary,
            metadata={
                "source": file_path,
                "format": "xlsx",
                "sheet_name": sheet_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "numeric_columns": list(numeric_cols) if len(numeric_cols) > 0 else []
            }
        )
        documents.append(document)

    return documents

# 使用示例
docs = smart_load_xlsx("sales_data.xlsx")
print(docs[0].page_content)
```

---

## 4. PPTX文档处理

### 4.1 使用python-pptx

```python
"""
python-pptx基础使用
适用场景：提取PowerPoint演示文稿内容
"""

from pptx import Presentation
from langchain.schema import Document

def load_pptx_with_python_pptx(file_path: str) -> list[Document]:
    """使用python-pptx加载PowerPoint文档"""
    prs = Presentation(file_path)
    documents = []

    for slide_idx, slide in enumerate(prs.slides):
        # 提取幻灯片文本
        text_content = []

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                if shape.text.strip():
                    text_content.append(shape.text)

        # 创建Document
        document = Document(
            page_content="\n".join(text_content),
            metadata={
                "source": file_path,
                "format": "pptx",
                "slide_number": slide_idx + 1,
                "total_slides": len(prs.slides)
            }
        )
        documents.append(document)

    return documents

# 使用示例
docs = load_pptx_with_python_pptx("presentation.pptx")
print(f"提取了 {len(docs)} 张幻灯片")
for i, doc in enumerate(docs[:3], 1):
    print(f"\n幻灯片 {i}:")
    print(doc.page_content[:200])
```

### 4.2 提取结构信息

```python
from pptx import Presentation

def extract_pptx_structure(file_path: str):
    """提取PowerPoint的结构信息"""
    prs = Presentation(file_path)

    print(f"=== 演示文稿结构 ===")
    print(f"总幻灯片数: {len(prs.slides)}\n")

    for slide_idx, slide in enumerate(prs.slides):
        print(f"幻灯片 {slide_idx + 1}:")

        # 提取标题
        if slide.shapes.title:
            print(f"  标题: {slide.shapes.title.text}")

        # 统计形状类型
        shape_types = {}
        for shape in slide.shapes:
            shape_type = shape.shape_type
            shape_types[shape_type] = shape_types.get(shape_type, 0) + 1

        print(f"  形状数量: {len(slide.shapes)}")

        # 提取表格
        tables = [shape for shape in slide.shapes if shape.has_table]
        if tables:
            print(f"  表格数量: {len(tables)}")

        # 提取图片
        pictures = [shape for shape in slide.shapes if shape.shape_type == 13]  # PICTURE
        if pictures:
            print(f"  图片数量: {len(pictures)}")

        print()

# 使用示例
extract_pptx_structure("presentation.pptx")
```

---

## 5. 统一Office文档加载器

### 5.1 通用Office加载器

```python
from typing import List
from langchain.schema import Document
import os

class UnifiedOfficeLoader:
    """统一的Office文档加载器"""

    def load(self, file_path: str) -> List[Document]:
        """根据文件扩展名自动选择加载器"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.docx':
            return self._load_docx(file_path)
        elif ext == '.xlsx':
            return self._load_xlsx(file_path)
        elif ext == '.pptx':
            return self._load_pptx(file_path)
        else:
            raise ValueError(f"不支持的Office格式: {ext}")

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载Word文档"""
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        return [Document(
            page_content=full_text,
            metadata={
                "source": file_path,
                "format": "docx",
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables)
            }
        )]

    def _load_xlsx(self, file_path: str) -> List[Document]:
        """加载Excel文档"""
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        documents = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text = df.to_string(index=False)

            documents.append(Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "format": "xlsx",
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            ))

        return documents

    def _load_pptx(self, file_path: str) -> List[Document]:
        """加载PowerPoint文档"""
        from pptx import Presentation

        prs = Presentation(file_path)
        documents = []

        for slide_idx, slide in enumerate(prs.slides):
            text_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content.append(shape.text)

            documents.append(Document(
                page_content="\n".join(text_content),
                metadata={
                    "source": file_path,
                    "format": "pptx",
                    "slide_number": slide_idx + 1,
                    "total_slides": len(prs.slides)
                }
            ))

        return documents

# 使用示例
loader = UnifiedOfficeLoader()

# 自动识别并加载
docs = loader.load("report.docx")
print(f"Word文档: {len(docs)} 个文档")

docs = loader.load("data.xlsx")
print(f"Excel文档: {len(docs)} 个工作表")

docs = loader.load("presentation.pptx")
print(f"PowerPoint文档: {len(docs)} 张幻灯片")
```

---

## 6. LangChain集成

### 6.1 使用LangChain的Office加载器

```python
from langchain.document_loaders import (
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)

# ===== Word文档 =====
docx_loader = Docx2txtLoader("report.docx")
docx_docs = docx_loader.load()

# ===== Excel文档 =====
xlsx_loader = UnstructuredExcelLoader("data.xlsx")
xlsx_docs = xlsx_loader.load()

# ===== PowerPoint文档 =====
pptx_loader = UnstructuredPowerPointLoader("presentation.pptx")
pptx_docs = pptx_loader.load()

print(f"Word: {len(docx_docs)} 文档")
print(f"Excel: {len(xlsx_docs)} 文档")
print(f"PowerPoint: {len(pptx_docs)} 文档")
```

---

## 7. 在RAG中的应用

### 7.1 构建Office文档知识库

```python
from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 1. 批量加载Office文档
loader = DirectoryLoader(
    "documents/",
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)
documents = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. 向量化并存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./office_kb"
)

print(f"Office文档知识库构建完成，包含 {len(chunks)} 个文本块")
```

---

## 总结

**Office文档处理的核心要点：**

1. **DOCX**: 使用python-docx，保留结构和格式
2. **XLSX**: 使用pandas，智能处理表格数据
3. **PPTX**: 使用python-pptx，按幻灯片组织内容
4. **统一接口**: 通过统一加载器处理所有Office格式
5. **元数据保留**: 记录文档结构、表格、图片等信息

**在RAG中的最佳实践:**
- 保留文档结构信息（标题、章节）
- 表格数据转换为可检索的文本
- 为每个工作表/幻灯片创建独立Document
- 添加丰富的元数据用于过滤

---

## 参考来源

> **参考来源：**
> - [python-docx Documentation](https://python-docx.readthedocs.io/) (2025)
> - [openpyxl Documentation](https://openpyxl.readthedocs.io/) (2025)
> - [python-pptx Documentation](https://python-pptx.readthedocs.io/) (2025)
> - [pandas Documentation](https://pandas.pydata.org/docs/) (2025)

---

**版本：** v1.0
**最后更新：** 2026-02-15
**下一步：** 阅读 [03_核心概念_06_HTML与Markdown解析.md](./03_核心概念_06_HTML与Markdown解析.md)
