# 实战代码 - 场景3: CSV_JSON结构化数据加载

> 使用 CSVLoader 和 JSONLoader 加载结构化数据的完整实战指南

---

## 核心目标

掌握 LangChain 结构化数据加载器的实战使用，包括 CSVLoader 和 JSONLoader 的使用、数据转换和 RAG 集成。

---

## 环境准备

### 1. 导入必要的库

```python
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import json
import csv

# 加载环境变量
load_dotenv()
```

---

## 场景 1: CSVLoader 基础使用

### 示例 1.1: 加载 CSV 文件

```python
from langchain_community.document_loaders import CSVLoader

# 加载 CSV 文件
loader = CSVLoader(file_path="data.csv", encoding="utf-8")
documents = loader.load()

# 查看结果
print(f"加载了 {len(documents)} 条记录")
print(f"第一条记录: {documents[0].page_content}")
print(f"元数据: {documents[0].metadata}")
```

**输出示例**:
```
加载了 100 条记录
第一条记录: name: John Doe
age: 30
city: New York
元数据: {'source': 'data.csv', 'row': 0}
```

**关键点**:
- CSVLoader 将每一行作为一个独立的 Document
- 元数据自动包含文件路径和行号
- 默认将所有列合并为文本

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 示例 1.2: 指定列作为内容

```python
from langchain_community.document_loaders import CSVLoader

# 只使用特定列作为内容
loader = CSVLoader(
    file_path="data.csv",
    source_column="name",  # 使用 name 列作为 source
    encoding="utf-8"
)
documents = loader.load()

print(f"第一条记录: {documents[0].page_content}")
print(f"元数据: {documents[0].metadata}")
```

**输出示例**:
```
第一条记录: name: John Doe
age: 30
city: New York
元数据: {'source': 'John Doe', 'row': 0}
```

### 示例 1.3: 懒加载模式

```python
from langchain_community.document_loaders import CSVLoader

# 懒加载 - 适合大 CSV 文件
loader = CSVLoader(file_path="large_data.csv", encoding="utf-8")

# 流式处理
for doc in loader.lazy_load():
    print(f"处理第 {doc.metadata['row']} 行")
    # 逐行处理,节省内存
```

**为什么使用懒加载?**
- 大 CSV 文件不会一次性加载到内存
- 可以提前终止处理
- 适合流式处理场景

[来源: reference/context7_langchain_01.md | LangChain 官方文档 - 懒加载最佳实践]

---

## 场景 2: JSONLoader 基础使用

### 示例 2.1: 加载 JSON 文件

```python
from langchain_community.document_loaders import JSONLoader

# 加载 JSON 文件
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".",  # 使用 jq 语法提取数据
    text_content=False
)
documents = loader.load()

# 查看结果
print(f"加载了 {len(documents)} 条记录")
print(f"第一条记录: {documents[0].page_content}")
```

**JSON 文件示例** (`data.json`):
```json
[
  {
    "name": "John Doe",
    "age": 30,
    "city": "New York"
  },
  {
    "name": "Jane Smith",
    "age": 25,
    "city": "Los Angeles"
  }
]
```

### 示例 2.2: 使用 jq 提取特定字段

```python
from langchain_community.document_loaders import JSONLoader

# 只提取 name 字段
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[] | .name",  # 提取每个对象的 name 字段
    text_content=False
)
documents = loader.load()

print(f"第一条记录: {documents[0].page_content}")
```

**输出示例**:
```
第一条记录: John Doe
```

### 示例 2.3: 嵌套 JSON 处理

```python
from langchain_community.document_loaders import JSONLoader

# 处理嵌套 JSON
loader = JSONLoader(
    file_path="nested_data.json",
    jq_schema=".users[] | {name: .name, email: .contact.email}",
    text_content=False
)
documents = loader.load()
```

**嵌套 JSON 示例** (`nested_data.json`):
```json
{
  "users": [
    {
      "name": "John Doe",
      "contact": {
        "email": "john@example.com",
        "phone": "123-456-7890"
      }
    }
  ]
}
```

---

## 场景 3: CSV 与 JSON 对比

### 示例 3.1: 相同数据的不同格式

```python
from langchain_community.document_loaders import CSVLoader, JSONLoader

# CSV 格式
csv_loader = CSVLoader(file_path="data.csv")
csv_docs = csv_loader.load()

# JSON 格式
json_loader = JSONLoader(file_path="data.json", jq_schema=".")
json_docs = json_loader.load()

# 对比
print(f"CSV 文档数: {len(csv_docs)}")
print(f"JSON 文档数: {len(json_docs)}")
```

**对比总结**:

| 特性 | CSVLoader | JSONLoader |
|------|-----------|------------|
| 数据结构 | 表格 | 嵌套对象 |
| 复杂度 | 简单 | 复杂 |
| 查询能力 | 有限 | 强大(jq) |
| 适用场景 | 简单表格数据 | 复杂结构数据 |
| 性能 | 快 | 较慢 |

---

## 场景 4: 数据转换与处理

### 示例 4.1: CSV 数据清洗

```python
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

def clean_csv_data(file_path: str):
    """加载并清洗 CSV 数据"""
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    # 清洗数据
    cleaned_docs = []
    for doc in documents:
        # 移除空白
        content = doc.page_content.strip()

        # 过滤空记录
        if not content:
            continue

        # 创建新文档
        cleaned_doc = Document(
            page_content=content,
            metadata={
                **doc.metadata,
                "cleaned": True
            }
        )
        cleaned_docs.append(cleaned_doc)

    return cleaned_docs

# 使用示例
documents = clean_csv_data("data.csv")
print(f"清洗后文档数: {len(documents)}")
```

### 示例 4.2: JSON 数据转换

```python
from langchain_community.document_loaders import JSONLoader
import json

def transform_json_data(file_path: str):
    """加载并转换 JSON 数据"""
    with open(file_path, "r") as f:
        data = json.load(f)

    # 转换为 Document
    documents = []
    for item in data:
        # 自定义格式化
        content = f"Name: {item['name']}\nAge: {item['age']}\nCity: {item['city']}"

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "name": item["name"]
            }
        )
        documents.append(doc)

    return documents

# 使用示例
documents = transform_json_data("data.json")
print(f"转换后文档数: {len(documents)}")
```

---

## 场景 5: 完整 RAG 管道

### 示例 5.1: CSV 到 RAG 系统

```python
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 加载 CSV
print("步骤 1: 加载 CSV...")
loader = CSVLoader(file_path="products.csv", encoding="utf-8")
documents = loader.load()
print(f"✓ 加载了 {len(documents)} 条记录")

# 2. 分割文档（可选，CSV 通常不需要分割）
print("\n步骤 2: 准备文档...")
# CSV 每行已经是独立的文档，通常不需要分割
chunks = documents
print(f"✓ 准备了 {len(chunks)} 个文档块")

# 3. 向量化
print("\n步骤 3: 向量化...")
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"✓ 向量化完成")

# 4. 构建 QA 链
print("\n步骤 4: 构建 QA 链...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    ),
    return_source_documents=True
)
print(f"✓ QA 链构建完成")

# 5. 查询
print("\n步骤 5: 查询...")
query = "What products are available?"
result = qa_chain.invoke({"query": query})

print(f"\n问题: {query}")
print(f"回答: {result['result']}")
print(f"\n来源记录:")
for i, doc in enumerate(result['source_documents']):
    print(f"  {i+1}. 行 {doc.metadata['row']}: {doc.page_content[:100]}...")
```

---

## 场景 6: 错误处理

### 示例 6.1: CSV 编码错误处理

```python
from langchain_community.document_loaders import CSVLoader
import chardet

def safe_load_csv(file_path: str):
    """安全加载 CSV 文件"""
    try:
        # 尝试 UTF-8
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()
        print(f"✓ 使用 UTF-8 编码成功加载")
        return documents
    except UnicodeDecodeError:
        print("✗ UTF-8 编码失败,尝试自动检测...")

        # 检测编码
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        print(f"检测到编码: {encoding}")

        # 使用检测到的编码
        loader = CSVLoader(file_path=file_path, encoding=encoding)
        documents = loader.load()
        print(f"✓ 使用 {encoding} 编码成功加载")
        return documents

# 使用示例
documents = safe_load_csv("data.csv")
```

### 示例 6.2: JSON 格式错误处理

```python
from langchain_community.document_loaders import JSONLoader
import json

def safe_load_json(file_path: str):
    """安全加载 JSON 文件"""
    try:
        # 验证 JSON 格式
        with open(file_path, "r") as f:
            data = json.load(f)

        # 加载文档
        loader = JSONLoader(file_path=file_path, jq_schema=".")
        documents = loader.load()
        print(f"✓ 成功加载: {file_path}")
        return documents
    except json.JSONDecodeError as e:
        print(f"✗ JSON 格式错误: {e}")
        return []
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return []

# 使用示例
documents = safe_load_json("data.json")
```

---

## 场景 7: 性能优化

### 示例 7.1: 批量加载 CSV

```python
from langchain_community.document_loaders import CSVLoader
from pathlib import Path
from typing import List
from langchain_core.documents import Document

def batch_load_csvs(directory: str) -> List[Document]:
    """批量加载目录中的所有 CSV 文件"""
    all_documents = []
    failed_files = []

    # 获取所有 CSV 文件
    csv_files = list(Path(directory).glob("*.csv"))
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    for file_path in csv_files:
        try:
            loader = CSVLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            all_documents.extend(documents)
            print(f"✓ {file_path.name}: {len(documents)} 条记录")
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            print(f"✗ {file_path.name}: {e}")

    # 总结
    print(f"\n总结:")
    print(f"  成功: {len(all_documents)} 条记录")
    print(f"  失败: {len(failed_files)} 个文件")

    return all_documents

# 使用示例
documents = batch_load_csvs("./csv_data")
```

### 示例 7.2: 流式处理大 CSV

```python
from langchain_community.document_loaders import CSVLoader

def process_large_csv(file_path: str):
    """使用懒加载处理大 CSV"""
    loader = CSVLoader(file_path=file_path)

    # 懒加载 + 流式处理
    for doc in loader.lazy_load():
        # 逐行处理
        process_record(doc)

def process_record(doc):
    """处理单条记录"""
    # 向量化、存储等操作
    pass

# 使用示例
process_large_csv("large_data.csv")
```

---

## 场景 8: 实战技巧

### 技巧 8.1: CSV 列映射

```python
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

def load_csv_with_column_mapping(file_path: str, column_mapping: dict):
    """加载 CSV 并映射列名"""
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    # 重新格式化内容
    mapped_docs = []
    for doc in documents:
        # 解析原始内容
        lines = doc.page_content.split("\n")
        data = {}
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                data[key] = value

        # 应用映射
        mapped_content = "\n".join([
            f"{new_key}: {data.get(old_key, '')}"
            for old_key, new_key in column_mapping.items()
        ])

        mapped_doc = Document(
            page_content=mapped_content,
            metadata=doc.metadata
        )
        mapped_docs.append(mapped_doc)

    return mapped_docs

# 使用示例
column_mapping = {
    "name": "产品名称",
    "price": "价格",
    "category": "类别"
}
documents = load_csv_with_column_mapping("products.csv", column_mapping)
```

### 技巧 8.2: JSON 数据扁平化

```python
from langchain_community.document_loaders import JSONLoader
import json

def flatten_json(nested_json: dict, parent_key: str = "", sep: str = "_"):
    """扁平化嵌套 JSON"""
    items = []
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_and_flatten_json(file_path: str):
    """加载并扁平化 JSON"""
    with open(file_path, "r") as f:
        data = json.load(f)

    # 扁平化
    if isinstance(data, list):
        flattened_data = [flatten_json(item) for item in data]
    else:
        flattened_data = [flatten_json(data)]

    # 转换为 Document
    documents = []
    for item in flattened_data:
        content = "\n".join([f"{k}: {v}" for k, v in item.items()])
        doc = Document(page_content=content, metadata={"source": file_path})
        documents.append(doc)

    return documents

# 使用示例
documents = load_and_flatten_json("nested_data.json")
```

---

## 总结

### 核心要点

1. **CSVLoader**: 简单、快速,适合表格数据
2. **JSONLoader**: 强大、灵活,适合复杂结构
3. **懒加载**: 大文件使用 `lazy_load()` 节省内存
4. **数据转换**: 根据需求自定义格式化
5. **RAG 集成**: 完整的加载 → 向量化 → 检索 → 生成流程
6. **错误处理**: 编码检测、格式验证
7. **性能优化**: 批量加载、流式处理

### 最佳实践

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 简单表格 | CSVLoader | 快速简单 |
| 复杂结构 | JSONLoader | 灵活强大 |
| 大文件 | lazy_load() | 节省内存 |
| 批量加载 | 错误处理 + 日志 | 提高健壮性 |
| RAG 应用 | 直接向量化 | 无需分割 |

### 常见问题

**Q1: CSV 和 JSON 如何选择?**
- 简单表格数据 → CSV
- 复杂嵌套结构 → JSON

**Q2: 如何处理大文件?**
- 使用 `lazy_load()` 流式加载
- 逐行处理,避免内存溢出

**Q3: 如何提升加载性能?**
- 使用正确的编码
- 批量处理时使用并发
- 对于 CSV,避免不必要的分割

---

## 下一步

理解了 CSV_JSON 结构化数据加载后,建议:

1. **07_实战代码_场景4_Office文档加载.md** - 学习 Office 文档加载
2. **07_实战代码_场景5_HTML_Markdown加载.md** - 学习 HTML/Markdown 加载
3. **07_实战代码_场景6_批量文档加载.md** - 学习批量加载策略

---

**数据来源**:
- [来源: reference/context7_langchain_01.md | LangChain 官方文档 - CSVLoader 示例]
- [来源: reference/fetch_1_8_b1b3cc_02.md | LangChain-OpenTutorial GitHub 教程]
- [来源: 03_核心概念_2_BaseLoader接口设计.md]
- [来源: 03_核心概念_4_懒加载模式.md]
