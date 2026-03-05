# 核心概念 8：RecursiveJsonSplitter

> JSON 结构感知的专用分块器

---

## 概述

RecursiveJsonSplitter 是 LangChain 中专门用于处理 JSON 数据的分块器。与通用的文本分块器不同，它基于 JSON 的结构进行分块，保留键值对的完整性和嵌套关系。这对于需要处理 API 响应、配置文件、结构化数据的 RAG 应用至关重要。

**核心特点**：
- 基于 JSON 结构分块
- 保留键值对完整性
- 支持嵌套 JSON
- 不继承自 TextSplitter（独立实现）
- 适合结构化数据处理

---

## 为什么需要 RecursiveJsonSplitter？

### 问题背景

在 RAG 开发中处理 JSON 数据时，我们经常遇到以下问题：

1. **结构被破坏**：使用通用分块器会在键值对中间切断
   ```json
   {
     "user": {
       "name": "Alice",
   // ← 在这里切断
       "age": 30
     }
   }
   ```

2. **语义丢失**：键和值分离，无法理解数据含义
   - 键名和值分离
   - 嵌套关系丢失
   - 数组元素被切断

3. **检索质量差**：无法按照数据结构进行检索
   - 无法检索完整的对象
   - 无法检索完整的数组
   - 难以理解数据的层级关系

### RecursiveJsonSplitter 的解决方案

RecursiveJsonSplitter 通过识别 JSON 结构，智能分块：

```python
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
    "users": [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "Bob", "age": 25, "email": "bob@example.com"}
    ],
    "metadata": {
        "version": "1.0",
        "created": "2026-01-01"
    }
}

splitter = RecursiveJsonSplitter(max_chunk_size=1000)

# 分块 JSON
chunks = splitter.split_json(json_data)

# 每个块保留完整的 JSON 结构
for i, chunk in enumerate(chunks):
    print(f"块 {i+1}:")
    print(chunk)
    print()
```

---

## 核心参数

### 1. max_chunk_size（最大块大小）

**类型**：`int`
**默认值**：`2000`
**单位**：字符数

**说明**：
- 每个块的最大字符数
- JSON 分块推荐值：1000-2000 字符
- 需要根据 JSON 复杂度调整

**示例**：
```python
# 小型 JSON 对象
splitter = RecursiveJsonSplitter(max_chunk_size=500)

# 大型 JSON 对象（推荐）
splitter = RecursiveJsonSplitter(max_chunk_size=1000)

# 复杂嵌套 JSON
splitter = RecursiveJsonSplitter(max_chunk_size=2000)
```

### 2. min_chunk_size（最小块大小）

**类型**：`int | None`
**默认值**：`None`

**说明**：
- 每个块的最小字符数
- 如果设置，会尝试合并小块
- 通常不需要设置

---

## 使用方法

### 方法 1：从 Python 字典分块

```python
from langchain_text_splitters import RecursiveJsonSplitter

json_data = {
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ],
    "metadata": {
        "version": "1.0"
    }
}

splitter = RecursiveJsonSplitter(max_chunk_size=1000)
chunks = splitter.split_json(json_data)

for chunk in chunks:
    print(chunk)
```

### 方法 2：从 JSON 字符串分块

```python
import json
from langchain_text_splitters import RecursiveJsonSplitter

json_string = """
{
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
}
"""

# 解析 JSON 字符串
json_data = json.loads(json_string)

# 分块
splitter = RecursiveJsonSplitter(max_chunk_size=1000)
chunks = splitter.split_json(json_data)
```

### 方法 3：从文件加载

```python
import json
from langchain_text_splitters import RecursiveJsonSplitter

# 读取 JSON 文件
with open("data.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 分块
splitter = RecursiveJsonSplitter(max_chunk_size=1000)
chunks = splitter.split_json(json_data)
```

---

## 实战示例

### 示例 1：API 响应处理

**场景**：处理 API 返回的 JSON 数据

```python
from langchain_text_splitters import RecursiveJsonSplitter
import json

# API 响应
api_response = {
    "status": "success",
    "data": {
        "users": [
            {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "profile": {
                    "bio": "Software engineer",
                    "location": "San Francisco"
                }
            },
            {
                "id": 2,
                "name": "Bob",
                "email": "bob@example.com",
                "profile": {
                    "bio": "Data scientist",
                    "location": "New York"
                }
            }
        ]
    },
    "pagination": {
        "page": 1,
        "total": 100
    }
}

# 分块
splitter = RecursiveJsonSplitter(max_chunk_size=500)
chunks = splitter.split_json(api_response)

print(f"总共分块: {len(chunks)} 个\n")

for i, chunk in enumerate(chunks):
    print(f"块 {i+1}:")
    print(json.dumps(chunk, indent=2, ensure_ascii=False))
    print()
```

### 示例 2：配置文件处理

**场景**：处理应用配置文件

```python
from langchain_text_splitters import RecursiveJsonSplitter
import json

# 配置文件
config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "credentials": {
            "username": "admin",
            "password": "secret"
        }
    },
    "api": {
        "endpoints": [
            {"path": "/users", "method": "GET"},
            {"path": "/users", "method": "POST"}
        ],
        "rate_limit": 1000
    },
    "logging": {
        "level": "INFO",
        "file": "/var/log/app.log"
    }
}

# 分块
splitter = RecursiveJsonSplitter(max_chunk_size=300)
chunks = splitter.split_json(config)

for i, chunk in enumerate(chunks):
    print(f"块 {i+1}:")
    print(json.dumps(chunk, indent=2))
    print()
```

### 示例 3：RAG 知识库构建

**场景**：构建基于 JSON 数据的知识库

```python
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import json

# JSON 数据
json_data = {
    "products": [
        {
            "id": 1,
            "name": "Laptop",
            "specs": {
                "cpu": "Intel i7",
                "ram": "16GB",
                "storage": "512GB SSD"
            },
            "price": 1200
        },
        {
            "id": 2,
            "name": "Mouse",
            "specs": {
                "type": "Wireless",
                "dpi": 1600
            },
            "price": 30
        }
    ]
}

# 1. 分块
splitter = RecursiveJsonSplitter(max_chunk_size=500)
chunks = splitter.split_json(json_data)

# 2. 转换为文本
texts = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]

# 3. 向量化和存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embeddings)

# 4. 检索
query = "笔记本电脑的配置"
results = vectorstore.similarity_search(query, k=2)

for result in results:
    print(f"检索结果:")
    print(result.page_content)
    print()
```

---

## 适用场景

### 1. API 响应处理

**场景**：处理 REST API 返回的 JSON 数据

**为什么使用 RecursiveJsonSplitter**：
- 保留 JSON 结构
- 便于理解 API 响应
- 支持嵌套对象和数组

### 2. 配置文件管理

**场景**：处理应用配置文件

**为什么使用 RecursiveJsonSplitter**：
- 保留配置的层级结构
- 便于检索特定配置项
- 支持复杂的配置结构

### 3. 结构化数据检索

**场景**：构建基于结构化数据的检索系统

**为什么使用 RecursiveJsonSplitter**：
- 保留数据的完整性
- 便于按照数据结构检索
- 支持复杂的数据关系

### 4. 日志分析

**场景**：处理 JSON 格式的日志文件

**为什么使用 RecursiveJsonSplitter**：
- 保留日志的结构信息
- 便于分析日志内容
- 支持嵌套的日志字段

---

## 常见问题

### Q1：RecursiveJsonSplitter 和 RecursiveCharacterTextSplitter 有什么区别？

**A**：

| 特性 | RecursiveJsonSplitter | RecursiveCharacterTextSplitter |
|------|----------------------|-------------------------------|
| 输入格式 | JSON（字典/列表） | 文本字符串 |
| 分块依据 | JSON 结构 | 字符分隔符 |
| 结构保留 | 是 | 否 |
| 适用场景 | 结构化数据 | 通用文本 |

**推荐**：
- JSON 数据 → RecursiveJsonSplitter
- 通用文本 → RecursiveCharacterTextSplitter

### Q2：如何处理超大的 JSON 对象？

**A**：

**方案 1：调整 max_chunk_size**
```python
# 增加块大小
splitter = RecursiveJsonSplitter(max_chunk_size=5000)
```

**方案 2：预处理 JSON**
```python
# 提取特定字段
filtered_data = {
    "users": json_data["users"][:100]  # 只取前100个用户
}
chunks = splitter.split_json(filtered_data)
```

### Q3：如何保留 JSON 的元数据？

**A**：

```python
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.schema import Document
import json

# 分块
splitter = RecursiveJsonSplitter(max_chunk_size=1000)
chunks = splitter.split_json(json_data)

# 创建 Document 对象，保留元数据
documents = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=json.dumps(chunk, ensure_ascii=False),
        metadata={
            "chunk_id": i,
            "source": "api_response",
            "type": "json"
        }
    )
    documents.append(doc)
```

### Q4：如何处理 JSON 数组？

**A**：

RecursiveJsonSplitter 会自动处理数组：

```python
json_data = {
    "items": [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
        {"id": 3, "name": "Item 3"}
    ]
}

splitter = RecursiveJsonSplitter(max_chunk_size=100)
chunks = splitter.split_json(json_data)

# 数组会被智能分块，保留每个元素的完整性
```

---

## 与 RAG 开发的联系

### 1. 结构化数据知识库

在 RAG 应用中，RecursiveJsonSplitter 帮助构建基于结构化数据的知识库：

```python
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import json

# JSON 数据列表
json_files = ["products.json", "users.json", "orders.json"]

splitter = RecursiveJsonSplitter(max_chunk_size=1000)

all_texts = []
for file_path in json_files:
    with open(file_path, "r") as f:
        json_data = json.load(f)
    chunks = splitter.split_json(json_data)
    texts = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    all_texts.extend(texts)

# 构建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(all_texts, embeddings)

print(f"知识库包含 {len(all_texts)} 个 JSON 块")
```

### 2. API 响应检索

RecursiveJsonSplitter 通过保留 JSON 结构，优化 API 响应检索：

```python
# 检索特定的 JSON 数据
query = "用户的邮箱地址"
results = vectorstore.similarity_search(query, k=3)

for result in results:
    # 解析 JSON
    data = json.loads(result.page_content)
    print(f"检索结果:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
```

---

## 最佳实践

### 1. 选择合适的 max_chunk_size

```python
# ✓ 推荐：根据 JSON 复杂度选择
# 简单 JSON
splitter = RecursiveJsonSplitter(max_chunk_size=500)

# 复杂嵌套 JSON
splitter = RecursiveJsonSplitter(max_chunk_size=2000)

# ✗ 不推荐：使用过小的 max_chunk_size
splitter = RecursiveJsonSplitter(max_chunk_size=100)
```

### 2. 验证 JSON 格式

```python
# ✓ 推荐：验证 JSON 格式
import json

try:
    json_data = json.loads(json_string)
    chunks = splitter.split_json(json_data)
except json.JSONDecodeError as e:
    print(f"JSON 格式错误: {e}")

# ✗ 不推荐：不验证直接分块
chunks = splitter.split_json(json_data)
```

### 3. 保留源信息

```python
# ✓ 推荐：保留 JSON 来源信息
chunks = splitter.split_json(json_data)
for i, chunk in enumerate(chunks):
    chunk["_metadata"] = {
        "source": "api_response",
        "chunk_id": i
    }

# ✗ 不推荐：不保留源信息
chunks = splitter.split_json(json_data)
```

### 4. 处理编码问题

```python
# ✓ 推荐：使用 ensure_ascii=False
texts = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]

# ✗ 不推荐：使用默认编码（中文会被转义）
texts = [json.dumps(chunk) for chunk in chunks]
```

---

## 数据来源

- [来源: reference/source_textsplitter_01_base.md | LangChain 源码]
- [来源: JSON 规范 | JSON 结构说明]
