---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-24
knowledge_point: 03_DocumentLoader文档加载
context7_query: DocumentLoader BaseLoader BlobLoader document loading
---

# Context7 文档：LangChain DocumentLoader

## 文档来源
- 库名称: LangChain
- 版本: latest (2026-02-17)
- 官方文档链接: https://docs.langchain.com/oss/python/integrations/document_loaders
- Context7 库 ID: /websites/langchain
- 总代码片段数: 26795
- 信任评分: 10/10
- 基准评分: 83/100

## 关键信息提取

### 1. DocumentLoader 概述

**标准接口**:
- 所有文档加载器实现 `BaseLoader` 接口
- 提供统一的 API 跨所有集成
- 从不同数据源(Slack, Notion, Google Drive)读取数据到 LangChain 的 Document 格式

**核心方法**:
- `load()`: 一次性加载所有文档
- `lazy_load()`: 流式懒加载文档(适合大数据集,管理内存)
- `loadAndSplit()`: 加载所有文档并分块

### 2. CSVLoader 使用示例

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    ...  # Integration-specific parameters here
)

# Load all documents
documents = loader.load()

# For large datasets, lazily load documents
for document in loader.lazy_load():
    print(document)
```

**关键特性**:
- 支持同步 `load()` 方法
- 支持懒加载 `lazy_load()` 方法
- 适合大数据集的内存管理

### 3. Azure Blob Storage 加载器

```python
loader = AzureBlobStorageFileLoader(
    conn_str=conn_str, container=container_name, blob_name=blob_name
)

documents = loader.load()
```

**用途**:
- 从 Azure Blob Storage 检索文档
- 使用连接字符串和 blob 名称
- 返回文档对象列表

### 4. WebBaseLoader 示例

**功能**:
- 使用 urllib 从 web URLs 加载 HTML
- 使用 BeautifulSoup 解析 HTML 为文本
- 支持自定义 HTML 到文本的解析

**自定义解析**:
```python
# 通过 bs_kwargs 传递参数到 BeautifulSoup 解析器
# 可以根据 class 名称或其他属性过滤和保留相关 HTML 元素
```

### 5. 接口设计(JavaScript 版本参考)

**共同 API**:
- `load()`: 一次性加载所有文档
- `loadAndSplit()`: 一次性加载所有文档并分块

**注意**: Python 版本还包含 `lazy_load()` 方法

## 设计理念

### 1. 标准化接口
- 所有加载器实现相同的 `BaseLoader` 接口
- 确保数据处理的一致性
- 无论数据源如何,都能统一处理

### 2. 懒加载优先
- `lazy_load()` 方法用于流式处理
- 避免大数据集的内存溢出
- 适合生产环境的大规模数据处理

### 3. 集成友好
- 支持多种数据源(文件、云存储、Web、数据库)
- 每个加载器可以定义自己的参数
- 但共享统一的 API

## 实际应用场景

### 1. RAG 应用
- 从各种数据源加载文档
- 转换为统一的 Document 格式
- 用于向量化和检索

### 2. 数据管道
- 批量加载文档
- 流式处理大数据集
- 与 TextSplitter 集成分块

### 3. 多源数据集成
- 从 Slack、Notion、Google Drive 等加载
- 统一处理不同格式的数据
- 保留元数据用于溯源

## 最佳实践

### 1. 大数据集处理
- 优先使用 `lazy_load()` 而不是 `load()`
- 避免一次性加载所有数据到内存
- 使用流式处理提升性能

### 2. 自定义解析
- 使用 BeautifulSoup 参数自定义 HTML 解析
- 根据需求过滤和保留相关内容
- 提取结构化信息

### 3. 元数据管理
- 保留文档来源信息
- 用于混合检索和过滤
- 支持溯源和审计

## 与 RAG 开发的关系

### 1. 数据加载阶段
- DocumentLoader 是 RAG 管道的第一步
- 负责从各种数据源加载原始数据
- 转换为统一的 Document 格式

### 2. 与其他组件的集成
- 与 TextSplitter 集成: `loadAndSplit()` 方法
- 与 VectorStore 集成: 加载的 Document 用于向量化
- 与 Retriever 集成: 提供检索的数据源

### 3. 生产环境考虑
- 使用懒加载处理大数据集
- 保留元数据用于过滤和溯源
- 支持多种数据源的统一处理

## 总结

LangChain 的 DocumentLoader 提供了:
1. **标准化接口**: 所有加载器实现 BaseLoader
2. **懒加载支持**: 适合大数据集的内存管理
3. **多源集成**: 支持文件、云存储、Web、数据库等
4. **RAG 友好**: 与 TextSplitter、VectorStore 无缝集成
5. **生产就绪**: 支持流式处理和大规模数据

这使得 LangChain 能够轻松集成各种数据源,为 RAG 应用提供统一的数据加载解决方案。
