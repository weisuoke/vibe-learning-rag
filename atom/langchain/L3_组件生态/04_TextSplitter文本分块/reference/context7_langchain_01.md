---
type: context7_documentation
library: langchain-text-splitters
version: latest
fetched_at: 2026-02-25
knowledge_point: 04_TextSplitter文本分块
context7_query: TextSplitter RecursiveCharacterTextSplitter usage best practices chunk_size chunk_overlap
---

# Context7 文档：LangChain TextSplitter

## 文档来源
- 库名称：LangChain
- Library ID：/websites/langchain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. RecursiveCharacterTextSplitter 初始化

**来源**：https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter

**核心参数**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 块大小（字符数）
    chunk_overlap=20,  # 块重叠（字符数）
    length_function=len,  # 长度计算函数
    is_separator_regex=False,  # 是否使用正则表达式分隔符
)
```

**使用方法**：
- `create_documents([text])`: 创建 Document 对象列表
- `split_documents(docs)`: 分割现有 Document 对象

### 2. 推荐配置

**来源**：https://docs.langchain.com/oss/python/langchain/rag

**RAG 应用推荐配置**：
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 推荐 1000 字符
    chunk_overlap=200,  # 推荐 200 字符（20% 重叠）
    add_start_index=True,  # 跟踪原始文档中的索引
)
```

**配置说明**：
- `chunk_size=1000`: 适合大多数 LLM 的 context window
- `chunk_overlap=200`: 20% 重叠率，保持上下文连续性
- `add_start_index=True`: 便于溯源和调试

### 3. 向量存储集成

**来源**：https://docs.langchain.com/oss/python/integrations/vectorstores/azure_cosmos_db_no_sql

**典型用法**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
docs = text_splitter.split_documents(data)

# 然后将 docs 存入向量数据库
```

**重叠配置**：
- 150-200 字符的重叠是常见选择
- 重叠有助于缓解信息在块边界处丢失的问题

### 4. RecursiveCharacterTextSplitter 策略

**来源**：https://docs.langchain.com/oss/javascript/langchain/knowledge-base

**核心策略**：
- **递归分割**：使用常见分隔符（如换行符）递归分割文档
- **分隔符优先级**：从大到小（段落 → 行 → 单词 → 字符）
- **块大小控制**：确保每个块达到适当大小
- **重叠处理**：在块之间保持重叠，避免分离重要上下文

**推荐用途**：
- 通用文本分块的首选方案
- 适合大多数文本类型
- 平衡了语义完整性和块大小控制

### 5. 参数详解

**来源**：https://docs.langchain.com/oss/javascript/integrations/splitters/recursive_text_splitter

**chunkSize（chunk_size）**：
- 定义块的最大大小
- 大小由 `lengthFunction` 确定
- 通常使用字符数或 token 数

**chunkOverlap（chunk_overlap）**：
- 指定块之间的目标重叠
- 对于缓解跨块的信息丢失至关重要
- 通常设置为 chunk_size 的 10-20%

**lengthFunction（length_function）**：
- 用于计算文本长度的函数
- 默认使用 `len()`（字符数）
- 可替换为 token 计数函数

### 6. 最佳实践总结

**chunk_size 选择**：
- 小文档（< 1000 字符）：chunk_size=500
- 中等文档（1000-5000 字符）：chunk_size=1000
- 大文档（> 5000 字符）：chunk_size=1500-2000

**chunk_overlap 选择**：
- 通常设置为 chunk_size 的 15-20%
- 最小值：100 字符
- 最大值：chunk_size 的 30%

**性能优化**：
- 使用 `add_start_index=True` 便于调试
- 对于大量文档，考虑批处理
- 使用合适的 `length_function` 控制 token 数量

### 7. 与 LangChain 生态集成

**文档加载 → 分块 → 向量化流程**：
```python
# 1. 加载文档
from langchain_community.document_loaders import TextLoader
loader = TextLoader("document.txt")
docs = loader.load()

# 2. 分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 3. 向量化和存储
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
```

**关键集成点**：
- 输入：Document 对象（来自 DocumentLoader）
- 输出：Document 对象列表（传递给 VectorStore）
- 元数据：自动保留和传递

### 8. 常见问题

**问题 1：块大小不一致**
- 原因：分隔符分布不均
- 解决：调整 chunk_size 或使用更细粒度的分隔符

**问题 2：上下文丢失**
- 原因：chunk_overlap 太小
- 解决：增加 chunk_overlap 到 chunk_size 的 20%

**问题 3：检索质量差**
- 原因：块太大或太小
- 解决：根据文档类型调整 chunk_size

### 9. 高级用法

**使用 token 计数**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,  # token 数量
    chunk_overlap=200
)
```

**自定义分隔符**：
```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
```

### 10. 性能考虑

**内存使用**：
- `split_documents()` 一次性加载所有文档
- 对于大文件，考虑流式处理

**处理速度**：
- RecursiveCharacterTextSplitter 性能良好
- 对于极大文档，可能需要优化

**可扩展性**：
- 支持批处理
- 可与异步操作结合
