---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/embeddings/embeddings.py
analyzed_files:
  - langchain_core/embeddings/embeddings.py
analyzed_at: 2026-02-25
knowledge_point: Embedding模型集成
---

# 源码分析：Embeddings 基类设计

## 分析的文件
- `langchain_core/embeddings/embeddings.py` (79行) - Embeddings 抽象基类

## 关键发现

### 1. 设计哲学：极简抽象接口

**核心设计**：
- 使用 ABC (Abstract Base Class) 定义接口
- 只定义 4 个方法：2 个抽象方法 + 2 个异步方法
- 遵循"接口隔离原则"：只暴露必需的方法

**代码结构**：
```python
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)
```

### 2. 文档与查询的分离设计

**为什么分离？**

LangChain 将 Embedding 操作分为两种：
1. **`embed_documents`** - 批量嵌入文档（用于索引构建）
2. **`embed_query`** - 单个查询嵌入（用于检索）

**设计理由**（来自源码注释）：
> "Usually the query embedding is identical to the document embedding, but the abstraction allows treating them independently."

**实际应用场景**：
- 某些模型对文档和查询使用不同的 prompt
- 某些模型对查询有特殊的优化（如 OpenAI 的 `text-embedding-ada-002`）
- 允许未来扩展（如查询重写、查询增强）

### 3. 异步支持的默认实现

**设计模式**：Adapter Pattern

LangChain 为异步方法提供了默认实现：
```python
async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    return await run_in_executor(None, self.embed_documents, texts)
```

**关键点**：
- 使用 `run_in_executor` 将同步方法包装为异步
- 子类可以选择覆盖异步方法以提供原生异步实现
- 这是一种"渐进式异步"设计：默认可用，但可优化

**性能考虑**（来自源码注释）：
> "By default, the asynchronous methods are implemented using the synchronous methods; however, implementations may choose to override the asynchronous methods with an async native implementation for performance reasons."

### 4. 类型系统设计

**输入输出类型**：
- `embed_documents(texts: list[str]) -> list[list[float]]`
  - 输入：字符串列表
  - 输出：二维浮点数列表（每个文档一个向量）

- `embed_query(text: str) -> list[float]`
  - 输入：单个字符串
  - 输出：一维浮点数列表（单个向量）

**设计优势**：
- 类型明确，易于理解
- 支持类型检查（mypy）
- 与 Python 标准库兼容（不依赖 numpy）

### 5. 文档字符串的设计

**关键信息**：
```python
"""Interface for embedding models.

This is an interface meant for implementing text embedding models.

Text embedding models are used to map text to a vector (a point in n-dimensional
space).

Texts that are similar will usually be mapped to points that are close to each
other in this space. The exact details of what's considered "similar" and how
"distance" is measured in this space are dependent on the specific embedding model.
```

**教学价值**：
- 清晰解释了 Embedding 的本质（文本 → 向量）
- 说明了相似性的概念（相似文本 → 相近向量）
- 强调了模型依赖性（不同模型有不同的相似度定义）

## 架构决策分析

### 决策1：为什么不使用 numpy？

**观察**：返回类型是 `list[list[float]]` 而非 `np.ndarray`

**可能原因**：
1. **依赖最小化**：核心抽象不依赖 numpy
2. **序列化友好**：Python 列表更容易序列化
3. **类型兼容性**：与 JSON、Pydantic 等工具兼容
4. **性能权衡**：对于小规模向量，列表性能足够

### 决策2：为什么不支持批量查询？

**观察**：`embed_query` 只接受单个字符串

**可能原因**：
1. **语义清晰**：查询通常是单个的
2. **简化接口**：避免接口复杂化
3. **实现灵活性**：子类可以自行优化批量查询

### 决策3：为什么异步方法不是抽象的？

**观察**：异步方法有默认实现

**设计理由**：
1. **向后兼容**：旧代码只需实现同步方法
2. **渐进式优化**：可以先用默认实现，后续优化
3. **降低实现门槛**：不强制要求异步实现

## 与 RAG 开发的联系

### 在 RAG 中的使用

1. **文档索引阶段**：
   ```python
   embeddings = OpenAIEmbeddings()
   doc_vectors = embeddings.embed_documents(documents)
   # 存储到向量数据库
   ```

2. **查询检索阶段**：
   ```python
   query_vector = embeddings.embed_query(user_query)
   # 在向量数据库中检索相似文档
   ```

### 设计对 RAG 的影响

1. **批量处理优化**：`embed_documents` 支持批量，提高索引构建效率
2. **查询优化**：`embed_query` 单独处理，允许查询特定优化
3. **异步支持**：在高并发 RAG 应用中，异步方法提高吞吐量

## 实现建议

### 如何实现自定义 Embeddings

```python
from langchain_core.embeddings import Embeddings

class MyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 实现批量嵌入逻辑
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # 实现单个查询嵌入逻辑
        return self._embed_single(text)

    def _embed_single(self, text: str) -> list[float]:
        # 实际的嵌入逻辑
        pass
```

### 性能优化建议

1. **覆盖异步方法**：如果有原生异步 API，覆盖 `aembed_*` 方法
2. **批量优化**：在 `embed_documents` 中使用批量 API
3. **缓存**：考虑使用 `CacheBackedEmbeddings` 包装器

## 总结

LangChain 的 Embeddings 基类设计体现了以下原则：
1. **极简主义**：只定义必需的接口
2. **灵活性**：允许文档和查询使用不同的嵌入策略
3. **渐进式优化**：提供默认实现，但允许优化
4. **类型安全**：明确的类型注解
5. **文档友好**：清晰的文档字符串

这种设计使得 LangChain 的 Embeddings 接口既简单易用，又足够灵活以支持各种实现。
