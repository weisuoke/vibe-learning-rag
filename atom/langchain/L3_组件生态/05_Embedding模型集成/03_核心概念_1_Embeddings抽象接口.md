# 核心概念 1：Embeddings 抽象接口

> LangChain Embeddings 基类的设计哲学与实现细节

---

## 元数据

**知识点层级**：L3_组件生态 > 05_Embedding模型集成 > 核心概念1
**难度等级**：⭐⭐⭐ (中级)
**预计学习时间**：30 分钟
**前置知识**：Python ABC、类型注解、异步编程基础
**适用场景**：理解 LangChain Embeddings 架构、实现自定义 Embeddings、RAG 系统开发

---

## 概述

LangChain 的 Embeddings 抽象接口是整个 Embedding 生态的基石。它通过极简的设计定义了所有 Embedding 模型必须遵循的契约，使得开发者可以无缝切换不同的 Embedding 提供商（OpenAI、HuggingFace、Cohere 等），而无需修改上层代码。

**核心价值**：
- **统一接口**：所有 Embedding 模型遵循相同的接口规范
- **灵活扩展**：轻松实现自定义 Embedding 模型
- **类型安全**：明确的类型注解，支持静态类型检查
- **异步支持**：内置异步方法，适配高并发场景

[来源: reference/source_embeddings_base_01.md]

---

## 1. Embeddings 基类设计

### 1.1 ABC 抽象基类架构

LangChain 使用 Python 的 `ABC`（Abstract Base Class）定义 Embeddings 接口，这是一种经典的面向对象设计模式。

**源码结构**：

```python
from abc import ABC, abstractmethod

class Embeddings(ABC):
    """Interface for embedding models.

    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in
    n-dimensional space).

    Texts that are similar will usually be mapped to points that are close
    to each other in this space. The exact details of what's considered
    "similar" and how "distance" is measured in this space are dependent
    on the specific embedding model.
    """

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

[来源: reference/source_embeddings_base_01.md]

**设计特点**：

1. **极简主义**：只定义 4 个方法（2 个抽象 + 2 个异步）
2. **强制实现**：子类必须实现 `embed_documents` 和 `embed_query`
3. **可选优化**：异步方法有默认实现，但可以覆盖优化
4. **文档友好**：清晰的 docstring 解释了 Embedding 的本质

### 1.2 设计哲学：接口隔离原则

LangChain 的 Embeddings 接口遵循 SOLID 原则中的**接口隔离原则**（Interface Segregation Principle）：

> "客户端不应该被迫依赖它不使用的方法"

**体现**：
- 只暴露必需的方法（embed_documents、embed_query）
- 不包含配置、初始化等实现细节
- 子类可以自由添加额外功能，但不影响接口契约

**对比其他设计**：

| 设计方案 | LangChain 方案 | 替代方案 |
|---------|---------------|---------|
| 方法数量 | 4 个（2 抽象 + 2 异步） | 可能包含 10+ 个方法 |
| 配置管理 | 子类自行处理 | 接口定义配置方法 |
| 批量处理 | embed_documents 内置 | 单独的 batch 方法 |
| 异步支持 | 默认实现 + 可覆盖 | 强制实现异步方法 |

[来源: reference/source_embeddings_base_01.md]

---

## 2. 核心方法详解

### 2.1 embed_documents() - 批量文档嵌入

**方法签名**：

```python
@abstractmethod
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """Embed search docs."""
```

**功能说明**：
- **输入**：字符串列表（多个文档）
- **输出**：二维浮点数列表（每个文档对应一个向量）
- **用途**：批量嵌入文档，用于构建向量索引

**典型使用场景**：

```python
from langchain_openai import OpenAIEmbeddings

# 初始化 Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 批量嵌入文档
documents = [
    "LangChain is a framework for developing LLM applications.",
    "RAG combines retrieval and generation for better answers.",
    "Vector databases store high-dimensional embeddings."
]

# 返回 3 个向量，每个向量 1536 维（text-embedding-3-small 默认维度）
doc_vectors = embeddings.embed_documents(documents)

print(f"嵌入了 {len(doc_vectors)} 个文档")
print(f"每个向量维度: {len(doc_vectors[0])}")
# 输出:
# 嵌入了 3 个文档
# 每个向量维度: 1536
```

[来源: reference/context7_langchain_openai_02.md]

**性能优化要点**：

1. **批量 API 调用**：大多数 Embedding 服务支持批量请求，减少网络开销
2. **并发控制**：避免一次性发送过多请求导致速率限制
3. **缓存机制**：对相同文档使用缓存避免重复计算

```python
# 使用 LangChain 的缓存包装器
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# 创建本地缓存存储
store = LocalFileStore("./embedding_cache")

# 包装 Embeddings 以启用缓存
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    store,
    namespace="openai_embeddings"
)

# 第一次调用会计算并缓存
vectors1 = cached_embeddings.embed_documents(documents)

# 第二次调用直接从缓存读取（速度快 100 倍）
vectors2 = cached_embeddings.embed_documents(documents)
```

### 2.2 embed_query() - 单个查询嵌入

**方法签名**：

```python
@abstractmethod
def embed_query(self, text: str) -> list[float]:
    """Embed query text."""
```

**功能说明**：
- **输入**：单个字符串（查询文本）
- **输出**：一维浮点数列表（单个向量）
- **用途**：嵌入用户查询，用于检索相似文档

**典型使用场景**：

```python
# 用户查询
user_query = "What is RAG in LangChain?"

# 嵌入查询
query_vector = embeddings.embed_query(user_query)

print(f"查询向量维度: {len(query_vector)}")
print(f"向量前 5 个值: {query_vector[:5]}")
# 输出:
# 查询向量维度: 1536
# 向量前 5 个值: [-0.0086, -0.0334, -0.0089, -0.0037, 0.0106]
```

[来源: reference/context7_langchain_openai_02.md]

### 2.3 为什么分离文档和查询嵌入？

LangChain 将 Embedding 操作分为 `embed_documents` 和 `embed_query` 两个方法，这是一个深思熟虑的设计决策。

**官方解释**（来自源码注释）：

> "Usually the query embedding is identical to the document embedding, but the abstraction allows treating them independently."

[来源: reference/source_embeddings_base_01.md]

**分离的理由**：

#### 理由 1：不同的优化策略

某些 Embedding 模型对文档和查询使用不同的处理策略：

```python
class AsymmetricEmbeddings(Embeddings):
    """非对称 Embedding 模型示例"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 文档使用完整的上下文编码
        return self.model.encode(
            texts,
            prompt="passage: ",  # 文档前缀
            normalize_embeddings=True
        )

    def embed_query(self, text: str) -> list[float]:
        # 查询使用简短的编码
        return self.model.encode(
            text,
            prompt="query: ",  # 查询前缀
            normalize_embeddings=True
        )
```

**实际案例**：
- **E5 模型**：要求查询前加 `query:` 前缀，文档前加 `passage:` 前缀
- **BGE 模型**：查询需要添加特殊指令 "Represent this sentence for searching relevant passages:"

#### 理由 2：批量 vs 单个处理

```python
# 文档嵌入：批量处理，优化吞吐量
doc_vectors = embeddings.embed_documents([doc1, doc2, doc3, ...])

# 查询嵌入：单个处理，优化延迟
query_vector = embeddings.embed_query(user_query)
```

**性能差异**：
- `embed_documents`：批量 API 调用，适合离线索引构建
- `embed_query`：单次调用，适合在线查询响应

#### 理由 3：未来扩展性

分离接口为未来的优化留下空间：

```python
class AdvancedEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        # 查询可以添加额外的处理
        expanded_query = self.query_expansion(text)  # 查询扩展
        rewritten_query = self.query_rewrite(expanded_query)  # 查询重写
        return self.model.encode(rewritten_query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 文档保持原样
        return self.model.encode(texts)
```

**在 RAG 中的应用**：

```python
# RAG 索引构建阶段
documents = load_documents("knowledge_base/")
doc_vectors = embeddings.embed_documents([doc.page_content for doc in documents])
vector_store.add_vectors(doc_vectors, documents)

# RAG 查询阶段
user_query = "How to use LangChain?"
query_vector = embeddings.embed_query(user_query)
similar_docs = vector_store.similarity_search(query_vector, k=5)
```

[来源: reference/source_embeddings_base_01.md]

---

## 3. 异步支持

### 3.1 默认异步实现

LangChain 为异步方法提供了开箱即用的默认实现：

```python
async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    """Asynchronous Embed search docs."""
    return await run_in_executor(None, self.embed_documents, texts)

async def aembed_query(self, text: str) -> list[float]:
    """Asynchronous Embed query text."""
    return await run_in_executor(None, self.embed_query, text)
```

**设计模式**：Adapter Pattern（适配器模式）

**工作原理**：
- 使用 `run_in_executor` 将同步方法包装为异步
- 在线程池中执行同步方法，避免阻塞事件循环
- 子类可以选择覆盖以提供原生异步实现

[来源: reference/source_embeddings_base_01.md]

### 3.2 何时使用异步方法？

**适用场景**：

1. **高并发 Web 应用**：

```python
from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings

app = FastAPI()
embeddings = OpenAIEmbeddings()

@app.post("/embed")
async def embed_text(text: str):
    # 使用异步方法避免阻塞其他请求
    vector = await embeddings.aembed_query(text)
    return {"vector": vector}
```

2. **批量异步处理**：

```python
import asyncio

async def embed_multiple_queries(queries: list[str]):
    # 并发处理多个查询
    tasks = [embeddings.aembed_query(q) for q in queries]
    vectors = await asyncio.gather(*tasks)
    return vectors

# 同时处理 10 个查询
queries = [f"Query {i}" for i in range(10)]
vectors = await embed_multiple_queries(queries)
```

3. **RAG 流式响应**：

```python
async def streaming_rag(query: str):
    # 异步嵌入查询
    query_vector = await embeddings.aembed_query(query)

    # 异步检索文档
    docs = await vector_store.asimilarity_search(query_vector)

    # 异步生成回答（流式）
    async for chunk in llm.astream(build_prompt(query, docs)):
        yield chunk
```

### 3.3 性能优化：覆盖异步方法

如果 Embedding 服务提供原生异步 API，应该覆盖默认实现：

```python
import httpx

class CustomAsyncEmbeddings(Embeddings):
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.client = httpx.AsyncClient()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 同步实现（使用 httpx 同步客户端）
        response = httpx.post(f"{self.api_url}/embed", json={"texts": texts})
        return response.json()["vectors"]

    def embed_query(self, text: str) -> list[float]:
        response = httpx.post(f"{self.api_url}/embed", json={"texts": [text]})
        return response.json()["vectors"][0]

    # 覆盖异步方法以使用原生异步 API
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.post(
            f"{self.api_url}/embed",
            json={"texts": texts}
        )
        return response.json()["vectors"]

    async def aembed_query(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.api_url}/embed",
            json={"texts": [text]}
        )
        return response.json()["vectors"][0]
```

**性能对比**：

| 实现方式 | 10 个查询耗时 | 100 个查询耗时 |
|---------|-------------|--------------|
| 默认异步（线程池） | ~2.5s | ~25s |
| 原生异步 API | ~0.8s | ~8s |
| 性能提升 | **3.1x** | **3.1x** |

[来源: reference/source_embeddings_base_01.md]

---

## 4. 类型系统

### 4.1 输入输出类型

LangChain 使用明确的类型注解定义接口契约：

```python
# 文档嵌入
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    #                        ↑ 输入          ↑ 输出
    #                   字符串列表      二维浮点数列表

# 查询嵌入
def embed_query(self, text: str) -> list[float]:
    #                     ↑ 输入    ↑ 输出
    #                  单个字符串  一维浮点数列表
```

**类型映射**：

| 方法 | 输入类型 | 输出类型 | 示例 |
|-----|---------|---------|------|
| `embed_documents` | `list[str]` | `list[list[float]]` | `[[-0.01, 0.02, ...], [0.03, -0.04, ...]]` |
| `embed_query` | `str` | `list[float]` | `[-0.01, 0.02, 0.03, ...]` |

### 4.2 为什么使用 list 而非 numpy.ndarray？

**设计决策**：返回 Python 原生 `list[float]` 而非 `numpy.ndarray`

**理由**：

1. **依赖最小化**：核心抽象不依赖 numpy
2. **序列化友好**：Python 列表可直接序列化为 JSON
3. **类型兼容性**：与 Pydantic、FastAPI 等工具无缝集成
4. **性能权衡**：对于小规模向量（<10K 维），列表性能足够

**实际使用**：

```python
# LangChain 返回 list
vectors = embeddings.embed_documents(texts)
type(vectors)  # <class 'list'>
type(vectors[0])  # <class 'list'>

# 如需 numpy，手动转换
import numpy as np
np_vectors = np.array(vectors)
type(np_vectors)  # <class 'numpy.ndarray'>
```

[来源: reference/source_embeddings_base_01.md]

### 4.3 类型检查与验证

使用 `mypy` 进行静态类型检查：

```python
from langchain_core.embeddings import Embeddings

def process_embeddings(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    # mypy 会检查类型是否匹配
    vectors = embeddings.embed_documents(texts)
    return vectors

# 类型错误示例
def wrong_usage(embeddings: Embeddings):
    # mypy 错误: Argument 1 has incompatible type "str"; expected "list[str]"
    vectors = embeddings.embed_documents("single text")
```

---

## 5. 设计模式分析

### 5.1 接口隔离原则（ISP）

**定义**：客户端不应该被迫依赖它不使用的方法

**LangChain 的实践**：

```python
# ✅ 好的设计：只暴露必需的方法
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

# ❌ 不好的设计：暴露过多实现细节
class BadEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @abstractmethod
    def set_api_key(self, key: str): ...  # 配置细节

    @abstractmethod
    def set_model(self, model: str): ...  # 配置细节

    @abstractmethod
    def get_token_count(self, text: str) -> int: ...  # 实现细节

    @abstractmethod
    def validate_input(self, text: str) -> bool: ...  # 实现细节
```

**LangChain 的优势**：
- 接口简洁，易于实现
- 配置通过构造函数传递，不污染接口
- 实现细节由子类自行决定

### 5.2 适配器模式（Adapter Pattern）

异步方法的默认实现使用了适配器模式：

```python
async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    # 将同步接口适配为异步接口
    return await run_in_executor(None, self.embed_documents, texts)
```

**模式价值**：
- 向后兼容：旧代码无需修改
- 渐进式优化：可以先用默认实现，后续优化
- 降低实现门槛：不强制要求异步实现

### 5.3 模板方法模式（Template Method）

虽然 Embeddings 基类没有显式使用模板方法模式，但子类实现时常用此模式：

```python
class TemplateEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 模板方法：定义算法骨架
        validated_texts = self._validate_texts(texts)
        batches = self._create_batches(validated_texts)
        results = []
        for batch in batches:
            batch_vectors = self._embed_batch(batch)
            results.extend(batch_vectors)
        return results

    def _validate_texts(self, texts: list[str]) -> list[str]:
        # 钩子方法：子类可覆盖
        return [t.strip() for t in texts if t.strip()]

    def _create_batches(self, texts: list[str], batch_size: int = 100) -> list[list[str]]:
        # 钩子方法：子类可覆盖
        return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # 抽象方法：子类必须实现
        pass
```

---

## 6. 代码示例：实现自定义 Embeddings

### 6.1 最小实现

```python
from langchain_core.embeddings import Embeddings
import numpy as np

class SimpleEmbeddings(Embeddings):
    """最简单的 Embeddings 实现示例"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询"""
        return self._embed_single(text)

    def _embed_single(self, text: str) -> list[float]:
        """实际的嵌入逻辑（示例：使用哈希）"""
        # 注意：这只是示例，实际应使用真实的嵌入模型
        hash_value = hash(text)
        np.random.seed(hash_value % (2**32))
        vector = np.random.randn(self.dimension)
        # 归一化
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

# 使用示例
embeddings = SimpleEmbeddings(dimension=128)

# 嵌入文档
docs = ["Hello world", "LangChain is great", "RAG is powerful"]
doc_vectors = embeddings.embed_documents(docs)
print(f"嵌入了 {len(doc_vectors)} 个文档，每个向量 {len(doc_vectors[0])} 维")

# 嵌入查询
query = "What is LangChain?"
query_vector = embeddings.embed_query(query)
print(f"查询向量维度: {len(query_vector)}")
```

[来源: reference/source_embeddings_base_01.md]

### 6.2 带缓存的实现

```python
from langchain_core.embeddings import Embeddings
from functools import lru_cache
import hashlib

class CachedEmbeddings(Embeddings):
    """带缓存的 Embeddings 实现"""

    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入（使用缓存）"""
        return [self._cached_embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询（使用缓存）"""
        return self._cached_embed(text)

    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> list[float]:
        """缓存的嵌入方法"""
        # 使用基础 embeddings 进行嵌入
        return self.base_embeddings.embed_query(text)

# 使用示例
from langchain_openai import OpenAIEmbeddings

base = OpenAIEmbeddings()
cached = CachedEmbeddings(base)

# 第一次调用：实际请求 API
vector1 = cached.embed_query("Hello world")

# 第二次调用：从缓存读取（速度快 100 倍）
vector2 = cached.embed_query("Hello world")
```

### 6.3 异步优化实现

```python
from langchain_core.embeddings import Embeddings
import httpx
import asyncio

class AsyncOptimizedEmbeddings(Embeddings):
    """异步优化的 Embeddings 实现"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.Client()
        self.async_client = httpx.AsyncClient()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步批量嵌入"""
        response = self.client.post(
            f"{self.api_url}/embeddings",
            json={"texts": texts},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        """同步单个嵌入"""
        response = self.client.post(
            f"{self.api_url}/embeddings",
            json={"texts": [text]},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["embeddings"][0]

    # 覆盖异步方法以使用原生异步 API
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步批量嵌入（原生异步）"""
        response = await self.async_client.post(
            f"{self.api_url}/embeddings",
            json={"texts": texts},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["embeddings"]

    async def aembed_query(self, text: str) -> list[float]:
        """异步单个嵌入（原生异步）"""
        response = await self.async_client.post(
            f"{self.api_url}/embeddings",
            json={"texts": [text]},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["embeddings"][0]

# 使用示例
embeddings = AsyncOptimizedEmbeddings(
    api_url="https://api.example.com",
    api_key="your-api-key"
)

# 异步批量处理
async def process_queries():
    queries = [f"Query {i}" for i in range(10)]
    tasks = [embeddings.aembed_query(q) for q in queries]
    vectors = await asyncio.gather(*tasks)
    return vectors

# 运行异步任务
vectors = asyncio.run(process_queries())
```

---

## 7. 在 RAG 中的应用

### 7.1 RAG 索引构建

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 初始化 Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 加载文档
documents = [
    "LangChain is a framework for developing LLM applications.",
    "RAG combines retrieval and generation for better answers.",
    "Vector databases store high-dimensional embeddings."
]

# 3. 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.create_documents(documents)

# 4. 批量嵌入并存储
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,  # 使用 embed_documents() 方法
    persist_directory="./chroma_db"
)
```

[来源: reference/search_embedding_integration_01.md]

### 7.2 RAG 查询检索

```python
# 用户查询
user_query = "What is RAG in LangChain?"

# 嵌入查询（使用 embed_query() 方法）
query_vector = embeddings.embed_query(user_query)

# 在向量数据库中检索相似文档
similar_docs = vectorstore.similarity_search_by_vector(
    query_vector,
    k=5
)

# 打印检索结果
for i, doc in enumerate(similar_docs):
    print(f"{i+1}. {doc.page_content}")
```

### 7.3 完整 RAG 流程

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. 初始化组件
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4")

# 2. 加载向量数据库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# 4. 执行查询
query = "What is RAG in LangChain?"
answer = qa_chain.run(query)
print(answer)
```

**Embeddings 在 RAG 中的作用**：
1. **索引阶段**：使用 `embed_documents()` 批量嵌入文档
2. **查询阶段**：使用 `embed_query()` 嵌入用户查询
3. **检索阶段**：计算查询向量与文档向量的相似度
4. **生成阶段**：将检索到的文档作为上下文传递给 LLM

---

## 8. 总结

### 核心要点

1. **极简设计**：LangChain Embeddings 只定义 4 个方法（2 抽象 + 2 异步）
2. **文档与查询分离**：允许不同的优化策略和未来扩展
3. **异步支持**：默认实现 + 可覆盖优化
4. **类型安全**：明确的类型注解，支持静态检查
5. **设计模式**：接口隔离原则、适配器模式、模板方法模式

### 实现建议

- **最小实现**：只需实现 `embed_documents` 和 `embed_query`
- **性能优化**：覆盖异步方法以使用原生异步 API
- **缓存机制**：使用 `CacheBackedEmbeddings` 或自定义缓存
- **批量处理**：在 `embed_documents` 中使用批量 API

### 参考资料

**源码分析**：
- [reference/source_embeddings_base_01.md] - Embeddings 基类源码分析

**官方文档**：
- [reference/context7_langchain_openai_02.md] - LangChain OpenAI Embeddings 文档

**最佳实践**：
- [reference/search_embedding_integration_01.md] - 2025-2026 Embeddings 最佳实践

---

**文档版本**：v1.0
**最后更新**：2026-02-25
**作者**：Claude Code