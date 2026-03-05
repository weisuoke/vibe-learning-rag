---
type: source_code_analysis
source: sourcecode/langchain/libs/partners/chroma/langchain_chroma/vectorstores.py
analyzed_files:
  - sourcecode/langchain/libs/partners/chroma/langchain_chroma/vectorstores.py
analyzed_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
---

# 源码分析：Chroma 集成实现

## 分析的文件

- `sourcecode/langchain/libs/partners/chroma/langchain_chroma/vectorstores.py` - Chroma 向量存储集成

## 关键发现

### 1. Chroma 特点

Chroma 是一个开源的向量数据库，专为 AI 应用设计：
- **本地优先**：支持本地文件系统存储
- **易于使用**：零配置启动
- **持久化**：支持数据持久化
- **适合场景**：本地开发、原型验证、中小规模应用

### 2. 核心依赖

```python
import chromadb
from chromadb import Search, Settings
from chromadb.api import CreateCollectionConfiguration
from langchain_core.vectorstores import VectorStore
```

Chroma 集成依赖：
- `chromadb` - Chroma 官方客户端
- `langchain_core` - LangChain 核心抽象

### 3. 辅助函数

#### 结果转换函数

```python
def _results_to_docs(results: Any) -> list[Document]:
    """将 Chroma 查询结果转换为 Document 列表"""
    return [doc for doc, _ in _results_to_docs_and_scores(results)]

def _results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
    """将 Chroma 查询结果转换为 (Document, score) 列表"""
    return [
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2]),
            result[3],  # distance score
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results["distances"][0],
            strict=False,
        )
        if result[0] is not None  # 过滤掉 None 内容
    ]

def _results_to_docs_and_vectors(results: Any) -> list[tuple[Document, np.ndarray]]:
    """将 Chroma 查询结果转换为 (Document, vector) 列表"""
    return [
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[3]),
            result[2],  # embedding vector
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["embeddings"][0],
            results["ids"][0],
            strict=False,
        )
        if result[0] is not None
    ]
```

#### 相似度计算

```python
def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """计算两个矩阵之间的余弦相似度

    Args:
        X: 第一个矩阵 (n_samples_X, n_features)
        Y: 第二个矩阵 (n_samples_Y, n_features)

    Returns:
        相似度矩阵 (n_samples_X, n_samples_Y)
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)

    # 检查维度匹配
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. "
            f"X has shape {X.shape} and Y has shape {Y.shape}."
        )

    # 计算范数
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)

    # 计算余弦相似度
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)

    # 处理 NaN 和 Inf
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0

    return similarity
```

#### MMR 算法

```python
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """计算最大边际相关性（MMR）

    MMR 平衡相关性和多样性：
    - lambda_mult = 1: 只考虑相关性（退化为普通检索）
    - lambda_mult = 0: 只考虑多样性（最大化差异）
    - lambda_mult = 0.5: 平衡相关性和多样性

    Args:
        query_embedding: 查询向量
        embedding_list: 候选向量列表
        lambda_mult: 相关性权重 (0-1)
        k: 返回结果数量

    Returns:
        选中的向量索引列表
    """
    if min(k, len(embedding_list)) <= 0:
        return []

    # 确保 query_embedding 是 2D
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # 计算与查询的相似度
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]

    # 选择最相似的作为第一个结果
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])

    # 迭代选择剩余结果
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1

        # 计算与已选择向量的相似度
        similarity_to_selected = cosine_similarity(embedding_list, selected)

        # 对每个候选向量计算 MMR 分数
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue

            # MMR 公式：λ * sim(q, d) - (1-λ) * max(sim(d, s))
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )

            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i

        idxs.append(idx_to_add)
        selected = np.vstack([selected, embedding_list[idx_to_add]])

    return idxs
```

## 代码片段

### Chroma 类初始化

```python
class Chroma(VectorStore):
    """Chroma vector store integration.

    Setup:
        Install langchain-chroma:

        ```bash
        pip install -U langchain-chroma
        ```

    Key init args:
        collection_name: str
            Name of the collection to use
        embedding_function: Embeddings
            Embedding function to use
        persist_directory: str | None
            Directory to persist the collection
        client_settings: Settings | None
            Chroma client settings
    """

    def __init__(
        self,
        collection_name: str = "langchain",
        embedding_function: Embeddings | None = None,
        persist_directory: str | None = None,
        client_settings: Settings | None = None,
        collection_metadata: dict | None = None,
        client: chromadb.Client | None = None,
    ):
        # 初始化 Chroma 客户端
        if client is not None:
            self._client = client
        else:
            if client_settings:
                self._client = chromadb.Client(client_settings)
            else:
                self._client = chromadb.Client()

        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata,
        )

        self._embedding_function = embedding_function
```

## 架构设计要点

1. **辅助函数分离**：将结果转换、相似度计算等逻辑提取为独立函数
2. **MMR 支持**：内置 MMR 算法，支持多样性检索
3. **灵活初始化**：支持多种初始化方式（客户端、设置、持久化目录）
4. **类型安全**：使用类型注解，便于 IDE 提示

## 优缺点分析

### 优点
1. **零配置启动**：无需额外服务，直接使用
2. **持久化支持**：数据可保存到本地文件系统
3. **完整功能**：支持 MMR、过滤、元数据等高级功能
4. **开发友好**：适合本地开发和原型验证

### 缺点
1. **性能限制**：大规模数据性能不如专业向量数据库
2. **单机限制**：不支持分布式部署
3. **并发限制**：高并发场景性能受限

## 适用场景

- ✅ 本地开发和测试
- ✅ 原型验证
- ✅ 中小规模应用（< 100K 文档）
- ✅ 需要持久化的场景
- ❌ 大规模生产环境
- ❌ 高并发场景
- ❌ 分布式部署

## 与其他后端的对比

| 特性 | Chroma | InMemory | FAISS | Pinecone |
|------|--------|----------|-------|----------|
| 持久化 | ✅ | ❌ | ✅ | ✅ |
| 部署复杂度 | 低 | 极低 | 低 | 中 |
| 性能 | 中 | 低 | 高 | 高 |
| 适合规模 | < 100K | < 1K | < 1M | 无限 |
| MMR 支持 | ✅ | ✅ | ❌ | ✅ |
| 成本 | 免费 | 免费 | 免费 | 付费 |
