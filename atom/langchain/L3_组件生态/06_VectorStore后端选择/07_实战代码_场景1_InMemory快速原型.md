# 实战代码 - 场景1：InMemory 快速原型

> **来源**：基于 `reference/source_inmemory_02.md` 源码分析

## 场景说明

InMemoryVectorStore 是 LangChain 提供的最简单的向量存储实现，适合：
- 快速原型验证
- 单元测试
- 教学演示
- 小规模数据（< 1000 文档）

**核心特点**：
- 零配置：无需安装额外依赖或启动服务
- 纯内存：使用 Python 字典存储向量和文档
- 完整功能：支持所有 VectorStore 接口
- 不持久化：程序重启后数据丢失

## 环境准备

```bash
# 安装依赖
pip install langchain-core langchain-openai numpy

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 完整代码示例

### 示例1：基础快速原型

```python
"""
InMemory 快速原型 - 基础示例
演示如何使用 InMemoryVectorStore 快速验证 RAG 想法
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()


def basic_prototype():
    """基础快速原型示例"""
    print("=" * 60)
    print("场景1：InMemory 快速原型 - 基础示例")
    print("=" * 60)

    # 1. 初始化 InMemoryVectorStore
    # 优点：零配置，直接使用
    print("\n[步骤1] 初始化 InMemoryVectorStore...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    print("✓ 初始化完成（无需任何配置）")

    # 2. 准备测试数据
    print("\n[步骤2] 准备测试数据...")
    documents = [
        Document(
            id="doc1",
            page_content="Python 是一种高级编程语言，广泛用于数据科学和机器学习。",
            metadata={"source": "python_intro.txt", "category": "programming"}
        ),
        Document(
            id="doc2",
            page_content="LangChain 是一个用于构建 LLM 应用的框架，支持多种向量存储后端。",
            metadata={"source": "langchain_intro.txt", "category": "framework"}
        ),
        Document(
            id="doc3",
            page_content="向量数据库用于存储和检索高维向量，是 RAG 系统的核心组件。",
            metadata={"source": "vector_db.txt", "category": "database"}
        ),
        Document(
            id="doc4",
            page_content="OpenAI 提供了强大的 embedding 模型，可以将文本转换为向量。",
            metadata={"source": "openai.txt", "category": "ai"}
        ),
        Document(
            id="doc5",
            page_content="RAG（检索增强生成）结合了检索和生成，提高了 LLM 的准确性。",
            metadata={"source": "rag.txt", "category": "technique"}
        ),
    ]
    print(f"✓ 准备了 {len(documents)} 个测试文档")

    # 3. 添加文档到向量存储
    print("\n[步骤3] 添加文档到向量存储...")
    start_time = time.time()
    ids = vector_store.add_documents(documents)
    elapsed = time.time() - start_time
    print(f"✓ 添加完成，耗时: {elapsed:.2f}秒")
    print(f"  文档 IDs: {ids}")

    # 4. 基础相似度检索
    print("\n[步骤4] 基础相似度检索...")
    query = "如何使用 LangChain 构建应用？"
    print(f"  查询: {query}")

    start_time = time.time()
    results = vector_store.similarity_search(query, k=3)
    elapsed = time.time() - start_time

    print(f"✓ 检索完成，耗时: {elapsed:.3f}秒")
    print(f"  返回 {len(results)} 个结果:")
    for i, doc in enumerate(results, 1):
        print(f"\n  结果 {i}:")
        print(f"    内容: {doc.page_content}")
        print(f"    元数据: {doc.metadata}")

    # 5. 带分数的检索
    print("\n[步骤5] 带分数的相似度检索...")
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print("✓ 检索完成，结果（按相似度排序）:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n  结果 {i} (相似度: {score:.4f}):")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category', 'N/A')}")

    # 6. 元数据过滤检索
    print("\n[步骤6] 元数据过滤检索...")

    # 定义过滤函数
    def filter_by_category(doc: Document) -> bool:
        """只返回 category 为 'framework' 或 'technique' 的文档"""
        return doc.metadata.get("category") in ["framework", "technique"]

    filtered_results = vector_store.similarity_search(
        query,
        k=5,
        filter=filter_by_category
    )

    print(f"✓ 过滤检索完成，返回 {len(filtered_results)} 个结果:")
    for i, doc in enumerate(filtered_results, 1):
        print(f"\n  结果 {i}:")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category')}")

    # 7. 检查内部存储
    print("\n[步骤7] 检查内部存储结构...")
    print(f"  存储的文档数量: {len(vector_store.store)}")
    print(f"  存储的文档 IDs: {list(vector_store.store.keys())}")

    # 查看第一个文档的存储结构
    first_id = list(vector_store.store.keys())[0]
    first_doc = vector_store.store[first_id]
    print(f"\n  第一个文档的存储结构:")
    print(f"    ID: {first_doc['id']}")
    print(f"    文本: {first_doc['text'][:50]}...")
    print(f"    向量维度: {len(first_doc['vector'])}")
    print(f"    元数据: {first_doc['metadata']}")

    return vector_store


def mmr_search_example():
    """MMR（最大边际相关性）检索示例"""
    print("\n" + "=" * 60)
    print("场景2：MMR 多样性检索")
    print("=" * 60)

    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    # 准备相似的文档（测试 MMR 的多样性）
    documents = [
        Document(
            page_content="Python 是一种编程语言，广泛用于数据科学。",
            metadata={"topic": "python"}
        ),
        Document(
            page_content="Python 语言简单易学，适合初学者。",
            metadata={"topic": "python"}
        ),
        Document(
            page_content="Python 有丰富的第三方库，如 NumPy 和 Pandas。",
            metadata={"topic": "python"}
        ),
        Document(
            page_content="Java 是一种面向对象的编程语言。",
            metadata={"topic": "java"}
        ),
        Document(
            page_content="JavaScript 主要用于 Web 前端开发。",
            metadata={"topic": "javascript"}
        ),
        Document(
            page_content="Go 语言以并发性能著称。",
            metadata={"topic": "go"}
        ),
    ]

    vector_store.add_documents(documents)

    query = "Python 编程语言"

    # 1. 普通相似度检索（可能返回很多相似的结果）
    print("\n[对比1] 普通相似度检索 (k=4):")
    normal_results = vector_store.similarity_search(query, k=4)
    for i, doc in enumerate(normal_results, 1):
        print(f"  {i}. {doc.page_content} (主题: {doc.metadata['topic']})")

    # 2. MMR 检索（平衡相关性和多样性）
    print("\n[对比2] MMR 检索 (k=4, lambda_mult=0.5):")
    print("  lambda_mult=0.5 表示相关性和多样性各占 50%")
    mmr_results = vector_store.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=6,  # 先检索 6 个候选
        lambda_mult=0.5  # 平衡相关性和多样性
    )
    for i, doc in enumerate(mmr_results, 1):
        print(f"  {i}. {doc.page_content} (主题: {doc.metadata['topic']})")

    # 3. 调整 lambda_mult 参数
    print("\n[对比3] MMR 检索 (lambda_mult=0.8，更注重相关性):")
    mmr_results_high = vector_store.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=6,
        lambda_mult=0.8  # 更注重相关性
    )
    for i, doc in enumerate(mmr_results_high, 1):
        print(f"  {i}. {doc.page_content} (主题: {doc.metadata['topic']})")

    print("\n[对比4] MMR 检索 (lambda_mult=0.2，更注重多样性):")
    mmr_results_low = vector_store.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=6,
        lambda_mult=0.2  # 更注重多样性
    )
    for i, doc in enumerate(mmr_results_low, 1):
        print(f"  {i}. {doc.page_content} (主题: {doc.metadata['topic']})")


def retriever_interface_example():
    """作为 Retriever 使用的示例"""
    print("\n" + "=" * 60)
    print("场景3：作为 Retriever 使用")
    print("=" * 60)

    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    # 添加文档
    documents = [
        Document(page_content="LangChain 支持多种向量存储后端。"),
        Document(page_content="Retriever 是 LangChain 中的检索抽象。"),
        Document(page_content="LCEL 可以将 Retriever 串联到链中。"),
    ]
    vector_store.add_documents(documents)

    # 1. 转换为 Retriever（默认配置）
    print("\n[示例1] 默认 Retriever:")
    retriever = vector_store.as_retriever()
    results = retriever.invoke("LangChain 检索")
    print(f"  返回 {len(results)} 个结果")
    for doc in results:
        print(f"    - {doc.page_content}")

    # 2. 配置 Retriever（使用 MMR）
    print("\n[示例2] MMR Retriever:")
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 3, "lambda_mult": 0.5}
    )
    results = mmr_retriever.invoke("LangChain 检索")
    print(f"  返回 {len(results)} 个结果")
    for doc in results:
        print(f"    - {doc.page_content}")

    # 3. 配置 Retriever（使用分数阈值）
    print("\n[示例3] 分数阈值 Retriever:")
    threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8, "k": 5}
    )
    results = threshold_retriever.invoke("LangChain 检索")
    print(f"  返回 {len(results)} 个结果（相似度 > 0.8）")
    for doc in results:
        print(f"    - {doc.page_content}")


def crud_operations_example():
    """CRUD 操作示例"""
    print("\n" + "=" * 60)
    print("场景4：CRUD 操作")
    print("=" * 60)

    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    # 1. Create - 添加文档
    print("\n[操作1] 添加文档:")
    docs = [
        Document(id="1", page_content="文档1"),
        Document(id="2", page_content="文档2"),
        Document(id="3", page_content="文档3"),
    ]
    ids = vector_store.add_documents(docs)
    print(f"  添加了 {len(ids)} 个文档: {ids}")
    print(f"  当前文档数: {len(vector_store.store)}")

    # 2. Read - 读取文档
    print("\n[操作2] 读取文档:")
    retrieved_docs = vector_store.get_by_ids(["1", "3"])
    print(f"  读取了 {len(retrieved_docs)} 个文档:")
    for doc in retrieved_docs:
        print(f"    ID: {doc.id}, 内容: {doc.page_content}")

    # 3. Update - 更新文档（先删除再添加）
    print("\n[操作3] 更新文档:")
    print("  删除文档 ID: 2")
    vector_store.delete(["2"])
    print(f"  当前文档数: {len(vector_store.store)}")

    print("  添加新文档 ID: 2")
    new_doc = Document(id="2", page_content="文档2（已更新）")
    vector_store.add_documents([new_doc])
    print(f"  当前文档数: {len(vector_store.store)}")

    # 验证更新
    updated_doc = vector_store.get_by_ids(["2"])[0]
    print(f"  更新后的内容: {updated_doc.page_content}")

    # 4. Delete - 删除文档
    print("\n[操作4] 删除文档:")
    print("  删除文档 ID: 1, 3")
    vector_store.delete(["1", "3"])
    print(f"  当前文档数: {len(vector_store.store)}")
    print(f"  剩余文档 IDs: {list(vector_store.store.keys())}")


def performance_test():
    """性能测试示例"""
    print("\n" + "=" * 60)
    print("场景5：性能测试")
    print("=" * 60)

    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    # 1. 测试不同数据规模的添加性能
    print("\n[测试1] 添加性能测试:")
    for size in [10, 50, 100]:
        docs = [
            Document(page_content=f"测试文档 {i}，内容较长以模拟真实场景。" * 5)
            for i in range(size)
        ]

        start_time = time.time()
        vector_store.add_documents(docs)
        elapsed = time.time() - start_time

        print(f"  添加 {size} 个文档: {elapsed:.2f}秒 ({elapsed/size*1000:.1f}ms/doc)")

    # 2. 测试检索性能
    print(f"\n[测试2] 检索性能测试 (总文档数: {len(vector_store.store)}):")
    query = "测试查询"

    # 测试不同 k 值
    for k in [5, 10, 20]:
        start_time = time.time()
        results = vector_store.similarity_search(query, k=k)
        elapsed = time.time() - start_time
        print(f"  检索 top-{k}: {elapsed*1000:.1f}ms")

    # 3. 内存占用估算
    print("\n[测试3] 内存占用估算:")
    import sys

    total_size = 0
    for doc_id, doc_data in vector_store.store.items():
        # 估算每个文档的内存占用
        doc_size = (
            sys.getsizeof(doc_id) +
            sys.getsizeof(doc_data['text']) +
            sys.getsizeof(doc_data['vector']) +
            sys.getsizeof(doc_data['metadata'])
        )
        total_size += doc_size

    print(f"  总文档数: {len(vector_store.store)}")
    print(f"  估算内存占用: {total_size / 1024:.2f} KB")
    print(f"  平均每文档: {total_size / len(vector_store.store) / 1024:.2f} KB")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("InMemory 快速原型 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    basic_prototype()
    mmr_search_example()
    retriever_interface_example()
    crud_operations_example()
    performance_test()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 运行说明

```bash
# 1. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 2. 运行示例
python 07_实战代码_场景1_InMemory快速原型.py
```

## 最佳实践

### 1. 适用场景

**推荐使用**：
- 快速验证 RAG 想法
- 单元测试和集成测试
- 教学演示和原型开发
- 小规模数据（< 1000 文档）

**不推荐使用**：
- 生产环境
- 大规模数据（> 1000 文档）
- 需要持久化的场景
- 高并发场景

### 2. 性能优化

```python
# 批量添加文档（而非逐个添加）
# 好的做法
vector_store.add_documents(documents)

# 不好的做法
for doc in documents:
    vector_store.add_documents([doc])  # 每次都要计算 embedding
```

### 3. 内存管理

```python
# 定期清理不需要的文档
vector_store.delete(old_doc_ids)

# 或者重新创建 vector_store
vector_store = InMemoryVectorStore(embeddings)
```

### 4. 测试技巧

```python
import pytest
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

@pytest.fixture
def vector_store():
    """测试夹具：创建 InMemoryVectorStore"""
    embeddings = OpenAIEmbeddings()
    return InMemoryVectorStore(embeddings)

def test_add_and_search(vector_store):
    """测试添加和检索功能"""
    # 添加文档
    docs = [Document(page_content="测试文档")]
    vector_store.add_documents(docs)

    # 检索
    results = vector_store.similarity_search("测试", k=1)
    assert len(results) == 1
    assert "测试" in results[0].page_content
```

## 常见问题

### Q1: InMemory 和 Chroma 如何选择？

**InMemory**：
- 零配置，立即可用
- 不持久化，程序重启数据丢失
- 适合测试和快速原型

**Chroma**：
- 需要安装 chromadb
- 支持持久化到本地文件
- 适合本地开发和中小规模应用

### Q2: 如何提高 InMemory 的检索速度？

InMemory 使用 numpy 计算余弦相似度，性能已经很好。如果需要更高性能：
1. 减少文档数量
2. 使用更小的 embedding 维度
3. 考虑切换到 FAISS

### Q3: InMemory 支持异步操作吗？

支持，但异步方法会直接调用同步方法：

```python
# 异步添加
await vector_store.aadd_documents(documents)

# 异步检索
results = await vector_store.asimilarity_search(query)
```

### Q4: 如何查看 InMemory 的内部存储？

```python
# 查看所有文档 ID
print(list(vector_store.store.keys()))

# 查看第一个文档的完整数据
first_id = list(vector_store.store.keys())[0]
print(vector_store.store[first_id])
# 输出: {'id': '...', 'vector': [...], 'text': '...', 'metadata': {...}}
```

### Q5: InMemory 的内存占用如何？

粗略估算：
- 每个文档的 embedding（1536 维）：约 6KB
- 文本内容：取决于长度
- 元数据：通常很小

1000 个文档大约占用 10-20MB 内存。

## 总结

InMemoryVectorStore 是最简单的向量存储实现，适合：
- **快速验证**：零配置，立即可用
- **单元测试**：轻量级，易于测试
- **教学演示**：代码简单，易于理解

**核心优势**：
- 零配置，无需外部依赖
- 完整功能，支持所有 VectorStore 接口
- 代码简单，易于调试

**主要限制**：
- 不持久化，程序重启数据丢失
- 内存限制，只能处理小规模数据
- 性能较差，大规模数据检索慢

**下一步**：
- 需要持久化 → 使用 Chroma
- 需要高性能 → 使用 FAISS
- 需要生产部署 → 使用 Qdrant 或 Pinecone
