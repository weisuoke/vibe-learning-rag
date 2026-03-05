# 实战代码 - 场景3：FAISS 高性能检索

> **来源**：基于 `reference/source_faiss_pinecone_milvus_05.md` 源码分析 + `reference/search_vectordb_production_01.md` 性能对比

## 场景说明

FAISS (Facebook AI Similarity Search) 是 Meta 开发的高性能向量检索库，适合：
- 本地高性能检索
- 静态数据集
- 大规模数据（< 1M 文档）
- 需要极致性能的场景

**核心特点**：
- 高性能：C++ 实现，性能极佳
- 本地运行：无需外部服务
- 丰富算法：支持多种索引类型（Flat, IVF, HNSW）
- 可持久化：支持保存和加载索引

## 环境准备

```bash
# 安装依赖
pip install langchain-community faiss-cpu langchain-openai numpy

# 如果有 GPU，可以安装 GPU 版本
# pip install faiss-gpu

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 完整代码示例

### 示例1：基础 FAISS 使用

```python
"""
FAISS 高性能检索 - 基础示例
演示如何使用 FAISS 进行高性能向量检索
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import time
import os

# 加载环境变量
load_dotenv()


def basic_faiss_usage():
    """基础 FAISS 使用示例"""
    print("=" * 60)
    print("场景1：FAISS 高性能检索 - 基础示例")
    print("=" * 60)

    # 1. 初始化 Embeddings
    print("\n[步骤1] 初始化 Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✓ 初始化完成")

    # 2. 准备测试数据
    print("\n[步骤2] 准备测试数据...")
    documents = [
        Document(
            page_content="FAISS 是 Facebook 开发的高性能向量检索库。",
            metadata={"source": "faiss_intro.txt", "category": "library"}
        ),
        Document(
            page_content="FAISS 支持多种索引类型，包括 Flat、IVF 和 HNSW。",
            metadata={"source": "faiss_index.txt", "category": "technical"}
        ),
        Document(
            page_content="FAISS 使用 C++ 实现，性能极佳。",
            metadata={"source": "faiss_perf.txt", "category": "performance"}
        ),
        Document(
            page_content="FAISS 适合静态数据集的高性能检索。",
            metadata={"source": "faiss_use.txt", "category": "usage"}
        ),
    ]
    print(f"✓ 准备了 {len(documents)} 个测试文档")

    # 3. 创建 FAISS 索引
    print("\n[步骤3] 创建 FAISS 索引...")
    start_time = time.time()
    
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    elapsed = time.time() - start_time
    print(f"✓ 索引创建完成，耗时: {elapsed:.2f}秒")

    # 4. 相似度检索
    print("\n[步骤4] 相似度检索...")
    query = "FAISS 的性能如何？"
    print(f"  查询: {query}")

    start_time = time.time()
    results = vector_store.similarity_search(query, k=3)
    elapsed = time.time() - start_time

    print(f"✓ 检索完成，耗时: {elapsed*1000:.2f}ms")
    print(f"  返回 {len(results)} 个结果:")
    for i, doc in enumerate(results, 1):
        print(f"\n  结果 {i}:")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category')}")

    # 5. 带分数的检索
    print("\n[步骤5] 带分数的相似度检索...")
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print("✓ 检索完成，结果（按相似度排序）:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n  结果 {i} (距离: {score:.4f}):")
        print(f"    内容: {doc.page_content}")
        print(f"    分类: {doc.metadata.get('category')}")

    return vector_store


def save_and_load_index():
    """保存和加载 FAISS 索引"""
    print("\n" + "=" * 60)
    print("场景2：保存和加载 FAISS 索引")
    print("=" * 60)

    # 1. 创建索引
    print("\n[步骤1] 创建 FAISS 索引...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    documents = [
        Document(page_content=f"文档 {i}：这是测试内容。" * 10)
        for i in range(100)
    ]
    
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"✓ 创建了包含 {len(documents)} 个文档的索引")

    # 2. 保存索引到本地
    print("\n[步骤2] 保存索引到本地...")
    index_path = "./faiss_index"
    
    start_time = time.time()
    vector_store.save_local(index_path)
    elapsed = time.time() - start_time
    
    print(f"✓ 索引保存完成，耗时: {elapsed:.2f}秒")
    print(f"  保存路径: {index_path}")
    
    # 检查文件
    if os.path.exists(index_path):
        files = os.listdir(index_path)
        print(f"  索引文件: {files}")

    # 3. 从本地加载索引
    print("\n[步骤3] 从本地加载索引...")
    
    start_time = time.time()
    loaded_vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # 注意：仅在信任索引文件时使用
    )
    elapsed = time.time() - start_time
    
    print(f"✓ 索引加载完成，耗时: {elapsed:.2f}秒")

    # 4. 验证加载的索引
    print("\n[步骤4] 验证加载的索引...")
    query = "测试查询"
    results = loaded_vector_store.similarity_search(query, k=3)
    
    print(f"  查询: {query}")
    print(f"  返回 {len(results)} 个结果")
    for i, doc in enumerate(results, 1):
        print(f"    {i}. {doc.page_content[:50]}...")

    return loaded_vector_store


def index_types_comparison():
    """不同索引类型对比"""
    print("\n" + "=" * 60)
    print("场景3：FAISS 索引类型对比")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 准备较大的数据集
    print("\n[准备] 生成测试数据集...")
    documents = [
        Document(page_content=f"文档 {i}：内容较长以模拟真实场景。" * 20)
        for i in range(500)
    ]
    print(f"✓ 生成了 {len(documents)} 个文档")

    # 1. Flat 索引（精确检索）
    print("\n[索引1] Flat 索引（精确检索）:")
    print("  特点：暴力搜索，100% 准确，适合小规模数据")
    
    start_time = time.time()
    flat_store = FAISS.from_documents(documents, embeddings)
    build_time = time.time() - start_time
    
    query = "测试查询"
    start_time = time.time()
    results = flat_store.similarity_search(query, k=10)
    search_time = time.time() - start_time
    
    print(f"  构建时间: {build_time:.2f}秒")
    print(f"  检索时间: {search_time*1000:.2f}ms")
    print(f"  返回结果: {len(results)} 个")

    # 注意：IVF 和 HNSW 索引需要更复杂的配置
    # 这里仅展示 Flat 索引的使用
    # 在实际应用中，可以使用 faiss 库直接创建不同类型的索引

    print("\n[说明] IVF 和 HNSW 索引:")
    print("  IVF (倒排索引):")
    print("    - 聚类加速，可调节准确率")
    print("    - 适合中等规模数据（10K - 1M）")
    print("    - 需要训练阶段")
    print("\n  HNSW (层次图):")
    print("    - 高性能近似检索")
    print("    - 适合大规模数据（> 100K）")
    print("    - 构建时间较长，检索极快")


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("场景4：FAISS 性能基准测试")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 测试不同数据规模
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n[测试] 数据规模: {size} 个文档")
        
        # 生成数据
        documents = [
            Document(page_content=f"文档 {i}：测试内容。" * 10)
            for i in range(size)
        ]
        
        # 构建索引
        start_time = time.time()
        vector_store = FAISS.from_documents(documents, embeddings)
        build_time = time.time() - start_time
        
        # 检索测试
        query = "测试查询"
        search_times = []
        
        for _ in range(10):  # 多次测试取平均
            start_time = time.time()
            vector_store.similarity_search(query, k=10)
            search_times.append(time.time() - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        print(f"  构建时间: {build_time:.2f}秒")
        print(f"  平均检索时间: {avg_search_time*1000:.2f}ms")
        print(f"  吞吐量: {1/avg_search_time:.1f} 查询/秒")


def merge_indexes():
    """合并多个 FAISS 索引"""
    print("\n" + "=" * 60)
    print("场景5：合并多个 FAISS 索引")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. 创建第一个索引
    print("\n[步骤1] 创建第一个索引...")
    docs1 = [
        Document(page_content=f"索引1 - 文档 {i}")
        for i in range(50)
    ]
    store1 = FAISS.from_documents(docs1, embeddings)
    print(f"✓ 索引1 包含 {len(docs1)} 个文档")

    # 2. 创建第二个索引
    print("\n[步骤2] 创建第二个索引...")
    docs2 = [
        Document(page_content=f"索引2 - 文档 {i}")
        for i in range(50)
    ]
    store2 = FAISS.from_documents(docs2, embeddings)
    print(f"✓ 索引2 包含 {len(docs2)} 个文档")

    # 3. 合并索引
    print("\n[步骤3] 合并索引...")
    store1.merge_from(store2)
    print(f"✓ 合并完成")

    # 4. 验证合并结果
    print("\n[步骤4] 验证合并结果...")
    query = "文档"
    results = store1.similarity_search(query, k=10)
    
    print(f"  查询: {query}")
    print(f"  返回 {len(results)} 个结果:")
    for i, doc in enumerate(results[:5], 1):
        print(f"    {i}. {doc.page_content}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("FAISS 高性能检索 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    basic_faiss_usage()
    save_and_load_index()
    index_types_comparison()
    performance_benchmark()
    merge_indexes()

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
python 07_实战代码_场景3_FAISS高性能检索.py
```

## 最佳实践

### 1. 索引类型选择

**Flat 索引**：
- 适合：< 10K 文档
- 优点：100% 准确
- 缺点：检索慢

**IVF 索引**：
- 适合：10K - 1M 文档
- 优点：速度快，可调准确率
- 缺点：需要训练

**HNSW 索引**：
- 适合：> 100K 文档
- 优点：检索极快
- 缺点：构建时间长

### 2. 持久化策略

```python
# 定期保存索引
vector_store.save_local("./faiss_index")

# 加载时注意安全性
vector_store = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # 仅信任的文件
)
```

### 3. 性能优化

```python
# 批量添加文档
vector_store.add_documents(documents)  # 好

# 避免逐个添加
for doc in documents:
    vector_store.add_documents([doc])  # 不好
```

## 常见问题

### Q1: FAISS 和 Chroma 如何选择？

**FAISS**：
- 性能极佳（C++ 实现）
- 不支持增量更新
- 适合静态数据

**Chroma**：
- 支持增量更新
- 支持元数据过滤
- 适合动态数据

### Q2: FAISS 支持元数据过滤吗？

不支持。FAISS 只存储向量，元数据需要单独管理。

### Q3: 如何更新 FAISS 索引？

FAISS 不支持增量更新，需要重建索引：

```python
# 1. 加载旧索引
old_store = FAISS.load_local(...)

# 2. 添加新文档
new_docs = [...]
old_store.add_documents(new_docs)

# 3. 保存新索引
old_store.save_local(...)
```

### Q4: FAISS 的内存占用如何？

粗略估算：
- 每个向量（1536 维）：约 6KB
- 1000 个文档：约 6MB
- 100K 个文档：约 600MB

### Q5: FAISS 支持 GPU 加速吗？

支持，安装 GPU 版本：

```bash
pip install faiss-gpu
```

## 总结

FAISS 是高性能向量检索的首选，适合：
- **高性能需求**：C++ 实现，性能极佳
- **静态数据**：不支持增量更新
- **本地部署**：无需外部服务

**核心优势**：
- 性能极佳（C++ 实现）
- 本地运行，无外部依赖
- 支持多种索引算法
- 可持久化到本地文件

**主要限制**：
- 不支持增量更新（需要重建索引）
- 不支持元数据过滤
- 不支持分布式

**下一步**：
- 需要增量更新 → 使用 Chroma 或 Qdrant
- 需要生产部署 → 使用 Qdrant 或 Pinecone
- 需要企业级 → 使用 Milvus
