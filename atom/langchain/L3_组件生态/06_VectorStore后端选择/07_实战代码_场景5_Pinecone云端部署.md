# 实战代码 - 场景5：Pinecone 云端部署

> **来源**：基于 `reference/context7_pinecone_02.md` 官方文档 + `reference/search_vectordb_production_01.md` 生产部署对比

## 场景说明

Pinecone 是一个完全托管的向量数据库服务，适合：
- 生产环境（完全托管）
- 大规模应用（无限扩展）
- 需要快速上线
- 多租户场景（命名空间）
- 需要元数据过滤

**核心特点**：
- 云托管：无需管理基础设施
- 自动扩展：根据负载自动扩展
- 高可用：内置高可用和备份
- Serverless：按使用量付费
- 多云支持：AWS、GCP、Azure

## 环境准备

```bash
# 安装依赖
pip install langchain-pinecone pinecone-client langchain-openai

# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

## 完整代码示例

### 示例1：基础云端部署

```python
"""
Pinecone 云端部署 - 基础示例
演示如何使用 Pinecone 进行云端部署
"""

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
import os

# 加载环境变量
load_dotenv()


def basic_cloud_deployment():
    """基础云端部署示例"""
    print("=" * 60)
    print("场景1：Pinecone 云端部署 - 基础示例")
    print("=" * 60)

    # 1. 初始化 Pinecone 客户端
    print("\n[步骤1] 初始化 Pinecone 客户端...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("✓ 客户端初始化完成")

    # 2. 创建或连接索引
    index_name = "langchain-demo"
    print(f"\n[步骤2] 创建或连接索引: {index_name}...")
    
    # 检查索引是否存在
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        # 创建 Serverless 索引
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small 维度
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✓ 索引 '{index_name}' 创建成功")
        
        # 等待索引准备就绪
        print("  等待索引准备就绪...")
        time.sleep(10)
    else:
        print(f"✓ 索引 '{index_name}' 已存在")

    # 3. 获取索引
    index = pc.Index(index_name)
    print(f"✓ 索引连接成功")
    
    # 查看索引统计
    stats = index.describe_index_stats()
    print(f"  索引统计: {stats}")

    # 4. 初始化 PineconeVectorStore
    print("\n[步骤3] 初始化 PineconeVectorStore...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    print("✓ 初始化完成")

    # 5. 添加文档
    print("\n[步骤4] 添加文档...")
    documents = [
        Document(
            page_content="Pinecone 是一个完全托管的向量数据库。",
            metadata={"source": "intro.txt", "category": "database"}
        ),
        Document(
            page_content="Pinecone 支持 Serverless 和 Pod-based 两种模式。",
            metadata={"source": "modes.txt", "category": "architecture"}
        ),
        Document(
            page_content="Pinecone 自动扩展，无需管理基础设施。",
            metadata={"source": "scaling.txt", "category": "features"}
        ),
    ]
    
    start_time = time.time()
    ids = vector_store.add_documents(documents)
    elapsed = time.time() - start_time
    
    print(f"✓ 添加了 {len(ids)} 个文档，耗时: {elapsed:.2f}秒")
    print(f"  文档 IDs: {ids}")

    # 6. 相似度检索
    print("\n[步骤5] 相似度检索...")
    query = "Pinecone 的特点是什么？"
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

    return vector_store, index


def namespace_isolation():
    """命名空间隔离示例"""
    print("\n" + "=" * 60)
    print("场景2：命名空间隔离（多租户）")
    print("=" * 60)

    # 初始化
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchain-demo"
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. 为不同租户创建命名空间
    print("\n[示例1] 为不同租户创建命名空间:")
    
    # 租户 A
    vector_store_a = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="tenant_a",
        text_key="text"
    )
    
    docs_a = [
        Document(page_content="租户 A 的文档 1"),
        Document(page_content="租户 A 的文档 2"),
    ]
    vector_store_a.add_documents(docs_a)
    print("  ✓ 租户 A 数据已添加到命名空间 'tenant_a'")

    # 租户 B
    vector_store_b = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="tenant_b",
        text_key="text"
    )
    
    docs_b = [
        Document(page_content="租户 B 的文档 1"),
        Document(page_content="租户 B 的文档 2"),
    ]
    vector_store_b.add_documents(docs_b)
    print("  ✓ 租户 B 数据已添加到命名空间 'tenant_b'")

    # 2. 隔离检索
    print("\n[示例2] 隔离检索:")
    
    query = "文档"
    
    # 只在租户 A 的命名空间中检索
    results_a = vector_store_a.similarity_search(query, k=5)
    print(f"  租户 A 检索结果 ({len(results_a)} 个):")
    for doc in results_a:
        print(f"    - {doc.page_content}")

    # 只在租户 B 的命名空间中检索
    results_b = vector_store_b.similarity_search(query, k=5)
    print(f"  租户 B 检索结果 ({len(results_b)} 个):")
    for doc in results_b:
        print(f"    - {doc.page_content}")


def metadata_filtering():
    """元数据过滤示例"""
    print("\n" + "=" * 60)
    print("场景3：元数据过滤")
    print("=" * 60)

    # 初始化
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchain-demo"
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="filtered",
        text_key="text"
    )

    # 添加带元数据的文档
    documents = [
        Document(
            page_content="Python 3.13 发布了新特性。",
            metadata={"language": "python", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="JavaScript ES2024 标准发布。",
            metadata={"language": "javascript", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="Python 教程：入门指南。",
            metadata={"language": "python", "year": 2023, "type": "tutorial"}
        ),
        Document(
            page_content="Go 语言并发编程。",
            metadata={"language": "go", "year": 2024, "type": "tutorial"}
        ),
    ]
    
    vector_store.add_documents(documents)

    # 1. 单条件过滤
    print("\n[示例1] 单条件过滤 (language=python):")
    results = vector_store.similarity_search(
        "编程语言",
        k=5,
        filter={"language": {"$eq": "python"}}
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")

    # 2. 多条件过滤（AND）
    print("\n[示例2] 多条件过滤 (language=python AND year=2024):")
    results = vector_store.similarity_search(
        "编程",
        k=5,
        filter={
            "$and": [
                {"language": {"$eq": "python"}},
                {"year": {"$eq": 2024}}
            ]
        }
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")

    # 3. 范围过滤
    print("\n[示例3] 范围过滤 (year >= 2024):")
    results = vector_store.similarity_search(
        "新特性",
        k=5,
        filter={"year": {"$gte": 2024}}
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")


def serverless_vs_pod():
    """Serverless vs Pod-based 对比"""
    print("\n" + "=" * 60)
    print("场景4：Serverless vs Pod-based 对比")
    print("=" * 60)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 1. Serverless 索引
    print("\n[模式1] Serverless 索引:")
    print("  特点：")
    print("    - 自动扩展")
    print("    - 按使用量付费")
    print("    - 适合不可预测负载")
    print("    - 配置简单")
    
    serverless_config = {
        "name": "serverless-index",
        "dimension": 1536,
        "metric": "cosine",
        "spec": ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    }
    print(f"  配置: {serverless_config}")

    # 2. Pod-based 索引
    print("\n[模式2] Pod-based 索引:")
    print("  特点：")
    print("    - 专用基础设施")
    print("    - 按资源预留付费")
    print("    - 适合可预测负载")
    print("    - 精细控制")
    
    from pinecone import PodSpec
    
    pod_config = {
        "name": "pod-index",
        "dimension": 1536,
        "metric": "cosine",
        "spec": PodSpec(
            environment="us-east1-gcp",
            pod_type="p1.x1",
            replicas=2,
            shards=1
        )
    }
    print(f"  配置: {pod_config}")

    # 3. 选择建议
    print("\n[选择建议]:")
    print("  Serverless:")
    print("    ✓ 开发和测试")
    print("    ✓ 不可预测的流量")
    print("    ✓ 快速上线")
    print("\n  Pod-based:")
    print("    ✓ 可预测的高流量")
    print("    ✓ 需要精细控制")
    print("    ✓ 成本优化（大规模）")


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("场景5：性能基准测试")
    print("=" * 60)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchain-demo"
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 测试不同数据规模
    sizes = [10, 50, 100]
    
    for size in sizes:
        print(f"\n[测试] 数据规模: {size} 个文档")
        
        namespace = f"benchmark_{size}"
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace,
            text_key="text"
        )
        
        # 生成数据
        documents = [
            Document(page_content=f"文档 {i}：测试内容。" * 10)
            for i in range(size)
        ]
        
        # 添加文档
        start_time = time.time()
        vector_store.add_documents(documents)
        build_time = time.time() - start_time
        
        # 等待索引更新
        time.sleep(2)
        
        # 检索测试
        query = "测试查询"
        search_times = []
        
        for _ in range(5):  # 多次测试取平均
            start_time = time.time()
            vector_store.similarity_search(query, k=10)
            search_times.append(time.time() - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        print(f"  添加时间: {build_time:.2f}秒")
        print(f"  平均检索时间: {avg_search_time*1000:.2f}ms")
        print(f"  吞吐量: {1/avg_search_time:.1f} 查询/秒")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Pinecone 云端部署 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    basic_cloud_deployment()
    namespace_isolation()
    metadata_filtering()
    serverless_vs_pod()
    performance_benchmark()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 运行说明

```bash
# 1. 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"

# 2. 运行示例
python 07_实战代码_场景5_Pinecone云端部署.py
```

## 最佳实践

### 1. 索引类型选择

**Serverless**（推荐）：
```python
spec = ServerlessSpec(
    cloud="aws",  # 或 "gcp", "azure"
    region="us-east-1"
)
```

**Pod-based**（高级）：
```python
spec = PodSpec(
    environment="us-east1-gcp",
    pod_type="p1.x1",
    replicas=2
)
```

### 2. 命名空间策略

```python
# 多租户隔离
tenant_a_store = PineconeVectorStore(
    index=index,
    namespace="tenant_a",
    ...
)

# 环境隔离
dev_store = PineconeVectorStore(
    index=index,
    namespace="dev",
    ...
)
```

### 3. 元数据设计

```python
# 好的元数据设计
metadata = {
    "source": "file.pdf",
    "page": 1,
    "category": "technical",
    "created_at": "2024-01-01"
}

# 支持的过滤运算符
filter = {
    "$and": [
        {"category": {"$eq": "technical"}},
        {"page": {"$gte": 1}}
    ]
}
```

### 4. 成本优化

```python
# 使用命名空间而非多个索引
# 好的做法
vector_store_a = PineconeVectorStore(namespace="a", ...)
vector_store_b = PineconeVectorStore(namespace="b", ...)

# 不好的做法（成本高）
# 为每个租户创建单独的索引
```

## 常见问题

### Q1: Pinecone 和 Qdrant 如何选择？

**Pinecone**：
- 完全托管，零运维
- 自动扩展
- 适合快速上线

**Qdrant**：
- 自托管，更灵活
- 成本更低（大规模）
- 适合需要控制的场景

### Q2: Pinecone 的成本如何？

- **Serverless**：按使用量付费
  - 免费额度：100K 向量
  - 超出后：按读写操作计费
- **Pod-based**：按资源预留付费
  - 固定月费
  - 适合可预测负载

### Q3: 如何优化 Pinecone 成本？

1. 使用命名空间而非多个索引
2. 选择合适的索引类型
3. 定期清理无用数据
4. 监控使用量

### Q4: Pinecone 支持多大规模的数据？

- Serverless：无限扩展
- Pod-based：取决于 pod 配置
- 实际案例：数十亿向量

### Q5: 如何备份 Pinecone 数据？

Pinecone 自动备份，无需手动操作。如需导出：

```python
# 导出所有向量
results = index.query(
    vector=[0] * 1536,
    top_k=10000,
    include_metadata=True
)
```

## 总结

Pinecone 是云端部署的首选向量数据库，适合：
- **快速上线**：完全托管，零运维
- **自动扩展**：无限扩展能力
- **多租户**：命名空间隔离

**核心优势**：
- 完全托管，无需运维
- 自动扩展，支持无限数据
- 高可用和备份
- 支持元数据过滤
- 支持命名空间隔离

**主要限制**：
- 付费服务（有免费额度）
- 依赖网络连接
- 数据存储在云端

**下一步**：
- 需要自托管 → 使用 Qdrant 或 Milvus
- 需要成本优化 → 使用开源方案
- 需要本地开发 → 使用 Chroma
