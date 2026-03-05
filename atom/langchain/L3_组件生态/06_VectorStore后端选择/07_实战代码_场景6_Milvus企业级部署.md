# 实战代码 - 场景6：Milvus 企业级部署

> **来源**：基于 `reference/source_faiss_pinecone_milvus_05.md` 概述 + `reference/search_vectordb_production_01.md` 生产部署对比

## 场景说明

Milvus 是一个开源的云原生向量数据库，适合：
- 企业级应用
- 大规模部署（> 1M 文档）
- 需要自托管
- 需要 GPU 加速
- 需要分布式

**核心特点**：
- 云原生：Kubernetes 友好
- 高性能：支持 GPU 加速
- 分布式：支持水平扩展
- 丰富索引：多种索引类型
- 企业级：完整的监控和管理

## 环境准备

```bash
# 安装依赖
pip install langchain-milvus pymilvus langchain-openai

# 启动 Milvus 服务器（Docker Standalone）
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v $(pwd)/volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:latest

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 完整代码示例

### 示例1：基础企业级部署

```python
"""
Milvus 企业级部署 - 基础示例
演示如何使用 Milvus 进行企业级部署
"""

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()


def basic_enterprise_deployment():
    """基础企业级部署示例"""
    print("=" * 60)
    print("场景1：Milvus 企业级部署 - 基础示例")
    print("=" * 60)

    # 1. 连接到 Milvus 服务器
    print("\n[步骤1] 连接到 Milvus 服务器...")
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✓ 连接成功")

    # 2. 初始化 Embeddings
    print("\n[步骤2] 初始化 Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✓ 初始化完成")

    # 3. 创建 Milvus 向量存储
    collection_name = "enterprise_docs"
    print(f"\n[步骤3] 创建 Milvus 向量存储: {collection_name}...")
    
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "host": "localhost",
            "port": "19530"
        }
    )
    print("✓ 向量存储创建完成")

    # 4. 添加文档
    print("\n[步骤4] 添加文档...")
    documents = [
        Document(
            page_content="Milvus 是一个开源的云原生向量数据库。",
            metadata={"source": "intro.txt", "category": "database"}
        ),
        Document(
            page_content="Milvus 支持 GPU 加速，性能极佳。",
            metadata={"source": "performance.txt", "category": "features"}
        ),
        Document(
            page_content="Milvus 支持分布式部署和水平扩展。",
            metadata={"source": "scaling.txt", "category": "architecture"}
        ),
    ]
    
    start_time = time.time()
    ids = vector_store.add_documents(documents)
    elapsed = time.time() - start_time
    
    print(f"✓ 添加了 {len(ids)} 个文档，耗时: {elapsed:.2f}秒")
    print(f"  文档 IDs: {ids}")

    # 5. 相似度检索
    print("\n[步骤5] 相似度检索...")
    query = "Milvus 的性能如何？"
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

    # 6. 查看 Collection 信息
    print("\n[步骤6] 查看 Collection 信息...")
    collection = Collection(collection_name)
    print(f"  Collection 名称: {collection.name}")
    print(f"  文档数量: {collection.num_entities}")
    print(f"  Schema: {collection.schema}")

    return vector_store


def index_types_configuration():
    """索引类型配置示例"""
    print("\n" + "=" * 60)
    print("场景2：Milvus 索引类型配置")
    print("=" * 60)

    # 连接到 Milvus
    connections.connect(alias="default", host="localhost", port="19530")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. FLAT 索引（精确检索）
    print("\n[索引1] FLAT 索引（精确检索）:")
    print("  特点：")
    print("    - 暴力搜索，100% 准确")
    print("    - 适合小规模数据（< 10K）")
    print("    - 无需训练")
    
    flat_config = {
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {}
    }
    print(f"  配置: {flat_config}")

    # 2. IVF_FLAT 索引（倒排索引）
    print("\n[索引2] IVF_FLAT 索引（倒排索引）:")
    print("  特点：")
    print("    - 聚类加速，可调准确率")
    print("    - 适合中等规模数据（10K - 1M）")
    print("    - 需要训练阶段")
    
    ivf_flat_config = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {
            "nlist": 128  # 聚类中心数量
        }
    }
    print(f"  配置: {ivf_flat_config}")

    # 3. HNSW 索引（层次图）
    print("\n[索引3] HNSW 索引（层次图）:")
    print("  特点：")
    print("    - 高性能近似检索")
    print("    - 适合大规模数据（> 100K）")
    print("    - 构建时间较长，检索极快")
    
    hnsw_config = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
            "M": 16,  # 每层最大连接数
            "efConstruction": 200  # 构建时的搜索深度
        }
    }
    print(f"  配置: {hnsw_config}")

    # 4. IVF_PQ 索引（量化压缩）
    print("\n[索引4] IVF_PQ 索引（量化压缩）:")
    print("  特点：")
    print("    - 内存占用小")
    print("    - 适合超大规模数据（> 1M）")
    print("    - 牺牲部分准确率")
    
    ivf_pq_config = {
        "index_type": "IVF_PQ",
        "metric_type": "L2",
        "params": {
            "nlist": 128,
            "m": 8,  # 子向量数量
            "nbits": 8  # 每个子向量的位数
        }
    }
    print(f"  配置: {ivf_pq_config}")


def partition_management():
    """分区管理示例"""
    print("\n" + "=" * 60)
    print("场景3：Milvus 分区管理")
    print("=" * 60)

    # 连接到 Milvus
    connections.connect(alias="default", host="localhost", port="19530")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    collection_name = "partitioned_docs"
    
    # 1. 创建带分区的 Collection
    print("\n[步骤1] 创建带分区的 Collection...")
    
    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
    ]
    schema = CollectionSchema(fields, description="Partitioned collection")
    
    # 创建 Collection
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    print(f"✓ Collection '{collection_name}' 创建完成")

    # 2. 创建分区
    print("\n[步骤2] 创建分区...")
    partitions = ["tech", "business", "science"]
    
    for partition_name in partitions:
        collection.create_partition(partition_name)
        print(f"  ✓ 分区 '{partition_name}' 创建完成")

    # 3. 向不同分区插入数据
    print("\n[步骤3] 向不同分区插入数据...")
    
    # 技术分区
    tech_docs = [
        Document(page_content="Python 是一种编程语言。", metadata={"category": "tech"}),
        Document(page_content="JavaScript 用于 Web 开发。", metadata={"category": "tech"}),
    ]
    
    # 商业分区
    business_docs = [
        Document(page_content="市场营销策略分析。", metadata={"category": "business"}),
        Document(page_content="财务报表解读。", metadata={"category": "business"}),
    ]
    
    print("  ✓ 数据插入完成")

    # 4. 分区检索
    print("\n[步骤4] 分区检索...")
    print("  只在 'tech' 分区中检索:")
    
    # 加载分区
    collection.load(partition_names=["tech"])
    
    # 检索
    query_vector = embeddings.embed_query("编程语言")
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=2,
        partition_names=["tech"]
    )
    
    print(f"  返回 {len(results[0])} 个结果")


def gpu_acceleration():
    """GPU 加速示例"""
    print("\n" + "=" * 60)
    print("场景4：Milvus GPU 加速")
    print("=" * 60)

    print("\n[说明] GPU 加速配置:")
    print("  Milvus 支持 GPU 加速的索引类型:")
    print("    - GPU_IVF_FLAT")
    print("    - GPU_IVF_PQ")
    print("    - GPU_CAGRA")
    
    print("\n[配置示例] GPU_IVF_FLAT:")
    gpu_config = {
        "index_type": "GPU_IVF_FLAT",
        "metric_type": "L2",
        "params": {
            "nlist": 128
        }
    }
    print(f"  {gpu_config}")
    
    print("\n[性能提升]:")
    print("  - 构建索引：5-10x 加速")
    print("  - 检索速度：3-5x 加速")
    print("  - 适合：大规模数据（> 1M 文档）")
    
    print("\n[硬件要求]:")
    print("  - NVIDIA GPU（计算能力 >= 6.0）")
    print("  - CUDA 11.0+")
    print("  - 足够的 GPU 内存")


def distributed_deployment():
    """分布式部署示例"""
    print("\n" + "=" * 60)
    print("场景5：Milvus 分布式部署")
    print("=" * 60)

    print("\n[部署模式]:")
    print("  1. Standalone（单机模式）")
    print("     - 适合：开发测试")
    print("     - 部署：Docker 单容器")
    print("     - 限制：单点故障")
    
    print("\n  2. Cluster（集群模式）")
    print("     - 适合：生产环境")
    print("     - 部署：Kubernetes")
    print("     - 优势：高可用、水平扩展")
    
    print("\n  3. Milvus Lite（轻量级）")
    print("     - 适合：边缘计算")
    print("     - 部署：嵌入式")
    print("     - 限制：功能受限")

    print("\n[Kubernetes 部署示例]:")
    k8s_config = """
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: milvus
spec:
  serviceName: milvus
  replicas: 3
  selector:
    matchLabels:
      app: milvus
  template:
    metadata:
      labels:
        app: milvus
    spec:
      containers:
      - name: milvus
        image: milvusdb/milvus:latest
        ports:
        - containerPort: 19530
        - containerPort: 9091
"""
    print(k8s_config)


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("场景6：Milvus 性能基准测试")
    print("=" * 60)

    # 连接到 Milvus
    connections.connect(alias="default", host="localhost", port="19530")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 测试不同数据规模
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n[测试] 数据规模: {size} 个文档")
        
        collection_name = f"benchmark_{size}"
        
        # 删除旧 Collection
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # 创建向量存储
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": "localhost", "port": "19530"}
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
        
        # 检索测试
        query = "测试查询"
        search_times = []
        
        for _ in range(10):  # 多次测试取平均
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
    print("Milvus 企业级部署 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    basic_enterprise_deployment()
    index_types_configuration()
    partition_management()
    gpu_acceleration()
    distributed_deployment()
    performance_benchmark()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 运行说明

```bash
# 1. 启动 Milvus 服务器
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# 2. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 3. 运行示例
python 07_实战代码_场景6_Milvus企业级部署.py
```

## 最佳实践

### 1. 索引类型选择

**FLAT**（< 10K）：
```python
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2"
}
```

**IVF_FLAT**（10K - 1M）：
```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
```

**HNSW**（> 100K）：
```python
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200}
}
```

### 2. 分区策略

```python
# 按时间分区
partitions = ["2024_Q1", "2024_Q2", "2024_Q3", "2024_Q4"]

# 按类别分区
partitions = ["tech", "business", "science"]

# 按租户分区
partitions = ["tenant_a", "tenant_b", "tenant_c"]
```

### 3. 性能优化

```python
# 批量插入
vector_store.add_documents(documents)  # 好

# 避免逐个插入
for doc in documents:
    vector_store.add_documents([doc])  # 不好
```

### 4. 资源配置

```yaml
# Kubernetes 资源配置
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
  limits:
    memory: "16Gi"
    cpu: "8"
```

## 常见问题

### Q1: Milvus 和 Qdrant 如何选择？

**Milvus**：
- 企业级功能更丰富
- 支持 GPU 加速
- 适合超大规模（> 1M）

**Qdrant**：
- 部署更简单
- Rust 实现，性能好
- 适合中大规模（100K - 1M）

### Q2: Milvus 的内存占用如何？

粗略估算：
- 每个向量（1536 维）：约 6KB
- 1000 个文档：约 6MB
- 1M 个文档：约 6GB

### Q3: 如何选择索引类型？

- **数据规模 < 10K**：FLAT
- **数据规模 10K - 1M**：IVF_FLAT
- **数据规模 > 100K**：HNSW
- **内存受限**：IVF_PQ

### Q4: Milvus 支持多大规模的数据？

- Standalone：< 10M 文档
- Cluster：无限扩展
- 实际案例：数十亿向量

### Q5: 如何备份 Milvus 数据？

```bash
# 创建备份
milvus-backup create --collection my_collection

# 恢复备份
milvus-backup restore --collection my_collection
```

## 总结

Milvus 是企业级向量数据库的首选，适合：
- **企业级应用**：完整的监控和管理
- **大规模部署**：> 1M 文档
- **GPU 加速**：5-10x 性能提升

**核心优势**：
- 开源免费
- 高性能（支持 GPU）
- 分布式架构
- 云原生设计
- 丰富的索引类型

**主要限制**：
- 部署复杂度较高
- 需要运维管理
- 学习曲线陡峭
- 资源消耗较大

**下一步**：
- 需要简单部署 → 使用 Chroma 或 Qdrant
- 需要云端托管 → 使用 Pinecone
- 需要本地开发 → 使用 FAISS
