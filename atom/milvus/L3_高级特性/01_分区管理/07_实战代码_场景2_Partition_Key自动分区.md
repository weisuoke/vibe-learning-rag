# 实战代码 - 场景2：Partition Key 自动分区

> **知识点**：分区管理 - Partition Key 自动分区
> **难度**：⭐⭐⭐☆☆
> **代码行数**：约 200 行
> **适用场景**：大规模多租户、自动数据路由、百万级租户支持

---

## 场景描述

在大规模 SaaS 应用中，我们经常需要支持数百万租户，每个租户的数据需要逻辑隔离。传统的手动分区管理方式有以下限制：
- 最多支持 1,024 个分区
- 需要手动创建和管理分区
- 插入和搜索时需要指定分区名称

**Partition Key（Milvus 2.6 核心特性）** 解决了这些问题：
- 自动根据字段值进行分区
- 支持数百万租户
- 无需手动管理分区
- 自动数据路由

本场景演示如何使用 Partition Key 实现自动分区管理。

---

## 核心概念

### 1. Partition Key 工作原理

```
用户插入数据 → Milvus 读取 Partition Key 字段值 → 哈希计算 → 路由到 16 个物理分区之一
```

**关键特性**：
- Milvus 自动创建 **16 个物理分区**（默认）
- 多个租户可能共享一个物理分区（逻辑隔离）
- 搜索时自动过滤，只扫描相关数据

### 2. Partition Key vs 手动分区

| 特性 | Partition Key | 手动分区 |
|------|---------------|----------|
| 最大租户数 | 数百万 | 1,024 |
| 管理方式 | 自动 | 手动 |
| 数据隔离 | 逻辑隔离 | 物理隔离 |
| 性能 | 高（避免 99% 无效扫描） | 高 |
| 适用场景 | 大规模多租户 | 中等数量租户 |

### 3. Partition Key 字段要求

- 必须是标量字段（INT64, VARCHAR）
- 在 Schema 中标记为 `is_partition_key=True`
- 每个 Collection 只能有一个 Partition Key 字段

---

## 完整代码示例

```python
"""
Milvus Partition Key 自动分区实战
场景：SaaS 应用多租户知识库

功能：
1. 使用 Partition Key 定义 Schema
2. 自动数据路由（无需指定分区）
3. 租户级别的数据隔离
4. 高效的租户搜索

依赖：
- pymilvus
- numpy
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import numpy as np
from typing import List, Dict, Any
import time


class PartitionKeyManager:
    """Partition Key 管理器 - 演示自动分区"""

    def __init__(self, collection_name: str = "saas_knowledge_base"):
        """
        初始化 Partition Key 管理器

        Args:
            collection_name: Collection 名称
        """
        self.collection_name = collection_name
        self.collection = None

    def connect_to_milvus(self, host: str = "localhost", port: str = "19530"):
        """连接到 Milvus"""
        print(f"连接到 Milvus: {host}:{port}")
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("✓ 连接成功")

    def create_collection_with_partition_key(self):
        """创建带 Partition Key 的 Collection"""
        # 检查 Collection 是否存在
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' 已存在，删除后重新创建")
            utility.drop_collection(self.collection_name)

        # 定义 Schema（关键：标记 tenant_id 为 Partition Key）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="tenant_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_partition_key=True  # 关键：标记为 Partition Key
            ),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="SaaS 多租户知识库 - 使用 Partition Key 自动分区"
        )

        # 创建 Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        print(f"✓ Collection '{self.collection_name}' 创建成功")
        print("✓ Partition Key 字段: tenant_id")
        print("✓ Milvus 将自动创建 16 个物理分区")

    def list_partitions(self):
        """列出所有分区（Partition Key 会自动创建物理分区）"""
        print("\n=== 分区列表 ===")
        partitions = self.collection.partitions
        print(f"物理分区数量: {len(partitions)}")
        for partition in partitions:
            print(f"- {partition.name}")
        return [p.name for p in partitions]

    def insert_tenant_data(
        self,
        tenant_id: str,
        doc_titles: List[str],
        doc_contents: List[str]
    ):
        """
        插入租户数据（无需指定分区，Milvus 自动路由）

        Args:
            tenant_id: 租户 ID
            doc_titles: 文档标题列表
            doc_contents: 文档内容列表
        """
        print(f"\n=== 插入租户 '{tenant_id}' 的数据 ===")

        # 生成随机 embedding
        embeddings = np.random.rand(len(doc_titles), 128).tolist()

        # 准备数据（关键：包含 tenant_id 字段）
        data = [
            [tenant_id] * len(doc_titles),  # tenant_id（Partition Key）
            doc_titles,
            doc_contents,
            embeddings
        ]

        # 插入数据（无需指定分区名称，Milvus 自动路由）
        insert_result = self.collection.insert(data=data)

        print(f"✓ 插入 {insert_result.insert_count} 条数据")
        print(f"✓ Milvus 自动根据 tenant_id='{tenant_id}' 路由到相应分区")

        return insert_result

    def create_index(self):
        """创建向量索引"""
        print("\n=== 创建索引 ===")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("✓ 索引创建成功")

    def load_collection(self):
        """加载整个 Collection（包括所有分区）"""
        print("\n=== 加载 Collection ===")
        self.collection.load()
        time.sleep(2)
        print("✓ Collection 已加载（包括所有分区）")

    def search_by_tenant(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ):
        """
        按租户搜索（自动过滤，只扫描该租户的数据）

        Args:
            tenant_id: 租户 ID
            query_embedding: 查询向量
            top_k: 返回结果数量
        """
        print(f"\n=== 搜索租户 '{tenant_id}' 的数据 ===")

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # 关键：使用 expr 过滤租户数据
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'tenant_id == "{tenant_id}"',  # 关键：过滤表达式
            output_fields=["tenant_id", "doc_title", "doc_content"]
        )

        print(f"找到 {len(results[0])} 条结果:")
        for i, hit in enumerate(results[0]):
            print(f"{i+1}. ID: {hit.id}, 距离: {hit.distance:.4f}")
            print(f"   租户: {hit.entity.get('tenant_id')}")
            print(f"   标题: {hit.entity.get('doc_title')}")
            print(f"   内容: {hit.entity.get('doc_content')[:50]}...")

        return results

    def search_across_tenants(
        self,
        tenant_ids: List[str],
        query_embedding: List[float],
        top_k: int = 5
    ):
        """
        跨租户搜索（搜索多个租户的数据）

        Args:
            tenant_ids: 租户 ID 列表
            query_embedding: 查询向量
            top_k: 返回结果数量
        """
        print(f"\n=== 跨租户搜索 {tenant_ids} ===")

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # 构建 IN 表达式
        tenant_list = ', '.join([f'"{tid}"' for tid in tenant_ids])
        expr = f'tenant_id in [{tenant_list}]'

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["tenant_id", "doc_title", "doc_content"]
        )

        print(f"找到 {len(results[0])} 条结果:")
        for i, hit in enumerate(results[0]):
            print(f"{i+1}. ID: {hit.id}, 距离: {hit.distance:.4f}")
            print(f"   租户: {hit.entity.get('tenant_id')}")
            print(f"   标题: {hit.entity.get('doc_title')}")

        return results

    def get_tenant_count(self, tenant_id: str) -> int:
        """获取租户的数据数量"""
        result = self.collection.query(
            expr=f'tenant_id == "{tenant_id}"',
            output_fields=["count(*)"]
        )
        count = len(result)
        print(f"租户 '{tenant_id}' 包含 {count} 条数据")
        return count

    def enable_partition_key_isolation(self):
        """
        启用 Partition Key 隔离（性能优化）
        注意：仅支持 HNSW 索引
        """
        print("\n=== 启用 Partition Key 隔离 ===")
        print("注意：Partition Key 隔离仅支持 HNSW 索引")
        print("当前示例使用 IVF_FLAT 索引，无法启用隔离")
        print("如需启用，请使用 HNSW 索引并设置 properties={'partitionkey.isolation': True}")


def main():
    """主函数 - 演示 Partition Key 自动分区"""

    # 1. 初始化
    manager = PartitionKeyManager()
    manager.connect_to_milvus()

    # 2. 创建带 Partition Key 的 Collection
    manager.create_collection_with_partition_key()

    # 3. 列出分区（Milvus 自动创建）
    manager.list_partitions()

    # 4. 插入多个租户的数据（无需指定分区）
    # 租户 A：技术公司
    manager.insert_tenant_data(
        tenant_id="tenant_a",
        doc_titles=["API 文档", "架构设计", "性能优化"],
        doc_contents=[
            "本文档介绍了 REST API 的使用方法...",
            "系统采用微服务架构...",
            "性能优化的关键在于..."
        ]
    )

    # 租户 B：教育机构
    manager.insert_tenant_data(
        tenant_id="tenant_b",
        doc_titles=["课程大纲", "教学计划", "考试安排"],
        doc_contents=[
            "本学期课程包括：数据结构、算法...",
            "教学计划：第一周讲解基础知识...",
            "期末考试时间：2026 年 6 月..."
        ]
    )

    # 租户 C：医疗机构
    manager.insert_tenant_data(
        tenant_id="tenant_c",
        doc_titles=["诊疗指南", "药品说明", "病例分析"],
        doc_contents=[
            "高血压诊疗指南：首先进行血压测量...",
            "阿司匹林：用于预防心血管疾病...",
            "病例：患者男性，65 岁，主诉胸痛..."
        ]
    )

    # 5. 创建索引
    manager.create_index()

    # 6. 加载 Collection
    manager.load_collection()

    # 7. 按租户搜索（只扫描该租户的数据）
    query_embedding = np.random.rand(128).tolist()
    manager.search_by_tenant("tenant_a", query_embedding, top_k=3)

    # 8. 跨租户搜索（搜索多个租户）
    manager.search_across_tenants(["tenant_a", "tenant_b"], query_embedding, top_k=5)

    # 9. 获取租户数据数量
    manager.get_tenant_count("tenant_a")
    manager.get_tenant_count("tenant_b")
    manager.get_tenant_count("tenant_c")

    # 10. Partition Key 隔离说明
    manager.enable_partition_key_isolation()

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
```

---

## 代码说明

### 1. 定义 Partition Key

```python
FieldSchema(
    name="tenant_id",
    dtype=DataType.VARCHAR,
    max_length=64,
    is_partition_key=True  # 关键：标记为 Partition Key
)
```

### 2. 插入数据（自动路由）

```python
# 无需指定分区名称，Milvus 自动根据 tenant_id 路由
collection.insert(data=[
    [tenant_id] * len(titles),  # Partition Key 字段
    titles,
    contents,
    embeddings
])
```

### 3. 按租户搜索（自动过滤）

```python
# 使用 expr 过滤租户数据
collection.search(
    data=[query_embedding],
    expr=f'tenant_id == "{tenant_id}"',  # 自动过滤
    ...
)
```

---

## 性能优化

### 1. 避免 99% 无效扫描

**来源**：`reference/search_分区管理_01.md`（Twitter/X）

**核心数据**：
- 使用 Partition Key 可以避免扫描 99% 的无关数据
- 多租户搜索延迟从秒级降至毫秒级

**示例**：
```python
# 不使用 Partition Key：扫描所有数据
results = collection.search(data=[query_embedding], limit=10)

# 使用 Partition Key：只扫描相关租户数据
results = collection.search(
    data=[query_embedding],
    expr='tenant_id == "tenant_a"',  # 只扫描 tenant_a 的数据
    limit=10
)
```

### 2. Partition Key 隔离（HNSW 索引）

**来源**：`reference/context7_milvus_01.md`

**启用方式**：
```python
from pymilvus import Collection, CollectionSchema

# 创建 Collection 时启用 Partition Key 隔离
collection = Collection(
    name="my_collection",
    schema=schema,
    properties={"partitionkey.isolation": True}  # 启用隔离
)

# 注意：仅支持 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200}
}
```

### 3. 推荐分区数量

**来源**：`reference/search_分区管理_03.md`（GitHub）

**推荐配置**：
- 推荐：100-200 个物理分区
- 默认：64 个物理分区
- 最大：根据系统资源调整

---

## RAG 应用场景

### 场景 1：SaaS 知识库

```python
# 每个企业客户一个 tenant_id
manager.insert_tenant_data(
    tenant_id="company_001",
    doc_titles=["产品手册", "技术文档"],
    doc_contents=[...]
)

# 搜索时只查询该企业的数据
results = manager.search_by_tenant("company_001", query_embedding)
```

### 场景 2：多语言文档库

```python
# 使用语言代码作为 Partition Key
FieldSchema(
    name="language",
    dtype=DataType.VARCHAR,
    max_length=10,
    is_partition_key=True
)

# 搜索时只查询特定语言
results = collection.search(
    data=[query_embedding],
    expr='language == "zh-CN"',  # 只搜索中文文档
    limit=10
)
```

### 场景 3：用户个人知识库

```python
# 使用 user_id 作为 Partition Key
FieldSchema(
    name="user_id",
    dtype=DataType.VARCHAR,
    max_length=64,
    is_partition_key=True
)

# 每个用户只能访问自己的数据
results = collection.search(
    data=[query_embedding],
    expr=f'user_id == "{current_user_id}"',
    limit=10
)
```

---

## 常见问题

### Q1: Partition Key 支持哪些数据类型？

**支持的类型**：
- `INT64`
- `VARCHAR`

**不支持的类型**：
- `FLOAT`
- `DOUBLE`
- `BOOL`
- `ARRAY`
- `JSON`

### Q2: 如何选择 Partition Key 字段？

**选择原则**：
1. **高基数**：字段值的种类要多（如 user_id, tenant_id）
2. **查询频繁**：经常用于过滤的字段
3. **数据分布均匀**：避免数据倾斜

**示例**：
```python
# 好的选择：tenant_id（高基数，查询频繁）
FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, is_partition_key=True)

# 不好的选择：status（低基数，只有 "active" 和 "inactive"）
FieldSchema(name="status", dtype=DataType.VARCHAR, is_partition_key=True)
```

### Q3: Partition Key 可以修改吗？

**不可以**。Partition Key 在 Collection 创建后无法修改。如需修改，必须：
1. 创建新的 Collection
2. 迁移数据
3. 删除旧 Collection

---

## 引用来源

### 官方文档（Context7）
- **Partition Key 定义**：`reference/context7_milvus_01.md`
  - Schema 定义：`is_partition_key=True`
  - 自动分区机制：16 个物理分区
  - Partition Key 隔离：`partitionkey.isolation`

### 多租户策略（Context7）
- **Partition Key-level 多租户**：`reference/context7_milvus_02.md`
  - 支持数百万租户
  - 逻辑隔离
  - 自动数据路由

### 性能优化（Twitter/X）
- **99% 无效扫描避免**：`reference/search_分区管理_01.md`
  - 延迟从秒级降至毫秒级
  - 只扫描相关数据

### 最佳实践（GitHub）
- **分区数量建议**：`reference/search_分区管理_03.md`
  - 推荐 100-200 个物理分区
  - 控制总数在 1000 以内

### 源码分析
- **Partition Key 测试**：`reference/source_分区管理_01.md`
  - `IsPartitionKey: true` 标记
  - 自动分区管理

---

## 下一步

- **场景 3**：多租户知识库 - 完整的 SaaS 应用多租户实现
- **场景 4**：时间序列分区 - 按时间分区管理历史数据
- **场景 5**：混合分区检索优化 - 结合分区和标量过滤提升性能

---

**版本**：v1.0
**最后更新**：2026-02-25
**适用 Milvus 版本**：2.6.x
