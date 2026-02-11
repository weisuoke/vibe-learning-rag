# 核心概念 2: Partition 数据操作

## 什么是 Partition 数据操作？

**Partition 数据操作**是指在 Milvus 分区中执行数据插入、查询、加载、释放等操作的过程，是使用 Partition 的核心环节。

### 核心定义

```python
# Partition 数据操作的本质
partition = collection.partition("partition_2024")

# 插入数据
partition.insert(data)

# 查询数据
results = collection.search(query_vector, partition_names=["partition_2024"])

# 加载/释放
partition.load()
partition.release()
```

**类比理解**:
- **前端类比**: 就像对数据库表的 CRUD 操作 `INSERT INTO partition_2024 VALUES (...)`
- **日常类比**: 就像往书架的特定层放书、从特定层找书、把书从手边拿走

---

## 1. 插入数据到分区

### 1.1 基础插入

```python
from pymilvus import Collection, connections
import numpy as np

# 连接和准备
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# 获取分区
partition = collection.partition("partition_2024")

# 准备数据
vectors = np.random.rand(100, 128).tolist()  # 100条128维向量
texts = [f"document_{i}" for i in range(100)]

# 插入数据
data = [vectors, texts]
result = partition.insert(data)

# 刷新到磁盘
collection.flush()

print(f"插入成功，ID范围: {result.primary_keys[0]} - {result.primary_keys[-1]}")
```

**关键点**:
- 数据格式必须与 Schema 一致
- 插入后需要 `flush()` 持久化
- 返回的 `primary_keys` 是自动生成的 ID

---

### 1.2 批量插入优化

```python
# 大批量数据插入优化
def batch_insert_to_partition(partition, vectors, texts, batch_size=1000):
    """批量插入数据到分区

    Args:
        partition: 目标分区
        vectors: 向量列表
        texts: 文本列表
        batch_size: 每批次大小
    """
    total = len(vectors)
    inserted_ids = []

    for i in range(0, total, batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        data = [batch_vectors, batch_texts]
        result = partition.insert(data)
        inserted_ids.extend(result.primary_keys)

        print(f"已插入 {min(i+batch_size, total)}/{total} 条数据")

    # 最后统一刷新
    collection.flush()
    print(f"批量插入完成，共 {len(inserted_ids)} 条数据")

    return inserted_ids

# 使用示例
vectors = np.random.rand(10000, 128).tolist()
texts = [f"doc_{i}" for i in range(10000)]

partition = collection.partition("partition_2024")
ids = batch_insert_to_partition(partition, vectors, texts, batch_size=1000)
```

**性能优化**:
- 批量大小推荐 1000-5000
- 批量插入比单条插入快 20-100 倍
- 最后统一 flush，减少 I/O 次数

---

### 1.3 动态分区插入

```python
from datetime import datetime

def insert_with_dynamic_partition(embedding, text, partition_strategy="time"):
    """根据策略动态选择分区插入

    Args:
        embedding: 向量
        text: 文本
        partition_strategy: 分区策略 (time/hash/round_robin)
    """
    if partition_strategy == "time":
        # 按时间分区
        month_str = datetime.now().strftime("%Y_%m")
        partition_name = f"partition_{month_str}"
    elif partition_strategy == "hash":
        # 按哈希分区
        hash_value = hash(text) % 10
        partition_name = f"partition_{hash_value}"
    elif partition_strategy == "round_robin":
        # 轮询分区
        partition_count = len(collection.partitions)
        partition_index = np.random.randint(0, partition_count)
        partition_name = collection.partitions[partition_index].name
    else:
        partition_name = "_default"

    # 确保分区存在
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)

    # 插入数据
    partition = collection.partition(partition_name)
    partition.insert([[embedding], [text]])

    return partition_name

# 使用示例
embedding = np.random.rand(128).tolist()
partition_name = insert_with_dynamic_partition(embedding, "test document", strategy="time")
print(f"数据插入到分区: {partition_name}")
```

---

## 2. 从分区查询数据

### 2.1 单分区检索

```python
# 从指定分区检索
query_vector = np.random.rand(1, 128).tolist()

search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}
}

results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,
    partition_names=["partition_2024"]  # 指定单个分区
)

# 输出结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 距离: {hit.distance:.4f}, 文本: {hit.entity.get('text')}")
```

**性能提升**:
- 单分区检索比全 Collection 检索快 3-10 倍
- 分区越小，检索越快

---

### 2.2 多分区检索

```python
# 从多个分区检索
partition_names = ["partition_2024_01", "partition_2024_02", "partition_2024_03"]

results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,
    partition_names=partition_names  # 指定多个分区
)

print(f"从 {len(partition_names)} 个分区检索到 {len(results[0])} 条结果")
```

**注意事项**:
- 多分区检索会并行执行
- 结果会合并并按相似度排序
- 分区越多，检索时间越长

---

### 2.3 时间范围查询

```python
from datetime import datetime, timedelta

def search_by_time_range(query_vector, start_date, end_date):
    """按时间范围查询

    Args:
        query_vector: 查询向量
        start_date: 开始日期
        end_date: 结束日期
    """
    # 生成时间范围内的分区列表
    partition_names = []
    current = start_date

    while current <= end_date:
        partition_name = f"partition_{current.strftime('%Y_%m')}"
        if collection.has_partition(partition_name):
            partition_names.append(partition_name)

        # 下一个月
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    print(f"检索分区: {partition_names}")

    # 执行检索
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        partition_names=partition_names
    )

    return results

# 使用示例：查询最近3个月的数据
start = datetime(2024, 1, 1)
end = datetime(2024, 3, 31)
query_vector = np.random.rand(128).tolist()

results = search_by_time_range(query_vector, start, end)
```

---

### 2.4 分区 + 标量过滤

```python
# 组合使用分区和标量过滤
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=10,
    partition_names=["partition_2024_01"],  # 分区过滤（粗粒度）
    expr='category == "tech" and views > 1000'  # 标量过滤（细粒度）
)

print(f"检索结果: {len(results[0])} 条")
```

**最佳实践**:
- Partition: 粗粒度过滤（时间、租户、地区）
- 标量过滤: 细粒度过滤（类别、状态、范围）
- 组合使用可获得最佳性能

---

## 3. 加载和释放分区

### 3.1 加载分区到内存

```python
# 加载分区（必须加载才能检索）
partition = collection.partition("partition_2024")
partition.load()

print(f"分区 {partition.name} 已加载到内存")

# 检查加载状态
print(f"加载状态: {partition.is_loaded}")
```

**关键点**:
- 分区必须加载到内存才能检索
- 加载会占用内存
- 加载时间取决于分区大小

---

### 3.2 释放分区内存

```python
# 释放分区内存
partition.release()

print(f"分区 {partition.name} 已从内存释放")

# 检查释放状态
print(f"加载状态: {partition.is_loaded}")
```

**使用场景**:
- 内存不足时释放不常用的分区
- 冷数据按需加载和释放
- 优化内存使用

---

### 3.3 批量加载和释放

```python
# 批量加载多个分区
def load_partitions(partition_names):
    """批量加载分区"""
    for name in partition_names:
        partition = collection.partition(name)
        if not partition.is_loaded:
            partition.load()
            print(f"✓ 加载分区: {name}")
        else:
            print(f"- 分区已加载: {name}")

# 批量释放多个分区
def release_partitions(partition_names):
    """批量释放分区"""
    for name in partition_names:
        partition = collection.partition(name)
        if partition.is_loaded:
            partition.release()
            print(f"✓ 释放分区: {name}")
        else:
            print(f"- 分区未加载: {name}")

# 使用示例
hot_partitions = ["partition_2024_01", "partition_2024_02"]
load_partitions(hot_partitions)

cold_partitions = ["partition_2023_01", "partition_2023_02"]
release_partitions(cold_partitions)
```

---

### 3.4 动态加载策略

```python
class PartitionLoadManager:
    """分区加载管理器"""

    def __init__(self, collection, max_loaded=5):
        self.collection = collection
        self.max_loaded = max_loaded
        self.loaded_partitions = []

    def load_partition(self, partition_name):
        """加载分区（LRU策略）"""
        partition = self.collection.partition(partition_name)

        # 如果已加载，移到最前面
        if partition_name in self.loaded_partitions:
            self.loaded_partitions.remove(partition_name)
            self.loaded_partitions.insert(0, partition_name)
            return

        # 如果超过最大加载数，释放最久未用的分区
        if len(self.loaded_partitions) >= self.max_loaded:
            oldest = self.loaded_partitions.pop()
            self.collection.partition(oldest).release()
            print(f"释放分区: {oldest}")

        # 加载新分区
        partition.load()
        self.loaded_partitions.insert(0, partition_name)
        print(f"加载分区: {partition_name}")

    def search_with_auto_load(self, query_vector, partition_names, **kwargs):
        """检索时自动加载分区"""
        # 确保所有分区都已加载
        for name in partition_names:
            self.load_partition(name)

        # 执行检索
        results = self.collection.search(
            data=[query_vector],
            partition_names=partition_names,
            **kwargs
        )

        return results

# 使用示例
manager = PartitionLoadManager(collection, max_loaded=3)

# 检索会自动加载分区
query_vector = np.random.rand(128).tolist()
results = manager.search_with_auto_load(
    query_vector,
    partition_names=["partition_2024_01"],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5
)
```

---

## 4. Query 操作（标量查询）

### 4.1 基础 Query

```python
# Query: 通过标量条件查询（不涉及向量检索）
results = collection.query(
    expr="id in [1, 2, 3, 4, 5]",
    output_fields=["id", "text"],
    partition_names=["partition_2024"]
)

for result in results:
    print(f"ID: {result['id']}, 文本: {result['text']}")
```

**Query vs Search**:
- Query: 标量条件查询，不涉及向量
- Search: 向量相似度检索

---

### 4.2 复杂条件 Query

```python
# 复杂条件查询
results = collection.query(
    expr='category == "tech" and views > 1000 and created_at > 1704067200',
    output_fields=["id", "text", "category", "views"],
    partition_names=["partition_2024_01", "partition_2024_02"],
    limit=100
)

print(f"查询到 {len(results)} 条结果")
```

---

## 5. 删除分区数据

### 5.1 按 ID 删除

```python
# 从指定分区删除数据
collection.delete(
    expr="id in [1, 2, 3, 4, 5]",
    partition_name="partition_2024"
)

collection.flush()
print("删除成功")
```

---

### 5.2 按条件删除

```python
# 按条件删除
collection.delete(
    expr='category == "spam" or views < 10',
    partition_name="partition_2024"
)

collection.flush()
print("批量删除成功")
```

---

### 5.3 清空分区

```python
# 清空整个分区（删除所有数据）
def clear_partition(partition_name):
    """清空分区的所有数据"""
    partition = collection.partition(partition_name)

    # 查询所有 ID
    results = collection.query(
        expr="id >= 0",
        output_fields=["id"],
        partition_names=[partition_name]
    )

    if len(results) == 0:
        print(f"分区 {partition_name} 已经是空的")
        return

    # 批量删除
    ids = [r["id"] for r in results]
    collection.delete(
        expr=f"id in {ids}",
        partition_name=partition_name
    )

    collection.flush()
    print(f"清空分区 {partition_name}，删除了 {len(ids)} 条数据")

# 使用示例
clear_partition("partition_2023_01")
```

---

## 6. 分区统计信息

### 6.1 获取分区数据量

```python
# 获取分区的数据量
partition = collection.partition("partition_2024")
num_entities = partition.num_entities

print(f"分区 {partition.name} 有 {num_entities} 条数据")
```

---

### 6.2 获取所有分区统计

```python
# 获取所有分区的统计信息
def get_partitions_stats():
    """获取所有分区的统计信息"""
    stats = []

    for partition in collection.partitions:
        info = {
            "name": partition.name,
            "num_entities": partition.num_entities,
            "is_loaded": partition.is_loaded,
            "description": partition.description
        }
        stats.append(info)

    return stats

# 使用示例
stats = get_partitions_stats()

print("分区统计:")
print(f"{'分区名':<30} {'数据量':<15} {'加载状态':<10}")
print("-" * 60)
for s in stats:
    print(f"{s['name']:<30} {s['num_entities']:<15} {'已加载' if s['is_loaded'] else '未加载':<10}")
```

---

## 7. 在实际应用中的使用

### 7.1 RAG 系统的分区操作

```python
# RAG 文档问答系统的分区操作

def insert_document_to_rag(doc_text, doc_metadata):
    """插入文档到 RAG 系统"""
    # 1. 生成 Embedding
    embedding = get_embedding(doc_text)

    # 2. 确定分区（按上传时间）
    month_str = datetime.now().strftime("%Y_%m")
    partition_name = f"partition_{month_str}"

    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)

    # 3. 插入数据
    partition = collection.partition(partition_name)
    data = [
        [embedding],
        [doc_text],
        [doc_metadata.get("category", "")],
        [int(datetime.now().timestamp())]
    ]
    partition.insert(data)
    collection.flush()

    return partition_name

def search_rag_documents(query, time_range_months=3):
    """检索 RAG 文档"""
    # 1. 生成查询 Embedding
    query_embedding = get_embedding(query)

    # 2. 确定检索的分区（最近N个月）
    partition_names = []
    for i in range(time_range_months):
        date = datetime.now() - timedelta(days=30*i)
        partition_name = f"partition_{date.strftime('%Y_%m')}"
        if collection.has_partition(partition_name):
            partition_names.append(partition_name)

    # 3. 加载分区
    for name in partition_names:
        partition = collection.partition(name)
        if not partition.is_loaded:
            partition.load()

    # 4. 执行检索
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
        partition_names=partition_names
    )

    return results
```

---

### 7.2 多租户系统的分区操作

```python
# 多租户系统的分区操作

def insert_tenant_document(tenant_id, doc_embedding, doc_text):
    """插入租户文档"""
    partition_name = f"tenant_{tenant_id}"

    # 确保租户分区存在
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)

    # 插入数据
    partition = collection.partition(partition_name)
    partition.insert([[doc_embedding], [doc_text]])
    collection.flush()

def search_tenant_documents(tenant_id, query_embedding):
    """检索租户文档（强制租户隔离）"""
    partition_name = f"tenant_{tenant_id}"

    # 加载租户分区
    partition = collection.partition(partition_name)
    if not partition.is_loaded:
        partition.load()

    # 只检索当前租户的分区
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        partition_names=[partition_name]  # 强制租户隔离
    )

    return results
```

---

## 总结

### 核心要点

1. **插入操作**: 使用 `partition.insert()` 插入数据到指定分区
2. **查询操作**: 使用 `collection.search()` 指定 `partition_names` 检索
3. **加载/释放**: 使用 `partition.load()` 和 `partition.release()` 管理内存
4. **Query 操作**: 使用 `collection.query()` 进行标量查询
5. **删除操作**: 使用 `collection.delete()` 删除分区数据

### 最佳实践

- ✅ 插入时明确指定分区
- ✅ 检索时明确指定分区
- ✅ 批量操作提升性能
- ✅ 动态加载和释放优化内存
- ✅ 组合使用分区和标量过滤

### 下一步

学习 **核心概念 3: Partition 性能优化**，了解如何通过分区优化检索性能。
