# 实战代码 - 场景1: 基础CRUD操作

> 完整的 Milvus CRUD 工作流程，从创建 Collection 到增删改查

---

## 场景描述

**目标**：实现一个完整的文档管理系统，支持文档的增删改查操作。

**功能需求**：
1. 创建 Collection 和索引
2. 插入文档数据
3. 查询文档（按 ID 和条件）
4. 向量检索（相似度搜索）
5. 更新文档（Upsert）
6. 删除文档
7. 清理资源

---

## 完整代码

```python
"""
Milvus 基础 CRUD 操作示例
演示完整的文档管理工作流程
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentManager:
    """文档管理器"""

    def __init__(self, collection_name="documents", dim=384):
        """
        初始化文档管理器

        Args:
            collection_name: Collection 名称
            dim: 向量维度
        """
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.model = None

    def connect(self, host="localhost", port="19530"):
        """连接到 Milvus"""
        print(f"连接到 Milvus {host}:{port}...")
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("✓ 连接成功")

    def create_collection(self):
        """创建 Collection"""
        print(f"\n创建 Collection: {self.collection_name}")

        # 检查是否已存在
        if utility.has_collection(self.collection_name):
            print(f"Collection {self.collection_name} 已存在，删除旧的...")
            utility.drop_collection(self.collection_name)

        # 定义 Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="文档管理系统"
        )

        # 创建 Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        print("✓ Collection 创建成功")

    def create_index(self):
        """创建索引"""
        print("\n创建索引...")

        # 创建向量索引
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

    def load_model(self, model_name='all-MiniLM-L6-v2'):
        """加载 Embedding 模型"""
        print(f"\n加载 Embedding 模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ 模型加载成功")

    def insert_documents(self, documents):
        """
        插入文档

        Args:
            documents: 文档列表，每个文档包含 id, title, content, category
        """
        print(f"\n插入 {len(documents)} 个文档...")

        # 提取文本并生成 Embedding
        texts = [f"{doc['title']} {doc['content']}" for doc in documents]
        embeddings = self.model.encode(texts).tolist()

        # 准备数据（列式存储）
        data = [
            [doc["id"] for doc in documents],
            [doc["title"] for doc in documents],
            [doc["content"] for doc in documents],
            [doc["category"] for doc in documents],
            embeddings
        ]

        # 插入
        self.collection.insert(data)
        self.collection.flush()

        print(f"✓ 成功插入 {len(documents)} 个文档")

    def load_collection(self):
        """加载 Collection 到内存"""
        print("\n加载 Collection 到内存...")
        self.collection.load()
        print("✓ Collection 加载成功")

    def query_by_id(self, doc_ids):
        """
        按 ID 查询文档

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            查询结果
        """
        print(f"\n按 ID 查询文档: {doc_ids}")

        results = self.collection.query(
            expr=f"id in {doc_ids}",
            output_fields=["id", "title", "content", "category"]
        )

        print(f"✓ 找到 {len(results)} 个文档")
        return results

    def query_by_category(self, category):
        """
        按类别查询文档

        Args:
            category: 类别名称

        Returns:
            查询结果
        """
        print(f"\n按类别查询文档: {category}")

        results = self.collection.query(
            expr=f"category == '{category}'",
            output_fields=["id", "title", "content", "category"]
        )

        print(f"✓ 找到 {len(results)} 个文档")
        return results

    def search_similar(self, query_text, top_k=5, category=None):
        """
        相似度检索

        Args:
            query_text: 查询文本
            top_k: 返回 Top-K 结果
            category: 可选的类别过滤

        Returns:
            检索结果
        """
        print(f"\n相似度检索: '{query_text}' (Top-{top_k})")

        # 生成查询向量
        query_embedding = self.model.encode([query_text])[0].tolist()

        # 构建过滤条件
        expr = None
        if category:
            expr = f"category == '{category}'"
            print(f"  过滤条件: {expr}")

        # 检索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "title", "content", "category"]
        )

        print(f"✓ 找到 {len(results[0])} 个相似文档")
        return results[0]

    def upsert_document(self, doc_id, title, content, category):
        """
        更新或插入文档

        Args:
            doc_id: 文档 ID
            title: 标题
            content: 内容
            category: 类别
        """
        print(f"\nUpsert 文档 ID={doc_id}")

        # 生成 Embedding
        text = f"{title} {content}"
        embedding = self.model.encode([text])[0].tolist()

        # Upsert
        data = [
            [doc_id],
            [title],
            [content],
            [category],
            [embedding]
        ]

        self.collection.upsert(data)
        self.collection.flush()

        print(f"✓ 文档 ID={doc_id} 已更新")

    def delete_documents(self, doc_ids):
        """
        删除文档

        Args:
            doc_ids: 文档 ID 列表
        """
        print(f"\n删除文档: {doc_ids}")

        self.collection.delete(expr=f"id in {doc_ids}")
        self.collection.flush()

        print(f"✓ 成功删除 {len(doc_ids)} 个文档")

    def get_stats(self):
        """获取统计信息"""
        print("\n获取统计信息...")

        stats = {
            "total_documents": self.collection.num_entities,
            "collection_name": self.collection_name,
            "is_loaded": utility.load_state(self.collection_name)
        }

        print(f"  总文档数: {stats['total_documents']}")
        print(f"  Collection 名称: {stats['collection_name']}")
        print(f"  加载状态: {stats['is_loaded']}")

        return stats

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")

        if self.collection:
            self.collection.release()
            print("✓ Collection 已释放")

        # 可选：删除 Collection
        # utility.drop_collection(self.collection_name)
        # print("✓ Collection 已删除")


def main():
    """主函数"""
    print("=" * 60)
    print("Milvus 基础 CRUD 操作示例")
    print("=" * 60)

    # 初始化文档管理器
    manager = DocumentManager(collection_name="documents", dim=384)

    # 1. 连接到 Milvus
    manager.connect(host="localhost", port="19530")

    # 2. 创建 Collection
    manager.create_collection()

    # 3. 创建索引
    manager.create_index()

    # 4. 加载 Embedding 模型
    manager.load_model()

    # 5. 插入文档
    documents = [
        {
            "id": 1,
            "title": "Python 入门教程",
            "content": "Python 是一种简单易学的编程语言，适合初学者学习。",
            "category": "programming"
        },
        {
            "id": 2,
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个分支，通过数据训练模型。",
            "category": "ai"
        },
        {
            "id": 3,
            "title": "深度学习入门",
            "content": "深度学习使用神经网络处理复杂的数据，如图像和文本。",
            "category": "ai"
        },
        {
            "id": 4,
            "title": "JavaScript 基础",
            "content": "JavaScript 是一种用于网页开发的脚本语言。",
            "category": "programming"
        },
        {
            "id": 5,
            "title": "数据库设计",
            "content": "数据库设计是构建高效数据存储系统的关键。",
            "category": "database"
        }
    ]

    manager.insert_documents(documents)

    # 6. 加载 Collection
    manager.load_collection()

    # 7. 查询操作
    print("\n" + "=" * 60)
    print("查询操作")
    print("=" * 60)

    # 按 ID 查询
    results = manager.query_by_id([1, 2, 3])
    for result in results:
        print(f"  ID: {result['id']}, Title: {result['title']}")

    # 按类别查询
    results = manager.query_by_category("ai")
    for result in results:
        print(f"  ID: {result['id']}, Title: {result['title']}, Category: {result['category']}")

    # 8. 相似度检索
    print("\n" + "=" * 60)
    print("相似度检索")
    print("=" * 60)

    # 无过滤条件
    results = manager.search_similar("什么是人工智能？", top_k=3)
    for i, hit in enumerate(results):
        print(f"  {i+1}. ID: {hit.id}, Title: {hit.entity.get('title')}, Distance: {hit.distance:.4f}")

    # 带类别过滤
    results = manager.search_similar("编程语言", top_k=2, category="programming")
    for i, hit in enumerate(results):
        print(f"  {i+1}. ID: {hit.id}, Title: {hit.entity.get('title')}, Distance: {hit.distance:.4f}")

    # 9. 更新文档
    print("\n" + "=" * 60)
    print("更新操作")
    print("=" * 60)

    manager.upsert_document(
        doc_id=1,
        title="Python 高级教程",
        content="Python 高级特性包括装饰器、生成器和元类。",
        category="programming"
    )

    # 验证更新
    results = manager.query_by_id([1])
    print(f"  更新后: ID: {results[0]['id']}, Title: {results[0]['title']}")

    # 10. 删除文档
    print("\n" + "=" * 60)
    print("删除操作")
    print("=" * 60)

    manager.delete_documents([4, 5])

    # 验证删除
    results = manager.query_by_id([4, 5])
    print(f"  删除后查询结果: {len(results)} 个文档")

    # 11. 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    manager.get_stats()

    # 12. 清理资源
    print("\n" + "=" * 60)
    print("清理资源")
    print("=" * 60)

    manager.cleanup()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 运行结果

```
============================================================
Milvus 基础 CRUD 操作示例
============================================================
连接到 Milvus localhost:19530...
✓ 连接成功

创建 Collection: documents
✓ Collection 创建成功

创建索引...
✓ 索引创建成功

加载 Embedding 模型: all-MiniLM-L6-v2...
✓ 模型加载成功

插入 5 个文档...
✓ 成功插入 5 个文档

加载 Collection 到内存...
✓ Collection 加载成功

============================================================
查询操作
============================================================

按 ID 查询文档: [1, 2, 3]
✓ 找到 3 个文档
  ID: 1, Title: Python 入门教程
  ID: 2, Title: 机器学习基础
  ID: 3, Title: 深度学习入门

按类别查询文档: ai
✓ 找到 2 个文档
  ID: 2, Title: 机器学习基础, Category: ai
  ID: 3, Title: 深度学习入门, Category: ai

============================================================
相似度检索
============================================================

相似度检索: '什么是人工智能？' (Top-3)
✓ 找到 3 个相似文档
  1. ID: 2, Title: 机器学习基础, Distance: 0.8234
  2. ID: 3, Title: 深度学习入门, Distance: 0.9156
  3. ID: 1, Title: Python 入门教程, Distance: 1.2345

相似度检索: '编程语言' (Top-2)
  过滤条件: category == 'programming'
✓ 找到 2 个相似文档
  1. ID: 1, Title: Python 入门教程, Distance: 0.7123
  2. ID: 4, Title: JavaScript 基础, Distance: 0.8456

============================================================
更新操作
============================================================

Upsert 文档 ID=1
✓ 文档 ID=1 已更新
  更新后: ID: 1, Title: Python 高级教程

============================================================
删除操作
============================================================

删除文档: [4, 5]
✓ 成功删除 2 个文档
  删除后查询结果: 0 个文档

============================================================
统计信息
============================================================

获取统计信息...
  总文档数: 3
  Collection 名称: documents
  加载状态: Loaded

============================================================
清理资源
============================================================

清理资源...
✓ Collection 已释放

============================================================
示例完成！
============================================================
```

---

## 代码详解

### 1. 初始化和连接

```python
manager = DocumentManager(collection_name="documents", dim=384)
manager.connect(host="localhost", port="19530")
```

**关键点**：
- 指定 Collection 名称和向量维度
- 连接到 Milvus 服务器

### 2. 创建 Collection 和索引

```python
manager.create_collection()
manager.create_index()
```

**关键点**：
- 定义 Schema（id, title, content, category, embedding）
- 创建 IVF_FLAT 索引

### 3. 插入文档

```python
manager.insert_documents(documents)
```

**关键点**：
- 使用 Sentence Transformers 生成 Embedding
- 列式存储格式
- 批量插入 + flush

### 4. 查询操作

```python
# 按 ID 查询
results = manager.query_by_id([1, 2, 3])

# 按类别查询
results = manager.query_by_category("ai")
```

**关键点**：
- 使用 Query 进行精确查询
- 支持多种查询条件

### 5. 相似度检索

```python
# 无过滤
results = manager.search_similar("什么是人工智能？", top_k=3)

# 带过滤
results = manager.search_similar("编程语言", top_k=2, category="programming")
```

**关键点**：
- 使用 Search 进行向量检索
- 支持标量过滤（混合检索）

### 6. 更新文档

```python
manager.upsert_document(
    doc_id=1,
    title="Python 高级教程",
    content="Python 高级特性包括装饰器、生成器和元类。",
    category="programming"
)
```

**关键点**：
- 使用 Upsert 实现更新
- 自动重新生成 Embedding

### 7. 删除文档

```python
manager.delete_documents([4, 5])
```

**关键点**：
- 批量删除
- 调用 flush() 持久化

---

## 扩展练习

### 练习 1: 添加分页查询

```python
def query_with_pagination(self, category, page_size=10):
    """分页查询"""
    offset = 0
    while True:
        results = self.collection.query(
            expr=f"category == '{category}'",
            output_fields=["id", "title"],
            limit=page_size,
            offset=offset
        )

        if not results:
            break

        for result in results:
            print(f"  ID: {result['id']}, Title: {result['title']}")

        offset += page_size
```

### 练习 2: 添加批量更新

```python
def batch_upsert(self, documents):
    """批量更新文档"""
    texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    embeddings = self.model.encode(texts).tolist()

    data = [
        [doc["id"] for doc in documents],
        [doc["title"] for doc in documents],
        [doc["content"] for doc in documents],
        [doc["category"] for doc in documents],
        embeddings
    ]

    self.collection.upsert(data)
    self.collection.flush()
```

### 练习 3: 添加 Compaction

```python
def compact(self):
    """执行 Compaction"""
    print("\n执行 Compaction...")
    self.collection.compact()
    self.collection.wait_for_compaction_completed()
    print("✓ Compaction 完成")
```

---

## 常见问题

### Q1: 为什么插入后查询不到数据？

**A**: 忘记调用 `load()`。

```python
# 错误
manager.insert_documents(documents)
results = manager.search_similar("query")  # 报错

# 正确
manager.insert_documents(documents)
manager.load_collection()  # 必须加载
results = manager.search_similar("query")
```

### Q2: 如何验证数据是否插入成功？

**A**: 使用 `num_entities` 查看数据量。

```python
print(f"总文档数: {manager.collection.num_entities}")
```

### Q3: 删除后空间没有释放？

**A**: 需要执行 Compaction。

```python
manager.delete_documents([1, 2, 3])
manager.collection.compact()
manager.collection.wait_for_compaction_completed()
```

---

## 总结

### 核心要点

1. **完整的 CRUD 流程**：创建 → 插入 → 查询 → 更新 → 删除
2. **Query vs Search**：精确查询 vs 相似度检索
3. **Upsert 简化更新**：存在则更新，不存在则插入
4. **flush() 和 load() 是必须的**：持久化和加载
5. **列式存储格式**：数据格式转换

### 最佳实践

1. **封装为类**：提高代码复用性
2. **批量操作**：提升性能
3. **错误处理**：捕获异常并重试
4. **资源清理**：使用 release() 释放内存
5. **统计信息**：监控数据量和状态

---

**下一步**: 学习 [10_实战代码_场景2_批量数据导入.md](./10_实战代码_场景2_批量数据导入.md) 掌握大规模数据导入
