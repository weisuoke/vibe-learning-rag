# 实战代码 - 场景4: Upsert与数据更新

> 掌握 Upsert 操作和数据更新策略，实现高效的数据同步

---

## 场景描述

**目标**：实现一个支持数据更新和同步的系统，使用 Upsert 简化更新逻辑。

**功能需求**：
1. 单条和批量 Upsert
2. 部分字段更新
3. 条件更新
4. 数据同步（从外部系统）
5. 冲突处理和错误恢复

---

## 完整代码

```python
"""
Milvus Upsert 与数据更新示例
演示各种数据更新场景
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import time
from datetime import datetime


class DataUpdateManager:
    """数据更新管理器"""

    def __init__(self, collection_name="knowledge_base", dim=384):
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.model = None

    def connect(self, host="localhost", port="19530"):
        """连接到 Milvus"""
        connections.connect(alias="default", host=host, port=port)
        print(f"✓ 已连接到 Milvus")

    def create_collection(self):
        """创建 Collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="version", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]

        schema = CollectionSchema(fields=fields, description="知识库")
        self.collection = Collection(name=self.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

        print(f"✓ Collection 创建成功")

    def load_model(self, model_name='all-MiniLM-L6-v2'):
        """加载 Embedding 模型"""
        self.model = SentenceTransformer(model_name)
        print("✓ 模型加载成功")

    def insert_initial_data(self):
        """插入初始数据"""
        documents = [
            {"id": 1, "title": "Python 基础", "content": "Python 是一种编程语言", "category": "编程", "version": 1},
            {"id": 2, "title": "机器学习", "content": "机器学习是 AI 的分支", "category": "AI", "version": 1},
            {"id": 3, "title": "深度学习", "content": "深度学习使用神经网络", "category": "AI", "version": 1},
        ]

        texts = [f"{d['title']} {d['content']}" for d in documents]
        embeddings = self.model.encode(texts).tolist()
        timestamp = int(time.time())

        data = [
            [d["id"] for d in documents],
            [d["title"] for d in documents],
            [d["content"] for d in documents],
            [d["category"] for d in documents],
            [d["version"] for d in documents],
            [timestamp] * len(documents),
            embeddings
        ]

        self.collection.insert(data)
        self.collection.flush()
        self.collection.load()

        print(f"✓ 已插入 {len(documents)} 个文档")

    # ============================================================
    # 场景 1: 单条 Upsert
    # ============================================================

    def upsert_single_document(self, doc_id: int, title: str, content: str, category: str):
        """单条 Upsert"""
        print(f"\n【场景1】单条 Upsert: ID={doc_id}")

        # 查询当前版本
        existing = self.collection.query(
            expr=f"id == {doc_id}",
            output_fields=["id", "version"]
        )

        version = existing[0]["version"] + 1 if existing else 1

        # 生成 Embedding
        text = f"{title} {content}"
        embedding = self.model.encode([text])[0].tolist()
        timestamp = int(time.time())

        # Upsert
        data = [
            [doc_id],
            [title],
            [content],
            [category],
            [version],
            [timestamp],
            [embedding]
        ]

        self.collection.upsert(data)
        self.collection.flush()

        action = "更新" if existing else "插入"
        print(f"✓ 文档 ID={doc_id} 已{action} (版本: {version})")

    # ============================================================
    # 场景 2: 批量 Upsert
    # ============================================================

    def batch_upsert(self, documents: List[Dict]):
        """批量 Upsert"""
        print(f"\n【场景2】批量 Upsert: {len(documents)} 个文档")

        # 查询现有文档的版本
        ids = [d["id"] for d in documents]
        existing = self.collection.query(
            expr=f"id in {ids}",
            output_fields=["id", "version"]
        )

        # 构建版本映射
        version_map = {e["id"]: e["version"] for e in existing}

        # 准备数据
        texts = [f"{d['title']} {d['content']}" for d in documents]
        embeddings = self.model.encode(texts).tolist()
        timestamp = int(time.time())

        data = [
            [d["id"] for d in documents],
            [d["title"] for d in documents],
            [d["content"] for d in documents],
            [d["category"] for d in documents],
            [version_map.get(d["id"], 0) + 1 for d in documents],
            [timestamp] * len(documents),
            embeddings
        ]

        # Upsert
        self.collection.upsert(data)
        self.collection.flush()

        new_count = len(documents) - len(existing)
        update_count = len(existing)
        print(f"✓ 新增: {new_count}, 更新: {update_count}")

    # ============================================================
    # 场景 3: 部分字段更新
    # ============================================================

    def update_fields(self, doc_id: int, updates: Dict):
        """更新指定字段"""
        print(f"\n【场景3】部分字段更新: ID={doc_id}")
        print(f"  更新字段: {list(updates.keys())}")

        # 查询原始数据
        results = self.collection.query(
            expr=f"id == {doc_id}",
            output_fields=["*"]
        )

        if not results:
            print(f"✗ 文档 ID={doc_id} 不存在")
            return

        original = results[0]

        # 更新字段
        for field, value in updates.items():
            original[field] = value

        # 如果更新了 title 或 content，重新生成 Embedding
        if "title" in updates or "content" in updates:
            text = f"{original['title']} {original['content']}"
            original["embedding"] = self.model.encode([text])[0].tolist()

        # 更新版本和时间戳
        original["version"] += 1
        original["updated_at"] = int(time.time())

        # Upsert
        data = [
            [original["id"]],
            [original["title"]],
            [original["content"]],
            [original["category"]],
            [original["version"]],
            [original["updated_at"]],
            [original["embedding"]]
        ]

        self.collection.upsert(data)
        self.collection.flush()

        print(f"✓ 文档 ID={doc_id} 已更新 (版本: {original['version']})")

    # ============================================================
    # 场景 4: 条件更新
    # ============================================================

    def conditional_update(self, condition: str, updates: Dict):
        """条件更新"""
        print(f"\n【场景4】条件更新")
        print(f"  条件: {condition}")
        print(f"  更新: {updates}")

        # 查询符合条件的文档
        results = self.collection.query(
            expr=condition,
            output_fields=["*"]
        )

        if not results:
            print("✗ 没有符合条件的文档")
            return

        print(f"  找到 {len(results)} 个文档")

        # 更新每个文档
        for doc in results:
            # 更新字段
            for field, value in updates.items():
                doc[field] = value

            # 如果更新了 title 或 content，重新生成 Embedding
            if "title" in updates or "content" in updates:
                text = f"{doc['title']} {doc['content']}"
                doc["embedding"] = self.model.encode([text])[0].tolist()

            # 更新版本和时间戳
            doc["version"] += 1
            doc["updated_at"] = int(time.time())

        # 批量 Upsert
        data = [
            [doc["id"] for doc in results],
            [doc["title"] for doc in results],
            [doc["content"] for doc in results],
            [doc["category"] for doc in results],
            [doc["version"] for doc in results],
            [doc["updated_at"] for doc in results],
            [doc["embedding"] for doc in results]
        ]

        self.collection.upsert(data)
        self.collection.flush()

        print(f"✓ 已更新 {len(results)} 个文档")

    # ============================================================
    # 场景 5: 数据同步
    # ============================================================

    def sync_from_external_system(self, external_data: List[Dict]):
        """从外部系统同步数据"""
        print(f"\n【场景5】数据同步: {len(external_data)} 个文档")

        # 查询现有文档
        ids = [d["id"] for d in external_data]
        existing = self.collection.query(
            expr=f"id in {ids}",
            output_fields=["id", "version", "updated_at"]
        )

        # 构建映射
        existing_map = {e["id"]: e for e in existing}

        # 过滤需要更新的文档
        to_update = []
        for doc in external_data:
            doc_id = doc["id"]
            if doc_id not in existing_map:
                # 新文档
                to_update.append(doc)
            else:
                # 检查是否需要更新（基于时间戳）
                if doc.get("updated_at", 0) > existing_map[doc_id]["updated_at"]:
                    to_update.append(doc)

        if not to_update:
            print("✓ 所有文档都是最新的")
            return

        print(f"  需要同步 {len(to_update)} 个文档")

        # 生成 Embedding
        texts = [f"{d['title']} {d['content']}" for d in to_update]
        embeddings = self.model.encode(texts).tolist()

        # 准备数据
        data = [
            [d["id"] for d in to_update],
            [d["title"] for d in to_update],
            [d["content"] for d in to_update],
            [d["category"] for d in to_update],
            [existing_map.get(d["id"], {}).get("version", 0) + 1 for d in to_update],
            [d.get("updated_at", int(time.time())) for d in to_update],
            embeddings
        ]

        # Upsert
        self.collection.upsert(data)
        self.collection.flush()

        print(f"✓ 同步完成")

    # ============================================================
    # 场景 6: 冲突处理
    # ============================================================

    def upsert_with_conflict_resolution(self, doc_id: int, title: str, content: str,
                                       category: str, strategy: str = "last_write_wins"):
        """带冲突处理的 Upsert"""
        print(f"\n【场景6】冲突处理 Upsert: ID={doc_id}")
        print(f"  策略: {strategy}")

        # 查询现有文档
        existing = self.collection.query(
            expr=f"id == {doc_id}",
            output_fields=["*"]
        )

        if not existing:
            # 新文档，直接插入
            self.upsert_single_document(doc_id, title, content, category)
            return

        original = existing[0]

        if strategy == "last_write_wins":
            # 最后写入获胜（默认策略）
            self.upsert_single_document(doc_id, title, content, category)

        elif strategy == "version_check":
            # 版本检查（乐观锁）
            expected_version = original["version"]
            # 这里假设客户端传入了期望的版本号
            # 实际应用中需要从参数传入
            print(f"  当前版本: {expected_version}")
            self.upsert_single_document(doc_id, title, content, category)

        elif strategy == "merge":
            # 合并策略（保留部分旧数据）
            print(f"  合并旧数据和新数据")
            # 这里可以实现自定义的合并逻辑
            merged_content = f"{original['content']} {content}"
            self.upsert_single_document(doc_id, title, merged_content, category)

    # ============================================================
    # 场景 7: 性能优化
    # ============================================================

    def benchmark_upsert_performance(self):
        """Upsert 性能测试"""
        print(f"\n【场景7】Upsert 性能测试")
        print("=" * 60)

        # 测试 1: 单条 Upsert
        print("\n测试 1: 单条 Upsert (100次)")
        start = time.time()
        for i in range(100):
            self.upsert_single_document(
                doc_id=1000 + i,
                title=f"测试文档 {i}",
                content=f"这是测试内容 {i}",
                category="测试"
            )
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  平均延迟: {elapsed * 10:.2f}ms")

        # 测试 2: 批量 Upsert
        print("\n测试 2: 批量 Upsert (100个文档)")
        documents = [
            {
                "id": 2000 + i,
                "title": f"批量文档 {i}",
                "content": f"这是批量内容 {i}",
                "category": "批量测试"
            }
            for i in range(100)
        ]

        start = time.time()
        self.batch_upsert(documents)
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  吞吐量: {100 / elapsed:.2f} 文档/秒")

    # ============================================================
    # 工具方法
    # ============================================================

    def get_document(self, doc_id: int):
        """获取文档"""
        results = self.collection.query(
            expr=f"id == {doc_id}",
            output_fields=["id", "title", "content", "category", "version", "updated_at"]
        )

        if results:
            doc = results[0]
            print(f"\n文档 ID={doc_id}:")
            print(f"  标题: {doc['title']}")
            print(f"  内容: {doc['content']}")
            print(f"  类别: {doc['category']}")
            print(f"  版本: {doc['version']}")
            print(f"  更新时间: {datetime.fromtimestamp(doc['updated_at'])}")
            return doc
        else:
            print(f"\n文档 ID={doc_id} 不存在")
            return None

    def list_all_documents(self):
        """列出所有文档"""
        results = self.collection.query(
            expr="id >= 0",
            output_fields=["id", "title", "version"],
            limit=100
        )

        print(f"\n所有文档 (共 {len(results)} 个):")
        for doc in results:
            print(f"  ID={doc['id']}: {doc['title']} (v{doc['version']})")

        return results


def main():
    """主函数"""
    print("=" * 60)
    print("Milvus Upsert 与数据更新示例")
    print("=" * 60)

    # 初始化
    manager = DataUpdateManager()
    manager.connect()
    manager.create_collection()
    manager.load_model()
    manager.insert_initial_data()

    # 场景 1: 单条 Upsert
    print("\n" + "=" * 60)
    print("场景 1: 单条 Upsert")
    print("=" * 60)

    # 更新现有文档
    manager.upsert_single_document(
        doc_id=1,
        title="Python 高级编程",
        content="Python 高级特性包括装饰器、生成器和元类",
        category="编程"
    )

    # 插入新文档
    manager.upsert_single_document(
        doc_id=10,
        title="数据库设计",
        content="数据库设计是构建高效系统的基础",
        category="数据库"
    )

    # 验证
    manager.get_document(1)
    manager.get_document(10)

    # 场景 2: 批量 Upsert
    print("\n" + "=" * 60)
    print("场景 2: 批量 Upsert")
    print("=" * 60)

    documents = [
        {"id": 2, "title": "机器学习进阶", "content": "机器学习的高级算法和技术", "category": "AI"},
        {"id": 3, "title": "深度学习实战", "content": "深度学习在实际项目中的应用", "category": "AI"},
        {"id": 11, "title": "Web 开发", "content": "现代 Web 开发技术栈", "category": "编程"},
    ]

    manager.batch_upsert(documents)
    manager.list_all_documents()

    # 场景 3: 部分字段更新
    print("\n" + "=" * 60)
    print("场景 3: 部分字段更新")
    print("=" * 60)

    manager.update_fields(1, {"category": "高级编程"})
    manager.update_fields(2, {"content": "机器学习包括监督学习、无监督学习和强化学习"})
    manager.get_document(1)

    # 场景 4: 条件更新
    print("\n" + "=" * 60)
    print("场景 4: 条件更新")
    print("=" * 60)

    manager.conditional_update(
        condition="category == 'AI'",
        updates={"category": "人工智能"}
    )
    manager.list_all_documents()

    # 场景 5: 数据同步
    print("\n" + "=" * 60)
    print("场景 5: 数据同步")
    print("=" * 60)

    external_data = [
        {
            "id": 1,
            "title": "Python 专家级编程",
            "content": "Python 专家级技巧和最佳实践",
            "category": "高级编程",
            "updated_at": int(time.time()) + 100  # 更新的时间戳
        },
        {
            "id": 12,
            "title": "云计算",
            "content": "云计算技术和服务",
            "category": "云技术",
            "updated_at": int(time.time())
        }
    ]

    manager.sync_from_external_system(external_data)
    manager.list_all_documents()

    # 场景 6: 冲突处理
    print("\n" + "=" * 60)
    print("场景 6: 冲突处理")
    print("=" * 60)

    manager.upsert_with_conflict_resolution(
        doc_id=1,
        title="Python 大师级编程",
        content="Python 大师级技巧",
        category="高级编程",
        strategy="last_write_wins"
    )

    # 场景 7: 性能测试
    print("\n" + "=" * 60)
    print("场景 7: 性能测试")
    print("=" * 60)

    # manager.benchmark_upsert_performance()  # 取消注释以运行性能测试

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 核心概念

### 1. Upsert 的优势

**传统方式**：
```python
# 检查 → 删除 → 插入（3步）
existing = collection.query(expr=f"id == {id}")
if existing:
    collection.delete(expr=f"id == {id}")
collection.insert(data)
```

**Upsert 方式**：
```python
# 一步完成
collection.upsert(data)
```

### 2. 版本控制

```python
# 每次更新增加版本号
version = existing["version"] + 1 if existing else 1
```

### 3. 时间戳管理

```python
# 记录更新时间
updated_at = int(time.time())
```

---

## 最佳实践

### 1. 批量 Upsert 优于单条

```python
# 慢：单条 Upsert
for doc in documents:
    collection.upsert([doc])
    collection.flush()

# 快：批量 Upsert
collection.upsert(documents)
collection.flush()
```

### 2. 版本控制防止冲突

```python
# 乐观锁：检查版本号
if current_version != expected_version:
    raise ConflictError("版本冲突")
```

### 3. 只更新必要字段

```python
# 查询原始数据 → 更新字段 → Upsert
original = collection.query(expr=f"id == {id}")[0]
original["title"] = new_title
collection.upsert([original])
```

---

## 性能优化

### 1. 批量大小

- 小批量（< 100）：适合实时更新
- 中批量（100-1000）：平衡性能和延迟
- 大批量（> 1000）：适合批量同步

### 2. 减少 Embedding 计算

```python
# 只在 title 或 content 变化时重新计算
if "title" in updates or "content" in updates:
    embedding = model.encode([text])[0]
```

### 3. 异步更新

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(upsert_doc, doc) for doc in documents]
```

---

## 总结

### 核心要点

1. **Upsert 简化更新逻辑**：一步完成插入或更新
2. **版本控制防止冲突**：乐观锁机制
3. **批量操作提升性能**：批量 Upsert
4. **部分字段更新**：查询 → 修改 → Upsert
5. **数据同步**：基于时间戳判断

### 最佳实践

1. **使用批量 Upsert**：提升性能
2. **实现版本控制**：防止冲突
3. **记录时间戳**：追踪更新时间
4. **错误处理**：重试机制
5. **性能监控**：基准测试

---

**下一步**: 学习 [13_实战代码_场景5_RAG数据管理实战.md](./13_实战代码_场景5_RAG数据管理实战.md)
