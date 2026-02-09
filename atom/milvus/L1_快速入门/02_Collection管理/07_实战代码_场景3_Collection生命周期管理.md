# 实战代码 - 场景3：Collection 生命周期管理

## 场景描述

演示 Collection 的完整生命周期管理，包括：
- 创建多个 Collection
- 管理 Collection 状态（加载/释放）
- Collection 信息查询
- 安全删除 Collection
- 错误处理

## 完整代码

```python
"""
场景3：Collection 生命周期管理
演示：多 Collection 管理、状态管理、安全删除
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Optional
import random

# ===== 1. Collection 管理器类 =====
class CollectionManager:
    """Collection 生命周期管理器"""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        """初始化管理器并连接到 Milvus"""
        connections.connect(host=host, port=port)
        print(f"✅ 已连接到 Milvus ({host}:{port})")

    def list_all_collections(self) -> List[str]:
        """列出所有 Collection"""
        collections = utility.list_collections()
        return collections

    def collection_exists(self, name: str) -> bool:
        """检查 Collection 是否存在"""
        return utility.has_collection(name)

    def get_collection_info(self, name: str) -> Dict:
        """获取 Collection 详细信息"""
        if not self.collection_exists(name):
            return {"error": f"Collection '{name}' 不存在"}

        collection = Collection(name)

        info = {
            "name": collection.name,
            "description": collection.description,
            "num_entities": collection.num_entities,
            "is_empty": collection.is_empty,
            "num_fields": len(collection.schema.fields),
            "fields": []
        }

        # 获取字段信息
        for field in collection.schema.fields:
            field_info = {
                "name": field.name,
                "type": str(field.dtype),
                "is_primary": field.is_primary
            }

            if field.dtype == DataType.FLOAT_VECTOR:
                field_info["dim"] = field.params.get("dim")
            elif field.dtype == DataType.VARCHAR:
                field_info["max_length"] = field.params.get("max_length")

            info["fields"].append(field_info)

        # 获取加载状态
        try:
            load_state = utility.load_state(name)
            info["load_state"] = str(load_state)
        except Exception as e:
            info["load_state"] = "unknown"

        return info

    def create_collection_safe(
        self,
        name: str,
        schema: CollectionSchema,
        overwrite: bool = False
    ) -> Optional[Collection]:
        """安全创建 Collection"""

        # 检查是否已存在
        if self.collection_exists(name):
            if overwrite:
                print(f"⚠️  Collection '{name}' 已存在，删除并重建")
                utility.drop_collection(name)
            else:
                print(f"⚠️  Collection '{name}' 已存在，返回现有 Collection")
                return Collection(name)

        # 创建新 Collection
        collection = Collection(name=name, schema=schema)
        print(f"✅ Collection '{name}' 创建成功")
        return collection

    def load_collection(self, name: str) -> bool:
        """加载 Collection 到内存"""
        if not self.collection_exists(name):
            print(f"❌ Collection '{name}' 不存在")
            return False

        try:
            collection = Collection(name)
            collection.load()
            print(f"✅ Collection '{name}' 已加载到内存")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def release_collection(self, name: str) -> bool:
        """释放 Collection 从内存"""
        if not self.collection_exists(name):
            print(f"❌ Collection '{name}' 不存在")
            return False

        try:
            collection = Collection(name)
            collection.release()
            print(f"✅ Collection '{name}' 已从内存释放")
            return True
        except Exception as e:
            print(f"❌ 释放失败: {e}")
            return False

    def drop_collection_safe(self, name: str, confirm: bool = False) -> bool:
        """安全删除 Collection"""
        if not self.collection_exists(name):
            print(f"⚠️  Collection '{name}' 不存在")
            return False

        # 获取 Collection 信息
        collection = Collection(name)
        num_entities = collection.num_entities

        # 显示警告信息
        print(f"\n⚠️  警告：即将删除 Collection '{name}'")
        print(f"   - 数据量: {num_entities} 条")
        print(f"   - 字段数: {len(collection.schema.fields)}")

        if not confirm:
            user_input = input("\n确认删除？(yes/no): ")
            if user_input.lower() != "yes":
                print("❌ 取消删除")
                return False

        # 执行删除
        try:
            utility.drop_collection(name)
            print(f"✅ Collection '{name}' 已删除")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False

    def print_all_collections(self):
        """打印所有 Collection 的信息"""
        collections = self.list_all_collections()

        if not collections:
            print("📭 没有 Collection")
            return

        print(f"\n📚 共有 {len(collections)} 个 Collection:\n")

        for name in collections:
            info = self.get_collection_info(name)
            print(f"Collection: {name}")
            print(f"  - 描述: {info.get('description', 'N/A')}")
            print(f"  - 数据量: {info.get('num_entities', 0)}")
            print(f"  - 字段数: {info.get('num_fields', 0)}")
            print(f"  - 加载状态: {info.get('load_state', 'unknown')}")
            print()


# ===== 2. 主程序 =====
def main():
    print("=" * 60)
    print("场景3：Collection 生命周期管理")
    print("=" * 60)

    # 创建管理器
    manager = CollectionManager()

    # ===== 步骤1：创建多个 Collection =====
    print("\n" + "=" * 60)
    print("步骤1：创建多个 Collection")
    print("=" * 60)

    # Collection 1: 文档检索
    doc_schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200)
        ],
        description="文档检索 Collection"
    )

    doc_collection = manager.create_collection_safe(
        name="documents",
        schema=doc_schema,
        overwrite=True
    )

    # Collection 2: 图片检索
    image_schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500)
        ],
        description="图片检索 Collection"
    )

    image_collection = manager.create_collection_safe(
        name="images",
        schema=image_schema,
        overwrite=True
    )

    # Collection 3: 用户画像
    user_schema = CollectionSchema(
        fields=[
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
            FieldSchema(name="profile_vector", dtype=DataType.FLOAT_VECTOR, dim=256),
            FieldSchema(name="age", dtype=DataType.INT8)
        ],
        description="用户画像 Collection"
    )

    user_collection = manager.create_collection_safe(
        name="users",
        schema=user_schema,
        overwrite=True
    )

    # ===== 步骤2：插入数据 =====
    print("\n" + "=" * 60)
    print("步骤2：插入数据到各个 Collection")
    print("=" * 60)

    # 插入文档数据
    doc_data = [
        {
            "id": i,
            "embedding": [random.random() for _ in range(128)],
            "title": f"文档 {i}"
        }
        for i in range(50)
    ]
    doc_collection.insert(doc_data)
    doc_collection.flush()
    print(f"✅ documents: 插入了 {len(doc_data)} 条数据")

    # 插入图片数据
    image_data = [
        {
            "id": i,
            "embedding": [random.random() for _ in range(512)],
            "url": f"https://example.com/image_{i}.jpg"
        }
        for i in range(30)
    ]
    image_collection.insert(image_data)
    image_collection.flush()
    print(f"✅ images: 插入了 {len(image_data)} 条数据")

    # 插入用户数据
    user_data = [
        {
            "user_id": f"USER_{i:04d}",
            "profile_vector": [random.random() for _ in range(256)],
            "age": random.randint(18, 60)
        }
        for i in range(20)
    ]
    user_collection.insert(user_data)
    user_collection.flush()
    print(f"✅ users: 插入了 {len(user_data)} 条数据")

    # ===== 步骤3：创建索引 =====
    print("\n" + "=" * 60)
    print("步骤3：为各个 Collection 创建索引")
    print("=" * 60)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }

    doc_collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ documents: 索引创建成功")

    image_collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ images: 索引创建成功")

    user_collection.create_index(field_name="profile_vector", index_params=index_params)
    print("✅ users: 索引创建成功")

    # ===== 步骤4：查看所有 Collection =====
    print("\n" + "=" * 60)
    print("步骤4：查看所有 Collection 信息")
    print("=" * 60)

    manager.print_all_collections()

    # ===== 步骤5：加载 Collection =====
    print("=" * 60)
    print("步骤5：加载 Collection 到内存")
    print("=" * 60)

    manager.load_collection("documents")
    manager.load_collection("images")
    # users Collection 暂不加载

    # ===== 步骤6：检查加载状态 =====
    print("\n" + "=" * 60)
    print("步骤6：检查 Collection 加载状态")
    print("=" * 60)

    for name in ["documents", "images", "users"]:
        load_state = utility.load_state(name)
        print(f"{name}: {load_state}")

    # ===== 步骤7：执行检索（仅加载的 Collection）=====
    print("\n" + "=" * 60)
    print("步骤7：执行检索")
    print("=" * 60)

    # 检索 documents
    query_vector = [[random.random() for _ in range(128)]]
    results = doc_collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["title"]
    )

    print("documents 检索结果:")
    for hit in results[0]:
        print(f"  - ID: {hit.id}, 标题: {hit.entity.get('title')}")

    # 尝试检索未加载的 Collection
    print("\n尝试检索未加载的 Collection (users):")
    try:
        query_vector = [[random.random() for _ in range(256)]]
        results = user_collection.search(
            data=query_vector,
            anns_field="profile_vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3
        )
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("   提示：Collection 未加载，无法检索")

    # ===== 步骤8：释放 Collection =====
    print("\n" + "=" * 60)
    print("步骤8：释放 Collection")
    print("=" * 60)

    manager.release_collection("documents")
    manager.release_collection("images")

    # ===== 步骤9：再次查看加载状态 =====
    print("\n" + "=" * 60)
    print("步骤9：再次查看加载状态")
    print("=" * 60)

    for name in ["documents", "images", "users"]:
        load_state = utility.load_state(name)
        print(f"{name}: {load_state}")

    # ===== 步骤10：删除 Collection =====
    print("\n" + "=" * 60)
    print("步骤10：删除 Collection")
    print("=" * 60)

    # 自动确认删除（演示用）
    manager.drop_collection_safe("images", confirm=True)

    # 需要用户确认删除（实际使用）
    # manager.drop_collection_safe("documents", confirm=False)

    # ===== 步骤11：最终状态 =====
    print("\n" + "=" * 60)
    print("步骤11：最终 Collection 列表")
    print("=" * 60)

    manager.print_all_collections()

    print("\n" + "=" * 60)
    print("🎉 场景3完成！")
    print("=" * 60)


# ===== 3. 运行主程序 =====
if __name__ == "__main__":
    main()
```

## 运行输出示例

```
============================================================
场景3：Collection 生命周期管理
============================================================
✅ 已连接到 Milvus (localhost:19530)

============================================================
步骤1：创建多个 Collection
============================================================
✅ Collection 'documents' 创建成功
✅ Collection 'images' 创建成功
✅ Collection 'users' 创建成功

============================================================
步骤2：插入数据到各个 Collection
============================================================
✅ documents: 插入了 50 条数据
✅ images: 插入了 30 条数据
✅ users: 插入了 20 条数据

============================================================
步骤3：为各个 Collection 创建索引
============================================================
✅ documents: 索引创建成功
✅ images: 索引创建成功
✅ users: 索引创建成功

============================================================
步骤4：查看所有 Collection 信息
============================================================

📚 共有 3 个 Collection:

Collection: documents
  - 描述: 文档检索 Collection
  - 数据量: 50
  - 字段数: 3
  - 加载状态: LoadState.NotLoad

Collection: images
  - 描述: 图片检索 Collection
  - 数据量: 30
  - 字段数: 3
  - 加载状态: LoadState.NotLoad

Collection: users
  - 描述: 用户画像 Collection
  - 数据量: 20
  - 字段数: 3
  - 加载状态: LoadState.NotLoad

============================================================
步骤5：加载 Collection 到内存
============================================================
✅ Collection 'documents' 已加载到内存
✅ Collection 'images' 已加载到内存

============================================================
步骤6：检查 Collection 加载状态
============================================================
documents: LoadState.Loaded
images: LoadState.Loaded
users: LoadState.NotLoad

============================================================
步骤7：执行检索
============================================================
documents 检索结果:
  - ID: 23, 标题: 文档 23
  - ID: 45, 标题: 文档 45
  - ID: 12, 标题: 文档 12

尝试检索未加载的 Collection (users):
❌ 错误: collection not loaded
   提示：Collection 未加载，无法检索

============================================================
步骤8：释放 Collection
============================================================
✅ Collection 'documents' 已从内存释放
✅ Collection 'images' 已从内存释放

============================================================
步骤9：再次查看加载状态
============================================================
documents: LoadState.NotLoad
images: LoadState.NotLoad
users: LoadState.NotLoad

============================================================
步骤10：删除 Collection
============================================================

⚠️  警告：即将删除 Collection 'images'
   - 数据量: 30 条
   - 字段数: 3

✅ Collection 'images' 已删除

============================================================
步骤11：最终 Collection 列表
============================================================

📚 共有 2 个 Collection:

Collection: documents
  - 描述: 文档检索 Collection
  - 数据量: 50
  - 字段数: 3
  - 加载状态: LoadState.NotLoad

Collection: users
  - 描述: 用户画像 Collection
  - 数据量: 20
  - 字段数: 3
  - 加载状态: LoadState.NotLoad

============================================================
🎉 场景3完成！
============================================================
```

## 关键要点

1. **Collection 管理器**：封装常用操作，提高代码复用性
2. **状态管理**：加载/释放 Collection，管理内存使用
3. **安全删除**：删除前显示警告信息，需要用户确认
4. **错误处理**：捕获异常，提供友好的错误提示
5. **批量管理**：同时管理多个 Collection

## 最佳实践

1. **使用管理器类**：封装 Collection 操作，便于维护
2. **检查存在性**：操作前检查 Collection 是否存在
3. **状态查询**：定期检查 Collection 的加载状态
4. **安全删除**：删除前确认，避免误删数据
5. **资源管理**：及时释放不用的 Collection，节省内存
