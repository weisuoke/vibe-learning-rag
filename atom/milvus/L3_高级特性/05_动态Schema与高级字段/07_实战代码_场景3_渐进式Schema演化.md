# 实战代码场景3：渐进式Schema演化

> 本文档演示使用 AddCollectionField 实现渐进式 Schema 演化，支持向后兼容

---

## 一、场景描述

### 1.1 业务需求

**场景**：电商产品目录系统，随着业务发展需要逐步添加新的产品属性字段。

**核心挑战**：
- 初期只有基础字段（ID、向量、标题），后续需要添加价格、库存、分类等字段
- 已有数百万条产品数据，不能重新创建 Collection
- 新字段添加后，旧数据和新数据需要共存
- 需要保证系统持续可用，不能长时间停机

**业务价值**：
- 支持业务快速迭代（无需重建 Collection）
- 保证数据一致性（旧数据自动填充默认值）
- 降低运维成本（无需数据迁移）

### 1.2 技术目标

1. **AddCollectionField 工作流**：掌握字段添加的完整流程
2. **Nullable 字段处理**：理解 nullable 字段的必要性
3. **默认值设置**：为新字段设置合理的默认值
4. **向后兼容性**：确保旧数据和新数据可以共存
5. **生产环境部署**：掌握生产环境的部署策略

### 1.3 预期效果

- 成功添加新字段到已存在的 Collection
- 旧数据的新字段自动填充默认值
- 新数据可以正常插入和查询
- 系统持续可用，无长时间停机

---

## 二、技术方案

### 2.1 Schema 演化策略

```
初始 Schema (Version 1):
┌─────────────────────────────────┐
│ id: INT64 (主键)                │
│ vector: FLOAT_VECTOR (768维)   │
│ title: VARCHAR (产品标题)       │
└─────────────────────────────────┘

添加字段后 (Version 2):
┌─────────────────────────────────┐
│ id: INT64 (主键)                │
│ vector: FLOAT_VECTOR (768维)   │
│ title: VARCHAR (产品标题)       │
│ price: DOUBLE (价格, nullable)  │ ← 新增
│ stock: INT64 (库存, nullable)   │ ← 新增
└─────────────────────────────────┘

再次添加字段 (Version 3):
┌─────────────────────────────────┐
│ id: INT64 (主键)                │
│ vector: FLOAT_VECTOR (768维)   │
│ title: VARCHAR (产品标题)       │
│ price: DOUBLE (价格, nullable)  │
│ stock: INT64 (库存, nullable)   │
│ category: VARCHAR (分类, nullable) │ ← 新增
└─────────────────────────────────┘
```

### 2.2 AddCollectionField 工作流

```
步骤1：停止写入
  ↓
步骤2：添加字段（AddCollectionField）
  ↓
步骤3：重新加载 Collection
  ↓
步骤4：验证字段
  ↓
步骤5：恢复写入
```

### 2.3 向后兼容性保证

**旧数据处理**：
- 新字段值为 NULL（如果没有默认值）
- 新字段值为默认值（如果设置了默认值）

**新数据处理**：
- 可以提供新字段的值
- 也可以不提供（使用默认值或 NULL）

**查询处理**：
- 使用 `dict.get()` 安全访问新字段
- 过滤新字段时需要考虑 NULL 值

---

## 三、完整代码

### 3.1 环境准备

```python
"""
渐进式Schema演化示例

依赖：
- pymilvus >= 2.6.0
- numpy

安装：
uv add pymilvus numpy
"""

from pymilvus import MilvusClient, DataType
import numpy as np
from typing import List, Dict, Any
import time

# 配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "product_catalog"
VECTOR_DIM = 768
```

### 3.2 创建初始 Collection（Version 1）

```python
def create_initial_collection(client: MilvusClient) -> None:
    """
    创建初始Collection（只有基础字段）

    [来源: reference/context7_pymilvus_01.md]
    """
    print("=" * 60)
    print("步骤1：创建初始Collection（Version 1）")
    print("=" * 60)

    # 删除已存在的Collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧Collection: {COLLECTION_NAME}")

    # 创建Schema（最小化）
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # 只添加基础字段
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 创建索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    # 创建Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    print(f"✓ Collection创建成功: {COLLECTION_NAME}")
    print(f"✓ Schema Version: 1")
    print(f"✓ 字段: id, vector, title")
```

### 3.3 插入初始数据（Version 1）

```python
def insert_initial_data(client: MilvusClient) -> None:
    """
    插入初始数据（只有基础字段）

    [来源: reference/context7_pymilvus_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤2：插入初始数据（Version 1）")
    print("=" * 60)

    data = []
    for i in range(100):
        data.append({
            "id": i,
            "vector": np.random.randn(VECTOR_DIM).tolist(),
            "title": f"Product {i}"
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"✓ 插入 {len(data)} 条初始数据")
    print(f"✓ 字段: id, vector, title")
```

### 3.4 添加新字段（Version 2）

```python
def add_fields_v2(client: MilvusClient) -> None:
    """
    添加新字段：price, stock（Version 2）

    [来源: reference/context7_milvus_02.md, reference/search_动态Schema_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤3：添加新字段（Version 2）")
    print("=" * 60)

    # 注意：生产环境中应该停止写入
    print("\n3.1 停止写入（生产环境必需）")
    print("  - 在生产环境中，应该停止所有写入操作")
    print("  - 本示例跳过此步骤")

    # 添加 price 字段
    print("\n3.2 添加 price 字段")
    try:
        client.add_collection_field(
            collection_name=COLLECTION_NAME,
            field_name="price",
            data_type=DataType.DOUBLE,
            nullable=True,  # 必须为 True
            default_value=0.0  # 默认值
        )
        print(f"✓ 添加字段成功: price (DOUBLE, nullable, default=0.0)")
    except Exception as e:
        print(f"✗ 添加字段失败: {e}")

    # 添加 stock 字段
    print("\n3.3 添加 stock 字段")
    try:
        client.add_collection_field(
            collection_name=COLLECTION_NAME,
            field_name="stock",
            data_type=DataType.INT64,
            nullable=True,  # 必须为 True
            default_value=0  # 默认值
        )
        print(f"✓ 添加字段成功: stock (INT64, nullable, default=0)")
    except Exception as e:
        print(f"✗ 添加字段失败: {e}")

    # 重新加载 Collection
    print("\n3.4 重新加载 Collection")
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
        time.sleep(2)  # 等待释放完成
        client.load_collection(collection_name=COLLECTION_NAME)
        time.sleep(2)  # 等待加载完成
        print(f"✓ Collection重新加载成功")
    except Exception as e:
        print(f"✗ Collection重新加载失败: {e}")

    # 验证字段
    print("\n3.5 验证字段")
    schema = client.describe_collection(collection_name=COLLECTION_NAME)
    field_names = [field["name"] for field in schema["fields"]]
    print(f"✓ 当前字段: {', '.join(field_names)}")

    # 恢复写入
    print("\n3.6 恢复写入（生产环境必需）")
    print("  - 在生产环境中，应该恢复所有写入操作")
    print("  - 本示例跳过此步骤")
```

### 3.5 验证向后兼容性

```python
def verify_backward_compatibility(client: MilvusClient) -> None:
    """
    验证向后兼容性（旧数据和新数据共存）

    [来源: reference/search_动态Schema_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤4：验证向后兼容性")
    print("=" * 60)

    query_vector = np.random.randn(VECTOR_DIM).tolist()

    # 查询旧数据（Version 1）
    print("\n4.1 查询旧数据（Version 1）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='id < 10',  # 查询前10条旧数据
        output_fields=["id", "title", "price", "stock"],
        limit=5
    )

    print(f"找到 {len(results[0])} 条旧数据")
    for hit in results[0]:
        print(f"  ID: {hit['id']}, Title: {hit['title']}, "
              f"Price: {hit.get('price', 'N/A')}, Stock: {hit.get('stock', 'N/A')}")

    print("\n说明:")
    print("  - 旧数据的 price 和 stock 字段值为默认值（0.0 和 0）")
    print("  - 这是因为添加字段时设置了 default_value")
```

### 3.6 插入新数据（Version 2）

```python
def insert_new_data_v2(client: MilvusClient) -> None:
    """
    插入新数据（包含新字段）

    [来源: reference/context7_pymilvus_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤5：插入新数据（Version 2）")
    print("=" * 60)

    data = []
    for i in range(100, 120):
        data.append({
            "id": i,
            "vector": np.random.randn(VECTOR_DIM).tolist(),
            "title": f"Product {i}",
            # 新字段
            "price": 99.99 + i,
            "stock": 100 + i
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"✓ 插入 {len(data)} 条新数据")
    print(f"✓ 字段: id, vector, title, price, stock")

    # 查询新数据
    print("\n5.1 查询新数据（Version 2）")
    query_vector = np.random.randn(VECTOR_DIM).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='id >= 100 and id < 110',
        output_fields=["id", "title", "price", "stock"],
        limit=5
    )

    print(f"找到 {len(results[0])} 条新数据")
    for hit in results[0]:
        print(f"  ID: {hit['id']}, Title: {hit['title']}, "
              f"Price: {hit.get('price', 'N/A')}, Stock: {hit.get('stock', 'N/A')}")
```

### 3.7 再次添加字段（Version 3）

```python
def add_fields_v3(client: MilvusClient) -> None:
    """
    再次添加新字段：category（Version 3）

    [来源: reference/context7_milvus_02.md]
    """
    print("\n" + "=" * 60)
    print("步骤6：再次添加新字段（Version 3）")
    print("=" * 60)

    # 添加 category 字段
    print("\n6.1 添加 category 字段")
    try:
        client.add_collection_field(
            collection_name=COLLECTION_NAME,
            field_name="category",
            data_type=DataType.VARCHAR,
            max_length=128,
            nullable=True,
            default_value="Uncategorized"
        )
        print(f"✓ 添加字段成功: category (VARCHAR, nullable, default='Uncategorized')")
    except Exception as e:
        print(f"✗ 添加字段失败: {e}")

    # 重新加载 Collection
    print("\n6.2 重新加载 Collection")
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
        time.sleep(2)
        client.load_collection(collection_name=COLLECTION_NAME)
        time.sleep(2)
        print(f"✓ Collection重新加载成功")
    except Exception as e:
        print(f"✗ Collection重新加载失败: {e}")

    # 验证字段
    print("\n6.3 验证字段")
    schema = client.describe_collection(collection_name=COLLECTION_NAME)
    field_names = [field["name"] for field in schema["fields"]]
    print(f"✓ 当前字段: {', '.join(field_names)}")
```

### 3.8 验证多版本共存

```python
def verify_multi_version_coexistence(client: MilvusClient) -> None:
    """
    验证多版本数据共存

    [来源: reference/search_RAG多租户_03.md]
    """
    print("\n" + "=" * 60)
    print("步骤7：验证多版本数据共存")
    print("=" * 60)

    query_vector = np.random.randn(VECTOR_DIM).tolist()

    # 查询所有数据
    print("\n7.1 查询所有版本的数据")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["id", "title", "price", "stock", "category"],
        limit=15
    )

    v1_count = 0  # 只有基础字段
    v2_count = 0  # 有 price, stock
    v3_count = 0  # 有 category

    for hit in results[0]:
        price = hit.get('price', None)
        stock = hit.get('stock', None)
        category = hit.get('category', None)

        if price is not None and price != 0.0:
            if category is not None and category != "Uncategorized":
                v3_count += 1
                print(f"  ID: {hit['id']} - Version 3 (price={price:.2f}, stock={stock}, category={category})")
            else:
                v2_count += 1
                print(f"  ID: {hit['id']} - Version 2 (price={price:.2f}, stock={stock})")
        else:
            v1_count += 1
            print(f"  ID: {hit['id']} - Version 1 (只有基础字段)")

    print(f"\n统计:")
    print(f"  - Version 1数据: {v1_count}条（只有基础字段）")
    print(f"  - Version 2数据: {v2_count}条（有 price, stock）")
    print(f"  - Version 3数据: {v3_count}条（有 category）")
    print(f"  - 说明: 多版本数据可以共存，查询时安全处理缺失字段")
```

### 3.9 主函数

```python
def main():
    """主函数"""
    print("渐进式Schema演化示例")
    print("=" * 60)

    # 连接Milvus
    client = MilvusClient(uri=MILVUS_URI)
    print(f"✓ 已连接到Milvus: {MILVUS_URI}\n")

    try:
        # 1. 创建初始Collection（Version 1）
        create_initial_collection(client)

        # 2. 插入初始数据（Version 1）
        insert_initial_data(client)

        # 3. 添加新字段（Version 2）
        add_fields_v2(client)

        # 4. 验证向后兼容性
        verify_backward_compatibility(client)

        # 5. 插入新数据（Version 2）
        insert_new_data_v2(client)

        # 6. 再次添加字段（Version 3）
        add_fields_v3(client)

        # 7. 验证多版本共存
        verify_multi_version_coexistence(client)

        print("\n" + "=" * 60)
        print("✓ 所有操作完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client.close()


if __name__ == "__main__":
    main()
```

---

## 四、运行验证

### 4.1 环境准备

```bash
# 1. 确保Milvus服务运行
docker ps | grep milvus

# 2. 安装依赖
uv add pymilvus numpy

# 3. 激活环境
source .venv/bin/activate
```

### 4.2 执行步骤

```bash
# 运行示例
python 07_实战代码_场景3_渐进式Schema演化.py
```

### 4.3 预期输出

```
渐进式Schema演化示例
============================================================
✓ 已连接到Milvus: http://localhost:19530

============================================================
步骤1：创建初始Collection（Version 1）
============================================================
✓ Collection创建成功: product_catalog
✓ Schema Version: 1
✓ 字段: id, vector, title

============================================================
步骤2：插入初始数据（Version 1）
============================================================
✓ 插入 100 条初始数据
✓ 字段: id, vector, title

============================================================
步骤3：添加新字段（Version 2）
============================================================

3.1 停止写入（生产环境必需）
  - 在生产环境中，应该停止所有写入操作
  - 本示例跳过此步骤

3.2 添加 price 字段
✓ 添加字段成功: price (DOUBLE, nullable, default=0.0)

3.3 添加 stock 字段
✓ 添加字段成功: stock (INT64, nullable, default=0)

3.4 重新加载 Collection
✓ Collection重新加载成功

3.5 验证字段
✓ 当前字段: id, vector, title, price, stock

...

============================================================
✓ 所有操作完成
============================================================
```

### 4.4 故障排查

**问题1：AddCollectionField 失败**
```
错误: field must be nullable
解决: 确保 nullable=True
```

**问题2：查询新字段时断言错误**
```
错误: column != nullptr
解决: 重新加载 Collection，等待 schema 同步
```

**问题3：并发 Insert 冲突**
```
错误: collection schema mismatch
解决: 停止写入再添加字段
```

---

## 五、关键要点

### 5.1 核心发现

1. **AddCollectionField 必须 nullable**：新字段必须设置 `nullable=True`
2. **重新加载 Collection**：添加字段后必须重新加载 Collection
3. **默认值机制**：设置默认值可以确保旧数据的一致性
4. **向后兼容性**：旧数据和新数据可以共存

### 5.2 最佳实践

1. **使用维护窗口**：在低流量时段添加字段
2. **停止写入**：添加字段前停止所有写入操作
3. **重新加载**：添加字段后重新加载 Collection
4. **测试验证**：测试所有查询类型（Search, Query, HybridSearch）
5. **监控指标**：监控 AddCollectionField 操作耗时和错误率

### 5.3 生产建议

1. **适用场景**：需要渐进式添加字段、不能重建 Collection
2. **不适用场景**：需要添加向量字段、需要删除字段
3. **性能影响**：nullable 字段聚合性能慢 4 倍
4. **并发控制**：避免并发 insert 和 AddCollectionField

---

## 参考资料

- [来源: reference/context7_milvus_02.md] - AddCollectionField API 文档
- [来源: reference/search_动态Schema_01.md] - 生产环境问题（Issue #43003, #45318, #41858）
- [来源: reference/source_动态Schema_01.md] - Schema 版本管理机制
