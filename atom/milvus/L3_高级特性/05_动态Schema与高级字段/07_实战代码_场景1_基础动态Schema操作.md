# 实战代码场景1：基础动态Schema操作

> 本文档演示 Milvus 2.6 动态 Schema 的基础操作，包括创建、插入、查询和过滤动态字段

---

## 一、场景描述

### 1.1 业务需求

**场景**：构建一个文档管理系统，需要存储来自不同来源的文档（PDF、网页、邮件），每种文档类型有不同的元数据字段。

**核心挑战**：
- 不同文档类型的元数据字段不同（PDF有页码，网页有URL，邮件有发件人）
- 需求快速迭代，经常需要添加新的元数据字段
- 不希望为每种文档类型创建单独的 Collection

**传统方案的问题**：
- 固定 Schema 需要预先定义所有字段
- 添加新字段需要使用 AddCollectionField，需要停机维护
- 为每种文档类型创建单独 Collection 导致管理复杂

**动态 Schema 解决方案**：
- 启用 `enable_dynamic_field=True`
- 不同文档可以有不同的动态字段
- 无需修改 Schema 即可添加新字段
- 所有文档存储在同一个 Collection

### 1.2 技术目标

1. **创建动态 Schema Collection**：启用动态字段支持
2. **插入多样化数据**：插入带有不同动态字段的文档
3. **查询动态字段**：查询并返回动态字段数据
4. **过滤动态字段**：使用动态字段进行过滤查询
5. **验证存储机制**：验证动态字段存储在 `$meta` 中

### 1.3 预期效果

- Collection 创建成功，动态字段启用
- 不同文档类型可以插入不同的动态字段
- 查询时可以访问动态字段
- 过滤动态字段时查询正常工作
- 性能满足基本需求（小规模数据）

---

## 二、技术方案

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ PDF文档  │  │ 网页文档  │  │ 邮件文档  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Milvus Collection                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  固定字段：                                      │   │
│  │  - id (INT64, 主键)                             │   │
│  │  - vector (FLOAT_VECTOR, 768维)                │   │
│  │  - text (VARCHAR, 文档内容)                     │   │
│  │  - source (VARCHAR, 来源类型)                   │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  动态字段（存储在 $meta 中）：                   │   │
│  │  - PDF: page_number, total_pages, file_path    │   │
│  │  - 网页: url, crawled_at, domain               │   │
│  │  - 邮件: sender, subject, received_at          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流设计

**插入流程**：
```
1. 应用层准备文档数据（包含固定字段 + 动态字段）
2. Milvus 接收数据
3. 固定字段存储在对应列中
4. 动态字段自动存储在 $meta 字段（JSON 格式）
5. 返回插入成功
```

**查询流程**：
```
1. 应用层发起查询（指定 output_fields 包含动态字段）
2. Milvus 执行向量检索
3. 从 $meta 字段中提取动态字段
4. 返回结果（包含固定字段 + 动态字段）
```

**过滤流程**：
```
1. 应用层发起过滤查询（filter 包含动态字段）
2. Milvus 解析过滤表达式
3. 扫描 $meta 字段进行过滤（全表扫描）
4. 返回符合条件的结果
```

### 2.3 Schema 设计

**固定字段**（高频访问，需要索引）：
- `id`: INT64, 主键
- `vector`: FLOAT_VECTOR, 768维
- `text`: VARCHAR, 文档内容
- `source`: VARCHAR, 来源类型（pdf/web/email）

**动态字段**（低频访问，灵活扩展）：
- PDF 文档：`page_number`, `total_pages`, `file_path`
- 网页文档：`url`, `crawled_at`, `domain`
- 邮件文档：`sender`, `subject`, `received_at`

### 2.4 关键技术点

**1. 动态字段启用**：
```python
schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=True  # 关键配置
)
```

**2. 动态字段插入**：
```python
data = {
    "id": 1,
    "vector": [0.1] * 768,
    "text": "Document content",
    "source": "pdf",
    # 动态字段（无需预先定义）
    "page_number": 5,
    "total_pages": 100
}
```

**3. 动态字段查询**：
```python
results = client.search(
    collection_name="documents",
    data=[query_vector],
    output_fields=["text", "source", "page_number"],  # 包含动态字段
    limit=10
)
```

**4. 动态字段过滤**：
```python
results = client.search(
    collection_name="documents",
    data=[query_vector],
    filter='source == "pdf" and page_number > 5',  # 过滤动态字段
    limit=10
)
```

---

## 三、完整代码

### 3.1 环境准备

```python
"""
基础动态Schema操作示例

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
COLLECTION_NAME = "dynamic_documents"
VECTOR_DIM = 768
```

### 3.2 创建动态Schema Collection

```python
def create_dynamic_collection(client: MilvusClient) -> None:
    """
    创建启用动态Schema的Collection

    [来源: reference/context7_pymilvus_01.md]
    """
    print("=" * 60)
    print("步骤1：创建动态Schema Collection")
    print("=" * 60)

    # 删除已存在的Collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除旧Collection: {COLLECTION_NAME}")

    # 创建Schema（启用动态字段）
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True  # 关键：启用动态字段
    )

    # 添加固定字段
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True
    )

    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM
    )

    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535
    )

    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=32
    )

    # 创建索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    index_params.add_index(
        field_name="source",
        index_type="INVERTED"  # 为固定字段创建索引
    )

    # 创建Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    print(f"✓ Collection创建成功: {COLLECTION_NAME}")
    print(f"✓ 动态字段已启用")
    print(f"✓ 固定字段: id, vector, text, source")
    print(f"✓ 动态字段: 任意字段（自动存储在$meta中）")
```

### 3.3 插入多样化数据

```python
def insert_diverse_documents(client: MilvusClient) -> None:
    """
    插入不同类型的文档（带有不同的动态字段）

    [来源: reference/context7_pymilvus_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤2：插入多样化数据")
    print("=" * 60)

    data = []

    # PDF 文档（带有 page_number, total_pages, file_path 动态字段）
    for i in range(10):
        data.append({
            "id": i,
            "vector": np.random.randn(VECTOR_DIM).tolist(),
            "text": f"PDF document content {i}",
            "source": "pdf",
            # 动态字段
            "page_number": i % 10 + 1,
            "total_pages": 100,
            "file_path": f"/documents/pdf_{i}.pdf"
        })

    # 网页文档（带有 url, crawled_at, domain 动态字段）
    for i in range(10, 20):
        data.append({
            "id": i,
            "vector": np.random.randn(VECTOR_DIM).tolist(),
            "text": f"Web page content {i}",
            "source": "web",
            # 动态字段（与PDF不同）
            "url": f"https://example.com/page_{i}",
            "crawled_at": "2026-02-25",
            "domain": "example.com"
        })

    # 邮件文档（带有 sender, subject, received_at 动态字段）
    for i in range(20, 30):
        data.append({
            "id": i,
            "vector": np.random.randn(VECTOR_DIM).tolist(),
            "text": f"Email content {i}",
            "source": "email",
            # 动态字段（与PDF和网页都不同）
            "sender": f"user{i}@example.com",
            "subject": f"Email subject {i}",
            "received_at": "2026-02-25T10:00:00Z"
        })

    # 插入数据
    client.insert(collection_name=COLLECTION_NAME, data=data)

    print(f"✓ 插入 {len(data)} 条文档")
    print(f"  - PDF文档: 10条（动态字段: page_number, total_pages, file_path）")
    print(f"  - 网页文档: 10条（动态字段: url, crawled_at, domain）")
    print(f"  - 邮件文档: 10条（动态字段: sender, subject, received_at）")
```

### 3.4 查询动态字段

```python
def query_dynamic_fields(client: MilvusClient) -> None:
    """
    查询并返回动态字段

    [来源: reference/context7_pymilvus_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤3：查询动态字段")
    print("=" * 60)

    # 准备查询向量
    query_vector = np.random.randn(VECTOR_DIM).tolist()

    # 查询PDF文档（包含PDF特有的动态字段）
    print("\n3.1 查询PDF文档（包含动态字段）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "pdf"',
        output_fields=["text", "source", "page_number", "total_pages", "file_path"],
        limit=3
    )

    for i, hit in enumerate(results[0]):
        print(f"\n结果 {i+1}:")
        print(f"  ID: {hit['id']}")
        print(f"  Text: {hit['text']}")
        print(f"  Source: {hit['source']}")
        print(f"  Page Number: {hit.get('page_number', 'N/A')}")  # 动态字段
        print(f"  Total Pages: {hit.get('total_pages', 'N/A')}")  # 动态字段
        print(f"  File Path: {hit.get('file_path', 'N/A')}")      # 动态字段
        print(f"  Distance: {hit['distance']:.4f}")

    # 查询网页文档（包含网页特有的动态字段）
    print("\n3.2 查询网页文档（包含动态字段）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "web"',
        output_fields=["text", "source", "url", "crawled_at", "domain"],
        limit=3
    )

    for i, hit in enumerate(results[0]):
        print(f"\n结果 {i+1}:")
        print(f"  ID: {hit['id']}")
        print(f"  Text: {hit['text']}")
        print(f"  Source: {hit['source']}")
        print(f"  URL: {hit.get('url', 'N/A')}")              # 动态字段
        print(f"  Crawled At: {hit.get('crawled_at', 'N/A')}")  # 动态字段
        print(f"  Domain: {hit.get('domain', 'N/A')}")        # 动态字段
        print(f"  Distance: {hit['distance']:.4f}")

    # 查询邮件文档（包含邮件特有的动态字段）
    print("\n3.3 查询邮件文档（包含动态字段）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "email"',
        output_fields=["text", "source", "sender", "subject", "received_at"],
        limit=3
    )

    for i, hit in enumerate(results[0]):
        print(f"\n结果 {i+1}:")
        print(f"  ID: {hit['id']}")
        print(f"  Text: {hit['text']}")
        print(f"  Source: {hit['source']}")
        print(f"  Sender: {hit.get('sender', 'N/A')}")        # 动态字段
        print(f"  Subject: {hit.get('subject', 'N/A')}")      # 动态字段
        print(f"  Received At: {hit.get('received_at', 'N/A')}")  # 动态字段
        print(f"  Distance: {hit['distance']:.4f}")
```

### 3.5 过滤动态字段

```python
def filter_dynamic_fields(client: MilvusClient) -> None:
    """
    使用动态字段进行过滤查询

    [来源: reference/context7_pymilvus_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤4：过滤动态字段")
    print("=" * 60)

    query_vector = np.random.randn(VECTOR_DIM).tolist()

    # 过滤PDF文档（page_number > 5）
    print("\n4.1 过滤PDF文档（page_number > 5）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "pdf" and page_number > 5',  # 过滤动态字段
        output_fields=["text", "page_number"],
        limit=5
    )

    print(f"找到 {len(results[0])} 条结果")
    for hit in results[0]:
        print(f"  ID: {hit['id']}, Page: {hit.get('page_number', 'N/A')}")

    # 过滤网页文档（domain == "example.com"）
    print("\n4.2 过滤网页文档（domain == 'example.com'）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "web" and domain == "example.com"',  # 过滤动态字段
        output_fields=["text", "domain"],
        limit=5
    )

    print(f"找到 {len(results[0])} 条结果")
    for hit in results[0]:
        print(f"  ID: {hit['id']}, Domain: {hit.get('domain', 'N/A')}")

    # 复合过滤（多个动态字段）
    print("\n4.3 复合过滤（source == 'pdf' and page_number >= 5 and page_number <= 8）")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "pdf" and page_number >= 5 and page_number <= 8',
        output_fields=["text", "page_number", "total_pages"],
        limit=10
    )

    print(f"找到 {len(results[0])} 条结果")
    for hit in results[0]:
        print(f"  ID: {hit['id']}, Page: {hit.get('page_number', 'N/A')}/{hit.get('total_pages', 'N/A')}")
```

### 3.6 验证存储机制

```python
def verify_storage_mechanism(client: MilvusClient) -> None:
    """
    验证动态字段存储在$meta中

    [来源: reference/source_动态Schema_01.md]
    """
    print("\n" + "=" * 60)
    print("步骤5：验证存储机制")
    print("=" * 60)

    # 查询所有字段（包括$meta）
    print("\n5.1 查询所有字段（使用 output_fields=['*']）")
    query_vector = np.random.randn(VECTOR_DIM).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter='source == "pdf"',
        output_fields=["*"],  # 返回所有字段
        limit=1
    )

    if results[0]:
        hit = results[0][0]
        print(f"\n文档 ID: {hit['id']}")
        print(f"固定字段:")
        print(f"  - text: {hit['text']}")
        print(f"  - source: {hit['source']}")
        print(f"\n动态字段（自动从$meta中提取）:")
        print(f"  - page_number: {hit.get('page_number', 'N/A')}")
        print(f"  - total_pages: {hit.get('total_pages', 'N/A')}")
        print(f"  - file_path: {hit.get('file_path', 'N/A')}")

        print(f"\n说明：")
        print(f"  - 动态字段实际存储在隐式的$meta字段中（JSON格式）")
        print(f"  - 查询时Milvus自动从$meta中提取动态字段")
        print(f"  - 用户无需手动操作$meta字段")
```

### 3.7 主函数

```python
def main():
    """主函数"""
    print("Milvus 动态Schema基础操作示例")
    print("=" * 60)

    # 连接Milvus
    client = MilvusClient(uri=MILVUS_URI)
    print(f"✓ 已连接到Milvus: {MILVUS_URI}\n")

    try:
        # 1. 创建动态Schema Collection
        create_dynamic_collection(client)

        # 2. 插入多样化数据
        insert_diverse_documents(client)

        # 3. 查询动态字段
        query_dynamic_fields(client)

        # 4. 过滤动态字段
        filter_dynamic_fields(client)

        # 5. 验证存储机制
        verify_storage_mechanism(client)

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
python 07_实战代码_场景1_基础动态Schema操作.py
```

### 4.3 预期输出

```
Milvus 动态Schema基础操作示例
============================================================
✓ 已连接到Milvus: http://localhost:19530

============================================================
步骤1：创建动态Schema Collection
============================================================
已删除旧Collection: dynamic_documents
✓ Collection创建成功: dynamic_documents
✓ 动态字段已启用
✓ 固定字段: id, vector, text, source
✓ 动态字段: 任意字段（自动存储在$meta中）

============================================================
步骤2：插入多样化数据
============================================================
✓ 插入 30 条文档
  - PDF文档: 10条（动态字段: page_number, total_pages, file_path）
  - 网页文档: 10条（动态字段: url, crawled_at, domain）
  - 邮件文档: 10条（动态字段: sender, subject, received_at）

============================================================
步骤3：查询动态字段
============================================================

3.1 查询PDF文档（包含动态字段）

结果 1:
  ID: 5
  Text: PDF document content 5
  Source: pdf
  Page Number: 6
  Total Pages: 100
  File Path: /documents/pdf_5.pdf
  Distance: 0.1234

...

============================================================
✓ 所有操作完成
============================================================
```

### 4.4 故障排查

**问题1：连接失败**
```
错误: Failed to connect to Milvus
解决: 检查Milvus服务是否运行，端口是否正确
```

**问题2：动态字段返回None**
```
原因: 该文档没有该动态字段
解决: 使用 hit.get('field_name', 'N/A') 安全访问
```

**问题3：过滤动态字段性能慢**
```
原因: 动态字段无法创建索引，需要全表扫描
解决: 将高频过滤字段定义为固定字段
```

---

## 五、关键要点

### 5.1 核心发现

1. **动态字段启用简单**：只需设置 `enable_dynamic_field=True`
2. **插入灵活**：不同文档可以有不同的动态字段
3. **查询透明**：动态字段和固定字段查询方式相同
4. **存储自动**：动态字段自动存储在 `$meta` 字段中

### 5.2 最佳实践

1. **核心字段固定**：将高频访问字段定义为固定字段
2. **业务字段动态**：将低频、多样化字段定义为动态字段
3. **安全访问**：使用 `dict.get()` 方法安全访问动态字段
4. **性能权衡**：动态字段无法创建索引，过滤性能较低

### 5.3 生产建议

1. **适用场景**：元数据不确定、多租户系统、快速迭代项目
2. **不适用场景**：高频聚合查询、固定Schema需求
3. **性能优化**：为固定字段创建索引，减少动态字段过滤
4. **监控指标**：查询延迟、内存占用、动态字段数量

---

## 参考资料

- [来源: reference/context7_pymilvus_01.md] - PyMilvus API 文档
- [来源: reference/source_动态Schema_01.md] - 动态Schema源码分析
- [来源: reference/search_动态Schema_01.md] - 生产环境问题和最佳实践
