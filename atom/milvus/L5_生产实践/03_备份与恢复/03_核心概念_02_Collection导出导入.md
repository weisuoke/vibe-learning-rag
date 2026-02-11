# 核心概念2：Collection 导出导入

> 手动数据迁移方案的完整指南

---

## 什么是 Collection 导出导入？

**Collection 导出导入** 是一种手动的数据备份和迁移方案，通过将 Collection 数据导出为文件，然后在目标环境导入。

**核心特性：**
- ✅ 灵活性高，可自定义格式
- ✅ 支持数据转换和清洗
- ✅ 跨版本、跨集群迁移
- ✅ 不依赖额外工具
- ✅ 适合一次性迁移

**与 Milvus Backup 的对比：**

| 维度 | Collection 导出导入 | Milvus Backup |
|------|-------------------|---------------|
| **易用性** | 需要编写脚本 | 开箱即用 |
| **灵活性** | 高（可自定义） | 中（固定格式） |
| **性能** | 中 | 高 |
| **索引备份** | 不支持 | 支持（可选） |
| **增量备份** | 需要自己实现 | 内置支持 |
| **适用场景** | 数据迁移、转换 | 定期备份 |

---

## 导出方案

### 1. 导出格式选择

**JSON 格式：**
- 优点：易读、易调试、跨语言
- 缺点：文件大、可能损失精度
- 适用：小数据量、调试

**Parquet 格式：**
- 优点：高效、保留精度、支持压缩
- 缺点：需要额外库
- 适用：大数据量、生产环境

**NumPy 格式：**
- 优点：精度高、速度快
- 缺点：只能存储向量
- 适用：纯向量数据

### 2. 完整导出方案

```python
import json
import pandas as pd
import numpy as np
from pymilvus import Collection, connections
from typing import Dict, List, Any
import os

class CollectionExporter:
    """Collection 导出器"""

    def __init__(self, host="localhost", port="19530"):
        """初始化连接"""
        connections.connect(host=host, port=port)

    def export_schema(self, collection_name: str, output_file: str):
        """导出 Schema"""
        collection = Collection(collection_name)
        schema = collection.schema

        schema_dict = {
            "collection_name": collection_name,
            "description": schema.description,
            "fields": []
        }

        # 导出字段定义
        for field in schema.fields:
            field_dict = {
                "name": field.name,
                "dtype": str(field.dtype),
                "description": field.description,
                "is_primary": field.is_primary,
                "auto_id": field.auto_id
            }

            # 向量字段的维度
            if hasattr(field, "params") and "dim" in field.params:
                field_dict["dim"] = field.params["dim"]

            schema_dict["fields"].append(field_dict)

        # 保存为 JSON
        with open(output_file, "w") as f:
            json.dump(schema_dict, f, indent=2)

        print(f"✅ Schema exported to {output_file}")

    def export_index_config(self, collection_name: str, output_file: str):
        """导出索引配置"""
        collection = Collection(collection_name)

        # 获取所有索引
        indexes = []
        for field in collection.schema.fields:
            if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                try:
                    index = collection.index(field.name)
                    if index:
                        indexes.append({
                            "field_name": field.name,
                            "index_type": index.params["index_type"],
                            "metric_type": index.params["metric_type"],
                            "params": index.params.get("params", {})
                        })
                except:
                    pass

        # 保存为 JSON
        with open(output_file, "w") as f:
            json.dump(indexes, f, indent=2)

        print(f"✅ Index config exported to {output_file}")

    def export_data_json(
        self,
        collection_name: str,
        output_file: str,
        batch_size: int = 10000
    ):
        """导出数据为 JSON 格式"""
        collection = Collection(collection_name)
        collection.load()

        total = collection.num_entities
        all_data = []

        print(f"Exporting {total} entities...")

        # 分批导出
        for offset in range(0, total, batch_size):
            batch = collection.query(
                expr="id >= 0",
                limit=batch_size,
                offset=offset,
                output_fields=["*"]
            )
            all_data.extend(batch)

            progress = min(offset + batch_size, total)
            print(f"Progress: {progress}/{total} ({progress*100//total}%)")

        # 保存为 JSON
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)

        print(f"✅ Data exported to {output_file}")

    def export_data_parquet(
        self,
        collection_name: str,
        output_file: str,
        batch_size: int = 10000
    ):
        """导出数据为 Parquet 格式（推荐）"""
        collection = Collection(collection_name)
        collection.load()

        total = collection.num_entities
        print(f"Exporting {total} entities to Parquet...")

        # 分批导出并追加
        for offset in range(0, total, batch_size):
            batch = collection.query(
                expr="id >= 0",
                limit=batch_size,
                offset=offset,
                output_fields=["*"]
            )

            # 转换为 DataFrame
            df = pd.DataFrame(batch)

            # 追加到 Parquet 文件
            if offset == 0:
                df.to_parquet(output_file, index=False)
            else:
                df.to_parquet(
                    output_file,
                    index=False,
                    engine="fastparquet",
                    append=True
                )

            progress = min(offset + batch_size, total)
            print(f"Progress: {progress}/{total} ({progress*100//total}%)")

        print(f"✅ Data exported to {output_file}")

    def export_collection(
        self,
        collection_name: str,
        output_dir: str,
        format: str = "parquet"
    ):
        """导出完整 Collection"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 导出 Schema
        schema_file = os.path.join(output_dir, "schema.json")
        self.export_schema(collection_name, schema_file)

        # 2. 导出索引配置
        index_file = os.path.join(output_dir, "index.json")
        self.export_index_config(collection_name, index_file)

        # 3. 导出数据
        if format == "json":
            data_file = os.path.join(output_dir, "data.json")
            self.export_data_json(collection_name, data_file)
        elif format == "parquet":
            data_file = os.path.join(output_dir, "data.parquet")
            self.export_data_parquet(collection_name, data_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"\n✅ Collection exported to {output_dir}")
        print(f"  - Schema: {schema_file}")
        print(f"  - Index: {index_file}")
        print(f"  - Data: {data_file}")

# 使用示例
exporter = CollectionExporter()
exporter.export_collection(
    collection_name="my_collection",
    output_dir="./backup/my_collection",
    format="parquet"
)
```

---

## 导入方案

### 1. 完整导入方案

```python
from pymilvus import (
    Collection, CollectionSchema, FieldSchema,
    DataType, connections, utility
)
import json
import pandas as pd

class CollectionImporter:
    """Collection 导入器"""

    def __init__(self, host="localhost", port="19530"):
        """初始化连接"""
        connections.connect(host=host, port=port)

    def import_schema(self, schema_file: str) -> CollectionSchema:
        """导入 Schema"""
        with open(schema_file, "r") as f:
            schema_dict = json.load(f)

        # 创建字段
        fields = []
        for field_dict in schema_dict["fields"]:
            # 解析数据类型
            dtype = self._parse_dtype(field_dict["dtype"])

            # 创建字段
            if "dim" in field_dict:
                # 向量字段
                field = FieldSchema(
                    name=field_dict["name"],
                    dtype=dtype,
                    dim=field_dict["dim"],
                    description=field_dict.get("description", "")
                )
            else:
                # 标量字段
                field = FieldSchema(
                    name=field_dict["name"],
                    dtype=dtype,
                    is_primary=field_dict.get("is_primary", False),
                    auto_id=field_dict.get("auto_id", False),
                    description=field_dict.get("description", "")
                )

            fields.append(field)

        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description=schema_dict.get("description", "")
        )

        return schema

    def _parse_dtype(self, dtype_str: str) -> DataType:
        """解析数据类型"""
        dtype_map = {
            "DataType.INT64": DataType.INT64,
            "DataType.FLOAT": DataType.FLOAT,
            "DataType.VARCHAR": DataType.VARCHAR,
            "DataType.FLOAT_VECTOR": DataType.FLOAT_VECTOR,
            "DataType.BINARY_VECTOR": DataType.BINARY_VECTOR,
        }
        return dtype_map.get(dtype_str, DataType.INT64)

    def import_index_config(self, collection: Collection, index_file: str):
        """导入索引配置"""
        with open(index_file, "r") as f:
            indexes = json.load(f)

        for index_config in indexes:
            print(f"Creating index on {index_config['field_name']}...")

            collection.create_index(
                field_name=index_config["field_name"],
                index_params={
                    "index_type": index_config["index_type"],
                    "metric_type": index_config["metric_type"],
                    "params": index_config["params"]
                }
            )

        print("✅ Indexes created")

    def import_data_json(
        self,
        collection: Collection,
        data_file: str,
        batch_size: int = 10000
    ):
        """从 JSON 导入数据"""
        with open(data_file, "r") as f:
            data = json.load(f)

        total = len(data)
        print(f"Importing {total} entities...")

        # 分批导入
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            collection.insert(batch)

            progress = min(i + batch_size, total)
            print(f"Progress: {progress}/{total} ({progress*100//total}%)")

        collection.flush()
        print("✅ Data imported")

    def import_data_parquet(
        self,
        collection: Collection,
        data_file: str,
        batch_size: int = 10000
    ):
        """从 Parquet 导入数据"""
        df = pd.read_parquet(data_file)
        total = len(df)
        print(f"Importing {total} entities...")

        # 分批导入
        for i in range(0, total, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch = batch_df.to_dict('records')
            collection.insert(batch)

            progress = min(i + batch_size, total)
            print(f"Progress: {progress}/{total} ({progress*100//total}%)")

        collection.flush()
        print("✅ Data imported")

    def import_collection(
        self,
        collection_name: str,
        input_dir: str,
        format: str = "parquet",
        drop_existing: bool = False
    ):
        """导入完整 Collection"""
        # 1. 删除已存在的 Collection（如果需要）
        if drop_existing and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Dropped existing collection: {collection_name}")

        # 2. 导入 Schema
        schema_file = os.path.join(input_dir, "schema.json")
        schema = self.import_schema(schema_file)

        # 3. 创建 Collection
        collection = Collection(collection_name, schema)
        print(f"✅ Collection created: {collection_name}")

        # 4. 导入数据
        if format == "json":
            data_file = os.path.join(input_dir, "data.json")
            self.import_data_json(collection, data_file)
        elif format == "parquet":
            data_file = os.path.join(input_dir, "data.parquet")
            self.import_data_parquet(collection, data_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # 5. 导入索引配置
        index_file = os.path.join(input_dir, "index.json")
        self.import_index_config(collection, index_file)

        # 6. 加载到内存
        collection.load()
        print(f"✅ Collection loaded: {collection_name}")

        print(f"\n✅ Collection imported successfully!")
        print(f"  Total entities: {collection.num_entities}")

# 使用示例
importer = CollectionImporter()
importer.import_collection(
    collection_name="my_collection_restored",
    input_dir="./backup/my_collection",
    format="parquet",
    drop_existing=True
)
```

---

## 高级功能

### 1. 增量导出

```python
def export_incremental(
    self,
    collection_name: str,
    output_file: str,
    last_export_time: int
):
    """增量导出（基于时间戳）"""
    collection = Collection(collection_name)
    collection.load()

    # 查询增量数据
    incremental_data = collection.query(
        expr=f"timestamp > {last_export_time}",
        output_fields=["*"]
    )

    print(f"Found {len(incremental_data)} new entities")

    # 保存增量数据
    df = pd.DataFrame(incremental_data)
    df.to_parquet(output_file, index=False)

    print(f"✅ Incremental data exported to {output_file}")
```

### 2. 数据转换

```python
def export_with_transformation(
    self,
    collection_name: str,
    output_file: str,
    transform_fn
):
    """导出时转换数据"""
    collection = Collection(collection_name)
    collection.load()

    data = collection.query(expr="id >= 0", output_fields=["*"])

    # 应用转换函数
    transformed_data = [transform_fn(item) for item in data]

    # 保存
    df = pd.DataFrame(transformed_data)
    df.to_parquet(output_file, index=False)

    print(f"✅ Transformed data exported to {output_file}")

# 使用示例：降维
def reduce_dimensions(item):
    """将 768 维向量降到 384 维"""
    item["vector"] = item["vector"][:384]
    return item

exporter.export_with_transformation(
    collection_name="my_collection",
    output_file="reduced.parquet",
    transform_fn=reduce_dimensions
)
```

### 3. 数据清洗

```python
def export_with_cleaning(
    self,
    collection_name: str,
    output_file: str,
    filter_fn
):
    """导出时清洗数据"""
    collection = Collection(collection_name)
    collection.load()

    data = collection.query(expr="id >= 0", output_fields=["*"])

    # 过滤数据
    cleaned_data = [item for item in data if filter_fn(item)]

    print(f"Filtered: {len(data)} → {len(cleaned_data)}")

    # 保存
    df = pd.DataFrame(cleaned_data)
    df.to_parquet(output_file, index=False)

    print(f"✅ Cleaned data exported to {output_file}")

# 使用示例：移除异常值
def is_valid(item):
    """检查数据是否有效"""
    # 检查向量范数
    vector = np.array(item["vector"])
    norm = np.linalg.norm(vector)
    return 0.1 < norm < 10.0  # 移除异常向量

exporter.export_with_cleaning(
    collection_name="my_collection",
    output_file="cleaned.parquet",
    filter_fn=is_valid
)
```

---

## 在 RAG 系统中的应用

### 场景1：知识库版本升级

```python
class RAGKnowledgeBaseUpgrader:
    """RAG 知识库升级器"""

    def upgrade_v1_to_v2(self, old_collection: str, new_collection: str):
        """从 v1 升级到 v2"""
        # 1. 导出旧版本数据
        exporter = CollectionExporter()
        exporter.export_collection(
            collection_name=old_collection,
            output_dir="./temp/v1_backup",
            format="parquet"
        )

        # 2. 转换数据格式
        df = pd.read_parquet("./temp/v1_backup/data.parquet")

        # v2 新增字段
        df["metadata"] = df.apply(self._extract_metadata, axis=1)
        df["chunk_id"] = range(len(df))

        # 保存转换后的数据
        df.to_parquet("./temp/v2_data.parquet", index=False)

        # 3. 创建新版本 Collection
        self._create_v2_collection(new_collection)

        # 4. 导入数据
        importer = CollectionImporter()
        collection = Collection(new_collection)
        importer.import_data_parquet(
            collection,
            "./temp/v2_data.parquet"
        )

        print(f"✅ Upgraded from {old_collection} to {new_collection}")

    def _extract_metadata(self, row):
        """提取元数据"""
        return {
            "source": row.get("source", "unknown"),
            "created_at": row.get("timestamp", 0),
            "version": "v2"
        }

    def _create_v2_collection(self, collection_name: str):
        """创建 v2 Collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields, description="RAG Knowledge Base v2")
        collection = Collection(collection_name, schema)

        # 创建索引
        collection.create_index(
            "vector",
            {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 200}}
        )

        return collection
```

### 场景2：多环境数据同步

```python
class MultiEnvironmentSync:
    """多环境数据同步"""

    def sync_dev_to_test(self):
        """从开发环境同步到测试环境"""
        # 1. 从开发环境导出
        dev_exporter = CollectionExporter(host="dev-milvus.com")
        dev_exporter.export_collection(
            collection_name="rag_docs",
            output_dir="./sync/dev_backup",
            format="parquet"
        )

        # 2. 导入到测试环境
        test_importer = CollectionImporter(host="test-milvus.com")
        test_importer.import_collection(
            collection_name="rag_docs",
            input_dir="./sync/dev_backup",
            format="parquet",
            drop_existing=True
        )

        print("✅ Synced from dev to test")

    def sync_with_validation(self):
        """带验证的同步"""
        # 导出
        exporter = CollectionExporter(host="source.com")
        exporter.export_collection("my_collection", "./sync/backup")

        # 导入
        importer = CollectionImporter(host="target.com")
        importer.import_collection("my_collection", "./sync/backup")

        # 验证
        source = Collection("my_collection")
        connections.connect(alias="target", host="target.com")
        target = Collection("my_collection", using="target")

        assert source.num_entities == target.num_entities
        print("✅ Validation passed")
```

---

## 性能优化

### 1. 并行导出

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_export(collection_name: str, output_dir: str, num_workers: int = 4):
    """并行导出"""
    collection = Collection(collection_name)
    total = collection.num_entities
    chunk_size = total // num_workers

    def export_chunk(worker_id):
        offset = worker_id * chunk_size
        limit = chunk_size if worker_id < num_workers - 1 else total - offset

        data = collection.query(
            expr="id >= 0",
            limit=limit,
            offset=offset,
            output_fields=["*"]
        )

        df = pd.DataFrame(data)
        df.to_parquet(f"{output_dir}/chunk_{worker_id}.parquet")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(export_chunk, i) for i in range(num_workers)]
        for future in futures:
            future.result()

    print(f"✅ Parallel export completed")
```

### 2. 压缩优化

```python
# 使用不同压缩算法
df.to_parquet("data.parquet", compression="snappy")  # 快速
df.to_parquet("data.parquet", compression="gzip")    # 平衡
df.to_parquet("data.parquet", compression="brotli")  # 高压缩率
```

---

## 总结

### 核心要点

1. **灵活性高**：可自定义格式和转换逻辑
2. **适合迁移**：跨版本、跨集群数据迁移
3. **需要编码**：需要自己编写导出导入脚本
4. **不备份索引**：只备份数据，需要重建索引

### 适用场景

- ✅ 一次性数据迁移
- ✅ 跨版本升级
- ✅ 数据转换和清洗
- ✅ 多环境数据同步

### 下一步

- 学习 [数据迁移策略](./03_核心概念_03_数据迁移策略.md)
- 实践 [Collection 导出导入场景](./07_实战代码_02_Collection导出导入.md)
