# 实战代码2：Collection 导出导入

> 手动数据迁移的完整实现

---

## 场景概述

本场景演示如何手动导出和导入 Collection 数据：
- 导出 Collection 的 Schema、数据和索引配置
- 支持 JSON 和 Parquet 两种格式
- 数据转换和清洗
- 跨版本、跨集群迁移

**适用场景：**
- 一次性数据迁移
- 跨版本升级
- 数据转换和清洗
- 不依赖 Backup 工具的场景

---

## 完整示例代码

### 示例1：基础导出导入

```python
#!/usr/bin/env python3
"""
Collection 基础导出导入示例
"""

import json
import pandas as pd
import numpy as np
from pymilvus import (
    Collection, CollectionSchema, FieldSchema,
    DataType, connections, utility
)
from typing import Dict, List, Optional
import os

class CollectionMigrator:
    """Collection 迁移工具"""

    def __init__(self, source_host="localhost", target_host="localhost"):
        """初始化"""
        self.source_host = source_host
        self.target_host = target_host

    def export_collection(
        self,
        collection_name: str,
        output_dir: str,
        format: str = "parquet"
    ):
        """导出 Collection"""
        print(f"=== 导出 Collection: {collection_name} ===")

        # 连接源环境
        connections.connect(alias="source", host=self.source_host, port=19530)
        collection = Collection(collection_name, using="source")
        collection.load()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 1. 导出 Schema
        print("[1/3] 导出 Schema...")
        self._export_schema(collection, output_dir)

        # 2. 导出索引配置
        print("[2/3] 导出索引配置...")
        self._export_index_config(collection, output_dir)

        # 3. 导出数据
        print("[3/3] 导出数据...")
        if format == "json":
            self._export_data_json(collection, output_dir)
        elif format == "parquet":
            self._export_data_parquet(collection, output_dir)
        else:
            raise ValueError(f"不支持的格式: {format}")

        print(f"✅ 导出完成: {output_dir}")

    def _export_schema(self, collection: Collection, output_dir: str):
        """导出 Schema"""
        schema = collection.schema
        schema_dict = {
            "collection_name": collection.name,
            "description": schema.description,
            "fields": []
        }

        for field in schema.fields:
            field_dict = {
                "name": field.name,
                "dtype": str(field.dtype),
                "description": field.description,
                "is_primary": field.is_primary,
                "auto_id": field.auto_id
            }

            # 向量字段的维度
            if hasattr(field, "params") and field.params:
                if "dim" in field.params:
                    field_dict["dim"] = field.params["dim"]
                if "max_length" in field.params:
                    field_dict["max_length"] = field.params["max_length"]

            schema_dict["fields"].append(field_dict)

        # 保存
        schema_file = os.path.join(output_dir, "schema.json")
        with open(schema_file, "w") as f:
            json.dump(schema_dict, f, indent=2)

        print(f"  Schema: {schema_file}")

    def _export_index_config(self, collection: Collection, output_dir: str):
        """导出索引配置"""
        indexes = []

        for field in collection.schema.fields:
            if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                try:
                    index = collection.index(field.name)
                    if index:
                        indexes.append({
                            "field_name": field.name,
                            "index_type": index.params.get("index_type"),
                            "metric_type": index.params.get("metric_type"),
                            "params": index.params.get("params", {})
                        })
                except:
                    pass

        # 保存
        index_file = os.path.join(output_dir, "index.json")
        with open(index_file, "w") as f:
            json.dump(indexes, f, indent=2)

        print(f"  索引配置: {index_file}")

    def _export_data_parquet(self, collection: Collection, output_dir: str):
        """导出数据为 Parquet 格式"""
        total = collection.num_entities
        batch_size = 10000

        print(f"  导出 {total} 条数据...")

        all_data = []
        for offset in range(0, total, batch_size):
            batch = collection.query(
                expr="id >= 0",
                limit=batch_size,
                offset=offset,
                output_fields=["*"]
            )
            all_data.extend(batch)

            progress = min(offset + batch_size, total)
            print(f"    进度: {progress}/{total} ({progress*100//total}%)")

        # 保存为 Parquet
        df = pd.DataFrame(all_data)
        data_file = os.path.join(output_dir, "data.parquet")
        df.to_parquet(data_file, index=False, compression="snappy")

        print(f"  数据文件: {data_file}")
        print(f"  文件大小: {os.path.getsize(data_file) / 1024 / 1024:.2f} MB")

    def import_collection(
        self,
        collection_name: str,
        input_dir: str,
        format: str = "parquet",
        drop_existing: bool = False
    ):
        """导入 Collection"""
        print(f"=== 导入 Collection: {collection_name} ===")

        # 连接目标环境
        connections.connect(alias="target", host=self.target_host, port=19530)

        # 1. 删除已存在的 Collection
        if drop_existing and utility.has_collection(collection_name, using="target"):
            utility.drop_collection(collection_name, using="target")
            print(f"  已删除现有 Collection: {collection_name}")

        # 2. 导入 Schema
        print("[1/4] 导入 Schema...")
        schema = self._import_schema(input_dir)
        collection = Collection(collection_name, schema, using="target")
        print(f"  ✅ Collection 已创建")

        # 3. 导入数据
        print("[2/4] 导入数据...")
        if format == "parquet":
            self._import_data_parquet(collection, input_dir)
        else:
            raise ValueError(f"不支持的格式: {format}")

        # 4. 导入索引配置
        print("[3/4] 创建索引...")
        self._import_index_config(collection, input_dir)

        # 5. 加载到内存
        print("[4/4] 加载 Collection...")
        collection.load()

        print(f"✅ 导入完成")
        print(f"  总数据量: {collection.num_entities}")

    def _import_schema(self, input_dir: str) -> CollectionSchema:
        """导入 Schema"""
        schema_file = os.path.join(input_dir, "schema.json")
        with open(schema_file, "r") as f:
            schema_dict = json.load(f)

        fields = []
        for field_dict in schema_dict["fields"]:
            dtype = self._parse_dtype(field_dict["dtype"])

            if "dim" in field_dict:
                # 向量字段
                field = FieldSchema(
                    name=field_dict["name"],
                    dtype=dtype,
                    dim=field_dict["dim"],
                    description=field_dict.get("description", "")
                )
            elif "max_length" in field_dict:
                # VARCHAR 字段
                field = FieldSchema(
                    name=field_dict["name"],
                    dtype=dtype,
                    max_length=field_dict["max_length"],
                    is_primary=field_dict.get("is_primary", False),
                    auto_id=field_dict.get("auto_id", False),
                    description=field_dict.get("description", "")
                )
            else:
                # 其他标量字段
                field = FieldSchema(
                    name=field_dict["name"],
                    dtype=dtype,
                    is_primary=field_dict.get("is_primary", False),
                    auto_id=field_dict.get("auto_id", False),
                    description=field_dict.get("description", "")
                )

            fields.append(field)

        schema = CollectionSchema(
            fields=fields,
            description=schema_dict.get("description", "")
        )

        return schema

    def _parse_dtype(self, dtype_str: str) -> DataType:
        """解析数据类型"""
        dtype_map = {
            "DataType.INT64": DataType.INT64,
            "DataType.INT32": DataType.INT32,
            "DataType.FLOAT": DataType.FLOAT,
            "DataType.DOUBLE": DataType.DOUBLE,
            "DataType.VARCHAR": DataType.VARCHAR,
            "DataType.FLOAT_VECTOR": DataType.FLOAT_VECTOR,
            "DataType.BINARY_VECTOR": DataType.BINARY_VECTOR,
            "DataType.JSON": DataType.JSON,
        }
        return dtype_map.get(dtype_str, DataType.INT64)

    def _import_data_parquet(self, collection: Collection, input_dir: str):
        """从 Parquet 导入数据"""
        data_file = os.path.join(input_dir, "data.parquet")
        df = pd.read_parquet(data_file)

        total = len(df)
        batch_size = 10000

        print(f"  导入 {total} 条数据...")

        for i in range(0, total, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch = batch_df.to_dict('records')
            collection.insert(batch)

            progress = min(i + batch_size, total)
            print(f"    进度: {progress}/{total} ({progress*100//total}%)")

        collection.flush()
        print(f"  ✅ 数据导入完成")

    def _import_index_config(self, collection: Collection, input_dir: str):
        """导入索引配置"""
        index_file = os.path.join(input_dir, "index.json")
        with open(index_file, "r") as f:
            indexes = json.load(f)

        for index_config in indexes:
            print(f"  创建索引: {index_config['field_name']}")

            collection.create_index(
                field_name=index_config["field_name"],
                index_params={
                    "index_type": index_config["index_type"],
                    "metric_type": index_config["metric_type"],
                    "params": index_config["params"]
                }
            )

        print(f"  ✅ 索引创建完成")


def main():
    """主函数"""
    migrator = CollectionMigrator(
        source_host="localhost",
        target_host="localhost"
    )

    # 1. 导出 Collection
    migrator.export_collection(
        collection_name="my_collection",
        output_dir="./migration/my_collection",
        format="parquet"
    )

    # 2. 导入到新 Collection
    migrator.import_collection(
        collection_name="my_collection_migrated",
        input_dir="./migration/my_collection",
        format="parquet",
        drop_existing=True
    )


if __name__ == "__main__":
    main()
```

**运行示例：**

```bash
python collection_migration.py
```

**输出：**

```
=== 导出 Collection: my_collection ===
[1/3] 导出 Schema...
  Schema: ./migration/my_collection/schema.json
[2/3] 导出索引配置...
  索引配置: ./migration/my_collection/index.json
[3/3] 导出数据...
  导出 1000000 条数据...
    进度: 10000/1000000 (1%)
    进度: 20000/1000000 (2%)
    ...
    进度: 1000000/1000000 (100%)
  数据文件: ./migration/my_collection/data.parquet
  文件大小: 2500.00 MB
✅ 导出完成: ./migration/my_collection

=== 导入 Collection: my_collection_migrated ===
[1/4] 导入 Schema...
  ✅ Collection 已创建
[2/4] 导入数据...
  导入 1000000 条数据...
    进度: 10000/1000000 (1%)
    ...
    进度: 1000000/1000000 (100%)
  ✅ 数据导入完成
[3/4] 创建索引...
  创建索引: vector
  ✅ 索引创建完成
[4/4] 加载 Collection...
✅ 导入完成
  总数据量: 1000000
```

---

### 示例2：带数据转换的迁移

```python
#!/usr/bin/env python3
"""
带数据转换的 Collection 迁移
"""

from typing import Callable, Dict, Any
import pandas as pd

class TransformingMigrator(CollectionMigrator):
    """支持数据转换的迁移工具"""

    def export_with_transform(
        self,
        collection_name: str,
        output_dir: str,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """导出并转换数据"""
        print(f"=== 导出并转换 Collection: {collection_name} ===")

        # 连接源环境
        connections.connect(alias="source", host=self.source_host, port=19530)
        collection = Collection(collection_name, using="source")
        collection.load()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 1. 导出 Schema（可能需要修改）
        print("[1/3] 导出 Schema...")
        self._export_schema(collection, output_dir)

        # 2. 导出索引配置
        print("[2/3] 导出索引配置...")
        self._export_index_config(collection, output_dir)

        # 3. 导出并转换数据
        print("[3/3] 导出并转换数据...")
        self._export_and_transform_data(collection, output_dir, transform_fn)

        print(f"✅ 导出完成: {output_dir}")

    def _export_and_transform_data(
        self,
        collection: Collection,
        output_dir: str,
        transform_fn: Callable
    ):
        """导出并转换数据"""
        total = collection.num_entities
        batch_size = 10000

        print(f"  处理 {total} 条数据...")

        all_data = []
        for offset in range(0, total, batch_size):
            batch = collection.query(
                expr="id >= 0",
                limit=batch_size,
                offset=offset,
                output_fields=["*"]
            )

            # 转换每条数据
            transformed_batch = [transform_fn(item) for item in batch]
            all_data.extend(transformed_batch)

            progress = min(offset + batch_size, total)
            print(f"    进度: {progress}/{total} ({progress*100//total}%)")

        # 保存
        df = pd.DataFrame(all_data)
        data_file = os.path.join(output_dir, "data.parquet")
        df.to_parquet(data_file, index=False, compression="snappy")

        print(f"  数据文件: {data_file}")


# 使用示例：向量降维
def reduce_vector_dimensions(item: Dict[str, Any]) -> Dict[str, Any]:
    """将 768 维向量降到 384 维"""
    if "vector" in item and len(item["vector"]) == 768:
        item["vector"] = item["vector"][:384]
    return item

# 使用示例：添加新字段
def add_metadata_field(item: Dict[str, Any]) -> Dict[str, Any]:
    """添加元数据字段"""
    item["metadata"] = {
        "source": "migration",
        "timestamp": int(time.time()),
        "version": "v2"
    }
    return item

# 使用示例：数据清洗
def clean_invalid_vectors(item: Dict[str, Any]) -> Dict[str, Any]:
    """清洗无效向量"""
    if "vector" in item:
        vector = np.array(item["vector"])
        norm = np.linalg.norm(vector)

        # 归一化异常向量
        if norm < 0.1 or norm > 10.0:
            item["vector"] = (vector / norm).tolist()

    return item


def main():
    """主函数"""
    migrator = TransformingMigrator(
        source_host="localhost",
        target_host="localhost"
    )

    # 导出并转换数据
    migrator.export_with_transform(
        collection_name="my_collection",
        output_dir="./migration/transformed",
        transform_fn=reduce_vector_dimensions
    )

    # 导入转换后的数据
    migrator.import_collection(
        collection_name="my_collection_reduced",
        input_dir="./migration/transformed",
        format="parquet"
    )


if __name__ == "__main__":
    main()
```

---

### 示例3：跨环境迁移

```python
#!/usr/bin/env python3
"""
跨环境 Collection 迁移
"""

import subprocess

class CrossEnvironmentMigrator:
    """跨环境迁移工具"""

    def migrate_dev_to_prod(
        self,
        collection_name: str,
        dev_host: str,
        prod_host: str
    ):
        """从开发环境迁移到生产环境"""
        print(f"=== 跨环境迁移: {collection_name} ===")
        print(f"  源: {dev_host}")
        print(f"  目标: {prod_host}")

        # 1. 从开发环境导出
        print("\n[1/4] 从开发环境导出...")
        dev_migrator = CollectionMigrator(source_host=dev_host)
        dev_migrator.export_collection(
            collection_name=collection_name,
            output_dir=f"./migration/{collection_name}",
            format="parquet"
        )

        # 2. 压缩数据
        print("\n[2/4] 压缩数据...")
        self._compress_data(f"./migration/{collection_name}")

        # 3. 传输到生产环境
        print("\n[3/4] 传输到生产环境...")
        self._transfer_to_prod(
            f"./migration/{collection_name}.tar.gz",
            prod_host
        )

        # 4. 在生产环境导入
        print("\n[4/4] 在生产环境导入...")
        prod_migrator = CollectionMigrator(target_host=prod_host)
        prod_migrator.import_collection(
            collection_name=collection_name,
            input_dir=f"./migration/{collection_name}",
            format="parquet"
        )

        print(f"\n✅ 跨环境迁移完成")

    def _compress_data(self, data_dir: str):
        """压缩数据"""
        subprocess.run([
            "tar", "-czf", f"{data_dir}.tar.gz", data_dir
        ], check=True)

        print(f"  压缩完成: {data_dir}.tar.gz")

    def _transfer_to_prod(self, archive_file: str, prod_host: str):
        """传输到生产环境"""
        # 使用 scp 传输
        subprocess.run([
            "scp", archive_file, f"{prod_host}:/tmp/"
        ], check=True)

        # 在生产环境解压
        subprocess.run([
            "ssh", prod_host,
            f"tar -xzf /tmp/{os.path.basename(archive_file)} -C /tmp/"
        ], check=True)

        print(f"  传输完成")


def main():
    """主函数"""
    migrator = CrossEnvironmentMigrator()

    migrator.migrate_dev_to_prod(
        collection_name="rag_knowledge_base",
        dev_host="dev-milvus.internal",
        prod_host="prod-milvus.internal"
    )


if __name__ == "__main__":
    main()
```

---

## 总结

### 核心要点

1. **手动迁移灵活性高**：可以自定义数据格式和转换逻辑
2. **Parquet 格式推荐**：比 JSON 更高效，保留精度
3. **支持数据转换**：在迁移过程中清洗和转换数据
4. **跨环境迁移**：开发、测试、生产环境间数据同步

### 适用场景

- ✅ 一次性数据迁移
- ✅ 跨版本升级
- ✅ 数据转换和清洗
- ✅ 不依赖 Backup 工具

### 下一步

- 学习 [跨集群数据迁移](./07_实战代码_03_跨集群数据迁移.md)
- 学习 [自动化备份系统](./07_实战代码_04_自动化备份系统.md)
