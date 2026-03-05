# 实战代码场景7：生产级Schema管理

> 本文档演示 Milvus 2.6 生产环境中的 Schema 版本管理、多版本共存、数据迁移和监控告警

---

## 一、场景描述

### 1.1 业务需求

**场景**：RAG 知识库系统已上线运行，随着业务迭代需要多次修改 Schema，要求在不停机的情况下完成 Schema 演化，并保证数据一致性。

**核心挑战**：
- Schema 变更需要版本管理和回滚能力
- 多版本数据需要共存（旧数据没有新字段）
- 变更过程中不能影响线上查询
- 需要监控 Schema 变更的影响

### 1.2 技术目标

1. **Schema 版本管理**：记录每次变更，支持审计
2. **安全变更流程**：维护窗口操作，完整验证
3. **多版本数据查询**：兼容新旧数据
4. **监控与回滚**：变更后监控关键指标

---

## 二、完整实战代码

```python
"""
生产级 Schema 管理实战
演示：版本管理、安全变更流程、多版本查询、监控
"""

import numpy as np
import time
import json
from datetime import datetime
from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import MilvusException

# ===== 配置 =====
MILVUS_URI = "http://localhost:19530"
DIM = 128
COLLECTION_NAME = "schema_mgmt_demo"
client = MilvusClient(MILVUS_URI)
np.random.seed(42)


# ============================================================
# 1. Schema 版本管理器
# ============================================================
print("=" * 60)
print("1. Schema 版本管理器")
print("=" * 60)


class SchemaVersionManager:
    """Schema 版本管理器：记录变更历史，支持审计和回滚规划"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.history: list[dict] = []
        self.current_version = 0

    def record_creation(self, fields: list[str]):
        """记录 Collection 创建"""
        self.current_version = 1
        self.history.append({
            "version": 1,
            "action": "CREATE",
            "fields": fields,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        print(f"  [v1] Collection 创建，字段: {fields}")

    def record_add_field(self, field_name: str, field_type: str,
                         nullable: bool = True, default_value=None) -> int:
        """记录字段添加"""
        self.current_version += 1
        record = {
            "version": self.current_version,
            "action": "ADD_FIELD",
            "field_name": field_name,
            "field_type": field_type,
            "nullable": nullable,
            "default_value": default_value,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        self.history.append(record)
        return self.current_version

    def mark_success(self, version: int):
        """标记变更成功"""
        for record in self.history:
            if record["version"] == version:
                record["status"] = "success"
                print(f"  [v{version}] ✅ 变更成功: {record['action']} "
                      f"{record.get('field_name', '')}")

    def mark_failed(self, version: int, error: str):
        """标记变更失败"""
        for record in self.history:
            if record["version"] == version:
                record["status"] = "failed"
                record["error"] = error
                print(f"  [v{version}] ❌ 变更失败: {error}")

    def get_history(self) -> list[dict]:
        """获取变更历史"""
        return self.history

    def print_history(self):
        """打印变更历史"""
        print(f"\n  Schema 变更历史 ({self.collection_name}):")
        print(f"  {'版本':<6} {'操作':<12} {'字段':<15} {'状态':<10} {'时间'}")
        print(f"  {'-'*70}")
        for r in self.history:
            field = r.get("field_name", ",".join(r.get("fields", [])))
            print(f"  v{r['version']:<5} {r['action']:<12} {field:<15} "
                  f"{r['status']:<10} {r['timestamp'][:19]}")


# 初始化版本管理器
version_mgr = SchemaVersionManager(COLLECTION_NAME)


# ============================================================
# 2. 创建初始 Collection（v1）
# ============================================================
print("\n" + "=" * 60)
print("2. 创建初始 Collection（v1）")
print("=" * 60)

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field("text", DataType.VARCHAR, max_length=1024)
schema.add_field("department", DataType.VARCHAR, max_length=64)
schema.add_field("created_at", DataType.INT64)

index_params = client.prepare_index_params()
index_params.add_index("vector", index_type="HNSW", metric_type="COSINE",
                       params={"M": 16, "efConstruction": 200})
index_params.add_index("department", index_type="INVERTED")

client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
version_mgr.record_creation(["id", "vector", "text", "department", "created_at"])

# 插入 v1 数据
v1_data = [
    {
        "id": i,
        "vector": np.random.randn(DIM).astype(np.float32).tolist(),
        "text": f"v1_document_{i}",
        "department": np.random.choice(["tech", "legal", "finance"]),
        "created_at": int(time.time()) - 86400 * 30,
        "schema_version": 1,  # 动态字段：记录数据版本
    }
    for i in range(200)
]
client.insert(COLLECTION_NAME, v1_data)
print(f"  插入 200 条 v1 数据")


# ============================================================
# 3. 安全变更流程（v2：添加 category 字段）
# ============================================================
print("\n" + "=" * 60)
print("3. 安全变更流程（v2）")
print("=" * 60)


def safe_add_field(client: MilvusClient, collection_name: str,
                   version_mgr: SchemaVersionManager,
                   field_name: str, data_type, **kwargs) -> bool:
    """
    生产安全的字段添加流程
    返回 True 表示成功，False 表示失败
    """
    version = version_mgr.record_add_field(
        field_name, str(data_type),
        nullable=kwargs.get("nullable", True),
        default_value=kwargs.get("default_value")
    )

    try:
        # Step 1: 添加字段
        print(f"\n  Step 1: 添加字段 {field_name}...")
        client.add_collection_field(
            collection_name=collection_name,
            field_name=field_name,
            data_type=data_type,
            **kwargs
        )

        # Step 2: Reload Collection
        print(f"  Step 2: Reload Collection...")
        client.release_collection(collection_name)
        client.load_collection(collection_name)
        time.sleep(1)

        # Step 3: 验证查询
        print(f"  Step 3: 验证查询...")
        query_vec = np.random.randn(DIM).astype(np.float32).tolist()

        # 普通搜索
        results = client.search(
            collection_name, data=[query_vec], limit=3,
            output_fields=["text", field_name]
        )
        assert len(results[0]) > 0, "搜索返回空结果"

        # 过滤查询
        results = client.search(
            collection_name, data=[query_vec], limit=3,
            filter=f"{field_name} is null",
            output_fields=["text", field_name]
        )

        version_mgr.mark_success(version)
        return True

    except Exception as e:
        version_mgr.mark_failed(version, str(e))
        print(f"  ⚠️ 变更失败，需要人工介入: {e}")
        return False


# 执行 v2 变更：添加 category 字段
print("  ⚠️ 模拟暂停写入...")
success = safe_add_field(
    client, COLLECTION_NAME, version_mgr,
    field_name="category",
    data_type=DataType.VARCHAR,
    max_length=100,
    nullable=True,
    default_value="uncategorized"
)
if success:
    print("  ✅ 模拟恢复写入")


# ============================================================
# 4. 继续变更（v3：添加 priority 字段）
# ============================================================
print("\n" + "=" * 60)
print("4. 继续变更（v3）")
print("=" * 60)

print("  ⚠️ 模拟暂停写入...")
safe_add_field(
    client, COLLECTION_NAME, version_mgr,
    field_name="priority",
    data_type=DataType.INT64,
    nullable=True,
    default_value=0
)
print("  ✅ 模拟恢复写入")


# ============================================================
# 5. 多版本数据共存查询
# ============================================================
print("\n" + "=" * 60)
print("5. 多版本数据共存")
print("=" * 60)

# 插入 v3 数据（包含新字段）
v3_data = [
    {
        "id": 1000 + i,
        "vector": np.random.randn(DIM).astype(np.float32).tolist(),
        "text": f"v3_document_{i}",
        "department": "tech",
        "created_at": int(time.time()),
        "category": np.random.choice(["tutorial", "api_doc", "changelog"]),
        "priority": np.random.choice([1, 2, 3]),
        "schema_version": 3,  # 动态字段：记录数据版本
    }
    for i in range(100)
]
client.insert(COLLECTION_NAME, v3_data)
print(f"  插入 100 条 v3 数据")

# 查询所有数据（v1 和 v3 共存）
query_vec = np.random.randn(DIM).astype(np.float32).tolist()

print("\n  查询结果（新旧数据共存）:")
results = client.search(
    COLLECTION_NAME, data=[query_vec], limit=6,
    output_fields=["text", "category", "priority", "schema_version"]
)
for r in results[0]:
    entity = r["entity"]
    sv = entity.get("schema_version", "?")
    cat = entity.get("category", "NULL")
    pri = entity.get("priority", "NULL")
    print(f"  id={r['id']}, schema_v={sv}, category={cat}, priority={pri}")

# 按版本过滤
print("\n  只查询 v1 数据:")
results = client.search(
    COLLECTION_NAME, data=[query_vec], limit=3,
    filter="schema_version == 1",
    output_fields=["text", "schema_version", "category"]
)
for r in results[0]:
    entity = r["entity"]
    print(f"  id={r['id']}, schema_v={entity.get('schema_version')}, "
          f"category={entity.get('category', 'NULL')}")

print("\n  只查询 v3 数据:")
results = client.search(
    COLLECTION_NAME, data=[query_vec], limit=3,
    filter="schema_version == 3",
    output_fields=["text", "schema_version", "category"]
)
for r in results[0]:
    entity = r["entity"]
    print(f"  id={r['id']}, schema_v={entity.get('schema_version')}, "
          f"category={entity.get('category')}")


# ============================================================
# 6. 变更历史审计
# ============================================================
print("\n" + "=" * 60)
print("6. 变更历史审计")
print("=" * 60)

version_mgr.print_history()

# 导出为 JSON（可持久化到文件或数据库）
history_json = json.dumps(version_mgr.get_history(), indent=2, ensure_ascii=False)
print(f"\n  变更历史 JSON（可持久化）:")
print(f"  共 {len(version_mgr.get_history())} 条记录")


# ============================================================
# 7. 监控指标采集
# ============================================================
print("\n" + "=" * 60)
print("7. 监控指标")
print("=" * 60)


def collect_metrics(client: MilvusClient, collection_name: str) -> dict:
    """采集 Collection 关键指标"""
    desc = client.describe_collection(collection_name)
    stats = client.get_collection_stats(collection_name)

    # 搜索延迟测试
    query_vec = np.random.randn(DIM).astype(np.float32).tolist()
    latencies = []
    for _ in range(10):
        start = time.time()
        client.search(collection_name, data=[query_vec], limit=10)
        latencies.append((time.time() - start) * 1000)

    metrics = {
        "collection": collection_name,
        "field_count": len(desc.get("fields", [])),
        "row_count": stats.get("row_count", 0),
        "avg_latency_ms": round(np.mean(latencies), 2),
        "p99_latency_ms": round(np.percentile(latencies, 99), 2),
        "schema_version": version_mgr.current_version,
        "timestamp": datetime.now().isoformat()
    }
    return metrics


metrics = collect_metrics(client, COLLECTION_NAME)
print(f"  字段数量:      {metrics['field_count']}")
print(f"  数据行数:      {metrics['row_count']}")
print(f"  平均搜索延迟:  {metrics['avg_latency_ms']} ms")
print(f"  P99 搜索延迟:  {metrics['p99_latency_ms']} ms")
print(f"  Schema 版本:   v{metrics['schema_version']}")


# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 60)
print("8. 总结")
print("=" * 60)

print("""
生产级 Schema 管理要点:

  1. 版本管理
     - 记录每次变更的字段、类型、时间、状态
     - 支持审计和问题追溯
     - 在数据中记录 schema_version（动态字段）

  2. 安全变更流程
     - 暂停写入 → 添加字段 → Reload → 验证 → 恢复写入
     - 每次变更都要验证所有查询类型
     - 失败时标记状态，人工介入

  3. 多版本共存
     - 旧数据新字段为 NULL 或默认值
     - 用 schema_version 动态字段区分数据版本
     - 查询时兼容新旧数据

  4. 监控告警
     - 变更后监控搜索延迟
     - 监控 schema mismatch 错误率
     - 定期采集关键指标
""")

# 清理
client.drop_collection(COLLECTION_NAME)
print("✅ 测试完成，Collection 已清理")
```

---

## 三、运行输出示例

```
============================================================
1. Schema 版本管理器
============================================================

============================================================
2. 创建初始 Collection（v1）
============================================================
  [v1] Collection 创建，字段: ['id', 'vector', 'text', 'department', 'created_at']
  插入 200 条 v1 数据

============================================================
3. 安全变更流程（v2）
============================================================
  ⚠️ 模拟暂停写入...
  Step 1: 添加字段 category...
  Step 2: Reload Collection...
  Step 3: 验证查询...
  [v2] ✅ 变更成功: ADD_FIELD category
  ✅ 模拟恢复写入

============================================================
6. 变更历史审计
============================================================
  Schema 变更历史 (schema_mgmt_demo):
  版本     操作           字段              状态       时间
  ----------------------------------------------------------------------
  v1     CREATE       id,vector,...    success    2026-02-25T10:00:00
  v2     ADD_FIELD    category         success    2026-02-25T10:00:01
  v3     ADD_FIELD    priority         success    2026-02-25T10:00:02

✅ 测试完成，Collection 已清理
```

---

## 四、生产环境部署建议

### 4.1 变更窗口选择

- 选择业务低峰期（如凌晨）
- 预留足够的验证时间
- 准备回滚方案（记录变更前的 Schema 状态）

### 4.2 回滚策略

AddCollectionField 目前**不支持删除字段**，回滚方案：
- 方案A：忽略新字段（不写入、不查询）
- 方案B：创建新 Collection，迁移数据
- 方案C：使用默认值填充，标记为废弃

### 4.3 持久化建议

将 `SchemaVersionManager` 的历史记录持久化到：
- 数据库（MySQL/PostgreSQL）
- 配置中心（etcd/Consul）
- 文件系统（JSON/YAML）

---

**版本**: v1.0
**最后更新**: 2026-02-25
**数据来源**: 源码分析 + Context7文档 + 网络搜索（2025-2026年资料）
