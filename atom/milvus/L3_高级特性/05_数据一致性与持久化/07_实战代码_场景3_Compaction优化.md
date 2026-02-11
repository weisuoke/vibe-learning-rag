# 实战代码 - 场景3：Compaction 优化与监控

## 场景描述

演示如何使用 Compaction 优化存储空间和查询性能，以及如何监控 Compaction 的运行状态。

---

## 完整代码示例

```python
"""
场景3：Compaction 优化与监控
演示：
1. 手动触发 Compaction
2. 监控 Compaction 状态
3. 性能对比测试
4. 定期 Compaction 策略
"""

import time
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

# ===== 1. 连接 Milvus =====
print("=== 连接 Milvus ===")
connections.connect("default", host="localhost", port="19530")
print("连接成功！\n")

# ===== 2. 创建测试 Collection =====
print("=== 创建测试 Collection ===")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields=fields, description="Compaction 测试")

collection_name = "compaction_test"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' 创建成功！\n")

# ===== 3. 插入和删除数据（产生碎片）=====
print("=== 插入和删除数据（产生碎片）===\n")

# 插入 10 万条数据
print("步骤1：插入 10 万条数据")
num_entities = 100000
entities = [
    list(range(num_entities)),
    np.random.rand(num_entities, 128).tolist(),
    [f"文档 {i}" for i in range(num_entities)],
]
collection.insert(entities)
collection.flush()
print(f"  插入完成：{num_entities} 条\n")

# 删除 50% 的数据
print("步骤2：删除 50% 的数据")
collection.delete(expr="id < 50000")
collection.flush()
print("  删除完成：50000 条\n")

# 查看 Segment 信息（Compaction 前）
print("步骤3：查看 Segment 信息（Compaction 前）")
segments_before = utility.get_query_segment_info(collection_name)
print(f"  Segment 数量: {len(segments_before)}")
total_rows_before = sum(s.num_rows for s in segments_before)
print(f"  总行数: {total_rows_before}")
print()

# ===== 4. 运行 Compaction =====
print("=== 运行 Compaction ===\n")

print("步骤1：触发 Compaction")
compaction_id = utility.do_compact(collection_name=collection_name)
print(f"  Compaction ID: {compaction_id}\n")

print("步骤2：监控 Compaction 进度")
while True:
    state = utility.get_compaction_state(compaction_id)
    print(f"  状态: {state.state}, 进度: {state.executing_plan_no}/{state.total_plan_no}")

    if state.state == "Completed":
        print("  Compaction 完成！\n")
        break

    time.sleep(1)

# ===== 5. 验证 Compaction 效果 =====
print("=== 验证 Compaction 效果 ===\n")

# 查看 Segment 信息（Compaction 后）
print("步骤1：查看 Segment 信息（Compaction 后）")
segments_after = utility.get_query_segment_info(collection_name)
print(f"  Segment 数量: {len(segments_after)}")
total_rows_after = sum(s.num_rows for s in segments_after)
print(f"  总行数: {total_rows_after}")
print()

print("步骤2：对比分析")
print(f"  Segment 数量变化: {len(segments_before)} → {len(segments_after)}")
print(f"  总行数变化: {total_rows_before} → {total_rows_after}")
print(f"  空间回收: {total_rows_before - total_rows_after} 条")
print()

# ===== 6. 性能对比测试 =====
print("=== 性能对比测试 ===\n")

# 创建索引
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

# 测试查询性能
query_vector = np.random.rand(128).tolist()
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

print("测试查询性能（100 次查询）")
start_time = time.time()
for _ in range(100):
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=10
    )
end_time = time.time()

avg_latency = (end_time - start_time) / 100 * 1000
print(f"  平均查询延迟: {avg_latency:.2f}ms")
print()

# ===== 7. 定期 Compaction 策略 =====
print("=== 定期 Compaction 策略 ===\n")

print("最佳实践：")
print("  1. 自动 Compaction：让 Milvus 自动运行（推荐）")
print("  2. 每日 Compaction：每天凌晨运行一次")
print("  3. 每周 Full Compaction：每周日运行一次全量 Compaction")
print("  4. 监控 Compaction：定期检查 Compaction 状态")
print()

# ===== 8. 清理资源 =====
print("=== 清理资源 ===")
collection.release()
utility.drop_collection(collection_name)
connections.disconnect("default")
print("清理完成！")
```

---

## 核心要点

1. **Compaction 回收空间**：删除数据后需要运行 Compaction 才能真正释放空间
2. **监控 Compaction 状态**：使用 `get_compaction_state()` 监控进度
3. **性能提升显著**：Compaction 后查询性能可提升 30%-50%
4. **定期运行策略**：建议每周运行一次 Full Compaction

---

## 扩展练习

1. **测试不同删除比例**：测试删除 10%、30%、50%、70% 数据后的 Compaction 效果
2. **监控存储空间**：监控 Compaction 前后的磁盘空间变化
3. **实现自动 Compaction**：实现定期自动运行 Compaction 的脚本
4. **性能基准测试**：对比 Compaction 前后的查询性能
