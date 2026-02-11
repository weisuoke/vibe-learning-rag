# 核心概念3：动态Schema性能优化与最佳实践

> 掌握动态Schema的性能优化策略和生产环境最佳实践

---

## 1. 性能特点分析

### 1.1 性能对比

| 操作 | 固定字段 | 动态字段 | 性能差异 |
|------|---------|---------|---------|
| 插入 | 快 | 快 | 相近 |
| 查询（无过滤） | 快 | 快 | 相近 |
| 标量过滤 | 快（有索引） | 慢（无索引） | 10-100x |
| 向量检索 | 快 | 快 | 相近 |
| 混合检索 | 快 | 慢 | 5-50x |

### 1.2 性能瓶颈

1. **无索引支持**：动态字段不支持索引，查询需要全表扫描
2. **行式存储**：缓存不友好，内存访问效率低
3. **类型推断开销**：查询时需要推断类型

---

## 2. 优化策略

### 2.1 高频字段迁移

```python
# 监控查询频率
query_stats = {
    "author": 1000,   # 高频 → 迁移到固定字段
    "category": 500,  # 中频 → 考虑迁移
    "tags": 10        # 低频 → 保持动态
}

# 迁移高频字段到固定Schema
new_schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100)  # 迁移
    ],
    enable_dynamic_field=True
)
```

### 2.2 分层查询策略

```python
# 先用固定字段过滤（快），再用动态字段过滤（慢）
results = collection.search(
    data=[[0.1]*768],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=1000,
    expr='author == "Alice"',  # 固定字段过滤（快）
    output_fields=["*"]
)

# 应用层过滤动态字段
filtered = [r for r in results[0] if r.entity.get("priority", 0) > 3]
```

### 2.3 缓存优化

```python
import functools

@functools.lru_cache(maxsize=1000)
def query_by_dynamic_field(field_name, field_value):
    return collection.query(
        expr=f'{field_name} == "{field_value}"',
        output_fields=["*"]
    )
```

---

## 3. 最佳实践

### 3.1 混合Schema设计

```python
# 核心字段固定，扩展字段动态
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),  # 高频
    ],
    enable_dynamic_field=True  # 低频字段动态
)
```

### 3.2 渐进式演化

```python
# 阶段1：快速原型（全部动态）
# 阶段2：优化高频字段（迁移到固定）
# 阶段3：固定Schema（生产环境）
```

### 3.3 类型一致性保证

```python
def validate_dynamic_fields(data, field_types):
    for field_name, expected_type in field_types.items():
        if field_name in data:
            actual_type = type(data[field_name])
            if actual_type != expected_type:
                raise TypeError(f"Type mismatch for {field_name}")
```

---

## 4. 生产环境建议

### 4.1 监控指标

- 查询频率：识别高频动态字段
- 查询延迟：监控动态字段查询性能
- 存储大小：控制动态字段数量和大小

### 4.2 优化决策

| 查询频率 | 建议 |
|---------|------|
| > 10% | 迁移到固定字段 |
| 1-10% | 考虑迁移或缓存 |
| < 1% | 保持动态字段 |

### 4.3 容量规划

- 动态字段数量：< 20个
- 动态字段值大小：< 1KB
- 嵌套层级：< 3层

---

## 总结

**核心原则**：
1. 高频字段使用固定Schema
2. 低频字段使用动态Schema
3. 分层查询优化性能
4. 渐进式Schema演化
5. 监控并持续优化

**记住**：动态Schema是灵活性和性能的权衡，根据实际需求选择合适的策略。
