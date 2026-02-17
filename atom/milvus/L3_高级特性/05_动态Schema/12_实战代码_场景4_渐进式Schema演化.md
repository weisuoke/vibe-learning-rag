# 实战代码 - 场景4：渐进式Schema演化

> 从快速原型到生产环境的Schema演化策略

---

## 场景描述

在实际项目中，Schema需求会随着项目发展而演化：
- **阶段1（原型期）**：快速迭代，全部使用动态Schema
- **阶段2（优化期）**：识别高频字段，迁移到固定Schema
- **阶段3（生产期）**：固定Schema，关闭动态字段

本场景演示完整的Schema演化过程。

---

## 完整代码

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

class SchemaEvolutionDemo:
    """Schema演化演示"""

    def __init__(self):
        self.model = None
        self.query_stats = {}  # 查询统计

    def setup(self):
        """初始化"""
        print("=" * 60)
        print("渐进式Schema演化 - 初始化")
        print("=" * 60)

        connections.connect(host="localhost", port="19530")
        print("✅ 已连接到Milvus")

        # 加载Embedding模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 模型加载完成")

    def phase1_prototype(self):
        """阶段1：快速原型（全部动态Schema）"""
        print("\n" + "=" * 60)
        print("阶段1：快速原型期")
        print("=" * 60)

        # 删除旧Collection
        if utility.has_collection("schema_v1"):
            utility.drop_collection("schema_v1")

        # 创建最小Schema（只有必需字段）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Prototype with full dynamic schema",
            enable_dynamic_field=True  # 全部使用动态字段
        )

        collection = Collection(name="schema_v1", schema=schema)
        print("✅ 已创建Collection: schema_v1")
        print("   - 固定字段: id, embedding")
        print("   - 动态字段: 所有业务字段")

        # 插入测试数据
        documents = [
            {
                "text": "Introduction to vector databases",
                "author": "Alice",
                "category": "database",
                "priority": 5
            },
            {
                "text": "Machine learning basics",
                "author": "Bob",
                "category": "AI",
                "difficulty": "beginner"
            },
            {
                "text": "RAG system architecture",
                "author": "Alice",
                "category": "AI",
                "priority": 8
            }
        ]

        data = []
        for doc in documents:
            embedding = self.model.encode(doc["text"]).tolist()
            item = {"embedding": embedding}
            item.update(doc)
            data.append(item)

        collection.insert(data)
        collection.flush()

        # 创建索引并加载
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        print(f"✅ 已插入 {len(data)} 条数据")
        print("\n特点:")
        print("  - 快速迭代，无需预先设计Schema")
        print("  - 所有字段都是动态的，灵活性最高")
        print("  - 适合快速验证想法")

        return collection

    def simulate_queries_v1(self, collection):
        """模拟查询（阶段1）"""
        print("\n" + "=" * 60)
        print("模拟查询并统计字段使用频率")
        print("=" * 60)

        # 模拟100次查询
        query_patterns = [
            ('author == "Alice"', 40),      # 40%的查询按author过滤
            ('category == "AI"', 30),       # 30%的查询按category过滤
            ('priority > 5', 20),           # 20%的查询按priority过滤
            ('difficulty == "beginner"', 10)  # 10%的查询按difficulty过滤
        ]

        self.query_stats = {}
        total_queries = 100

        for expr, count in query_patterns:
            field_name = expr.split()[0]
            self.query_stats[field_name] = count

            # 模拟查询
            for _ in range(count):
                results = collection.query(
                    expr=expr,
                    output_fields=["*"],
                    limit=10
                )

        print("✅ 模拟完成100次查询\n")
        print("字段查询频率统计:")
        for field, count in sorted(self.query_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_queries * 100
            print(f"  {field}: {count}次 ({percentage:.0f}%)")

        print("\n分析:")
        print("  - author: 高频字段（40%）→ 应迁移到固定Schema")
        print("  - category: 中频字段（30%）→ 应迁移到固定Schema")
        print("  - priority: 低频字段（20%）→ 可保持动态")
        print("  - difficulty: 低频字段（10%）→ 可保持动态")

    def phase2_optimization(self):
        """阶段2：优化期（高频字段迁移到固定Schema）"""
        print("\n" + "=" * 60)
        print("阶段2：优化期")
        print("=" * 60)

        # 删除旧Collection
        if utility.has_collection("schema_v2"):
            utility.drop_collection("schema_v2")

        # 创建优化后的Schema（高频字段固定）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),  # 高频 → 固定
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)  # 中频 → 固定
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Optimized with high-frequency fields fixed",
            enable_dynamic_field=True  # 低频字段保持动态
        )

        collection = Collection(name="schema_v2", schema=schema)
        print("✅ 已创建Collection: schema_v2")
        print("   - 固定字段: id, embedding, text, author, category")
        print("   - 动态字段: priority, difficulty等低频字段")

        # 迁移数据
        old_collection = Collection("schema_v1")
        old_data = old_collection.query(expr="id > 0", output_fields=["*"], limit=1000)

        new_data = []
        for item in old_data:
            new_item = {
                "embedding": item["embedding"],
                "text": item.get("text", ""),
                "author": item.get("author", ""),
                "category": item.get("category", "")
            }

            # 低频字段保持动态
            for key in ["priority", "difficulty"]:
                if key in item:
                    new_item[key] = item[key]

            new_data.append(new_item)

        collection.insert(new_data)
        collection.flush()

        # 为固定字段创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        # 为高频字段创建索引
        collection.create_index(field_name="author", index_params={"index_type": "TRIE"})
        collection.create_index(field_name="category", index_params={"index_type": "TRIE"})

        collection.load()

        print(f"✅ 已迁移 {len(new_data)} 条数据")
        print("\n特点:")
        print("  - 高频字段有索引，查询性能提升10-100倍")
        print("  - 低频字段保持动态，保留灵活性")
        print("  - 平衡性能和灵活性")

        return collection

    def compare_performance(self, collection_v1, collection_v2):
        """性能对比"""
        print("\n" + "=" * 60)
        print("性能对比测试")
        print("=" * 60)

        import time

        # 测试1：按author查询（高频字段）
        print("\n测试1：按author查询")

        # V1（动态字段，无索引）
        start = time.time()
        for _ in range(10):
            collection_v1.query(expr='author == "Alice"', output_fields=["*"])
        v1_time = time.time() - start

        # V2（固定字段，有索引）
        start = time.time()
        for _ in range(10):
            collection_v2.query(expr='author == "Alice"', output_fields=["*"])
        v2_time = time.time() - start

        print(f"  V1（动态字段）: {v1_time:.3f}秒")
        print(f"  V2（固定字段）: {v2_time:.3f}秒")
        print(f"  性能提升: {v1_time / v2_time:.1f}x")

        # 测试2：按priority查询（低频字段）
        print("\n测试2：按priority查询")

        # V1（动态字段）
        start = time.time()
        for _ in range(10):
            collection_v1.query(expr='priority > 5', output_fields=["*"])
        v1_time = time.time() - start

        # V2（动态字段）
        start = time.time()
        for _ in range(10):
            collection_v2.query(expr='priority > 5', output_fields=["*"])
        v2_time = time.time() - start

        print(f"  V1（动态字段）: {v1_time:.3f}秒")
        print(f"  V2（动态字段）: {v2_time:.3f}秒")
        print(f"  性能差异: {abs(v1_time - v2_time) / v1_time * 100:.1f}%（相近）")

    def phase3_production(self):
        """阶段3：生产期（固定Schema）"""
        print("\n" + "=" * 60)
        print("阶段3：生产期")
        print("=" * 60)

        # 删除旧Collection
        if utility.has_collection("schema_v3"):
            utility.drop_collection("schema_v3")

        # 创建生产Schema（所有字段固定）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="priority", dtype=DataType.INT64),
            FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=50)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Production with all fields fixed",
            enable_dynamic_field=False  # 关闭动态字段
        )

        collection = Collection(name="schema_v3", schema=schema)
        print("✅ 已创建Collection: schema_v3")
        print("   - 固定字段: 所有字段")
        print("   - 动态字段: 已关闭")

        # 迁移数据
        old_collection = Collection("schema_v2")
        old_data = old_collection.query(expr="id > 0", output_fields=["*"], limit=1000)

        new_data = []
        for item in old_data:
            new_item = {
                "embedding": item["embedding"],
                "text": item.get("text", ""),
                "author": item.get("author", ""),
                "category": item.get("category", ""),
                "priority": item.get("priority", 0),
                "difficulty": item.get("difficulty", "")
            }
            new_data.append(new_item)

        collection.insert(new_data)
        collection.flush()

        # 为所有字段创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.create_index(field_name="author", index_params={"index_type": "TRIE"})
        collection.create_index(field_name="category", index_params={"index_type": "TRIE"})

        collection.load()

        print(f"✅ 已迁移 {len(new_data)} 条数据")
        print("\n特点:")
        print("  - 所有字段都有索引，性能最优")
        print("  - 类型安全，编译时检查")
        print("  - 适合生产环境")

        return collection

    def evolution_summary(self):
        """演化总结"""
        print("\n" + "=" * 60)
        print("Schema演化总结")
        print("=" * 60)

        summary = """
阶段1：快速原型期
  Schema策略: 最小固定字段 + 全部动态字段
  优势: 快速迭代，灵活性最高
  劣势: 查询性能低，无类型检查
  适用: 项目初期，需求不明确

阶段2：优化期
  Schema策略: 核心固定字段 + 高频固定字段 + 低频动态字段
  优势: 平衡性能和灵活性
  劣势: 需要数据迁移
  适用: 需求逐渐明确，开始优化性能

阶段3：生产期
  Schema策略: 所有字段固定，关闭动态字段
  优势: 性能最优，类型安全
  劣势: 灵活性最低
  适用: 需求稳定，生产环境

演化决策矩阵:
  查询频率 > 10%  → 迁移到固定字段
  查询频率 1-10%  → 考虑迁移或缓存
  查询频率 < 1%   → 保持动态字段

性能提升:
  高频字段迁移: 10-100x
  所有字段固定: 20-50x
        """

        print(summary)

    def cleanup(self):
        """清理资源"""
        print("\n" + "=" * 60)
        print("清理资源")
        print("=" * 60)

        for collection_name in ["schema_v1", "schema_v2", "schema_v3"]:
            if utility.has_collection(collection_name):
                Collection(collection_name).release()

        connections.disconnect("default")
        print("✅ 资源清理完成")

    def run(self):
        """运行完整演示"""
        try:
            # 1. 初始化
            self.setup()

            # 2. 阶段1：快速原型
            collection_v1 = self.phase1_prototype()

            # 3. 模拟查询并统计
            self.simulate_queries_v1(collection_v1)

            # 4. 阶段2：优化期
            collection_v2 = self.phase2_optimization()

            # 5. 性能对比
            self.compare_performance(collection_v1, collection_v2)

            # 6. 阶段3：生产期
            collection_v3 = self.phase3_production()

            # 7. 演化总结
            self.evolution_summary()

            # 8. 清理
            self.cleanup()

            print("\n" + "=" * 60)
            print("✅ 渐进式Schema演化演示完成！")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo = SchemaEvolutionDemo()
    demo.run()
```

---

## 演化策略

### 决策流程图

```
项目启动
    ↓
阶段1：快速原型
  - 最小固定字段
  - 全部动态字段
    ↓
监控查询频率
    ↓
识别高频字段（> 10%）
    ↓
阶段2：优化期
  - 高频字段 → 固定
  - 低频字段 → 动态
    ↓
持续监控
    ↓
需求稳定？
    ↓ 是
阶段3：生产期
  - 所有字段固定
  - 关闭动态字段
```

### 迁移检查清单

**阶段1 → 阶段2**：
- [ ] 统计字段查询频率
- [ ] 识别高频字段（> 10%）
- [ ] 设计新Schema（高频字段固定）
- [ ] 创建新Collection
- [ ] 迁移数据
- [ ] 创建索引
- [ ] 性能测试
- [ ] 切换应用
- [ ] 删除旧Collection

**阶段2 → 阶段3**：
- [ ] 确认需求稳定
- [ ] 设计最终Schema（所有字段固定）
- [ ] 创建新Collection
- [ ] 迁移数据
- [ ] 创建所有索引
- [ ] 性能测试
- [ ] 切换应用
- [ ] 删除旧Collection

---

## 监控指标

### 1. 查询频率统计

```python
class QueryMonitor:
    """查询监控器"""

    def __init__(self):
        self.field_query_count = {}
        self.total_queries = 0

    def record_query(self, expr):
        """记录查询"""
        self.total_queries += 1

        # 解析查询表达式，提取字段名
        fields = self.extract_fields(expr)
        for field in fields:
            self.field_query_count[field] = self.field_query_count.get(field, 0) + 1

    def extract_fields(self, expr):
        """从表达式中提取字段名"""
        import re
        # 简化版：提取 field_name == value 中的 field_name
        pattern = r'(\w+)\s*(?:==|!=|>|<|>=|<=|in|not in)'
        return re.findall(pattern, expr)

    def get_statistics(self):
        """获取统计信息"""
        stats = []
        for field, count in self.field_query_count.items():
            percentage = count / self.total_queries * 100
            stats.append({
                "field": field,
                "count": count,
                "percentage": percentage,
                "recommendation": self.get_recommendation(percentage)
            })

        return sorted(stats, key=lambda x: x["percentage"], reverse=True)

    def get_recommendation(self, percentage):
        """获取建议"""
        if percentage > 10:
            return "迁移到固定字段"
        elif percentage > 1:
            return "考虑迁移或缓存"
        else:
            return "保持动态字段"

# 使用示例
monitor = QueryMonitor()

# 记录查询
monitor.record_query('author == "Alice"')
monitor.record_query('category == "AI"')
monitor.record_query('author == "Bob"')

# 获取统计
stats = monitor.get_statistics()
for stat in stats:
    print(f"{stat['field']}: {stat['percentage']:.1f}% - {stat['recommendation']}")
```

### 2. 性能监控

```python
import time

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.query_times = {}

    def measure_query(self, field, query_func):
        """测量查询性能"""
        start = time.time()
        result = query_func()
        elapsed = time.time() - start

        if field not in self.query_times:
            self.query_times[field] = []

        self.query_times[field].append(elapsed)
        return result

    def get_average_time(self, field):
        """获取平均查询时间"""
        if field not in self.query_times:
            return 0

        times = self.query_times[field]
        return sum(times) / len(times)

    def get_slow_fields(self, threshold=0.1):
        """获取慢查询字段"""
        slow_fields = []
        for field, times in self.query_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > threshold:
                slow_fields.append({
                    "field": field,
                    "avg_time": avg_time,
                    "query_count": len(times)
                })

        return sorted(slow_fields, key=lambda x: x["avg_time"], reverse=True)
```

---

## 最佳实践

### 1. 渐进式演化，不要一步到位

```python
# ❌ 错误：直接从全动态跳到全固定
# 阶段1 → 阶段3（跳过阶段2）

# ✅ 正确：渐进式演化
# 阶段1 → 阶段2 → 阶段3
```

### 2. 基于数据决策，不要凭感觉

```python
# ❌ 错误：凭感觉决定哪些字段固定
# "我觉得author应该固定"

# ✅ 正确：基于查询统计决定
# 监控显示author查询频率40% → 固定
```

### 3. 保留灵活性，不要过早优化

```python
# ❌ 错误：项目初期就固定所有字段
# 需求还不明确，过早固定会限制灵活性

# ✅ 正确：初期使用动态Schema
# 快速迭代，需求明确后再优化
```

### 4. 平滑迁移，不要影响业务

```python
# ✅ 正确的迁移流程
# 1. 创建新Collection
# 2. 双写（同时写入新旧Collection）
# 3. 迁移历史数据
# 4. 切换读取（从新Collection读取）
# 5. 停止双写
# 6. 删除旧Collection
```

---

## 总结

**核心价值**：
1. 适应项目不同阶段的需求
2. 平衡灵活性和性能
3. 降低迁移风险
4. 基于数据决策

**演化路径**：
- 阶段1：快速原型（全动态）
- 阶段2：优化期（混合）
- 阶段3：生产期（全固定）

**决策依据**：
- 查询频率 > 10%：迁移到固定
- 查询频率 1-10%：考虑迁移
- 查询频率 < 1%：保持动态

**记住**：Schema演化是一个持续的过程，需要根据实际情况不断调整。
