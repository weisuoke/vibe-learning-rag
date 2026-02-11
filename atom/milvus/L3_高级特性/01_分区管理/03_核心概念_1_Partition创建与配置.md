# 核心概念 1: Partition 创建与配置

## 什么是 Partition 创建与配置？

**Partition 创建与配置**是指在 Milvus Collection 中创建分区、设计分区命名策略、配置分区参数的过程，是使用 Partition 的第一步。

### 核心定义

```python
# Partition 创建的本质
partition = collection.create_partition(
    partition_name="partition_2024_01"  # 分区名称
)
```

**类比理解**:
- **前端类比**: 就像在数据库中创建表分区 `CREATE TABLE users PARTITION BY RANGE (date)`
- **日常类比**: 就像在图书馆的书架上划分不同的层，每层存放不同类型的书

---

## 1. Partition 创建 API

### 1.1 基础创建

```python
from pymilvus import Collection, connections

# 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 获取 Collection
collection = Collection("my_collection")

# 创建分区
partition = collection.create_partition("partition_2024")

print(f"分区创建成功: {partition.name}")
print(f"分区描述: {partition.description}")
```

**关键点**:
- 分区名必须唯一
- 分区名只能包含字母、数字、下划线
- 分区名长度限制为 255 个字符
- 每个 Collection 默认有一个 `_default` 分区

---

### 1.2 带描述的创建

```python
# 创建带描述的分区
partition = collection.create_partition(
    partition_name="partition_2024_01",
    description="2024年1月的数据分区"
)

print(f"分区名: {partition.name}")
print(f"分区描述: {partition.description}")
```

**最佳实践**:
- 使用有意义的描述
- 描述应包含分区的用途、时间范围、数据特征
- 便于后续维护和理解

---

### 1.3 检查分区是否存在

```python
# 检查分区是否存在
if collection.has_partition("partition_2024"):
    print("分区已存在")
    partition = collection.partition("partition_2024")
else:
    print("分区不存在，创建新分区")
    partition = collection.create_partition("partition_2024")
```

**防御性编程**:
- 创建前先检查是否存在
- 避免重复创建导致错误
- 提高代码健壮性

---

### 1.4 获取已有分区

```python
# 方式1：通过名称获取
partition = collection.partition("partition_2024")

# 方式2：列出所有分区
partitions = collection.partitions
for p in partitions:
    print(f"分区名: {p.name}, 数据量: {p.num_entities}")

# 方式3：获取分区名列表
partition_names = [p.name for p in collection.partitions]
print(f"所有分区: {partition_names}")
```

---

## 2. Partition 命名策略

### 2.1 时间分区命名

**按年分区**:
```python
# 格式: partition_YYYY
partitions = ["partition_2022", "partition_2023", "partition_2024"]

for name in partitions:
    collection.create_partition(name)
```

**适用场景**:
- 数据跨度多年
- 查询通常按年范围
- 数据量巨大，按年分区合适

---

**按月分区** (推荐):
```python
# 格式: partition_YYYY_MM
from datetime import datetime, timedelta

def create_monthly_partitions(start_date, months):
    """创建指定月数的分区"""
    partitions = []
    for i in range(months):
        date = start_date + timedelta(days=30*i)
        partition_name = f"partition_{date.strftime('%Y_%m')}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
        partitions.append(partition_name)
    return partitions

# 创建最近12个月的分区
start = datetime(2023, 1, 1)
partitions = create_monthly_partitions(start, 12)
print(f"创建了 {len(partitions)} 个月度分区")
```

**适用场景**:
- 数据按月增长
- 查询通常按月或季度范围
- 平衡分区数量和粒度

---

**按周分区**:
```python
# 格式: partition_YYYY_WW
import datetime

def create_weekly_partitions(year, weeks):
    """创建指定周数的分区"""
    partitions = []
    for week in range(1, weeks + 1):
        partition_name = f"partition_{year}_W{week:02d}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
        partitions.append(partition_name)
    return partitions

# 创建2024年的52周分区
partitions = create_weekly_partitions(2024, 52)
```

**适用场景**:
- 数据更新频繁
- 查询通常按周范围
- 需要更细粒度的时间控制

---

**按天分区** (谨慎使用):
```python
# 格式: partition_YYYY_MM_DD
from datetime import datetime, timedelta

def create_daily_partitions(start_date, days):
    """创建指定天数的分区"""
    partitions = []
    for i in range(days):
        date = start_date + timedelta(days=i)
        partition_name = f"partition_{date.strftime('%Y_%m_%d')}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
        partitions.append(partition_name)
    return partitions

# 创建最近30天的分区
start = datetime.now() - timedelta(days=30)
partitions = create_daily_partitions(start, 30)
```

**适用场景**:
- 数据量巨大，按天分区合适
- 查询通常精确到天
- 有足够的管理能力维护大量分区

**注意事项**:
- 1年 = 365个分区，管理复杂度高
- 只在确实需要时使用
- 考虑定期合并或清理旧分区

---

### 2.2 租户分区命名

**按租户 ID 分区**:
```python
# 格式: tenant_{tenant_id}
def create_tenant_partition(tenant_id, company_name):
    """为新租户创建分区"""
    partition_name = f"tenant_{tenant_id}"
    if not collection.has_partition(partition_name):
        partition = collection.create_partition(
            partition_name,
            description=f"租户: {company_name}"
        )
        print(f"为租户 {company_name} 创建分区: {partition_name}")
        return partition
    else:
        print(f"租户分区已存在: {partition_name}")
        return collection.partition(partition_name)

# 示例
create_tenant_partition("001", "公司A")
create_tenant_partition("002", "公司B")
create_tenant_partition("003", "公司C")
```

**适用场景**:
- SaaS 多租户系统
- 需要数据隔离
- 每个租户独立管理

---

**按租户类型分区**:
```python
# 格式: tenant_{type}_{id}
def create_typed_tenant_partition(tenant_type, tenant_id, company_name):
    """按租户类型创建分区"""
    partition_name = f"tenant_{tenant_type}_{tenant_id}"
    if not collection.has_partition(partition_name):
        partition = collection.create_partition(
            partition_name,
            description=f"{tenant_type}租户: {company_name}"
        )
        return partition
    return collection.partition(partition_name)

# 示例：区分企业版和个人版
create_typed_tenant_partition("enterprise", "001", "大型企业A")
create_typed_tenant_partition("personal", "001", "个人用户A")
```

**适用场景**:
- 租户有不同类型（企业版、个人版）
- 不同类型需要不同的配置或策略
- 便于按类型统计和管理

---

### 2.3 类别分区命名

**按业务类别分区**:
```python
# 格式: category_{category_name}
categories = ["tech", "finance", "healthcare", "education", "retail"]

for category in categories:
    partition_name = f"category_{category}"
    if not collection.has_partition(partition_name):
        collection.create_partition(
            partition_name,
            description=f"{category}类别的数据"
        )

print(f"创建了 {len(categories)} 个类别分区")
```

**适用场景**:
- 数据有明确的类别划分
- 查询通常按类别过滤
- 类别数量有限（< 50个）

---

**按地区分区**:
```python
# 格式: region_{region_code}
regions = {
    "beijing": "北京",
    "shanghai": "上海",
    "guangzhou": "广州",
    "shenzhen": "深圳"
}

for code, name in regions.items():
    partition_name = f"region_{code}"
    if not collection.has_partition(partition_name):
        collection.create_partition(
            partition_name,
            description=f"{name}地区的数据"
        )
```

**适用场景**:
- 数据有地域特征
- 查询通常按地区过滤
- 需要地域隔离或优化

---

### 2.4 混合分区命名

**时间 + 类别**:
```python
# 格式: {category}_YYYY_MM
def create_category_time_partition(category, year, month):
    """创建类别+时间分区"""
    partition_name = f"{category}_{year}_{month:02d}"
    if not collection.has_partition(partition_name):
        collection.create_partition(
            partition_name,
            description=f"{category}类别 {year}年{month}月的数据"
        )
    return partition_name

# 示例：为每个类别创建月度分区
categories = ["tech", "finance", "healthcare"]
for category in categories:
    for month in range(1, 13):
        create_category_time_partition(category, 2024, month)

# 结果：tech_2024_01, tech_2024_02, ..., healthcare_2024_12
```

**适用场景**:
- 数据同时有类别和时间特征
- 查询通常同时指定类别和时间范围
- 需要更细粒度的分区控制

**注意事项**:
- 分区数量 = 类别数 × 时间分区数
- 容易导致分区爆炸，谨慎使用
- 只在确实需要时使用

---

## 3. Partition 配置最佳实践

### 3.1 分区数量控制

**推荐范围**:
```python
# 根据数据规模确定分区数量
def recommend_partition_count(total_entities):
    """根据数据量推荐分区数量"""
    if total_entities < 1_000_000:
        return 1, 10  # 1-10个分区
    elif total_entities < 10_000_000:
        return 10, 50  # 10-50个分区
    else:
        return 50, 200  # 50-200个分区

# 示例
total = 5_000_000
min_count, max_count = recommend_partition_count(total)
print(f"推荐分区数量: {min_count}-{max_count} 个")
```

**分区大小控制**:
```python
# 推荐每个分区的数据量
RECOMMENDED_PARTITION_SIZE = {
    "min": 100_000,      # 最小 10万条
    "optimal": 1_000_000,  # 最优 100万条
    "max": 10_000_000    # 最大 1000万条
}

def calculate_partition_count(total_entities, target_size=1_000_000):
    """根据目标分区大小计算分区数量"""
    count = max(1, total_entities // target_size)
    return count

# 示例
total = 50_000_000
count = calculate_partition_count(total)
print(f"建议创建 {count} 个分区")
```

---

### 3.2 分区命名规范

**命名规范**:
```python
# 好的命名规范
GOOD_NAMES = [
    "partition_2024_01",      # 清晰的时间标识
    "tenant_company_a",       # 清晰的租户标识
    "category_tech",          # 清晰的类别标识
    "region_beijing"          # 清晰的地区标识
]

# 不好的命名
BAD_NAMES = [
    "p1",                     # 无意义
    "partition",              # 太通用
    "data_2024",              # 不够具体
    "test"                    # 临时命名
]

# 命名验证函数
import re

def validate_partition_name(name):
    """验证分区名是否符合规范"""
    # 规则1：只包含字母、数字、下划线
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        return False, "只能包含字母、数字、下划线"

    # 规则2：长度限制
    if len(name) > 255:
        return False, "长度不能超过255个字符"

    # 规则3：不能以数字开头
    if name[0].isdigit():
        return False, "不能以数字开头"

    # 规则4：不能是保留字
    if name in ["_default", "default"]:
        return False, "不能使用保留字"

    return True, "命名符合规范"

# 测试
names = ["partition_2024", "123_partition", "partition-2024", "_default"]
for name in names:
    valid, message = validate_partition_name(name)
    print(f"{name}: {message}")
```

---

### 3.3 分区创建工具函数

**通用分区创建器**:
```python
from typing import List, Optional
from datetime import datetime

class PartitionManager:
    """分区管理器"""

    def __init__(self, collection):
        self.collection = collection

    def create_time_partitions(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "month"
    ) -> List[str]:
        """创建时间分区

        Args:
            start_date: 开始日期
            end_date: 结束日期
            granularity: 粒度 (year/month/week/day)

        Returns:
            创建的分区名列表
        """
        partitions = []
        current = start_date

        while current <= end_date:
            if granularity == "year":
                partition_name = f"partition_{current.strftime('%Y')}"
                current = datetime(current.year + 1, 1, 1)
            elif granularity == "month":
                partition_name = f"partition_{current.strftime('%Y_%m')}"
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)
            elif granularity == "week":
                week = current.isocalendar()[1]
                partition_name = f"partition_{current.year}_W{week:02d}"
                current += timedelta(weeks=1)
            elif granularity == "day":
                partition_name = f"partition_{current.strftime('%Y_%m_%d')}"
                current += timedelta(days=1)
            else:
                raise ValueError(f"不支持的粒度: {granularity}")

            if not self.collection.has_partition(partition_name):
                self.collection.create_partition(partition_name)
                partitions.append(partition_name)

        return partitions

    def create_tenant_partitions(
        self,
        tenant_ids: List[str],
        tenant_names: Optional[List[str]] = None
    ) -> List[str]:
        """创建租户分区

        Args:
            tenant_ids: 租户ID列表
            tenant_names: 租户名称列表（可选）

        Returns:
            创建的分区名列表
        """
        partitions = []
        for i, tenant_id in enumerate(tenant_ids):
            partition_name = f"tenant_{tenant_id}"
            description = None
            if tenant_names and i < len(tenant_names):
                description = f"租户: {tenant_names[i]}"

            if not self.collection.has_partition(partition_name):
                self.collection.create_partition(partition_name, description=description)
                partitions.append(partition_name)

        return partitions

    def create_category_partitions(
        self,
        categories: List[str]
    ) -> List[str]:
        """创建类别分区

        Args:
            categories: 类别列表

        Returns:
            创建的分区名列表
        """
        partitions = []
        for category in categories:
            partition_name = f"category_{category}"
            if not self.collection.has_partition(partition_name):
                self.collection.create_partition(
                    partition_name,
                    description=f"{category}类别的数据"
                )
                partitions.append(partition_name)

        return partitions

    def list_partitions(self) -> List[dict]:
        """列出所有分区及其信息"""
        partitions_info = []
        for partition in self.collection.partitions:
            info = {
                "name": partition.name,
                "description": partition.description,
                "num_entities": partition.num_entities
            }
            partitions_info.append(info)
        return partitions_info

# 使用示例
manager = PartitionManager(collection)

# 创建月度分区
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
time_partitions = manager.create_time_partitions(start, end, granularity="month")
print(f"创建了 {len(time_partitions)} 个月度分区")

# 创建租户分区
tenant_ids = ["001", "002", "003"]
tenant_names = ["公司A", "公司B", "公司C"]
tenant_partitions = manager.create_tenant_partitions(tenant_ids, tenant_names)
print(f"创建了 {len(tenant_partitions)} 个租户分区")

# 列出所有分区
partitions_info = manager.list_partitions()
for info in partitions_info:
    print(f"分区: {info['name']}, 数据量: {info['num_entities']}")
```

---

## 4. Schema 设计与分区

### 4.1 分区友好的 Schema 设计

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

# 设计支持分区的 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),

    # 时间字段（用于时间分区）
    FieldSchema(name="created_at", dtype=DataType.INT64),

    # 租户字段（用于租户分区）
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=50),

    # 类别字段（用于类别分区）
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),

    # 其他字段
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]

schema = CollectionSchema(fields, description="支持多种分区策略的 Schema")
collection = Collection("multi_partition_collection", schema)
```

**设计原则**:
- 包含分区相关的字段（时间、租户、类别）
- 字段类型选择合适（时间用 INT64 存储时间戳）
- 字段长度合理（VARCHAR 长度不要过大）

---

### 4.2 插入数据时确定分区

```python
from datetime import datetime

def insert_with_auto_partition(embedding, text, tenant_id, category):
    """插入数据时自动确定分区"""

    # 策略1：按时间分区
    current_month = datetime.now().strftime("%Y_%m")
    partition_name = f"partition_{current_month}"

    # 策略2：按租户分区
    # partition_name = f"tenant_{tenant_id}"

    # 策略3：按类别分区
    # partition_name = f"category_{category}"

    # 确保分区存在
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)

    # 插入到指定分区
    partition = collection.partition(partition_name)
    data = [
        [embedding],
        [int(datetime.now().timestamp())],  # created_at
        [tenant_id],
        [category],
        [text]
    ]
    partition.insert(data)

    return partition_name

# 使用示例
embedding = [0.1] * 128
partition_name = insert_with_auto_partition(
    embedding=embedding,
    text="示例文档",
    tenant_id="001",
    category="tech"
)
print(f"数据插入到分区: {partition_name}")
```

---

## 5. 在实际应用中的使用

### 5.1 RAG 系统的分区策略

```python
# RAG 文档问答系统的分区设计

# 策略1：按文档上传时间分区（推荐）
def create_rag_time_partitions():
    """为 RAG 系统创建时间分区"""
    manager = PartitionManager(collection)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    partitions = manager.create_time_partitions(start, end, granularity="month")
    return partitions

# 策略2：按文档类型分区
def create_rag_category_partitions():
    """为 RAG 系统创建类别分区"""
    categories = ["technical_doc", "user_manual", "faq", "blog_post"]
    manager = PartitionManager(collection)
    partitions = manager.create_category_partitions(categories)
    return partitions

# 策略3：按租户分区（多租户 RAG）
def create_rag_tenant_partitions(tenant_ids):
    """为多租户 RAG 系统创建租户分区"""
    manager = PartitionManager(collection)
    partitions = manager.create_tenant_partitions(tenant_ids)
    return partitions
```

---

### 5.2 推荐系统的分区策略

```python
# 推荐系统的分区设计

# 策略1：按商品类别分区
def create_recommendation_category_partitions():
    """为推荐系统创建类别分区"""
    categories = ["electronics", "clothing", "food", "books", "toys"]
    manager = PartitionManager(collection)
    partitions = manager.create_category_partitions(categories)
    return partitions

# 策略2：按地区分区
def create_recommendation_region_partitions():
    """为推荐系统创建地区分区"""
    regions = ["beijing", "shanghai", "guangzhou", "shenzhen"]
    partitions = []
    for region in regions:
        partition_name = f"region_{region}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
            partitions.append(partition_name)
    return partitions
```

---

## 总结

### 核心要点

1. **创建 API**: 使用 `collection.create_partition()` 创建分区
2. **命名策略**: 根据业务特征选择合适的命名策略（时间/租户/类别）
3. **数量控制**: 推荐 10-200 个分区，避免分区爆炸
4. **命名规范**: 使用有意义的、规范的分区名
5. **工具函数**: 使用 PartitionManager 简化分区管理

### 最佳实践

- ✅ 创建前检查分区是否存在
- ✅ 使用有意义的分区名和描述
- ✅ 根据查询模式设计分区策略
- ✅ 控制分区数量在合理范围
- ✅ 使用工具函数简化分区管理

### 下一步

学习 **核心概念 2: Partition 数据操作**，了解如何在分区中插入、查询、管理数据。
