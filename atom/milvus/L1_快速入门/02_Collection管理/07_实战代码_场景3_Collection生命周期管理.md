# Collection管理 - 实战代码场景3：Collection生命周期管理

> 多租户RAG系统的Collection管理：工厂模式 + 按需加载 + 自动清理 + 健康检查

---

## 场景描述

**应用场景：** 多租户RAG系统的Collection生命周期管理

**需求：**
- 支持多租户（每个租户独立Collection）
- 按需加载Collection（Lazy Loading）
- 自动清理不活跃的Collection
- 健康检查和监控
- 生产环境最佳实践

**技术栈：**
- Milvus 2.6
- pymilvus 2.6+
- Python 3.9+

**重要说明：**
根据Milvus官方建议和生产实践，建议Collection数量控制在1000以内以保证最佳性能。对于大规模多租户场景（>1000租户），推荐使用Partition Key策略而非独立Collection。

---

## 完整代码实现

```python
"""
Milvus 2.6 Collection生命周期管理 - 多租户RAG系统
演示：Collection工厂 + 按需加载 + 自动清理 + 健康检查
"""

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import numpy as np
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta

# ===== 1. Collection管理器类 =====
class CollectionManager:
    """
    Collection生命周期管理器
    
    功能：
    - Collection工厂模式
    - 按需加载（Lazy Loading）
    - 自动清理不活跃Collection
    - 健康检查
    """
    
    def __init__(self, max_loaded: int = 100):
        """
        初始化管理器
        
        Args:
            max_loaded: 最大同时加载的Collection数量
        """
        self.max_loaded = max_loaded
        self.loaded_collections: Dict[str, Dict] = {}
        self.schema_template = self._create_schema_template()
        
        print(f"✅ CollectionManager 初始化完成")
        print(f"   - 最大加载数: {max_loaded}")
    
    def _create_schema_template(self) -> CollectionSchema:
        """创建标准Schema模板"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=768),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        return CollectionSchema(fields, description="Tenant document collection")
    
    def create_tenant_collection(self, tenant_id: str) -> Collection:
        """
        为租户创建Collection
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            Collection对象
        """
        collection_name = f"tenant_{tenant_id}"
        
        # 检查是否已存在
        if utility.has_collection(collection_name):
            print(f"⚠️  租户 {tenant_id} 的Collection已存在")
            return Collection(collection_name)
        
        # 创建Collection
        collection = Collection(collection_name, self.schema_template)
        
        # 创建索引
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        print(f"✅ 租户 {tenant_id} 的Collection创建成功")
        return collection
    
    def get_collection(self, tenant_id: str, auto_create: bool = True) -> Optional[Collection]:
        """
        获取租户的Collection（按需加载）
        
        Args:
            tenant_id: 租户ID
            auto_create: 如果不存在是否自动创建
            
        Returns:
            Collection对象或None
        """
        collection_name = f"tenant_{tenant_id}"
        
        # 1. 检查缓存
        if collection_name in self.loaded_collections:
            # 更新最后访问时间
            self.loaded_collections[collection_name]["last_access"] = time.time()
            return self.loaded_collections[collection_name]["collection"]
        
        # 2. 检查是否存在
        if not utility.has_collection(collection_name):
            if auto_create:
                collection = self.create_tenant_collection(tenant_id)
            else:
                print(f"⚠️  租户 {tenant_id} 的Collection不存在")
                return None
        else:
            collection = Collection(collection_name)
        
        # 3. 加载到内存
        if not collection.is_loaded:
            collection.load()
            print(f"✅ 租户 {tenant_id} 的Collection已加载到内存")
        
        # 4. 缓存
        self.loaded_collections[collection_name] = {
            "collection": collection,
            "last_access": time.time(),
            "tenant_id": tenant_id
        }
        
        # 5. 检查是否需要清理
        if len(self.loaded_collections) > self.max_loaded:
            self._cleanup_inactive()
        
        return collection
    
    def release_collection(self, tenant_id: str):
        """释放租户的Collection（释放内存）"""
        collection_name = f"tenant_{tenant_id}"
        
        if collection_name in self.loaded_collections:
            collection = self.loaded_collections[collection_name]["collection"]
            collection.release()
            del self.loaded_collections[collection_name]
            print(f"✅ 租户 {tenant_id} 的Collection已释放")
    
    def _cleanup_inactive(self):
        """清理不活跃的Collection"""
        if len(self.loaded_collections) <= self.max_loaded:
            return
        
        # 按最后访问时间排序
        sorted_collections = sorted(
            self.loaded_collections.items(),
            key=lambda x: x[1]["last_access"]
        )
        
        # 释放最旧的Collection
        to_release = len(self.loaded_collections) - self.max_loaded
        for i in range(to_release):
            collection_name, info = sorted_collections[i]
            tenant_id = info["tenant_id"]
            self.release_collection(tenant_id)
            print(f"🧹 自动清理: 释放租户 {tenant_id} 的Collection")
    
    def health_check(self) -> Dict:
        """健康检查"""
        total_collections = len(utility.list_collections())
        loaded_count = len(self.loaded_collections)
        
        health_status = {
            "status": "healthy",
            "total_collections": total_collections,
            "loaded_collections": loaded_count,
            "max_loaded": self.max_loaded,
            "memory_usage_percent": (loaded_count / self.max_loaded * 100) if self.max_loaded > 0 else 0
        }
        
        # 检查是否超载
        if loaded_count > self.max_loaded * 0.9:
            health_status["status"] = "warning"
            health_status["message"] = "接近最大加载数"
        
        return health_status
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_collections": len(utility.list_collections()),
            "loaded_collections": len(self.loaded_collections),
            "tenant_stats": []
        }
        
        for collection_name, info in self.loaded_collections.items():
            collection = info["collection"]
            stats["tenant_stats"].append({
                "tenant_id": info["tenant_id"],
                "num_entities": collection.num_entities,
                "last_access": datetime.fromtimestamp(info["last_access"]).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return stats

# ===== 2. 演示：多租户Collection管理 =====
print("=" * 70)
print("Milvus 2.6 Collection生命周期管理演示")
print("=" * 70)

# 连接到Milvus
connections.connect("default", host="localhost", port="19530")
print("✅ 已连接到Milvus")

# 创建管理器
manager = CollectionManager(max_loaded=5)

# ===== 3. 场景1：创建多个租户Collection =====
print("\n" + "=" * 70)
print("场景1: 创建多个租户Collection")
print("=" * 70)

tenant_ids = ["tenant_001", "tenant_002", "tenant_003"]

for tenant_id in tenant_ids:
    collection = manager.get_collection(tenant_id, auto_create=True)
    print(f"✅ 租户 {tenant_id}: Collection已就绪")

# ===== 4. 场景2：插入数据 =====
print("\n" + "=" * 70)
print("场景2: 为租户插入数据")
print("=" * 70)

def generate_mock_data(tenant_id: str, count: int = 5):
    """生成模拟数据"""
    texts = [f"{tenant_id} 的文档 {i+1}" for i in range(count)]
    vectors = [np.random.rand(768).tolist() for _ in range(count)]
    sources = [f"doc_{i+1}.pdf" for i in range(count)]
    timestamps = [int(time.time()) for _ in range(count)]
    return texts, vectors, sources, timestamps

for tenant_id in tenant_ids:
    collection = manager.get_collection(tenant_id)
    texts, vectors, sources, timestamps = generate_mock_data(tenant_id)
    
    collection.insert([texts, vectors, sources, timestamps])
    collection.flush()
    
    print(f"✅ 租户 {tenant_id}: 已插入 {len(texts)} 条数据")

# ===== 5. 场景3：按需加载测试 =====
print("\n" + "=" * 70)
print("场景3: 按需加载测试（创建更多租户）")
print("=" * 70)

# 创建更多租户（超过max_loaded）
additional_tenants = [f"tenant_{i:03d}" for i in range(4, 10)]

for tenant_id in additional_tenants:
    collection = manager.get_collection(tenant_id, auto_create=True)
    print(f"✅ 租户 {tenant_id}: Collection已创建")

print(f"\n当前加载的Collection数: {len(manager.loaded_collections)}")
print(f"最大加载数: {manager.max_loaded}")

# ===== 6. 场景4：健康检查 =====
print("\n" + "=" * 70)
print("场景4: 健康检查")
print("=" * 70)

health = manager.health_check()
print(f"健康状态: {health['status']}")
print(f"总Collection数: {health['total_collections']}")
print(f"已加载Collection数: {health['loaded_collections']}")
print(f"内存使用率: {health['memory_usage_percent']:.1f}%")

if "message" in health:
    print(f"⚠️  警告: {health['message']}")

# ===== 7. 场景5：统计信息 =====
print("\n" + "=" * 70)
print("场景5: 统计信息")
print("=" * 70)

stats = manager.get_statistics()
print(f"总Collection数: {stats['total_collections']}")
print(f"已加载Collection数: {stats['loaded_collections']}")
print(f"\n租户详情:")

for tenant_stat in stats['tenant_stats']:
    print(f"  - 租户 {tenant_stat['tenant_id']}:")
    print(f"      记录数: {tenant_stat['num_entities']}")
    print(f"      最后访问: {tenant_stat['last_access']}")

# ===== 8. 场景6：检索测试 =====
print("\n" + "=" * 70)
print("场景6: 多租户检索测试")
print("=" * 70)

query_vector = np.random.rand(768).tolist()

for tenant_id in ["tenant_001", "tenant_002"]:
    collection = manager.get_collection(tenant_id)
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=3,
        output_fields=["text", "source"]
    )
    
    print(f"\n租户 {tenant_id} 的检索结果:")
    for i, hit in enumerate(results[0], 1):
        print(f"  {i}. {hit.entity.get('text')} (相似度: {hit.distance:.4f})")

# ===== 9. 场景7：清理测试 =====
print("\n" + "=" * 70)
print("场景7: 手动清理测试")
print("=" * 70)

# 释放特定租户
manager.release_collection("tenant_001")

print(f"清理后加载的Collection数: {len(manager.loaded_collections)}")

# ===== 10. 清理所有测试Collection =====
print("\n" + "=" * 70)
print("清理: 删除所有测试Collection")
print("=" * 70)

all_tenants = tenant_ids + additional_tenants
for tenant_id in all_tenants:
    collection_name = f"tenant_{tenant_id}"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🧹 已删除: {collection_name}")

print("\n" + "=" * 70)
print("🎉 Collection生命周期管理演示完成！")
print("=" * 70)
```

---

## 关键设计模式

### 1. 工厂模式

```python
def create_tenant_collection(self, tenant_id: str) -> Collection:
    """为租户创建标准化的Collection"""
    collection_name = f"tenant_{tenant_id}"
    collection = Collection(collection_name, self.schema_template)
    
    # 统一的索引配置
    collection.create_index(...)
    
    return collection
```

**优势：**
- 统一的Collection创建流程
- 标准化的Schema和索引配置
- 易于维护和扩展

### 2. 按需加载（Lazy Loading）

```python
def get_collection(self, tenant_id: str) -> Collection:
    """只在需要时才加载Collection"""
    # 1. 检查缓存
    if collection_name in self.loaded_collections:
        return self.loaded_collections[collection_name]
    
    # 2. 加载到内存
    collection.load()
    
    # 3. 缓存
    self.loaded_collections[collection_name] = collection
    
    return collection
```

**优势：**
- 节省内存
- 提高系统响应速度
- 支持大量租户

### 3. 自动清理（LRU策略）

```python
def _cleanup_inactive(self):
    """清理最久未访问的Collection"""
    sorted_collections = sorted(
        self.loaded_collections.items(),
        key=lambda x: x[1]["last_access"]
    )
    
    # 释放最旧的Collection
    for collection_name, info in sorted_collections[:to_release]:
        self.release_collection(info["tenant_id"])
```

**优势：**
- 自动内存管理
- 保持系统稳定
- 避免内存溢出

---

## 生产环境最佳实践

### 1. Collection数量限制

**重要建议：**
```python
# ✅ 推荐：<1000个Collection
# 适用场景：中小型多租户系统

# ⚠️  谨慎：1000-10000个Collection
# 需要：严格的内存管理和监控

# ❌ 不推荐：>10000个Collection
# 替代方案：使用Partition Key策略
```

**Partition Key替代方案：**
```python
# 对于大规模多租户（>1000租户），使用Partition Key
fields = [
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]

# 检索时自动按tenant_id分区
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    expr="tenant_id == 'tenant_001'",
    limit=10
)
```

### 2. 监控指标

```python
def get_monitoring_metrics(self) -> Dict:
    """获取监控指标"""
    return {
        "total_collections": len(utility.list_collections()),
        "loaded_collections": len(self.loaded_collections),
        "memory_usage_percent": self._calculate_memory_usage(),
        "avg_collection_size": self._calculate_avg_size(),
        "inactive_collections": self._count_inactive()
    }
```

**关键指标：**
- Collection总数
- 已加载Collection数
- 内存使用率
- 平均Collection大小
- 不活跃Collection数

### 3. 健康检查

```python
def health_check(self) -> Dict:
    """健康检查"""
    health = {
        "status": "healthy",
        "checks": {
            "connection": self._check_connection(),
            "memory": self._check_memory(),
            "collections": self._check_collections()
        }
    }
    
    # 判断整体状态
    if any(check == "unhealthy" for check in health["checks"].values()):
        health["status"] = "unhealthy"
    
    return health
```

---

## 性能优化建议

### 1. 连接池管理

```python
class ConnectionPool:
    """Milvus连接池"""
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = []
    
    def get_connection(self):
        # 实现连接池逻辑
        pass
```

### 2. 批量操作

```python
def batch_create_collections(self, tenant_ids: List[str]):
    """批量创建Collection"""
    for tenant_id in tenant_ids:
        self.create_tenant_collection(tenant_id)
        
        # 每创建10个Collection，暂停一下
        if len(tenant_ids) % 10 == 0:
            time.sleep(0.1)
```

### 3. 异步加载

```python
import asyncio

async def async_load_collection(self, tenant_id: str):
    """异步加载Collection"""
    collection = await asyncio.to_thread(
        self.get_collection, tenant_id
    )
    return collection
```

---

## 常见问题

### Q1: 为什么建议Collection数量<1000？

**A:** 根据Milvus官方建议和生产实践：
- 每个Collection会占用一定的管理资源
- Collection数量过多会影响元数据管理性能
- 建议<1000个Collection以保证最佳性能

### Q2: 如何支持>1000个租户？

**A:** 使用Partition Key策略：
- 单个Collection + Partition Key字段
- 可以支持百万级租户
- 性能更好，管理更简单

### Q3: 如何监控Collection的内存占用？

**A:** 
```python
# 查看Collection统计
collection.num_entities  # 记录数
collection.is_loaded  # 是否已加载

# 估算内存占用
memory_mb = collection.num_entities * vector_dim * 4 / 1024 / 1024
```

---

## 下一步

- **深度掌握**：[09_化骨绵掌](./09_化骨绵掌.md)
- **返回导航**：[00_概览](./00_概览.md)
