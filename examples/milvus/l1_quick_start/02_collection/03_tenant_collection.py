"""
Milvus 2.6 Collection生命周期管理 - 多租户系统
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
    CollectionManager生命周期管理器

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
        return CollectionSchema(fields=fields, description="Tenant document collection")
    
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

    def get_collection(self, tenant_id: str, auto_create: bool = True) -> Optional[Collection]:
        """
        获取租户的Collection (按需加载)
        
        Args:
            tenant_id: 租户ID
            auto_create: 如果Collection不存在，是否自动创建

        Returns:
            Collection对象或None
        """
        collection_name = f"tenant_{tenant_id}"

        # 1. 检查缓存
        if collection_name in self.loaded_collections:
            # 更新最后访问时间
            self.loaded_collections[collection_name]['last_access'] = time.time()
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
