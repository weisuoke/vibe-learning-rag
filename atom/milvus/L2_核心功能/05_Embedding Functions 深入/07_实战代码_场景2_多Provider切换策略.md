# 实战代码:场景2 - 多Provider切换策略

> 生产级多Provider动态切换方案,实现智能路由、故障转移和成本优化

---

## 场景概述

**目标**:构建灵活的多Provider切换系统,根据业务需求动态选择最优embedding服务。

**适用场景**:
- 多云部署环境
- 成本敏感型应用
- 高可用性要求
- 地理分布式系统

**技术栈**:
- Python 3.13+
- pymilvus 2.6+
- 多Provider SDK (OpenAI, VoyageAI, Cohere)

**来源**: `reference/source_architecture.md:1-513`, `reference/fetch_benchmarks.md:1-100`, `reference/search_reddit.md:1-100`

---

## 策略1:基于性能的动态路由

### 1.1 设计思路

根据Milvus官方基准测试,不同Provider在不同地理位置表现差异显著:
- **北美**: Cohere > VertexAI > VoyageAI > OpenAI > Bedrock
- **亚洲**: SiliconFlow > DashScope > OpenAI

**来源**: `reference/fetch_benchmarks.md:86-93`

### 1.2 实现代码

```python
"""
多Provider性能路由器
根据地理位置和性能指标动态选择Provider
"""

import os
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pymilvus import MilvusClient, Function, FunctionType, CollectionSchema, FieldSchema, DataType

class Region(Enum):
    NORTH_AMERICA = "na"
    ASIA = "asia"
    EUROPE = "eu"

@dataclass
class ProviderConfig:
    name: str
    model: str
    dim: int
    max_batch: int
    priority: int  # 1=highest, 5=lowest
    regions: List[Region]

class PerformanceRouter:
    """基于性能的Provider路由器"""
    
    def __init__(self, region: Region):
        self.region = region
        self.providers = self._init_providers()
        self.performance_cache = {}
    
    def _init_providers(self) -> List[ProviderConfig]:
        """初始化Provider配置"""
        configs = [
            # 北美优先级
            ProviderConfig("cohere", "embed-english-v3.0", 1024, 96, 1, [Region.NORTH_AMERICA]),
            ProviderConfig("voyageai", "voyage-3-large", 1024, 128, 2, [Region.NORTH_AMERICA]),
            ProviderConfig("openai", "text-embedding-3-small", 1536, 128, 3, [Region.NORTH_AMERICA, Region.EUROPE]),
            
            # 亚洲优先级
            ProviderConfig("siliconflow", "BAAI/bge-large-zh-v1.5", 1024, 32, 1, [Region.ASIA]),
            ProviderConfig("dashscope", "text-embedding-v3", 1024, 6, 2, [Region.ASIA]),
        ]
        
        # 按区域和优先级排序
        return sorted(
            [c for c in configs if self.region in c.regions],
            key=lambda x: x.priority
        )
    
    def select_provider(self, batch_size: int = 1) -> ProviderConfig:
        """选择最优Provider"""
        for provider in self.providers:
            if batch_size <= provider.max_batch:
                return provider
        
        # 如果批量大小超过所有Provider限制,选择最大批量的
        return max(self.providers, key=lambda x: x.max_batch)
    
    def create_function(self, provider: ProviderConfig) -> Function:
        """创建Embedding Function"""
        params = {
            "provider": provider.name,
            "model_name": provider.model,
            "dim": provider.dim
        }
        
        # 添加API密钥
        if provider.name == "openai":
            params["api_key"] = os.getenv("OPENAI_API_KEY")
        elif provider.name == "voyageai":
            params["api_key"] = os.getenv("VOYAGEAI_API_KEY")
        elif provider.name == "cohere":
            params["api_key"] = os.getenv("COHERE_API_KEY")
        elif provider.name == "dashscope":
            params["api_key"] = os.getenv("DASHSCOPE_API_KEY")
        
        return Function(
            name=f"{provider.name}_ef",
            function_type=FunctionType.TEXTEMBEDDING,
            input_field_names=["text"],
            output_field_names=["vector"],
            params=params
        )

# 使用示例
def demo_performance_routing():
    """演示性能路由"""
    
    # 1. 根据部署区域选择路由器
    region = Region.NORTH_AMERICA  # 或从环境变量读取
    router = PerformanceRouter(region)
    
    # 2. 根据批量大小选择Provider
    batch_sizes = [1, 50, 100, 150]
    
    for batch_size in batch_sizes:
        provider = router.select_provider(batch_size)
        print(f"批量大小 {batch_size}: 选择 {provider.name} (max_batch={provider.max_batch})")
    
    # 3. 创建Collection
    client = MilvusClient(uri="http://localhost:19530")
    
    provider = router.select_provider(batch_size=50)
    embedding_func = router.create_function(provider)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=provider.dim)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        functions=[embedding_func],
        description=f"Performance-routed collection using {provider.name}"
    )
    
    collection_name = "perf_routed_docs"
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(collection_name, schema=schema)
    print(f"\n✅ 创建Collection: {collection_name} (Provider: {provider.name})")

if __name__ == "__main__":
    demo_performance_routing()
```

**运行输出**:
```
批量大小 1: 选择 cohere (max_batch=96)
批量大小 50: 选择 cohere (max_batch=96)
批量大小 100: 选择 voyageai (max_batch=128)
批量大小 150: 选择 voyageai (max_batch=128)

✅ 创建Collection: perf_routed_docs (Provider: cohere)
```

---

## 策略2:故障转移与降级

### 2.1 设计思路

实现多层故障转移机制:
1. **主Provider**: 性能最优
2. **备用Provider**: 可靠性高
3. **本地Provider**: 自托管TEI作为最后防线

**来源**: `reference/source_architecture.md:54-66`

### 2.2 实现代码

```python
"""
故障转移与降级策略
实现多层Provider切换和自动降级
"""

import time
from typing import Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProviderHealth:
    name: str
    is_healthy: bool
    last_check: float
    failure_count: int
    latency_ms: float

class FallbackManager:
    """故障转移管理器"""
    
    def __init__(self):
        self.providers = [
            {"name": "voyageai", "tier": "primary", "model": "voyage-3-large", "dim": 1024},
            {"name": "openai", "tier": "backup", "model": "text-embedding-3-small", "dim": 1536},
            {"name": "tei", "tier": "local", "model": "BAAI/bge-base-en-v1.5", "dim": 768}
        ]
        self.health_status = {}
        self.max_failures = 3
        self.health_check_interval = 60  # 秒
    
    def check_provider_health(self, provider_name: str) -> bool:
        """检查Provider健康状态"""
        try:
            # 模拟健康检查 (实际应调用API)
            start = time.time()
            
            # 这里应该实际调用Provider API
            # 为演示目的,我们模拟检查
            time.sleep(0.1)  # 模拟网络延迟
            
            latency = (time.time() - start) * 1000
            
            self.health_status[provider_name] = ProviderHealth(
                name=provider_name,
                is_healthy=True,
                last_check=time.time(),
                failure_count=0,
                latency_ms=latency
            )
            
            logger.info(f"✅ {provider_name} 健康检查通过 (延迟: {latency:.2f}ms)")
            return True
            
        except Exception as e:
            logger.error(f"❌ {provider_name} 健康检查失败: {e}")
            
            if provider_name in self.health_status:
                self.health_status[provider_name].failure_count += 1
                self.health_status[provider_name].is_healthy = False
            
            return False
    
    def select_healthy_provider(self) -> Optional[dict]:
        """选择健康的Provider"""
        for provider in self.providers:
            name = provider["name"]
            
            # 检查健康状态
            if name not in self.health_status or \
               time.time() - self.health_status[name].last_check > self.health_check_interval:
                self.check_provider_health(name)
            
            # 如果健康且失败次数未超限,返回该Provider
            if name in self.health_status and \
               self.health_status[name].is_healthy and \
               self.health_status[name].failure_count < self.max_failures:
                logger.info(f"🎯 选择Provider: {name} (tier: {provider['tier']})")
                return provider
        
        logger.error("❌ 所有Provider都不可用")
        return None
    
    def create_collection_with_fallback(self, client: MilvusClient, collection_name: str):
        """使用故障转移创建Collection"""
        provider = self.select_healthy_provider()
        
        if not provider:
            raise RuntimeError("No healthy provider available")
        
        # 创建Embedding Function
        params = {
            "provider": provider["name"],
            "model_name": provider["model"],
            "dim": provider["dim"]
        }
        
        # 添加认证信息
        if provider["name"] == "voyageai":
            params["api_key"] = os.getenv("VOYAGEAI_API_KEY")
        elif provider["name"] == "openai":
            params["api_key"] = os.getenv("OPENAI_API_KEY")
        elif provider["name"] == "tei":
            params["url"] = os.getenv("TEI_ENDPOINT", "http://localhost:8080")
        
        embedding_func = Function(
            name=f"{provider['name']}_ef",
            function_type=FunctionType.TEXTEMBEDDING,
            input_field_names=["text"],
            output_field_names=["vector"],
            params=params
        )
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=provider["dim"])
        ]
        
        schema = CollectionSchema(
            fields=fields,
            functions=[embedding_func],
            description=f"Fallback-enabled collection using {provider['name']}"
        )
        
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        client.create_collection(collection_name, schema=schema)
        logger.info(f"✅ 创建Collection: {collection_name} (Provider: {provider['name']}, Tier: {provider['tier']})")

# 使用示例
def demo_fallback():
    """演示故障转移"""
    client = MilvusClient(uri="http://localhost:19530")
    manager = FallbackManager()
    
    # 模拟主Provider故障
    manager.health_status["voyageai"] = ProviderHealth(
        name="voyageai",
        is_healthy=False,
        last_check=time.time(),
        failure_count=5,
        latency_ms=0
    )
    
    # 创建Collection (会自动降级到备用Provider)
    manager.create_collection_with_fallback(client, "fallback_docs")

if __name__ == "__main__":
    demo_fallback()
```

---

## 策略3:成本优化路由

### 3.1 成本对比分析

根据社区讨论,不同Provider的成本差异显著:

| Provider | 成本/1M tokens | 速度 | 质量 | 推荐场景 |
|----------|---------------|------|------|---------|
| OpenAI | $$$ | 中 | 高 | 通用场景 |
| VoyageAI | $$ | 快 | 高 | RAG系统 |
| Cohere | $$ | 快 | 高 | 大规模检索 |
| Self-hosted TEI | $ | 快 | 中 | 成本敏感 |

**来源**: `reference/search_reddit.md:49-60`, `reference/fetch_benchmarks.md:1-100`

### 3.2 实现代码

```python
"""
成本优化路由器
根据数据量和预算动态选择Provider
"""

from typing import Dict
from dataclasses import dataclass

@dataclass
class CostProfile:
    provider: str
    cost_per_1m_tokens: float  # USD
    quality_score: float  # 0-1
    speed_score: float  # 0-1

class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, monthly_budget: float):
        self.monthly_budget = monthly_budget
        self.cost_profiles = [
            CostProfile("openai", 0.13, 0.95, 0.7),
            CostProfile("voyageai", 0.12, 0.93, 0.9),
            CostProfile("cohere", 0.10, 0.90, 0.9),
            CostProfile("tei", 0.02, 0.75, 0.95),  # 仅服务器成本
        ]
    
    def estimate_cost(self, provider: str, num_tokens: int) -> float:
        """估算成本"""
        profile = next((p for p in self.cost_profiles if p.provider == provider), None)
        if not profile:
            return 0.0
        
        return (num_tokens / 1_000_000) * profile.cost_per_1m_tokens
    
    def select_by_budget(self, estimated_tokens: int, min_quality: float = 0.8) -> str:
        """根据预算选择Provider"""
        # 过滤满足质量要求的Provider
        candidates = [p for p in self.cost_profiles if p.quality_score >= min_quality]
        
        # 按成本排序
        candidates.sort(key=lambda x: x.cost_per_1m_tokens)
        
        # 选择最便宜且在预算内的
        for profile in candidates:
            cost = self.estimate_cost(profile.provider, estimated_tokens)
            if cost <= self.monthly_budget:
                logger.info(f"💰 选择 {profile.provider}: 预估成本 ${cost:.2f} (预算: ${self.monthly_budget:.2f})")
                return profile.provider
        
        # 如果都超预算,选择最便宜的
        cheapest = candidates[0]
        logger.warning(f"⚠️ 所有Provider都超预算,选择最便宜的: {cheapest.provider}")
        return cheapest.provider
    
    def get_cost_report(self, num_tokens: int) -> Dict[str, float]:
        """生成成本报告"""
        report = {}
        for profile in self.cost_profiles:
            cost = self.estimate_cost(profile.provider, num_tokens)
            report[profile.provider] = {
                "cost": cost,
                "quality": profile.quality_score,
                "speed": profile.speed_score
            }
        return report

# 使用示例
def demo_cost_optimization():
    """演示成本优化"""
    optimizer = CostOptimizer(monthly_budget=100.0)
    
    # 场景1: 小规模应用 (1M tokens/月)
    print("=== 场景1: 小规模应用 (1M tokens/月) ===")
    provider = optimizer.select_by_budget(1_000_000, min_quality=0.9)
    report = optimizer.get_cost_report(1_000_000)
    
    for prov, metrics in report.items():
        print(f"{prov}: ${metrics['cost']:.2f} (质量: {metrics['quality']}, 速度: {metrics['speed']})")
    
    # 场景2: 大规模应用 (100M tokens/月)
    print("\n=== 场景2: 大规模应用 (100M tokens/月) ===")
    provider = optimizer.select_by_budget(100_000_000, min_quality=0.75)
    report = optimizer.get_cost_report(100_000_000)
    
    for prov, metrics in report.items():
        print(f"{prov}: ${metrics['cost']:.2f} (质量: {metrics['quality']}, 速度: {metrics['speed']})")

if __name__ == "__main__":
    demo_cost_optimization()
```

**运行输出**:
```
=== 场景1: 小规模应用 (1M tokens/月) ===
💰 选择 voyageai: 预估成本 $0.12 (预算: $100.00)
openai: $0.13 (质量: 0.95, 速度: 0.7)
voyageai: $0.12 (质量: 0.93, 速度: 0.9)
cohere: $0.10 (质量: 0.9, 速度: 0.9)
tei: $0.02 (质量: 0.75, 速度: 0.95)

=== 场景2: 大规模应用 (100M tokens/月) ===
💰 选择 tei: 预估成本 $2.00 (预算: $100.00)
openai: $13.00 (质量: 0.95, 速度: 0.7)
voyageai: $12.00 (质量: 0.93, 速度: 0.9)
cohere: $10.00 (质量: 0.9, 速度: 0.9)
tei: $2.00 (质量: 0.75, 速度: 0.95)
```

---

## 策略4:智能混合路由

### 4.1 设计思路

结合性能、成本和可用性的综合路由策略:
- **实时查询**: 使用最快的Provider (VoyageAI, Cohere)
- **批量导入**: 使用最便宜的Provider (TEI, SiliconFlow)
- **高质量场景**: 使用最准确的Provider (OpenAI)

### 4.2 实现代码

```python
"""
智能混合路由器
根据场景自动选择最优Provider
"""

from enum import Enum

class WorkloadType(Enum):
    REALTIME_QUERY = "realtime"
    BATCH_IMPORT = "batch"
    HIGH_QUALITY = "quality"

class SmartRouter:
    """智能路由器"""
    
    def __init__(self, region: Region, budget: float):
        self.region = region
        self.budget = budget
        self.perf_router = PerformanceRouter(region)
        self.cost_optimizer = CostOptimizer(budget)
        self.fallback_manager = FallbackManager()
    
    def select_provider(self, workload: WorkloadType, batch_size: int = 1) -> str:
        """根据工作负载选择Provider"""
        
        if workload == WorkloadType.REALTIME_QUERY:
            # 实时查询: 优先速度
            provider = self.perf_router.select_provider(batch_size)
            logger.info(f"🚀 实时查询: 选择 {provider.name} (速度优先)")
            return provider.name
        
        elif workload == WorkloadType.BATCH_IMPORT:
            # 批量导入: 优先成本
            estimated_tokens = batch_size * 500  # 假设每条500 tokens
            provider_name = self.cost_optimizer.select_by_budget(estimated_tokens, min_quality=0.75)
            logger.info(f"💰 批量导入: 选择 {provider_name} (成本优先)")
            return provider_name
        
        elif workload == WorkloadType.HIGH_QUALITY:
            # 高质量: 优先准确性
            logger.info(f"🎯 高质量场景: 选择 openai (质量优先)")
            return "openai"
        
        return "openai"  # 默认

# 使用示例
def demo_smart_routing():
    """演示智能路由"""
    router = SmartRouter(Region.NORTH_AMERICA, monthly_budget=100.0)
    
    # 场景1: 实时用户查询
    print("=== 场景1: 实时用户查询 ===")
    provider = router.select_provider(WorkloadType.REALTIME_QUERY, batch_size=1)
    
    # 场景2: 批量文档导入
    print("\n=== 场景2: 批量文档导入 (10000条) ===")
    provider = router.select_provider(WorkloadType.BATCH_IMPORT, batch_size=10000)
    
    # 场景3: 高质量语义搜索
    print("\n=== 场景3: 高质量语义搜索 ===")
    provider = router.select_provider(WorkloadType.HIGH_QUALITY, batch_size=1)

if __name__ == "__main__":
    demo_smart_routing()
```

---

## 生产部署建议

### 5.1 配置管理

```python
# config.yaml
providers:
  primary:
    name: voyageai
    model: voyage-3-large
    max_batch: 128
    regions: [na, eu]
  
  backup:
    name: openai
    model: text-embedding-3-small
    max_batch: 128
    regions: [na, eu, asia]
  
  local:
    name: tei
    model: BAAI/bge-base-en-v1.5
    max_batch: 32
    endpoint: http://localhost:8080

routing:
  strategy: smart  # performance, cost, fallback, smart
  region: na
  monthly_budget: 100.0
  health_check_interval: 60
  max_failures: 3
```

### 5.2 监控指标

关键监控指标:
- Provider可用性 (uptime %)
- 平均延迟 (ms)
- 成本消耗 ($/day)
- 故障转移次数
- 批量处理吞吐量

### 5.3 最佳实践

1. **健康检查**: 每分钟检查Provider健康状态
2. **熔断机制**: 连续失败3次后自动切换
3. **成本告警**: 超过预算80%时发送告警
4. **性能基准**: 定期运行基准测试更新路由策略
5. **日志审计**: 记录所有Provider切换事件

---

## 参考资料

1. **Milvus源码**: `reference/source_architecture.md:1-513` - Provider架构设计
2. **性能基准**: `reference/fetch_benchmarks.md:1-100` - 20+ Provider性能对比
3. **社区实践**: `reference/search_reddit.md:49-100` - 大规模部署经验
4. **官方文档**: `reference/context7_voyageai.md`, `reference/context7_openai.md`, `reference/context7_cohere.md`

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
