# 核心概念2：Token优化技术

> **降低50-80%成本的核心技术**

---

## 为什么需要Token优化？

### 成本压力

```python
# 场景：企业RAG系统
queries_per_day = 1000
avg_tokens_per_query = 10000  # 未优化
cost_per_day = (1000 * 10000 / 1_000_000) * 10 = $100
cost_per_month = $3000

# 优化后
avg_tokens_per_query = 2500  # 4x优化
cost_per_month = $750  # 节省75%
```

---

## 核心优化技术

### 技术1：智能分块（Semantic Chunking）

**原理**：按语义边界分块，而非固定大小

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunking(text: str, chunk_size: int = 500) -> list[str]:
    """语义分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    return splitter.split_text(text)

# 效果：减少30%无效分块
```

### 技术2：语义缓存（Semantic Caching）

**原理**：缓存相似查询的结果

```python
from functools import lru_cache
import hashlib

class SemanticCache:
    """语义缓存"""
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}
        self.threshold = similarity_threshold

    def get(self, query: str, embedding_func) -> str | None:
        """获取缓存"""
        query_embedding = embedding_func(query)

        for cached_query, (cached_embedding, result) in self.cache.items():
            similarity = cosine_similarity(query_embedding, cached_embedding)
            if similarity > self.threshold:
                return result

        return None

    def set(self, query: str, embedding, result: str):
        """设置缓存"""
        self.cache[query] = (embedding, result)

# 效果：命中率80%时节省80%成本
```

### 技术3：上下文蒸馏（Context Distillation）

**原理**：使用小模型提取关键信息

```python
def context_distillation(documents: list[str], query: str) -> str:
    """上下文蒸馏"""
    # 使用小模型（如GPT-3.5）提取关键信息
    prompt = f"""从以下文档中提取与查询相关的关键信息：

查询：{query}

文档：
{documents}

关键信息（保留核心事实和数据）："""

    # 使用便宜的模型
    summary = cheap_llm.generate(prompt, max_tokens=500)
    return summary

# 效果：Token减少70%，成本降低90%
```

### 技术4：选择性上下文（Selective Context）

**原理**：只保留高相关性内容

```python
def selective_context(
    query: str,
    documents: list[str],
    threshold: float = 0.7
) -> list[str]:
    """选择性上下文"""
    # 计算相关性分数
    scores = [calculate_relevance(query, doc) for doc in documents]

    # 只保留高分文档
    selected = [
        doc for doc, score in zip(documents, scores)
        if score > threshold
    ]

    return selected

# 效果：减少40%无关内容
```

---

## 组合优化策略

### 策略1：分层优化

```python
class LayeredOptimization:
    """分层Token优化"""
    def __init__(self):
        self.cache = SemanticCache()

    def optimize(self, query: str, documents: list[str]) -> str:
        """分层优化流程"""
        # 1. 检查缓存
        cached = self.cache.get(query)
        if cached:
            return cached

        # 2. 选择性上下文
        selected = selective_context(query, documents, threshold=0.7)

        # 3. 语义分块
        chunks = []
        for doc in selected:
            chunks.extend(semantic_chunking(doc, chunk_size=500))

        # 4. 上下文蒸馏
        distilled = context_distillation(chunks, query)

        # 5. 缓存结果
        self.cache.set(query, distilled)

        return distilled

# 效果：综合节省70-80%Token
```

### 策略2：动态优化

```python
def dynamic_optimization(
    query: str,
    documents: list[str],
    budget: int = 4000  # Token预算
) -> str:
    """动态Token优化"""
    current_tokens = sum(count_tokens(doc) for doc in documents)

    if current_tokens <= budget:
        # 预算充足，不优化
        return "\n\n".join(documents)

    # 计算压缩比
    compression_ratio = current_tokens / budget

    if compression_ratio < 2:
        # 轻度压缩：选择性上下文
        return selective_context(query, documents)
    elif compression_ratio < 4:
        # 中度压缩：语义分块 + 选择
        chunks = semantic_chunking("\n\n".join(documents))
        return selective_context(query, chunks)
    else:
        # 重度压缩：上下文蒸馏
        return context_distillation(documents, query)

# 效果：自适应优化，平衡成本和质量
```

---

## 实际效果对比

### 优化前后对比

| 技术 | Token减少 | 成本降低 | 质量影响 | 延迟影响 |
|------|----------|---------|---------|---------|
| **智能分块** | 30% | 30% | +5% | 0 |
| **语义缓存** | 80%（命中时） | 80% | 0 | -50% |
| **上下文蒸馏** | 70% | 90% | -5% | +20% |
| **选择性上下文** | 40% | 40% | +10% | 0 |
| **组合策略** | 75% | 80% | +5% | +10% |

### 实际案例

```python
# 案例：企业文档问答系统

# 原始配置
original_config = {
    "documents": 10,
    "tokens_per_doc": 1000,
    "total_tokens": 10000,
    "cost_per_query": 0.10,
    "queries_per_day": 1000,
    "monthly_cost": 3000
}

# 优化后配置
optimized_config = {
    "documents": 5,  # 选择性上下文
    "tokens_per_doc": 500,  # 语义分块
    "total_tokens": 2500,
    "cost_per_query": 0.025,
    "queries_per_day": 1000,
    "monthly_cost": 750,  # 节省75%
    "cache_hit_rate": 0.6,  # 60%命中率
    "effective_monthly_cost": 300  # 考虑缓存后
}

# 最终节省：90%
```

---

## 监控与调优

### 关键指标

```python
class TokenOptimizationMetrics:
    """Token优化指标"""
    def __init__(self):
        self.metrics = {
            "original_tokens": [],
            "optimized_tokens": [],
            "compression_ratio": [],
            "quality_score": [],
            "cache_hit_rate": []
        }

    def log(self, original: int, optimized: int, quality: float, cache_hit: bool):
        """记录指标"""
        self.metrics["original_tokens"].append(original)
        self.metrics["optimized_tokens"].append(optimized)
        self.metrics["compression_ratio"].append(original / optimized)
        self.metrics["quality_score"].append(quality)
        self.metrics["cache_hit_rate"].append(1 if cache_hit else 0)

    def report(self):
        """生成报告"""
        return {
            "avg_compression": sum(self.metrics["compression_ratio"]) / len(self.metrics["compression_ratio"]),
            "avg_quality": sum(self.metrics["quality_score"]) / len(self.metrics["quality_score"]),
            "cache_hit_rate": sum(self.metrics["cache_hit_rate"]) / len(self.metrics["cache_hit_rate"]),
            "token_savings": 1 - sum(self.metrics["optimized_tokens"]) / sum(self.metrics["original_tokens"])
        }
```

---

## 最佳实践

### 实践1：渐进式优化

```python
# 从保守到激进
optimization_levels = {
    "conservative": {
        "selective_context": True,
        "semantic_chunking": False,
        "distillation": False,
        "expected_savings": "30%"
    },
    "moderate": {
        "selective_context": True,
        "semantic_chunking": True,
        "distillation": False,
        "expected_savings": "50%"
    },
    "aggressive": {
        "selective_context": True,
        "semantic_chunking": True,
        "distillation": True,
        "expected_savings": "75%"
    }
}
```

### 实践2：A/B测试

```python
def ab_test_optimization():
    """A/B测试优化策略"""
    # 对照组：无优化
    control_group = run_queries(optimization=None)

    # 实验组：优化
    test_group = run_queries(optimization="moderate")

    # 对比结果
    results = {
        "cost_reduction": calculate_cost_reduction(control_group, test_group),
        "quality_change": calculate_quality_change(control_group, test_group),
        "latency_change": calculate_latency_change(control_group, test_group)
    }

    return results
```

---

## 总结

### 核心要点

1. **智能分块**：按语义边界分块，减少30%无效内容
2. **语义缓存**：命中率80%时节省80%成本
3. **上下文蒸馏**：使用小模型提取关键信息，降低90%成本
4. **选择性上下文**：只保留高相关性内容，减少40%Token
5. **组合策略**：综合节省70-80%Token

### 记忆口诀

**"分缓蒸选，层层优化"**

### 下一步

理解了Token优化技术后，接下来学习：
- **上下文压缩**：LLMLingua等高级技术
- **文档排序**：解决Lost in the Middle
- **动态窗口**：自适应调整

---

**记住**：Token优化不是一次性的，而是持续监控和调整的过程！
