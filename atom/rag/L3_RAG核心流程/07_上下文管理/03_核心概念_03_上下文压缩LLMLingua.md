# 核心概念3：上下文压缩LLMLingua

> **20x压缩，性能提升21.4%的黑科技**

---

## 什么是LLMLingua？

**LLMLingua** 是Microsoft Research在2023-2024年提出的上下文压缩技术，通过智能删除冗余token，实现高压缩比的同时保持甚至提升性能。

**核心论文**：
- LLMLingua (2023): 20x压缩，RAG场景提升21.4%
- LongLLMLingua (2024): 解决Lost in the Middle，4x压缩提升17.1%
- LLMLingua-2 (2024): 数据蒸馏方法

---

## 核心原理

### 粗到细压缩（Coarse-to-Fine）

```
原始文本
  ↓
[粗粒度压缩] 句子级别
  ↓ 保留重要句子
[细粒度压缩] 词级别
  ↓ 删除冗余词
压缩文本
```

### 重要性评分

```python
# 使用小型LLM评估每个token的重要性
def calculate_importance(token, context):
    """计算token重要性"""
    # 1. 困惑度（Perplexity）
    perplexity = model.perplexity(token, context)

    # 2. 自信息（Self-Information）
    self_info = -log(model.prob(token | context))

    # 3. 上下文依赖
    dependency = model.dependency_score(token, context)

    # 综合评分
    importance = (perplexity + self_info + dependency) / 3
    return importance
```

---

## LLMLingua系列对比

### 版本演进

| 版本 | 发布时间 | 核心创新 | 最佳压缩比 | 性能提升 |
|------|---------|---------|-----------|---------|
| **LLMLingua** | 2023.10 | 粗到细压缩 | 20x | +21.4% |
| **LongLLMLingua** | 2024.01 | 解决Lost in Middle | 4x | +17.1% |
| **LLMLingua-2** | 2024.03 | 数据蒸馏 | 10x | +18.5% |

### LLMLingua（基础版）

**特点**：
- 粗到细两阶段压缩
- 基于困惑度的重要性评分
- 适用于通用场景

**实验数据**：
```python
# 压缩比 vs 性能
compression_ratios = [2, 4, 10, 20]
performance = [1.052, 1.089, 1.123, 1.214]

# 结论：压缩比越高，性能越好（反直觉！）
```

### LongLLMLingua（长上下文优化）

**特点**：
- 针对Lost in the Middle问题
- 问题感知压缩
- 保留关键信息在首尾

**实验数据**：
```python
# 4x压缩效果
baseline_recall = 0.65  # 无压缩
compressed_recall = 0.82  # 4x压缩

improvement = (0.82 - 0.65) / 0.65 = 26%
```

### LLMLingua-2（数据蒸馏）

**特点**：
- 使用GPT-4生成训练数据
- 训练专门的压缩模型
- 更快的压缩速度

**性能对比**：
```python
# 压缩速度（tokens/秒）
llmlingua_speed = 100
llmlingua2_speed = 500  # 5x faster

# 质量保持
llmlingua_quality = 0.95
llmlingua2_quality = 0.96  # 略有提升
```

---

## 实战应用

### 基础使用

```python
from llmlingua import PromptCompressor

# 初始化压缩器
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    device="cuda"
)

# 压缩上下文
original_prompt = """
RAG（Retrieval-Augmented Generation）是一种非常强大的技术，
它结合了检索和生成两个部分。首先，它会从知识库中检索相关的文档，
然后将这些文档作为上下文传递给大语言模型，最后生成答案。
这种方法非常有效，因为它可以利用外部知识，而不仅仅依赖模型的内部知识。
"""

compressed = compressor.compress_prompt(
    original_prompt,
    rate=0.5,  # 压缩到50%
    force_tokens=['\n', '?', '!', '.']  # 保留标点
)

print(f"原始: {len(original_prompt)} 字符")
print(f"压缩后: {len(compressed['compressed_prompt'])} 字符")
print(f"压缩比: {compressed['ratio']}")
```

### 集成LangChain

```python
from langchain.prompts import PromptTemplate
from llmlingua import PromptCompressor

class LLMLinguaCompressor:
    """LangChain集成"""
    def __init__(self, compression_rate: float = 0.5):
        self.compressor = PromptCompressor()
        self.rate = compression_rate

    def compress_documents(self, documents: list[str]) -> str:
        """压缩文档列表"""
        # 合并文档
        combined = "\n\n".join(documents)

        # 压缩
        result = self.compressor.compress_prompt(
            combined,
            rate=self.rate
        )

        return result['compressed_prompt']

# 使用
compressor = LLMLinguaCompressor(compression_rate=0.25)  # 4x压缩
compressed_context = compressor.compress_documents(retrieved_docs)
```

### RAG完整流程

```python
def rag_with_compression(query: str, documents: list[str]) -> str:
    """带压缩的RAG流程"""
    # 1. 压缩文档
    compressor = PromptCompressor()
    compressed_docs = []

    for doc in documents:
        result = compressor.compress_prompt(
            doc,
            rate=0.25,  # 4x压缩
            question=query  # 问题感知压缩
        )
        compressed_docs.append(result['compressed_prompt'])

    # 2. 构建prompt
    context = "\n\n".join(compressed_docs)
    prompt = f"""基于以下上下文回答问题。

上下文：
{context}

问题：{query}

答案："""

    # 3. 调用LLM
    response = llm.generate(prompt)
    return response
```

---

## 压缩策略

### 策略1：固定压缩比

```python
def fixed_ratio_compression(text: str, ratio: float = 0.25) -> str:
    """固定压缩比"""
    result = compressor.compress_prompt(text, rate=ratio)
    return result['compressed_prompt']

# 适用场景：预算固定，需要可预测的成本
```

### 策略2：目标Token数

```python
def target_tokens_compression(text: str, target_tokens: int) -> str:
    """压缩到目标Token数"""
    current_tokens = count_tokens(text)
    ratio = target_tokens / current_tokens

    result = compressor.compress_prompt(text, rate=ratio)
    return result['compressed_prompt']

# 适用场景：Context Window有限，需要精确控制
```

### 策略3：问题感知压缩

```python
def question_aware_compression(
    text: str,
    question: str,
    ratio: float = 0.25
) -> str:
    """问题感知压缩"""
    result = compressor.compress_prompt(
        text,
        rate=ratio,
        question=question  # 保留与问题相关的内容
    )
    return result['compressed_prompt']

# 适用场景：RAG系统，需要保留相关信息
```

### 策略4：分层压缩

```python
def layered_compression(documents: list[str], query: str) -> str:
    """分层压缩策略"""
    compressed = []

    for i, doc in enumerate(documents):
        # 根据相关性分数决定压缩比
        relevance = calculate_relevance(query, doc)

        if relevance > 0.9:
            ratio = 0.7  # 高相关：轻度压缩
        elif relevance > 0.7:
            ratio = 0.5  # 中相关：中度压缩
        else:
            ratio = 0.25  # 低相关：重度压缩

        result = compressor.compress_prompt(doc, rate=ratio)
        compressed.append(result['compressed_prompt'])

    return "\n\n".join(compressed)

# 适用场景：差异化处理，平衡质量和成本
```

---

## 性能优化

### 优化1：批量压缩

```python
def batch_compression(documents: list[str], batch_size: int = 10) -> list[str]:
    """批量压缩提升效率"""
    compressed = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # 批量处理
        results = compressor.compress_prompt_batch(batch, rate=0.25)
        compressed.extend([r['compressed_prompt'] for r in results])

    return compressed
```

### 优化2：缓存压缩结果

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_compression(text: str, ratio: float) -> str:
    """缓存压缩结果"""
    result = compressor.compress_prompt(text, rate=ratio)
    return result['compressed_prompt']

# 对于重复文档，避免重复压缩
```

### 优化3：异步压缩

```python
import asyncio

async def async_compression(documents: list[str]) -> list[str]:
    """异步压缩"""
    tasks = [
        asyncio.to_thread(compressor.compress_prompt, doc, rate=0.25)
        for doc in documents
    ]
    results = await asyncio.gather(*tasks)
    return [r['compressed_prompt'] for r in results]

# 并行处理多个文档
```

---

## 实验数据

### 压缩比 vs 性能

```python
# 实验设置
dataset = "NaturalQuestions"
baseline_accuracy = 0.65

# 不同压缩比的效果
results = {
    "no_compression": {"accuracy": 0.65, "cost": 1.0},
    "2x_compression": {"accuracy": 0.68, "cost": 0.5},
    "4x_compression": {"accuracy": 0.76, "cost": 0.25},
    "10x_compression": {"accuracy": 0.73, "cost": 0.1},
    "20x_compression": {"accuracy": 0.79, "cost": 0.05}
}

# 结论：4x-10x是最佳平衡点
```

### RAG场景对比

```python
# 场景：企业文档问答
documents_per_query = 10
tokens_per_document = 1000

# 无压缩
baseline_tokens = 10 * 1000 = 10000
baseline_cost = 0.10
baseline_accuracy = 0.75

# 4x压缩
compressed_tokens = 10 * 250 = 2500
compressed_cost = 0.025
compressed_accuracy = 0.88

# 提升
cost_reduction = (0.10 - 0.025) / 0.10 = 75%
accuracy_improvement = (0.88 - 0.75) / 0.75 = 17.3%
```

---

## 常见问题

### Q1: 为什么压缩后性能反而提升？

**A**: 三个原因：
1. **冗余去除**：自然语言有大量冗余，去除后更清晰
2. **噪音过滤**：压缩过程过滤了无关信息
3. **注意力集中**：更短的上下文让LLM更专注

### Q2: 什么时候不应该使用压缩？

**A**: 以下场景慎用：
- 上下文本身很短（<1K tokens）
- 需要保留所有细节（法律文档）
- 实时性要求极高（压缩有延迟）

### Q3: 如何选择压缩比？

**A**: 根据场景：
- **通用RAG**: 4x（最佳平衡）
- **成本敏感**: 10x-20x
- **质量优先**: 2x
- **实验验证**: A/B测试

### Q4: LLMLingua vs 简单截断？

**A**: 对比：
```python
# 简单截断
truncated = text[:target_length]
# 问题：可能截断关键信息

# LLMLingua
compressed = compressor.compress_prompt(text, rate=0.25)
# 优势：智能保留关键信息
```

---

## 最佳实践

### 实践1: 渐进式压缩

```python
# 从保守到激进
compression_levels = {
    "conservative": 0.7,  # 30%压缩
    "moderate": 0.5,      # 50%压缩
    "aggressive": 0.25    # 75%压缩
}

# 根据实际效果调整
```

### 实践2: 监控质量

```python
def compress_with_monitoring(text: str, ratio: float) -> dict:
    """带监控的压缩"""
    result = compressor.compress_prompt(text, rate=ratio)

    # 监控指标
    metrics = {
        "original_tokens": count_tokens(text),
        "compressed_tokens": count_tokens(result['compressed_prompt']),
        "actual_ratio": result['ratio'],
        "compression_time": result['time']
    }

    return {
        "compressed": result['compressed_prompt'],
        "metrics": metrics
    }
```

### 实践3: A/B测试

```python
def ab_test_compression():
    """A/B测试压缩效果"""
    # 对照组：无压缩
    control = run_rag(compression=None)

    # 实验组：4x压缩
    test = run_rag(compression=0.25)

    # 对比
    print(f"成本降低: {(control.cost - test.cost) / control.cost * 100}%")
    print(f"质量变化: {(test.accuracy - control.accuracy) / control.accuracy * 100}%")
```

---

## 2026年趋势

### 趋势1: 更智能的压缩

```
2024: 基于困惑度
2025: 问题感知压缩
2026: 多模态压缩（文本+图片）
```

### 趋势2: 实时压缩

```
当前: 离线压缩（100 tokens/秒）
未来: 实时压缩（1000+ tokens/秒）
```

### 趋势3: 自适应压缩

```
当前: 固定压缩比
未来: 根据查询复杂度自动调整
```

---

## 总结

### 核心要点

1. **原理**：粗到细压缩 + 重要性评分
2. **效果**：20x压缩，性能提升21.4%
3. **最佳实践**：4x压缩是RAG最佳平衡点
4. **集成**：LangChain/LlamaIndex无缝集成

### 记忆口诀

**"粗细两阶段，智能去冗余，4x最平衡"**

### 下一步

理解了LLMLingua后，接下来学习：
- **Lost in the Middle问题**：位置偏差分析
- **文档排序策略**：首尾放置
- **ReRank重排序**：二次精排

---

**记住**：LLMLingua不是简单的截断，而是智能的信息提炼！
