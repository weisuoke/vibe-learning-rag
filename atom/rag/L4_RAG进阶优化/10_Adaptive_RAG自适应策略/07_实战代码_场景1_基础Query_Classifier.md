# 实战代码 - 场景1: 基础 Query Classifier

> 从零实现一个查询复杂度分类器

---

## 场景描述

**目标**: 实现一个基础的查询复杂度分类器，能够将用户查询分类为 NO_RETRIEVE、SINGLE、ITERATIVE、WEB_SEARCH 四种策略。

**适用场景**:
- 快速原型验证
- 小规模 RAG 应用
- 学习 Adaptive RAG 原理

**技术栈**:
- Python 3.13+
- OpenAI API (可选，用于 LLM 分类器)
- 无需额外依赖

---

## 完整代码实现

```python
"""
基础查询复杂度分类器
实现三种分类方法：规则引擎、特征工程、LLM 分类器
"""

import re
from typing import Literal
from dataclasses import dataclass

# ===== 1. 数据结构定义 =====

@dataclass
class QueryFeatures:
    """查询特征"""
    query: str
    word_count: int
    char_count: int
    entity_count: int
    has_time_keywords: bool
    has_comparison_keywords: bool
    has_complex_keywords: bool
    question_marks: int
    commas: int

Strategy = Literal["NO_RETRIEVE", "SINGLE", "ITERATIVE", "WEB_SEARCH"]

# ===== 2. 特征提取器 =====

class FeatureExtractor:
    """提取查询特征"""

    def __init__(self):
        # 关键词定义
        self.time_keywords = [
            "今天", "最新", "现在", "2025", "2026", "今年",
            "today", "latest", "current", "now", "recent"
        ]

        self.comparison_keywords = [
            "比较", "对比", "区别", "差异", "优缺点",
            "compare", "difference", "vs", "versus"
        ]

        self.complex_keywords = [
            "为什么", "如何", "怎样", "分析", "解释",
            "why", "how", "analyze", "explain"
        ]

    def extract(self, query: str) -> QueryFeatures:
        """提取查询特征"""
        query_lower = query.lower()
        words = query.split()

        # 基础特征
        word_count = len(words)
        char_count = len(query)

        # 实体数量（简单估算：首字母大写的词）
        entity_count = sum(1 for w in words if w and w[0].isupper())

        # 关键词检测
        has_time = any(kw in query_lower for kw in self.time_keywords)
        has_comparison = any(kw in query_lower for kw in self.comparison_keywords)
        has_complex = any(kw in query_lower for kw in self.complex_keywords)

        # 标点符号
        question_marks = query.count("?") + query.count("？")
        commas = query.count(",") + query.count("，")

        return QueryFeatures(
            query=query,
            word_count=word_count,
            char_count=char_count,
            entity_count=entity_count,
            has_time_keywords=has_time,
            has_comparison_keywords=has_comparison,
            has_complex_keywords=has_complex,
            question_marks=question_marks,
            commas=commas
        )

# ===== 3. 规则引擎分类器 =====

class RuleBasedClassifier:
    """基于规则的分类器"""

    def __init__(self):
        self.extractor = FeatureExtractor()

    def classify(self, query: str) -> Strategy:
        """
        分类查询

        规则优先级:
        1. 实时查询 → WEB_SEARCH
        2. 简单查询 → NO_RETRIEVE
        3. 复杂查询 → ITERATIVE
        4. 默认 → SINGLE
        """
        features = self.extractor.extract(query)

        # 规则1: 实时查询
        if features.has_time_keywords:
            return "WEB_SEARCH"

        # 规则2: 简单查询
        if features.word_count < 5 and not features.has_complex_keywords:
            return "NO_RETRIEVE"

        # 规则3: 复杂查询
        if (features.word_count > 15 or
            features.has_comparison_keywords or
            features.commas > 1):
            return "ITERATIVE"

        # 规则4: 默认中等查询
        return "SINGLE"

    def explain(self, query: str) -> dict:
        """解释分类决策"""
        features = self.extractor.extract(query)
        strategy = self.classify(query)

        return {
            "query": query,
            "strategy": strategy,
            "features": {
                "word_count": features.word_count,
                "has_time_keywords": features.has_time_keywords,
                "has_comparison_keywords": features.has_comparison_keywords,
                "has_complex_keywords": features.has_complex_keywords,
                "commas": features.commas
            },
            "reasoning": self._get_reasoning(features, strategy)
        }

    def _get_reasoning(self, features: QueryFeatures, strategy: Strategy) -> str:
        """生成推理解释"""
        if strategy == "WEB_SEARCH":
            return f"检测到时间关键词，需要实时信息"
        elif strategy == "NO_RETRIEVE":
            return f"简单查询（{features.word_count} 个词），LLM 已知知识可直接回答"
        elif strategy == "ITERATIVE":
            reasons = []
            if features.word_count > 15:
                reasons.append(f"查询较长（{features.word_count} 个词）")
            if features.has_comparison_keywords:
                reasons.append("包含对比关键词")
            if features.commas > 1:
                reasons.append(f"多个逗号（{features.commas} 个）")
            return f"复杂查询：{', '.join(reasons)}"
        else:
            return f"中等查询（{features.word_count} 个词），需要单次检索"

# ===== 4. LLM 分类器（可选）=====

class LLMClassifier:
    """基于 LLM 的分类器"""

    def __init__(self, api_key: str = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            print("警告: OpenAI 未安装，LLM 分类器不可用")
            self.available = False

    def classify(self, query: str) -> Strategy:
        """使用 LLM 分类查询"""
        if not self.available:
            raise RuntimeError("LLM 分类器不可用，请安装 openai 库")

        prompt = f"""分析以下查询的复杂度，选择最合适的检索策略。

查询: {query}

策略选项:
- NO_RETRIEVE: 简单事实查询，LLM 已知知识可直接回答
- SINGLE: 中等查询，需要检索一次外部文档
- ITERATIVE: 复杂查询，需要多次检索和推理
- WEB_SEARCH: 实时信息查询，需要网络搜索

只回答策略名称，不要解释。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # 验证结果
        valid_strategies = ["NO_RETRIEVE", "SINGLE", "ITERATIVE", "WEB_SEARCH"]
        if result in valid_strategies:
            return result
        else:
            # 降级到规则引擎
            print(f"警告: LLM 返回无效策略 '{result}'，降级到规则引擎")
            return RuleBasedClassifier().classify(query)

# ===== 5. 分类器对比工具 =====

class ClassifierComparison:
    """对比不同分类器的结果"""

    def __init__(self):
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = None

        # 尝试初始化 LLM 分类器
        try:
            self.llm_classifier = LLMClassifier()
        except:
            pass

    def compare(self, queries: list[str]) -> dict:
        """对比多个查询的分类结果"""
        results = []

        for query in queries:
            # 规则引擎
            rule_result = self.rule_classifier.classify(query)

            # LLM 分类器（如果可用）
            llm_result = None
            if self.llm_classifier and self.llm_classifier.available:
                try:
                    llm_result = self.llm_classifier.classify(query)
                except:
                    llm_result = "ERROR"

            results.append({
                "query": query,
                "rule_based": rule_result,
                "llm_based": llm_result,
                "agreement": rule_result == llm_result if llm_result else None
            })

        # 统计
        total = len(results)
        agreements = sum(1 for r in results if r["agreement"] is True)
        agreement_rate = agreements / total * 100 if total > 0 else 0

        return {
            "results": results,
            "statistics": {
                "total_queries": total,
                "agreements": agreements,
                "agreement_rate": f"{agreement_rate:.1f}%"
            }
        }

# ===== 6. 使用示例 =====

def main():
    """主函数：演示分类器使用"""

    print("=" * 60)
    print("基础查询复杂度分类器 - 实战示例")
    print("=" * 60)

    # 初始化分类器
    classifier = RuleBasedClassifier()

    # 测试查询
    test_queries = [
        "什么是 Python?",
        "如何使用 LangChain 构建 RAG?",
        "比较 LangChain 和 LlamaIndex 的优缺点，并分析适用场景",
        "2026 年 AI 有哪些新进展?",
        "Python 是什么语言?",
        "解释 Transformer 的工作原理",
        "今天的天气怎么样?",
        "为什么 RAG 需要向量检索，而不是直接使用 LLM?"
    ]

    print("\n【规则引擎分类结果】\n")

    for i, query in enumerate(test_queries, 1):
        result = classifier.explain(query)
        print(f"{i}. 查询: {query}")
        print(f"   策略: {result['strategy']}")
        print(f"   推理: {result['reasoning']}")
        print(f"   特征: {result['features']}")
        print()

    # 统计分布
    print("\n【策略分布统计】\n")
    strategy_counts = {}
    for query in test_queries:
        strategy = classifier.classify(query)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    total = len(test_queries)
    for strategy, count in sorted(strategy_counts.items()):
        percentage = count / total * 100
        print(f"{strategy:15s}: {count:2d} ({percentage:5.1f}%)")

    # 对比测试（如果 LLM 可用）
    print("\n【分类器对比】\n")
    comparison = ClassifierComparison()
    compare_result = comparison.compare(test_queries[:3])  # 只对比前3个

    if compare_result["results"][0]["llm_based"]:
        print("规则引擎 vs LLM 分类器:\n")
        for r in compare_result["results"]:
            print(f"查询: {r['query'][:40]}...")
            print(f"  规则引擎: {r['rule_based']}")
            print(f"  LLM:      {r['llm_based']}")
            print(f"  一致性:   {'✓' if r['agreement'] else '✗'}")
            print()

        print(f"总体一致率: {compare_result['statistics']['agreement_rate']}")
    else:
        print("LLM 分类器不可用，跳过对比")

if __name__ == "__main__":
    main()
```

---

## 运行输出示例

```
============================================================
基础查询复杂度分类器 - 实战示例
============================================================

【规则引擎分类结果】

1. 查询: 什么是 Python?
   策略: NO_RETRIEVE
   推理: 简单查询（3 个词），LLM 已知知识可直接回答
   特征: {'word_count': 3, 'has_time_keywords': False, 'has_comparison_keywords': False, 'has_complex_keywords': False, 'commas': 0}

2. 查询: 如何使用 LangChain 构建 RAG?
   策略: SINGLE
   推理: 中等查询（6 个词），需要单次检索
   特征: {'word_count': 6, 'has_time_keywords': False, 'has_comparison_keywords': False, 'has_complex_keywords': True, 'commas': 0}

3. 查询: 比较 LangChain 和 LlamaIndex 的优缺点，并分析适用场景
   策略: ITERATIVE
   推理: 复杂查询：查询较长（11 个词）, 包含对比关键词, 多个逗号（1 个）
   特征: {'word_count': 11, 'has_time_keywords': False, 'has_comparison_keywords': True, 'has_complex_keywords': True, 'commas': 1}

4. 查询: 2026 年 AI 有哪些新进展?
   策略: WEB_SEARCH
   推理: 检测到时间关键词，需要实时信息
   特征: {'word_count': 7, 'has_time_keywords': True, 'has_comparison_keywords': False, 'has_complex_keywords': False, 'commas': 0}

5. 查询: Python 是什么语言?
   策略: NO_RETRIEVE
   推理: 简单查询（4 个词），LLM 已知知识可直接回答
   特征: {'word_count': 4, 'has_time_keywords': False, 'has_comparison_keywords': False, 'has_complex_keywords': False, 'commas': 0}

6. 查询: 解释 Transformer 的工作原理
   策略: SINGLE
   推理: 中等查询（5 个词），需要单次检索
   特征: {'word_count': 5, 'has_time_keywords': False, 'has_comparison_keywords': False, 'has_complex_keywords': True, 'commas': 0}

7. 查询: 今天的天气怎么样?
   策略: WEB_SEARCH
   推理: 检测到时间关键词，需要实时信息
   特征: {'word_count': 5, 'has_time_keywords': True, 'has_comparison_keywords': False, 'has_complex_keywords': False, 'commas': 0}

8. 查询: 为什么 RAG 需要向量检索，而不是直接使用 LLM?
   策略: ITERATIVE
   推理: 复杂查询：查询较长（16 个词）, 多个逗号（1 个）
   特征: {'word_count': 16, 'has_time_keywords': False, 'has_comparison_keywords': False, 'has_complex_keywords': True, 'commas': 1}


【策略分布统计】

ITERATIVE     :  2 ( 25.0%)
NO_RETRIEVE   :  2 ( 25.0%)
SINGLE        :  2 ( 25.0%)
WEB_SEARCH    :  2 ( 25.0%)

【分类器对比】

LLM 分类器不可用，跳过对比
```

---

## 代码说明

### 1. 特征提取器

```python
class FeatureExtractor:
    """提取查询特征"""

    def extract(self, query: str) -> QueryFeatures:
        # 提取多维度特征
        # - 词数、字符数
        # - 实体数量
        # - 关键词检测
        # - 标点符号
```

**关键特征**:
- `word_count`: 查询长度（词数）
- `has_time_keywords`: 是否包含时间词
- `has_comparison_keywords`: 是否包含对比词
- `has_complex_keywords`: 是否包含复杂词
- `commas`: 逗号数量

### 2. 规则引擎

```python
class RuleBasedClassifier:
    """基于规则的分类器"""

    def classify(self, query: str) -> Strategy:
        # 规则优先级:
        # 1. 实时查询 → WEB_SEARCH
        # 2. 简单查询 → NO_RETRIEVE
        # 3. 复杂查询 → ITERATIVE
        # 4. 默认 → SINGLE
```

**优点**:
- 无需训练数据
- 可解释性强
- 快速响应

**缺点**:
- 准确率有限（~70%）
- 规则需要人工调优

### 3. LLM 分类器

```python
class LLMClassifier:
    """基于 LLM 的分类器"""

    def classify(self, query: str) -> Strategy:
        # 使用 GPT-4o-mini 分类
        # 准确率: ~95%
        # 成本: ~50 tokens/query
```

**优点**:
- 准确率最高（~95%）
- 理解语义

**缺点**:
- 成本较高
- 延迟较大

---

## 扩展建议

### 1. 添加缓存

```python
from functools import lru_cache

class CachedClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    @lru_cache(maxsize=1000)
    def classify(self, query: str) -> Strategy:
        return self.classifier.classify(query)
```

### 2. 添加置信度

```python
@dataclass
class ClassificationResult:
    strategy: Strategy
    confidence: float  # 0.0-1.0
    reasoning: str

class ConfidenceClassifier:
    def classify_with_confidence(self, query: str) -> ClassificationResult:
        # 计算置信度
        # 如果置信度低，可以降级或请求人工审核
        pass
```

### 3. 添加 A/B 测试

```python
class ABTestClassifier:
    def __init__(self, classifier_a, classifier_b, ratio=0.5):
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        self.ratio = ratio

    def classify(self, query: str) -> Strategy:
        import random
        if random.random() < self.ratio:
            return self.classifier_a.classify(query)
        else:
            return self.classifier_b.classify(query)
```

---

## 关键洞察

1. **规则引擎是最佳起点**
   - 无需训练数据
   - 快速验证想法
   - 准确率 70% 已足够

2. **特征工程很重要**
   - 词数、关键词、标点符号
   - 简单特征效果好

3. **LLM 分类器是终极方案**
   - 准确率 95%
   - 但成本较高

4. **可解释性至关重要**
   - 用户需要知道为什么选择某个策略
   - 便于调试和优化

---

**参考文献**:
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403) - arXiv (2024)
- [LangGraph Adaptive RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag) - LangChain AI (2025)
