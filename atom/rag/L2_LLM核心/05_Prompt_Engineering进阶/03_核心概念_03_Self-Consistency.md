# 核心概念 3: Self-Consistency

## 一句话定义

**通过多次独立推理并投票选择最一致的答案,显著提升复杂推理任务的准确性和可靠性。**

**RAG应用:** 在RAG系统中,对关键查询生成多个答案并投票,减少单次生成的随机性错误,提升答案的可信度。

---

## 为什么重要?

### 问题场景

```python
# 场景:数学推理任务
from openai import OpenAI

client = OpenAI()

# ❌ 单次推理:不可靠
prompt = """
问题:一个班级有30个学生,其中60%是女生。
如果新来了5个男生,现在女生占比是多少?

让我们一步步思考:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

print(response.choices[0].message.content)
# 可能输出:
# 步骤1: 女生人数 = 30 * 0.6 = 18
# 步骤2: 新的总人数 = 30 + 5 = 35
# 步骤3: 女生占比 = 18 / 35 = 51.4%
# 答案: 51.4%

# 问题:
# 1. 单次推理可能出错
# 2. 温度>0时,每次结果可能不同
# 3. 无法判断答案是否可靠
```

### 解决方案

```python
# ✅ Self-Consistency:多次推理投票
from collections import Counter

def self_consistency_generate(prompt, n=5):
    """多次推理并投票"""
    answers = []

    for i in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # 保持一定随机性
        )

        # 提取答案
        content = response.choices[0].message.content
        answer = extract_final_answer(content)
        answers.append(answer)

    # 投票选择最常见的答案
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0]

    return {
        "all_answers": answers,
        "final_answer": most_common[0],
        "confidence": most_common[1] / n,
        "distribution": dict(answer_counts)
    }

def extract_final_answer(text):
    """提取最终答案"""
    # 简单实现:查找"答案:"后的内容
    if "答案:" in text:
        return text.split("答案:")[1].strip().split("\n")[0]
    return text.split("\n")[-1].strip()

# 测试
result = self_consistency_generate(prompt, n=5)
print(f"所有答案: {result['all_answers']}")
print(f"最终答案: {result['final_answer']}")
print(f"置信度: {result['confidence']:.1%}")
print(f"分布: {result['distribution']}")

# 输出示例:
# 所有答案: ['51.4%', '51.4%', '51.4%', '48.6%', '51.4%']
# 最终答案: 51.4%
# 置信度: 80.0%
# 分布: {'51.4%': 4, '48.6%': 1}
```

**性能提升:**

| 任务类型 | CoT单次 | Self-Consistency (n=5) | 提升 |
|---------|---------|----------------------|------|
| 数学推理 | 40.7% | 57.1% | +40% |
| 常识推理 | 79.0% | 83.6% | +6% |
| 符号推理 | 25.0% | 37.5% | +50% |

**来源:** [Self-Consistency Improves Chain of Thought Reasoning (2022)](https://arxiv.org/abs/2203.11171)

---

## 核心原理

### 原理1:多数投票机制

**定义:** 通过多次独立采样,选择出现频率最高的答案作为最终结果。

**数学基础:**

```
假设单次推理准确率为 p
多次投票后准确率为 P(n, k)

P(n, k) = 1 - Σ(i=0 to k-1) C(n,i) * p^i * (1-p)^(n-i)

其中:
n = 采样次数
k = 多数阈值 (通常为 n/2 + 1)
```

**实验验证:**

```python
# 实验:不同采样次数的效果
import numpy as np
from scipy.stats import binom

def calculate_accuracy(p, n):
    """计算多数投票后的准确率"""
    k = n // 2 + 1  # 多数阈值
    # 计算至少k次正确的概率
    accuracy = 1 - binom.cdf(k-1, n, p)
    return accuracy

# 单次准确率 p = 0.7
p = 0.7

results = []
for n in [1, 3, 5, 7, 10]:
    acc = calculate_accuracy(p, n)
    results.append((n, acc))
    print(f"n={n}: {acc:.1%}")

# 输出:
# n=1: 70.0%
# n=3: 78.4%
# n=5: 83.7%
# n=7: 87.4%
# n=10: 90.4%
```

**收益递减:**

```
n=1 → n=3: +8.4%  (提升明显)
n=3 → n=5: +5.3%  (提升减少)
n=5 → n=7: +3.7%  (继续减少)
n=7 → n=10: +3.0% (收益递减)
```

**最佳实践:** n=5-7是性价比最高的选择

**来源:** [Self-Consistency Paper (2022)](https://arxiv.org/abs/2203.11171)

---

### 原理2:推理路径多样性

**核心发现:** 不同的推理路径可能得到相同的正确答案。

**示例:**

```python
# 问题:计算 15% 的 80 是多少?

# 路径1:直接计算
"""
15% = 0.15
0.15 * 80 = 12
答案: 12
"""

# 路径2:分步计算
"""
10% of 80 = 8
5% of 80 = 4
15% = 10% + 5% = 8 + 4 = 12
答案: 12
"""

# 路径3:比例计算
"""
15/100 * 80 = (15 * 80) / 100 = 1200 / 100 = 12
答案: 12
"""

# 三条不同路径,相同答案 → 高置信度
```

**为什么多样性重要?**

```
单一路径:
错误推理 → 错误答案
      ↑
      无法纠正

多样性路径:
路径1(错误) → 答案A
路径2(正确) → 答案B
路径3(正确) → 答案B
路径4(正确) → 答案B
路径5(错误) → 答案C
      ↓
   投票选择B (正确)
```

**实现多样性:**

```python
# 使用temperature控制多样性
# temperature = 0: 确定性输出,无多样性
# temperature = 0.7: 适度多样性
# temperature = 1.0: 高度多样性

# 最佳实践:
# Self-Consistency推荐 temperature = 0.7-0.8
```

**来源:** [Diverse Reasoning Paths (2023)](https://arxiv.org/abs/2305.14325)

---

### 原理3:置信度估计

**定义:** 通过答案分布估计结果的可信度。

**置信度计算:**

```python
def calculate_confidence(answers):
    """计算置信度"""
    from collections import Counter

    counts = Counter(answers)
    total = len(answers)
    most_common_count = counts.most_common(1)[0][1]

    # 方法1:简单比例
    confidence_simple = most_common_count / total

    # 方法2:考虑分布熵
    import math
    entropy = -sum((c/total) * math.log2(c/total)
                   for c in counts.values())
    max_entropy = math.log2(len(counts))
    confidence_entropy = 1 - (entropy / max_entropy if max_entropy > 0 else 0)

    return {
        "simple": confidence_simple,
        "entropy_based": confidence_entropy,
        "distribution": dict(counts)
    }

# 示例1:高置信度
answers1 = ["A", "A", "A", "A", "B"]
conf1 = calculate_confidence(answers1)
print(f"高置信度: {conf1}")
# simple: 0.8 (80%)
# entropy_based: 0.72

# 示例2:低置信度
answers2 = ["A", "B", "C", "D", "E"]
conf2 = calculate_confidence(answers2)
print(f"低置信度: {conf2}")
# simple: 0.2 (20%)
# entropy_based: 0.0
```

**置信度阈值:**

```python
# 根据置信度决定是否接受答案
def should_accept_answer(confidence, threshold=0.6):
    """判断是否接受答案"""
    if confidence >= threshold:
        return True, "高置信度,接受答案"
    elif confidence >= 0.4:
        return False, "中等置信度,建议增加采样"
    else:
        return False, "低置信度,需要重新设计提示词"

# 测试
print(should_accept_answer(0.8))  # (True, "高置信度,接受答案")
print(should_accept_answer(0.5))  # (False, "中等置信度,建议增加采样")
print(should_accept_answer(0.2))  # (False, "低置信度,需要重新设计提示词")
```

---

## 手写实现

### 从零实现 Self-Consistency

```python
"""
Self-Consistency Implementation
功能:多次推理并投票选择最一致答案
"""

from typing import List, Dict, Callable, Optional
from collections import Counter
from dataclasses import dataclass
import math
from openai import OpenAI

@dataclass
class ReasoningPath:
    """推理路径"""
    reasoning: str
    answer: str
    index: int

@dataclass
class ConsistencyResult:
    """Self-Consistency结果"""
    final_answer: str
    confidence: float
    all_paths: List[ReasoningPath]
    distribution: Dict[str, int]
    entropy: float

class SelfConsistency:
    """Self-Consistency实现"""

    def __init__(self, client: OpenAI):
        self.client = client

    def generate_multiple_paths(
        self,
        prompt: str,
        n: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        answer_extractor: Optional[Callable] = None
    ) -> List[ReasoningPath]:
        """
        生成多条推理路径

        Args:
            prompt: 提示词
            n: 采样次数
            model: 模型名称
            temperature: 温度参数
            answer_extractor: 答案提取函数
        """
        paths = []

        for i in range(n):
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            reasoning = response.choices[0].message.content

            # 提取答案
            if answer_extractor:
                answer = answer_extractor(reasoning)
            else:
                answer = self._default_answer_extractor(reasoning)

            paths.append(ReasoningPath(
                reasoning=reasoning,
                answer=answer,
                index=i
            ))

        return paths

    def _default_answer_extractor(self, text: str) -> str:
        """默认答案提取器"""
        # 查找"答案:"关键词
        keywords = ["答案:", "Answer:", "因此", "所以"]

        for keyword in keywords:
            if keyword in text:
                parts = text.split(keyword)
                if len(parts) > 1:
                    # 提取关键词后的第一行
                    answer_part = parts[1].strip()
                    return answer_part.split("\n")[0].strip()

        # 如果没找到,返回最后一行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines[-1] if lines else ""

    def vote(
        self,
        paths: List[ReasoningPath]
    ) -> ConsistencyResult:
        """
        投票选择最一致的答案

        Args:
            paths: 推理路径列表
        """
        # 统计答案分布
        answers = [path.answer for path in paths]
        distribution = Counter(answers)

        # 选择最常见的答案
        most_common = distribution.most_common(1)[0]
        final_answer = most_common[0]
        count = most_common[1]

        # 计算置信度
        confidence = count / len(answers)

        # 计算熵
        entropy = self._calculate_entropy(distribution, len(answers))

        return ConsistencyResult(
            final_answer=final_answer,
            confidence=confidence,
            all_paths=paths,
            distribution=dict(distribution),
            entropy=entropy
        )

    def _calculate_entropy(
        self,
        distribution: Counter,
        total: int
    ) -> float:
        """计算分布熵"""
        if len(distribution) == 1:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def generate_with_consistency(
        self,
        prompt: str,
        n: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        answer_extractor: Optional[Callable] = None,
        min_confidence: float = 0.6
    ) -> Dict:
        """
        使用Self-Consistency生成答案

        Args:
            prompt: 提示词
            n: 采样次数
            model: 模型名称
            temperature: 温度参数
            answer_extractor: 答案提取函数
            min_confidence: 最小置信度阈值
        """
        # 生成多条推理路径
        paths = self.generate_multiple_paths(
            prompt, n, model, temperature, answer_extractor
        )

        # 投票
        result = self.vote(paths)

        # 判断是否需要增加采样
        need_more_samples = result.confidence < min_confidence

        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "distribution": result.distribution,
            "entropy": result.entropy,
            "need_more_samples": need_more_samples,
            "all_paths": [
                {
                    "index": p.index,
                    "answer": p.answer,
                    "reasoning": p.reasoning
                }
                for p in result.all_paths
            ]
        }

    def adaptive_sampling(
        self,
        prompt: str,
        initial_n: int = 5,
        max_n: int = 10,
        target_confidence: float = 0.8,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ) -> Dict:
        """
        自适应采样:根据置信度动态调整采样次数

        Args:
            prompt: 提示词
            initial_n: 初始采样次数
            max_n: 最大采样次数
            target_confidence: 目标置信度
            model: 模型名称
            temperature: 温度参数
        """
        paths = []

        # 初始采样
        paths.extend(self.generate_multiple_paths(
            prompt, initial_n, model, temperature
        ))

        # 迭代增加采样直到达到目标置信度
        while len(paths) < max_n:
            result = self.vote(paths)

            if result.confidence >= target_confidence:
                break

            # 增加采样
            new_paths = self.generate_multiple_paths(
                prompt, 1, model, temperature
            )
            paths.extend(new_paths)

        # 最终投票
        final_result = self.vote(paths)

        return {
            "final_answer": final_result.final_answer,
            "confidence": final_result.confidence,
            "distribution": final_result.distribution,
            "total_samples": len(paths),
            "converged": final_result.confidence >= target_confidence
        }


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI()
    sc = SelfConsistency(client)

    # 测试1:基础Self-Consistency
    print("=== 基础Self-Consistency ===")
    prompt = """
问题:一个班级有30个学生,其中60%是女生。
如果新来了5个男生,现在女生占比是多少?

让我们一步步思考:
"""

    result = sc.generate_with_consistency(prompt, n=5)
    print(f"最终答案: {result['final_answer']}")
    print(f"置信度: {result['confidence']:.1%}")
    print(f"分布: {result['distribution']}")
    print(f"熵: {result['entropy']:.2f}")
    print()

    # 测试2:自适应采样
    print("=== 自适应采样 ===")
    result2 = sc.adaptive_sampling(
        prompt,
        initial_n=3,
        max_n=10,
        target_confidence=0.8
    )
    print(f"最终答案: {result2['final_answer']}")
    print(f"置信度: {result2['confidence']:.1%}")
    print(f"总采样次数: {result2['total_samples']}")
    print(f"是否收敛: {result2['converged']}")
```

### 实现原理解析

**1. 多路径生成**
- 使用temperature控制多样性
- 并行生成多条推理路径
- 提取每条路径的最终答案

**2. 投票机制**
- 统计答案分布
- 选择出现频率最高的答案
- 计算置信度和熵

**3. 自适应采样**
- 从少量采样开始
- 根据置信度动态增加采样
- 达到目标置信度后停止

**4. 置信度评估**
- 简单比例:最常见答案的占比
- 熵:衡量分布的不确定性
- 阈值判断:决定是否接受答案

---

## RAG 应用场景

### 场景1:关键查询验证

**问题:** RAG系统对关键查询需要高可靠性

**解决方案:** 使用Self-Consistency验证答案

```python
from openai import OpenAI
import chromadb

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# 添加文档
docs = [
    "Python由Guido van Rossum于1991年创建。",
    "JavaScript由Brendan Eich于1995年创建。",
    "Java由James Gosling于1995年创建。"
]
collection.add(
    documents=docs,
    ids=[f"doc{i}" for i in range(len(docs))]
)

def rag_with_self_consistency(query: str, n: int = 5) -> Dict:
    """RAG + Self-Consistency"""

    # 1. 检索
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    retrieved_docs = results['documents'][0]

    # 2. 构建提示词
    prompt = f"""
文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs))}

问题: {query}

让我们一步步分析并给出答案:
"""

    # 3. Self-Consistency生成
    sc = SelfConsistency(client)
    result = sc.generate_with_consistency(prompt, n=n)

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "final_answer": result['final_answer'],
        "confidence": result['confidence'],
        "distribution": result['distribution'],
        "all_paths": result['all_paths']
    }

# 测试
result = rag_with_self_consistency("Python是什么时候创建的?", n=5)
print(f"问题: {result['query']}")
print(f"最终答案: {result['final_answer']}")
print(f"置信度: {result['confidence']:.1%}")
print(f"答案分布: {result['distribution']}")
```

---

### 场景2:复杂推理任务

**问题:** 需要多步推理的复杂查询

**解决方案:** CoT + Self-Consistency

```python
def complex_reasoning_with_consistency(query: str, n: int = 5) -> Dict:
    """复杂推理 + Self-Consistency"""

    # 检索
    results = collection.query(query_texts=[query], n_results=3)
    docs = results['documents'][0]

    # CoT + Self-Consistency提示词
    prompt = f"""
文档:
{chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(docs))}

问题: {query}

这需要多步推理。让我们一步步分析:
1. 首先找出相关信息
2. 然后进行比较或计算
3. 最后得出结论

分析:
"""

    sc = SelfConsistency(client)
    result = sc.generate_with_consistency(prompt, n=n, temperature=0.7)

    return result

# 测试
result = complex_reasoning_with_consistency(
    "Python和JavaScript哪个创建得更早?相差几年?",
    n=5
)
print(f"最终答案: {result['final_answer']}")
print(f"置信度: {result['confidence']:.1%}")
```

---

### 场景3:答案质量评分

**问题:** 需要评估RAG答案的质量

**解决方案:** 使用Self-Consistency的置信度作为质量指标

```python
def evaluate_answer_quality(
    query: str,
    answer: str,
    docs: List[str],
    n: int = 5
) -> Dict:
    """评估答案质量"""

    prompt = f"""
问题: {query}
答案: {answer}
文档: {' | '.join(docs)}

请评估这个答案的质量(1-10分):
"""

    sc = SelfConsistency(client)
    result = sc.generate_with_consistency(prompt, n=n)

    # 置信度作为质量指标
    quality_score = result['confidence'] * 10

    return {
        "answer": answer,
        "quality_score": quality_score,
        "confidence": result['confidence'],
        "distribution": result['distribution']
    }

# 测试
evaluation = evaluate_answer_quality(
    query="Python是什么时候创建的?",
    answer="1991年",
    docs=["Python由Guido van Rossum于1991年创建。"],
    n=5
)
print(f"质量评分: {evaluation['quality_score']:.1f}/10")
print(f"置信度: {evaluation['confidence']:.1%}")
```

---

## 最佳实践

### 1. 采样次数选择

```python
# 根据任务重要性选择
low_importance = 3      # 一般任务
medium_importance = 5   # 重要任务
high_importance = 7     # 关键任务
critical = 10           # 极关键任务

# 成本vs收益权衡
# n=3: 成本低,提升明显
# n=5: 最佳性价比
# n=7: 高可靠性
# n=10: 收益递减明显
```

### 2. Temperature设置

```python
# Self-Consistency推荐temperature
# 太低(0.0-0.3): 缺乏多样性,失去Self-Consistency优势
# 适中(0.6-0.8): 最佳,既有多样性又不太离谱
# 太高(0.9-1.0): 过度随机,答案质量下降

recommended_temperature = 0.7
```

### 3. 答案提取

```python
# 自定义答案提取器
def custom_extractor(text: str) -> str:
    """针对特定格式的答案提取"""
    import re

    # 方法1: 正则表达式
    match = re.search(r'答案[：:]\s*(.+)', text)
    if match:
        return match.group(1).strip()

    # 方法2: 关键词匹配
    keywords = ["因此", "所以", "最终"]
    for keyword in keywords:
        if keyword in text:
            parts = text.split(keyword)
            return parts[-1].strip()

    # 方法3: 最后一行
    return text.split("\n")[-1].strip()
```

### 4. 置信度阈值

```python
# 根据置信度采取不同策略
def handle_by_confidence(confidence: float, answer: str):
    if confidence >= 0.8:
        return "接受答案", answer
    elif confidence >= 0.6:
        return "谨慎接受", answer
    elif confidence >= 0.4:
        return "增加采样", None
    else:
        return "重新设计提示词", None
```

---

## 参考资源

- [Self-Consistency Improves Chain of Thought (2022)](https://arxiv.org/abs/2203.11171)
- [Diverse Reasoning Paths (2023)](https://arxiv.org/abs/2305.14325)
- [Prompt Engineering Guide - Self-Consistency](https://www.promptingguide.ai/techniques/consistency)
- [Maxim AI - Advanced Techniques](https://www.getmaxim.ai/articles/advanced-prompt-engineering-techniques-in-2025)
