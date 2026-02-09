# 核心概念：一致性检测与NLI

> 使用自然语言推理检测生成内容与检索内容的一致性

---

## 原理讲解

### 什么是一致性检测？

**一致性检测**是验证 LLM 生成的答案是否与检索到的文档内容一致的过程。

**核心问题：**
```
检索文档：「Python 3.9 于 2020 年 10 月 5 日发布」
LLM 生成：「Python 3.9 于 2021 年发布」

如何自动判断这两段文本是否一致？
```

### 什么是 NLI（自然语言推理）？

**NLI（Natural Language Inference）** 是判断两段文本之间逻辑关系的任务。

**三种关系：**

1. **Entailment（蕴含）**：hypothesis 可以从 premise 推导出
   ```
   Premise: "Python 3.9 于 2020 年 10 月发布"
   Hypothesis: "Python 3.9 在 2020 年发布"
   关系: Entailment ✅
   ```

2. **Contradiction（矛盾）**：hypothesis 与 premise 矛盾
   ```
   Premise: "Python 3.9 于 2020 年 10 月发布"
   Hypothesis: "Python 3.9 于 2021 年发布"
   关系: Contradiction ❌
   ```

3. **Neutral（中立）**：hypothesis 与 premise 无关
   ```
   Premise: "Python 3.9 于 2020 年 10 月发布"
   Hypothesis: "Python 很流行"
   关系: Neutral ⚪
   ```

### NLI 在 RAG 中的应用

**核心思想：** 将检索文档作为 premise，生成答案作为 hypothesis，使用 NLI 模型判断关系。

```python
# RAG 一致性检测流程
def check_consistency(answer, retrieved_doc):
    # 1. 构建句子对
    premise = retrieved_doc
    hypothesis = answer

    # 2. NLI 模型推理
    scores = nli_model.predict([(premise, hypothesis)])
    # 输出：[contradiction_score, neutral_score, entailment_score]

    # 3. 提取蕴含分数
    entailment_score = scores[0][2]

    # 4. 判断一致性
    if entailment_score > 0.8:
        return "高度一致"
    elif entailment_score > 0.6:
        return "基本一致"
    else:
        return "可能不一致"
```

### 为什么 NLI 适合一致性检测？

**优势：**

1. **语义理解**：不是简单的字符串匹配，而是理解语义
   ```
   文档："Python 3.9 于 2020 年 10 月发布"
   答案："Python 3.9 在 2020 年秋季发布"
   → NLI 可以识别"10月"和"秋季"的语义关系
   ```

2. **逻辑推理**：可以处理隐含推理
   ```
   文档："Python 3.9 新增了字典合并运算符"
   答案："Python 3.9 支持字典合并"
   → NLI 可以推理出"新增"意味着"支持"
   ```

3. **预训练模型**：无需训练，直接使用
   - 大量预训练 NLI 模型可用（DeBERTa、RoBERTa 等）
   - 在通用数据集上训练，泛化能力强

**局限：**

1. **不是100%准确**：模型也会误判
2. **上下文依赖**：缺乏更广泛的上下文信息
3. **领域适应性**：在专业领域可能不够准确

---

## 手写实现

### 实现1：简单的 NLI 一致性检测器

```python
from sentence_transformers import CrossEncoder
from typing import List, Dict

class SimpleConsistencyChecker:
    """
    简单的一致性检测器
    使用预训练 NLI 模型检测答案与文档的一致性
    """

    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-base'):
        """
        初始化 NLI 模型

        Args:
            model_name: 预训练 NLI 模型名称
        """
        print(f"加载 NLI 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        print("模型加载完成")

    def check(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        检查两段文本的一致性

        Args:
            premise: 前提（检索文档）
            hypothesis: 假设（生成答案）

        Returns:
            包含三个分数的字典：contradiction, neutral, entailment
        """
        # NLI 模型推理
        scores = self.model.predict([(premise, hypothesis)])

        # 解析分数
        result = {
            'contradiction': float(scores[0][0]),  # 矛盾分数
            'neutral': float(scores[0][1]),        # 中立分数
            'entailment': float(scores[0][2])      # 蕴含分数
        }

        return result

    def is_consistent(self, premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
        """
        判断是否一致

        Args:
            premise: 前提
            hypothesis: 假设
            threshold: 蕴含分数阈值

        Returns:
            True 表示一致，False 表示不一致
        """
        scores = self.check(premise, hypothesis)
        return scores['entailment'] >= threshold


# 使用示例
if __name__ == "__main__":
    checker = SimpleConsistencyChecker()

    # 测试1：一致的情况
    doc1 = "Python 3.9 于 2020 年 10 月 5 日发布"
    answer1 = "Python 3.9 在 2020 年发布"

    scores1 = checker.check(doc1, answer1)
    print(f"\n测试1：一致的情况")
    print(f"文档: {doc1}")
    print(f"答案: {answer1}")
    print(f"分数: {scores1}")
    print(f"是否一致: {checker.is_consistent(doc1, answer1)}")

    # 测试2：矛盾的情况
    doc2 = "Python 3.9 于 2020 年 10 月 5 日发布"
    answer2 = "Python 3.9 于 2021 年发布"

    scores2 = checker.check(doc2, answer2)
    print(f"\n测试2：矛盾的情况")
    print(f"文档: {doc2}")
    print(f"答案: {answer2}")
    print(f"分数: {scores2}")
    print(f"是否一致: {checker.is_consistent(doc2, answer2)}")

    # 测试3：中立的情况
    doc3 = "Python 3.9 于 2020 年 10 月 5 日发布"
    answer3 = "Python 是一种流行的编程语言"

    scores3 = checker.check(doc3, answer3)
    print(f"\n测试3：中立的情况")
    print(f"文档: {doc3}")
    print(f"答案: {answer3}")
    print(f"分数: {scores3}")
    print(f"是否一致: {checker.is_consistent(doc3, answer3)}")
```

**预期输出：**
```
加载 NLI 模型: cross-encoder/nli-deberta-v3-base
模型加载完成

测试1：一致的情况
文档: Python 3.9 于 2020 年 10 月 5 日发布
答案: Python 3.9 在 2020 年发布
分数: {'contradiction': 0.02, 'neutral': 0.13, 'entailment': 0.85}
是否一致: True

测试2：矛盾的情况
文档: Python 3.9 于 2020 年 10 月 5 日发布
答案: Python 3.9 于 2021 年发布
分数: {'contradiction': 0.78, 'neutral': 0.15, 'entailment': 0.07}
是否一致: False

测试3：中立的情况
文档: Python 3.9 于 2020 年 10 月 5 日发布
答案: Python 是一种流行的编程语言
分数: {'contradiction': 0.05, 'neutral': 0.82, 'entailment': 0.13}
是否一致: False
```

### 实现2：多文档一致性检测器

```python
from typing import List, Dict
import numpy as np

class MultiDocConsistencyChecker(SimpleConsistencyChecker):
    """
    多文档一致性检测器
    检查答案与多个检索文档的一致性
    """

    def check_multi_docs(
        self,
        answer: str,
        docs: List[str],
        aggregation: str = 'max'
    ) -> Dict[str, float]:
        """
        检查答案与多个文档的一致性

        Args:
            answer: 生成的答案
            docs: 检索到的多个文档
            aggregation: 聚合方式 ('max', 'mean', 'min')

        Returns:
            聚合后的一致性分数
        """
        if not docs:
            return {'contradiction': 0.0, 'neutral': 1.0, 'entailment': 0.0}

        # 检查答案与每个文档的一致性
        all_scores = []
        for doc in docs:
            scores = self.check(doc, answer)
            all_scores.append(scores)

        # 聚合分数
        if aggregation == 'max':
            # 取最大蕴含分数（最乐观）
            entailment_scores = [s['entailment'] for s in all_scores]
            best_idx = np.argmax(entailment_scores)
            return all_scores[best_idx]

        elif aggregation == 'mean':
            # 取平均分数
            return {
                'contradiction': np.mean([s['contradiction'] for s in all_scores]),
                'neutral': np.mean([s['neutral'] for s in all_scores]),
                'entailment': np.mean([s['entailment'] for s in all_scores])
            }

        elif aggregation == 'min':
            # 取最小蕴含分数（最保守）
            entailment_scores = [s['entailment'] for s in all_scores]
            worst_idx = np.argmin(entailment_scores)
            return all_scores[worst_idx]

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def check_sentence_level(
        self,
        answer: str,
        docs: List[str]
    ) -> List[Dict[str, any]]:
        """
        句子级别的一致性检测
        将答案拆分成句子，逐句检测

        Args:
            answer: 生成的答案
            docs: 检索到的文档

        Returns:
            每个句子的一致性检测结果
        """
        # 简单的句子分割（实际应用中可以使用 nltk 或 spacy）
        sentences = [s.strip() for s in answer.split('。') if s.strip()]

        results = []
        for sentence in sentences:
            scores = self.check_multi_docs(sentence, docs, aggregation='max')
            results.append({
                'sentence': sentence,
                'scores': scores,
                'is_consistent': scores['entailment'] >= 0.7
            })

        return results


# 使用示例
if __name__ == "__main__":
    checker = MultiDocConsistencyChecker()

    # 测试：多文档一致性检测
    docs = [
        "Python 3.9 于 2020 年 10 月 5 日发布",
        "Python 3.9 新增了字典合并运算符 |",
        "Python 3.9 改进了类型提示功能"
    ]

    answer = "Python 3.9 在 2020 年发布，新增了字典合并运算符"

    print("=== 多文档一致性检测 ===")
    print(f"答案: {answer}\n")

    # 不同聚合方式
    for agg in ['max', 'mean', 'min']:
        scores = checker.check_multi_docs(answer, docs, aggregation=agg)
        print(f"{agg.upper()} 聚合:")
        print(f"  蕴含分数: {scores['entailment']:.2f}")
        print(f"  是否一致: {scores['entailment'] >= 0.7}\n")

    # 句子级别检测
    print("=== 句子级别检测 ===")
    sentence_results = checker.check_sentence_level(answer, docs)
    for i, result in enumerate(sentence_results, 1):
        print(f"句子{i}: {result['sentence']}")
        print(f"  蕴含分数: {result['scores']['entailment']:.2f}")
        print(f"  是否一致: {result['is_consistent']}\n")
```

---

## RAG应用场景

### 场景1：问答系统的答案验证

**需求：** 在文档问答系统中，验证 LLM 生成的答案是否基于检索文档

```python
def qa_with_consistency_check(query: str, retriever, llm, checker):
    """
    带一致性检测的问答系统
    """
    # 1. 检索相关文档
    docs = retriever.search(query, top_k=3)

    # 2. 生成答案
    context = "\n\n".join([doc.content for doc in docs])
    prompt = f"基于以下文档回答问题：\n{context}\n\n问题：{query}\n答案："
    answer = llm.generate(prompt)

    # 3. 一致性检测
    scores = checker.check_multi_docs(answer, [doc.content for doc in docs])

    # 4. 根据一致性分数决定是否返回答案
    if scores['entailment'] >= 0.8:
        return {
            "answer": answer,
            "confidence": "高",
            "consistency_score": scores['entailment']
        }
    elif scores['entailment'] >= 0.6:
        return {
            "answer": f"（不太确定）{answer}",
            "confidence": "中",
            "consistency_score": scores['entailment']
        }
    else:
        return {
            "answer": "抱歉，我对这个答案不够确定",
            "confidence": "低",
            "consistency_score": scores['entailment']
        }
```

### 场景2：事实验证系统

**需求：** 验证新闻文章或社交媒体内容的事实准确性

```python
def fact_verification(claim: str, evidence_docs: List[str], checker):
    """
    事实验证系统
    """
    # 检查声明与证据文档的一致性
    scores = checker.check_multi_docs(claim, evidence_docs, aggregation='mean')

    # 判断事实准确性
    if scores['entailment'] >= 0.8:
        verdict = "支持"
        explanation = "证据文档支持该声明"
    elif scores['contradiction'] >= 0.6:
        verdict = "反驳"
        explanation = "证据文档与该声明矛盾"
    else:
        verdict = "无法判断"
        explanation = "证据不足以判断该声明的真伪"

    return {
        "claim": claim,
        "verdict": verdict,
        "explanation": explanation,
        "scores": scores
    }
```

### 场景3：多轮对话的一致性监控

**需求：** 在多轮对话中，确保 LLM 的回答前后一致

```python
class DialogueConsistencyMonitor:
    """
    对话一致性监控器
    """

    def __init__(self, checker):
        self.checker = checker
        self.history = []  # 存储对话历史

    def add_turn(self, query: str, answer: str, docs: List[str]):
        """添加一轮对话"""
        self.history.append({
            "query": query,
            "answer": answer,
            "docs": docs
        })

    def check_current_consistency(self) -> Dict:
        """
        检查当前回答与历史回答的一致性
        """
        if len(self.history) < 2:
            return {"consistent": True, "issues": []}

        current = self.history[-1]
        issues = []

        # 检查当前回答与之前回答的一致性
        for i, prev in enumerate(self.history[:-1]):
            scores = self.checker.check(prev['answer'], current['answer'])

            if scores['contradiction'] >= 0.6:
                issues.append({
                    "turn": i + 1,
                    "previous_answer": prev['answer'],
                    "current_answer": current['answer'],
                    "contradiction_score": scores['contradiction']
                })

        return {
            "consistent": len(issues) == 0,
            "issues": issues
        }
```

---

## 完整代码示例

### 端到端的 RAG 一致性检测系统

```python
"""
完整的 RAG 一致性检测系统
演示：从检索到生成到验证的完整流程
"""

from sentence_transformers import CrossEncoder
from openai import OpenAI
import os
from typing import List, Dict

# ===== 1. 初始化组件 =====
print("=== 初始化 RAG 系统 ===")

# NLI 模型
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
print("✓ NLI 模型加载完成")

# LLM 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✓ LLM 客户端初始化完成")

# ===== 2. 模拟检索文档 =====
print("\n=== 模拟检索 ===")

query = "Python 3.9 有什么新特性？"
print(f"查询: {query}")

retrieved_docs = [
    "Python 3.9 于 2020 年 10 月 5 日发布，这是一个重要的版本更新。",
    "Python 3.9 新增了字典合并运算符 |，可以方便地合并两个字典。",
    "Python 3.9 改进了类型提示功能，支持使用内置集合类型作为泛型。"
]

print(f"检索到 {len(retrieved_docs)} 个文档")

# ===== 3. 生成答案 =====
print("\n=== 生成答案 ===")

context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
prompt = f"""基于以下文档回答问题：

{context}

问题：{query}

要求：严格基于文档内容回答，不要添加文档中没有的信息。

答案："""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

answer = response.choices[0].message.content
print(f"生成答案: {answer}")

# ===== 4. 一致性检测 =====
print("\n=== 一致性检测 ===")

# 检查答案与每个文档的一致性
consistency_scores = []
for i, doc in enumerate(retrieved_docs):
    scores = nli_model.predict([(doc, answer)])
    entailment_score = scores[0][2]
    consistency_scores.append(entailment_score)

    print(f"文档{i+1} 一致性: {entailment_score:.2f}")

# 计算平均一致性
avg_consistency = sum(consistency_scores) / len(consistency_scores)
print(f"\n平均一致性分数: {avg_consistency:.2f}")

# ===== 5. 决策 =====
print("\n=== 决策 ===")

threshold = 0.7

if avg_consistency >= threshold:
    print(f"✓ 一致性检测通过（{avg_consistency:.2f} >= {threshold}）")
    print(f"最终答案: {answer}")
else:
    print(f"✗ 一致性检测未通过（{avg_consistency:.2f} < {threshold}）")
    print("最终答案: 抱歉，我对这个答案不够确定")

# ===== 6. 详细分析 =====
print("\n=== 详细分析 ===")

# 句子级别检测
sentences = [s.strip() + '。' for s in answer.split('。') if s.strip()]
print(f"答案包含 {len(sentences)} 个句子\n")

for i, sentence in enumerate(sentences, 1):
    print(f"句子{i}: {sentence}")

    # 检查该句子与所有文档的一致性
    sentence_scores = []
    for doc in retrieved_docs:
        scores = nli_model.predict([(doc, sentence)])
        sentence_scores.append(scores[0][2])

    max_score = max(sentence_scores)
    best_doc_idx = sentence_scores.index(max_score)

    print(f"  最佳匹配: 文档{best_doc_idx + 1}")
    print(f"  一致性分数: {max_score:.2f}")
    print(f"  状态: {'✓ 通过' if max_score >= threshold else '✗ 未通过'}\n")
```

**运行输出示例：**
```
=== 初始化 RAG 系统 ===
✓ NLI 模型加载完成
✓ LLM 客户端初始化完成

=== 模拟检索 ===
查询: Python 3.9 有什么新特性？
检索到 3 个文档

=== 生成答案 ===
生成答案: Python 3.9 于 2020 年 10 月 5 日发布，主要新特性包括：新增了字典合并运算符 |，可以方便地合并两个字典；改进了类型提示功能，支持使用内置集合类型作为泛型。

=== 一致性检测 ===
文档1 一致性: 0.82
文档2 一致性: 0.88
文档3 一致性: 0.85

平均一致性分数: 0.85

=== 决策 ===
✓ 一致性检测通过（0.85 >= 0.7）
最终答案: Python 3.9 于 2020 年 10 月 5 日发布，主要新特性包括：新增了字典合并运算符 |，可以方便地合并两个字典；改进了类型提示功能，支持使用内置集合类型作为泛型。

=== 详细分析 ===
答案包含 2 个句子

句子1: Python 3.9 于 2020 年 10 月 5 日发布，主要新特性包括：新增了字典合并运算符 |，可以方便地合并两个字典。
  最佳匹配: 文档2
  一致性分数: 0.88
  状态: ✓ 通过

句子2: 改进了类型提示功能，支持使用内置集合类型作为泛型。
  最佳匹配: 文档3
  一致性分数: 0.85
  状态: ✓ 通过
```

---

## 关键要点

### 1. NLI 模型的选择

**推荐模型：**
- `cross-encoder/nli-deberta-v3-base`：平衡性能和准确率
- `cross-encoder/nli-roberta-large`：更高准确率，但更慢
- `cross-encoder/nli-MiniLM2-L6-H768`：更快，但准确率略低

### 2. 阈值设置

**不同场景的推荐阈值：**
- 高风险（医疗、法律）：0.85-0.9
- 中风险（客服、问答）：0.7-0.8
- 低风险（推荐、娱乐）：0.6-0.7

### 3. 性能优化

**优化策略：**
- 批量处理：一次检测多个句子对
- 缓存结果：相同查询不重复检测
- 异步检测：不阻塞用户响应

### 4. 局限性

**NLI 模型的局限：**
- 不是100%准确，需要设置合理阈值
- 对复杂语义和隐含推理可能误判
- 需要结合其他方法（关键词匹配、语义相似度）

---

## 总结

**一致性检测与 NLI 的核心价值：**

1. **自动化验证**：无需人工审核，自动检测幻觉
2. **语义理解**：不是简单的字符串匹配，而是理解语义关系
3. **即插即用**：使用预训练模型，无需训练
4. **灵活调整**：通过阈值控制严格程度

**在 RAG 开发中的应用：**

- 问答系统的答案验证
- 事实验证系统
- 多轮对话的一致性监控
- 内容审核和质量保障

**记住：**

> **NLI 一致性检测是幻觉检测的核心技术，但不是唯一方法。**
>
> **需要结合引用溯源、多策略缓解，构建完整的质量保障体系。**
