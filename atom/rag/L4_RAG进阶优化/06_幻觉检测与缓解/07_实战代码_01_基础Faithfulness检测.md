# 实战代码1：基础Faithfulness检测

> **使用RAGAS框架实现基础的Faithfulness评估**

---

## 场景描述

构建一个基础的RAG幻觉检测系统，使用RAGAS框架评估生成内容的Faithfulness分数，快速筛选出可能存在幻觉的回答。

**适用场景：**
- 原型开发阶段
- 快速评估RAG系统质量
- 离线批量评估

---

## 环境准备

### 安装依赖

```bash
# 安装RAGAS和相关依赖
pip install ragas openai datasets python-dotenv

# 可选：安装LangChain（用于RAG集成）
pip install langchain langchain-openai
```

### 配置API密钥

```bash
# 创建.env文件
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选：自定义端点
EOF
```

---

## 完整代码实现

```python
"""
基础Faithfulness检测系统
使用RAGAS框架评估RAG生成内容的忠实度
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_openai import ChatOpenAI
import json

# 加载环境变量
load_dotenv()


class FaithfulnessDetector:
    """基础Faithfulness检测器"""

    def __init__(self, model_name: str = "gpt-4o-mini", threshold: float = 0.7):
        """
        初始化检测器

        Args:
            model_name: 使用的LLM模型
            threshold: Faithfulness阈值
        """
        self.model_name = model_name
        self.threshold = threshold
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def evaluate_single(
        self, question: str, contexts: List[str], answer: str
    ) -> Dict:
        """
        评估单个问答对的Faithfulness

        Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            answer: 生成的回答

        Returns:
            评估结果字典
        """
        # 准备数据
        data = {
            "question": [question],
            "contexts": [contexts],
            "answer": [answer],
        }

        # 转换为Dataset
        dataset = Dataset.from_dict(data)

        # 评估Faithfulness
        result = evaluate(dataset, metrics=[faithfulness], llm=self.llm)

        faithfulness_score = result["faithfulness"]

        return {
            "question": question,
            "answer": answer,
            "faithfulness_score": faithfulness_score,
            "passed": faithfulness_score >= self.threshold,
            "status": "passed" if faithfulness_score >= self.threshold else "rejected",
        }

    def evaluate_batch(self, samples: List[Dict]) -> Dict:
        """
        批量评估多个问答对

        Args:
            samples: 样本列表，每个样本包含question, contexts, answer

        Returns:
            批量评估结果
        """
        # 准备批量数据
        data = {
            "question": [s["question"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "answer": [s["answer"] for s in samples],
        }

        # 转换为Dataset
        dataset = Dataset.from_dict(data)

        # 批量评估
        result = evaluate(dataset, metrics=[faithfulness], llm=self.llm)

        # 解析结果
        faithfulness_scores = result["faithfulness"]

        results = []
        for i, sample in enumerate(samples):
            score = faithfulness_scores[i] if isinstance(faithfulness_scores, list) else faithfulness_scores
            results.append(
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "faithfulness_score": score,
                    "passed": score >= self.threshold,
                }
            )

        # 统计
        avg_score = sum(r["faithfulness_score"] for r in results) / len(results)
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)

        return {
            "results": results,
            "avg_faithfulness": avg_score,
            "pass_rate": pass_rate,
            "total_samples": len(results),
            "passed_samples": sum(1 for r in results if r["passed"]),
        }


class RAGWithFaithfulness:
    """集成Faithfulness检测的RAG系统"""

    def __init__(self, threshold: float = 0.7):
        """
        初始化RAG系统

        Args:
            threshold: Faithfulness阈值
        """
        self.detector = FaithfulnessDetector(threshold=threshold)
        self.threshold = threshold

    def retrieve_contexts(self, question: str) -> List[str]:
        """
        检索相关上下文（模拟实现）

        Args:
            question: 用户问题

        Returns:
            上下文列表
        """
        # 这里是模拟实现，实际应该调用向量数据库
        mock_contexts = {
            "什么是Python?": [
                "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
                "Python语法简洁，易于学习，广泛用于数据科学和Web开发。",
            ],
            "Python有什么特点?": [
                "Python语法简洁，易于学习。",
                "Python是解释型语言，支持多种编程范式。",
            ],
        }

        return mock_contexts.get(question, ["未找到相关上下文"])

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        生成回答（模拟实现）

        Args:
            question: 用户问题
            contexts: 检索到的上下文

        Returns:
            生成的回答
        """
        # 这里是模拟实现，实际应该调用LLM
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        context_text = "\n".join(contexts)
        prompt = f"""
基于以下上下文回答问题：

上下文：
{context_text}

问题：{question}

要求：
1. 仅基于上下文回答
2. 不要添加上下文中没有的信息
3. 如果上下文不足以回答，说"基于提供的信息，我无法回答这个问题"
"""

        response = llm.invoke(prompt)
        return response.content

    def query(self, question: str) -> Dict:
        """
        完整的RAG查询流程（集成Faithfulness检测）

        Args:
            question: 用户问题

        Returns:
            查询结果
        """
        # 1. 检索
        contexts = self.retrieve_contexts(question)

        # 2. 生成
        answer = self.generate_answer(question, contexts)

        # 3. Faithfulness检测
        detection_result = self.detector.evaluate_single(question, contexts, answer)

        # 4. 根据检测结果决定是否返回
        if detection_result["passed"]:
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "faithfulness_score": detection_result["faithfulness_score"],
                "status": "passed",
            }
        else:
            return {
                "question": question,
                "answer": "抱歉，我无法基于提供的信息回答这个问题。",
                "original_answer": answer,
                "contexts": contexts,
                "faithfulness_score": detection_result["faithfulness_score"],
                "status": "rejected",
                "reason": f"Faithfulness分数过低 ({detection_result['faithfulness_score']:.2f} < {self.threshold})",
            }


def demo_single_evaluation():
    """演示单个样本评估"""
    print("=" * 60)
    print("演示1：单个样本Faithfulness评估")
    print("=" * 60)

    detector = FaithfulnessDetector(threshold=0.7)

    # 测试样本1：完全一致
    question1 = "什么是Python?"
    contexts1 = ["Python是一种高级编程语言，由Guido van Rossum创建。"]
    answer1 = "Python是一种由Guido van Rossum创建的高级编程语言。"

    result1 = detector.evaluate_single(question1, contexts1, answer1)
    print(f"\n样本1：")
    print(f"  问题: {result1['question']}")
    print(f"  回答: {result1['answer']}")
    print(f"  Faithfulness分数: {result1['faithfulness_score']:.2f}")
    print(f"  状态: {result1['status']}")

    # 测试样本2：部分幻觉
    question2 = "什么是Python?"
    contexts2 = ["Python是一种高级编程语言。"]
    answer2 = "Python是一种由Guido创建的高级编程语言，广泛用于AI开发。"

    result2 = detector.evaluate_single(question2, contexts2, answer2)
    print(f"\n样本2：")
    print(f"  问题: {result2['question']}")
    print(f"  回答: {result2['answer']}")
    print(f"  Faithfulness分数: {result2['faithfulness_score']:.2f}")
    print(f"  状态: {result2['status']}")


def demo_batch_evaluation():
    """演示批量评估"""
    print("\n" + "=" * 60)
    print("演示2：批量Faithfulness评估")
    print("=" * 60)

    detector = FaithfulnessDetector(threshold=0.7)

    # 准备测试样本
    samples = [
        {
            "question": "什么是Python?",
            "contexts": ["Python是一种高级编程语言，由Guido van Rossum创建。"],
            "answer": "Python是一种由Guido van Rossum创建的高级编程语言。",
        },
        {
            "question": "Python有什么特点?",
            "contexts": ["Python语法简洁，易于学习。"],
            "answer": "Python语法简洁，易于学习，广泛用于数据科学。",
        },
        {
            "question": "谁创建了Python?",
            "contexts": ["Python由Guido van Rossum创建。"],
            "answer": "Python由James Gosling创建。",
        },
    ]

    # 批量评估
    batch_result = detector.evaluate_batch(samples)

    print(f"\n批量评估结果：")
    print(f"  总样本数: {batch_result['total_samples']}")
    print(f"  通过样本数: {batch_result['passed_samples']}")
    print(f"  平均Faithfulness: {batch_result['avg_faithfulness']:.2f}")
    print(f"  通过率: {batch_result['pass_rate']:.1%}")

    print(f"\n详细结果：")
    for i, result in enumerate(batch_result["results"], 1):
        status_icon = "✅" if result["passed"] else "❌"
        print(f"  {status_icon} 样本{i}: {result['faithfulness_score']:.2f}")


def demo_rag_integration():
    """演示RAG集成"""
    print("\n" + "=" * 60)
    print("演示3：RAG系统集成Faithfulness检测")
    print("=" * 60)

    rag = RAGWithFaithfulness(threshold=0.7)

    # 测试查询
    questions = ["什么是Python?", "Python有什么特点?"]

    for question in questions:
        print(f"\n问题: {question}")
        result = rag.query(question)

        print(f"  状态: {result['status']}")
        print(f"  Faithfulness分数: {result['faithfulness_score']:.2f}")
        print(f"  回答: {result['answer']}")

        if result["status"] == "rejected":
            print(f"  原始回答: {result['original_answer']}")
            print(f"  拒绝原因: {result['reason']}")


def demo_threshold_tuning():
    """演示阈值调优"""
    print("\n" + "=" * 60)
    print("演示4：Faithfulness阈值调优")
    print("=" * 60)

    # 测试样本
    question = "什么是Python?"
    contexts = ["Python是一种高级编程语言。"]
    answer = "Python是一种高级编程语言，广泛用于数据科学。"

    # 测试不同阈值
    thresholds = [0.5, 0.7, 0.9]

    print(f"\n测试样本：")
    print(f"  问题: {question}")
    print(f"  回答: {answer}")

    for threshold in thresholds:
        detector = FaithfulnessDetector(threshold=threshold)
        result = detector.evaluate_single(question, contexts, answer)

        status_icon = "✅" if result["passed"] else "❌"
        print(
            f"\n  阈值={threshold}: {status_icon} (分数: {result['faithfulness_score']:.2f})"
        )


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("基础Faithfulness检测系统")
    print("=" * 60)

    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ 错误：未设置OPENAI_API_KEY环境变量")
        print("请创建.env文件并设置API密钥")
        return

    try:
        # 运行演示
        demo_single_evaluation()
        demo_batch_evaluation()
        demo_rag_integration()
        demo_threshold_tuning()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

## 代码说明

### 核心类

1. **FaithfulnessDetector**
   - 封装RAGAS Faithfulness评估逻辑
   - 支持单个和批量评估
   - 可配置阈值

2. **RAGWithFaithfulness**
   - 集成Faithfulness检测的完整RAG系统
   - 自动检测并拒绝低分回答

### 关键功能

1. **单个评估**：`evaluate_single()`
   - 评估单个问答对
   - 返回详细结果

2. **批量评估**：`evaluate_batch()`
   - 批量评估多个样本
   - 计算统计指标

3. **RAG集成**：`query()`
   - 完整的RAG流程
   - 自动检测和过滤

---

## 运行示例

### 基础运行

```bash
python faithfulness_detection.py
```

### 自定义阈值

```python
# 严格模式（医疗、法律场景）
detector = FaithfulnessDetector(threshold=0.9)

# 宽松模式（通用问答）
detector = FaithfulnessDetector(threshold=0.6)
```

---

## 输出示例

```
============================================================
演示1：单个样本Faithfulness评估
============================================================

样本1：
  问题: 什么是Python?
  回答: Python是一种由Guido van Rossum创建的高级编程语言。
  Faithfulness分数: 1.00
  状态: passed

样本2：
  问题: 什么是Python?
  回答: Python是一种由Guido创建的高级编程语言，广泛用于AI开发。
  Faithfulness分数: 0.67
  状态: rejected

============================================================
演示2：批量Faithfulness评估
============================================================

批量评估结果：
  总样本数: 3
  通过样本数: 2
  平均Faithfulness: 0.78
  通过率: 66.7%

详细结果：
  ✅ 样本1: 1.00
  ✅ 样本2: 0.75
  ❌ 样本3: 0.00
```

---

## 性能优化

### 1. 批量处理

```python
# 一次评估多个样本，提速3-5倍
samples = [...]  # 100个样本
result = detector.evaluate_batch(samples)
```

### 2. 异步评估

```python
import asyncio

async def async_evaluate(samples):
    tasks = [detector.evaluate_single_async(s) for s in samples]
    return await asyncio.gather(*tasks)
```

### 3. 缓存结果

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_evaluate(question, contexts_tuple, answer):
    return detector.evaluate_single(question, list(contexts_tuple), answer)
```

---

## 常见问题

### Q1: RAGAS评估速度慢怎么办？

**A:** 3种优化方法：
1. 使用更快的模型（gpt-4o-mini而非gpt-4o）
2. 批量评估（一次评估多个样本）
3. 缓存结果（相同输入直接返回缓存）

### Q2: 如何调整阈值？

**A:** 根据场景设置：
- 医疗/法律：0.9+
- 企业知识库：0.7-0.8
- 通用问答：0.6-0.7

### Q3: 评估成本如何？

**A:** 成本估算：
- gpt-4o-mini：~$0.0005/样本
- gpt-4o：~$0.002/样本
- 建议：开发用mini，生产用4o

---

## 扩展建议

1. **添加日志记录**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

2. **集成监控**
   ```python
   from prometheus_client import Histogram
   faithfulness_score = Histogram('faithfulness_score', 'Faithfulness scores')
   ```

3. **持久化结果**
   ```python
   import json
   with open('evaluation_results.json', 'w') as f:
       json.dump(results, f, indent=2)
   ```

---

**记住：这是最基础的Faithfulness检测实现，适合快速原型开发和离线评估。生产环境建议结合NLI验证和声明分解。**
