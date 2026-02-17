# 实战代码5：Groundedness评估

> RAG Triad评估体系实现

---

## 代码示例

```python
"""
Groundedness评估实战
演示：RAG Triad评估、幻觉检测、质量监控
"""

from openai import OpenAI
import os
from typing import Dict, List
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 基础Groundedness检查 =====

def basic_groundedness_check(answer: str, context: str) -> float:
    """
    基础Groundedness检查
    """
    prompt = f"""
评估答案是否基于上下文（0-1分）：

上下文：
{context}

答案：
{answer}

评分标准：
- 1.0：完全基于上下文
- 0.7-0.9：主要基于上下文
- 0.4-0.6：部分基于上下文
- 0.1-0.3：少量基于上下文
- 0.0：完全不基于上下文

只返回数字分数：
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是评估专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.0

def example_basic_check():
    """基础检查示例"""
    print("=== 基础Groundedness检查 ===\n")
    
    context = "Python是一种解释型语言"
    
    # 测试1：完全基于上下文
    answer1 = "Python是一种解释型语言"
    score1 = basic_groundedness_check(answer1, context)
    print(f"答案1: {answer1}")
    print(f"Groundedness: {score1:.2f}\n")
    
    # 测试2：添加了额外信息
    answer2 = "Python是世界上最好的编程语言"
    score2 = basic_groundedness_check(answer2, context)
    print(f"答案2: {answer2}")
    print(f"Groundedness: {score2:.2f}\n")

# ===== 2. RAG Triad评估器 =====

class RAGTriadEvaluator:
    """RAG Triad评估器"""
    
    def __init__(self):
        self.client = client
    
    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """评估上下文相关性"""
        prompt = f"""
评估上下文对问题的相关性（0-1分）：

问题：{query}
上下文：{context}

只返回数字分数：
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是评估专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.0
    
    def evaluate_groundedness(self, answer: str, context: str) -> float:
        """评估Groundedness"""
        return basic_groundedness_check(answer, context)
    
    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """评估答案相关性"""
        prompt = f"""
评估答案对问题的相关性（0-1分）：

问题：{query}
答案：{answer}

只返回数字分数：
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是评估专家"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.0
    
    def evaluate_full_triad(
        self,
        query: str,
        context: str,
        answer: str
    ) -> Dict:
        """完整RAG Triad评估"""
        context_rel = self.evaluate_context_relevance(query, context)
        groundedness = self.evaluate_groundedness(answer, context)
        answer_rel = self.evaluate_answer_relevance(query, answer)
        
        overall = (context_rel + groundedness + answer_rel) / 3
        
        return {
            "context_relevance": context_rel,
            "groundedness": groundedness,
            "answer_relevance": answer_rel,
            "overall_score": overall,
            "passed": overall >= 0.75
        }

def example_rag_triad():
    """RAG Triad评估示例"""
    print("=== RAG Triad评估 ===\n")
    
    evaluator = RAGTriadEvaluator()
    
    query = "Python有什么特点？"
    context = "Python是一种解释型、面向对象的编程语言。"
    answer = "Python是一种解释型语言，支持面向对象编程。"
    
    result = evaluator.evaluate_full_triad(query, context, answer)
    
    print(f"上下文相关性: {result['context_relevance']:.2f}")
    print(f"Groundedness: {result['groundedness']:.2f}")
    print(f"答案相关性: {result['answer_relevance']:.2f}")
    print(f"综合分数: {result['overall_score']:.2f}")
    print(f"是否通过: {'✅' if result['passed'] else '❌'}")

# ===== 3. 幻觉检测 =====

def detect_hallucination_rules(answer: str, context: str) -> Dict:
    """基于规则的幻觉检测"""
    issues = []
    
    # 规则1：检查数字
    import re
    answer_numbers = set(re.findall(r'\d+', answer))
    context_numbers = set(re.findall(r'\d+', context))
    
    hallucinated_numbers = answer_numbers - context_numbers
    if hallucinated_numbers:
        issues.append({
            "type": "hallucinated_numbers",
            "values": list(hallucinated_numbers)
        })
    
    # 规则2：检查绝对性词汇
    absolute_words = ["总是", "从不", "所有", "没有", "必须", "一定"]
    found_absolutes = [w for w in absolute_words if w in answer]
    if found_absolutes and not any(w in context for w in found_absolutes):
        issues.append({
            "type": "absolute_statements",
            "values": found_absolutes
        })
    
    return {
        "has_hallucination": len(issues) > 0,
        "issues": issues,
        "confidence": 1.0 - (len(issues) * 0.2)
    }

def detect_hallucination_llm(answer: str, context: str) -> Dict:
    """使用LLM进行幻觉检测"""
    prompt = f"""
分析答案中是否存在幻觉（不基于上下文的信息）。

上下文：
{context}

答案：
{answer}

以JSON格式返回：
{{
  "grounded_statements": ["句子1"],
  "hallucinated_statements": ["句子2"],
  "overall_score": 0.0-1.0
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是幻觉检测专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "解析失败"}

def example_hallucination_detection():
    """幻觉检测示例"""
    print("\n=== 幻觉检测 ===\n")
    
    context = "Python是一种编程语言"
    answer = "Python是世界上最好的编程语言，有1000万用户"
    
    # 规则检测
    result_rules = detect_hallucination_rules(answer, context)
    print("规则检测结果:")
    print(f"  有幻觉: {result_rules['has_hallucination']}")
    print(f"  问题: {result_rules['issues']}")
    
    # LLM检测
    result_llm = detect_hallucination_llm(answer, context)
    print("\nLLM检测结果:")
    print(json.dumps(result_llm, ensure_ascii=False, indent=2))

# ===== 4. 质量监控系统 =====

class QualityMonitor:
    """质量监控系统"""
    
    def __init__(self):
        self.metrics = []
        self.evaluator = RAGTriadEvaluator()
    
    def log_generation(
        self,
        session_id: str,
        query: str,
        context: str,
        answer: str
    ):
        """记录生成质量"""
        scores = self.evaluator.evaluate_full_triad(query, context, answer)
        
        record = {
            "session_id": session_id,
            "query": query,
            "answer_length": len(answer),
            "scores": scores
        }
        
        self.metrics.append(record)
        
        if scores["overall_score"] < 0.75:
            print(f"⚠️  低质量生成: {session_id}, 分数: {scores['overall_score']:.2f}")
    
    def get_statistics(self) -> Dict:
        """获取统计数据"""
        if not self.metrics:
            return {"error": "No data"}
        
        avg_groundedness = sum(
            r["scores"]["groundedness"] for r in self.metrics
        ) / len(self.metrics)
        
        avg_relevance = sum(
            r["scores"]["answer_relevance"] for r in self.metrics
        ) / len(self.metrics)
        
        pass_rate = sum(
            1 for r in self.metrics if r["scores"]["overall_score"] >= 0.75
        ) / len(self.metrics)
        
        return {
            "total_generations": len(self.metrics),
            "avg_groundedness": avg_groundedness,
            "avg_relevance": avg_relevance,
            "pass_rate": pass_rate
        }

def example_quality_monitoring():
    """质量监控示例"""
    print("\n=== 质量监控 ===\n")
    
    monitor = QualityMonitor()
    
    # 记录多次生成
    test_cases = [
        {
            "query": "Python有什么特点？",
            "context": "Python是解释型语言",
            "answer": "Python是解释型语言"
        },
        {
            "query": "什么是RAG？",
            "context": "RAG是检索增强生成",
            "answer": "RAG是最好的技术"
        }
    ]
    
    for i, case in enumerate(test_cases):
        monitor.log_generation(
            f"session_{i}",
            case["query"],
            case["context"],
            case["answer"]
        )
    
    # 获取统计
    stats = monitor.get_statistics()
    print("\n质量统计:")
    print(f"  总生成数: {stats['total_generations']}")
    print(f"  平均Groundedness: {stats['avg_groundedness']:.2f}")
    print(f"  平均相关性: {stats['avg_relevance']:.2f}")
    print(f"  通过率: {stats['pass_rate']:.2%}")

# ===== 5. 完整评估流程 =====

class CompleteEvaluationSystem:
    """完整评估系统"""
    
    def __init__(self):
        self.evaluator = RAGTriadEvaluator()
        self.monitor = QualityMonitor()
    
    def evaluate_generation(
        self,
        session_id: str,
        query: str,
        context: str,
        answer: str,
        domain: str = "general"
    ) -> Dict:
        """完整评估流程"""
        # 1. RAG Triad评估
        triad = self.evaluator.evaluate_full_triad(query, context, answer)
        
        # 2. 幻觉检测
        hallucination = detect_hallucination_rules(answer, context)
        
        # 3. 获取阈值
        thresholds = self.get_thresholds(domain)
        
        # 4. 综合判断
        passed = (
            triad["groundedness"] >= thresholds["groundedness"] and
            triad["answer_relevance"] >= thresholds["answer"] and
            not hallucination["has_hallucination"]
        )
        
        # 5. 记录监控
        self.monitor.log_generation(session_id, query, context, answer)
        
        return {
            "passed": passed,
            "triad_scores": triad,
            "hallucination_check": hallucination,
            "thresholds": thresholds,
            "recommendation": "通过" if passed else "需要重新生成"
        }
    
    def get_thresholds(self, domain: str) -> Dict:
        """获取质量阈值"""
        thresholds = {
            "medical": {"groundedness": 0.90, "answer": 0.85},
            "legal": {"groundedness": 0.90, "answer": 0.85},
            "financial": {"groundedness": 0.85, "answer": 0.80},
            "general": {"groundedness": 0.75, "answer": 0.75}
        }
        return thresholds.get(domain, thresholds["general"])

def example_complete_evaluation():
    """完整评估示例"""
    print("\n=== 完整评估流程 ===\n")
    
    system = CompleteEvaluationSystem()
    
    result = system.evaluate_generation(
        "session_123",
        "Python有什么特点？",
        "Python是一种解释型、面向对象的编程语言。",
        "Python是一种解释型语言，支持面向对象编程。",
        domain="general"
    )
    
    print(f"是否通过: {'✅' if result['passed'] else '❌'}")
    print(f"推荐: {result['recommendation']}")
    print(f"\nTriad分数:")
    print(f"  Groundedness: {result['triad_scores']['groundedness']:.2f}")
    print(f"  答案相关性: {result['triad_scores']['answer_relevance']:.2f}")
    print(f"\n幻觉检查:")
    print(f"  有幻觉: {result['hallucination_check']['has_hallucination']}")

# ===== 运行所有示例 =====

if __name__ == "__main__":
    example_basic_check()
    example_rag_triad()
    example_hallucination_detection()
    example_quality_monitoring()
    example_complete_evaluation()
```

---

## 关键要点

1. **RAG Triad**：Context Relevance + Groundedness + Answer Relevance
2. **幻觉检测**：规则检测 + LLM检测
3. **质量监控**：实时追踪质量指标
4. **阈值设置**：根据领域设置合适阈值
5. **完整流程**：评估→检测→监控一体化

---

**版本：** v1.0
**最后更新：** 2026-02-16
