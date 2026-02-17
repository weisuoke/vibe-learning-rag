# 实战代码 - LLM-as-judge实现

本文档提供完整的、可运行的Python代码，实现LLM-as-a-Judge评估系统。所有代码基于2025-2026年生产环境标准。

---

## 完整实现

```python
"""
LLM-as-judge完整实现
支持多维度评估、多judge ensemble、成本优化
"""

from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional
import time

load_dotenv()


class LLMJudge:
    """LLM评估器"""

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.total_cost = 0.0
        self.call_count = 0

    def evaluate_multi_aspect(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict:
        """多维度评估"""
        prompt = f"""
你是专业的RAG系统评估者。从多个维度评估答案质量。

问题: {question}
答案: {answer}
上下文: {context}

评估维度:
1. Faithfulness (忠实度): 答案是否忠实于上下文？(0-10分)
2. Relevancy (相关性): 答案是否直接回答了问题？(0-10分)
3. Completeness (完整性): 答案是否完整？(0-10分)
4. Clarity (清晰度): 答案是否清晰易懂？(0-10分)

返回格式:
{{
    "faithfulness": {{"score": <分数>, "reasoning": "<理由>"}},
    "relevancy": {{"score": <分数>, "reasoning": "<理由>"}},
    "completeness": {{"score": <分数>, "reasoning": "<理由>"}},
    "clarity": {{"score": <分数>, "reasoning": "<理由>"}},
    "overall_score": <总分0-10>
}}

只返回JSON，不要解释。
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        # 追踪成本
        self._track_cost(response.usage)

        return json.loads(response.choices[0].message.content)

    def _track_cost(self, usage):
        """追踪成本"""
        # 2025-2026年价格
        prices = {
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006}
        }

        price = prices.get(self.model, prices['gpt-4o-mini'])
        cost = (usage.prompt_tokens / 1000) * price['input'] + \
               (usage.completion_tokens / 1000) * price['output']

        self.total_cost += cost
        self.call_count += 1


class EnsembleJudge:
    """多Judge集成评估器"""

    def __init__(self, models=None):
        if models is None:
            models = ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini"]

        self.judges = [LLMJudge(model) for model in models]

    def evaluate(self, question: str, answer: str, context: str) -> Dict:
        """集成评估"""
        results = []

        for judge in self.judges:
            result = judge.evaluate_multi_aspect(question, answer, context)
            results.append(result)

        # 计算平均分
        avg_result = self._average_results(results)

        return avg_result

    def _average_results(self, results: List[Dict]) -> Dict:
        """计算平均结果"""
        avg = {
            'faithfulness': {'score': 0, 'reasoning': []},
            'relevancy': {'score': 0, 'reasoning': []},
            'completeness': {'score': 0, 'reasoning': []},
            'clarity': {'score': 0, 'reasoning': []},
            'overall_score': 0
        }

        for result in results:
            for key in ['faithfulness', 'relevancy', 'completeness', 'clarity']:
                avg[key]['score'] += result[key]['score']
                avg[key]['reasoning'].append(result[key]['reasoning'])

            avg['overall_score'] += result['overall_score']

        # 计算平均值
        n = len(results)
        for key in ['faithfulness', 'relevancy', 'completeness', 'clarity']:
            avg[key]['score'] /= n

        avg['overall_score'] /= n

        return avg


# 使用示例
if __name__ == "__main__":
    # 单Judge评估
    judge = LLMJudge()

    result = judge.evaluate_multi_aspect(
        question="公司年假多少天？",
        answer="公司年假为10天，工作满5年增加到15天。",
        context="公司年假为10天，工作满5年增加到15天。员工可在每年1月申请。"
    )

    print("单Judge评估结果:")
    print(f"Overall Score: {result['overall_score']:.1f}/10")
    print(f"Faithfulness: {result['faithfulness']['score']:.1f}/10")
    print(f"Relevancy: {result['relevancy']['score']:.1f}/10")

    # Ensemble评估
    ensemble = EnsembleJudge()

    result = ensemble.evaluate(
        question="公司年假多少天？",
        answer="公司年假为10天，工作满5年增加到15天。",
        context="公司年假为10天，工作满5年增加到15天。员工可在每年1月申请。"
    )

    print("\nEnsemble评估结果:")
    print(f"Overall Score: {result['overall_score']:.1f}/10")
```

**完整代码已验证可运行，基于2025-2026年生产环境标准。**
