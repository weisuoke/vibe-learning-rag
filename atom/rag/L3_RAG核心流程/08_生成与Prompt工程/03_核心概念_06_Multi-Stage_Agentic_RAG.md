# 核心概念6：Multi-Stage Agentic RAG

> Draft → Critique → Synthesis多阶段验证流程 - 2025-2026生产标准

---

## 概述

Multi-Stage Agentic RAG是2025-2026年RAG系统质量保证的核心技术，通过多阶段验证和自我批评机制，显著降低幻觉率并提升答案质量。

**核心思想：** 不是一次生成就完成，而是通过Draft（草稿）→ Critique（批评）→ Synthesis（综合）的迭代流程持续改进。

**来源：** Stack AI "Complete Guide to Prompt Engineering for RAG" (2025年11月)
https://www.stack-ai.com/blog/prompt-engineering-for-rag-pipelines-the-complete-guide-to-prompt-engineering-for-retrieval-augmented-generation

---

## 1. 三阶段流程

### 1.1 基本架构

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MultiStageRAG:
    """
    多阶段RAG生成器
    """
    
    def __init__(self):
        self.client = client
    
    def generate_multi_stage(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        三阶段生成流程
        """
        # 阶段1：Draft（草稿生成）
        draft = self.stage1_draft(query, context)
        
        # 阶段2：Critique（自我批评）
        critique = self.stage2_critique(draft, context, query)
        
        # 阶段3：Synthesis（综合改进）
        final = self.stage3_synthesis(draft, critique, context, query)
        
        return {
            "draft": draft,
            "critique": critique,
            "final": final
        }
    
    def stage1_draft(self, query: str, context: str) -> str:
        """
        阶段1：生成初稿
        """
        prompt = f"""
基于以下参考资料回答问题：

参考资料：
{context}

问题：{query}

要求：
1. 基于参考资料回答
2. 保持客观准确
3. 添加引用标记

回答：
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def stage2_critique(
        self,
        draft: str,
        context: str,
        query: str
    ) -> str:
        """
        阶段2：自我批评
        """
        critique_prompt = f"""
作为严格的评审专家，评估以下答案的质量。

参考资料：
{context}

问题：{query}

答案：
{draft}

请从以下角度批评：
1. **准确性**：是否完全基于参考资料？有无添加额外信息？
2. **完整性**：是否充分回答了问题？有无遗漏关键信息？
3. **清晰性**：表达是否清晰？逻辑是否连贯？
4. **引用**：引用是否准确？是否标注了所有关键信息的来源？

批评（请具体指出问题）：
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是严格的评审专家"},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def stage3_synthesis(
        self,
        draft: str,
        critique: str,
        context: str,
        query: str
    ) -> str:
        """
        阶段3：综合改进
        """
        synthesis_prompt = f"""
基于批评意见改进答案。

原答案：
{draft}

批评意见：
{critique}

参考资料：
{context}

问题：{query}

要求：
1. 针对批评意见进行改进
2. 保持基于参考资料
3. 确保引用准确
4. 提升清晰度和完整性

改进后的答案：
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content

# 使用示例
rag = MultiStageRAG()

query = "什么是RAG？"
context = """
RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
RAG通过检索相关文档来增强LLM的生成能力。
RAG可以减少幻觉，提高答案的准确性。
"""

result = rag.generate_multi_stage(query, context)

print("=== 阶段1：初稿 ===")
print(result["draft"])

print("\n=== 阶段2：批评 ===")
print(result["critique"])

print("\n=== 阶段3：最终答案 ===")
print(result["final"])
```

---

## 2. 高级多阶段模式

### 2.1 五阶段流程

```python
class AdvancedMultiStageRAG:
    """
    高级五阶段RAG
    """
    
    def generate_five_stage(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        五阶段生成流程
        """
        # 阶段1：Query Analysis（查询分析）
        analysis = self.stage1_query_analysis(query)
        
        # 阶段2：Draft Generation（草稿生成）
        draft = self.stage2_draft(query, context, analysis)
        
        # 阶段3：Fact Checking（事实检查）
        fact_check = self.stage3_fact_check(draft, context)
        
        # 阶段4：Critique（批评）
        critique = self.stage4_critique(draft, fact_check, query)
        
        # 阶段5：Final Synthesis（最终综合）
        final = self.stage5_synthesis(draft, fact_check, critique, context, query)
        
        return {
            "analysis": analysis,
            "draft": draft,
            "fact_check": fact_check,
            "critique": critique,
            "final": final
        }
    
    def stage1_query_analysis(self, query: str) -> dict:
        """
        阶段1：查询分析
        """
        analysis_prompt = f"""
分析以下问题的关键要素：

问题：{query}

请识别：
1. 核心问题是什么？
2. 需要什么类型的信息？
3. 回答的关键要点是什么？

以JSON格式返回：
{{
  "core_question": "...",
  "info_type": "...",
  "key_points": ["...", "..."]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是查询分析专家"},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"error": "解析失败"}
    
    def stage2_draft(
        self,
        query: str,
        context: str,
        analysis: dict
    ) -> str:
        """
        阶段2：基于分析生成草稿
        """
        key_points = analysis.get("key_points", [])
        key_points_str = "\n".join([f"- {p}" for p in key_points])
        
        prompt = f"""
基于查询分析和参考资料回答问题。

查询分析：
核心问题：{analysis.get("core_question", query)}
关键要点：
{key_points_str}

参考资料：
{context}

问题：{query}

回答：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def stage3_fact_check(self, draft: str, context: str) -> dict:
        """
        阶段3：事实检查
        """
        check_prompt = f"""
逐句检查答案中的事实是否基于参考资料。

参考资料：
{context}

答案：
{draft}

对每个事实陈述进行检查，返回JSON格式：
{{
  "facts": [
    {{
      "statement": "...",
      "grounded": true/false,
      "source": "..."
    }}
  ]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是事实检查专家"},
                {"role": "user", "content": check_prompt}
            ],
            temperature=0
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"error": "解析失败"}
    
    def stage4_critique(
        self,
        draft: str,
        fact_check: dict,
        query: str
    ) -> str:
        """
        阶段4：综合批评
        """
        ungrounded = [
            f["statement"] for f in fact_check.get("facts", [])
            if not f.get("grounded", True)
        ]
        
        critique_prompt = f"""
基于事实检查结果，批评答案的质量。

答案：
{draft}

问题：{query}

事实检查发现的问题：
{chr(10).join([f"- {s}" for s in ungrounded]) if ungrounded else "无"}

请提供改进建议：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是评审专家"},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def stage5_synthesis(
        self,
        draft: str,
        fact_check: dict,
        critique: str,
        context: str,
        query: str
    ) -> str:
        """
        阶段5：最终综合
        """
        synthesis_prompt = f"""
基于所有反馈生成最终答案。

原答案：
{draft}

事实检查结果：
{fact_check}

批评意见：
{critique}

参考资料：
{context}

问题：{query}

要求：
1. 修正所有不基于参考资料的内容
2. 实施批评意见中的改进建议
3. 确保答案完整、准确、清晰
4. 添加准确的引用标记

最终答案：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
```

---

## 3. Chain-of-Thought集成

### 3.1 CoT + Multi-Stage

```python
class CoTMultiStageRAG:
    """
    结合Chain-of-Thought的多阶段RAG
    """
    
    def generate_with_cot(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        CoT + Multi-Stage生成
        """
        # 阶段1：CoT推理
        reasoning = self.stage1_cot_reasoning(query, context)
        
        # 阶段2：基于推理生成答案
        draft = self.stage2_draft_with_reasoning(query, context, reasoning)
        
        # 阶段3：验证推理链
        verification = self.stage3_verify_reasoning(reasoning, context)
        
        # 阶段4：最终答案
        final = self.stage4_final_answer(draft, verification, context, query)
        
        return {
            "reasoning": reasoning,
            "draft": draft,
            "verification": verification,
            "final": final
        }
    
    def stage1_cot_reasoning(self, query: str, context: str) -> str:
        """
        阶段1：Chain-of-Thought推理
        """
        cot_prompt = f"""
使用Chain-of-Thought方法分析问题。

参考资料：
{context}

问题：{query}

请按以下步骤思考：
1. 问题的核心是什么？
2. 参考资料中有哪些相关信息？
3. 如何组织这些信息来回答问题？
4. 推理链是什么？

思考过程：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是推理专家"},
                {"role": "user", "content": cot_prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    def stage2_draft_with_reasoning(
        self,
        query: str,
        context: str,
        reasoning: str
    ) -> str:
        """
        阶段2：基于推理生成答案
        """
        draft_prompt = f"""
基于推理过程生成答案。

推理过程：
{reasoning}

参考资料：
{context}

问题：{query}

答案：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": draft_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def stage3_verify_reasoning(
        self,
        reasoning: str,
        context: str
    ) -> dict:
        """
        阶段3：验证推理链
        """
        verify_prompt = f"""
验证推理链的每一步是否基于参考资料。

推理过程：
{reasoning}

参考资料：
{context}

验证结果（JSON格式）：
{{
  "steps": [
    {{
      "step": "...",
      "valid": true/false,
      "reason": "..."
    }}
  ]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是验证专家"},
                {"role": "user", "content": verify_prompt}
            ],
            temperature=0
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"error": "解析失败"}
    
    def stage4_final_answer(
        self,
        draft: str,
        verification: dict,
        context: str,
        query: str
    ) -> str:
        """
        阶段4：基于验证结果生成最终答案
        """
        invalid_steps = [
            s for s in verification.get("steps", [])
            if not s.get("valid", True)
        ]
        
        final_prompt = f"""
基于验证结果生成最终答案。

原答案：
{draft}

验证发现的问题：
{chr(10).join([f"- {s['step']}: {s['reason']}" for s in invalid_steps]) if invalid_steps else "无"}

参考资料：
{context}

问题：{query}

最终答案（修正所有问题）：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
```

---

## 4. 自适应阶段数

### 4.1 动态阶段调整

```python
class AdaptiveMultiStageRAG:
    """
    自适应多阶段RAG
    """
    
    def generate_adaptive(
        self,
        query: str,
        context: str,
        max_iterations: int = 5,
        quality_threshold: float = 0.85
    ) -> dict:
        """
        自适应多阶段生成
        """
        iterations = []
        current_answer = None
        
        for i in range(max_iterations):
            # 生成或改进答案
            if i == 0:
                current_answer = self.initial_draft(query, context)
            else:
                current_answer = self.improve_answer(
                    current_answer,
                    iterations[-1]["critique"],
                    context,
                    query
                )
            
            # 评估质量
            quality = self.evaluate_quality(current_answer, context, query)
            
            # 生成批评
            critique = self.generate_critique(current_answer, context, query)
            
            iterations.append({
                "iteration": i + 1,
                "answer": current_answer,
                "quality": quality,
                "critique": critique
            })
            
            # 如果质量达标，提前结束
            if quality >= quality_threshold:
                break
        
        return {
            "final_answer": current_answer,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "final_quality": iterations[-1]["quality"]
        }
    
    def initial_draft(self, query: str, context: str) -> str:
        """
        初始草稿
        """
        prompt = f"""
参考资料：{context}

问题：{query}

回答：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def improve_answer(
        self,
        current: str,
        critique: str,
        context: str,
        query: str
    ) -> str:
        """
        改进答案
        """
        improve_prompt = f"""
基于批评改进答案。

当前答案：
{current}

批评：
{critique}

参考资料：
{context}

问题：{query}

改进后的答案：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": improve_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def evaluate_quality(
        self,
        answer: str,
        context: str,
        query: str
    ) -> float:
        """
        评估质量（0-1分）
        """
        eval_prompt = f"""
评估答案质量（0-1分）：

参考资料：{context}
问题：{query}
答案：{answer}

综合评分（只返回数字）：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是评估专家"},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5
    
    def generate_critique(
        self,
        answer: str,
        context: str,
        query: str
    ) -> str:
        """
        生成批评
        """
        critique_prompt = f"""
批评答案的不足：

参考资料：{context}
问题：{query}
答案：{answer}

批评：
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是评审专家"},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
```

---

## 5. 生产环境优化

### 5.1 并行化处理

```python
import asyncio
from typing import List

class ParallelMultiStageRAG:
    """
    并行化多阶段RAG
    """
    
    async def generate_parallel(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        并行生成多个草稿，选择最佳
        """
        # 并行生成3个草稿
        drafts = await asyncio.gather(
            self.async_draft(query, context, temperature=0.1),
            self.async_draft(query, context, temperature=0.2),
            self.async_draft(query, context, temperature=0.3)
        )
        
        # 并行评估所有草稿
        evaluations = await asyncio.gather(*[
            self.async_evaluate(draft, context, query)
            for draft in drafts
        ])
        
        # 选择最佳草稿
        best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i])
        best_draft = drafts[best_idx]
        
        # 对最佳草稿进行批评和改进
        critique = await self.async_critique(best_draft, context, query)
        final = await self.async_improve(best_draft, critique, context, query)
        
        return {
            "drafts": drafts,
            "evaluations": evaluations,
            "best_draft": best_draft,
            "critique": critique,
            "final": final
        }
    
    async def async_draft(
        self,
        query: str,
        context: str,
        temperature: float
    ) -> str:
        """
        异步生成草稿
        """
        # 简化实现
        return f"Draft with temperature {temperature}"
    
    async def async_evaluate(
        self,
        draft: str,
        context: str,
        query: str
    ) -> float:
        """
        异步评估
        """
        # 简化实现
        return 0.85
    
    async def async_critique(
        self,
        draft: str,
        context: str,
        query: str
    ) -> str:
        """
        异步批评
        """
        # 简化实现
        return "Critique"
    
    async def async_improve(
        self,
        draft: str,
        critique: str,
        context: str,
        query: str
    ) -> str:
        """
        异步改进
        """
        # 简化实现
        return "Improved answer"
```

---

## 6. 效果对比

### 6.1 2025-2026实验数据

**来源：** Stack AI Multi-Stage RAG Study (2025年11月)

| 方法 | Groundedness | Answer Quality | Hallucination Rate |
|------|--------------|----------------|-------------------|
| Single-Stage | 0.72 | 0.68 | 28% |
| 3-Stage | 0.87 | 0.84 | 12% |
| 5-Stage | 0.91 | 0.89 | 7% |
| Adaptive | 0.89 | 0.87 | 9% |

**结论：**
- 3-Stage相比Single-Stage：Groundedness提升21%，幻觉率降低57%
- 5-Stage效果最佳，但成本较高
- Adaptive在质量和成本间取得平衡

---

## 总结

### 核心原则

1. **迭代改进**：不是一次生成，而是持续改进
2. **自我批评**：通过批评发现问题
3. **事实验证**：每个阶段都验证事实
4. **质量优先**：以质量为导向，不是速度
5. **自适应**：根据质量动态调整阶段数

### 2025-2026标准配置

```python
# Multi-Stage RAG生产配置
MULTI_STAGE_CONFIG_2026 = {
    "stages": 3,  # 3阶段（Draft → Critique → Synthesis）
    "quality_threshold": 0.85,
    "max_iterations": 5,
    "parallel_drafts": 3,
    "cot_integration": True,
    "fact_checking": True
}
```

---

**版本：** v1.0 (2025-2026最新标准)
**最后更新：** 2026-02-16
**参考来源：**
- Stack AI "Prompt Engineering for RAG" (2025-11)
- https://www.stack-ai.com/blog/prompt-engineering-for-rag-pipelines-the-complete-guide-to-prompt-engineering-for-retrieval-augmented-generation
