from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

class MultiLayerValidator:
    """多层验证器"""

    def __init__(self):
        self.client = OpenAI()

    def layer1_grounding_check(self, answer: str, context: str) -> dict:
        """第一层：事实基础检查"""
        print("=== 第1层：事实基础检查 ===")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是事实核查专家"},
                {"role": "user", "content": f"""
                    检查答案中的每个事实是否都能在上下文中找到。
                 
                    答案：{answer}

                    上下文：{context}

                    返回 JSON:
                    {{
                        "is_grounded": true/false,
                        "unsupported_claims": ["不支持的声明1", "声明2"],
                        "confidence": 0.0-1.0
                    }}
                """}
            ],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        print(f"  事实基础: {'✅ 通过' if result['is_grounded'] else '❌ 失败'}")
        if result['unsupported_claims']:
            print(f" 不支持的声明：{result['unsupported_claims']}")

        return result
    
    def layer2_consistency_check(self, answer: str, context: str) -> dict:
        """第2层：一致性检查"""

        print("\n=== 第2层：一致性检查 ===")

        # Use Chat Completions API since we send role-based messages and ask for JSON output.
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是逻辑一致性专家"},
                {"role": "user", "content": f"""
                    检查答案是否与上下文一致，是否有逻辑矛盾。
                 
                    答案: {answer}

                    上下文：{context}

                    返回 JSON:

                    {{
                        "is_consistent": true/false,
                        "contradictions": ["矛盾点1", "矛盾点2"],
                        "confidence": 0.0-1.0
                    }}
                """}
            ],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        print(f"  一致性: {'✅ 通过' if result['is_consistent'] else '❌ 失败'}")
        if result["contradictions"]:
            print(f"  矛盾点: {result['contradictions']}")

        return result
    
    def layer3_comleteness_check(self, answer: str, question: str, context: str) -> dict:
        """第三层：完整性检查"""
        print("\n=== 第3层：完整性检查 ===")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是答案完整性专家"},
                {"role": "user", "content": f"""
                    检查答案是否完整回答了问题。
                 
                    问题：{question}

                    答案：{answer}

                    上下文：{context}

                    返回 JSON：
                    {{
                        "is_complete": true/false,
                        "missing_aspects": ["缺失方面1", "方面2"],
                        "confidence": 0.0-1.0
                    }}
                """}
            ],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        print(f"  完整性: {'✅ 通过' if result['is_complete'] else '❌ 失败'}")
        if result['missing_aspects']:
            print(f"  缺失方面: {result['missing_aspects']}")

        return result
    
    def validate(self, answer: str, question: str, context: str) -> dict:
        """执行多层验证"""
        print("=== 开始多层验证 ===\n")

        # 第一层：事实基础
        layer1 = self.layer1_grounding_check(answer, context)

        # 第二层：一致性
        layer2 = self.layer2_consistency_check(answer, context)

        # 第三层：完整性
        layer3 = self.layer3_comleteness_check(answer, question, context)

        # 综合评估
        all_passed = (
            layer1['is_grounded'] and
            layer2['is_consistent'] and
            layer3['is_complete']
        )

        overall_confidence = (
            layer1['confidence'] +
            layer2['confidence'] + 
            layer3['confidence']
        ) / 3

        print(f"\n=== 验证结果 ===")
        print(f"总体通过: {'✅ 是' if all_passed else '❌ 否'}")
        print(f"综合置信度: {overall_confidence:.2f}")

        return {
            "validation_passed": all_passed,
            "overall_confidence": overall_confidence,
            "layer1_grounding": layer1,
            "layer2_consistency": layer2,
            "layer3_completeness": layer3
        }
    
# 测试
validator = MultiLayerValidator()

answer = "RAG 是检索增强生成技术，结合检索和生成两个过程，能够访问最新信息。"
question = "什么是 RAG？"
context = """
文档1：RAG（检索增强生成）是一种结合检索和生成的技术。
文档2：RAG 的核心优势是能够访问最新信息和私有数据。
"""

validation_result = validator.validate(answer, question, context)
