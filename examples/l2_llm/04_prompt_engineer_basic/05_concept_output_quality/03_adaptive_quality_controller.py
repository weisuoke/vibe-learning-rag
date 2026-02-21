from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

class AdaptiveQualityController:
    """自适应质量控制"""

    def __init__(self):
        self.client = OpenAI()

    def assess_risk_level(self, question: str) -> str:
        """评估问题的风险级别"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是风险评估专家"},
                {"role": "user", "content": f"""
                    评估这个问题的风险级别。
                 
                    问题：{question}

                    风险级别定义：
                    - LOW：一般性问题，错误影响小
                    - MEDIUM：重要问题，错误有一定影响
                    - HIGH：关键问题，错误影响大（如医疗、法律、金融）

                    返回 JSON：
                    {{
                        "risk_level": "LOW/MEDIUM/HIGH",
                        "reason": "评估理由"
                    }}
                """}
            ]
        )

        result = json.loads(response.choices[0].message.content)
        print(f"=== 风险评估 ===")
        print(f"风险级别: {result['risk_level']}")
        print(f"理由: {result['reason']}\n")

        return result['risk_level']
    
    def get_quality_strategy(self, risk_level: str) -> dict:
        """根据风险级别获取质量策略"""

        strategies = {
            "LOW": {
                "confidence_threshold": 0.6,
                "validation_layers": 1,
                "require_sources": False,
                "allow_speculation": True
            },
            "MEDIUM": {
                "confidence_threshold": 0.75,
                "validation_layers": 2,
                "require_sources": True,
                "allow_speculation": False
            },
            "HIGH": {
                "confidence_threshold": 0.9,
                "validation_layers": 3,
                "require_sources": True,
                "allow_speculation": False,
                "require_disclaimer": True
            }
        }
        
        return strategies[risk_level]
    
    def adaptive_rag_query(self, question: str, context: str) -> dict:
        """自适应质量控制的 RAG 查询"""

        # 步骤 1：评估风险
        risk_level = self.assess_risk_level(question)

        # 步骤 2：获取质量策略
        strategy = self.get_quality_strategy(risk_level)
        print(f"=== 质量策略 ===")
        print(f"置信度阈值: {strategy['confidence_threshold']}")
        print(f"验证层数：{strategy['validation_layers']}")
        print(f"要求来源：{strategy['require_sources']}")
        print(f"允许推测：{strategy['allow_speculation']}\n")

        # 步骤 3：构建 Prompt
        prompt = f"""
            任务：基于上下文回答

            上下文：{context}

            质量要求（风险级别：{risk_level}）:
            - 置信度阈值：{strategy['confidence_threshold']}
            - 必须标注来源：{strategy['require_sources']}
            - 允许推测：{strategy['allow_speculation']}

            约束：
            - 如果置信度低于 {strategy['confidence_threshold']}，明确说明不确定
            - {'必须标注信息来源' if strategy['require_sources'] else ''}
            - {'不要添加推测或猜测' if not strategy['allow_speculation'] else ''}

            返回 JSON：
            {{
                "answer": "答案",
                "confidence": 0.0-1.0,
                "sources": ["来源"],
                "risk_level": "{risk_level}",
                "meets_threshold": true/false
            }}
        """

        # 步骤 4： 生成答案
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是自适应 RAG 助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)

        # 步骤 5：验证
        if not result["meets_threshold"]:
            print(f"⚠️  警告：置信度 {result['confidence']} 低于阈值 {strategy['confidence_threshold']}")

            if risk_level == "HIGH":
                print("❌ 高风险问题，拒绝回答")
                return None
            
        return result
    
# 测试不同风险级别的问题
controller = AdaptiveQualityController()

# 低风险问题
print("=" * 50)
print("测试 1：低风险问题")
print("=" * 50)

result1 = controller.adaptive_rag_query(
    question="什么是 RAG？",
    context="RAG 是检索增强生成技术..."
)

if result1:
    print(f"\n答案: {result1['answer']}")
    print(f"置信度: {result1['confidence']}")

# 高风险问题
print("\n" + "=" * 50)
print("测试 2：高风险问题")
print("=" * 50)
result2 = controller.adaptive_rag_query(
    question="这个药物的副作用是什么？",
    context="某些研究表明..."
)

if result2:
    print(f"\n答案: {result2['answer']}")
    print(f"置信度: {result2['confidence']}")
else:
    print("\n❌ 由于风险级别高且置信度不足，拒绝回答")

