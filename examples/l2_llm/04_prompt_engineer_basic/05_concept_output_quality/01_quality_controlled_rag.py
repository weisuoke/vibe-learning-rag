from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# 质量控制 Prompt
QUALITY_CONTROLLED_PROMPT = """
任务：基于检索到的文档回答用户问题

上下文：
{context}

问题：{question}

约束条件：
- 答案必须完全基于上下文，不能编造信息
- 如果上下文不足，必须明确说明
- 答案长度: 50-200 字
- 必须标注信息来源

验证清单（在回答前自我检查）：
- [ ] 答案的每个事实都能在上下文中找到？
- [ ] 是否包含任何推测或猜测？
- [ ] 是否标注了来源？
- [ ] 长度是否符合要求？

返回格式：
{{
  "answer": "基于上下文的答案",
  "sources": ["来源1", "来源2"],
  "confidence": 0.0-1.0,
  "has_sufficient_context": true/false,
  "validation_passed": true/false,
  "validation_notes": "验证说明"
}}

如果验证失败，返回：
{{
  "error": "验证失败的原因",
  "suggestion": "需要什么额外信息",
  "validation_passed": false
}}
"""

def quality_controlled_rag(question: str, context: str) -> dict:
    """带质量控制的 RAG 查询"""

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是严格的 RAG 助手，总是返回 JSON 格式"},
            {"role": "user", "content": QUALITY_CONTROLLED_PROMPT.format(
                context=context,
                question=question
            )}
        ],
        temperature=0.1 # 低温度确保稳定
    )

    result = json.loads(response.choices[0].message.content)

        # 后处理验证
    if result.get("validation_passed"):
        print("✅ 质量验证通过")
        return result
    else:
        print(f"❌ 质量验证失败: {result.get('error')}")
        print(f"💡 建议: {result.get('suggestion')}")
        return None
    
# 测试
context = """
文档1：RAG（检索增强生成）是一种结合检索和生成的技术。
文档2：RAG 的核心优势是能够访问最新信息和私有数据。
文档3：典型应用包括知识库问答、文档分析、智能客服。
"""

result = quality_controlled_rag("什么是RAG?", context)

if result:
    print(f"\n答案: {result['answer']}")
    print(f"来源: {result['sources']}")
    print(f"置信度: {result['confidence']}")
    print(f"上下文充足: {result['has_sufficient_context']}")
    print(f"验证说明: {result['validation_notes']}")