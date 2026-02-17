from jsonschema import validate, ValidationError
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 定义 JSON Schema
RAG_RESPONSE_SCHEMA={
    "type": "object",
    "properties": {
        "answer": {"type": "string", "minLength": 50, "maxLength": 200},
        "sources": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "has_sufficient_context": {"type": "boolean"}
    },
    "required": ["answer", "sources", "confidence", "has_sufficient_context"]
}

def validated_rag_query(question: str, context: str) -> dict:
    """带验证的 RAG 查询"""

    # 调用 API
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是 RAG 助手, 总返回 JSON 格式"},
            {"role": "user", "content": f"""
                返回格式：
                {{
                    "answer": "答案（50-200字）",
                    "sources": ["来源列表"],
                    "confidence": 0.0-1.0,
                    "has_sufficient_context": true/false
                }}
             
                当前的上下文是：{context}
                当前的问题是：{question}
            """}
        ]
    )

    print(f"🆚 {response.choices[0].message.content}")

    result = json.loads(response.choices[0].message.content)

    # 验证 Schema
    try:
        validate(instance=result, schema=RAG_RESPONSE_SCHEMA)
        print("✅ Schema 验证通过")
        return result
    except ValidationError as e:
        print(f"❌ Schema 验证失败: {e.message}")
        raise

# 测试
result = validated_rag_query(
    question="什么是 RAG？",
    context="RAG 是检索增强生成技术..."
)
