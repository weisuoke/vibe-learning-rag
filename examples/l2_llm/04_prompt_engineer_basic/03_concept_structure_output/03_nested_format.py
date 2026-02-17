NESTED_FORMAT = """
返回嵌套的 JSON 结构

{{
    "query_analysis": {{
        "original_query": "原始问题",
        "intent": "用户意图",
        "entities": ["实体1", "实体2"]
    }},
    "retrieval_results": [
        {{
            "doc_id": "文档ID",
            "content": "文档内容",
            "relevance_score": 0.0-1.0,
            "metadata": {{
                "source": "来源",
                "date": "日期"
            }}
        }}
    ],
    "generated_answer": {{
        "answer": "答案内容",
        "confidence": 0.0-1.0,
        "reasoning": "推理过程"
    }},
    "recommendations": {{
        "follow_up_questions": ["问题1", "问题2"],
        "related_topics": ["主题1", "主题2"]
    }}
}}
"""

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def complex_rag_query(question: str, documents: list) -> dict:
    """复杂的 RAG 查询 (嵌套结构)"""

    docs_text = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(documents)])

    response = client.chat.completions.create(
        model="gpt-4o",
        # Must be a JSON-serializable dict; a set here will crash during request serialization.
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "你是高级 RAG 系统。只返回合法 JSON（不要 Markdown 代码块，不要额外解释）。",
            },
            {"role": "user", "content": f"""
                {NESTED_FORMAT}

                文档：
                {docs_text}

                问题：
                {question}
            """},
        ],
        temperature=0.1
    )

    content = response.choices[0].message.content or ""
    if not content.strip():
        # JSONDecodeError at (1,1) commonly means the model returned an empty string.
        # Dump the response to make debugging easier.
        raise RuntimeError(
            "Empty model response content; cannot parse JSON. "
            f"response={response.model_dump()}"
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try extracting a JSON object from surrounding text (e.g., accidental preface).
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise RuntimeError(
            "Model did not return valid JSON. Raw content:\n" + content
        )

# 测试
docs = [
    "RAG 是检索增强生成技术...",
    "向量数据库用于存储 embeddings...",
    "LangChain 是 RAG 框架..."
]

result = complex_rag_query("什么是 RAG?", docs)

print("=== 复杂嵌套结构 ===")
print(json.dumps(result, indent=2, ensure_ascii=False))
