from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 定义信息提取格式
EXTRACTION_FORMAT= """
从文档中提取以下信息，返回 JSON 格式：

{{
    "documents": [
        {{
            "doc_id": "文档编号",
            "title": "文档标题",
            "key_points": ["要点1", "要点2", "要点3"],
            "technical_terms": ["术语1", "术语2"],
            "relevance_score": 0.0-1.0
        }}
    ],
    "summary": "所有文档的综合总结",
    "total_docs": 文档总数
}}
"""

def extract_structured_info(documents: list) -> dict:
    """提取结构化信息"""
    docs_text = "\n\n".join([
        f"## 文档 {i+1}\n{doc}"
        for i, doc in enumerate(documents)
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是文档分析专家，只返回合法 JSON（不要 Markdown 代码块，不要额外解释）。"},
            {"role": "user", "content": f"""
                {EXTRACTION_FORMAT}

                文档内容
                {docs_text}
            """}
        ],
        temperature=0.1
    )

    return json.loads(response.choices[0].message.content)

# 测试
docs = [
    "RAG 是检索增强生成技术，核心流程包括检索、注入、生成。",
    "向量数据库用于存储 embeddings，支持语义检索。",
    "LangChain 是流行的 RAG 框架，提供了完整的工具链。"
]

result = extract_structured_info(docs)

print("=== 结构化信息提取 ===")
print(f"文档总数: {result['total_docs']}")
print(f"综合总结: {result['summary']}")
print("\n文档详情:")
for doc in result['documents']:
    print(f"\n文档 {doc['doc_id']}:")
    print(f"  标题: {doc['title']}")
    print(f"  要点: {doc['key_points']}")
    print(f"  术语: {doc['technical_terms']}")
    print(f"  相关度: {doc['relevance_score']}")