from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def flexible_format_query(question: str, context: str, format_type: str = "json") -> str:
    """支持多种输出格式"""
    format_instructions = {
        "json": """
            返回 JSON 格式:
            {{
                "answer": "答案",
                "sources": ["来源"]
            }}
        """,
        "xml": """
            返回 XML 格式:
            <response>
                <answer>答案</answer>
                <sources>
                    <source>[来源1]</source>
                    <source>[来源2]</source>
                </sources>
            </response>
        """,
        "markdown": """
            返回 Markdown 格式：

            ## 答案
            [答案内容]

            ## 来源
            - [来源1]
            - [来源2]
        """
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"} if format_type == "json" else None,
        messages=[
            {"role": "system", "content": f"你总是返回 {format_type.upper()} 格式"},
            {"role": "user", "content": f"""
                {format_instructions[format_type]}

                上下文：{context}
                问题：{question}
            """}
        ]
    )

    return response.choices[0].message.content

# 测试不同格式
context = "RAG 是检索增强生成技术..."
question = "什么是 RAG？"

print("=== JSON 格式 ===")
print(flexible_format_query(question, context, "json"))

print("\n=== XML 格式 ===")
print(flexible_format_query(question, context, "xml"))

print("\n=== Markdown 格式 ===")
print(flexible_format_query(question, context, "markdown"))