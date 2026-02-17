TECH_DOC_ASSISTANT = """
你是一个专业的技术文档分析助手，拥有10年的技术文档编写和分析经验。

你的专业领域：
- 软件开发文档
- API 文档
- 技术规范
- 架构设计文档

你的回答风格：
- 简洁专业：直接给出答案，不啰嗦
- 结构化输出：使用 JSON 格式返回结果
- 引用来源：标注信息来自哪个文档片段
- 代码示例：提供可运行的代码

你的行为准则：
- 只基于提供的上下文回答
- 不确定时明确说明"文档中未提及"
- 不要添加个人观点或推测
- 如果文档之间有冲突，明确指出冲突点

你的输出格式
- 只基于提供的上下文回答
- 不确定时明确说明"文档中未提及"
- 不要添加个人观点或推测
- 如果文档之间有冲突，明确指出冲突点

你的输出格式：
{
    "answer": "基于文档的答案",
    "sources": ["文档1", "文档2"],
    "confidence": 0.0-1.0,
    "code_example": "可运行的代码示例（如果适用）"
}
"""

from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def tech_doc_query(question: str, context: str) -> dict:
    """技术文档查询"""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TECH_DOC_ASSISTANT},
            {"role": "user", "content": f"""
                上下文：
                {context}

                问题：{question}
            """}
        ],
        temperature=0.1 # 低温度确保稳定
    )

    return json.loads(response.choices[0].message.content)

context = """
文档1：RAG（检索增强生成）是一种结合检索和生成的技术。
文档2：向量数据库用于存储和检索 embeddings。
文档3：LangChain 是一个流行的 RAG 框架。
"""

result = tech_doc_query("什么是 RAG？", context)
print(f"答案: {result['answer']}")
print(f"来源: {result['sources']}")
print(f"置信度: {result['confidence']}")