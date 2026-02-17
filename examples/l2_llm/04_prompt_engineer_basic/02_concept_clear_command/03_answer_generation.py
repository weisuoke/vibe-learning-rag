ANSWER_GENERATION_PROMPT="""
任务：基于检索到的文档回答用户问题

步骤：
1. 仔细阅读所有文档片段
2. 识别与问题直接相关的信息
3. 综合信息形成答案
4. 验证答案的准确性和完整性

答案要求:
- 长度: 50-200字
- 只基于文档内容，不编造信息
- 如果文档不足，明确说明
- 标注信息来源

当前输入：
- 问题：{question}
- 文档：{documents}

输出格式：
{{
    "answer": "基于文档的答案",
    "sources": ["文档1", "文档2"],
    "confidence": 0.0-1.0,
    "has_sufficient_context": true/false
}}

如果验证失败，必须返回JSON如下：
{{
  "error": "验证失败的原因",
  "suggestion": "需要什么额外信息"
}}

验证清单:
- [ ] 答案的每个事实都能在文档中找到？
- [ ] 答案长度在 50-200 字之间？
- [ ] 是否标注了来源？
- [ ] 如果文档不足，是否明确说明？
"""

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_answer(question: str, documents: list) -> dict:
    """生成答案"""
    # 构建文档上下文
    docs_text = "\n\n".join([
        f"## 文档 {i+1}\n{doc['content']}"
        for i, doc in enumerate(documents)
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是专业的 RAG 助手"},
            {"role": "user", "content": ANSWER_GENERATION_PROMPT.format(
                question=question,
                documents=docs_text
            )}
        ],
        temperature=0.1
    )

    print(f"{response.choices[0].message.content}")

    return json.loads(response.choices[0].message.content)

# 测试
docs = [
    {"content": "RAG 是检索增强生成技术，结合检索和生成两个过程。"},
    {"content": "RAG 的核心优势是能够访问最新信息和私有数据。"}
]

result = generate_answer("什么是 CAG？", docs)
if "error" in result:
    # Example prompt can intentionally return an error object when context is insufficient.
    print(f"错误: {result.get('error')}")
    print(f"建议: {result.get('suggestion')}")
else:
    print(f"答案: {result['answer']}")
    print(f"来源: {result['sources']}")
    print(f"置信度: {result['confidence']}")
