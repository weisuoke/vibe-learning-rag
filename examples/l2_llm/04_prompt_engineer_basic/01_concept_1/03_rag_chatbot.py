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

load_dotenv()

class RAGChatbot:
    """带角色设定的 RAG 聊天机器人"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.client = OpenAI()

    def chat(self, user_message: str, context: str = "") -> str:
        """多轮对话"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加历史对话
        messages.extend(self.conversation_history)

        # 添加当前消息
        if context:
            user_content = f"上下文：\n{context}\n\n问题：{user_message}"
        else:
            user_content = user_message

        messages.append({"role": "user", "content": user_content})

        # 调用 API
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1
        )

        assistant_message = response.choices[0].message.content

        # 保存对话历史
        self.conversation_history.append({"role": "user", "content": user_content})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

# 使用示例
chatbot = RAGChatbot(TECH_DOC_ASSISTANT)

# 第一轮对话
response1 = chatbot.chat("什么事RAG?", context="RAG 是检索增强生成...")
print(f"回答1：{response1}")

# 第二轮对话
response2 = chatbot.chat("它有什么优势？")
print(f"回答2：{response2}")

# 第三列对话
response3 = chatbot.chat("给我一个示例代码")
print(f"回答3: {response3}")