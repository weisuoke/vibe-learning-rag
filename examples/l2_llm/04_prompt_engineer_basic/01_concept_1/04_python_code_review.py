from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 定义角色
ROLE_PROMPT = """
你是一个专业的 Python 开发专家，擅长代码审查和优化。

你的回答风格：
- 直接指出问题
- 提供改进建议
- 给出优化后的代码

你的约束：
- 只关注代码质量、性能和安全
- 不讨论业务逻辑
"""

# 使用角色
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": """
            审查这段代码：
            ```python
            def get_user(id):
                query = f"SELECT * FROM users WHERE id = {id}"
                return db.execute(query)
            ```
        """}
    ]
)

print(response.choices[0].message.content)