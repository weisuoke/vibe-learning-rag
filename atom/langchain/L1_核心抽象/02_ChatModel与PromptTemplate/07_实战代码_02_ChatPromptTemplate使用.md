# 实战代码 02: ChatPromptTemplate 使用

> **本文目标**: 通过可运行的代码示例，掌握 ChatPromptTemplate 的实战应用

---

## 概述

本文提供完整的、可直接运行的 Python 代码示例，涵盖 ChatPromptTemplate 的常见使用场景。

---

## 1. 基础模板使用

### 1.1 创建简单模板

```python
"""
示例1: 创建和使用基础模板
演示：from_messages() 方法的基本用法
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ===== 1. 创建模板 =====
print("=== 创建模板 ===")
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

print(f"输入变量: {template.input_variables}")
print(f"消息数量: {len(template.messages)}")

# ===== 2. 格式化消息 =====
print("\n=== 格式化消息 ===")
messages = template.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})

for i, msg in enumerate(messages):
    print(f"{i+1}. {msg.type}: {msg.content}")

# ===== 3. 与模型组合 =====
print("\n=== 与模型组合 ===")
model = ChatOpenAI(model="gpt-4o-mini")
chain = template | model

response = chain.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})

print(f"回答: {response.content}")
```

### 1.2 多种创建方式

```python
"""
示例2: 不同的模板创建方式
演示：from_messages(), from_template() 等方法
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ===== 方式1: from_messages (推荐) =====
print("=== 方式1: from_messages ===")
template1 = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
print(f"变量: {template1.input_variables}")

# ===== 方式2: from_template (单消息) =====
print("\n=== 方式2: from_template ===")
template2 = ChatPromptTemplate.from_template("{question}")
print(f"变量: {template2.input_variables}")

# ===== 方式3: 使用消息对象 =====
print("\n=== 方式3: 使用消息对象 ===")
template3 = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是{role}"),
    HumanMessage(content="{question}")
])
print(f"变量: {template3.input_variables}")

# ===== 方式4: 混合使用 =====
print("\n=== 方式4: 混合使用 ===")
template4 = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    SystemMessage(content="你的专长是{skill}"),
    ("human", "{question}")
])
print(f"变量: {template4.input_variables}")
```

---

## 2. 变量替换

### 2.1 基础变量替换

```python
"""
示例3: 变量替换
演示：如何使用变量占位符
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# ===== 1. 单个变量 =====
print("=== 单个变量 ===")
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{question}")
])

chain = template | model
response = chain.invoke({"question": "你好"})
print(f"回答: {response.content}\n")

# ===== 2. 多个变量 =====
print("=== 多个变量 ===")
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，专长是{skill}"),
    ("human", "{question}")
])

chain = template | model
response = chain.invoke({
    "role": "Python专家",
    "skill": "代码优化",
    "question": "如何优化代码？"
})
print(f"回答: {response.content}\n")

# ===== 3. 变量验证 =====
print("=== 变量验证 ===")
try:
    # 缺少变量会报错
    response = chain.invoke({"role": "助手"})
except KeyError as e:
    print(f"❌ 错误: 缺少变量 {e}")
```

### 2.2 类型转换

```python
"""
示例4: 变量类型转换
演示：不同类型的变量如何处理
"""
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "数据：{data}")
])

# ===== 测试不同类型 =====
print("=== 类型转换测试 ===\n")

# 字符串
messages = template.invoke({"data": "文本"})
print(f"字符串: {messages[0].content}")

# 数字
messages = template.invoke({"data": 123})
print(f"数字: {messages[0].content}")

# 列表
messages = template.invoke({"data": [1, 2, 3]})
print(f"列表: {messages[0].content}")

# 字典
messages = template.invoke({"data": {"key": "value"}})
print(f"字典: {messages[0].content}")

# None
messages = template.invoke({"data": None})
print(f"None: {messages[0].content}")
```

---

## 3. 部分应用（Partial）

### 3.1 静态部分应用

```python
"""
示例5: 静态部分应用
演示：预填充固定变量
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# ===== 1. 创建基础模板 =====
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，专长是{skill}"),
    ("human", "{question}")
])

print(f"原始变量: {template.input_variables}")

# ===== 2. 部分应用 =====
partial_template = template.partial(
    role="Python专家",
    skill="代码优化"
)

print(f"部分应用后变量: {partial_template.input_variables}")

# ===== 3. 使用部分应用的模板 =====
chain = partial_template | model

response = chain.invoke({"question": "如何优化这段代码？"})
print(f"\n回答: {response.content}")
```

### 3.2 动态部分应用

```python
"""
示例6: 动态部分应用
演示：使用函数生成动态值
"""
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 定义动态函数 =====
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_user_context():
    """获取用户上下文"""
    return "开发环境"

# ===== 2. 创建模板 =====
template = ChatPromptTemplate.from_messages([
    ("system", "当前时间：{current_time}，环境：{context}"),
    ("human", "{question}")
])

# ===== 3. 动态部分应用 =====
partial_template = template.partial(
    current_time=get_current_time,
    context=get_user_context
)

# ===== 4. 使用 =====
model = ChatOpenAI(model="gpt-4o-mini")
chain = partial_template | model

print("=== 第一次调用 ===")
response1 = chain.invoke({"question": "现在几点？"})
print(f"回答: {response1.content}\n")

import time
time.sleep(2)

print("=== 第二次调用（2秒后）===")
response2 = chain.invoke({"question": "现在几点？"})
print(f"回答: {response2.content}")
```

---

## 4. MessagesPlaceholder 使用

### 4.1 基础用法

```python
"""
示例7: MessagesPlaceholder 基础用法
演示：注入动态消息列表
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ===== 1. 创建带 MessagesPlaceholder 的模板 =====
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

print(f"输入变量: {template.input_variables}")

# ===== 2. 准备历史消息 =====
history = [
    HumanMessage(content="我叫张三"),
    AIMessage(content="你好张三！"),
    HumanMessage(content="我喜欢Python"),
    AIMessage(content="Python很棒！")
]

# ===== 3. 使用模板 =====
model = ChatOpenAI(model="gpt-4o-mini")
chain = template | model

response = chain.invoke({
    "history": history,
    "question": "我叫什么名字？"
})

print(f"\n回答: {response.content}")
```

### 4.2 可选的 MessagesPlaceholder

```python
"""
示例8: 可选的 MessagesPlaceholder
演示：optional=True 的使用
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 创建可选的 MessagesPlaceholder =====
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-4o-mini")
chain = template | model

# ===== 2. 不提供 history =====
print("=== 不提供 history ===")
response = chain.invoke({"question": "你好"})
print(f"回答: {response.content}\n")

# ===== 3. 提供 history =====
print("=== 提供 history ===")
history = [
    HumanMessage(content="我叫李四"),
    AIMessage(content="你好李四！")
]

response = chain.invoke({
    "history": history,
    "question": "我叫什么名字？"
})
print(f"回答: {response.content}")
```

---

## 5. 模板组合

### 5.1 使用 + 操作符

```python
"""
示例9: 模板组合
演示：使用 + 操作符组合模板
"""
from langchain_core.prompts import ChatPromptTemplate

# ===== 1. 创建子模板 =====
system_template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}")
])

context_template = ChatPromptTemplate.from_messages([
    ("human", "上下文：{context}")
])

question_template = ChatPromptTemplate.from_messages([
    ("human", "问题：{question}")
])

# ===== 2. 组合模板 =====
combined_template = system_template + context_template + question_template

print(f"组合后的变量: {combined_template.input_variables}")

# ===== 3. 使用组合模板 =====
messages = combined_template.invoke({
    "role": "助手",
    "context": "这是背景信息",
    "question": "这是问题"
})

print("\n=== 生成的消息 ===")
for i, msg in enumerate(messages):
    print(f"{i+1}. {msg.type}: {msg.content}")
```

### 5.2 条件组合

```python
"""
示例10: 条件组合
演示：根据条件动态组合模板
"""
from langchain_core.prompts import ChatPromptTemplate

def create_template(include_context: bool, include_examples: bool):
    """根据条件创建模板"""
    # 基础模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手")
    ])

    # 添加上下文
    if include_context:
        context_template = ChatPromptTemplate.from_messages([
            ("human", "上下文：{context}")
        ])
        template = template + context_template

    # 添加示例
    if include_examples:
        examples_template = ChatPromptTemplate.from_messages([
            ("human", "示例：{examples}")
        ])
        template = template + examples_template

    # 添加问题
    question_template = ChatPromptTemplate.from_messages([
        ("human", "问题：{question}")
    ])
    template = template + question_template

    return template

# ===== 测试不同组合 =====
print("=== 只有问题 ===")
template1 = create_template(False, False)
print(f"变量: {template1.input_variables}\n")

print("=== 问题 + 上下文 ===")
template2 = create_template(True, False)
print(f"变量: {template2.input_variables}\n")

print("=== 问题 + 上下文 + 示例 ===")
template3 = create_template(True, True)
print(f"变量: {template3.input_variables}")
```

---

## 6. 实用模板模式

### 6.1 角色模板工厂

```python
"""
示例11: 角色模板工厂
演示：创建不同角色的模板
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def create_role_template(role: str, personality: str, style: str):
    """创建角色模板"""
    return ChatPromptTemplate.from_messages([
        ("system", f"""你是{role}，具有以下特征：
- 性格：{personality}
- 说话风格：{style}
"""),
        ("human", "{question}")
    ])

# ===== 创建不同角色 =====
python_expert = create_role_template(
    role="Python专家",
    personality="严谨、专业",
    style="技术性强、简洁"
)

friendly_teacher = create_role_template(
    role="编程老师",
    personality="耐心、友好",
    style="通俗易懂、鼓励性"
)

# ===== 使用不同角色 =====
model = ChatOpenAI(model="gpt-4o-mini")

print("=== Python专家 ===")
chain1 = python_expert | model
response1 = chain1.invoke({"question": "什么是装饰器？"})
print(f"{response1.content}\n")

print("=== 编程老师 ===")
chain2 = friendly_teacher | model
response2 = chain2.invoke({"question": "什么是装饰器？"})
print(f"{response2.content}")
```

### 6.2 Few-shot 模板

```python
"""
示例12: Few-shot 学习模板
演示：使用示例引导模型
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ===== 1. 创建 Few-shot 模板 =====
template = ChatPromptTemplate.from_messages([
    ("system", "你是{task}助手"),
    MessagesPlaceholder(variable_name="examples"),
    ("human", "{input}")
])

# ===== 2. 准备示例 =====
sentiment_examples = [
    HumanMessage(content="这个产品很好"),
    AIMessage(content="正面"),
    HumanMessage(content="这个产品很差"),
    AIMessage(content="负面"),
    HumanMessage(content="这个产品还不错"),
    AIMessage(content="正面")
]

# ===== 3. 使用模板 =====
model = ChatOpenAI(model="gpt-4o-mini")
chain = template | model

test_cases = [
    "这个产品质量一般",
    "非常满意这次购物",
    "完全不推荐"
]

print("=== 情感分类 ===")
for test in test_cases:
    response = chain.invoke({
        "task": "情感分类",
        "examples": sentiment_examples,
        "input": test
    })
    print(f"输入: {test}")
    print(f"分类: {response.content}\n")
```

---

## 7. 完整应用示例

### 7.1 可配置的聊天机器人

```python
"""
示例13: 可配置的聊天机器人
演示：使用模板构建灵活的聊天应用
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional

load_dotenv()

class ConfigurableChatBot:
    """可配置的聊天机器人"""

    def __init__(
        self,
        role: str = "助手",
        personality: str = "友好",
        max_history: int = 10
    ):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.max_history = max_history
        self.history: List = []

        # 创建模板
        self.template = ChatPromptTemplate.from_messages([
            ("system", f"你是{role}，性格{personality}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        self.chain = self.template | self.model

    def chat(self, question: str) -> str:
        """发送消息"""
        # 限制历史长度
        history = self.history[-self.max_history:]

        # 调用模型
        response = self.chain.invoke({
            "history": history,
            "question": question
        })

        # 更新历史
        self.history.append(HumanMessage(content=question))
        self.history.append(response)

        return response.content

    def clear(self):
        """清空历史"""
        self.history = []

# ===== 使用示例 =====
print("=== 可配置聊天机器人 ===\n")

# 创建Python专家机器人
bot = ConfigurableChatBot(
    role="Python专家",
    personality="专业、严谨",
    max_history=6
)

print("用户: 什么是装饰器？")
print(f"AI: {bot.chat('什么是装饰器？')}\n")

print("用户: 能举个例子吗？")
print(f"AI: {bot.chat('能举个例子吗？')}\n")

print(f"历史消息数: {len(bot.history)}")
```

---

## 8. 最佳实践

### 8.1 模板复用

```python
"""
最佳实践：模板复用
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ 好：定义一次，多处使用
ASSISTANT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-4o-mini")
chain = ASSISTANT_TEMPLATE | model

# 场景1
response1 = chain.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})

# 场景2
response2 = chain.invoke({
    "role": "JavaScript专家",
    "question": "什么是闭包？"
})

# ❌ 不好：每次都创建新模板
def bad_ask(role, question):
    template = ChatPromptTemplate.from_messages([
        ("system", f"你是{role}"),
        ("human", "{question}")
    ])
    chain = template | model
    return chain.invoke({"question": question})
```

---

## 检查清单

完成本节实战后，你应该能够：

- [ ] 使用 from_messages() 创建模板
- [ ] 使用变量占位符
- [ ] 使用部分应用（partial）
- [ ] 使用 MessagesPlaceholder
- [ ] 组合多个模板
- [ ] 创建角色模板工厂
- [ ] 实现 Few-shot 学习
- [ ] 构建可配置的聊天机器人
- [ ] 应用模板复用最佳实践

---

**下一步**: 阅读 `07_实战代码_03_对话历史管理.md` 学习对话历史的管理技巧
