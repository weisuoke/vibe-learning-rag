# 核心概念10：与ChatPromptTemplate关系

> 本文档是 PromptTemplate高级用法 知识点的第10个核心概念

---

## 概念定义

**ChatPromptTemplate 是专门用于对话场景的模板类，与 PromptTemplate 在消息结构、组合方式和变量处理上有本质区别。**

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## 核心区别

### 1. 数据结构差异

**PromptTemplate**：
- 输出单个字符串
- 适用于简单的文本生成场景
- 直接传递给 LLM

**ChatPromptTemplate**：
- 输出消息数组（Message List）
- 每条消息有角色（system、user、assistant）
- 适用于对话场景

[来源: reference/context7_langchain_03.md | LangChain 官方文档]

### 2. 创建方式对比

#### PromptTemplate 创建

```python
from langchain_core.prompts import PromptTemplate

# 方式1：from_template
prompt = PromptTemplate.from_template("Say {foo}")

# 方式2：显式指定
prompt = PromptTemplate(
    template="Say {foo}",
    input_variables=["foo"]
)
```

[来源: reference/source_prompttemplate_01.md | 源码分析]

#### ChatPromptTemplate 创建

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式1：from_messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

# 方式2：使用消息模板
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant"
)
human_template = HumanMessagePromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## 组合方式差异

### PromptTemplate 组合

**使用 `+` 操作符**：

```python
from langchain_core.prompts import PromptTemplate

# 组合两个 PromptTemplate
prompt1 = PromptTemplate.from_template("Hello {name}. ")
prompt2 = PromptTemplate.from_template("You are {age} years old.")

combined = prompt1 + prompt2
# 结果: "Hello {name}. You are {age} years old."

print(combined.format(name="Alice", age=25))
# 输出: "Hello Alice. You are 25 years old."
```

[来源: reference/source_prompttemplate_01.md | 源码分析 prompt.py:142-184]

**关键特性**：
- 自动合并 `input_variables`（取并集）
- 自动合并 `partial_variables`（检查冲突）
- 要求两个模板的 `template_format` 必须一致

### ChatPromptTemplate 组合

**使用 `+` 操作符**：

```python
from langchain_core.prompts import ChatPromptTemplate

# 组合两个 ChatPromptTemplate
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", "Tell me about {topic}")
])

combined = prompt1 + prompt2
# 结果: 包含两条消息的 ChatPromptTemplate

print(combined.invoke({"topic": "AI"}))
# 输出: [SystemMessage(content="You are a helpful assistant"),
#       HumanMessage(content="Tell me about AI")]
```

[来源: reference/search_composition_01.md | 社区资料]

**关键特性**：
- 合并消息列表
- 保持消息顺序
- 自动合并变量

---

## partial_variables 兼容性问题

### 问题描述

**ChatPromptTemplate 在某些版本中对 `partial_variables` 的支持存在兼容性问题。**

[来源: reference/search_partial_01.md | GitHub Issue #17560]

### 问题示例

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# ❌ 在某些版本中可能不工作
prompt = ChatPromptTemplate.from_messages([
    ("system", "Today is {date}"),
    ("user", "{query}")
], partial_variables={"date": get_current_date})

# 可能报错或 partial_variables 不生效
```

[来源: reference/search_partial_01.md | GitHub Issue #17560]

### 解决方案

**方案1：使用 PromptTemplate 构建消息**

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# 为每个消息单独创建 PromptTemplate
system_prompt = PromptTemplate(
    template="Today is {date}",
    input_variables=[],
    partial_variables={"date": get_current_date}
)

user_prompt = PromptTemplate(
    template="{query}",
    input_variables=["query"]
)

# 转换为消息模板
system_message = SystemMessagePromptTemplate(prompt=system_prompt)
user_message = HumanMessagePromptTemplate(prompt=user_prompt)

# 组合为 ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    user_message
])

# 使用
result = chat_prompt.invoke({"query": "What's the weather?"})
print(result)
```

[来源: reference/search_partial_01.md | 社区解决方案]

**方案2：在调用时传递 partial 变量**

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

prompt = ChatPromptTemplate.from_messages([
    ("system", "Today is {date}"),
    ("user", "{query}")
])

# 在调用时传递所有变量（包括 partial 变量）
result = prompt.invoke({
    "date": datetime.now().strftime("%Y-%m-%d"),
    "query": "What's the weather?"
})
```

[来源: reference/search_partial_01.md | 社区解决方案]

**方案3：使用最新版本**

```bash
# 更新到最新版本以获得更好的支持
uv add langchain-core --upgrade
```

---

## 使用场景对比

### 何时使用 PromptTemplate

**适用场景**：
1. **简单文本生成**：单次输入输出
2. **模板复用**：需要频繁组合和复用模板
3. **非对话场景**：如文本分类、摘要生成
4. **需要高级模板功能**：如 Mustache、Jinja2 格式

**示例**：

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# 文本分类场景
prompt = PromptTemplate.from_template(
    "Classify the following text into one of these categories: {categories}\n\n"
    "Text: {text}\n\n"
    "Category:"
)

llm = OpenAI()
chain = prompt | llm

result = chain.invoke({
    "categories": "positive, negative, neutral",
    "text": "I love this product!"
})
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### 何时使用 ChatPromptTemplate

**适用场景**：
1. **对话系统**：多轮对话、聊天机器人
2. **需要系统提示**：设定 AI 角色和行为
3. **多角色交互**：模拟多人对话
4. **Few-shot 学习**：提供示例对话

**示例**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 对话场景
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Always provide clear explanations."),
    ("user", "How do I {task} in {language}?")
])

llm = ChatOpenAI()
chain = prompt | llm

result = chain.invoke({
    "task": "sort a list",
    "language": "Python"
})
```

[来源: reference/context7_langchain_03.md | LangChain 官方文档]

---

## 迁移模式

### 从 PromptTemplate 迁移到 ChatPromptTemplate

**场景**：需要将简单的文本生成改为对话格式

**迁移步骤**：

```python
# ===== 原始 PromptTemplate =====
from langchain_core.prompts import PromptTemplate

old_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

# ===== 迁移到 ChatPromptTemplate =====
from langchain_core.prompts import ChatPromptTemplate

new_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# 使用方式相同
result = new_prompt.invoke({"question": "What is AI?"})
```

[来源: reference/search_composition_01.md | 社区最佳实践]

### 从 ChatPromptTemplate 迁移到 PromptTemplate

**场景**：需要将对话格式简化为单个字符串

**迁移步骤**：

```python
# ===== 原始 ChatPromptTemplate =====
from langchain_core.prompts import ChatPromptTemplate

old_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# ===== 迁移到 PromptTemplate =====
from langchain_core.prompts import PromptTemplate

new_prompt = PromptTemplate.from_template(
    "System: You are a helpful assistant.\n\n"
    "User: {question}\n\n"
    "Assistant:"
)

# 使用方式相同
result = new_prompt.invoke({"question": "What is AI?"})
```

[来源: reference/search_composition_01.md | 社区最佳实践]

---

## 完整实战示例

### 示例1：PromptTemplate 与 ChatPromptTemplate 对比

```python
"""
PromptTemplate vs ChatPromptTemplate 对比示例
演示：两种模板的使用差异
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

load_dotenv()

# ===== 1. PromptTemplate 示例 =====
print("=== PromptTemplate 示例 ===")

prompt_template = PromptTemplate.from_template(
    "Translate the following English text to {language}: {text}"
)

llm = OpenAI(temperature=0)
chain = prompt_template | llm

result = chain.invoke({
    "language": "French",
    "text": "Hello, how are you?"
})

print(f"PromptTemplate 结果: {result}")
print()

# ===== 2. ChatPromptTemplate 示例 =====
print("=== ChatPromptTemplate 示例 ===")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator."),
    ("user", "Translate the following English text to {language}: {text}")
])

chat_llm = ChatOpenAI(temperature=0)
chat_chain = chat_prompt | chat_llm

chat_result = chat_chain.invoke({
    "language": "French",
    "text": "Hello, how are you?"
})

print(f"ChatPromptTemplate 结果: {chat_result.content}")
print()

# ===== 3. 组合示例 =====
print("=== 组合示例 ===")

# PromptTemplate 组合
prompt1 = PromptTemplate.from_template("Hello {name}. ")
prompt2 = PromptTemplate.from_template("You are {age} years old.")
combined_prompt = prompt1 + prompt2

print(f"PromptTemplate 组合: {combined_prompt.format(name='Alice', age=25)}")

# ChatPromptTemplate 组合
chat1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant")
])
chat2 = ChatPromptTemplate.from_messages([
    ("user", "Tell me about {topic}")
])
combined_chat = chat1 + chat2

print(f"ChatPromptTemplate 组合: {combined_chat.invoke({'topic': 'AI'})}")
```

**运行输出示例**：
```
=== PromptTemplate 示例 ===
PromptTemplate 结果: Bonjour, comment allez-vous?

=== ChatPromptTemplate 示例 ===
ChatPromptTemplate 结果: Bonjour, comment allez-vous?

=== 组合示例 ===
PromptTemplate 组合: Hello Alice. You are 25 years old.
ChatPromptTemplate 组合: [SystemMessage(content='You are a helpful assistant'), HumanMessage(content='Tell me about AI')]
```

[来源: reference/context7_langchain_01.md + reference/context7_langchain_03.md]

### 示例2：partial_variables 兼容性处理

```python
"""
partial_variables 兼容性处理示例
演示：如何在 ChatPromptTemplate 中正确使用 partial_variables
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI

load_dotenv()

# ===== 1. 定义动态函数 =====
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

# ===== 2. 方案1：使用 PromptTemplate 构建消息 =====
print("=== 方案1：使用 PromptTemplate 构建消息 ===")

system_prompt = PromptTemplate(
    template="You are a helpful assistant. Today is {date} and the current time is {time}.",
    input_variables=[],
    partial_variables={
        "date": get_current_date,
        "time": get_current_time
    }
)

user_prompt = PromptTemplate(
    template="{query}",
    input_variables=["query"]
)

system_message = SystemMessagePromptTemplate(prompt=system_prompt)
user_message = HumanMessagePromptTemplate(prompt=user_prompt)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    user_message
])

llm = ChatOpenAI(temperature=0)
chain = chat_prompt | llm

result = chain.invoke({"query": "What's the weather like?"})
print(f"结果: {result.content}")
print()

# ===== 3. 方案2：在调用时传递变量 =====
print("=== 方案2：在调用时传递变量 ===")

simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today is {date} and the current time is {time}."),
    ("user", "{query}")
])

result2 = simple_prompt.invoke({
    "date": get_current_date(),
    "time": get_current_time(),
    "query": "What's the weather like?"
})

print(f"格式化结果: {result2}")
```

**运行输出示例**：
```
=== 方案1：使用 PromptTemplate 构建消息 ===
结果: I don't have real-time weather information...

=== 方案2：在调用时传递变量 ===
格式化结果: [SystemMessage(content='You are a helpful assistant. Today is 2026-02-26 and the current time is 10:30:45.'), HumanMessage(content="What's the weather like?")]
```

[来源: reference/search_partial_01.md | 社区解决方案]

---

## 双重类比

### 前端类比

| 概念 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 前端类比 | 模板字符串 (`Hello ${name}`) | React 组件数组 (`[<Header />, <Body />]`) |
| 数据结构 | 单个字符串 | 组件列表 |
| 组合方式 | 字符串拼接 | 数组合并 |
| 适用场景 | 简单文本渲染 | 复杂 UI 结构 |

### 日常生活类比

| 概念 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 日常类比 | 填空题 | 对话剧本 |
| 结构 | 单句话 | 多人对话 |
| 使用方式 | 填入答案 | 分配角色台词 |
| 适用场景 | 简单问答 | 戏剧表演 |

[来源: CLAUDE_LANGCHAIN.md | LangChain 类比对照表]

---

## 关键要点总结

1. **数据结构差异**：
   - PromptTemplate → 单个字符串
   - ChatPromptTemplate → 消息数组

2. **组合方式**：
   - 两者都支持 `+` 操作符
   - PromptTemplate 合并字符串
   - ChatPromptTemplate 合并消息列表

3. **partial_variables 兼容性**：
   - ChatPromptTemplate 在某些版本中存在兼容性问题
   - 推荐使用 PromptTemplate 构建消息或在调用时传递变量

4. **使用场景**：
   - PromptTemplate：简单文本生成、模板复用
   - ChatPromptTemplate：对话系统、多角色交互

5. **迁移策略**：
   - 从 PromptTemplate 到 ChatPromptTemplate：拆分为多条消息
   - 从 ChatPromptTemplate 到 PromptTemplate：合并为单个字符串

---

## 参考资料

- [LangChain 官方文档 - PromptTemplate](https://docs.langchain.com/)
- [LangChain 官方文档 - ChatPromptTemplate](https://docs.langchain.com/)
- [GitHub Issue #17560 - partial_variables 兼容性问题](https://github.com/langchain-ai/langchain/issues/17560)
- [LangChain 源码 - prompt.py](sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py)

---

**文档版本**: v1.0
**最后更新**: 2026-02-26
**知识点**: PromptTemplate高级用法
**层级**: L3_组件生态
