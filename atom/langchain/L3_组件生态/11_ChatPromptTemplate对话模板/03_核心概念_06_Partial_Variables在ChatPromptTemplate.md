# 核心概念6：Partial Variables 在 ChatPromptTemplate

> 本文档深入讲解 LangChain ChatPromptTemplate 中 Partial Variables（部分变量）的应用，包括与 PromptTemplate 的区别、实际应用场景、兼容性问题处理和最佳实践。

---

## 什么是 Partial Variables？

**Partial Variables（部分变量）** 是 ChatPromptTemplate 的预填充机制，允许在创建模板时设置部分变量的值，使用时只需传递剩余的动态变量。

**核心价值**：
- 减少重复传递固定参数
- 支持动态值（如时间戳）的自动注入
- 提升模板复用性
- 简化调用代码

[来源: reference/context7_langchain_03.md | LangChain 官方文档]

---

## 1. 基础用法

### 1.1 使用 partial() 方法

ChatPromptTemplate 提供 `partial()` 方法来预填充部分变量。

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建包含多个变量的模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant named {name}."),
    ("human", "Hi I'm {user}"),
    ("ai", "Hi there, {user}, I'm {name}."),
    ("human", "{input}"),
])

# 预填充部分变量
partial_template = template.partial(name="R2D2", user="Lucy")

# 使用时只需传递剩余变量
messages = partial_template.format_messages(input="hello")
print(messages)
```

**输出**：
```
[
    SystemMessage(content="You are an AI assistant named R2D2."),
    HumanMessage(content="Hi I'm Lucy"),
    AIMessage(content="Hi there, Lucy, I'm R2D2."),
    HumanMessage(content="hello")
]
```

**关键特性**：
- `partial()` 返回新的模板实例（不修改原模板）
- 可以预填充多个变量
- 预填充后的变量不需要在 `format_messages()` 中传递

[来源: reference/context7_langchain_03.md | LangChain 官方文档]

### 1.2 支持的值类型

Partial Variables 支持两种类型的值：

#### 1.2.1 固定字符串值

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{input}")
])

# 预填充固定值
specialized_template = template.partial(role="Python programming")

messages = specialized_template.format_messages(input="How to use list comprehension?")
```

**使用场景**：
- 系统角色定义
- 固定的上下文信息
- 预设的配置参数

#### 1.2.2 动态函数值

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

template = ChatPromptTemplate.from_messages([
    ("system", "Current date: {date}, time: {time}"),
    ("human", "{input}")
])

# 预填充动态值（使用函数）
dynamic_template = template.partial(
    date=get_current_date,
    time=get_current_time
)

# 每次调用时自动获取最新值
messages = dynamic_template.format_messages(input="What's the weather?")
```

**使用场景**：
- 时间戳（日期、时间）
- 动态配置（从环境变量读取）
- 计数器或序列号
- 随机值或 UUID

[来源: reference/search_partial_01.md | LangChain 社区实践]

---

## 2. 与 PromptTemplate 的区别

### 2.1 基本相似性

ChatPromptTemplate 和 PromptTemplate 都支持 Partial Variables，核心机制相同：

```python
# PromptTemplate 的 partial
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Language: {language}, Query: {query}",
    input_variables=["query"],
    partial_variables={"language": "Python"}
)

# ChatPromptTemplate 的 partial
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Language: {language}"),
    ("human", "{query}")
])
chat_prompt_partial = chat_prompt.partial(language="Python")
```

**共同特性**：
- 都支持固定值和函数值
- 都返回新的模板实例
- 都自动合并 partial_variables

[来源: reference/source_chatprompt_01.md | 源码分析]

### 2.2 关键区别

#### 区别1：消息级别的预填充

ChatPromptTemplate 的 partial 作用于整个消息列表：

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are {role}. Your expertise: {expertise}"),
    ("human", "User: {user_name}"),
    ("human", "{input}")
])

# 可以预填充跨多个消息的变量
partial_template = template.partial(
    role="AI Assistant",
    expertise="Python programming",
    user_name="Alice"
)

# 使用时只需传递 input
messages = partial_template.format_messages(input="Help me debug")
```

**优势**：
- 支持跨消息的变量预填充
- 适合复杂对话场景
- 更灵活的变量管理

#### 区别2：与 MessagesPlaceholder 的交互

ChatPromptTemplate 的 partial 可以与 MessagesPlaceholder 配合：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are {role}."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# 预填充 role，但保留 history 和 input 为动态
partial_template = template.partial(role="helpful assistant")

# 使用时传递 history 和 input
messages = partial_template.format_messages(
    history=[("human", "Hi"), ("ai", "Hello!")],
    input="How are you?"
)
```

**注意**：
- MessagesPlaceholder 的变量不能被 partial
- 只能预填充字符串模板中的变量

[来源: reference/source_chatprompt_01.md | 源码分析]

---

## 3. 兼容性问题与解决方案

### 3.1 历史兼容性问题

ChatPromptTemplate 的 Partial Variables 功能在早期版本中存在一些问题：

#### Issue #17560 (v0.1.9 - 2024)

**问题描述**：
- ChatPromptTemplate 的 partial_variables 不工作
- HumanMessagePromptTemplate.from_template 传入 partial 后仍报缺少变量错误

**示例**：
```python
# ❌ 在 v0.1.9 中会报错
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Date: {date}"),
    ("human", "{input}")
])

partial_template = template.partial(date="2024-01-01")
# 报错：Missing required input variable: date
```

**解决方案**：
- 升级到 v0.2.x 或更新版本
- 该问题已在后续版本中修复

[来源: reference/search_partial_01.md | GitHub Issue #17560]

#### Issue #6431 (2023)

**问题描述**：
- ChatPromptTemplate 使用 partial_variables 时出现验证错误
- 输入变量验证不匹配

**解决方案**：
- 已通过 PR 修复
- partial 变量现在可以正确注入

[来源: reference/search_partial_01.md | GitHub Issue #6431]

#### Issue #30049 (2025 - 最新)

**问题描述**：
- ChatPromptTemplate 在连接（concatenating）时，部分初始化的 partial variables 被忽略
- 导致变量处理不一致
- 影响模板组合使用

**示例**：
```python
# ⚠️ 在某些版本中可能有问题
template1 = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}")
]).partial(role="assistant")

template2 = ChatPromptTemplate.from_messages([
    ("human", "{input}")
])

# 组合后 partial 变量可能丢失
combined = template1 + template2
# 可能需要重新传递 role
```

**状态**：
- 2025年新发现的问题
- 需要进一步调查
- 建议在使用前测试

[来源: reference/search_partial_01.md | GitHub Issue #30049]

### 3.2 最佳实践

#### 实践1：版本选择

```python
# ✅ 推荐使用最新版本
# pip install langchain-core>=0.2.0

from langchain_core.prompts import ChatPromptTemplate

# 在生产环境使用前测试
template = ChatPromptTemplate.from_messages([
    ("system", "Date: {date}"),
    ("human", "{input}")
])

partial_template = template.partial(date="2026-02-26")

# 验证是否正常工作
try:
    messages = partial_template.format_messages(input="test")
    print("✅ Partial variables working correctly")
except Exception as e:
    print(f"❌ Error: {e}")
```

#### 实践2：避免模板组合时的问题

```python
from langchain_core.prompts import ChatPromptTemplate

# ✅ 方案1：在组合后再 partial
template1 = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}")
])

template2 = ChatPromptTemplate.from_messages([
    ("human", "{input}")
])

combined = template1 + template2
# 在组合后预填充
final_template = combined.partial(role="assistant")

# ✅ 方案2：使用 extend 而不是 +
base = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}")
]).partial(role="assistant")

base.extend([("human", "{input}")])
```

[来源: reference/search_partial_01.md | 社区最佳实践]

---

## 4. 实际应用场景

### 4.1 场景1：时间戳注入

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建带时间戳的模板
template = ChatPromptTemplate.from_messages([
    ("system", "Current time: {timestamp}. You are a helpful assistant."),
    ("human", "{input}")
])

# 预填充时间戳（每次调用自动更新）
timestamped_template = template.partial(timestamp=get_timestamp)

# 使用
messages = timestamped_template.format_messages(input="What's the weather?")
# System message 会包含当前时间
```

**优势**：
- 自动注入最新时间
- 无需手动传递
- 适合日志和审计场景

### 4.2 场景2：系统角色配置

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义不同角色的模板
base_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Your expertise: {expertise}. Tone: {tone}"),
    ("human", "{input}")
])

# 创建专门的助手
python_assistant = base_template.partial(
    role="Python programming expert",
    expertise="Python, FastAPI, async programming",
    tone="technical and precise"
)

data_analyst = base_template.partial(
    role="Data analyst",
    expertise="pandas, numpy, data visualization",
    tone="analytical and clear"
)

# 使用不同的助手
python_response = python_assistant.format_messages(
    input="How to use async/await?"
)

data_response = data_analyst.format_messages(
    input="How to analyze time series data?"
)
```

**优势**：
- 创建可复用的专门模板
- 统一管理系统配置
- 易于维护和扩展

### 4.3 场景3：用户上下文管理

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建用户特定的模板
template = ChatPromptTemplate.from_messages([
    ("system", "User: {user_name} (ID: {user_id}, Plan: {plan})"),
    ("human", "{input}")
])

# 为特定用户创建模板
user_template = template.partial(
    user_name="Alice",
    user_id="12345",
    plan="Premium"
)

# 在整个会话中使用
messages = user_template.format_messages(input="Show my usage stats")
```

**优势**：
- 自动注入用户信息
- 支持个性化响应
- 简化多租户系统

### 4.4 场景4：RAG 系统配置

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

# RAG 系统模板
rag_template = ChatPromptTemplate.from_messages([
    ("system", """You are a RAG assistant.
Date: {date}
Knowledge base: {kb_name}
Max context: {max_context} tokens
Instructions: {instructions}"""),
    ("human", "Context: {context}"),
    ("human", "Question: {question}")
])

# 预填充系统配置
configured_rag = rag_template.partial(
    date=get_date,
    kb_name="Company Documentation",
    max_context=4000,
    instructions="Answer based on the provided context. If unsure, say so."
)

# 使用时只需传递动态内容
messages = configured_rag.format_messages(
    context="[Retrieved documents...]",
    question="What is our refund policy?"
)
```

**优势**：
- 集中管理 RAG 配置
- 动态注入日期等信息
- 简化 RAG 调用代码

[来源: reference/context7_langchain_03.md | LangChain 官方文档]

---

## 5. 高级技巧

### 5.1 链式 Partial

可以多次调用 partial 来逐步填充变量：

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}, Language: {language}, Tone: {tone}"),
    ("human", "{input}")
])

# 第一次 partial
step1 = template.partial(role="assistant")

# 第二次 partial
step2 = step1.partial(language="English")

# 第三次 partial
final = step2.partial(tone="friendly")

# 使用
messages = final.format_messages(input="Hello")
```

**使用场景**：
- 分阶段配置模板
- 根据条件逐步填充
- 支持配置继承

### 5.2 条件 Partial

根据条件选择不同的 partial 值：

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Mode: {mode}, Level: {level}"),
    ("human", "{input}")
])

# 根据用户级别选择配置
def create_user_template(user_level):
    if user_level == "beginner":
        return template.partial(
            mode="tutorial",
            level="beginner-friendly"
        )
    elif user_level == "advanced":
        return template.partial(
            mode="expert",
            level="technical"
        )
    else:
        return template.partial(
            mode="standard",
            level="intermediate"
        )

# 使用
beginner_template = create_user_template("beginner")
advanced_template = create_user_template("advanced")
```

### 5.3 与 LCEL 集成

Partial Variables 可以与 LCEL 管道无缝集成：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 创建预填充的模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant."),
    ("human", "{input}")
]).partial(role="helpful")

# 集成到 LCEL 管道
chain = template | ChatOpenAI(model="gpt-4")

# 使用
response = chain.invoke({"input": "Hello"})
```

**优势**：
- 简化管道配置
- 减少重复代码
- 提升可读性

[来源: reference/context7_langchain_02.md | LangChain LCEL 文档]

---

## 6. 注意事项与限制

### 6.1 不能 Partial 的变量

以下类型的变量不能使用 partial：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}"),
    MessagesPlaceholder("history"),  # ❌ 不能 partial
    ("human", "{input}")
])

# ❌ 这会报错
# template.partial(history=[...])

# ✅ 正确做法：在 format_messages 时传递
messages = template.format_messages(
    role="assistant",  # 可以 partial
    history=[("human", "Hi")],  # 必须在这里传递
    input="Hello"
)
```

**限制**：
- MessagesPlaceholder 的变量必须在 format_messages 时传递
- 图片列表模板的变量不支持 partial

### 6.2 变量冲突

如果多次 partial 同一个变量，后面的会覆盖前面的：

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Role: {role}"),
    ("human", "{input}")
])

# 第一次 partial
step1 = template.partial(role="assistant")

# 第二次 partial 同一个变量（会覆盖）
step2 = step1.partial(role="expert")  # "expert" 覆盖 "assistant"

messages = step2.format_messages(input="Hello")
# System message: "Role: expert"
```

### 6.3 性能考虑

使用函数作为 partial 值时，函数会在每次 format_messages 时调用：

```python
from langchain_core.prompts import ChatPromptTemplate
import time

call_count = 0

def expensive_function():
    global call_count
    call_count += 1
    time.sleep(0.1)  # 模拟耗时操作
    return f"Value {call_count}"

template = ChatPromptTemplate.from_messages([
    ("system", "Data: {data}"),
    ("human", "{input}")
]).partial(data=expensive_function)

# 每次调用都会执行 expensive_function
messages1 = template.format_messages(input="test1")  # call_count = 1
messages2 = template.format_messages(input="test2")  # call_count = 2
```

**建议**：
- 避免在 partial 函数中执行耗时操作
- 考虑使用缓存机制
- 对于固定值，使用字符串而不是函数

---

## 7. 总结

### 核心要点

1. **基本用法**：使用 `partial()` 方法预填充变量
2. **值类型**：支持固定字符串和动态函数
3. **与 PromptTemplate 的区别**：支持跨消息的变量预填充
4. **兼容性**：注意早期版本的问题，建议使用 v0.2.x+
5. **应用场景**：时间戳、系统配置、用户上下文、RAG 系统
6. **限制**：MessagesPlaceholder 变量不能 partial

### 最佳实践

- ✅ 使用最新版本的 LangChain
- ✅ 在生产环境使用前测试 partial 功能
- ✅ 避免在 partial 函数中执行耗时操作
- ✅ 使用链式 partial 实现分阶段配置
- ✅ 与 LCEL 管道集成以简化代码
- ❌ 不要尝试 partial MessagesPlaceholder 变量
- ❌ 注意模板组合时的 partial 变量处理

### 下一步学习

- 学习模板组合与扩展（核心概念7）
- 了解 MessagesPlaceholder 的高级用法
- 探索 LCEL 管道中的模板应用
- 研究生产环境的模板管理策略

---

**参考资料**：
- [LangChain 官方文档 - ChatPromptTemplate](https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate)
- [GitHub Issue #17560 - partial_variables not working](https://github.com/langchain-ai/langchain/issues/17560)
- [GitHub Issue #30049 - Partial variables ignored when concatenating](https://github.com/langchain-ai/langchain/issues/30049)
- [源码分析 - chat.py](reference/source_chatprompt_01.md)

**版本信息**：
- LangChain Core: v0.2+
- 文档更新时间: 2026-02-26
