---
type: context7_documentation
library: langchain
version: v0.2
fetched_at: 2026-02-26
knowledge_point: 11_ChatPromptTemplate对话模板
context7_query: ChatPromptTemplate from_messages MessagesPlaceholder message types roles
---

# Context7 文档：LangChain ChatPromptTemplate

## 文档来源
- 库名称：LangChain
- 版本：v0.2
- 官方文档链接：https://python.langchain.com/v0.2/

## 关键信息提取

### 1. Create ChatPromptTemplate from Messages List

**来源**: https://python.langchain.com/v0.2/api_reference/_modules/langchain_core/prompts/chat

**核心内容**: 使用 `from_messages` 类方法创建 ChatPromptTemplate

**代码示例**:
```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Create using message objects
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
chat_prompt_from_objects = ChatPromptTemplate.from_messages(messages)

# Create using prompt template objects
prompt_templates = [
    SystemMessagePromptTemplate.from_template("You are {ai_role}"),
    HumanMessagePromptTemplate.from_template("I want to know about {subject}")
]
chat_prompt_from_templates = ChatPromptTemplate.from_messages(prompt_templates)
```

**关键特性**:
- 支持多种消息格式（message objects, prompt template objects）
- 灵活定义复杂对话结构
- 支持不同角色和消息类型

---

### 2. Use MessagesPlaceholder in ChatPromptTemplate

**来源**: https://python.langchain.com/v0.2/docs/concepts

**核心内容**: 使用 MessagesPlaceholder 动态插入消息列表

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
```

**关键特性**:
- 动态插入消息列表
- 适用于对话历史管理
- 灵活的消息构建

---

### 3. Create Chat Prompt Template from Messages

**来源**: https://python.langchain.com/v0.2/api_reference/langchain/agents/langchain.agents.schema.AgentScratchPadChatPromptTemplate

**核心内容**: from_messages 方法支持多种消息格式

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

# Instantiation from a list of message templates
template = ChatPromptTemplate.from_messages([
    ("human", "Hello, how are you?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "That's good to hear."),
])

# Instantiation from mixed message formats
from langchain_core.messages import SystemMessage
template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    ("human", "Hello, how are you?"),
])
```

**支持的消息格式**:
1. 元组格式: `(message_type, content)` - 如 `("human", "Hello")`
2. BaseMessage 对象: `SystemMessage(content="...")`
3. 字符串模板: 支持变量替换
4. 混合格式: 可以组合使用

---

### 4. Building a Prompt with Chat History using MessagesPlaceholder

**来源**: https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.MessagesPlaceholder

**核心内容**: 使用 MessagesPlaceholder 构建包含对话历史的提示词

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

prompt.invoke(
   {
       "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
       "question": "now multiply that by 4"
   }
)

# Output:
# ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="what's 5 + 2"),
#     AIMessage(content="5 + 2 is 7"),
#     HumanMessage(content="now multiply that by 4"),
# ])
```

**关键特性**:
- 系统消息 + 历史消息 + 用户问题的完整结构
- 自动转换元组为消息对象
- 支持动态对话历史注入

---

### 5. Create ChatPromptTemplate with System Message and MessagesPlaceholder

**来源**: https://python.langchain.com/v0.2/docs/tutorials/chatbot

**核心内容**: 结合系统消息和 MessagesPlaceholder 构建对话模板

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model
```

**应用场景**:
- 对话式 AI 系统
- 聊天机器人
- 需要保持上下文的应用

---

## 核心概念总结

### 1. 消息类型
- **SystemMessage**: 系统消息，定义 AI 的角色和行为
- **HumanMessage**: 用户消息
- **AIMessage**: AI 回复消息
- **ChatMessage**: 自定义角色消息

### 2. 创建方法
- **from_messages()**: 从消息列表创建（推荐）
- **from_template()**: 从单个模板字符串创建
- **直接构造**: 使用构造函数

### 3. 消息格式
- **元组格式**: `("role", "content")` - 最简洁
- **消息对象**: `SystemMessage(content="...")` - 最灵活
- **模板对象**: `SystemMessagePromptTemplate.from_template(...)` - 支持变量

### 4. MessagesPlaceholder
- **用途**: 动态插入消息列表
- **场景**: 对话历史、Few-shot 示例
- **特性**: 自动转换、灵活注入

### 5. 与 LCEL 集成
- 使用 `|` 操作符链式组合
- 与 ChatModel 无缝集成
- 支持流式处理

---

## 实际应用模式

### 模式 1: 基础对话模板
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
```

### 模式 2: 带历史记录的对话
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
```

### 模式 3: Few-shot 学习
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("human", "Hello"),
    ("ai", "Bonjour"),
    ("human", "Goodbye"),
    ("ai", "Au revoir"),
    ("human", "{input}")
])
```

### 模式 4: LCEL 链式组合
```python
chain = prompt | model | output_parser
result = chain.invoke({"input": "..."})
```

---

## 最佳实践

1. **优先使用 from_messages()**: 比直接构造更简洁
2. **使用元组格式**: 对于简单消息，元组格式最简洁
3. **MessagesPlaceholder 用于动态内容**: 对话历史、示例列表
4. **系统消息放在最前面**: 定义 AI 的角色和行为
5. **与 LCEL 结合**: 构建完整的处理链

---

## 与 PromptTemplate 的区别

| 特性 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 输出格式 | 字符串 | 消息列表 |
| 适用场景 | 文本生成 | 对话系统 |
| 消息类型 | 无 | System/Human/AI/Chat |
| 历史管理 | 不支持 | MessagesPlaceholder |
| LCEL 集成 | 支持 | 支持 |

---

## 下一步调研方向

基于官方文档，需要进一步调研:
1. **多模态支持**: 图片消息的实际使用
2. **Partial Variables**: 在 ChatPromptTemplate 中的应用
3. **模板组合**: 使用 `+` 操作符组合模板
4. **社区实践**: 实际项目中的使用模式
5. **性能优化**: 大规模对话历史的处理
