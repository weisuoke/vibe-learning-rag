---
type: context7_documentation
library: langchain
version: v0.2 / OSS Python
fetched_at: 2026-02-26
knowledge_point: 11_ChatPromptTemplate对话模板
context7_query: ChatPromptTemplate message types roles system human ai
---

# Context7 文档：LangChain 消息类型与角色

## 文档来源
- 库名称：LangChain OSS Python
- 版本：v0.2
- 官方文档链接：https://docs.langchain.com/oss/python/

## 关键信息提取

### 1. Create message list for chat invocation

**来源**: https://docs.langchain.com/oss/python/integrations/chat/zhipuai

**核心内容**: 构建包含 AI、System、Human 消息的列表

**代码示例**:
```python
messages = [
    AIMessage(content="Hi."),
    SystemMessage(content="Your role is a poet."),
    HumanMessage(content="Write a short poem about AI in four lines."),
]
```

**关键特性**:
- **SystemMessage**: 设置角色（如 poet）
- **HumanMessage**: 提供任务
- **AIMessage**: AI 的回复

---

### 2. Chain Model with ChatPromptTemplate

**来源**: https://docs.langchain.com/oss/python/integrations/chat/gradientai

**核心内容**: 使用 system 和 human 角色创建提示模板

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        (
            "system",
            'You are a knowledgeable assistant. Carefully read the provided context and answer the user\'s question. If the answer is present in the context, cite the relevant sentence. If not, reply with "Not found in context."',
        ),
        ("human", "Context: {context}\nQuestion: {question}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "context": (
            "The Eiffel Tower is located in Paris and was completed in 1889. "
            "It was designed by Gustave Eiffel's engineering company. "
            "The tower is one of the most recognizable structures in the world. "
            "The Statue of Liberty was a gift from France to the United States."
        ),
        "question": "Who designed the Eiffel Tower and when was it completed?",
    }
)
```

**关键特性**:
- 使用元组格式定义消息
- system 消息定义 AI 的行为规则
- human 消息包含变量（context, question）
- 与 LLM 链式组合

---

### 3. Invoke ChatVertexAI for Chat Completions

**来源**: https://docs.langchain.com/oss/python/integrations/chat/google_vertex_ai

**核心内容**: 使用消息列表调用 LLM

**代码示例**:
```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence."
    ),
    ("human", "I love programming.")
]
ai_msg = llm.invoke(messages)
ai_msg
```

**关键特性**:
- 简洁的元组格式
- system 消息定义翻译任务
- 直接传递给 LLM

---

### 4. Create Chat Prompt Template with Tool Binding

**来源**: https://docs.langchain.com/oss/python/integrations/tools/tilores

**核心内容**: 使用 placeholder 消息角色

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# specifying tool_choice will force the model to call this tool.
model_with_tools = model.bind_tools([search_tool], tool_choice=search_tool.name)

model_chain = prompt | model_with_tools
```

**关键特性**:
- **placeholder 角色**: 动态插入消息列表
- 与工具绑定集成
- 支持复杂的消息流

---

### 5. Invoke ChatGoogleGenerativeAI Model with Messages

**来源**: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai

**核心内容**: 使用元组格式调用模型

**代码示例**:
```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = model.invoke(messages)
ai_msg
```

**关键特性**:
- 元组格式：`(role, content)`
- 支持的角色：system, human, ai
- 返回 AIMessage 对象

---

## 消息角色总结

### 1. System 角色
**用途**: 定义 AI 的角色、行为规则、任务说明

**典型场景**:
- 设置 AI 的身份（如 "You are a helpful assistant"）
- 定义任务规则（如 "Translate English to French"）
- 提供上下文约束（如 "Answer based on the provided context"）

**最佳实践**:
- 放在消息列表的最前面
- 清晰明确地定义 AI 的行为
- 避免过于复杂的指令

### 2. Human 角色
**用途**: 用户的输入、问题、请求

**典型场景**:
- 用户的问题
- 需要处理的文本
- 任务输入

**最佳实践**:
- 支持变量替换（如 `{user_input}`）
- 可以包含上下文信息
- 保持简洁明了

### 3. AI 角色
**用途**: AI 的回复、示例回答

**典型场景**:
- Few-shot 学习示例
- 对话历史中的 AI 回复
- 预设的回答模式

**最佳实践**:
- 用于 Few-shot 示例
- 展示期望的回答格式
- 提供对话上下文

### 4. Placeholder 角色
**用途**: 动态插入消息列表

**典型场景**:
- 对话历史
- 工具调用结果
- 动态消息流

**最佳实践**:
- 使用 MessagesPlaceholder 类
- 支持 optional 参数
- 灵活注入消息

---

## 消息格式对比

| 格式 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| 元组格式 | `("human", "Hello")` | 简洁、易读 | 功能有限 |
| 消息对象 | `HumanMessage(content="Hello")` | 功能完整、灵活 | 代码较长 |
| 模板对象 | `HumanMessagePromptTemplate.from_template("{input}")` | 支持变量 | 最复杂 |

---

## 实际应用模式

### 模式 1: 简单问答
```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
]
```

### 模式 2: 翻译任务
```python
messages = [
    ("system", "You are a translator. Translate English to French."),
    ("human", "{text}")
]
```

### 模式 3: 上下文问答
```python
messages = [
    ("system", "Answer based on the context."),
    ("human", "Context: {context}\nQuestion: {question}")
]
```

### 模式 4: Few-shot 学习
```python
messages = [
    ("system", "You are a translator."),
    ("human", "Hello"),
    ("ai", "Bonjour"),
    ("human", "{input}")
]
```

### 模式 5: 工具集成
```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}"),
    ("placeholder", "{messages}")  # 工具调用结果
]
```

---

## 与 LCEL 集成

### 基础链式组合
```python
chain = prompt | llm
result = chain.invoke({"question": "..."})
```

### 完整处理链
```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "..."})
```

### 流式处理
```python
for chunk in chain.stream({"input": "..."}):
    print(chunk)
```

---

## 最佳实践总结

1. **System 消息优先**: 始终在最前面定义 AI 的角色
2. **使用元组格式**: 对于简单场景，元组格式最简洁
3. **变量命名清晰**: 使用有意义的变量名（如 `{question}` 而非 `{q}`）
4. **Few-shot 示例**: 使用 AI 角色提供示例回答
5. **Placeholder 用于动态内容**: 对话历史、工具结果等
6. **LCEL 链式组合**: 使用 `|` 操作符构建完整处理链

---

## 下一步调研方向

基于官方文档，需要进一步调研:
1. **Partial Variables 在 ChatPromptTemplate 中的应用**
2. **模板组合**: 使用 `+` 操作符组合模板
3. **多模态支持**: 图片消息的实际使用
4. **社区实践**: 实际项目中的使用模式
5. **性能优化**: 大规模对话历史的处理
